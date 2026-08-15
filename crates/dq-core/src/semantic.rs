//! LLM ve embedding model gerektirmeyen Turkce anlamsal on-isleme.
//!
//! Bu modul amac:
//! 1. Sorguyi terim bazli genisletmek (es anlamli kelimeler eklemek).
//! 2. Soru tipini tespit etmek (sayi, tarih, kisi, tanim, prosedur vb.).
//! 3. Sorgudan varlik (entity) cikarimak (tarih, tutar, kod vb.).
//!
//! Tofas kullanimi: BM25 ve dense modellerle birlesik kullanilarak
//! hem anahtar kelime hem de hafif anlamsal genisletme saglar.

use std::collections::HashSet;

use crate::text::tokenize_for_search;

// ---------------------------------------------------------------------------
// Turkce es anlamli kelime sozlugu (kucuk harf, tireli).
// Gerektiginde genisletilebilir; amac model dosyasi olmadan
// sorgu kapsamini artirmaktir.
// ---------------------------------------------------------------------------
const SYNONYMS: &[(&str, &[&str])] = &[
    ("proje", &["girisim", "calisma", "uygulama", "faaliyet"]),
    ("sistem", &["altyapi", "cerceve", "yapi", "ortam"]),
    ("geliştirme", &["yazilim", "kodlama", "implementasyon", "tasarim"]),
    ("test", &["deneme", "sinama", "test-etme", "dogrulama"]),
    ("analiz", &["inceleme", "cozumleme", "degerlendirme", "calisma"]),
    ("belge", &["dosya", "rapor", "kaynak", "metin", "dokuman"]),
    ("veri", &["bilgi", "icerik", "kayit", "veri-seti"]),
    ("model", &["modeller", "yapay-zeka", "ai", "ml", " algoritma"]),
    ("yapay-zeka", &["ai", "ml", "makine-ogrenmesi", "derin-ogrenme"]),
    ("makine-ogrenmesi", &["ai", "yapay-zeka", "ml", "model"]),
    ("güvenlik", &["kisitlama", "yetki", "erişim", "korunma"]),
    ("yetki", &["izin", "yetkilendirme", "rol", "access"]),
    ("kullanici", &["kisi", "kullanici-hesabi", "hesap", "profil"]),
    ("soru", &["sorgu", "istek", "dilek", "soru-cevap"]),
    ("cevap", &["sonuc", "cikti", "yanit", "cozum"]),
    ("islem", &["process", "calisma", "gorev", "operator"]),
    ("hata", &["bug", "sorun", "problem", "ariza", "eksik"]),
    ("doküman", &["belge", "dosya", "rapor", "kaynak"]),
    ("teknoloji", &["tech", "yazilim", "donanim", "sistem"]),
    ("guncelleme", &["versiyon", "surum", "yeni", "degisiklik"]),
    ("tarih", &["zaman", "tarih-araligi", "gun", "ay", "yil"]),
    ("saat", &["saat-dakika", "zaman", "sure"]),
    ("fiyat", &["ucret", "tutar", "maliyet", "fiyatlandirma"]),
    ("para", &["tl", "usd", "eur", "miktar", "tutar"]),
    ("sure", &["zaman", "periyot", "surec", "suresi"]),
    ("yil", &["yıl", "sene", "dönem"]),
    ("ay", &["aylik", "ay-suresi"]),
    ("gun", &["gün", "gunluk", "gune"]),
    ("hafta", &["haftalik", "hafta-suresi"]),
    ("sayi", &["numara", "rakam", "adet", "miktar"]),
    ("adet", &["sayi", "miktar", "numara"]),
    ("miktar", &["adet", "sayi", "tutar", "oran"]),
    ("oran", &["yuzde", "oranlamak", "pay"]),
    ("yuzde", &["%", "oran", "pay"]),
    ("minimum", &["en-az", "alt-sinir", "min"]),
    ("maksimum", &["en-cok", "ust-sinir", "max"]),
    ("kontrol", &["denetim", "kontrol-etme", "gözden-gecirme"]),
    ("rapor", &["belge", "dosya", "cikti", "sonuc"]),
    ("kurallar", &["kural-seti", "policy", "yonetmelik"]),
    ("sikayet", &["problem", "sorun", "hata", "ariza"]),
    ("talimat", &["kullanici-kilavuzu", "rehber", "dokuman", "prosedur"]),
    ("gorev", &["is", "gorev", "surec", "operator"]),
];

// ---------------------------------------------------------------------------
// Soru kelimeleri ve cevap tipi tahmini.
// ---------------------------------------------------------------------------
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuestionType {
    Unknown,
    Definition, // "X nedir?", "X tanımı?"
    Procedure,  // "Nasıl Y?", "Yapılışı?"
    Causal,     // "Neden X?", "Sebebi?"
    Temporal,   // "Ne zaman?", "Hangi tarihte?"
    Numeric,    // "Kaç?", "Ne kadar?", "Tutarı?"
    Person,     // "Kim?", "Kisi?"
    Location,   // "Nerede?", "Yeri?"
    Selection,  // "Hangi?", " hangisi?"
    List,       // "Neler?", "Liste?", "Örnekler?"
}

#[derive(Debug, Clone)]
pub struct QuestionAnalysis {
    pub qtype: QuestionType,
    pub entities: Vec<Entity>,
    pub expanded_terms: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum Entity {
    Date(String),
    Number(f64, Option<String>), // value, unit
    Percentage(f64),
    Email(String),
    Url(String),
    Code(String),
    Time(String),
}

// ---------------------------------------------------------------------------
// Ana API: sorguyu analiz et ve genislet.
// ---------------------------------------------------------------------------
pub fn analyze(query: &str) -> QuestionAnalysis {
    let qtype = detect_question_type(query);
    let entities = extract_entities(query);
    let expanded_terms = expand_query(query);

    QuestionAnalysis {
        qtype,
        entities,
        expanded_terms,
    }
}

// ---------------------------------------------------------------------------
// Soru tipi tespiti
// ---------------------------------------------------------------------------
pub fn detect_question_type(query: &str) -> QuestionType {
    let q = tr_lower(query);

    if q.contains("ne zaman") || q.contains("hangi tarih") || q.contains("tarihinde") {
        return QuestionType::Temporal;
    }
    if q.contains("neden") || q.contains("sebebi") || q.contains("nedeni") || q.contains("neden?") {
        return QuestionType::Causal;
    }
    if q.contains("nasıl") || q.contains("nasil") || q.contains("yapılır") || q.contains("yapilir") {
        return QuestionType::Procedure;
    }
    if q.contains("kim") || q.contains("kisi") || q.contains("kişi") {
        return QuestionType::Person;
    }
    if q.contains("nerede") || q.contains("yeri") || q.contains("konum") {
        return QuestionType::Location;
    }
    if q.contains("kaç") || q.contains("kac") || q.contains("ne kadar") || q.contains("miktar") {
        return QuestionType::Numeric;
    }
    if q.contains("hangi") || q.contains("hangisi") {
        return QuestionType::Selection;
    }
    if q.contains("neler") || q.contains("liste") || q.contains("örnek") || q.contains("ornek") {
        return QuestionType::List;
    }
    if q.contains("nedir") || q.contains("tanım") || q.contains("tanim") || q.contains("aciklama") {
        return QuestionType::Definition;
    }

    QuestionType::Unknown
}

// ---------------------------------------------------------------------------
// Varlik (entity) cikarimi: model gerektirmeyen regex tabanli.
// ---------------------------------------------------------------------------
pub fn extract_entities(query: &str) -> Vec<Entity> {
    let mut out = Vec::new();
    let q = query;

    // Tarih: dd.mm.yyyy, dd/mm/yyyy, dd-mm-yyyy
    static DATE_RE: once_cell::sync::Lazy<regex::Regex> =
        once_cell::sync::Lazy::new(|| regex::Regex::new(r"\b\d{1,2}[./\-]\d{1,2}[./\-]\d{2,4}\b").unwrap());
    for m in DATE_RE.find_iter(q) {
        out.push(Entity::Date(m.as_str().to_string()));
    }

    // Zaman: HH:MM, HH:MM:SS
    static TIME_RE: once_cell::sync::Lazy<regex::Regex> =
        once_cell::sync::Lazy::new(|| regex::Regex::new(r"\b\d{1,2}:\d{2}(:\d{2})?\b").unwrap());
    for m in TIME_RE.find_iter(q) {
        out.push(Entity::Time(m.as_str().to_string()));
    }

    // Yüzde: %12, 12%
    static PCT_RE: once_cell::sync::Lazy<regex::Regex> =
        once_cell::sync::Lazy::new(|| regex::Regex::new(r"\b(\d+[.,]?\d*)\s*%|%\s*(\d+[.,]?\d*)\b").unwrap());
    for m in PCT_RE.find_iter(q) {
        let s = m.as_str().replace('%', "").replace(',', ".").trim().to_string();
        if let Ok(v) = s.parse::<f64>() {
            out.push(Entity::Percentage(v));
        }
    }

    // Sayi + birim: 250 saat, 500 km, 1.5 kg, 100 TL
    static NUM_UNIT_RE: once_cell::sync::Lazy<regex::Regex> =
        once_cell::sync::Lazy::new(|| {
            regex::Regex::new(r"\b(\d+[.,]?\d*)\s*(saat|dakika|gün|hafta|ay|yıl|yil|km|m|kg|g|mb|gb|tl|usd|eur|adet|miktar|oran|derece)?\b")
                .unwrap()
        });
    for m in NUM_UNIT_RE.find_iter(q) {
        let s = m.as_str().replace(',', ".").trim().to_string();
        let parts: Vec<&str> = s.split_whitespace().collect();
        if let Some(num_str) = parts.first() {
            if let Ok(v) = num_str.parse::<f64>() {
                let unit = parts.get(1).map(|u| u.to_string());
                out.push(Entity::Number(v, unit));
            }
        }
    }

    // E-posta
    static EMAIL_RE: once_cell::sync::Lazy<regex::Regex> =
        once_cell::sync::Lazy::new(|| regex::Regex::new(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b").unwrap());
    for m in EMAIL_RE.find_iter(q) {
        out.push(Entity::Email(m.as_str().to_string()));
    }

    // URL
    static URL_RE: once_cell::sync::Lazy<regex::Regex> =
        once_cell::sync::Lazy::new(|| regex::Regex::new(r"\bhttps?://[^\s,)\]]+\b").unwrap());
    for m in URL_RE.find_iter(q) {
        out.push(Entity::Url(m.as_str().to_string()));
    }

    // Kod benzeri: MADDE 7, MADDE 12.3, FIG 2.1, ABC123
    static CODE_RE: once_cell::sync::Lazy<regex::Regex> =
        once_cell::sync::Lazy::new(|| regex::Regex::new(r"\b([A-ZÇĞİÖŞÜ]{2,}[\s\-]?\d+[\.\d]*)\b").unwrap());
    for m in CODE_RE.find_iter(q) {
        out.push(Entity::Code(m.as_str().to_string()));
    }

    out
}

// ---------------------------------------------------------------------------
// Sorgu genisletme: es anlamli kelimeleri ekle.
// ---------------------------------------------------------------------------
pub fn expand_query(query: &str) -> Vec<String> {
    let terms = tokenize_for_search(query);
    let mut expanded = Vec::new();
    let mut seen = HashSet::new();

    for t in &terms {
        if seen.insert(t.clone()) {
            expanded.push(t.clone());
        }
        // Sozlukte ara
        for (key, synonyms) in SYNONYMS {
            if *key == *t || synonyms.contains(&t.as_str()) {
                for syn in *synonyms {
                    if syn != key && !seen.contains(*syn) {
                        seen.insert(syn.to_string());
                        expanded.push(syn.to_string());
                    }
                }
                // Anahtar kelimeyi de ekle
                if key != t && !seen.contains(*key) {
                    seen.insert(key.to_string());
                    expanded.push(key.to_string());
                }
            }
        }
    }

    expanded
}

// ---------------------------------------------------------------------------
// Yardimci: Turkce kucuk harf (text.rs'den re-export).
// ---------------------------------------------------------------------------
fn tr_lower(s: &str) -> String {
    crate::text::tr_lower(s)
}

// ---------------------------------------------------------------------------
// Testler
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_temporal_question() {
        assert_eq!(detect_question_type("Ne zaman yapılır?"), QuestionType::Temporal);
    }

    #[test]
    fn detects_numeric_question() {
        assert_eq!(detect_question_type("Kaç saatte bir?"), QuestionType::Numeric);
    }

    #[test]
    fn extracts_date_entity() {
        let ents = extract_entities("Tarih 25.12.2024");
        assert!(ents.iter().any(|e| matches!(e, Entity::Date(_))));
    }

    #[test]
    fn extracts_percentage_entity() {
        let ents = extract_entities("Oran %15");
        assert!(ents.iter().any(|e| matches!(e, Entity::Percentage(15.0))));
    }

    #[test]
    fn expands_synonyms() {
        let terms = expand_query("yapay zeka modeli test edildi");
        assert!(terms.contains(&"ai".to_string()) || terms.contains(&"yapay-zeka".to_string()));
        assert!(terms.contains(&"ml".to_string()) || terms.contains(&"test".to_string()));
    }
}
