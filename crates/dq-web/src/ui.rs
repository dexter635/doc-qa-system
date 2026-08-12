//! Kucuk, tekrar kullanilan gorunum yardimcilari (rozet siniflari vb.).

pub fn classification_class(c: &str) -> String {
    format!("badge {}", c.to_ascii_lowercase())
}

pub fn classification_label(c: &str) -> &'static str {
    match c.to_ascii_lowercase().as_str() {
        "unclassified" => "TASNİF DIŞI",
        "restricted" => "HİZMETE ÖZEL",
        "confidential" => "ÖZEL",
        "secret" => "GİZLİ",
        "top_secret" => "ÇOK GİZLİ",
        _ => "BİLİNMİYOR",
    }
}

pub fn status_class(s: &str) -> String {
    format!("badge status-{}", s.to_ascii_lowercase())
}

pub fn status_label(s: &str) -> &'static str {
    match s {
        "pending" => "Bekliyor",
        "processing" => "İşleniyor",
        "ready" => "Hazır",
        "failed" => "Başarısız",
        _ => s_static(),
    }
}

fn s_static() -> &'static str {
    "?"
}

pub fn kind_class(k: &str) -> String {
    format!("badge kind-{}", k.to_ascii_lowercase())
}

pub fn kind_label(k: &str) -> &'static str {
    match k {
        "grounded" => "Kaynaklı Cevap",
        "refused" => "Bilgi Bulunamadı",
        "blocked" => "Engellendi",
        _ => "?",
    }
}

pub fn lang_label(l: &str) -> &'static str {
    match l {
        "tr" => "Türkçe",
        "en" => "İngilizce",
        _ => "Bilinmiyor",
    }
}
