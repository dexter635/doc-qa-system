use once_cell::sync::Lazy;
use regex::Regex;
use unicode_normalization::UnicodeNormalization;

use crate::types::Lang;

/// Turkce'ye duyarli kucuk harfe cevirme.
///
/// `str::to_lowercase` "I" harfini "i" yapar; Turkce'de dogrusu "ı"dir.
/// Arama ve normalizasyon tutarliligi icin bu fonksiyon kullanilmalidir.
pub fn tr_lower(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            'I' => out.push('ı'),
            'İ' => out.push('i'),
            _ => out.extend(ch.to_lowercase()),
        }
    }
    out
}

/// ASCII 'I' -> 'i' esleyen, Turkce harfleri (ç, ğ, ı, ö, ş, ü) oldugu gibi
/// koruyan kucuk harfe cevirme.
///
/// `tr_lower`'dan farki: karma TR/EN metinlerde (ör. guardrail kural
/// eslestirme) ASCII "Ignore" gibi kelimeleri Turkce kurala gore "ıgnore"
/// yaparak regex eslesmesini kirmaz. Duz `str::to_lowercase` ile ayni
/// davranir, yalnizca 'İ' (noktali buyuk I) tek karakter 'i'ye indirgenir.
pub fn casefold(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            'İ' => out.push('i'),
            _ => out.extend(ch.to_lowercase()),
        }
    }
    out
}

/// Turkce'ye duyarli buyuk harfe cevirme.
pub fn tr_upper(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            'i' => out.push('İ'),
            'ı' => out.push('I'),
            _ => out.extend(ch.to_uppercase()),
        }
    }
    out
}

/// Aksanlari sadelestirir: "şğçöüı" -> "sgcoui".
///
/// OCR ciktilarinda Turkce karakterler sik bozuldugu icin BM25 tarafinda
/// hem asil hem de sadelestirilmis bicim indekslenir.
pub fn fold_diacritics(s: &str) -> String {
    s.nfd()
        .filter(|c| !unicode_normalization::char::is_combining_mark(*c))
        .map(|c| match c {
            'ı' => 'i',
            'İ' => 'I',
            'ğ' => 'g',
            'Ğ' => 'G',
            'ş' => 's',
            'Ş' => 'S',
            _ => c,
        })
        .collect()
}

static WS_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"[ \t\x0b\x0c\r]+").unwrap());
static MULTI_NL_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\n{3,}").unwrap());
static HYPHEN_BREAK_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(\p{L})[-\u{00ad}]\n(\p{Ll})").unwrap());
static SOFT_WRAP_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(\p{Ll},?)\n(\p{Ll})").unwrap());

/// PDF/OCR ciktisini temizler: satir sonu tirelerini birlestirir, yumusak
/// satir kirilmalarini kaldirir, fazla bosluklari sadelestirir.
pub fn clean_extracted_text(raw: &str) -> String {
    let s = raw.replace('\u{0000}', " ").replace("\r\n", "\n");
    let s = s.replace('\u{fb01}', "fi").replace('\u{fb02}', "fl");
    let s = HYPHEN_BREAK_RE.replace_all(&s, "$1$2").into_owned();
    let s = SOFT_WRAP_RE.replace_all(&s, "$1 $2").into_owned();
    let s = WS_RE.replace_all(&s, " ").into_owned();
    let s = MULTI_NL_RE.replace_all(&s, "\n\n").into_owned();
    s.lines()
        .map(|l| l.trim_end())
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string()
}

/// Yaklasik token sayisi. Gercek tokenizer yerine, hem TR hem EN icin
/// olculmus ~3.6 karakter/token orani kullanilir (butce planlamasi icin yeterli).
pub fn estimate_tokens(s: &str) -> usize {
    let chars = s.chars().count();
    if chars == 0 {
        return 0;
    }
    ((chars as f32) / 3.6).ceil() as usize
}

/// Cumle sonu sayilmamasi gereken yaygin kisaltmalar.
const ABBREVIATIONS: &[&str] = &[
    "vb", "vs", "bkz", "örn", "orn", "dr", "prof", "doç", "doc", "md", "no", "nr", "sy", "tbl",
    "şek", "sek", "fig", "eq", "etc", "e.g", "i.e", "mr", "mrs", "vol", "pp", "ref",
];

/// Metni cumlelere ayirir. Kaynak dogrulama (groundedness) cumle bazinda
/// yapildigi icin bu bolme kalitesi dogrudan guardrail'i etkiler.
///
/// Rust `regex` lookaround desteklemedigi icin elle taranir; ayrica bu yol
/// kisaltma ve ondalik sayi istisnalarini da ele almamizi saglar.
pub fn split_sentences(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut out: Vec<String> = Vec::new();
    let mut buf = String::new();
    let mut i = 0usize;

    while i < chars.len() {
        let c = chars[i];
        buf.push(c);

        let is_terminator = matches!(c, '.' | '!' | '?' | '…');
        let is_para_break = c == '\n' && chars.get(i + 1).is_some_and(|n| *n == '\n');

        if is_para_break {
            push_sentence(&mut out, &mut buf);
            i += 1;
            while chars.get(i).is_some_and(|n| n.is_whitespace()) {
                i += 1;
            }
            continue;
        }

        if is_terminator {
            // "3.14" veya "Madde 4.2" gibi sayilarda bolme yapma.
            let prev_digit = i > 0 && chars[i - 1].is_ascii_digit();
            let next_digit = chars.get(i + 1).is_some_and(|n| n.is_ascii_digit());
            let after_ws = chars.get(i + 1).is_none_or(|n| n.is_whitespace());
            let starts_new = chars[i + 1..]
                .iter()
                .find(|n| !n.is_whitespace())
                .is_none_or(|n| n.is_uppercase() || n.is_ascii_digit() || *n == '-' || *n == '•');

            if !(prev_digit && next_digit)
                && after_ws
                && starts_new
                && !ends_with_abbreviation(&buf)
            {
                push_sentence(&mut out, &mut buf);
                i += 1;
                while chars.get(i).is_some_and(|n| n.is_whitespace()) {
                    i += 1;
                }
                continue;
            }
        }
        i += 1;
    }
    push_sentence(&mut out, &mut buf);
    out
}

fn push_sentence(out: &mut Vec<String>, buf: &mut String) {
    let t = buf.trim().to_string();
    buf.clear();
    if t.is_empty() {
        return;
    }
    if t.chars().count() < 3 {
        if let Some(last) = out.last_mut() {
            last.push(' ');
            last.push_str(&t);
            return;
        }
    }
    out.push(t);
}

fn ends_with_abbreviation(buf: &str) -> bool {
    let trimmed = buf.trim_end_matches('.');
    let last_word: String = trimmed
        .chars()
        .rev()
        .take_while(|c| c.is_alphanumeric() || *c == '.')
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    let lw = tr_lower(&last_word);
    ABBREVIATIONS.contains(&lw.as_str())
}

const TR_STOPWORDS: &[&str] = &[
    "ve",
    "ile",
    "bir",
    "bu",
    "için",
    "olarak",
    "daha",
    "gibi",
    "olan",
    "veya",
    "her",
    "ancak",
    "ise",
    "de",
    "da",
    "ki",
    "en",
    "çok",
    "sonra",
    "kadar",
    "tarafından",
    "üzere",
    "göre",
];
const EN_STOPWORDS: &[&str] = &[
    "the", "and", "for", "with", "that", "this", "from", "are", "was", "have", "has", "not",
    "which", "will", "shall", "been", "were", "their", "these", "such",
];

/// Basit ama guvenilir dil tespiti: stopword sayimi + Turkce'ye ozgu karakterler.
pub fn detect_lang(text: &str) -> Lang {
    let lower = tr_lower(text);
    let words: Vec<&str> = lower
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .take(600)
        .collect();
    if words.len() < 3 {
        return Lang::Unknown;
    }
    let mut tr = 0usize;
    let mut en = 0usize;
    for w in &words {
        if TR_STOPWORDS.contains(w) {
            tr += 1;
        }
        if EN_STOPWORDS.contains(w) {
            en += 1;
        }
    }
    let tr_chars = lower.chars().filter(|c| "çğıöşü".contains(*c)).count();
    let tr_score = tr as f32 + (tr_chars as f32) * 0.12;
    let en_score = en as f32;
    if tr_score < 1.0 && en_score < 1.0 {
        return Lang::Unknown;
    }
    if tr_score >= en_score {
        Lang::Tr
    } else {
        Lang::En
    }
}

/// UTF-8 sinirlarini bozmadan kirpar.
pub fn truncate_chars(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        return s.to_string();
    }
    let mut out: String = s.chars().take(max_chars.saturating_sub(1)).collect();
    out.push('…');
    out
}

/// Kelime bazli, tekrarlari sadelestirilmis token listesi (BM25 icin).
pub fn tokenize_for_search(text: &str) -> Vec<String> {
    fold_diacritics(&tr_lower(text))
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() >= 2 && w.chars().count() <= 40)
        .map(|w| w.to_string())
        .collect()
}

/// Karakter n-gram Jaccard benzerligi (0..1). Model gerektirmeyen, hizli bir
/// metin ortusme olcusudur; groundedness dogrulamasinda kullanilir.
pub fn ngram_similarity(a: &str, b: &str, n: usize) -> f32 {
    let ga = char_ngrams(a, n);
    let gb = char_ngrams(b, n);
    if ga.is_empty() || gb.is_empty() {
        return 0.0;
    }
    let inter = ga.intersection(&gb).count() as f32;
    let union = ga.union(&gb).count() as f32;
    if union == 0.0 {
        0.0
    } else {
        inter / union
    }
}

/// `a` icindeki n-gram'larin ne kadarinin `b` icinde bulundugu (0..1).
/// Asimetriktir: kisa bir cumlenin uzun bir baglamda desteklenip
/// desteklenmedigini olcmek icin Jaccard'dan daha uygundur.
pub fn containment(a: &str, b: &str, n: usize) -> f32 {
    let ga = char_ngrams(a, n);
    let gb = char_ngrams(b, n);
    if ga.is_empty() {
        return 0.0;
    }
    let inter = ga.intersection(&gb).count() as f32;
    inter / ga.len() as f32
}

fn char_ngrams(s: &str, n: usize) -> std::collections::HashSet<String> {
    let norm = fold_diacritics(&tr_lower(s));
    let cleaned: String = norm
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { ' ' })
        .collect();
    let chars: Vec<char> = WS_RE.replace_all(cleaned.trim(), " ").chars().collect();
    let mut set = std::collections::HashSet::new();
    if chars.len() < n {
        if !chars.is_empty() {
            set.insert(chars.iter().collect());
        }
        return set;
    }
    for w in chars.windows(n) {
        set.insert(w.iter().collect::<String>());
    }
    set
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn turkish_lowercase_is_correct() {
        assert_eq!(tr_lower("İSTANBUL"), "istanbul");
        assert_eq!(tr_lower("IŞIK"), "ışık");
    }

    #[test]
    fn diacritics_are_folded() {
        assert_eq!(fold_diacritics("şğçöüı"), "sgcoui");
    }

    #[test]
    fn hyphenated_line_breaks_are_joined() {
        let cleaned = clean_extracted_text("bakı-\nmında yapılan");
        assert!(cleaned.contains("bakımında"), "got: {cleaned}");
    }

    #[test]
    fn detects_languages() {
        assert_eq!(
            detect_lang("Bu belge için hazırlanan bakım talimatı ve ekleri ile birlikte"),
            Lang::Tr
        );
        assert_eq!(
            detect_lang("The maintenance procedure for this system shall be performed with care"),
            Lang::En
        );
    }

    #[test]
    fn sentences_are_split() {
        let s = split_sentences("Birinci cümle. İkinci cümle! Üçüncü mü?");
        assert_eq!(s.len(), 3);
    }

    #[test]
    fn containment_detects_support() {
        let ctx = "Motorun periyodik bakımı 500 saatte bir yapılır ve yağ değişimi zorunludur.";
        assert!(containment("Periyodik bakım 500 saatte bir yapılır.", ctx, 4) > 0.6);
        assert!(containment("Uçağın azami hızı 900 km/s olarak ölçülmüştür.", ctx, 4) < 0.3);
    }

    #[test]
    fn truncate_respects_utf8() {
        assert_eq!(truncate_chars("çğüöşı", 3).chars().count(), 3);
    }
}
