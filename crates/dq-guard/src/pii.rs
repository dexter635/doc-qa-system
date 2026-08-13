//! Kisisel veri (PII) tespiti ve maskeleme.
//!
//! Regex tek basina cok fazla yanlis pozitif uretir: 11 haneli her sayi TC
//! kimlik numarasi degildir. Bu yuzden desen eslesmesinden sonra ilgili
//! *dogrulama algoritmasi* (TCKN saglama, Luhn, IBAN mod-97) uygulanir.

use once_cell::sync::Lazy;
use regex::Regex;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PiiKind {
    Tckn,
    Iban,
    CreditCard,
    Email,
    Phone,
    IpAddress,
}

impl PiiKind {
    pub fn label(&self) -> &'static str {
        match self {
            PiiKind::Tckn => "TC Kimlik No",
            PiiKind::Iban => "IBAN",
            PiiKind::CreditCard => "Kart No",
            PiiKind::Email => "E-posta",
            PiiKind::Phone => "Telefon",
            PiiKind::IpAddress => "IP Adresi",
        }
    }

    fn mask(&self) -> &'static str {
        match self {
            PiiKind::Tckn => "[TCKN-MASKELENDI]",
            PiiKind::Iban => "[IBAN-MASKELENDI]",
            PiiKind::CreditCard => "[KART-MASKELENDI]",
            PiiKind::Email => "[EPOSTA-MASKELENDI]",
            PiiKind::Phone => "[TELEFON-MASKELENDI]",
            PiiKind::IpAddress => "[IP-MASKELENDI]",
        }
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct PiiMatch {
    pub kind: PiiKind,
    pub start: usize,
    pub end: usize,
}

static TCKN_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\b[1-9]\d{10}\b").unwrap());
static IBAN_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\b[A-Z]{2}\d{2}[ ]?(?:[A-Z0-9]{4}[ ]?){2,7}[A-Z0-9]{1,4}\b").unwrap()
});
static CARD_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\b(?:\d{4}[ -]?){3}\d{4}\b").unwrap());
static EMAIL_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b").unwrap());
static PHONE_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?:\+90|0)?[ ]?\(?5\d{2}\)?[ ]?\d{3}[ -]?\d{2}[ -]?\d{2}\b").unwrap()
});
static IP_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\b(?:\d{1,3}\.){3}\d{1,3}\b").unwrap());

/// TC Kimlik Numarasi saglama algoritmasi.
pub fn is_valid_tckn(s: &str) -> bool {
    let d: Vec<u32> = s.chars().filter_map(|c| c.to_digit(10)).collect();
    if d.len() != 11 || d[0] == 0 {
        return false;
    }
    let odd: u32 = d[0] + d[2] + d[4] + d[6] + d[8];
    let even: u32 = d[1] + d[3] + d[5] + d[7];
    let d10 = ((odd * 7) as i64 - even as i64).rem_euclid(10) as u32;
    if d10 != d[9] {
        return false;
    }
    let sum10: u32 = d[..10].iter().sum();
    sum10 % 10 == d[10]
}

/// Luhn saglamasi (kredi karti numaralari).
pub fn is_valid_luhn(s: &str) -> bool {
    let digits: Vec<u32> = s.chars().filter_map(|c| c.to_digit(10)).collect();
    if digits.len() < 13 || digits.len() > 19 {
        return false;
    }
    let mut sum = 0u32;
    for (i, d) in digits.iter().rev().enumerate() {
        let mut v = *d;
        if i % 2 == 1 {
            v *= 2;
            if v > 9 {
                v -= 9;
            }
        }
        sum += v;
    }
    sum % 10 == 0
}

/// IBAN mod-97 saglamasi (ISO 13616).
pub fn is_valid_iban(s: &str) -> bool {
    let cleaned: String = s.chars().filter(|c| !c.is_whitespace()).collect();
    if cleaned.len() < 15 || cleaned.len() > 34 {
        return false;
    }
    let (head, tail) = cleaned.split_at(4);
    let rearranged = format!("{tail}{head}");
    let mut remainder = 0u32;
    for c in rearranged.chars() {
        let value = if c.is_ascii_digit() {
            c.to_digit(10).unwrap()
        } else if c.is_ascii_alphabetic() {
            c.to_ascii_uppercase() as u32 - 'A' as u32 + 10
        } else {
            return false;
        };
        remainder = if value > 9 {
            (remainder * 100 + value) % 97
        } else {
            (remainder * 10 + value) % 97
        };
    }
    remainder == 1
}

fn is_valid_ip(s: &str) -> bool {
    s.split('.')
        .filter_map(|p| p.parse::<u32>().ok())
        .filter(|n| *n <= 255)
        .count()
        == 4
}

/// Metindeki dogrulanmis PII eslesmelerini dondurur.
pub fn detect(text: &str) -> Vec<PiiMatch> {
    let mut out: Vec<PiiMatch> = Vec::new();

    let push = |kind: PiiKind, start: usize, end: usize, out: &mut Vec<PiiMatch>| {
        // Ic ice eslesmeleri (IBAN icindeki kart deseni gibi) tekilleştir.
        if out.iter().any(|m| start < m.end && m.start < end) {
            return;
        }
        out.push(PiiMatch { kind, start, end });
    };

    for m in IBAN_RE.find_iter(text) {
        if is_valid_iban(m.as_str()) {
            push(PiiKind::Iban, m.start(), m.end(), &mut out);
        }
    }
    for m in CARD_RE.find_iter(text) {
        if is_valid_luhn(m.as_str()) {
            push(PiiKind::CreditCard, m.start(), m.end(), &mut out);
        }
    }
    for m in TCKN_RE.find_iter(text) {
        if is_valid_tckn(m.as_str()) {
            push(PiiKind::Tckn, m.start(), m.end(), &mut out);
        }
    }
    for m in EMAIL_RE.find_iter(text) {
        push(PiiKind::Email, m.start(), m.end(), &mut out);
    }
    for m in PHONE_RE.find_iter(text) {
        push(PiiKind::Phone, m.start(), m.end(), &mut out);
    }
    for m in IP_RE.find_iter(text) {
        if is_valid_ip(m.as_str()) {
            push(PiiKind::IpAddress, m.start(), m.end(), &mut out);
        }
    }

    out.sort_by_key(|m| m.start);
    out
}

/// Tespit edilen PII'yi maskeler; maskelenen turleri de dondurur.
pub fn redact(text: &str) -> (String, Vec<PiiKind>) {
    let matches = detect(text);
    if matches.is_empty() {
        return (text.to_string(), Vec::new());
    }
    let mut out = String::with_capacity(text.len());
    let mut cursor = 0usize;
    let mut kinds = Vec::new();
    for m in &matches {
        if m.start < cursor {
            continue;
        }
        out.push_str(&text[cursor..m.start]);
        out.push_str(m.kind.mask());
        cursor = m.end;
        if !kinds.contains(&m.kind) {
            kinds.push(m.kind);
        }
    }
    out.push_str(&text[cursor..]);
    (out, kinds)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_tckn_checksum() {
        assert!(is_valid_tckn("10000000146"));
        assert!(!is_valid_tckn("12345678901"));
        assert!(!is_valid_tckn("00000000000"));
    }

    #[test]
    fn validates_luhn() {
        assert!(is_valid_luhn("4111 1111 1111 1111"));
        assert!(!is_valid_luhn("4111 1111 1111 1112"));
    }

    #[test]
    fn validates_iban() {
        assert!(is_valid_iban("TR330006100519786457841326"));
        assert!(!is_valid_iban("TR330006100519786457841327"));
    }

    #[test]
    fn random_11_digit_number_is_not_flagged() {
        // Yanlis pozitif kontrolu: parca numarasi TCKN sanilmamalidir.
        let found = detect("Parça numarası 12345678901 olarak kayıtlıdır.");
        assert!(found.iter().all(|m| m.kind != PiiKind::Tckn));
    }

    #[test]
    fn redacts_email_and_tckn() {
        let (out, kinds) = redact("İlgili: ahmet@ornek.com, TCKN 10000000146");
        assert!(out.contains("[EPOSTA-MASKELENDI]"));
        assert!(out.contains("[TCKN-MASKELENDI]"));
        assert_eq!(kinds.len(), 2);
    }
}
