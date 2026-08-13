//! BM25 seyrek (sparse) arama.
//!
//! Yalnizca gomme tabanli arama, belge numarasi / parca kodu / kisaltma gibi
//! *tam eslesme* gerektiren sorgularda basarisiz olur ("MADDE 7 nedir?").
//! BM25 bu bosugu kapatir ve hibrit fuzyonun ikinci ayagini olusturur.

use std::collections::HashMap;

use dq_core::text::{fold_diacritics, tokenize_for_search, tr_lower};

const K1: f32 = 1.4;
const B: f32 = 0.72;

#[derive(Default)]
pub struct Bm25Index {
    /// terim -> (satir, terim frekansi)
    postings: HashMap<String, Vec<(usize, u32)>>,
    doc_len: Vec<f32>,
    avg_len: f32,
}

impl Bm25Index {
    pub fn new() -> Self {
        Self::default()
    }

    /// Satirlarin eklenme sirasi vektor indeksi ile ayni olmalidir.
    pub fn push(&mut self, text: &str, heading: Option<&str>) {
        let row = self.doc_len.len();
        let mut tokens = tokenize_for_search(text);
        if let Some(h) = heading {
            // Baslik terimleri iki kez sayilir: baslikta gecen terim daha ayirt edici.
            let mut ht = tokenize_for_search(h);
            tokens.append(&mut ht.clone());
            tokens.append(&mut ht);
        }
        let len = tokens.len() as f32;
        let mut tf: HashMap<String, u32> = HashMap::new();
        for t in tokens {
            *tf.entry(t).or_insert(0) += 1;
        }
        for (term, count) in tf {
            self.postings.entry(term).or_default().push((row, count));
        }
        self.doc_len.push(len);
        let total: f32 = self.doc_len.iter().sum();
        self.avg_len = if self.doc_len.is_empty() {
            0.0
        } else {
            total / self.doc_len.len() as f32
        };
    }

    pub fn len(&self) -> usize {
        self.doc_len.len()
    }

    pub fn is_empty(&self) -> bool {
        self.doc_len.is_empty()
    }

    pub fn clear(&mut self) {
        self.postings.clear();
        self.doc_len.clear();
        self.avg_len = 0.0;
    }

    pub fn search(
        &self,
        query: &str,
        k: usize,
        allow: &dyn Fn(usize) -> bool,
    ) -> Vec<(usize, f32)> {
        if self.doc_len.is_empty() || k == 0 {
            return Vec::new();
        }
        let n = self.doc_len.len() as f32;
        let mut scores: HashMap<usize, f32> = HashMap::new();

        for term in query_terms(query) {
            let Some(posting) = self.postings.get(&term) else {
                continue;
            };
            let df = posting.len() as f32;
            // Robertson/Sparck-Jones IDF; negatif olmamasi icin +1.0 ile kaydirilir.
            let idf = (((n - df + 0.5) / (df + 0.5)) + 1.0).ln();
            for (row, tf) in posting {
                if !allow(*row) {
                    continue;
                }
                let dl = self.doc_len[*row];
                let denom = *tf as f32 + K1 * (1.0 - B + B * dl / self.avg_len.max(1.0));
                let contrib = idf * (*tf as f32 * (K1 + 1.0)) / denom.max(f32::EPSILON);
                *scores.entry(*row).or_insert(0.0) += contrib;
            }
        }

        let mut out: Vec<(usize, f32)> = scores.into_iter().collect();
        out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        out.truncate(k);
        out
    }
}

/// Sorgu terimleri: hem sadelestirilmis hem de kisa ekleri atilmis bicimler.
///
/// Turkce sondan eklemeli oldugu icin "bakımı" ve "bakım" farkli token olur.
/// Tam bir stemmer yerine, uzun kelimelerde son ekleri kirpan hafif bir
/// yaklasim kullanilir; yanlis pozitif maliyeti fuzyon tarafindan sogurulur.
fn query_terms(query: &str) -> Vec<String> {
    let base = tokenize_for_search(query);
    let mut out: Vec<String> = Vec::with_capacity(base.len() * 2);
    for t in base {
        if t.chars().count() >= 7 {
            let trimmed: String = t.chars().take(t.chars().count() - 2).collect();
            out.push(trimmed);
        }
        out.push(t);
    }
    out.dedup();
    out
}

/// Metinde sorgu terimlerinin gectigi yerleri isaretlemek icin yardimci
/// (arayuzde vurgulama ve snippet secimi).
pub fn highlight_terms(query: &str) -> Vec<String> {
    tokenize_for_search(query)
        .into_iter()
        .filter(|t| t.chars().count() >= 3)
        .collect()
}

/// Bir metnin sorgu terimlerini en yogun iceren bolumunu dondurur.
pub fn best_snippet(text: &str, query: &str, max_chars: usize) -> String {
    let terms = highlight_terms(query);
    if terms.is_empty() || text.chars().count() <= max_chars {
        return dq_core::text::truncate_chars(text, max_chars);
    }
    let hay = fold_diacritics(&tr_lower(text));
    let chars: Vec<char> = text.chars().collect();
    let hay_chars: Vec<char> = hay.chars().collect();
    let window = max_chars.min(chars.len());

    let mut best_start = 0usize;
    let mut best_score = -1i32;
    let step = (window / 4).max(1);
    let mut start = 0usize;
    while start + window <= hay_chars.len() {
        let slice: String = hay_chars[start..start + window].iter().collect();
        let score = terms.iter().filter(|t| slice.contains(t.as_str())).count() as i32;
        if score > best_score {
            best_score = score;
            best_start = start;
        }
        start += step;
    }
    let end = (best_start + window).min(chars.len());
    let mut out: String = chars[best_start..end].iter().collect();
    if best_start > 0 {
        out.insert(0, '…');
    }
    if end < chars.len() {
        out.push('…');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_term_match_ranks_first() {
        let mut idx = Bm25Index::new();
        idx.push("Motorun yağ değişimi 250 saatte bir yapılır.", None);
        idx.push("MADDE 7 - Personel güvenlik belgesi zorunludur.", None);
        idx.push("Uçuş öncesi kontrol listesi uygulanır.", None);
        let hits = idx.search("MADDE 7", 3, &|_| true);
        assert_eq!(hits[0].0, 1);
    }

    #[test]
    fn filter_excludes_rows() {
        let mut idx = Bm25Index::new();
        idx.push("gizli plan", None);
        idx.push("açık plan", None);
        let hits = idx.search("plan", 5, &|i| i == 1);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, 1);
    }

    #[test]
    fn snippet_centers_on_query() {
        let text = format!(
            "{} önemli bilgi: yağ değişimi 250 saat. {}",
            "a".repeat(300),
            "b".repeat(300)
        );
        let s = best_snippet(&text, "yağ değişimi", 120);
        assert!(s.contains("yağ değişimi"), "snippet: {s}");
    }
}
