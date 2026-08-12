//! LLM'siz cikarimsal (extractive) cevap uretici.
//!
//! LLM servisi kapaliysa veya hata verirse sistem tamamen durmaz: en alakali
//! cumleler kaynaklariyla birlikte dogrudan alintılanır. Uretilen metin
//! belgeden birebir kopya oldugu icin *tanim geregi* halusinasyon icermez;
//! karsiliginda akicilik kaybedilir. Cevap bu modda acikca isaretlenir.

use dq_core::text::{split_sentences, tokenize_for_search};
use dq_core::{Lang, ScoredChunk};
use std::collections::HashSet;

pub struct ExtractiveAnswer {
    pub text: String,
    /// Kullanilan kaynaklarin 1 tabanli numaralari.
    pub used_markers: Vec<usize>,
}

/// Sorgu terimleriyle en cok ortusen cumleleri secer.
pub fn answer(question: &str, chunks: &[ScoredChunk], lang: Lang, max_sentences: usize) -> ExtractiveAnswer {
    let q_terms: HashSet<String> = tokenize_for_search(question)
        .into_iter()
        .filter(|t| t.chars().count() >= 3)
        .collect();

    let mut scored: Vec<(f32, usize, String)> = Vec::new();
    for (i, sc) in chunks.iter().enumerate() {
        for sentence in split_sentences(&sc.chunk.text) {
            if sentence.chars().count() < 25 {
                continue;
            }
            let s_terms: HashSet<String> = tokenize_for_search(&sentence).into_iter().collect();
            let overlap = q_terms.intersection(&s_terms).count() as f32;
            if overlap == 0.0 {
                continue;
            }
            // Kaynak siralamasi da hesaba katilir; ust siradaki chunk'lardan
            // gelen cumleler oncelenir.
            let rank_bonus = 1.0 / (1.0 + i as f32);
            let coverage = overlap / q_terms.len().max(1) as f32;
            scored.push((coverage + 0.35 * rank_bonus, i + 1, sentence));
        }
    }

    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    scored.dedup_by(|a, b| a.2 == b.2);
    scored.truncate(max_sentences);

    if scored.is_empty() {
        return ExtractiveAnswer {
            text: crate::prompts::refusal(lang).to_string(),
            used_markers: Vec::new(),
        };
    }

    // Okunabilirlik icin kaynak sirasina gore diz.
    scored.sort_by_key(|(_, marker, _)| *marker);

    let mut used = Vec::new();
    let mut lines = Vec::new();
    for (_, marker, sentence) in scored {
        if !used.contains(&marker) {
            used.push(marker);
        }
        lines.push(format!("{} [{}]", sentence.trim(), marker));
    }

    let prefix = match lang {
        Lang::En => "Relevant excerpts found in the documents:",
        _ => "Belgelerde bulunan ilgili bölümler:",
    };

    ExtractiveAnswer {
        text: format!("{prefix}\n\n- {}", lines.join("\n- ")),
        used_markers: used,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dq_core::{Chunk, Classification};
    use uuid::Uuid;

    fn sc(text: &str) -> ScoredChunk {
        ScoredChunk {
            chunk: Chunk {
                id: Uuid::new_v4(),
                doc_id: Uuid::new_v4(),
                ordinal: 0,
                page_from: 3,
                page_to: 3,
                text: text.into(),
                heading_path: None,
                token_estimate: 20,
                lang: Lang::Tr,
                classification: Classification::Unclassified,
                confidence: 1.0,
            },
            score: 0.8,
            dense_score: Some(0.8),
            sparse_score: None,
            rerank_score: None,
            doc_filename: "bakim.pdf".into(),
        }
    }

    #[test]
    fn picks_sentence_matching_query() {
        let chunks = vec![sc("Genel hükümler bu bölümde açıklanmıştır. Motorun yağ değişimi 250 saatte bir yapılmalıdır.")];
        let a = answer("yağ değişimi kaç saatte bir", &chunks, Lang::Tr, 3);
        assert!(a.text.contains("250 saatte"));
        assert!(a.text.contains("[1]"));
        assert_eq!(a.used_markers, vec![1]);
    }

    #[test]
    fn refuses_when_nothing_matches() {
        let chunks = vec![sc("Bu bölümde personel izin süreleri anlatılır ve tablolar verilir.")];
        let a = answer("uçağın azami irtifası", &chunks, Lang::Tr, 3);
        assert!(a.used_markers.is_empty());
    }
}
