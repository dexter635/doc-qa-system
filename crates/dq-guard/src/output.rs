//! Cikti tarafi guardrail: kaynak dogrulama (groundedness), PII maskeleme,
//! gizlilik damgalama ve dil tutarliligi.
//!
//! Bu katman LLM'e "guven" duymaz: modelin urettigi her cumle, kendisine
//! verilen kaynak metinlerle n-gram ortusmesi uzerinden dogrulanir. Yeterince
//! desteklenmeyen cevaplar otomatik olarak reddedilir; boylece halusinasyon
//! kullaniciya degil, guardrail'e carpar.

use once_cell::sync::Lazy;
use regex::Regex;

use dq_core::config::GuardrailConfig;
use dq_core::text::{containment, detect_lang, split_sentences};
use dq_core::{Answer, AnswerKind, Citation, Classification, Groundedness, Lang, ScoredChunk};
use uuid::Uuid;

static MARKER_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\[(\d+)\]").unwrap());

/// Minimum ortusme esigi: bir cumlenin kaynagi "destekliyor" sayilmasi icin
/// gereken n-gram containment orani. Deneysel olarak secildi; cok yuksek
/// olursa parafraz edilen dogru cevaplar da reddedilir.
const SUPPORT_CONTAINMENT_THRESHOLD: f32 = 0.30;
const NGRAM_N: usize = 4;
/// Bu uzunlugun altindaki cumleler (baslik, "Sonuç:" gibi) dogrulamaya girmez.
const MIN_SENTENCE_CHARS: usize = 12;

pub struct OutputGuard {
    cfg: GuardrailConfig,
}

pub struct OutputResult {
    pub text: String,
    pub kind: AnswerKind,
    pub citations: Vec<Citation>,
    pub groundedness: Groundedness,
    pub classification: Classification,
    pub warnings: Vec<String>,
}

impl OutputGuard {
    pub fn new(cfg: GuardrailConfig) -> Self {
        Self { cfg }
    }

    /// LLM'in urettigi ham metni (veya cikarimsal yedegi) degerlendirir.
    ///
    /// `context_chunks` LLM'e verilen sirayla ayni olmalidir: `[n]` isareti
    /// `context_chunks[n-1]`'e karsilik gelir.
    pub fn evaluate(
        &self,
        raw_answer: &str,
        context_chunks: &[ScoredChunk],
        query_lang: Lang,
    ) -> OutputResult {
        let mut warnings = Vec::new();
        let max_classification = context_chunks
            .iter()
            .map(|c| c.chunk.classification)
            .max()
            .unwrap_or(Classification::Unclassified);

        if context_chunks.is_empty() || raw_answer.trim().is_empty() {
            return self.refuse(query_lang, max_classification, "Baglam bulunamadi.".into());
        }

        let sentences = split_sentences(raw_answer);
        let mut supported = 0usize;
        let mut counted = 0usize;
        let mut unsupported_sentences = Vec::new();
        let mut used_markers: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();

        for sentence in &sentences {
            let clean = MARKER_RE.replace_all(sentence, "").trim().to_string();
            if clean.chars().count() < MIN_SENTENCE_CHARS {
                continue;
            }
            counted += 1;

            let markers: Vec<usize> = MARKER_RE
                .captures_iter(sentence)
                .filter_map(|c| c[1].parse::<usize>().ok())
                .filter(|m| *m >= 1 && *m <= context_chunks.len())
                .collect();

            let best_score = if markers.is_empty() {
                // Kaynak isareti yoksa tum baglama karsi en iyi ortusme aranir;
                // yine de bulunursa "isaretsiz destek" olarak dusuk agirlik verilir.
                context_chunks
                    .iter()
                    .map(|c| containment(&clean, &c.chunk.text, NGRAM_N))
                    .fold(0f32, f32::max)
                    * 0.85
            } else {
                for m in &markers {
                    used_markers.insert(*m);
                }
                markers
                    .iter()
                    .map(|m| containment(&clean, &context_chunks[*m - 1].chunk.text, NGRAM_N))
                    .fold(0f32, f32::max)
            };

            if best_score >= SUPPORT_CONTAINMENT_THRESHOLD {
                supported += 1;
            } else {
                unsupported_sentences.push(sentence.trim().to_string());
            }
        }

        let support_ratio = if counted == 0 {
            0.0
        } else {
            supported as f32 / counted as f32
        };
        let top_score = context_chunks.first().map(|c| c.score).unwrap_or(0.0);

        let passed = counted > 0
            && support_ratio >= self.cfg.min_support_ratio
            && top_score >= self.cfg.min_top_score;

        if !passed {
            let reason = if top_score < self.cfg.min_top_score {
                format!(
                    "En iyi kaynak skoru ({top_score:.2}) esigin ({:.2}) altinda.",
                    self.cfg.min_top_score
                )
            } else {
                format!(
                    "Cevabin yalnizca %{:.0}'i kaynaklarla dogrulanabildi (esik %{:.0}).",
                    support_ratio * 100.0,
                    self.cfg.min_support_ratio * 100.0
                )
            };
            return self.refuse(query_lang, max_classification, reason);
        }

        // Yalnizca fiilen kullanilan (ve dogrulanan) kaynaklar atif olarak dondurulur.
        let citations: Vec<Citation> = if self.cfg.require_citations && !used_markers.is_empty() {
            used_markers
                .into_iter()
                .filter_map(|m| context_chunks.get(m - 1).map(|c| (m, c)))
                .map(|(marker, c)| Citation {
                    marker,
                    doc_id: c.chunk.doc_id,
                    doc_filename: c.doc_filename.clone(),
                    chunk_id: c.chunk.id,
                    page_from: c.chunk.page_from,
                    page_to: c.chunk.page_to,
                    snippet: dq_core::text::truncate_chars(&c.chunk.text, 240),
                    score: c.score,
                })
                .collect()
        } else {
            // Isaretsiz ama destekli cevap: en iyi kaynagi tek atif olarak ekle.
            context_chunks
                .first()
                .map(|c| {
                    vec![Citation {
                        marker: 1,
                        doc_id: c.chunk.doc_id,
                        doc_filename: c.doc_filename.clone(),
                        chunk_id: c.chunk.id,
                        page_from: c.chunk.page_from,
                        page_to: c.chunk.page_to,
                        snippet: dq_core::text::truncate_chars(&c.chunk.text, 240),
                        score: c.score,
                    }]
                })
                .unwrap_or_default()
        };

        if self.cfg.require_citations && citations.is_empty() {
            warnings.push("Cevapta kaynak atfi bulunamadi.".into());
        }

        let mut text = raw_answer.trim().to_string();
        if self.cfg.redact_pii_in_answer {
            let (redacted, kinds) = crate::pii::redact(&text);
            if !kinds.is_empty() {
                warnings.push(format!(
                    "Ciktida kisisel veri maskelendi ({}).",
                    kinds
                        .iter()
                        .map(|k| k.label())
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
                text = redacted;
            }
        }

        if self.cfg.enforce_language_match {
            let answer_lang = detect_lang(&text);
            if query_lang != Lang::Unknown
                && answer_lang != Lang::Unknown
                && answer_lang != query_lang
            {
                warnings.push("Cevap dili soru diliyle tam eslesmeyebilir.".into());
            }
        }

        let classification = if self.cfg.stamp_classification {
            citations
                .iter()
                .map(|c| context_chunk_classification(context_chunks, c.chunk_id))
                .max()
                .unwrap_or(max_classification)
        } else {
            Classification::Unclassified
        };

        OutputResult {
            text,
            kind: AnswerKind::Grounded,
            citations,
            groundedness: Groundedness {
                support_ratio,
                unsupported_sentences,
                top_score,
                passed: true,
            },
            classification,
            warnings,
        }
    }

    fn refuse(&self, lang: Lang, _classification: Classification, reason: String) -> OutputResult {
        OutputResult {
            text: dq_llm_refusal(lang),
            kind: AnswerKind::Refused,
            citations: Vec::new(),
            groundedness: Groundedness {
                support_ratio: 0.0,
                unsupported_sentences: Vec::new(),
                top_score: 0.0,
                passed: false,
            },
            // Reddedilen cevap bilgi ifsa etmedigi icin dusuk dereceli damgalanir.
            classification: Classification::Unclassified,
            warnings: vec![reason],
        }
    }
}

fn context_chunk_classification(chunks: &[ScoredChunk], chunk_id: Uuid) -> Classification {
    chunks
        .iter()
        .find(|c| c.chunk.id == chunk_id)
        .map(|c| c.chunk.classification)
        .unwrap_or(Classification::Unclassified)
}

/// Modele bagimli olmadan sabit ret metni. `dq-llm`'deki metinle ayni
/// olmasi icin burada tekrar tanimlanir (dq-guard, dq-llm'e bagimli degildir).
fn dq_llm_refusal(lang: Lang) -> String {
    match lang {
        Lang::En => {
            "This information is not present in the uploaded documents. Try a different question or upload the relevant document.".to_string()
        }
        _ => "Bu bilgi yüklenen belgelerde bulunmuyor. Farklı bir soru sorabilir veya ilgili belgeyi yükleyebilirsiniz.".to_string(),
    }
}

/// Bir `Answer` degerini bu sonuca gore olusturur (server katmani icin yardimci).
pub fn into_answer(
    query_id: Uuid,
    result: OutputResult,
    lang: Lang,
    model: String,
    cached: bool,
    latency_ms: u64,
) -> Answer {
    Answer {
        query_id,
        kind: result.kind,
        text: result.text,
        citations: result.citations,
        groundedness: result.groundedness,
        lang,
        classification: result.classification,
        cached,
        latency_ms,
        model,
        warnings: result.warnings,
        trace: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dq_core::{Chunk, Classification};

    fn chunk(text: &str, score: f32, classification: Classification) -> ScoredChunk {
        ScoredChunk {
            chunk: Chunk {
                id: Uuid::new_v4(),
                doc_id: Uuid::new_v4(),
                ordinal: 0,
                page_from: 4,
                page_to: 4,
                text: text.into(),
                heading_path: None,
                token_estimate: 30,
                lang: Lang::Tr,
                classification,
                confidence: 1.0,
                parent_id: None,
                chunk_type: dq_core::ChunkType::Standalone,
            },
            score,
            dense_score: Some(score),
            sparse_score: None,
            rerank_score: None,
            doc_filename: "bakim.pdf".into(),
        }
    }

    fn cfg() -> GuardrailConfig {
        GuardrailConfig::default()
    }

    #[test]
    fn grounded_answer_passes_with_citation() {
        let guard = OutputGuard::new(cfg());
        let ctx = vec![chunk(
            "Motorun periyodik bakımı 500 saatte bir yapılır ve yağ değişimi zorunludur.",
            0.6,
            Classification::Restricted,
        )];
        let raw = "Periyodik bakım 500 saatte bir yapılır ve yağ değişimi zorunludur. [1]";
        let res = guard.evaluate(raw, &ctx, Lang::Tr);
        assert_eq!(res.kind, AnswerKind::Grounded);
        assert_eq!(res.citations.len(), 1);
        assert_eq!(res.classification, Classification::Restricted);
    }

    #[test]
    fn hallucinated_answer_is_refused() {
        let guard = OutputGuard::new(cfg());
        let ctx = vec![chunk(
            "Motorun periyodik bakımı 500 saatte bir yapılır.",
            0.6,
            Classification::Unclassified,
        )];
        let raw = "Uçağın azami hızı 900 km/s olarak test edilmiştir. [1]";
        let res = guard.evaluate(raw, &ctx, Lang::Tr);
        assert_eq!(res.kind, AnswerKind::Refused);
        assert!(res.groundedness.support_ratio < 0.5);
    }

    #[test]
    fn low_top_score_forces_refusal_even_if_text_matches() {
        let mut cfg = cfg();
        cfg.min_top_score = 0.9;
        let guard = OutputGuard::new(cfg);
        let ctx = vec![chunk(
            "Motorun periyodik bakımı 500 saatte bir yapılır.",
            0.3,
            Classification::Unclassified,
        )];
        let raw = "Motorun periyodik bakımı 500 saatte bir yapılır. [1]";
        let res = guard.evaluate(raw, &ctx, Lang::Tr);
        assert_eq!(res.kind, AnswerKind::Refused);
    }

    #[test]
    fn pii_in_answer_is_redacted() {
        let guard = OutputGuard::new(cfg());
        let ctx = vec![chunk(
            "Sorumlu personelin TC kimlik numarası 10000000146 olarak kayıtlıdır.",
            0.6,
            Classification::Secret,
        )];
        let raw = "Sorumlu personelin TC kimlik numarası 10000000146 olarak kayıtlıdır. [1]";
        let res = guard.evaluate(raw, &ctx, Lang::Tr);
        assert_eq!(res.kind, AnswerKind::Grounded);
        assert!(res.text.contains("[TCKN-MASKELENDI]"));
    }
}
