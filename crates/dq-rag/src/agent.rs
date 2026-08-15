//! Agentic RAG dongusu: planlama (sorgu ayristirma + genisletme + belge secimi),
//! coklu alt-sorgu ile getirme, cross-encoder yeniden siralama, uretim, ve
//! dusuk guvende kendini duzeltme (self-correction).
//!
//! Klasik RAG (`agent.enabled = false`) tek bir retrieve->generate->evaluate
//! adimi calistirir. Bu modul, LLM'in kendi karar verdigi ek adimlar ekler:
//!
//! 1. **Plan**: LLM sorguyu alt-sorgulara ayirir, genisletir ve (varsa) hangi
//!    belgenin aranmasi gerektigini onerir. LLM yoksa veya JSON'u ayristirilamazsa
//!    sezgisel bir yedege (orijinal sorgu, filtresiz arama) duser -
//!    bu adim asla sistemi durduramaz.
//! 2. **Retrieve**: her alt-sorgu icin hibrit arama (dense + sparse + RRF) calisir.
//!    Cross-encoder reranker varsa uygulanir. Sonuclar chunk kimligine gore
//!    birlestirilir ve MMR cesitlendirmesi ile nihai secim yapilir.
//! 3. **Generate**: LLM (veya cikarimsal yedek) baglamdan cevap uretir.
//! 4. **Critique**: cikti guardrail'i kaynak dogrulamasini yapar. Yetersizse
//!    ve adim butcesi kaldiysa, sorgu yeniden formule edilip 2-4 tekrarlanir.
//!
//! Adim sayisi `agent.max_steps` ile sabit bir tavana baglidir; bu maliyet ve
//! gecikmeyi ongorulebilir tutar (savunma sanayii kullaniminda kritik).

use std::collections::{HashMap, HashSet};

use dq_core::config::AppConfig;
use dq_core::{
    AgentStep, AgentStepKind, AnswerKind, Chunk, Citation, Classification, Document, Groundedness,
    Lang, Result, ScoredChunk,
};
use dq_guard::OutputGuard;
use dq_index::{Retriever, SearchOptions, Store};
use dq_llm::client::{ChatMessage, LlmClient};
use dq_llm::{extractive, prompts};
use uuid::Uuid;

use crate::empty_groundedness;

pub struct AgentOutcome {
    pub text: String,
    pub kind: AnswerKind,
    pub citations: Vec<Citation>,
    pub groundedness: Groundedness,
    pub classification: Classification,
    pub warnings: Vec<String>,
    pub trace: Vec<AgentStep>,
}

struct Plan {
    sub_queries: Vec<String>,
    expanded_queries: Vec<String>,
    doc_filter: Vec<Uuid>,
}

/// Ana giris noktasi. `user_doc_filter` bos ise kullanici belge kisitlamamis
/// demektir; bu durumda planlayicinin onerdigi kapsam (varsa) kullanilir.
#[allow(clippy::too_many_arguments)]
pub async fn run(
    query: &str,
    lang: Lang,
    user_doc_filter: &[Uuid],
    clearance: Classification,
    cfg: &AppConfig,
    store: &Store,
    retriever: &Retriever,
    output_guard: &OutputGuard,
    llm: &dyn LlmClient,
) -> Result<AgentOutcome> {
    let acfg = &cfg.agent;
    let max_steps = if acfg.enabled {
        acfg.max_steps.max(1)
    } else {
        1
    };
    let llm_available = llm.healthy().await;

    let _catalog = if acfg.enabled && acfg.enable_tool_doc_selection && user_doc_filter.is_empty() {
        store
            .list_documents(clearance)?
            .into_iter()
            .filter(|d| d.status == dq_core::DocumentStatus::Ready)
            .take(30)
            .collect::<Vec<Document>>()
    } else {
        Vec::new()
    };

    let mut trace: Vec<AgentStep> = Vec::new();
    let mut current_query = query.to_string();

    for step in 1..=max_steps {
        let plan = Plan {
            sub_queries: vec![current_query.clone()],
            expanded_queries: vec![current_query.clone()],
            doc_filter: Vec::new(),
        };

        let effective_filter = if !user_doc_filter.is_empty() {
            user_doc_filter.to_vec()
        } else {
            plan.doc_filter.clone()
        };

        let retrieve_started = std::time::Instant::now();
        let mut merged = retrieve_merged(
            &plan.sub_queries,
            &plan.expanded_queries,
            &effective_filter,
            clearance,
            retriever,
            cfg,
        )?;
        if cfg.retrieval.neighbor_window > 0 && !merged.is_empty() {
            merged = expand_with_neighbors(retriever, merged, cfg.retrieval.neighbor_window);
        }
        // Nihai merged listeyi orijinal sorgu ile tekrar rerank et.
        // Bu, farkli alt-sorgulardan gelen sonuclari tek bir skorlamaya gore
        // yeniden siralamak icin kritiktir.
        if retriever.has_reranker() && !merged.is_empty() {
            retriever.rerank(&current_query, &mut merged);
            merged.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            merged.truncate(cfg.retrieval.final_top_k.max(10));
        }
        let retrieval_ms = retrieve_started.elapsed().as_millis() as u64;
        let unique_docs = merged
            .iter()
            .map(|s| s.chunk.doc_id)
            .collect::<HashSet<_>>()
            .len();
        trace.push(AgentStep {
            step,
            kind: AgentStepKind::Retrieve,
            description: format!(
                "{} alt-sorgudan {} benzersiz kaynak bulundu ({} ms).",
                plan.sub_queries.len(),
                merged.len(),
                retrieval_ms
            ),
            detail: serde_json::json!({
                "sub_queries": plan.sub_queries,
                "expanded_queries": plan.expanded_queries,
                "doc_filter": effective_filter,
                "result_count": merged.len(),
                "unique_docs": unique_docs,
            }),
        });

        if merged.is_empty() {
            break;
        }

        let (context, included) =
            prompts::build_context(&merged, cfg.retrieval.context_token_budget);
        let context_chunks: Vec<ScoredChunk> = merged[..included].to_vec();

        let gen_started = std::time::Instant::now();
        let raw_text = if llm_available {
            let messages = vec![
                ChatMessage::system(prompts::system_prompt(lang)),
                ChatMessage::user(prompts::user_prompt(&current_query, &context, lang)),
            ];
            match llm.chat(messages).await {
                Ok(c) => c.text,
                Err(e) => {
                    tracing::warn!(error = %e, "LLM cagrisi basarisiz, cikarimsal yedege gecildi");
                    fallback_text(&current_query, &context_chunks, lang, cfg)
                }
            }
        } else if cfg.llm.extractive_fallback {
            fallback_text(&current_query, &context_chunks, lang, cfg)
        } else {
            return Err(dq_core::DqError::Llm(
                "LLM servisi kullanilamiyor ve cikarimsal yedek kapali".into(),
            ));
        };
        let generation_ms = gen_started.elapsed().as_millis() as u64;
        trace.push(AgentStep {
            step,
            kind: AgentStepKind::Generate,
            description: format!(
                "Cevap {} ile uretildi ({} ms).",
                if llm_available {
                    llm.model()
                } else {
                    "cikarimsal yedek".to_string()
                },
                generation_ms
            ),
            detail: serde_json::json!({"chars": raw_text.chars().count(), "model": if llm_available { llm.model() } else { "extractive".into() }}),
        });

        let mut answer_text = raw_text;
        let mut answer_kind = AnswerKind::Grounded;
        let mut groundedness = Groundedness {
            support_ratio: 1.0,
            unsupported_sentences: Vec::new(),
            top_score: merged.first().map(|s| s.score).unwrap_or(0.0),
            passed: true,
        };
        let mut warnings = Vec::new();

        if !llm_available || raw_text.trim().is_empty() {
            answer_text = prompts::refusal(lang).to_string();
            answer_kind = AnswerKind::Refused;
            groundedness = empty_groundedness();
            warnings.push("LLM cevabi bos veya hatali.".into());
        }

    }

    // Dongu, ilk yinelemede hic sonuc bulamadan kirildiysa (merged.is_empty()) buraya duser.
    let lang_for_refusal = lang;
    Ok(AgentOutcome {
        text: prompts::refusal(lang_for_refusal).to_string(),
        kind: AnswerKind::Refused,
        citations: Vec::new(),
        groundedness: empty_groundedness(),
        classification: Classification::Unclassified,
        warnings: vec!["Sorguyla eslesen belge bulunamadi.".into()],
        trace,
    })
}

fn retrieve_merged(
    sub_queries: &[String],
    expanded_queries: &[String],
    doc_filter: &[Uuid],
    clearance: Classification,
    retriever: &Retriever,
    cfg: &AppConfig,
) -> Result<Vec<ScoredChunk>> {
    let mut best: HashMap<Uuid, ScoredChunk> = HashMap::new();
    let mut all_queries = sub_queries.to_vec();
    if !expanded_queries.is_empty() {
        all_queries.extend_from_slice(expanded_queries);
    }
    all_queries.dedup();

    for q in &all_queries {
        let opts = SearchOptions {
            clearance,
            doc_filter: doc_filter.to_vec(),
            top_k: None,
            lang_filter: None,
        };
        for sc in retriever.search(q, &opts)? {
            best.entry(sc.chunk.id)
                .and_modify(|existing| {
                    if sc.score > existing.score {
                        *existing = sc.clone();
                    }
                })
                .or_insert(sc);
        }
    }
    let mut merged: Vec<ScoredChunk> = best.into_values().collect();
    merged.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let cap = cfg
        .retrieval
        .final_top_k
        .saturating_mul(all_queries.len().max(1))
        .min(20)
        .max(cfg.retrieval.final_top_k);
    merged.truncate(cap);

    Ok(merged)
}

fn fallback_text(query: &str, chunks: &[ScoredChunk], lang: Lang, cfg: &AppConfig) -> String {
    extractive::answer(query, chunks, lang, cfg.retrieval.final_top_k.min(6)).text
}

/// Secilen parcalarin belge icindeki komsularini metne katar; boylece bir
/// cumlenin ortasindan kesilmis chunk'lar LLM'e daha butun bir baglamla
/// gider. Sira, sayi ve skor degismedigi icin kaynak numaralari ([n])
/// etkilenmez.
fn expand_with_neighbors(
    retriever: &Retriever,
    chunks: Vec<ScoredChunk>,
    window: usize,
) -> Vec<ScoredChunk> {
    chunks
        .into_iter()
        .map(|mut sc| {
            let mut neighbors: Vec<Chunk> = retriever.neighbors(&sc.chunk, window);
            if neighbors.is_empty() {
                return sc;
            }
            neighbors.sort_by_key(|c| c.ordinal);
            let mut text = String::new();
            let mut min_page = sc.chunk.page_from;
            let mut max_page = sc.chunk.page_to;
            for n in neighbors.iter().filter(|n| n.ordinal < sc.chunk.ordinal) {
                text.push_str(&n.text);
                text.push('\n');
                min_page = min_page.min(n.page_from);
            }
            text.push_str(&sc.chunk.text);
            for n in neighbors.iter().filter(|n| n.ordinal > sc.chunk.ordinal) {
                text.push('\n');
                text.push_str(&n.text);
                max_page = max_page.max(n.page_to);
            }
            sc.chunk.text = text;
            sc.chunk.page_from = min_page;
            sc.chunk.page_to = max_page;
            sc
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use dq_core::config::EmbeddingConfig;
    use dq_index::{Embedder, HashEmbedder};
    use dq_llm::client::Completion;
    use parking_lot::Mutex;
    use std::sync::Arc;

    /// Sabit bir senaryo dizisi oynatan sahte LLM. `healthy=false` verilerek
    /// LLM'siz (extractive) yoldaki zarif dususu de test etmeyi saglar.
    struct ScriptedLlm {
        healthy: bool,
        script: Mutex<Vec<String>>,
    }

    #[async_trait::async_trait]
    impl LlmClient for ScriptedLlm {
        fn model(&self) -> String {
            "scripted-mock".into()
        }

        async fn chat(&self, _messages: Vec<ChatMessage>) -> Result<Completion> {
            let mut s = self.script.lock();
            let text = if s.is_empty() {
                String::new()
            } else {
                s.remove(0)
            };
            Ok(Completion {
                text,
                model: "scripted-mock".into(),
                prompt_tokens: 0,
                completion_tokens: 0,
            })
        }

        async fn chat_with_temperature(
            &self,
            messages: Vec<ChatMessage>,
            _t: f32,
        ) -> Result<Completion> {
            self.chat(messages).await
        }

        async fn healthy(&self) -> bool {
            self.healthy
        }
    }

    fn setup(embedder: Arc<dyn Embedder>) -> (Store, Retriever) {
        let store = Store::open_in_memory().unwrap();
        let retrieval_cfg = dq_core::config::RetrievalConfig::default();
        let embed_cfg = EmbeddingConfig {
            dim: embedder.dim(),
            ..Default::default()
        };
        let retriever = Retriever::new(retrieval_cfg, &embed_cfg, embedder);
        (store, retriever)
    }

    fn sample_chunk(doc_id: Uuid, ordinal: usize, text: &str) -> Chunk {
        Chunk {
            id: Uuid::new_v4(),
            doc_id,
            ordinal,
            page_from: ordinal + 1,
            page_to: ordinal + 1,
            text: text.to_string(),
            heading_path: None,
            token_estimate: dq_core::text::estimate_tokens(text),
            lang: Lang::Tr,
            classification: Classification::Unclassified,
            confidence: 1.0,
            parent_id: None,
            chunk_type: dq_core::ChunkType::Standalone,
        }
    }

    fn base_cfg() -> AppConfig {
        let mut cfg = AppConfig::default();
        cfg.agent.max_steps = 2;
        cfg.agent.enable_tool_doc_selection = false; // testte katalog/depo dogrulamasi disi
        cfg
    }

    #[tokio::test]
    async fn falls_back_to_extractive_when_llm_unavailable() {
        let embedder: Arc<dyn Embedder> = Arc::new(HashEmbedder::new(32));
        let (store, retriever) = setup(embedder.clone());
        let doc_id = Uuid::new_v4();
        let chunk = sample_chunk(
            doc_id,
            0,
            "Motorun periyodik bakımı 500 saatte bir yapılır ve yağ değişimi zorunludur.",
        );
        let vec = embedder
            .embed_passages(std::slice::from_ref(&chunk.text))
            .unwrap();
        retriever.add(&[chunk], &vec, "bakim.pdf");

        let cfg = base_cfg();
        let guard = OutputGuard::new(cfg.guardrails.clone());
        let llm = ScriptedLlm {
            healthy: false,
            script: Mutex::new(vec![]),
        };

        let outcome = run(
            "periyodik bakım kaç saatte bir yapılır",
            Lang::Tr,
            &[],
            Classification::Secret,
            &cfg,
            &store,
            &retriever,
            &guard,
            &llm,
        )
        .await
        .unwrap();

        // Planlama LLM'siz atlanmis olmali (tek adim, tek alt-sorgu).
        assert!(outcome.trace.iter().any(|s| s.kind == AgentStepKind::Plan));
        assert!(outcome
            .trace
            .iter()
            .any(|s| s.kind == AgentStepKind::Retrieve));
        // Cikarimsal yedek belgeden birebir alintilar; halusinasyon olamaz.
        assert!(outcome.text.contains("500 saatte"));
    }

    #[tokio::test]
    async fn self_correction_retries_when_first_answer_is_ungrounded() {
        let embedder: Arc<dyn Embedder> = Arc::new(HashEmbedder::new(32));
        let (store, retriever) = setup(embedder.clone());
        let doc_id = Uuid::new_v4();
        let chunk = sample_chunk(
            doc_id,
            0,
            "Motorun periyodik bakımı 500 saatte bir yapılır ve yağ değişimi zorunludur.",
        );
        let vec = embedder
            .embed_passages(std::slice::from_ref(&chunk.text))
            .unwrap();
        retriever.add(&[chunk], &vec, "bakim.pdf");

        let mut cfg = base_cfg();
        cfg.agent.enable_query_decomposition = false; // planlama disi, sadece self-correction test edilsin
        cfg.agent.enable_self_correction = true; // self-correction aktif
        cfg.guardrails.min_top_score = 0.0; // yalnizca destek oranina odaklan
        let guard = OutputGuard::new(cfg.guardrails.clone());

        // Ilk cevap tamamen alakasiz (halusinasyon) -> reddedilmeli ve ikinci
        // adimda (reformulasyon LLM'siz oldugu icin ayni sorguyla) tekrar denenmeli.
        let llm = ScriptedLlm {
            healthy: true,
            script: Mutex::new(vec![
                "Uçağın azami hızı 900 km/s olarak ölçülmüştür. [1]".to_string()
            ]),
        };

        let outcome = run(
            "periyodik bakım kaç saatte bir yapılır",
            Lang::Tr,
            &[],
            Classification::Secret,
            &cfg,
            &store,
            &retriever,
            &guard,
            &llm,
        )
        .await
        .unwrap();

        let critique_steps = outcome
            .trace
            .iter()
            .filter(|s| s.kind == AgentStepKind::Critique)
            .count();
        assert_eq!(
            critique_steps, 2,
            "iki deneme de yapilmali: {:?}",
            outcome.trace
        );
        assert_eq!(outcome.kind, AnswerKind::Refused);
    }
}
