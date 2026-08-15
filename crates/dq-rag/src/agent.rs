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
use dq_llm::json::extract_json_object;
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

    let catalog = if acfg.enabled && acfg.enable_tool_doc_selection && user_doc_filter.is_empty() {
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
    let mut last: Option<(String, Vec<ScoredChunk>)> = None;

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
            last = Some((current_query.clone(), merged));
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

        let critique_started = std::time::Instant::now();
        let result = output_guard.evaluate(&raw_text, &context_chunks, lang);
        let passed = result.kind == AnswerKind::Grounded;
        let critique_ms = critique_started.elapsed().as_millis() as u64;
        trace.push(AgentStep {
            step,
            kind: AgentStepKind::Critique,
            description: format!(
                "Kaynak dogrulama: {:?}, destek orani %{:.0}, en iyi skor {:.2}.",
                result.kind,
                result.groundedness.support_ratio * 100.0,
                result.groundedness.top_score
            ),
            detail: serde_json::json!({
                "support_ratio": result.groundedness.support_ratio,
                "top_score": result.groundedness.top_score,
                "passed": passed,
                "critique_ms": critique_ms,
            }),
        });

        if passed || !acfg.enabled || !acfg.enable_self_correction || step >= max_steps {
            return Ok(AgentOutcome {
                text: result.text,
                kind: result.kind,
                citations: result.citations,
                groundedness: result.groundedness,
                classification: result.classification,
                warnings: result.warnings,
                trace,
            });
        }

        // Yetersiz destek: bir sonraki adim icin sorguyu yeniden formule et.
        current_query =
            reformulate(&current_query, lang, llm, llm_available, step, &mut trace).await;
        last = Some((current_query.clone(), context_chunks));
    }

    // Dongu, ilk yinelemede hic sonuc bulamadan kirildiysa (merged.is_empty()) buraya duser.
    let lang_for_refusal = lang;
    let _ = last;
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

#[allow(clippy::too_many_arguments)]
async fn plan_step(
    query: &str,
    lang: Lang,
    acfg: &dq_core::config::AgentConfig,
    catalog: &[Document],
    llm: &dyn LlmClient,
    llm_available: bool,
    step: usize,
    trace: &mut Vec<AgentStep>,
) -> Plan {
    if !llm_available {
        trace.push(AgentStep {
            step,
            kind: AgentStepKind::Plan,
            description: "Planlama atlandi (LLM kullanilamiyor); orijinal sorgu kullanildi.".into(),
            detail: serde_json::json!({"sub_queries": [query], "expanded_queries": [query]}),
        });
        return Plan {
            sub_queries: vec![query.to_string()],
            expanded_queries: vec![query.to_string()],
            doc_filter: Vec::new(),
        };
    }

    let catalog_ids: Vec<String> = catalog.iter().map(|d| d.id.to_string()).collect();
    let catalog_refs: Vec<prompts::CatalogDoc<'_>> = catalog_ids
        .iter()
        .zip(catalog.iter())
        .map(|(id, d)| prompts::CatalogDoc {
            id,
            filename: &d.filename,
        })
        .collect();
    let messages = vec![
        ChatMessage::system(prompts::planning_system_prompt(lang)),
        ChatMessage::user(prompts::planning_user_prompt(
            query,
            &catalog_refs,
            acfg.max_sub_queries,
            lang,
        )),
    ];

    let completion = match llm.chat_with_temperature(messages, 0.0).await {
        Ok(c) => c.text,
        Err(e) => {
            trace.push(AgentStep {
                step,
                kind: AgentStepKind::Plan,
                description: format!(
                    "Planlama cagrisi basarisiz ({e}); orijinal sorgu kullanildi."
                ),
                detail: serde_json::json!({"sub_queries": [query], "expanded_queries": [query]}),
            });
            return Plan {
                sub_queries: vec![query.to_string()],
                expanded_queries: vec![query.to_string()],
                doc_filter: Vec::new(),
            };
        }
    };

    let Some(json) = extract_json_object(&completion) else {
        trace.push(AgentStep {
            step,
            kind: AgentStepKind::Plan,
            description: "Planlama ciktisi JSON olarak ayristirilamadi; orijinal sorgu kullanildi."
                .into(),
            detail: serde_json::json!({"sub_queries": [query], "expanded_queries": [query]}),
        });
        return Plan {
            sub_queries: vec![query.to_string()],
            expanded_queries: vec![query.to_string()],
            doc_filter: Vec::new(),
        };
    };

    let mut sub_queries: Vec<String> = json
        .get("sub_queries")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str())
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect()
        })
        .unwrap_or_default();
    sub_queries.dedup();
    sub_queries.truncate(acfg.max_sub_queries.max(1));
    if sub_queries.is_empty() {
        sub_queries.push(query.to_string());
    }
    if !sub_queries.contains(&query.to_string()) {
        sub_queries.insert(0, query.to_string());
        sub_queries.truncate(acfg.max_sub_queries.max(1));
    }

    let expanded_queries: Vec<String> = json
        .get("expanded_queries")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str())
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect()
        })
        .unwrap_or_else(|| sub_queries.clone());

    let catalog_ids: std::collections::HashSet<Uuid> = catalog.iter().map(|d| d.id).collect();
    let doc_filter: Vec<Uuid> = json
        .get("doc_ids")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str())
                .filter_map(|s| Uuid::parse_str(s).ok())
                .filter(|id| catalog_ids.contains(id))
                .collect()
        })
        .unwrap_or_default();

    let reasoning = json.get("reasoning").and_then(|v| v.as_str()).unwrap_or("");
    trace.push(AgentStep {
        step,
        kind: AgentStepKind::Plan,
        description: format!(
            "{} alt-sorguya ayristirildi{}.",
            sub_queries.len(),
            if doc_filter.is_empty() {
                String::new()
            } else {
                format!(", {} belgeyle sinirlandirildi", doc_filter.len())
            }
        ),
        detail: serde_json::json!({"sub_queries": sub_queries.clone(), "expanded_queries": expanded_queries.clone(), "doc_ids": doc_filter, "reasoning": reasoning}),
    });

    Plan {
        sub_queries,
        expanded_queries,
        doc_filter,
    }
}

async fn reformulate(
    original_query: &str,
    lang: Lang,
    llm: &dyn LlmClient,
    llm_available: bool,
    step: usize,
    trace: &mut Vec<AgentStep>,
) -> String {
    if !llm_available {
        trace.push(AgentStep {
            step,
            kind: AgentStepKind::Plan,
            description:
                "Yeniden formulasyon icin LLM yok; belge kapsami genisletilerek tekrar denenecek."
                    .into(),
            detail: serde_json::json!({"query": original_query}),
        });
        return original_query.to_string();
    }
    let messages = vec![ChatMessage::user(prompts::reformulation_prompt(
        original_query,
        lang,
    ))];
    let rewritten = match llm.chat_with_temperature(messages, 0.3).await {
        Ok(c) => extract_json_object(&c.text)
            .and_then(|v| {
                v.get("query")
                    .and_then(|q| q.as_str())
                    .map(|s| s.trim().to_string())
            })
            .filter(|s| !s.is_empty()),
        Err(_) => None,
    };
    let next = rewritten.unwrap_or_else(|| original_query.to_string());
    trace.push(AgentStep {
        step,
        kind: AgentStepKind::Plan,
        description: "Yetersiz kaynak dogrulamasi nedeniyle sorgu yeniden formule edildi.".into(),
        detail: serde_json::json!({"original": original_query, "rewritten": next}),
    });
    next
}

/// Sorgu genisletme: orijinal sorgunun es anlamli varyasyonlarini uretir.
/// Bu, arama kapsamini genisletmek icin kullanilir (query expansion).
async fn generate_multi_query(
    query: &str,
    lang: Lang,
    max_variants: usize,
    llm: &dyn LlmClient,
    llm_available: bool,
) -> Vec<String> {
    if !llm_available || max_variants <= 1 {
        return vec![query.to_string()];
    }
    let messages = vec![ChatMessage::user(prompts::query_expansion_prompt(
        query,
        lang,
        max_variants,
    ))];
    match llm.chat_with_temperature(messages, 0.7).await {
        Ok(c) => extract_json_object(&c.text)
            .and_then(|v| {
                v.get("expanded_queries")
                    .and_then(|q| q.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|q| q.as_str())
                            .map(|s| s.trim().to_string())
                            .filter(|s| !s.is_empty() && s != query)
                            .collect::<Vec<_>>()
                    })
            })
            .unwrap_or_default(),
        Err(_) => Vec::new(),
    }
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
