//! RAG orkestrasyonu.
//!
//! Akis: guardrail (girdi) -> onbellek -> hibrit getirme -> baglam olusturma
//! -> yerel LLM (veya cikarimsal yedek) -> guardrail (cikti) -> onbellek yaz
//! -> denetim kaydi. Her adim ayri bir crate'te test edildigi icin bu dosya
//! yalnizca *sirlama* mantigini icerir.

use std::sync::Arc;
use std::time::Instant;

use dq_core::config::AppConfig;
use dq_core::{
    Answer, AnswerKind, Classification, Document, DocumentStatus, DqError, Groundedness, Result,
    UserContext,
};
use dq_guard::{InputGuard, OutputGuard};
use dq_index::{AnswerCache, CacheHit, CacheKey, Embedder, Retriever, Store};
use dq_ingest::Ingestor;
use dq_llm::client::LlmClient;
use uuid::Uuid;

mod agent;

pub struct UploadOutcome {
    pub document: Document,
    pub warnings: Vec<String>,
}

pub struct Pipeline {
    cfg: AppConfig,
    store: Arc<Store>,
    retriever: Arc<Retriever>,
    embedder: Arc<dyn Embedder>,
    ingestor: Ingestor,
    input_guard: InputGuard,
    output_guard: OutputGuard,
    cache: AnswerCache,
    llm: Arc<dyn LlmClient>,
    /// Gomme modeli yuklenemedigi icin yedege dusuldugunde kullaniciya
    /// gosterilecek uyari (bkz. `dq_index::embed::build`).
    embedding_fallback_warning: Option<String>,
}

impl Pipeline {
    pub fn new(
        cfg: AppConfig,
        store: Arc<Store>,
        retriever: Arc<Retriever>,
        embedder: Arc<dyn Embedder>,
        llm: Arc<dyn LlmClient>,
        embedding_fallback_warning: Option<String>,
    ) -> Result<Self> {
        let ingestor = Ingestor::new(cfg.ingest.clone(), &cfg.ocr);
        let input_guard = InputGuard::new(cfg.guardrails.clone())?;
        let output_guard = OutputGuard::new(cfg.guardrails.clone());
        let cache = AnswerCache::new(cfg.cache.clone());
        Ok(Self {
            cfg,
            store,
            retriever,
            embedder,
            ingestor,
            input_guard,
            output_guard,
            cache,
            llm,
            embedding_fallback_warning,
        })
    }

    pub fn store(&self) -> &Store {
        &self.store
    }

    pub fn retriever(&self) -> &Retriever {
        &self.retriever
    }

    pub fn cache(&self) -> &AnswerCache {
        &self.cache
    }

    pub fn ocr_engine_name(&self) -> &'static str {
        self.ingestor.ocr_engine_name()
    }

    /// Saglik denetimi ucnoktasi icin: yerel LLM servisine ulasilabiliyor mu?
    pub async fn llm_healthy(&self) -> bool {
        self.llm.healthy().await
    }

    /// Bir belgeyi isler, gomer ve indekse ekler.
    ///
    /// CPU-yogun (PDF/OCR/embedding) oldugu icin cagiran taraf bunu
    /// `tokio::task::spawn_blocking` icinde calistirmalidir.
    pub fn ingest_document(
        &self,
        bytes: &[u8],
        filename: &str,
        classification: Classification,
        owner: &str,
    ) -> Result<UploadOutcome> {
        if bytes.len() as u64 > self.cfg.ingest.max_file_bytes {
            return Err(DqError::PayloadTooLarge {
                size: bytes.len() as u64,
                limit: self.cfg.ingest.max_file_bytes,
            });
        }

        let content_hash = dq_core::ids::content_hash(bytes);
        if let Some(existing) = self.store.find_document_by_hash(&content_hash)? {
            return Ok(UploadOutcome {
                document: existing,
                warnings: vec!["Bu belge zaten yuklenmis (icerik ozdes); tekrar islenmedi.".into()],
            });
        }

        let doc_id = dq_core::ids::new_id();
        let outcome = self.ingestor.ingest(bytes, doc_id, classification)?;

        let now = chrono::Utc::now();
        let mut document = Document {
            id: doc_id,
            filename: dq_ingest::sanitize_filename(filename),
            mime: outcome.kind.mime().to_string(),
            content_hash,
            size_bytes: bytes.len() as u64,
            page_count: outcome.page_count,
            lang: outcome.lang,
            classification,
            status: DocumentStatus::Processing,
            owner: owner.to_string(),
            created_at: now,
            updated_at: now,
            error: None,
            avg_confidence: outcome.avg_confidence,
        };
        self.store.insert_document(&document)?;

        let texts: Vec<String> = outcome.chunks.iter().map(|c| c.text.clone()).collect();
        let vectors = match self.embedder.embed_passages(&texts) {
            Ok(v) => v,
            Err(e) => {
                document.status = DocumentStatus::Failed;
                document.error = Some(e.to_string());
                document.updated_at = chrono::Utc::now();
                // Basarisiz belgeyi kaydet ki kullanici nedenini gorsun, ancak hatayi da dondur.
                let _ = self.store.update_document(&document);
                return Err(e);
            }
        };

        self.store
            .insert_chunks(&outcome.chunks, &vectors, &self.cfg.embedding.model)?;
        self.retriever
            .add(&outcome.chunks, &vectors, &document.filename);

        document.status = DocumentStatus::Ready;
        document.updated_at = chrono::Utc::now();
        self.store.update_document(&document)?;

        // Yeni belge, daha once "bilgi yok" denen sorulari cevaplanabilir hale
        // getirebilir; onbellek tazelenmezse eski ret cevabi sunulmaya devam eder.
        self.cache.invalidate_all(&self.store)?;

        let mut warnings = outcome.warnings;
        if let Some(w) = &self.embedding_fallback_warning {
            warnings.push(w.clone());
        }
        Ok(UploadOutcome { document, warnings })
    }

    pub fn list_documents(&self, clearance: Classification) -> Result<Vec<Document>> {
        self.store.list_documents(clearance)
    }

    pub fn get_document(&self, id: Uuid, clearance: Classification) -> Result<Option<Document>> {
        match self.store.get_document(id)? {
            Some(d) if d.classification.readable_by(clearance) => Ok(Some(d)),
            Some(_) => Err(DqError::Forbidden(
                "Bu belgeyi goruntuleme yetkiniz yok".into(),
            )),
            None => Ok(None),
        }
    }

    pub fn delete_document(&self, id: Uuid) -> Result<bool> {
        let deleted = self.store.delete_document(id)?;
        if deleted {
            self.retriever.rebuild(&self.store)?;
            self.cache.invalidate_all(&self.store)?;
        }
        Ok(deleted)
    }

    pub fn rebuild_index(&self) -> Result<()> {
        self.retriever.rebuild(&self.store)
    }

    /// Soru-cevap uc noktasi.
    pub async fn ask(
        &self,
        raw_query: &str,
        user: &UserContext,
        doc_filter: Vec<Uuid>,
    ) -> Result<Answer> {
        let started = Instant::now();

        let verdict = self.input_guard.check(raw_query)?;
        if !verdict.allowed {
            self.store.append_audit(
                &user.username,
                "ask",
                None,
                "blocked",
                serde_json::json!({"reasons": verdict.reasons, "score": verdict.injection_score}),
            )?;
            return Ok(Answer {
                query_id: dq_core::ids::new_id(),
                kind: AnswerKind::Blocked,
                text: verdict.reasons.join(" "),
                citations: vec![],
                groundedness: empty_groundedness(),
                lang: verdict.lang,
                classification: Classification::Unclassified,
                cached: false,
                latency_ms: elapsed_ms(started),
                model: self.llm.model(),
                warnings: verdict.reasons,
                trace: Vec::new(),
            });
        }

        let query = verdict.sanitized_query.clone();
        let cache_key = CacheKey {
            query: query.clone(),
            clearance: user.clearance,
            doc_filter: doc_filter.clone(),
            model: self.llm.model(),
        };
        let query_vec = if self.cfg.cache.semantic_enabled {
            self.embedder.embed_query(&query).ok()
        } else {
            None
        };

        if let Some(hit) = self
            .cache
            .get(&self.store, &cache_key, query_vec.as_deref())?
        {
            let mut answer = match hit {
                CacheHit::Exact(a) => a,
                CacheHit::Semantic(a, _sim) => a,
            };
            answer.cached = true;
            answer.latency_ms = elapsed_ms(started);
            self.store.append_audit(
                &user.username,
                "ask",
                None,
                "cache_hit",
                serde_json::json!({"query": query}),
            )?;
            return Ok(answer);
        }

        let outcome = agent::run(
            &query,
            verdict.lang,
            &doc_filter,
            user.clearance,
            &self.cfg,
            &self.store,
            &self.retriever,
            self.llm.as_ref(),
        )
        .await?;

        let mut answer = Answer {
            query_id: dq_core::ids::new_id(),
            kind: outcome.kind,
            text: outcome.text,
            citations: outcome.citations,
            groundedness: outcome.groundedness,
            lang: verdict.lang,
            classification: outcome.classification,
            cached: false,
            latency_ms: elapsed_ms(started),
            model: self.llm.model(),
            warnings: outcome.warnings,
            trace: outcome.trace,
        };
        if verdict.injection_score > 0.0 {
            answer.warnings.extend(verdict.reasons.clone());
        }

        self.cache
            .put(&self.store, &cache_key, &answer, query_vec.as_deref())?;
        self.store.append_audit(
            &user.username,
            "ask",
            None,
            match answer.kind {
                AnswerKind::Grounded => "answered",
                AnswerKind::Refused => "refused",
                AnswerKind::Blocked => "blocked",
            },
            serde_json::json!({"query": query, "citations": answer.citations.len()}),
        )?;

        Ok(answer)
    }
}

fn empty_groundedness() -> Groundedness {
    Groundedness {
        support_ratio: 0.0,
        unsupported_sentences: Vec::new(),
        top_score: 0.0,
        passed: false,
    }
}

fn elapsed_ms(started: Instant) -> u64 {
    started.elapsed().as_millis() as u64
}
