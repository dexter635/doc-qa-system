//! Hibrit geri getirme (retrieval): yogun + seyrek arama, RRF fuzyonu,
//! MMR cesitlendirme ve istege bagli cross-encoder yeniden siralama.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use dq_core::config::{EmbeddingConfig, RetrievalConfig};
use dq_core::{semantic, Chunk, ChunkType, Classification, DqError, Lang, Result, ScoredChunk};
use parking_lot::RwLock;
use uuid::Uuid;

use crate::bm25::Bm25Index;
use crate::embed::{cosine, Embedder};
use crate::store::Store;
use crate::vector::{FlatIndex, VectorIndex};

struct Entry {
    chunk: Chunk,
    filename: String,
}

#[derive(Default)]
struct Inner {
    entries: Vec<Entry>,
    dense: FlatIndex,
    sparse: Bm25Index,
    by_doc: HashMap<Uuid, Vec<usize>>,
}

pub struct SearchOptions {
    pub clearance: Classification,
    /// Bos ise tum belgelerde arama yapilir.
    pub doc_filter: Vec<Uuid>,
    pub top_k: Option<usize>,
    /// Dili filtresi; None = tum diller.
    pub lang_filter: Option<Lang>,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            clearance: Classification::TopSecret,
            doc_filter: Vec::new(),
            top_k: None,
            lang_filter: None,
        }
    }
}

pub struct Retriever {
    cfg: RetrievalConfig,
    embedder: Arc<dyn Embedder>,
    reranker: Option<parking_lot::Mutex<fastembed::TextRerank>>,
    inner: RwLock<Inner>,
}

impl Retriever {
    pub fn new(
        cfg: RetrievalConfig,
        embed_cfg: &EmbeddingConfig,
        embedder: Arc<dyn Embedder>,
    ) -> Self {
        let reranker = build_reranker(embed_cfg);
        Self {
            cfg,
            inner: RwLock::new(Inner {
                dense: FlatIndex::new(embedder.dim()),
                ..Default::default()
            }),
            embedder,
            reranker,
        }
    }

    pub fn has_reranker(&self) -> bool {
        self.reranker.is_some()
    }

    pub fn len(&self) -> usize {
        self.inner.read().entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Indeksi veritabanindan bastan kurar. Acilista ve belge silmede cagrilir.
    pub fn rebuild(&self, store: &Store) -> Result<()> {
        let rows = store.load_all_chunks()?;
        let mut inner = Inner {
            dense: FlatIndex::with_capacity(self.embedder.dim(), rows.len()),
            ..Default::default()
        };
        for (chunk, vec, filename) in rows {
            let idx = inner.entries.len();
            inner.by_doc.entry(chunk.doc_id).or_default().push(idx);
            inner
                .sparse
                .push(&chunk.text, chunk.heading_path.as_deref());
            inner.dense.push(&vec);
            inner.entries.push(Entry { chunk, filename });
        }
        tracing::info!(chunks = inner.entries.len(), "indeks yeniden kuruldu");
        *self.inner.write() = inner;
        Ok(())
    }

    /// Yeni islenen bir belgeyi indekse ekler (tam yeniden kurulum gerekmez).
    pub fn add(&self, chunks: &[Chunk], vectors: &[Vec<f32>], filename: &str) {
        let mut inner = self.inner.write();
        for (chunk, vec) in chunks.iter().zip(vectors) {
            let idx = inner.entries.len();
            inner.by_doc.entry(chunk.doc_id).or_default().push(idx);
            inner
                .sparse
                .push(&chunk.text, chunk.heading_path.as_deref());
            inner.dense.push(vec);
            inner.entries.push(Entry {
                chunk: chunk.clone(),
                filename: filename.to_string(),
            });
        }
    }

    pub fn search(&self, query: &str, opts: &SearchOptions) -> Result<Vec<ScoredChunk>> {
        let query = query.trim();
        if query.is_empty() {
            return Err(DqError::BadRequest("bos sorgu".into()));
        }
        let inner = self.inner.read();
        if inner.entries.is_empty() {
            return Ok(Vec::new());
        }

        // Yetki ve belge filtresi tek bir izin fonksiyonunda toplanir; bu
        // filtre arama *sirasinda* uygulanir, sonradan degil. Aksi halde
        // yetkisiz icerik aday listesine girip skorlari etkilerdi.
        let doc_filter: HashSet<Uuid> = opts.doc_filter.iter().copied().collect();
        let lang_filter = opts.lang_filter;
        let allow = |i: usize| -> bool {
            let e = &inner.entries[i];
            if !e.chunk.classification.readable_by(opts.clearance) {
                return false;
            }
            if let Some(lang) = lang_filter {
                if e.chunk.lang != lang {
                    return false;
                }
            }
            doc_filter.is_empty() || doc_filter.contains(&e.chunk.doc_id)
        };

        let qvec = self.embedder.embed_query(query)?;
        let expanded = semantic::expand_query(query);
        let dense_hits = inner.dense.search(&qvec, self.cfg.dense_top_k, &allow);
        let sparse_hits = inner.sparse.search_expanded(query, self.cfg.sparse_top_k, &allow, &expanded);

        let fused = reciprocal_rank_fusion(&dense_hits, &sparse_hits, self.cfg.rrf_k);
        if fused.is_empty() {
            return Ok(Vec::new());
        }

        let dense_map: HashMap<usize, f32> = dense_hits.iter().copied().collect();
        let sparse_map: HashMap<usize, f32> = sparse_hits.iter().copied().collect();
        let max_sparse = sparse_hits
            .first()
            .map(|(_, s)| *s)
            .unwrap_or(1.0)
            .max(1e-6);

        let candidates: Vec<usize> = fused
            .iter()
            .take(self.cfg.rerank_candidates.max(self.cfg.final_top_k))
            .map(|(i, _)| *i)
            .collect();

        let final_k = opts.top_k.unwrap_or(self.cfg.final_top_k);

        // Cesitlilik: ayni paragrafin komsu parcalari baglami doldurup
        // farkli bilgi kaynaklarini disari itmemeli.
        let selected = mmr_select(
            &candidates,
            &|i| inner.dense.row(i),
            &qvec,
            self.cfg.mmr_lambda,
            final_k.max(self.cfg.final_top_k),
        );

        let mut out: Vec<ScoredChunk> = selected
            .iter()
            .map(|&i| {
                let e = &inner.entries[i];
                ScoredChunk {
                    chunk: e.chunk.clone(),
                    score: dense_map.get(&i).copied().unwrap_or(0.0),
                    dense_score: dense_map.get(&i).copied(),
                    sparse_score: sparse_map.get(&i).map(|s| s / max_sparse),
                    rerank_score: None,
                    doc_filename: e.filename.clone(),
                }
            })
            .collect();

        if let Some(rr) = &self.reranker {
            rerank_in_place(rr, query, &mut out);
        } else {
            // Cross-encoder yoksa nihai skor dense ve normalize sparse'in
            // agirlikli birlesimi olur.
            for s in out.iter_mut() {
                let d = s.dense_score.unwrap_or(0.0);
                let sp = s.sparse_score.unwrap_or(0.0);
                s.score = 0.75 * d + 0.25 * sp;
            }
        }

        out.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        out.retain(|s| s.score >= self.cfg.min_score);
        out.truncate(final_k);
        Ok(out)
    }

    /// Secilen chunk'in belge icindeki komsularini getirir; cevap uretilirken
    /// cumlenin kesildigi durumlarda baglami tamamlar.
    pub fn neighbors(&self, chunk: &Chunk, window: usize) -> Vec<Chunk> {
        if window == 0 {
            return Vec::new();
        }
        let inner = self.inner.read();
        let Some(indices) = inner.by_doc.get(&chunk.doc_id) else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for &i in indices {
            let c = &inner.entries[i].chunk;
            let diff = c.ordinal.abs_diff(chunk.ordinal);
            if diff > 0 && diff <= window {
                out.push(c.clone());
            }
        }
        out.sort_by_key(|c| c.ordinal);
        out
    }

    /// Verilen child chunk'larin parent'larini getirir.
    /// Parent-child retrieval (RAGFlow, LlamaIndex tarzi): ince taneli child
    /// chunk'lar yerine genis baglamli parent chunk'lar LLM'e verilir.
    pub fn parent_chunks(&self, child_ids: &[Uuid]) -> Vec<Chunk> {
        if child_ids.is_empty() {
            return Vec::new();
        }
        let inner = self.inner.read();
        let mut parents = Vec::new();
        let mut seen = HashSet::new();

        for &child_id in child_ids {
            let Some(child_idx) = inner.entries.iter().position(|e| e.chunk.id == child_id) else {
                continue;
            };
            let child = &inner.entries[child_idx].chunk;

            let parent_id = match child.chunk_type {
                ChunkType::Child => child.parent_id,
                _ => None,
            };

            if let Some(pid) = parent_id {
                if seen.insert(pid) {
                    if let Some(p_idx) = inner.entries.iter().position(|e| e.chunk.id == pid) {
                        parents.push(inner.entries[p_idx].chunk.clone());
                    }
                }
            }
        }
        parents
    }

    /// Disaridan verilen ScoredChunk listesini cross-encoder ile yeniden siralar.
    /// Agent tarafinda coklu alt-sorgu sonuclarini birlestirdikten sonra
    /// tek bir skorlamaya gore yeniden siralamak icin kullanilir.
    pub fn rerank(&self, query: &str, items: &mut [ScoredChunk]) {
        if items.is_empty() {
            return;
        }
        let Some(rr) = self.reranker.as_ref() else {
            return;
        };
        let docs: Vec<&str> = items.iter().map(|s| s.chunk.text.as_str()).collect();
        let mut guard = rr.lock();
        if let Ok(results) = guard.rerank(query, docs, false, None) {
            for r in results {
                if let Some(slot) = items.get_mut(r.index) {
                    let normalized = 1.0 / (1.0 + (-r.score).exp());
                    slot.rerank_score = Some(normalized);
                    slot.score = normalized;
                }
            }
        }
    }
}

/// Reciprocal Rank Fusion: farkli olcekteki skorlari normalize etmeye
/// calismak yerine yalnizca *siralamayi* kullanir; pratikte skor
/// normalizasyonundan daha kararlidir.
fn reciprocal_rank_fusion(
    dense: &[(usize, f32)],
    sparse: &[(usize, f32)],
    k: f32,
) -> Vec<(usize, f32)> {
    let mut acc: HashMap<usize, f32> = HashMap::new();
    for (rank, (idx, _)) in dense.iter().enumerate() {
        *acc.entry(*idx).or_insert(0.0) += 1.0 / (k + rank as f32 + 1.0);
    }
    for (rank, (idx, _)) in sparse.iter().enumerate() {
        *acc.entry(*idx).or_insert(0.0) += 1.0 / (k + rank as f32 + 1.0);
    }
    let mut out: Vec<(usize, f32)> = acc.into_iter().collect();
    out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    out
}

/// Maximal Marginal Relevance: alaka ile cesitliligi dengeler.
fn mmr_select<'a>(
    candidates: &[usize],
    vec_of: &dyn Fn(usize) -> &'a [f32],
    query: &[f32],
    lambda: f32,
    k: usize,
) -> Vec<usize> {
    if candidates.len() <= k {
        return candidates.to_vec();
    }
    let mut selected: Vec<usize> = Vec::with_capacity(k);
    let mut pool: Vec<usize> = candidates.to_vec();

    while selected.len() < k && !pool.is_empty() {
        let mut best_pos = 0usize;
        let mut best_score = f32::NEG_INFINITY;
        for (pos, &cand) in pool.iter().enumerate() {
            let relevance = cosine(vec_of(cand), query);
            let redundancy = selected
                .iter()
                .map(|&s| cosine(vec_of(cand), vec_of(s)))
                .fold(0f32, f32::max);
            let score = lambda * relevance - (1.0 - lambda) * redundancy;
            if score > best_score {
                best_score = score;
                best_pos = pos;
            }
        }
        selected.push(pool.remove(best_pos));
    }
    selected
}

fn build_reranker(cfg: &EmbeddingConfig) -> Option<parking_lot::Mutex<fastembed::TextRerank>> {
    if cfg.reranker_model.trim().is_empty() {
        return None;
    }
    let model = match cfg.reranker_model.to_ascii_lowercase().as_str() {
        "bge-reranker-base" => fastembed::RerankerModel::BGERerankerBase,
        "jina-reranker-v1-turbo-en" => fastembed::RerankerModel::JINARerankerV1TurboEn,
        other => {
            tracing::warn!(model = other, "bilinmeyen reranker modeli, devre disi");
            return None;
        }
    };
    let opts = fastembed::RerankInitOptions::new(model)
        .with_cache_dir(cfg.cache_dir.clone())
        .with_show_download_progress(!cfg.offline);
    match fastembed::TextRerank::try_new(opts) {
        Ok(r) => {
            tracing::info!(model = %cfg.reranker_model, "cross-encoder reranker hazir");
            Some(parking_lot::Mutex::new(r))
        }
        Err(e) => {
            tracing::warn!(error = %e, "reranker yuklenemedi, RRF skorlari kullanilacak");
            None
        }
    }
}

fn rerank_in_place(
    reranker: &parking_lot::Mutex<fastembed::TextRerank>,
    query: &str,
    items: &mut [ScoredChunk],
) {
    let docs: Vec<&str> = items.iter().map(|s| s.chunk.text.as_str()).collect();
    let mut guard = reranker.lock();
    match guard.rerank(query, docs, false, None) {
        Ok(results) => {
            for r in results {
                if let Some(slot) = items.get_mut(r.index) {
                    // Cross-encoder logit'i 0..1 araligina sikistirilir.
                    let normalized = 1.0 / (1.0 + (-r.score).exp());
                    slot.rerank_score = Some(normalized);
                    slot.score = normalized;
                }
            }
        }
        Err(e) => tracing::warn!(error = %e, "yeniden siralama basarisiz, RRF skorlari korunuyor"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rrf_prefers_items_ranked_by_both() {
        let dense = vec![(1usize, 0.9f32), (2, 0.8), (3, 0.7)];
        let sparse = vec![(3usize, 5.0f32), (1, 4.0)];
        let fused = reciprocal_rank_fusion(&dense, &sparse, 60.0);
        assert_eq!(fused[0].0, 1);
    }

    #[test]
    fn mmr_drops_near_duplicates() {
        let vectors: Vec<Vec<f32>> = vec![vec![1.0, 0.0], vec![0.99, 0.14], vec![0.0, 1.0]];
        let query = vec![1.0, 0.0];
        // Dusuk lambda cesitliligi agirlikli oldugu icin (lambda->0 = saf
        // cesitlilik, lambda->1 = saf alaka) burada 0.3 kullanilir; 0.6 gibi
        // yuksek bir deger alaka baskin oldugundan neredeyse-yinelenen ogeyi
        // eleyemez (bu, MMR formulunun beklenen davranisidir, hata degildir).
        let picked = mmr_select(&[0, 1, 2], &|i| vectors[i].as_slice(), &query, 0.3, 2);
        assert!(picked.contains(&0));
        assert!(picked.contains(&2), "cesitlilik saglanmadi: {picked:?}");
    }
}
