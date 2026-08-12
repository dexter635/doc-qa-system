//! Cok katmanli cevap onbellegi.
//!
//! Katmanlar:
//! 1. Sureç ici LRU (TTL'li) - en hizli, yeniden baslatmada kaybolur.
//! 2. SQLite kalici onbellek - yeniden baslatmayi asar.
//! 3. Anlamsal onbellek - "yağ değişimi ne zaman?" ile "yağ değişim aralığı
//!    nedir?" ayni cevabi paylasabilir.
//!
//! Guvenlik notu: onbellek anahtari kullanicinin *yetki seviyesini* ve belge
//! filtresini icerir. Aksi halde dusuk yetkili bir kullanici, yuksek yetkili
//! birinin sorgusuna verilen cevabi onbellekten okuyabilirdi.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use dq_core::config::CacheConfig;
use dq_core::ids::key_of;
use dq_core::text::tr_lower;
use dq_core::{Answer, Classification, Result};
use parking_lot::Mutex;
use uuid::Uuid;

use crate::embed::cosine;
use crate::store::Store;

pub struct CacheKey {
    pub query: String,
    pub clearance: Classification,
    pub doc_filter: Vec<Uuid>,
    pub model: String,
}

impl CacheKey {
    pub fn scope(&self) -> String {
        let mut docs: Vec<String> = self.doc_filter.iter().map(|d| d.to_string()).collect();
        docs.sort();
        format!("{}|{}|{}", self.model, self.clearance as i64, docs.join(","))
    }

    pub fn hash(&self) -> String {
        let normalized = normalize_query(&self.query);
        key_of(&[&normalized, &self.scope()])
    }
}

/// Yazim/bosluk farklarinin onbellegi kacirmasini engeller.
fn normalize_query(q: &str) -> String {
    let lower = tr_lower(q);
    lower
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .trim_end_matches(['?', '.', '!'])
        .to_string()
}

struct MemEntry {
    value: String,
    at: Instant,
}

pub struct AnswerCache {
    cfg: CacheConfig,
    mem: Mutex<HashMap<String, MemEntry>>,
    hits: Mutex<CacheStats>,
}

#[derive(Debug, Default, Clone, Copy, serde::Serialize)]
pub struct CacheStats {
    pub exact_hits: u64,
    pub semantic_hits: u64,
    pub misses: u64,
}

pub enum CacheHit {
    Exact(Answer),
    Semantic(Answer, f32),
}

impl AnswerCache {
    pub fn new(cfg: CacheConfig) -> Self {
        Self {
            cfg,
            mem: Mutex::new(HashMap::new()),
            hits: Mutex::new(CacheStats::default()),
        }
    }

    pub fn stats(&self) -> CacheStats {
        *self.hits.lock()
    }

    pub fn get(
        &self,
        store: &Store,
        key: &CacheKey,
        query_vec: Option<&[f32]>,
    ) -> Result<Option<CacheHit>> {
        if !self.cfg.enabled {
            return Ok(None);
        }
        let hashed = key.hash();

        if let Some(v) = self.mem_get(&hashed) {
            if let Ok(answer) = serde_json::from_str::<Answer>(&v) {
                self.hits.lock().exact_hits += 1;
                return Ok(Some(CacheHit::Exact(answer)));
            }
        }

        if let Some(v) = store.cache_get(&hashed)? {
            self.mem_put(&hashed, &v);
            if let Ok(answer) = serde_json::from_str::<Answer>(&v) {
                self.hits.lock().exact_hits += 1;
                return Ok(Some(CacheHit::Exact(answer)));
            }
        }

        if self.cfg.semantic_enabled {
            if let Some(qv) = query_vec {
                let candidates =
                    store.cache_candidates(&key.scope(), key.clearance, self.cfg.ttl_secs)?;
                let mut best: Option<(f32, String)> = None;
                for (_k, vec, answer) in candidates {
                    if vec.len() != qv.len() {
                        continue;
                    }
                    let sim = cosine(&vec, qv);
                    if sim >= self.cfg.semantic_threshold
                        && best.as_ref().map(|(b, _)| sim > *b).unwrap_or(true)
                    {
                        best = Some((sim, answer));
                    }
                }
                if let Some((sim, answer)) = best {
                    if let Ok(a) = serde_json::from_str::<Answer>(&answer) {
                        self.hits.lock().semantic_hits += 1;
                        return Ok(Some(CacheHit::Semantic(a, sim)));
                    }
                }
            }
        }

        self.hits.lock().misses += 1;
        Ok(None)
    }

    pub fn put(
        &self,
        store: &Store,
        key: &CacheKey,
        answer: &Answer,
        query_vec: Option<&[f32]>,
    ) -> Result<()> {
        if !self.cfg.enabled {
            return Ok(());
        }
        // Reddedilen veya bloklanan cevaplar onbelleklenmez: kaynak eklendikce
        // ayni soru dogru cevabi almalidir.
        if answer.kind != dq_core::AnswerKind::Grounded {
            return Ok(());
        }
        let hashed = key.hash();
        let payload = serde_json::to_string(answer)?;
        self.mem_put(&hashed, &payload);
        store.cache_put(
            &hashed,
            &key.scope(),
            &key.query,
            &payload,
            key.clearance,
            query_vec,
        )
    }

    /// Belge eklendiginde/silindiginde tum onbellek gecersizlesir.
    pub fn invalidate_all(&self, store: &Store) -> Result<()> {
        self.mem.lock().clear();
        store.cache_clear()
    }

    fn mem_get(&self, key: &str) -> Option<String> {
        let mut mem = self.mem.lock();
        let ttl = Duration::from_secs(self.cfg.ttl_secs);
        match mem.get(key) {
            Some(e) if e.at.elapsed() < ttl => Some(e.value.clone()),
            Some(_) => {
                mem.remove(key);
                None
            }
            None => None,
        }
    }

    fn mem_put(&self, key: &str, value: &str) {
        let mut mem = self.mem.lock();
        if mem.len() as u64 >= self.cfg.memory_capacity {
            // En eski girdiyi dusur (basit LRU yaklasimı).
            if let Some(oldest) = mem
                .iter()
                .min_by_key(|(_, v)| v.at)
                .map(|(k, _)| k.clone())
            {
                mem.remove(&oldest);
            }
        }
        mem.insert(
            key.to_string(),
            MemEntry {
                value: value.to_string(),
                at: Instant::now(),
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_ignores_case_and_punctuation() {
        let a = CacheKey {
            query: "Yağ değişimi ne zaman?".into(),
            clearance: Classification::Restricted,
            doc_filter: vec![],
            model: "m".into(),
        };
        let b = CacheKey {
            query: "yağ  değişimi ne zaman".into(),
            clearance: Classification::Restricted,
            doc_filter: vec![],
            model: "m".into(),
        };
        assert_eq!(a.hash(), b.hash());
    }

    #[test]
    fn key_separates_clearance_levels() {
        let low = CacheKey {
            query: "plan".into(),
            clearance: Classification::Restricted,
            doc_filter: vec![],
            model: "m".into(),
        };
        let high = CacheKey {
            query: "plan".into(),
            clearance: Classification::Secret,
            doc_filter: vec![],
            model: "m".into(),
        };
        assert_ne!(low.hash(), high.hash());
    }

    #[test]
    fn doc_filter_order_does_not_matter() {
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        let k1 = CacheKey {
            query: "q".into(),
            clearance: Classification::Unclassified,
            doc_filter: vec![a, b],
            model: "m".into(),
        };
        let k2 = CacheKey {
            query: "q".into(),
            clearance: Classification::Unclassified,
            doc_filter: vec![b, a],
            model: "m".into(),
        };
        assert_eq!(k1.hash(), k2.hash());
    }
}
