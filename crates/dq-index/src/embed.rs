//! Gomme (embedding) katmani.
//!
//! Iki uygulama sunulur:
//! - [`FastEmbedder`]: ONNX uzerinde yerel calisan cok dilli model (uretim).
//! - [`HashEmbedder`]: model dosyasi olmadan calisan deterministik yedek.
//!
//! Yedek uygulamanin varlik sebebi, kapali agda model indirilememesi
//! durumunda sistemin tamamen durmamasidir; kalite dususu API uzerinden
//! acikca raporlanir.

use dq_core::config::EmbeddingConfig;
use dq_core::text::{fold_diacritics, tr_lower};
use dq_core::{DqError, Result};
use parking_lot::Mutex;

pub trait Embedder: Send + Sync {
    fn dim(&self) -> usize;
    fn model_name(&self) -> String;
    /// Modelin gercek anlamsal model mi yoksa yedek mi oldugunu belirtir.
    fn is_fallback(&self) -> bool {
        false
    }
    fn embed_passages(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
    fn embed_query(&self, text: &str) -> Result<Vec<f32>>;
}

/// Vektoru birim uzunluga getirir; boylece kosinus benzerligi = ic carpim.
pub fn normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Birim vektorler icin kosinus benzerligi.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

// ---------------------------------------------------------------------------
// fastembed tabanli uretim modeli
// ---------------------------------------------------------------------------

pub struct FastEmbedder {
    inner: Mutex<fastembed::TextEmbedding>,
    dim: usize,
    model: String,
    query_prefix: String,
    passage_prefix: String,
    batch_size: usize,
}

impl FastEmbedder {
    pub fn new(cfg: &EmbeddingConfig) -> Result<Self> {
        let model = map_model(&cfg.model)?;
        if cfg.offline {
            std::env::set_var("HF_HUB_OFFLINE", "1");
        }
        let opts = fastembed::TextInitOptions::new(model)
            .with_cache_dir(cfg.cache_dir.clone())
            .with_show_download_progress(!cfg.offline);
        let inner = fastembed::TextEmbedding::try_new(opts)
            .map_err(|e| DqError::Embedding(format!("model yuklenemedi: {e}")))?;

        let embedder = Self {
            inner: Mutex::new(inner),
            dim: cfg.dim,
            model: cfg.model.clone(),
            query_prefix: cfg.query_prefix.clone(),
            passage_prefix: cfg.passage_prefix.clone(),
            batch_size: cfg.batch_size.max(1),
        };
        // Konfigurasyondaki boyut ile modelin gercek boyutu tutmazsa vektor
        // deposu sessizce bozulur; baslangicta dogrula.
        let probe = embedder.embed_query("test")?;
        if probe.len() != cfg.dim {
            return Err(DqError::Embedding(format!(
                "embedding.dim={} ancak model {} boyut uretiyor",
                cfg.dim,
                probe.len()
            )));
        }
        Ok(embedder)
    }

    fn run(&self, inputs: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let mut guard = self.inner.lock();
        let mut out = guard
            .embed(inputs, Some(self.batch_size))
            .map_err(|e| DqError::Embedding(format!("gomme uretilemedi: {e}")))?;
        for v in out.iter_mut() {
            normalize(v);
        }
        Ok(out)
    }
}

impl Embedder for FastEmbedder {
    fn dim(&self) -> usize {
        self.dim
    }

    fn model_name(&self) -> String {
        self.model.clone()
    }

    fn embed_passages(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let inputs: Vec<String> = texts
            .iter()
            .map(|t| format!("{}{}", self.passage_prefix, t))
            .collect();
        self.run(inputs)
    }

    fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let inputs = vec![format!("{}{}", self.query_prefix, text)];
        self.run(inputs)?
            .into_iter()
            .next()
            .ok_or_else(|| DqError::Embedding("bos gomme sonucu".into()))
    }
}

/// Konfigurasyondaki model adini fastembed enum'una eslestirir.
fn map_model(name: &str) -> Result<fastembed::EmbeddingModel> {
    use fastembed::EmbeddingModel as M;
    let key = name.to_ascii_lowercase().replace('_', "-");
    Ok(match key.as_str() {
        "multilingual-e5-small" => M::MultilingualE5Small,
        "multilingual-e5-base" => M::MultilingualE5Base,
        "multilingual-e5-large" => M::MultilingualE5Large,
        "bge-small-en-v1.5" => M::BGESmallENV15,
        "bge-base-en-v1.5" => M::BGEBaseENV15,
        "all-minilm-l6-v2" => M::AllMiniLML6V2,
        other => {
            return Err(DqError::Config(format!(
                "bilinmeyen embedding modeli: {other}"
            )))
        }
    })
}

// ---------------------------------------------------------------------------
// Model gerektirmeyen yedek
// ---------------------------------------------------------------------------

/// Karakter n-gram'larini sabit boyutlu bir uzaya hash'leyen deterministik
/// gomme. Anlamsal degil sozdizimsel benzerlik olcer; yalnizca model
/// erisilemedigi durumda ve testlerde kullanilir.
pub struct HashEmbedder {
    dim: usize,
}

impl HashEmbedder {
    pub fn new(dim: usize) -> Self {
        Self { dim: dim.max(32) }
    }

    fn encode(&self, text: &str) -> Vec<f32> {
        let norm = fold_diacritics(&tr_lower(text));
        let chars: Vec<char> = norm
            .chars()
            .map(|c| if c.is_alphanumeric() { c } else { ' ' })
            .collect();
        let mut v = vec![0f32; self.dim];
        for n in [3usize, 4, 5] {
            if chars.len() < n {
                continue;
            }
            for w in chars.windows(n) {
                if w.iter().all(|c| *c == ' ') {
                    continue;
                }
                let gram: String = w.iter().collect();
                let h = blake3::hash(gram.as_bytes());
                let bytes = h.as_bytes();
                let idx = (u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize)
                    % self.dim;
                let sign = if bytes[4] & 1 == 0 { 1.0 } else { -1.0 };
                v[idx] += sign;
            }
        }
        normalize(&mut v);
        v
    }
}

impl Embedder for HashEmbedder {
    fn dim(&self) -> usize {
        self.dim
    }
    fn model_name(&self) -> String {
        format!("hash-ngram-{}", self.dim)
    }
    fn is_fallback(&self) -> bool {
        true
    }
    fn embed_passages(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|t| self.encode(t)).collect())
    }
    fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        Ok(self.encode(text))
    }
}

/// Konfigurasyona gore gomme motorunu kurar; model yuklenemezse
/// (kapali ag, eksik dosya) yedek motora duser ve durumu loglar.
pub fn build(cfg: &EmbeddingConfig) -> (std::sync::Arc<dyn Embedder>, Option<String>) {
    match FastEmbedder::new(cfg) {
        Ok(e) => {
            tracing::info!(model = %e.model_name(), dim = e.dim(), "gomme modeli hazir");
            (std::sync::Arc::new(e), None)
        }
        Err(e) => {
            let msg =
                format!("Gomme modeli yuklenemedi ({e}); dusuk kaliteli yedek gomme kullaniliyor.");
            tracing::error!("{msg}");
            (std::sync::Arc::new(HashEmbedder::new(cfg.dim)), Some(msg))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_embedder_is_deterministic_and_normalized() {
        let e = HashEmbedder::new(128);
        let a = e.embed_query("periyodik bakım").unwrap();
        let b = e.embed_query("periyodik bakım").unwrap();
        assert_eq!(a, b);
        let norm: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4);
    }

    #[test]
    fn similar_text_scores_higher() {
        let e = HashEmbedder::new(256);
        let q = e.embed_query("periyodik bakım süresi").unwrap();
        let near = e.embed_query("periyodik bakım aralığı").unwrap();
        let far = e
            .embed_query("uçuş kontrol yüzeyleri kalibrasyonu")
            .unwrap();
        assert!(cosine(&q, &near) > cosine(&q, &far));
    }
}
