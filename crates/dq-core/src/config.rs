use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{DqError, Result};
use crate::types::Classification;

/// Uygulamanin tum ayarlari. `config/default.toml` dosyasindan okunur,
/// `DQ_` onekli ortam degiskenleri ile ezilir.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AppConfig {
    pub server: ServerConfig,
    pub storage: StorageConfig,
    pub ingest: IngestConfig,
    pub ocr: OcrConfig,
    pub embedding: EmbeddingConfig,
    pub retrieval: RetrievalConfig,
    pub cache: CacheConfig,
    pub llm: LlmConfig,
    pub guardrails: GuardrailConfig,
    pub auth: AuthConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    /// CORS icin izin verilen originler. Bos = ayni origin disinda hepsi kapali.
    pub allowed_origins: Vec<String>,
    /// Tek istekte kabul edilen en buyuk govde (bayt).
    pub max_body_bytes: u64,
    /// Dakikada kullanici basina istek limiti.
    pub rate_limit_per_min: u32,
    pub request_timeout_secs: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct StorageConfig {
    /// SQLite veritabani yolu.
    pub db_path: PathBuf,
    /// Yuklenen orijinal dosyalarin saklandigi dizin.
    pub blob_dir: PathBuf,
    /// Model dosyalarinin (ONNX/rten) bulundugu dizin - cevrimdisi calisma icin.
    pub model_dir: PathBuf,
    /// HNSW indeksinin diske yazildigi dizin.
    pub index_dir: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct IngestConfig {
    pub max_file_bytes: u64,
    pub max_pages: usize,
    /// Hedef chunk buyuklugu (yaklasik token).
    pub chunk_tokens: usize,
    /// Ardisik chunk'lar arasi ortusme (token).
    pub chunk_overlap_tokens: usize,
    /// Bu uzunlugun altindaki chunk'lar bir oncekine eklenir.
    pub min_chunk_tokens: usize,
    /// PDF sayfasindan cikan metin bu karakter sayisinin altindaysa OCR'a dusulur.
    pub ocr_fallback_char_threshold: usize,
    /// PDF sayfasi raster'a cevrilirken kullanilacak DPI.
    pub render_dpi: f32,
    pub allowed_mime: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OcrConfig {
    /// "ocrs" (saf Rust, cevrimdisi) | "tesseract" (harici binary) | "none"
    pub engine: String,
    /// Tesseract kullanilacaksa dil listesi.
    pub tesseract_langs: String,
    pub tesseract_bin: String,
    /// Bu guvenin altindaki satirlar atilir.
    pub min_line_confidence: f32,
    /// Goruntu on isleme (deskew/binarize) uygulansin mi?
    pub preprocess: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EmbeddingConfig {
    /// fastembed model adi (cok dilli oneri: multilingual-e5-small).
    pub model: String,
    pub dim: usize,
    /// E5 ailesi icin gerekli onekler.
    pub query_prefix: String,
    pub passage_prefix: String,
    pub batch_size: usize,
    /// Modelin indirilecegi/okunacagi dizin (air-gapped kurulumda onceden doldurulur).
    pub cache_dir: PathBuf,
    /// Ag erisimi kapali ise model dizinde yoksa hata verilir.
    pub offline: bool,
    /// Cross-encoder yeniden siralayici (bos ise devre disi).
    pub reranker_model: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RetrievalConfig {
    /// Dense aramadan alinacak aday sayisi.
    pub dense_top_k: usize,
    /// BM25 aramadan alinacak aday sayisi.
    pub sparse_top_k: usize,
    /// Fusion sonrasi rerank'e girecek aday sayisi.
    pub rerank_candidates: usize,
    /// LLM baglamina girecek nihai chunk sayisi.
    pub final_top_k: usize,
    /// Reciprocal Rank Fusion sabiti.
    pub rrf_k: f32,
    /// Bu skorun altindaki adaylar elenir (0..1 normalize skor).
    pub min_score: f32,
    /// Secilen chunk'in komsularini baglama ekle (context expansion).
    pub neighbor_window: usize,
    /// Baglamin toplam token butcesi.
    pub context_token_budget: usize,
    /// MMR cesitlilik agirligi (1.0 = sadece benzerlik, 0.0 = sadece cesitlilik).
    pub mmr_lambda: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CacheConfig {
    pub enabled: bool,
    /// Bellek ici tam eslesme onbellegi kapasitesi.
    pub memory_capacity: u64,
    pub ttl_secs: u64,
    /// Anlamsal onbellek: sorgu vektoru bu benzerligin uzerindeyse cevap yeniden kullanilir.
    pub semantic_enabled: bool,
    pub semantic_threshold: f32,
    /// Gomme vektorleri de onbelleklenir (ayni belge tekrar yuklenirse).
    pub embedding_cache_enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LlmConfig {
    /// OpenAI uyumlu yerel sunucu adresi (llama.cpp server / Ollama / vLLM).
    pub base_url: String,
    pub model: String,
    /// Bos birakilirsa Authorization basligi gonderilmez (yerel kurulum).
    pub api_key: String,
    pub temperature: f32,
    pub top_p: f32,
    pub max_tokens: u32,
    pub timeout_secs: u64,
    /// LLM erisilemezse cikarimsal (extractive) yedek cevap uretilsin mi?
    pub extractive_fallback: bool,
    /// Saglik kontrolu icin baslangicta yoklama yapilsin mi?
    pub probe_on_start: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GuardrailConfig {
    /// Girdi tarafi
    pub max_query_chars: usize,
    pub block_prompt_injection: bool,
    pub injection_threshold: f32,
    pub block_pii_in_query: bool,
    /// Cikti tarafi
    pub require_citations: bool,
    /// Cevabin desteklenme orani bu esigin altindaysa cevap reddedilir.
    pub min_support_ratio: f32,
    /// En iyi kaynak skoru bu esigin altindaysa "bilgi yok" denir.
    pub min_top_score: f32,
    /// Ciktida PII maskelensin mi?
    pub redact_pii_in_answer: bool,
    /// Cevap, kaynaklarin en yuksek gizlilik derecesini tasir ve damgalanir.
    pub stamp_classification: bool,
    /// Kullanici yetkisinin uzerindeki belgeler aramadan tamamen cikarilir.
    pub enforce_clearance: bool,
    /// Yasakli konu/anahtar kelime listesi (regex).
    pub denylist_patterns: Vec<String>,
    /// Cevap dili sorgu dili ile ayni olmali.
    pub enforce_language_match: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AuthConfig {
    pub enabled: bool,
    /// JWT imzalama anahtari. Uretimde MUTLAKA ortam degiskeninden verilmeli.
    pub jwt_secret: String,
    pub token_ttl_secs: u64,
    /// Baslangicta olusturulacak yerel kullanicilar.
    pub users: Vec<SeedUser>,
    /// Auth kapaliyken kullanilan varsayilan yetki seviyesi.
    pub anonymous_clearance: Classification,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeedUser {
    pub username: String,
    /// Argon2 hash. Bos ise `password` alanindaki duz metin ilk aciliste hashlenir.
    #[serde(default)]
    pub password_hash: String,
    #[serde(default)]
    pub password: String,
    pub clearance: Classification,
    #[serde(default)]
    pub roles: Vec<String>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".into(),
            port: 8080,
            allowed_origins: vec!["http://127.0.0.1:8081".into(), "http://localhost:8081".into()],
            max_body_bytes: 64 * 1024 * 1024,
            rate_limit_per_min: 60,
            request_timeout_secs: 300,
        }
    }
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            db_path: PathBuf::from("data/docqa.sqlite3"),
            blob_dir: PathBuf::from("data/blobs"),
            model_dir: PathBuf::from("models"),
            index_dir: PathBuf::from("data/index"),
        }
    }
}

impl Default for IngestConfig {
    fn default() -> Self {
        Self {
            max_file_bytes: 50 * 1024 * 1024,
            max_pages: 500,
            chunk_tokens: 380,
            chunk_overlap_tokens: 64,
            min_chunk_tokens: 40,
            ocr_fallback_char_threshold: 120,
            render_dpi: 200.0,
            allowed_mime: vec![
                "application/pdf".into(),
                "image/jpeg".into(),
                "image/png".into(),
            ],
        }
    }
}

impl Default for OcrConfig {
    fn default() -> Self {
        Self {
            engine: "ocrs".into(),
            tesseract_langs: "tur+eng".into(),
            tesseract_bin: "tesseract".into(),
            min_line_confidence: 0.35,
            preprocess: true,
        }
    }
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            model: "multilingual-e5-small".into(),
            dim: 384,
            query_prefix: "query: ".into(),
            passage_prefix: "passage: ".into(),
            batch_size: 16,
            cache_dir: PathBuf::from("models/embeddings"),
            offline: false,
            reranker_model: String::new(),
        }
    }
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            dense_top_k: 40,
            sparse_top_k: 40,
            rerank_candidates: 24,
            final_top_k: 6,
            rrf_k: 60.0,
            min_score: 0.18,
            neighbor_window: 1,
            context_token_budget: 3000,
            mmr_lambda: 0.72,
        }
    }
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            memory_capacity: 512,
            ttl_secs: 3600,
            semantic_enabled: true,
            semantic_threshold: 0.96,
            embedding_cache_enabled: true,
        }
    }
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            base_url: "http://127.0.0.1:8000/v1".into(),
            model: "qwen2.5-7b-instruct".into(),
            api_key: String::new(),
            temperature: 0.1,
            top_p: 0.9,
            max_tokens: 900,
            timeout_secs: 120,
            extractive_fallback: true,
            probe_on_start: true,
        }
    }
}

impl Default for GuardrailConfig {
    fn default() -> Self {
        Self {
            max_query_chars: 1000,
            block_prompt_injection: true,
            injection_threshold: 0.5,
            block_pii_in_query: false,
            require_citations: true,
            min_support_ratio: 0.5,
            min_top_score: 0.22,
            redact_pii_in_answer: true,
            stamp_classification: true,
            enforce_clearance: true,
            denylist_patterns: vec![],
            enforce_language_match: true,
        }
    }
}

impl Default for AuthConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            jwt_secret: String::new(),
            token_ttl_secs: 8 * 3600,
            users: vec![],
            anonymous_clearance: Classification::Restricted,
        }
    }
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            server: Default::default(),
            storage: Default::default(),
            ingest: Default::default(),
            ocr: Default::default(),
            embedding: Default::default(),
            retrieval: Default::default(),
            cache: Default::default(),
            llm: Default::default(),
            guardrails: Default::default(),
            auth: Default::default(),
        }
    }
}

impl AppConfig {
    /// Dosyadan yukler; dosya yoksa varsayilanlari kullanir. Ardindan
    /// `DQ_` onekli ortam degiskenlerini uygular ve dogrular.
    pub fn load(path: Option<&Path>) -> Result<Self> {
        let mut cfg = match path {
            Some(p) if p.exists() => {
                let raw = std::fs::read_to_string(p)
                    .map_err(|e| DqError::Config(format!("{} okunamadi: {e}", p.display())))?;
                toml::from_str::<AppConfig>(&raw)
                    .map_err(|e| DqError::Config(format!("{} ayristirilamadi: {e}", p.display())))?
            }
            Some(p) => {
                tracing::warn!(path = %p.display(), "config dosyasi yok, varsayilanlar kullaniliyor");
                AppConfig::default()
            }
            None => AppConfig::default(),
        };
        cfg.apply_env();
        cfg.validate()?;
        Ok(cfg)
    }

    /// Konteyner/servis kurulumlarinda sik kullanilan alanlar icin env override.
    fn apply_env(&mut self) {
        if let Ok(v) = std::env::var("DQ_SERVER_HOST") {
            self.server.host = v;
        }
        if let Some(v) = env_parse::<u16>("DQ_SERVER_PORT") {
            self.server.port = v;
        }
        if let Ok(v) = std::env::var("DQ_LLM_BASE_URL") {
            self.llm.base_url = v;
        }
        if let Ok(v) = std::env::var("DQ_LLM_MODEL") {
            self.llm.model = v;
        }
        if let Ok(v) = std::env::var("DQ_LLM_API_KEY") {
            self.llm.api_key = v;
        }
        if let Ok(v) = std::env::var("DQ_JWT_SECRET") {
            self.auth.jwt_secret = v;
        }
        if let Ok(v) = std::env::var("DQ_DB_PATH") {
            self.storage.db_path = PathBuf::from(v);
        }
        if let Ok(v) = std::env::var("DQ_MODEL_DIR") {
            self.storage.model_dir = PathBuf::from(v);
        }
        if let Ok(v) = std::env::var("DQ_OCR_ENGINE") {
            self.ocr.engine = v;
        }
        if let Some(v) = env_parse::<bool>("DQ_EMBEDDING_OFFLINE") {
            self.embedding.offline = v;
        }
        if let Some(v) = env_parse::<bool>("DQ_AUTH_ENABLED") {
            self.auth.enabled = v;
        }
    }

    fn validate(&self) -> Result<()> {
        if self.auth.enabled && self.auth.jwt_secret.len() < 32 {
            return Err(DqError::Config(
                "auth.jwt_secret en az 32 karakter olmali (DQ_JWT_SECRET ortam degiskeni ile verin)"
                    .into(),
            ));
        }
        if self.embedding.dim == 0 {
            return Err(DqError::Config("embedding.dim sifir olamaz".into()));
        }
        if self.ingest.chunk_overlap_tokens >= self.ingest.chunk_tokens {
            return Err(DqError::Config(
                "ingest.chunk_overlap_tokens, chunk_tokens degerinden kucuk olmali".into(),
            ));
        }
        if !(0.0..=1.0).contains(&self.retrieval.mmr_lambda) {
            return Err(DqError::Config("retrieval.mmr_lambda 0..1 araliginda olmali".into()));
        }
        if self.retrieval.final_top_k == 0 {
            return Err(DqError::Config("retrieval.final_top_k sifir olamaz".into()));
        }
        for p in &self.guardrails.denylist_patterns {
            regex::Regex::new(p)
                .map_err(|e| DqError::Config(format!("gecersiz denylist regex '{p}': {e}")))?;
        }
        Ok(())
    }

    /// Calisma dizinlerini olusturur.
    pub fn ensure_dirs(&self) -> Result<()> {
        for d in [
            &self.storage.blob_dir,
            &self.storage.model_dir,
            &self.storage.index_dir,
            &self.embedding.cache_dir,
        ] {
            std::fs::create_dir_all(d)
                .map_err(|e| DqError::Config(format!("{} olusturulamadi: {e}", d.display())))?;
        }
        if let Some(parent) = self.storage.db_path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| DqError::Config(format!("{} olusturulamadi: {e}", parent.display())))?;
            }
        }
        Ok(())
    }
}

fn env_parse<T: std::str::FromStr>(key: &str) -> Option<T> {
    std::env::var(key).ok().and_then(|v| v.parse::<T>().ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_valid_when_auth_disabled() {
        let mut cfg = AppConfig::default();
        cfg.auth.enabled = false;
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn rejects_weak_jwt_secret() {
        let cfg = AppConfig::default();
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn rejects_overlap_larger_than_chunk() {
        let mut cfg = AppConfig::default();
        cfg.auth.enabled = false;
        cfg.ingest.chunk_overlap_tokens = cfg.ingest.chunk_tokens;
        assert!(cfg.validate().is_err());
    }
}
