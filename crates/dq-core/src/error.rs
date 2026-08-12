use std::fmt;

/// Sistemin tum katmanlarinda kullanilan hata tipi.
///
/// Hatalar HTTP katmaninda `status_code()` ile dogrudan cevaba cevrilir;
/// bu sayede sunucu tarafinda tekrar eden `match` bloklari olusmaz.
#[derive(Debug, thiserror::Error)]
pub enum DqError {
    #[error("gecersiz istek: {0}")]
    BadRequest(String),

    #[error("kaynak bulunamadi: {0}")]
    NotFound(String),

    #[error("yetkisiz erisim: {0}")]
    Unauthorized(String),

    #[error("bu islem icin yetkiniz yok: {0}")]
    Forbidden(String),

    #[error("desteklenmeyen dosya tipi: {0}")]
    UnsupportedMedia(String),

    #[error("dosya cok buyuk: {size} bayt (limit {limit} bayt)")]
    PayloadTooLarge { size: u64, limit: u64 },

    #[error("guardrail bloklandi: {0}")]
    GuardrailBlocked(String),

    #[error("hiz limiti asildi, {retry_after_secs} saniye sonra deneyin")]
    RateLimited { retry_after_secs: u64 },

    #[error("belge isleme hatasi: {0}")]
    Ingest(String),

    #[error("OCR hatasi: {0}")]
    Ocr(String),

    #[error("gomme (embedding) hatasi: {0}")]
    Embedding(String),

    #[error("depolama hatasi: {0}")]
    Storage(String),

    #[error("LLM hatasi: {0}")]
    Llm(String),

    #[error("konfigurasyon hatasi: {0}")]
    Config(String),

    #[error("dahili hata: {0}")]
    Internal(String),
}

impl DqError {
    pub fn internal(e: impl fmt::Display) -> Self {
        DqError::Internal(e.to_string())
    }

    /// Kullaniciya donen HTTP durum kodu.
    pub fn status_code(&self) -> u16 {
        match self {
            DqError::BadRequest(_) => 400,
            DqError::Unauthorized(_) => 401,
            DqError::Forbidden(_) => 403,
            DqError::NotFound(_) => 404,
            DqError::PayloadTooLarge { .. } => 413,
            DqError::UnsupportedMedia(_) => 415,
            DqError::GuardrailBlocked(_) => 422,
            DqError::RateLimited { .. } => 429,
            DqError::Ingest(_) | DqError::Ocr(_) => 422,
            _ => 500,
        }
    }

    /// Makine tarafindan okunabilir hata kodu (istemci bu koda gore dallanir).
    pub fn code(&self) -> &'static str {
        match self {
            DqError::BadRequest(_) => "bad_request",
            DqError::NotFound(_) => "not_found",
            DqError::Unauthorized(_) => "unauthorized",
            DqError::Forbidden(_) => "forbidden",
            DqError::UnsupportedMedia(_) => "unsupported_media_type",
            DqError::PayloadTooLarge { .. } => "payload_too_large",
            DqError::GuardrailBlocked(_) => "guardrail_blocked",
            DqError::RateLimited { .. } => "rate_limited",
            DqError::Ingest(_) => "ingest_failed",
            DqError::Ocr(_) => "ocr_failed",
            DqError::Embedding(_) => "embedding_failed",
            DqError::Storage(_) => "storage_failed",
            DqError::Llm(_) => "llm_failed",
            DqError::Config(_) => "config_error",
            DqError::Internal(_) => "internal_error",
        }
    }

    /// 5xx hatalarin detayi kullaniciya sizmamalidir (bilgi ifsasi riski).
    pub fn public_message(&self) -> String {
        if self.status_code() >= 500 {
            "Beklenmeyen bir sunucu hatasi olustu. Islem kimligi ile birlikte sistem yoneticisine basvurun.".to_string()
        } else {
            self.to_string()
        }
    }
}

impl From<anyhow::Error> for DqError {
    fn from(e: anyhow::Error) -> Self {
        DqError::Internal(format!("{e:#}"))
    }
}

impl From<std::io::Error> for DqError {
    fn from(e: std::io::Error) -> Self {
        DqError::Internal(format!("io: {e}"))
    }
}

impl From<serde_json::Error> for DqError {
    fn from(e: serde_json::Error) -> Self {
        DqError::Internal(format!("json: {e}"))
    }
}

pub type Result<T> = std::result::Result<T, DqError>;
