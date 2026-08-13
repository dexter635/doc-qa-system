//! `DqError` -> HTTP yanit cevrimi.
//!
//! 5xx hatalarda ayrintili neden yalnizca sunucu loguna yazilir; istemciye
//! `public_message()` ile genellestirilmis bir mesaj donduruluer (bilgi
//! ifsasi/OWASP A05 - Security Misconfiguration onlemi).

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use dq_core::DqError;
use serde_json::json;

pub struct ApiError(pub DqError);

impl From<DqError> for ApiError {
    fn from(e: DqError) -> Self {
        ApiError(e)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status =
            StatusCode::from_u16(self.0.status_code()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        if status.is_server_error() {
            tracing::error!(error = %self.0, "sunucu hatasi");
        }
        let mut body = json!({
            "error": {
                "code": self.0.code(),
                "message": self.0.public_message(),
            }
        });
        if let DqError::RateLimited { retry_after_secs } = &self.0 {
            body["error"]["retry_after_secs"] = json!(retry_after_secs);
        }
        (status, Json(body)).into_response()
    }
}

pub type ApiResult<T> = std::result::Result<T, ApiError>;
