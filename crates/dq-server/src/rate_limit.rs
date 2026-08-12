//! Sabit pencereli, kullanici/IP basina istek hiz siniri.
//!
//! Amac: acik veya kaba kuvvet (brute-force) denemelerinin ve yanlislikla
//! olusan istek firtinalarinin sistemi (ozellikle LLM ve OCR gibi pahali
//! islemleri) tuketmesini onlemek (OWASP API4:2023 - Unrestricted Resource
//! Consumption).

use std::collections::HashMap;
use std::net::SocketAddr;
use std::time::{Duration, Instant};

use axum::extract::{ConnectInfo, Request, State};
use axum::middleware::Next;
use axum::response::Response;
use parking_lot::Mutex;

use dq_core::DqError;

use crate::error::ApiError;
use crate::state::AppState;

pub struct RateLimiter {
    limit_per_min: u32,
    buckets: Mutex<HashMap<String, Vec<Instant>>>,
}

impl RateLimiter {
    pub fn new(limit_per_min: u32) -> Self {
        Self {
            limit_per_min,
            buckets: Mutex::new(HashMap::new()),
        }
    }

    /// `Ok(())` izin verildi; `Err(saniye)` ise bu kadar saniye sonra tekrar denenmeli.
    fn check(&self, key: &str) -> std::result::Result<(), u64> {
        if self.limit_per_min == 0 {
            return Ok(());
        }
        let now = Instant::now();
        let window = Duration::from_secs(60);
        let mut buckets = self.buckets.lock();
        let entry = buckets.entry(key.to_string()).or_default();
        entry.retain(|t| now.duration_since(*t) < window);
        if entry.len() as u32 >= self.limit_per_min {
            let oldest = entry.first().copied().unwrap_or(now);
            let retry = window.saturating_sub(now.duration_since(oldest)).as_secs().max(1);
            return Err(retry);
        }
        entry.push(now);
        Ok(())
    }
}

pub async fn rate_limit_mw(
    State(state): State<AppState>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    req: Request,
    next: Next,
) -> std::result::Result<Response, ApiError> {
    // Oturum acmis kullanicilar icin belirtec, acilmamislar icin istemci IP'si
    // anahtar olarak kullanilir; boylece paylasilan bir NAT arkasindaki
    // kullanicilar birbirini kilitlemez.
    let key = req
        .headers()
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
        .unwrap_or_else(|| addr.ip().to_string());

    match state.limiter.check(&key) {
        Ok(()) => Ok(next.run(req).await),
        Err(retry_after_secs) => Err(ApiError(DqError::RateLimited { retry_after_secs })),
    }
}
