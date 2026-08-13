use std::sync::Arc;

use dq_core::config::AppConfig;
use dq_rag::Pipeline;
use metrics_exporter_prometheus::PrometheusHandle;

use crate::auth::AuthService;
use crate::rate_limit::RateLimiter;

#[derive(Clone)]
pub struct AppState {
    pub pipeline: Arc<Pipeline>,
    pub cfg: Arc<AppConfig>,
    pub auth: Arc<AuthService>,
    pub limiter: Arc<RateLimiter>,
    pub metrics: Arc<PrometheusHandle>,
}
