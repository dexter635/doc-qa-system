//! Prometheus metrikleri.
//!
//! Amac: kapali ag/on-prem dagitimda bile standart bir gozlemlenebilirlik
//! yuzeyi sunmak. `/metrics` ucnoktasi kimlik dogrulamasiz servis edilir
//! (Prometheus scraping'in yaygin pratigi budur); agda bu yolun sadece
//! izleme sistemine acik olmasi gerekir (ag seviyesinde kisitlanmalidir).

use std::time::Duration;

use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};

/// Global Prometheus kaydediciyi kurar ve metin ciktisini uretmek icin
/// kullanilacak tutamaci (handle) dondurur.
pub fn install() -> PrometheusHandle {
    let builder = PrometheusBuilder::new();
    builder
        .set_buckets_for_metric(
            metrics_exporter_prometheus::Matcher::Full("http_request_duration_seconds".to_string()),
            &[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
        )
        .expect("gecerli histogram kovalari")
        .install_recorder()
        .expect("prometheus kaydedicisi yalnizca bir kez kurulmali")
}

/// HTTP istek sayaci ve gecikme histogramini kaydeder.
pub fn record_http(method: &str, path: &str, status: u16, elapsed: Duration) {
    let status_label = status.to_string();
    metrics::counter!(
        "http_requests_total",
        "method" => method.to_string(),
        "path" => path.to_string(),
        "status" => status_label.clone(),
    )
    .increment(1);
    metrics::histogram!(
        "http_request_duration_seconds",
        "method" => method.to_string(),
        "path" => path.to_string(),
    )
    .record(elapsed.as_secs_f64());
}

/// RAG isleminin (soru-cevap) sonucunu metriklere yazar.
pub fn record_ask(kind: &str, cached: bool, agent_steps: usize, latency: Duration) {
    metrics::counter!("dq_ask_total", "kind" => kind.to_string(), "cached" => cached.to_string())
        .increment(1);
    metrics::histogram!("dq_ask_duration_seconds", "kind" => kind.to_string())
        .record(latency.as_secs_f64());
    metrics::histogram!("dq_agent_steps").record(agent_steps as f64);
}

pub fn record_ingest(status: &str, duration: Duration) {
    metrics::counter!("dq_ingest_total", "status" => status.to_string()).increment(1);
    metrics::histogram!("dq_ingest_duration_seconds").record(duration.as_secs_f64());
}

/// Her istegi sayar ve gecikmesini olcer.
///
/// Not: yol etiketi olarak istegin ham path'i kullanilir (rota sablonu degil).
/// Bu uygulamada belge/denetim kimlikleri sinirli sayida oldugu icin
/// kardinalite riski dusuktur; buyuk olcekli kurulumlarda `MatchedPath`
/// normalizasyonu eklenebilir.
pub async fn metrics_mw(
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    let method = req.method().to_string();
    let path = req.uri().path().to_string();
    let started = std::time::Instant::now();
    let response = next.run(req).await;
    record_http(
        &method,
        &path,
        response.status().as_u16(),
        started.elapsed(),
    );
    response
}
