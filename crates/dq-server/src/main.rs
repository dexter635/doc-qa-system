mod auth;
mod error;
mod metrics;
mod rate_limit;
mod routes;
mod state;

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use dq_core::config::AppConfig;
use dq_index::{embed, Retriever, Store};
use dq_llm::client::{LlmClient, OpenAiCompatClient};
use dq_rag::Pipeline;
use tower_http::catch_panic::CatchPanicLayer;
use tower_http::cors::{AllowOrigin, CorsLayer};

use crate::auth::AuthService;
use crate::rate_limit::RateLimiter;
use crate::state::AppState;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dq_core::telemetry::init("info,dq_server=debug,tower_http=info");
    let metrics_handle = Arc::new(metrics::install());

    let config_path =
        std::env::var("DQ_CONFIG").unwrap_or_else(|_| "config/default.toml".to_string());
    let cfg = AppConfig::load(Some(std::path::Path::new(&config_path)))?;
    cfg.ensure_dirs()?;

    let store = Arc::new(Store::open(&cfg.storage.db_path)?);

    // Konfigurasyondaki tohum kullanicilar veritabanina yazilir; duz metin
    // parolalar yalnizca ilk aciliste Argon2 ile hash'lenir, disk uzerinde
    // hic saklanmaz.
    for seed in &cfg.auth.users {
        let hash = if !seed.password_hash.is_empty() {
            seed.password_hash.clone()
        } else if !seed.password.is_empty() {
            AuthService::hash_password(&seed.password)?
        } else {
            tracing::warn!(user = %seed.username, "parolasi olmayan tohum kullanici atlandi");
            continue;
        };
        store.upsert_user(&seed.username, &hash, seed.clearance, &seed.roles)?;
    }

    let (embedder, embed_warning) = embed::build(&cfg.embedding);
    if let Some(w) = &embed_warning {
        tracing::error!("{w}");
    }

    let retriever = Arc::new(Retriever::new(
        cfg.retrieval.clone(),
        &cfg.embedding,
        embedder.clone(),
    ));
    retriever.rebuild(&store)?;
    tracing::info!(chunks = retriever.len(), "indeks hazir");

    let llm: Arc<dyn LlmClient> = Arc::new(OpenAiCompatClient::new(&cfg.llm)?);
    if cfg.llm.probe_on_start {
        if llm.healthy().await {
            tracing::info!(base_url = %cfg.llm.base_url, "yerel LLM servisine ulasildi");
        } else {
            tracing::warn!(
                base_url = %cfg.llm.base_url,
                "yerel LLM servisine ulasilamadi; cikarimsal yedek moduna dusulecek"
            );
        }
    }

    let pipeline = Arc::new(Pipeline::new(
        cfg.clone(),
        store.clone(),
        retriever.clone(),
        embedder.clone(),
        llm.clone(),
        embed_warning,
    )?);
    let auth = Arc::new(AuthService::new(
        &cfg.auth.jwt_secret,
        cfg.auth.token_ttl_secs,
        cfg.auth.enabled,
        cfg.auth.anonymous_clearance,
    ));
    let limiter = Arc::new(RateLimiter::new(cfg.server.rate_limit_per_min));

    let state = AppState {
        pipeline,
        cfg: Arc::new(cfg.clone()),
        auth,
        limiter,
        metrics: metrics_handle,
    };

    let app = routes::router(state.clone())
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            rate_limit::rate_limit_mw,
        ))
        .layer(axum::middleware::from_fn(metrics::metrics_mw))
        .layer(tower_http::trace::TraceLayer::new_for_http())
        .layer(tower_http::limit::RequestBodyLimitLayer::new(
            cfg.server.max_body_bytes as usize,
        ))
        .layer(tower_http::timeout::TimeoutLayer::with_status_code(
            axum::http::StatusCode::REQUEST_TIMEOUT,
            Duration::from_secs(cfg.server.request_timeout_secs),
        ))
        .layer(CatchPanicLayer::custom(handle_panic))
        .layer(cors_layer(&cfg.server.allowed_origins));

    // Uretimde tek konteyner/tek port dagitimi icin: derlenmis frontend
    // burada servis edilir, /api disindaki eslesmeyen her yol bu dizine duser.
    let app = if let Some(dir) = &cfg.server.static_dir {
        tracing::info!(dir = %dir.display(), "statik frontend dosyalari servis ediliyor");
        app.fallback_service(tower_http::services::ServeDir::new(dir))
    } else {
        app
    };

    let addr: SocketAddr = format!("{}:{}", cfg.server.host, cfg.server.port).parse()?;
    tracing::info!(%addr, auth_enabled = cfg.auth.enabled, "sunucu baslatiliyor");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    let grace = Duration::from_secs(cfg.server.shutdown_grace_secs);
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(shutdown_signal(grace))
    .await?;
    Ok(())
}

/// Yalnizca konfigurasyonda beyaz listeye alinmis originlere izin verir.
/// Bos liste, aynı-origin disinda tum cross-origin erisimi kapatir.
fn cors_layer(allowed: &[String]) -> CorsLayer {
    if allowed.is_empty() {
        return CorsLayer::new();
    }
    let origins: Vec<axum::http::HeaderValue> =
        allowed.iter().filter_map(|o| o.parse().ok()).collect();
    CorsLayer::new()
        .allow_origin(AllowOrigin::list(origins))
        .allow_methods(tower_http::cors::Any)
        .allow_headers(tower_http::cors::Any)
}

/// Tek bir istekteki panic'in tum sunucuyu dusurmesini engeller (OWASP
/// dayaniklilik geregi); istemciye tutarli JSON hata bicimi dondurulur.
/// Panic detayi yalnizca sunucu logunda kalir, istemciye sizdirilmaz.
fn handle_panic(err: Box<dyn std::any::Any + Send + 'static>) -> Response {
    let detail = if let Some(s) = err.downcast_ref::<&str>() {
        s.to_string()
    } else if let Some(s) = err.downcast_ref::<String>() {
        s.clone()
    } else {
        "bilinmeyen panic".to_string()
    };
    tracing::error!(panic = %detail, "istek isleme sirasinda panic yakalandi");
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        axum::Json(serde_json::json!({
            "error": {
                "code": "internal_error",
                "message": "Beklenmeyen bir sunucu hatasi olustu. Islem kimligi ile birlikte sistem yoneticisine basvurun."
            }
        })),
    )
        .into_response()
}

/// Ctrl+C veya (Unix'te) SIGTERM alindiginda devam eden isteklerin
/// tamamlanmasi icin bir sure daha bekleyip zarifce kapanmayi tetikler.
async fn shutdown_signal(grace: Duration) {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("Ctrl+C isleyicisi kurulamadi");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("SIGTERM isleyicisi kurulamadi")
            .recv()
            .await;
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => tracing::info!("Ctrl+C alindi, zarifce kapatiliyor"),
        _ = terminate => tracing::info!("SIGTERM alindi, zarifce kapatiliyor"),
    }
    tracing::info!(
        grace_secs = grace.as_secs(),
        "devam eden istekler icin bekleniyor"
    );

    // Zarif kapanma asilirsa (uzun suren bir istek/gorev takilirsa) sureci
    // sonsuza dek beklemek yerine belirtilen sure sonunda zorla sonlandir.
    tokio::spawn(async move {
        tokio::time::sleep(grace).await;
        tracing::warn!("zarif kapanma suresi asildi, surec zorla sonlandiriliyor");
        std::process::exit(1);
    });
}
