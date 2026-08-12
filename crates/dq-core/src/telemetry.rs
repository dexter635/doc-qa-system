use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

/// Loglamayi baslatir.
///
/// `DQ_LOG_FORMAT=json` verildiginde makine tarafindan islenebilir JSON log
/// uretir (SIEM entegrasyonu icin); aksi halde insan okunur formattadir.
pub fn init(default_filter: &str) {
    let filter = EnvFilter::try_from_env("DQ_LOG")
        .or_else(|_| EnvFilter::try_new(default_filter))
        .unwrap_or_else(|_| EnvFilter::new("info"));

    let json = std::env::var("DQ_LOG_FORMAT").map(|v| v == "json").unwrap_or(false);

    let registry = tracing_subscriber::registry().with(filter);
    if json {
        registry
            .with(tracing_subscriber::fmt::layer().json().with_current_span(true))
            .init();
    } else {
        registry
            .with(tracing_subscriber::fmt::layer().with_target(false).compact())
            .init();
    }
}

/// Sure olcumu icin basit yardimci.
pub struct Timer(std::time::Instant);

impl Timer {
    pub fn start() -> Self {
        Timer(std::time::Instant::now())
    }
    pub fn elapsed_ms(&self) -> u64 {
        self.0.elapsed().as_millis() as u64
    }
}

impl Default for Timer {
    fn default() -> Self {
        Self::start()
    }
}
