//! Backend API istemcisi. Rust tiplerini dogrudan `dq-core`den almak yerine
//! burada kopyalanir: WASM hedefi icin gereksiz bagimlilik (tracing-subscriber,
//! blake3 vb.) yuklememek ve derleme suresini kisaltmak icindir. Sozlesme
//! JSON uzerinden kurulur, Rust tipi uzerinden degil.

use serde::{Deserialize, Serialize};

const BASE: &str = "/api";

#[derive(Debug, Clone)]
pub enum ApiErr {
    /// Sunucunun dondurdugu yapilandirilmis hata (kod + mesaj).
    Api { code: String, message: String },
    /// Aga ulasilamadi / JSON ayristirilamadi.
    Network(String),
}

impl std::fmt::Display for ApiErr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ApiErr::Api { message, .. } => write!(f, "{message}"),
            ApiErr::Network(m) => write!(f, "Baglanti hatasi: {m}"),
        }
    }
}

pub type ApiResult<T> = Result<T, ApiErr>;

#[derive(Debug, Clone, Deserialize)]
pub struct Document {
    pub id: String,
    pub filename: String,
    pub mime: String,
    pub size_bytes: u64,
    pub page_count: usize,
    pub lang: String,
    pub classification: String,
    pub status: String,
    pub owner: String,
    pub created_at: String,
    pub error: Option<String>,
    pub avg_confidence: f32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct UploadResponse {
    pub document: Document,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Citation {
    pub marker: usize,
    pub doc_filename: String,
    pub page_from: usize,
    pub page_to: usize,
    pub snippet: String,
    pub score: f32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Groundedness {
    pub support_ratio: f32,
    pub unsupported_sentences: Vec<String>,
    pub top_score: f32,
    pub passed: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AgentStep {
    pub step: usize,
    pub kind: String,
    pub description: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Answer {
    pub query_id: String,
    pub kind: String,
    pub text: String,
    pub citations: Vec<Citation>,
    pub groundedness: Groundedness,
    pub lang: String,
    pub classification: String,
    pub cached: bool,
    pub latency_ms: u64,
    pub model: String,
    pub warnings: Vec<String>,
    #[serde(default)]
    pub trace: Vec<AgentStep>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LoginResponse {
    pub token: String,
    pub username: String,
    pub clearance: String,
    pub roles: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AuditEntry {
    pub at: String,
    pub actor: String,
    pub action: String,
    pub subject: Option<String>,
    pub outcome: String,
    pub detail: serde_json::Value,
}

#[derive(Serialize)]
struct LoginRequest<'a> {
    username: &'a str,
    password: &'a str,
}

#[derive(Serialize)]
struct AskRequest<'a> {
    query: &'a str,
    doc_ids: &'a [String],
}

async fn parse_response<T: for<'de> Deserialize<'de>>(
    resp: gloo_net::http::Response,
) -> ApiResult<T> {
    if resp.ok() {
        resp.json::<T>()
            .await
            .map_err(|e| ApiErr::Network(format!("yanit ayristirilamadi: {e}")))
    } else {
        #[derive(Deserialize)]
        struct ErrBody {
            error: ErrDetail,
        }
        #[derive(Deserialize)]
        struct ErrDetail {
            code: String,
            message: String,
        }
        match resp.json::<ErrBody>().await {
            Ok(b) => Err(ApiErr::Api {
                code: b.error.code,
                message: b.error.message,
            }),
            Err(_) => Err(ApiErr::Network(format!("HTTP {}", resp.status()))),
        }
    }
}

fn auth_header(token: Option<&str>) -> String {
    token.map(|t| format!("Bearer {t}")).unwrap_or_default()
}

pub async fn health() -> ApiResult<serde_json::Value> {
    let resp = gloo_net::http::Request::get(&format!("{BASE}/health"))
        .send()
        .await
        .map_err(|e| ApiErr::Network(e.to_string()))?;
    parse_response(resp).await
}

pub async fn login(username: &str, password: &str) -> ApiResult<LoginResponse> {
    let resp = gloo_net::http::Request::post(&format!("{BASE}/auth/login"))
        .json(&LoginRequest { username, password })
        .map_err(|e| ApiErr::Network(e.to_string()))?
        .send()
        .await
        .map_err(|e| ApiErr::Network(e.to_string()))?;
    parse_response(resp).await
}

pub async fn list_documents(token: &str) -> ApiResult<Vec<Document>> {
    let resp = gloo_net::http::Request::get(&format!("{BASE}/documents"))
        .header("Authorization", &auth_header(Some(token)))
        .send()
        .await
        .map_err(|e| ApiErr::Network(e.to_string()))?;
    parse_response(resp).await
}

pub async fn delete_document(token: &str, id: &str) -> ApiResult<()> {
    let resp = gloo_net::http::Request::delete(&format!("{BASE}/documents/{id}"))
        .header("Authorization", &auth_header(Some(token)))
        .send()
        .await
        .map_err(|e| ApiErr::Network(e.to_string()))?;
    parse_response::<serde_json::Value>(resp).await.map(|_| ())
}

pub async fn upload_document(
    token: &str,
    file: web_sys::File,
    classification: &str,
) -> ApiResult<UploadResponse> {
    let form =
        web_sys::FormData::new().map_err(|_| ApiErr::Network("FormData olusturulamadi".into()))?;
    form.append_with_blob_and_filename("file", &file, &file.name())
        .map_err(|_| ApiErr::Network("dosya forma eklenemedi".into()))?;
    form.append_with_str("classification", classification)
        .map_err(|_| ApiErr::Network("siniflandirma forma eklenemedi".into()))?;

    let resp = gloo_net::http::Request::post(&format!("{BASE}/documents"))
        .header("Authorization", &auth_header(Some(token)))
        .body(form)
        .map_err(|e| ApiErr::Network(e.to_string()))?
        .send()
        .await
        .map_err(|e| ApiErr::Network(e.to_string()))?;
    parse_response(resp).await
}

pub async fn ask(token: &str, query: &str, doc_ids: &[String]) -> ApiResult<Answer> {
    let resp = gloo_net::http::Request::post(&format!("{BASE}/ask"))
        .header("Authorization", &auth_header(Some(token)))
        .json(&AskRequest { query, doc_ids })
        .map_err(|e| ApiErr::Network(e.to_string()))?
        .send()
        .await
        .map_err(|e| ApiErr::Network(e.to_string()))?;
    parse_response(resp).await
}

pub async fn audit_log(token: &str) -> ApiResult<Vec<AuditEntry>> {
    let resp = gloo_net::http::Request::get(&format!("{BASE}/audit"))
        .header("Authorization", &auth_header(Some(token)))
        .send()
        .await
        .map_err(|e| ApiErr::Network(e.to_string()))?;
    parse_response(resp).await
}

pub async fn audit_verify(token: &str) -> ApiResult<serde_json::Value> {
    let resp = gloo_net::http::Request::get(&format!("{BASE}/audit/verify"))
        .header("Authorization", &auth_header(Some(token)))
        .send()
        .await
        .map_err(|e| ApiErr::Network(e.to_string()))?;
    parse_response(resp).await
}
