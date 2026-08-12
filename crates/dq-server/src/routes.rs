//! HTTP uc noktalari.

use axum::extract::{Multipart, Path, State};
use axum::http::HeaderMap;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use dq_core::{Answer, Classification, DqError, Document, UserContext};

use crate::auth::AuthService;
use crate::error::{ApiError, ApiResult};
use crate::state::AppState;

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/api/health", get(health))
        .route("/api/auth/login", post(login))
        .route("/api/documents", get(list_documents).post(upload_document))
        .route("/api/documents/{id}", get(get_document).delete(delete_document))
        .route("/api/ask", post(ask))
        .route("/api/audit", get(audit_log))
        .route("/api/audit/verify", get(audit_verify))
        .with_state(state)
}

/// `Authorization: Bearer <token>` basligini dogrular. Auth kapaliysa
/// konfigurasyondaki anonim yetki seviyesiyle sanal bir kullanici dondurur.
fn current_user(state: &AppState, headers: &HeaderMap) -> ApiResult<UserContext> {
    let token = headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.strip_prefix("Bearer "));
    state.auth.verify_token_or_anonymous(token).map_err(ApiError)
}

fn require_role(user: &UserContext, role: &str) -> ApiResult<()> {
    if user.is_admin() || user.roles.iter().any(|r| r == role) {
        Ok(())
    } else {
        Err(ApiError(DqError::Forbidden(format!(
            "Bu islem icin '{role}' rolu gerekiyor"
        ))))
    }
}

fn parse_classification(s: &str) -> ApiResult<Classification> {
    serde_json::from_value(serde_json::Value::String(s.to_string()))
        .map_err(|_| ApiError(DqError::BadRequest(format!("Gecersiz gizlilik derecesi: {s}"))))
}

async fn health(State(state): State<AppState>) -> Json<serde_json::Value> {
    let llm_ok = state.pipeline.llm_healthy().await;
    let chunk_count = state.pipeline.store().chunk_count().unwrap_or(0);
    Json(serde_json::json!({
        "status": "ok",
        "ocr_engine": state.pipeline.ocr_engine_name(),
        "embedding_model": state.cfg.embedding.model,
        "llm_model": state.cfg.llm.model,
        "llm_healthy": llm_ok,
        "chunk_count": chunk_count,
        "cache": state.pipeline.cache().stats(),
    }))
}

#[derive(Deserialize)]
struct LoginRequest {
    username: String,
    password: String,
}

#[derive(Serialize)]
struct LoginResponse {
    token: String,
    username: String,
    clearance: String,
    roles: Vec<String>,
}

async fn login(State(state): State<AppState>, Json(req): Json<LoginRequest>) -> ApiResult<Json<LoginResponse>> {
    if !state.auth.enabled() {
        return Err(ApiError(DqError::BadRequest("Kimlik dogrulama devre disi".into())));
    }
    // Kullanici bulunamasa bile ayni maliyette bir hash dogrulamasi yapilir;
    // boylece yanit suresi kullanici adinin var olup olmadigini sizdirmaz.
    const DUMMY_HASH: &str = "$argon2id$v=19$m=19456,t=2,p=1$c29tZXNhbHRzb21lc2FsdA$Y5S3nWn0m8pQ4b6r6qq0mS0kS0kS0kS0kS0kS0kS0kQ";
    let found = state.pipeline.store().get_user(&req.username)?;
    let (hash, user) = match &found {
        Some((h, u)) => (h.as_str(), Some(u)),
        None => (DUMMY_HASH, None),
    };
    let password_ok = AuthService::verify_password(&req.password, hash);
    let Some(user) = user.filter(|_| password_ok) else {
        return Err(ApiError(DqError::Unauthorized("Kullanici adi veya parola hatali".into())));
    };
    let token = state.auth.issue_token(user)?;
    state
        .pipeline
        .store()
        .append_audit(&user.username, "login", None, "success", serde_json::json!({}))?;
    Ok(Json(LoginResponse {
        token,
        username: user.username.clone(),
        clearance: user.clearance.label_tr().to_string(),
        roles: user.roles.clone(),
    }))
}

#[derive(Serialize)]
struct UploadResponse {
    document: Document,
    warnings: Vec<String>,
}

async fn upload_document(
    State(state): State<AppState>,
    headers: HeaderMap,
    mut multipart: Multipart,
) -> ApiResult<Json<UploadResponse>> {
    let user = current_user(&state, &headers)?;
    require_role(&user, "user")?;

    let mut file_bytes: Option<Vec<u8>> = None;
    let mut filename = "belge".to_string();
    let mut classification = Classification::Unclassified;

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| ApiError(DqError::BadRequest(format!("Multipart istegi bozuk: {e}"))))?
    {
        match field.name().unwrap_or("") {
            "file" => {
                filename = field.file_name().unwrap_or("belge").to_string();
                let data = field
                    .bytes()
                    .await
                    .map_err(|e| ApiError(DqError::BadRequest(format!("Dosya okunamadi: {e}"))))?;
                file_bytes = Some(data.to_vec());
            }
            "classification" => {
                let v = field.text().await.unwrap_or_default();
                if !v.trim().is_empty() {
                    classification = parse_classification(&v)?;
                }
            }
            _ => {}
        }
    }

    let bytes = file_bytes.ok_or_else(|| ApiError(DqError::BadRequest("'file' alani gerekli".into())))?;

    // Bell-LaPadula "no write up": kullanici kendi yetki seviyesinin
    // uzerinde bir gizlilik derecesi iddia edemez.
    if classification > user.clearance {
        return Err(ApiError(DqError::Forbidden(
            "Belgeyi kendi yetki seviyenizin uzerinde bir gizlilik derecesiyle isaretleyemezsiniz".into(),
        )));
    }

    let owner = user.username.clone();
    let pipeline = state.pipeline.clone();
    let outcome = tokio::task::spawn_blocking(move || pipeline.ingest_document(&bytes, &filename, classification, &owner))
        .await
        .map_err(|e| ApiError(DqError::Internal(format!("Isleme gorevi calistirilamadi: {e}"))))??;

    Ok(Json(UploadResponse {
        document: outcome.document,
        warnings: outcome.warnings,
    }))
}

async fn list_documents(State(state): State<AppState>, headers: HeaderMap) -> ApiResult<Json<Vec<Document>>> {
    let user = current_user(&state, &headers)?;
    Ok(Json(state.pipeline.list_documents(user.clearance)?))
}

async fn get_document(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<Uuid>,
) -> ApiResult<Json<Document>> {
    let user = current_user(&state, &headers)?;
    let doc = state
        .pipeline
        .get_document(id, user.clearance)?
        .ok_or_else(|| ApiError(DqError::NotFound("Belge bulunamadi".into())))?;
    Ok(Json(doc))
}

async fn delete_document(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<Uuid>,
) -> ApiResult<Json<serde_json::Value>> {
    let user = current_user(&state, &headers)?;
    require_role(&user, "admin")?;
    let deleted = state.pipeline.delete_document(id)?;
    state.pipeline.store().append_audit(
        &user.username,
        "delete_document",
        Some(&id.to_string()),
        if deleted { "deleted" } else { "not_found" },
        serde_json::json!({}),
    )?;
    Ok(Json(serde_json::json!({ "deleted": deleted })))
}

#[derive(Deserialize)]
struct AskRequest {
    query: String,
    #[serde(default)]
    doc_ids: Vec<Uuid>,
}

async fn ask(State(state): State<AppState>, headers: HeaderMap, Json(req): Json<AskRequest>) -> ApiResult<Json<Answer>> {
    let user = current_user(&state, &headers)?;
    let answer = state.pipeline.ask(&req.query, &user, req.doc_ids).await?;
    Ok(Json(answer))
}

async fn audit_log(State(state): State<AppState>, headers: HeaderMap) -> ApiResult<Json<Vec<serde_json::Value>>> {
    let user = current_user(&state, &headers)?;
    require_role(&user, "admin")?;
    Ok(Json(state.pipeline.store().recent_audit(200)?))
}

async fn audit_verify(State(state): State<AppState>, headers: HeaderMap) -> ApiResult<Json<serde_json::Value>> {
    let user = current_user(&state, &headers)?;
    require_role(&user, "admin")?;
    let broken_at = state.pipeline.store().verify_audit_chain()?;
    Ok(Json(serde_json::json!({
        "intact": broken_at.is_none(),
        "broken_at_index": broken_at,
    })))
}
