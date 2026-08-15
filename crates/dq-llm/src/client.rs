//! Yerel LLM istemcileri.
//!
//! Iki mod desteklenir:
//! - `OpenAiCompatClient`: OpenAI uyumlu `/chat/completions` uclari icin
//!   (Ollama, llama.cpp server, vLLM vb.).
//! - `LocalLlmClient`: Rust `llm` crate'i ile yerel GGUF dosyalarindan
//!   tek container, offline inference. Tek bir model yukler ve donusturur.
//!
//! Bulut saglayicisina hicbir cagri yapilmaz.

use std::path::PathBuf;
use std::time::Duration;

use async_trait::async_trait;
use dq_core::config::LlmConfig;
use dq_core::{DqError, Result};
use serde::{Deserialize, Serialize};

use super::chat_template::ChatTemplate;

#[derive(Debug, Clone, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".into(),
            content: content.into(),
        }
    }
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".into(),
            content: content.into(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Completion {
    pub text: String,
    pub model: String,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
}

#[async_trait]
pub trait LlmClient: Send + Sync {
    fn model(&self) -> String;
    async fn chat(&self, messages: Vec<ChatMessage>) -> Result<Completion>;
    /// `chat`'in ayni sozlesmesi ancak sicaklik (temperature) gecici olarak
    /// ezilir. Yapisal JSON uretimi (planlama/yeniden formulasyon) gibi
    /// deterministiklik gerektiren cagrilar icin kullanilir.
    async fn chat_with_temperature(
        &self,
        messages: Vec<ChatMessage>,
        temperature: f32,
    ) -> Result<Completion>;
    /// Servisin ayakta olup olmadigi. Basarisizsa sistem cikarimsal moda duser.
    async fn healthy(&self) -> bool;
}

/// LLM olmadigi durumlarda kullanilan "null nesne" istemci.
/// Sistemin dusmesini onlemek icin tum istekleri uygun sekilde reddeder
/// ve cikarimsal (extractive) yedek moduna gecmesini saglar.
#[derive(Clone)]
pub struct NullLlmClient;

#[async_trait]
impl LlmClient for NullLlmClient {
    fn model(&self) -> String {
        "none".into()
    }

    async fn chat(&self, _messages: Vec<ChatMessage>) -> Result<Completion> {
        Err(DqError::Llm("LLM devre disi; cikarimsal yedek modu aktif".into()))
    }

    async fn chat_with_temperature(
        &self,
        _messages: Vec<ChatMessage>,
        _temperature: f32,
    ) -> Result<Completion> {
        Err(DqError::Llm("LLM devre disi; cikarimsal yedek modu aktif".into()))
    }

    async fn healthy(&self) -> bool {
        false
    }
}

impl NullLlmClient {
    pub fn new() -> Self {
        Self
    }
}

/// Gecici (transient) hatalarda denenecek en fazla deneme sayisi (ilk deneme dahil).
const MAX_ATTEMPTS: u32 = 3;
/// Denemeler arasi baslangic bekleme suresi; her denemede ikiye katlanir.
const BASE_BACKOFF: Duration = Duration::from_millis(400);

pub struct OpenAiCompatClient {
    http: reqwest::Client,
    base_url: String,
    model: String,
    api_key: String,
    temperature: f32,
    top_p: f32,
    max_tokens: u32,
}

impl OpenAiCompatClient {
    pub fn new(cfg: &LlmConfig) -> Result<Self> {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(cfg.timeout_secs))
            .build()
            .map_err(|e| DqError::Llm(format!("http istemcisi kurulamadi: {e}")))?;
        Ok(Self {
            http,
            base_url: cfg.base_url.trim_end_matches('/').to_string(),
            model: cfg.model.clone(),
            api_key: cfg.api_key.clone(),
            temperature: cfg.temperature,
            top_p: cfg.top_p,
            max_tokens: cfg.max_tokens,
        })
    }
}

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: &'a [ChatMessage],
    temperature: f32,
    top_p: f32,
    max_tokens: u32,
    stream: bool,
}

#[derive(Deserialize)]
struct ChatResponse {
    #[serde(default)]
    model: String,
    choices: Vec<Choice>,
    #[serde(default)]
    usage: Option<Usage>,
}

#[derive(Deserialize)]
struct Choice {
    message: ResponseMessage,
}

#[derive(Deserialize)]
struct ResponseMessage {
    #[serde(default)]
    content: String,
}

#[derive(Deserialize, Default)]
struct Usage {
    #[serde(default)]
    prompt_tokens: u32,
    #[serde(default)]
    completion_tokens: u32,
}

#[async_trait]
impl LlmClient for OpenAiCompatClient {
    fn model(&self) -> String {
        self.model.clone()
    }

    async fn chat(&self, messages: Vec<ChatMessage>) -> Result<Completion> {
        self.chat_retrying(messages, self.temperature).await
    }

    async fn chat_with_temperature(
        &self,
        messages: Vec<ChatMessage>,
        temperature: f32,
    ) -> Result<Completion> {
        self.chat_retrying(messages, temperature).await
    }

    async fn healthy(&self) -> bool {
        let mut req = self.http.get(format!("{}/models", self.base_url));
        if !self.api_key.is_empty() {
            req = req.bearer_auth(&self.api_key);
        }
        match tokio::time::timeout(Duration::from_secs(5), req.send()).await {
            Ok(Ok(r)) => r.status().is_success(),
            _ => false,
        }
    }
}

impl OpenAiCompatClient {
    /// Gecici hatalarda (ag hatasi, 429, 5xx) ussel geri cekilme ile yeniden
    /// dener; istemci hatalarinda (4xx, 429 haric) hemen vazgecer - tekrar
    /// denemek sonucu degistirmeyecek bir isteği bosuna cogaltmamak icin.
    async fn chat_retrying(
        &self,
        messages: Vec<ChatMessage>,
        temperature: f32,
    ) -> Result<Completion> {
        let mut attempt = 0u32;
        let mut backoff = BASE_BACKOFF;
        loop {
            attempt += 1;
            match self.chat_once(&messages, temperature).await {
                Ok(c) => return Ok(c),
                Err(e) if attempt < MAX_ATTEMPTS && is_retryable(&e) => {
                    tracing::warn!(attempt, error = %e, "LLM cagrisi basarisiz, yeniden deneniyor");
                    tokio::time::sleep(backoff).await;
                    backoff *= 2;
                }
                Err(e) => return Err(e),
            }
        }
    }

    async fn chat_once(&self, messages: &[ChatMessage], temperature: f32) -> Result<Completion> {
        let body = ChatRequest {
            model: &self.model,
            messages,
            temperature,
            top_p: self.top_p,
            max_tokens: self.max_tokens,
            stream: false,
        };

        let mut req = self
            .http
            .post(format!("{}/chat/completions", self.base_url))
            .json(&body);
        if !self.api_key.is_empty() {
            req = req.bearer_auth(&self.api_key);
        }

        let resp = req
            .send()
            .await
            .map_err(|e| DqError::Llm(format!("LLM'e ulasilamadi: {e}")))?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(DqError::Llm(format!(
                "LLM {status} dondu: {}",
                dq_core::text::truncate_chars(&text, 300)
            )));
        }

        let parsed: ChatResponse = resp
            .json()
            .await
            .map_err(|e| DqError::Llm(format!("LLM cevabi ayristirilamadi: {e}")))?;

        let text = parsed
            .choices
            .into_iter()
            .next()
            .map(|c| c.message.content)
            .unwrap_or_default();

        if text.trim().is_empty() {
            return Err(DqError::Llm("LLM bos cevap dondu".into()));
        }

        let usage = parsed.usage.unwrap_or_default();
        Ok(Completion {
            text,
            model: if parsed.model.is_empty() {
                self.model.clone()
            } else {
                parsed.model
            },
            prompt_tokens: usage.prompt_tokens,
            completion_tokens: usage.completion_tokens,
        })
    }
}

/// Hata mesaji, ag/gecici bir sorunu mu (yeniden denenebilir) yoksa kalici bir
/// istemci hatasini mi (400 gibi - yeniden denemek sonucu degistirmez) yansitiyor.
fn is_retryable(e: &DqError) -> bool {
    let msg = e.to_string();
    if msg.contains("ulasilamadi") {
        return true; // baglanti/ag hatasi
    }
    // "LLM 5xx/429 dondu: ..." bicimindeki mesajlardan durum kodunu yakala.
    for code in ["500", "502", "503", "504", "429"] {
        if msg.contains(code) {
            return true;
        }
    }
    false
}

#[derive(Clone)]
pub struct LocalLlmClient {
    model_path: PathBuf,
    model_name: String,
}

impl LocalLlmClient {
    pub fn new(model_path: impl Into<PathBuf>, model_name: impl Into<String>) -> Self {
        Self {
            model_path: model_path.into(),
            model_name: model_name.into(),
        }
    }
}

#[async_trait]
impl LlmClient for LocalLlmClient {
    fn model(&self) -> String {
        self.model_name.clone()
    }

    async fn chat(&self, messages: Vec<ChatMessage>) -> Result<Completion> {
        self.chat_with_temperature(messages, 0.0).await
    }

    async fn chat_with_temperature(
        &self,
        messages: Vec<ChatMessage>,
        temperature: f32,
    ) -> Result<Completion> {
        let prompt = ChatTemplate::to_prompt(&messages);
        let model_path = self.model_path.clone();
        let model_name = self.model_name.clone();
        let temp = temperature;

        let result = tokio::task::spawn_blocking(move || -> Result<Completion> {
            use llm::Model;
            use llm::models::Llama;
            use std::fs;

            let params = llm::InferenceParameters::from(llm::ModelParameters {
                temperature: temp,
                top_p: 0.9,
                top_k: 40,
                ..Default::default()
            });

            let model = Llama::load_from_file(
                std::path::Path::new(&model_path),
                params,
                llm::load::LoaderCallback::new(|_| {}),
            )
            .map_err(|e| DqError::Llm(format!("GGUF model yuklenemedi: {e}")))?;

            let mut session = llm::InferenceSession::new(&model);
            let mut generated = String::new();

            let mut rng = rand::thread_rng();
            let mut tokens = model
                .tokenizer()
                .encode(&prompt, llm::TokenEncodeSpecialTokensPolicy::Never)
                .map_err(|e| DqError::Llm(format!("tokenize hatasi: {e}")))?;

            for token in model.infer(
                &mut session,
                &model.special_symbols(),
                tokens,
                &llm::InferenceRequest::new(params, 256),
                &mut rng,
                |t| {
                    if let Some(t) = t {
                        generated.push_str(t);
                    }
                },
            ) {
                match token {
                    Ok(llm::InferenceToken::Token(t)) => {
                        tokens.push(t);
                    }
                    Ok(llm::InferenceToken::Eot) => break,
                    Err(_) => break,
                }
            }

            Ok(Completion {
                text: generated,
                model: model_name,
                prompt_tokens: 0,
                completion_tokens: 0,
            })
        })
        .await
        .map_err(|e| DqError::Llm(format!("inference panik: {e}")))??;

        Ok(result)
    }

    async fn healthy(&self) -> bool {
        self.model_path.exists()
    }
}

mod chat_template {
    pub fn to_prompt(messages: &[super::ChatMessage]) -> String {
        let mut prompt = String::new();
        for m in messages {
            match m.role.as_str() {
                "system" => prompt.push_str(&format!("<|system|>\n{}\n", m.content)),
                "user" => prompt.push_str(&format!("<|user|>\n{}\n", m.content)),
                "assistant" => prompt.push_str(&format!("<|assistant|>\n{}\n", m.content)),
                _ => prompt.push_str(&format!("<|user|>\n{}\n", m.content)),
            }
        }
        prompt.push_str("<|assistant|>\n");
        prompt
    }
}

/// Hata mesaji, ag/gecici bir sorunu mu (yeniden denenebilir) yoksa kalici bir
/// istemci hatasini mi (400 gibi - yeniden denemek sonucu degistirmez) yansitiyor.
fn is_retryable(e: &DqError) -> bool {
    let msg = e.to_string();
    if msg.contains("ulasilamadi") {
        return true; // baglanti/ag hatasi
    }
    // "LLM 5xx/429 dondu: ..." bicimindeki mesajlardan durum kodunu yakala.
    for code in ["500", "502", "503", "504", "429"] {
        if msg.contains(code) {
            return true;
        }
    }
    false
}
