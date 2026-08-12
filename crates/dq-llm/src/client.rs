//! Yerel LLM istemcisi (OpenAI uyumlu `/chat/completions`).
//!
//! Yerel calisan llama.cpp server, Ollama ve vLLM ayni sozlesmeyi konustugu
//! icin tek bir istemci uc motoru da destekler; boylece model degistirmek
//! konfigurasyon degisikligine iner. Bulut saglayicisina hicbir cagri yapilmaz.

use std::time::Duration;

use async_trait::async_trait;
use dq_core::config::LlmConfig;
use dq_core::{DqError, Result};
use serde::{Deserialize, Serialize};

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
    /// Servisin ayakta olup olmadigi. Basarisizsa sistem cikarimsal moda duser.
    async fn healthy(&self) -> bool;
}

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
        let body = ChatRequest {
            model: &self.model,
            messages: &messages,
            temperature: self.temperature,
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
