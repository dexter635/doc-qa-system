//! Yerel LLM erisimi, istem sablonlari ve LLM'siz yedek cevap ureteci.

pub mod client;
pub mod extractive;
pub mod json;
pub mod prompts;

pub use client::{ChatMessage, Completion, LlmClient, OpenAiCompatClient};
