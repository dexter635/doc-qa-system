//! Guardrail katmani: girdi (prompt injection/PII) ve cikti
//! (groundedness/PII/gizlilik damgalama) denetimleri.

pub mod input;
pub mod output;
pub mod pii;

pub use input::{InjectionCategory, InputGuard, InputVerdict, TriggeredRule};
pub use output::{into_answer, OutputGuard, OutputResult};
pub use pii::{detect as detect_pii, redact as redact_pii, PiiKind, PiiMatch};
