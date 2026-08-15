//! Ortak tipler, hata modeli, konfigurasyon ve telemetri.
//!
//! `dq-core` diger tum crate'lerin bagimli oldugu temel katmandir. Burada
//! is mantigi yoktur; sadece paylasilan sozlesmeler bulunur.

pub mod config;
pub mod error;
pub mod ids;
pub mod semantic;
pub mod telemetry;
pub mod text;
pub mod types;

pub use error::{DqError, Result};
pub use types::*;
