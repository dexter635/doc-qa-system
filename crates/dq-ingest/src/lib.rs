//! Belge alimi (ingest): tur tespiti, metin cikarimi, OCR ve parcalama.

pub mod chunk;
pub mod detect;
pub mod imageproc;
pub mod ocr;
pub mod pdf;
pub mod pipeline;

pub use detect::{sanitize_filename, sniff, FileKind};
pub use pipeline::{IngestOutcome, Ingestor};
