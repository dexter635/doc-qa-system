use dq_core::{DqError, Result};

/// Dosyanin gercek turu. Istemcinin gonderdigi `Content-Type` basligina
/// guvenilmez; tur her zaman icerikten (magic bytes) tespit edilir.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileKind {
    Pdf,
    Jpeg,
    Png,
}

impl FileKind {
    pub fn mime(&self) -> &'static str {
        match self {
            FileKind::Pdf => "application/pdf",
            FileKind::Jpeg => "image/jpeg",
            FileKind::Png => "image/png",
        }
    }

    pub fn is_image(&self) -> bool {
        matches!(self, FileKind::Jpeg | FileKind::Png)
    }
}

/// Icerigin ilk baytlarindan dosya turunu tespit eder.
pub fn sniff(bytes: &[u8]) -> Result<FileKind> {
    if bytes.len() < 8 {
        return Err(DqError::UnsupportedMedia("dosya cok kisa".into()));
    }
    // PDF basligi ilk 1 KB icinde herhangi bir yerde olabilir (RFC disi ama yaygin).
    let head = &bytes[..bytes.len().min(1024)];
    if head.windows(5).any(|w| w == b"%PDF-") {
        return Ok(FileKind::Pdf);
    }
    if bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
        return Ok(FileKind::Jpeg);
    }
    if bytes.starts_with(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]) {
        return Ok(FileKind::Png);
    }
    Err(DqError::UnsupportedMedia(
        "yalnizca PDF, JPEG ve PNG desteklenmektedir".into(),
    ))
}

/// Yuklenen dosya adini guvenli hale getirir (path traversal ve kontrol
/// karakterlerine karsi).
pub fn sanitize_filename(name: &str) -> String {
    let base = name
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or("belge")
        .trim()
        .trim_matches('.');
    let cleaned: String = base
        .chars()
        .map(|c| {
            if c.is_control() || matches!(c, '<' | '>' | ':' | '"' | '|' | '?' | '*' | '\0') {
                '_'
            } else {
                c
            }
        })
        .collect();
    let cleaned = cleaned.trim();
    if cleaned.is_empty() {
        "belge".to_string()
    } else {
        dq_core::text::truncate_chars(cleaned, 180)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_types() {
        assert_eq!(sniff(b"%PDF-1.7\n....").unwrap(), FileKind::Pdf);
        assert_eq!(sniff(&[0xFF, 0xD8, 0xFF, 0xE0, 0, 0, 0, 0]).unwrap(), FileKind::Jpeg);
        assert!(sniff(b"MZ\x90\x00\x03\x00\x00\x00").is_err());
    }

    #[test]
    fn filename_traversal_is_blocked() {
        assert_eq!(sanitize_filename("../../etc/passwd"), "passwd");
        assert_eq!(sanitize_filename("C:\\gizli\\plan.pdf"), "plan.pdf");
        assert_eq!(sanitize_filename("   "), "belge");
    }
}
