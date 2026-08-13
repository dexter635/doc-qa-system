//! Belge isleme hatti: bayt dizisi -> sayfa metinleri -> chunk'lar.

use dq_core::config::{IngestConfig, OcrConfig};
use dq_core::text::{clean_extracted_text, detect_lang};
use dq_core::{Chunk, Classification, DqError, ExtractionMethod, Lang, PageText, Result};
use uuid::Uuid;

use crate::detect::{self, FileKind};
use crate::imageproc;
use crate::ocr::{self, OcrEngine};

pub struct IngestOutcome {
    pub kind: FileKind,
    pub pages: Vec<PageText>,
    pub chunks: Vec<Chunk>,
    pub page_count: usize,
    pub lang: Lang,
    pub avg_confidence: f32,
    /// Kullaniciya gosterilecek kalite uyarilari (bos sayfa, OCR yok, vb.).
    pub warnings: Vec<String>,
}

pub struct Ingestor {
    cfg: IngestConfig,
    ocr: Box<dyn OcrEngine>,
}

impl Ingestor {
    pub fn new(cfg: IngestConfig, ocr_cfg: &OcrConfig) -> Self {
        let engine = ocr::build(ocr_cfg);
        tracing::info!(engine = engine.name(), "OCR motoru secildi");
        Self { cfg, ocr: engine }
    }

    pub fn ocr_engine_name(&self) -> &'static str {
        self.ocr.name()
    }

    /// Bir dosyayi bastan sona isler. CPU-yogun oldugu icin cagiran taraf
    /// bunu `spawn_blocking` icinde calistirmalidir.
    pub fn ingest(
        &self,
        bytes: &[u8],
        doc_id: Uuid,
        classification: Classification,
    ) -> Result<IngestOutcome> {
        if bytes.len() as u64 > self.cfg.max_file_bytes {
            return Err(DqError::PayloadTooLarge {
                size: bytes.len() as u64,
                limit: self.cfg.max_file_bytes,
            });
        }
        let kind = detect::sniff(bytes)?;
        if !self.cfg.allowed_mime.iter().any(|m| m == kind.mime()) {
            return Err(DqError::UnsupportedMedia(kind.mime().into()));
        }

        let mut warnings = Vec::new();
        let pages = match kind {
            FileKind::Pdf => self.ingest_pdf(bytes, &mut warnings)?,
            _ => self.ingest_image(bytes, kind, &mut warnings)?,
        };

        if pages.iter().all(|p| p.text.trim().is_empty()) {
            return Err(DqError::Ingest(
                "Belgeden hic metin cikarilamadi. Taranmis belge ise OCR motorunun kurulu oldugundan emin olun.".into(),
            ));
        }

        let combined: String = pages
            .iter()
            .map(|p| p.text.as_str())
            .collect::<Vec<_>>()
            .join("\n")
            .chars()
            .take(20_000)
            .collect();
        let lang = detect_lang(&combined);
        if lang == Lang::Unknown {
            warnings.push("Belge dili kesin tespit edilemedi.".into());
        }

        let avg_confidence = if pages.is_empty() {
            0.0
        } else {
            pages.iter().map(|p| p.confidence).sum::<f32>() / pages.len() as f32
        };
        if avg_confidence < 0.7 {
            warnings.push(format!(
                "Dusuk metin cikarim guveni (%{:.0}); cevaplar hatali olabilir.",
                avg_confidence * 100.0
            ));
        }

        let chunks = crate::chunk::build_chunks(doc_id, &pages, &self.cfg, classification, lang);
        if chunks.is_empty() {
            return Err(DqError::Ingest(
                "Belge indekslenebilir icerik uretmedi".into(),
            ));
        }

        Ok(IngestOutcome {
            kind,
            page_count: pages.len(),
            lang,
            avg_confidence,
            pages,
            chunks,
            warnings,
        })
    }

    fn ingest_pdf(&self, bytes: &[u8], warnings: &mut Vec<String>) -> Result<Vec<PageText>> {
        let total_pages = crate::pdf::page_count(bytes)?;
        if total_pages == 0 {
            return Err(DqError::Ingest("PDF sayfa icermiyor".into()));
        }
        if total_pages > self.cfg.max_pages {
            return Err(DqError::Ingest(format!(
                "PDF {total_pages} sayfa iceriyor; limit {}",
                self.cfg.max_pages
            )));
        }

        let raw_pages = crate::pdf::page_texts(bytes).unwrap_or_else(|e| {
            tracing::warn!(error = %e, "PDF metin katmani okunamadi, tamamen OCR'a dusuluyor");
            vec![String::new(); total_pages]
        });

        let mut pages: Vec<PageText> = raw_pages
            .into_iter()
            .enumerate()
            .map(|(i, t)| PageText {
                page_no: i + 1,
                text: clean_extracted_text(&t),
                method: ExtractionMethod::PdfText,
                confidence: 1.0,
                lang: Lang::Unknown,
            })
            .collect();

        // Metin katmani zayif olan sayfalar taranmis kabul edilir.
        let needs_ocr: Vec<usize> = pages
            .iter()
            .filter(|p| p.text.chars().count() < self.cfg.ocr_fallback_char_threshold)
            .map(|p| p.page_no)
            .collect();

        if !needs_ocr.is_empty() {
            if !self.ocr.available() {
                warnings.push(format!(
                    "{} sayfada metin katmani yok ve OCR motoru kullanilamiyor; bu sayfalar atlandi.",
                    needs_ocr.len()
                ));
            } else {
                let images = crate::pdf::page_images(bytes, &needs_ocr)?;
                if images.is_empty() {
                    warnings.push(
                        "Taranmis sayfalardaki goruntuler cikarilamadi (desteklenmeyen sikistirma)."
                            .into(),
                    );
                }
                for (page_no, imgs) in images {
                    let mut texts = Vec::new();
                    let mut conf_sum = 0f32;
                    let mut conf_n = 0usize;
                    for img in imgs {
                        if imageproc::is_blank(&img) {
                            continue;
                        }
                        let prepared = imageproc::preprocess(&imageproc::cap_dimensions(img, 4000));
                        match self.ocr.recognize(&prepared, Lang::Unknown) {
                            Ok(res) if !res.text.trim().is_empty() => {
                                conf_sum += res.confidence;
                                conf_n += 1;
                                texts.push(res.text);
                            }
                            Ok(_) => {}
                            Err(e) => tracing::warn!(page = page_no, error = %e, "OCR basarisiz"),
                        }
                    }
                    if texts.is_empty() {
                        continue;
                    }
                    if let Some(slot) = pages.iter_mut().find(|p| p.page_no == page_no) {
                        let ocr_text = clean_extracted_text(&texts.join("\n"));
                        slot.method = if slot.text.trim().is_empty() {
                            ExtractionMethod::Ocr
                        } else {
                            ExtractionMethod::Hybrid
                        };
                        slot.confidence = if conf_n == 0 {
                            0.0
                        } else {
                            conf_sum / conf_n as f32
                        };
                        slot.text = if slot.text.trim().is_empty() {
                            ocr_text
                        } else {
                            format!("{}\n{}", slot.text, ocr_text)
                        };
                    }
                }
            }
        }

        for p in pages.iter_mut() {
            p.lang = detect_lang(&p.text);
        }
        pages.retain(|p| !p.text.trim().is_empty());
        Ok(pages)
    }

    fn ingest_image(
        &self,
        bytes: &[u8],
        kind: FileKind,
        warnings: &mut Vec<String>,
    ) -> Result<Vec<PageText>> {
        if !self.ocr.available() {
            return Err(DqError::Ocr(
                "Resim belgeleri icin OCR gerekiyor ancak motor kullanilamiyor. Tesseract'i kurun (tur+eng dil paketleri ile).".into(),
            ));
        }
        let img = imageproc::load(bytes, kind)?;
        if imageproc::is_blank(&img) {
            return Err(DqError::Ingest("Goruntu bos gorunuyor".into()));
        }
        let prepared = imageproc::preprocess(&imageproc::cap_dimensions(img, 4000));
        let res = self.ocr.recognize(&prepared, Lang::Unknown)?;
        if res.dropped_words > 0 {
            warnings.push(format!(
                "{} kelime dusuk OCR guveni nedeniyle atildi.",
                res.dropped_words
            ));
        }
        let text = clean_extracted_text(&res.text);
        if text.trim().is_empty() {
            return Err(DqError::Ocr("Goruntuden metin okunamadi".into()));
        }
        Ok(vec![PageText {
            page_no: 1,
            lang: detect_lang(&text),
            text,
            method: ExtractionMethod::Ocr,
            confidence: res.confidence,
        }])
    }
}
