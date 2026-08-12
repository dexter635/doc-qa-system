//! OCR motoru soyutlamasi.
//!
//! Sistem tek bir motora bagli kalmamalidir: Tesseract Turkce'de en iyi
//! sonucu verir ancak harici bir binary gerektirir. Motor calistirilamazsa
//! sistem cokmez, ilgili sayfa dusuk guvenle isaretlenir.

use std::io::Write;
use std::process::{Command, Stdio};

use dq_core::{DqError, Lang, Result};
use image::DynamicImage;

#[derive(Debug, Clone)]
pub struct OcrPage {
    pub text: String,
    /// 0..1 arasi ortalama kelime guveni.
    pub confidence: f32,
    /// Guven esiginin altinda kalip atilan kelime sayisi.
    pub dropped_words: usize,
}

pub trait OcrEngine: Send + Sync {
    fn name(&self) -> &'static str;
    fn recognize(&self, img: &DynamicImage, lang: Lang) -> Result<OcrPage>;
    /// Motor bu makinede kullanilabilir mi?
    fn available(&self) -> bool;
}

/// OCR'in kapali oldugu kurulumlar icin bos motor.
pub struct NoopOcr;

impl OcrEngine for NoopOcr {
    fn name(&self) -> &'static str {
        "noop"
    }
    fn recognize(&self, _img: &DynamicImage, _lang: Lang) -> Result<OcrPage> {
        Err(DqError::Ocr("OCR devre disi (ocr.engine = \"none\")".into()))
    }
    fn available(&self) -> bool {
        false
    }
}

/// Harici `tesseract` binary'sini TSV modunda calistiran motor.
///
/// TSV cikti secilmesinin sebebi kelime bazinda guven skoru vermesidir;
/// bu skor hem chunk kalitesini hem de cevabin guven esigini besler.
pub struct TesseractOcr {
    bin: String,
    langs: String,
    min_conf: f32,
}

impl TesseractOcr {
    pub fn new(bin: impl Into<String>, langs: impl Into<String>, min_conf: f32) -> Self {
        Self {
            bin: bin.into(),
            langs: langs.into(),
            min_conf,
        }
    }

    fn lang_arg(&self, lang: Lang) -> String {
        match lang {
            Lang::Tr => "tur".into(),
            Lang::En => "eng".into(),
            Lang::Unknown => self.langs.clone(),
        }
    }
}

impl OcrEngine for TesseractOcr {
    fn name(&self) -> &'static str {
        "tesseract"
    }

    fn available(&self) -> bool {
        Command::new(&self.bin)
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }

    fn recognize(&self, img: &DynamicImage, lang: Lang) -> Result<OcrPage> {
        let dir = tempfile::tempdir().map_err(|e| DqError::Ocr(format!("gecici dizin: {e}")))?;
        let input = dir.path().join("page.png");
        {
            let mut f = std::fs::File::create(&input)
                .map_err(|e| DqError::Ocr(format!("gecici dosya: {e}")))?;
            let mut buf = Vec::new();
            img.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
                .map_err(|e| DqError::Ocr(format!("PNG yazilamadi: {e}")))?;
            f.write_all(&buf).map_err(|e| DqError::Ocr(format!("yazma: {e}")))?;
        }

        let out = Command::new(&self.bin)
            .arg(&input)
            .arg("stdout")
            .arg("-l")
            .arg(self.lang_arg(lang))
            .arg("--psm")
            .arg("3")
            .arg("--oem")
            .arg("1")
            .arg("tsv")
            .stderr(Stdio::piped())
            .output()
            .map_err(|e| DqError::Ocr(format!("tesseract calistirilamadi: {e}")))?;

        if !out.status.success() {
            let err = String::from_utf8_lossy(&out.stderr);
            return Err(DqError::Ocr(format!("tesseract hatasi: {}", err.trim())));
        }
        Ok(parse_tsv(&String::from_utf8_lossy(&out.stdout), self.min_conf))
    }
}

/// Tesseract TSV ciktisini metne cevirir, kelime guvenlerini toplar.
fn parse_tsv(tsv: &str, min_conf: f32) -> OcrPage {
    let mut lines: Vec<String> = Vec::new();
    let mut current_line_key = (0i64, 0i64, 0i64, 0i64);
    let mut current: Vec<String> = Vec::new();
    let mut conf_sum = 0f32;
    let mut conf_n = 0usize;
    let mut dropped = 0usize;

    for (i, row) in tsv.lines().enumerate() {
        if i == 0 {
            continue; // baslik satiri
        }
        let cols: Vec<&str> = row.split('\t').collect();
        if cols.len() < 12 {
            continue;
        }
        let level: i64 = cols[0].parse().unwrap_or(0);
        if level != 5 {
            continue; // yalnizca kelime seviyesi
        }
        let key = (
            cols[1].parse().unwrap_or(0),
            cols[2].parse().unwrap_or(0),
            cols[3].parse().unwrap_or(0),
            cols[4].parse().unwrap_or(0),
        );
        let conf: f32 = cols[10].parse().unwrap_or(-1.0) / 100.0;
        let text = cols[11].trim();
        if text.is_empty() {
            continue;
        }
        if key != current_line_key {
            if !current.is_empty() {
                lines.push(current.join(" "));
                current.clear();
            }
            current_line_key = key;
        }
        if conf < min_conf {
            dropped += 1;
            continue;
        }
        conf_sum += conf;
        conf_n += 1;
        current.push(text.to_string());
    }
    if !current.is_empty() {
        lines.push(current.join(" "));
    }

    OcrPage {
        text: lines.join("\n"),
        confidence: if conf_n == 0 { 0.0 } else { conf_sum / conf_n as f32 },
        dropped_words: dropped,
    }
}

/// Konfigurasyona gore motoru secer. Istenen motor yoksa OCR devre disi
/// birakilir ve uyari loglanir (sistem yine de calisir).
pub fn build(cfg: &dq_core::config::OcrConfig) -> Box<dyn OcrEngine> {
    match cfg.engine.as_str() {
        "none" => Box::new(NoopOcr),
        _ => {
            let engine = TesseractOcr::new(
                cfg.tesseract_bin.clone(),
                cfg.tesseract_langs.clone(),
                cfg.min_line_confidence,
            );
            if engine.available() {
                Box::new(engine)
            } else {
                tracing::warn!(
                    bin = %cfg.tesseract_bin,
                    "tesseract bulunamadi; taranmis belgeler metne cevrilemeyecek"
                );
                Box::new(NoopOcr)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tsv_is_parsed_into_lines() {
        let tsv = "level\tpage\tblock\tpar\tline\tword\tleft\ttop\twidth\theight\tconf\ttext\n\
5\t1\t1\t1\t1\t1\t0\t0\t10\t10\t95\tBakım\n\
5\t1\t1\t1\t1\t2\t0\t0\t10\t10\t90\ttalimatı\n\
5\t1\t1\t1\t2\t1\t0\t0\t10\t10\t20\tçöp\n\
5\t1\t1\t1\t2\t2\t0\t0\t10\t10\t88\tMadde\n";
        let page = parse_tsv(tsv, 0.35);
        assert_eq!(page.text, "Bakım talimatı\nMadde");
        assert_eq!(page.dropped_words, 1);
        assert!(page.confidence > 0.9);
    }
}
