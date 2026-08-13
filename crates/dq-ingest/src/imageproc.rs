//! OCR oncesi goruntu hazirligi.
//!
//! Amac: dusuk cozunurluklu / dusuk kontrastli taramalarda OCR dogrulugunu
//! artirmak. Adimlar bilincli olarak sade tutulmustur; agir islemler
//! (deskew, dewarp) OCR kalitesine katkisi olcuye deger olmadikca eklenmez.

use dq_core::{DqError, Result};
use image::{DynamicImage, GenericImageView, GrayImage};

use crate::detect::FileKind;

/// Kaynak tuketimini sinirlayarak goruntuyu bellekten yukler.
pub fn load(bytes: &[u8], kind: FileKind) -> Result<DynamicImage> {
    let format = match kind {
        FileKind::Jpeg => image::ImageFormat::Jpeg,
        FileKind::Png => image::ImageFormat::Png,
        FileKind::Pdf => return Err(DqError::Ingest("PDF goruntu olarak yuklenemez".into())),
    };
    let mut reader = image::ImageReader::with_format(std::io::Cursor::new(bytes), format);
    // Dekompresyon bombasi korumasi.
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(20_000);
    limits.max_image_height = Some(20_000);
    limits.max_alloc = Some(512 * 1024 * 1024);
    reader.limits(limits);
    reader
        .decode()
        .map_err(|e| DqError::Ingest(format!("goruntu cozulemedi: {e}")))
}

/// OCR icin normalize edilmis goruntu uretir.
pub fn preprocess(img: &DynamicImage) -> DynamicImage {
    let gray = img.to_luma8();
    let gray = upscale_if_small(gray);
    let gray = if is_low_contrast(&gray) {
        binarize_otsu(&gray)
    } else {
        gray
    };
    DynamicImage::ImageLuma8(gray)
}

/// Tesseract ~300 DPI'a denk gelen metin yuksekligi bekler. Kucuk goruntuler
/// buyutulmezse karakterler tanimsiz kalir.
fn upscale_if_small(img: GrayImage) -> GrayImage {
    const TARGET_MIN_DIM: u32 = 1400;
    let (w, h) = img.dimensions();
    let min_dim = w.min(h);
    if min_dim >= TARGET_MIN_DIM || min_dim == 0 {
        return img;
    }
    let scale = (TARGET_MIN_DIM as f32 / min_dim as f32).min(3.0);
    let nw = ((w as f32) * scale).round() as u32;
    let nh = ((h as f32) * scale).round() as u32;
    image::imageops::resize(&img, nw, nh, image::imageops::FilterType::Lanczos3)
}

fn is_low_contrast(img: &GrayImage) -> bool {
    let hist = histogram(img);
    let total: u64 = hist.iter().sum();
    if total == 0 {
        return false;
    }
    let mean: f64 = hist
        .iter()
        .enumerate()
        .map(|(i, c)| i as f64 * *c as f64)
        .sum::<f64>()
        / total as f64;
    let var: f64 = hist
        .iter()
        .enumerate()
        .map(|(i, c)| (i as f64 - mean).powi(2) * *c as f64)
        .sum::<f64>()
        / total as f64;
    var.sqrt() < 55.0
}

/// Otsu esikleme: histogrami iki sinifa ayirip sinif ici varyansi en aza indirir.
pub fn binarize_otsu(img: &GrayImage) -> GrayImage {
    let hist = histogram(img);
    let total: u64 = hist.iter().sum();
    if total == 0 {
        return img.clone();
    }
    let sum_all: f64 = hist
        .iter()
        .enumerate()
        .map(|(i, c)| i as f64 * *c as f64)
        .sum();

    let (mut w_b, mut sum_b, mut best_var, mut threshold) = (0u64, 0f64, -1f64, 128usize);
    for (t, &count) in hist.iter().enumerate() {
        w_b += count;
        if w_b == 0 {
            continue;
        }
        let w_f = total - w_b;
        if w_f == 0 {
            break;
        }
        sum_b += t as f64 * count as f64;
        let m_b = sum_b / w_b as f64;
        let m_f = (sum_all - sum_b) / w_f as f64;
        let between = w_b as f64 * w_f as f64 * (m_b - m_f).powi(2);
        if between > best_var {
            best_var = between;
            threshold = t;
        }
    }

    let mut out = img.clone();
    for p in out.pixels_mut() {
        p.0[0] = if (p.0[0] as usize) > threshold {
            255
        } else {
            0
        };
    }
    out
}

fn histogram(img: &GrayImage) -> [u64; 256] {
    let mut hist = [0u64; 256];
    for p in img.pixels() {
        hist[p.0[0] as usize] += 1;
    }
    hist
}

/// Goruntunun anlamli icerik tasiyip tasimadigi (bos sayfa tespiti).
pub fn is_blank(img: &DynamicImage) -> bool {
    let gray = img.to_luma8();
    let bin = binarize_otsu(&gray);
    let dark = bin.pixels().filter(|p| p.0[0] < 128).count();
    let total = (bin.width() * bin.height()) as usize;
    total == 0 || (dark as f32 / total as f32) < 0.002
}

/// Goruntuyu OCR icin makul bir ust sinira indirger.
pub fn cap_dimensions(img: DynamicImage, max_dim: u32) -> DynamicImage {
    let (w, h) = img.dimensions();
    if w.max(h) <= max_dim {
        return img;
    }
    let scale = max_dim as f32 / w.max(h) as f32;
    img.resize(
        ((w as f32) * scale) as u32,
        ((h as f32) * scale) as u32,
        image::imageops::FilterType::Lanczos3,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn otsu_separates_two_levels() {
        let mut img = GrayImage::new(10, 10);
        for (i, p) in img.pixels_mut().enumerate() {
            p.0[0] = if i % 2 == 0 { 20 } else { 230 };
        }
        let bin = binarize_otsu(&img);
        let values: std::collections::HashSet<u8> = bin.pixels().map(|p| p.0[0]).collect();
        assert!(values.is_subset(&[0u8, 255u8].into_iter().collect()));
    }

    #[test]
    fn blank_page_is_detected() {
        let img = DynamicImage::ImageLuma8(GrayImage::from_pixel(50, 50, image::Luma([255])));
        assert!(is_blank(&img));
    }
}
