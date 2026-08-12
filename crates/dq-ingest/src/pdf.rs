//! PDF metin ve gomulu goruntu cikarimi.
//!
//! Iki asamali strateji:
//! 1. `pdf-extract` ile metin katmani okunur (dijital dogmus PDF'ler).
//! 2. Metin yetersizse (taranmis belge) sayfadaki gomulu goruntuler
//!    `lopdf` ile cikarilip OCR'a gonderilir.
//!
//! Boylece harici bir yerel kutuphaneye (pdfium/mupdf) bagimlilik olmadan
//! taranmis PDF'ler de islenebilir.

use std::collections::BTreeMap;

use dq_core::{DqError, Result};
use image::DynamicImage;
use lopdf::{Dictionary, Document, Object};

/// PDF'in her sayfasindaki gomulu metin katmani.
///
/// `pdf-extract` bozuk PDF'lerde panikleyebildigi icin cagri `catch_unwind`
/// ile izole edilir; tek bir hatali belge sunucuyu dusurmemelidir.
pub fn page_texts(bytes: &[u8]) -> Result<Vec<String>> {
    let owned = bytes.to_vec();
    let result = std::panic::catch_unwind(move || pdf_extract::extract_text_from_mem_by_pages(&owned));
    match result {
        Ok(Ok(pages)) => Ok(pages),
        Ok(Err(e)) => Err(DqError::Ingest(format!("PDF metni okunamadi: {e}"))),
        Err(_) => Err(DqError::Ingest(
            "PDF ayristirilamadi (bozuk veya desteklenmeyen yapi)".into(),
        )),
    }
}

/// Sayfa sayisi. Metin cikarimi basarisiz olsa bile bilinmelidir.
pub fn page_count(bytes: &[u8]) -> Result<usize> {
    let doc = load(bytes)?;
    Ok(doc.get_pages().len())
}

/// Sayfa numarasi (1'den baslar) -> o sayfadaki gomulu goruntuler.
///
/// Taranmis belgelerde sayfa basina genellikle tek bir buyuk goruntu bulunur.
pub fn page_images(bytes: &[u8], only_pages: &[usize]) -> Result<BTreeMap<usize, Vec<DynamicImage>>> {
    let doc = load(bytes)?;
    let mut out: BTreeMap<usize, Vec<DynamicImage>> = BTreeMap::new();

    for (page_no, page_id) in doc.get_pages() {
        let page_no = page_no as usize;
        if !only_pages.is_empty() && !only_pages.contains(&page_no) {
            continue;
        }
        let Some(resources) = resolve_resources(&doc, page_id) else {
            continue;
        };
        let Ok(xobjects) = resources.get(b"XObject").and_then(|o| resolve_dict(&doc, o)) else {
            continue;
        };

        for (_, obj_ref) in xobjects.iter() {
            let Ok(id) = obj_ref.as_reference() else { continue };
            let Ok(stream) = doc.get_object(id).and_then(|o| o.as_stream()) else {
                continue;
            };
            let is_image = stream
                .dict
                .get(b"Subtype")
                .and_then(|o| o.as_name())
                .map(|n| n == b"Image")
                .unwrap_or(false);
            if !is_image {
                continue;
            }
            match decode_image_stream(stream) {
                Ok(img) => out.entry(page_no).or_default().push(img),
                Err(e) => tracing::debug!(page = page_no, error = %e, "gomulu goruntu cozulemedi"),
            }
        }
    }
    Ok(out)
}

fn load(bytes: &[u8]) -> Result<Document> {
    Document::load_mem(bytes).map_err(|e| DqError::Ingest(format!("PDF acilamadi: {e}")))
}

/// Sayfanin `Resources` sozlugu; yoksa `Parent` zinciri yukari dogru taranir.
fn resolve_resources(doc: &Document, page_id: lopdf::ObjectId) -> Option<Dictionary> {
    let mut current = page_id;
    for _ in 0..8 {
        let dict = doc.get_object(current).ok()?.as_dict().ok()?;
        if let Some(res) = dict.get(b"Resources").ok() {
            if let Ok(d) = resolve_dict(doc, res) {
                return Some(d);
            }
        }
        let parent = dict.get(b"Parent").ok()?.as_reference().ok()?;
        current = parent;
    }
    None
}

fn resolve_dict(doc: &Document, obj: &Object) -> std::result::Result<Dictionary, lopdf::Error> {
    match obj {
        Object::Reference(id) => doc.get_object(*id)?.as_dict().cloned(),
        other => other.as_dict().cloned(),
    }
}

fn decode_image_stream(stream: &lopdf::Stream) -> Result<DynamicImage> {
    let filters = stream_filters(stream);
    let width = stream
        .dict
        .get(b"Width")
        .and_then(|o| o.as_i64())
        .map_err(|_| DqError::Ingest("goruntu genisligi yok".into()))? as u32;
    let height = stream
        .dict
        .get(b"Height")
        .and_then(|o| o.as_i64())
        .map_err(|_| DqError::Ingest("goruntu yuksekligi yok".into()))? as u32;

    // Kaynak tuketimine karsi ust sinir (goruntu bombasi korumasi).
    if width == 0 || height == 0 || (width as u64) * (height as u64) > 80_000_000 {
        return Err(DqError::Ingest("gecersiz veya asiri buyuk goruntu".into()));
    }

    if filters.iter().any(|f| f == "DCTDecode") {
        return image::load_from_memory_with_format(&stream.content, image::ImageFormat::Jpeg)
            .map_err(|e| DqError::Ingest(format!("JPEG cozulemedi: {e}")));
    }
    if filters.iter().any(|f| f == "JPXDecode") {
        return Err(DqError::Ingest("JPEG2000 goruntuler desteklenmiyor".into()));
    }

    let s = stream.clone();
    let data = s
        .decompressed_content()
        .map_err(|e| DqError::Ingest(format!("akis acilamadi: {e}")))?;

    let bpc = stream.dict.get(b"BitsPerComponent").and_then(|o| o.as_i64()).unwrap_or(8);
    let components = color_components(stream);

    match (bpc, components) {
        (8, 1) => {
            let buf = image::GrayImage::from_raw(width, height, data)
                .ok_or_else(|| DqError::Ingest("gri tonlama tamponu uyumsuz".into()))?;
            Ok(DynamicImage::ImageLuma8(buf))
        }
        (8, 3) => {
            let buf = image::RgbImage::from_raw(width, height, data)
                .ok_or_else(|| DqError::Ingest("RGB tamponu uyumsuz".into()))?;
            Ok(DynamicImage::ImageRgb8(buf))
        }
        (1, 1) => {
            // 1 bit/piksel: satir basi baytlara hizalidir.
            let row_bytes = width.div_ceil(8) as usize;
            let mut pixels = Vec::with_capacity((width * height) as usize);
            for y in 0..height as usize {
                for x in 0..width as usize {
                    let byte = data.get(y * row_bytes + x / 8).copied().unwrap_or(0xFF);
                    let bit = (byte >> (7 - (x % 8))) & 1;
                    pixels.push(if bit == 1 { 255u8 } else { 0u8 });
                }
            }
            let buf = image::GrayImage::from_raw(width, height, pixels)
                .ok_or_else(|| DqError::Ingest("1-bit tampon uyumsuz".into()))?;
            Ok(DynamicImage::ImageLuma8(buf))
        }
        _ => Err(DqError::Ingest(format!(
            "desteklenmeyen goruntu formati (bpc={bpc}, bilesen={components})"
        ))),
    }
}

fn stream_filters(stream: &lopdf::Stream) -> Vec<String> {
    let Ok(obj) = stream.dict.get(b"Filter") else {
        return Vec::new();
    };
    match obj {
        Object::Name(n) => vec![String::from_utf8_lossy(n).to_string()],
        Object::Array(items) => items
            .iter()
            .filter_map(|o| o.as_name().ok())
            .map(|n| String::from_utf8_lossy(n).to_string())
            .collect(),
        _ => Vec::new(),
    }
}

fn color_components(stream: &lopdf::Stream) -> usize {
    match stream.dict.get(b"ColorSpace") {
        Ok(Object::Name(n)) => match n.as_slice() {
            b"DeviceRGB" | b"CalRGB" => 3,
            b"DeviceCMYK" => 4,
            _ => 1,
        },
        Ok(Object::Array(items)) => {
            // [/Indexed /DeviceRGB 255 <palet>] gibi yapilar tek bilesenlidir.
            match items.first().and_then(|o| o.as_name().ok()) {
                Some(b"Indexed") | Some(b"ICCBased") => 1,
                _ => 1,
            }
        }
        _ => 1,
    }
}
