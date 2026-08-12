//! Yapiya duyarli parcalama (chunking).
//!
//! Sabit boyutlu pencereleme yerine belge yapisi (baslik / paragraf / sayfa)
//! korunur. Sebep: teknik dokumanlarda "3.2 Periyodik Bakim" gibi basliklar
//! chunk'in anlamini belirler; baslik kaybolursa hem gomme kalitesi hem de
//! kullaniciya gosterilen kaynak baglami bozulur.

use dq_core::config::IngestConfig;
use dq_core::text::{estimate_tokens, split_sentences};
use dq_core::{Chunk, Classification, Lang, PageText};
use once_cell::sync::Lazy;
use regex::Regex;
use uuid::Uuid;

static NUMBERED_HEADING: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^\s*(\d+(?:\.\d+){0,4})\.?\s+(\S.{0,110})$").unwrap());
static KEYWORD_HEADING: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)^\s*(madde|b[öo]l[üu]m|k[ıi]s[ıi]m|ek|annex|section|chapter|appendix|part)\s+([0-9IVXA-Z]+)")
        .unwrap()
});

#[derive(Debug, Clone)]
struct Block {
    text: String,
    page: usize,
    heading_level: Option<usize>,
    confidence: f32,
}

/// Satirin baslik olup olmadigini ve seviyesini tahmin eder.
fn heading_level(line: &str) -> Option<usize> {
    let t = line.trim();
    if t.is_empty() || t.chars().count() > 130 {
        return None;
    }
    if let Some(c) = NUMBERED_HEADING.captures(t) {
        let dots = c.get(1).map(|m| m.as_str().matches('.').count()).unwrap_or(0);
        // "1.5 Litre" gibi olcu ifadelerini baslik sanmamak icin: baslik
        // satiri cumle noktalama isareti ile bitmemelidir.
        if !t.ends_with('.') || t.split_whitespace().count() <= 12 {
            return Some((dots + 1).min(5));
        }
    }
    if KEYWORD_HEADING.is_match(t) {
        return Some(1);
    }
    let letters: Vec<char> = t.chars().filter(|c| c.is_alphabetic()).collect();
    if letters.len() >= 4 {
        let upper = letters.iter().filter(|c| c.is_uppercase()).count();
        if upper as f32 / letters.len() as f32 > 0.85 && !t.ends_with('.') {
            return Some(1);
        }
    }
    None
}

fn to_blocks(pages: &[PageText]) -> Vec<Block> {
    let mut blocks = Vec::new();
    for page in pages {
        let mut buf: Vec<String> = Vec::new();
        let flush = |buf: &mut Vec<String>, blocks: &mut Vec<Block>| {
            if buf.is_empty() {
                return;
            }
            blocks.push(Block {
                text: buf.join(" "),
                page: page.page_no,
                heading_level: None,
                confidence: page.confidence,
            });
            buf.clear();
        };

        for line in page.text.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                flush(&mut buf, &mut blocks);
                continue;
            }
            if let Some(level) = heading_level(trimmed) {
                flush(&mut buf, &mut blocks);
                blocks.push(Block {
                    text: trimmed.to_string(),
                    page: page.page_no,
                    heading_level: Some(level),
                    confidence: page.confidence,
                });
                continue;
            }
            buf.push(trimmed.to_string());
        }
        flush(&mut buf, &mut blocks);
    }
    blocks
}

/// Sayfa metinlerini indekslenebilir chunk'lara cevirir.
pub fn build_chunks(
    doc_id: Uuid,
    pages: &[PageText],
    cfg: &IngestConfig,
    classification: Classification,
    doc_lang: Lang,
) -> Vec<Chunk> {
    let blocks = to_blocks(pages);
    let mut chunks: Vec<Chunk> = Vec::new();
    let mut heading_stack: Vec<(usize, String)> = Vec::new();

    let mut buf: Vec<String> = Vec::new();
    let mut buf_tokens = 0usize;
    let mut page_from = pages.first().map(|p| p.page_no).unwrap_or(1);
    let mut page_to = page_from;
    let mut conf_sum = 0f32;
    let mut conf_n = 0usize;
    let mut pending_overlap: Vec<String> = Vec::new();

    macro_rules! flush_chunk {
        () => {
            if buf_tokens >= cfg.min_chunk_tokens && !buf.is_empty() {
                let text = buf.join("\n");
                let heading_path = if heading_stack.is_empty() {
                    None
                } else {
                    Some(
                        heading_stack
                            .iter()
                            .map(|(_, h)| h.as_str())
                            .collect::<Vec<_>>()
                            .join(" > "),
                    )
                };
                let confidence = if conf_n == 0 { 1.0 } else { conf_sum / conf_n as f32 };
                chunks.push(Chunk {
                    id: Uuid::new_v4(),
                    doc_id,
                    ordinal: chunks.len(),
                    page_from,
                    page_to,
                    token_estimate: estimate_tokens(&text),
                    lang: if doc_lang == Lang::Unknown {
                        dq_core::text::detect_lang(&text)
                    } else {
                        doc_lang
                    },
                    classification,
                    confidence,
                    heading_path,
                    text,
                });
                pending_overlap = tail_sentences(&buf, cfg.chunk_overlap_tokens);
                buf.clear();
                buf_tokens = 0;
                conf_sum = 0.0;
                conf_n = 0;
            }
        };
    }

    for block in &blocks {
        if let Some(level) = block.heading_level {
            // Yeni baslik: mevcut chunk yeterince doluysa kapat.
            if buf_tokens >= cfg.chunk_tokens / 2 {
                flush_chunk!();
            }
            heading_stack.retain(|(l, _)| *l < level);
            heading_stack.push((level, block.text.clone()));
            if buf.is_empty() {
                page_from = block.page;
            }
            page_to = block.page;
            // Baslik metni chunk icerigine de girer; gomme sinyali guclenir.
            buf.push(block.text.clone());
            buf_tokens += estimate_tokens(&block.text);
            continue;
        }

        let block_tokens = estimate_tokens(&block.text);
        if buf.is_empty() {
            if !pending_overlap.is_empty() {
                let ov = pending_overlap.join(" ");
                buf_tokens += estimate_tokens(&ov);
                buf.push(ov);
                pending_overlap.clear();
            }
            page_from = block.page;
        }
        page_to = block.page;
        conf_sum += block.confidence;
        conf_n += 1;

        if block_tokens > cfg.chunk_tokens {
            // Cok uzun paragraf: cumle bazinda bolunur.
            for sentence in split_sentences(&block.text) {
                let st = estimate_tokens(&sentence);
                if buf_tokens + st > cfg.chunk_tokens && buf_tokens >= cfg.min_chunk_tokens {
                    flush_chunk!();
                    if !pending_overlap.is_empty() {
                        let ov = pending_overlap.join(" ");
                        buf_tokens += estimate_tokens(&ov);
                        buf.push(ov);
                        pending_overlap.clear();
                    }
                    page_from = block.page;
                }
                buf_tokens += st;
                buf.push(sentence);
            }
            continue;
        }

        if buf_tokens + block_tokens > cfg.chunk_tokens && buf_tokens >= cfg.min_chunk_tokens {
            flush_chunk!();
            if !pending_overlap.is_empty() {
                let ov = pending_overlap.join(" ");
                buf_tokens += estimate_tokens(&ov);
                buf.push(ov);
                pending_overlap.clear();
            }
            page_from = block.page;
        }
        buf_tokens += block_tokens;
        buf.push(block.text.clone());
    }

    // Son parca minimum esigin altinda olsa bile kaybedilmemelidir.
    if !buf.is_empty() {
        let text = buf.join("\n");
        if estimate_tokens(&text) >= 8 {
            let heading_path = if heading_stack.is_empty() {
                None
            } else {
                Some(
                    heading_stack
                        .iter()
                        .map(|(_, h)| h.as_str())
                        .collect::<Vec<_>>()
                        .join(" > "),
                )
            };
            let confidence = if conf_n == 0 { 1.0 } else { conf_sum / conf_n as f32 };
            chunks.push(Chunk {
                id: Uuid::new_v4(),
                doc_id,
                ordinal: chunks.len(),
                page_from,
                page_to,
                token_estimate: estimate_tokens(&text),
                lang: if doc_lang == Lang::Unknown {
                    dq_core::text::detect_lang(&text)
                } else {
                    doc_lang
                },
                classification,
                confidence,
                heading_path,
                text,
            });
        }
    }

    chunks
}

/// Onceki chunk'in sonundan, verilen token butcesi kadar cumle dondurur.
fn tail_sentences(buf: &[String], overlap_tokens: usize) -> Vec<String> {
    if overlap_tokens == 0 {
        return Vec::new();
    }
    let joined = buf.join(" ");
    let sentences = split_sentences(&joined);
    let mut out: Vec<String> = Vec::new();
    let mut tokens = 0usize;
    for s in sentences.iter().rev() {
        let t = estimate_tokens(s);
        if tokens + t > overlap_tokens && !out.is_empty() {
            break;
        }
        tokens += t;
        out.push(s.clone());
    }
    out.reverse();
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use dq_core::ExtractionMethod;

    fn page(no: usize, text: &str) -> PageText {
        PageText {
            page_no: no,
            text: text.to_string(),
            method: ExtractionMethod::PdfText,
            confidence: 1.0,
            lang: Lang::Tr,
        }
    }

    #[test]
    fn detects_headings() {
        assert_eq!(heading_level("3.2 Periyodik Bakım"), Some(2));
        assert_eq!(heading_level("MADDE 7"), Some(1));
        assert_eq!(heading_level("GENEL HÜKÜMLER"), Some(1));
        assert_eq!(heading_level("Bu bir normal cümledir ve başlık değildir."), None);
    }

    #[test]
    fn chunks_carry_heading_path() {
        let cfg = IngestConfig::default();
        let body = "Motorun periyodik bakımı 500 saatte bir yapılır. ".repeat(30);
        let text = format!("3. BAKIM\n\n3.2 Periyodik Bakım\n\n{body}");
        let chunks = build_chunks(
            Uuid::new_v4(),
            &[page(1, &text)],
            &cfg,
            Classification::Restricted,
            Lang::Tr,
        );
        assert!(!chunks.is_empty());
        let path = chunks[0].heading_path.as_deref().unwrap_or("");
        assert!(path.contains("BAKIM"), "path: {path}");
        assert!(chunks.iter().all(|c| c.token_estimate <= cfg.chunk_tokens + 60));
    }

    #[test]
    fn page_ranges_are_tracked() {
        let cfg = IngestConfig::default();
        let chunks = build_chunks(
            Uuid::new_v4(),
            &[page(1, &"Birinci sayfa metni. ".repeat(60)), page(2, &"İkinci sayfa metni. ".repeat(60))],
            &cfg,
            Classification::Unclassified,
            Lang::Tr,
        );
        assert!(chunks.iter().any(|c| c.page_from == 1));
        assert!(chunks.iter().any(|c| c.page_to == 2));
    }
}
