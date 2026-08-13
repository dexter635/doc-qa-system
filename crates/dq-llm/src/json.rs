//! Zayif/guclu yerel modellerden gelen serbest metin icinden JSON cikarimi.
//!
//! Yerel modeller (ozellikle kucuk parametreli olanlar) "structured output"
//! / function-calling'i her zaman duzgun desteklemez; JSON'u kod bloguna
//! sarabilir, oncesine/sonrasina aciklama ekleyebilir. Bu yuzden JSON, ciktinin
//! *tamami* olarak degil, ilk dengeli suslu parantez blogu olarak aranir.

/// Metindeki ilk dengeli `{...}` blogunu bulup ayristirmayi dener.
///
/// Basarisiz olursa `None` doner; cagiran taraf bunu bir sezgisel (heuristic)
/// yedekle karsilamalidir - bu asla panic'e ya da hataya yol acmamalidir,
/// cunku model ciktisi dogal olarak guvenilmezdir.
pub fn extract_json_object(text: &str) -> Option<serde_json::Value> {
    let bytes = text.as_bytes();
    let start = text.find('{')?;
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escaped = false;

    for (i, &b) in bytes.iter().enumerate().skip(start) {
        let c = b as char;
        if in_string {
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_string = false;
            }
            continue;
        }
        match c {
            '"' => in_string = true,
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    let candidate = &text[start..=i];
                    return serde_json::from_str(candidate).ok();
                }
            }
            _ => {}
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_plain_json() {
        let v = extract_json_object(r#"{"a": 1, "b": [1,2]}"#).unwrap();
        assert_eq!(v["a"], 1);
    }

    #[test]
    fn extracts_json_wrapped_in_prose_and_code_fence() {
        let text = "Elbette, iste plan:\n```json\n{\"sub_queries\": [\"a\", \"b\"]}\n```\nUmarim yardimci olur.";
        let v = extract_json_object(text).unwrap();
        assert_eq!(v["sub_queries"][0], "a");
    }

    #[test]
    fn handles_nested_braces_and_strings_with_braces() {
        let text =
            r#"{"reasoning": "kullanici {ornek} istedi", "sub_queries": ["x"], "meta": {"n": 2}}"#;
        let v = extract_json_object(text).unwrap();
        assert_eq!(v["meta"]["n"], 2);
    }

    #[test]
    fn returns_none_for_garbage() {
        assert!(extract_json_object("bu metinde hic json yok").is_none());
    }

    #[test]
    fn returns_none_for_unbalanced_braces() {
        assert!(extract_json_object("{\"a\": 1").is_none());
    }
}
