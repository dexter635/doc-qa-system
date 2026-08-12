//! Leptos `on:input`/`on:change` olaylarindan DOM degerini okuma yardimcilari.
//!
//! Leptos surumleri arasinda `event_target_value` gibi yardimcilarin ihrac
//! yolu degisebildigi icin, dogrudan `web_sys` ile yazmak daha kararlidir.

use wasm_bindgen::JsCast;

pub fn input_value(ev: &web_sys::Event) -> String {
    ev.target()
        .and_then(|t| t.dyn_into::<web_sys::HtmlInputElement>().ok())
        .map(|el| el.value())
        .unwrap_or_default()
}

pub fn textarea_value(ev: &web_sys::Event) -> String {
    ev.target()
        .and_then(|t| t.dyn_into::<web_sys::HtmlTextAreaElement>().ok())
        .map(|el| el.value())
        .unwrap_or_default()
}

pub fn select_value(ev: &web_sys::Event) -> String {
    ev.target()
        .and_then(|t| t.dyn_into::<web_sys::HtmlSelectElement>().ok())
        .map(|el| el.value())
        .unwrap_or_default()
}

/// `<input type="file">` degisiminden ilk secilen dosyayi dondurur.
pub fn first_file(ev: &web_sys::Event) -> Option<web_sys::File> {
    let input = ev.target()?.dyn_into::<web_sys::HtmlInputElement>().ok()?;
    let list = input.files()?;
    list.get(0)
}
