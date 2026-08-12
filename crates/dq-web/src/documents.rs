use leptos::prelude::*;
use leptos::task::spawn_local;

use crate::api::{self, Document};
use crate::auth::AuthState;
use crate::dom::{first_file, select_value};
use crate::ui;

const CLASSIFICATIONS: [&str; 5] = ["unclassified", "restricted", "confidential", "secret", "top_secret"];

fn clearance_rank(c: &str) -> usize {
    CLASSIFICATIONS.iter().position(|x| *x == c.to_ascii_lowercase()).unwrap_or(0)
}

fn reload_documents(
    token: String,
    documents: RwSignal<Vec<Document>>,
    loading: RwSignal<bool>,
    error: RwSignal<Option<String>>,
) {
    loading.set(true);
    spawn_local(async move {
        match api::list_documents(&token).await {
            Ok(docs) => {
                documents.set(docs);
                error.set(None);
            }
            Err(e) => error.set(Some(e.to_string())),
        }
        loading.set(false);
    });
}

#[component]
pub fn DocumentsView(auth: AuthState) -> impl IntoView {
    let documents = RwSignal::new(Vec::<Document>::new());
    let loading = RwSignal::new(true);
    let error = RwSignal::new(Option::<String>::None);

    let selected_file = RwSignal::new(Option::<web_sys::File>::None);
    let selected_name = RwSignal::new(String::new());
    let classification = RwSignal::new("unclassified".to_string());
    let upload_busy = RwSignal::new(false);
    let upload_msg = RwSignal::new(Option::<(bool, String)>::None);

    reload_documents(auth.token.clone(), documents, loading, error);

    let allowed_classifications: Vec<&'static str> = CLASSIFICATIONS
        .iter()
        .copied()
        .filter(|c| clearance_rank(c) <= clearance_rank(&auth.clearance))
        .collect();

    let on_file_change = move |ev: web_sys::Event| {
        if let Some(f) = first_file(&ev) {
            selected_name.set(f.name());
            selected_file.set(Some(f));
        }
    };

    let token_for_upload = auth.token.clone();
    let token_for_upload_reload = auth.token.clone();
    let do_upload = move |_| {
        let Some(file) = selected_file.get_untracked() else {
            upload_msg.set(Some((false, "Önce bir dosya seçin.".into())));
            return;
        };
        let token = token_for_upload.clone();
        let token_reload = token_for_upload_reload.clone();
        let cls = classification.get_untracked();
        upload_busy.set(true);
        upload_msg.set(None);
        spawn_local(async move {
            match api::upload_document(&token, file, &cls).await {
                Ok(res) => {
                    let mut msg = format!("'{}' yüklendi ve işleniyor.", res.document.filename);
                    if !res.warnings.is_empty() {
                        msg.push_str(&format!(" Uyarılar: {}", res.warnings.join(" ")));
                    }
                    upload_msg.set(Some((true, msg)));
                    selected_file.set(None);
                    selected_name.set(String::new());
                    reload_documents(token_reload, documents, loading, error);
                }
                Err(e) => upload_msg.set(Some((false, e.to_string()))),
            }
            upload_busy.set(false);
        });
    };

    let is_admin = auth.is_admin();

    view! {
        <div class="card">
            <h2>"Belge Yükle"</h2>
            <div class="row">
                <input type="file" accept=".pdf,.jpg,.jpeg,.png" on:change=on_file_change />
                <select on:change=move |ev| classification.set(select_value(&ev))>
                    {allowed_classifications.iter().map(|c| {
                        let c = c.to_string();
                        view! { <option value=c.clone()>{ui::classification_label(&c)}</option> }
                    }).collect::<Vec<_>>()}
                </select>
                <button class="btn" on:click=do_upload disabled=move || upload_busy.get() || selected_file.get().is_none()>
                    {move || if upload_busy.get() { "Yükleniyor…" } else { "Yükle" }}
                </button>
            </div>
            {move || (!selected_name.get().is_empty()).then(|| view! { <p class="mono">{selected_name.get()}</p> })}
            {move || upload_msg.get().map(|(ok, m)| {
                let cls = if ok { "msg ok" } else { "msg error" };
                view! { <div class=cls>{m}</div> }
            })}
        </div>

        <div class="card">
            <h2>"Belgeler"</h2>
            {move || error.get().map(|e| view! { <div class="msg error">{e}</div> })}
            {move || {
                if loading.get() {
                    view! { <p class="empty"><span class="spinner"></span>" Yükleniyor…"</p> }.into_any()
                } else if documents.get().is_empty() {
                    view! { <p class="empty">"Henüz belge yüklenmedi."</p> }.into_any()
                } else {
                    let rows = documents.get().into_iter().map(|d| {
                        let id = d.id.clone();
                        let token_row = auth.token.clone();
                        view! {
                            <tr>
                                <td>{d.filename.clone()}</td>
                                <td><span class=ui::status_class(&d.status)>{ui::status_label(&d.status)}</span></td>
                                <td><span class=ui::classification_class(&d.classification)>{ui::classification_label(&d.classification)}</span></td>
                                <td>{ui::lang_label(&d.lang)}</td>
                                <td>{d.page_count}</td>
                                <td>{format!("%{:.0}", d.avg_confidence * 100.0)}</td>
                                <td>{d.owner.clone()}</td>
                                <td>
                                    {is_admin.then(|| {
                                        let id_del = id.clone();
                                        let token_del = token_row.clone();
                                        view! {
                                            <button class="btn danger" on:click=move |_| {
                                                let id = id_del.clone();
                                                let token = token_del.clone();
                                                spawn_local(async move {
                                                    if api::delete_document(&token, &id).await.is_ok() {
                                                        reload_documents(token.clone(), documents, loading, error);
                                                    }
                                                });
                                            }>"Sil"</button>
                                        }
                                    })}
                                </td>
                            </tr>
                        }
                    }).collect::<Vec<_>>();
                    view! {
                        <table>
                            <thead>
                                <tr>
                                    <th>"Dosya"</th><th>"Durum"</th><th>"Gizlilik"</th>
                                    <th>"Dil"</th><th>"Sayfa"</th><th>"Güven"</th><th>"Sahip"</th><th></th>
                                </tr>
                            </thead>
                            <tbody>{rows}</tbody>
                        </table>
                    }.into_any()
                }
            }}
        </div>
    }
}
