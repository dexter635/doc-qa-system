use leptos::prelude::*;
use leptos::task::spawn_local;

use crate::api::{self, AuditEntry};
use crate::auth::AuthState;

#[component]
pub fn AuditView(auth: AuthState) -> impl IntoView {
    let entries = RwSignal::new(Vec::<AuditEntry>::new());
    let loading = RwSignal::new(true);
    let error = RwSignal::new(Option::<String>::None);
    let chain_status = RwSignal::new(Option::<(bool, String)>::None);

    let token = auth.token.clone();
    spawn_local(async move {
        match api::audit_log(&token).await {
            Ok(list) => entries.set(list),
            Err(e) => error.set(Some(e.to_string())),
        }
        loading.set(false);
    });

    let token_verify = auth.token.clone();
    let verify = move |_| {
        let token = token_verify.clone();
        chain_status.set(None);
        spawn_local(async move {
            match api::audit_verify(&token).await {
                Ok(v) => {
                    let intact = v.get("intact").and_then(|b| b.as_bool()).unwrap_or(false);
                    let msg = if intact {
                        "Denetim kaydı zinciri sağlam; kurcalama tespit edilmedi.".to_string()
                    } else {
                        format!(
                            "UYARI: zincir bozulmuş. İlk bozulma indeksi: {:?}",
                            v.get("broken_at_index")
                        )
                    };
                    chain_status.set(Some((intact, msg)));
                }
                Err(e) => chain_status.set(Some((false, e.to_string()))),
            }
        });
    };

    view! {
        <div class="card">
            <div class="row" style="justify-content: space-between;">
                <h2>"Denetim Kaydı"</h2>
                <button class="btn secondary" on:click=verify>"Zincirin bütünlüğünü doğrula"</button>
            </div>
            {move || chain_status.get().map(|(ok, m)| {
                let cls = if ok { "msg ok" } else { "msg error" };
                view! { <div class=cls>{m}</div> }
            })}
            {move || error.get().map(|e| view! { <div class="msg error">{e}</div> })}
            {move || {
                if loading.get() {
                    view! { <p class="empty"><span class="spinner"></span>" Yükleniyor…"</p> }.into_any()
                } else if entries.get().is_empty() {
                    view! { <p class="empty">"Kayıt yok."</p> }.into_any()
                } else {
                    let rows = entries.get().into_iter().map(|e| view! {
                        <tr>
                            <td class="mono">{e.at.clone()}</td>
                            <td>{e.actor.clone()}</td>
                            <td>{e.action.clone()}</td>
                            <td class="mono">{e.subject.clone().unwrap_or_default()}</td>
                            <td>{e.outcome.clone()}</td>
                        </tr>
                    }).collect::<Vec<_>>();
                    view! {
                        <table>
                            <thead>
                                <tr><th>"Zaman"</th><th>"Kullanıcı"</th><th>"Eylem"</th><th>"Konu"</th><th>"Sonuç"</th></tr>
                            </thead>
                            <tbody>{rows}</tbody>
                        </table>
                    }.into_any()
                }
            }}
        </div>
    }
}
