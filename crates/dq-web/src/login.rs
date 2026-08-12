use leptos::prelude::*;
use leptos::task::spawn_local;

use crate::api;
use crate::auth::AuthState;
use crate::dom::input_value;

#[component]
pub fn LoginView(on_login: impl Fn(AuthState) + 'static + Copy) -> impl IntoView {
    let username = RwSignal::new(String::new());
    let password = RwSignal::new(String::new());
    let error = RwSignal::new(Option::<String>::None);
    let busy = RwSignal::new(false);

    let submit = move |ev: web_sys::SubmitEvent| {
        ev.prevent_default();
        if busy.get_untracked() {
            return;
        }
        let u = username.get_untracked();
        let p = password.get_untracked();
        if u.trim().is_empty() || p.is_empty() {
            error.set(Some("Kullanıcı adı ve parola gerekli.".into()));
            return;
        }
        error.set(None);
        busy.set(true);
        spawn_local(async move {
            match api::login(&u, &p).await {
                Ok(resp) => {
                    let state: AuthState = resp.into();
                    crate::auth::save(&state);
                    on_login(state);
                }
                Err(e) => error.set(Some(e.to_string())),
            }
            busy.set(false);
        });
    };

    view! {
        <div class="login-wrap">
            <div class="card login-card">
                <h2>"Belge Analiz ve Soru-Cevap Sistemi"</h2>
                <p class="mono">"Yerel dağıtım — savunma sanayii kullanımı"</p>
                <form on:submit=submit>
                    <div class="row">
                        <input type="text" placeholder="Kullanıcı adı"
                            prop:value=move || username.get()
                            on:input=move |ev| username.set(input_value(&ev)) />
                        <input type="password" placeholder="Parola"
                            prop:value=move || password.get()
                            on:input=move |ev| password.set(input_value(&ev)) />
                        <button type="submit" class="btn" disabled=move || busy.get()>
                            {move || if busy.get() { "Giriş yapılıyor…" } else { "Giriş yap" }}
                        </button>
                    </div>
                </form>
                {move || error.get().map(|e| view! { <div class="msg error">{e}</div> })}
            </div>
        </div>
    }
}
