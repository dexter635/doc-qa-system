use leptos::prelude::*;

use crate::ask::AskView;
use crate::audit::AuditView;
use crate::auth::AuthState;
use crate::documents::DocumentsView;
use crate::login::LoginView;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tab {
    Documents,
    Ask,
    Audit,
}

#[component]
pub fn App() -> impl IntoView {
    let auth = RwSignal::new(crate::auth::load());
    let tab = RwSignal::new(Tab::Ask);

    let logout = move |_| {
        crate::auth::clear();
        auth.set(None);
    };

    view! {
        <div class="shell">
            {move || match auth.get() {
                None => view! { <LoginView on_login=move |state: AuthState| auth.set(Some(state)) /> }.into_any(),
                Some(state) => {
                    let is_admin = state.is_admin();
                    view! {
                        <header class="topbar">
                            <div class="brand"><span class="dot"></span>"Belge Analiz Sistemi"</div>
                            <nav class="tabs">
                                <button class:active=move || tab.get() == Tab::Documents
                                    on:click=move |_| tab.set(Tab::Documents)>"Belgeler"</button>
                                <button class:active=move || tab.get() == Tab::Ask
                                    on:click=move |_| tab.set(Tab::Ask)>"Soru-Cevap"</button>
                                {is_admin.then(|| view! {
                                    <button class:active=move || tab.get() == Tab::Audit
                                        on:click=move |_| tab.set(Tab::Audit)>"Denetim"</button>
                                })}
                            </nav>
                            <div class="user-box">
                                <span><strong>{state.username.clone()}</strong>" · " {state.clearance.clone()}</span>
                                <button class="btn secondary" on:click=logout>"Çıkış"</button>
                            </div>
                        </header>
                        <main class="content">
                            {move || match tab.get() {
                                Tab::Documents => view! { <DocumentsView auth=state.clone() /> }.into_any(),
                                Tab::Ask => view! { <AskView auth=state.clone() /> }.into_any(),
                                Tab::Audit => view! { <AuditView auth=state.clone() /> }.into_any(),
                            }}
                        </main>
                    }.into_any()
                }
            }}
        </div>
    }
}
