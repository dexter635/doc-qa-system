use std::collections::HashSet;

use leptos::prelude::*;
use leptos::task::spawn_local;

use crate::api::{self, Answer, Document};
use crate::auth::AuthState;
use crate::dom::textarea_value;
use crate::ui;

#[component]
pub fn AskView(auth: AuthState) -> impl IntoView {
    let documents = RwSignal::new(Vec::<Document>::new());
    let selected_docs = RwSignal::new(HashSet::<String>::new());
    let query = RwSignal::new(String::new());
    let answer = RwSignal::new(Option::<Answer>::None);
    let busy = RwSignal::new(false);
    let error = RwSignal::new(Option::<String>::None);

    let token_for_docs = auth.token.clone();
    spawn_local(async move {
        if let Ok(docs) = api::list_documents(&token_for_docs).await {
            documents.set(docs.into_iter().filter(|d| d.status == "ready").collect());
        }
    });

    let toggle_doc = move |id: String| {
        selected_docs.update(|set| {
            if !set.remove(&id) {
                set.insert(id);
            }
        });
    };

    let token_for_ask = auth.token.clone();
    let submit = move |ev: web_sys::SubmitEvent| {
        ev.prevent_default();
        let q = query.get_untracked();
        if q.trim().is_empty() || busy.get_untracked() {
            return;
        }
        let token = token_for_ask.clone();
        let doc_ids: Vec<String> = selected_docs.get_untracked().into_iter().collect();
        busy.set(true);
        error.set(None);
        answer.set(None);
        spawn_local(async move {
            match api::ask(&token, &q, &doc_ids).await {
                Ok(a) => answer.set(Some(a)),
                Err(e) => error.set(Some(e.to_string())),
            }
            busy.set(false);
        });
    };

    view! {
        <div class="card">
            <h2>"Soru Sor"</h2>
            {move || {
                let docs = documents.get();
                (!docs.is_empty()).then(|| view! {
                    <div>
                        <h3>"Belge kapsamı (boş bırakılırsa tümü)"</h3>
                        <div class="doc-picker">
                            {docs.into_iter().map(|d| {
                                let id = d.id.clone();
                                let id_check = id.clone();
                                view! {
                                    <label>
                                        <input type="checkbox"
                                            on:change=move |_| toggle_doc(id_check.clone()) />
                                        {d.filename.clone()}
                                    </label>
                                }
                            }).collect::<Vec<_>>()}
                        </div>
                    </div>
                })
            }}
            <form on:submit=submit>
                <textarea placeholder="Örn: Motorun periyodik bakım aralığı nedir?"
                    prop:value=move || query.get()
                    on:input=move |ev| query.set(textarea_value(&ev))></textarea>
                <div class="row" style="margin-top: 0.6rem;">
                    <button type="submit" class="btn" disabled=move || busy.get()>
                        {move || if busy.get() { "Aranıyor…" } else { "Sor" }}
                    </button>
                    {move || busy.get().then(|| view! { <span class="spinner"></span> })}
                </div>
            </form>
            {move || error.get().map(|e| view! { <div class="msg error">{e}</div> })}
        </div>

        {move || answer.get().map(|a| view! { <AnswerCard answer=a /> })}
    }
}

#[component]
fn AnswerCard(answer: Answer) -> impl IntoView {
    let support_pct = (answer.groundedness.support_ratio * 100.0).round() as i32;
    view! {
        <div class="card">
            <div class="row" style="justify-content: space-between;">
                <span class=ui::kind_class(&answer.kind)>{ui::kind_label(&answer.kind)}</span>
                <span class=ui::classification_class(&answer.classification)>
                    {ui::classification_label(&answer.classification)}
                </span>
            </div>
            <p class="answer-text">{answer.text.clone()}</p>
            <div class="answer-meta">
                <span>"Süre: " {answer.latency_ms} "ms"</span>
                <span>{if answer.cached { "Önbellekten" } else { "Yeni üretildi" }}</span>
                <span>"Dil: " {ui::lang_label(&answer.lang)}</span>
                <span>
                    "Kaynak doğrulama: "
                    <span class="progress"><div style=format!("width: {support_pct}%")></div></span>
                    {format!(" %{support_pct}")}
                </span>
            </div>
            {(!answer.warnings.is_empty()).then(|| view! {
                <div class="msg warn">
                    {answer.warnings.iter().map(|w| view! { <div>{w.clone()}</div> }).collect::<Vec<_>>()}
                </div>
            })}
            {(!answer.citations.is_empty()).then(|| view! {
                <div>
                    <h3>"Kaynaklar"</h3>
                    {answer.citations.iter().map(|c| view! {
                        <div class="citation">
                            <div class="src">
                                {format!("[{}] {} · s. {}-{} · skor {:.2}", c.marker, c.doc_filename, c.page_from, c.page_to, c.score)}
                            </div>
                            <div>{c.snippet.clone()}</div>
                        </div>
                    }).collect::<Vec<_>>()}
                </div>
            })}
            {(!answer.trace.is_empty()).then(|| view! {
                <details class="agent-trace">
                    <summary>{format!("Ajan adımları ({})", answer.trace.len())}</summary>
                    <ol>
                        {answer.trace.iter().map(|s| view! {
                            <li>
                                <span class=format!("badge step-{}", s.kind)>{s.kind.clone()}</span>
                                " "{s.description.clone()}
                            </li>
                        }).collect::<Vec<_>>()}
                    </ol>
                </details>
            })}
        </div>
    }
}
