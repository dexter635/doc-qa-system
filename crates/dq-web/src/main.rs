mod api;
mod app;
mod ask;
mod audit;
mod auth;
mod documents;
mod dom;
mod login;
mod ui;

use app::App;

fn main() {
    console_error_panic_hook::set_once();
    leptos::mount::mount_to_body(App);
}
