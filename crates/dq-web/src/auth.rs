//! Kalici oturum durumu (localStorage). Sunucu JWT'yi dogrular; burada
//! yalnizca arayuzu suslemek icin kullanici bilgisi tutulur.

use serde::{Deserialize, Serialize};

use crate::api::LoginResponse;

const STORAGE_KEY: &str = "dq_auth_v1";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AuthState {
    pub token: String,
    pub username: String,
    pub clearance: String,
    pub roles: Vec<String>,
}

impl AuthState {
    pub fn is_admin(&self) -> bool {
        self.roles.iter().any(|r| r == "admin")
    }
}

impl From<LoginResponse> for AuthState {
    fn from(r: LoginResponse) -> Self {
        Self {
            token: r.token,
            username: r.username,
            clearance: r.clearance,
            roles: r.roles,
        }
    }
}

fn storage() -> Option<web_sys::Storage> {
    web_sys::window()?.local_storage().ok()?
}

pub fn load() -> Option<AuthState> {
    let raw = storage()?.get_item(STORAGE_KEY).ok()??;
    serde_json::from_str(&raw).ok()
}

pub fn save(state: &AuthState) {
    if let (Some(s), Ok(raw)) = (storage(), serde_json::to_string(state)) {
        let _ = s.set_item(STORAGE_KEY, &raw);
    }
}

pub fn clear() {
    if let Some(s) = storage() {
        let _ = s.remove_item(STORAGE_KEY);
    }
}
