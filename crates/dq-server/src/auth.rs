//! Kimlik dogrulama: JWT tabanli oturum + Argon2 parola hash'leme.

use argon2::password_hash::{PasswordHash, PasswordHasher, PasswordVerifier, SaltString};
use argon2::Argon2;
use dq_core::{Classification, DqError, Result, UserContext};
use jsonwebtoken::{decode, encode, DecodingKey, EncodingKey, Header, Validation};
use rand_core::OsRng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
struct Claims {
    sub: String,
    clearance: i64,
    roles: Vec<String>,
    /// Unix zaman damgasi (saniye); `jsonwebtoken` bunu otomatik dogrular.
    exp: usize,
    iat: usize,
}

pub struct AuthService {
    encoding: EncodingKey,
    decoding: DecodingKey,
    ttl_secs: u64,
    enabled: bool,
    anonymous_clearance: Classification,
}

impl AuthService {
    pub fn new(
        secret: &str,
        ttl_secs: u64,
        enabled: bool,
        anonymous_clearance: Classification,
    ) -> Self {
        Self {
            encoding: EncodingKey::from_secret(secret.as_bytes()),
            decoding: DecodingKey::from_secret(secret.as_bytes()),
            ttl_secs,
            enabled,
            anonymous_clearance,
        }
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    pub fn hash_password(password: &str) -> Result<String> {
        let salt = SaltString::generate(&mut OsRng);
        Argon2::default()
            .hash_password(password.as_bytes(), &salt)
            .map(|h| h.to_string())
            .map_err(|e| DqError::Internal(format!("parola hash'lenemedi: {e}")))
    }

    pub fn verify_password(password: &str, hash: &str) -> bool {
        let Ok(parsed) = PasswordHash::new(hash) else {
            return false;
        };
        Argon2::default()
            .verify_password(password.as_bytes(), &parsed)
            .is_ok()
    }

    pub fn issue_token(&self, user: &UserContext) -> Result<String> {
        let now = chrono::Utc::now().timestamp() as usize;
        let claims = Claims {
            sub: user.username.clone(),
            clearance: user.clearance as i64,
            roles: user.roles.clone(),
            iat: now,
            exp: now + self.ttl_secs as usize,
        };
        encode(&Header::default(), &claims, &self.encoding)
            .map_err(|e| DqError::Internal(format!("token uretilemedi: {e}")))
    }

    /// Auth kapaliysa varsayilan (anonim) kullaniciyi dondurur; acikken
    /// gecerli bir JWT gerektirir.
    pub fn verify_token_or_anonymous(&self, token: Option<&str>) -> Result<UserContext> {
        if !self.enabled {
            return Ok(UserContext {
                username: "anonim".into(),
                clearance: self.anonymous_clearance,
                roles: vec!["user".into()],
            });
        }
        let token =
            token.ok_or_else(|| DqError::Unauthorized("Oturum belirteci gerekli".into()))?;
        let data =
            decode::<Claims>(token, &self.decoding, &Validation::default()).map_err(|e| {
                DqError::Unauthorized(format!("Gecersiz veya suresi dolmus belirteç: {e}"))
            })?;
        Ok(UserContext {
            username: data.claims.sub,
            clearance: Classification::from_i64(data.claims.clearance),
            roles: data.claims.roles,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn password_hash_roundtrips() {
        let hash = AuthService::hash_password("guclu-bir-parola").unwrap();
        assert!(AuthService::verify_password("guclu-bir-parola", &hash));
        assert!(!AuthService::verify_password("yanlis", &hash));
    }

    #[test]
    fn token_roundtrips_and_rejects_tampering() {
        let auth = AuthService::new(
            "0".repeat(32).as_str(),
            3600,
            true,
            Classification::Restricted,
        );
        let user = UserContext {
            username: "analist".into(),
            clearance: Classification::Secret,
            roles: vec!["user".into()],
        };
        let token = auth.issue_token(&user).unwrap();
        let verified = auth.verify_token_or_anonymous(Some(&token)).unwrap();
        assert_eq!(verified.username, "analist");
        assert_eq!(verified.clearance, Classification::Secret);

        let mut tampered = token.clone();
        tampered.push('x');
        assert!(auth.verify_token_or_anonymous(Some(&tampered)).is_err());
    }

    #[test]
    fn disabled_auth_returns_anonymous() {
        let auth = AuthService::new(
            "0".repeat(32).as_str(),
            3600,
            false,
            Classification::Restricted,
        );
        let user = auth.verify_token_or_anonymous(None).unwrap();
        assert_eq!(user.clearance, Classification::Restricted);
    }
}
