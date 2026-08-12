use uuid::Uuid;

/// Yeni rastgele kimlik.
pub fn new_id() -> Uuid {
    Uuid::new_v4()
}

/// Icerik hash'i (blake3, hex). Ayni dosyanin tekrar islenmesini engellemek
/// ve onbellek anahtari uretmek icin kullanilir.
pub fn content_hash(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex().to_string()
}

/// Birden fazla parcadan deterministik anahtar uretir (onbellek anahtarlari icin).
pub fn key_of(parts: &[&str]) -> String {
    let mut hasher = blake3::Hasher::new();
    for p in parts {
        hasher.update(p.as_bytes());
        hasher.update(&[0x1f]);
    }
    hasher.finalize().to_hex().to_string()
}

/// Denetim kaydi zinciri icin SHA-256 tabanli hash (uzun omurlu, standart).
pub fn audit_hash(prev_hash: &str, payload: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(prev_hash.as_bytes());
    h.update(b"|");
    h.update(payload.as_bytes());
    format!("{:x}", h.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_is_stable() {
        assert_eq!(content_hash(b"abc"), content_hash(b"abc"));
        assert_ne!(content_hash(b"abc"), content_hash(b"abd"));
    }

    #[test]
    fn key_is_unambiguous() {
        assert_ne!(key_of(&["ab", "c"]), key_of(&["a", "bc"]));
    }
}
