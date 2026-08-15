//! SQLite tabanli kalici depolama.
//!
//! Tek dosyalik gomulu veritabani secildi: kapali ag / tek makine kurulumunda
//! harici bir servis (Postgres, Qdrant) isletmek operasyonel yuk getirir.
//! Vektorler de ayni dosyada BLOB olarak tutulur, boylece yedekleme tek
//! dosyanin kopyalanmasina indirgenir.

use std::path::Path;

use chrono::{DateTime, Utc};
use dq_core::ids::audit_hash;
use dq_core::{
    AuditEvent, Chunk, ChunkType, Classification, Document, DocumentStatus, DqError, Lang, Result,
    UserContext,
};
use parking_lot::Mutex;
use rusqlite::{params, Connection, OptionalExtension};
use uuid::Uuid;

pub struct Store {
    conn: Mutex<Connection>,
}

impl Store {
    pub fn open(path: &Path) -> Result<Self> {
        let conn = Connection::open(path).map_err(|e| DqError::Storage(e.to_string()))?;
        Self::configure(&conn)?;
        let store = Self {
            conn: Mutex::new(conn),
        };
        store.migrate()?;
        Ok(store)
    }

    pub fn open_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory().map_err(|e| DqError::Storage(e.to_string()))?;
        Self::configure(&conn)?;
        let store = Self {
            conn: Mutex::new(conn),
        };
        store.migrate()?;
        Ok(store)
    }

    fn configure(conn: &Connection) -> Result<()> {
        conn.execute_batch(
            "PRAGMA journal_mode = WAL;
             PRAGMA synchronous = NORMAL;
             PRAGMA foreign_keys = ON;
             PRAGMA busy_timeout = 5000;",
        )
        .map_err(|e| DqError::Storage(e.to_string()))
    }

    fn migrate(&self) -> Result<()> {
        let conn = self.conn.lock();
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS documents (
                id             TEXT PRIMARY KEY,
                filename       TEXT NOT NULL,
                mime           TEXT NOT NULL,
                content_hash   TEXT NOT NULL,
                size_bytes     INTEGER NOT NULL,
                page_count     INTEGER NOT NULL DEFAULT 0,
                lang           TEXT NOT NULL DEFAULT 'und',
                classification INTEGER NOT NULL DEFAULT 0,
                status         TEXT NOT NULL DEFAULT 'pending',
                owner          TEXT NOT NULL DEFAULT '',
                created_at     TEXT NOT NULL,
                updated_at     TEXT NOT NULL,
                error          TEXT,
                avg_confidence REAL NOT NULL DEFAULT 1.0
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_hash ON documents(content_hash);

            CREATE TABLE IF NOT EXISTS chunks (
                id             TEXT PRIMARY KEY,
                doc_id         TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                ordinal        INTEGER NOT NULL,
                page_from      INTEGER NOT NULL,
                page_to        INTEGER NOT NULL,
                text           TEXT NOT NULL,
                heading_path   TEXT,
                token_estimate INTEGER NOT NULL,
                lang           TEXT NOT NULL,
                classification INTEGER NOT NULL,
                confidence     REAL NOT NULL DEFAULT 1.0,
                parent_id      TEXT,
                chunk_type     TEXT NOT NULL DEFAULT 'standalone'
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id, ordinal);
            CREATE INDEX IF NOT EXISTS idx_chunks_parent ON chunks(parent_id) WHERE parent_id IS NOT NULL;

            CREATE TABLE IF NOT EXISTS embeddings (
                chunk_id TEXT PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
                dim      INTEGER NOT NULL,
                model    TEXT NOT NULL,
                vec      BLOB NOT NULL
            );

            CREATE TABLE IF NOT EXISTS qa_cache (
                key        TEXT PRIMARY KEY,
                scope      TEXT NOT NULL,
                query      TEXT NOT NULL,
                answer     TEXT NOT NULL,
                clearance  INTEGER NOT NULL,
                query_vec  BLOB,
                created_at TEXT NOT NULL,
                hits       INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_cache_scope ON qa_cache(scope);

            CREATE TABLE IF NOT EXISTS users (
                username      TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                clearance     INTEGER NOT NULL,
                roles         TEXT NOT NULL DEFAULT '[]',
                created_at    TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS audit (
                id        TEXT PRIMARY KEY,
                seq       INTEGER,
                at        TEXT NOT NULL,
                actor     TEXT NOT NULL,
                action    TEXT NOT NULL,
                subject   TEXT,
                outcome   TEXT NOT NULL,
                detail    TEXT NOT NULL,
                prev_hash TEXT NOT NULL,
                hash      TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_audit_at ON audit(at);
            
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER NOT NULL PRIMARY KEY
            );
            INSERT OR IGNORE INTO schema_version VALUES (2);
            "#,
        )
        .map_err(|e| DqError::Storage(e.to_string()))?;
        Ok(())
    }

    // ---------------- documents ----------------

    pub fn insert_document(&self, doc: &Document) -> Result<()> {
        let conn = self.conn.lock();
        conn.execute(
            "INSERT INTO documents (id, filename, mime, content_hash, size_bytes, page_count,
                lang, classification, status, owner, created_at, updated_at, error, avg_confidence)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14)",
            params![
                doc.id.to_string(),
                doc.filename,
                doc.mime,
                doc.content_hash,
                doc.size_bytes as i64,
                doc.page_count as i64,
                doc.lang.code(),
                doc.classification as i64,
                doc.status.as_str(),
                doc.owner,
                doc.created_at.to_rfc3339(),
                doc.updated_at.to_rfc3339(),
                doc.error,
                doc.avg_confidence,
            ],
        )
        .map_err(map_sqlite)?;
        Ok(())
    }

    pub fn update_document(&self, doc: &Document) -> Result<()> {
        let conn = self.conn.lock();
        conn.execute(
            "UPDATE documents SET page_count=?2, lang=?3, status=?4, updated_at=?5,
                    error=?6, avg_confidence=?7, classification=?8
             WHERE id=?1",
            params![
                doc.id.to_string(),
                doc.page_count as i64,
                doc.lang.code(),
                doc.status.as_str(),
                doc.updated_at.to_rfc3339(),
                doc.error,
                doc.avg_confidence,
                doc.classification as i64,
            ],
        )
        .map_err(map_sqlite)?;
        Ok(())
    }

    pub fn find_document_by_hash(&self, hash: &str) -> Result<Option<Document>> {
        let conn = self.conn.lock();
        conn.query_row(
            "SELECT id, filename, mime, content_hash, size_bytes, page_count, lang, classification,
                    status, owner, created_at, updated_at, error, avg_confidence
             FROM documents WHERE content_hash = ?1",
            params![hash],
            row_to_document,
        )
        .optional()
        .map_err(map_sqlite)
    }

    pub fn get_document(&self, id: Uuid) -> Result<Option<Document>> {
        let conn = self.conn.lock();
        conn.query_row(
            "SELECT id, filename, mime, content_hash, size_bytes, page_count, lang, classification,
                    status, owner, created_at, updated_at, error, avg_confidence
             FROM documents WHERE id = ?1",
            params![id.to_string()],
            row_to_document,
        )
        .optional()
        .map_err(map_sqlite)
    }

    /// Kullanicinin yetki seviyesinde gorebilecegi belgeler.
    pub fn list_documents(&self, clearance: Classification) -> Result<Vec<Document>> {
        let conn = self.conn.lock();
        let mut stmt = conn
            .prepare(
                "SELECT id, filename, mime, content_hash, size_bytes, page_count, lang, classification,
                        status, owner, created_at, updated_at, error, avg_confidence
                 FROM documents WHERE classification <= ?1 ORDER BY created_at DESC",
            )
            .map_err(map_sqlite)?;
        let rows = stmt
            .query_map(params![clearance as i64], row_to_document)
            .map_err(map_sqlite)?;
        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(map_sqlite)
    }

    pub fn delete_document(&self, id: Uuid) -> Result<bool> {
        let conn = self.conn.lock();
        let n = conn
            .execute(
                "DELETE FROM documents WHERE id = ?1",
                params![id.to_string()],
            )
            .map_err(map_sqlite)?;
        Ok(n > 0)
    }

    // ---------------- chunks & embeddings ----------------

    pub fn insert_chunks(&self, chunks: &[Chunk], vectors: &[Vec<f32>], model: &str) -> Result<()> {
        if chunks.len() != vectors.len() {
            return Err(DqError::Storage("chunk ve vektor sayisi uyusmuyor".into()));
        }
        let mut conn = self.conn.lock();
        let tx = conn.transaction().map_err(map_sqlite)?;
        {
            let mut chunk_stmt = tx
                .prepare(
                    "INSERT INTO chunks (id, doc_id, ordinal, page_from, page_to, text,
                        heading_path, token_estimate, lang, classification, confidence)
                     VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11)",
                )
                .map_err(map_sqlite)?;
            let mut vec_stmt = tx
                .prepare("INSERT INTO embeddings (chunk_id, dim, model, vec) VALUES (?1,?2,?3,?4)")
                .map_err(map_sqlite)?;

            for (c, v) in chunks.iter().zip(vectors) {
                chunk_stmt
                    .execute(params![
                        c.id.to_string(),
                        c.doc_id.to_string(),
                        c.ordinal as i64,
                        c.page_from as i64,
                        c.page_to as i64,
                        c.text,
                        c.heading_path,
                        c.token_estimate as i64,
                        c.lang.code(),
                        c.classification as i64,
                        c.confidence,
                    ])
                    .map_err(map_sqlite)?;
                vec_stmt
                    .execute(params![
                        c.id.to_string(),
                        v.len() as i64,
                        model,
                        encode_vec(v)
                    ])
                    .map_err(map_sqlite)?;
            }
        }
        tx.commit().map_err(map_sqlite)?;
        Ok(())
    }

    /// Indeks kurulumu icin tum chunk'lari ve vektorlerini yukler.
    pub fn load_all_chunks(&self) -> Result<Vec<(Chunk, Vec<f32>, String)>> {
        let conn = self.conn.lock();
        let mut stmt = conn
            .prepare(
                "SELECT c.id, c.doc_id, c.ordinal, c.page_from, c.page_to, c.text, c.heading_path,
                        c.token_estimate, c.lang, c.classification, c.confidence, c.parent_id, c.chunk_type, e.vec, d.filename
                 FROM chunks c
                 JOIN embeddings e ON e.chunk_id = c.id
                 JOIN documents  d ON d.id = c.doc_id
                 WHERE d.status = 'ready'
                 ORDER BY c.doc_id, c.ordinal",
            )
            .map_err(map_sqlite)?;
        let rows = stmt
            .query_map([], |row| {
                let chunk = Chunk {
                    id: parse_uuid(row.get::<_, String>(0)?),
                    doc_id: parse_uuid(row.get::<_, String>(1)?),
                    ordinal: row.get::<_, i64>(2)? as usize,
                    page_from: row.get::<_, i64>(3)? as usize,
                    page_to: row.get::<_, i64>(4)? as usize,
                    text: row.get(5)?,
                    heading_path: row.get(6)?,
                    token_estimate: row.get::<_, i64>(7)? as usize,
                    lang: Lang::from_code(&row.get::<_, String>(8)?),
                    classification: Classification::from_i64(row.get::<_, i64>(9)?),
                    confidence: row.get(10)?,
                    parent_id: row.get::<_, Option<String>>(11)?.map(parse_uuid),
                    chunk_type: match row.get::<_, String>(12)?.as_str() {
                        "parent" => ChunkType::Parent,
                        "child" => ChunkType::Child,
                        _ => ChunkType::Standalone,
                    },
                };
                let blob: Vec<u8> = row.get(13)?;
                let filename: String = row.get(14)?;
                Ok((chunk, decode_vec(&blob), filename))
            })
            .map_err(map_sqlite)?;
        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(map_sqlite)
    }

    /// Bir belgenin belirli sirali chunk'ini getirir (komsu genisletme icin).
    pub fn get_chunk_by_ordinal(&self, doc_id: Uuid, ordinal: usize) -> Result<Option<Chunk>> {
        let conn = self.conn.lock();
        conn.query_row(
            "SELECT id, doc_id, ordinal, page_from, page_to, text, heading_path, token_estimate,
                    lang, classification, confidence, parent_id, chunk_type
             FROM chunks WHERE doc_id = ?1 AND ordinal = ?2",
            params![doc_id.to_string(), ordinal as i64],
            |row| {
                Ok(Chunk {
                    id: parse_uuid(row.get::<_, String>(0)?),
                    doc_id: parse_uuid(row.get::<_, String>(1)?),
                    ordinal: row.get::<_, i64>(2)? as usize,
                    page_from: row.get::<_, i64>(3)? as usize,
                    page_to: row.get::<_, i64>(4)? as usize,
                    text: row.get(5)?,
                    heading_path: row.get(6)?,
                    token_estimate: row.get::<_, i64>(7)? as usize,
                    lang: Lang::from_code(&row.get::<_, String>(8)?),
                    classification: Classification::from_i64(row.get::<_, i64>(9)?),
                    confidence: row.get(10)?,
                    parent_id: row.get::<_, Option<String>>(11)?.map(parse_uuid),
                    chunk_type: match row.get::<_, String>(12)?.as_str() {
                        "parent" => ChunkType::Parent,
                        "child" => ChunkType::Child,
                        _ => ChunkType::Standalone,
                    },
                })
            },
        )
        .optional()
        .map_err(map_sqlite)
    }

    pub fn chunk_count(&self) -> Result<usize> {
        let conn = self.conn.lock();
        conn.query_row("SELECT COUNT(*) FROM chunks", [], |r| r.get::<_, i64>(0))
            .map(|n| n as usize)
            .map_err(map_sqlite)
    }

    // ---------------- kullanicilar ----------------

    pub fn upsert_user(
        &self,
        username: &str,
        password_hash: &str,
        clearance: Classification,
        roles: &[String],
    ) -> Result<()> {
        let conn = self.conn.lock();
        conn.execute(
            "INSERT INTO users (username, password_hash, clearance, roles, created_at)
             VALUES (?1,?2,?3,?4,?5)
             ON CONFLICT(username) DO UPDATE SET
                password_hash=excluded.password_hash,
                clearance=excluded.clearance,
                roles=excluded.roles",
            params![
                username,
                password_hash,
                clearance as i64,
                serde_json::to_string(roles).unwrap_or_else(|_| "[]".into()),
                Utc::now().to_rfc3339(),
            ],
        )
        .map_err(map_sqlite)?;
        Ok(())
    }

    pub fn get_user(&self, username: &str) -> Result<Option<(String, UserContext)>> {
        let conn = self.conn.lock();
        conn.query_row(
            "SELECT username, password_hash, clearance, roles FROM users WHERE username = ?1",
            params![username],
            |row| {
                let roles: String = row.get(3)?;
                Ok((
                    row.get::<_, String>(1)?,
                    UserContext {
                        username: row.get(0)?,
                        clearance: Classification::from_i64(row.get::<_, i64>(2)?),
                        roles: serde_json::from_str(&roles).unwrap_or_default(),
                    },
                ))
            },
        )
        .optional()
        .map_err(map_sqlite)
    }

    // ---------------- denetim kaydi ----------------

    /// Denetim kaydini hash zinciri ile ekler; sonradan silme/degistirme
    /// `verify_audit_chain` ile tespit edilebilir.
    pub fn append_audit(
        &self,
        actor: &str,
        action: &str,
        subject: Option<&str>,
        outcome: &str,
        detail: serde_json::Value,
    ) -> Result<AuditEvent> {
        let conn = self.conn.lock();
        let prev_hash: String = conn
            .query_row(
                "SELECT hash FROM audit ORDER BY rowid DESC LIMIT 1",
                [],
                |r| r.get(0),
            )
            .optional()
            .map_err(map_sqlite)?
            .unwrap_or_else(|| "GENESIS".to_string());

        let id = Uuid::new_v4();
        let at = Utc::now();
        let detail_str = detail.to_string();
        let payload = format!(
            "{}|{}|{}|{}|{}|{}",
            id,
            at.to_rfc3339(),
            actor,
            action,
            subject.unwrap_or(""),
            detail_str
        );
        let hash = audit_hash(&prev_hash, &payload);

        conn.execute(
            "INSERT INTO audit (id, at, actor, action, subject, outcome, detail, prev_hash, hash)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9)",
            params![
                id.to_string(),
                at.to_rfc3339(),
                actor,
                action,
                subject,
                outcome,
                detail_str,
                prev_hash,
                hash
            ],
        )
        .map_err(map_sqlite)?;

        Ok(AuditEvent {
            id,
            at,
            actor: actor.to_string(),
            action: action.to_string(),
            subject: subject.map(|s| s.to_string()),
            outcome: outcome.to_string(),
            detail,
            prev_hash,
            hash,
        })
    }

    /// Zincirin butunlugunu dogrular; bozulan ilk kaydin sirasini dondurur.
    pub fn verify_audit_chain(&self) -> Result<Option<usize>> {
        let conn = self.conn.lock();
        let mut stmt = conn
            .prepare(
                "SELECT id, at, actor, action, subject, detail, prev_hash, hash
                 FROM audit ORDER BY rowid ASC",
            )
            .map_err(map_sqlite)?;
        let rows = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, Option<String>>(4)?,
                    row.get::<_, String>(5)?,
                    row.get::<_, String>(6)?,
                    row.get::<_, String>(7)?,
                ))
            })
            .map_err(map_sqlite)?;

        let mut expected_prev = "GENESIS".to_string();
        for (i, r) in rows.enumerate() {
            let (id, at, actor, action, subject, detail, prev_hash, hash) =
                r.map_err(map_sqlite)?;
            if prev_hash != expected_prev {
                return Ok(Some(i));
            }
            let payload = format!(
                "{}|{}|{}|{}|{}|{}",
                id,
                at,
                actor,
                action,
                subject.unwrap_or_default(),
                detail
            );
            if audit_hash(&prev_hash, &payload) != hash {
                return Ok(Some(i));
            }
            expected_prev = hash;
        }
        Ok(None)
    }

    pub fn recent_audit(&self, limit: usize) -> Result<Vec<serde_json::Value>> {
        let conn = self.conn.lock();
        let mut stmt = conn
            .prepare(
                "SELECT at, actor, action, subject, outcome, detail
                 FROM audit ORDER BY rowid DESC LIMIT ?1",
            )
            .map_err(map_sqlite)?;
        let rows = stmt
            .query_map(params![limit as i64], |row| {
                Ok(serde_json::json!({
                    "at": row.get::<_, String>(0)?,
                    "actor": row.get::<_, String>(1)?,
                    "action": row.get::<_, String>(2)?,
                    "subject": row.get::<_, Option<String>>(3)?,
                    "outcome": row.get::<_, String>(4)?,
                    "detail": serde_json::from_str::<serde_json::Value>(
                        &row.get::<_, String>(5)?).unwrap_or(serde_json::Value::Null),
                }))
            })
            .map_err(map_sqlite)?;
        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(map_sqlite)
    }

    // ---------------- onbellek ----------------

    pub fn cache_get(&self, key: &str) -> Result<Option<String>> {
        let conn = self.conn.lock();
        let found: Option<String> = conn
            .query_row(
                "SELECT answer FROM qa_cache WHERE key = ?1",
                params![key],
                |r| r.get(0),
            )
            .optional()
            .map_err(map_sqlite)?;
        if found.is_some() {
            let _ = conn.execute(
                "UPDATE qa_cache SET hits = hits + 1 WHERE key = ?1",
                params![key],
            );
        }
        Ok(found)
    }

    pub fn cache_put(
        &self,
        key: &str,
        scope: &str,
        query: &str,
        answer: &str,
        clearance: Classification,
        query_vec: Option<&[f32]>,
    ) -> Result<()> {
        let conn = self.conn.lock();
        conn.execute(
            "INSERT INTO qa_cache (key, scope, query, answer, clearance, query_vec, created_at, hits)
             VALUES (?1,?2,?3,?4,?5,?6,?7,0)
             ON CONFLICT(key) DO UPDATE SET answer=excluded.answer, created_at=excluded.created_at",
            params![
                key,
                scope,
                query,
                answer,
                clearance as i64,
                query_vec.map(encode_vec),
                Utc::now().to_rfc3339(),
            ],
        )
        .map_err(map_sqlite)?;
        Ok(())
    }

    /// Anlamsal onbellek taramasi icin ayni kapsam ve yetki seviyesindeki
    /// kayitlarin sorgu vektorlerini dondurur.
    pub fn cache_candidates(
        &self,
        scope: &str,
        clearance: Classification,
        max_age_secs: u64,
    ) -> Result<Vec<(String, Vec<f32>, String)>> {
        let cutoff = Utc::now() - chrono::Duration::seconds(max_age_secs as i64);
        let conn = self.conn.lock();
        let mut stmt = conn
            .prepare(
                "SELECT key, query_vec, answer FROM qa_cache
                 WHERE scope = ?1 AND clearance <= ?2 AND created_at >= ?3 AND query_vec IS NOT NULL",
            )
            .map_err(map_sqlite)?;
        let rows = stmt
            .query_map(
                params![scope, clearance as i64, cutoff.to_rfc3339()],
                |row| {
                    let blob: Vec<u8> = row.get(1)?;
                    Ok((
                        row.get::<_, String>(0)?,
                        decode_vec(&blob),
                        row.get::<_, String>(2)?,
                    ))
                },
            )
            .map_err(map_sqlite)?;
        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(map_sqlite)
    }

    /// Belge eklendiginde/silindiginde eski cevaplar gecersizlesir.
    pub fn cache_clear(&self) -> Result<()> {
        let conn = self.conn.lock();
        conn.execute("DELETE FROM qa_cache", [])
            .map_err(map_sqlite)?;
        Ok(())
    }

    pub fn cache_stats(&self) -> Result<(usize, usize)> {
        let conn = self.conn.lock();
        conn.query_row(
            "SELECT COUNT(*), COALESCE(SUM(hits),0) FROM qa_cache",
            [],
            |r| Ok((r.get::<_, i64>(0)? as usize, r.get::<_, i64>(1)? as usize)),
        )
        .map_err(map_sqlite)
    }
}

fn row_to_document(row: &rusqlite::Row<'_>) -> rusqlite::Result<Document> {
    Ok(Document {
        id: parse_uuid(row.get::<_, String>(0)?),
        filename: row.get(1)?,
        mime: row.get(2)?,
        content_hash: row.get(3)?,
        size_bytes: row.get::<_, i64>(4)? as u64,
        page_count: row.get::<_, i64>(5)? as usize,
        lang: Lang::from_code(&row.get::<_, String>(6)?),
        classification: Classification::from_i64(row.get::<_, i64>(7)?),
        status: DocumentStatus::from_str_lossy(&row.get::<_, String>(8)?),
        owner: row.get(9)?,
        created_at: parse_time(row.get::<_, String>(10)?),
        updated_at: parse_time(row.get::<_, String>(11)?),
        error: row.get(12)?,
        avg_confidence: row.get(13)?,
    })
}

fn parse_uuid(s: String) -> Uuid {
    Uuid::parse_str(&s).unwrap_or_else(|_| Uuid::nil())
}

fn parse_time(s: String) -> DateTime<Utc> {
    DateTime::parse_from_rfc3339(&s)
        .map(|d| d.with_timezone(&Utc))
        .unwrap_or_else(|_| Utc::now())
}

/// f32 vektoru little-endian BLOB'a cevirir.
pub fn encode_vec(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

pub fn decode_vec(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn map_sqlite(e: rusqlite::Error) -> DqError {
    DqError::Storage(e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_doc() -> Document {
        Document {
            id: Uuid::new_v4(),
            filename: "test.pdf".into(),
            mime: "application/pdf".into(),
            content_hash: "hash1".into(),
            size_bytes: 100,
            page_count: 1,
            lang: Lang::Tr,
            classification: Classification::Restricted,
            status: DocumentStatus::Ready,
            owner: "analist".into(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            error: None,
            avg_confidence: 1.0,
        }
    }

    #[test]
    fn documents_roundtrip() {
        let store = Store::open_in_memory().unwrap();
        let doc = sample_doc();
        store.insert_document(&doc).unwrap();
        let got = store.get_document(doc.id).unwrap().unwrap();
        assert_eq!(got.filename, "test.pdf");
        assert_eq!(got.classification, Classification::Restricted);
    }

    #[test]
    fn clearance_filters_document_list() {
        let store = Store::open_in_memory().unwrap();
        let mut secret = sample_doc();
        secret.classification = Classification::Secret;
        secret.content_hash = "h2".into();
        store.insert_document(&secret).unwrap();
        store.insert_document(&sample_doc()).unwrap();

        assert_eq!(
            store
                .list_documents(Classification::Restricted)
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            store.list_documents(Classification::Secret).unwrap().len(),
            2
        );
    }

    #[test]
    fn vector_blob_roundtrip() {
        let v = vec![0.5f32, -0.25, 1.0];
        assert_eq!(decode_vec(&encode_vec(&v)), v);
    }

    #[test]
    fn audit_chain_detects_tampering() {
        let store = Store::open_in_memory().unwrap();
        store
            .append_audit("a", "login", None, "ok", serde_json::json!({}))
            .unwrap();
        store
            .append_audit("a", "query", Some("q1"), "ok", serde_json::json!({"n":1}))
            .unwrap();
        assert_eq!(store.verify_audit_chain().unwrap(), None);

        store
            .conn
            .lock()
            .execute("UPDATE audit SET actor='b' WHERE actor='a'", [])
            .unwrap();
        assert!(store.verify_audit_chain().unwrap().is_some());
    }
}
