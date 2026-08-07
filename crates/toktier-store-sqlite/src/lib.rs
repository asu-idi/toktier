//! SQLite persistence for the toktier session store.
//!
//! One store maps to one SQLite database file that is owned exclusively
//! by this Rust layer: the connection takes `locking_mode=EXCLUSIVE`, so
//! no other process (including any Python `sqlite3` handle) can read or
//! write the file while it is open here. All session and node payloads
//! travel as store format v1 records; this crate adds no format of its
//! own beyond the table shapes below.
//!
//! Load discipline mirrors the store's import semantics:
//! * corrupt node records are skipped silently and counted in
//!   `import_rejects` (a missing node can only ever cause a miss);
//! * corrupt session records abort the load loudly with the record's
//!   contract error code (sessions are explicit state, corruption there
//!   must not pass unnoticed);
//! * every session tail is re-encoded through the resolved engine and
//!   must reproduce the recorded ids bit-exactly before it is accepted.

#![deny(unsafe_code)]

use std::fmt;
use std::path::Path;

use rusqlite::{Connection, OptionalExtension};
use toktier_store_core::{
    KeyId, SemanticFingerprint, SessionEncoder, SessionHandle, SessionStore, StoreConfig,
    StoreError, FORMAT_NAME,
};

/// Errors of the SQLite tier.
#[derive(Debug)]
pub enum DbError {
    /// Underlying SQLite failure.
    Sqlite(rusqlite::Error),
    /// A store-format or store-semantics failure (carries the contract
    /// error code).
    Store(StoreError),
    /// The database file does not carry the expected format marker, has
    /// an inconsistent schema, or contains an unrepresentable value.
    Schema(String),
    /// No engine was resolved for a fingerprint required by a stored
    /// session.
    MissingEngine(SemanticFingerprint),
}

impl fmt::Display for DbError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DbError::Sqlite(e) => write!(f, "sqlite error: {e}"),
            DbError::Store(e) => write!(f, "{e}"),
            DbError::Schema(msg) => write!(f, "schema error: {msg}"),
            DbError::MissingEngine(fp) => {
                write!(f, "no engine resolved for fingerprint {:02x?}", &fp[..4])
            }
        }
    }
}

impl std::error::Error for DbError {}

impl From<rusqlite::Error> for DbError {
    fn from(e: rusqlite::Error) -> DbError {
        DbError::Sqlite(e)
    }
}

impl From<StoreError> for DbError {
    fn from(e: StoreError) -> DbError {
        DbError::Store(e)
    }
}

/// Maps fingerprints to the engines that own them during a load.
pub trait EngineResolver {
    fn resolve(&self, fingerprint: &SemanticFingerprint) -> Option<&dyn SessionEncoder>;
}

/// A single engine serving every fingerprint (single-tokenizer stores).
pub struct SingleEngine<'a>(pub &'a dyn SessionEncoder);

impl EngineResolver for SingleEngine<'_> {
    fn resolve(&self, _fingerprint: &SemanticFingerprint) -> Option<&dyn SessionEncoder> {
        Some(self.0)
    }
}

const SCHEMA_SQL: &str = "
CREATE TABLE IF NOT EXISTS store_meta (k TEXT PRIMARY KEY, v TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS fingerprints (
    id INTEGER PRIMARY KEY CHECK (id BETWEEN 0 AND 4294967295),
    fingerprint BLOB NOT NULL,
    seal_guard_chars INTEGER NOT NULL CHECK (seal_guard_chars >= 0));
CREATE TABLE IF NOT EXISTS nodes (node_key BLOB PRIMARY KEY, rec BLOB NOT NULL);
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY CHECK (id >= 0),
    key_id INTEGER NOT NULL CHECK (key_id BETWEEN 0 AND 4294967295),
    rec BLOB NOT NULL,
    sidecar BLOB NOT NULL);
";

fn sqlite_i64_to_u64(column: &str, value: i64) -> Result<u64, DbError> {
    u64::try_from(value).map_err(|_| {
        DbError::Schema(format!(
            "{column} value {value} cannot be represented as u64"
        ))
    })
}

fn sqlite_i64_to_u32(column: &str, value: i64) -> Result<u32, DbError> {
    u32::try_from(value).map_err(|_| {
        DbError::Schema(format!(
            "{column} value {value} cannot be represented as u32"
        ))
    })
}

fn u64_to_sqlite_i64(column: &str, value: u64) -> Result<i64, DbError> {
    i64::try_from(value).map_err(|_| {
        DbError::Schema(format!(
            "{column} value {value} exceeds sqlite integer range"
        ))
    })
}

fn u64_to_usize(column: &str, value: u64) -> Result<usize, DbError> {
    usize::try_from(value).map_err(|_| {
        DbError::Schema(format!(
            "{column} value {value} cannot be represented as usize"
        ))
    })
}

/// An exclusively-owned store database file.
pub struct StoreDb {
    conn: Connection,
}

impl StoreDb {
    /// Open (creating if needed) a store database and take exclusive
    /// ownership of the file for the lifetime of this handle.
    pub fn open(path: &Path) -> Result<StoreDb, DbError> {
        let conn = Connection::open(path)?;
        // WAL for crash safety, EXCLUSIVE so the file has one owner.
        // Both pragmas answer with a row; read it rather than assume.
        let _mode: String = conn.query_row("PRAGMA journal_mode=WAL", [], |r| r.get(0))?;
        let _lock: String = conn.query_row("PRAGMA locking_mode=EXCLUSIVE", [], |r| r.get(0))?;
        conn.execute_batch(SCHEMA_SQL)?;
        Ok(StoreDb { conn })
    }

    /// Persist the full store state (configuration, fingerprints, sealed
    /// nodes, sessions) in one transaction, replacing prior content.
    pub fn save(&mut self, store: &SessionStore) -> Result<(), DbError> {
        let cfg = store.config().clone();
        let tx = self.conn.transaction()?;
        tx.execute("DELETE FROM store_meta", [])?;
        tx.execute("DELETE FROM fingerprints", [])?;
        tx.execute("DELETE FROM nodes", [])?;
        tx.execute("DELETE FROM sessions", [])?;
        let meta: [(&str, String); 6] = [
            ("format", FORMAT_NAME.to_string()),
            ("block_chars", cfg.block_chars.to_string()),
            ("tail_soft_cap_bytes", cfg.tail_soft_cap_bytes.to_string()),
            ("tail_hard_cap_bytes", cfg.tail_hard_cap_bytes.to_string()),
            ("node_tail_cap_bytes", cfg.node_tail_cap_bytes.to_string()),
            ("max_sessions", cfg.max_sessions.to_string()),
        ];
        for (k, v) in &meta {
            tx.execute("INSERT INTO store_meta VALUES (?1, ?2)", (k, v))?;
        }
        for (id, fingerprint, guard) in store.export_fingerprints() {
            let guard = u64_to_sqlite_i64("fingerprints.seal_guard_chars", guard)?;
            tx.execute(
                "INSERT INTO fingerprints VALUES (?1, ?2, ?3)",
                (i64::from(id), fingerprint.as_slice(), guard),
            )?;
        }
        for (node_key, rec) in store.export_node_items()? {
            tx.execute(
                "INSERT INTO nodes VALUES (?1, ?2)",
                (node_key.as_slice(), rec.as_slice()),
            )?;
        }
        for handle in store.list_handles() {
            let info = store.session_info(handle)?;
            let rec = store.export_session(handle)?;
            let sidecar = store.export_session_sidecar(handle)?;
            let saved_id = u64_to_sqlite_i64("sessions.id", handle.0)?;
            tx.execute(
                "INSERT INTO sessions VALUES (?1, ?2, ?3, ?4)",
                (
                    saved_id,
                    i64::from(info.key_id.0),
                    rec.as_slice(),
                    sidecar.as_slice(),
                ),
            )?;
        }
        tx.commit()?;
        Ok(())
    }

    /// Rebuild a store from the database.
    ///
    /// Returns the store together with the mapping from saved session
    /// ids to fresh handles (handles are process-local and never
    /// persistent).
    pub fn load(
        &self,
        engines: &dyn EngineResolver,
    ) -> Result<(SessionStore, Vec<(i64, SessionHandle)>), DbError> {
        let format: Option<String> = self
            .conn
            .query_row("SELECT v FROM store_meta WHERE k = 'format'", [], |r| {
                r.get(0)
            })
            .optional()?;
        match format {
            Some(f) if f == FORMAT_NAME => {}
            other => {
                return Err(DbError::Schema(format!(
                    "unsupported store format marker: {other:?}"
                )))
            }
        }
        let meta_u64 = |k: &str| -> Result<u64, DbError> {
            let v: String =
                self.conn
                    .query_row("SELECT v FROM store_meta WHERE k = ?1", [k], |r| r.get(0))?;
            v.parse::<u64>().map_err(|_| {
                DbError::Schema(format!(
                    "store_meta.v for {k} value {v:?} is not a valid u64"
                ))
            })
        };
        let meta_usize = |k: &str| -> Result<usize, DbError> {
            let value = meta_u64(k)?;
            u64_to_usize(&format!("store_meta.v for {k}"), value)
        };
        let cfg = StoreConfig {
            block_chars: meta_u64("block_chars")?,
            tail_soft_cap_bytes: meta_usize("tail_soft_cap_bytes")?,
            tail_hard_cap_bytes: meta_usize("tail_hard_cap_bytes")?,
            node_tail_cap_bytes: meta_usize("node_tail_cap_bytes")?,
            max_sessions: meta_usize("max_sessions")?,
        };
        let mut store = SessionStore::new(cfg)?;

        // Fingerprints, in saved id order.
        let mut key_map: Vec<(u32, KeyId, SemanticFingerprint)> = Vec::new();
        {
            let mut stmt = self.conn.prepare(
                "SELECT id, fingerprint, seal_guard_chars FROM fingerprints ORDER BY id",
            )?;
            let rows = stmt.query_map([], |r| {
                Ok((
                    r.get::<_, i64>(0)?,
                    r.get::<_, Vec<u8>>(1)?,
                    r.get::<_, i64>(2)?,
                ))
            })?;
            for row in rows {
                let (id, fp_raw, guard) = row?;
                let id = sqlite_i64_to_u32("fingerprints.id", id)?;
                let fp: SemanticFingerprint = fp_raw.try_into().map_err(|_| {
                    DbError::Schema(format!("fingerprint row {id} is not 32 bytes"))
                })?;
                let guard = sqlite_i64_to_u64("fingerprints.seal_guard_chars", guard)?;
                let kid = store.register_fingerprint(fp, guard)?;
                key_map.push((id, kid, fp));
            }
        }

        // Nodes: silent per-record verification, rejects counted.
        {
            let mut stmt = self.conn.prepare("SELECT node_key, rec FROM nodes")?;
            let rows = stmt.query_map([], |r| {
                Ok((r.get::<_, Vec<u8>>(0)?, r.get::<_, Vec<u8>>(1)?))
            })?;
            for row in rows {
                let (node_key, rec) = row?;
                store.import_node_item(&node_key, &rec);
            }
        }

        // Sessions: loud on any rejection.
        let mut handle_map = Vec::new();
        {
            let mut stmt = self
                .conn
                .prepare("SELECT id, key_id, rec, sidecar FROM sessions ORDER BY id")?;
            let rows = stmt.query_map([], |r| {
                Ok((
                    r.get::<_, i64>(0)?,
                    r.get::<_, i64>(1)?,
                    r.get::<_, Vec<u8>>(2)?,
                    r.get::<_, Vec<u8>>(3)?,
                ))
            })?;
            for row in rows {
                let (sid, key_id, rec, sidecar) = row?;
                let saved_id = sqlite_i64_to_u64("sessions.id", sid)?;
                let key_id = sqlite_i64_to_u32("sessions.key_id", key_id)?;
                let Some((_, kid, fp)) = key_map.iter().find(|(id, _, _)| *id == key_id).copied()
                else {
                    return Err(DbError::Schema(format!(
                        "session {saved_id} references unknown key id {key_id}"
                    )));
                };
                let engine = engines.resolve(&fp).ok_or(DbError::MissingEngine(fp))?;
                let handle = store.import_session_with_sidecar(kid, &rec, &sidecar, engine)?;
                handle_map.push((sid, handle));
            }
        }
        Ok((store, handle_map))
    }
}
