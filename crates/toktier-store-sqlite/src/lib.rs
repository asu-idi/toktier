//! SQLite persistence for the toktier session store.
//!
//! This is an internal supporting crate of TokTier, versioned with the
//! workspace and carrying no independent API stability promise; use the
//! `toktier` package for the supported Rust surface.
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

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::path::Path;
#[cfg(unix)]
use std::path::PathBuf;

use rusqlite::{Connection, OptionalExtension};
use toktier_store_core::{
    KeyId, SemanticFingerprint, SessionEncoder, SessionHandle, SessionStore, StoreConfig,
    StoreError, FORMAT_NAME,
};

/// Errors of the SQLite tier.
#[derive(Debug)]
pub enum DbError {
    /// Filesystem operation outside SQLite failed.
    Io(std::io::Error),
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
            DbError::Io(e) => write!(f, "io error: {e}"),
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

impl From<std::io::Error> for DbError {
    fn from(e: std::io::Error) -> DbError {
        DbError::Io(e)
    }
}

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

/// One stable name restored with the plaintext needed to resume incremental
/// TKFR/content tracking after a process restart.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecoveredNamedSession {
    pub name: String,
    pub handle: SessionHandle,
    pub transcript: String,
}

/// Borrowed durable-session input used by an atomic save. The transcript is
/// passed directly to SQLite instead of being cloned first.
#[derive(Debug, Clone, Copy)]
pub struct NamedSessionRef<'a> {
    pub name: &'a str,
    pub handle: SessionHandle,
    pub transcript: &'a str,
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
CREATE TABLE IF NOT EXISTS named_sessions (
    name TEXT PRIMARY KEY,
    session_id INTEGER NOT NULL CHECK (session_id >= 0),
    FOREIGN KEY(session_id) REFERENCES sessions(id));
CREATE TABLE IF NOT EXISTS named_session_recovery (
    name TEXT PRIMARY KEY,
    transcript BLOB NOT NULL,
    binding BLOB NOT NULL,
    FOREIGN KEY(name) REFERENCES named_sessions(name));
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

#[cfg(unix)]
fn sqlite_sidecar(path: &Path, suffix: &str) -> PathBuf {
    let mut value = path.as_os_str().to_os_string();
    value.push(suffix);
    PathBuf::from(value)
}

#[cfg(unix)]
fn prepare_private_database(path: &Path) -> Result<Vec<PathBuf>, DbError> {
    use std::fs::{self, OpenOptions};
    use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};

    let mut sidecars = Vec::new();
    for suffix in ["-wal", "-shm"] {
        let sidecar = sqlite_sidecar(path, suffix);
        if !sidecar.try_exists()? {
            sidecars.push(sidecar);
        }
    }

    // create_new gives this layer exact set-on-create semantics: files a
    // caller supplied retain their mode, while new databases never pass
    // through an umask-derived visibility window.
    let mut options = OpenOptions::new();
    options.read(true).write(true).create_new(true).mode(0o600);
    match options.open(path) {
        Ok(file) => file.set_permissions(fs::Permissions::from_mode(0o600))?,
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(error) => return Err(error.into()),
    }
    Ok(sidecars)
}

#[cfg(unix)]
fn set_materialized_sidecars_private(sidecars: &[PathBuf]) -> Result<(), DbError> {
    use std::fs;
    use std::os::unix::fs::PermissionsExt;

    for sidecar in sidecars {
        if sidecar.try_exists()? {
            fs::set_permissions(sidecar, fs::Permissions::from_mode(0o600))?;
        }
    }
    Ok(())
}

impl StoreDb {
    /// Open (creating if needed) a store database and take exclusive
    /// ownership of the file for the lifetime of this handle.
    pub fn open(path: &Path) -> Result<StoreDb, DbError> {
        #[cfg(unix)]
        let new_sidecars = prepare_private_database(path)?;
        let conn = Connection::open(path)?;
        // WAL for crash safety, EXCLUSIVE so the file has one owner.
        // Both pragmas answer with a row; read it rather than assume.
        let _mode: String = conn.query_row("PRAGMA journal_mode=WAL", [], |r| r.get(0))?;
        let _lock: String = conn.query_row("PRAGMA locking_mode=EXCLUSIVE", [], |r| r.get(0))?;
        conn.execute_batch(SCHEMA_SQL)?;
        #[cfg(unix)]
        {
            // SQLite owns WAL/SHM creation and may remove or recreate them.
            // Tighten instances materialized during open; the Rust session
            // path's 0700 store home is the durable lifecycle guarantee.
            set_materialized_sidecars_private(&new_sidecars)?;
        }
        Ok(StoreDb { conn })
    }

    /// Persist the full store state (configuration, fingerprints, sealed
    /// nodes, sessions) in one transaction, replacing prior content.
    pub fn save(&mut self, store: &SessionStore) -> Result<(), DbError> {
        self.save_named(store, &[])
    }

    /// Persist the full store and its public stable-name mapping in the same
    /// transaction. Session handles remain process-local; names are rebound
    /// to fresh handles by [`Self::load_named_recoverable`].
    pub fn save_named(
        &mut self,
        store: &SessionStore,
        names: &[(String, SessionHandle)],
    ) -> Result<(), DbError> {
        self.save_named_inner(store, names, &HashMap::new())
    }

    /// Persist stable names together with the historical plaintext needed to
    /// recover a sealed session's incremental TKFR/content digest state.
    /// The binding is generated from the store's incremental digest state;
    /// transcript length is checked without rescanning historical bytes. Full
    /// digest/tail/checkpoint verification occurs before restart state is
    /// admitted by [`Self::load_named_recoverable`].
    pub fn save_named_recoverable(
        &mut self,
        store: &SessionStore,
        sessions: &[NamedSessionRef<'_>],
    ) -> Result<(), DbError> {
        let names = sessions
            .iter()
            .map(|session| (session.name.to_owned(), session.handle))
            .collect::<Vec<_>>();
        let transcripts = sessions
            .iter()
            .map(|session| (session.name, session.transcript))
            .collect::<HashMap<_, _>>();
        self.save_named_inner(store, &names, &transcripts)
    }

    fn save_named_inner(
        &mut self,
        store: &SessionStore,
        names: &[(String, SessionHandle)],
        transcripts: &HashMap<&str, &str>,
    ) -> Result<(), DbError> {
        let cfg = store.config().clone();
        let tx = self.conn.transaction()?;
        tx.execute("DELETE FROM named_session_recovery", [])?;
        tx.execute("DELETE FROM named_sessions", [])?;
        tx.execute("DELETE FROM store_meta", [])?;
        tx.execute("DELETE FROM fingerprints", [])?;
        tx.execute("DELETE FROM nodes", [])?;
        tx.execute("DELETE FROM sessions", [])?;
        let meta: [(&str, String); 8] = [
            ("format", FORMAT_NAME.to_string()),
            ("block_chars", cfg.block_chars.to_string()),
            ("tail_soft_cap_bytes", cfg.tail_soft_cap_bytes.to_string()),
            ("tail_hard_cap_bytes", cfg.tail_hard_cap_bytes.to_string()),
            ("node_tail_cap_bytes", cfg.node_tail_cap_bytes.to_string()),
            ("max_sessions", cfg.max_sessions.to_string()),
            (
                "recovery_tracking",
                u8::from(store.recovery_tracking_enabled()).to_string(),
            ),
            (
                "content_tracking",
                u8::from(store.content_tracking_enabled()).to_string(),
            ),
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
        let handles = store.list_handles();
        let known_handles = handles.iter().copied().collect::<HashSet<_>>();
        for handle in handles {
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
        for (name, handle) in names {
            if name.is_empty() || name.len() > 1024 {
                return Err(DbError::Schema(
                    "session names must contain 1..=1024 bytes".to_owned(),
                ));
            }
            if !known_handles.contains(handle) {
                return Err(DbError::Schema(format!(
                    "named session {name:?} references unknown handle {}",
                    handle.0
                )));
            }
            tx.execute(
                "INSERT INTO named_sessions VALUES (?1, ?2)",
                (
                    name,
                    u64_to_sqlite_i64("named_sessions.session_id", handle.0)?,
                ),
            )?;
            if let Some(transcript) = transcripts.get(name.as_str()) {
                let binding = store.export_recovery_binding(*handle)?.ok_or_else(|| {
                    DbError::Schema(format!(
                        "named session {name:?} has no complete recovery binding"
                    ))
                })?;
                let expected_bytes = store
                    .content_index_entry(*handle)?
                    .ok_or_else(|| {
                        DbError::Schema(format!(
                            "named session {name:?} has no complete content binding"
                        ))
                    })?
                    .byte_length;
                if u64::try_from(transcript.len()).ok() != Some(expected_bytes) {
                    return Err(DbError::Schema(format!(
                        "named session {name:?} transcript length does not match its binding"
                    )));
                }
                tx.execute(
                    "INSERT INTO named_session_recovery VALUES (?1, ?2, ?3)",
                    (name, transcript.as_bytes(), binding.as_slice()),
                )?;
            }
        }
        if transcripts.len() != names.len() && !transcripts.is_empty() {
            return Err(DbError::Schema(
                "recoverable named-session persistence requires one transcript per name".to_owned(),
            ));
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
        let optional_bool = |key: &str| -> Result<bool, DbError> {
            let value: Option<String> = self
                .conn
                .query_row("SELECT v FROM store_meta WHERE k = ?1", [key], |row| {
                    row.get(0)
                })
                .optional()?;
            match value.as_deref() {
                None | Some("0") => Ok(false),
                Some("1") => Ok(true),
                Some(other) => Err(DbError::Schema(format!(
                    "store_meta.v for {key} value {other:?} is not 0 or 1"
                ))),
            }
        };
        if optional_bool("recovery_tracking")? {
            store.enable_recovery_tracking()?;
        }
        if optional_bool("content_tracking")? {
            store.enable_content_tracking()?;
        }

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

    /// Load stable names and restore their incremental recovery/content state
    /// from transcript + TKFR bytes captured in the same SQLite transaction.
    /// Missing, malformed, or mismatched recovery material is a loud failure.
    pub fn load_named_recoverable(
        &self,
        engines: &dyn EngineResolver,
    ) -> Result<(SessionStore, Vec<RecoveredNamedSession>), DbError> {
        let (mut store, handles) = self.load(engines)?;
        let mut sessions = Vec::new();
        let mut statement = self.conn.prepare(
            "SELECT n.name, n.session_id, r.transcript, r.binding
             FROM named_sessions AS n
             LEFT JOIN named_session_recovery AS r ON r.name = n.name
             ORDER BY n.name",
        )?;
        let rows = statement.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, Option<Vec<u8>>>(2)?,
                row.get::<_, Option<Vec<u8>>>(3)?,
            ))
        })?;
        for row in rows {
            let (name, saved_id, transcript, binding) = row?;
            let handle = handles
                .iter()
                .find_map(|(candidate, handle)| (*candidate == saved_id).then_some(*handle))
                .ok_or_else(|| {
                    DbError::Schema(format!(
                        "named session {name:?} references missing session {saved_id}"
                    ))
                })?;
            let transcript = transcript.ok_or_else(|| {
                DbError::Schema(format!(
                    "named session {name:?} has no restart-recovery transcript"
                ))
            })?;
            let binding = binding.ok_or_else(|| {
                DbError::Schema(format!(
                    "named session {name:?} has no restart-recovery binding"
                ))
            })?;
            let transcript = String::from_utf8(transcript).map_err(|_| {
                DbError::Schema(format!(
                    "named session {name:?} recovery transcript is not UTF-8"
                ))
            })?;
            store.restore_tracking_with_binding(handle, &transcript, &binding)?;
            sessions.push(RecoveredNamedSession {
                name,
                handle,
                transcript,
            });
        }
        Ok((store, sessions))
    }
}
