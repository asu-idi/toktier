use std::collections::{HashMap, HashSet};
#[cfg(all(feature = "sqlite", unix))]
use std::fs;
use std::path::Path;
#[cfg(all(feature = "sqlite", unix))]
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use toktier_routing_core::{NativeRouter, ReferenceEngine};
use toktier_store_core::{ContentDigest, KeyId, SessionHandle, SessionStore, StoreConfig};
#[cfg(feature = "sqlite")]
use toktier_store_sqlite::{NamedSessionRef, RecoveredNamedSession, SingleEngine, StoreDb};

use crate::manifest::LocalArtifact;
use crate::{
    Backend, Certification, Encoding, Error, ErrorCode, ExecutionFacts, Result, RoutePlan,
    TokenBuffer, TokenPatch,
};

#[derive(Debug)]
pub(crate) struct TokenizerInner {
    pub(crate) artifact: LocalArtifact,
    pub(crate) reference: Arc<ReferenceEngine>,
    pub(crate) router: Arc<NativeRouter>,
    pub(crate) plan: RoutePlan,
    pub(crate) store: std::sync::Mutex<SessionStoreState>,
    pub(crate) gpu_detail: Option<(String, i32)>,
    #[cfg(feature = "jit")]
    pub(crate) jit_detail: Option<crate::JitArtifact>,
}

pub(crate) struct SessionStoreState {
    pub(crate) store: SessionStore,
    pub(crate) key: KeyId,
    names: HashMap<String, SessionHandle>,
    leased: HashSet<String>,
    /// Full plaintext is retained only for durable named sessions. Store-v1
    /// intentionally omits sealed-prefix text, so restart-safe delta appends
    /// need this caller-owned recovery input to revalidate TKFR-v1.
    transcripts: Option<HashMap<String, String>>,
    /// A durable write failed after the in-memory mutation. The process may
    /// continue serving stateless requests, but this session store is sealed
    /// until restart reloads the last atomic SQLite commit.
    persistence_fault: Option<String>,
    #[cfg(feature = "sqlite")]
    database: Option<StoreDb>,
}

impl std::fmt::Debug for SessionStoreState {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SessionStoreState")
            .field("names", &self.names.len())
            .field("leased", &self.leased.len())
            .finish_non_exhaustive()
    }
}

impl SessionStoreState {
    pub(crate) fn new(
        router: Arc<NativeRouter>,
        fingerprint: [u8; 32],
        seal_guard: u64,
        database_path: Option<&Path>,
        seed_digest_overlap: bool,
    ) -> Result<Self> {
        #[cfg(not(feature = "sqlite"))]
        let _ = &router;
        #[cfg(feature = "sqlite")]
        if let Some(path) = database_path {
            if let Some(parent) = path.parent() {
                ensure_store_home(parent)?;
            }
            let populated = path.metadata().is_ok_and(|metadata| metadata.len() > 0);
            let database = StoreDb::open(path)?;
            if populated {
                let (mut store, recovered) =
                    database.load_named_recoverable(&SingleEngine(router.as_ref()))?;
                set_seed_overlap(&mut store, seed_digest_overlap);
                let key = store
                    .export_fingerprints()
                    .into_iter()
                    .find_map(|(id, candidate, _)| (candidate == fingerprint).then_some(KeyId(id)))
                    .ok_or_else(|| {
                        Error::new(
                            ErrorCode::SessionStateMismatch,
                            "persistent store does not contain this tokenizer fingerprint",
                        )
                    })?;
                let names = recovered
                    .iter()
                    .map(|session| (session.name.clone(), session.handle))
                    .collect();
                let transcripts = recovered
                    .into_iter()
                    .map(
                        |RecoveredNamedSession {
                             name, transcript, ..
                         }| (name, transcript),
                    )
                    .collect();
                return Ok(Self {
                    store,
                    key,
                    names,
                    leased: HashSet::new(),
                    transcripts: Some(transcripts),
                    persistence_fault: None,
                    database: Some(database),
                });
            }
            let mut store = new_store(true, seed_digest_overlap)?;
            let key = store.register_fingerprint(fingerprint, seal_guard)?;
            let mut state = Self {
                store,
                key,
                names: HashMap::new(),
                leased: HashSet::new(),
                transcripts: Some(HashMap::new()),
                persistence_fault: None,
                database: Some(database),
            };
            state.persist()?;
            return Ok(state);
        }
        #[cfg(not(feature = "sqlite"))]
        if database_path.is_some() {
            return Err(Error::new(
                ErrorCode::ConfigInvalid,
                "persistent sessions require the sqlite feature",
            ));
        }

        let mut store = new_store(false, seed_digest_overlap)?;
        let key = store.register_fingerprint(fingerprint, seal_guard)?;
        Ok(Self {
            store,
            key,
            names: HashMap::new(),
            leased: HashSet::new(),
            transcripts: None,
            persistence_fault: None,
            #[cfg(feature = "sqlite")]
            database: None,
        })
    }

    fn persist(&mut self) -> Result<()> {
        self.ensure_healthy()?;
        #[cfg(feature = "sqlite")]
        if let Some(database) = &mut self.database {
            let transcripts = self.transcripts.as_ref().ok_or_else(|| {
                Error::new(
                    ErrorCode::Internal,
                    "persistent session state lost its recovery transcripts",
                )
            })?;
            let sessions = self
                .names
                .iter()
                .map(|(name, handle)| {
                    let transcript = transcripts.get(name).ok_or_else(|| {
                        Error::new(
                            ErrorCode::SessionStateMismatch,
                            format!("persistent session {name:?} has no recovery transcript"),
                        )
                    })?;
                    Ok(NamedSessionRef {
                        name,
                        handle: *handle,
                        transcript,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            if let Err(error) = database.save_named_recoverable(&self.store, &sessions) {
                let error = Error::from(error);
                self.persistence_fault = Some(error.to_string());
                return Err(error);
            }
        }
        Ok(())
    }

    pub(crate) fn ensure_healthy(&self) -> Result<()> {
        match &self.persistence_fault {
            Some(cause) => Err(Error::new(
                ErrorCode::SessionStateMismatch,
                format!(
                    "persistent session state is sealed after a failed atomic save; restart to reload the last committed state: {cause}"
                ),
            )),
            None => Ok(()),
        }
    }
}

#[cfg(feature = "sqlite")]
fn ensure_store_home(path: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::{DirBuilderExt, PermissionsExt};

        let mut current = PathBuf::new();
        for component in path.components() {
            current.push(component.as_os_str());
            let mut builder = fs::DirBuilder::new();
            builder.mode(0o700);
            match builder.create(&current) {
                Ok(()) => fs::set_permissions(&current, fs::Permissions::from_mode(0o700))
                    .map_err(|error| {
                        Error::new(ErrorCode::Io, error.to_string()).with_path(&current)
                    })?,
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
                Err(error) => {
                    return Err(Error::new(ErrorCode::Io, error.to_string()).with_path(&current));
                }
            }
        }
        // Preserve create_dir_all's validation of a pre-existing final path
        // without changing any directory this call did not create.
        fs::create_dir_all(path)
            .map_err(|error| Error::new(ErrorCode::Io, error.to_string()).with_path(path))?;
    }
    #[cfg(not(unix))]
    std::fs::create_dir_all(path)
        .map_err(|error| Error::new(ErrorCode::Io, error.to_string()).with_path(path))?;
    Ok(())
}

fn new_store(persistent: bool, seed_digest_overlap: bool) -> Result<SessionStore> {
    let mut store = SessionStore::new(StoreConfig::default())?;
    if persistent {
        store.enable_recovery_tracking()?;
    }
    store.enable_content_tracking()?;
    set_seed_overlap(&mut store, seed_digest_overlap);
    Ok(store)
}

/// Install the bounded-pool overlap runner when the runtime opted in
/// (PLAN/162 WP5/WP6). Recovery hashing and durable serialization stay
/// serial in the durable tier; only the seed content-digest scan moves
/// next to the seed encode.
fn set_seed_overlap(store: &mut SessionStore, enabled: bool) {
    if enabled {
        store.set_seed_overlap(Some(Arc::new(toktier_routing_core::RayonSeedOverlap)));
    }
}

/// Compact observability snapshot for one named session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionStats {
    pub name: String,
    pub revision: Option<u64>,
    pub token_count: u64,
    pub closed: bool,
}

/// One non-cloneable, single-writer agent session.
///
/// Mutation requires one exclusive Rust borrow, so duplicate in-process
/// writers are rejected by the type system before the runtime revision gate:
///
/// ```compile_fail
/// use toktier::Session;
/// fn invalid(session: &mut Session) {
///     let first = &mut *session;
///     let second = &mut *session;
///     let _ = first.append("a");
///     let _ = second.append("b");
/// }
/// ```
pub struct Session {
    tokenizer: Arc<TokenizerInner>,
    name: String,
    handle: Option<SessionHandle>,
    revision: Option<u64>,
    closed: bool,
}

impl std::fmt::Debug for Session {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Session")
            .field("name", &self.name)
            .field("revision", &self.revision)
            .field("closed", &self.closed)
            .finish_non_exhaustive()
    }
}

impl Session {
    pub(crate) fn open(tokenizer: Arc<TokenizerInner>, name: String) -> Result<Self> {
        validate_name(&name)?;
        let (handle, revision) = {
            let mut state = tokenizer
                .store
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            state.ensure_healthy()?;
            if !state.leased.insert(name.clone()) {
                return Err(Error::new(
                    ErrorCode::SessionRevisionConflict,
                    format!("session {name:?} already has a writer in this process"),
                ));
            }
            let handle = state.names.get(&name).copied();
            let revision = handle
                .map(|handle| state.store.revision(handle))
                .transpose()?;
            (handle, revision)
        };
        Ok(Self {
            tokenizer,
            name,
            handle,
            revision,
            closed: false,
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn revision(&self) -> Option<u64> {
        self.revision
    }

    /// Seed a newly named session and return its initial complete stream.
    pub fn seed(&mut self, text: &str) -> Result<Encoding> {
        self.seed_observed(text).map(|(encoding, _)| encoding)
    }

    pub(crate) fn seed_observed(&mut self, text: &str) -> Result<(Encoding, Duration)> {
        self.require_open()?;
        if self.handle.is_some() {
            return Err(Error::new(
                ErrorCode::SessionRevisionConflict,
                "seed requires a new session; use overwrite for an existing name",
            ));
        }
        let (handle, revision, ids, persistence) = {
            let mut state = self.lock_store();
            state.ensure_healthy()?;
            let key = state.key;
            let put = state.store.put(key, text, self.tokenizer.router.as_ref())?;
            state.names.insert(self.name.clone(), put.handle);
            if let Some(transcripts) = &mut state.transcripts {
                transcripts.insert(self.name.clone(), text.to_owned());
            }
            // The returned Encoding shares the store's adopted allocation
            // (or its one cached materialization) instead of copying the
            // complete row again.
            let ids = state.store.shared_all_ids(put.handle)?;
            let persistence_started = Instant::now();
            state.persist()?;
            (put.handle, put.revision, ids, persistence_started.elapsed())
        };
        self.handle = Some(handle);
        self.revision = Some(revision);
        Ok((
            Encoding::from_buffer(
                TokenBuffer::from_shared(ids),
                session_facts(&self.tokenizer, "session_seed", text.len()),
            ),
            persistence,
        ))
    }

    /// Append only new text and return the exact BPE suffix replacement.
    pub fn append(&mut self, delta: &str) -> Result<TokenPatch> {
        self.append_observed(delta).map(|(patch, _)| patch)
    }

    pub(crate) fn append_observed(&mut self, delta: &str) -> Result<(TokenPatch, Duration)> {
        self.require_open()?;
        let handle = self.handle.ok_or_else(|| {
            Error::new(ErrorCode::InvalidArgument, "seed the session before append")
        })?;
        let expected_revision = self
            .revision
            .ok_or_else(|| Error::new(ErrorCode::Internal, "seeded session has no revision"))?;
        let (patch, persistence) = {
            let mut state = self.lock_store();
            state.ensure_healthy()?;
            let patch = state.store.append_patch(
                handle,
                delta,
                expected_revision,
                self.tokenizer.router.as_ref(),
            )?;
            if let Some(transcripts) = &mut state.transcripts {
                let transcript = transcripts.get_mut(&self.name).ok_or_else(|| {
                    Error::new(
                        ErrorCode::SessionStateMismatch,
                        "persistent session is missing its recovery transcript",
                    )
                })?;
                transcript.push_str(delta);
            }
            let persistence_started = Instant::now();
            state.persist()?;
            (patch, persistence_started.elapsed())
        };
        self.revision = Some(patch.revision);
        let execution = ExecutionFacts {
            backend: backend_for_path(&patch.path),
            path: patch.path.clone(),
            source: Some("native_session".to_owned()),
            input_bytes: delta.len() as u64,
            certification: self.tokenizer.plan.certification,
        };
        Ok((
            TokenPatch::new(
                patch.replace_from,
                patch.replacement_ids,
                patch.revision,
                patch.token_count,
                execution,
            ),
            persistence,
        ))
    }

    /// Compatibility operation for callers that still submit the complete
    /// transcript. Prefix identity is checked with the store's native content
    /// digest before only the suffix is repaired.
    pub fn encode_transcript(&mut self, complete_text: &str) -> Result<Encoding> {
        self.require_open()?;
        let handle = self.handle.ok_or_else(|| {
            Error::new(
                ErrorCode::InvalidArgument,
                "seed the session before compatibility encode",
            )
        })?;
        let prefix_len = {
            let state = self.lock_store();
            state.ensure_healthy()?;
            let expected = state.store.content_index_entry(handle)?.ok_or_else(|| {
                Error::new(
                    ErrorCode::SessionStateMismatch,
                    "session has no recoverable content binding; seed or reopen with history",
                )
            })?;
            let prefix_len = usize::try_from(expected.byte_length).map_err(|_| {
                Error::new(
                    ErrorCode::InvalidArgument,
                    "session text length exceeds usize",
                )
            })?;
            let prefix = complete_text.as_bytes().get(..prefix_len).ok_or_else(|| {
                Error::new(
                    ErrorCode::SessionStateMismatch,
                    "complete transcript is shorter than the stored session",
                )
            })?;
            std::str::from_utf8(prefix).map_err(|_| {
                Error::new(
                    ErrorCode::SessionStateMismatch,
                    "stored byte length is not a UTF-8 boundary in the candidate transcript",
                )
            })?;
            let observed = ContentDigest::from_bytes(prefix)?.entry();
            if observed != expected {
                return Err(Error::new(
                    ErrorCode::SessionStateMismatch,
                    "complete transcript does not extend the stored session exactly",
                ));
            }
            prefix_len
        };
        let delta = &complete_text[prefix_len..];
        self.append(delta)?;
        self.snapshot()
    }

    /// Materialize the complete token stream explicitly.
    pub fn snapshot(&self) -> Result<Encoding> {
        self.require_open()?;
        let handle = self.handle.ok_or_else(|| {
            Error::new(
                ErrorCode::InvalidArgument,
                "seed the session before snapshot",
            )
        })?;
        let mut state = self.lock_store();
        state.ensure_healthy()?;
        // Explicit materialization point: the complete row is served from
        // the generation-keyed shared snapshot, so repeated snapshots of
        // an unchanged session return the same immutable allocation.
        let ids = state.store.shared_all_ids(handle)?;
        Ok(Encoding::from_buffer(
            TokenBuffer::from_shared(ids),
            session_facts(&self.tokenizer, "session_snapshot", 0),
        ))
    }

    /// Replace an existing named session with a new genesis stream.
    pub fn overwrite(&mut self, text: &str) -> Result<Encoding> {
        self.require_open()?;
        let (handle, revision, ids) = {
            let mut state = self.lock_store();
            state.ensure_healthy()?;
            if let Some(old) = self.handle {
                state.store.evict(old);
            }
            let key = state.key;
            let put = state.store.put(key, text, self.tokenizer.router.as_ref())?;
            state.names.insert(self.name.clone(), put.handle);
            if let Some(transcripts) = &mut state.transcripts {
                transcripts.insert(self.name.clone(), text.to_owned());
            }
            let ids = state.store.shared_all_ids(put.handle)?;
            state.persist()?;
            (put.handle, put.revision, ids)
        };
        self.handle = Some(handle);
        self.revision = Some(revision);
        Ok(Encoding::from_buffer(
            TokenBuffer::from_shared(ids),
            session_facts(&self.tokenizer, "session_overwrite", text.len()),
        ))
    }

    /// Fork this session under a new stable name. Sealed content-addressed
    /// nodes are shared; the new revision lineage starts at zero.
    pub fn fork(&self, new_name: impl Into<String>) -> Result<Session> {
        self.require_open()?;
        let new_name = new_name.into();
        validate_name(&new_name)?;
        let handle = self.handle.ok_or_else(|| {
            Error::new(ErrorCode::InvalidArgument, "seed the session before fork")
        })?;
        let forked = {
            let mut state = self.lock_store();
            state.ensure_healthy()?;
            if state.names.contains_key(&new_name) || state.leased.contains(&new_name) {
                return Err(Error::new(
                    ErrorCode::SessionRevisionConflict,
                    format!("session {new_name:?} already exists"),
                ));
            }
            let forked = state.store.fork(handle)?;
            state.names.insert(new_name.clone(), forked);
            let forked_transcript = state
                .transcripts
                .as_ref()
                .and_then(|transcripts| transcripts.get(&self.name).cloned());
            if let Some(transcripts) = &mut state.transcripts {
                let transcript = forked_transcript.ok_or_else(|| {
                    Error::new(
                        ErrorCode::SessionStateMismatch,
                        "persistent source session has no recovery transcript",
                    )
                })?;
                transcripts.insert(new_name.clone(), transcript);
            }
            state.leased.insert(new_name.clone());
            state.persist()?;
            forked
        };
        Ok(Session {
            tokenizer: Arc::clone(&self.tokenizer),
            name: new_name,
            handle: Some(forked),
            revision: Some(0),
            closed: false,
        })
    }

    /// Delete this session's persistent/in-memory state.
    pub fn delete(mut self) -> Result<()> {
        self.require_open()?;
        {
            let mut state = self.lock_store();
            state.ensure_healthy()?;
            if let Some(handle) = self.handle {
                state.store.evict(handle);
            }
            state.names.remove(&self.name);
            if let Some(transcripts) = &mut state.transcripts {
                transcripts.remove(&self.name);
            }
            state.leased.remove(&self.name);
            state.persist()?;
        }
        self.closed = true;
        Ok(())
    }

    /// Release the single-writer lease while retaining reopenable state.
    pub fn close(mut self) -> Result<()> {
        self.release_lease();
        self.closed = true;
        Ok(())
    }

    pub fn stats(&self) -> Result<SessionStats> {
        let token_count = match self.handle {
            Some(handle) if !self.closed => {
                let state = self.lock_store();
                state.ensure_healthy()?;
                state.store.session_info(handle)?.token_count
            }
            _ => 0,
        };
        Ok(SessionStats {
            name: self.name.clone(),
            revision: self.revision,
            token_count,
            closed: self.closed,
        })
    }

    fn lock_store(&self) -> std::sync::MutexGuard<'_, SessionStoreState> {
        self.tokenizer
            .store
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn require_open(&self) -> Result<()> {
        if self.closed {
            Err(Error::new(
                ErrorCode::InvalidArgument,
                format!("session {:?} is closed", self.name),
            ))
        } else {
            Ok(())
        }
    }

    fn release_lease(&self) {
        self.lock_store().leased.remove(&self.name);
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        if !self.closed {
            self.release_lease();
            self.closed = true;
        }
    }
}

fn validate_name(name: &str) -> Result<()> {
    if name.is_empty() || name.len() > 1024 || name.contains('\0') {
        Err(Error::new(
            ErrorCode::InvalidArgument,
            "session name must contain 1..=1024 non-NUL UTF-8 bytes",
        ))
    } else {
        Ok(())
    }
}

fn session_facts(tokenizer: &TokenizerInner, path: &str, input_bytes: usize) -> ExecutionFacts {
    ExecutionFacts {
        backend: if tokenizer.plan.backends.contains(&Backend::FastCpu) {
            Backend::FastCpu
        } else {
            Backend::HuggingFace
        },
        path: path.to_owned(),
        source: Some("native_session".to_owned()),
        input_bytes: input_bytes as u64,
        certification: tokenizer.plan.certification,
    }
}

fn backend_for_path(path: &str) -> Backend {
    if path.contains("gpu") {
        Backend::Gpu
    } else if path.contains("hf") || path.contains("fallback") {
        Backend::HuggingFace
    } else {
        Backend::FastCpu
    }
}

#[allow(dead_code)]
fn _assert_session_send() {
    fn assert_send<T: Send>() {}
    assert_send::<Session>();
    let _ = Certification::Reference;
}

#[cfg(all(test, feature = "sqlite", unix))]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;
    use toktier_routing_core::BACKEND_REFERENCE;

    const TOKENIZER_JSON: &[u8] = br#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":null,"model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":false,"vocab":{"a":0},"merges":[]}}"#;

    fn reference_router() -> Arc<NativeRouter> {
        let reference = Arc::new(ReferenceEngine::from_bytes(TOKENIZER_JSON).unwrap());
        Arc::new(
            NativeRouter::new(
                vec![BACKEND_REFERENCE.to_owned()],
                vec![0],
                reference,
                None,
                false,
                None,
                false,
                true,
            )
            .unwrap(),
        )
    }

    fn permission_bits(path: &Path) -> u32 {
        fs::metadata(path).unwrap().permissions().mode() & 0o777
    }

    #[test]
    fn fresh_rust_store_home_and_database_are_owner_only() {
        let temporary = tempfile::tempdir().unwrap();
        let existing = temporary.path().join("caller-owned");
        fs::create_dir(&existing).unwrap();
        fs::set_permissions(&existing, fs::Permissions::from_mode(0o750)).unwrap();
        ensure_store_home(&existing).unwrap();
        assert_eq!(permission_bits(&existing), 0o750);

        let store_home = temporary.path().join("runtime-home/sessions");
        let database_path = store_home.join("test.sqlite3");
        let _state =
            SessionStoreState::new(reference_router(), [7; 32], 0, Some(&database_path), false)
                .unwrap();

        assert_eq!(permission_bits(&store_home), 0o700);
        assert_eq!(permission_bits(&database_path), 0o600);
        for suffix in ["-wal", "-shm"] {
            let mut sidecar = database_path.as_os_str().to_os_string();
            sidecar.push(suffix);
            let sidecar = PathBuf::from(sidecar);
            if sidecar.exists() {
                assert_eq!(permission_bits(&sidecar), 0o600);
            }
        }
    }
}
