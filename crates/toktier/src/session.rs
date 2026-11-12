use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use toktier_routing_core::{NativeRouter, ReferenceEngine};
use toktier_store_core::{ContentDigest, KeyId, SessionHandle, SessionStore, StoreConfig};
#[cfg(feature = "sqlite")]
use toktier_store_sqlite::{NamedSessionRef, RecoveredNamedSession, SingleEngine, StoreDb};

use crate::manifest::{LocalArtifact, Registry};
use crate::{
    Backend, Certification, Encoding, Error, ErrorCode, ExecutionFacts, Policy, ReasonCode, Result,
    RoutePlan, TokenBuffer, TokenPatch,
};

#[derive(Debug)]
pub(crate) struct TokenizerInner {
    pub(crate) artifact: LocalArtifact,
    pub(crate) reference: Arc<ReferenceEngine>,
    pub(crate) router: Arc<NativeRouter>,
    pub(crate) plan: RoutePlan,
    /// The policy the plan was made under; the chunked path reads it
    /// for its evidence gate.
    pub(crate) policy: Policy,
    pub(crate) registry: Arc<Registry>,
    /// The chunk-parallel admission and counters, built on first use.
    pub(crate) chunked: crate::chunked::State,
    pub(crate) store: std::sync::Mutex<SessionStoreState>,
    pub(crate) gpu_detail: Option<(String, i32)>,
    /// What a local verification of this route would be filed under,
    /// when the route is one a campaign has not judged.
    pub(crate) gpu_verification_key: Option<crate::verify_local::VerificationKey>,
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

    /// Session slots the core still has free, for a durable store; `None`
    /// in memory mode, where the core's own eviction applies.
    ///
    /// The core makes room for a new session by dropping its least
    /// recently used one. A durable store must not let that happen: the
    /// dropped session's name and transcript would still be in this
    /// owner's tables, and the next atomic save would refuse them. So a
    /// durable store asks first and declines the call that needs the slot.
    fn durable_free_slots(&self) -> Option<usize> {
        self.transcripts.as_ref()?;
        Some(
            self.store
                .config()
                .max_sessions
                .saturating_sub(self.store.list_handles().len()),
        )
    }

    /// Whether a content lookup may open its temporary session without
    /// making the core drop a named one.
    pub(crate) fn lookup_has_a_slot(&self) -> bool {
        self.durable_free_slots().is_none_or(|free| free >= 1)
    }

    /// Decline a new named session when a durable store is at its bound.
    /// One slot stays free for `Tokenizer::lookup`, whose hit opens a
    /// temporary session, so the bound on named sessions is one below the
    /// core's capacity. Nothing has been changed when this returns an
    /// error, so the store stays usable.
    fn ensure_room_for_a_named_session(&self) -> Result<()> {
        match self.durable_free_slots() {
            Some(free) if free <= 1 => Err(Error::new(
                ErrorCode::InvalidArgument,
                format!(
                    "the durable session store holds {} named sessions, its capacity; delete a session before creating another",
                    self.names.len()
                ),
            )),
            _ => Ok(()),
        }
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
    // The store home is created the way every other private directory in
    // this crate is: one component at a time, each one owner-only.
    crate::fsutil::create_private_dir_all(path)
}

#[cfg(test)]
thread_local! {
    /// A smaller session capacity for the stores this test thread creates,
    /// so the bound can be reached in a few calls. Test builds only.
    static TEST_MAX_SESSIONS: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

fn store_config() -> StoreConfig {
    #[cfg(test)]
    if let Some(max_sessions) = TEST_MAX_SESSIONS.with(std::cell::Cell::get) {
        return StoreConfig {
            max_sessions,
            ..StoreConfig::default()
        };
    }
    StoreConfig::default()
}

fn new_store(persistent: bool, seed_digest_overlap: bool) -> Result<SessionStore> {
    let mut store = SessionStore::new(store_config())?;
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
            state.ensure_room_for_a_named_session()?;
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
            reason: reason_for_repair_path(&patch.path),
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
            // Replacing a stored session gives its own slot back first, so
            // only a name with nothing stored needs a free one.
            let replaces_a_stored_session = self
                .handle
                .is_some_and(|old| state.store.revision(old).is_ok());
            if !replaces_a_stored_session {
                state.ensure_room_for_a_named_session()?;
            }
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
            state.ensure_room_for_a_named_session()?;
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
        reason: reason_for_repair_path(path),
    }
}

/// Why a session repair ran where it did.
///
/// The repair engine does not carry a ledger code the way the request
/// router does; it names its outcome in the path itself, and each name
/// has exactly one meaning. `engine_guard` is the `R_INPUT_GUARD_ROUTED`
/// stage of that name in `docs/contracts/routing.md` Section 5.2, and
/// both full re-encodes that follow an unusable cut are that section's
/// `R_SESSION_NO_SAFE_CUT`. An accelerated or repaired path took the
/// admitted route and has nothing to record.
fn reason_for_repair_path(path: &str) -> Option<ReasonCode> {
    match path {
        "hf_full_engine_guard" => Some(ReasonCode::InputGuardRouted),
        "hf_full_no_safe_cut" | "hf_full_window_covers_all" => Some(ReasonCode::SessionNoSafeCut),
        // The stored tail did not describe itself consistently, so the
        // accumulated text was re-encoded from the reference. Registered
        // in the frozen namespace as `R_INVALID_PRIOR_STATE` in 0.2.7;
        // it used to travel as an `Other` string that appeared in
        // neither face's vocabulary.
        "hf_full_invalid_prior_state" => Some(ReasonCode::InvalidPriorState),
        _ => None,
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
    use std::fs;
    use std::os::unix::fs::PermissionsExt;
    use std::path::PathBuf;
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

    /// The key releases 0.2.4 to 0.3.1 of this crate wrote: no serving
    /// generation beyond `v1` and no source component.
    fn released_key(artifact_sha256: &[u8], fast_cpu: bool) -> [u8; 32] {
        use sha2::{Digest, Sha256};
        let mut digest = Sha256::new();
        digest.update(b"toktier.rust.fingerprint.v1\0");
        for component in [
            artifact_sha256,
            b"tokenizers" as &[u8],
            b"0.22.2",
            b"toktier-rust-serving-v1",
            if fast_cpu { b"fast_cpu" } else { b"hf" },
            if fast_cpu {
                b"toktier-fast-repair-v1" as &[u8]
            } else {
                b""
            },
        ] {
            digest.update((component.len() as u32).to_le_bytes());
            digest.update(component);
        }
        digest.finalize().into()
    }

    #[test]
    fn a_store_written_under_an_earlier_release_key_is_refused() {
        use crate::runtime::{semantic_fingerprint, semantic_fingerprint_for};
        let artifact = [b'a'; 64];
        for fast_cpu in [true, false] {
            let current = semantic_fingerprint(&artifact, fast_cpu);
            assert_ne!(current, released_key(&artifact, fast_cpu));
            assert_ne!(
                semantic_fingerprint_for(&artifact, fast_cpu, "0".repeat(64).as_str()),
                semantic_fingerprint_for(&artifact, fast_cpu, "1".repeat(64).as_str()),
                "the key must follow the build's source identity"
            );
            let temporary = tempfile::tempdir().unwrap();
            let old = temporary.path().join("sessions/released.sqlite3");
            drop(
                SessionStoreState::new(
                    reference_router(),
                    released_key(&artifact, fast_cpu),
                    0,
                    Some(&old),
                    false,
                )
                .unwrap(),
            );
            let refused = SessionStoreState::new(reference_router(), current, 0, Some(&old), false)
                .unwrap_err();
            assert_eq!(refused.code(), ErrorCode::SessionStateMismatch);
            assert!(
                refused
                    .message()
                    .contains("does not contain this tokenizer fingerprint"),
                "{}",
                refused.message()
            );
            let own = temporary.path().join("sessions/own.sqlite3");
            drop(
                SessionStoreState::new(reference_router(), current, 0, Some(&own), false).unwrap(),
            );
            SessionStoreState::new(reference_router(), current, 0, Some(&own), false).unwrap();
        }
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

    /// The repair engine names its outcome in the path, so this crate's
    /// reading of it has to stay tied to the paths the engine emits: the
    /// two full re-encodes that follow an unusable cut, the guard route,
    /// the one outcome with no frozen code, and the repaired paths that
    /// have nothing to explain.
    #[test]
    fn a_repair_path_reads_as_the_reason_the_engine_meant() {
        assert_eq!(
            reason_for_repair_path("hf_full_engine_guard"),
            Some(ReasonCode::InputGuardRouted)
        );
        for path in ["hf_full_no_safe_cut", "hf_full_window_covers_all"] {
            assert_eq!(
                reason_for_repair_path(path),
                Some(ReasonCode::SessionNoSafeCut),
                "for {path}"
            );
        }
        assert_eq!(
            reason_for_repair_path("hf_full_invalid_prior_state"),
            Some(ReasonCode::InvalidPriorState)
        );
        for path in [
            "gigatoken_repair",
            "gigatoken_repair_noop",
            "accelerated_with_reconstructed_spans",
            "accelerated_with_lazy_span_checkpoints",
        ] {
            assert_eq!(reason_for_repair_path(path), None, "for {path}");
        }
    }

    // ---- named sessions at the store's bound -------------------------------
    //
    // These tests load a shipped artifact (from `TOKTIER_TEST_ARTIFACTS` or
    // the default cache, as the lifecycle tests do) and give the stores they
    // create a session capacity of four, so the bound is reached in a few
    // calls. A durable store then holds at most three named sessions.

    const SMALL_CAPACITY: usize = 4;

    fn artifact_root() -> Option<PathBuf> {
        let root = std::env::var_os("TOKTIER_TEST_ARTIFACTS")
            .map(PathBuf::from)
            .or_else(|| {
                std::env::var_os("HOME")
                    .map(PathBuf::from)
                    .map(|home| home.join(".cache/toktier/artifacts"))
            })
            .filter(|root| root.join("qwen3_8b-b968826d9c46/tokenizer.json").is_file());
        if root.is_none() {
            eprintln!("skip: no qwen3_8b artifact under TOKTIER_TEST_ARTIFACTS or the cache");
        }
        root
    }

    /// Sets the test-only capacity for this thread until dropped.
    struct SmallCapacity;

    impl SmallCapacity {
        fn set() -> Self {
            TEST_MAX_SESSIONS.with(|bound| bound.set(Some(SMALL_CAPACITY)));
            SmallCapacity
        }
    }

    impl Drop for SmallCapacity {
        fn drop(&mut self) {
            TEST_MAX_SESSIONS.with(|bound| bound.set(None));
        }
    }

    fn qwen(root: &Path, home: Option<&Path>) -> (crate::Runtime, crate::Tokenizer) {
        let mut builder = crate::Runtime::builder()
            .artifact_cache(root)
            .device(crate::Device::Cpu);
        if let Some(home) = home {
            builder = builder.home(home);
        }
        let runtime = builder.build().unwrap();
        let tokenizer = runtime.load("qwen3_8b").unwrap();
        assert_eq!(
            tokenizer
                .inner
                .store
                .lock()
                .unwrap()
                .store
                .config()
                .max_sessions,
            SMALL_CAPACITY
        );
        (runtime, tokenizer)
    }

    /// A distinct ASCII text of `len` bytes. A lookup hit needs a sealed
    /// block, so the texts it is asked about are longer than 4,096.
    fn text(index: usize, len: usize) -> String {
        const WORDS: [&str; 8] = [
            "river", "stone", "maple", "cloud", "lantern", "harbor", "meadow", "copper",
        ];
        let mut state = index as u64 ^ 0x9E37_79B9_7F4A_7C15;
        let mut out = format!("Document {index}. ");
        while out.len() < len {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            out.push_str(WORDS[(state >> 33) as usize % WORDS.len()]);
            out.push(if state >> 60 == 0 { '\n' } else { ' ' });
        }
        out.truncate(len);
        out
    }

    fn seed_new(tokenizer: &crate::Tokenizer, name: &str, text: &str) -> Result<Vec<u32>> {
        let mut session = tokenizer.open_session(name)?;
        let ids = session.seed(text)?.ids().to_vec();
        session.close()?;
        Ok(ids)
    }

    /// Seed each (name, text) in order and stop at the first refusal, which
    /// is returned with the number seeded before it.
    fn seed_until_declined(
        tokenizer: &crate::Tokenizer,
        texts: &[(String, String)],
    ) -> (usize, Option<Error>) {
        for (seeded, (name, text)) in texts.iter().enumerate() {
            match seed_new(tokenizer, name, text) {
                Ok(ids) => assert_eq!(ids, tokenizer.encode(text).unwrap().ids(), "{name}"),
                Err(error) => return (seeded, Some(error)),
            }
        }
        (texts.len(), None)
    }

    fn assert_declined(error: &Error, held: usize) {
        assert_eq!(error.code(), ErrorCode::InvalidArgument, "{error}");
        assert_eq!(
            error.message(),
            format!(
                "the durable session store holds {held} named sessions, its capacity; delete a session before creating another"
            )
        );
    }

    fn assert_healthy(tokenizer: &crate::Tokenizer, sessions: u64) {
        tokenizer
            .inner
            .store
            .lock()
            .unwrap()
            .ensure_healthy()
            .unwrap();
        let stats = tokenizer.store_stats();
        assert_eq!(stats.session_count, sessions);
        assert_eq!(stats.sessions_evicted, 0);
    }

    fn snapshot_of(tokenizer: &crate::Tokenizer, name: &str) -> Option<Vec<u32>> {
        let session = tokenizer.open_session(name).unwrap();
        session
            .revision()
            .map(|_| session.snapshot().unwrap().ids().to_vec())
    }

    #[test]
    fn a_full_durable_store_declines_new_names_and_keeps_working() {
        let Some(root) = artifact_root() else {
            return;
        };
        let _capacity = SmallCapacity::set();
        let temporary = tempfile::tempdir().unwrap();
        let home = temporary.path().join("home");
        let named = |index: usize| (format!("s{index}"), text(index, 80));
        let expected = {
            let (_runtime, tokenizer) = qwen(&root, Some(&home));
            let fill: Vec<_> = (0..=SMALL_CAPACITY).map(named).collect();
            let (seeded, declined) = seed_until_declined(&tokenizer, &fill);
            assert_eq!(seeded, SMALL_CAPACITY - 1, "first refusal: {declined:?}");
            assert_declined(&declined.unwrap(), SMALL_CAPACITY - 1);
            assert_healthy(&tokenizer, 3);

            // A stored session keeps working.
            let mut s0 = tokenizer.open_session("s0").unwrap();
            s0.append(" and one more turn").unwrap();
            let grown = format!("{} and one more turn", named(0).1);
            assert_eq!(
                s0.snapshot().unwrap().ids(),
                tokenizer.encode(&grown).unwrap().ids()
            );

            // The other two calls that create a name are declined as well.
            assert_declined(&s0.fork("f0").unwrap_err(), 3);
            let mut unseeded = tokenizer.open_session("n0").unwrap();
            assert_declined(&unseeded.overwrite(&text(40, 80)).unwrap_err(), 3);
            drop(unseeded);
            s0.close().unwrap();

            // Replacing a stored name gives its slot back first.
            let mut s1 = tokenizer.open_session("s1").unwrap();
            let replaced = text(41, 80);
            assert_eq!(
                s1.overwrite(&replaced).unwrap().ids(),
                tokenizer.encode(&replaced).unwrap().ids()
            );
            s1.close().unwrap();
            assert_healthy(&tokenizer, 3);

            // `delete` makes room, and the declined session value can seed.
            let mut s3 = tokenizer.open_session("s3").unwrap();
            assert_declined(&s3.seed(&named(3).1).unwrap_err(), 3);
            tokenizer.open_session("s2").unwrap().delete().unwrap();
            assert_eq!(
                s3.seed(&named(3).1).unwrap().ids(),
                tokenizer.encode(&named(3).1).unwrap().ids()
            );
            s3.close().unwrap();
            assert_declined(&seed_new(&tokenizer, "s4", &named(4).1).unwrap_err(), 3);
            assert_healthy(&tokenizer, 3);

            [("s0", grown), ("s1", replaced), ("s3", named(3).1)].map(|(name, text)| {
                let ids = snapshot_of(&tokenizer, name).unwrap();
                assert_eq!(ids, tokenizer.encode(&text).unwrap().ids(), "{name}");
                (name, ids)
            })
        };

        // A fresh runtime over the same home reads the same names and ids.
        let (_runtime, tokenizer) = qwen(&root, Some(&home));
        assert_healthy(&tokenizer, 3);
        for (name, ids) in expected {
            assert_eq!(snapshot_of(&tokenizer, name), Some(ids), "{name}");
        }
        for name in ["s2", "s4", "f0", "n0"] {
            assert_eq!(snapshot_of(&tokenizer, name), None, "{name}");
        }
    }

    #[test]
    fn a_lookup_hit_at_the_bound_evicts_no_named_session() {
        let Some(root) = artifact_root() else {
            return;
        };
        let _capacity = SmallCapacity::set();
        let temporary = tempfile::tempdir().unwrap();
        let (_runtime, tokenizer) = qwen(&root, Some(&temporary.path().join("home")));
        // As many names as the core has slots: the last one is declined.
        let fill: Vec<_> = (0..SMALL_CAPACITY)
            .map(|index| (format!("s{index}"), text(index, 6000)))
            .collect();
        let (seeded, _) = seed_until_declined(&tokenizer, &fill);
        assert!(seeded >= SMALL_CAPACITY - 1);

        let query = format!("{} and a question that follows", fill[0].1);
        let hit = tokenizer.lookup(&query).unwrap().expect("a lookup hit");
        assert_eq!(hit.ids(), tokenizer.encode(&query).unwrap().ids());
        assert_healthy(&tokenizer, 3);

        let mut s1 = tokenizer.open_session("s1").unwrap();
        s1.append(" one more turn").unwrap();
        assert_eq!(
            s1.snapshot().unwrap().ids(),
            tokenizer
                .encode(&format!("{} one more turn", fill[1].1))
                .unwrap()
                .ids()
        );
        assert_healthy(&tokenizer, 3);
    }

    #[test]
    fn a_file_with_every_slot_taken_opens_and_frees_slots_on_delete() {
        let Some(root) = artifact_root() else {
            return;
        };
        let _capacity = SmallCapacity::set();
        let temporary = tempfile::tempdir().unwrap();
        let home = temporary.path().join("home");
        let texts: Vec<String> = (0..4).map(|index| text(index, 6000)).collect();
        {
            // Write a file in which named sessions take every slot, as a
            // build without the named-session bound could: three through
            // the public calls, the fourth straight into the tables, then
            // one atomic save.
            let (_runtime, tokenizer) = qwen(&root, Some(&home));
            for (index, text) in texts.iter().take(3).enumerate() {
                seed_new(&tokenizer, &format!("s{index}"), text).unwrap();
            }
            let mut state = tokenizer.inner.store.lock().unwrap();
            let key = state.key;
            let put = state
                .store
                .put(key, &texts[3], tokenizer.inner.router.as_ref())
                .unwrap();
            state.names.insert("s3".to_owned(), put.handle);
            state
                .transcripts
                .as_mut()
                .unwrap()
                .insert("s3".to_owned(), texts[3].clone());
            state.persist().unwrap();
        }

        let (runtime, tokenizer) = qwen(&root, Some(&home));
        assert_healthy(&tokenizer, 4);
        // A hit would need a fifth slot, so the lookup answers a miss.
        let query = format!("{} and a question that follows", texts[0]);
        assert!(tokenizer.lookup(&query).unwrap().is_none());
        assert_healthy(&tokenizer, 4);
        assert_declined(&seed_new(&tokenizer, "n0", &texts[0]).unwrap_err(), 4);
        let s0 = tokenizer.open_session("s0").unwrap();
        assert_declined(&s0.fork("f0").unwrap_err(), 4);
        drop(s0);
        let mut s1 = tokenizer.open_session("s1").unwrap();
        s1.append(" one more turn").unwrap();
        s1.close().unwrap();
        assert_healthy(&tokenizer, 4);

        // One deletion frees the lookup's slot; the next one makes room
        // for a new name.
        tokenizer.open_session("s3").unwrap().delete().unwrap();
        let hit = tokenizer.lookup(&query).unwrap().expect("a lookup hit");
        assert_eq!(hit.ids(), tokenizer.encode(&query).unwrap().ids());
        assert_healthy(&tokenizer, 3);
        assert_declined(&seed_new(&tokenizer, "n0", &texts[3]).unwrap_err(), 3);
        tokenizer.open_session("s2").unwrap().delete().unwrap();
        assert_eq!(
            seed_new(&tokenizer, "n0", &texts[3]).unwrap(),
            tokenizer.encode(&texts[3]).unwrap().ids()
        );
        assert_healthy(&tokenizer, 3);
        drop((runtime, tokenizer));

        let (_runtime, tokenizer) = qwen(&root, Some(&home));
        assert_healthy(&tokenizer, 3);
        assert_eq!(
            snapshot_of(&tokenizer, "n0").unwrap(),
            tokenizer.encode(&texts[3]).unwrap().ids()
        );
        assert_eq!(
            snapshot_of(&tokenizer, "s1").unwrap(),
            tokenizer
                .encode(&format!("{} one more turn", texts[1]))
                .unwrap()
                .ids()
        );
    }
}
