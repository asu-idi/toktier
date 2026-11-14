use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use toktier_routing_core::{NativeRouter, ReferenceEngine};
use toktier_store_core::{ContentDigest, KeyId, SessionHandle, SessionStore, StoreConfig};
#[cfg(feature = "sqlite")]
use toktier_store_sqlite::{NamedSessionRef, RecoveredNamedSession, SingleEngine, StoreDb};

use crate::manifest::{LocalArtifact, Registry};
use crate::prefix_index::{KeyParams, PrefixIndex, Seal};
use crate::property_guard::GuardedEncoder;
use crate::{
    Backend, Certification, Encoding, Error, ErrorCode, ExecutionFacts, Policy, ReasonCode, Result,
    RoutePlan, TokenBuffer, TokenPatch,
};

#[derive(Debug)]
pub(crate) struct TokenizerInner {
    pub(crate) artifact: LocalArtifact,
    pub(crate) reference: Arc<ReferenceEngine>,
    pub(crate) router: Arc<NativeRouter>,
    /// On a tokenizer whose property data is guarded
    /// (`property_guard::Handling::Guard`), the reference-only router a
    /// text holding a listed code point goes to; `None` otherwise.
    pub(crate) guard_router: Option<Arc<NativeRouter>>,
    pub(crate) plan: RoutePlan,
    /// The policy the plan was made under; the chunked path reads it
    /// for its evidence gate.
    pub(crate) policy: Policy,
    pub(crate) registry: Arc<Registry>,
    /// The chunk-parallel admission and counters, built on first use.
    pub(crate) chunked: crate::chunked::State,
    pub(crate) store: std::sync::Mutex<SessionStoreState>,
    /// What a lookup computes its block keys from: the store's semantic
    /// fingerprint and block size, readable without the store lock.
    pub(crate) prefix_key: KeyParams,
    /// Whether that index keeps anything (`lookup_index_blocks > 0`).
    pub(crate) prefix_index_on: bool,
    pub(crate) gpu_detail: Option<(String, i32)>,
    /// What a local verification of this route would be filed under,
    /// when the route is one a campaign has not judged.
    pub(crate) gpu_verification_key: Option<crate::verify_local::VerificationKey>,
    #[cfg(feature = "jit")]
    pub(crate) jit_detail: Option<crate::JitArtifact>,
}

impl TokenizerInner {
    /// The router `text` goes to, and whether the property-data guard
    /// sent it there (`property_guard::router_for`).
    pub(crate) fn router_for(&self, text: &str) -> (&NativeRouter, bool) {
        crate::property_guard::router_for(&self.router, self.guard_router.as_deref(), text)
    }

    /// The engine every store operation of this tokenizer runs against.
    pub(crate) fn session_engine(&self) -> GuardedEncoder<'_> {
        GuardedEncoder::new(&self.router, self.guard_router.as_deref())
    }
}

pub(crate) struct SessionStoreState {
    pub(crate) store: SessionStore,
    pub(crate) key: KeyId,
    names: HashMap<String, SessionHandle>,
    /// The index `Tokenizer::lookup` reads: the completed blocks of the
    /// sessions this store holds.
    pub(crate) index: PrefixIndex,
    /// The core's `sessions_evicted` when the tables last followed it.
    seen_evictions: u64,
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
        guard_router: Option<Arc<NativeRouter>>,
        fingerprint: [u8; 32],
        seal_guard: u64,
        database_path: Option<&Path>,
        seed_digest_overlap: bool,
        lookup_index_blocks: usize,
    ) -> Result<Self> {
        #[cfg(not(feature = "sqlite"))]
        let _ = (&router, &guard_router);
        #[cfg(feature = "sqlite")]
        if let Some(path) = database_path {
            if let Some(parent) = path.parent() {
                ensure_store_home(parent)?;
            }
            let populated = path.metadata().is_ok_and(|metadata| metadata.len() > 0);
            let database = StoreDb::open(path)?;
            if populated {
                let engine = GuardedEncoder::new(router.as_ref(), guard_router.as_deref());
                let (mut store, recovered) =
                    database.load_named_recoverable(&SingleEngine(&engine))?;
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
                let mut index = new_index(&store, fingerprint, lookup_index_blocks);
                for session in &recovered {
                    // The seals a stored session had when each of its
                    // blocks was completed are not in the file, so these
                    // blocks offer no seal; the session's current seal
                    // serves the blocks it completes from now on.
                    let seal = seal_of(&store, session.handle)?;
                    index.insert_session(session.handle, &session.transcript, seal);
                }
                let seen_evictions = store.stats().sessions_evicted;
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
                    index,
                    seen_evictions,
                    leased: HashSet::new(),
                    transcripts: Some(transcripts),
                    persistence_fault: None,
                    database: Some(database),
                });
            }
            let mut store = new_store(true, seed_digest_overlap)?;
            let key = store.register_fingerprint(fingerprint, seal_guard)?;
            let index = new_index(&store, fingerprint, lookup_index_blocks);
            let mut state = Self {
                store,
                key,
                names: HashMap::new(),
                index,
                seen_evictions: 0,
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
        let index = new_index(&store, fingerprint, lookup_index_blocks);
        Ok(Self {
            store,
            key,
            names: HashMap::new(),
            index,
            seen_evictions: 0,
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

    /// Decline a new named session when a durable store is at its bound.
    /// The bound on named sessions is one below the core's capacity (the
    /// 0.3.2 bound, kept as it is). Nothing has been changed when this
    /// returns an error, so the store stays usable.
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

    /// Follow the core after a call that may have made it drop its least
    /// recently used session to make room (`put` and `fork`, memory mode):
    /// release what each dropped session registered in the index and, in
    /// memory mode, forget the names that point at dropped sessions.
    fn follow_core_evictions(&mut self) {
        let evicted = self.store.stats().sessions_evicted;
        if evicted == self.seen_evictions {
            return;
        }
        self.seen_evictions = evicted;
        let store = &self.store;
        for handle in self.index.tracked_handles() {
            if store.revision(handle).is_err() {
                self.index.remove_session(handle);
            }
        }
        if self.transcripts.is_none() {
            self.names
                .retain(|_, handle| store.revision(*handle).is_ok());
        }
    }

    /// Register what a successful `append_patch` of `delta` added, and
    /// release the session's track when the core's text length does not
    /// agree with what the index was fed (the index then offers nothing
    /// for that session rather than something unchecked).
    fn index_after_append(&mut self, handle: SessionHandle, delta: &str) -> Result<()> {
        let seal = seal_of(&self.store, handle)?;
        self.index.append(handle, delta, seal);
        self.index_check_length(handle);
        Ok(())
    }

    /// Release `handle`'s track unless the index was fed exactly as many
    /// bytes as the core holds for it.
    fn index_check_length(&mut self, handle: SessionHandle) {
        let Some(fed) = self.index.tracked_bytes(handle) else {
            return;
        };
        let held = self
            .store
            .session_info(handle)
            .ok()
            .and_then(|info| info.stable_prefix_bytes.checked_add(info.tail_bytes));
        if held != Some(fed) {
            self.index.remove_session(handle);
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

/// The core's configuration. `node_tail_cap_bytes: 0` makes the core
/// write no block record: it takes the core's existing "skip the record
/// and detach the chain" branch at a session's first completed block.
/// The crate finds stored prefixes through its own index of the live
/// sessions (`prefix_index`) and calls none of the core functions that
/// read records (`SessionStore::lookup`, `export_node_items`,
/// `import_node_item`), so the records would only be written; ids, seals
/// and session records are unchanged by the setting.
fn store_config() -> StoreConfig {
    #[cfg(test)]
    if let Some(max_sessions) = TEST_MAX_SESSIONS.with(std::cell::Cell::get) {
        return StoreConfig {
            max_sessions,
            node_tail_cap_bytes: 0,
            ..StoreConfig::default()
        };
    }
    StoreConfig {
        node_tail_cap_bytes: 0,
        ..StoreConfig::default()
    }
}

/// The default of `RuntimeBuilder::lookup_index_blocks`: entries of
/// 4,096-character blocks, about 1.07 billion characters of live session
/// text.
pub(crate) const DEFAULT_LOOKUP_INDEX_BLOCKS: usize = 262_144;

fn new_index(store: &SessionStore, fingerprint: [u8; 32], max_blocks: usize) -> PrefixIndex {
    PrefixIndex::new(
        KeyParams {
            fingerprint,
            block_chars: store.config().block_chars,
        },
        max_blocks,
    )
}

/// The session's current seal point, as the core reports it.
pub(crate) fn seal_of(store: &SessionStore, handle: SessionHandle) -> Result<Seal> {
    let info = store.session_info(handle)?;
    Ok(Seal {
        chars: info.safe_char,
        bytes: info.stable_prefix_bytes,
        ids: info.sealed_tokens,
    })
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
            let mut handle = state.names.get(&name).copied();
            if state.transcripts.is_none()
                && handle.is_some_and(|handle| state.store.revision(handle).is_err())
            {
                // Memory mode: the core dropped this session to make room
                // for another, so the name is a new name again.
                state.names.remove(&name);
                handle = None;
            }
            let revision = match handle
                .map(|handle| state.store.revision(handle))
                .transpose()
            {
                Ok(revision) => revision,
                Err(error) => {
                    // A failed open keeps no lease, so a later open of the
                    // same name is not refused as a second writer.
                    state.leased.remove(&name);
                    return Err(error.into());
                }
            };
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
        let guard_routed;
        let (handle, revision, ids, persistence) = {
            let mut state = self.lock_store();
            state.ensure_healthy()?;
            state.ensure_room_for_a_named_session()?;
            let key = state.key;
            let engine = self.tokenizer.session_engine();
            let put = state.store.put(key, text, &engine)?;
            guard_routed = engine.guard_routed();
            state.follow_core_evictions();
            state.names.insert(self.name.clone(), put.handle);
            let seal = seal_of(&state.store, put.handle)?;
            state.index.insert_session(put.handle, text, seal);
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
                session_facts(&self.tokenizer, "session_seed", text.len(), guard_routed),
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
        let guard_routed;
        let (patch, persistence) = {
            let mut state = self.lock_store();
            state.ensure_healthy()?;
            let engine = self.tokenizer.session_engine();
            let appended = state
                .store
                .append_patch(handle, delta, expected_revision, &engine);
            guard_routed = engine.guard_routed();
            let patch = match appended {
                Ok(patch) => patch,
                Err(error) => {
                    // A refused append changes nothing, and the track
                    // stays. Should the core have taken some of the text
                    // before failing, the lengths no longer agree and
                    // the track is released.
                    state.index_check_length(handle);
                    return Err(error.into());
                }
            };
            state.index_after_append(handle, delta)?;
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
        let mut execution = ExecutionFacts {
            backend: backend_for_path(&patch.path),
            path: patch.path.clone(),
            source: Some("native_session".to_owned()),
            input_bytes: delta.len() as u64,
            certification: self.tokenizer.plan.certification,
            reason: reason_for_repair_path(&patch.path),
        };
        if guard_routed {
            crate::runtime::mark_guard_routed(&mut execution);
        }
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
            session_facts(&self.tokenizer, "session_snapshot", 0, false),
        ))
    }

    /// Replace an existing named session with a new genesis stream.
    pub fn overwrite(&mut self, text: &str) -> Result<Encoding> {
        self.require_open()?;
        let guard_routed;
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
                state.index.remove_session(old);
            }
            let key = state.key;
            let engine = self.tokenizer.session_engine();
            let put = state.store.put(key, text, &engine)?;
            guard_routed = engine.guard_routed();
            state.follow_core_evictions();
            state.names.insert(self.name.clone(), put.handle);
            let seal = seal_of(&state.store, put.handle)?;
            state.index.insert_session(put.handle, text, seal);
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
            session_facts(
                &self.tokenizer,
                "session_overwrite",
                text.len(),
                guard_routed,
            ),
        ))
    }

    /// Fork this session under a new stable name. The fork starts with the
    /// source's text and ids, and the lookup index counts it on every block
    /// the source has; the new revision lineage starts at zero.
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
            state.index.fork(handle, forked);
            state.follow_core_evictions();
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
                state.index.remove_session(handle);
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

fn session_facts(
    tokenizer: &TokenizerInner,
    path: &str,
    input_bytes: usize,
    guard_routed: bool,
) -> ExecutionFacts {
    let mut facts = ExecutionFacts {
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
    };
    if guard_routed {
        crate::runtime::mark_guard_routed(&mut facts);
    }
    facts
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
                    None,
                    released_key(&artifact, fast_cpu),
                    0,
                    Some(&old),
                    false,
                    DEFAULT_LOOKUP_INDEX_BLOCKS,
                )
                .unwrap(),
            );
            let refused = SessionStoreState::new(
                reference_router(),
                None,
                current,
                0,
                Some(&old),
                false,
                DEFAULT_LOOKUP_INDEX_BLOCKS,
            )
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
                SessionStoreState::new(
                    reference_router(),
                    None,
                    current,
                    0,
                    Some(&own),
                    false,
                    DEFAULT_LOOKUP_INDEX_BLOCKS,
                )
                .unwrap(),
            );
            SessionStoreState::new(
                reference_router(),
                None,
                current,
                0,
                Some(&own),
                false,
                DEFAULT_LOOKUP_INDEX_BLOCKS,
            )
            .unwrap();
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
        let _state = SessionStoreState::new(
            reference_router(),
            None,
            [7; 32],
            0,
            Some(&database_path),
            false,
            DEFAULT_LOOKUP_INDEX_BLOCKS,
        )
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
        // A lookup needs no session slot: it hits, with exact ids, and
        // drops nothing.
        let query = format!("{} and a question that follows", texts[0]);
        let hit = tokenizer.lookup(&query).unwrap().expect("a lookup hit");
        assert_eq!(hit.ids(), tokenizer.encode(&query).unwrap().ids());
        assert_healthy(&tokenizer, 4);
        assert_declined(&seed_new(&tokenizer, "n0", &texts[0]).unwrap_err(), 4);
        let s0 = tokenizer.open_session("s0").unwrap();
        assert_declined(&s0.fork("f0").unwrap_err(), 4);
        drop(s0);
        let mut s1 = tokenizer.open_session("s1").unwrap();
        s1.append(" one more turn").unwrap();
        s1.close().unwrap();
        assert_healthy(&tokenizer, 4);

        // The file holds one named session more than the bound allows: one
        // deletion brings it to the bound, the next one makes room for a
        // new name. The lookup needs no slot at either point.
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

    #[test]
    fn an_evicted_name_in_memory_mode_is_a_new_name_again() {
        let Some(root) = artifact_root() else {
            return;
        };
        let _capacity = SmallCapacity::set();
        let (_runtime, tokenizer) = qwen(&root, None);
        // In memory mode the core keeps its own eviction: past the
        // capacity the least recently used session, `s0`, is dropped.
        for index in 0..=SMALL_CAPACITY {
            seed_new(&tokenizer, &format!("s{index}"), &text(index, 80)).unwrap();
        }
        let stats = tokenizer.store_stats();
        assert_eq!(stats.session_count, SMALL_CAPACITY as u64);
        assert_eq!(stats.sessions_evicted, 1);

        let mut s0 = tokenizer.open_session("s0").unwrap();
        assert_eq!(s0.revision(), None);
        let again = text(50, 80);
        assert_eq!(
            s0.seed(&again).unwrap().ids(),
            tokenizer.encode(&again).unwrap().ids()
        );
        s0.close().unwrap();
        assert_eq!(
            snapshot_of(&tokenizer, "s0").unwrap(),
            tokenizer.encode(&again).unwrap().ids()
        );
    }

    #[test]
    fn a_failed_open_keeps_no_lease() {
        let Some(root) = artifact_root() else {
            return;
        };
        let _capacity = SmallCapacity::set();
        let temporary = tempfile::tempdir().unwrap();
        let (_runtime, tokenizer) = qwen(&root, Some(&temporary.path().join("home")));
        seed_new(&tokenizer, "s0", &text(0, 80)).unwrap();
        // A name whose stored session the core no longer holds, which no
        // public call produces in a durable store: opening it fails.
        {
            let mut state = tokenizer.inner.store.lock().unwrap();
            let handle = state.names["s0"];
            assert!(state.store.evict(handle));
        }
        for _ in 0..2 {
            let error = tokenizer.open_session("s0").unwrap_err();
            assert_eq!(error.code(), ErrorCode::InvalidArgument, "{error}");
            assert!(!tokenizer.inner.store.lock().unwrap().leased.contains("s0"));
        }
    }

    // ---- the lookup index and the store's memory --------------------------
    //
    // BEGIN session memory tests. The helpers right below read what the
    // index change added; every test after them goes through them or the
    // public calls, so the same tests can be compiled against a tree
    // without the change by giving these helpers stand-in bodies.

    fn reused_ids(tokenizer: &crate::Tokenizer) -> u64 {
        tokenizer.store_stats().lookup_reused_ids
    }

    fn index_bytes(tokenizer: &crate::Tokenizer) -> u64 {
        tokenizer.store_stats().index_bytes
    }

    fn with_index_blocks(builder: crate::RuntimeBuilder, blocks: usize) -> crate::RuntimeBuilder {
        builder.lookup_index_blocks(blocks)
    }

    /// The index equals one rebuilt from the sessions the store holds
    /// now (`texts` gives the current text of every name), and keeps its
    /// own invariants.
    fn assert_index_matches(tokenizer: &crate::Tokenizer, texts: &HashMap<String, String>) {
        let state = tokenizer.inner.store.lock().unwrap();
        state.index.check_invariants().unwrap();
        let mut rebuilt = PrefixIndex::new(state.index.params(), usize::MAX);
        for (name, handle) in &state.names {
            assert!(
                state.store.revision(*handle).is_ok(),
                "{name} names a session the core does not hold"
            );
            let text = texts
                .get(name)
                .unwrap_or_else(|| panic!("no text for {name}"));
            rebuilt.insert_session(*handle, text, Seal::default());
        }
        assert_eq!(
            state.store.list_handles().len(),
            state.names.len(),
            "every session the core holds is named"
        );
        assert!(
            rebuilt.shape() == state.index.shape(),
            "the index differs from one rebuilt from the live sessions"
        );
    }

    fn core_node_count(tokenizer: &crate::Tokenizer) -> u64 {
        tokenizer
            .inner
            .store
            .lock()
            .unwrap()
            .store
            .stats()
            .node_count
    }

    // ---- shared setup ---------------------------------------------------------

    /// A runtime that runs the fast CPU engine whatever the registry's
    /// binding of this tree is, so sessions get seal points.
    fn experimental(
        root: &Path,
        home: Option<&Path>,
        index_blocks: Option<usize>,
    ) -> (crate::Runtime, crate::Tokenizer) {
        experimental_family(root, home, index_blocks, "qwen3_8b")
    }

    fn experimental_family(
        root: &Path,
        home: Option<&Path>,
        index_blocks: Option<usize>,
        family: &str,
    ) -> (crate::Runtime, crate::Tokenizer) {
        let mut builder = crate::Runtime::builder()
            .artifact_cache(root)
            .device(crate::Device::Cpu)
            .policy(Policy::Experimental);
        if let Some(home) = home {
            builder = builder.home(home);
        }
        if let Some(blocks) = index_blocks {
            builder = with_index_blocks(builder, blocks);
        }
        let runtime = builder.build().unwrap();
        let tokenizer = runtime.load(family).unwrap();
        assert!(
            tokenizer.plan().backends.contains(&Backend::FastCpu),
            "the experimental plan runs the fast CPU engine"
        );
        (runtime, tokenizer)
    }

    /// A small deterministic generator (SplitMix64).
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }

        fn below(&mut self, bound: usize) -> usize {
            (self.next() % bound.max(1) as u64) as usize
        }
    }

    #[derive(Clone, Copy)]
    enum Flavor {
        Ascii,
        Multibyte,
        /// Pieces that NFC normalization changes (decomposed accents,
        /// conjoining Hangul jamo, the Angstrom sign).
        Nfc,
    }

    /// `chars` characters of word-like text.
    fn prose(seed: u64, chars: usize, flavor: Flavor) -> String {
        const ASCII: [&str; 10] = [
            "river ",
            "stone ",
            "maple, ",
            "cloud ",
            "lantern. ",
            "harbor ",
            "meadow ",
            "copper ",
            "42 ",
            "\n",
        ];
        const MULTIBYTE: [&str; 6] = [
            "caf\u{e9} ",
            "\u{65e5}\u{672c}\u{8a9e}\u{306e}\u{6587}\u{7ae0} ",
            "\u{41f}\u{440}\u{438}\u{432}\u{435}\u{442} ",
            "\u{1f642} ",
            "na\u{ef}ve ",
            "\u{3b1}\u{3b2}\u{3b3} ",
        ];
        const NFC: [&str; 5] = [
            "e\u{301}te\u{301} ",
            "A\u{30a}ngstro\u{308}m ",
            "\u{1100}\u{1161}\u{11a8} ",
            "\u{212b} ",
            "u\u{308}ber ",
        ];
        let mut rng = Rng(seed ^ 0x51ED_2701);
        let mut out = String::new();
        let mut count = 0usize;
        while count < chars {
            let piece = match (flavor, rng.below(4)) {
                (Flavor::Multibyte, 0) | (Flavor::Nfc, 0) => MULTIBYTE[rng.below(MULTIBYTE.len())],
                (Flavor::Nfc, 1) => NFC[rng.below(NFC.len())],
                _ => ASCII[rng.below(ASCII.len())],
            };
            out.push_str(piece);
            count += piece.chars().count();
        }
        out.chars().take(chars).collect()
    }

    fn take_chars(text: &str, chars: usize) -> String {
        text.chars().take(chars).collect()
    }

    fn ids_of(tokenizer: &crate::Tokenizer, text: &str) -> Vec<u32> {
        tokenizer.encode(text).unwrap().ids().to_vec()
    }

    /// A lookup's answer: `None`, or ids that must equal `encode`.
    fn lookup_checked(tokenizer: &crate::Tokenizer, query: &str) -> Option<Vec<u32>> {
        let found = tokenizer.lookup(query).unwrap()?;
        assert_eq!(found.ids(), ids_of(tokenizer, query), "lookup ids");
        Some(found.ids().to_vec())
    }

    /// Seed `name` with `first` and grow it by `turns` appends of
    /// `turn_chars` characters; every snapshot equals `encode`.
    fn grown_session(
        tokenizer: &crate::Tokenizer,
        name: &str,
        seed: u64,
        first: usize,
        turns: usize,
        turn_chars: usize,
        flavor: Flavor,
    ) -> String {
        let mut text = prose(seed, first, flavor);
        let mut session = tokenizer.open_session(name).unwrap();
        session.seed(&text).unwrap();
        for turn in 0..turns {
            let delta = prose(seed * 1_000 + turn as u64, turn_chars, flavor);
            session.append(&delta).unwrap();
            text.push_str(&delta);
            assert_eq!(
                session.snapshot().unwrap().ids(),
                ids_of(tokenizer, &text),
                "{name} after turn {turn}"
            );
        }
        session.close().unwrap();
        text
    }

    #[test]
    fn the_core_writes_no_block_record() {
        let Some(root) = artifact_root() else {
            return;
        };
        let (_runtime, tokenizer) = experimental(&root, None, None);
        let a = grown_session(&tokenizer, "a", 1, 6_000, 6, 1_500, Flavor::Ascii);
        assert_eq!(core_node_count(&tokenizer), 0, "after seeds and appends");
        seed_new(&tokenizer, "b", &prose(2, 20_000, Flavor::Multibyte)).unwrap();
        let source = tokenizer.open_session("a").unwrap();
        let mut fork = source.fork("a-fork").unwrap();
        fork.append(&prose(3, 9_000, Flavor::Ascii)).unwrap();
        fork.close().unwrap();
        source.close().unwrap();
        assert_eq!(core_node_count(&tokenizer), 0, "after a fork");
        for query in [format!("{a} and a question"), take_chars(&a, 9_000)] {
            lookup_checked(&tokenizer, &query).expect("a hit");
        }
        assert_eq!(core_node_count(&tokenizer), 0, "after lookups");
        assert!(reused_ids(&tokenizer) > 0);
    }

    #[test]
    fn deleted_overwritten_and_dropped_sessions_release_their_entries() {
        let Some(root) = artifact_root() else {
            return;
        };
        {
            let (_runtime, tokenizer) = experimental(&root, None, None);
            let empty = index_bytes(&tokenizer);
            let mut texts = Vec::new();
            for index in 0..6u64 {
                let name = format!("s{index}");
                let flavor = if index % 2 == 0 {
                    Flavor::Ascii
                } else {
                    Flavor::Nfc
                };
                texts.push((
                    name.clone(),
                    grown_session(&tokenizer, &name, 10 + index, 5_000, 4, 2_000, flavor),
                ));
            }
            assert!(tokenizer.store_stats().node_count > 0);
            assert!(index_bytes(&tokenizer) > empty);
            for (name, _) in &texts {
                tokenizer.open_session(name).unwrap().delete().unwrap();
            }
            let stats = tokenizer.store_stats();
            assert_eq!(stats.node_count, 0, "entries after every delete");
            assert_eq!(stats.session_count, 0);
            assert_eq!(
                index_bytes(&tokenizer),
                empty,
                "index bytes after every delete"
            );
            for (name, text) in &texts {
                assert!(
                    lookup_checked(&tokenizer, &format!("{text} and more")).is_none(),
                    "{name} was deleted"
                );
            }

            // `overwrite`: the old text misses, the new one hits.
            let old = grown_session(&tokenizer, "o", 20, 5_000, 3, 2_000, Flavor::Ascii);
            let new = prose(21, 12_000, Flavor::Multibyte);
            let mut session = tokenizer.open_session("o").unwrap();
            assert_eq!(
                session.overwrite(&new).unwrap().ids(),
                ids_of(&tokenizer, &new)
            );
            session.close().unwrap();
            assert!(lookup_checked(&tokenizer, &format!("{old} and more")).is_none());
            lookup_checked(&tokenizer, &format!("{new} and more")).expect("a hit");
            assert_eq!(tokenizer.store_stats().node_count, 2);
            assert_index_matches(&tokenizer, &HashMap::from([("o".to_owned(), new.clone())]));
        }

        // Sessions the core drops at its capacity (memory mode).
        let _capacity = SmallCapacity::set();
        let (_runtime, tokenizer) = experimental(&root, None, None);
        let mut texts = HashMap::new();
        for index in 0..=SMALL_CAPACITY as u64 {
            let name = format!("c{index}");
            texts.insert(
                name.clone(),
                grown_session(
                    &tokenizer,
                    &name,
                    30 + index,
                    5_000,
                    2,
                    2_000,
                    Flavor::Ascii,
                ),
            );
        }
        let stats = tokenizer.store_stats();
        assert_eq!(stats.sessions_evicted, 1);
        assert_eq!(
            stats.node_count,
            2 * SMALL_CAPACITY as u64,
            "entries of the live sessions only"
        );
        assert!(lookup_checked(&tokenizer, &format!("{} and more", texts["c0"])).is_none());
        texts.remove("c0");
        lookup_checked(&tokenizer, &format!("{} and more", texts["c1"])).expect("a hit");
        assert_index_matches(&tokenizer, &texts);
    }

    /// A random replay of every session call and of lookups over texts
    /// that share prefixes; ids equal `encode` at every step.
    #[test]
    fn a_random_replay_keeps_ids_exact_and_the_index_consistent() {
        let Some(root) = artifact_root() else {
            return;
        };
        for seed in [11u64, 12, 13] {
            replay(&root, seed, 300);
        }
    }

    fn replay(root: &Path, seed: u64, steps: usize) {
        let (_runtime, tokenizer) = experimental(root, None, None);
        let pool: Vec<String> = (0..3u64)
            .map(|index| {
                let flavor = [Flavor::Ascii, Flavor::Multibyte, Flavor::Nfc][index as usize];
                prose(seed * 10 + index, 26_000, flavor)
            })
            .collect();
        let mut rng = Rng(seed);
        let mut live: HashMap<String, String> = HashMap::new();
        let mut gone: Vec<String> = Vec::new();
        let mut next_name = 0usize;
        // Lookups made and lookups that hit, to compare with the counters.
        let (mut lookups, mut hits) = (0u64, 0u64);
        // A text that follows one of the pool texts, or not, from `at`.
        let continuation = |rng: &mut Rng, text: &str| -> String {
            let len = 200 + rng.below(3_000);
            let chars = text.chars().count();
            let base = &pool[rng.below(pool.len())];
            if rng.below(3) != 0 && base.starts_with(text) && chars + len <= 26_000 {
                base.chars().skip(chars).take(len).collect()
            } else {
                prose(rng.next(), len, Flavor::Nfc)
            }
        };
        let fresh_text = |rng: &mut Rng| -> String {
            let base = &pool[rng.below(pool.len())];
            let mut text = take_chars(base, 1_000 + rng.below(12_000));
            if rng.below(3) == 0 {
                text.push_str(&prose(rng.next(), 500, Flavor::Ascii));
            }
            text
        };
        for step in 0..steps {
            let names: Vec<String> = {
                let mut names: Vec<String> = live.keys().cloned().collect();
                names.sort();
                names
            };
            let pick = |rng: &mut Rng| names[rng.below(names.len())].clone();
            match rng.below(12) {
                0 | 1 if live.len() < 12 => {
                    let name = format!("r{next_name}");
                    next_name += 1;
                    let text = fresh_text(&mut rng);
                    assert_eq!(
                        seed_new(&tokenizer, &name, &text).unwrap(),
                        ids_of(&tokenizer, &text)
                    );
                    live.insert(name, text);
                }
                2..=4 if !names.is_empty() => {
                    let name = pick(&mut rng);
                    if live[&name].chars().count() > 30_000 {
                        continue;
                    }
                    let delta = continuation(&mut rng, &live[&name]);
                    let mut session = tokenizer.open_session(&name).unwrap();
                    session.append(&delta).unwrap();
                    let text = live.get_mut(&name).unwrap();
                    text.push_str(&delta);
                    assert_eq!(
                        session.snapshot().unwrap().ids(),
                        ids_of(&tokenizer, text),
                        "step {step}"
                    );
                    session.close().unwrap();
                }
                5 if !names.is_empty() => {
                    let name = pick(&mut rng);
                    let delta = continuation(&mut rng, &live[&name]);
                    let complete = format!("{}{delta}", live[&name]);
                    let mut session = tokenizer.open_session(&name).unwrap();
                    assert_eq!(
                        session.encode_transcript(&complete).unwrap().ids(),
                        ids_of(&tokenizer, &complete),
                        "step {step}"
                    );
                    session.close().unwrap();
                    live.insert(name, complete);
                }
                6 if !names.is_empty() => {
                    let name = pick(&mut rng);
                    let text = fresh_text(&mut rng);
                    let mut session = tokenizer.open_session(&name).unwrap();
                    assert_eq!(
                        session.overwrite(&text).unwrap().ids(),
                        ids_of(&tokenizer, &text)
                    );
                    session.close().unwrap();
                    gone.push(std::mem::replace(live.get_mut(&name).unwrap(), text));
                }
                7 if !names.is_empty() && live.len() < 12 => {
                    let source = pick(&mut rng);
                    let name = format!("r{next_name}");
                    next_name += 1;
                    let session = tokenizer.open_session(&source).unwrap();
                    let fork = session.fork(&name).unwrap();
                    assert_eq!(
                        fork.snapshot().unwrap().ids(),
                        ids_of(&tokenizer, &live[&source])
                    );
                    fork.close().unwrap();
                    session.close().unwrap();
                    live.insert(name, live[&source].clone());
                }
                8 if !names.is_empty() => {
                    let name = pick(&mut rng);
                    tokenizer.open_session(&name).unwrap().delete().unwrap();
                    gone.push(live.remove(&name).unwrap());
                }
                9 if !names.is_empty() => {
                    let name = pick(&mut rng);
                    assert_eq!(
                        snapshot_of(&tokenizer, &name).unwrap(),
                        ids_of(&tokenizer, &live[&name])
                    );
                }
                _ => {
                    // A lookup: a live, gone or pool text, cut or extended.
                    let source = match rng.below(3) {
                        0 if !names.is_empty() => live[&pick(&mut rng)].clone(),
                        1 if !gone.is_empty() => gone[rng.below(gone.len())].clone(),
                        _ => pool[rng.below(pool.len())].clone(),
                    };
                    let chars = source.chars().count();
                    let mut query = take_chars(&source, rng.below(chars + 1));
                    if rng.below(2) == 0 {
                        query.push_str(&prose(rng.next(), rng.below(3_000), Flavor::Nfc));
                    }
                    let found = lookup_checked(&tokenizer, &query);
                    lookups += 1;
                    hits += u64::from(found.is_some());
                    let shares_a_block = live.values().any(|text| {
                        take_chars(text, 4_096).chars().count() == 4_096
                            && query.starts_with(&take_chars(text, 4_096))
                    });
                    assert_eq!(
                        found.is_some(),
                        shares_a_block,
                        "step {step}: hit iff a live session has the first block"
                    );
                }
            }
        }
        assert_index_matches(&tokenizer, &live);
        assert_eq!(core_node_count(&tokenizer), 0);
        assert!(
            reused_ids(&tokenizer) > 0,
            "seed {seed}: lookups reused sealed ids"
        );
        let stats = tokenizer.store_stats();
        assert!(hits > 0, "seed {seed}: lookups hit");
        assert_eq!(stats.lookups, lookups, "seed {seed}: lookups counted");
        assert_eq!(stats.lookup_hits, hits, "seed {seed}: hits counted");
        assert_eq!(
            stats.lookup_misses,
            lookups - hits,
            "seed {seed}: misses counted"
        );
    }

    #[test]
    fn a_durable_restart_rebuilds_the_index_and_keeps_no_block_rows() {
        let Some(root) = artifact_root() else {
            return;
        };
        let temporary = tempfile::tempdir().unwrap();
        let home = temporary.path().join("home");
        let (d0, d1) = {
            let (_runtime, tokenizer) = experimental(&root, Some(&home), None);
            let d0 = grown_session(&tokenizer, "d0", 40, 3_000, 8, 1_500, Flavor::Nfc);
            let d1 = grown_session(&tokenizer, "d1", 41, 3_000, 8, 1_500, Flavor::Ascii);
            (d0, d1)
        };

        // The file holds no block rows.
        let file = home.join("sessions/qwen3_8b.sqlite3");
        let copy = temporary.path().join("copy.sqlite3");
        fs::copy(&file, &copy).unwrap();
        for suffix in ["-wal", "-shm"] {
            let mut from = file.as_os_str().to_os_string();
            from.push(suffix);
            let mut to = copy.as_os_str().to_os_string();
            to.push(suffix);
            if Path::new(&from).exists() {
                fs::copy(&from, &to).unwrap();
            }
        }
        {
            let (_runtime, tokenizer) = experimental(&root, None, None);
            let (store, _) = StoreDb::open(&copy)
                .unwrap()
                .load(&SingleEngine(tokenizer.inner.router.as_ref()))
                .unwrap();
            assert_eq!(
                store.stats().node_count,
                0,
                "block rows in the durable file"
            );
            assert_eq!(store.stats().session_count, 2);
        }

        {
            let (_runtime, tokenizer) = experimental(&root, Some(&home), None);
            assert_eq!(core_node_count(&tokenizer), 0);
            // A hit at once after the restart, exact, with nothing reused
            // yet: the stored blocks carry no seal.
            lookup_checked(&tokenizer, &format!("{d0} and a question")).expect("a hit");
            lookup_checked(&tokenizer, &take_chars(&d1, 10_000)).expect("a hit");
            assert_eq!(reused_ids(&tokenizer), 0);
            assert_index_matches(
                &tokenizer,
                &HashMap::from([("d0".to_owned(), d0.clone()), ("d1".to_owned(), d1.clone())]),
            );
            // Once d0 completes another block, its ids are reused again.
            let mut session = tokenizer.open_session("d0").unwrap();
            let delta = prose(42, 5_000, Flavor::Nfc);
            session.append(&delta).unwrap();
            session.close().unwrap();
            let grown = format!("{d0}{delta}");
            lookup_checked(&tokenizer, &format!("{grown} and a question")).expect("a hit");
            assert!(reused_ids(&tokenizer) > 0, "reuse after the next block");
            tokenizer.open_session("d1").unwrap().delete().unwrap();
        }

        let (_runtime, tokenizer) = experimental(&root, Some(&home), None);
        assert!(
            lookup_checked(&tokenizer, &format!("{d1} and a question")).is_none(),
            "a deleted session misses after a restart"
        );
        lookup_checked(&tokenizer, &format!("{d0} and a question")).expect("a hit");
    }

    /// Creators, deleters, appenders and lookups at once on a full store:
    /// a durable store at its 1,023 named sessions and a memory store at
    /// its capacity of 1,024.
    #[test]
    fn concurrent_calls_at_the_bound_stay_exact_and_consistent() {
        let Some(root) = artifact_root() else {
            return;
        };
        let temporary = tempfile::tempdir().unwrap();
        for durable in [true, false] {
            let home = temporary.path().join("home");
            concurrency_at_the_bound(&root, durable.then_some(home.as_path()));
        }
    }

    fn concurrency_at_the_bound(root: &Path, home: Option<&Path>) {
        use std::sync::{Arc as SyncArc, Barrier, Mutex};
        let durable = home.is_some();
        let (_runtime, tokenizer) = experimental(root, home, None);
        let capacity = tokenizer
            .inner
            .store
            .lock()
            .unwrap()
            .store
            .config()
            .max_sessions;
        let fill = if durable { capacity - 1 } else { capacity };
        let mut texts: HashMap<String, String> = HashMap::new();
        // Long sessions the lookups and appenders use, then short ones.
        for index in 0..4u64 {
            let name = format!("lk{index}");
            texts.insert(
                name.clone(),
                grown_session(
                    &tokenizer,
                    &name,
                    50 + index,
                    4_000,
                    3,
                    1_500,
                    Flavor::Ascii,
                ),
            );
        }
        for index in 0..2u64 {
            let name = format!("ap{index}");
            texts.insert(
                name.clone(),
                grown_session(&tokenizer, &name, 60 + index, 4_000, 2, 1_500, Flavor::Nfc),
            );
        }
        for index in texts.len()..fill {
            let name = format!("f{index}");
            let text = text(index, 48);
            seed_new(&tokenizer, &name, &text).unwrap();
            texts.insert(name, text);
        }
        assert_eq!(tokenizer.store_stats().session_count, fill as u64);
        let evicted_before = tokenizer.store_stats().sessions_evicted;
        let queries: Vec<String> = (0..4)
            .map(|index| texts[&format!("lk{index}")].clone())
            .collect();
        let texts = SyncArc::new(Mutex::new(texts));
        let created = SyncArc::new(std::sync::atomic::AtomicU64::new(0));
        let threads = 8;
        let barrier = SyncArc::new(Barrier::new(threads));
        let (done, finished) = std::sync::mpsc::channel();
        for worker in 0..threads {
            let tokenizer = tokenizer.clone();
            let barrier = SyncArc::clone(&barrier);
            let texts = SyncArc::clone(&texts);
            let created = SyncArc::clone(&created);
            let queries = queries.clone();
            let done = done.clone();
            std::thread::spawn(move || {
                barrier.wait();
                match worker {
                    0 | 1 => {
                        for turn in 0..4u64 {
                            let name = format!("c{worker}-{turn}");
                            let text = prose(100 + worker as u64 * 10 + turn, 6_000, Flavor::Ascii);
                            match seed_new(&tokenizer, &name, &text) {
                                Ok(ids) => {
                                    assert_eq!(ids, ids_of(&tokenizer, &text));
                                    created.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                                    texts.lock().unwrap().insert(name, text);
                                }
                                Err(error) => {
                                    assert!(durable, "memory mode refuses nothing: {error}");
                                    assert_eq!(error.code(), ErrorCode::InvalidArgument, "{error}");
                                }
                            }
                        }
                    }
                    2 => {
                        for index in 10..16usize {
                            let name = format!("f{index}");
                            tokenizer.open_session(&name).unwrap().delete().unwrap();
                            texts.lock().unwrap().remove(&name);
                        }
                    }
                    3 | 4 => {
                        let name = format!("ap{}", worker - 3);
                        let mut stored = texts.lock().unwrap()[&name].clone();
                        let mut session = tokenizer.open_session(&name).unwrap();
                        for turn in 0..4u64 {
                            let delta = prose(200 + worker as u64 * 10 + turn, 1_500, Flavor::Nfc);
                            match session.append(&delta) {
                                Ok(_) => {
                                    stored.push_str(&delta);
                                    texts.lock().unwrap().insert(name.clone(), stored.clone());
                                    match session.snapshot() {
                                        Ok(snapshot) => {
                                            assert_eq!(snapshot.ids(), ids_of(&tokenizer, &stored))
                                        }
                                        Err(error) => {
                                            assert!(!durable, "{error}");
                                            break;
                                        }
                                    }
                                }
                                Err(error) => {
                                    // Memory mode: the core may drop this
                                    // session to make room for a creator's.
                                    assert!(!durable, "{error}");
                                    break;
                                }
                            }
                        }
                        session.close().unwrap();
                    }
                    _ => {
                        let mut rng = Rng(worker as u64);
                        for _ in 0..12 {
                            let source = &queries[rng.below(queries.len())];
                            let chars = source.chars().count();
                            let mut query =
                                take_chars(source, 4_096 + rng.below(chars - 4_096 + 1));
                            query.push_str(&prose(rng.next(), rng.below(2_000), Flavor::Ascii));
                            let found = lookup_checked(&tokenizer, &query);
                            if durable {
                                assert!(found.is_some(), "a stored session's prefix hits");
                            }
                        }
                    }
                }
                done.send(worker).unwrap();
            });
        }
        drop(done);
        for _ in 0..threads {
            finished
                .recv_timeout(Duration::from_secs(900))
                .expect("every worker finishes in time");
        }
        let stats = tokenizer.store_stats();
        let created = created.load(std::sync::atomic::Ordering::SeqCst);
        if durable {
            assert_eq!(stats.sessions_evicted, 0);
        } else {
            // Each creation past the capacity drops one session; a lookup
            // drops none.
            assert!(stats.sessions_evicted - evicted_before <= created);
        }
        assert_index_matches(&tokenizer, &texts.lock().unwrap());
        assert_eq!(core_node_count(&tokenizer), 0);
    }

    #[test]
    fn memory_mode_names_follow_the_core() {
        let Some(root) = artifact_root() else {
            return;
        };
        let _capacity = SmallCapacity::set();
        let (_runtime, tokenizer) = experimental(&root, None, None);
        for index in 0..3 * SMALL_CAPACITY {
            seed_new(&tokenizer, &format!("s{index}"), &text(index, 80)).unwrap();
            assert!(tokenizer.inner.store.lock().unwrap().names.len() <= SMALL_CAPACITY);
        }
        // `s0` was dropped long ago and never opened again: a fork may
        // take its name.
        let source = tokenizer
            .open_session(format!("s{}", 3 * SMALL_CAPACITY - 1))
            .unwrap();
        let fork = source.fork("s0").unwrap();
        assert_eq!(
            fork.snapshot().unwrap().ids(),
            ids_of(&tokenizer, &text(3 * SMALL_CAPACITY - 1, 80))
        );
        fork.close().unwrap();
        source.close().unwrap();
        assert!(tokenizer.inner.store.lock().unwrap().names.len() <= SMALL_CAPACITY);
    }

    /// Pins the clause that declines `overwrite` of a session value whose
    /// state the core no longer holds when a durable store is full.
    #[test]
    fn overwrite_of_an_unstored_session_value_is_declined_at_the_bound() {
        let Some(root) = artifact_root() else {
            return;
        };
        let _capacity = SmallCapacity::set();
        let temporary = tempfile::tempdir().unwrap();
        let (_runtime, tokenizer) = qwen(&root, Some(&temporary.path().join("home")));
        for index in 0..SMALL_CAPACITY - 1 {
            seed_new(&tokenizer, &format!("s{index}"), &text(index, 80)).unwrap();
        }
        let mut s0 = tokenizer.open_session("s0").unwrap();
        {
            // The core loses s0's state and its slot is taken by another
            // session, which no public call does in a durable store.
            let mut state = tokenizer.inner.store.lock().unwrap();
            let handle = state.names["s0"];
            assert!(state.store.evict(handle));
            let key = state.key;
            state
                .store
                .put(key, &text(90, 80), tokenizer.inner.router.as_ref())
                .unwrap();
        }
        assert_declined(
            &s0.overwrite(&text(91, 80)).unwrap_err(),
            SMALL_CAPACITY - 1,
        );
        // Declined before any change: nothing was saved, nothing sealed.
        let state = tokenizer.inner.store.lock().unwrap();
        state.ensure_healthy().unwrap();
        assert_eq!(state.store.list_handles().len(), SMALL_CAPACITY - 1);
    }
    /// A session whose sixth block was completed by a call that made no
    /// new seal, so that block carries the session's current seal, which
    /// lies inside it. Returns the text, the handle, that seal, the length
    /// in characters after the call that crossed into the sixth block, and
    /// the text's length (the sixth block's end).
    fn session_with_its_seal_in_the_sixth_block(
        tokenizer: &crate::Tokenizer,
        flavor: Flavor,
    ) -> (String, SessionHandle, Seal, usize, usize) {
        let mut text = grown_session(tokenizer, "h", 7, 3_000, 9, 1_500, flavor);
        let mut session = tokenizer.open_session("h").unwrap();
        let crossed = 5 * 4_096 + 3_000;
        let first = prose(701, crossed - 16_500, flavor);
        session.append(&first).unwrap();
        text.push_str(&first);
        let end = 6 * 4_096;
        let second = prose(702, end - crossed, flavor);
        session.append(&second).unwrap();
        text.push_str(&second);
        assert_eq!(session.snapshot().unwrap().ids(), ids_of(tokenizer, &text));
        session.close().unwrap();
        let state = tokenizer.inner.store.lock().unwrap();
        let handle = state.names["h"];
        let current = seal_of(&state.store, handle).unwrap();
        assert!(
            current.chars as usize > 5 * 4_096 && (current.chars as usize) < crossed,
            "the current seal lies in the sixth block: {current:?}"
        );
        (text, handle, current, crossed, end)
    }

    /// The byte offset of character `chars` of `text`.
    fn byte_at_char(text: &str, chars: usize) -> usize {
        text.char_indices()
            .nth(chars)
            .map_or(text.len(), |(byte, _)| byte)
    }

    /// A holder whose seal does not agree with what the store holds is
    /// dropped and counted, nothing is reused, and the ids are those of
    /// `encode`. Each case registers the session again with one field of
    /// its sixth block's seal altered.
    #[test]
    fn holders_that_disagree_with_the_store_are_not_used() {
        let Some(root) = artifact_root() else {
            return;
        };
        for (which, flavor) in [Flavor::Ascii, Flavor::Multibyte].into_iter().enumerate() {
            let (_runtime, tokenizer) = experimental(&root, None, None);
            let (text, handle, current, crossed, end) =
                session_with_its_seal_in_the_sixth_block(&tokenizer, flavor);
            assert_eq!(
                current.bytes as usize,
                byte_at_char(&text, current.chars as usize)
            );
            assert_eq!(text.chars().count(), end);
            let query = format!("{text}{}", prose(703, 400, flavor));
            let expected = ids_of(&tokenizer, &query);
            let rejects = || tokenizer.store_stats().checksum_rejects;

            // As stored: the sixth block carries the current seal, and a
            // lookup reuses its ids.
            let before = reused_ids(&tokenizer);
            assert_eq!(tokenizer.lookup(&query).unwrap().unwrap().ids(), expected);
            assert_eq!(
                reused_ids(&tokenizer) - before,
                current.ids,
                "flavor {which}: reuse as stored"
            );
            assert_eq!(rejects(), 0);

            // Register the session again so that its sixth block carries
            // `altered`.
            let register_with = |altered: Seal| {
                let mut state = tokenizer.inner.store.lock().unwrap();
                state.index.remove_session(handle);
                let cut = byte_at_char(&text, crossed);
                state.index.insert_session(handle, &text[..cut], altered);
                state.index.append(handle, &text[cut..], current);
            };
            let mut cases: Vec<(&str, Seal)> = vec![
                (
                    "ids not in the seal log",
                    Seal {
                        ids: current.ids - 1,
                        ..current
                    },
                ),
                (
                    "chars not in the seal log",
                    Seal {
                        chars: current.chars - 1,
                        ..current
                    },
                ),
                (
                    "more ids than are sealed",
                    Seal {
                        ids: current.ids + 5,
                        ..current
                    },
                ),
                (
                    "bytes beyond the stable prefix",
                    Seal {
                        bytes: byte_at_char(&text, current.chars as usize + 3) as u64,
                        ..current
                    },
                ),
            ];
            if let Some(inside) = (1..current.bytes as usize)
                .rev()
                .find(|byte| !text.is_char_boundary(*byte))
            {
                cases.push((
                    "bytes inside a character",
                    Seal {
                        bytes: inside as u64,
                        ..current
                    },
                ));
            }
            for (what, altered) in cases {
                register_with(altered);
                let rejected = rejects();
                let reused = reused_ids(&tokenizer);
                let found = tokenizer.lookup(&query).unwrap().expect("a hit");
                assert_eq!(found.ids(), expected, "flavor {which}, {what}: ids");
                assert_eq!(
                    rejects(),
                    rejected + 1,
                    "flavor {which}, {what}: counted as a reject"
                );
                assert_eq!(
                    reused_ids(&tokenizer),
                    reused,
                    "flavor {which}, {what}: nothing reused"
                );
                assert_eq!(tokenizer.lookup(&query).unwrap().unwrap().ids(), expected);
                assert_eq!(
                    rejects(),
                    rejected + 1,
                    "flavor {which}, {what}: dropped after the reject"
                );
            }

            // A seal that lies beyond the block it is listed for (the
            // second block).
            {
                let mut state = tokenizer.inner.store.lock().unwrap();
                state.index.remove_session(handle);
                let cut = byte_at_char(&text, 5_000);
                state.index.insert_session(handle, &text[..cut], current);
                state.index.append(handle, &text[cut..], current);
            }
            let other = format!(
                "{}{}",
                &text[..byte_at_char(&text, 8_192)],
                prose(704, 30_000, flavor)
            );
            let expected_other = ids_of(&tokenizer, &other);
            let rejected = rejects();
            assert_eq!(
                tokenizer.lookup(&other).unwrap().unwrap().ids(),
                expected_other,
                "flavor {which}: a seal beyond its block"
            );
            assert_eq!(
                rejects(),
                rejected + 1,
                "flavor {which}: a seal beyond its block is a reject"
            );

            // The core no longer holds the session; its track is still
            // there.
            register_with(current);
            assert!(tokenizer.inner.store.lock().unwrap().store.evict(handle));
            let rejected = rejects();
            assert_eq!(tokenizer.lookup(&query).unwrap().unwrap().ids(), expected);
            assert_eq!(
                rejects(),
                rejected + 1,
                "flavor {which}: a session the core dropped is a reject"
            );
            tokenizer
                .inner
                .store
                .lock()
                .unwrap()
                .index
                .check_invariants()
                .unwrap();
        }
    }

    /// The byte offset is the one field of a holder the seal log does not
    /// hold: it is confirmed on the query. A holder whose offset is another
    /// character boundary than the one its seal's character count names is
    /// dropped and counted, and the ids are those of `encode`; an older
    /// seal of the log with its own offset is still reused.
    #[test]
    fn a_holder_whose_byte_offset_disagrees_with_its_characters_is_not_used() {
        let Some(root) = artifact_root() else {
            return;
        };
        for (which, flavor) in [Flavor::Ascii, Flavor::Multibyte].into_iter().enumerate() {
            let (_runtime, tokenizer) = experimental(&root, None, None);
            let (text, handle, current, crossed, _end) =
                session_with_its_seal_in_the_sixth_block(&tokenizer, flavor);
            let older = {
                let state = tokenizer.inner.store.lock().unwrap();
                let sidecar = state.store.export_session_sidecar(handle).unwrap();
                let log = toktier_store_core::SessionSidecar::from_bytes(&sidecar)
                    .unwrap()
                    .seal_log;
                let (chars, ids) = log
                    .iter()
                    .copied()
                    .filter(|&(chars, ids)| ids > 0 && chars + 3 < current.chars)
                    .max_by_key(|&(_, ids)| ids)
                    .expect("an older seal in the log");
                Seal {
                    chars,
                    bytes: byte_at_char(&text, chars as usize) as u64,
                    ids,
                }
            };
            let query = format!("{text}{}", prose(705, 400, flavor));
            let expected = ids_of(&tokenizer, &query);
            let rejects = || tokenizer.store_stats().checksum_rejects;
            let register_with = |altered: Seal| {
                let mut state = tokenizer.inner.store.lock().unwrap();
                state.index.remove_session(handle);
                let cut = byte_at_char(&text, crossed);
                state.index.insert_session(handle, &text[..cut], altered);
                state.index.append(handle, &text[cut..], current);
            };

            // An older seal of the log, recorded as it was: reused.
            register_with(older);
            let reused = reused_ids(&tokenizer);
            assert_eq!(tokenizer.lookup(&query).unwrap().unwrap().ids(), expected);
            assert_eq!(
                reused_ids(&tokenizer) - reused,
                older.ids,
                "flavor {which}: an older seal is reused"
            );
            assert_eq!(rejects(), 0);

            let cases = [
                (
                    "bytes three characters below the seal",
                    Seal {
                        bytes: byte_at_char(&text, current.chars as usize - 3) as u64,
                        ..current
                    },
                ),
                (
                    "an older seal with bytes three characters above it",
                    Seal {
                        bytes: byte_at_char(&text, older.chars as usize + 3) as u64,
                        ..older
                    },
                ),
            ];
            for (what, altered) in cases {
                register_with(altered);
                let rejected = rejects();
                let reused = reused_ids(&tokenizer);
                let found = tokenizer.lookup(&query).unwrap().expect("a hit");
                assert_eq!(found.ids(), expected, "flavor {which}, {what}: ids");
                assert_eq!(
                    rejects(),
                    rejected + 1,
                    "flavor {which}, {what}: counted as a reject"
                );
                assert_eq!(
                    reused_ids(&tokenizer),
                    reused,
                    "flavor {which}, {what}: nothing reused"
                );
                assert_eq!(tokenizer.lookup(&query).unwrap().unwrap().ids(), expected);
                assert_eq!(
                    rejects(),
                    rejected + 1,
                    "flavor {which}, {what}: dropped after the reject"
                );
            }
        }
    }

    /// The byte offset of a holder is checked only within its seal's own
    /// block: an offset past the end of that block is refused before any
    /// character is counted, so the work under the store lock stays within
    /// one block however far the offset points. The case is a seal of the
    /// first block given the byte offset of the current seal in the sixth
    /// block, which passes every check before the count. A seal recorded
    /// as it was is still counted from the start of its block and reused.
    #[test]
    fn a_holder_byte_offset_beyond_its_seal_block_is_refused_before_counting() {
        let Some(root) = artifact_root() else {
            return;
        };
        let counted = || crate::runtime::SEALED_PREFIX_COUNTED.with(|total| total.replace(0));
        for (which, flavor) in [Flavor::Ascii, Flavor::Multibyte].into_iter().enumerate() {
            let (_runtime, tokenizer) = experimental(&root, None, None);
            let (text, handle, current, crossed, _end) =
                session_with_its_seal_in_the_sixth_block(&tokenizer, flavor);
            let (first_chars, first_ids) = {
                let state = tokenizer.inner.store.lock().unwrap();
                let sidecar = state.store.export_session_sidecar(handle).unwrap();
                toktier_store_core::SessionSidecar::from_bytes(&sidecar)
                    .unwrap()
                    .seal_log
                    .iter()
                    .copied()
                    .filter(|&(_, ids)| ids > 0)
                    .min_by_key(|&(chars, _)| chars)
                    .expect("a seal in the log")
            };
            assert!(
                first_chars < 4_096,
                "flavor {which}: the first seal lies in the first block: {first_chars}"
            );
            let query = format!("{text}{}", prose(706, 400, flavor));
            let expected = ids_of(&tokenizer, &query);
            let rejects = || tokenizer.store_stats().checksum_rejects;
            let register_with = |altered: Seal| {
                let mut state = tokenizer.inner.store.lock().unwrap();
                state.index.remove_session(handle);
                let cut = byte_at_char(&text, crossed);
                state.index.insert_session(handle, &text[..cut], altered);
                state.index.append(handle, &text[cut..], current);
            };

            // The current seal, as recorded: counted within its own block
            // and reused.
            register_with(current);
            counted();
            let reused = reused_ids(&tokenizer);
            assert_eq!(tokenizer.lookup(&query).unwrap().unwrap().ids(), expected);
            assert_eq!(reused_ids(&tokenizer) - reused, current.ids);
            assert_eq!(
                counted(),
                current.chars - 5 * 4_096,
                "flavor {which}: a recorded seal is counted from the start of its block"
            );
            assert_eq!(rejects(), 0);

            // The first seal with the current seal's byte offset.
            let altered = Seal {
                chars: first_chars,
                ids: first_ids,
                bytes: current.bytes,
            };
            register_with(altered);
            let reused = reused_ids(&tokenizer);
            let found = tokenizer.lookup(&query).unwrap().expect("a hit");
            let scanned = counted();
            eprintln!(
                "flavor {which}: seal at {first_chars} characters with the byte offset of {} characters: {scanned} characters counted",
                current.chars
            );
            assert_eq!(found.ids(), expected, "flavor {which}: ids");
            assert_eq!(rejects(), 1, "flavor {which}: counted as a reject");
            assert_eq!(
                reused_ids(&tokenizer),
                reused,
                "flavor {which}: nothing reused"
            );
            assert_eq!(
                scanned, 0,
                "flavor {which}: refused before any character is counted"
            );
        }
    }

    /// `store_stats().index_bytes` comes from running totals of the
    /// holder and key slots. Through seeds, appends, overwrites, forks,
    /// deletes, sessions the core drops at its capacity, blocks declined
    /// at the index cap, lookups and rejected holders, the totals equal a
    /// recount over every entry and track after each call (the crate's
    /// own test builds also compare them inside every `store_stats`).
    #[test]
    fn index_size_totals_equal_a_recount_after_every_call() {
        let Some(root) = artifact_root() else {
            return;
        };
        let _capacity = SmallCapacity::set();
        for (seed, index_blocks) in [(41u64, None), (42, Some(12)), (43, Some(3))] {
            let (_runtime, tokenizer) = experimental(&root, None, index_blocks);
            let pool: Vec<String> = (0..3u64)
                .map(|index| {
                    let flavor = [Flavor::Ascii, Flavor::Multibyte, Flavor::Nfc][index as usize];
                    prose(seed * 10 + index, 24_000, flavor)
                })
                .collect();
            let mut rng = Rng(seed);
            let check = |step: usize, what: &str| {
                let stats = tokenizer.store_stats();
                let state = tokenizer.inner.store.lock().unwrap();
                assert_eq!(
                    stats.index_bytes,
                    state.index.approx_bytes_by_walk(),
                    "seed {seed}, step {step} ({what}): index bytes and a recount"
                );
                state.index.check_invariants().unwrap();
            };
            for step in 0..240 {
                let name = format!("t{}", rng.below(9));
                let base = &pool[rng.below(pool.len())];
                let what = match rng.below(9) {
                    0 | 1 => {
                        let text = take_chars(base, 4_000 + rng.below(14_000));
                        let mut session = tokenizer.open_session(&name).unwrap();
                        if session.revision().is_none() {
                            session.seed(&text).unwrap();
                        } else {
                            session.overwrite(&text).unwrap();
                        }
                        session.close().unwrap();
                        "seed or overwrite"
                    }
                    2..=4 => {
                        let mut session = tokenizer.open_session(&name).unwrap();
                        let delta = prose(rng.next(), 300 + rng.below(5_000), Flavor::Ascii);
                        if session.revision().is_none() {
                            session.seed(&delta).unwrap();
                        } else {
                            session.append(&delta).unwrap();
                        }
                        session.close().unwrap();
                        "append"
                    }
                    5 => {
                        let target = format!("t{}", rng.below(9));
                        let free = target != name
                            && tokenizer
                                .open_session(&target)
                                .unwrap()
                                .revision()
                                .is_none();
                        let session = tokenizer.open_session(&name).unwrap();
                        if session.revision().is_some() && free {
                            session.fork(&target).unwrap().close().unwrap();
                        }
                        session.close().unwrap();
                        "fork"
                    }
                    6 => {
                        let session = tokenizer.open_session(&name).unwrap();
                        if session.revision().is_some() {
                            session.delete().unwrap();
                        } else {
                            session.close().unwrap();
                        }
                        "delete"
                    }
                    7 => {
                        // A listed holder dropped as a reject.
                        let mut state = tokenizer.inner.store.lock().unwrap();
                        if let Some(&handle) = state.names.get(&name) {
                            let text = take_chars(base, 4_096);
                            let key = crate::prefix_index::Walker::new(state.index.params(), &text)
                                .next_key()
                                .map(|(key, _)| key);
                            if let Some(key) = key {
                                state.index.reject_holder(&key, handle);
                            }
                        }
                        "reject"
                    }
                    _ => {
                        let query = format!(
                            "{}{}",
                            take_chars(base, rng.below(24_000)),
                            prose(rng.next(), 500, Flavor::Nfc)
                        );
                        lookup_checked(&tokenizer, &query);
                        "lookup"
                    }
                };
                check(step, what);
            }
            let stats = tokenizer.store_stats();
            assert!(
                stats.sessions_evicted > 0,
                "seed {seed}: the core dropped sessions"
            );
            if index_blocks == Some(3) {
                assert!(
                    stats.index_blocks_declined > 0,
                    "seed {seed}: blocks declined"
                );
            }
            let names: Vec<String> = tokenizer
                .inner
                .store
                .lock()
                .unwrap()
                .names
                .keys()
                .cloned()
                .collect();
            for name in names {
                let session = tokenizer.open_session(&name).unwrap();
                if session.revision().is_some() {
                    session.delete().unwrap();
                }
            }
            check(usize::MAX, "after deleting everything");
            assert_eq!(tokenizer.store_stats().node_count, 0);
        }
    }

    // END session memory tests.
}
