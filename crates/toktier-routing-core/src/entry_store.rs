//! Native facade entry store: naming, content lookup, persistence, and LRU.

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine as _;
use serde::{Deserialize, Serialize};
use toktier_store_core::{
    ContentDigest, ContentIndexEntry, KeyId, RecoveryBindingV1, SessionHandle, SessionRecordV1,
    SessionStore, StoreConfig, StoreError,
};

use crate::NativeRouter;

pub const AUTO_MIN_BYTES: usize = 4096;
pub const DEFAULT_CACHE_BUDGET_BYTES: usize = 128 * 1024 * 1024;
pub const MAX_AUTO_ENTRIES: usize = 1024;
const NATIVE_MAX_SESSIONS: usize = 4096;
const META_NAME: &str = "meta.json";
const INDEX_NAME: &str = "index.json";
const ENTRIES_DIR: &str = "entries";
const META_FORMAT: u64 = 1;
const RECOVERY_SUFFIX: &str = ".binding";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EntryKind {
    Session,
    Auto,
}

#[derive(Debug, Clone)]
struct Entry {
    kind: EntryKind,
    byte_length: u64,
    text: Option<String>,
    handle: Option<SessionHandle>,
    revision: u64,
    index: ContentIndexEntry,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EntryStoreStats {
    pub session_hits: u64,
    pub session_appends: u64,
    pub session_overwrites: u64,
    pub session_misses: u64,
    pub auto_hits: u64,
    pub auto_appends: u64,
    pub auto_misses: u64,
    pub collision_rejects: u64,
    pub degraded: u64,
    pub index_rebuilds: u64,
    pub entries_evicted: u64,
    pub extra: BTreeMap<String, u64>,
}

#[derive(Debug)]
pub enum EntryStoreOpenError {
    StateMismatch(String),
    Io(std::io::Error),
    Store(StoreError),
}

impl std::fmt::Display for EntryStoreOpenError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::StateMismatch(message) => formatter.write_str(message),
            Self::Io(error) => error.fmt(formatter),
            Self::Store(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for EntryStoreOpenError {}

impl From<std::io::Error> for EntryStoreOpenError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<StoreError> for EntryStoreOpenError {
    fn from(error: StoreError) -> Self {
        Self::Store(error)
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct MetaPayload {
    format: u64,
    fingerprint: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct IndexPayload {
    format: u64,
    entries: BTreeMap<String, IndexRow>,
}

#[derive(Debug, Serialize, Deserialize)]
struct IndexRow {
    bytes: u64,
    end: String,
    marks: Vec<(u64, String)>,
}

impl IndexRow {
    fn from_entry(entry: &ContentIndexEntry) -> Self {
        Self {
            bytes: entry.byte_length,
            end: hex(&entry.end_digest),
            marks: entry
                .marks
                .iter()
                .map(|(position, digest)| (*position, hex(digest)))
                .collect(),
        }
    }

    fn into_entry(self) -> Option<ContentIndexEntry> {
        Some(ContentIndexEntry {
            byte_length: self.bytes,
            end_digest: unhex16(&self.end)?,
            marks: self
                .marks
                .into_iter()
                .map(|(position, digest)| Some((position, unhex16(&digest)?)))
                .collect::<Option<Vec<_>>>()?,
        })
    }
}

pub struct NativeEntryStore {
    fingerprint: [u8; 32],
    router: Arc<NativeRouter>,
    directory: Option<PathBuf>,
    budget: usize,
    entries: HashMap<String, Entry>,
    lru: VecDeque<String>,
    index: BTreeMap<String, ContentIndexEntry>,
    stats: EntryStoreStats,
    store: SessionStore,
    key_id: KeyId,
}

impl std::fmt::Debug for NativeEntryStore {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("NativeEntryStore")
            .field("directory", &self.directory)
            .field("entries", &self.entries.len())
            .field("budget", &self.budget)
            .finish_non_exhaustive()
    }
}

impl NativeEntryStore {
    pub fn open(
        fingerprint: [u8; 32],
        router: Arc<NativeRouter>,
        directory: Option<PathBuf>,
        cache_budget_bytes: usize,
        seal_end_guard_chars: u64,
    ) -> Result<Self, EntryStoreOpenError> {
        let mut store = SessionStore::new(StoreConfig {
            max_sessions: NATIVE_MAX_SESSIONS,
            ..StoreConfig::default()
        })?;
        if directory.is_some() {
            store.enable_recovery_tracking()?;
        }
        store.enable_content_tracking()?;
        let key_id = store.register_fingerprint(fingerprint, seal_end_guard_chars)?;
        let mut result = Self {
            fingerprint,
            router,
            directory,
            budget: cache_budget_bytes,
            entries: HashMap::new(),
            lru: VecDeque::new(),
            index: BTreeMap::new(),
            stats: EntryStoreStats::default(),
            store,
            key_id,
        };
        if let Some(directory) = result.directory.clone() {
            result.open_directory(&directory)?;
        }
        Ok(result)
    }

    pub fn stats(&self) -> &EntryStoreStats {
        &self.stats
    }

    pub fn native_stats(&self) -> toktier_store_core::StatsSnapshot {
        self.store.stats()
    }

    pub fn entries_len(&self) -> usize {
        self.entries.len()
    }

    pub fn resident_bytes(&self) -> u64 {
        self.entries
            .values()
            .filter(|entry| entry.text.is_some())
            .map(|entry| entry.byte_length)
            .sum()
    }

    fn open_directory(&mut self, directory: &Path) -> Result<(), EntryStoreOpenError> {
        ensure_private_dir(directory)?;
        let meta_path = directory.join(META_NAME);
        let fresh = !meta_path.is_file();
        if fresh {
            let payload = serde_json::to_vec(&MetaPayload {
                format: META_FORMAT,
                fingerprint: hex(&self.fingerprint),
            })
            .map_err(|error| EntryStoreOpenError::StateMismatch(error.to_string()))?;
            atomic_write(&meta_path, &payload)?;
        } else {
            let payload: MetaPayload =
                serde_json::from_slice(&fs::read(&meta_path)?).map_err(|error| {
                    EntryStoreOpenError::StateMismatch(format!(
                        "the store's metadata file cannot be read: {error}"
                    ))
                })?;
            if payload.format != META_FORMAT {
                return Err(EntryStoreOpenError::StateMismatch(format!(
                    "the store metadata names unsupported format {}",
                    payload.format
                )));
            }
            if payload.fingerprint != hex(&self.fingerprint) {
                return Err(EntryStoreOpenError::StateMismatch(
                    "the store was written under a different semantic fingerprint".to_owned(),
                ));
            }
        }
        ensure_private_dir(&directory.join(ENTRIES_DIR))?;
        self.load_or_rebuild_index(fresh)?;
        Ok(())
    }

    fn load_or_rebuild_index(&mut self, fresh: bool) -> Result<(), EntryStoreOpenError> {
        let directory = self.directory.as_ref().expect("persistent store");
        let index_path = directory.join(INDEX_NAME);
        let loaded = fs::read(&index_path)
            .ok()
            .and_then(|raw| serde_json::from_slice::<IndexPayload>(&raw).ok())
            .filter(|payload| payload.format == META_FORMAT)
            .and_then(|payload| {
                payload
                    .entries
                    .into_iter()
                    .map(|(name, row)| Some((name, row.into_entry()?)))
                    .collect::<Option<BTreeMap<_, _>>>()
            });
        let loaded_from_disk = loaded.is_some();
        self.index = match loaded {
            Some(index) => index,
            None => {
                let rebuilt = self.rebuild_index()?;
                if !fresh {
                    self.stats.index_rebuilds += 1;
                }
                rebuilt
            }
        };
        for (name, row) in &self.index {
            let kind = if name.starts_with("s-") {
                EntryKind::Session
            } else {
                EntryKind::Auto
            };
            self.entries.insert(
                name.clone(),
                Entry {
                    kind,
                    byte_length: row.byte_length,
                    text: None,
                    handle: None,
                    revision: 0,
                    index: row.clone(),
                },
            );
            self.lru.push_back(name.clone());
        }
        if !loaded_from_disk {
            self.persist_index();
        }
        Ok(())
    }

    fn rebuild_index(&self) -> Result<BTreeMap<String, ContentIndexEntry>, EntryStoreOpenError> {
        let mut index = BTreeMap::new();
        let entries_dir = self
            .directory
            .as_ref()
            .expect("persistent store")
            .join(ENTRIES_DIR);
        let mut paths = fs::read_dir(entries_dir)?
            .filter_map(Result::ok)
            .map(|item| item.path())
            .filter(|path| path.extension().and_then(|value| value.to_str()) == Some("rec"))
            .collect::<Vec<_>>();
        paths.sort();
        for path in paths {
            let Ok(record) = fs::read(&path)
                .map_err(io_store)
                .and_then(|raw| SessionRecordV1::from_bytes(&raw))
            else {
                continue;
            };
            let Some(name) = path.file_stem().and_then(|value| value.to_str()) else {
                continue;
            };
            let row = if record.stable_prefix_bytes == 0 {
                ContentDigest::from_bytes(record.tail_text.as_bytes())?.entry()
            } else {
                let binding_path = path.with_file_name(format!("{name}{RECOVERY_SUFFIX}"));
                let Ok(binding) = fs::read(binding_path)
                    .map_err(io_store)
                    .and_then(|raw| RecoveryBindingV1::from_bytes(&raw))
                else {
                    continue;
                };
                let full_bytes = record
                    .stable_prefix_bytes
                    .checked_add(record.tail_text.len() as u64);
                if binding.record_hash != record.curr_block_hash
                    || Some(binding.index_entry.byte_length) != full_bytes
                {
                    continue;
                }
                binding.index_entry
            };
            index.insert(name.to_owned(), row);
        }
        Ok(index)
    }

    fn record_path(&self, name: &str) -> PathBuf {
        self.directory
            .as_ref()
            .expect("persistent store")
            .join(ENTRIES_DIR)
            .join(format!("{name}.rec"))
    }

    fn recovery_path(&self, name: &str) -> PathBuf {
        self.directory
            .as_ref()
            .expect("persistent store")
            .join(ENTRIES_DIR)
            .join(format!("{name}{RECOVERY_SUFFIX}"))
    }

    fn count_extra(&mut self, name: &str) {
        *self.stats.extra.entry(name.to_owned()).or_default() += 1;
    }

    fn persist_index(&mut self) {
        let Some(directory) = &self.directory else {
            return;
        };
        let payload = IndexPayload {
            format: META_FORMAT,
            entries: self
                .index
                .iter()
                .map(|(name, entry)| (name.clone(), IndexRow::from_entry(entry)))
                .collect(),
        };
        let result = serde_json::to_vec(&payload)
            .map_err(|error| std::io::Error::other(error.to_string()))
            .and_then(|raw| atomic_write(&directory.join(INDEX_NAME), &raw));
        if result.is_err() {
            self.count_extra("persist_failures");
        }
    }

    fn persist_entry(&mut self, name: &str) {
        if self.directory.is_none() {
            return;
        }
        let Some(handle) = self.entries.get(name).and_then(|entry| entry.handle) else {
            return;
        };
        let result = (|| -> Result<(), Box<dyn std::error::Error>> {
            let record = self.store.export_session(handle)?;
            let binding = self
                .store
                .export_recovery_binding(handle)?
                .ok_or("native session has no recovery binding")?;
            atomic_write(&self.record_path(name), &record)?;
            atomic_write(&self.recovery_path(name), &binding)?;
            Ok(())
        })();
        if result.is_err() {
            self.count_extra("persist_failures");
        }
    }

    fn resident(&mut self, name: &str, candidate: Option<&str>) -> bool {
        if self
            .entries
            .get(name)
            .is_some_and(|entry| entry.text.is_some() && entry.handle.is_some())
        {
            return true;
        }
        if self.directory.is_none() {
            return false;
        }
        let result = (|| -> Result<(SessionHandle, String, u64, ContentIndexEntry), StoreError> {
            let raw = fs::read(self.record_path(name)).map_err(io_store)?;
            let view = SessionRecordV1::from_bytes(&raw)?;
            let (handle, historical) = if view.stable_prefix_bytes == 0 {
                let handle = self
                    .store
                    .import_session(self.key_id, &raw, &*self.router)?;
                (handle, view.tail_text.clone())
            } else {
                let candidate = candidate.ok_or_else(|| {
                    StoreError::InvalidInput(
                        "candidate text is required for sealed recovery".into(),
                    )
                })?;
                let binding = fs::read(self.recovery_path(name)).map_err(io_store)?;
                let (handle, historical_chars) = self.store.import_session_with_binding_candidate(
                    self.key_id,
                    &raw,
                    candidate,
                    &binding,
                    &*self.router,
                )?;
                let end = byte_offset_at_char(candidate, historical_chars);
                (handle, candidate[..end].to_owned())
            };
            let row = self
                .store
                .content_index_entry(handle)?
                .ok_or_else(|| StoreError::Internal("import lost content index".into()))?;
            Ok((handle, historical, view.revision, row))
        })();
        let Ok((handle, historical, revision, row)) = result else {
            self.drop_entry(name, false);
            return false;
        };
        if let Some(entry) = self.entries.get_mut(name) {
            entry.handle = Some(handle);
            entry.byte_length = row.byte_length;
            entry.text = Some(historical);
            entry.revision = revision;
            entry.index = row.clone();
            self.index.insert(name.to_owned(), row);
        }
        // Keep the recovered row pinned until its caller has read or mutated
        // the native session.  Evicting it here under a zero-byte budget
        // would return `true` while immediately clearing the very handle and
        // text that `encode_session`/`serve_from_index` are about to use.
        true
    }

    fn touch(&mut self, name: &str) {
        if let Some(position) = self.lru.iter().position(|value| value == name) {
            self.lru.remove(position);
        }
        self.lru.push_back(name.to_owned());
    }

    fn evict_over_budget(&mut self, pin: &str) {
        while self.resident_bytes() > self.budget as u64 {
            // Non-resident persistent rows remain in the LRU so they can be
            // recovered later, but they cannot reduce the resident budget.
            // Skipping them by repeatedly moving them to the back can spin
            // forever when the only resident row is the freshly written pin
            // (most visibly with a zero-byte budget).  Evict an actual
            // resident victim, then evict the pin as the final choice after
            // its IDs have already been materialized for the caller.
            let victim = self
                .lru
                .iter()
                .find(|name| {
                    name.as_str() != pin
                        && self
                            .entries
                            .get(*name)
                            .is_some_and(|entry| entry.text.is_some())
                })
                .cloned()
                .or_else(|| {
                    self.entries
                        .get(pin)
                        .is_some_and(|entry| entry.text.is_some())
                        .then(|| pin.to_owned())
                });
            let Some(name) = victim else { break };
            self.evict_entry(&name);
        }
    }

    fn evict_entry(&mut self, name: &str) {
        self.stats.entries_evicted += 1;
        let handle = self.entries.get(name).and_then(|entry| entry.handle);
        if let Some(handle) = handle {
            self.store.evict(handle);
        }
        if self.directory.is_none() {
            self.entries.remove(name);
            self.index.remove(name);
            self.lru.retain(|value| value != name);
        } else if let Some(entry) = self.entries.get_mut(name) {
            entry.text = None;
            entry.handle = None;
        }
    }

    fn drop_entry(&mut self, name: &str, remove_file: bool) {
        if let Some(entry) = self.entries.remove(name) {
            if let Some(handle) = entry.handle {
                self.store.evict(handle);
            }
        }
        self.index.remove(name);
        self.lru.retain(|value| value != name);
        if remove_file && self.directory.is_some() {
            let _ = fs::remove_file(self.record_path(name));
            let _ = fs::remove_file(self.recovery_path(name));
        }
    }

    fn cap_auto_entries(&mut self) {
        while self
            .entries
            .values()
            .filter(|entry| entry.kind == EntryKind::Auto)
            .count()
            > MAX_AUTO_ENTRIES
        {
            let Some(name) = self.lru.iter().find_map(|name| {
                self.entries
                    .get(name)
                    .is_some_and(|entry| entry.kind == EntryKind::Auto)
                    .then(|| name.clone())
            }) else {
                break;
            };
            self.drop_entry(&name, true);
        }
    }

    fn write_entry(&mut self, name: String, kind: EntryKind, text: &str) -> Option<Vec<u32>> {
        if let Some(handle) = self.entries.get(&name).and_then(|entry| entry.handle) {
            self.store.evict(handle);
        }
        let put = self.store.put(self.key_id, text, &*self.router);
        let Ok(put) = put else {
            self.drop_entry(&name, false);
            self.stats.degraded += 1;
            return None;
        };
        let row = match self.store.content_index_entry(put.handle) {
            Ok(Some(row)) => row,
            _ => {
                self.store.evict(put.handle);
                self.stats.degraded += 1;
                return None;
            }
        };
        let ids = match self.store.all_ids(put.handle) {
            Ok(ids) => ids,
            Err(_) => {
                self.store.evict(put.handle);
                self.stats.degraded += 1;
                return None;
            }
        };
        let entry = Entry {
            kind,
            byte_length: row.byte_length,
            text: Some(text.to_owned()),
            handle: Some(put.handle),
            revision: put.revision,
            index: row.clone(),
        };
        self.entries.insert(name.clone(), entry);
        self.index.insert(name.clone(), row);
        self.touch(&name);
        self.cap_auto_entries();
        self.persist_entry(&name);
        self.persist_index();
        self.evict_over_budget(&name);
        Some(ids)
    }

    fn append_entry(&mut self, name: &str, delta: &str) -> Option<Vec<u32>> {
        let (handle, revision) = self
            .entries
            .get(name)
            .and_then(|entry| Some((entry.handle?, entry.revision)))?;
        let outcome = match self.store.append(handle, delta, revision, &*self.router) {
            Ok(outcome) => outcome,
            Err(_) => {
                self.drop_entry(name, false);
                self.stats.degraded += 1;
                return None;
            }
        };
        let row = match self.store.content_index_entry(handle) {
            Ok(Some(row)) => row,
            _ => {
                self.drop_entry(name, false);
                self.stats.degraded += 1;
                return None;
            }
        };
        if let Some(entry) = self.entries.get_mut(name) {
            entry.text.as_mut()?.push_str(delta);
            entry.byte_length = row.byte_length;
            entry.revision = outcome.revision;
            entry.index = row.clone();
        }
        self.index.insert(name.to_owned(), row);
        self.persist_entry(name);
        self.persist_index();
        self.evict_over_budget(name);
        Some(outcome.all_ids)
    }

    pub fn encode_session(&mut self, session_id: &str, text: &str) -> Option<Vec<u32>> {
        let name = entry_name(EntryKind::Session, session_id);
        if self.entries.contains_key(&name) && !self.resident(&name, Some(text)) {
            self.stats.session_overwrites += 1;
            return self.write_entry(name, EntryKind::Session, text);
        }
        if !self.entries.contains_key(&name) {
            self.stats.session_misses += 1;
            return self.write_entry(name, EntryKind::Session, text);
        }
        self.touch(&name);
        let stored = self.entries.get(&name)?.text.as_ref()?;
        if stored == text {
            self.stats.session_hits += 1;
            let ids = self.entry_ids(&name);
            self.evict_over_budget(&name);
            return ids;
        }
        if text.starts_with(stored) {
            let delta = text[stored.len()..].to_owned();
            let ids = self.append_entry(&name, &delta);
            if ids.is_some() {
                self.stats.session_appends += 1;
            }
            return ids;
        }
        self.stats.session_overwrites += 1;
        self.write_entry(name, EntryKind::Session, text)
    }

    pub fn encode_auto(&mut self, text: &str) -> Option<Vec<u32>> {
        if let Some(ids) = self.serve_from_index(text) {
            return Some(ids);
        }
        if text.len() < AUTO_MIN_BYTES {
            return None;
        }
        self.stats.auto_misses += 1;
        let put = match self.store.put(self.key_id, text, &*self.router) {
            Ok(put) => put,
            Err(_) => {
                self.stats.degraded += 1;
                return None;
            }
        };
        let row = match self.store.content_index_entry(put.handle) {
            Ok(Some(row)) => row,
            _ => {
                self.store.evict(put.handle);
                self.stats.degraded += 1;
                return None;
            }
        };
        let name = entry_name(EntryKind::Auto, &hex(&row.end_digest));
        if self.entries.contains_key(&name) {
            self.drop_entry(&name, false);
        }
        let ids = match self.store.all_ids(put.handle) {
            Ok(ids) => ids,
            Err(_) => {
                self.store.evict(put.handle);
                self.stats.degraded += 1;
                return None;
            }
        };
        self.entries.insert(
            name.clone(),
            Entry {
                kind: EntryKind::Auto,
                byte_length: row.byte_length,
                text: Some(text.to_owned()),
                handle: Some(put.handle),
                revision: put.revision,
                index: row.clone(),
            },
        );
        self.index.insert(name.clone(), row);
        self.touch(&name);
        self.cap_auto_entries();
        self.persist_entry(&name);
        self.persist_index();
        self.evict_over_budget(&name);
        Some(ids)
    }

    fn serve_from_index(&mut self, text: &str) -> Option<Vec<u32>> {
        let digests = self.query_digests(text.as_bytes());
        let mut candidates = self
            .index
            .iter()
            .filter(|(_name, row)| {
                row.byte_length > 0 && digests.get(&row.byte_length) == Some(&row.end_digest)
            })
            .map(|(name, row)| (row.byte_length, name.clone()))
            .collect::<Vec<_>>();
        candidates.sort_by(|left, right| right.0.cmp(&left.0).then(left.1.cmp(&right.1)));
        for (_length, name) in candidates {
            if !self.entries.contains_key(&name) {
                self.index.remove(&name);
                continue;
            }
            if !self.resident(&name, Some(text)) {
                continue;
            }
            let stored = self.entries.get(&name)?.text.as_ref()?;
            if !text.starts_with(stored) {
                self.stats.collision_rejects += 1;
                continue;
            }
            self.touch(&name);
            let stored_len = self.entries.get(&name)?.text.as_ref()?.len();
            if stored_len == text.len() {
                self.stats.auto_hits += 1;
                let ids = self.entry_ids(&name);
                self.evict_over_budget(&name);
                return ids;
            }
            return self.serve_extension(&name, text, stored_len);
        }
        None
    }

    fn serve_extension(&mut self, name: &str, text: &str, stored_len: usize) -> Option<Vec<u32>> {
        let delta = &text[stored_len..];
        if self.entries.get(name)?.kind == EntryKind::Auto {
            let ids = self.append_entry(name, delta);
            if ids.is_some() {
                self.stats.auto_appends += 1;
            }
            return ids;
        }
        let handle = self.entries.get(name)?.handle?;
        let fork = match self.store.fork(handle) {
            Ok(fork) => fork,
            Err(_) => {
                self.stats.degraded += 1;
                return None;
            }
        };
        let outcome = match self.store.append(fork, delta, 0, &*self.router) {
            Ok(outcome) => outcome,
            Err(_) => {
                self.store.evict(fork);
                self.stats.degraded += 1;
                return None;
            }
        };
        self.stats.auto_appends += 1;
        if text.len() >= AUTO_MIN_BYTES {
            self.adopt_fork(fork, text, outcome.revision);
        } else {
            self.store.evict(fork);
        }
        Some(outcome.all_ids)
    }

    fn adopt_fork(&mut self, fork: SessionHandle, text: &str, revision: u64) {
        let Ok(Some(row)) = self.store.content_index_entry(fork) else {
            self.store.evict(fork);
            self.stats.degraded += 1;
            return;
        };
        let name = entry_name(EntryKind::Auto, &hex(&row.end_digest));
        if self.entries.contains_key(&name) {
            self.drop_entry(&name, false);
        }
        self.entries.insert(
            name.clone(),
            Entry {
                kind: EntryKind::Auto,
                byte_length: row.byte_length,
                text: Some(text.to_owned()),
                handle: Some(fork),
                revision,
                index: row.clone(),
            },
        );
        self.index.insert(name.clone(), row);
        self.touch(&name);
        self.cap_auto_entries();
        self.persist_entry(&name);
        self.persist_index();
        self.evict_over_budget(&name);
    }

    fn entry_ids(&mut self, name: &str) -> Option<Vec<u32>> {
        let handle = self.entries.get(name)?.handle?;
        match self.store.all_ids(handle) {
            Ok(ids) => Some(ids),
            Err(_) => {
                self.drop_entry(name, false);
                self.stats.degraded += 1;
                None
            }
        }
    }

    fn query_digests(&self, data: &[u8]) -> BTreeMap<u64, [u8; 16]> {
        let lengths = self
            .index
            .values()
            .flat_map(|entry| {
                std::iter::once(entry.byte_length)
                    .chain(entry.marks.iter().map(|(position, _digest)| *position))
            })
            .filter(|length| *length > 0 && *length <= data.len() as u64)
            .collect::<BTreeSet<_>>();
        let mut result = BTreeMap::new();
        let mut state = ContentDigest::empty();
        let mut consumed = 0usize;
        for length in lengths {
            let end = length as usize;
            if state.append(&data[consumed..end]).is_err() {
                return BTreeMap::new();
            }
            consumed = end;
            result.insert(length, state.entry().end_digest);
        }
        result
    }
}

fn entry_name(kind: EntryKind, token: &str) -> String {
    match kind {
        EntryKind::Session => format!("s-{}", URL_SAFE_NO_PAD.encode(token.as_bytes())),
        EntryKind::Auto => format!("a-{token}"),
    }
}

fn byte_offset_at_char(value: &str, chars: usize) -> usize {
    if chars == 0 {
        return 0;
    }
    value
        .char_indices()
        .nth(chars)
        .map_or(value.len(), |(offset, _)| offset)
}

fn ensure_private_dir(path: &Path) -> std::io::Result<()> {
    fs::create_dir_all(path)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(path, fs::Permissions::from_mode(0o700))?;
    }
    Ok(())
}

fn atomic_write(path: &Path, data: &[u8]) -> std::io::Result<()> {
    let temporary = path.with_file_name(format!(
        ".{}.{}.tmp",
        path.file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("state"),
        std::process::id()
    ));
    let mut options = fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let result = (|| {
        let mut file = options.open(&temporary)?;
        file.write_all(data)?;
        fs::rename(&temporary, path)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

fn hex(raw: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(raw.len() * 2);
    for byte in raw {
        output.push(DIGITS[(byte >> 4) as usize] as char);
        output.push(DIGITS[(byte & 0x0f) as usize] as char);
    }
    output
}

fn unhex16(value: &str) -> Option<[u8; 16]> {
    if value.len() != 32 {
        return None;
    }
    let mut output = [0u8; 16];
    for (index, slot) in output.iter_mut().enumerate() {
        let start = index * 2;
        *slot = u8::from_str_radix(&value[start..start + 2], 16).ok()?;
    }
    Some(output)
}

fn io_store(error: std::io::Error) -> StoreError {
    StoreError::Internal(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ReferenceEngine, BACKEND_REFERENCE};
    use tokenizers::models::bpe::BPE;
    use tokenizers::Tokenizer;

    fn router() -> Arc<NativeRouter> {
        let model = BPE::builder()
            .vocab_and_merges([("a".to_owned(), 0)], Vec::new())
            .build()
            .unwrap();
        let tokenizer = Tokenizer::new(model);
        let reference = Arc::new(
            ReferenceEngine::from_bytes(tokenizer.to_string(false).unwrap().as_bytes()).unwrap(),
        );
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

    #[test]
    fn zero_budget_recovers_appends_then_evicts_without_spinning() {
        let directory = tempfile::tempdir().unwrap();
        let mut store =
            NativeEntryStore::open([7; 32], router(), Some(directory.path().to_owned()), 0, 0)
                .unwrap();
        assert_eq!(store.encode_session("chat", "aaaa"), Some(vec![0; 4]));
        assert_eq!(store.resident_bytes(), 0);
        assert_eq!(store.encode_session("chat", "aaaaaa"), Some(vec![0; 6]));
        assert_eq!(store.stats().session_appends, 1);
        assert_eq!(store.resident_bytes(), 0);
        assert!(store.stats().entries_evicted >= 2);

        drop(store);
        let mut reopened =
            NativeEntryStore::open([7; 32], router(), Some(directory.path().to_owned()), 0, 0)
                .unwrap();
        assert_eq!(reopened.encode_session("chat", "aaaaaa"), Some(vec![0; 6]));
        assert_eq!(reopened.stats().session_hits, 1);
        assert_eq!(reopened.resident_bytes(), 0);
    }

    #[test]
    fn content_prefix_extension_stays_inside_the_native_store() {
        let mut store = NativeEntryStore::open([9; 32], router(), None, 1 << 20, 0).unwrap();
        let base = "a".repeat(AUTO_MIN_BYTES + 17);
        let grown = format!("{base}aaaa");
        assert_eq!(store.encode_auto(&base).unwrap().len(), AUTO_MIN_BYTES + 17);
        assert_eq!(
            store.encode_auto(&grown).unwrap().len(),
            AUTO_MIN_BYTES + 21
        );
        assert_eq!(store.stats().auto_misses, 1);
        assert_eq!(store.stats().auto_appends, 1);
    }
}
