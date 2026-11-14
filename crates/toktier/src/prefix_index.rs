//! The index `Tokenizer::lookup` reads: one entry per completed block of
//! the text of every live session.
//!
//! A block is `block_chars` characters (4,096 by default). The key of
//! block `k` of a text is a chained SHA-256 over the key of block `k - 1`,
//! the block's index, the store's semantic fingerprint and the block's
//! bytes, in the layout of the store core's node keys under a domain of
//! its own. A query whose first `k` blocks hash to a key therefore has the
//! same first `k` blocks as every session that registered that key.
//!
//! An entry holds how many live sessions have that block prefix (`refs`)
//! and up to [`MAX_LISTED_HOLDERS`] of them with a seal point: a place
//! where the session's sealed token prefix ends. The seal a block carries
//! is the one its session had *before* the call that completed the block,
//! so it was certified on a text no longer than the block's end, and it
//! holds for every text with that block prefix: the ids of such a text are
//! the session's sealed ids up to the seal followed by the encoding of the
//! text from the seal on. The index copies no text and no ids; whoever
//! uses a holder checks it against the store first.
//!
//! Entries are released with the last session that has them, so the
//! index follows the live sessions. It holds at most `max_blocks` entries:
//! a session whose next block would need a new entry beyond that stops
//! registering blocks for the rest of its life (its handle's life).
//!
//! This module calls neither the store nor an engine; it is plain
//! bookkeeping, kept consistent by its owner.

use std::collections::HashMap;

use sha2::{Digest, Sha256};
use toktier_store_core::SessionHandle;

/// Domain of the block keys. The keys stay in this process: they are
/// never stored, and never compared with the store core's own records.
const DOMAIN: &[u8] = b"toktier.rust.prefix-index.v1\0";

/// Seal holders listed per entry. One usable holder serves a hit; the
/// others stand in when it goes. A small fixed number keeps every
/// per-block step constant-time when many sessions share a prefix.
pub(crate) const MAX_LISTED_HOLDERS: usize = 4;

/// Below this many slots a map is not shrunk after removals.
const SHRINK_FLOOR: usize = 1024;

pub(crate) type BlockKey = [u8; 32];

/// What the block keys are computed from. Immutable for a store, so a
/// query's keys can be computed without the store lock.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct KeyParams {
    pub(crate) fingerprint: [u8; 32],
    pub(crate) block_chars: u64,
}

impl KeyParams {
    /// Hash state for block `index` whose predecessor has key `parent`,
    /// before the block's bytes.
    fn block_hasher(&self, parent: Option<&BlockKey>, index: u64) -> Sha256 {
        let mut hasher = Sha256::new();
        hasher.update(DOMAIN);
        match parent {
            Some(parent) => {
                hasher.update([1u8]);
                hasher.update(parent);
            }
            None => {
                hasher.update([0u8]);
                hasher.update([0u8; 32]);
            }
        }
        hasher.update(index.to_le_bytes());
        hasher.update(self.fingerprint);
        hasher
    }

    fn block_chars_usize(&self) -> usize {
        usize::try_from(self.block_chars).unwrap_or(usize::MAX)
    }
}

/// The byte offset after the first `limit` characters of `text` (the
/// whole text when it is shorter), and how many characters that is.
fn split_after_chars(text: &str, limit: usize) -> (usize, usize) {
    let mut taken = 0usize;
    for (offset, _) in text.char_indices() {
        if taken == limit {
            return (offset, taken);
        }
        taken += 1;
    }
    (text.len(), taken)
}

/// A seal point of one session: its first `ids` sealed tokens cover its
/// first `bytes` UTF-8 bytes, which are `chars` characters. The zero seal
/// means nothing is sealed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct Seal {
    pub(crate) chars: u64,
    pub(crate) bytes: u64,
    pub(crate) ids: u64,
}

impl Seal {
    pub(crate) fn is_zero(&self) -> bool {
        self.ids == 0
    }
}

/// A live session and the seal point it offers for one block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Holder {
    pub(crate) handle: SessionHandle,
    pub(crate) seal: Seal,
}

#[derive(Debug)]
struct Entry {
    /// Live sessions whose text has this block prefix.
    refs: u32,
    /// At most `MAX_LISTED_HOLDERS` of them, each with a non-zero seal.
    holders: Vec<Holder>,
}

/// The incremental hashing state of a text fed in pieces.
#[derive(Debug, Clone)]
struct Rolling {
    /// Hash state of the block in progress (its prefix part included).
    partial: Sha256,
    /// Characters of the block in progress fed so far.
    partial_chars: u64,
    /// Blocks completed so far.
    depth: u64,
}

impl Rolling {
    fn new(params: &KeyParams) -> Self {
        Self {
            partial: params.block_hasher(None, 0),
            partial_chars: 0,
            depth: 0,
        }
    }

    /// Feed `text`; push the key of every block it completes.
    fn feed(&mut self, params: &KeyParams, text: &str, completed: &mut Vec<BlockKey>) {
        let block_chars = params.block_chars_usize();
        let mut rest = text;
        while !rest.is_empty() {
            let need = block_chars - self.partial_chars as usize;
            let (end, taken) = split_after_chars(rest, need);
            self.partial.update(&rest.as_bytes()[..end]);
            self.partial_chars += taken as u64;
            rest = &rest[end..];
            if self.partial_chars == params.block_chars {
                self.depth += 1;
                let finished = std::mem::replace(&mut self.partial, Sha256::new());
                let key: BlockKey = finished.finalize().into();
                self.partial = params.block_hasher(Some(&key), self.depth);
                self.partial_chars = 0;
                completed.push(key);
            }
        }
    }
}

/// What the index keeps for one live session.
#[derive(Debug, Clone)]
struct Track {
    /// The key of every block registered for this session, in order.
    keys: Vec<BlockKey>,
    rolling: Rolling,
    /// UTF-8 bytes of the session's text fed so far.
    bytes: u64,
    /// The session's seal after its latest call; the blocks later calls
    /// complete carry it.
    seal: Seal,
    /// False once a block was declined at the cap: the track then
    /// registers nothing more.
    attached: bool,
}

/// Counters of the index and of the lookups it served.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct IndexCounters {
    pub(crate) lookups: u64,
    pub(crate) hits: u64,
    pub(crate) misses: u64,
    /// Holders that did not agree with the store when a lookup checked
    /// them; each was dropped from its entry.
    pub(crate) holder_rejects: u64,
    /// Blocks not registered because the index was at its cap.
    pub(crate) blocks_declined: u64,
    /// Token ids lookups took from sessions instead of encoding them.
    pub(crate) reused_ids: u64,
}

#[derive(Debug)]
pub(crate) struct PrefixIndex {
    params: KeyParams,
    max_blocks: usize,
    entries: HashMap<BlockKey, Entry>,
    tracks: HashMap<SessionHandle, Track>,
    pub(crate) counters: IndexCounters,
}

impl PrefixIndex {
    /// An empty index holding at most `max_blocks` entries; 0 keeps no
    /// entries and no tracks at all.
    pub(crate) fn new(params: KeyParams, max_blocks: usize) -> Self {
        Self {
            params,
            max_blocks,
            entries: HashMap::new(),
            tracks: HashMap::new(),
            counters: IndexCounters::default(),
        }
    }

    pub(crate) fn params(&self) -> KeyParams {
        self.params
    }

    /// Whether the index keeps anything (`max_blocks > 0`).
    pub(crate) fn enabled(&self) -> bool {
        self.max_blocks > 0
    }

    pub(crate) fn entry_count(&self) -> usize {
        self.entries.len()
    }

    #[cfg(test)]
    pub(crate) fn track_count(&self) -> usize {
        self.tracks.len()
    }

    /// An estimate of the bytes the index holds: its map slots, the
    /// listed holders and the key lists of the tracks.
    pub(crate) fn approx_bytes(&self) -> u64 {
        let entry_slot = std::mem::size_of::<BlockKey>() + std::mem::size_of::<Entry>() + 1;
        let track_slot = std::mem::size_of::<SessionHandle>() + std::mem::size_of::<Track>() + 1;
        let holders: usize = self
            .entries
            .values()
            .map(|entry| entry.holders.capacity() * std::mem::size_of::<Holder>())
            .sum();
        let keys: usize = self
            .tracks
            .values()
            .map(|track| track.keys.capacity() * std::mem::size_of::<BlockKey>())
            .sum();
        (std::mem::size_of::<Self>()
            + self.entries.capacity() * entry_slot
            + holders
            + self.tracks.capacity() * track_slot
            + keys) as u64
    }

    /// Whether `handle` has a track.
    #[cfg(test)]
    pub(crate) fn is_tracked(&self, handle: SessionHandle) -> bool {
        self.tracks.contains_key(&handle)
    }

    /// The handles that have a track.
    pub(crate) fn tracked_handles(&self) -> Vec<SessionHandle> {
        self.tracks.keys().copied().collect()
    }

    /// Bytes of text fed for `handle`, when it has a track.
    pub(crate) fn tracked_bytes(&self, handle: SessionHandle) -> Option<u64> {
        self.tracks.get(&handle).map(|track| track.bytes)
    }

    /// Register a session stored in one call (a seed, an overwrite, or a
    /// session loaded from a durable file): every block `text` completes
    /// is counted with no holder, and `seal_after`, the session's seal
    /// once the call returned, is kept for the blocks later calls
    /// complete.
    pub(crate) fn insert_session(&mut self, handle: SessionHandle, text: &str, seal_after: Seal) {
        if !self.enabled() {
            return;
        }
        // A handle is never reused; a track already present would be a
        // bookkeeping slip, and replacing it keeps the counts right.
        self.remove_session(handle);
        let mut track = Track {
            keys: Vec::new(),
            rolling: Rolling::new(&self.params),
            bytes: 0,
            seal: Seal::default(),
            attached: true,
        };
        self.extend(handle, &mut track, text, seal_after);
        self.tracks.insert(handle, track);
    }

    /// Account for `delta` appended to a session. The blocks it completes
    /// carry the seal the session had before this call.
    pub(crate) fn append(&mut self, handle: SessionHandle, delta: &str, seal_after: Seal) {
        let Some(mut track) = self.tracks.remove(&handle) else {
            return;
        };
        self.extend(handle, &mut track, delta, seal_after);
        self.tracks.insert(handle, track);
    }

    fn extend(&mut self, handle: SessionHandle, track: &mut Track, text: &str, seal_after: Seal) {
        track.bytes += text.len() as u64;
        if track.attached {
            let mut completed = Vec::new();
            track.rolling.feed(&self.params, text, &mut completed);
            let seal_before = track.seal;
            for key in completed {
                let full = self.entries.len() >= self.max_blocks;
                match self.entries.get_mut(&key) {
                    Some(entry) => {
                        entry.refs += 1;
                        if !seal_before.is_zero() {
                            list_holder(
                                entry,
                                Holder {
                                    handle,
                                    seal: seal_before,
                                },
                            );
                        }
                    }
                    None if full => {
                        // At the cap: this session registers nothing more;
                        // what it registered stays until it goes.
                        self.counters.blocks_declined += 1;
                        track.attached = false;
                        break;
                    }
                    None => {
                        let mut holders = Vec::new();
                        if !seal_before.is_zero() {
                            holders.push(Holder {
                                handle,
                                seal: seal_before,
                            });
                        }
                        self.entries.insert(key, Entry { refs: 1, holders });
                    }
                }
                track.keys.push(key);
            }
        }
        track.seal = seal_after;
    }

    /// A fork has every block of its source, and offers the source's
    /// seals where the source is listed.
    pub(crate) fn fork(&mut self, source: SessionHandle, fork: SessionHandle) {
        let Some(mut track) = self.tracks.get(&source).cloned() else {
            return;
        };
        self.remove_session(fork);
        for (registered, key) in track.keys.iter().enumerate() {
            let Some(entry) = self.entries.get_mut(key) else {
                // Not reachable while the counts are right. Keep the fork
                // to the keys it was counted for, and let it register
                // nothing more.
                track.keys.truncate(registered);
                track.attached = false;
                break;
            };
            entry.refs += 1;
            let listed = entry
                .holders
                .iter()
                .find(|holder| holder.handle == source)
                .map(|holder| holder.seal);
            if let Some(seal) = listed {
                list_holder(entry, Holder { handle: fork, seal });
            }
        }
        self.tracks.insert(fork, track);
    }

    /// Release everything `handle` registered.
    pub(crate) fn remove_session(&mut self, handle: SessionHandle) {
        let Some(track) = self.tracks.remove(&handle) else {
            return;
        };
        for key in &track.keys {
            if let Some(entry) = self.entries.get_mut(key) {
                entry.holders.retain(|holder| holder.handle != handle);
                entry.refs = entry.refs.saturating_sub(1);
                if entry.refs == 0 {
                    self.entries.remove(key);
                }
            }
        }
        shrink_after_removal(&mut self.entries);
        shrink_after_removal(&mut self.tracks);
    }

    /// The listed holders of the entry for `key`; `None` when no live
    /// session has that block prefix.
    pub(crate) fn probe(&self, key: &BlockKey) -> Option<&[Holder]> {
        self.entries.get(key).map(|entry| entry.holders.as_slice())
    }

    /// Whether `handle` is still a listed holder of `key`.
    pub(crate) fn is_listed(&self, key: &BlockKey, handle: SessionHandle) -> bool {
        self.entries
            .get(key)
            .is_some_and(|entry| entry.holders.iter().any(|holder| holder.handle == handle))
    }

    /// Drop a holder that did not agree with the store, and count it.
    pub(crate) fn reject_holder(&mut self, key: &BlockKey, handle: SessionHandle) {
        if let Some(entry) = self.entries.get_mut(key) {
            entry.holders.retain(|holder| holder.handle != handle);
        }
        self.counters.holder_rejects += 1;
    }

    /// Every invariant the index keeps by itself: an entry exists iff its
    /// count is positive, the count equals the number of tracks listing
    /// its key, at most four holders per entry, each one tracked and
    /// counted for that key with a non-zero seal, and no more entries than
    /// the cap.
    #[cfg(test)]
    pub(crate) fn check_invariants(&self) -> std::result::Result<(), String> {
        let mut counted: HashMap<BlockKey, u32> = HashMap::new();
        for (handle, track) in &self.tracks {
            let mut seen = std::collections::HashSet::new();
            for key in &track.keys {
                if !seen.insert(*key) {
                    return Err(format!("handle {} lists a key twice", handle.0));
                }
                *counted.entry(*key).or_default() += 1;
            }
        }
        if self.entries.len() > self.max_blocks {
            return Err(format!(
                "{} entries above the cap {}",
                self.entries.len(),
                self.max_blocks
            ));
        }
        if counted.len() != self.entries.len() {
            return Err(format!(
                "{} keys tracked, {} entries",
                counted.len(),
                self.entries.len()
            ));
        }
        for (key, entry) in &self.entries {
            if entry.refs == 0 {
                return Err("an entry with no reference".to_owned());
            }
            if counted.get(key) != Some(&entry.refs) {
                return Err(format!(
                    "entry counts {} references, tracks {:?}",
                    entry.refs,
                    counted.get(key)
                ));
            }
            if entry.holders.len() > MAX_LISTED_HOLDERS {
                return Err(format!("{} listed holders", entry.holders.len()));
            }
            for holder in &entry.holders {
                if holder.seal.is_zero() {
                    return Err("a listed holder with the zero seal".to_owned());
                }
                let listed_by_track = self
                    .tracks
                    .get(&holder.handle)
                    .is_some_and(|track| track.keys.contains(key));
                if !listed_by_track {
                    return Err(format!(
                        "holder {} is not a live session with this block",
                        holder.handle.0
                    ));
                }
            }
        }
        Ok(())
    }

    /// The count of the entry for `key`.
    #[cfg(test)]
    pub(crate) fn refs(&self, key: &BlockKey) -> Option<u32> {
        self.entries.get(key).map(|entry| entry.refs)
    }

    /// The registered shape: each entry's count and each track's keys.
    #[cfg(test)]
    pub(crate) fn shape(
        &self,
    ) -> (
        std::collections::BTreeMap<BlockKey, u32>,
        std::collections::BTreeMap<u64, Vec<BlockKey>>,
    ) {
        (
            self.entries
                .iter()
                .map(|(key, entry)| (*key, entry.refs))
                .collect(),
            self.tracks
                .iter()
                .map(|(handle, track)| (handle.0, track.keys.clone()))
                .collect(),
        )
    }
}

fn list_holder(entry: &mut Entry, holder: Holder) {
    if entry
        .holders
        .iter()
        .any(|listed| listed.handle == holder.handle)
    {
        return;
    }
    if entry.holders.len() < MAX_LISTED_HOLDERS {
        entry.holders.push(holder);
        return;
    }
    // Keep the youngest sessions: the oldest is the likeliest to be
    // deleted or dropped next.
    if let Some(oldest) = entry
        .holders
        .iter_mut()
        .min_by_key(|listed| listed.handle)
        .filter(|listed| listed.handle < holder.handle)
    {
        *oldest = holder;
    }
}

/// Give memory back once a map is at most a quarter full; an empty map
/// returns to no allocation at all.
fn shrink_after_removal<K: std::hash::Hash + Eq, V>(map: &mut HashMap<K, V>) {
    if map.is_empty() {
        map.shrink_to_fit();
    } else if map.capacity() > SHRINK_FLOOR && map.len() * 4 < map.capacity() {
        map.shrink_to(map.len() * 2);
    }
}

/// The key of each complete block of a query, one at a time, computed
/// without any lock.
pub(crate) struct Walker<'a> {
    params: KeyParams,
    text: &'a str,
    byte: usize,
    parent: Option<BlockKey>,
    depth: u64,
}

impl<'a> Walker<'a> {
    pub(crate) fn new(params: KeyParams, text: &'a str) -> Self {
        Self {
            params,
            text,
            byte: 0,
            parent: None,
            depth: 0,
        }
    }

    /// The key of the next complete block and the byte of the query it
    /// ends at; `None` when fewer than `block_chars` characters remain.
    pub(crate) fn next_key(&mut self) -> Option<(BlockKey, usize)> {
        let rest = &self.text[self.byte..];
        let block_chars = self.params.block_chars_usize();
        let (end, taken) = split_after_chars(rest, block_chars);
        if taken < block_chars {
            return None;
        }
        let mut hasher = self.params.block_hasher(self.parent.as_ref(), self.depth);
        hasher.update(&rest.as_bytes()[..end]);
        let key: BlockKey = hasher.finalize().into();
        self.parent = Some(key);
        self.depth += 1;
        self.byte += end;
        Some((key, self.byte))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    const BLOCK: u64 = 8;

    fn params() -> KeyParams {
        KeyParams {
            fingerprint: [7; 32],
            block_chars: BLOCK,
        }
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
            (self.next() % bound as u64) as usize
        }
    }

    /// Pieces of one to four bytes, so block ends fall inside and between
    /// multi-byte characters' positions.
    const PIECES: [&str; 8] = [
        "a",
        "b",
        "\u{e9}",
        "\u{65e5}\u{672c}",
        "x y",
        "\u{1f642}",
        "e\u{301}",
        "\n",
    ];

    fn random_text(rng: &mut Rng, pieces: usize) -> String {
        (0..pieces)
            .map(|_| PIECES[rng.below(PIECES.len())])
            .collect()
    }

    /// Split `text` into random pieces at character boundaries.
    fn random_split<'t>(rng: &mut Rng, text: &'t str) -> Vec<&'t str> {
        let mut cuts: Vec<usize> = text
            .char_indices()
            .map(|(offset, _)| offset)
            .filter(|_| rng.below(5) == 0)
            .collect();
        cuts.push(text.len());
        cuts.dedup();
        let mut out = Vec::new();
        let mut start = 0;
        for cut in cuts {
            if cut > start {
                out.push(&text[start..cut]);
                start = cut;
            }
        }
        out
    }

    /// The key of every complete block of `text`, computed the plain way:
    /// one fresh hash per block over the whole chain.
    fn naive_keys(params: &KeyParams, text: &str) -> Vec<BlockKey> {
        let chars: Vec<char> = text.chars().collect();
        let mut keys: Vec<BlockKey> = Vec::new();
        for (index, block) in chars.chunks_exact(BLOCK as usize).enumerate() {
            let bytes: String = block.iter().collect();
            let mut hasher = Sha256::new();
            hasher.update(DOMAIN);
            match keys.last() {
                Some(parent) => {
                    hasher.update([1u8]);
                    hasher.update(parent);
                }
                None => {
                    hasher.update([0u8]);
                    hasher.update([0u8; 32]);
                }
            }
            hasher.update((index as u64).to_le_bytes());
            hasher.update(params.fingerprint);
            hasher.update(bytes.as_bytes());
            keys.push(hasher.finalize().into());
        }
        keys
    }

    /// A seal point at character `chars` of `text`, with `ids` sealed ids.
    fn seal_at(text: &str, chars: usize, ids: u64) -> Seal {
        let bytes = text
            .char_indices()
            .nth(chars)
            .map_or(text.len(), |(offset, _)| offset);
        Seal {
            chars: chars as u64,
            bytes: bytes as u64,
            ids,
        }
    }

    /// The naive model: every live session's text and the history of its
    /// seals, with the text length (in characters) each was made on.
    #[derive(Clone)]
    struct ModelSession {
        text: String,
        /// (seal, characters of the text when the seal was made).
        seals: Vec<(Seal, u64)>,
        /// The seal each block carries, by the rule "the seal before the
        /// call that completed it" (zero when the creating call did).
        block_seals: Vec<Seal>,
        /// `naive_keys` of the text, recomputed on every change.
        keys: Vec<BlockKey>,
    }

    impl ModelSession {
        fn current_seal(&self) -> Seal {
            self.seals.last().map_or(Seal::default(), |(seal, _)| *seal)
        }

        fn chars(&self) -> u64 {
            self.text.chars().count() as u64
        }

        /// Apply one call: grow the text, assign the seal before the call
        /// to the blocks the call completes, then take `seal_after`.
        fn apply(&mut self, delta: &str, seal_after: Seal, created: bool) {
            let before = if created {
                Seal::default()
            } else {
                self.current_seal()
            };
            self.text.push_str(delta);
            self.keys = naive_keys(&params(), &self.text);
            let blocks = (self.chars() / BLOCK) as usize;
            while self.block_seals.len() < blocks {
                self.block_seals.push(before);
            }
            if self.current_seal() != seal_after || self.seals.is_empty() {
                self.seals.push((seal_after, self.chars()));
            }
        }
    }

    /// A seal after a call: at or after the previous one, at most the
    /// text's length, ids growing with the characters and unique per
    /// session (`salt`).
    fn next_seal(rng: &mut Rng, text: &str, previous: Seal, salt: u64) -> Seal {
        let chars = text.chars().count();
        if chars == 0 || rng.below(4) == 0 {
            return previous;
        }
        let low = previous.chars as usize;
        let at = low + rng.below(chars - low + 1);
        if at == 0 || at as u64 == previous.chars {
            return previous;
        }
        seal_at(text, at, (at as u64) * 1_000 + salt)
    }

    struct Harness {
        index: PrefixIndex,
        live: BTreeMap<u64, ModelSession>,
        next_handle: u64,
        /// Per key: sealed holders registered since the entry was created
        /// (a replacement in the list can only happen past four).
        sealed_registrations: HashMap<BlockKey, usize>,
        /// The keys of recently removed texts.
        dead_texts: Vec<Vec<BlockKey>>,
        last_dead: String,
    }

    impl Harness {
        fn new(max_blocks: usize) -> Self {
            Self {
                index: PrefixIndex::new(params(), max_blocks),
                live: BTreeMap::new(),
                next_handle: 1,
                sealed_registrations: HashMap::new(),
                dead_texts: Vec::new(),
                last_dead: String::new(),
            }
        }

        fn note_registrations(&mut self, session: &ModelSession, first_block: usize) {
            for (block, key) in session.keys.iter().enumerate().skip(first_block) {
                if !session.block_seals[block].is_zero() {
                    *self.sealed_registrations.entry(*key).or_default() += 1;
                }
            }
        }

        fn insert(&mut self, rng: &mut Rng, text: String) {
            let handle = self.next_handle;
            self.next_handle += 1;
            let seal = next_seal(rng, &text, Seal::default(), handle);
            let mut session = ModelSession {
                text: String::new(),
                seals: Vec::new(),
                block_seals: Vec::new(),
                keys: Vec::new(),
            };
            session.apply(&text, seal, true);
            self.index
                .insert_session(SessionHandle(handle), &text, seal);
            self.live.insert(handle, session);
        }

        fn append(&mut self, rng: &mut Rng, handle: u64, delta: &str) {
            let mut session = self.live[&handle].clone();
            let blocks_before = session.block_seals.len();
            let grown = format!("{}{delta}", session.text);
            let seal = next_seal(rng, &grown, session.current_seal(), handle);
            session.apply(delta, seal, false);
            // Feed the delta in random pieces; only the last call carries
            // the new seal, the others keep the current one.
            let pieces = random_split(rng, delta);
            let mut seal_so_far = self.live[&handle].current_seal();
            let mut model = self.live[&handle].clone();
            for (position, piece) in pieces.iter().enumerate() {
                let seal_after = if position + 1 == pieces.len() {
                    seal
                } else {
                    seal_so_far
                };
                self.index.append(SessionHandle(handle), piece, seal_after);
                model.apply(piece, seal_after, false);
                seal_so_far = seal_after;
            }
            // Pieces that keep the seal give every block the seal before
            // the whole delta, so the model of one call and of the pieces
            // agree.
            assert_eq!(model.block_seals, session.block_seals);
            self.note_registrations(&session, blocks_before);
            self.live.insert(handle, session);
        }

        fn fork(&mut self, source: u64) {
            let handle = self.next_handle;
            self.next_handle += 1;
            let session = self.live[&source].clone();
            self.index
                .fork(SessionHandle(source), SessionHandle(handle));
            for key in &session.keys {
                if self.index.is_listed(key, SessionHandle(source)) {
                    *self.sealed_registrations.entry(*key).or_default() += 1;
                }
            }
            self.live.insert(handle, session);
        }

        fn remove(&mut self, handle: u64) {
            let session = self.live.remove(&handle).unwrap();
            self.index.remove_session(SessionHandle(handle));
            self.last_dead = session.text.clone();
            for key in &session.keys {
                if self.index.probe(key).is_none() {
                    self.sealed_registrations.remove(key);
                }
            }
            self.dead_texts.push(session.keys);
            if self.dead_texts.len() > 64 {
                self.dead_texts.remove(0);
            }
        }

        /// Compare the index with the model.
        fn check(&self) {
            self.index.check_invariants().unwrap();
            let mut refs: HashMap<BlockKey, u32> = HashMap::new();
            let mut best: HashMap<BlockKey, u64> = HashMap::new();
            let mut block_end: HashMap<BlockKey, u64> = HashMap::new();
            for session in self.live.values() {
                for (block, key) in session.keys.iter().enumerate() {
                    *refs.entry(*key).or_default() += 1;
                    let seal = session.block_seals[block];
                    let top = best.entry(*key).or_default();
                    *top = (*top).max(seal.ids);
                    block_end.insert(*key, (block as u64 + 1) * BLOCK);
                }
            }
            assert_eq!(self.index.entry_count(), refs.len(), "entries");
            for (key, count) in &refs {
                assert_eq!(self.index.refs(key), Some(*count), "refs");
            }
            assert_eq!(self.index.track_count(), self.live.len());
            for handle in self.live.keys() {
                assert!(self.index.is_tracked(SessionHandle(*handle)));
            }
            for (key, end) in &block_end {
                let holders = self.index.probe(key).unwrap();
                assert!(holders.len() <= MAX_LISTED_HOLDERS);
                for holder in holders {
                    let session = &self.live[&holder.handle.0];
                    let made_on = session
                        .seals
                        .iter()
                        .find(|(seal, _)| *seal == holder.seal)
                        .map(|(_, chars)| *chars)
                        .expect("a listed seal is one of its session's seals");
                    assert!(
                        made_on < *end,
                        "a seal made on {made_on} characters serves a block ending at {end}"
                    );
                }
                let registrations = self.sealed_registrations.get(key).copied().unwrap_or(0);
                if registrations <= MAX_LISTED_HOLDERS {
                    let listed_best = holders.iter().map(|h| h.seal.ids).max().unwrap_or(0);
                    assert_eq!(listed_best, best[key], "best listed seal");
                }
            }
            for keys in &self.dead_texts {
                for key in keys {
                    assert_eq!(self.index.probe(key).is_some(), refs.contains_key(key));
                }
            }
        }
    }

    /// A new text: a prefix of a live text (cut anywhere) and new pieces.
    fn related_text(rng: &mut Rng, harness: &Harness) -> String {
        let mut text = match harness
            .live
            .values()
            .nth(rng.below(harness.live.len().max(1)))
        {
            Some(other) if rng.below(3) != 0 => {
                let chars = other.text.chars().count();
                other.text.chars().take(rng.below(chars + 1)).collect()
            }
            _ => String::new(),
        };
        text.push_str(&{
            let pieces = rng.below(40);
            random_text(rng, pieces)
        });
        text
    }

    #[test]
    fn the_index_follows_a_naive_model_of_the_live_sessions() {
        for seed in [1u64, 2, 0xDEC0DE] {
            let mut rng = Rng(seed);
            let mut harness = Harness::new(usize::MAX);
            for _ in 0..20_000 {
                let handles: Vec<u64> = harness.live.keys().copied().collect();
                match rng.below(10) {
                    0 | 1 if harness.live.len() < 24 => {
                        let text = related_text(&mut rng, &harness);
                        harness.insert(&mut rng, text);
                    }
                    2..=5 if !handles.is_empty() => {
                        let handle = handles[rng.below(handles.len())];
                        if harness.live[&handle].text.chars().count() > 240 {
                            harness.remove(handle);
                            harness.check();
                            continue;
                        }
                        let delta = {
                            let pieces = 1 + rng.below(20);
                            random_text(&mut rng, pieces)
                        };
                        harness.append(&mut rng, handle, &delta);
                    }
                    6 if !handles.is_empty() && harness.live.len() < 24 => {
                        harness.fork(handles[rng.below(handles.len())]);
                    }
                    7 | 8 if !handles.is_empty() => {
                        harness.remove(handles[rng.below(handles.len())]);
                    }
                    _ => {
                        // Probe a query: a live or dead text, extended.
                        let source = if !handles.is_empty() && rng.below(2) == 0 {
                            harness.live[&handles[rng.below(handles.len())]]
                                .text
                                .clone()
                        } else {
                            harness.last_dead.clone()
                        };
                        let query = format!("{source}{}", random_text(&mut rng, 5));
                        let mut walker = Walker::new(params(), &query);
                        for key in naive_keys(&params(), &query) {
                            let (walked, _) = walker.next_key().unwrap();
                            assert_eq!(walked, key);
                            let live = harness.live.values().any(|s| s.keys.contains(&key));
                            assert_eq!(harness.index.probe(&key).is_some(), live);
                        }
                        assert!(walker.next_key().is_none());
                    }
                }
                harness.check();
            }
            let handles: Vec<u64> = harness.live.keys().copied().collect();
            for handle in handles {
                harness.remove(handle);
                harness.check();
            }
        }
    }

    #[test]
    fn removing_every_session_leaves_nothing() {
        let mut rng = Rng(9);
        let mut harness = Harness::new(usize::MAX);
        let empty_bytes = harness.index.approx_bytes();
        for _ in 0..300 {
            let text = related_text(&mut rng, &harness);
            harness.insert(&mut rng, text);
            let handles: Vec<u64> = harness.live.keys().copied().collect();
            let handle = handles[rng.below(handles.len())];
            let delta = random_text(&mut rng, 30);
            harness.append(&mut rng, handle, &delta);
            if rng.below(3) == 0 {
                harness.fork(handle);
            }
        }
        assert!(harness.index.entry_count() > 0);
        let handles: Vec<u64> = harness.live.keys().copied().collect();
        for handle in handles {
            harness.remove(handle);
        }
        harness.check();
        assert_eq!(harness.index.entry_count(), 0);
        assert_eq!(harness.index.track_count(), 0);
        assert_eq!(harness.index.approx_bytes(), empty_bytes);
    }

    #[test]
    fn the_cap_declines_new_entries_and_keeps_counting_existing_ones() {
        let seal = |text: &str| seal_at(text, 1, 5);
        let a = "abcdefgh".repeat(4); // 4 blocks
        let mut index = PrefixIndex::new(params(), 3);
        index.insert_session(SessionHandle(1), &a[..8], seal(&a));
        index.append(SessionHandle(1), &a[8..24], seal(&a));
        assert_eq!(index.entry_count(), 3);
        // The fourth block needs a new entry: declined, and the track is
        // detached for good.
        index.append(SessionHandle(1), &a[24..], seal(&a));
        assert_eq!(index.counters.blocks_declined, 1);
        assert_eq!(index.entry_count(), 3);
        index.append(SessionHandle(1), &"zzzzzzzz".repeat(2), seal(&a));
        assert_eq!(index.counters.blocks_declined, 1);
        assert_eq!(index.shape().1[&1].len(), 3);
        index.check_invariants().unwrap();

        // A second session on the same prefix is counted on the existing
        // entries and declined at its first new one.
        let mut b = a[..24].to_owned();
        b.push_str("qqqqqqqq");
        index.insert_session(SessionHandle(2), &b, Seal::default());
        assert_eq!(index.counters.blocks_declined, 2);
        let (entries, tracks) = index.shape();
        assert!(entries.values().all(|refs| *refs == 2));
        assert_eq!(tracks[&2].len(), 3);
        index.check_invariants().unwrap();

        // Room returns once sessions are removed.
        index.remove_session(SessionHandle(1));
        index.remove_session(SessionHandle(2));
        assert_eq!(index.entry_count(), 0);
        index.insert_session(SessionHandle(3), &b, Seal::default());
        assert_eq!(index.entry_count(), 3);
        assert_eq!(index.counters.blocks_declined, 3);
        index.remove_session(SessionHandle(3));
        index.insert_session(SessionHandle(4), &a[..24], Seal::default());
        assert_eq!(index.entry_count(), 3);
        assert_eq!(index.counters.blocks_declined, 3);
        index.check_invariants().unwrap();

        // The option at 0 keeps nothing.
        let mut off = PrefixIndex::new(params(), 0);
        off.insert_session(SessionHandle(1), &a, Seal::default());
        off.append(SessionHandle(1), &a, seal(&a));
        off.fork(SessionHandle(1), SessionHandle(2));
        assert!(!off.enabled());
        assert_eq!(off.entry_count(), 0);
        assert_eq!(off.tracked_handles(), Vec::new());
        assert_eq!(off.counters.blocks_declined, 0);
    }

    #[test]
    fn many_sessions_on_one_prefix_keep_short_lists_and_right_counts() {
        let mut rng = Rng(1024);
        let prefix: String = random_text(&mut rng, 25 * BLOCK as usize)
            .chars()
            .take(25 * BLOCK as usize)
            .collect();
        let mut index = PrefixIndex::new(params(), usize::MAX);
        let mut handles = Vec::new();
        for handle in 1..=1024u64 {
            // Seed one block short of the prefix with a seal, then
            // complete the prefix by an append so that the last blocks
            // list holders.
            let head: String = prefix.chars().take(23 * BLOCK as usize + 3).collect();
            index.insert_session(SessionHandle(handle), &head, seal_at(&head, 10, handle));
            index.append(
                SessionHandle(handle),
                &prefix[head.len()..],
                seal_at(&prefix, 12, handle + 5_000),
            );
            handles.push(handle);
        }
        let keys = naive_keys(&params(), &prefix);
        assert_eq!(keys.len(), 25);
        let check = |index: &PrefixIndex, live: usize| {
            index.check_invariants().unwrap();
            let (entries, _) = index.shape();
            for key in &keys {
                assert_eq!(entries.get(key).copied().unwrap_or(0) as usize, live);
                if live > 0 {
                    assert!(index.probe(key).unwrap().len() <= MAX_LISTED_HOLDERS);
                }
            }
        };
        check(&index, 1024);
        // The listed holders of the appended blocks are the youngest.
        let listed: Vec<u64> = index
            .probe(&keys[24])
            .unwrap()
            .iter()
            .map(|h| h.handle.0)
            .collect();
        assert_eq!(listed.len(), MAX_LISTED_HOLDERS);
        assert!(listed.iter().all(|handle| *handle > 1020));
        while !handles.is_empty() {
            let handle = handles.swap_remove(rng.below(handles.len()));
            index.remove_session(SessionHandle(handle));
            check(&index, handles.len());
        }
        assert_eq!(index.entry_count(), 0);
    }

    #[test]
    fn the_walker_and_a_track_compute_the_same_keys() {
        for seed in 0..200u64 {
            let mut rng = Rng(seed);
            let text = {
                let pieces = rng.below(120);
                random_text(&mut rng, pieces)
            };
            let expected = naive_keys(&params(), &text);
            let mut walker = Walker::new(params(), &text);
            let mut walked = Vec::new();
            let mut last_end = 0;
            while let Some((key, end)) = walker.next_key() {
                assert!(text.is_char_boundary(end) && end > last_end);
                last_end = end;
                walked.push(key);
            }
            assert_eq!(walked, expected);
            assert_eq!(
                text[..last_end].chars().count() as u64,
                expected.len() as u64 * BLOCK
            );

            let mut index = PrefixIndex::new(params(), usize::MAX);
            let pieces = random_split(&mut rng, &text);
            let (first, rest) = pieces.split_first().map_or(("", &[][..]), |(f, r)| (*f, r));
            index.insert_session(SessionHandle(1), first, Seal::default());
            for piece in rest {
                index.append(SessionHandle(1), piece, Seal::default());
            }
            assert_eq!(
                index.shape().1.get(&1).cloned().unwrap_or_default(),
                expected
            );
            assert_eq!(
                index.tracked_bytes(SessionHandle(1)),
                Some(text.len() as u64)
            );
        }
    }
}
