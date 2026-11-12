//! A durable session store at its bound, through the public API with the
//! real capacity: 1,023 named sessions are stored, creating one more is
//! declined with `InvalidArgument` before anything changes, and every other
//! call keeps working, across restarts.
//!
//! Artifact tier: the test loads `qwen3_8b` from `TOKTIER_TEST_ARTIFACTS`
//! (or the default cache) and is skipped when it is not there. Most texts
//! are short so that the 1,023 saves stay quick; the ones a lookup is asked
//! about are longer than one sealed block.
#![cfg(feature = "sqlite")]

use std::path::{Path, PathBuf};

use toktier::{Device, Error, ErrorCode, Runtime, Tokenizer};

const BOUND: usize = 1023;

fn artifact_root() -> Option<PathBuf> {
    std::env::var_os("TOKTIER_TEST_ARTIFACTS")
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME")
                .map(PathBuf::from)
                .map(|home| home.join(".cache/toktier/artifacts"))
        })
        .filter(|root| root.join("qwen3_8b-b968826d9c46/tokenizer.json").is_file())
}

fn load(root: &Path, home: &Path) -> (Runtime, Tokenizer) {
    let runtime = Runtime::builder()
        .artifact_cache(root)
        .device(Device::Cpu)
        .home(home)
        .build()
        .unwrap();
    let tokenizer = runtime.load("qwen3_8b").unwrap();
    (runtime, tokenizer)
}

/// A distinct ASCII text of `len` bytes.
fn text(salt: u64, index: u64, len: usize) -> String {
    const WORDS: [&str; 8] = [
        "river", "stone", "maple", "cloud", "lantern", "harbor", "meadow", "copper",
    ];
    let mut state = (salt << 32) ^ index ^ 0x9E37_79B9_7F4A_7C15;
    let mut out = format!("Document {salt}-{index}. ");
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

/// What session `s-<index>` is seeded with: long for the ones a lookup is
/// asked about, short for the rest.
fn seed_text(index: u64) -> String {
    text(1, index, if index == 7 || index == 8 { 6000 } else { 48 })
}

fn assert_declined(error: &Error) {
    assert_eq!(error.code(), ErrorCode::InvalidArgument, "{error}");
    assert_eq!(
        error.message(),
        "the durable session store holds 1023 named sessions, its capacity; delete a session before creating another"
    );
}

struct Store {
    tokenizer: Tokenizer,
    _runtime: Runtime,
}

impl Store {
    fn seed(&self, name: &str, text: &str) -> toktier::Result<()> {
        let mut session = self.tokenizer.open_session(name)?;
        let ids = session.seed(text)?.ids().to_vec();
        assert_eq!(ids, self.encode(text), "{name}");
        session.close()
    }

    fn encode(&self, text: &str) -> Vec<u32> {
        self.tokenizer.encode(text).unwrap().ids().to_vec()
    }

    fn append(&self, name: &str, stored: &mut String, delta: &str) {
        let mut session = self.tokenizer.open_session(name).unwrap();
        session.append(delta).unwrap();
        stored.push_str(delta);
        assert_eq!(
            session.snapshot().unwrap().ids(),
            self.encode(stored),
            "{name}"
        );
        session.close().unwrap();
    }

    fn delete(&self, name: &str) {
        let session = self.tokenizer.open_session(name).unwrap();
        assert!(session.revision().is_some(), "{name} is stored");
        session.delete().unwrap();
    }

    fn lookup_hits(&self, query: &str) {
        let hit = self.tokenizer.lookup(query).unwrap().expect("a lookup hit");
        assert_eq!(hit.ids(), self.encode(query));
    }

    fn assert_sessions(&self, count: u64) {
        let stats = self.tokenizer.store_stats();
        assert_eq!(stats.session_count, count);
        assert_eq!(stats.sessions_evicted, 0);
    }

    fn assert_stored(&self, name: &str, text: &str) {
        let session = self.tokenizer.open_session(name).unwrap();
        assert!(session.revision().is_some(), "{name} is stored");
        assert_eq!(
            session.snapshot().unwrap().ids(),
            self.encode(text),
            "{name}"
        );
        session.close().unwrap();
    }
}

#[test]
fn a_durable_store_holds_1023_named_sessions_and_declines_the_next() {
    let Some(root) = artifact_root() else {
        eprintln!("skip: no qwen3_8b artifact under TOKTIER_TEST_ARTIFACTS or the cache");
        return;
    };
    let temporary = tempfile::tempdir().unwrap();
    let home = temporary.path().join("runtime-home");
    let open = || {
        let (runtime, tokenizer) = load(&root, &home);
        Store {
            tokenizer,
            _runtime: runtime,
        }
    };
    let mut stored: std::collections::HashMap<String, String> = (0..BOUND as u64)
        .map(|i| (format!("s-{i}"), seed_text(i)))
        .collect();

    let store = open();
    // Seed until declined: 1,023 are stored, the 1,024th is declined.
    for index in 0..=BOUND as u64 {
        let name = format!("s-{index}");
        let result = store.seed(&name, &seed_text(index));
        if index < BOUND as u64 {
            result.unwrap();
        } else {
            assert_declined(&result.unwrap_err());
        }
    }
    store.assert_sessions(BOUND as u64);

    // Nothing is sealed: stored sessions keep working, and a lookup hit
    // evicts nothing.
    store.append("s-3", stored.get_mut("s-3").unwrap(), " a later turn");
    store.lookup_hits(&format!("{} and a question", stored["s-7"]));
    store.assert_sessions(BOUND as u64);

    // `delete` makes room for exactly one.
    store.delete("s-5");
    stored.remove("s-5");
    let new_a = text(2, 0, 48);
    store.seed("new-a", &new_a).unwrap();
    stored.insert("new-a".to_owned(), new_a);
    assert_declined(&store.seed("new-b", &text(2, 1, 48)).unwrap_err());

    // The other calls that create a name are declined while full.
    let source = store.tokenizer.open_session("s-10").unwrap();
    assert_declined(&source.fork("fk-0").unwrap_err());
    source.close().unwrap();
    let mut unseeded = store.tokenizer.open_session("brand-new").unwrap();
    assert_declined(&unseeded.overwrite(&text(2, 4, 48)).unwrap_err());
    drop(unseeded);

    // Overwriting a stored name and the full-transcript call work.
    let replaced = text(2, 6, 48);
    let mut s6 = store.tokenizer.open_session("s-6").unwrap();
    assert_eq!(
        s6.overwrite(&replaced).unwrap().ids(),
        store.encode(&replaced)
    );
    s6.close().unwrap();
    stored.insert("s-6".to_owned(), replaced);
    let complete = format!("{} and the full transcript", stored["s-4"]);
    let mut s4 = store.tokenizer.open_session("s-4").unwrap();
    assert_eq!(
        s4.encode_transcript(&complete).unwrap().ids(),
        store.encode(&complete)
    );
    s4.close().unwrap();
    stored.insert("s-4".to_owned(), complete);
    store.assert_sessions(BOUND as u64);
    drop(store);

    // A restart reads the same 1,023 sessions with the changes made above.
    let store = open();
    store.assert_sessions(BOUND as u64);
    for name in ["s-0", "new-a", "s-3", "s-6", "s-4"] {
        store.assert_stored(name, &stored[name]);
    }
    for name in ["s-5", "new-b", "fk-0", "brand-new"] {
        let session = store.tokenizer.open_session(name).unwrap();
        assert_eq!(session.revision(), None, "{name}");
    }
    assert_declined(&store.seed("new-c", &text(2, 2, 48)).unwrap_err());
    store.lookup_hits(&format!("{} and another question", stored["s-8"]));
    store.append(
        "s-9",
        stored.get_mut("s-9").unwrap(),
        " a turn after restart",
    );
    for j in 0..10u64 {
        let old = format!("s-{}", 20 + j);
        store.delete(&old);
        stored.remove(&old);
        let fresh = text(3, j, 48);
        store.seed(&format!("r-{j}"), &fresh).unwrap();
        stored.insert(format!("r-{j}"), fresh);
    }
    store.assert_sessions(BOUND as u64);
    drop(store);

    let store = open();
    store.assert_sessions(BOUND as u64);
    for name in ["r-0", "r-9", "s-9", "s-30", "s-1022", "new-a"] {
        store.assert_stored(name, &stored[name]);
    }
}
