//! SQLite tier battery: bit-exact roundtrip, corruption rejection with
//! contract behavior, and exclusive file ownership.

use std::path::PathBuf;

use tempfile::TempDir;
use toktier_store_core::testing::MockEncoder;
use toktier_store_core::{SessionEncoder, SessionStore, StoreConfig};
use toktier_store_sqlite::{DbError, SingleEngine, StoreDb};

fn fp(tag: u8) -> [u8; 32] {
    let mut out = [0u8; 32];
    out[0] = tag;
    out
}

fn cfg64() -> StoreConfig {
    StoreConfig {
        block_chars: 64,
        ..StoreConfig::default()
    }
}

fn tmp_db(name: &str) -> (TempDir, PathBuf) {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join(name);
    (dir, path)
}

fn assert_schema_value_error(error: DbError, column: &str, value: &str) {
    match error {
        DbError::Schema(message) => {
            assert!(message.contains(column), "missing column in {message:?}");
            assert!(message.contains(value), "missing value in {message:?}");
        }
        other => panic!("unexpected error {other:?}"),
    }
}

fn corrupt_integers(path: &std::path::Path, sql: &str) {
    let conn = rusqlite::Connection::open(path).unwrap();
    // Model corruption of a database created before the checks existed.
    conn.pragma_update(None, "ignore_check_constraints", true)
        .unwrap();
    conn.execute_batch(sql).unwrap();
}

fn load_error(path: &std::path::Path, enc: &MockEncoder) -> DbError {
    let db = StoreDb::open(path).unwrap();
    match db.load(&SingleEngine(enc)) {
        Err(error) => error,
        Ok(_) => panic!("corrupt integer value must abort the load"),
    }
}

/// Corrupt one sqlite blob out-of-band. The store database is
/// Rust-exclusive while open; this helper opens the file only after the
/// owning handle is dropped, standing in for on-disk corruption.
fn flip_blob_byte(path: &std::path::Path, table: &str, at_end: bool) {
    let conn = rusqlite::Connection::open(path).unwrap();
    let (pk_col, blob_col) = match table {
        "nodes" => ("node_key", "rec"),
        "sessions" => ("id", "rec"),
        other => panic!("unknown table {other}"),
    };
    let keys: Vec<rusqlite::types::Value> = conn
        .prepare(&format!("SELECT {pk_col} FROM {table}"))
        .unwrap()
        .query_map([], |r| r.get(0))
        .unwrap()
        .map(|v| v.unwrap())
        .collect();
    assert!(!keys.is_empty(), "no rows to corrupt in {table}");
    for key in keys {
        let rec: Vec<u8> = conn
            .query_row(
                &format!("SELECT {blob_col} FROM {table} WHERE {pk_col} = ?1"),
                [&key],
                |r| r.get(0),
            )
            .unwrap();
        let mut bad = rec.clone();
        let ix = if at_end { bad.len() - 1 } else { 0 };
        bad[ix] ^= 1;
        conn.execute(
            &format!("UPDATE {table} SET {blob_col} = ?1 WHERE {pk_col} = ?2"),
            (bad.as_slice(), &key),
        )
        .unwrap();
    }
}

fn build_store(enc: &MockEncoder) -> (SessionStore, u64, Vec<u32>, String) {
    let mut store = SessionStore::new(cfg64()).unwrap();
    let kid = store.register_fingerprint(fp(1), 0).unwrap();
    let mut acc = "Persistent session content. ".repeat(15);
    let put = store.put(kid, &acc, enc).unwrap();
    store
        .append(
            put.handle,
            " with a tail delta \u{1f642}",
            put.revision,
            enc,
        )
        .unwrap();
    acc.push_str(" with a tail delta \u{1f642}");
    let ids = store.all_ids(put.handle).unwrap();
    (store, put.handle.0, ids, acc)
}

#[test]
fn sqlite_roundtrip_bit_exact_and_post_load_append() {
    let enc = MockEncoder::default();
    let (store, h, ids, acc) = build_store(&enc);
    let (_dir, path) = tmp_db("store.db");
    {
        let mut db = StoreDb::open(&path).unwrap();
        db.save(&store).unwrap();
    }
    let db = StoreDb::open(&path).unwrap();
    let (mut store2, hmap) = db.load(&SingleEngine(&enc)).unwrap();
    assert_eq!(hmap.len(), 1);
    let (old_id, new_handle) = hmap[0];
    assert_eq!(u64::try_from(old_id).unwrap(), h);
    assert_eq!(store2.all_ids(new_handle).unwrap(), ids);
    assert_eq!(
        store2.export_node_items().unwrap(),
        store.export_node_items().unwrap()
    );
    // Full-fidelity restore: the chain continues after load.
    assert!(store2.session_info(new_handle).unwrap().chain_ok);
    let rev = store2.revision(new_handle).unwrap();
    let out = store2
        .append(new_handle, " post-load delta.", rev, &enc)
        .unwrap();
    let want = enc.encode(&format!("{acc} post-load delta.")).unwrap().ids;
    assert_eq!(out.all_ids, want);
}

#[test]
fn sqlite_corrupted_node_blobs_reject_silently_sessions_loudly() {
    let enc = MockEncoder::default();
    let (store, _h, ids, _acc) = build_store(&enc);
    assert!(store.stats().node_count > 0);
    let (_dir, path) = tmp_db("store.db");
    {
        let mut db = StoreDb::open(&path).unwrap();
        db.save(&store).unwrap();
    }
    // Corrupt every node blob: load succeeds, nodes rejected + counted,
    // lookups miss (prefer miss over wrong).
    flip_blob_byte(&path, "nodes", false);
    {
        let db = StoreDb::open(&path).unwrap();
        let (mut store2, hmap) = db.load(&SingleEngine(&enc)).unwrap();
        assert_eq!(
            store2.stats().node_count,
            0,
            "corrupt nodes must be rejected"
        );
        assert!(store2.stats().import_rejects > 0);
        let kid = toktier_store_core::KeyId(0);
        let text = "Persistent session content. ".repeat(15);
        assert!(store2.lookup(kid, &text, &enc).unwrap().is_none());
        // Sessions themselves survived (their blobs are intact).
        assert_eq!(store2.all_ids(hmap[0].1).unwrap(), ids);
    }
    // Corrupt a session record blob: the load is loud.
    flip_blob_byte(&path, "sessions", true);
    let db = StoreDb::open(&path).unwrap();
    match db.load(&SingleEngine(&enc)) {
        Err(DbError::Store(e)) => assert!(e.is_rejection(), "unexpected error {e:?}"),
        Err(other) => panic!("unexpected load error {other:?}"),
        Ok(_) => panic!("corrupt session record must abort the load"),
    }
}

#[test]
fn sqlite_file_is_rust_exclusive_while_open() {
    let enc = MockEncoder::default();
    let (store, ..) = build_store(&enc);
    let (_dir, path) = tmp_db("store.db");
    let mut db = StoreDb::open(&path).unwrap();
    db.save(&store).unwrap();
    // With locking_mode=EXCLUSIVE and a completed write, a second
    // connection cannot read the database while this handle lives.
    let other = rusqlite::Connection::open(&path).unwrap();
    other
        .busy_timeout(std::time::Duration::from_millis(50))
        .unwrap();
    let read: Result<i64, _> = other.query_row("SELECT COUNT(*) FROM sessions", [], |r| r.get(0));
    assert!(read.is_err(), "second connection must be locked out");
    drop(db);
    let read_after: i64 = other
        .query_row("SELECT COUNT(*) FROM sessions", [], |r| r.get(0))
        .unwrap();
    assert_eq!(read_after, 1);
}

#[test]
fn sqlite_missing_engine_is_reported() {
    let enc = MockEncoder::default();
    let (store, ..) = build_store(&enc);
    let (_dir, path) = tmp_db("store.db");
    {
        let mut db = StoreDb::open(&path).unwrap();
        db.save(&store).unwrap();
    }
    struct NoEngines;
    impl toktier_store_sqlite::EngineResolver for NoEngines {
        fn resolve(&self, _fp: &[u8; 32]) -> Option<&dyn SessionEncoder> {
            None
        }
    }
    let db = StoreDb::open(&path).unwrap();
    assert!(matches!(
        db.load(&NoEngines),
        Err(DbError::MissingEngine(_))
    ));
}

#[test]
fn sqlite_save_rejects_unsigned_values_outside_integer_range() {
    let mut store = SessionStore::new(cfg64()).unwrap();
    store.register_fingerprint(fp(1), u64::MAX).unwrap();
    let (_dir, path) = tmp_db("store.db");
    let mut db = StoreDb::open(&path).unwrap();
    let error = db.save(&store).unwrap_err();
    assert_schema_value_error(
        error,
        "fingerprints.seal_guard_chars",
        &u64::MAX.to_string(),
    );
}

#[test]
fn sqlite_load_rejects_corrupt_integer_values() {
    let enc = MockEncoder::default();
    let (store, ..) = build_store(&enc);
    let (_dir, path) = tmp_db("store.db");
    {
        let mut db = StoreDb::open(&path).unwrap();
        db.save(&store).unwrap();
    }

    corrupt_integers(&path, "UPDATE fingerprints SET seal_guard_chars = -1");
    assert_schema_value_error(
        load_error(&path, &enc),
        "fingerprints.seal_guard_chars",
        "-1",
    );

    corrupt_integers(
        &path,
        "UPDATE fingerprints SET seal_guard_chars = 0, id = 4294967296",
    );
    assert_schema_value_error(load_error(&path, &enc), "fingerprints.id", "4294967296");

    corrupt_integers(
        &path,
        "UPDATE fingerprints SET id = 0; UPDATE sessions SET key_id = 4294967296",
    );
    assert_schema_value_error(load_error(&path, &enc), "sessions.key_id", "4294967296");

    corrupt_integers(&path, "UPDATE sessions SET key_id = 0, id = -1");
    assert_schema_value_error(load_error(&path, &enc), "sessions.id", "-1");
}
