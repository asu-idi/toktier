# Session-seed differential profiler

This standalone diagnostic project decomposes the Rust `Session::seed` path
without editing source files covered by TokTier's CPU/GPU certification
digests. It reproduces the exact historical 4 MiB Qwen3-8B lifecycle payload
and emits one JSON observation per fresh process.

The result distinguishes direct observations from differential estimates. In
particular, counterfactual precomputed encoders preserve exact IDs, spans,
boundary certificates, store configuration, and integrity work while removing
the routed full-tokenization computation. Post-text factorial cells expose
interactions between block chaining and certified tail sealing; they must not
be added as if independent.

Build and run through `run.py`; it checks the product commit/source identities,
uses a fresh process for every retained sample, keeps correctness checks
outside the timer, and writes raw JSONL plus aggregate JSON.

```bash
cargo build --release --manifest-path tools/seed-breakdown/Cargo.toml
python3 tools/seed-breakdown/run.py \
  --artifact /path/to/qwen3_8b/tokenizer.json \
  --output /path/to/cpu.json \
  --samples 31 --device cpu --numactl
```

On an RTX 5090 host, replace `--device cpu --numactl` with `--device gpu`.
The GPU default set records ID-only encode, encode with character offsets,
in-memory seed, and SQLite seed. To derive the labelled median contrasts:

```bash
python3 tools/seed-breakdown/analyze.py \
  --cpu /path/to/cpu.json \
  --gpu /path/to/gpu.json \
  --output /path/to/analysis.json
```

Set `TOKTIER_PROFILE_PRODUCT_COMMIT` when the measured source snapshot has no
Git metadata. The recorded source identities remain the authoritative binding.

## W3 direct cells

`--suite w3` selects the PLAN/163 W3 cells. They measure post-encode stages
directly, so their exact ID rows are precomputed before the timer starts:

- `store_seed_soa_shape[_unicode]` and `store_seed_lazy_shape[_unicode]`:
  the complete state-seed shape (text in, session committed, complete row
  returned) over the real corrected-Gigatoken engine, through the retained
  W4a structure-of-arrays adoption versus the W4b lazy shared-buffer
  adoption respectively. Both record seal counters and the store's
  full-row materialization count.
- `store_seed_lazy_shape_overlap[_unicode]`: the same lazy seed shape with
  the W4c bounded-pool digest overlap installed (CHANGE-162 C5). The worker
  count and pool shape are recorded per observation, the content digest is
  asserted byte-identical to the direct serial scan inside the process, and
  overlap readings never mix with the serial cell.
- `store_seed_concurrent4[_overlap]`: four independent stores and engine
  sets seed the same payload from four caller threads; the wall clock spans
  barrier release to the last committed seed, with per-lane latencies in
  the details. The overlap variant contrasts multi-request throughput under
  the shared bounded pool.
- `store_seed_overlap_longrun`: forty overlap seeds in one process with
  resident memory recorded after the first and last rounds, for worker-leak
  and RSS-stability evidence.
- `spans_direct[_unicode]`: the production known-ID span bridge
  (`spans_for_ids`), with the raw byte-length table prewarmed.
- `spans_soa_direct[_unicode]`: the production one-pass structure-of-arrays
  bridge (`spans_soa_for_ids`, the W4a WP1 landing), compared element for
  element against the retained pair bridge.
- `spans_soa_proto[_unicode]`: a one-pass structure-of-arrays prototype
  (ASCII direct fill; Unicode streaming UTF-8 boundary merge).
- `spans_lazy_closure[_unicode]`: allocation-free streaming byte-length
  closure check over the whole row.
- `spans_lazy_tail_window[_unicode]`: spans for only the mutable-tail window
  (ASCII suffix back-projection; Unicode anchored streaming merge).
- `spans_checkpoint_build[_unicode]` and `spans_checkpoint_window[_unicode]`:
  sparse cumulative checkpoints every 4096 tokens, and one arbitrary window
  rebuilt from its nearest checkpoint.
- `payload_digest_seed_shape` / `payload_digest_append_shape`: the product
  payload digest over the whole row and over the post-seal sealed+tail shape.
- `payload_digest_incremental_proto`: a cloned sealed-prefix hasher state fed
  only the tail; the digest must be bit-identical to the product function.
- `payload_digest_incremental_direct`: the production incremental payload
  hasher (`PayloadHasher`, the W4a A1 landing) over the same append shape.
- `payload_digest_chunked`: the same digest bytes fed through a 16 KiB stack
  buffer instead of one 4-byte update per ID.
- `added_gate_scan`: one Aho-Corasick pass with the router's added-token
  pattern set.

Every prototype cell asserts element-for-element (spans) or bit-for-bit
(digest) equality against the product implementation inside the measured
process; a failed comparison is a process error, not a data row. The Unicode
payload is generated from ASCII escapes only, is normalization-stable for the
artifact, and its SHA-256 is recorded in each observation. All W3 cells are
host-side CPU measurements and run beneath the routing registry, so they also
run on an intermediate tree whose certificates will be regenerated in a later
batch; `runtime_build_certified` is recorded honestly per row.
