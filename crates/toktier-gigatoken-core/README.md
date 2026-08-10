# TokTier Gigatoken core

This private Rust workspace crate contains the request-time subset of the
modified Gigatoken build already shipped in TokTier's wheel.  It is derived
from Gigatoken commit `34a15995fc930c3807cd176bfd8ee91c166ee2fe` and includes
the corrections recorded in
`packaging/fast_cpu/gigatoken-toktier-pinned-1.patch`.

The upstream code remains MIT licensed; its license and TokTier modification
notice are retained under `vendor/`.  TokTier's Apache-2.0 license applies to
the surrounding integration, routing, and bindings.

This source copy exists so the corrected BPE engine can execute inside the
single Rust request pipeline.  It intentionally excludes upstream Python
bindings, training, file readers, and batch orchestration.
