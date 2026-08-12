# Support matrix

Coverage is decided by **tokenizer content**, not by model name. Each entry
below is a specific tokenizer artifact frozen at a recorded SHA-256; all
evidence refers to those exact bytes. A model repository is covered exactly when
the tokenizer it ships matches a listed artifact.

Snapshot of the upstream audit: 2026-08-04
(470 repositories examined). The
machine-readable identities this package ships (family, pinned repository and
revision, per-file sha256) are printed by `toktier inspect --json`. The
repositories that share a certified artifact are enumerated under
[Verified model repositories](#verified-model-repositories) below; the full
audit table, with the per-segment comparison behind every row, lives in the
audit records behind this document and does not ship inside the package.

## Status vocabulary

| Status | Meaning |
|---|---|
| `certified` | Evidence on file for this artifact and this backend, with the backend identified by a binary digest. The prebuilt GPU kernel delivery in this release is in this state on its judged architectures (sm_89 and sm_120). |
| `certified_source` | Evidence binds source identity, build flags, and exact toolchain rather than one platform-specific binary. The integrated corrected-Gigatoken CPU engine and the locally compiled GPU JIT use this state; each also binds its backend-specific inputs. |
| `reference-only` | No accelerated route is offered. The reference implementation runs and the reason is reported by `explain()`. This is also the state when the installed reference version differs from the certified one. |
| `experimental` | Available only by explicit opt-in; no certified exact-ID claim applies. Fastokens is exposed only in this state. |
| `unsupported` | The named engine is known not to load or represent this artifact and is never planned. |

Statuses are recorded per artifact **and** per backend: a family can be
`certified` on the CPU reference path and `reference-only` on GPU, and the
registry keeps both.

## Certified CPU fast repair

The default CPU full-encode and repair-window engine is the corrected
Gigatoken `0.10.0+toktier.pinned.1` implementation compiled directly into the
single private `toktier._native` extension in the core wheel. Its exact HF
loader/oracle companions (`tokenizers==0.22.2`, `transformers==4.57.6`) are
mandatory base dependencies of that wheel -- the shipped extras are
`fastokens`, `gpu`, and `gpu-jit` -- and there is no separately installed
Gigatoken distribution or second native extension. Its route is
certified for **11 unique
tokenizer artifacts** after 41,800,181,401 document-artifact checks over
3,800,016,491 documents and 12,328,592,579,973 characters, with zero observed
token-ID divergence from Hugging Face `tokenizers==0.22.2`.

Those artifacts cover **12 model families**. NVIDIA's Nemotron-Terminal family
is the additional lineage: its 8B, 14B, and 32B repositories ship the
`qwen3_8b` `tokenizer.json` byte-for-byte
(`aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`). It
inherits the same artifact-bound certificate; it is not counted as a twelfth
independent tokenizer implementation.

The route opens only when the integrated module, domain-separated source
digest, Cargo release flags, exact rustc, repair-table digest, patch, oracle
version, and tokenizer artifact all match the registry. Otherwise the facade
runs HF and gives the failed axis in `explain()`. Fastokens 0.3.1 is a separate
explicit experimental full-session adapter and has no TokTier exact-ID
guarantee.

The exact engine binding and native-equivalence record are in
[`tools/fast_cpu_binding.json`](../tools/fast_cpu_binding.json). A focused
released-API rerun over every row is in
[`readings/fast_cpu_focused_parity.json`](../readings/fast_cpu_focused_parity.json),
and the one-call native-front-end rerun is in
[`readings/fast_cpu_native_frontend_parity.json`](../readings/fast_cpu_native_frontend_parity.json).

## Byte-level BPE families

| Family id | Anchor repository | Artifact sha256 (prefix) | CPU fast repair | GPU status |
|---|---|---|---|---|
| `qwen3_8b` | `Qwen/Qwen3-8B` | aeb13307a71a | certified (Gigatoken) | certified (prebuilt) |
| `qwen3_5_08b` | `Qwen/Qwen3.5-0.8B` | 5f9e4d4901a9 | certified (Gigatoken) | certified (prebuilt) |
| `llama_3_1_8b` | `meta-llama/Llama-3.1-8B` | 76e48799b099 | certified (Gigatoken) | certified (prebuilt) |
| `deepseek_v3` | `deepseek-ai/DeepSeek-V3` | 621ac2e32d0d | certified (Gigatoken) | certified (prebuilt) |
| `deepseek_v4_flash` | `deepseek-ai/DeepSeek-V4-Flash` | 8f9f37ca37fd | certified (Gigatoken) | certified (prebuilt) |
| `gpt_oss_120b` | `openai/gpt-oss-120b` | 0614fe83cada | certified (Gigatoken) | certified (prebuilt) |
| `glm_5_2` | `zai-org/GLM-5.2` | 19e773648cb4 | certified (Gigatoken) | certified (prebuilt) |
| `minimax_m3` | `MiniMaxAI/MiniMax-M3` | bb1f1626cf01 | certified (Gigatoken) | certified (prebuilt) |
| `ministral_3_8b` | `mistralai/Ministral-3-8B-Instruct-2512-BF16` | d5f6046775b1 | certified (Gigatoken) | certified (prebuilt) |
| `nemotron_3_nano_4b` | `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` | 623c34567aeb | certified (Gigatoken) | certified (prebuilt) |
| `olmo_3_7b` | `allenai/Olmo-3-1025-7B` | 73fd5254624f | certified (Gigatoken) | certified (prebuilt) |
| `hy3` | `tencent/Hy3` | 446e0b59cd94 | unsupported by Gigatoken; HF | certified (prebuilt) |
| `kimi_k3` | `moonshotai/Kimi-K3` | 773c2476259d | unsupported by Gigatoken; HF | certified (prebuilt) |
| `ling_3_0_flash` | `inclusionAI/Ling-3.0-flash` | 40fb9d7d7795 | unsupported by Gigatoken; HF | certified (prebuilt) |
| `laguna_s_2_1` | `poolside/Laguna-S-2.1` | 809240f7a182 | unsupported by Gigatoken; HF | certified (prebuilt) |

Pinned upstream revisions are recorded per row as
`support.repo_revision.{family}` in the generated table; the package verifies
them at fetch time.

GPU status names the prebuilt kernel delivery: `certified (prebuilt)` is
binary-digest bound to the shipped fatbin, source/build bound to its Rust
request host, and judged on sm_89 and sm_120. The JIT delivery of the same
kernels remains `certified_source`;
the registry records both, per delivery, and the loader verifies the
delivery it actually runs.

`kimi_k3` note: the upstream repository ships a `tiktoken.model` rather than a
`tokenizer.json`. The frozen artifact is a conversion of that file, produced on
the installing machine from pinned upstream bytes; comparisons for sibling
repositories, and for that repository itself, are made at `tiktoken.model`
level. The recorded zero-divergence check between the conversion and the
upstream form covers 2,220,065 texts and is evidence on file, not a check this
package re-runs: it needs the upstream tokenizer runtime, which this package
deliberately does not depend on. What ships instead is
`toktier artifacts check-conversion`, which re-runs on your machine the three
properties that bind your bytes to that evidence -- the conversion is
deterministic, its output is the pinned sha256 and byte count, and the reserved
added-token block is contiguous and complete.

## WordPiece families

| Family id | Anchor repository | Artifact sha256 (prefix) | CPU fast repair | GPU status |
|---|---|---|---|---|
| `bert_cased` | `google-bert/bert-base-cased` | 5b3360be30cd | reference-only (not packaged) | n/a |
| `bert_uncased` | `google-bert/bert-base-uncased` | ce64fce797c2 | reference-only (not packaged) | n/a |
| `bert_multilingual_cased` | `google-bert/bert-base-multilingual-cased` | f4a4d5bf7301 | reference-only (not packaged) | n/a |

## Availability in this package

Recorded differential evidence and package availability are separate facts,
and for three rows the two currently differ. The artifact manifest shipped inside the package is
generated from the differential-campaign registry
(`tables/support_registry.json`) and therefore carries exactly its 15
families. The three WordPiece families have
boundary-predicate evidence, a second evidence family that has not been wired into
the shipped registry yet, so no packaged artifact identity exists for them:
`toktier artifacts fetch` and `toktier.load()` report `ARTIFACT_NOT_FOUND`
for these family ids in this release. Extending the shipped registry and
manifest to carry boundary-predicate evidence is release work tracked for a
later version; until then, `toktier inspect` is the authoritative listing of
what this package resolves.

## Sibling repositories

Many model repositories ship a tokenizer that is byte-identical to one of the
anchors above, or identical after JSON canonicalisation, or identical in its
core segments (`model`, `pre_tokenizer`, `normalizer`, `decoder`) while the
added-token table or chat-template post-processor differs. The generated table
records that comparison per repository, using these categories:

| Category | Meaning for coverage |
|---|---|
| identical | byte-identical to the anchor — the certificate carries over |
| equivalent | identical after canonicalisation (serialisation form differs); loads to the same tokenizer, recorded separately for transparency |
| added-token face | core segments equal, added/special-token table or post-processor differs; the kernel-facing tokenizer is the certified one, and the added-token table is regenerated for that repository by the added-token frontend |
| different | core segments differ — not covered |
| unverified | gated or otherwise not retrievable at audit time; listed as unresolved rather than assumed |
| n/a | repository ships no comparable tokenizer file |

Matching by repository name is not reliable — quantised or derived twins
sometimes ship a different serialisation than their main repository, and
occasionally a different tokenizer altogether. Matching by content hash is what
the package does at load time.

The repositories that landed in the identical and equivalent categories are
enumerated in the next section, with the digest each comparison was made
against.

## Verified model repositories

The categories above are a vocabulary; this section is the enumeration. It
names every repository the upstream audit placed in one of the identical or
equivalent categories, that is, every repository whose coverage follows from
artifact identity rather than from a campaign of its own.

The reasoning chain is short, and it is worth stating in full because
everything below rests on it:

1. A certificate is issued for an artifact, not for a model name. The evidence
   for a family names one tokenizer file and its sha256, and every campaign
   reading refers to those bytes.
2. If another repository ships a file with the same sha256, the two are the
   same bytes. Anything established about the bytes holds for both, so the
   certificate carries over unchanged. This is the whole of the argument for
   the `identical` rows, and it is why the tables below repeat the digest on
   every line: on an `identical` row that digest and the one in the family
   matrix above are the same string, and that string is the claim.
3. Where the bytes differ but the difference is confined to JSON formatting, or
   to serialisation form (an absent default field, the legacy string form of
   the merges table), the file loads to the same tokenizer. The certificate
   carries over, on a premise one step weaker than byte equality, so those rows
   are labelled separately rather than folded into `identical`.
4. Nothing else carries over. A repository whose core segments differ, or whose
   added-token table differs, is not listed as verified here, however small the
   difference looks; the exceptions are enumerated at the end of this section.

| Basis | What was compared | Rows |
|---|---|---|
| identical | `tokenizer.json` sha256 equal to the certified artifact | 150 |
| identical (source file) | `kimi_k3` lineage: upstream `tiktoken.model` sha256 equal to the conversion source of the certified artifact | 12 |
| equivalent (canonicalisation) | JSON content equal after canonicalisation; byte difference is formatting only | 10 |
| equivalent (serialisation) | loads to the same tokenizer; byte difference is legacy serialisation form | 38 |

**210 repositories** are verified on these bases, spread over 15 of the 18
artifacts in the family matrix. Nine of them are cross-vendor: three
`nvidia/Nemotron-Terminal-*` repositories ship the `qwen3_8b` artifact
verbatim, and six `nvidia/Llama-3.x-Nemotron-*` repositories carry the
`llama_3_1_8b` tokenizer in a different serialisation. They are listed under
the artifact that covers them, not under the organisation that published them,
which is the same rule the loader applies.

The executable counterpart is `toktier.from_pretrained(repo_id)`. It resolves
and hashes the repository's tokenizer file, then selects the canonical anchor
only for an exact entry in the packaged, root-digested sibling registry. The
accelerated engines receive the canonical bytes that were certified, not an
unreviewed alternate serialisation. Unknown or changed content stays on the HF
reference route.

### Counts per artifact

The last column is not part of the verified total. It counts repositories whose
`model`, `pre_tokenizer`, `normalizer` and `decoder` segments equal the
certified artifact while the added-token table or the post-processor differs.
The kernel-facing tokenizer in those repositories is the certified one, but the
loaded tokenizer is not the certified object, so they are recorded and not
counted.

| Artifact | Verified sub-versions | identical | equivalent (canonicalisation) | equivalent (serialisation) | Core-equal only (added-token face) |
|---|---|---|---|---|---|
| `qwen3_8b` | 110 | 79 | 0 | 31 | 18 |
| `qwen3_5_08b` | 19 | 19 | 0 | 0 | 5 |
| `llama_3_1_8b` | 13 | 3 | 5 | 5 | 6 |
| `deepseek_v3` | 2 | 2 | 0 | 0 | 10 |
| `deepseek_v4_flash` | 6 | 6 | 0 | 0 | 0 |
| `gpt_oss_120b` | 3 | 3 | 0 | 0 | 0 |
| `glm_5_2` | 6 | 6 | 0 | 0 | 0 |
| `minimax_m3` | 1 | 1 | 0 | 0 | 11 |
| `ministral_3_8b` | 2 | 2 | 0 | 0 | 5 |
| `nemotron_3_nano_4b` | 19 | 16 | 3 | 0 | 1 |
| `olmo_3_7b` | 9 | 7 | 0 | 2 | 0 |
| `hy3` | 1 | 1 | 0 | 0 | 21 |
| `kimi_k3` | 12 | 12 * | 0 | 0 | 0 |
| `bert_cased` | 4 | 2 | 2 | 0 | 0 |
| `bert_uncased` | 3 | 3 | 0 | 0 | 0 |
| `bert_multilingual_cased` | 0 | 0 | 0 | 0 | 0 |
| **total** | **210** | 162 | 10 | 38 | 77 |

`*` `kimi_k3` rows are compared at `tiktoken.model` level, for the reason given
under the family matrix above.

`bert_multilingual_cased` has no sibling in the audited set: the audit found no
other official repository shipping that artifact.

### Repositories

Rows are grouped by the artifact that covers them and sorted by repository
name. The anchor repository of each family is not repeated here; it is the row
in the family matrix above. The revision column records where the repository's
default branch stood at the snapshot. `from_pretrained()` expands it to the
audited full commit and uses that immutable revision by default; callers may
request another revision, but the resulting file must still match by content
before acceleration is admitted.

#### `qwen3_8b` (110 repositories)

| Repository | Revision at snapshot | tokenizer.json sha256 | Basis |
|---|---|---|---|
| `nvidia/Nemotron-Terminal-14B` | `fa6f29a51305` | `aeb13307a71a` | identical |
| `nvidia/Nemotron-Terminal-32B` | `a6794afe7fcc` | `aeb13307a71a` | identical |
| `nvidia/Nemotron-Terminal-8B` | `bb1413579351` | `aeb13307a71a` | identical |
| `Qwen/Qwen3-0.6B` | `c1899de289a0` | `aeb13307a71a` | identical |
| `Qwen/Qwen3-0.6B-FP8` | `e5be08033360` | `aeb13307a71a` | identical |
| `Qwen/Qwen3-0.6B-GPTQ-Int8` | `d3f20e7e7182` | `aeb13307a71a` | identical |
| `Qwen/Qwen3-0.6B-MLX-4bit` | `173234aa840d` | `aeb13307a71a` | identical |
| `Qwen/Qwen3-0.6B-MLX-6bit` | `3818e758c8ed` | `aeb13307a71a` | identical |
| `Qwen/Qwen3-0.6B-MLX-8bit` | `e53d3ae02ebc` | `aeb13307a71a` | identical |
| `Qwen/Qwen3-0.6B-MLX-bf16` | `bc82a1060abf` | `aeb13307a71a` | identical |
| `Qwen/Qwen3-1.7B` | `70d244cc86cc` | `aeb13307a71a` | identical |
| `Qwen/Qwen3-1.7B-FP8` | `1641e6c1b620` | `aeb13307a71a` | identical |
| `Qwen/Qwen3-1.7B-GPTQ-Int8` | `382fed3fa21d` | `aeb13307a71a` | identical |
| `Qwen/Qwen3-1.7B-MLX-4bit` | `21457c6f51ed` | `aeb13307a71a` | identical |
| `Qwen/Qwen3-1.7B-MLX-6bit` | `51d30dc1eaca` | `aeb13307a71a` | identical |
| `Qwen/Qwen3-1.7B-MLX-8bit` | `95400cbada1b` | `aeb13307a71a` | identical |
| `Qwen/Qwen3-1.7B-MLX-bf16` | `720c04346ea2` | `aeb13307a71a` | identical |
| `Qwen/Qwen3-14B` | `40c069824f42` | `aeb13307a71a` | identical |
| `Qwen/Qwen3-14B-AWQ` | `31c69efc2946` | `aeb13307a71a` | identical |
| `Qwen/Qwen3-14B-FP8` | `9a283b4a5efb` | `aeb13307a71a` | identical |
| `Qwen/Qwen3-14B-MLX-4bit` | `ba63a5141812` | `aeb13307a71a` | identical |
| `Qwen/Qwen3-14B-MLX-6bit` | `02c5e9b6ecab` | `aeb13307a71a` | identical |
| `Qwen/Qwen3-14B-MLX-8bit` | `7f11bdbf4e52` | `aeb13307a71a` | identical |
| `Qwen/Qwen3-14B-MLX-bf16` | `edbe64998ef1` | `aeb13307a71a` | identical |
| `Qwen/Qwen3-235B-A22B` | `8efa61729e24` | `aeb13307a71a` | identical |
| `Qwen/Qwen3-235B-A22B-FP8` | `39eb2b067ea6` | `aeb13307a71a` | identical |
| `Qwen/Qwen3-235B-A22B-GPTQ-Int4` | `dff138951f38` | `aeb13307a71a` | identical |
| `Qwen/Qwen3-235B-A22B-Instruct-2507` | `ac9c66cc9b46` | `aeb13307a71a` | identical |
| `Qwen/Qwen3-235B-A22B-Instruct-2507-FP8` | `e156cb4efae4` | `aeb13307a71a` | identical |
| `Qwen/Qwen3-235B-A22B-MLX-6bit` | `a9b4800972d7` | `aeb13307a71a` | identical |
| `Qwen/Qwen3-235B-A22B-MLX-8bit` | `06d7d9cb6602` | `aeb13307a71a` | identical |
| `Qwen/Qwen3-235B-A22B-MLX-bf16` | `284ea931f422` | `aeb13307a71a` | identical |
| `Qwen/Qwen3-235B-A22B-Thinking-2507` | `6cbffae6d8e2` | `19564a48c4f7` | equivalent (serialisation) |
| `Qwen/Qwen3-235B-A22B-Thinking-2507-FP8` | `f07f63f2bbd7` | `aeb13307a71a` | identical |
| `Qwen/Qwen3-30B-A3B` | `ad44e777bcd1` | `aeb13307a71a` | identical |
| `Qwen/Qwen3-30B-A3B-FP8` | `d206ba732169` | `aeb13307a71a` | identical |
| `Qwen/Qwen3-30B-A3B-GPTQ-Int4` | `9b534e4318b7` | `aeb13307a71a` | identical |
| `Qwen/Qwen3-30B-A3B-Instruct-2507` | `0d7cf23991f4` | `aeb13307a71a` | identical |
| `Qwen/Qwen3-30B-A3B-Instruct-2507-FP8` | `5a5a776300a4` | `aeb13307a71a` | identical |
| `Qwen/Qwen3-30B-A3B-MLX-4bit` | `4e2776a4cc73` | `aeb13307a71a` | identical |
| `Qwen/Qwen3-30B-A3B-MLX-6bit` | `b92024737fc3` | `aeb13307a71a` | identical |
| `Qwen/Qwen3-30B-A3B-MLX-8bit` | `ef6823f1ecb0` | `aeb13307a71a` | identical |
| `Qwen/Qwen3-30B-A3B-MLX-bf16` | `c8e239e419e3` | `aeb13307a71a` | identical |
| `Qwen/Qwen3-30B-A3B-Thinking-2507` | `144afc2f379b` | `19564a48c4f7` | equivalent (serialisation) |
| `Qwen/Qwen3-30B-A3B-Thinking-2507-FP8` | `60d80c83c53c` | `aeb13307a71a` | identical |
| `Qwen/Qwen3-32B` | `9216db5781bf` | `aeb13307a71a` | identical |
| `Qwen/Qwen3-32B-AWQ` | `0499c3ac83fd` | `aeb13307a71a` | identical |
| `Qwen/Qwen3-32B-FP8` | `aa55da1ecc13` | `aeb13307a71a` | identical |
| `Qwen/Qwen3-32B-MLX-4bit` | `ceb4c4aad016` | `aeb13307a71a` | identical |
| `Qwen/Qwen3-32B-MLX-6bit` | `864389199428` | `aeb13307a71a` | identical |
| `Qwen/Qwen3-32B-MLX-8bit` | `5fd4a0905f0e` | `aeb13307a71a` | identical |
| `Qwen/Qwen3-32B-MLX-bf16` | `cf84a682b91f` | `aeb13307a71a` | identical |
| `Qwen/Qwen3-4B` | `1cfa9a720891` | `aeb13307a71a` | identical |
| `Qwen/Qwen3-4B-AWQ` | `74d4bd2bd4bf` | `aeb13307a71a` | identical |
| `Qwen/Qwen3-4B-FP8` | `96b30dc13593` | `aeb13307a71a` | identical |
| `Qwen/Qwen3-4B-Instruct-2507` | `cdbee75f17c0` | `aeb13307a71a` | identical |
| `Qwen/Qwen3-4B-Instruct-2507-FP8` | `8591804019c8` | `aeb13307a71a` | identical |
| `Qwen/Qwen3-4B-MLX-4bit` | `52a5ab34fa60` | `aeb13307a71a` | identical |
| `Qwen/Qwen3-4B-MLX-6bit` | `d73d5dba88e4` | `aeb13307a71a` | identical |
| `Qwen/Qwen3-4B-MLX-8bit` | `315fb4813e1d` | `aeb13307a71a` | identical |
| `Qwen/Qwen3-4B-MLX-bf16` | `94e68620a4b9` | `aeb13307a71a` | identical |
| `Qwen/Qwen3-4B-SafeRL` | `1b95ccb88cab` | `aeb13307a71a` | identical |
| `Qwen/Qwen3-4B-Thinking-2507` | `768f209d9ea8` | `aeb13307a71a` | identical |
| `Qwen/Qwen3-4B-Thinking-2507-FP8` | `953532f94270` | `aeb13307a71a` | identical |
| `Qwen/Qwen3-8B-AWQ` | `4da05a8edb55` | `aeb13307a71a` | identical |
| `Qwen/Qwen3-8B-FP8` | `220b46e3b218` | `aeb13307a71a` | identical |
| `Qwen/Qwen3-8B-MLX-4bit` | `383413e909f3` | `aeb13307a71a` | identical |
| `Qwen/Qwen3-8B-MLX-6bit` | `dccec1206305` | `aeb13307a71a` | identical |
| `Qwen/Qwen3-8B-MLX-8bit` | `6a20f8c9329f` | `aeb13307a71a` | identical |
| `Qwen/Qwen3-8B-MLX-bf16` | `6766fd4b8101` | `aeb13307a71a` | identical |
| `Qwen/Qwen3-Coder-30B-A3B-Instruct` | `b2cff646eb4b` | `19564a48c4f7` | equivalent (serialisation) |
| `Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8` | `dcaee4d4dfc5` | `aeb13307a71a` | identical |
| `Qwen/Qwen3-Coder-480B-A35B-Instruct` | `9d90cf8fca1b` | `19564a48c4f7` | equivalent (serialisation) |
| `Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8` | `003f183a92fb` | `aeb13307a71a` | identical |
| `Qwen/Qwen3-Coder-Next` | `a7fbcb5c0e12` | `19564a48c4f7` | equivalent (serialisation) |
| `Qwen/Qwen3-Coder-Next-Base` | `1b6df59d5f75` | `19564a48c4f7` | equivalent (serialisation) |
| `Qwen/Qwen3-Coder-Next-FP8` | `da6e2ed27304` | `19564a48c4f7` | equivalent (serialisation) |
| `Qwen/Qwen3-Next-80B-A3B-Instruct` | `9c7f2fbe8446` | `aeb13307a71a` | identical |
| `Qwen/Qwen3-Next-80B-A3B-Instruct-FP8` | `c5f5f263bdd5` | `aeb13307a71a` | identical |
| `Qwen/Qwen3-Next-80B-A3B-Thinking` | `e502dd4100cc` | `aeb13307a71a` | identical |
| `Qwen/Qwen3-Next-80B-A3B-Thinking-FP8` | `1a28d48a94e7` | `aeb13307a71a` | identical |
| `Qwen/Qwen3-Reranker-0.6B` | `e61197ed4502` | `aeb13307a71a` | identical |
| `Qwen/Qwen3-Reranker-4B` | `22e683669bc0` | `aeb13307a71a` | identical |
| `Qwen/Qwen3-Reranker-8B` | `77d193c791ed` | `aeb13307a71a` | identical |
| `Qwen/Qwen3-VL-235B-A22B-Instruct` | `710c13861be6` | `a5d85b6dcc53` | equivalent (serialisation) |
| `Qwen/Qwen3-VL-235B-A22B-Instruct-FP8` | `7fbcd8c9e2ad` | `ba85e4e5222d` | equivalent (serialisation) |
| `Qwen/Qwen3-VL-235B-A22B-Thinking` | `6664affde684` | `a5d85b6dcc53` | equivalent (serialisation) |
| `Qwen/Qwen3-VL-235B-A22B-Thinking-FP8` | `4208801232c8` | `ba85e4e5222d` | equivalent (serialisation) |
| `Qwen/Qwen3-VL-2B-Instruct` | `89644892e4d8` | `a5d85b6dcc53` | equivalent (serialisation) |
| `Qwen/Qwen3-VL-2B-Instruct-FP8` | `46485250d885` | `ba85e4e5222d` | equivalent (serialisation) |
| `Qwen/Qwen3-VL-2B-Thinking` | `33e0ad94c327` | `a5d85b6dcc53` | equivalent (serialisation) |
| `Qwen/Qwen3-VL-2B-Thinking-FP8` | `bc71e10812c1` | `ba85e4e5222d` | equivalent (serialisation) |
| `Qwen/Qwen3-VL-30B-A3B-Instruct` | `9c4b90e1e4ba` | `a5d85b6dcc53` | equivalent (serialisation) |
| `Qwen/Qwen3-VL-30B-A3B-Instruct-FP8` | `d9748a51ae66` | `ba85e4e5222d` | equivalent (serialisation) |
| `Qwen/Qwen3-VL-30B-A3B-Thinking` | `d0ed0380729b` | `a5d85b6dcc53` | equivalent (serialisation) |
| `Qwen/Qwen3-VL-30B-A3B-Thinking-FP8` | `cf3058e421b6` | `ba85e4e5222d` | equivalent (serialisation) |
| `Qwen/Qwen3-VL-32B-Instruct` | `0cfaf48183f5` | `a5d85b6dcc53` | equivalent (serialisation) |
| `Qwen/Qwen3-VL-32B-Instruct-FP8` | `4bf2c2f39c37` | `ba85e4e5222d` | equivalent (serialisation) |
| `Qwen/Qwen3-VL-32B-Thinking` | `7edd10ffd119` | `a5d85b6dcc53` | equivalent (serialisation) |
| `Qwen/Qwen3-VL-32B-Thinking-FP8` | `3eee143c9b35` | `ba85e4e5222d` | equivalent (serialisation) |
| `Qwen/Qwen3-VL-4B-Instruct` | `ebb281ec70b0` | `a5d85b6dcc53` | equivalent (serialisation) |
| `Qwen/Qwen3-VL-4B-Instruct-FP8` | `fefbb44cbcce` | `ba85e4e5222d` | equivalent (serialisation) |
| `Qwen/Qwen3-VL-4B-Thinking` | `1de27d8c51f1` | `a5d85b6dcc53` | equivalent (serialisation) |
| `Qwen/Qwen3-VL-4B-Thinking-FP8` | `219b8e195ea3` | `ba85e4e5222d` | equivalent (serialisation) |
| `Qwen/Qwen3-VL-8B-Instruct` | `0c351dd01ed8` | `a5d85b6dcc53` | equivalent (serialisation) |
| `Qwen/Qwen3-VL-8B-Instruct-FP8` | `9cdc6310a8cb` | `ba85e4e5222d` | equivalent (serialisation) |
| `Qwen/Qwen3-VL-8B-Thinking` | `92f3c4b4fead` | `a5d85b6dcc53` | equivalent (serialisation) |
| `Qwen/Qwen3-VL-8B-Thinking-FP8` | `a6638e84662f` | `ba85e4e5222d` | equivalent (serialisation) |
| `Qwen/Qwen3-VL-Reranker-2B` | `4bd860ac4f15` | `aeb13307a71a` | identical |
| `Qwen/Qwen3-VL-Reranker-8B` | `b212dc8c91a8` | `aeb13307a71a` | identical |

#### `qwen3_5_08b` (19 repositories)

| Repository | Revision at snapshot | tokenizer.json sha256 | Basis |
|---|---|---|---|
| `Qwen/Qwen3.5-122B-A10B` | `dc4d348443bc` | `5f9e4d4901a9` | identical |
| `Qwen/Qwen3.5-122B-A10B-FP8` | `a099dee70ccf` | `5f9e4d4901a9` | identical |
| `Qwen/Qwen3.5-122B-A10B-GPTQ-Int4` | `30cd92cba970` | `5f9e4d4901a9` | identical |
| `Qwen/Qwen3.5-27B` | `fc05daec18b0` | `5f9e4d4901a9` | identical |
| `Qwen/Qwen3.5-27B-FP8` | `97f5941bf617` | `5f9e4d4901a9` | identical |
| `Qwen/Qwen3.5-27B-GPTQ-Int4` | `8f0c09f227ae` | `5f9e4d4901a9` | identical |
| `Qwen/Qwen3.5-2B` | `15852e8c1636` | `5f9e4d4901a9` | identical |
| `Qwen/Qwen3.5-35B-A3B` | `59d61f3ce65a` | `5f9e4d4901a9` | identical |
| `Qwen/Qwen3.5-35B-A3B-FP8` | `9d1823d2dee6` | `5f9e4d4901a9` | identical |
| `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4` | `3af5ca2972fa` | `5f9e4d4901a9` | identical |
| `Qwen/Qwen3.5-397B-A17B` | `8472618112ab` | `5f9e4d4901a9` | identical |
| `Qwen/Qwen3.5-397B-A17B-FP8` | `ea5b4f81096f` | `5f9e4d4901a9` | identical |
| `Qwen/Qwen3.5-397B-A17B-GPTQ-Int4` | `df333de344e0` | `5f9e4d4901a9` | identical |
| `Qwen/Qwen3.5-4B` | `851bf6e806ef` | `5f9e4d4901a9` | identical |
| `Qwen/Qwen3.5-9B` | `c20223623576` | `5f9e4d4901a9` | identical |
| `Qwen/Qwen3.6-27B` | `6a9e13bd6fc8` | `5f9e4d4901a9` | identical |
| `Qwen/Qwen3.6-27B-FP8` | `e89b16ebf198` | `5f9e4d4901a9` | identical |
| `Qwen/Qwen3.6-35B-A3B` | `995ad96eacd9` | `5f9e4d4901a9` | identical |
| `Qwen/Qwen3.6-35B-A3B-FP8` | `95a723d08a94` | `5f9e4d4901a9` | identical |

#### `llama_3_1_8b` (13 repositories)

| Repository | Revision at snapshot | tokenizer.json sha256 | Basis |
|---|---|---|---|
| `meta-llama/Llama-3.1-405B` | `b906e4dc842a` | `76e48799b099` | identical |
| `meta-llama/Llama-3.1-405B-FP8` | `cb4739810a51` | `76e48799b099` | identical |
| `meta-llama/Llama-3.1-405B-Instruct` | `be673f326cab` | `79e3e522635f` | equivalent (canonicalisation) |
| `meta-llama/Llama-3.1-405B-Instruct-FP8` | `64a54b704768` | `79e3e522635f` | equivalent (canonicalisation) |
| `meta-llama/Llama-3.1-70B` | `349b2ddb53ce` | `76e48799b099` | identical |
| `meta-llama/Llama-3.1-70B-Instruct` | `1605565b47bb` | `79e3e522635f` | equivalent (canonicalisation) |
| `meta-llama/Llama-3.1-8B-Instruct` | `0e9e39f249a1` | `79e3e522635f` | equivalent (canonicalisation) |
| `nvidia/Llama-3.1-Nemotron-70B-Instruct-HF` | `031d4042f36a` | `79e3e522635f` | equivalent (canonicalisation) |
| `nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1` | `d552708a9d57` | `6b9e4e7fb171` | equivalent (serialisation) |
| `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` | `54641c1611fc` | `6b9e4e7fb171` | equivalent (serialisation) |
| `nvidia/Llama-3_1-Nemotron-Ultra-253B-v1` | `5b47def5b895` | `6b9e4e7fb171` | equivalent (serialisation) |
| `nvidia/Llama-3_3-Nemotron-Super-49B-v1` | `387156d8d686` | `6b9e4e7fb171` | equivalent (serialisation) |
| `nvidia/Llama-3_3-Nemotron-Super-49B-v1_5` | `420ba7d28211` | `6b9e4e7fb171` | equivalent (serialisation) |

#### `deepseek_v3` (2 repositories)

| Repository | Revision at snapshot | tokenizer.json sha256 | Basis |
|---|---|---|---|
| `deepseek-ai/DeepSeek-V3-0324` | `e9b33add7688` | `621ac2e32d0d` | identical |
| `deepseek-ai/DeepSeek-V3-Base` | `afb92e1fa402` | `621ac2e32d0d` | identical |

#### `deepseek_v4_flash` (6 repositories)

| Repository | Revision at snapshot | tokenizer.json sha256 | Basis |
|---|---|---|---|
| `deepseek-ai/DeepSeek-V4-Flash-0731` | `7872f01b1d1f` | `8f9f37ca37fd` | identical |
| `deepseek-ai/DeepSeek-V4-Flash-Base` | `8855555deef2` | `8f9f37ca37fd` | identical |
| `deepseek-ai/DeepSeek-V4-Flash-DSpark` | `62af8fffb2f7` | `8f9f37ca37fd` | identical |
| `deepseek-ai/DeepSeek-V4-Pro` | `b5968e9190ef` | `8f9f37ca37fd` | identical |
| `deepseek-ai/DeepSeek-V4-Pro-Base` | `98730c030fbd` | `8f9f37ca37fd` | identical |
| `deepseek-ai/DeepSeek-V4-Pro-DSpark` | `7c09739fd136` | `8f9f37ca37fd` | identical |

#### `gpt_oss_120b` (3 repositories)

| Repository | Revision at snapshot | tokenizer.json sha256 | Basis |
|---|---|---|---|
| `openai/gpt-oss-20b` | `6cee5e81ee83` | `0614fe83cada` | identical |
| `openai/gpt-oss-safeguard-120b` | `3c7391182603` | `0614fe83cada` | identical |
| `openai/gpt-oss-safeguard-20b` | `8a11e17b25c9` | `0614fe83cada` | identical |

#### `glm_5_2` (6 repositories)

| Repository | Revision at snapshot | tokenizer.json sha256 | Basis |
|---|---|---|---|
| `zai-org/GLM-4.7-Flash` | `7dd20894a642` | `19e773648cb4` | identical |
| `zai-org/GLM-5` | `4e6698ba8e85` | `19e773648cb4` | identical |
| `zai-org/GLM-5-FP8` | `4f96cc5eec29` | `19e773648cb4` | identical |
| `zai-org/GLM-5.1` | `26e1bd6e011f` | `19e773648cb4` | identical |
| `zai-org/GLM-5.1-FP8` | `f396cf805182` | `19e773648cb4` | identical |
| `zai-org/GLM-5.2-FP8` | `ba978f7d347e` | `19e773648cb4` | identical |

#### `minimax_m3` (1 repositories)

| Repository | Revision at snapshot | tokenizer.json sha256 | Basis |
|---|---|---|---|
| `MiniMaxAI/MiniMax-M3-MXFP8` | `c5454eb03678` | `bb1f1626cf01` | identical |

#### `ministral_3_8b` (2 repositories)

| Repository | Revision at snapshot | tokenizer.json sha256 | Basis |
|---|---|---|---|
| `mistralai/Ministral-3-14B-Instruct-2512-BF16` | `3cea74c1ebaf` | `d5f6046775b1` | identical |
| `mistralai/Ministral-3-3B-Instruct-2512-BF16` | `b6d637bef239` | `d5f6046775b1` | identical |

#### `nemotron_3_nano_4b` (19 repositories)

| Repository | Revision at snapshot | tokenizer.json sha256 | Basis |
|---|---|---|---|
| `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16` | `97ab8012882a` | `623c34567aeb` | identical |
| `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | `2d59de1cbd51` | `c6021eb6847e` | equivalent (canonicalisation) |
| `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8` | `f8dc1c0afee9` | `c6021eb6847e` | equivalent (canonicalisation) |
| `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4` | `ce1b118ae66e` | `c6021eb6847e` | equivalent (canonicalisation) |
| `nvidia/NVIDIA-Nemotron-3-Nano-4B-FP8` | `3fe6dab75665` | `623c34567aeb` | identical |
| `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-Base-BF16` | `46cc6113d364` | `623c34567aeb` | identical |
| `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` | `d51eab0d1f97` | `623c34567aeb` | identical |
| `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8` | `7d7e5797b8a3` | `623c34567aeb` | identical |
| `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` | `4f0cf9daaeb7` | `623c34567aeb` | identical |
| `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-Base-BF16` | `1493b9c9ca30` | `623c34567aeb` | identical |
| `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16` | `624ba927cfbe` | `623c34567aeb` | identical |
| `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-GenRM` | `116af7cb1a23` | `623c34567aeb` | identical |
| `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4` | `183968f87ae4` | `623c34567aeb` | identical |
| `nvidia/NVIDIA-Nemotron-Labs-3-Elastic-30B-A3B-BF16` | `4bcc4326801e` | `623c34567aeb` | identical |
| `nvidia/NVIDIA-Nemotron-Labs-3-Elastic-30B-A3B-FP8` | `0515e708cb8c` | `623c34567aeb` | identical |
| `nvidia/NVIDIA-Nemotron-Labs-3-Elastic-30B-A3B-NVFP4` | `a8b6f7c6dba5` | `623c34567aeb` | identical |
| `nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-BF16` | `8fe5546888e9` | `623c34567aeb` | identical |
| `nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-FP8` | `e437e1e77cfe` | `623c34567aeb` | identical |
| `nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-NVFP4` | `1d370e47fbc5` | `623c34567aeb` | identical |

#### `olmo_3_7b` (9 repositories)

| Repository | Revision at snapshot | tokenizer.json sha256 | Basis |
|---|---|---|---|
| `allenai/Olmo-3-1125-32B` | `c2b61dae89a1` | `73fd5254624f` | identical |
| `allenai/Olmo-3-32B-Think` | `f2edda15216e` | `73fd5254624f` | identical |
| `allenai/Olmo-3-32B-Think-DPO` | `97604023f260` | `73fd5254624f` | identical |
| `allenai/Olmo-3-32B-Think-SFT` | `a6d7f3cf497c` | `ae12cb7a47a4` | equivalent (serialisation) |
| `allenai/Olmo-3-7B-Think` | `d97e442d7cc6` | `73fd5254624f` | identical |
| `allenai/Olmo-3-7B-Think-DPO` | `7b18bf927b43` | `73fd5254624f` | identical |
| `allenai/Olmo-3-7B-Think-SFT` | `6ff857587e04` | `ae12cb7a47a4` | equivalent (serialisation) |
| `allenai/Olmo-3.1-32B-Think` | `832c3f543499` | `73fd5254624f` | identical |
| `allenai/Olmo-3.1-7B-RL-Zero-Math` | `364820b002fc` | `73fd5254624f` | identical |

#### `hy3` (1 repositories)

| Repository | Revision at snapshot | tokenizer.json sha256 | Basis |
|---|---|---|---|
| `tencent/Hy3-FP8` | `ecc1d8e194e0` | `446e0b59cd94` | identical |

#### `kimi_k3` (12 repositories)

| Repository | Revision at snapshot | tiktoken.model sha256 | Basis |
|---|---|---|---|
| `moonshotai/Kimi-K2-Base` | `ce72df012259` | `b6c497a7469b` | identical (source file) |
| `moonshotai/Kimi-K2-Instruct` | `fd1984e2b7a3` | `b6c497a7469b` | identical (source file) |
| `moonshotai/Kimi-K2-Instruct-0905` | `ac6c49f04883` | `b6c497a7469b` | identical (source file) |
| `moonshotai/Kimi-K2-Thinking` | `a51ccc050d73` | `b6c497a7469b` | identical (source file) |
| `moonshotai/Kimi-K2.5` | `4d01dfe0332d` | `b6c497a7469b` | identical (source file) |
| `moonshotai/Kimi-K2.6` | `7eb5002f6aad` | `b6c497a7469b` | identical (source file) |
| `moonshotai/Kimi-K2.7-Code` | `74797c9c6237` | `b6c497a7469b` | identical (source file) |
| `moonshotai/Kimi-Linear-48B-A3B-Base` | `3b171c17bfc4` | `b6c497a7469b` | identical (source file) |
| `moonshotai/Kimi-Linear-48B-A3B-Instruct` | `e1df551a4471` | `b6c497a7469b` | identical (source file) |
| `moonshotai/Kimi-VL-A3B-Instruct` | `398eede0903c` | `b6c497a7469b` | identical (source file) |
| `moonshotai/Kimi-VL-A3B-Thinking` | `7d99e220af61` | `b6c497a7469b` | identical (source file) |
| `moonshotai/Kimi-VL-A3B-Thinking-2506` | `aa1730989e75` | `b6c497a7469b` | identical (source file) |

#### `bert_cased` (4 repositories)

| Repository | Revision at snapshot | tokenizer.json sha256 | Basis |
|---|---|---|---|
| `google-bert/bert-base-cased-finetuned-mrpc` | `f150c1d609d1` | `5b3360be30cd` | identical |
| `google-bert/bert-large-cased` | `06fa25dc1b6e` | `a17c4dbf7a87` | equivalent (canonicalisation) |
| `google-bert/bert-large-cased-whole-word-masking` | `4110364c3591` | `5b3360be30cd` | identical |
| `google-bert/bert-large-cased-whole-word-masking-finetuned-squad` | `de462ace242f` | `a17c4dbf7a87` | equivalent (canonicalisation) |

#### `bert_uncased` (3 repositories)

| Repository | Revision at snapshot | tokenizer.json sha256 | Basis |
|---|---|---|---|
| `google-bert/bert-large-uncased` | `6da4b6a26a18` | `ce64fce797c2` | identical |
| `google-bert/bert-large-uncased-whole-word-masking` | `bf1420893378` | `ce64fce797c2` | identical |
| `google-bert/bert-large-uncased-whole-word-masking-finetuned-squad` | `979de3ccf2f3` | `ce64fce797c2` | identical |

### Recorded and not counted

The same sweep produced four outcomes that are not coverage. They are listed
because a support matrix that only reports its successes cannot be checked.

| Outcome | Repositories | Why it is not in the list above |
|---|---|---|
| core-equal, added-token face | 77 | The four core segments equal a certified artifact; the added-token table or the chat-template post-processor differs. Adoption needs the added-token table re-exported for that repository, not a new campaign. |
| different | 83 | Core segments differ. Some are near misses — same pre-tokenizer and same merges, with a handful of in-vocabulary special slots renamed — but a renamed slot changes token ids, so the certificate does not carry over. |
| no comparable tokenizer file | 32 | The repository ships `tekken.json`, a bare `tiktoken.model`, or configuration only. GGUF repositories are excluded throughout, since the tokenizer is embedded in the container. |
| gated, unverified | 13 | The file face was not retrievable with the credentials available at the snapshot. Unresolved, not assumed. |

The thirteen gated repositories are named, because "unverified" is only honest
if it is specific. All are `meta-llama` Llama 3.2 and 3.3 repositories:
`Llama-3.2-1B`, `Llama-3.2-1B-Instruct`,
`Llama-3.2-1B-Instruct-QLORA_INT4_EO8`,
`Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8`, `Llama-3.2-3B`,
`Llama-3.2-3B-Instruct`, `Llama-3.2-3B-Instruct-QLORA_INT4_EO8`,
`Llama-3.2-3B-Instruct-SpinQuant_INT4_EO8`, `Llama-3.2-11B-Vision`,
`Llama-3.2-11B-Vision-Instruct`, `Llama-3.2-90B-Vision`,
`Llama-3.2-90B-Vision-Instruct` and `Llama-3.3-70B-Instruct`. Their
relationship to `llama_3_1_8b` is plausible and unmeasured; the comparison is
pending access and no claim is made in the meantime. Re-running the sweep with
access resolves these thirteen rows and nothing else.

Two artifacts in the family matrix above, `ling_3_0_flash` and
`laguna_s_2_1`, were admitted after this snapshot and were therefore not part
of the sweep. No sub-versions are listed for them. The reconnaissance that
preceded their admission did note byte-identical quantised twins for both, but
that observation was not carried through the revision-and-digest bookkeeping
this section reports, so nothing is listed and nothing is counted.

A further 39 repositories were examined and fall outside the certified set by
design: the `google/gemma-4-*` line and `FacebookAI/xlm-roberta-*` are the
documented structural exclusions below, and the `openai-community/gpt2` line is
a benchmark reference rather than a certified family.

The accounting closes on the snapshot total: 470 repositories examined = 16
anchors + 210 verified sub-versions + 77 core-equal + 83 different + 32 without
a comparable file + 13 gated + 39 out of scope.

### What the snapshot is and is not

The comparison is dated. Upstream tokenizer files are not immutable: in the six
months before the snapshot, 66 commits in the audited repositories modified an
already-published `tokenizer.json`. Most were same-week release clean-ups, but
at least one arrived eight months after the model shipped, replacing two
in-vocabulary tokens while sibling repositories kept the old values. Two anchor
repositories, `MiniMaxAI/MiniMax-M3` and `tencent/Hy3`, had themselves advanced
past their pinned revision by the snapshot, in both cases without changing the
tokenizer bytes.

Two consequences, both already reflected in how this package works. Coverage is
bound to a repository *and* a revision *and* a file digest, which is why the
tables above carry all three. And a row here is a statement about what was
observed on 2026-08-04, not a promise about the repository's current contents:
what the package actually enforces is the digest comparison it performs at load
time, which is unaffected by anything upstream does afterwards.

## Documented exclusions

| Subject | Reason |
|---|---|
| `google/gemma-4-*` | The pipeline provides no pre-tokenization boundaries, so the boundary condition the accelerated path relies on has nothing to attach to. Listed for completeness; the reference path handles these normally through Hugging Face. |
| `FacebookAI/xlm-roberta-*` | Unigram model, outside the BPE/WordPiece scope of the current certification. |

These are structural exclusions rather than gaps awaiting a campaign, and they
are stated here so that the boundary of the claim is visible without reading the
registry.

## How coverage is extended

A new family needs a certification campaign: splitter pattern table, boundary
predicate check, differential fuzzing, replay campaigns, and — where the
pre-tokenizer structure matches an existing kernel group — GPU kernel support.
Families under reconnaissance are listed in [`../ROADMAP.md`](../ROADMAP.md).

<!--
Placeholder keys used in this file (registry: README.md):
  number  support.repos_audited
  number  support.sha.{family}              — 15 BPE rows + 3 WordPiece rows
  pending support.status.{family}.cpu       — per family, reference/CPU backend
  pending support.status.{family}.gpu       — per family, GPU backend
  pending support.repo_revision.{family}    — generated table only
  pending support.audit_snapshot_date
-->
