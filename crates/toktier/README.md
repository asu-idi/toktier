# TokTier for Rust

`toktier` is the Python-free serving API for
[TokTier](https://github.com/asu-idi/toktier). It exposes the same verified
artifacts, corrected CPU repair, prebuilt CUDA delivery, routing policy, and
state store without a GIL or Python-shaped token collection.

```rust,no_run
use toktier::{ArtifactManager, Device, Revision, Runtime};

fn main() -> toktier::Result<()> {
    let artifacts = ArtifactManager::builder()
        .cache("/var/cache/toktier/artifacts")
        .build()?;
    let runtime = Runtime::builder()
        .artifacts(artifacts)
        .device(Device::Auto)
        .build()?;
    let tokenizer = runtime.from_pretrained(
        "Qwen/Qwen3-8B",
        Revision::commit("b968826d9c46dd6066d109eabc6255188de91218")?,
    )?;
    let encoded = tokenizer.encode("hello world")?;
    println!("{:?}", encoded.ids());

    let mut session = tokenizer.open_session("agent-42")?;
    let seed = session.seed("user: hello\n")?;
    let mut downstream = seed.ids().to_vec();
    let patch = session.append("assistant: hi\n")?;
    downstream.truncate(patch.keep_tokens() as usize);
    downstream.extend_from_slice(patch.replacement_ids());
    Ok(())
}
```

The crate is published on crates.io from 0.2.0 onward and tracks the package
version, so `cargo add toktier` resolves it from the registry; a local
checkout can also be consumed as a workspace/path dependency. Rust can fetch
an immutable revision, verify and mirror it, export/import the Python-v1
air-gap format, or operate with network access disabled. The optional `jit`
feature invokes an exact `nvcc` directly and loads its authenticated product
through the same Rust CUDA Driver host as prebuilt delivery; no shell, Python,
PyTorch, or Ninja participates.

The API returns immutable continuous `u32` buffers, a continuous
values/offsets representation for batches, and suffix-replacement patches for
agent appends. Accelerated paths are admitted only when the embedded registry
and the bytes observed in this process prove the complete binding. Otherwise
they fall to the frozen `tokenizers==0.22.2` reference.

See [the Rust API guide](https://github.com/asu-idi/toktier/blob/main/docs/rust-api.md),
the [lifecycle and distribution guide](https://github.com/asu-idi/toktier/blob/main/docs/rust-lifecycle.md),
and the runnable [`examples`](examples/).

Licensed under the [Apache License 2.0](LICENSE).
