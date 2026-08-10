#[cfg(feature = "jit")]
fn main() -> toktier::Result<()> {
    use toktier::{Device, GpuDelivery, Policy, Runtime};

    let cache = std::env::var_os("TOKTIER_ARTIFACT_CACHE")
        .map(std::path::PathBuf::from)
        .ok_or_else(|| std::io::Error::other("set TOKTIER_ARTIFACT_CACHE"))?;
    let runtime = Runtime::builder()
        .artifact_cache(cache)
        .device(Device::Cuda(0))
        .gpu_delivery(GpuDelivery::Jit)
        .gpu_min_bytes(0)
        .policy(Policy::Experimental)
        .accept_uncertified_jit(true)?
        .build()?;
    let tokenizer = runtime.load("qwen3_8b")?;
    let reference = Runtime::builder()
        .artifact_cache(
            std::env::var_os("TOKTIER_ARTIFACT_CACHE")
                .ok_or_else(|| std::io::Error::other("set TOKTIER_ARTIFACT_CACHE"))?,
        )
        .device(Device::Cpu)
        .policy(Policy::Reference)
        .build()?
        .load("qwen3_8b")?;
    let text = "TokTier direct Rust JIT probe — 你好";
    let encoding = tokenizer.encode(text)?;
    assert_eq!(encoding.ids(), reference.encode(text)?.ids());
    println!("ids={:?}", encoding.ids());
    println!("execution={:?}", encoding.execution());
    println!("jit={:#?}", tokenizer.jit_facts());
    Ok(())
}

#[cfg(not(feature = "jit"))]
fn main() {
    eprintln!("build with --features jit");
}
