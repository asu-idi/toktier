use toktier::{Device, GpuDelivery, Policy, Runtime};

fn main() -> toktier::Result<()> {
    let tokenizer = Runtime::builder()
        .device(Device::Cuda(0))
        .gpu_delivery(GpuDelivery::Prebuilt)
        .gpu_min_bytes(0)
        .build()?
        .load("qwen3_8b")?;
    let text = "TokTier on CUDA. 中🙂 ".repeat(8_192);
    let encoded = tokenizer.encode(&text)?;
    let reference = Runtime::builder()
        .device(Device::Cpu)
        .policy(Policy::Reference)
        .build()?
        .load("qwen3_8b")?;
    assert_eq!(encoded.ids(), reference.encode(&text)?.ids());
    println!(
        "gpu={:?} backend={:?} tokens={}",
        tokenizer.gpu_facts(),
        encoded.execution().backend,
        encoded.ids().len()
    );
    Ok(())
}
