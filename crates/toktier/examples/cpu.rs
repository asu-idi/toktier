use toktier::{DecodeOptions, Device, Runtime};

fn main() -> toktier::Result<()> {
    let runtime = Runtime::builder().device(Device::Cpu).build()?;
    let tokenizer = runtime.load("qwen3_8b")?;
    let encoded = tokenizer.encode("Hello from Rust. 你好！")?;
    println!("family={} ids={:?}", tokenizer.family(), encoded.ids());
    println!(
        "decoded={}",
        tokenizer.decode(encoded.token_buffer(), DecodeOptions::default())?
    );
    println!("route={:?}", tokenizer.plan().backends);
    Ok(())
}
