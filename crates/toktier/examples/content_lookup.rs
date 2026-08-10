use toktier::{Device, Runtime};

fn main() -> toktier::Result<()> {
    let tokenizer = Runtime::builder()
        .device(Device::Cpu)
        .build()?
        .load("qwen3_8b")?;
    let text = "shared prefix ".repeat(2_000);
    tokenizer.open_session("producer")?.seed(&text)?;
    let hit = tokenizer.lookup(&text)?.expect("seeded content must hit");
    assert_eq!(hit.ids(), tokenizer.encode(&text)?.ids());
    println!("content hit: {} ids", hit.ids().len());
    Ok(())
}
