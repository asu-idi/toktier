use toktier::{Device, Runtime};

fn main() -> toktier::Result<()> {
    let tokenizer = Runtime::builder()
        .device(Device::Cpu)
        .build()?
        .load("qwen3_8b")?;
    let mut session = tokenizer.open_session("agent-42")?;
    let seed = session.seed("The agent considered an internation")?;
    let mut downstream = seed.ids().to_vec();
    let patch = session.append("alization proposal. 你好")?;
    downstream.truncate(patch.keep_tokens() as usize);
    downstream.extend_from_slice(patch.replacement_ids());
    assert_eq!(downstream, session.snapshot()?.ids());
    println!(
        "revision={} keep={} replacement={} total={}",
        patch.revision(),
        patch.keep_tokens(),
        patch.replacement_ids().len(),
        patch.token_count()
    );
    Ok(())
}
