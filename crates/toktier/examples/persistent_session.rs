use toktier::{Device, Runtime};

fn main() -> toktier::Result<()> {
    let home = tempfile::tempdir()?;
    let history = "persistent multilingual agent history 中🙂. ".repeat(256);
    let delta = " first turn";

    let before = {
        let runtime = Runtime::builder()
            .home(home.path())
            .device(Device::Cpu)
            .build()?;
        let tokenizer = runtime.load("qwen3_8b")?;
        let mut session = tokenizer.open_session("agent-42")?;
        let seed = session.seed(&history)?;
        let mut ids = seed.ids().to_vec();
        let patch = session.append(delta)?;
        ids.truncate(patch.keep_tokens() as usize);
        ids.extend_from_slice(patch.replacement_ids());
        session.close()?;
        ids
    };

    let runtime = Runtime::builder()
        .home(home.path())
        .device(Device::Cpu)
        .build()?;
    let tokenizer = runtime.load("qwen3_8b")?;
    let mut session = tokenizer.open_session("agent-42")?;
    let patch = session.append(" after restart")?;
    let mut after = before;
    after.truncate(patch.keep_tokens() as usize);
    after.extend_from_slice(patch.replacement_ids());

    let expected = tokenizer.encode(&format!("{history}{delta} after restart"))?;
    assert_eq!(after, expected.ids());
    let fork = session.fork("agent-42-branch")?;
    assert_eq!(fork.snapshot()?.ids(), after);
    println!("restart/fork exact: {} ids", after.len());
    Ok(())
}
