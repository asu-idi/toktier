use toktier::{Device, Policy, Runtime};

fn main() -> toktier::Result<()> {
    let family = std::env::var("TOKTIER_FAMILY").unwrap_or_else(|_| "qwen3_8b".to_owned());
    let certified = Runtime::builder()
        .device(Device::Cpu)
        .build()?
        .load(&family)?;
    let reference = Runtime::builder()
        .device(Device::Cpu)
        .policy(Policy::Reference)
        .build()?
        .load(&family)?;
    let cases = [
        String::new(),
        "hello world".to_owned(),
        " leading and trailing ".to_owned(),
        "中文🙂 café e\u{301} \r\n\t".to_owned(),
        "<|im_start|>user\nhello<|im_end|>".to_owned(),
        "a".repeat(65_537),
    ];
    let mut checks = 0usize;
    for text in cases {
        let fast = certified.encode(&text)?;
        let hf = reference.encode(&text)?;
        assert_eq!(fast.ids(), hf.ids(), "one-shot divergence");
        assert_eq!(
            certified.decode(fast.ids(), Default::default())?,
            reference.decode(hf.ids(), Default::default())?
        );
        checks += fast.ids().len();
    }

    let seed_text = "agent transcript 中🙂. ".repeat(1_024);
    let mut session = certified.open_session("campaign")?;
    let seed = session.seed(&seed_text)?;
    let mut reconstructed = seed.ids().to_vec();
    let mut complete = seed_text;
    let deltas = [
        " short".to_owned(),
        "\n".repeat(1_500),
        " final🙂".to_owned(),
    ];
    for delta in deltas {
        complete.push_str(&delta);
        let patch = session.append(&delta)?;
        reconstructed.truncate(patch.keep_tokens() as usize);
        reconstructed.extend_from_slice(patch.replacement_ids());
        let expected = reference.encode(&complete)?;
        assert_eq!(reconstructed, expected.ids(), "patch divergence");
        checks += reconstructed.len();
    }
    println!("{family}: {checks} exact-ID checks, zero divergence");
    Ok(())
}
