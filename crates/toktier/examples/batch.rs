use toktier::{Device, Runtime};

fn main() -> toktier::Result<()> {
    let tokenizer = Runtime::builder()
        .device(Device::Cpu)
        .build()?
        .load("qwen3_8b")?;
    let documents = ["alpha", "beta gamma", "你好"];
    let batch = tokenizer.encode_batch(&documents)?;
    for row in 0..batch.len() {
        println!("row {row}: {:?}", batch.row(row)?);
    }
    println!(
        "values={} offsets={:?}",
        batch.values().len(),
        batch.offsets()
    );
    Ok(())
}
