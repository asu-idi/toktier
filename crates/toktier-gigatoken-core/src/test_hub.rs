//! Local-only tokenizer fixture discovery for retained upstream tests.

use std::path::PathBuf;

fn cache_root() -> PathBuf {
    std::env::var_os("HF_HUB_CACHE")
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("HF_HOME")
                .map(PathBuf::from)
                .map(|path| path.join("hub"))
        })
        .or_else(|| {
            std::env::var_os("HOME")
                .map(PathBuf::from)
                .map(|path| path.join(".cache/huggingface/hub"))
        })
        .unwrap_or_default()
}

pub(crate) fn hf_tokenizer_json(repo_id: &str) -> Option<PathBuf> {
    let repo = cache_root().join(format!("models--{}", repo_id.replace('/', "--")));
    let revision = std::fs::read_to_string(repo.join("refs/main")).ok()?;
    let path = repo
        .join("snapshots")
        .join(revision.trim())
        .join("tokenizer.json");
    path.is_file().then_some(path)
}

pub(crate) fn gpt2_tokenizer_json() -> PathBuf {
    hf_tokenizer_json("openai-community/gpt2").unwrap_or_else(|| {
        let root = std::env::var_os("GIGATOKEN_UPSTREAM_ROOT")
            .map(PathBuf::from)
            .expect(
                "GPT-2 is not cached; set GIGATOKEN_UPSTREAM_ROOT to the audited upstream tree",
            );
        root.join("tests/fixtures/gpt2_tokenizer.json")
    })
}
