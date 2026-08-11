//! Audited Rust core of TokTier's corrected Gigatoken build.
//!
//! This is an internal supporting crate of TokTier, versioned with the
//! workspace and carrying no independent API stability promise; use the
//! `toktier` package for the supported Rust surface.
//!
//! The implementation is vendored from upstream commit
//! `34a15995fc930c3807cd176bfd8ee91c166ee2fe`, with TokTier's pinned repair
//! patch applied.  Python bindings, training, batch/file input, and unrelated
//! packaging code are deliberately excluded: this crate is the in-process
//! BPE engine used below TokTier's native router.

// Keep the audited upstream control flow and data layout recognizable. These
// lints point at intentional SIMD/index loops, tokenizer-construction argument
// sets, and a large precompiled-normalizer value; rewriting them solely for
// style would enlarge the downstream semantic delta we must certify.
#![allow(
    clippy::collapsible_if,
    clippy::large_enum_variant,
    clippy::needless_range_loop,
    clippy::nonminimal_bool,
    clippy::redundant_slicing,
    clippy::single_range_in_vec_init,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::unnecessary_cast
)]

mod input;
mod token;

#[cfg(all(test, feature = "upstream-tests"))]
mod test_hub;

pub mod bpe;
pub mod pretokenize;

pub mod load_tokenizer {
    pub mod hf;
    pub mod tiktoken;
}

pub use bpe::Tokenizer;
pub use load_tokenizer::hf::load_hf_bpe;
