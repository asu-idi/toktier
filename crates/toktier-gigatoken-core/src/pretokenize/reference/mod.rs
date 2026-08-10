//! Reference pretokenizers: superseded designs kept as test oracles.
//!
//! Nothing in here runs in the library's encode path — the production
//! pretokenizers are the mask scanners in [`super::fast`] (see
//! `pretokenize_as_iter` / `PretokenizerType`). These implementations are
//! retained as independent oracles for the differential tests behind the
//! `upstream-tests` feature:
//!
//! - [`state_machine`]: the byte-class DFA, the original correctness
//!   reference (also exercised from other modules' tests).
//! - [`combinator`]: winnow parser-combinator implementation.
//! - [`simd`]: first-generation portable-SIMD prototype.

// The superseded baselines keep their full vendored surface; only the
// `upstream-tests` differential tier exercises every entry point.
#[allow(dead_code)]
pub mod combinator;
#[allow(dead_code)]
pub mod simd;
pub mod state_machine;
