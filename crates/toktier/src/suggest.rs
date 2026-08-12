//! Close-match suggestions for a mistyped family id.
//!
//! The Python facade answers an unknown family with the closest valid
//! ids, computed by `difflib.get_close_matches(word, families, n=3)`
//! (cutoff 0.6). The Rust facade answers the same question, so it
//! computes the same ranking rather than an approximation of it: the
//! similarity below is `difflib.SequenceMatcher.ratio` -- twice the
//! total size of the matching blocks over the combined length -- with
//! no junk heuristic, which is what `get_close_matches` uses for
//! inputs this short (autojunk needs a 200-element sequence).
//!
//! Ties follow Python's `heapq.nlargest` over `(ratio, candidate)`
//! pairs: equal ratios rank the lexicographically larger id first.

/// Ratio below which a candidate is not offered at all.
const CUTOFF: f64 = 0.6;

/// How many suggestions the message may carry.
const LIMIT: usize = 3;

/// The closest candidates to `word`, best first, at most [`LIMIT`].
pub(crate) fn close_matches<'a>(
    word: &str,
    candidates: impl IntoIterator<Item = &'a str>,
) -> Vec<&'a str> {
    let target = word.chars().collect::<Vec<_>>();
    let mut scored = candidates
        .into_iter()
        .filter_map(|candidate| {
            let ratio = ratio(&candidate.chars().collect::<Vec<_>>(), &target);
            (ratio >= CUTOFF).then_some((ratio, candidate))
        })
        .collect::<Vec<_>>();
    // Descending by ratio, then by candidate, matching `nlargest` over
    // the same tuples. Ratios come from identical arithmetic on both
    // sides, so equal scores compare equal here as they do there.
    scored.sort_by(|left, right| {
        right
            .0
            .partial_cmp(&left.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| right.1.cmp(left.1))
    });
    scored.truncate(LIMIT);
    scored.into_iter().map(|(_, candidate)| candidate).collect()
}

/// `difflib.SequenceMatcher(None, a, b).ratio()`.
fn ratio(a: &[char], b: &[char]) -> f64 {
    let total = a.len() + b.len();
    if total == 0 {
        return 1.0;
    }
    2.0 * matched(a, b) as f64 / total as f64
}

/// Total size of the matching blocks, the `M` in `2M / T`.
fn matched(a: &[char], b: &[char]) -> usize {
    let mut total = 0;
    let mut queue = vec![(0, a.len(), 0, b.len())];
    while let Some((alo, ahi, blo, bhi)) = queue.pop() {
        let (i, j, size) = longest_match(a, b, alo, ahi, blo, bhi);
        if size == 0 {
            continue;
        }
        total += size;
        if alo < i && blo < j {
            queue.push((alo, i, blo, j));
        }
        if i + size < ahi && j + size < bhi {
            queue.push((i + size, ahi, j + size, bhi));
        }
    }
    total
}

/// The earliest longest matching block in `a[alo..ahi]`/`b[blo..bhi]`,
/// returned as `(i, j, size)`; the same dynamic program difflib runs,
/// including its preference for the earliest `i` then the earliest `j`.
fn longest_match(
    a: &[char],
    b: &[char],
    alo: usize,
    ahi: usize,
    blo: usize,
    bhi: usize,
) -> (usize, usize, usize) {
    let (mut best_i, mut best_j, mut best_size) = (alo, blo, 0usize);
    // Run lengths ending at each `j`, carried one `i` at a time.
    let mut previous = vec![0usize; b.len() + 1];
    let mut current = vec![0usize; b.len() + 1];
    for (i, left) in a.iter().enumerate().take(ahi).skip(alo) {
        current[blo..bhi].fill(0);
        for j in blo..bhi {
            if *left != b[j] {
                continue;
            }
            // Only run lengths from inside the window carry over, which
            // is what difflib's per-window `j2len` map holds.
            let run = if j == blo { 1 } else { previous[j - 1] + 1 };
            current[j] = run;
            if run > best_size {
                best_i = i + 1 - run;
                best_j = j + 1 - run;
                best_size = run;
            }
        }
        std::mem::swap(&mut previous, &mut current);
    }
    (best_i, best_j, best_size)
}

#[cfg(test)]
mod tests {
    use super::{close_matches, ratio};

    fn score(a: &str, b: &str) -> f64 {
        ratio(
            &a.chars().collect::<Vec<_>>(),
            &b.chars().collect::<Vec<_>>(),
        )
    }

    #[test]
    fn ratio_matches_the_python_reference_values() {
        // Values read from difflib.SequenceMatcher(None, a, b).ratio().
        let cases = [
            ("qwen3_8b", "qwen3-8b", 0.875),
            ("qwen3_8b", "qwen3_8b", 1.0),
            ("llama_31_8b", "llama_3_8b", 0.9523809523809523),
            ("gpt_oss_120b", "kimi_k3", 0.10526315789473684),
            ("abcd", "dcba", 0.25),
            ("", "abc", 0.0),
        ];
        for (left, right, expected) in cases {
            let observed = score(left, right);
            assert!(
                (observed - expected).abs() < 1e-12,
                "{left} vs {right}: {observed} != {expected}"
            );
        }
    }

    #[test]
    fn suggestions_are_ranked_and_capped() {
        let families = [
            "bert_base_cased",
            "deepseek_v3",
            "gpt_oss_120b",
            "kimi_k3",
            "llama_31_8b",
            "qwen3_8b",
        ];
        assert_eq!(close_matches("qwen3-8b", families), vec!["qwen3_8b"]);
        assert_eq!(close_matches("llama31_8b", families), vec!["llama_31_8b"]);
        assert!(close_matches("zzzzzzzz", families).is_empty());
        assert!(close_matches("bert_base", families).len() <= 3);
    }

    #[test]
    fn the_shipped_roster_answers_typos_the_way_python_does() {
        // Expectations produced by difflib.get_close_matches(query,
        // families, n=3) over the shipped family ids.
        let families = [
            "deepseek_v3",
            "deepseek_v4_flash",
            "glm_5_2",
            "gpt_oss_120b",
            "hy3",
            "kimi_k3",
            "laguna_s_2_1",
            "ling_3_0_flash",
            "llama_3_1_8b",
            "minimax_m3",
            "ministral_3_8b",
            "nemotron_3_nano_4b",
            "olmo_3_7b",
            "qwen3_5_08b",
            "qwen3_8b",
        ];
        let cases: [(&str, Vec<&str>); 22] = [
            ("qwen3-8b", vec!["qwen3_8b", "qwen3_5_08b"]),
            ("qwen38b", vec!["qwen3_8b", "qwen3_5_08b"]),
            ("Qwen3_8B", vec!["qwen3_8b", "qwen3_5_08b"]),
            ("llama_31_8b", vec!["llama_3_1_8b", "olmo_3_7b"]),
            ("llama3_1_8b", vec!["llama_3_1_8b"]),
            ("deepseek", vec!["deepseek_v3", "deepseek_v4_flash"]),
            ("deepseekv3", vec!["deepseek_v3", "deepseek_v4_flash"]),
            ("kimi", vec!["kimi_k3"]),
            ("kimi_k2", vec!["kimi_k3"]),
            ("gpt-oss-120b", vec!["gpt_oss_120b"]),
            ("minimax", vec!["minimax_m3"]),
            ("ministral", vec!["ministral_3_8b"]),
            ("nemotron", vec!["nemotron_3_nano_4b"]),
            ("olmo3_7b", vec!["olmo_3_7b"]),
            ("glm52", vec!["glm_5_2"]),
            ("hy", vec!["hy3"]),
            ("ling_3_0", vec!["ling_3_0_flash"]),
            ("laguna", vec!["laguna_s_2_1"]),
            ("zzzz", vec![]),
            ("qwen", vec!["qwen3_8b"]),
            ("qwen3_5_8b", vec!["qwen3_5_08b", "qwen3_8b"]),
            ("bert_base_cased", vec![]),
        ];
        for (query, expected) in cases {
            assert_eq!(close_matches(query, families), expected, "{query}");
        }
    }

    #[test]
    fn equal_scores_rank_the_larger_id_first() {
        // Both candidates score identically against the query, so the
        // order is decided the way heapq.nlargest decides it.
        let observed = close_matches("aab", ["aac", "aad"]);
        assert_eq!(observed, vec!["aad", "aac"]);
    }
}
