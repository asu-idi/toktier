//! What version of its text semantics this binary actually carries.
//!
//! Three libraries under the reference engine decide where text is cut:
//! the `regex` crate's Unicode tables, the Oniguruma library `onig`
//! binds, and `unicode-segmentation`'s tables. Their correct behaviour is
//! defined by an evolving external standard, so a new package version of
//! one of them is not by itself a change of behaviour -- and a change of
//! behaviour is what can move ids. The certificate is about the second.
//!
//! The comparison has to happen here rather than in the build script.
//! A build script cannot link the libraries the crate being built links:
//! its own build dependencies are a separate resolution for the host,
//! possibly of other versions. So the build script hands over what it
//! knows -- the judged behaviour version of each unit, and whether the
//! package versions moved -- and this module asks the linked libraries
//! themselves.
//!
//! Fail-closed in every direction: a unit whose version cannot be read
//! here falls back to the build script's exact package comparison; a
//! record that names no unit at all leaves the whole core on that exact
//! comparison; and a package that resolves from somewhere other than the
//! judged one never reaches this module, because no probe of a library's
//! version can speak for a copy that is not the judged one.

use std::sync::OnceLock;

use crate::diagnostics::BehaviorVersion;

/// Separators of the one-line values the build script emits. Both are
/// ASCII separators, so neither can appear in a version or a report.
const FIELD_SEPARATOR: char = '\u{1f}';
const RECORD_SEPARATOR: char = '\u{1e}';

/// The Unicode versions `regex` can be asked about, newest first. The
/// engine accepts `\p{Age=X}` for every version its tables carry, so the
/// first one that compiles is the version of those tables.
const AGE_CANDIDATES: &[&str] = &[
    "23.0", "22.0", "21.0", "20.0", "19.0", "18.0", "17.0", "16.0", "15.1", "15.0", "14.0", "13.0",
    "12.1", "12.0", "11.0", "10.0",
];

/// One unit, as the record judged it and as this binary reads it.
struct Reading {
    unit: String,
    judged: String,
    source: String,
    observed: Option<String>,
}

/// The judged behaviour versions, read from the record the build script
/// carried over.
fn judged_readings() -> Vec<Reading> {
    let recorded = env!("TOKTIER_RUST_API_JUDGED_BEHAVIOR");
    if recorded.is_empty() {
        return Vec::new();
    }
    recorded
        .split(FIELD_SEPARATOR)
        .filter_map(|record| {
            let mut fields = record.split(RECORD_SEPARATOR);
            let unit = fields.next()?;
            let judged = fields.next()?;
            let source = fields.next()?;
            if unit.is_empty() || judged.is_empty() {
                return None;
            }
            Some(Reading {
                observed: observe(unit),
                unit: unit.to_owned(),
                judged: judged.to_owned(),
                source: source.to_owned(),
            })
        })
        .collect()
}

/// The package moves the build script saw for one unit, if any.
fn drift_for(unit: &str) -> Option<&'static str> {
    let recorded = env!("TOKTIER_RUST_API_R2_DRIFT");
    recorded.split(FIELD_SEPARATOR).find_map(|record| {
        let (name, report) = record.split_once(RECORD_SEPARATOR)?;
        (name == unit).then_some(report)
    })
}

/// The version of one unit as the libraries this binary links report it.
///
/// `None` for a unit this release has no way to ask, which is the same
/// answer as a probe that failed: either way the build script's exact
/// comparison is what is left.
fn observe(unit: &str) -> Option<String> {
    match unit {
        // The tables are not versioned in the API, but the engine
        // accepts `\p{Age=X}` exactly for the versions they carry, so
        // the newest one that compiles is their version.
        "regex" => AGE_CANDIDATES
            .iter()
            .find(|age| regex::Regex::new(&format!(r"\p{{Age={age}}}")).is_ok())
            .map(|age| (*age).to_owned()),
        // Oniguruma publishes no Unicode version of its own; its library
        // version is what its own history ties those tables to, and the
        // binding reports the library it actually linked.
        "onig" => Some(onig::version()).filter(|version| !version.is_empty()),
        "unicode-segmentation" => {
            let (major, minor, patch) = unicode_segmentation::UNICODE_VERSION;
            Some(format!("{major}.{minor}.{patch}"))
        }
        _ => None,
    }
}

/// How a unit says which version of its tables this binary carries.
fn carries(unit: &str, observed: &str) -> String {
    match unit {
        "regex" => format!("carries Unicode {observed} (regex tables)"),
        "onig" => format!("links Oniguruma {observed}"),
        "unicode-segmentation" => format!("reads segmentation tables {observed}"),
        other => format!("reads {other} {observed}"),
    }
}

/// How a unit says that a package move left its tables where they were.
fn unchanged(unit: &str, version: &str) -> String {
    match unit {
        "regex" => format!("Unicode data version unchanged ({version})"),
        "onig" => format!("Oniguruma library version unchanged ({version})"),
        "unicode-segmentation" => format!("segmentation table version unchanged ({version})"),
        other => format!("{other} behaviour version unchanged ({version})"),
    }
}

/// How to see, on one's own text, what the accelerated engines would do
/// differently from the reference engine in this binary.
///
/// The command takes the measurement and records it; it is the same one
/// the `supported_untested` label points at, and it is the only thing
/// this crate offers here, because whether a particular text is affected
/// is a question about that text and nobody else can answer it.
const COMPARE_HINT: &str = "To compare the accelerated engine with this binary's reference \
                            engine on your own text, run: toktier-rust verify-local --engine \
                            cpu --family <family> --input <path>";

/// The same command, offered where the two sides are already known to
/// disagree somewhere and the open question is whether it reaches the
/// caller's own text.
const AFFECTED_HINT: &str = "To see whether your text is affected, run: toktier-rust \
                             verify-local --engine cpu --family <family> --input <path>";

fn readings() -> &'static Vec<Reading> {
    static READINGS: OnceLock<Vec<Reading>> = OnceLock::new();
    READINGS.get_or_init(judged_readings)
}

/// How the certified core of this build compares with the judged one:
/// `verified`, an `unlocated:` line, or a `mismatched:` line saying what
/// differs and what aligns it.
///
/// The build script answered for every core package whose version is the
/// whole of its behaviour. What is left is the units this binary can ask
/// directly, and they are asked once per process.
pub(crate) fn core_closure() -> &'static str {
    static CORE: OnceLock<String> = OnceLock::new();
    CORE.get_or_init(|| {
        decide(
            env!("TOKTIER_RUST_API_CORE_CLOSURE_STATIC"),
            readings(),
            &drift_for,
        )
    })
}

/// The core answer, from the build script's half and this binary's half.
///
/// Taken apart from where the two halves come from so that each state
/// can be exercised: the build script already refused; a unit whose
/// tables moved; a unit that cannot be read while its packages moved;
/// and everything agreeing.
fn decide(
    recorded: &str,
    readings: &[Reading],
    drift: &dyn Fn(&str) -> Option<&'static str>,
) -> String {
    if recorded != "verified" {
        return recorded.to_owned();
    }
    for reading in readings {
        match &reading.observed {
            Some(observed) if observed != &reading.judged => {
                return format!(
                    "mismatched: the reference engine in this binary {} where the certified \
                     evidence was taken on {}; some code points may tokenize differently; the \
                     accelerated engines are held on the reference route. {AFFECTED_HINT}",
                    carries(&reading.unit, observed),
                    reading.judged
                );
            }
            Some(_) => {}
            None => {
                if let Some(moved) = drift(&reading.unit) {
                    return format!(
                        "mismatched: the behaviour version of {} could not be read in this \
                         binary, so the package versions are compared instead: {moved}. \
                         {AFFECTED_HINT}",
                        reading.unit
                    );
                }
            }
        }
    }
    "verified".to_owned()
}

pub(crate) fn core_closure_verified() -> bool {
    core_closure() == "verified"
}

/// What this build compiles that the judged build did not, where the
/// difference is reported rather than gating.
///
/// Two kinds of thing arrive here: packages outside the certified core,
/// and core packages whose package version moved while the behaviour
/// version this binary reads stayed where the evidence was taken. Both
/// are named, with the commands that align them; neither is called
/// harmless.
pub(crate) fn dependency_advisory() -> Option<&'static str> {
    static ADVISORY: OnceLock<Option<String>> = OnceLock::new();
    ADVISORY
        .get_or_init(|| {
            let mut parts = Vec::new();
            let outside = env!("TOKTIER_RUST_API_DEPENDENCY_ADVISORY");
            if !outside.is_empty() {
                parts.push(outside.to_owned());
            }
            for reading in readings() {
                let Some(observed) = &reading.observed else {
                    continue;
                };
                if observed != &reading.judged {
                    continue;
                }
                if let Some(drift) = drift_for(&reading.unit) {
                    parts.push(format!(
                        "{drift}: {}; the certificate holds. {COMPARE_HINT}",
                        unchanged(&reading.unit, observed)
                    ));
                }
            }
            (!parts.is_empty()).then(|| parts.join(". "))
        })
        .as_deref()
}

/// The behaviour version of each unit, as a reader of `doctor` sees it.
pub(crate) fn facts() -> Vec<BehaviorVersion> {
    readings()
        .iter()
        .map(|reading| BehaviorVersion {
            unit: reading.unit.clone(),
            observed: reading.observed.clone(),
            judged: reading.judged.clone(),
            source: reading.source.clone(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{
        carries, decide, drift_for, observe, readings, unchanged, Reading, AGE_CANDIDATES,
        FIELD_SEPARATOR, RECORD_SEPARATOR,
    };

    fn reading(unit: &str, judged: &str, observed: Option<&str>) -> Reading {
        Reading {
            unit: unit.to_owned(),
            judged: judged.to_owned(),
            source: "probe:regex-age".to_owned(),
            observed: observed.map(str::to_owned),
        }
    }

    fn no_drift(_: &str) -> Option<&'static str> {
        None
    }

    /// The four states of the core answer, each asked directly.
    #[test]
    fn the_core_answer_has_four_states() {
        // The build script already refused: its words stand, whatever
        // the probes say.
        assert_eq!(
            decide(
                "mismatched: the judged build compiled memchr 2.8.3",
                &[reading("regex", "16.0", Some("17.0"))],
                &no_drift,
            ),
            "mismatched: the judged build compiled memchr 2.8.3"
        );
        // Tables that moved: named, with both versions and what follows.
        let moved = decide(
            "verified",
            &[reading("regex", "16.0", Some("17.0"))],
            &no_drift,
        );
        assert_eq!(
            moved,
            "mismatched: the reference engine in this binary carries Unicode 17.0 (regex \
             tables) where the certified evidence was taken on 16.0; some code points may \
             tokenize differently; the accelerated engines are held on the reference route. \
             To see whether your text is affected, run: toktier-rust verify-local --engine cpu \
             --family <family> --input <path>"
        );
        // A unit that cannot be read falls back to the package
        // comparison the build script made, and only refuses when that
        // comparison found something.
        assert_eq!(
            decide("verified", &[reading("regex", "16.0", None)], &no_drift),
            "verified"
        );
        assert_eq!(
            decide("verified", &[reading("regex", "16.0", None)], &|unit| {
                (unit == "regex").then_some("regex 1.13.1 -> 1.14.0")
            }),
            "mismatched: the behaviour version of regex could not be read in this binary, so \
             the package versions are compared instead: regex 1.13.1 -> 1.14.0. To see whether \
             your text is affected, run: toktier-rust verify-local --engine cpu --family \
             <family> --input <path>"
        );
        // Everything agreeing.
        assert_eq!(
            decide(
                "verified",
                &[
                    reading("regex", "16.0", Some("16.0")),
                    reading("onig", "6.9.10", Some("6.9.10")),
                ],
                &no_drift,
            ),
            "verified"
        );
    }

    /// A record that names no unit at all leaves the answer where the
    /// build script left it, which is the exact comparison.
    #[test]
    fn a_record_with_no_units_leaves_the_exact_answer_standing() {
        assert_eq!(decide("verified", &[], &no_drift), "verified");
        assert_eq!(
            decide("unlocated: no lockfile", &[], &no_drift),
            "unlocated: no lockfile"
        );
    }

    /// The record and the libraries agree in this workspace. The judged
    /// values are written by `tools/generate_judged_closure.py` from the
    /// vendored sources of the same packages this build links, so a lock
    /// that moved without the record being regenerated is red here.
    #[test]
    fn every_judged_unit_reads_the_way_the_record_says() {
        let readings = readings();
        assert!(!readings.is_empty(), "no behaviour unit was recorded");
        for reading in readings {
            let observed = reading
                .observed
                .as_deref()
                .unwrap_or_else(|| panic!("{} could not be read", reading.unit));
            assert_eq!(
                observed, reading.judged,
                "{} reads {observed} and the record judges {}",
                reading.unit, reading.judged
            );
        }
    }

    /// The three units this release knows how to ask, and nothing else.
    #[test]
    fn a_unit_this_release_cannot_ask_reads_as_unreadable() {
        assert!(observe("regex").is_some());
        assert!(observe("onig").is_some());
        assert!(observe("unicode-segmentation").is_some());
        assert!(observe("unicode_categories").is_none());
        assert!(observe("").is_none());
    }

    /// The age probe finds the newest version the tables carry, not the
    /// newest one it asks about.
    #[test]
    fn the_age_probe_answers_from_the_tables_rather_than_the_candidates() {
        let observed = observe("regex").expect("a regex age");
        assert!(AGE_CANDIDATES.contains(&observed.as_str()));
        assert!(
            regex::Regex::new(&format!(r"\p{{Age={observed}}}")).is_ok(),
            "the answer does not compile"
        );
        let newer = AGE_CANDIDATES
            .iter()
            .take_while(|age| **age != observed)
            .collect::<Vec<_>>();
        for age in newer {
            assert!(
                regex::Regex::new(&format!(r"\p{{Age={age}}}")).is_err(),
                "{age} compiles and is newer than the answer"
            );
        }
    }

    /// The unit sentences name the unit and both sides, and never say
    /// that a package outside the reading cannot matter.
    #[test]
    fn the_unit_sentences_say_which_tables_and_which_version() {
        assert_eq!(
            carries("regex", "17.0"),
            "carries Unicode 17.0 (regex tables)"
        );
        assert_eq!(carries("onig", "6.9.11"), "links Oniguruma 6.9.11");
        assert_eq!(
            unchanged("regex", "16.0"),
            "Unicode data version unchanged (16.0)"
        );
        assert_eq!(
            unchanged("a-new-unit", "1.0"),
            "a-new-unit behaviour version unchanged (1.0)"
        );
    }

    /// The drift line is read by unit, and a unit that is not in it has
    /// no drift rather than the first one's.
    #[test]
    fn the_drift_line_is_read_by_unit() {
        // The separators are the ones the build script writes.
        assert_eq!(FIELD_SEPARATOR, '\u{1f}');
        assert_eq!(RECORD_SEPARATOR, '\u{1e}');
        // This workspace's own build has no drift at all, which is what
        // a build inside the source workspace should read.
        assert!(drift_for("regex").is_none());
        assert!(drift_for("a-unit-that-is-not-recorded").is_none());
    }
}
