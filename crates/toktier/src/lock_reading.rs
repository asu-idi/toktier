//! Which lockfile the build script reads for this build.
//!
//! The rules `build.rs` applies to the value of `TOKTIER_CARGO_LOCK` and to
//! the lockfile it then locates, written without reading the process
//! environment so that they can be tested. `build.rs` takes this file in
//! with `#[path]`; the library compiles it into its test build only, beside
//! the build-support module it shares the lockfile types with.
//!
//! The standing of the property data (`property_guard.rs`) believes the
//! lockfile it is handed, so these rules decide which lockfile that is:
//!
//! - a value of `TOKTIER_CARGO_LOCK` that is empty, not absolute, or that
//!   resolves inside this crate's own directory is refused
//!   ([`refuse_named`]). Cargo runs a build script in the package's own
//!   directory, so a relative value names a file inside this crate, not one
//!   where it was typed; and nothing inside this crate's directory is a
//!   lockfile of the build that consumes it (a packaged copy ships the
//!   lockfile recorded when it was published);
//! - [`located_by`] names how the lockfile that was read was found; a
//!   lockfile found only above this crate's own manifest is reported as
//!   `manifest`, which the library never counts as read.

use std::ffi::OsStr;
use std::fmt;
use std::fs;
use std::path::{Component, Path, PathBuf};

#[cfg(test)]
use crate::build_support::{GoverningLock, CARGO_LOCK_ENV};
#[cfg(not(test))]
use crate::source_identity::{GoverningLock, CARGO_LOCK_ENV};

/// Why a value of `TOKTIER_CARGO_LOCK` is not believed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Refusal {
    /// The variable is set to nothing.
    Empty,
    /// The value is not an absolute path.
    NotAbsolute(PathBuf),
    /// The value resolves inside this crate's own directory.
    InsideCrate(PathBuf),
}

impl fmt::Display for Refusal {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => write!(formatter, "{CARGO_LOCK_ENV} is set to an empty value"),
            Self::NotAbsolute(path) => write!(
                formatter,
                "{CARGO_LOCK_ENV}={} is not an absolute path (Cargo runs this build script in \
                 the crate's own directory, so a relative path would name a file there)",
                path.display()
            ),
            Self::InsideCrate(path) => write!(
                formatter,
                "{CARGO_LOCK_ENV}={} lies inside this crate's own directory, which holds no \
                 lockfile of the build that consumes the crate",
                path.display()
            ),
        }
    }
}

/// The path with `.` dropped and each `..` folded into its parent, without
/// touching the file system.
fn lexical(path: &Path) -> PathBuf {
    let mut folded = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                folded.pop();
            }
            other => folded.push(other.as_os_str()),
        }
    }
    folded
}

/// The path with every symbolic link resolved when it exists, its lexical
/// form otherwise.
fn resolved(path: &Path) -> PathBuf {
    fs::canonicalize(path).unwrap_or_else(|_| lexical(path))
}

/// Whether the value of `TOKTIER_CARGO_LOCK` is refused, and why.
///
/// `manifest_dir` is this crate's `CARGO_MANIFEST_DIR`. A value is refused
/// when it is empty, when it is not an absolute path, and when it resolves
/// (following `..` and symbolic links) inside `manifest_dir`. A refused
/// value is read as no lockfile at all: every reading of the build script
/// is taken as missing.
pub fn refuse_named(value: &OsStr, manifest_dir: &Path) -> Option<Refusal> {
    if value.is_empty() {
        return Some(Refusal::Empty);
    }
    let path = Path::new(value);
    if !path.is_absolute() {
        return Some(Refusal::NotAbsolute(path.to_owned()));
    }
    let inside = resolved(path).starts_with(resolved(manifest_dir))
        || lexical(path).starts_with(lexical(manifest_dir));
    inside.then(|| Refusal::InsideCrate(path.to_owned()))
}

/// How the lockfile the build script read was located, in the word
/// `src/property_guard.rs` reads.
///
/// - `named`: `TOKTIER_CARGO_LOCK` named it (`named` is whether the
///   variable is set, after [`refuse_named`] let it through);
/// - `out_dir`: it is the `Cargo.lock` of a directory above `out_dir`;
/// - `manifest`: it was found only above this crate's own manifest (the
///   search tries `OUT_DIR`'s ancestors first), which the library never
///   counts as read;
/// - `none`: nothing was located, or the named file is this package's own
///   record of the judged build.
pub fn located_by(governing: &GoverningLock, named: bool, out_dir: &Path) -> &'static str {
    match governing {
        GoverningLock::Found(_) if named => "named",
        GoverningLock::Found(path)
            if out_dir
                .ancestors()
                .any(|ancestor| ancestor.join("Cargo.lock") == *path) =>
        {
            "out_dir"
        }
        GoverningLock::Found(_) => "manifest",
        GoverningLock::JudgedRecordNamed(_) | GoverningLock::Missing => "none",
    }
}

/// Why the build reads no lockfile, for the build warning: `refusal` when
/// the variable was refused, `named` the value of the variable, `governing`
/// what the search located.
pub fn unread_reason(
    refusal: Option<&Refusal>,
    named: Option<&Path>,
    governing: &GoverningLock,
) -> String {
    if let Some(refusal) = refusal {
        return refusal.to_string();
    }
    match (governing, named) {
        (GoverningLock::JudgedRecordNamed(path), _) => format!(
            "{CARGO_LOCK_ENV} names {}, this package's own record of the judged build rather \
             than a lockfile of this build",
            path.display()
        ),
        (GoverningLock::Found(path), _) => format!(
            "the only lockfile found, {}, lies above this crate's own manifest, and the build \
             cannot tell whether it governs this build",
            path.display()
        ),
        (GoverningLock::Missing, Some(path)) => format!(
            "{CARGO_LOCK_ENV} names {}, which is not a file",
            path.display()
        ),
        (GoverningLock::Missing, None) => {
            "no lockfile governing this build was located from OUT_DIR".to_owned()
        }
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::OsStr;
    use std::fs;
    use std::path::{Path, PathBuf};

    use super::{located_by, refuse_named, unread_reason, GoverningLock, Refusal};

    /// A crate directory with a `Cargo.lock` and a `data/` record inside it,
    /// a project beside it with its own lockfile, and a symbolic link from
    /// the project into the crate.
    struct Layout {
        _root: tempfile::TempDir,
        crate_dir: PathBuf,
        project: PathBuf,
    }

    fn layout() -> Layout {
        let root = tempfile::tempdir().unwrap();
        let crate_dir = root.path().join("registry/toktier-0.3.3");
        let project = root.path().join("project");
        fs::create_dir_all(crate_dir.join("data/build")).unwrap();
        fs::create_dir_all(&project).unwrap();
        for file in [
            crate_dir.join("Cargo.lock"),
            crate_dir.join("data/build/judged_dependencies.lock"),
            project.join("Cargo.lock"),
        ] {
            fs::write(file, "version = 4\n").unwrap();
        }
        #[cfg(unix)]
        std::os::unix::fs::symlink(&crate_dir, project.join("linked-crate")).unwrap();
        Layout {
            _root: root,
            crate_dir,
            project,
        }
    }

    fn refuse(value: &str, crate_dir: &Path) -> Option<Refusal> {
        refuse_named(OsStr::new(value), crate_dir)
    }

    #[test]
    fn relative_and_empty_values_are_refused() {
        let layout = layout();
        for value in [
            "Cargo.lock",
            "./Cargo.lock",
            "../../Cargo.lock",
            "project/Cargo.lock",
        ] {
            assert_eq!(
                refuse(value, &layout.crate_dir),
                Some(Refusal::NotAbsolute(PathBuf::from(value))),
                "{value}"
            );
        }
        assert_eq!(refuse("", &layout.crate_dir), Some(Refusal::Empty));
        for refusal in [
            Refusal::Empty,
            Refusal::NotAbsolute(PathBuf::from("Cargo.lock")),
        ] {
            assert!(
                refusal.to_string().contains("TOKTIER_CARGO_LOCK"),
                "{refusal}"
            );
        }
        assert!(Refusal::NotAbsolute(PathBuf::from("Cargo.lock"))
            .to_string()
            .contains("absolute path"));
    }

    #[test]
    fn absolute_values_inside_the_crate_are_refused() {
        let layout = layout();
        let crate_dir = &layout.crate_dir;
        let inside = [
            // The packaged copy's own shipped lockfile.
            crate_dir.join("Cargo.lock"),
            // Its record of the judged build.
            crate_dir.join("data/build/judged_dependencies.lock"),
            // The same reached through `..`.
            crate_dir.join("data/../Cargo.lock"),
            layout
                .project
                .join("../registry/toktier-0.3.3/./Cargo.lock"),
            // A file that does not exist inside the crate.
            crate_dir.join("missing/Cargo.lock"),
        ];
        for path in inside {
            let value = path.to_str().unwrap();
            assert_eq!(
                refuse(value, crate_dir),
                Some(Refusal::InsideCrate(path.clone())),
                "{value}"
            );
        }
        // The same reached through a symbolic link from elsewhere.
        #[cfg(unix)]
        {
            let linked = layout.project.join("linked-crate/Cargo.lock");
            assert_eq!(
                refuse(linked.to_str().unwrap(), crate_dir),
                Some(Refusal::InsideCrate(linked.clone()))
            );
        }
        assert!(Refusal::InsideCrate(crate_dir.join("Cargo.lock"))
            .to_string()
            .contains("inside this crate's own directory"));
    }

    #[test]
    fn an_absolute_value_elsewhere_is_believed() {
        let layout = layout();
        for path in [
            layout.project.join("Cargo.lock"),
            layout.project.join("../project/Cargo.lock"),
            // A missing file elsewhere is not refused here; the search then
            // finds nothing and the build reads no lockfile.
            layout.project.join("missing.lock"),
            PathBuf::from("/nonexistent/Cargo.lock"),
        ] {
            assert_eq!(
                refuse(path.to_str().unwrap(), &layout.crate_dir),
                None,
                "{}",
                path.display()
            );
        }
        // A sibling whose name only starts like the crate's directory.
        let sibling = layout
            .crate_dir
            .with_file_name("toktier-0.3.3-other/Cargo.lock");
        assert_eq!(refuse(sibling.to_str().unwrap(), &layout.crate_dir), None);
    }

    #[test]
    fn located_by_names_each_way_a_lockfile_was_found() {
        let out_dir = Path::new("/work/app/target/release/build/toktier-0123/out");
        let above_out_dir = PathBuf::from("/work/app/Cargo.lock");
        let above_manifest = PathBuf::from("/src/toktier/Cargo.lock");
        // Named by the variable, wherever it lies.
        for path in [&above_out_dir, &above_manifest] {
            assert_eq!(
                located_by(&GoverningLock::Found(path.clone()), true, out_dir),
                "named"
            );
        }
        // Found above OUT_DIR.
        assert_eq!(
            located_by(&GoverningLock::Found(above_out_dir), false, out_dir),
            "out_dir"
        );
        // Found only above the crate's manifest (never counted as read).
        assert_eq!(
            located_by(&GoverningLock::Found(above_manifest), false, out_dir),
            "manifest"
        );
        // The named judged record, and nothing at all.
        for named in [true, false] {
            assert_eq!(
                located_by(
                    &GoverningLock::JudgedRecordNamed(PathBuf::from("/x/judged.lock")),
                    named,
                    out_dir
                ),
                "none"
            );
            assert_eq!(located_by(&GoverningLock::Missing, named, out_dir), "none");
        }
    }

    #[test]
    fn each_unread_reading_says_why() {
        let refused = Refusal::NotAbsolute(PathBuf::from("Cargo.lock"));
        let named = Path::new("/nonexistent/Cargo.lock");
        let cases = [
            (
                unread_reason(
                    Some(&refused),
                    Some(Path::new("Cargo.lock")),
                    &GoverningLock::Missing,
                ),
                "not an absolute path",
            ),
            (
                unread_reason(None, Some(named), &GoverningLock::Missing),
                "which is not a file",
            ),
            (
                unread_reason(
                    None,
                    Some(named),
                    &GoverningLock::JudgedRecordNamed(named.to_owned()),
                ),
                "record of the judged build",
            ),
            (
                unread_reason(
                    None,
                    None,
                    &GoverningLock::Found(PathBuf::from("/src/toktier/Cargo.lock")),
                ),
                "above this crate's own manifest",
            ),
            (
                unread_reason(None, None, &GoverningLock::Missing),
                "no lockfile governing this build was located",
            ),
        ];
        for (reason, expected) in cases {
            assert!(reason.contains(expected), "{reason}");
        }
    }
}
