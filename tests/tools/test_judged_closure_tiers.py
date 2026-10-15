"""The four gates that keep the certified core mechanical.

The tier split decides which packages can withhold a certificate, so the
tool that derives it refuses rather than guesses: a text-semantics name it
has not been told about, a third-party package an edge of ours still pins
exactly, a package classified as text-semantics core that nothing links,
and a new facade source file nobody has placed on or off the encode path.

Whether the shipped record still describes this tree is a question for
Cargo, so it is asked where Cargo can be asked: `python3
tools/generate_judged_closure.py --check`, which CI runs. These tests run
against an isolated home and do not invoke Cargo.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import generate_judged_closure as closure  # noqa: E402


class FakeGraph:
    """A resolve graph small enough to state in a test.

    The real one comes from `cargo metadata`; what `classify` asks of it
    is which packages are workspace members, which edges leave them, what
    an engine crate reaches, which packages are proc macros, and what our
    own manifests require.
    """

    def __init__(
        self,
        *,
        members: dict[str, list[str]],
        reachable: list[str],
        requirements: dict[str, list[str]],
        proc_macros: tuple[str, ...] = (),
        versions: dict[str, str] | None = None,
    ) -> None:
        self._members = members
        self._reachable = reachable
        self._requirements = requirements
        self._proc_macros = proc_macros
        self._versions = versions or {}
        self.workspace_members = set(members)
        self.nodes: dict[str, closure.Node] = {
            name: name
            for name in {
                *members,
                *(edge for edges in members.values() for edge in edges),
                *reachable,
            }
        }

    def name(self, identifier: str) -> str:
        return identifier

    def version(self, identifier: str) -> str:
        return self._versions.get(identifier, "1.0.0")

    def is_proc_macro(self, identifier: str) -> bool:
        return identifier in self._proc_macros

    def id_of(self, name: str) -> str | None:
        return name if name in self.nodes else None

    def edges(self, identifier: str, kinds: set[str | None]) -> list[str]:
        return self._members.get(identifier, [])

    def reachable(self, roots: list[str], kinds: set[str | None]) -> set[str]:
        return set(self._reachable)

    def own_requirements(self, name: str) -> list[str]:
        return self._requirements.get(name, [])


def _graph(
    *,
    members: dict[str, list[str]] | None = None,
    reachable: list[str] | None = None,
    requirements: dict[str, list[str]] | None = None,
    proc_macros: tuple[str, ...] = (),
    versions: dict[str, str] | None = None,
) -> FakeGraph:
    return FakeGraph(
        members=members if members is not None else {"toktier": ["serde"]},
        reachable=reachable if reachable is not None else ["serde"],
        requirements=(
            requirements if requirements is not None else {"serde": ["^1.0"]}
        ),
        proc_macros=proc_macros,
        versions=versions,
    )


def _compiled(*names: str) -> set[tuple[str, str]]:
    return {(name, "1.0.0") for name in names}


def test_every_shipped_package_carries_a_tier_and_a_criterion() -> None:
    import json

    document = json.loads(closure.OUTPUT.read_text(encoding="utf-8"))
    assert document["schema"] == "toktier.rust_compiled_closure.v2"
    assert document["tier_rule"]["statement"]
    for entry in document["packages"]:
        assert entry["tier"] in {"core", "implementation", "periphery"}
        assert entry["criterion"]
        if entry["tier"] == "core":
            assert entry["criterion"] in {"R0", "R2"}
        elif entry["tier"] == "implementation":
            assert entry["criterion"] == "R1"
        else:
            assert entry["criterion"].startswith("periphery")


def test_a_workspace_member_is_core_and_a_direct_dependency_is_implementation() -> None:
    graph = _graph(requirements={"serde": ["^1.0"]})
    tiers, criteria = closure.classify(_compiled("toktier", "serde"), [graph])
    assert tiers[("toktier", "1.0.0")] == "core"
    assert criteria[("toktier", "1.0.0")] == "R0"
    # `serde` is named from encode-path sources of this repository, so the
    # reference test finds it; since 0.3.0 it is an implementation library
    # on a caret edge, reported rather than gating.
    assert tiers[("serde", "1.0.0")] == "implementation"
    assert criteria[("serde", "1.0.0")] == "R1"


def test_a_direct_dependency_no_encode_path_source_names_is_periphery() -> None:
    graph = _graph(
        members={"toktier": ["a-package-nothing-names"]},
        reachable=["a-package-nothing-names"],
        requirements={"a-package-nothing-names": ["^1.0.0"]},
    )
    tiers, criteria = closure.classify(
        _compiled("toktier", "a-package-nothing-names"), [graph]
    )
    assert tiers[("a-package-nothing-names", "1.0.0")] == "periphery"
    assert (
        criteria[("a-package-nothing-names", "1.0.0")]
        == "periphery:lifecycle-only-direct-dependency"
    )


def test_a_second_copy_of_a_core_package_is_judged_on_its_own() -> None:
    """Only the version an own edge resolves is the copy that edge names;
    `base64 0.13.1` arrives under `spm_precompiled` and is periphery."""

    graph = _graph(versions={"serde": "1.0.0"}, requirements={"serde": ["^1.0"]})
    compiled = {("toktier", "1.0.0"), ("serde", "1.0.0"), ("serde", "0.9.0")}
    tiers, criteria = closure.classify(compiled, [graph])
    assert criteria[("serde", "1.0.0")] == "R1"
    assert tiers[("serde", "0.9.0")] == "periphery"


def test_the_name_net_refuses_an_unclassified_text_semantics_package() -> None:
    graph = _graph(
        members={"toktier": ["serde"]},
        reachable=["serde", "unicode-something-new"],
        requirements={"serde": ["^1.0"]},
    )
    with pytest.raises(closure.GenerationError) as refusal:
        closure.classify(
            _compiled("toktier", "serde", "unicode-something-new"), [graph]
        )
    assert "unicode-something-new" in str(refusal.value)
    assert "TEXT_SEMANTICS_TABLE" in str(refusal.value)


def test_a_third_party_package_on_an_exact_edge_is_refused() -> None:
    graph = _graph(requirements={"serde": ["=1.0.0"]})
    with pytest.raises(closure.GenerationError) as refusal:
        closure.classify(_compiled("toktier", "serde"), [graph])
    assert "exact (`=`) edge" in str(refusal.value)
    assert "serde" in str(refusal.value)


def test_a_reference_library_on_an_exact_edge_is_refused_too() -> None:
    graph = _graph(
        members={"toktier": ["serde", "regex"]},
        reachable=["serde", "regex"],
        requirements={"serde": ["^1.0"], "regex": ["=1.0.0"]},
    )
    with pytest.raises(closure.GenerationError) as refusal:
        closure.classify(_compiled("toktier", "serde", "regex"), [graph])
    assert "regex" in str(refusal.value)
    assert "certified reference versions" in str(refusal.value)


def test_the_certified_reference_versions_always_hold_the_judged_one() -> None:
    assert closure.certified_reference_versions("tokenizers", "0.22.2") == ["0.22.2"]
    assert closure.certified_reference_versions("a-unit-not-in-the-table", "1.2.3") == [
        "1.2.3"
    ]
    for unit in closure.CERTIFIED_REFERENCE_VERSIONS:
        assert closure.BEHAVIOUR_SOURCES[unit] == "crate-version"
    for unit, source in closure.BEHAVIOUR_SOURCES.items():
        if source == "crate-version":
            assert unit in closure.CERTIFIED_REFERENCE_VERSIONS


def test_a_text_semantics_core_package_nothing_links_is_refused() -> None:
    graph = _graph(
        members={"toktier": ["serde"]},
        reachable=["serde"],
        requirements={"serde": ["^1.0"]},
    )
    with pytest.raises(closure.GenerationError) as refusal:
        closure.classify(_compiled("toktier", "serde", "regex"), [graph])
    assert "reachable through normal edges from an engine crate" in str(refusal.value)
    assert "regex" in str(refusal.value)


def test_a_table_entry_this_build_no_longer_compiles_is_refused() -> None:
    with pytest.raises(closure.GenerationError) as refusal:
        closure.check_table_covers(_compiled("toktier", "serde"))
    assert "no longer compiles" in str(refusal.value)
    assert "regex" in str(refusal.value)


def test_the_shipped_closure_covers_every_classified_package() -> None:
    import json

    document = json.loads(closure.OUTPUT.read_text(encoding="utf-8"))
    closure.check_table_covers(
        {(entry["name"], entry["version"]) for entry in document["packages"]}
    )


def test_an_unplaced_facade_source_file_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(closure, "FACADE_ENCODE_SOURCES", ("lib.rs",), raising=True)
    with pytest.raises(closure.GenerationError) as refusal:
        closure.check_facade_source_classification()
    message = str(refusal.value)
    assert "runtime.rs" in message
    assert "FACADE_ENCODE_SOURCES" in message


def test_a_classified_facade_source_file_that_is_gone_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        closure,
        "FACADE_LIFECYCLE_SOURCES",
        (*closure.FACADE_LIFECYCLE_SOURCES, "a_file_that_is_not_here.rs"),
        raising=True,
    )
    with pytest.raises(closure.GenerationError) as refusal:
        closure.check_facade_source_classification()
    assert "a_file_that_is_not_here.rs" in str(refusal.value)


def test_the_behaviour_units_of_the_shipped_record_agree_with_the_table() -> None:
    import json

    document = json.loads(closure.OUTPUT.read_text(encoding="utf-8"))
    probed = {
        entry["name"]: entry
        for entry in document["packages"]
        if "behavior_unit" in entry
    }
    assert set(probed) == set(closure.BEHAVIOUR_UNITS)
    for name, entry in probed.items():
        assert entry["tier"] == "core"
        assert entry["criterion"] == "R2"
        assert entry["behavior_unit"] == closure.BEHAVIOUR_UNITS[name]
        assert (
            entry["behavior_source"]
            == closure.BEHAVIOUR_SOURCES[entry["behavior_unit"]]
        )
        assert entry["behavior_version"]
        # A unit read by crate version carries its certified reference
        # versions, the judged one first; a probed unit carries none.
        if entry["behavior_source"] == "crate-version":
            assert entry["certified_versions"][0] == entry["version"]
            assert entry["behavior_version"] == f"crate:{entry['version']}"
            assert (
                document["tier_rule"]["certified_reference_versions"][
                    entry["behavior_unit"]
                ]
                == entry["certified_versions"]
            )
        else:
            assert "certified_versions" not in entry
    # The embedded reference engine is a reference library since 0.3.0.
    assert probed["tokenizers"]["behavior_source"] == "crate-version"
    # One version per unit, whatever the package versions are.
    for unit in set(closure.BEHAVIOUR_UNITS.values()):
        versions = {
            entry["behavior_version"]
            for entry in probed.values()
            if entry["behavior_unit"] == unit
        }
        assert len(versions) == 1
