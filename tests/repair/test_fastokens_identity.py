"""The pinned Fastokens adapter: identity by import package and its assurance.

Each cell below builds a real site directory in ``tmp_path`` -- a pure Python
``fastokens`` package with a stand-in ``Tokenizer`` and one or two
``*.dist-info`` directories with a RECORD naming its files -- puts it on
``sys.path`` and lets the adapter resolve it exactly as it would resolve an
installed wheel. The registry node is supplied per test, so a state can be
reached without the published wheel being present, and its absence is a
state of its own.
"""

from __future__ import annotations

import base64
import hashlib
import importlib
import json
import re
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest

from toktier.errors import BackendUnavailable
from toktier.repair import fastokens as adapter
from toktier.repair.registry import RepairFamily

ROOT = Path(__file__).resolve().parents[2]
PINNED = "toktier-fastokens"
UPSTREAM = "fastokens"
PINNED_VERSION = "0.3.1.2"
UPSTREAM_VERSION = "0.3.1"

_INIT = '''"""Stand-in for the fastokens import package (tests only)."""

class _Encoding:
    def __init__(self, ids):
        self.ids = ids


class Tokenizer:
    @staticmethod
    def from_file(path):
        return Tokenizer()

    def encode(self, text, add_special_tokens=False):
        return _Encoding([ord(c) % 256 for c in text.encode("utf-8").decode("latin-1")])
'''


def _record_line(relative: str, payload: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=")
    return f"{relative},sha256={digest.decode()},{len(payload)}"


def install(site: Path, name: str, version: str, *, flavour: str) -> Path:
    """Write the package files and a dist-info whose RECORD names them."""
    package = site / "fastokens"
    package.mkdir(exist_ok=True)
    files = {
        "fastokens/__init__.py": _INIT.encode(),
        "fastokens/_native.abi3.so": f"engine bytes of {flavour}\n".encode(),
        "fastokens/_native.pyi": b"class Tokenizer: ...\n",
    }
    for relative, payload in files.items():
        (site / relative).write_bytes(payload)
    info = site / f"{name.replace('-', '_')}-{version}.dist-info"
    info.mkdir(exist_ok=True)
    (info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
    )
    lines = [_record_line(rel, payload) for rel, payload in files.items()]
    lines.append(f"{info.name}/METADATA,,")
    lines.append(f"{info.name}/RECORD,,")
    (info / "RECORD").write_text("\n".join(lines) + "\n")
    return package


def tree_digest(package: Path) -> str:
    return adapter._hash_tree(package)


def make_entry(
    engine_digest: str,
    *,
    guard: bool = True,
    families: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    shipped = json.loads(
        (ROOT / "tools" / "fastokens_binding.json").read_text(encoding="utf-8")
    )
    entry = {
        key: shipped[key]
        for key in (
            "backend",
            "admission",
            "distribution",
            "recognised_distributions",
            "upstream",
            "source",
            "sdist",
            "oracle",
            "guard",
            "evidence",
        )
    }
    entry["families"] = families if families is not None else shipped["families"]
    wheel = dict(shipped["known_wheels"][0])
    wheel["engine_digest"] = engine_digest
    entry["known_wheels"] = [wheel]
    if not guard:
        del entry["guard"]
    return entry


@pytest.fixture
def site(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    root = tmp_path / "site"
    root.mkdir()
    monkeypatch.syspath_prepend(str(root))
    monkeypatch.chdir(tmp_path)
    sys.modules.pop("fastokens", None)
    importlib.invalidate_caches()
    yield root
    sys.modules.pop("fastokens", None)
    importlib.invalidate_caches()


def _spec(family: str = "qwen3_8b") -> RepairFamily:
    registry = json.loads(
        (ROOT / "tables" / "support_registry.json").read_text(encoding="utf-8")
    )
    sha = next(
        row["artifact_sha256"]
        for row in registry["artifacts"]
        if row["family"] == family
    )
    return RepairFamily(
        family=family,
        artifact_sha256=sha,
        margin=1,
        effective_l_max=1,
        has_normalizer=False,
        source_table_sha256="0" * 64,
    )


class _Reference:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        self.calls += 1
        encoded = text.encode("utf-8")
        return list(range(len(encoded))), [(i, i + 1) for i in range(len(encoded))]


class _Vocab:
    def get_vocab(self) -> dict[str, int]:
        return {chr(0x100 + i): i for i in range(256)}


def _open(
    monkeypatch: pytest.MonkeyPatch,
    entry: dict[str, Any] | None,
    *,
    oracle: str = "0.22.2",
    family: str = "qwen3_8b",
) -> adapter.FastokensFullRepair:
    monkeypatch.setattr(adapter, "pinned_engine_entry", lambda: entry)
    monkeypatch.setattr("toktier._oracle.oracle_version", lambda: oracle)
    monkeypatch.setattr(adapter, "_byte_lengths_from_hf", lambda tokenizer: [1] * 256)
    monkeypatch.setattr(
        adapter, "_spans_from_ids", lambda ids, lengths, text: [(0, 1)] * len(ids)
    )
    return adapter.FastokensFullRepair.open(
        spec=_spec(family),
        tokenizer_path=Path("/nonexistent/tokenizer.json"),
        hf_tokenizer=_Vocab(),
        reference_encode=_Reference(),
    )


# ---------------------------------------------------------------- identity


def test_pinned_distribution_alone_is_resolved_by_import_package(site: Path) -> None:
    """U1 first half: only the pinned distribution installed, and it is seen."""
    package = install(site, PINNED, PINNED_VERSION, flavour="pinned")
    identity = adapter.fastokens_identity()
    assert identity.available
    assert identity.package_dir == package.resolve()
    assert identity.distribution == PINNED
    assert identity.version == PINNED_VERSION
    assert identity.engine_digest == tree_digest(package)
    assert identity.imported_tree_matches_record
    assert identity.coinstalled == ()
    assert adapter.fastokens_distribution_identity() == (
        PINNED_VERSION,
        tree_digest(package),
    )


def test_engine_digest_matches_the_record_based_construction(site: Path) -> None:
    """The v1 digest domain is kept: hashing the tree equals hashing RECORD paths."""
    package = install(site, PINNED, PINNED_VERSION, flavour="pinned")
    digest = hashlib.sha256(b"toktier.fastokens.distribution.v1\0")
    for relative in sorted(
        ("fastokens/__init__.py", "fastokens/_native.abi3.so", "fastokens/_native.pyi")
    ):
        raw = (site / relative).read_bytes()
        digest.update(len(relative.encode()).to_bytes(4, "little"))
        digest.update(relative.encode())
        digest.update(hashlib.sha256(raw).digest())
    assert tree_digest(package) == digest.hexdigest()


def test_absent_package_reports_not_installed(site: Path) -> None:
    """U5: nothing installed."""
    identity = adapter.fastokens_identity()
    assert not identity.available
    assert identity.owners == ()
    assert adapter.fastokens_distribution_identity() == (None, None)


def test_orphaned_metadata_is_told_apart_from_absence(site: Path) -> None:
    """U10: a dist-info whose files were removed by uninstalling the other one."""
    package = install(site, UPSTREAM, UPSTREAM_VERSION, flavour="upstream")
    for path in list(package.iterdir()):
        path.unlink()
    package.rmdir()
    importlib.invalidate_caches()
    identity = adapter.fastokens_identity()
    assert not identity.available
    assert [owner.label for owner in identity.orphaned] == [
        f"{UPSTREAM} {UPSTREAM_VERSION}"
    ]
    assert adapter.fastokens_distribution_identity() == (UPSTREAM_VERSION, None)


def test_coinstalled_distributions_are_attributed_by_bytes(site: Path) -> None:
    """U4a: upstream first, pinned last; the bytes on disk belong to the pinned one."""
    install(site, UPSTREAM, UPSTREAM_VERSION, flavour="upstream")
    package = install(site, PINNED, PINNED_VERSION, flavour="pinned")
    identity = adapter.fastokens_identity()
    assert identity.distribution == PINNED
    assert [owner.label for owner in identity.coinstalled] == [
        f"{UPSTREAM} {UPSTREAM_VERSION}"
    ]
    assert identity.engine_digest == tree_digest(package)
    assert identity.imported_tree_matches_record


def test_coinstalled_reverse_order_attributes_to_upstream(site: Path) -> None:
    """U4b: pinned first, upstream last; the bytes on disk are upstream's."""
    install(site, PINNED, PINNED_VERSION, flavour="pinned")
    install(site, UPSTREAM, UPSTREAM_VERSION, flavour="upstream")
    identity = adapter.fastokens_identity()
    assert identity.distribution == UPSTREAM
    assert [owner.label for owner in identity.coinstalled] == [
        f"{PINNED} {PINNED_VERSION}"
    ]


def test_mixed_bytes_have_no_owner_but_stay_verifiable(site: Path) -> None:
    """U4c: two RECORDs, and the files on disk match neither completely."""
    install(site, UPSTREAM, UPSTREAM_VERSION, flavour="upstream")
    install(site, PINNED, PINNED_VERSION, flavour="pinned")
    (site / "fastokens" / "_native.pyi").write_bytes(b"# edited after install\n")
    identity = adapter.fastokens_identity()
    assert identity.owner is None
    assert identity.verifiable
    # Nothing owns the bytes, so nothing is co-installed *with* an
    # owner; the state is that both RECORDs describe other bytes.
    assert identity.coinstalled == ()
    assert len(identity.unowned) == 2


def test_one_distribution_whose_record_does_not_match_is_not_coinstalled(
    site: Path,
) -> None:
    """D1: a single installation is never reported as sharing with itself.

    A RECORD entry that no longer describes the file beside it is the
    metadata disagreeing with the bytes. Reporting the one installed
    distribution as co-installed said a second installation was here,
    and phrased one distribution as several.
    """
    install(site, PINNED, PINNED_VERSION, flavour="pinned")
    record = next((site).glob(f"{PINNED.replace('-', '_')}-*.dist-info/RECORD"))
    lines = record.read_text(encoding="utf-8").splitlines()
    first, _, size = lines[0].rpartition(",")
    name, _, _digest = first.rpartition(",")
    record.write_text(
        "\n".join([f"{name},sha256=" + "A" * 43 + f",{size}", *lines[1:]]) + "\n",
        encoding="utf-8",
    )

    identity = adapter.fastokens_identity()

    assert identity.owner is None
    assert identity.verifiable
    assert identity.coinstalled == ()
    assert [owner.label for owner in identity.unowned] == [
        f"{PINNED} {PINNED_VERSION}"
    ]
    advisory = adapter.assess(
        identity,
        entry=make_entry(tree_digest(site / "fastokens")),
        guard=None,
        oracle_version="0.22.2",
        family=None,
        artifact_sha256=None,
    ).advisory
    assert advisory is not None
    assert "the RECORD of 'toktier-fastokens 0.3.1.2' names" in advisory
    # One distribution is spoken of in the singular, and the sentence
    # does not claim two of anything.
    for plural in ("the RECORDs", "the distributions", "neither"):
        assert plural not in advisory


def test_a_shadowing_copy_is_not_verifiable(site: Path, tmp_path: Path) -> None:
    """U7 / G11c: a copy earlier on sys.path than the recorded one."""
    install(site, PINNED, PINNED_VERSION, flavour="pinned")
    shadow = tmp_path / "shadow"
    shadow.mkdir()
    copy = shadow / "fastokens"
    copy.mkdir()
    for path in (site / "fastokens").iterdir():
        copy.joinpath(path.name).write_bytes(path.read_bytes())
    (copy / "_native.abi3.so").write_bytes(b"other engine bytes\n")
    sys.path.insert(0, str(shadow))
    importlib.invalidate_caches()
    try:
        identity = adapter.fastokens_identity()
    finally:
        sys.path.remove(str(shadow))
        importlib.invalidate_caches()
    assert identity.available
    assert identity.package_dir == copy.resolve()
    assert identity.shadowed
    assert not identity.verifiable
    assert identity.owner is None
    # The recorded distribution is bypassed, not shared with: nothing is
    # co-installed, and the shadow state carries the report on its own.
    assert identity.coinstalled == ()
    assert adapter._advisory(identity) is None


def test_a_package_without_any_metadata_is_not_verifiable(site: Path) -> None:
    """A bare directory on sys.path: importable, recorded by nothing."""
    package = site / "fastokens"
    package.mkdir()
    (package / "__init__.py").write_text(_INIT)
    identity = adapter.fastokens_identity()
    assert identity.available
    assert identity.owners == ()
    assert not identity.verifiable


# ------------------------------------------------------------- state machine


def test_s1_certified_pinned_is_true_in_the_guarded_sense(
    site: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = install(site, PINNED, PINNED_VERSION, flavour="pinned")
    repair = _open(monkeypatch, make_entry(tree_digest(package)))
    stats = repair.stats()
    assert stats["certification"] == "experimental"
    assert stats["engine_assurance"] == "certified_pinned"
    assert stats["exact_id_guarantee"] is True
    assert stats["assurance_reason"] is None
    assert stats["engine_distribution"] == PINNED
    assert stats["engine_version"] == PINNED_VERSION
    assert stats["engine_digest"] == tree_digest(package)
    assert stats["config_id"] == "toktier-fastokens-full-experimental-v2"
    assert stats["advisory"] is None
    basis = stats["guarantee_basis"]
    assert isinstance(basis, dict)
    assert basis["evidence_id"] == "ev-fastokens-pinned-v1"
    assert basis["known_wheel"]["engine_digest"] == tree_digest(package)
    assert basis["mismatch_guarded"] == 0
    assert basis["families"] == 15
    # The count and the list are the same evidence set, so a reader
    # holding this object can draw the boundary without the registry.
    assert isinstance(basis["family_ids"], list)
    assert len(basis["family_ids"]) == basis["families"]
    assert basis["family_ids"] == sorted(basis["family_ids"])
    assert "qwen3_8b" in basis["family_ids"]
    statement = basis["statement"]
    assert "routed to that reference by the adapter's Unicode guard" in statement
    assert "other builds of the same source are not covered" in statement
    guard = stats["unicode_guard"]
    assert guard == {
        "id": "toktier-fastokens-unicode-skew-guard-v1",
        "codepoints": 154,
        "active": True,
    }
    known = cast(dict[str, str], stats["known_wheel"])
    assert known["filename"].startswith("toktier_fastokens-0.3.1.2-")


def test_s2_unrecognized_build_when_the_digest_is_not_listed(
    site: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(site, PINNED, PINNED_VERSION, flavour="self-built")
    repair = _open(monkeypatch, make_entry("0" * 64))
    stats = repair.stats()
    assert stats["engine_assurance"] == "unrecognized_build"
    assert stats["exact_id_guarantee"] is False
    assert stats["guarantee_basis"] is None
    assert stats["known_wheel"] is None
    reason = str(stats["assurance_reason"])
    assert "not among the wheels toktier published" in reason
    # The remedy has to move a same-version installation, which a plain
    # `pip install` leaves in place, and it names the distribution the
    # registry records rather than a version written here.
    assert (
        'pip install --force-reinstall --no-deps --only-binary :all: '
        '"toktier-fastokens==0.3.1.2"' in reason
    )
    # The guard is unconditional: it is active in this state as well.
    assert cast(dict[str, object], stats["unicode_guard"])["active"] is True


def test_s2_prime_self_built_bytes_that_match_are_certified(
    site: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The label follows the bytes, not who built them."""
    package = install(site, PINNED, PINNED_VERSION, flavour="pinned")
    repair = _open(monkeypatch, make_entry(tree_digest(package)))
    assert repair.stats()["engine_assurance"] == "certified_pinned"


def test_s3_upstream_build(site: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    install(site, UPSTREAM, UPSTREAM_VERSION, flavour="upstream")
    repair = _open(monkeypatch, make_entry("0" * 64))
    stats = repair.stats()
    assert stats["engine_assurance"] == "upstream_build"
    assert stats["exact_id_guarantee"] is False
    assert stats["engine_distribution"] == UPSTREAM
    assert stats["engine_version"] == UPSTREAM_VERSION
    assert "do not carry over" in str(stats["assurance_reason"])


def test_s4a_coinstalled_with_pinned_bytes_reports_and_does_not_refuse(
    site: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(site, UPSTREAM, UPSTREAM_VERSION, flavour="upstream")
    package = install(site, PINNED, PINNED_VERSION, flavour="pinned")
    repair = _open(monkeypatch, make_entry(tree_digest(package)))
    stats = repair.stats()
    assert stats["engine_assurance"] == "certified_pinned"
    assert stats["exact_id_guarantee"] is True
    advisory = str(stats["advisory"])
    assert "'fastokens 0.3.1' is also installed" in advisory
    assert "belong to toktier-fastokens 0.3.1.2" in advisory
    assert "pip uninstall -y fastokens toktier-fastokens" in advisory


def test_s4b_coinstalled_with_upstream_bytes(
    site: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(site, PINNED, PINNED_VERSION, flavour="pinned")
    install(site, UPSTREAM, UPSTREAM_VERSION, flavour="upstream")
    repair = _open(monkeypatch, make_entry("0" * 64))
    stats = repair.stats()
    assert stats["engine_assurance"] == "upstream_build"
    assert "'toktier-fastokens 0.3.1.2' is also installed" in str(stats["advisory"])


def test_s4c_mixed_bytes_are_unrecognized_with_an_advisory(
    site: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(site, UPSTREAM, UPSTREAM_VERSION, flavour="upstream")
    install(site, PINNED, PINNED_VERSION, flavour="pinned")
    (site / "fastokens" / "_native.pyi").write_bytes(b"# edited after install\n")
    repair = _open(monkeypatch, make_entry("0" * 64))
    stats = repair.stats()
    assert stats["engine_assurance"] == "unrecognized_build"
    assert stats["engine_distribution"] is None
    advisory = str(stats["advisory"])
    assert "the RECORDs of" in advisory and "name the fastokens files" in advisory
    assert "are not the ones they recorded" in advisory


def test_s5_not_installed_refuses_as_before(
    site: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(BackendUnavailable) as caught:
        _open(monkeypatch, None)
    assert "not installed" in str(caught.value)
    assert caught.value.details == {"backend": "fastokens", "extra": "fastokens"}


def test_s6_guard_disabled_when_the_registry_carries_no_guard(
    site: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = install(site, PINNED, PINNED_VERSION, flavour="pinned")
    repair = _open(monkeypatch, make_entry(tree_digest(package), guard=False))
    stats = repair.stats()
    assert stats["engine_assurance"] == "guard_disabled"
    assert stats["exact_id_guarantee"] is False
    assert stats["unicode_guard"] == {
        "id": "toktier-fastokens-unicode-skew-guard-v1",
        "codepoints": 0,
        "active": False,
    }


def test_s6_a_guard_whose_digest_disagrees_is_not_compiled(
    site: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = install(site, PINNED, PINNED_VERSION, flavour="pinned")
    entry = make_entry(tree_digest(package))
    entry["guard"] = dict(entry["guard"])
    entry["guard"]["ranges"] = [*entry["guard"]["ranges"], ["U+0041", "U+0041"]]
    assert adapter.compile_unicode_guard(entry) is None
    repair = _open(monkeypatch, entry)
    assert repair.stats()["engine_assurance"] == "guard_disabled"


def test_s7_shadowed_import_refuses(
    site: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = install(site, PINNED, PINNED_VERSION, flavour="pinned")
    shadow = tmp_path / "shadow"
    copy = shadow / "fastokens"
    copy.mkdir(parents=True)
    for path in package.iterdir():
        copy.joinpath(path.name).write_bytes(path.read_bytes())
    (copy / "_native.abi3.so").write_bytes(b"other engine bytes\n")
    monkeypatch.syspath_prepend(str(shadow))
    importlib.invalidate_caches()
    with pytest.raises(BackendUnavailable) as caught:
        _open(monkeypatch, make_entry(tree_digest(package)))
    assert "could not be verified" in str(caught.value)
    assert caught.value.details["engine_assurance"] == "unverifiable"
    assert caught.value.details["imported_from"] == str(copy.resolve())
    assert str(package.resolve()) in caught.value.details["recorded_at"]
    # The message names both paths and then one thing to do about the
    # relation between them; which copy to keep stays the reader's call.
    remedy = str(caught.value.details["remedy"])
    assert "remove the shadowing entry from sys.path" in remedy
    assert remedy in str(caught.value)
    assert "pip install" not in remedy


def test_a_package_no_distribution_records_names_the_published_one(
    site: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other unverifiable arm used to stop at the diagnosis.

    A bare ``fastokens/`` directory on ``sys.path`` is importable and
    recorded by nothing, so the digest describes bytes no distribution
    vouches for. Saying only that left a reader with a state and no move.
    """
    package = site / "fastokens"
    package.mkdir()
    (package / "__init__.py").write_text(_INIT)
    importlib.invalidate_caches()

    with pytest.raises(BackendUnavailable) as caught:
        _open(monkeypatch, make_entry(tree_digest(package)))

    assert caught.value.details["engine_assurance"] == "unverifiable"
    assert caught.value.details["recorded_at"] == []
    remedy = str(caught.value.details["remedy"])
    assert "no installed distribution records" in str(caught.value)
    assert remedy in str(caught.value)
    # The distribution and version come from the registry node, in the
    # same replacement form the unrecognized-build state prints.
    assert (
        'pip install --force-reinstall --no-deps --only-binary :all: '
        '"toktier-fastokens==0.3.1.2"' in remedy
    )


def test_s8_oracle_mismatch(site: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package = install(site, PINNED, PINNED_VERSION, flavour="pinned")
    repair = _open(monkeypatch, make_entry(tree_digest(package)), oracle="0.22.3")
    stats = repair.stats()
    assert stats["engine_assurance"] == "oracle_mismatch"
    assert stats["exact_id_guarantee"] is False
    assert "tokenizers 0.22.3" in str(stats["assurance_reason"])
    assert "(0.22.2)" in str(stats["assurance_reason"])


def test_s9_family_outside_evidence(
    site: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = install(site, PINNED, PINNED_VERSION, flavour="pinned")
    entry = make_entry(
        tree_digest(package),
        families=[{"family": "llama_3_1_8b", "artifact_sha256": "1" * 64}],
    )
    repair = _open(monkeypatch, entry)
    stats = repair.stats()
    assert stats["engine_assurance"] == "family_outside_evidence"
    assert stats["exact_id_guarantee"] is False
    assert stats["assurance_reason"] == (
        "no pinned-build reading is on file for this family"
    )


def test_s10_orphaned_metadata_refuses_with_a_reinstall_hint(
    site: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = install(site, UPSTREAM, UPSTREAM_VERSION, flavour="upstream")
    for path in list(package.iterdir()):
        path.unlink()
    package.rmdir()
    importlib.invalidate_caches()
    with pytest.raises(BackendUnavailable) as caught:
        _open(monkeypatch, make_entry("0" * 64))
    assert "files are missing" in str(caught.value)
    assert caught.value.details["orphaned"] == ["fastokens 0.3.1"]
    assert "pip uninstall -y fastokens toktier-fastokens" in str(caught.value)


def test_a_missing_registry_node_is_fail_closed(
    site: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(site, PINNED, PINNED_VERSION, flavour="pinned")
    repair = _open(monkeypatch, None)
    stats = repair.stats()
    assert stats["engine_assurance"] == "unrecognized_build"
    assert stats["exact_id_guarantee"] is False
    assert cast(dict[str, object], stats["unicode_guard"])["active"] is False
    # D5: the premise that failed is this build's registry, so the
    # sentence names it and prints no command for the installed engine,
    # which is the published wheel and needs no replacing.
    reason = str(stats["assurance_reason"])
    assert "carries no engine_distributions node" in reason
    assert "unicode_guard.active reads false" in reason
    assert "pip install" not in reason


def test_a_directly_constructed_adapter_carries_no_assurance() -> None:
    """The constructor without a report answers the weaker state."""
    repair = adapter.FastokensFullRepair(
        spec=_spec(),
        engine=object(),  # type: ignore[arg-type]
        engine_version="0.3.1",
        engine_digest="e" * 64,
        hf_tokenizer=_Vocab(),
        reference_encode=_Reference(),
    )
    stats = repair.stats()
    assert stats["engine_assurance"] == "unrecognized_build"
    assert stats["exact_id_guarantee"] is False
    assert cast(dict[str, object], stats["unicode_guard"])["active"] is False


# ---------------------------------------------------------------------- guard


def test_the_guard_routes_a_hit_to_the_reference(
    site: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """U11: a guarded code point answers from the reference, on its own path."""
    package = install(site, PINNED, PINNED_VERSION, flavour="pinned")
    repair = _open(monkeypatch, make_entry(tree_digest(package)))
    reference = repair._reference_encode
    assert isinstance(reference, _Reference)
    ids, _spans, kept, path = repair("prefix ", [1], [(0, 1)], "a̴࢘ tail")
    assert path == "hf_full_fastokens_unicode_skew_guard"
    assert kept == 0
    assert reference.calls == 1
    assert ids == list(range(len("prefix a̴࢘ tail".encode())))
    last = repair.stats()["last"]
    assert isinstance(last, dict)
    assert last["reason"] == "unicode_skew_guard"
    assert last["detail"] == {"codepoint": "U+0898", "position": 8}
    # A plain request goes to the engine.
    _, _, _, path = repair("prefix ", [1], [(0, 1)], "plain tail")
    assert path == "fastokens_full_experimental"
    assert repair.stats()["path_counts"] == {
        "fastokens_full_experimental": 1,
        "hf_full_fastokens_unicode_skew_guard": 1,
    }


def _shipped_guard() -> tuple[dict[str, Any], re.Pattern[str]]:
    binding = json.loads(
        (ROOT / "tools" / "fastokens_binding.json").read_text(encoding="utf-8")
    )
    entry = {"backend": "fastokens", "guard": binding["guard"]}
    pattern = adapter.compile_unicode_guard(entry)
    assert pattern is not None
    return binding, pattern


def test_the_shipped_guard_is_the_full_domain_set() -> None:
    """U12: 154 code points, equal to the full-domain reading, a superset of 108."""
    binding, pattern = _shipped_guard()
    reading = json.loads(
        (ROOT / "readings" / "fastokens_pinned_guard_full_domain.json").read_text(
            encoding="utf-8"
        )
    )
    codepoints = [int(cp[2:], 16) for cp in reading["codepoints"]]
    assert len(codepoints) == 154 == binding["guard"]["codepoints"]
    assert reading["set_sha256"] == binding["guard"]["set_sha256"]
    matched = [cp for cp in range(0x110000) if pattern.match(chr(cp))]
    assert matched == codepoints
    archived = [int(cp[2:], 16) for cp in reading["archived_guard"]["codepoints"]]
    assert len(archived) == 108
    assert set(archived) < set(codepoints)
    # Every code point in the set fires in the shape the evidence probed.
    for cp in codepoints:
        assert pattern.search("a" + chr(cp) + "̴") is not None
    # The set stays quiet on plain text.
    assert pattern.search("The quick brown fox, 123 -- café 中文") is None


def test_the_widened_domain_explains_the_46_new_code_points() -> None:
    """The 46 additions are exactly those Python's Unicode tables do not know.

    "Python's Unicode tables" carries a version, and the reading says which
    one it means: the archived probe domain was filtered through CPython
    3.12's tables, Unicode 15.0.0. An interpreter carrying a different
    edition draws the assigned/unassigned line elsewhere -- under Unicode
    13.0.0, the edition CPython 3.10 ships, 50 of the 108 archived code
    points are not assigned yet -- so the exact correspondence belongs to
    the tables the reading names. This cell reads that version back out of
    the reading and asks for the full 46/46 and 108/108 match only when the
    running tables are those. On any other edition it keeps the half that
    holds in both directions -- whatever these tables do know of the
    archived set, they still give a reordering combining class -- and counts
    the ones they do not know.

    The product does not depend on any of this: the shipped guard is
    compiled from the 154 code points the registry carries, and the hot path
    never consults ``unicodedata``. What follows corroborates how the
    archived set came to be the smaller one, rather than standing in for the
    guarantee itself.
    """
    import unicodedata

    reading = json.loads(
        (ROOT / "readings" / "fastokens_pinned_guard_full_domain.json").read_text(
            encoding="utf-8"
        )
    )
    derived_with = reading["archived_guard"]["derived_with"]
    named = re.search(r"Unicode (\d+\.\d+\.\d+)", derived_with)
    assert named is not None, derived_with
    same_tables = unicodedata.unidata_version == named.group(1)

    added = [int(cp[2:], 16) for cp in reading["comparison"]["only_in_full_domain"]]
    assert len(added) == 46
    if same_tables:
        for cp in added:
            # Unassigned in CPython's tables, hence outside the archived probe
            # domain; combining class 0 there, while the engine reorders them.
            assert unicodedata.category(chr(cp)) == "Cn"
            assert unicodedata.combining(chr(cp)) == 0

    archived = [int(cp[2:], 16) for cp in reading["archived_guard"]["codepoints"]]
    assert len(archived) == 108
    unknown = [cp for cp in archived if unicodedata.category(chr(cp)) == "Cn"]
    known = [cp for cp in archived if unicodedata.category(chr(cp)) != "Cn"]
    assert len(known) + len(unknown) == len(archived)
    for cp in known:
        assert unicodedata.combining(chr(cp)) > 1, (
            f"U+{cp:04X} is assigned under Unicode "
            f"{unicodedata.unidata_version} yet its combining class is "
            f"{unicodedata.combining(chr(cp))}"
        )
    if same_tables:
        assert unknown == [], (
            f"{len(unknown)} of the archived code points are unassigned under "
            "the tables the reading names"
        )
        assert len(known) == 108
