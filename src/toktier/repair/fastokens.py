"""Explicit experimental full-reencode callback backed by Fastokens.

Fastokens is useful as a fast BPE implementation, but TokTier does not grant it
the corrected-Gigatoken exact-ID certificate.  This adapter is therefore never
automatic: the caller must select it together with ``EXPERIMENTAL`` policy.
It re-encodes the complete session tail on every append and reports that fact.
If its token-byte spans cannot be reconstructed, it falls back to the HF
reference callback instead of committing malformed session state.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol

from ..errors import BackendUnavailable
from ..policy import BACKEND_FASTOKENS
from .gigatoken import (
    ReferenceEncode,
    WindowUnsupported,
    _byte_lengths_from_hf,
    _spans_from_ids,
)
from .registry import RepairFamily

__all__ = ["FastokensFullRepair", "fastokens_distribution_identity"]


def fastokens_distribution_identity() -> tuple[str | None, str | None]:
    """Installed version and content digest, without importing Fastokens."""
    # Keep importlib.metadata off ``import toktier``: it pulls in the socket
    # module on CPython.  Probing an explicitly requested optional backend may
    # load it; importing the public package may not.
    from importlib.metadata import PackageNotFoundError, distribution

    try:
        dist = distribution("fastokens")
    except PackageNotFoundError:
        return None, None
    digest = hashlib.sha256(b"toktier.fastokens.distribution.v1\0")
    code_files = sorted(
        (
            item
            for item in (dist.files or ())
            if item.parts
            and item.parts[0] == "fastokens"
            and "__pycache__" not in item.parts
        ),
        key=str,
    )
    if not code_files:
        return dist.version, None
    try:
        for item in code_files:
            raw = Path(str(dist.locate_file(item))).read_bytes()
            name = str(item).encode("utf-8")
            digest.update(len(name).to_bytes(4, "little"))
            digest.update(name)
            digest.update(hashlib.sha256(raw).digest())
    except OSError:
        return dist.version, None
    return dist.version, digest.hexdigest()


class _Encoding(Protocol):
    @property
    def ids(self) -> Sequence[int]: ...


class _Tokenizer(Protocol):
    def encode(
        self, text: str, add_special_tokens: bool = False
    ) -> _Encoding: ...


class _Factory(Protocol):
    @staticmethod
    def from_file(path: str) -> _Tokenizer: ...


class _FastTokenizer(Protocol):
    def get_vocab(self) -> Mapping[str, int]: ...


class FastokensFullRepair:
    """Full-session Fastokens callback, explicitly outside certification."""

    def __init__(
        self,
        *,
        spec: RepairFamily,
        engine: _Tokenizer,
        engine_version: str,
        engine_digest: str,
        hf_tokenizer: _FastTokenizer,
        reference_encode: ReferenceEncode,
    ) -> None:
        self.spec = spec
        self._engine = engine
        self._engine_version = engine_version
        self._engine_digest = engine_digest
        self._reference_encode = reference_encode
        self._byte_lengths = tuple(_byte_lengths_from_hf(hf_tokenizer))
        self._path_counts: dict[str, int] = {}
        self._last: dict[str, object] | None = None

    @classmethod
    def open(
        cls,
        *,
        spec: RepairFamily,
        tokenizer_path: Path,
        hf_tokenizer: _FastTokenizer,
        reference_encode: ReferenceEncode,
    ) -> FastokensFullRepair:
        """Load Fastokens from the already verified tokenizer artifact."""
        from importlib import import_module
        from importlib.metadata import PackageNotFoundError

        try:
            installed, digest = fastokens_distribution_identity()
            if installed is None:
                raise PackageNotFoundError("fastokens")
            if digest is None:
                raise OSError("Fastokens distribution content cannot be hashed")
            module = import_module("fastokens")
            factory: _Factory = module.Tokenizer
            engine = factory.from_file(str(tokenizer_path))
        except (PackageNotFoundError, ImportError, AttributeError) as error:
            raise BackendUnavailable(
                "the experimental Fastokens repair backend is not installed",
                details={
                    "backend": BACKEND_FASTOKENS,
                    "extra": "fastokens",
                },
            ) from error
        except (OSError, RuntimeError, ValueError) as error:
            raise BackendUnavailable(
                f"Fastokens could not load the verified artifact: {error}",
                details={
                    "backend": BACKEND_FASTOKENS,
                    "stage": "engine_load",
                },
            ) from error
        try:
            return cls(
                spec=spec,
                engine=engine,
                engine_version=installed,
                engine_digest=digest,
                hf_tokenizer=hf_tokenizer,
                reference_encode=reference_encode,
            )
        except (AttributeError, TypeError, ValueError, WindowUnsupported) as error:
            raise BackendUnavailable(
                f"Fastokens cannot use this verified artifact: {error}",
                details={
                    "backend": BACKEND_FASTOKENS,
                    "stage": "span_table",
                    "family": spec.family,
                },
            ) from error

    @property
    def config_id(self) -> str:
        return "toktier-fastokens-full-experimental-v1"

    def _count(self, path: str) -> None:
        self._path_counts[path] = self._path_counts.get(path, 0) + 1

    def _reference_full(
        self, text: str, *, reason: str, detail: object | None = None
    ) -> tuple[list[int], list[tuple[int, int]], int, str]:
        ids, spans = self._reference_encode(text)
        path = f"hf_full_fastokens_{reason}"
        self._count(path)
        self._last = {
            "path": path,
            "reason": reason,
            "detail": detail,
            "input_chars": len(text),
        }
        return ids, spans, 0, path

    def __call__(
        self,
        tail_text: str,
        tail_ids: list[int],
        tail_spans: Sequence[tuple[int, int]],
        delta: str,
    ) -> tuple[list[int], list[tuple[int, int]], int, str]:
        text = tail_text + delta
        if not delta:
            path = "fastokens_full_experimental_noop"
            self._count(path)
            self._last = {"path": path, "kept_tokens": len(tail_ids)}
            return list(tail_ids), list(tail_spans), len(tail_ids), path
        try:
            ids = [
                int(value)
                for value in self._engine.encode(
                    text, add_special_tokens=False
                ).ids
            ]
            spans = _spans_from_ids(ids, self._byte_lengths, text)
        except (
            OSError,
            RuntimeError,
            UnicodeError,
            ValueError,
            WindowUnsupported,
        ) as error:
            return self._reference_full(
                text,
                reason="guard",
                detail={"error": type(error).__name__, "message": str(error)},
            )
        path = "fastokens_full_experimental"
        self._count(path)
        self._last = {
            "path": path,
            "input_chars": len(text),
            "delta_chars": len(delta),
            "kept_tokens": 0,
        }
        return ids, spans, 0, path

    def stats(self) -> dict[str, object]:
        return {
            "backend": BACKEND_FASTOKENS,
            "engine": "fastokens",
            "engine_version": self._engine_version,
            "engine_digest": self._engine_digest,
            "config_id": self.config_id,
            "certification": "experimental",
            "exact_id_guarantee": False,
            "mode": "full_reencode",
            "family": self.spec.family,
            "artifact_sha256": self.spec.artifact_sha256,
            "path_counts": dict(sorted(self._path_counts.items())),
            "last": dict(self._last) if self._last is not None else None,
        }
