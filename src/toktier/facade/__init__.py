"""High-level facade surface: ``load``, ``Tokenizer``, ``Encoding``.

Contract reference: ``docs/contracts/facade.md``. Import cost note: this
package imports no oracle, no native store and no accelerator runtime;
construction loads the oracle and may perform a lightweight GPU probe, while
the native store and GPU engine remain lazy until their first relevant use.
"""

from __future__ import annotations

from .api import Encoding, Tokenizer, load

__all__ = ["Encoding", "Tokenizer", "load"]
