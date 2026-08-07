"""High-level facade surface: ``load``, ``Tokenizer``, ``Encoding``.

Contract reference: ``docs/contracts/facade.md``. Import cost note: this
package imports no oracle, no native store and no accelerator runtime;
those load at construction and first use respectively.
"""

from __future__ import annotations

from .api import Encoding, Tokenizer, load

__all__ = ["Encoding", "Tokenizer", "load"]
