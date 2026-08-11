# Evidence carry-over records for the 0.2 line

Machine-readable `evidence_carryover.v1` records for this minor version are
added here as JSON files. Records are add-only: after review they are not
edited, replaced, or deleted. The original registry and readings records stay
in their existing locations and are referenced through `carried_evidence`
JSON pointers.

An empty directory means that no certification evidence has yet used the
exception channel on this minor line. `tools/verify_carryover.py --check`
validates every JSON record recursively from `evidence/carryover/`.
