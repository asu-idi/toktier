# Fastokens license materials

TokTier can call [Fastokens](https://github.com/crusoecloud/fastokens) as an
explicit experimental full-session backend. Fastokens is installed separately
from its own distribution and is not bundled in the TokTier wheel.

The files in this directory reproduce the `LICENSE` and `NOTICES.txt` text
from Fastokens `v0.3.1`, commit
`fe854299553524f2156a22036a2cb4d1f2ef4d97`. They are retained here so the
optional integration's license and upstream attributions remain available in
source distributions and repository archives. Their text is byte-identical
after ignoring the conventional terminal newline added by this repository.

TokTier does not grant this adapter an exact-ID certificate. It is reachable
only with `policy="experimental", repair_backend="fastokens"` and reports
`exact_id_guarantee: false`.
