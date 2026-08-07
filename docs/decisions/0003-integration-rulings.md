# 0003 - Integration rulings (2026-08-05)

1. **Native module name** - The private native module is `toktier._native`.
2. **Python 3.10 TOML support** - The conditional `tomli>=2.0` dependency is accepted for Python versions earlier than 3.11.
3. **Configuration file discovery** - The configuration file is read only when `TOKTIER_HOME` is set; this clarifies Section 6 of `config.md`.
4. **Log levels** - `log_level` values are normalized to upper case and validated against the standard logging level names.
5. **Experimental file key** - An `experimental` key in a configuration file raises `CONFIG_INVALID`.
6. **Malformed artifact manifests** - A malformed artifact manifest raises `REGISTRY_INVALID`.
7. **Missing hub dependency** - If `huggingface_hub` is missing at fetch time, the operation raises `ARTIFACT_NOT_FOUND` with `details.missing` identifying the dependency.
8. **CLI entry point** - The `[project.scripts]` entry `toktier=toktier.cli:main` arrives with the CLI lane.
