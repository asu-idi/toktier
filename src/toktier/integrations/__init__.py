"""Adapters that let serving frameworks call toktier as a tokenizer backend.

Each submodule is imported on demand and keeps its framework imports
deferred, so ``import toktier`` stays free of framework dependencies.
"""
