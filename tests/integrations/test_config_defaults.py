"""Every default of the SGLang adapter and scheduler configurations.

A wrapper built without options must serve exactly as before: no
chunking, no process pool, no shadow check, identity required. Each
field default is pinned here, so changing one is a reviewed edit of
this table rather than a silent change of behavior.
"""

from __future__ import annotations

import dataclasses

from toktier.integrations.sglang import AdapterConfig
from toktier.integrations.sglang_sched import SchedulerConfig

ADAPTER_DEFAULTS: dict[str, object] = {
    "family": None,
    "repo_id": None,
    "session_key": "auto",
    "store": None,
    "per_process_store": True,
    "shadow": "off",
    "shadow_rate": 1.0,
    "shadow_dump_dir": None,
    "shadow_queue_size": 256,
    "timing_log": None,
    "stats_file": None,
    "explain_every": 0,
    "rid_separator": None,
    "long_request_encoder": None,
    "long_request_min_chars": 1 << 30,
    "chunked_min_chars": 0,
    "chunked_k": 0,
    "device": "cpu",
    "identity": "require",
    "scheduler": None,
}

SCHEDULER_DEFAULTS: dict[str, object] = {
    "workers": 0,
    "cores": (),
    "policy": "srpt",
    "aging": 1.0,
    "chunk_min_chars": 131_072,
    "chunk_target": 65_536,
    "kmax": 16,
    "k_rule": "fixed",
    "affinity": True,
    "ms_fixed": 0.2,
    "ms_per_char": 3.0e-5,
    "non_ascii_ratio": 5.0,
    "ms_per_char_session": 1.0e-5,
    "calibrate": True,
    "calibrate_alpha": 0.05,
    "worker_timeout_s": 60.0,
    "start_timeout_s": 300.0,
    "rayon_threads": 1,
    "loader": None,
    "loader_path": None,
}


def _defaults(config: object) -> dict[str, object]:
    assert dataclasses.is_dataclass(config)
    return {
        field.name: getattr(config, field.name)
        for field in dataclasses.fields(config)
    }


def test_adapter_config_defaults_are_the_table() -> None:
    assert _defaults(AdapterConfig()) == ADAPTER_DEFAULTS


def test_scheduler_config_defaults_are_the_table() -> None:
    assert _defaults(SchedulerConfig()) == SCHEDULER_DEFAULTS


def test_default_adapter_turns_nothing_on() -> None:
    config = AdapterConfig()
    assert config.scheduler is None
    assert config.chunked_min_chars == 0
    assert config.shadow == "off"
    assert config.identity == "require"
    assert SchedulerConfig().workers == 0
