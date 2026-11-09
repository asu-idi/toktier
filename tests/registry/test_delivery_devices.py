"""Per record, the prebuilt GPU delivery judges every device the JIT one does.

``gpu_delivery="auto"`` prefers the prebuilt delivery whenever it can
serve; ``tools/generate_registry.py --check`` keeps "prebuilt judged
devices include the JIT judged devices" an invariant of the registry, so
that preference never trades a judged device for an unjudged one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import generate_registry

ROOT = Path(__file__).resolve().parents[2]
SHIPPED = ROOT / "tables" / "support_registry.json"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _row(document: dict[str, Any], family: str) -> dict[str, Any]:
    return next(item for item in document["artifacts"] if item["family"] == family)


def _dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")



def test_prebuilt_devices_include_the_jit_devices_on_the_shipped_registry() -> None:
    assert generate_registry.delivery_device_problems(SHIPPED) == []


def test_a_jit_device_outside_the_prebuilt_judgement_is_refused(
    tmp_path: Path,
) -> None:
    document = _load(SHIPPED)
    record = _row(document, "deepseek_v3")
    deliveries = record["backends"]["gpu"]["deliveries"]
    deliveries["prebuilt"]["devices"] = [
        device for device in deliveries["prebuilt"]["devices"] if device != "sm_89"
    ]
    path = tmp_path / "registry.json"
    _dump(path, document)
    problems = generate_registry.delivery_device_problems(path)
    assert problems == [
        "deepseek_v3: the JIT delivery is judged on ['sm_89'] where the prebuilt "
        'delivery is not; gpu_delivery="auto" would prefer an unjudged prebuilt image'
    ]
