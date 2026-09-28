from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aegis_staircases.contracts import canonical_hash


ROOT = Path(__file__).resolve().parents[2]
EVALUATION_CONFIG_PATH = ROOT / "configs" / "evaluation.v1.json"


def load_evaluation_config(path: Path = EVALUATION_CONFIG_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def expected_main_source_ids(count: int = 500) -> list[str]:
    return [f"arxiv2k:{index:04d}" for index in range(count)]


def validate_locked_inputs(gate_material: dict[str, Any], evaluation_lock: dict[str, Any]) -> None:
    """Require current ordered input hashes to match the gate-bound representation lock."""
    current = gate_material.get("inputs")
    locked = evaluation_lock.get("inputs")
    if not isinstance(current, dict) or not isinstance(locked, dict):
        raise RuntimeError("Gate or evaluation lock is missing ordered input hashes")
    human_hashes = current.get("human_text_hashes")
    generated_outputs = current.get("generated_outputs")
    if locked.get("human_text_hashes") != human_hashes:
        raise RuntimeError("Current human input hashes differ from the representation lock")
    if locked.get("generated_outputs") != generated_outputs:
        raise RuntimeError("Current generated input hashes differ from the representation lock")
    if locked.get("human_text_hashes_sha256") != canonical_hash(human_hashes):
        raise RuntimeError("Locked human input-hash aggregate is invalid")
    if locked.get("generated_text_contract_sha256") != canonical_hash(generated_outputs):
        raise RuntimeError("Locked generated input-hash aggregate is invalid")
