#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aegis_staircases.contracts import read_jsonl, sha256_file, validate_queue  # noqa: E402
from aegis_staircases.prompts import build_prompt, prompt_contract, validate_exemplars  # noqa: E402


def main() -> None:
    manifest_dir = ROOT / "data" / "manifests"
    lock = json.loads((manifest_dir / "manifest-lock.json").read_text(encoding="utf-8"))
    queue = read_jsonl(manifest_dir / "freegen_queue.jsonl")
    humans = read_jsonl(manifest_dir / "human_sources.jsonl")
    exemplars = json.loads((manifest_dir / "exemplars.json").read_text(encoding="utf-8"))
    validate_queue(queue)
    validate_exemplars(exemplars)

    expected_hashes = {
        "human_sources_sha256": manifest_dir / "human_sources.jsonl",
        "freegen_queue_sha256": manifest_dir / "freegen_queue.jsonl",
        "exemplars_sha256": manifest_dir / "exemplars.json",
    }
    for field, path in expected_hashes.items():
        actual = sha256_file(path)
        if actual != lock[field]:
            raise RuntimeError(f"{field} mismatch: {actual} != {lock[field]}")
    if len(humans) != 2000 or len({row["source_id"] for row in humans}) != 2000:
        raise RuntimeError("Human source manifest is not exactly 2,000 unique sources")
    cohort_counts = Counter(row["cohort"] for row in queue)
    if cohort_counts != {"main": 500, "pilot": 50}:
        raise RuntimeError(f"Unexpected queue cohorts: {dict(cohort_counts)}")
    if prompt_contract(exemplars) != lock["prompt_contract"]:
        raise RuntimeError("Prompt contract mismatch")
    for row in queue:
        if "human_text" in row or "text" in row:
            raise RuntimeError(f"Target human text leaked into generation queue: {row['source_id']}")
        prompt = build_prompt(row, exemplars)
        human = humans[int(row["compact_index"])]
        if human["human_sha256"] != row["human_sha256"]:
            raise RuntimeError(f"Human link mismatch: {row['source_id']}")
        if human["human_text"] in prompt:
            raise RuntimeError(f"Target human text leaked into rendered prompt: {row['source_id']}")
    print(json.dumps({"status": "ok", "cohorts": dict(cohort_counts), "target_text_leakage": 0}, indent=2))


if __name__ == "__main__":
    main()
