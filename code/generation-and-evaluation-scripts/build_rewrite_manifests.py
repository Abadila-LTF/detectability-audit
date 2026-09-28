#!/usr/bin/env python3
"""Build the CLEAN-RW-v1 rewrite queue from the frozen V5 queue and verified human sources."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aegis_staircases.contracts import (  # noqa: E402
    atomic_write_json,
    canonical_hash,
    read_jsonl,
    sha256_file,
    sha256_text,
    validate_queue,
    write_jsonl,
)
from aegis_staircases.prompts import validate_exemplars  # noqa: E402
from aegis_staircases.rewrite_prompts import rewrite_prompt_contract  # noqa: E402

MANIFESTS = ROOT / "data" / "manifests"
REWRITE_QUEUE = MANIFESTS / "rewrite_queue.jsonl"
REWRITE_LOCK = MANIFESTS / "rewrite-manifest-lock.json"


def build_rewrite_queue() -> list[dict]:
    queue = read_jsonl(MANIFESTS / "freegen_queue.jsonl")
    validate_queue(queue)
    humans = {row["source_id"]: row for row in read_jsonl(MANIFESTS / "human_sources.jsonl")}
    rows = []
    for row in queue:
        human = humans[row["source_id"]]
        if human["human_sha256"] != row["human_sha256"] or sha256_text(human["human_text"]) != row["human_sha256"]:
            raise RuntimeError(f"Human source hash mismatch for {row['source_id']}")
        if human["raw_label"] != 0 or human["raw_row_index"] % 2 != 0:
            raise RuntimeError(f"Source is not a verified even-row human text: {row['source_id']}")
        rows.append({**row, "source_text": human["human_text"]})
    return rows


def main() -> None:
    exemplars = json.loads((MANIFESTS / "exemplars.json").read_text(encoding="utf-8"))
    validate_exemplars(exemplars)
    rows = build_rewrite_queue()
    write_jsonl(REWRITE_QUEUE, rows)
    lock = {
        "schema_version": "aegis.clean-rewrite.manifest-lock.v1",
        "run_id": "clean-rw-v1",
        "freegen_queue_sha256": sha256_file(MANIFESTS / "freegen_queue.jsonl"),
        "human_sources_sha256": sha256_file(MANIFESTS / "human_sources.jsonl"),
        "exemplars_sha256": sha256_file(MANIFESTS / "exemplars.json"),
        "v5_manifest_lock_sha256": sha256_file(MANIFESTS / "manifest-lock.json"),
        "rewrite_queue_sha256": sha256_file(REWRITE_QUEUE),
        "rewrite_queue_contract_sha256": canonical_hash(rows),
        "counts": {
            "rows": len(rows),
            "main": sum(row["cohort"] == "main" for row in rows),
            "pilot": sum(row["cohort"] == "pilot" for row in rows),
        },
        "prompt_contract": rewrite_prompt_contract(exemplars),
    }
    atomic_write_json(REWRITE_LOCK, lock)
    print(json.dumps(lock, indent=2))


if __name__ == "__main__":
    main()
