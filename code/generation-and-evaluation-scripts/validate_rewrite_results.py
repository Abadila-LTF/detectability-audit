#!/usr/bin/env python3
"""Strict CLEAN-RW-v1 validation (docs/PREREGISTRATION-RW-v1.md).

Mirrors the locked scripts/validate_results.py::validate_result_cell check-for-check, with
the rewrite config, rewrite queue, rewrite prompt, and frozen rewrite cell owners. The
`panel` command is the four-cell global rewrite gate (analogue of validate_main_panel.py).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aegis_staircases.contracts import (  # noqa: E402
    atomic_write_json,
    canonical_hash,
    normalize_whitespace,
    read_jsonl,
    sha256_file,
    sha256_text,
)
from aegis_staircases.evaluation_contract import expected_main_source_ids  # noqa: E402
from aegis_staircases.rewrite_prompts import build_rewrite_prompt, rewrite_prompt_contract  # noqa: E402

CONFIG_PATH = ROOT / "configs" / "experiment.rw-v1.json"
REWRITE_CELLS = ("base", "sft", "dpo", "rlvr")
MAIN_SPLITS = {"test": 79, "train": 352, "val": 69}


def validate_rewrite_cell(
    result_root: Path, *, cell: str, expected_phase: str | None = None, root: Path = ROOT
) -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    queue = read_jsonl(root / config["manifests"]["queue"])
    lock = json.loads((root / config["manifests"]["lock"]).read_text(encoding="utf-8"))
    exemplars = json.loads((root / "data" / "manifests" / "exemplars.json").read_text(encoding="utf-8"))
    if lock["prompt_contract"] != rewrite_prompt_contract(exemplars):
        raise RuntimeError("Rewrite prompt contract differs from the manifest lock")
    if sha256_file(root / config["manifests"]["queue"]) != lock["rewrite_queue_sha256"]:
        raise RuntimeError("Rewrite queue hash differs from the manifest lock")
    queue_by_id = {row["source_id"]: row for row in queue}
    if cell not in REWRITE_CELLS:
        raise RuntimeError(f"Unknown rewrite cell: {cell}")
    account_alias = config["cell_accounts"][cell]

    result_root = result_root.expanduser().resolve()
    contract = json.loads((result_root / "_contract.json").read_text(encoding="utf-8"))
    complete = json.loads((result_root / "_complete.json").read_text(encoding="utf-8"))
    if contract["run_id"] != config["run_id"] or complete["run_id"] != config["run_id"]:
        raise RuntimeError("Run ID mismatch")
    if contract["suite"] != "tulu" or contract["cell"] != cell:
        raise RuntimeError("Suite/cell mismatch")
    if contract["account_alias"] != account_alias:
        raise RuntimeError("Modal account alias mismatch")
    if contract.get("task") != "rewrite":
        raise RuntimeError("Contract is not a rewrite contract")
    if contract["rewrite_queue_sha256"] != lock["rewrite_queue_sha256"]:
        raise RuntimeError("Contract rewrite queue hash mismatch")
    if contract["prompt_contract"] != lock["prompt_contract"]:
        raise RuntimeError("Contract prompt contract mismatch")
    phase = str(contract["phase"])
    cohort = str(contract["cohort"])
    if expected_phase is not None and phase != expected_phase:
        raise RuntimeError(f"Phase mismatch: {phase} != {expected_phase}")
    if phase not in {"smoke", "pilot", "main"}:
        raise RuntimeError(f"Unknown phase: {phase}")
    if cohort != ("main" if phase == "main" else "pilot"):
        raise RuntimeError(f"Cohort/phase mismatch: {cohort}/{phase}")
    if contract["config_sha256"] != sha256_file(CONFIG_PATH):
        raise RuntimeError("Config hash mismatch")
    contract_hash = contract["contract_sha256"]
    if canonical_hash({k: v for k, v in contract.items() if k != "contract_sha256"}) != contract_hash:
        raise RuntimeError("Stored contract SHA-256 is not the canonical contract hash")
    model = config["models"]["tulu"][cell]
    if contract["model"]["hf_id"] != model["hf_id"] or contract["model"]["revision"] != model["revision"]:
        raise RuntimeError("Pinned model mismatch")
    source_ids = contract["source_ids"]
    if len(source_ids) != len(set(source_ids)):
        raise RuntimeError("Duplicate source IDs in contract")
    expected_count = {"smoke": 3, "pilot": 50, "main": 500}[phase]
    expected_ids = (
        expected_main_source_ids()
        if phase == "main"
        else [row["source_id"] for row in queue if row["cohort"] == "pilot"][:expected_count]
    )
    if source_ids != expected_ids:
        raise RuntimeError(f"{phase} result must contain the frozen {phase} IDs in order")

    fixed_tokens = int(config["decoding"]["fixed_new_tokens"])
    records = []
    for source_id in source_ids:
        expected = queue_by_id[source_id]
        source_dir = result_root / source_id.replace(":", "_")
        attempt_files = sorted(source_dir.glob("attempt-*.json"))
        if [path.name for path in attempt_files] != ["attempt-00.json"]:
            raise RuntimeError(f"Expected exactly attempt-00 for {source_id}: {attempt_files}")
        record = json.loads(attempt_files[0].read_text(encoding="utf-8"))
        if record["contract_sha256"] != contract_hash:
            raise RuntimeError(f"Contract hash mismatch for {source_id}")
        for field, value in (
            ("run_id", config["run_id"]),
            ("suite", "tulu"),
            ("cell", cell),
            ("cohort", cohort),
            ("phase", phase),
            ("model_hf_id", model["hf_id"]),
            ("model_revision", model["revision"]),
        ):
            if record[field] != value:
                raise RuntimeError(f"{field} mismatch for {source_id}")
        for field in ("source_id", "human_sha256", "topic_sha256", "compact_index", "split_origin"):
            if record[field] != expected[field]:
                raise RuntimeError(f"{field} mismatch for {source_id}")
        if record["attempt"] != 0 or record["status"] != "accepted":
            raise RuntimeError(f"Endpoint was not accepted on sole attempt: {source_id}")
        if record["output_tokens"] != fixed_tokens or len(record["output_token_ids"]) != fixed_tokens:
            raise RuntimeError(f"Fixed token contract failed: {source_id}")
        if not record["exact_token_count"] or not record["normalized_output"]:
            raise RuntimeError(f"Empty or inexact output: {source_id}")
        if normalize_whitespace(record["raw_output"]) != record["normalized_output"]:
            raise RuntimeError(f"Normalization mismatch: {source_id}")
        if sha256_text(record["raw_output"]) != record["raw_output_sha256"]:
            raise RuntimeError(f"Raw output hash mismatch: {source_id}")
        if sha256_text(record["normalized_output"]) != record["normalized_output_sha256"]:
            raise RuntimeError(f"Normalized output hash mismatch: {source_id}")
        if sha256_text(build_rewrite_prompt(expected, exemplars)) != record["prompt_sha256"]:
            raise RuntimeError(f"Rendered rewrite prompt hash mismatch: {source_id}")
        records.append(record)

    if complete["contract_sha256"] != contract_hash:
        raise RuntimeError("Completion summary contract hash mismatch")
    for field, value in (("suite", "tulu"), ("cell", cell), ("cohort", cohort), ("phase", phase)):
        if complete[field] != value:
            raise RuntimeError(f"Completion summary {field} mismatch")
    if (
        not complete["complete"]
        or complete["requested_sources"] != len(source_ids)
        or complete["completed_sources"] != len(source_ids)
        or complete["accepted_sources"] != len(source_ids)
        or complete["exhausted_sources"] != 0
    ):
        raise RuntimeError("Completion summary does not report full acceptance")
    split_counts = Counter(record["split_origin"] for record in records)
    if phase == "main" and dict(split_counts) != MAIN_SPLITS:
        raise RuntimeError(f"Main split-role counts changed: {dict(split_counts)}")
    return {
        "status": "ok",
        "run_id": config["run_id"],
        "suite": "tulu",
        "cell": cell,
        "account_alias": account_alias,
        "cohort": cohort,
        "phase": phase,
        "sources": len(records),
        "attempts": len(records),
        "split_counts": dict(sorted(split_counts.items())),
        "fixed_tokens_each": fixed_tokens,
        "word_count_min": min(r["actual_words"] for r in records),
        "word_count_median": statistics.median(r["actual_words"] for r in records),
        "word_count_max": max(r["actual_words"] for r in records),
        "elapsed_seconds": complete["elapsed_seconds"],
        "model_revision": model["revision"],
        "contract_sha256": contract_hash,
    }


def validate_rewrite_panel(result_roots: Sequence[Path], *, root: Path = ROOT) -> dict[str, Any]:
    """Four-cell global rewrite gate over 2,000 main endpoints; no scoring."""
    roots: dict[str, Path] = {}
    for unresolved in result_roots:
        result_root = unresolved.expanduser().resolve()
        contract = json.loads((result_root / "_contract.json").read_text(encoding="utf-8"))
        cell = str(contract["cell"])
        if contract["suite"] != "tulu" or cell in roots:
            raise RuntimeError(f"Unexpected or duplicate rewrite cell: {contract['suite']}/{cell}")
        roots[cell] = result_root
    if sorted(roots) != sorted(REWRITE_CELLS):
        raise RuntimeError(f"Global rewrite gate needs exactly {REWRITE_CELLS}; got {sorted(roots)}")
    expected_ids = expected_main_source_ids()
    human_rows = {row["source_id"]: row for row in read_jsonl(root / "data" / "manifests" / "human_sources.jsonl")}
    human_hashes = []
    for source_id in expected_ids:
        if sha256_text(human_rows[source_id]["human_text"]) != human_rows[source_id]["human_sha256"]:
            raise RuntimeError(f"Frozen human text hash mismatch: {source_id}")
        human_hashes.append(human_rows[source_id]["human_sha256"])
    reports, contracts, outputs = [], [], []
    for cell in REWRITE_CELLS:
        report = validate_rewrite_cell(roots[cell], cell=cell, expected_phase="main", root=root)
        contract = json.loads((roots[cell] / "_contract.json").read_text(encoding="utf-8"))
        contracts.append({"suite": "tulu", "cell": cell, "sha256": contract["contract_sha256"]})
        outputs.append(
            {
                "suite": "tulu",
                "cell": cell,
                "output_hashes": [
                    json.loads((roots[cell] / s.replace(":", "_") / "attempt-00.json").read_text(encoding="utf-8"))[
                        "normalized_output_sha256"
                    ]
                    for s in expected_ids
                ],
            }
        )
        reports.append(report)
    material = {
        "schema_version": "aegis.clean-rewrite.global-gate.v1",
        "run_id": "clean-rw-v1",
        "experiment_config_sha256": sha256_file(CONFIG_PATH),
        "source_ids_sha256": canonical_hash(expected_ids),
        "contracts": contracts,
        "inputs": {"human_text_hashes": human_hashes, "generated_outputs": outputs},
    }
    return {
        "status": "GLOBAL_REWRITE_GATE_PASSED",
        "run_id": "clean-rw-v1",
        "scoring_unlocked": True,
        "cells": reports,
        "cell_count": len(reports),
        "source_count_per_cell": len(expected_ids),
        "gate_material": material,
        "gate_sha256": canonical_hash(material),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    cell_parser = sub.add_parser("cell")
    cell_parser.add_argument("result_root", type=Path)
    cell_parser.add_argument("--cell", required=True, choices=REWRITE_CELLS)
    cell_parser.add_argument("--phase", choices=["smoke", "pilot", "main"])
    panel_parser = sub.add_parser("panel")
    panel_parser.add_argument("result_roots", nargs="+", type=Path)
    panel_parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "cell":
        result = validate_rewrite_cell(args.result_root, cell=args.cell, expected_phase=args.phase)
    else:
        result = validate_rewrite_panel(args.result_roots)
        if args.output:
            atomic_write_json(args.output.expanduser().resolve(), result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
