#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aegis_staircases.contracts import (  # noqa: E402
    canonical_hash,
    normalize_whitespace,
    read_jsonl,
    sha256_file,
    sha256_text,
)
from aegis_staircases.evaluation_contract import expected_main_source_ids  # noqa: E402
from aegis_staircases.prompts import build_prompt  # noqa: E402


def validate_result_cell(
    result_root: Path,
    *,
    suite: str,
    cell: str,
    account_alias: str,
    expected_phase: str | None = None,
    root: Path = ROOT,
) -> dict[str, object]:
    """Validate one complete result tree and return its machine-readable summary."""
    config_path = root / "configs" / "experiment.v1.json"
    manifest_dir = root / "data" / "manifests"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    queue = read_jsonl(manifest_dir / "freegen_queue.jsonl")
    exemplars = json.loads((manifest_dir / "exemplars.json").read_text(encoding="utf-8"))
    queue_by_id = {row["source_id"]: row for row in queue}

    result_root = result_root.expanduser().resolve()
    contract = json.loads((result_root / "_contract.json").read_text(encoding="utf-8"))
    complete = json.loads((result_root / "_complete.json").read_text(encoding="utf-8"))
    if contract["run_id"] != config["run_id"] or complete["run_id"] != config["run_id"]:
        raise RuntimeError("Run ID mismatch")
    if contract["suite"] != suite or contract["cell"] != cell:
        raise RuntimeError("Suite/cell mismatch")
    if contract["account_alias"] != account_alias:
        raise RuntimeError("Modal account alias mismatch")
    phase = str(contract["phase"])
    cohort = str(contract["cohort"])
    if expected_phase is not None and phase != expected_phase:
        raise RuntimeError(f"Phase mismatch: {phase} != {expected_phase}")
    if phase not in {"smoke", "pilot", "main"}:
        raise RuntimeError(f"Unknown phase: {phase}")
    expected_cohort = "main" if phase == "main" else "pilot"
    if cohort != expected_cohort:
        raise RuntimeError(f"Cohort/phase mismatch: {cohort}/{phase}")
    if contract["config_sha256"] != sha256_file(config_path):
        raise RuntimeError("Config hash mismatch")
    contract_hash = contract["contract_sha256"]
    unhashed_contract = {key: value for key, value in contract.items() if key != "contract_sha256"}
    if canonical_hash(unhashed_contract) != contract_hash:
        raise RuntimeError("Stored contract SHA-256 is not the canonical contract hash")
    model = config["models"][suite][cell]
    if contract["model"]["hf_id"] != model["hf_id"] or contract["model"]["revision"] != model["revision"]:
        raise RuntimeError("Pinned model mismatch")
    source_ids = contract["source_ids"]
    if len(source_ids) != len(set(source_ids)):
        raise RuntimeError("Duplicate source IDs in contract")
    if phase == "main" and source_ids != expected_main_source_ids():
        raise RuntimeError("Main result must contain arxiv2k:0000-0499 in compact-index order")
    if any(source_id not in queue_by_id for source_id in source_ids):
        raise RuntimeError("Contract contains a source absent from the frozen queue")

    fixed_tokens = int(config["decoding"]["fixed_new_tokens"])
    records = []
    for source_id in source_ids:
        expected = queue_by_id[source_id]
        source_dir = result_root / source_id.replace(":", "_")
        attempt_files = sorted(source_dir.glob("attempt-*.json"))
        if [path.name for path in attempt_files] != ["attempt-00.json"]:
            raise RuntimeError(f"Expected exactly attempt-00 for {source_id}: {attempt_files}")
        record = json.loads(attempt_files[0].read_text(encoding="utf-8"))
        if record["contract_sha256"] != contract["contract_sha256"]:
            raise RuntimeError(f"Contract hash mismatch for {source_id}")
        for field, expected_value in (
            ("run_id", config["run_id"]),
            ("suite", suite),
            ("cell", cell),
            ("cohort", cohort),
            ("phase", phase),
            ("model_hf_id", model["hf_id"]),
            ("model_revision", model["revision"]),
        ):
            if record[field] != expected_value:
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
        if sha256_text(build_prompt(expected, exemplars)) != record["prompt_sha256"]:
            raise RuntimeError(f"Rendered prompt hash mismatch: {source_id}")
        records.append(record)

    if complete["contract_sha256"] != contract_hash:
        raise RuntimeError("Completion summary contract hash mismatch")
    for field, expected_value in (
        ("suite", suite),
        ("cell", cell),
        ("cohort", cohort),
        ("phase", phase),
    ):
        if complete[field] != expected_value:
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
    if phase == "main" and split_counts != Counter({"train": 352, "val": 69, "test": 79}):
        raise RuntimeError(f"Main split-role counts changed: {dict(split_counts)}")
    result = {
        "status": "ok",
        "run_id": config["run_id"],
        "suite": suite,
        "cell": cell,
        "account_alias": account_alias,
        "cohort": cohort,
        "phase": phase,
        "sources": len(records),
        "attempts": len(records),
        "split_counts": dict(sorted(split_counts.items())),
        "fixed_tokens_each": fixed_tokens,
        "word_count_min": min(record["actual_words"] for record in records),
        "word_count_median": statistics.median(record["actual_words"] for record in records),
        "word_count_max": max(record["actual_words"] for record in records),
        "elapsed_seconds": complete["elapsed_seconds"],
        "model_revision": model["revision"],
        "contract_sha256": contract["contract_sha256"],
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a downloaded CLEAN-FG result cell")
    parser.add_argument("result_root", type=Path, help="Directory containing _contract.json and source folders")
    parser.add_argument("--suite", required=True, choices=["tulu", "pythia"])
    parser.add_argument("--cell", required=True)
    parser.add_argument(
        "--account-alias", required=True, choices=["account_1", "account_2_only"]
    )
    parser.add_argument("--phase", choices=["smoke", "pilot", "main"])
    args = parser.parse_args()
    result = validate_result_cell(
        args.result_root,
        suite=args.suite,
        cell=args.cell,
        account_alias=args.account_alias,
        expected_phase=args.phase,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
