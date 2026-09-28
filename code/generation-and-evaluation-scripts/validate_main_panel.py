#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from aegis_staircases.contracts import (  # noqa: E402
    atomic_write_json,
    canonical_hash,
    read_jsonl,
    sha256_file,
    sha256_text,
)
from aegis_staircases.evaluation_contract import (  # noqa: E402
    expected_main_source_ids,
    load_evaluation_config,
)
from scripts.validate_results import validate_result_cell  # noqa: E402


def validate_main_panel(result_roots: Sequence[Path], *, root: Path = ROOT) -> dict[str, Any]:
    """Apply the all-nine-cell operational gate without computing representations or scores."""
    evaluation_path = root / "configs" / "evaluation.v1.json"
    experiment_path = root / "configs" / "experiment.v1.json"
    evaluation = load_evaluation_config(evaluation_path)
    expected_specs = evaluation["global_gate"]["cells"]
    expected_keys = {(spec["suite"], spec["cell"]) for spec in expected_specs}
    roots_by_key: dict[tuple[str, str], Path] = {}
    for unresolved in result_roots:
        result_root = unresolved.expanduser().resolve()
        contract_path = result_root / "_contract.json"
        if not contract_path.is_file():
            raise RuntimeError(f"Missing _contract.json: {result_root}")
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        key = (str(contract.get("suite")), str(contract.get("cell")))
        if key in roots_by_key:
            raise RuntimeError(f"Duplicate main result cell: {key}")
        roots_by_key[key] = result_root

    actual_keys = set(roots_by_key)
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    if missing or unexpected:
        raise RuntimeError(
            "Global main gate is closed; "
            f"missing={missing or 'none'}, unexpected={unexpected or 'none'}"
        )
    if len(roots_by_key) != 9:
        raise RuntimeError(f"Global main gate requires exactly 9 cells, found {len(roots_by_key)}")

    expected_ids = expected_main_source_ids(int(evaluation["global_gate"]["source_count_per_cell"]))
    human_rows = {
        row["source_id"]: row
        for row in read_jsonl(root / "data" / "manifests" / "human_sources.jsonl")
    }
    human_hashes = []
    for source_id in expected_ids:
        row = human_rows[source_id]
        if sha256_text(row["human_text"]) != row["human_sha256"]:
            raise RuntimeError(f"Frozen human text hash mismatch: {source_id}")
        human_hashes.append(row["human_sha256"])
    cell_reports = []
    contract_hashes = []
    generated_outputs = []
    for spec in expected_specs:
        key = (spec["suite"], spec["cell"])
        result_root = roots_by_key[key]
        report = validate_result_cell(
            result_root,
            suite=spec["suite"],
            cell=spec["cell"],
            account_alias=spec["account_alias"],
            expected_phase="main",
            root=root,
        )
        contract = json.loads((result_root / "_contract.json").read_text(encoding="utf-8"))
        if contract["source_ids"] != expected_ids:
            raise RuntimeError(f"Frozen source order changed in {key}")
        if report["split_counts"] != {"test": 79, "train": 352, "val": 69}:
            raise RuntimeError(f"Frozen split-role counts changed in {key}")
        contract_hashes.append(
            {"suite": spec["suite"], "cell": spec["cell"], "sha256": contract["contract_sha256"]}
        )
        generated_outputs.append(
            {
                "suite": spec["suite"],
                "cell": spec["cell"],
                "output_hashes": [
                    json.loads(
                        (
                            result_root
                            / source_id.replace(":", "_")
                            / "attempt-00.json"
                        ).read_text(encoding="utf-8")
                    )["normalized_output_sha256"]
                    for source_id in expected_ids
                ],
            }
        )
        cell_reports.append(report)

    gate_material = {
        "schema_version": evaluation["schema_version"],
        "run_id": evaluation["run_id"],
        "experiment_config_sha256": sha256_file(experiment_path),
        "evaluation_config_sha256": sha256_file(evaluation_path),
        "source_ids_sha256": canonical_hash(expected_ids),
        "contracts": contract_hashes,
        "inputs": {
            "human_text_hashes": human_hashes,
            "generated_outputs": generated_outputs,
        },
    }
    return {
        "status": "GLOBAL_MAIN_GATE_PASSED",
        "run_id": evaluation["run_id"],
        "scoring_unlocked": True,
        "cells": cell_reports,
        "cell_count": len(cell_reports),
        "source_count_per_cell": len(expected_ids),
        "split_source_counts": evaluation["splits"]["expected_source_counts"],
        "gate_material": gate_material,
        "gate_sha256": canonical_hash(gate_material),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the all-nine-cell CLEAN-FG-v5 main-panel gate without scoring"
    )
    parser.add_argument("result_roots", nargs="+", type=Path, help="Exactly nine main result roots")
    parser.add_argument("--output", type=Path, help="Optional path for an auditable gate report")
    args = parser.parse_args()
    report = validate_main_panel(args.result_roots)
    if args.output:
        atomic_write_json(args.output.expanduser().resolve(), report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
