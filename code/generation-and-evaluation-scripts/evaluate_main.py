#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from aegis_staircases.contracts import atomic_write_json, canonical_hash, sha256_file  # noqa: E402
from aegis_staircases.evaluation import (  # noqa: E402
    array_contract_hash,
    auroc_to_dprime,
    exact_page_test,
    exact_spearman_test,
    fit_detector,
    load_evaluation_config,
    paired_source_bootstrap,
    pythia_ceiling_audit,
    sourcewise_auc_contributions,
    tulu_ceiling_audit,
)
from aegis_staircases.evaluation_contract import validate_locked_inputs  # noqa: E402
from scripts.validate_main_panel import validate_main_panel  # noqa: E402


def _load_views(
    path: Path, gate: dict[str, Any], evaluation: dict[str, Any]
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    views_path = path.expanduser().resolve()
    manifest_path = views_path.with_suffix(".manifest.json")
    if not views_path.is_file() or not manifest_path.is_file():
        raise RuntimeError("Both the frozen .npz views and its .manifest.json are required")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["global_gate_sha256"] != gate["gate_sha256"]:
        raise RuntimeError("RoBERTa views were built from a different global gate")
    if manifest["evaluation_config_sha256"] != sha256_file(ROOT / "configs" / "evaluation.v1.json"):
        raise RuntimeError("RoBERTa views were built under a different evaluation contract")
    if manifest["artifact"]["sha256"] != sha256_file(views_path):
        raise RuntimeError("RoBERTa view artifact hash mismatch")
    if manifest["representation"] != evaluation["representation"]:
        raise RuntimeError("RoBERTa representation contract mismatch")
    lock_config = evaluation["evaluation_lock"]
    lock_path = views_path.with_suffix(lock_config["artifact_suffix"])
    if not lock_path.is_file():
        raise RuntimeError("Gate-bound evaluation lock is missing")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if manifest["evaluation_lock"]["path"] != lock_path.name:
        raise RuntimeError("View manifest points to a different evaluation lock")
    if manifest["evaluation_lock"]["sha256"] != sha256_file(lock_path):
        raise RuntimeError("Evaluation-lock file hash mismatch")
    lock_material = {key: value for key, value in lock.items() if key != "lock_sha256"}
    if canonical_hash(lock_material) != lock["lock_sha256"]:
        raise RuntimeError("Evaluation-lock canonical hash mismatch")
    if manifest["evaluation_lock"]["lock_sha256"] != lock["lock_sha256"]:
        raise RuntimeError("View manifest evaluation-lock identity mismatch")
    validate_locked_inputs(gate["gate_material"], lock)
    if lock["global_gate"] != gate["gate_material"] or lock["global_gate_sha256"] != gate["gate_sha256"]:
        raise RuntimeError("Evaluation lock was built from a different nine-cell gate")
    expected_configs = [
        {"path": "configs/experiment.v1.json", "sha256": sha256_file(ROOT / "configs" / "experiment.v1.json")},
        {"path": "configs/evaluation.v1.json", "sha256": sha256_file(ROOT / "configs" / "evaluation.v1.json")},
    ]
    if lock["configs"] != expected_configs:
        raise RuntimeError("Evaluation config hashes changed after representation extraction")
    expected_code = [
        {"path": relative, "sha256": sha256_file(ROOT / relative)}
        for relative in lock_config["code_files"]
    ]
    if lock["code"] != expected_code:
        raise RuntimeError("Evaluator code hashes changed after representation extraction")
    expected_packages = {
        distribution: importlib.metadata.version(distribution)
        for distribution in lock_config["package_distributions"]
    }
    if lock["packages"] != {
        "python": platform.python_version(),
        "distributions": expected_packages,
    }:
        raise RuntimeError("Evaluation package versions changed after representation extraction")
    if lock["roberta"]["hf_id"] != evaluation["representation"]["model_hf_id"]:
        raise RuntimeError("Evaluation-lock RoBERTa identity mismatch")
    if lock["roberta"]["revision"] != evaluation["representation"]["model_revision"]:
        raise RuntimeError("Evaluation-lock RoBERTa revision mismatch")
    arrays = np.load(views_path, allow_pickle=False)
    required_arrays = {
        "human_embeddings",
        "generated_embeddings",
        "source_ids",
        "split_origins",
        "suites",
        "cells",
    }
    if set(arrays.files) != required_arrays:
        raise RuntimeError(f"Unexpected RoBERTa view arrays: {arrays.files}")
    if arrays["human_embeddings"].shape != (500, 768):
        raise RuntimeError("Human representation shape changed")
    if arrays["generated_embeddings"].shape != (9, 500, 768):
        raise RuntimeError("Generated representation shape changed")
    specs = evaluation["global_gate"]["cells"]
    expected_suites = [spec["suite"] for spec in specs]
    expected_cells = [spec["cell"] for spec in specs]
    if arrays["suites"].tolist() != expected_suites or arrays["cells"].tolist() != expected_cells:
        raise RuntimeError("Representation cell order changed")
    expected_ids = [f"arxiv2k:{index:04d}" for index in range(500)]
    if arrays["source_ids"].tolist() != expected_ids:
        raise RuntimeError("Representation source order changed")
    split_counts = {
        split: int(np.sum(arrays["split_origins"] == split)) for split in ("train", "val", "test")
    }
    if split_counts != evaluation["splits"]["expected_source_counts"]:
        raise RuntimeError(f"Representation split-role counts changed: {split_counts}")
    if lock["embedding_cache"]["path"] != views_path.name:
        raise RuntimeError("Evaluation lock names a different embedding cache")
    if lock["embedding_cache"]["sha256"] != sha256_file(views_path):
        raise RuntimeError("Evaluation-lock embedding cache hash mismatch")
    actual_array_hashes = {name: array_contract_hash(arrays[name]) for name in arrays.files}
    if lock["embedding_cache"]["arrays"] != actual_array_hashes:
        raise RuntimeError("Evaluation-lock per-array hashes do not match the embedding cache")
    for split in ("train", "val", "test"):
        ids = arrays["source_ids"][arrays["split_origins"] == split].tolist()
        if lock["splits"][split]["source_ids"] != ids:
            raise RuntimeError(f"Evaluation-lock {split} IDs changed")
        if lock["splits"][split]["count"] != len(ids):
            raise RuntimeError(f"Evaluation-lock {split} count changed")
        if lock["splits"][split]["source_ids_sha256"] != canonical_hash(ids):
            raise RuntimeError(f"Evaluation-lock {split} ID hash changed")
    return arrays, manifest, lock


def evaluate(
    result_roots: Sequence[Path], views_path: Path, output: Path
) -> dict[str, Any]:
    # The global result gate and the gate-bound representation hash are checked
    # before fitting a detector or exposing a test score.
    gate = validate_main_panel(result_roots)
    if gate["status"] != "GLOBAL_MAIN_GATE_PASSED" or not gate["scoring_unlocked"]:
        raise RuntimeError("Global main gate did not pass")
    evaluation = load_evaluation_config()
    arrays, views_manifest, evaluation_lock = _load_views(views_path, gate, evaluation)

    classifier = evaluation["classifier"]
    specs = evaluation["global_gate"]["cells"]
    split_origins = arrays["split_origins"].tolist()
    test_mask = arrays["split_origins"] == "test"
    test_source_ids = arrays["source_ids"][test_mask].tolist()
    fits = {}
    test_score_pairs = {}
    cell_results = {}
    for index, spec in enumerate(specs):
        key = f"{spec['suite']}/{spec['cell']}"
        fit = fit_detector(
            arrays["human_embeddings"],
            arrays["generated_embeddings"][index],
            split_origins,
            c_grid=classifier["c_grid"],
            max_iter=int(classifier["max_iter"]),
            tolerance=float(classifier["tolerance"]),
            random_seed=int(classifier["random_seed"]),
        )
        fits[key] = fit
        test_score_pairs[key] = (fit.human_scores["test"], fit.generated_scores["test"])
        contributions = sourcewise_auc_contributions(
            fit.human_scores["test"], fit.generated_scores["test"]
        )
        cell_results[key] = {
            "suite": spec["suite"],
            "cell": spec["cell"],
            "selected_c": fit.selected_c,
            "validation_auroc": fit.validation_auroc,
            "validation_grid": fit.validation_grid,
            "test_auroc": fit.test_auroc,
            "test_dprime": auroc_to_dprime(fit.test_auroc),
            "test_human_scores": fit.human_scores["test"].tolist(),
            "test_generated_scores": fit.generated_scores["test"].tolist(),
            "test_sourcewise_auroc_contributions": contributions.tolist(),
        }

    tulu_names = evaluation["statistics"]["tulu"]["order"]
    tulu_keys = [f"tulu/{name}" for name in tulu_names]
    tulu_contributions = np.column_stack(
        [cell_results[key]["test_sourcewise_auroc_contributions"] for key in tulu_keys]
    )
    tulu_aurocs = [cell_results[key]["test_auroc"] for key in tulu_keys]
    page = exact_page_test(tulu_contributions)
    alpha = float(evaluation["statistics"]["alpha"])
    tulu_ceiling = tulu_ceiling_audit(
        tulu_names, tulu_contributions, tulu_aurocs, alpha=alpha
    )
    if tulu_ceiling["status"] == "WITHHELD_CEILING":
        tulu_verdict = "WITHHELD_CEILING"
    elif tulu_ceiling["status"] == "ROBUST_TO_CEILING_ORDER":
        admissible = tulu_ceiling["admissible"]
        robust_support = all(
            item["direction"] == "increasing" and item["reject_at_alpha"] for item in admissible
        )
        tulu_verdict = "DIRECTIONAL_SUPPORTED" if robust_support else "DIRECTIONAL_NOT_SUPPORTED"
    elif page["direction"] == "increasing" and page["p_value"] < alpha:
        tulu_verdict = "DIRECTIONAL_SUPPORTED"
    else:
        tulu_verdict = "DIRECTIONAL_NOT_SUPPORTED"

    pythia = evaluation["statistics"]["pythia"]
    pythia_names = pythia["order"]
    pythia_keys = [f"pythia/{name}" for name in pythia_names]
    pythia_aurocs = [cell_results[key]["test_auroc"] for key in pythia_keys]
    log_parameters = [math.log10(int(value)) for value in pythia["parameters"]]
    spearman = exact_spearman_test(log_parameters, pythia_aurocs, alternative="less")
    pythia_ceiling = pythia_ceiling_audit(
        pythia_names, pythia["parameters"], pythia_aurocs, alpha=alpha
    )
    if pythia_ceiling["status"] == "WITHHELD_CEILING":
        pythia_verdict = "WITHHELD_CEILING"
    elif pythia_ceiling["status"] == "ROBUST_TO_CEILING_ORDER":
        admissible = pythia_ceiling["admissible"]
        robust_support = all(
            item["direction"] == "negative" and item["reject_at_alpha"] for item in admissible
        )
        pythia_verdict = "DIRECTIONAL_SUPPORTED" if robust_support else "DIRECTIONAL_NOT_SUPPORTED"
    elif spearman["rho"] is not None and spearman["rho"] < 0 and spearman["p_value"] < alpha:
        pythia_verdict = "DIRECTIONAL_SUPPORTED"
    else:
        pythia_verdict = "DIRECTIONAL_NOT_SUPPORTED"

    contrast_pairs = {
        "tulu/sft-base": ("tulu/sft", "tulu/base"),
        "tulu/dpo-sft": ("tulu/dpo", "tulu/sft"),
        "tulu/rlvr-dpo": ("tulu/rlvr", "tulu/dpo"),
        "tulu/rlvr-base": ("tulu/rlvr", "tulu/base"),
    }
    bootstrap_config = evaluation["statistics"]["bootstrap"]
    bootstrap = paired_source_bootstrap(
        test_score_pairs,
        replicates=int(bootstrap_config["replicates"]),
        seed=int(bootstrap_config["seed"]),
        contrasts=contrast_pairs,
        quantile_method=bootstrap_config["quantile_method"],
    )

    try:
        import sklearn
    except ImportError as error:  # pragma: no cover - fit_detector already gives a clearer error
        raise RuntimeError("Install the exact 'evaluation' extra before evaluation") from error
    result = {
        "schema_version": "aegis.clean-staircase.main-evaluation.v1",
        "run_id": evaluation["run_id"],
        "status": "COMPLETE",
        "global_gate_sha256": gate["gate_sha256"],
        "views_sha256": views_manifest["artifact"]["sha256"],
        "evaluation_config_sha256": sha256_file(ROOT / "configs" / "evaluation.v1.json"),
        "evaluation_lock": evaluation_lock,
        "test_source_ids": test_source_ids,
        "test_source_count": len(test_source_ids),
        "cells": cell_results,
        "tulu": {
            "order": tulu_names,
            "ordinary_exact_page": page,
            "ceiling_audit": tulu_ceiling,
            "verdict": tulu_verdict,
        },
        "pythia": {
            "order": pythia_names,
            "parameters": pythia["parameters"],
            "ordinary_exact_spearman": spearman,
            "ceiling_audit": pythia_ceiling,
            "verdict": pythia_verdict,
        },
        "paired_source_bootstrap": bootstrap,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "test_scores_released": True,
    }
    output = output.expanduser().resolve()
    atomic_write_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen CLEAN-FG-v5 evaluation only after the all-nine-cell gate"
    )
    parser.add_argument("result_roots", nargs="+", type=Path, help="Exactly nine main result roots")
    parser.add_argument("--views", type=Path, required=True, help="Gate-bound RoBERTa .npz")
    parser.add_argument("--output", type=Path, required=True, help="Evaluation JSON path")
    args = parser.parse_args()
    result = evaluate(args.result_roots, args.views, args.output)
    summary = {
        "status": result["status"],
        "run_id": result["run_id"],
        "global_gate_sha256": result["global_gate_sha256"],
        "tulu_verdict": result["tulu"]["verdict"],
        "pythia_verdict": result["pythia"]["verdict"],
        "output": str(args.output.expanduser().resolve()),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
