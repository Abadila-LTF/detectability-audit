#!/usr/bin/env python3
"""CLEAN-RW-v1 evaluator (docs/AMENDMENT-2026-09-28-REWRITE-EVALUATOR.md).

Builds the four rewrite views with the locked encoder, writes and then verifies a
rewrite evaluation lock over the amended code set, refits the CLEAN-FG-v5 Tulu
comparators from the locked FG views (asserting the released test AUROCs exactly), and
runs the preregistered Page test, contrasts, and rewrite-vs-free-generation dissociation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from aegis_staircases.contracts import atomic_write_json, canonical_hash, read_jsonl, sha256_file, sha256_text  # noqa: E402
from aegis_staircases.evaluation import (  # noqa: E402
    array_contract_hash,
    auroc_to_dprime,
    exact_page_test,
    fit_detector,
    paired_source_bootstrap,
    sourcewise_auc_contributions,
    tulu_ceiling_audit,
)
from aegis_staircases.evaluation_contract import load_evaluation_config  # noqa: E402
from aegis_staircases.views import add_model_boundaries, content_token_ids, tokenize_content_ids  # noqa: E402
from scripts.build_roberta_views import (  # noqa: E402
    _code_hashes,
    _encode_texts,
    _package_versions,
    _resolve_roberta_snapshot,
)
from scripts.validate_main_panel import validate_main_panel  # noqa: E402
from scripts.validate_rewrite_results import REWRITE_CELLS, validate_rewrite_panel  # noqa: E402

RW_EVALUATION_CONFIG = ROOT / "configs" / "evaluation.rw-v1.json"
FG_GATE_SHA256 = "546e7e2a78d6a3976e4903e00e30e9f298057d87f5985e4304764e126d3359cf"
FG_ARTIFACTS = {
    128: ("downloads/evaluation/roberta-views.npz", "docs/evidence/evaluation-2026-09-28"),
    64: ("downloads/evaluation/headroom-64/roberta-views.npz", "docs/evidence/headroom-2026-09-28/view-64"),
    32: ("downloads/evaluation/headroom-32/roberta-views.npz", "docs/evidence/headroom-2026-09-28/view-32"),
}
CONFIG_FILES = [
    "configs/experiment.v1.json",
    "configs/evaluation.v1.json",
    "configs/experiment.rw-v1.json",
    "configs/evaluation.rw-v1.json",
]
SOURCE_IDS = [f"arxiv2k:{index:04d}" for index in range(500)]


def _rw_roots(result_roots: Sequence[Path]) -> dict[str, Path]:
    roots = {}
    for unresolved in result_roots:
        root = unresolved.expanduser().resolve()
        roots[json.loads((root / "_contract.json").read_text(encoding="utf-8"))["cell"]] = root
    return roots


def descent_authorized(content_tokens: int, previous: Path | None) -> None:
    if content_tokens == 128:
        return
    if content_tokens not in (64, 32) or previous is None:
        raise RuntimeError("64/32-token rewrite views require --descent-from the previous rewrite evaluation")
    prior = json.loads(previous.read_text(encoding="utf-8"))
    if prior["content_tokens"] != {64: 128, 32: 64}[content_tokens]:
        raise RuntimeError("Descent must proceed 128 -> 64 -> 32")
    if not any(cell["test_auroc"] == 1.0 for cell in prior["rewrite"]["cells"].values()):
        raise RuntimeError("No rewrite cell was at ceiling; descent is not authorized")


def paired_view_gate(tokenizer: Any, humans: list[str], rw_texts: dict[str, list[str]], content_tokens: int) -> dict[str, Any]:
    failures, human_min, generated_min, pairs = [], None, None, 0
    for cell, texts in rw_texts.items():
        for source_id, human, text in zip(SOURCE_IDS, humans, texts):
            counts = []
            for value in (human, text):
                total = len(tokenize_content_ids(tokenizer, value))
                counts.append(total)
                if total < content_tokens:
                    failures.append({"endpoint": f"tulu/{cell}/{source_id}", "tokens": total})
                    continue
                _, mask = add_model_boundaries(tokenizer, content_token_ids(tokenizer, value, content_tokens))
                if sum(mask) != content_tokens:
                    failures.append({"endpoint": f"tulu/{cell}/{source_id}", "mask": sum(mask)})
            human_min = counts[0] if human_min is None else min(human_min, counts[0])
            generated_min = counts[1] if generated_min is None else min(generated_min, counts[1])
            pairs += 1
    return {"status": "ok" if not failures and pairs == 2000 else "failed", "content_tokens": content_tokens,
            "pairs": pairs, "human_tokens_min": human_min, "generated_tokens_min": generated_min, "failures": failures}


def _fg_views(content_tokens: int) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    npz, evidence = FG_ARTIFACTS[content_tokens]
    npz_path = ROOT / npz
    manifest = json.loads((ROOT / evidence / "roberta-views.manifest.json").read_text(encoding="utf-8"))
    released = json.loads((ROOT / evidence / "evaluation.json").read_text(encoding="utf-8"))
    if sha256_file(npz_path) != manifest["artifact"]["sha256"]:
        raise RuntimeError(f"FG {content_tokens}-token views differ from the released manifest")
    if released["views_sha256"] != manifest["artifact"]["sha256"]:
        raise RuntimeError("Released FG evaluation is bound to different views")
    return npz_path, manifest, released


def _lock_material(content_tokens: int, rw_gate: dict, fg_gate: dict, fg_npz: Path, views_path: Path,
                   arrays: dict, roberta: dict, evaluation: dict) -> dict[str, Any]:
    splits = arrays["split_origins"].tolist()
    return {
        "schema_version": "aegis.clean-rewrite.evaluation-lock.v1",
        "run_id": "clean-rw-v1",
        "content_tokens": content_tokens,
        "rewrite_gate": rw_gate["gate_material"],
        "rewrite_gate_sha256": rw_gate["gate_sha256"],
        "fg_gate_sha256": fg_gate["gate_sha256"],
        "fg_views": {"path": str(fg_npz.relative_to(ROOT)), "sha256": sha256_file(fg_npz)},
        "configs": [{"path": p, "sha256": sha256_file(ROOT / p)} for p in CONFIG_FILES],
        "code": _code_hashes(evaluation["evaluation_lock"]["code_files"]),
        "packages": {"python": platform.python_version(),
                     "distributions": _package_versions(evaluation["evaluation_lock"]["package_distributions"])},
        "roberta": roberta,
        "splits": {s: {"source_ids": [i for i, r in zip(SOURCE_IDS, splits) if r == s]} for s in ("train", "val", "test")},
        "embedding_cache": {"path": views_path.name, "sha256": sha256_file(views_path),
                            "arrays": {name: array_contract_hash(array) for name, array in arrays.items()}},
    }


def build(rw_roots: Sequence[Path], fg_roots: Sequence[Path], content_tokens: int, output_dir: Path) -> Path:
    import torch
    from transformers import AutoModel, AutoTokenizer

    evaluation = load_evaluation_config(RW_EVALUATION_CONFIG)
    rw_gate = validate_rewrite_panel(rw_roots)
    fg_gate = validate_main_panel(fg_roots)
    if rw_gate["status"] != "GLOBAL_REWRITE_GATE_PASSED" or fg_gate["gate_sha256"] != FG_GATE_SHA256:
        raise RuntimeError("Rewrite or free-generation global gate failed")
    representation = evaluation["representation"]
    torch.manual_seed(int(evaluation["classifier"]["random_seed"]))
    torch.use_deterministic_algorithms(True)
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    snapshot, files, weight_file = _resolve_roberta_snapshot(
        representation["model_hf_id"], representation["model_revision"], evaluation["evaluation_lock"],
        (ROOT / ".cache" / "huggingface").resolve())
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), use_fast=True, local_files_only=True, trust_remote_code=False)
    model = AutoModel.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False,
                                      use_safetensors=weight_file.endswith(".safetensors"))
    model.eval()
    model.to("cpu")

    human_rows = {row["source_id"]: row for row in read_jsonl(ROOT / "data" / "manifests" / "human_sources.jsonl")}
    humans = [human_rows[s]["human_text"] for s in SOURCE_IDS]
    split_origins = [human_rows[s]["split_origin"] for s in SOURCE_IDS]
    roots = _rw_roots(rw_roots)
    rw_texts = {}
    for cell in REWRITE_CELLS:
        texts = []
        for source_id in SOURCE_IDS:
            record = json.loads((roots[cell] / source_id.replace(":", "_") / "attempt-00.json").read_text(encoding="utf-8"))
            if sha256_text(record["normalized_output"]) != record["normalized_output_sha256"]:
                raise RuntimeError(f"Rewrite text hash mismatch: {cell}/{source_id}")
            texts.append(record["normalized_output"])
        rw_texts[cell] = texts
    output_dir = output_dir.expanduser().resolve()
    view_gate = paired_view_gate(tokenizer, humans, rw_texts, content_tokens)
    atomic_write_json(output_dir / "paired-view-gate.json", view_gate)
    if view_gate["status"] != "ok":
        raise RuntimeError(f"Rewrite paired-view gate failed at {content_tokens} tokens")

    batch = int(representation["batch_size"])
    human_embeddings = _encode_texts(humans, tokenizer, model, torch, content_tokens=content_tokens, batch_size=batch)
    generated = np.stack([_encode_texts(rw_texts[c], tokenizer, model, torch, content_tokens=content_tokens,
                                        batch_size=batch) for c in REWRITE_CELLS], axis=0)
    fg_npz, _, _ = _fg_views(content_tokens)
    fg_arrays = np.load(fg_npz, allow_pickle=False)
    if not np.array_equal(fg_arrays["human_embeddings"], human_embeddings.astype(np.float32)):
        raise RuntimeError("Rewrite human view does not reproduce the locked FG human view bit-for-bit")
    arrays = {
        "human_embeddings": human_embeddings.astype(np.float32, copy=False),
        "generated_embeddings": generated.astype(np.float32, copy=False),
        "source_ids": np.asarray(SOURCE_IDS),
        "split_origins": np.asarray(split_origins),
        "cells": np.asarray(list(REWRITE_CELLS)),
    }
    views_path = output_dir / "rewrite-views.npz"
    temporary = views_path.with_name(views_path.name + f".tmp-{os.getpid()}.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, views_path)
    roberta = {"hf_id": representation["model_hf_id"], "revision": representation["model_revision"],
               "weight_file": weight_file, "files": files}
    material = _lock_material(content_tokens, rw_gate, fg_gate, fg_npz, views_path, arrays, roberta, evaluation)
    atomic_write_json(output_dir / "rewrite-views.lock.json", {**material, "lock_sha256": canonical_hash(material)})
    return views_path


def verify_lock(rw_roots: Sequence[Path], fg_roots: Sequence[Path], content_tokens: int, views_path: Path) -> tuple[dict, dict]:
    evaluation = load_evaluation_config(RW_EVALUATION_CONFIG)
    lock = json.loads(views_path.with_name("rewrite-views.lock.json").read_text(encoding="utf-8"))
    material = {k: v for k, v in lock.items() if k != "lock_sha256"}
    if canonical_hash(material) != lock["lock_sha256"]:
        raise RuntimeError("Rewrite lock canonical hash mismatch")
    rw_gate = validate_rewrite_panel(rw_roots)
    fg_gate = validate_main_panel(fg_roots)
    arrays = {name: value for name, value in np.load(views_path, allow_pickle=False).items()}
    fg_npz, _, _ = _fg_views(content_tokens)
    expected = _lock_material(content_tokens, rw_gate, fg_gate, fg_npz, views_path, arrays, lock["roberta"], evaluation)
    for key in expected:
        if expected[key] != lock[key]:
            raise RuntimeError(f"Rewrite evaluation lock check failed on '{key}'")
    return lock, arrays


def _bootstrap_samples(scores: dict[str, tuple], replicates: int, seed: int, chunk_size: int = 256) -> dict[str, np.ndarray]:
    """Exact mirror of the locked paired_source_bootstrap resampling, returning the samples."""
    names = list(scores)
    kernels = []
    for name in names:
        human = np.asarray(scores[name][0], dtype=float)
        generated = np.asarray(scores[name][1], dtype=float)
        kernel = (generated[:, None] > human[None, :]).astype(float)
        kernel += 0.5 * (generated[:, None] == human[None, :])
        kernels.append(kernel)
    n = kernels[0].shape[0]
    rng = np.random.default_rng(seed)
    samples = {name: np.empty(replicates, dtype=float) for name in names}
    probabilities = np.full(n, 1.0 / n)
    for start in range(0, replicates, chunk_size):
        stop = min(start + chunk_size, replicates)
        counts = rng.multinomial(n, probabilities, size=stop - start).astype(float)
        for name, kernel in zip(names, kernels):
            samples[name][start:stop] = np.sum((counts @ kernel) * counts, axis=1) / (n**2)
    return samples


def _decision(estimate: float, ci: list[float], aurocs: Sequence[float]) -> str:
    ceilings = [value == 1.0 for value in aurocs]
    pairs = [ceilings[i:i + 2] for i in range(0, len(ceilings), 2)]
    if any(all(pair) for pair in pairs):
        return "WITHHELD_CEILING"
    if any(ceilings):
        return "CEILING_CENSORED"
    if ci[0] > 0:
        return "DISSOCIATION_RW_MORE_DETECTABLE"
    if ci[1] < 0:
        return "DISSOCIATION_RW_LESS_DETECTABLE"
    return "NO_DISSOCIATION"


def score(rw_roots: Sequence[Path], fg_roots: Sequence[Path], content_tokens: int, views_path: Path, output: Path) -> dict:
    evaluation = load_evaluation_config(RW_EVALUATION_CONFIG)
    lock, arrays = verify_lock(rw_roots, fg_roots, content_tokens, views_path)
    classifier = evaluation["classifier"]
    splits = arrays["split_origins"].tolist()
    fg_npz, _, released = _fg_views(content_tokens)
    fg_arrays = np.load(fg_npz, allow_pickle=False)
    fg_cells = fg_arrays["cells"].tolist()
    stages = list(REWRITE_CELLS)
    fits, pairs, cells = {}, {}, {"rewrite": {}, "free_generation": {}}
    for family, human, generated_by_stage in (
        ("rewrite", arrays["human_embeddings"], {s: arrays["generated_embeddings"][i] for i, s in enumerate(stages)}),
        ("free_generation", fg_arrays["human_embeddings"],
         {s: fg_arrays["generated_embeddings"][fg_cells.index(s)] for s in stages}),
    ):
        for stage in stages:
            fit = fit_detector(human, generated_by_stage[stage], splits, c_grid=classifier["c_grid"],
                               max_iter=int(classifier["max_iter"]), tolerance=float(classifier["tolerance"]),
                               random_seed=int(classifier["random_seed"]))
            key = f"{'rw' if family == 'rewrite' else 'fg'}/{stage}"
            fits[key] = fit
            pairs[key] = (fit.human_scores["test"], fit.generated_scores["test"])
            cells[family][stage] = {
                "selected_c": fit.selected_c, "validation_auroc": fit.validation_auroc,
                "validation_grid": fit.validation_grid, "test_auroc": fit.test_auroc,
                "test_dprime": auroc_to_dprime(fit.test_auroc),
                "test_human_scores": fit.human_scores["test"].tolist(),
                "test_generated_scores": fit.generated_scores["test"].tolist(),
                "test_sourcewise_auroc_contributions": sourcewise_auc_contributions(
                    fit.human_scores["test"], fit.generated_scores["test"]).tolist(),
            }
    for stage in stages:
        if cells["free_generation"][stage]["test_auroc"] != released["cells"][f"tulu/{stage}"]["test_auroc"]:
            raise RuntimeError(f"Refit FG comparator for {stage} does not reproduce the released test AUROC")

    alpha = float(evaluation["statistics"]["alpha"])
    contributions = np.column_stack([cells["rewrite"][s]["test_sourcewise_auroc_contributions"] for s in stages])
    aurocs = [cells["rewrite"][s]["test_auroc"] for s in stages]
    page = exact_page_test(contributions)
    ceiling = tulu_ceiling_audit(stages, contributions, aurocs, alpha=alpha)
    if ceiling["status"] == "WITHHELD_CEILING":
        verdict = "WITHHELD_CEILING"
    elif ceiling["status"] == "ROBUST_TO_CEILING_ORDER":
        verdict = ("DIRECTIONAL_SUPPORTED" if all(i["direction"] == "increasing" and i["reject_at_alpha"]
                                                   for i in ceiling["admissible"]) else "DIRECTIONAL_NOT_SUPPORTED")
    elif page["direction"] == "increasing" and page["p_value"] < alpha:
        verdict = "DIRECTIONAL_SUPPORTED"
    else:
        verdict = "DIRECTIONAL_NOT_SUPPORTED"

    boot_cfg = evaluation["statistics"]["bootstrap"]
    contrasts = {
        "rw/sft-base": ("rw/sft", "rw/base"), "rw/dpo-sft": ("rw/dpo", "rw/sft"),
        "rw/rlvr-dpo": ("rw/rlvr", "rw/dpo"), "rw/rlvr-base": ("rw/rlvr", "rw/base"),
        **{f"dissociation/{s}": (f"rw/{s}", f"fg/{s}") for s in stages},
    }
    bootstrap = paired_source_bootstrap(pairs, replicates=int(boot_cfg["replicates"]), seed=int(boot_cfg["seed"]),
                                        contrasts=contrasts, quantile_method=boot_cfg["quantile_method"])
    samples = _bootstrap_samples(pairs, int(boot_cfg["replicates"]), int(boot_cfg["seed"]))

    def interval(values: np.ndarray) -> list[float]:
        return [float(v) for v in np.quantile(values, [0.025, 0.975], method=boot_cfg["quantile_method"])]

    for name in pairs:
        if interval(samples[name]) != bootstrap["cells"][name]["auroc_ci95"]:
            raise RuntimeError(f"Mirrored bootstrap does not reproduce the locked bootstrap for {name}")
    for label, (hi, lo) in contrasts.items():
        if interval(samples[hi] - samples[lo]) != bootstrap["contrasts"][label]["auroc_difference_ci95"]:
            raise RuntimeError(f"Mirrored bootstrap does not reproduce the locked contrast {label}")

    dissociation = {}
    for stage in stages:
        c = bootstrap["contrasts"][f"dissociation/{stage}"]
        pair_aurocs = [cells["rewrite"][stage]["test_auroc"], cells["free_generation"][stage]["test_auroc"]]
        dissociation[stage] = {"delta": c["auroc_difference"], "delta_ci95": c["auroc_difference_ci95"],
                               "rw_auroc": pair_aurocs[0], "fg_auroc": pair_aurocs[1],
                               "decision": _decision(c["auroc_difference"], c["auroc_difference_ci95"], pair_aurocs)}
    did_samples = (samples["rw/rlvr"] - samples["fg/rlvr"]) - (samples["rw/base"] - samples["fg/base"])
    did = dissociation["rlvr"]["delta"] - dissociation["base"]["delta"]
    did_ci = interval(did_samples)
    did_aurocs = [dissociation["rlvr"]["rw_auroc"], dissociation["rlvr"]["fg_auroc"],
                  dissociation["base"]["rw_auroc"], dissociation["base"]["fg_auroc"]]
    result = {
        "schema_version": "aegis.clean-rewrite.evaluation.v1",
        "run_id": "clean-rw-v1",
        "status": "COMPLETE",
        "content_tokens": content_tokens,
        "evaluation_lock_sha256": lock["lock_sha256"],
        "rewrite_gate_sha256": lock["rewrite_gate_sha256"],
        "fg_gate_sha256": lock["fg_gate_sha256"],
        "fg_views_sha256": lock["fg_views"]["sha256"],
        "rewrite_views_sha256": lock["embedding_cache"]["sha256"],
        "test_source_count": int(np.sum(arrays["split_origins"] == "test")),
        "rewrite": {"order": stages, "cells": cells["rewrite"], "ordinary_exact_page": page,
                    "ceiling_audit": ceiling, "verdict": verdict},
        "free_generation_comparator": {"cells": cells["free_generation"],
                                       "released_test_aurocs_reproduced": True},
        "paired_source_bootstrap": bootstrap,
        "dissociation": {"per_stage": dissociation,
                         "difference_in_differences_rlvr_minus_base": {
                             "estimate": did, "ci95": did_ci,
                             "decision": _decision(did, did_ci, did_aurocs)}},
        "runtime": {"python": platform.python_version()},
    }
    atomic_write_json(output.expanduser().resolve(), result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rw-roots", nargs=4, type=Path, required=True)
    parser.add_argument("--fg-roots", nargs=9, type=Path, required=True)
    parser.add_argument("--content-tokens", type=int, choices=[128, 64, 32], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--descent-from", type=Path)
    args = parser.parse_args()
    descent_authorized(args.content_tokens, args.descent_from)
    views = build(args.rw_roots, args.fg_roots, args.content_tokens, args.output_dir)
    result = score(args.rw_roots, args.fg_roots, args.content_tokens, views, args.output_dir / "evaluation.json")
    print(json.dumps({"status": result["status"], "content_tokens": args.content_tokens,
                      "rewrite_verdict": result["rewrite"]["verdict"],
                      "ceiling_cells": [s for s, c in result["rewrite"]["cells"].items() if c["test_auroc"] == 1.0],
                      "any_rewrite_ceiling": any(c["test_auroc"] == 1.0 for c in result["rewrite"]["cells"].values())},
                     indent=2))


if __name__ == "__main__":
    main()
