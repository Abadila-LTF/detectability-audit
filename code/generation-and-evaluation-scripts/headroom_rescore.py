#!/usr/bin/env python3
"""Headroom rescoring driver (docs/AMENDMENT-2026-09-28-HEADROOM-RESCORE.md).

Runs the locked global gate, view builder, and evaluator unchanged, injecting a derived
evaluation config whose only difference from configs/evaluation.v1.json is
representation.content_tokens (64, or 32 when the descent rule triggers).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import scripts.build_roberta_views as build_module  # noqa: E402
import scripts.evaluate_main as evaluate_module  # noqa: E402
from aegis_staircases.contracts import atomic_write_json, canonical_hash, read_jsonl, sha256_file  # noqa: E402
from aegis_staircases.evaluation_contract import load_evaluation_config  # noqa: E402
from aegis_staircases.headroom import (  # noqa: E402
    FLOOR_VIEW,
    derive_headroom_config,
    needs_descent,
    select_primary_view,
)
from aegis_staircases.views import add_model_boundaries, content_token_ids, tokenize_content_ids  # noqa: E402
from scripts.validate_main_panel import validate_main_panel  # noqa: E402

PARENT_CONFIG = ROOT / "configs" / "evaluation.v1.json"
ROBERTA_SNAPSHOT = (
    ROOT / ".cache" / "huggingface" / "models--FacebookAI--roberta-base" / "snapshots"
    / "e2da8e2f811d1448a5b465c236feacd80ffbac7b"
)


def paired_view_gate(result_roots: Sequence[Path], content_tokens: int) -> dict[str, Any]:
    """All 4,500 pairs must yield exactly N content IDs recovered through the Amendment A mask."""
    from transformers import AutoTokenizer

    gate = validate_main_panel(result_roots)
    if gate["status"] != "GLOBAL_MAIN_GATE_PASSED":
        raise RuntimeError("Global main gate did not pass")
    tokenizer = AutoTokenizer.from_pretrained(str(ROBERTA_SNAPSHOT), use_fast=True, local_files_only=True)
    humans = {row["source_id"]: row["human_text"] for row in read_jsonl(ROOT / "data" / "manifests" / "human_sources.jsonl")}
    failures = []
    human_min = generated_min = None
    pairs = 0
    for unresolved in result_roots:
        root = unresolved.expanduser().resolve()
        contract = json.loads((root / "_contract.json").read_text(encoding="utf-8"))
        for source_id in contract["source_ids"]:
            record = json.loads((root / source_id.replace(":", "_") / "attempt-00.json").read_text(encoding="utf-8"))
            counts = []
            for text in (humans[source_id], record["normalized_output"]):
                total = len(tokenize_content_ids(tokenizer, text))
                counts.append(total)
                if total < content_tokens:
                    failures.append({"endpoint": f"{contract['suite']}/{contract['cell']}/{source_id}", "tokens": total})
                    continue
                content = content_token_ids(tokenizer, text, content_tokens)
                _, mask = add_model_boundaries(tokenizer, content)
                if sum(mask) != content_tokens:
                    failures.append({"endpoint": f"{contract['suite']}/{contract['cell']}/{source_id}", "mask": sum(mask)})
            human_min = counts[0] if human_min is None else min(human_min, counts[0])
            generated_min = counts[1] if generated_min is None else min(generated_min, counts[1])
            pairs += 1
    return {
        "status": "ok" if not failures and pairs == 4500 else "failed",
        "content_tokens": content_tokens,
        "pairs": pairs,
        "human_tokens_min": human_min,
        "generated_tokens_min": generated_min,
        "failures": failures,
        "global_gate_sha256": gate["gate_sha256"],
    }


def rescore(result_roots: Sequence[Path], content_tokens: int, output_dir: Path, parent_evaluation: Path | None) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    if content_tokens == FLOOR_VIEW:
        if parent_evaluation is None:
            raise RuntimeError("The 32-token view requires --descent-from <64-token evaluation.json>")
        evaluation_64 = json.loads(parent_evaluation.read_text(encoding="utf-8"))
        if not needs_descent(evaluation_64):
            raise RuntimeError("Descent rule did not trigger at 64 tokens; 32 tokens is not authorized")
    parent = load_evaluation_config(PARENT_CONFIG)
    derived = derive_headroom_config(parent, content_tokens)

    view_gate = paired_view_gate(result_roots, content_tokens)
    atomic_write_json(output_dir / "paired-view-gate.json", view_gate)
    if view_gate["status"] != "ok":
        raise RuntimeError(f"Paired-view gate failed at {content_tokens} tokens")

    # Inject the derived config into the locked modules; their code runs unchanged.
    build_module.load_evaluation_config = lambda *args, **kwargs: derived
    evaluate_module.load_evaluation_config = lambda *args, **kwargs: derived
    views_path = output_dir / "roberta-views.npz"
    manifest = build_module.build_views(result_roots, views_path)
    evaluation_path = output_dir / "evaluation.json"
    result = evaluate_module.evaluate(result_roots, views_path, evaluation_path)
    envelope = {
        "schema_version": "aegis.clean-staircase.headroom-rescore.v1",
        "amendment": "docs/AMENDMENT-2026-09-28-HEADROOM-RESCORE.md",
        "content_tokens": content_tokens,
        "parent_config_path": "configs/evaluation.v1.json",
        "parent_config_sha256": sha256_file(PARENT_CONFIG),
        "derived_config_canonical_sha256": canonical_hash(derived),
        "derived_config": derived,
        "paired_view_gate": view_gate,
        "global_gate_sha256": result["global_gate_sha256"],
        "views_sha256": manifest["artifact"]["sha256"],
        "evaluation_lock_sha256": manifest["evaluation_lock"]["lock_sha256"],
        "evaluation_json_sha256": sha256_file(evaluation_path),
        "ceiling_cells": sorted(k for k, v in result["cells"].items() if v["test_auroc"] == 1.0),
        "tulu_verdict": result["tulu"]["verdict"],
        "pythia_verdict": result["pythia"]["verdict"],
    }
    if content_tokens == 64:
        envelope["descent_to_32_triggered"] = needs_descent(result)
    atomic_write_json(output_dir / "headroom.json", envelope)
    return envelope


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("rescore")
    run.add_argument("result_roots", nargs="+", type=Path)
    run.add_argument("--content-tokens", type=int, required=True, choices=[64, 32])
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--descent-from", type=Path)
    select = sub.add_parser("select")
    select.add_argument("--view", action="append", required=True, help="N=path/to/evaluation.json")
    args = parser.parse_args()
    if args.command == "rescore":
        print(json.dumps(rescore(args.result_roots, args.content_tokens, args.output_dir, args.descent_from), indent=2))
    else:
        evaluations = {}
        for item in args.view:
            view, path = item.split("=", 1)
            evaluations[int(view)] = json.loads(Path(path).read_text(encoding="utf-8"))
        print(json.dumps({"views": sorted(evaluations), "primary_headroom_view": select_primary_view(evaluations)}))


if __name__ == "__main__":
    main()
