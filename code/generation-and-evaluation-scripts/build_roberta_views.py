#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from huggingface_hub import list_repo_files, snapshot_download


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
from aegis_staircases.evaluation import array_contract_hash  # noqa: E402
from aegis_staircases.evaluation_contract import load_evaluation_config  # noqa: E402
from aegis_staircases.views import (  # noqa: E402
    add_model_boundaries,
    content_token_ids,
)
from scripts.validate_main_panel import validate_main_panel  # noqa: E402


def _package_versions(distributions: Sequence[str]) -> dict[str, str]:
    versions = {}
    for distribution in distributions:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeError(f"Evaluation-lock package is missing: {distribution}") from error
    return versions


def _code_hashes(paths: Sequence[str]) -> list[dict[str, str]]:
    records = []
    for relative in paths:
        path = ROOT / relative
        if not path.is_file():
            raise RuntimeError(f"Evaluation-lock code file is missing: {relative}")
        records.append({"path": relative, "sha256": sha256_file(path)})
    return records


def _resolve_roberta_snapshot(
    model_id: str, revision: str, lock_config: dict[str, Any], cache_dir: Path
) -> tuple[Path, list[dict[str, Any]], str]:
    available = set(list_repo_files(repo_id=model_id, revision=revision))
    weight_file = next(
        (name for name in lock_config["weight_preference"] if name in available), None
    )
    if weight_file is None:
        raise RuntimeError("Pinned RoBERTa revision has no supported unsharded weight artifact")
    selected = [name for name in lock_config["roberta_snapshot_files"] if name in available]
    if "config.json" not in selected:
        raise RuntimeError("Pinned RoBERTa config.json is missing")
    if "tokenizer.json" not in selected and not {"vocab.json", "merges.txt"}.issubset(selected):
        raise RuntimeError("Pinned RoBERTa tokenizer artifacts are incomplete")
    selected.append(weight_file)
    snapshot = Path(
        snapshot_download(
            repo_id=model_id,
            revision=revision,
            cache_dir=str(cache_dir),
            allow_patterns=selected,
        )
    ).resolve()
    if snapshot.name != revision:
        raise RuntimeError(f"Resolved snapshot commit changed: {snapshot.name} != {revision}")
    files = []
    for relative in sorted(selected):
        path = snapshot / relative
        if not path.is_file():
            raise RuntimeError(f"Selected RoBERTa artifact was not downloaded: {relative}")
        files.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return snapshot, files, weight_file


def _cell_roots(result_roots: Sequence[Path]) -> dict[tuple[str, str], Path]:
    roots: dict[tuple[str, str], Path] = {}
    for unresolved in result_roots:
        root = unresolved.expanduser().resolve()
        contract = json.loads((root / "_contract.json").read_text(encoding="utf-8"))
        key = (contract["suite"], contract["cell"])
        if key in roots:
            raise RuntimeError(f"Duplicate cell root: {key}")
        roots[key] = root
    return roots


def _encode_texts(
    texts: Sequence[str], tokenizer: Any, model: Any, torch: Any, *, content_tokens: int, batch_size: int
) -> np.ndarray:
    batches = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        model_rows = []
        content_masks = []
        for text in batch_texts:
            content = content_token_ids(tokenizer, text, content_tokens)
            model_ids, content_mask = add_model_boundaries(tokenizer, content)
            if sum(content_mask) != content_tokens:
                raise RuntimeError("Boundary construction changed the frozen content-token count")
            model_rows.append(model_ids)
            content_masks.append(content_mask)
        lengths = {len(row) for row in model_rows}
        if len(lengths) != 1:
            raise RuntimeError("Pinned RoBERTa boundary construction produced ragged fixed views")
        input_ids = torch.tensor(model_rows, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        pooling_mask = torch.tensor(content_masks, dtype=torch.bool)
        with torch.inference_mode():
            hidden = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            pooled = (hidden * pooling_mask.unsqueeze(-1)).sum(dim=1) / float(content_tokens)
        batches.append(pooled.detach().cpu().to(dtype=torch.float32).numpy())
    return np.concatenate(batches, axis=0)


def build_views(
    result_roots: Sequence[Path], output: Path, *, cache_dir: Path | None = None
) -> dict[str, Any]:
    # This call is deliberately first: incomplete panels cannot download RoBERTa,
    # build real representations, or expose any downstream score.
    gate = validate_main_panel(result_roots)
    if gate["status"] != "GLOBAL_MAIN_GATE_PASSED" or not gate["scoring_unlocked"]:
        raise RuntimeError("Global main gate did not pass")

    try:
        import torch
        import transformers
        from transformers import AutoModel, AutoTokenizer
    except ImportError as error:  # pragma: no cover - depends on optional environment
        raise RuntimeError("Install the exact 'evaluation' extra before building representations") from error

    evaluation = load_evaluation_config()
    representation = evaluation["representation"]
    lock_config = evaluation["evaluation_lock"]
    if representation["device"] != "cpu":
        raise RuntimeError("Evaluation v1 freezes representation extraction on CPU")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    torch.manual_seed(int(evaluation["classifier"]["random_seed"]))
    torch.use_deterministic_algorithms(True)
    model_id = representation["model_hf_id"]
    revision = representation["model_revision"]
    resolved_cache = (
        cache_dir.expanduser().resolve()
        if cache_dir is not None
        else (ROOT / ".cache" / "huggingface").resolve()
    )
    snapshot, snapshot_files, weight_file = _resolve_roberta_snapshot(
        model_id, revision, lock_config, resolved_cache
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot), use_fast=True, local_files_only=True, trust_remote_code=False
    )
    model = AutoModel.from_pretrained(
        str(snapshot),
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=weight_file.endswith(".safetensors"),
    )
    model.eval()
    model.to("cpu")
    model_commit = getattr(model.config, "_commit_hash", None)
    tokenizer_commit = tokenizer.init_kwargs.get("_commit_hash")

    human_rows = {
        row["source_id"]: row
        for row in read_jsonl(ROOT / "data" / "manifests" / "human_sources.jsonl")
    }
    source_ids = [f"arxiv2k:{index:04d}" for index in range(500)]
    humans = [human_rows[source_id]["human_text"] for source_id in source_ids]
    for source_id, text in zip(source_ids, humans):
        if sha256_text(text) != human_rows[source_id]["human_sha256"]:
            raise RuntimeError(f"Human text hash mismatch before representation: {source_id}")
    split_origins = [human_rows[source_id]["split_origin"] for source_id in source_ids]

    roots = _cell_roots(result_roots)
    cell_specs = evaluation["global_gate"]["cells"]
    generated_texts: list[list[str]] = []
    generated_hash_contract = []
    for spec in cell_specs:
        root = roots[(spec["suite"], spec["cell"])]
        texts = []
        hashes = []
        for source_id in source_ids:
            record = json.loads(
                (root / source_id.replace(":", "_") / "attempt-00.json").read_text(encoding="utf-8")
            )
            text = record["normalized_output"]
            if sha256_text(text) != record["normalized_output_sha256"]:
                raise RuntimeError(f"Generated text hash mismatch before representation: {spec}/{source_id}")
            texts.append(text)
            hashes.append(record["normalized_output_sha256"])
        generated_texts.append(texts)
        generated_hash_contract.append(
            {"suite": spec["suite"], "cell": spec["cell"], "output_hashes": hashes}
        )
    current_inputs = {
        "human_text_hashes": [human_rows[source_id]["human_sha256"] for source_id in source_ids],
        "generated_outputs": generated_hash_contract,
    }
    if gate["gate_material"]["inputs"] != current_inputs:
        raise RuntimeError("Representation inputs changed after the global gate")

    content_tokens = int(representation["content_tokens"])
    batch_size = int(representation["batch_size"])
    human_embeddings = _encode_texts(
        humans, tokenizer, model, torch, content_tokens=content_tokens, batch_size=batch_size
    )
    generated_embeddings = np.stack(
        [
            _encode_texts(
                texts, tokenizer, model, torch, content_tokens=content_tokens, batch_size=batch_size
            )
            for texts in generated_texts
        ],
        axis=0,
    )
    if human_embeddings.shape != (500, 768) or generated_embeddings.shape != (9, 500, 768):
        raise RuntimeError(
            f"Unexpected frozen representation shapes: {human_embeddings.shape}, {generated_embeddings.shape}"
        )

    output = output.expanduser().resolve()
    if output.suffix != ".npz":
        raise ValueError("Representation output must end in .npz")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".tmp-{os.getpid()}.npz")
    arrays = {
        "human_embeddings": human_embeddings.astype(np.float32, copy=False),
        "generated_embeddings": generated_embeddings.astype(np.float32, copy=False),
        "source_ids": np.asarray(source_ids),
        "split_origins": np.asarray(split_origins),
        "suites": np.asarray([spec["suite"] for spec in cell_specs]),
        "cells": np.asarray([spec["cell"] for spec in cell_specs]),
    }
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, output)
    split_ids = {
        split: [source_id for source_id, role in zip(source_ids, split_origins) if role == split]
        for split in ("train", "val", "test")
    }
    package_versions = _package_versions(lock_config["package_distributions"])
    lock_material = {
        "schema_version": "aegis.clean-staircase.evaluation-lock.v1",
        "run_id": evaluation["run_id"],
        "global_gate": gate["gate_material"],
        "global_gate_sha256": gate["gate_sha256"],
        "configs": [
            {"path": "configs/experiment.v1.json", "sha256": sha256_file(ROOT / "configs" / "experiment.v1.json")},
            {"path": "configs/evaluation.v1.json", "sha256": sha256_file(ROOT / "configs" / "evaluation.v1.json")},
        ],
        "code": _code_hashes(lock_config["code_files"]),
        "packages": {"python": platform.python_version(), "distributions": package_versions},
        "roberta": {
            "hf_id": model_id,
            "revision": revision,
            "weight_file": weight_file,
            "files": snapshot_files,
        },
        "splits": {
            split: {
                "count": len(ids),
                "source_ids": ids,
                "source_ids_sha256": canonical_hash(ids),
            }
            for split, ids in split_ids.items()
        },
        "inputs": {
            "source_ids_sha256": canonical_hash(source_ids),
            "human_text_hashes": current_inputs["human_text_hashes"],
            "human_text_hashes_sha256": canonical_hash(current_inputs["human_text_hashes"]),
            "generated_outputs": current_inputs["generated_outputs"],
            "generated_text_contract_sha256": canonical_hash(generated_hash_contract),
        },
        "embedding_cache": {
            "path": output.name,
            "sha256": sha256_file(output),
            "arrays": {name: array_contract_hash(array) for name, array in arrays.items()},
        },
    }
    evaluation_lock = {**lock_material, "lock_sha256": canonical_hash(lock_material)}
    lock_path = output.with_suffix(lock_config["artifact_suffix"])
    atomic_write_json(lock_path, evaluation_lock)
    manifest = {
        "schema_version": "aegis.clean-staircase.roberta-views.v1",
        "run_id": evaluation["run_id"],
        "global_gate_sha256": gate["gate_sha256"],
        "evaluation_config_sha256": sha256_file(ROOT / "configs" / "evaluation.v1.json"),
        "representation": representation,
        "model_resolved_commit": model_commit,
        "tokenizer_resolved_commit": tokenizer_commit,
        "content_tokenization": "add_special_tokens=False; first 128 IDs; boundaries added afterward",
        "pooling": "last-hidden mean over the 128 content positions only",
        "source_ids_sha256": canonical_hash(source_ids),
        "human_text_hashes_sha256": canonical_hash(
            [human_rows[source_id]["human_sha256"] for source_id in source_ids]
        ),
        "generated_text_contract_sha256": canonical_hash(generated_hash_contract),
        "arrays": {
            "human_embeddings": list(human_embeddings.shape),
            "generated_embeddings": list(generated_embeddings.shape),
            "source_ids": [500],
            "split_origins": [500],
            "suites": [9],
            "cells": [9],
        },
        "artifact": {"path": output.name, "sha256": sha256_file(output)},
        "evaluation_lock": {
            "path": lock_path.name,
            "sha256": sha256_file(lock_path),
            "lock_sha256": evaluation_lock["lock_sha256"],
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "distributions": package_versions,
            "device": "cpu",
        },
        "contains_scores": False,
    }
    atomic_write_json(output.with_suffix(".manifest.json"), manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the exact frozen RoBERTa views after the all-nine-cell main gate"
    )
    parser.add_argument("result_roots", nargs="+", type=Path, help="Exactly nine main result roots")
    parser.add_argument("--output", type=Path, required=True, help="Output .npz path")
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    manifest = build_views(args.result_roots, args.output, cache_dir=args.cache_dir)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
