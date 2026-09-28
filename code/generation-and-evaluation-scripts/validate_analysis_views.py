#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from huggingface_hub import snapshot_download  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

from aegis_staircases.contracts import read_jsonl, sha256_file  # noqa: E402
from aegis_staircases.views import content_token_ids, tokenize_content_ids  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the local fixed RoBERTa analysis view")
    parser.add_argument("result_roots", nargs="+", type=Path)
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    config = json.loads((ROOT / "configs" / "experiment.v1.json").read_text(encoding="utf-8"))
    humans = {
        row["source_id"]: row
        for row in read_jsonl(ROOT / "data" / "manifests" / "human_sources.jsonl")
    }
    view = config["decoding"]["analysis_view"]
    tokenizer_dir = Path(
        snapshot_download(
            repo_id=view["tokenizer_hf_id"],
            revision=view["tokenizer_revision"],
            cache_dir=str(ROOT / ".cache" / "huggingface"),
            allow_patterns=[
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "vocab.json",
                "merges.txt",
            ],
        )
    )
    tokenizer_path = tokenizer_dir / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    required = int(view["tokens"])

    counts = []
    failures = []
    seen_endpoints = set()
    for unresolved_root in args.result_roots:
        root = unresolved_root.expanduser().resolve()
        contract = json.loads((root / "_contract.json").read_text(encoding="utf-8"))
        for source_id in contract["source_ids"]:
            endpoint_id = f"{contract['suite']}/{contract['cell']}/{source_id}"
            if endpoint_id in seen_endpoints:
                raise RuntimeError(f"Duplicate endpoint: {endpoint_id}")
            seen_endpoints.add(endpoint_id)
            record_path = root / source_id.replace(":", "_") / "attempt-00.json"
            record = json.loads(record_path.read_text(encoding="utf-8"))
            human_ids = tokenize_content_ids(tokenizer, humans[source_id]["human_text"])
            generated_ids = tokenize_content_ids(tokenizer, record["normalized_output"])
            human_count = len(human_ids)
            generated_count = len(generated_ids)
            if human_count >= required and content_token_ids(
                tokenizer, humans[source_id]["human_text"], required
            ) != human_ids[:required]:
                raise RuntimeError(f"Human content-view mismatch: {endpoint_id}")
            if generated_count >= required and content_token_ids(
                tokenizer, record["normalized_output"], required
            ) != generated_ids[:required]:
                raise RuntimeError(f"Generated content-view mismatch: {endpoint_id}")
            counts.append((human_count, generated_count))
            if human_count < required or generated_count < required:
                failures.append(
                    {
                        "endpoint_id": endpoint_id,
                        "human_tokens": human_count,
                        "generated_tokens": generated_count,
                    }
                )

    result = {
        "status": "ok" if not failures else "failed",
        "run_id": config["run_id"],
        "tokenizer_hf_id": view["tokenizer_hf_id"],
        "tokenizer_revision": view["tokenizer_revision"],
        "tokenizer_json_sha256": sha256_file(tokenizer_path),
        "required_content_tokens": required,
        "pairs": len(counts),
        "human_tokens_min": min(value[0] for value in counts),
        "generated_tokens_min": min(value[1] for value in counts),
        "failures": failures,
    }
    print(json.dumps(result, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
