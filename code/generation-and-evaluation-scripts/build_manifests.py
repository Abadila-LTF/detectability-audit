#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aegis_staircases.contracts import (  # noqa: E402
    atomic_write_json,
    inclusive_word_bounds,
    sha256_file,
    sha256_text,
    validate_queue,
    word_count,
    write_jsonl,
)
from aegis_staircases.prompts import PROMPT_VERSION, prompt_contract  # noqa: E402


CONFIG_PATH = ROOT / "configs" / "experiment.v1.json"
OUTPUT_DIR = ROOT / "data" / "manifests"


def require_hash(path: Path, expected: str, label: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"{label} SHA-256 mismatch: {actual} != {expected}")


def load_raw_sources(path: Path, contract: dict) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != int(contract["expected_raw_rows"]):
        raise RuntimeError(f"Expected {contract['expected_raw_rows']} raw rows, found {len(rows)}")
    if set(rows[0]) != {"text", "generated"}:
        raise RuntimeError(f"Unexpected source columns: {list(rows[0])}")
    labels = [int(row["generated"]) for row in rows]
    if labels.count(0) != int(contract["expected_human_rows"]) or labels.count(1) != int(contract["expected_human_rows"]):
        raise RuntimeError("Expected 2,000 human and 2,000 legacy-AI rows")
    if labels != [index % 2 for index in range(len(labels))]:
        raise RuntimeError("Source rows are not the audited alternating human/legacy-AI sequence")
    humans = [row["text"] for row in rows if int(row["generated"]) == int(contract["human_label"])]
    if len(humans) != int(contract["expected_human_rows"]):
        raise RuntimeError("Filtered human count mismatch")
    return humans


def load_topics(path: Path, humans: list[str]) -> dict[int, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(humans):
        raise RuntimeError(f"Expected {len(humans)} topics, found {len(rows)}")
    topics: dict[int, dict[str, str]] = {}
    for row in rows:
        index = int(row["original_index"])
        if index in topics:
            raise RuntimeError(f"Duplicate topic index: {index}")
        if index < 0 or index >= len(humans):
            raise RuntimeError(f"Topic index out of range: {index}")
        if row["original_text"] != humans[index]:
            raise RuntimeError(f"Topic source mismatch at compact human index {index}")
        topic = row["topic"].strip()
        if not topic:
            raise RuntimeError(f"Empty topic at compact human index {index}")
        topics[index] = {"topic": topic, "topic_sha256": sha256_text(topic)}
    if set(topics) != set(range(len(humans))):
        raise RuntimeError("Topic indices do not cover 0..1999 exactly")
    return topics


def load_canonical_manifest(path: Path, humans: list[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(humans):
        raise RuntimeError("Canonical source manifest row count mismatch")
    for index, (row, human_text) in enumerate(zip(rows, humans)):
        expected_id = f"arxiv2k:{index:04d}"
        if row["source_id"] != expected_id or int(row["original_index"]) != index:
            raise RuntimeError(f"Canonical identity mismatch at {index}")
        if row["human_sha256"] != sha256_text(human_text):
            raise RuntimeError(f"Canonical human hash mismatch at {index}")
        if int(row["human_word_count"]) != word_count(human_text):
            raise RuntimeError(f"Canonical word count mismatch at {index}")
        if row["split_origin"] not in {"train", "val", "test"}:
            raise RuntimeError(f"Invalid split at {index}")
    return rows


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    contract = config["source_contract"]
    if config["prompt"]["version"] != PROMPT_VERSION:
        raise RuntimeError("Config and code prompt versions differ")

    source_path = (ROOT / contract["source_csv"]).resolve()
    topics_path = (ROOT / contract["topics_csv"]).resolve()
    canonical_path = (ROOT / contract["canonical_source_manifest"]).resolve()
    exemplar_selection_path = (ROOT / contract["exemplar_selection_report"]).resolve()
    require_hash(source_path, contract["source_csv_sha256"], "Source CSV")
    require_hash(topics_path, contract["topics_csv_sha256"], "Topics CSV")
    require_hash(canonical_path, contract["canonical_source_manifest_sha256"], "Canonical source manifest")
    require_hash(
        exemplar_selection_path,
        contract["exemplar_selection_report_sha256"],
        "Exemplar selection report",
    )

    humans = load_raw_sources(source_path, contract)
    topics = load_topics(topics_path, humans)
    canonical_rows = load_canonical_manifest(canonical_path, humans)
    exemplar_selection = json.loads(exemplar_selection_path.read_text(encoding="utf-8"))

    lower = float(config["decoding"]["lower_word_ratio"])
    upper = float(config["decoding"]["upper_word_ratio"])
    human_rows: list[dict] = []
    for index, (human_text, canonical) in enumerate(zip(humans, canonical_rows)):
        target = word_count(human_text)
        low, high = inclusive_word_bounds(target, lower, upper)
        human_rows.append(
            {
                "source_id": f"arxiv2k:{index:04d}",
                "compact_index": index,
                "raw_row_index": 2 * index,
                "raw_label": 0,
                "split_origin": canonical["split_origin"],
                "human_text": human_text,
                "human_sha256": sha256_text(human_text),
                "human_word_count": target,
                "topic": topics[index]["topic"],
                "topic_sha256": topics[index]["topic_sha256"],
                "low_words": low,
                "high_words": high,
            }
        )

    exemplar_indices = [int(value) for value in contract["exemplar_compact_indices"]]
    main_start, main_end = (int(value) for value in contract["main_compact_indices"])
    main_indices = list(range(main_start, main_end + 1))
    pilot_candidates = [
        row["compact_index"]
        for row in human_rows
        if row["compact_index"] >= int(contract["pilot_min_compact_index"])
        and row["split_origin"] == contract["pilot_split"]
        and row["compact_index"] not in exemplar_indices
    ]
    pilot_indices = pilot_candidates[: int(contract["pilot_size"])]
    if len(pilot_indices) != int(contract["pilot_size"]):
        raise RuntimeError("Not enough pilot sources")
    if set(main_indices) & set(pilot_indices) or set(main_indices) & set(exemplar_indices) or set(pilot_indices) & set(exemplar_indices):
        raise RuntimeError("Main, pilot, and exemplar identities must be disjoint")
    selected = exemplar_selection["selected"]
    if exemplar_indices != [int(selected["compact_index"])]:
        raise RuntimeError("Configured exemplar differs from the frozen tokenizer-aware selection report")
    selected_index = exemplar_indices[0]
    selected_row = human_rows[selected_index]
    selected_canonical = canonical_rows[selected_index]
    if selected["source_id"] != selected_row["source_id"] or selected["human_sha256"] != selected_row["human_sha256"]:
        raise RuntimeError("Frozen exemplar selection identity/hash mismatch")
    eligibility = exemplar_selection["eligibility"]
    if (
        selected_row["split_origin"] != eligibility["split_origin"]
        or selected_canonical["primary_common_include"].strip().lower() != "true"
        or not int(eligibility["minimum_words"])
        <= selected_row["human_word_count"]
        <= int(eligibility["maximum_words"])
    ):
        raise RuntimeError("Frozen exemplar no longer satisfies its prospective eligibility rule")

    queue_rows: list[dict] = []
    for cohort, indices in (("main", main_indices), ("pilot", pilot_indices)):
        for index in indices:
            row = human_rows[index]
            queue_rows.append(
                {
                    "source_id": row["source_id"],
                    "compact_index": index,
                    "split_origin": row["split_origin"],
                    "cohort": cohort,
                    "human_sha256": row["human_sha256"],
                    "topic": row["topic"],
                    "topic_sha256": row["topic_sha256"],
                    "target_words": row["human_word_count"],
                    "low_words": row["low_words"],
                    "high_words": row["high_words"],
                }
            )
    validate_queue(queue_rows)

    exemplars = [
        {
            "source_id": human_rows[index]["source_id"],
            "compact_index": index,
            "split_origin": human_rows[index]["split_origin"],
            "human_sha256": human_rows[index]["human_sha256"],
            "word_count": human_rows[index]["human_word_count"],
            "topic": human_rows[index]["topic"],
            "topic_sha256": human_rows[index]["topic_sha256"],
            "text": human_rows[index]["human_text"],
        }
        for index in exemplar_indices
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    human_path = OUTPUT_DIR / "human_sources.jsonl"
    queue_path = OUTPUT_DIR / "freegen_queue.jsonl"
    exemplars_path = OUTPUT_DIR / "exemplars.json"
    write_jsonl(human_path, human_rows)
    write_jsonl(queue_path, queue_rows)
    atomic_write_json(exemplars_path, exemplars)

    prompt_info = prompt_contract(exemplars)
    lock = {
        "schema_version": "aegis.clean-staircase.manifest-lock.v1",
        "run_id": config["run_id"],
        "config_sha256": sha256_file(CONFIG_PATH),
        "source_csv_sha256": sha256_file(source_path),
        "topics_csv_sha256": sha256_file(topics_path),
        "canonical_source_manifest_sha256": sha256_file(canonical_path),
        "exemplar_selection_report_sha256": sha256_file(exemplar_selection_path),
        "human_sources_sha256": sha256_file(human_path),
        "freegen_queue_sha256": sha256_file(queue_path),
        "exemplars_sha256": sha256_file(exemplars_path),
        "prompt_contract": prompt_info,
        "counts": {
            "raw_source_rows": 4000,
            "human_sources": len(human_rows),
            "main_sources": len(main_indices),
            "pilot_sources": len(pilot_indices),
            "exemplars": len(exemplars),
            "queue_rows": len(queue_rows),
        },
        "main_source_ids": [human_rows[index]["source_id"] for index in main_indices],
        "pilot_source_ids": [human_rows[index]["source_id"] for index in pilot_indices],
        "exemplar_source_ids": [human_rows[index]["source_id"] for index in exemplar_indices],
        "exemplar_selection": exemplar_selection,
    }
    atomic_write_json(OUTPUT_DIR / "manifest-lock.json", lock)
    print(json.dumps({"status": "ok", "counts": lock["counts"], "prompt": prompt_info}, indent=2))


if __name__ == "__main__":
    main()
