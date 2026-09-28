from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Iterable


REVISION_RE = re.compile(r"^[0-9a-f]{40}$")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_hash(value: Any) -> str:
    return sha256_text(canonical_json(value))


def word_count(text: str) -> int:
    return len(text.split())


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def inclusive_word_bounds(target_words: int, lower: float, upper: float) -> tuple[int, int]:
    if target_words <= 0:
        raise ValueError("target_words must be positive")
    low = math.ceil(target_words * lower)
    high = math.floor(target_words * upper)
    if low > high:
        raise ValueError("invalid word bounds")
    return low, high


def stable_seed(*parts: object, base_seed: int = 0) -> int:
    material = "|".join(str(part) for part in (base_seed, *parts))
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big") % (2**31 - 1)


def validate_revision(revision: str) -> None:
    if not REVISION_RE.fullmatch(revision):
        raise ValueError(f"Model revision must be a 40-character lowercase commit SHA: {revision!r}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(canonical_json(row) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def validate_queue(rows: list[dict[str, Any]]) -> None:
    required = {
        "source_id",
        "compact_index",
        "split_origin",
        "cohort",
        "human_sha256",
        "topic",
        "topic_sha256",
        "target_words",
        "low_words",
        "high_words",
    }
    keys: set[tuple[str, str]] = set()
    for row in rows:
        missing = required - set(row)
        if missing:
            raise ValueError(f"Queue row missing fields: {sorted(missing)}")
        key = (str(row["cohort"]), str(row["source_id"]))
        if key in keys:
            raise ValueError(f"Duplicate queue key: {key}")
        keys.add(key)
        expected_id = f"arxiv2k:{int(row['compact_index']):04d}"
        if row["source_id"] != expected_id:
            raise ValueError(f"Source ID mismatch: {row['source_id']} != {expected_id}")
        if sha256_text(str(row["topic"])) != row["topic_sha256"]:
            raise ValueError(f"Topic hash mismatch for {row['source_id']}")
        if not int(row["low_words"]) <= int(row["target_words"]) <= int(row["high_words"]):
            raise ValueError(f"Invalid bounds for {row['source_id']}")
