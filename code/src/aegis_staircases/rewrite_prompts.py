"""CLEAN-RW-v1 plain rewrite prompt (docs/PREREGISTRATION-RW-v1.md).

Mirrors the V5 one-shot serialization (same example bundle, same TASK/Topic/Abstract
layout) and adds the full source abstract plus a rewrite instruction. Identical for all
four Tulu stages; no chat template; no target-length field.
"""

from __future__ import annotations

from typing import Any

from .contracts import canonical_hash, sha256_text
from .prompts import EXAMPLE_TEMPLATE, exemplar_bundle

REWRITE_PROMPT_VERSION = "rewrite-plain-full-source-1shot-fixed256-rw-v1"

REWRITE_PROMPT_TEMPLATE = """Rewrite the source abstract below as one new scientific abstract.

Use the example only to understand the requested genre and level of detail. Do not copy the example's wording or claims. Keep the scientific content of the source abstract, but express it in new wording. Return only the new abstract, with no title, labels, notes, or explanation.

{examples}

TASK
Topic: {topic}
Source abstract:
{source}

Abstract:
"""


def build_rewrite_prompt(row: dict[str, Any], exemplars: list[dict[str, Any]]) -> str:
    source = row["source_text"]
    if sha256_text(source) != row["human_sha256"]:
        raise ValueError(f"Rewrite source text hash mismatch for {row['source_id']}")
    return REWRITE_PROMPT_TEMPLATE.format(
        examples=exemplar_bundle(exemplars),
        topic=row["topic"],
        source=source,
    )


def rewrite_prompt_contract(exemplars: list[dict[str, Any]]) -> dict[str, Any]:
    bundle = exemplar_bundle(exemplars)
    return {
        "prompt_version": REWRITE_PROMPT_VERSION,
        "template_sha256": sha256_text(REWRITE_PROMPT_TEMPLATE),
        "example_template_sha256": sha256_text(EXAMPLE_TEMPLATE),
        "exemplar_bundle_sha256": sha256_text(bundle),
        "exemplar_contract_sha256": canonical_hash(exemplars),
        "surface": "plain_completion",
        "task": "rewrite",
    }
