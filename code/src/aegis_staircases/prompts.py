from __future__ import annotations

from typing import Any

from .contracts import canonical_hash, sha256_text, word_count


PROMPT_VERSION = "freegen-plain-full-human-1shot-fixed256-v5"

PROMPT_TEMPLATE = """Write one new scientific abstract for the task below.

Use the example only to understand the requested genre and level of detail. Do not copy its wording or claims. Return only the new abstract, with no title, labels, notes, or explanation.

{examples}

TASK
Topic: {topic}

Abstract:
"""

EXAMPLE_TEMPLATE = """EXAMPLE {number}
Topic: {topic}
Abstract:
{text}"""


def validate_exemplars(exemplars: list[dict[str, Any]]) -> None:
    if len(exemplars) != 1:
        raise ValueError("Exactly one full exemplar is required")
    for exemplar in exemplars:
        if word_count(exemplar["text"]) != int(exemplar["word_count"]):
            raise ValueError(f"Exemplar word count mismatch for {exemplar['source_id']}")
        if sha256_text(exemplar["text"]) != exemplar["human_sha256"]:
            raise ValueError(f"Exemplar human hash mismatch for {exemplar['source_id']}")
        if sha256_text(exemplar["topic"]) != exemplar["topic_sha256"]:
            raise ValueError(f"Exemplar topic hash mismatch for {exemplar['source_id']}")


def exemplar_bundle(exemplars: list[dict[str, Any]]) -> str:
    validate_exemplars(exemplars)
    return "\n\n".join(
        EXAMPLE_TEMPLATE.format(
            number=number,
            topic=exemplar["topic"],
            text=exemplar["text"],
        )
        for number, exemplar in enumerate(exemplars, 1)
    )


def build_prompt(row: dict[str, Any], exemplars: list[dict[str, Any]]) -> str:
    return PROMPT_TEMPLATE.format(
        examples=exemplar_bundle(exemplars),
        topic=row["topic"],
    )


def prompt_contract(exemplars: list[dict[str, Any]]) -> dict[str, Any]:
    bundle = exemplar_bundle(exemplars)
    return {
        "prompt_version": PROMPT_VERSION,
        "template_sha256": sha256_text(PROMPT_TEMPLATE),
        "example_template_sha256": sha256_text(EXAMPLE_TEMPLATE),
        "exemplar_bundle_sha256": sha256_text(bundle),
        "exemplar_contract_sha256": canonical_hash(exemplars),
        "surface": "plain_completion",
    }
