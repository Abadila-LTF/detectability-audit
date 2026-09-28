from __future__ import annotations

from typing import Any, Sequence


def tokenize_content_ids(tokenizer: Any, text: str) -> list[int]:
    """Tokenize content with special-token insertion disabled."""
    encoded = tokenizer.encode(text, add_special_tokens=False)
    token_ids = encoded.ids if hasattr(encoded, "ids") else encoded
    return [int(token_id) for token_id in token_ids]


def content_token_ids(tokenizer: Any, text: str, required: int = 128) -> list[int]:
    """Return the frozen first-N content-token view, without model boundaries."""
    if required <= 0:
        raise ValueError("required must be positive")
    ids = tokenize_content_ids(tokenizer, text)
    if len(ids) < required:
        raise ValueError(f"Text has {len(ids)} content tokens; {required} are required")
    return ids[:required]


def add_model_boundaries(tokenizer: Any, content_ids: Sequence[int]) -> tuple[list[int], list[bool]]:
    """Add RoBERTa boundaries after truncation and identify content positions exactly."""
    ids = [int(token_id) for token_id in content_ids]
    model_ids = [int(token_id) for token_id in tokenizer.build_inputs_with_special_tokens(ids)]
    special_mask = [
        int(value)
        for value in tokenizer.get_special_tokens_mask(model_ids, already_has_special_tokens=True)
    ]
    if len(model_ids) != len(special_mask):
        raise RuntimeError("Tokenizer boundary mask length mismatch")
    content_mask = [value == 0 for value in special_mask]
    recovered = [token_id for token_id, keep in zip(model_ids, content_mask) if keep]
    if recovered != ids:
        raise RuntimeError("Tokenizer changed content IDs while adding model boundaries")
    return model_ids, content_mask
