"""Pre-committed headroom-rescore rules (docs/AMENDMENT-2026-09-28-HEADROOM-RESCORE.md)."""

from __future__ import annotations

import copy
from typing import Any, Dict, Mapping, Optional

PARENT_VIEW = 128
HEADROOM_VIEWS = (64, 32)
FLOOR_VIEW = 32
DESCENT_CEILING_CELLS = 3
PYTHIA_CELLS = ("70m", "410m", "1.4b", "2.8b", "6.9b")


def derive_headroom_config(parent: Mapping[str, Any], content_tokens: int) -> Dict[str, Any]:
    """Return the parent evaluation config with only representation.content_tokens changed."""
    if content_tokens not in HEADROOM_VIEWS:
        raise ValueError(f"Headroom views are limited to {HEADROOM_VIEWS}; got {content_tokens}")
    if int(parent["representation"]["content_tokens"]) != PARENT_VIEW:
        raise ValueError("Headroom configs derive only from the frozen 128-token parent")
    derived = copy.deepcopy(dict(parent))
    derived["representation"]["content_tokens"] = content_tokens
    assert_only_view_differs(parent, derived)
    return derived


def assert_only_view_differs(parent: Mapping[str, Any], derived: Mapping[str, Any]) -> None:
    reference = copy.deepcopy(dict(derived))
    reference["representation"]["content_tokens"] = parent["representation"]["content_tokens"]
    if reference != parent:
        raise RuntimeError("Derived headroom config differs from its parent beyond content_tokens")


def ceiling_cells(evaluation: Mapping[str, Any]) -> list:
    return sorted(key for key, cell in evaluation["cells"].items() if cell["test_auroc"] == 1.0)


def needs_descent(evaluation_64: Mapping[str, Any]) -> bool:
    """Descent to 32 tokens iff three or more of nine cells are at unrounded AUROC 1 at 64."""
    if len(evaluation_64["cells"]) != 9:
        raise ValueError("Descent rule is defined over the nine-cell panel")
    return len(ceiling_cells(evaluation_64)) >= DESCENT_CEILING_CELLS


def pythia_below_ceiling(evaluation: Mapping[str, Any]) -> bool:
    return all(evaluation["cells"][f"pythia/{cell}"]["test_auroc"] < 1.0 for cell in PYTHIA_CELLS)


def select_primary_view(evaluations: Mapping[int, Mapping[str, Any]]) -> Optional[int]:
    """Longest evaluated view with all five Pythia cells below ceiling, else None."""
    allowed = {PARENT_VIEW, *HEADROOM_VIEWS}
    if not set(evaluations) <= allowed:
        raise ValueError(f"Unexpected views: {sorted(evaluations)}")
    for view in sorted(evaluations, reverse=True):
        if pythia_below_ceiling(evaluations[view]):
            return view
    return None
