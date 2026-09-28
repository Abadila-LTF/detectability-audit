from __future__ import annotations

import itertools
import math
from collections import Counter
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any, Mapping, Sequence

import numpy as np

from aegis_staircases.contracts import canonical_json, sha256_bytes
from aegis_staircases.evaluation_contract import (  # re-exported for evaluation callers
    expected_main_source_ids,
    load_evaluation_config,
)


def array_contract_hash(array: np.ndarray) -> dict[str, Any]:
    """Hash an array's semantic dtype/shape/content rather than its container bytes."""
    contiguous = np.ascontiguousarray(array)
    header = canonical_json({"dtype": str(contiguous.dtype), "shape": list(contiguous.shape)})
    digest = sha256_bytes(header.encode("utf-8") + b"\0" + contiguous.tobytes(order="C"))
    return {"dtype": str(contiguous.dtype), "shape": list(contiguous.shape), "sha256": digest}
from aegis_staircases.views import (  # re-exported for evaluation callers
    add_model_boundaries,
    content_token_ids,
    tokenize_content_ids,
)


def average_ranks(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("values must be a nonempty one-dimensional sequence")
    if not np.isfinite(array).all():
        raise ValueError("rank inputs must be finite")
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.size, dtype=float)
    start = 0
    while start < array.size:
        end = start + 1
        while end < array.size and array[order[end]] == array[order[start]]:
            end += 1
        average = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average
        start = end
    return ranks


def binary_auroc(human_scores: Sequence[float], generated_scores: Sequence[float]) -> float:
    human = np.asarray(human_scores, dtype=float)
    generated = np.asarray(generated_scores, dtype=float)
    if human.ndim != 1 or generated.ndim != 1 or human.size == 0 or generated.size == 0:
        raise ValueError("Both score arrays must be nonempty and one-dimensional")
    if not np.isfinite(human).all() or not np.isfinite(generated).all():
        raise ValueError("Scores must be finite")
    comparisons = (generated[:, None] > human[None, :]).astype(float)
    comparisons += 0.5 * (generated[:, None] == human[None, :])
    return float(comparisons.mean())


def sourcewise_auc_contributions(
    human_scores: Sequence[float], generated_scores: Sequence[float]
) -> np.ndarray:
    """Decompose paired-source AUROC so the source-contribution mean is the AUROC."""
    human = np.asarray(human_scores, dtype=float)
    generated = np.asarray(generated_scores, dtype=float)
    if human.ndim != 1 or generated.ndim != 1 or human.shape != generated.shape or not human.size:
        raise ValueError("Paired human/generated score arrays must have the same nonzero length")
    comparisons = (generated[:, None] > human[None, :]).astype(float)
    comparisons += 0.5 * (generated[:, None] == human[None, :])
    contributions = 0.5 * (comparisons.mean(axis=1) + comparisons.mean(axis=0))
    auc = binary_auroc(human, generated)
    if not math.isclose(float(contributions.mean()), auc, rel_tol=0.0, abs_tol=1e-15):
        raise RuntimeError("Sourcewise AUROC decomposition failed its mean identity")
    return contributions


def _page_scaled_statistic(block_values: np.ndarray) -> tuple[int, list[tuple[int, ...]]]:
    values = np.asarray(block_values, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 2:
        raise ValueError("Page input must be a nonempty blocks-by-conditions matrix")
    weights = np.arange(1, values.shape[1] + 1, dtype=int)
    rank_rows: list[tuple[int, ...]] = []
    scaled_statistic = 0
    for row in values:
        scaled_ranks_array = average_ranks(row) * 2.0
        rounded = np.rint(scaled_ranks_array).astype(int)
        if not np.array_equal(scaled_ranks_array, rounded.astype(float)):
            raise RuntimeError("Page average ranks were not representable in half-rank units")
        scaled_ranks = tuple(int(value) for value in rounded)
        rank_rows.append(scaled_ranks)
        scaled_statistic += int(np.dot(weights, rounded))
    return scaled_statistic, rank_rows


def exact_page_test(block_values: Sequence[Sequence[float]]) -> dict[str, Any]:
    """One-sided Page test using an exact, tie-aware dynamic program.

    Within each source block all k! assignments are counted. Repeated outcomes caused
    by ties retain their permutation multiplicity, and block distributions are then
    convolved exactly with integer counts.
    """
    values = np.asarray(block_values, dtype=float)
    observed_scaled, rank_rows = _page_scaled_statistic(values)
    conditions = values.shape[1]
    weights = tuple(range(1, conditions + 1))
    permutations = tuple(itertools.permutations(range(conditions)))
    outcome_cache: dict[tuple[int, ...], Counter[int]] = {}
    distribution: Counter[int] = Counter({0: 1})
    for ranks in rank_rows:
        outcomes = outcome_cache.get(ranks)
        if outcomes is None:
            outcomes = Counter(
                sum(weights[position] * ranks[permutation[position]] for position in range(conditions))
                for permutation in permutations
            )
            outcome_cache[ranks] = outcomes
        updated: Counter[int] = Counter()
        for partial, partial_count in distribution.items():
            for addition, addition_count in outcomes.items():
                updated[partial + addition] += partial_count * addition_count
        distribution = updated

    denominator = math.factorial(conditions) ** values.shape[0]
    if sum(distribution.values()) != denominator:
        raise RuntimeError("Page exact distribution lost permutation mass")
    numerator = sum(count for statistic, count in distribution.items() if statistic >= observed_scaled)
    null_mean_scaled = sum(statistic * count for statistic, count in distribution.items()) / denominator
    if observed_scaled > null_mean_scaled:
        direction = "increasing"
    elif observed_scaled < null_mean_scaled:
        direction = "decreasing"
    else:
        direction = "flat"
    return {
        "blocks": int(values.shape[0]),
        "conditions": int(conditions),
        "statistic": observed_scaled / 2.0,
        "statistic_scaled_by_2": observed_scaled,
        "null_mean": null_mean_scaled / 2.0,
        "direction": direction,
        "alternative": "increasing",
        "p_value": float(numerator / denominator),
        "exact_tail_numerator": str(numerator),
        "exact_denominator": str(denominator),
        "tie_aware": True,
        "method": "exact_page_dynamic_program",
    }


def spearman_rho(x: Sequence[float], y: Sequence[float]) -> float | None:
    x_ranks = average_ranks(x)
    y_ranks = average_ranks(y)
    x_centered = x_ranks - x_ranks.mean()
    y_centered = y_ranks - y_ranks.mean()
    denominator = float(np.sqrt(np.dot(x_centered, x_centered) * np.dot(y_centered, y_centered)))
    if denominator == 0.0:
        return None
    return float(np.dot(x_centered, y_centered) / denominator)


def exact_spearman_test(
    x: Sequence[float], y: Sequence[float], *, alternative: str = "less"
) -> dict[str, Any]:
    if alternative not in {"less", "greater"}:
        raise ValueError("alternative must be 'less' or 'greater'")
    x_values = tuple(float(value) for value in x)
    y_values = tuple(float(value) for value in y)
    if len(x_values) != len(y_values) or len(x_values) < 2:
        raise ValueError("Spearman inputs must have the same length of at least two")
    observed = spearman_rho(x_values, y_values)
    denominator = math.factorial(len(y_values))
    if observed is None:
        return {
            "rho": None,
            "p_value": None,
            "exact_tail_numerator": None,
            "exact_denominator": denominator,
            "permutations": denominator,
            "alternative": alternative,
            "method": "exact_permutation_spearman",
        }
    numerator = 0
    tolerance = 1e-15
    for permutation in itertools.permutations(range(len(y_values))):
        permuted = [y_values[index] for index in permutation]
        statistic = spearman_rho(x_values, permuted)
        if statistic is None:
            raise RuntimeError("A nonconstant observed Spearman vector became constant under permutation")
        if alternative == "less" and statistic <= observed + tolerance:
            numerator += 1
        if alternative == "greater" and statistic >= observed - tolerance:
            numerator += 1
    return {
        "rho": observed,
        "p_value": float(numerator / denominator),
        "exact_tail_numerator": numerator,
        "exact_denominator": denominator,
        "permutations": denominator,
        "alternative": alternative,
        "method": "exact_permutation_spearman",
    }


def auroc_to_dprime(auroc: float) -> dict[str, Any]:
    value = float(auroc)
    if not 0.0 <= value <= 1.0:
        raise ValueError("AUROC must lie in [0, 1]")
    if value == 1.0:
        return {"value": None, "censoring": "right", "reason": "unrounded_auroc_equals_one"}
    if value == 0.0:
        return {"value": None, "censoring": "left", "reason": "unrounded_auroc_equals_zero"}
    return {
        "value": math.sqrt(2.0) * NormalDist().inv_cdf(value),
        "censoring": None,
        "reason": None,
    }


def _direction(value: float | None, *, tolerance: float = 1e-15) -> str:
    if value is None or abs(value) <= tolerance:
        return "flat"
    return "negative" if value < 0 else "positive"


def pythia_ceiling_audit(
    names: Sequence[str], parameters: Sequence[int], aurocs: Sequence[float], *, alpha: float = 0.05
) -> dict[str, Any]:
    if not (len(names) == len(parameters) == len(aurocs) == 5):
        raise ValueError("The frozen Pythia audit requires exactly five aligned cells")
    x = [math.log10(int(value)) for value in parameters]
    values = [float(value) for value in aurocs]
    ordinary = exact_spearman_test(x, values, alternative="less")
    ceiling_indices = [index for index, value in enumerate(values) if value == 1.0]
    result: dict[str, Any] = {
        "ceiling_cells": [names[index] for index in ceiling_indices],
        "ordinary_observed": ordinary,
        "enumeration_count": math.factorial(len(ceiling_indices)) if len(ceiling_indices) >= 2 else 0,
        "operationalization": (
            "Every globally consistent relative order among AUROC==1 cells is placed above all "
            "finite cells; the exact 120-permutation Spearman test is recomputed for each order."
        ),
    }
    if len(ceiling_indices) < 2:
        result.update({"status": "NO_AMBIGUOUS_CEILING_ORDER", "admissible": []})
        return result

    finite_max = max((value for value in values if value < 1.0), default=0.0)
    admissible = []
    for ordered_indices in itertools.permutations(ceiling_indices):
        adjusted = values.copy()
        for hidden_rank, cell_index in enumerate(ordered_indices, 1):
            adjusted[cell_index] = finite_max + hidden_rank
        test = exact_spearman_test(x, adjusted, alternative="less")
        admissible.append(
            {
                "ceiling_order_low_to_high": [names[index] for index in ordered_indices],
                "rho": test["rho"],
                "p_value": test["p_value"],
                "direction": _direction(test["rho"]),
                "reject_at_alpha": bool(test["p_value"] < alpha),
            }
        )
    directions = sorted({entry["direction"] for entry in admissible})
    decisions = sorted({entry["reject_at_alpha"] for entry in admissible})
    withheld = len(directions) > 1 or len(decisions) > 1
    result.update(
        {
            "status": "WITHHELD_CEILING" if withheld else "ROBUST_TO_CEILING_ORDER",
            "rho_range": [min(entry["rho"] for entry in admissible), max(entry["rho"] for entry in admissible)],
            "p_value_range": [
                min(entry["p_value"] for entry in admissible),
                max(entry["p_value"] for entry in admissible),
            ],
            "directions": directions,
            "alpha_decisions": decisions,
            "admissible": admissible,
        }
    )
    return result


def tulu_ceiling_audit(
    names: Sequence[str], contributions: Sequence[Sequence[float]], aurocs: Sequence[float], *, alpha: float = 0.05
) -> dict[str, Any]:
    values = np.asarray(contributions, dtype=float)
    if len(names) != 4 or values.ndim != 2 or values.shape[1] != 4 or len(aurocs) != 4:
        raise ValueError("The frozen Tulu audit requires four aligned cells")
    ordinary = exact_page_test(values)
    ceiling_indices = [index for index, value in enumerate(aurocs) if float(value) == 1.0]
    result: dict[str, Any] = {
        "ceiling_cells": [names[index] for index in ceiling_indices],
        "ordinary_observed": ordinary,
        "enumeration_count": math.factorial(len(ceiling_indices)) if len(ceiling_indices) >= 2 else 0,
        "operationalization": (
            "For each globally consistent relative order among AUROC==1 cells, only those ceiling "
            "ties are broken in the same order in every source block and above every finite cell; "
            "non-ceiling values and their ties are preserved before recomputing the exact Page test."
        ),
    }
    if len(ceiling_indices) < 2:
        result.update({"status": "NO_AMBIGUOUS_CEILING_ORDER", "admissible": []})
        return result

    admissible = []
    for ordered_indices in itertools.permutations(ceiling_indices):
        adjusted = values.copy()
        for hidden_rank, cell_index in enumerate(ordered_indices, 1):
            adjusted[:, cell_index] = 1.0 + hidden_rank
        test = exact_page_test(adjusted)
        admissible.append(
            {
                "ceiling_order_low_to_high": [names[index] for index in ordered_indices],
                "statistic": test["statistic"],
                "p_value": test["p_value"],
                "direction": test["direction"],
                "reject_at_alpha": bool(test["p_value"] < alpha),
            }
        )
    directions = sorted({entry["direction"] for entry in admissible})
    decisions = sorted({entry["reject_at_alpha"] for entry in admissible})
    withheld = len(directions) > 1 or len(decisions) > 1
    result.update(
        {
            "status": "WITHHELD_CEILING" if withheld else "ROBUST_TO_CEILING_ORDER",
            "statistic_range": [
                min(entry["statistic"] for entry in admissible),
                max(entry["statistic"] for entry in admissible),
            ],
            "p_value_range": [
                min(entry["p_value"] for entry in admissible),
                max(entry["p_value"] for entry in admissible),
            ],
            "directions": directions,
            "alpha_decisions": decisions,
            "admissible": admissible,
        }
    )
    return result


def paired_source_bootstrap(
    scores: Mapping[str, tuple[Sequence[float], Sequence[float]]],
    *,
    replicates: int = 10000,
    seed: int = 730241,
    contrasts: Mapping[str, tuple[str, str]] | None = None,
    quantile_method: str = "linear",
    chunk_size: int = 256,
) -> dict[str, Any]:
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    names = list(scores)
    if not names:
        raise ValueError("At least one cell is required")
    kernels = []
    source_count: int | None = None
    observed: dict[str, float] = {}
    for name in names:
        human = np.asarray(scores[name][0], dtype=float)
        generated = np.asarray(scores[name][1], dtype=float)
        if human.ndim != 1 or generated.shape != human.shape or human.size == 0:
            raise ValueError(f"Invalid paired scores for {name}")
        if source_count is None:
            source_count = int(human.size)
        elif human.size != source_count:
            raise ValueError("Every cell must use the same paired source count")
        kernel = (generated[:, None] > human[None, :]).astype(float)
        kernel += 0.5 * (generated[:, None] == human[None, :])
        kernels.append(kernel)
        observed[name] = float(kernel.mean())
    assert source_count is not None

    rng = np.random.default_rng(seed)
    samples = {name: np.empty(replicates, dtype=float) for name in names}
    probabilities = np.full(source_count, 1.0 / source_count)
    for start in range(0, replicates, chunk_size):
        stop = min(start + chunk_size, replicates)
        counts = rng.multinomial(source_count, probabilities, size=stop - start).astype(float)
        for name, kernel in zip(names, kernels):
            samples[name][start:stop] = np.sum((counts @ kernel) * counts, axis=1) / (source_count**2)

    def interval(values: np.ndarray) -> list[float]:
        return [
            float(value)
            for value in np.quantile(values, [0.025, 0.975], method=quantile_method)
        ]

    result: dict[str, Any] = {
        "replicates": replicates,
        "seed": seed,
        "paired_source_count": source_count,
        "same_resample_across_all_cells": True,
        "cells": {
            name: {"auroc": observed[name], "auroc_ci95": interval(samples[name])}
            for name in names
        },
        "contrasts": {},
    }
    for label, (higher, lower) in (contrasts or {}).items():
        if higher not in samples or lower not in samples:
            raise ValueError(f"Unknown contrast cell in {label}: {(higher, lower)}")
        differences = samples[higher] - samples[lower]
        result["contrasts"][label] = {
            "auroc_difference": observed[higher] - observed[lower],
            "auroc_difference_ci95": interval(differences),
        }
    return result


@dataclass
class DetectorFit:
    selected_c: float
    validation_auroc: float
    test_auroc: float
    validation_grid: list[dict[str, float]]
    human_scores: dict[str, np.ndarray]
    generated_scores: dict[str, np.ndarray]


def fit_detector(
    human_embeddings: np.ndarray,
    generated_embeddings: np.ndarray,
    split_origins: Sequence[str],
    *,
    c_grid: Sequence[float] = (0.01, 0.1, 1.0, 10.0, 100.0),
    max_iter: int = 10000,
    tolerance: float = 0.0001,
    random_seed: int = 730241,
) -> DetectorFit:
    """Fit the frozen train-only-scaled detector and select C on validation AUROC."""
    try:
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
    except ImportError as error:  # pragma: no cover - exercised only without evaluation extras
        raise RuntimeError("Install the exact 'evaluation' extra before fitting detectors") from error
    import warnings

    human = np.asarray(human_embeddings, dtype=float)
    generated = np.asarray(generated_embeddings, dtype=float)
    splits = np.asarray(split_origins)
    if human.ndim != 2 or generated.shape != human.shape or human.shape[0] != splits.size:
        raise ValueError("Human/generated embeddings and source split labels must align")
    if set(splits) != {"train", "val", "test"}:
        raise ValueError("split_origins must contain train, val, and test")
    if sorted(float(value) for value in c_grid) != list(float(value) for value in c_grid):
        raise ValueError("c_grid must be in ascending order for the smallest-C tie break")

    train_sources = splits == "train"
    scaler = StandardScaler().fit(np.concatenate([human[train_sources], generated[train_sources]], axis=0))
    human_scaled = scaler.transform(human)
    generated_scaled = scaler.transform(generated)
    train_x = np.concatenate([human_scaled[train_sources], generated_scaled[train_sources]], axis=0)
    train_y = np.concatenate(
        [np.zeros(train_sources.sum(), dtype=int), np.ones(train_sources.sum(), dtype=int)]
    )
    val_sources = splits == "val"
    validation_grid = []
    best_model = None
    best_c = None
    best_auc = -math.inf
    for c_value in c_grid:
        model = LogisticRegression(
            C=float(c_value),
            penalty="l2",
            solver="lbfgs",
            max_iter=max_iter,
            tol=tolerance,
            random_state=random_seed,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(train_x, train_y)
        human_val = model.decision_function(human_scaled[val_sources])
        generated_val = model.decision_function(generated_scaled[val_sources])
        auc = binary_auroc(human_val, generated_val)
        validation_grid.append({"c": float(c_value), "auroc": auc})
        if auc > best_auc:
            best_auc = auc
            best_c = float(c_value)
            best_model = model
    if best_model is None or best_c is None:
        raise RuntimeError("No detector candidate was fitted")

    human_scores: dict[str, np.ndarray] = {}
    generated_scores: dict[str, np.ndarray] = {}
    for split in ("train", "val", "test"):
        mask = splits == split
        human_scores[split] = np.asarray(best_model.decision_function(human_scaled[mask]), dtype=float)
        generated_scores[split] = np.asarray(
            best_model.decision_function(generated_scaled[mask]), dtype=float
        )
    return DetectorFit(
        selected_c=best_c,
        validation_auroc=best_auc,
        test_auroc=binary_auroc(human_scores["test"], generated_scores["test"]),
        validation_grid=validation_grid,
        human_scores=human_scores,
        generated_scores=generated_scores,
    )
