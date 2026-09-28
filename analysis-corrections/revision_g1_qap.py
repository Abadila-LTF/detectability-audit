"""
Revision (MDPI Technologies, Reviewer 3) — dependence-aware test for Gate G1.

The 45 pairwise (cosine, transfer) points share models, so they are not
independent and the analytic Spearman p-value in geometry_gates.json is
anti-conservative. This script runs a QAP (quadratic assignment procedure)
permutation test: the 10 model labels of the predictor matrix are permuted
(rows and columns jointly), Spearman rho over the selected pairs is
recomputed, and the observed rho is compared with the permutation null.

With n = 10 models the full permutation group (10! = 3,628,800) is enumerated
exactly; a 100,000-draw Monte Carlo version is reported alongside.

Inputs (read-only, DATA_ROOT):
  analysis/results/geometry_gates.json
      gate_G1.whitened_cosine_matrix.values, gate_G1.raw_cosine_matrix.values,
      gate_G1.transfer_matrix.values (the data that produced rho = 0.6731),
      G1_norm_corrected.per_model_whitened_norms
  analysis/results/transfer_matrix_3seed_avg.json
      meanpool_mean, aegis_mean (sensitivity: 3-seed averaged transfer)

Output: OUT_DIR/g1_qap_revision.json
"""

import itertools
import json
from pathlib import Path

import numpy as np

DATA_ROOT = Path("/Users/abadila/Desktop/ethics/AEGIS-Multi")
RESULTS_DIR = DATA_ROOT / "analysis" / "results"
OUT_DIR = Path(__file__).resolve().parent.parent / "analysis" / "results"
N_MC = 100_000
SEED = 20260928


def rankdata(x):
    """Average ranks (ties -> mean rank), numpy only."""
    x = np.asarray(x, dtype=float)
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(1, len(x) + 1)
    return (np.bincount(inv, weights=ranks) / counts)[inv]


def spearman(x, y):
    rx, ry = rankdata(x), rankdata(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    return float(rx @ ry / np.sqrt((rx @ rx) * (ry @ ry)))


def qap(pred, target, idx, label):
    """QAP test of Spearman(pred[idx], target[idx]), permuting model labels of pred.

    The selected pair set is closed under relabelling (upper triangle with a
    symmetric predictor, or all ordered off-diagonal pairs), so the multiset of
    predictor values — hence their average ranks — is permutation-invariant.
    We therefore rank once into a matrix and index it by permuted labels.
    """
    n = pred.shape[0]
    rows, cols = idx
    full_offdiag = len(rows) == n * (n - 1)
    assert full_offdiag or np.allclose(pred, pred.T), "triangle selection needs a symmetric predictor"

    obs = spearman(pred[rows, cols], target[rows, cols])

    rank_mat = np.zeros((n, n))
    rank_mat[rows, cols] = rankdata(pred[rows, cols])
    if not full_offdiag:
        rank_mat[cols, rows] = rank_mat[rows, cols]
    base = rank_mat[rows, cols]
    mu, sd = base.mean(), np.sqrt(((base - base.mean()) ** 2).sum())
    ry = rankdata(target[rows, cols])
    ry = (ry - ry.mean()) / np.sqrt(((ry - ry.mean()) ** 2).sum())

    def stats_for(perms):
        return ((rank_mat[perms[:, rows], perms[:, cols]] - mu) / sd) @ ry

    assert abs(stats_for(np.arange(n)[None, :])[0] - obs) < 1e-9

    out = {"label": label, "n_pairs": int(len(rows)), "observed_rho": round(obs, 4)}

    rng = np.random.default_rng(SEED)
    perms = np.array([rng.permutation(n) for _ in range(N_MC)])
    null = stats_for(perms)
    out["monte_carlo"] = {
        "n_permutations": N_MC,
        "seed": SEED,
        "p_one_sided": (int((null >= obs - 1e-12).sum()) + 1) / (N_MC + 1),
        "p_two_sided_abs": (int((np.abs(null) >= abs(obs) - 1e-12).sum()) + 1) / (N_MC + 1),
        "null_mean": round(float(null.mean()), 4),
        "null_sd": round(float(null.std()), 4),
        "null_q95": round(float(np.quantile(null, 0.95)), 4),
        "null_q99": round(float(np.quantile(null, 0.99)), 4),
    }

    ge, ge_abs, total, s1, s2 = 0, 0, 0, 0.0, 0.0
    it = itertools.permutations(range(n))
    while True:
        chunk = list(itertools.islice(it, 400_000))
        if not chunk:
            break
        st = stats_for(np.array(chunk, dtype=np.int64))
        ge += int((st >= obs - 1e-12).sum())
        ge_abs += int((np.abs(st) >= abs(obs) - 1e-12).sum())
        total += len(st)
        s1 += float(st.sum())
        s2 += float((st ** 2).sum())
    out["exact"] = {
        "n_permutations": total,
        "n_ge_observed": ge,
        "p_one_sided": ge / total,
        "p_two_sided_abs": ge_abs / total,
        "null_mean": round(s1 / total, 4),
        "null_sd": round(float(np.sqrt(s2 / total - (s1 / total) ** 2)), 4),
    }
    return out


def main():
    gg = json.load(open(RESULTS_DIR / "geometry_gates.json"))
    g1 = gg["gate_G1"]
    models = g1["transfer_matrix"]["models"]
    assert models == g1["whitened_cosine_matrix"]["models"]
    n = len(models)
    W = np.array(g1["whitened_cosine_matrix"]["values"])
    R = np.array(g1["raw_cosine_matrix"]["values"])
    T = np.array(g1["transfer_matrix"]["values"])  # T[i, j] = probe trained on i, tested on j

    tm = json.load(open(RESULTS_DIR / "transfer_matrix_3seed_avg.json"))
    assert tm["models"] == models
    T_mp3 = np.array(tm["meanpool_mean"])
    T_ag3 = np.array(tm["aegis_mean"])

    norms_d = gg["G1_norm_corrected"]["per_model_whitened_norms"]
    norms = np.array([norms_d[m] for m in models])
    NC = W * norms[None, :]                          # cos(A,B) * ||Δ_B||, asymmetric
    NCs = W * (norms[None, :] + norms[:, None]) / 2  # symmetric variant

    triu = np.triu_indices(n, k=1)
    tril = (triu[1], triu[0])
    offd = np.where(~np.eye(n, dtype=bool))
    Tsym = (T + T.T) / 2

    tests = [
        ("primary", W, T, triu,
         "Whitened cosine vs transfer, upper triangle (train i -> test j, i<j) — the manuscript's rho=0.673"),
        ("raw_cosine", R, T, triu, "Raw (unwhitened) cosine vs transfer, upper triangle"),
        ("lower_triangle", W, T, tril, "Whitened cosine vs transfer, lower triangle (train j -> test i)"),
        ("symmetrised", W, Tsym, triu, "Whitened cosine vs symmetrised transfer (T+T')/2"),
        ("all_ordered_90", W, T, offd, "Whitened cosine vs transfer, all 90 ordered off-diagonal pairs"),
        ("meanpool_3seed", W, T_mp3, triu,
         "Whitened cosine vs 3-seed MeanPool transfer (transfer_matrix_3seed_avg.meanpool_mean), upper"),
        ("aegis_3seed", W, T_ag3, triu,
         "Whitened cosine vs 3-seed AEGIS transfer (transfer_matrix_3seed_avg.aegis_mean), upper"),
        ("norm_corrected_90", NC, T, offd, "Norm-corrected cos x ||Δ_target|| vs transfer, 90 ordered pairs"),
        ("norm_corrected_sym", NCs, T, triu, "Norm-corrected symmetric cos x mean||Δ|| vs transfer, upper"),
    ]

    results = {
        "experiment": "G1_QAP_permutation_revision",
        "reviewer": "R3: 45 pairwise comparisons are not independent",
        "method": ("QAP: jointly permute row/column model labels of the predictor matrix; recompute Spearman rho "
                   "over the selected pairs. Exact enumeration of all 10! = 3,628,800 relabellings (identity "
                   "included) plus 100,000 Monte Carlo draws (p = (#{rho_perm >= rho_obs} + 1)/(B + 1))."),
        "inputs": {
            "geometry_gates.json": ["gate_G1.whitened_cosine_matrix.values", "gate_G1.raw_cosine_matrix.values",
                                    "gate_G1.transfer_matrix.values", "G1_norm_corrected.per_model_whitened_norms"],
            "transfer_matrix_3seed_avg.json": ["meanpool_mean", "aegis_mean"],
        },
        "data_root": str(DATA_ROOT),
        "models": models,
        "reported_in_source": {
            "gate_G1.whitened_cosine_vs_transfer": g1["whitened_cosine_vs_transfer"],
            "gate_G1.raw_cosine_vs_transfer": g1["raw_cosine_vs_transfer"],
            "G1_norm_corrected.norm_corrected_rho": gg["G1_norm_corrected"]["norm_corrected_rho"],
            "G1_norm_corrected.norm_corrected_symmetric_rho": gg["G1_norm_corrected"]["norm_corrected_symmetric_rho"],
        },
        "tests": {},
    }
    for key, pred, targ, idx, label in tests:
        r = qap(pred, targ, idx, label)
        print(f"{key:<20s} rho={r['observed_rho']:+.4f}  exact p={r['exact']['p_one_sided']:.2e} "
              f"({r['exact']['n_ge_observed']}/{r['exact']['n_permutations']})  "
              f"MC p={r['monte_carlo']['p_one_sided']:.2e}  null sd={r['exact']['null_sd']}")
        results["tests"][key] = r

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "g1_qap_revision.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
