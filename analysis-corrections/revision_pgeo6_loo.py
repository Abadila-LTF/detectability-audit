"""
Revision (MDPI Technologies, Reviewer 1) — held-out evaluation of the additive
vector model Δ(M,c) ≈ μ + α_M + β_c (P-geo-6), which was reported in-sample.

Δ(M,c) is recomputed exactly as in scripts/track4c_pgeo6.py: StandardScaler fit
on the train split, Δ = mean(AI) − mean(human) of the scaled train embeddings
(near-copy exclusions applied). In-sample R² is re-derived and checked against
pgeo6_additive_model.json before any held-out analysis.

Leave-one-cell-out (20 cells = 10 models × 2 conditions): fit the additive model
on the 19 remaining cells and predict the held-out Δ. In a two-way layout with
one cell missing the least-squares additive fit gives the closed form
    Δ̂(M,c) = Δ(M,c') + [mean_{M'≠M} Δ(M',c) − mean_{M'≠M} Δ(M',c')].
(verified numerically against a generic least-squares fit below).

d′ values are NOT recomputed; they are read from pgeo6_additive_model.json
(dprime_map), i.e. the same d′ the in-sample analysis used.

Output: OUT_DIR/pgeo6_loo_revision.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

DATA_ROOT = Path("/Users/abadila/Desktop/ethics/AEGIS-Multi")
EMBED_CACHE = DATA_ROOT / "data" / "embed_cache"
SPLITS_DIR = DATA_ROOT / "data" / "splits"
EXCLUSION_DIR = DATA_ROOT / "data" / "exclusions"
RESULTS_DIR = DATA_ROOT / "analysis" / "results"
OUT_DIR = Path(__file__).resolve().parent.parent / "analysis" / "results"

MODELS = ["claude-sonnet", "deepseek-r1", "gemini-flash", "gpt-4o", "gpt-4o-mini",
          "llama-3.1-8b", "llama-3.3-70b", "llama-3.3-70b-instruct", "mistral-large", "qwen-72b"]
CONDITIONS = ["generate-lf", "rewrite-lf"]  # sorted, as in track4c


def load_labels(prompt, model, split):
    """Identical to track4c_pgeo6.load_labels."""
    df = pd.read_csv(SPLITS_DIR / prompt / f"{split}_{model}.csv")
    excl_path = EXCLUSION_DIR / f"{model}_near_copies.csv"
    if excl_path.exists():
        excl_idx = set(pd.read_csv(excl_path)["original_index"].tolist())
        all_csv = DATA_ROOT / "data" / "processed" / prompt / "all.csv"
        if all_csv.exists():
            all_df = pd.read_csv(all_csv)
            excluded = set(all_df[(all_df["model"] == model) &
                                  (all_df["original_index"].isin(excl_idx))]["text"].astype(str).values)
            df = df[~df["text"].astype(str).isin(excluded)]
    return df["label"].values


def compute_delta(model, cond):
    X = np.load(EMBED_CACHE / cond / f"train_{model}.npy")
    y = load_labels(cond, model, "train")
    X = X[:len(y)]
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = 1.0  # sklearn StandardScaler convention
    Xs = (X - mu) / sd
    return Xs[y == 1].mean(axis=0) - Xs[y == 0].mean(axis=0)


def additive_fit(D, mask):
    """Generic LS fit of D[m,c,:] = mu + a_m + b_c over cells where mask is True."""
    nm, nc, d = D.shape
    cells = [(m, c) for m in range(nm) for c in range(nc) if mask[m, c]]
    X = np.zeros((len(cells), 1 + nm + nc))
    for k, (m, c) in enumerate(cells):
        X[k, 0] = 1; X[k, 1 + m] = 1; X[k, 1 + nm + c] = 1
    Y = np.stack([D[m, c] for m, c in cells])
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)  # min-norm; predictions are identifiable
    def predict(m, c):
        x = np.zeros(1 + nm + nc); x[0] = 1; x[1 + m] = 1; x[1 + nm + c] = 1
        return x @ beta
    return predict


def r2(y, yhat):
    y, yhat = np.asarray(y), np.asarray(yhat)
    return float(1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def main():
    pg = json.load(open(RESULTS_DIR / "pgeo6_additive_model.json"))
    dmap = pg["dprime_map"]

    D = np.stack([np.stack([compute_delta(m, c) for c in CONDITIONS]) for m in MODELS])  # (10, 2, d)
    nm, nc, d = D.shape
    print(f"Δ tensor: {D.shape}")

    # --- in-sample reproduction (track4c) ---
    grand = D.mean(axis=(0, 1))
    pred_in = D.mean(axis=1, keepdims=True) + D.mean(axis=0, keepdims=True) - grand
    ss_tot = ((D - grand) ** 2).sum()
    r2_in = 1 - ((D - pred_in) ** 2).sum() / ss_tot
    model_norms = {m: float(np.linalg.norm(D[i].mean(0) - grand)) for i, m in enumerate(MODELS)}
    print(f"in-sample vector R² = {r2_in:.4f} (stored {pg['r2_vector_space']})")
    assert abs(r2_in - pg["r2_vector_space"]) < 5e-4, "failed to reproduce stored in-sample R²"
    for m in MODELS:
        assert abs(model_norms[m] - pg["model_norms"][m]) < 5e-3, m

    cells = [(i, j) for i in range(nm) for j in range(nc)]
    keys = [f"{MODELS[i]}/{CONDITIONS[j]}" for i, j in cells]
    dprime = np.array([dmap[k] for k in keys])
    norm_meas = np.array([np.linalg.norm(D[i, j]) for i, j in cells])
    norm_pred_in = np.array([np.linalg.norm(pred_in[i, j]) for i, j in cells])
    rho_in = spearmanr(norm_pred_in, dprime)
    print(f"in-sample ρ(‖Δ̂‖, d′) = {rho_in[0]:.4f} (stored {pg['norm_vs_dprime']['rho']})")

    # --- leave-one-cell-out ---
    loo_add, loo_model_only, loo_task_only = [], [], []
    for i, j in cells:
        jo = 1 - j
        others = [k for k in range(nm) if k != i]
        pred = D[i, jo] + D[others, j].mean(0) - D[others, jo].mean(0)
        mask = np.ones((nm, nc), bool); mask[i, j] = False
        gen = additive_fit(D, mask)(i, j)
        assert np.allclose(gen, pred, atol=1e-6), "closed form != LS fit"
        loo_add.append(pred)
        loo_model_only.append(D[i, jo])                 # baseline: same model, other condition
        loo_task_only.append(D[others, j].mean(0))      # baseline: other models, same condition
    loo_add, loo_model_only, loo_task_only = map(np.array, (loo_add, loo_model_only, loo_task_only))
    Dflat = np.array([D[i, j] for i, j in cells])

    def vec_r2(P):
        # PRESS-style: 1 − Σ‖Δ − Δ̂_(−cell)‖² / Σ‖Δ − grand mean‖²
        return float(1 - ((Dflat - P) ** 2).sum() / ((Dflat - grand) ** 2).sum())

    def summarise(P, name):
        npred = np.linalg.norm(P, axis=1)
        cos = np.einsum("kd,kd->k", P, Dflat) / (npred * norm_meas)
        s_all = spearmanr(npred, dprime); p_all = pearsonr(npred, dprime)
        within = {}
        for j, c in enumerate(CONDITIONS):
            idx = [k for k, (_, jj) in enumerate(cells) if jj == j]
            s = spearmanr(npred[idx], dprime[idx]); p = pearsonr(npred[idx], dprime[idx])
            within[c] = {"n": len(idx), "spearman_rho": round(float(s[0]), 4), "spearman_p": round(float(s[1]), 6),
                         "pearson_r": round(float(p[0]), 4), "pearson_p": round(float(p[1]), 6),
                         "norm_r2_heldout": round(r2(norm_meas[idx], npred[idx]), 4)}
        return {
            "predictor": name,
            "vector_r2_heldout": round(vec_r2(P), 4),
            "norm_r2_heldout": round(r2(norm_meas, npred), 4),
            "norm_pearson_pred_vs_measured": round(float(pearsonr(npred, norm_meas)[0]), 4),
            "mean_cosine_pred_vs_measured": round(float(cos.mean()), 4),
            "min_cosine_pred_vs_measured": round(float(cos.min()), 4),
            "dprime_spearman_rho": round(float(s_all[0]), 4), "dprime_spearman_p": round(float(s_all[1]), 6),
            "dprime_pearson_r": round(float(p_all[0]), 4), "dprime_pearson_p": round(float(p_all[1]), 6),
            "within_condition": within,
            "per_cell": {k: {"norm_measured": round(float(a), 4), "norm_predicted": round(float(b), 4),
                             "cosine": round(float(c), 4), "dprime": float(dp)}
                         for k, a, b, c, dp in zip(keys, norm_meas, npred, cos, dprime)},
        }

    add = summarise(loo_add, "additive (mu + alpha_M + beta_c), leave-one-cell-out")
    mo = summarise(loo_model_only, "baseline: model-only (Δ of same model in the other condition)")
    to = summarise(loo_task_only, "baseline: task-only (mean Δ of the other 9 models in the same condition)")
    for s in (add, mo, to):
        print(f"{s['predictor'][:40]:<40s} vecR2={s['vector_r2_heldout']:+.4f} normR2={s['norm_r2_heldout']:+.4f} "
              f"ρ(‖Δ̂‖,d′)={s['dprime_spearman_rho']:+.4f} (p={s['dprime_spearman_p']:.2e}) "
              f"within: " + ", ".join(f"{c}:{v['spearman_rho']:+.3f}" for c, v in s["within_condition"].items()))

    s_meas = spearmanr(norm_meas, dprime)
    within_meas = {c: round(float(spearmanr(norm_meas[[k for k, (_, jj) in enumerate(cells) if jj == j]],
                                            dprime[[k for k, (_, jj) in enumerate(cells) if jj == j]])[0]), 4)
                   for j, c in enumerate(CONDITIONS)}
    print(f"reference: measured ‖Δ‖ vs d′ ρ={s_meas[0]:.4f}; within {within_meas}")

    results = {
        "experiment": "P-geo-6_additive_model_LOO_revision",
        "reviewer": "R1: additive model evaluated in-sample",
        "inputs": {
            "embeddings": "data/embed_cache/{generate-lf,rewrite-lf}/train_<model>.npy",
            "labels": "data/splits/<cond>/train_<model>.csv (+ data/exclusions/*_near_copies.csv via data/processed/<cond>/all.csv)",
            "dprime": "analysis/results/pgeo6_additive_model.json -> dprime_map",
            "reproduction_check": "analysis/results/pgeo6_additive_model.json -> r2_vector_space, model_norms, norm_vs_dprime",
        },
        "data_root": str(DATA_ROOT),
        "embedding_dim": int(d),
        "in_sample_reproduction": {
            "vector_r2": round(float(r2_in), 4), "stored_vector_r2": pg["r2_vector_space"],
            "spearman_norm_pred_vs_dprime": round(float(rho_in[0]), 4), "stored": pg["norm_vs_dprime"]["rho"],
            "norm_r2_in_sample": round(r2(norm_meas, norm_pred_in), 4),
        },
        "reference_measured_norm_vs_dprime": {"spearman_rho": round(float(s_meas[0]), 4),
                                              "spearman_p": round(float(s_meas[1]), 6),
                                              "within_condition_spearman": within_meas},
        "loo_additive": add,
        "loo_baseline_model_only": mo,
        "loo_baseline_task_only": to,
        "note": ("With only 2 conditions, a held-out cell's model effect is estimated from that model's single "
                 "other cell, so LOO is a strict test of additivity (the interaction term is fully out-of-sample)."),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "pgeo6_loo_revision.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
