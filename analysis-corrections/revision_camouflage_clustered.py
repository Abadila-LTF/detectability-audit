"""
Revision (MDPI Technologies, Reviewers 1 & 3) — source- and generator-aware
analysis of the camouflage effect (BLEU(source, rewrite) vs probe P(AI)).

Per-pair data are not stored in camouflage_bleu_confidence.json (summaries
only), so they are regenerated with the ORIGINAL code: probe confidences and
the pair construction are imported from scripts/track3_camouflage.py
(read-only import), BLEU from scripts/bleu_utils.py.

ALIGNMENT AUDIT. track3.load_source_texts() keys sources by row position in
AEGIS/data/rewritten_texts_file_2K_1.csv (first 2,000 rows). That file
alternates human (generated=0) and AI (generated=1) rows, so row k is NOT the
human abstract with original_index k. We therefore run every analysis under
two pairings:
  "original_track3"  — exactly as published (reproduces rho = −0.2078)
  "aligned"          — source = human text with the same original_index in
                       data/processed/rewrite-lf/all.csv (model == "human")

Analyses (per pairing):
  (a) pooled Spearman rho with a cluster bootstrap over source documents
      (original_index; all pairs from a resampled source enter together),
      B = 5,000, percentile 95% CI; generator-cluster bootstrap as sensitivity.
  (b) per-generator Spearman rho with bootstrap CIs (one pair per source per
      generator, so resampling pairs == resampling sources).
  (c) within-cluster centring on global ranks: rank BLEU and P(AI) over all
      pairs, subtract (i) source means, (ii) generator means, (iii) both
      (two-way demeaning by alternating projections), then Pearson on the
      residual ranks; CIs from the same source-cluster bootstrap.

Output: OUT_DIR/camouflage_clustered_revision.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True  # do not write __pycache__ into DATA_ROOT
DATA_ROOT = Path("/Users/abadila/Desktop/ethics/AEGIS-Multi")
sys.path.insert(0, str(DATA_ROOT / "scripts"))
import track3_camouflage as t3  # noqa: E402
from bleu_utils import compute_bleu4, edit_similarity  # noqa: E402

RESULTS_DIR = DATA_ROOT / "analysis" / "results"
OUT_DIR = Path(__file__).resolve().parent.parent / "analysis" / "results"
B = 5000
SEED = 20260928


# ---------------------------------------------------------------- statistics
def rankdata(x):
    x = np.asarray(x, dtype=float)
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x)); r[order] = np.arange(1, len(x) + 1)
    return (np.bincount(inv, weights=r) / counts)[inv]


def pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    den = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / den) if den > 0 else float("nan")


def spearman(x, y):
    return pearson(rankdata(x), rankdata(y))


def demean(v, groups, n_iter=1):
    """Subtract group means; with a list of groupings, alternate (two-way FE)."""
    v = v.astype(float).copy()
    for _ in range(n_iter):
        for g in groups:
            sums = np.bincount(g, weights=v); cnt = np.bincount(g)
            v -= (sums / np.maximum(cnt, 1))[g]
    return v


def within_corr(bleu, conf, groups, n_iter=1, singleton_group=None):
    rb, rc = rankdata(bleu), rankdata(conf)
    keep = np.ones(len(rb), bool)
    if singleton_group is not None:  # drop sources contributing a single pair (no within info)
        cnt = np.bincount(singleton_group)
        keep = cnt[singleton_group] >= 2
    rb, rc = rb[keep], rc[keep]
    gs = [np.unique(g[keep], return_inverse=True)[1] for g in groups]
    return pearson(demean(rb, gs, n_iter), demean(rc, gs, n_iter))


def ci(vals):
    vals = np.asarray(vals)
    vals = vals[np.isfinite(vals)]
    return [round(float(np.quantile(vals, 0.025)), 4), round(float(np.quantile(vals, 0.975)), 4)]


def boot_p(vals, obs):
    """Two-sided bootstrap p for H0: stat = 0 (fraction of resamples crossing 0, doubled)."""
    vals = np.asarray(vals); vals = vals[np.isfinite(vals)]
    frac = (vals >= 0).mean() if obs < 0 else (vals <= 0).mean()
    return float(min(1.0, 2 * max(frac, 1 / len(vals))))


# ---------------------------------------------------------------- data
def build_pairs():
    sources_t3 = t3.load_source_texts()
    all_df = pd.read_csv(DATA_ROOT / "data" / "processed" / "rewrite-lf" / "all.csv")
    human = all_df[all_df["model"] == "human"].set_index("original_index")["text"].astype(str).to_dict()
    rows = []
    for model in t3.MODELS:
        print(f"  {model}: fitting probe (track3.get_probe_confidence) ...", flush=True)
        probs, labels, test_df = t3.get_probe_confidence(model)
        ai_probs = probs[labels == 1]
        ai_texts = test_df[test_df["label"] == 1]["text"].astype(str).values
        model_all = all_df[all_df["model"] == model]
        for text, prob in zip(ai_texts, ai_probs):
            match = model_all[model_all["text"].astype(str) == text]
            oidx = int(match.iloc[0]["original_index"]) if len(match) else -1
            if oidx < 0 or oidx not in sources_t3:  # identical inclusion rule to track3
                continue
            rows.append({
                "model": model, "original_index": oidx, "conf": float(prob),
                "bleu_t3": compute_bleu4(sources_t3[oidx], text),
                "edit_t3": edit_similarity(sources_t3[oidx], text),
                "bleu_aligned": compute_bleu4(human[oidx], text),
                "edit_aligned": edit_similarity(human[oidx], text),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- analysis
def analyse(df, bcol, rng):
    bleu = df[bcol].values; conf = df["conf"].values
    src = np.unique(df["original_index"].values, return_inverse=True)[1]
    gen = np.unique(df["model"].values, return_inverse=True)[1]
    gen_names = sorted(df["model"].unique())
    n_src = src.max() + 1

    obs = {
        "pooled_spearman": spearman(bleu, conf),
        "within_source": within_corr(bleu, conf, [src], singleton_group=src),
        "within_generator": within_corr(bleu, conf, [gen]),
        "within_source_and_generator": within_corr(bleu, conf, [src, gen], n_iter=50, singleton_group=src),
    }
    per_gen_obs = {g: spearman(bleu[gen == k], conf[gen == k]) for k, g in enumerate(gen_names)}
    obs["mean_of_per_generator_rho"] = float(np.mean(list(per_gen_obs.values())))

    # source-cluster bootstrap
    members = [np.where(src == s)[0] for s in range(n_src)]
    boots = {k: [] for k in obs}
    for _ in range(B):
        draw = rng.integers(0, n_src, n_src)
        idx = np.concatenate([members[s] for s in draw])
        new_src = np.repeat(np.arange(n_src), [len(members[s]) for s in draw])  # duplicates = distinct clusters
        b, c, g = bleu[idx], conf[idx], gen[idx]
        boots["pooled_spearman"].append(spearman(b, c))
        boots["within_source"].append(within_corr(b, c, [new_src], singleton_group=new_src))
        boots["within_generator"].append(within_corr(b, c, [g]))
        boots["within_source_and_generator"].append(
            within_corr(b, c, [new_src, g], n_iter=20, singleton_group=new_src))
        boots["mean_of_per_generator_rho"].append(np.mean([spearman(b[g == k], c[g == k]) for k in range(len(gen_names))]))

    # generator-cluster bootstrap (10 clusters; coarse sensitivity only)
    gboot = []
    for _ in range(B):
        draw = rng.integers(0, len(gen_names), len(gen_names))
        idx = np.concatenate([np.where(gen == k)[0] for k in draw])
        gboot.append(spearman(bleu[idx], conf[idx]))

    # per-generator bootstrap
    per_gen = {}
    for k, g in enumerate(gen_names):
        ii = np.where(gen == k)[0]
        bs = [spearman(bleu[ii[r]], conf[ii[r]]) for r in rng.integers(0, len(ii), (B, len(ii)))]
        per_gen[g] = {"n_pairs": int(len(ii)), "rho": round(per_gen_obs[g], 4), "ci95": ci(bs),
                      "boot_p_two_sided": round(boot_p(bs, per_gen_obs[g]), 4),
                      "bleu_mean": round(float(bleu[ii].mean()), 4), "conf_mean": round(float(conf[ii].mean()), 4)}

    # between-generator ecological correlation (10 points)
    gm_b = [bleu[gen == k].mean() for k in range(len(gen_names))]
    gm_c = [conf[gen == k].mean() for k in range(len(gen_names))]

    sizes = np.bincount(src)
    out = {
        "n_pairs": int(len(df)), "n_sources": int(n_src), "n_generators": len(gen_names),
        "pairs_per_source": {"min": int(sizes.min()), "median": float(np.median(sizes)), "max": int(sizes.max()),
                             "n_sources_with_>=2_pairs": int((sizes >= 2).sum())},
        "bleu_mean": round(float(bleu.mean()), 4), "bleu_median": round(float(np.median(bleu)), 4),
        "source_cluster_bootstrap": {k: {"estimate": round(obs[k], 4), "ci95": ci(boots[k]),
                                         "boot_p_two_sided": round(boot_p(boots[k], obs[k]), 4)} for k in obs},
        "generator_cluster_bootstrap_pooled_spearman": {"estimate": round(obs["pooled_spearman"], 4), "ci95": ci(gboot),
                                                        "boot_p_two_sided": round(boot_p(gboot, obs["pooled_spearman"]), 4)},
        "per_generator": per_gen,
        "n_generators_negative": int(sum(v < 0 for v in per_gen_obs.values())),
        "n_generators_ci_excludes_0": int(sum(v["ci95"][1] < 0 or v["ci95"][0] > 0 for v in per_gen.values())),
        "between_generator_spearman_of_means": {"rho": round(spearman(np.array(gm_b), np.array(gm_c)), 4), "n": len(gen_names)},
    }
    return out


def per_model_bleu_json_bootstrap(rng):
    """Corroboration: the two correctly aligned GNN-specialist files (bleu_confidence_<model>.json)."""
    out = {}
    for m in ["llama-3.1-8b", "llama-3.3-70b"]:
        d = json.load(open(RESULTS_DIR / f"bleu_confidence_{m}.json"))
        b, c = np.array(d["bleu_values"]), np.array(d["confidence_values"])
        obs = spearman(b, c)
        bs = [spearman(b[r], c[r]) for r in rng.integers(0, len(b), (B, len(b)))]
        out[m] = {"file": f"bleu_confidence_{m}.json", "n_pairs": int(len(b)), "rho": round(obs, 4),
                  "stored_spearman_rho": d["spearman_rho"], "ci95": ci(bs), "bleu4_mean": d["bleu4_mean"],
                  "note": "GNN specialist confidence; one pair per source, so pair bootstrap == source bootstrap"}
    return out


def main():
    rng = np.random.default_rng(SEED)
    stored = json.load(open(RESULTS_DIR / "camouflage_bleu_confidence.json"))
    df = build_pairs()

    # reproduction check vs stored summary
    repro = {"pooled_n": int(len(df)), "stored_pooled_n": stored["pooled"]["n_pairs"],
             "pooled_rho_original_pairing": round(spearman(df["bleu_t3"].values, df["conf"].values), 4),
             "stored_pooled_rho": stored["pooled"]["spearman_bleu_vs_conf"]["rho"], "per_model": {}}
    for m, g in df.groupby("model"):
        repro["per_model"][m] = {"rho": round(spearman(g["bleu_t3"].values, g["conf"].values), 4),
                                 "stored_rho": stored["per_model"][m]["spearman_bleu_vs_conf"]["rho"],
                                 "bleu_mean": round(float(g["bleu_t3"].mean()), 4),
                                 "stored_bleu_mean": stored["per_model"][m]["bleu4_mean"]}
    print(json.dumps(repro, indent=1))
    assert repro["pooled_n"] == repro["stored_pooled_n"]
    assert abs(repro["pooled_rho_original_pairing"] - repro["stored_pooled_rho"]) < 1e-3

    # alignment audit
    src_csv = pd.read_csv(t3.SOURCE_DATA, nrows=2000)
    all_df = pd.read_csv(DATA_ROOT / "data" / "processed" / "rewrite-lf" / "all.csv")
    human = all_df[all_df["model"] == "human"].set_index("original_index")["text"].astype(str)
    norm = lambda t: " ".join(str(t).lower().split())  # noqa: E731
    first_words = lambda t: norm(t).split()[:12]  # noqa: E731
    same = sum(first_words(src_csv.iloc[k]["text"]) == first_words(human[k]) for k in range(2000))
    audit = {
        "source_file": str(t3.SOURCE_DATA),
        "source_file_columns": list(src_csv.columns),
        "generated_flag_counts_first_2000_rows": {str(k): int(v) for k, v in src_csv["generated"].value_counts().items()},
        "rows_whose_first_12_words_match_human_text_at_same_original_index": int(same),
        "bleu_mean_original_pairing": round(float(df["bleu_t3"].mean()), 4),
        "bleu_mean_aligned_pairing": round(float(df["bleu_aligned"].mean()), 4),
        "edit_sim_mean_original_pairing": round(float(df["edit_t3"].mean()), 4),
        "edit_sim_mean_aligned_pairing": round(float(df["edit_aligned"].mean()), 4),
        "explanation": ("track3.load_source_texts() uses dict(enumerate(df['text'])) over a file whose rows alternate "
                        "human/AI; key k therefore points to the human abstract of a DIFFERENT paper (k even) or to an "
                        "old AI rewrite (k odd), not to the source of the rewrite with original_index k."),
    }
    print(json.dumps(audit, indent=1))

    results = {
        "experiment": "camouflage_clustered_revision",
        "reviewers": "R1 & R3: pooled rho over 2,989 pairs ignores clustering by source and generator",
        "inputs": {
            "stored_summary": "analysis/results/camouflage_bleu_confidence.json (pooled.*, per_model.*)",
            "regenerated_from": ["scripts/track3_camouflage.py (get_probe_confidence, load_source_texts, get_test_pairs)",
                                 "scripts/bleu_utils.py", "data/embed_cache/rewrite-lf/*", "data/splits/rewrite-lf/*",
                                 "data/processed/rewrite-lf/all.csv", str(t3.SOURCE_DATA)],
            "corroboration": ["analysis/results/bleu_confidence_llama-3.1-8b.json",
                              "analysis/results/bleu_confidence_llama-3.3-70b.json"],
        },
        "data_root": str(DATA_ROOT),
        "bootstrap": {"B": B, "seed": SEED, "ci": "percentile 95%"},
        "reproduction_of_stored_result": repro,
        "alignment_audit": audit,
        "original_track3_pairing": analyse(df, "bleu_t3", rng),
        "aligned_pairing": analyse(df, "bleu_aligned", rng),
        "aligned_pairing_edit_similarity": analyse(df, "edit_aligned", rng),
        "corroboration_gnn_specialist_files": per_model_bleu_json_bootstrap(rng),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "camouflage_clustered_revision.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    df.to_csv(OUT_DIR / "camouflage_pairs_revision.csv", index=False)
    for k in ["original_track3_pairing", "aligned_pairing", "aligned_pairing_edit_similarity"]:
        r = results[k]
        print(f"\n== {k} ==")
        for kk, v in r["source_cluster_bootstrap"].items():
            print(f"  {kk:<32s} {v['estimate']:+.4f} CI {v['ci95']}  p={v['boot_p_two_sided']}")
        print(f"  generator-cluster CI {r['generator_cluster_bootstrap_pooled_spearman']['ci95']}")
        print(f"  between-generator rho {r['between_generator_spearman_of_means']}")
        for g, v in r["per_generator"].items():
            print(f"    {g:<24s} rho={v['rho']:+.4f} CI {v['ci95']} bleu={v['bleu_mean']}")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
