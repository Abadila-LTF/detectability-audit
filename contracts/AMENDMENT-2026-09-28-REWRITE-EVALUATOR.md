# CLEAN-RW-v1 evaluator amendment: extending the frozen evaluator to the rewrite suite

Date: 2026-09-28

Status: prospective. It is committed **before any rewrite scoring**. At the time of this commit, rewrite generation was at the smoke stage; no rewrite representation, detector, or score exists. It implements the evaluation contract fixed in `docs/PREREGISTRATION-RW-v1.md`, which is authoritative if the two ever disagree.

## Why an amendment is needed

The locked CLEAN-FG-v5 evaluator cannot score the rewrite suite unchanged:
- `scripts/build_roberta_views.py` and `scripts/evaluate_main.py` are hard-wired to the nine-cell FG panel (shape `(9, 500, 768)`, the nine-cell gate, and the Pythia statistics).
- `scripts/validate_results.py` is bound to `configs/experiment.v1.json` and the FG prompt.

No locked file is edited. The rewrite suite gets new files that **call the locked library functions**.

## New files (added to the rewrite evaluation lock)

| File | Role |
|---|---|
| `configs/evaluation.rw-v1.json` | Derived from `configs/evaluation.v1.json`: the same representation, classifier, splits, bootstrap, and Tulu statistics. Changes: the global-gate cells become the four rewrite cells, the Pythia block is removed, and headroom and dissociation blocks are added. The lock `code_files` list is the V1 list plus the four rewrite files. |
| `scripts/validate_rewrite_results.py` | Strict per-cell validation that mirrors the locked `validate_result_cell` check for check, plus the four-cell global rewrite gate (`GLOBAL_REWRITE_GATE_PASSED`). |
| `scripts/evaluate_rewrite.py` | View builder, lock writer and verifier, and scorer (described below). |
| `src/aegis_staircases/rewrite_prompts.py` | The frozen rewrite prompt; its hash is in the preregistration. |
| `src/aegis_staircases/headroom.py` | The Move 2 headroom rules module; locked here because the rewrite descent rule inherits from it. |

## Procedure (`scripts/evaluate_rewrite.py`, one view per invocation)

1. **Descent authorization.** 128 tokens is always allowed. 64 requires a 128-token rewrite evaluation with at least one rewrite cell at AUROC 1. 32 requires the same at 64. Nothing is scored below 32.
2. **Gates.**
   - The four-cell global rewrite gate must pass.
   - The CLEAN-FG-v5 nine-cell gate must pass with the released gate hash `546e7e2a…`.
   - A rewrite paired-view gate over all 2,000 rewrite/human pairs at the evaluated view must pass (every text has at least *N* content tokens, and the Amendment A mask recovers *N*).
3. **Views.** The builder uses the locked `_resolve_roberta_snapshot` and `_encode_texts` with the frozen representation (CPU, float32, batch 16, deterministic torch, seed 730241) to produce 500 human and 4 × 500 rewrite embeddings.
   - The human embeddings must **reproduce the locked CLEAN-FG-v5 human view bit-for-bit** at the same view. Otherwise the run stops.
4. **Lock.** The lock records:
   - the four config hashes (V5 experiment and evaluation, rewrite experiment and evaluation);
   - the hashes of every file in the amended `code_files` set;
   - package versions and the RoBERTa artifacts;
   - both gate identities;
   - the FG view artifact hash, the split IDs, and the per-array embedding hashes.

   Before scoring, every component is recomputed and must match. This is the lock check on the amended set.
5. **Free-generation comparators.** The four CLEAN-FG-v5 Tulu detectors are refit from the locked FG views at the same view. The view file hash must equal the released manifest. Each refit's test AUROC must **exactly equal** the released value (`docs/evidence/evaluation-2026-09-28/evaluation.json` at 128; `docs/evidence/headroom-2026-09-28/view-{64,32}/evaluation.json` at 64 and 32). Otherwise the run stops.
6. **Rewrite statistics.** These use the locked functions:
   - `fit_detector` per rewrite cell;
   - `sourcewise_auc_contributions`;
   - `exact_page_test` for Base < SFT < DPO < RLVR;
   - `tulu_ceiling_audit`.

   The verdict logic is copied from `scripts/evaluate_main.py`: `DIRECTIONAL_SUPPORTED`, `DIRECTIONAL_NOT_SUPPORTED`, or `WITHHELD_CEILING`.
7. **Bootstrap and dissociation.**
   - The locked `paired_source_bootstrap` runs over all eight cells (4 RW + 4 FG) with the same resample. Contrasts: the four rewrite staircase contrasts and Δ_s = AUROC_RW(s) − AUROC_FG(s) for each stage.
   - The difference-in-differences Δ_RLVR − Δ_Base needs the bootstrap samples, which the locked function does not return. The evaluator therefore mirrors the locked resampling exactly and **asserts that the mirrored per-cell and contrast intervals equal the locked function's output**. Only then does it compute the difference-in-differences interval from the same samples.
   - Decisions follow the preregistration:
     - a CI excluding 0 is a dissociation, with its direction;
     - AUROC 1 on both sides of a stage is `WITHHELD_CEILING`;
     - AUROC 1 on exactly one side is `CEILING_CENSORED`, with no decision.

## Unchanged

The following are all unchanged, as are every locked file and both V5 configs:
- the probe, C grid, and selection rule;
- the splits;
- d′ and its right-censoring;
- the bootstrap settings;
- the Page test and ceiling audit.
