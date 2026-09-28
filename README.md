# How Detectable Is Machine-Generated Text? — Release Package

Companion release for:

> Alaktif, A., Chergui, M., Faiq, G., Ammoumou, A.
> *How Detectable Is Machine-Generated Text? A Preregistered Audit of
> Post-Training, Scale, and Task Effects.* Under revision at
> MDPI Technologies (technologies-4588510).

This package contains everything needed to audit and reproduce the
paper's evaluation: the preregistered contracts, the frozen
configuration files, the generation and evaluation code, the derived
per-experiment results (including per-source probe scores), and the
corrected statistical analyses.

## Layout

| Path | Contents |
|---|---|
| `contracts/` | Preregistration documents (free-generation and rewrite protocols), the evaluation contract, and all dated amendments (base-weights provenance with per-file SHA-256 verification against the official repository; evaluator boundary-mask fix; headroom rescore with the pre-committed descent rule; rewrite evaluator) |
| `configs/` | The frozen experiment and evaluation configuration files, byte-identical to those whose hashes every generated endpoint records |
| `code/` | Generation and evaluation source (Modal runners, validity gates, view builder, locked evaluator, exact Page / permutation-Spearman tests, paired source bootstrap) |
| `results/` | `evaluation.json` for each analysis view (128/64/32 tokens) for the free-generation nine-cell panel and the four-cell rewrite panel: per-cell AUROC, d-prime with ceiling flags, bootstrap CIs, trend tests with ceiling audits, dissociation contrasts, and per-source test scores |
| `analysis-corrections/` | Dependence-aware reanalyses of the ten-generator corpus: exact QAP permutation test for the cosine-transfer association (with and without the replication endpoint), source-clustered bootstrap for the camouflage correlation (with per-pair data), and leave-one-out validation of the additive vector model |
| `figures/` | Scripts regenerating the paper's clean-data figures from `results/` |

## Provenance guarantees

- Every generated endpoint records the git commit, configuration hash,
  model revision, and decoding contract it was produced under.
- The base model's weights were verified file-by-file (SHA-256) against
  the official `meta-llama/Llama-3.1-8B` repository metadata at the
  pinned revision before any use; the verification procedure and hashes
  are in `contracts/AMENDMENT-2026-09-28-BASE-WEIGHTS-MIRROR.md`.
- Scoring is gated: the evaluator refuses to run until all cells of a
  panel exist and every validity gate has passed, and its decision
  rules (tests, ceiling handling, verdict criteria) were committed
  before any score existed.

## Licenses

Code: MIT (see `LICENSE`). Configuration and contract documents:
CC-BY-4.0. Derived results contain no full generated texts; per-source
records carry only identifiers, scores, and statistics. Model weights
are not redistributed here; obtain them from their official
repositories under their respective licenses (Llama 3.1 Community
License; Apache-2.0 for Pythia; Ai2 releases for Tulu-3 and OLMo-2).
