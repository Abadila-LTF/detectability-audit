# CLEAN-FG-v5 evaluation contract

Status: prospective and frozen before the nine-cell global main gate. This document specifies code that has been tested only on synthetic inputs. It contains no main-panel detector result or scientific verdict.

## Release barrier

Real representation extraction and real scoring are closed until all nine main cells exist and pass strict validation:

- Tulu: Base, SFT, DPO, RLVR;
- Pythia: 70M, 410M, 1.4B, 2.8B, 6.9B;
- every cell contains the identical ordered IDs `arxiv2k:0000` through `arxiv2k:0499`;
- every endpoint has one accepted attempt with exactly 256 native-model tokens;
- original source roles are exactly 352 train, 69 validation, and 79 test IDs;
- cell ownership, model revision, run/config hash, prompt hash, phase, cohort, source hash, and completion summary all match the frozen contracts.

`scripts/validate_main_panel.py` applies this barrier without computing a representation or score. `scripts/build_roberta_views.py` calls it before importing the model stack or downloading weights. `scripts/evaluate_main.py` calls it again, then requires a representation artifact whose manifest is bound to the same gate hash and evaluation-config hash. An eight-cell panel cannot build real views, fit a detector, or release a test AUROC through these scripts.

## Representation

The representation model and tokenizer are `FacebookAI/roberta-base` at commit `e2da8e2f811d1448a5b465c236feacd80ffbac7b`. For every human and generated text:

1. tokenize with `add_special_tokens=False`;
2. require at least 128 content IDs and retain the first 128;
3. add RoBERTa model boundaries only after truncation (boundary positions are identified on the boundary-added IDs per [`AMENDMENT-2026-09-28-EVALUATOR-BOUNDARY-MASK.md`](AMENDMENT-2026-09-28-EVALUATOR-BOUNDARY-MASK.md));
4. run the frozen model in evaluation mode on CPU;
5. average the last hidden states at the 128 content positions only.

Boundary positions never enter the mean. The human view is computed once per source and paired with each cell's generated view. The output contains 500 human vectors and a 9 × 500 generated tensor, each with 768 float32 dimensions. Its manifest records the model revision, gate hash, config hash, input-text contract hashes, artifact hash, array shapes, and exact runtime versions. It contains no detector scores.

## Evaluation lock

The view builder writes a deterministic lock beside the cache by replacing `.npz` with `.lock.json` (for example, `roberta-views.npz` → `roberta-views.lock.json`). The view manifest points to both the lock file hash and its canonical content hash. The lock contains:

- the experiment and evaluation config hashes;
- hashes of every evaluator/validator script and supporting module named in `configs/evaluation.v1.json`;
- all nine generation contract hashes inside the complete global-gate material;
- the ordered 500 human text hashes and all nine ordered 500-element generated-output hash lists, recomputed from the current manifests/result roots;
- exact Python and distribution versions for NumPy, SciPy, scikit-learn, PyTorch, Transformers, Tokenizers, and Hugging Face Hub;
- the pinned RoBERTa repository/revision plus SHA-256 and byte size for every locally loaded config, tokenizer, and weight file;
- the exact train, validation, and test source-ID lists, counts, and hashes;
- the same exact ordered human/generated input hashes plus their aggregate contract hashes;
- the compressed embedding-cache hash and a semantic dtype/shape/content hash for every array inside it.

No timestamp or machine-specific absolute path enters the canonical lock identity. Before scoring, `scripts/evaluate_main.py` recomputes the gate—including every current ordered human and generated input hash—then verifies the config, code, package, cache, per-array, split-ID, and input-text components against the lock. Thus a result whose text and self-hash were changed without changing its generation contract still closes the lock instead of silently reusing stale embeddings. The complete verified lock is embedded in the final evaluation JSON rather than referenced only by path or gate hash. Consequently, changing evaluation code or an input text after view extraction requires rebuilding and relocking the representations before any test score can be released.

## Detector

A separate binary detector is trained for every cell. Generated text is the positive class. Each source contributes its human and generated member to the same original role.

- `StandardScaler` is fitted only on the 704 training observations (352 paired sources).
- L2 logistic regression uses `lbfgs`, tolerance `1e-4`, and at most 10,000 iterations.
- Candidate `C` values are `0.01, 0.1, 1, 10, 100` in ascending order.
- `C` is chosen by validation AUROC on 138 observations (69 pairs). Exact AUROC ties retain the smallest `C`.
- The selected already-trained model is not refitted on train plus validation.
- Test AUROC is computed once on the 158 observations from the 79 held-out paired sources.

AUROC is the primary cell metric. For a finite AUROC, the reported descriptive separation is

`d′ = sqrt(2) × Phi^-1(AUROC)`.

If the unrounded AUROC is exactly 1, d′ is `null` and right-censored. It is never replaced by a finite cap. The evaluation JSON preserves the 79 human scores, generated scores, and sourcewise contributions for audit.

## Paired source contribution and Tulu Page test

For cell `c`, test-source `i`, AI score `a_ci`, human score `h_ci`, and tie-aware comparison `K(a,h) = 1[a>h] + 0.5×1[a=h]`, define

`q_ci = 0.5 × (mean_j K(a_ci,h_cj) + mean_j K(a_cj,h_ci))`.

The mean of the 79 `q_ci` values equals that cell's test AUROC exactly. The four contributions for source `i` form one Tulu Page-test block.

The primary Tulu alternative is Base < SFT < DPO < RLVR. The ordinary result is a one-sided exact Page test over 79 blocks. Ranks are averaged within ties. For every block, all 4! condition assignments are counted with their tie multiplicities; the block distributions are convolved with an integer-count dynamic program. No asymptotic Page approximation is used. Adjacent AUROC contrasts (SFT−Base, DPO−SFT, RLVR−DPO) and the endpoint contrast (RLVR−Base) accompany the ordered test.

## Pythia scale test

The primary Pythia statistic is Spearman correlation between test detectability and `log10(parameters)` over the frozen 70M, 410M, 1.4B, 2.8B, and 6.9B order. The one-sided alternative is rho < 0. Its p-value enumerates all 5! = 120 assignments exactly, retaining permutation multiplicity if detectability values tie.

## Paired bootstrap

Cell and contrast intervals use 10,000 percentile bootstrap replicates with seed `730241`. Each replicate samples the 79 test source IDs with replacement. The identical sampled source-count vector is applied to every cell, and both members of every human/generated source pair travel together. Detectors and scalers remain fixed; this is a held-out paired-source uncertainty interval, not a training-pipeline bootstrap. The 2.5th and 97.5th percentiles use NumPy's `linear` quantile rule.

## Ceiling adjudication

The ordinary observed Page and Spearman results are always retained. Ceiling adjudication begins only when an unrounded cell AUROC equals exactly 1.

For Pythia, every globally consistent relative ordering of two or more ceiling cells is enumerated. Ceiling cells are ranked above all finite cells, the exact 120-permutation Spearman test is recomputed for every admissible ceiling order, and the rho and p-value ranges are reported.

For Tulu, every globally consistent relative ordering of two or more ceiling cells is likewise enumerated. Because an AUROC-1 cell has source contribution 1 in every block, only the ceiling-cell ties are broken, in the same relative order in all 79 source blocks and above finite cells. Non-ceiling contribution values and their mutual ties are preserved. The exact Page statistic and p-value are recomputed for every admissible order, and their ranges are reported. This is the prospective operationalization of beyond-ceiling ordering; it does not impute a finite d′ or a hidden score magnitude.

If the directional sign or the reject/do-not-reject decision at alpha `0.05` changes across admissible ceiling orders, the directional verdict is `WITHHELD_CEILING`. Otherwise the result is labeled robust to the enumerated ceiling order. A ceiling-affected failure to support the alternative is never reported as evidence of no effect.

## Reproducible execution

Install the separately pinned evaluation environment only when the global panel is ready:

```bash
python3 -m pip install -e '.[evaluation]'
```

The evaluation extra pins NumPy 2.0.2, SciPy 1.14.1, scikit-learn 1.5.2, PyTorch 2.7.1, and Transformers 4.57.3. NumPy is intentionally not a base dependency, so generation/result validators remain lightweight.

The release sequence is:

```bash
python3 scripts/validate_main_panel.py <nine-main-roots> --output <gate-report.json>
python3 scripts/build_roberta_views.py <nine-main-roots> --output <roberta-views.npz>
python3 scripts/evaluate_main.py <nine-main-roots> --views <roberta-views.npz> --output <evaluation.json>
```

The second command creates `<roberta-views.lock.json>` and `<roberta-views.manifest.json>` beside the cache. Do not run the last two commands while Tulu Base is absent. Do not inspect, summarize, plot, or promote test scores until the global gate has passed and the gate-bound evaluation artifact is complete.
