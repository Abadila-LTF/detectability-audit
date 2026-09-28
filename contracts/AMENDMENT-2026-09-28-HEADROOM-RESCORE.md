# CLEAN-FG-v5 amendment: headroom rescoring of the nine gated cells

Date: 2026-09-28

Status: prospective. The user authorized this amendment ("Move 2") on 2026-09-28. It was recorded after the released 128-token evaluation (`docs/evidence/evaluation-2026-09-28/evaluation.json`, commit `2ec4089`) and **before any reduced-view representation, detector fit, or score existed**. No new text is generated. The amendment rescores the existing 4,500 gated endpoints.

## Precedence

- **The released 128-token results remain the primary instrument-strength findings.** They are not replaced, revised, or re-labelled. Those results are Tulu `DIRECTIONAL_SUPPORTED`, Pythia `WITHHELD_CEILING`, with 7/9 cells at test AUROC 1.
- **The headroom results are the saturation remedy specified by this amendment.**
- **Provenance of the remedy:** the V5 preregistration (`docs/PREREGISTRATION.md`) and `docs/EVALUATION-v1.md` specify ceiling handling only as right-censoring and verdict withholding at the 128-token view. They do not specify a reduced-view instrument. The reduced-view remedy is therefore specified here, prospectively relative to its own outputs but after the 128-token release. It must be cited that way.

## Motivation

At the frozen 128-token view, seven of nine cells have unrounded test AUROC exactly 1:
- Tulu DPO and RLVR;
- all five Pythia cells.

Orderings among ceiling cells are therefore undefined.
- The Pythia scale test is `WITHHELD_CEILING`: the 120 admissible orders span ρ ∈ [-1, 1].
- The Tulu Page verdict relies on DPO/RLVR sitting exactly at ceiling while Base/SFT fall short by 1/6,241.

A shorter analysis view gives the frozen probe less evidence per text, which is intended to restore headroom below the ceiling.

## Specification

### Views

- **Headroom views:** the first **64** RoBERTa content tokens, and, if the descent rule triggers, the first **32**.
- **No view shorter than 32 tokens** may be built or scored.
- **Unchanged from 128-token scoring:**
  - the tokenizer and model, `FacebookAI/roberta-base@e2da8e2f811d1448a5b465c236feacd80ffbac7b`;
  - the content rule: `add_special_tokens=False`, then the first *N* content IDs, applied identically to human and generated text;
  - boundaries added after truncation, identified with the Amendment A mask method (`docs/AMENDMENT-2026-09-28-EVALUATOR-BOUNDARY-MASK.md`);
  - mean pooling over the *N* content positions only;
  - CPU, float32, batch size 16.

### Probe and statistics

The frozen evaluation v1 protocol is applied without change:
- the same 352/69/79 source-paired splits;
- train-only `StandardScaler`;
- L2 logistic regression (`lbfgs`, tolerance 1e-4, max 10,000 iterations, seed 730241);
- C chosen from {0.01, 0.1, 1, 10, 100} by validation AUROC, with the smallest C on ties and no refit;
- a single test AUROC;
- d′ = √2·Φ⁻¹(AUROC), right-censored at AUROC 1;
- 10,000-replicate paired source bootstrap, seed 730241, shared across cells;
- the Tulu contrasts;
- the same ceiling adjudication.

### Mechanism

`scripts/headroom_rescore.py`, a new file outside the evaluation lock, runs the **locked** code unchanged: `scripts/validate_main_panel.py`, `scripts/build_roberta_views.py::build_views`, and `scripts/evaluate_main.py::evaluate`.

It injects a derived evaluation configuration into those two modules. The derived configuration equals `configs/evaluation.v1.json` except that `representation.content_tokens` is 64 or 32; `src/aegis_staircases/headroom.py` enforces that no other field differs.

Each view's outputs are written beside a headroom envelope. The envelope records:
- *N*;
- the canonical hash of the derived configuration;
- the parent config file hash;
- the view, lock, manifest, and evaluation artifact hashes.

The embedded `evaluation_config_sha256` fields refer to the parent `configs/evaluation.v1.json` file, from which the derived configuration differs only in `content_tokens`.

### Paired-view sanity gate (per view, before scoring)

The gate covers all 4,500 human/generated pairs (9 cells × 500 frozen main sources). Every human and generated text must have at least *N* content tokens. For every text, the Amendment A boundary mask must recover exactly the *N* content IDs. Any failure stops the rescoring.

### Descent rule (pre-committed)

1. Score the 64-token view.
2. If **three or more of the nine cells** have unrounded test AUROC exactly 1.0 at 64 tokens, also build, gate, and score the 32-token view.
3. Otherwise, stop at 64. No further descent is ever made below 32.

### Primary headroom analysis (pre-committed)

- **Selection:** among the evaluated views {128, 64, and 32 if triggered}, the primary headroom view is the **longest view at which all five Pythia cells have test AUROC < 1.0**.
- **Tulu:** the Tulu ordering is read at that same view.
- **If no evaluated view has all five Pythia cells below ceiling:**
  - there is no primary headroom view;
  - every evaluated view is reported descriptively with its ceiling flags and verdicts as produced;
  - the headroom remedy is recorded as exhausted at the lowest evaluated view;
  - no headroom verdict is claimed for either family.

### Pre-committed tests at the primary headroom view

These are the tests the 128-token suite already runs, applied unchanged:
- **Pythia:** the exact 120-permutation Spearman correlation between test AUROC and log10(parameters), one-sided alternative ρ < 0. The ceiling audit and `WITHHELD_CEILING` rule apply if any cell there is still at ceiling. By the selection rule, none is.
- **Tulu:** the exact one-sided tie-aware Page test for Base < SFT < DPO < RLVR over the 79 test-source blocks of sourcewise AUROC contributions. The Tulu ceiling audit and withholding rule apply unchanged.
- **α = 0.05.**

### Reporting

For every evaluated view, report:
- the full nine-cell table: selected C, validation AUROC, test AUROC, bootstrap 95% CI, d′ with the CI endpoints' d′ transform, and ceiling flags;
- the Tulu contrasts;
- both trend-test outcomes, including ceiling audits.

Results are appended to this file and to `state/STATUS.md`.

## Results (appended 2026-09-28, after execution)

**Execution record**
- Amendment commit: `6f08378`.
- Driver and rules commit: `665d19e`.
- Global gate reused: `546e7e2a78d6a3976e4903e00e30e9f298057d87f5985e4304764e126d3359cf`, re-validated inside every run.
- Evidence: `docs/evidence/headroom-2026-09-28/view-{64,32}/`.
- Numbers are copied unrounded from the produced `evaluation.json` files.

| View | Paired-view gate | Views SHA-256 | Lock SHA-256 | `evaluation.json` SHA-256 |
|---|---|---|---|---|
| 64 | ok: 4,500 pairs, 0 failures | `0d0fd7c427f29dba…` | `3d5ff4296055fcc9…` | `539faa01a182fae310ff8e75f9b3e4337f762e9e29b71b4a6d952ad825464e88` |
| 32 | ok: 4,500 pairs, 0 failures | `806df7dfd80a6372…` | `6060640b3ab0194b…` | `defc1ada1b0ad245321eeb82ac1f90f121dfe86a3836afae66f3cf9e12b3c567` |

**Descent and selection**
- **Descent:** at 64 tokens, four cells had test AUROC exactly 1 (Tulu DPO, Tulu RLVR, Pythia 1.4B, Pythia 2.8B). Four is at least three, so the 32-token view was built and scored.
- **Primary headroom view: 32 tokens.** At 64, two Pythia cells remained at ceiling. At 32, no cell of the nine is at ceiling. The `select` command returned `primary_headroom_view = 32`.

### 64-token view (secondary)

The d′ CI endpoints are the √2·Φ⁻¹ transform of the AUROC bootstrap endpoints; an endpoint of 1 gives +∞ (right-censored).

| Cell | C | Validation AUROC | Test AUROC | AUROC 95% CI | d′ | d′ 95% CI | Ceiling |
|---|---:|---:|---:|---|---:|---|---|
| Tulu Base | 10.0 | 0.9987397605545053 | 0.9937510014420766 | [0.9793302355391764, 1.0] | 3.53236929855913 | [2.885148987095444, +∞] | no |
| Tulu SFT | 10.0 | 0.9995799201848351 | 0.9998397692677455 | [0.9990386156064733, 1.0] | 5.088583877702719 | [4.386762161398643, +∞] | no |
| Tulu DPO | 10.0 | 0.9978996009241756 | 1.0 | [1.0, 1.0] | null (right-censored) | — | **YES** |
| Tulu RLVR | 10.0 | 0.9976895610165931 | 1.0 | [1.0, 1.0] | null (right-censored) | — | **YES** |
| Pythia 70M | 10.0 | 1.0 | 0.9996795385354911 | [0.99855792340971, 1.0] | 4.827669623952761 | [4.21410559662458, +∞] | no |
| Pythia 410M | 0.1 | 1.0 | 0.9995193078032366 | [0.9980772312129467, 1.0] | 4.669153549401031 | [4.08787188028797, +∞] | no |
| Pythia 1.4B | 10.0 | 0.9987397605545053 | 1.0 | [1.0, 1.0] | null (right-censored) | — | **YES** |
| Pythia 2.8B | 100.0 | 0.9957992018483511 | 1.0 | [1.0, 1.0] | null (right-censored) | — | **YES** |
| Pythia 6.9B | 1.0 | 0.9985297206469229 | 0.9996795385354911 | [0.99855792340971, 1.0] | 4.827669623952761 | [4.21410559662458, +∞] | no |

Tulu contrasts (AUROC difference, 95% CI):

| Contrast | Difference | 95% CI |
|---|---:|---|
| SFT − Base | 0.006088767825668917 | [0.0, 0.020189072264060193] |
| DPO − SFT | 0.00016023073225446272 | [0.0, 0.0009613843935266653] |
| RLVR − DPO | 0.0 | [0.0, 0.0] |
| RLVR − Base | 0.00624899855792338 | [0.0, 0.02066976446082358] |

**Tulu Page test**
- Ordinary: L = 2097.5 (null mean 1975.0), exact p = 1.838179139006728e-25, direction increasing.
- Ceiling audit: `ROBUST_TO_CEILING_ORDER`. Ceiling cells are DPO and RLVR, giving 2 admissible orders: L ∈ [2271.0, 2350.0], p ∈ [1.0104427946587314e-97, 1.1723457384111715e-40].
- Verdict: `DIRECTIONAL_SUPPORTED`.

**Pythia Spearman test**
- Ordinary: ρ = 0.3689323936863109, exact one-sided p = 0.7666666666666667 (92/120).
- Ceiling audit: `ROBUST_TO_CEILING_ORDER`. Ceiling cells are 1.4B and 2.8B, giving 2 orders: ρ ∈ [0.30779350562554625, 0.41039134083406165], p ∈ [0.6833333333333333, 0.7833333333333333], direction positive.
- Verdict: `DIRECTIONAL_NOT_SUPPORTED`.

### 32-token view (primary headroom view)

| Cell | C | Validation AUROC | Test AUROC | AUROC 95% CI | d′ | d′ 95% CI | Ceiling |
|---|---:|---:|---:|---|---:|---|---|
| Tulu Base | 10.0 | 0.9731148918294475 | 0.9767665438231052 | [0.9506489344656305, 0.9945521551033488] | 2.815878993579067 | [2.33511903433428, 3.6006252910169954] | no |
| Tulu SFT | 0.1 | 0.9897080445284604 | 0.9753244672328153 | [0.9485659349463227, 0.9940714629065854] | 2.7797019517973482 | [2.306731052230663, 3.558684125647197] | no |
| Tulu DPO | 10.0 | 0.9873976055450535 | 0.9735619291780163 | [0.9435987822464349, 0.9942316936388399] | 2.7378200000078654 | [2.242549251961525, 3.572317902770257] | no |
| Tulu RLVR | 1.0 | 0.9871875656374711 | 0.9855792340970998 | [0.9701970838006729, 0.9958380067296907] | 3.090969880208423 | [2.663951308416306, 3.7315973475501316] | no |
| Pythia 70M | 100.0 | 0.987607645452636 | 0.9945521551033488 | [0.9860599262938632, 1.0] | 3.6006252910169954 | [3.1098113700277734, +∞] | no |
| Pythia 410M | 0.1 | 0.9813064482251628 | 0.9690754686748918 | [0.9455215510334882, 0.9871815414196443] | 2.6408688328289056 | [2.266841867299212, 3.1560550783410575] | no |
| Pythia 1.4B | 1.0 | 0.9655534551564797 | 0.9708380067296908 | [0.947925012017305, 0.9879826950809165] | 2.677466683434837 | [2.298180556791168, 3.1912797413916194] | no |
| Pythia 2.8B | 1.0 | 0.9819365679479101 | 0.9822143887197564 | [0.9665117769588207, 0.9940714629065854] | 2.972387232687201 | [2.590601734472037, 3.558684125647197] | no |
| Pythia 6.9B | 0.1 | 0.9674438143247217 | 0.9786893126101587 | [0.9607394648293542, 0.992469155584041] | 2.8671786131363812 | [2.4880717087169484, 3.4378009301314387] | no |

Tulu contrasts (AUROC difference, 95% CI):

| Contrast | Difference | 95% CI |
|---|---:|---|
| SFT − Base | -0.0014420765902899424 | [-0.032847300112161526, 0.028360839609037014] |
| DPO − SFT | -0.0017625380547989788 | [-0.026922768787053426, 0.0230732254446403] |
| RLVR − DPO | 0.012017304919083482 | [-0.0075308444159589705, 0.0362121454895048] |
| RLVR − Base | 0.008812690273994561 | [-0.012658227848101333, 0.03557122256048717] |

**Pre-committed tests at the primary headroom view (32 tokens)**
- **Tulu:** exact one-sided Page test, Base < SFT < DPO < RLVR, 79 blocks. L = 2054.0 (null mean 1975.0), exact p = **0.00037091825333338474**, direction increasing. Ceiling audit: `NO_AMBIGUOUS_CEILING_ORDER`. Verdict **`DIRECTIONAL_SUPPORTED`**.
- **Pythia:** exact 120-permutation Spearman, alternative ρ < 0. ρ = **-0.1**, exact p = **0.475** (57/120). Ceiling audit: `NO_AMBIGUOUS_CEILING_ORDER`. Verdict **`DIRECTIONAL_NOT_SUPPORTED`**.

### Reading notes (descriptive only)

- **Tulu:** at 32 tokens the mean test AUROCs are not monotone. Base 0.97677 > SFT 0.97532 > DPO 0.97356 < RLVR 0.98558. The Page test is on sourcewise contributions and reaches p = 0.00037; every adjacent and endpoint contrast CI includes 0.
- **Pythia:** at 32 tokens no cell is at ceiling. The exact scale test does not support declining detectability with parameter count (ρ = -0.1, p = 0.475).
- **Precedence:** these are the headroom-remedy results. The 128-token results remain the primary instrument-strength findings.
