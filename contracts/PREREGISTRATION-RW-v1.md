# CLEAN-RW-v1 preregistration: Tulu rewrite staircase

Date: 2026-09-28

Status: frozen before any rewrite generation. No rewrite text exists.
- The user authorized this extension ("Move 1") on 2026-09-28.
- It is recorded after the CLEAN-FG-v5 128-token release and after the Move 2 headroom rescore (`docs/AMENDMENT-2026-09-28-HEADROOM-RESCORE.md`).

## Question

Under one shared plain rewrite treatment, does linear-probe detectability change across the Tulu Base→SFT→DPO→RLVR post-training staircase? Per stage, does rewrite detectability dissociate from free-generation detectability?

Pythia is excluded; scale is addressed by CLEAN-FG-v5 and Move 2.

## What is mirrored from CLEAN-FG-v5 (unchanged)

The only change is the task: rewrite instead of topic-conditioned free generation. Everything below is taken from `configs/experiment.v1.json` (SHA-256 `8c34041d…`) and `docs/PREREGISTRATION.md`.

**Sources and splits**
- The same frozen source identities:
  - main: `arxiv2k:0000`–`arxiv2k:0499`, with their original 352/69/79 train/validation/test roles;
  - pilot: the same 50 frozen training IDs ≥ 500;
  - smoke: the first 3 pilot IDs.

**Models and runtime**
- The same four Tulu checkpoints and revisions.
- Base keeps the frozen identity `meta-llama/Llama-3.1-8B@d04e592…`. It is served only through the verified acquisition channel and cache-time hash gate of `docs/AMENDMENT-2026-09-28-BASE-WEIGHTS-MIRROR.md`, which runs on every Base launch.
- The same runtime: vLLM 0.21.0, CUDA 12.9 image, Python 3.12, FP16, no quantization, one A10, `max_model_len` 2048, batch 8, `enforce_eager`.

**Decoding and acceptance**
- The same decoding: temperature 1.0, top-p 0.95, unrestricted top-k, repetition penalty 1.0, n = 1, and exactly **one attempt with `min_tokens = max_tokens = 256` native tokens and EOS ignored**.
- The deterministic seed is `stable_seed(run_id, suite, cell, source_id, attempt=0, base_seed=730241)`. Here run_id is `clean-rw-v1`, so seeds differ from V5 by construction.
- The same acceptance rule: a nonempty output with exactly 256 recorded token IDs. There is no rejection, retry, or selection.

**Analysis view**
- The same paired analysis view: the first 128 content tokens under `FacebookAI/roberta-base@e2da8e2f…`, applied symmetrically to each rewrite and its own human source.
- Boundaries use the Amendment A mask method.

**Account ownership, unchanged**
- `account_2_only`: Base, SFT, RLVR.
- `account_1`: DPO.
- Each cell is downloaded through its owner's wrapper and never resumed across accounts.

## What changes

| Item | CLEAN-RW-v1 |
|---|---|
| Run ID / results root | `clean-rw-v1` (results Volume `aegis-clean-staircases-v1`, root `clean-rw-v1/`) |
| Modal app | `aegis-clean-rewrite-v1` |
| Experiment config | `configs/experiment.rw-v1.json`, SHA-256 `b3886efad45b8ebf636f635dd9d077b390a92e78d75edf64648c977a129c9b50`. It is derived from V5 and changes only the fields in this table. |
| Queue | `data/manifests/rewrite_queue.jsonl`, SHA-256 `f58ec327941c9b5ef46cd36639bdaf8b47b64ed798e67d6b6b6970380f4d7a87` (550 rows = the frozen V5 queue rows + `source_text`) |
| Manifest lock | `data/manifests/rewrite-manifest-lock.json` |
| Source text | The source text is the verified human source: `human_sources.jsonl`, taken from even raw rows of the audited alternating human/legacy-AI CSV. `build_rewrite_manifests.py` re-verifies `raw_label = 0`, an even `raw_row_index`, and `sha256(human_text) = human_sha256`. |
| Cells | Tulu Base, SFT, DPO, RLVR only |

## Prompt contract (frozen)

The prompt is a single plain-completion template, identical for all four stages. It has no chat template and no target-length field. It mirrors the V5 one-shot serialization: the same single full exemplar (`arxiv2k:0760`) in the same `EXAMPLE 1 / Topic / Abstract` bundle, and the same `TASK / Topic / … / Abstract:` layout. It adds the full source abstract and a rewrite instruction.

| Contract field | Value |
|---|---|
| Version | `rewrite-plain-full-source-1shot-fixed256-rw-v1` |
| Code | `src/aegis_staircases/rewrite_prompts.py` |
| Template SHA-256 | `082c882c6245bc1294906e11641df2e9b662b633fc512e61e8d18bccd408ee98` |
| Example-template SHA-256 | `cfd3f11122e180f656b5b9ca2fd028acb3d28bfbdaf3e42735e8b51afd55fbb0` (identical to V5) |
| Exemplar-bundle SHA-256 | `40eb65a10a4bc4c4a7d2aac1f43b14081f65b739b873a421e23659c662410708` |
| Exemplar-contract SHA-256 | `abb8f033250eb0c10c212b8672562fb3a921573b756bb545bcac66fa80190bba` |

Exact template (`{examples}` is the V5 exemplar bundle; `{topic}` is the frozen topic; `{source}` is the full verified source abstract):

```text
Rewrite the source abstract below as one new scientific abstract.

Use the example only to understand the requested genre and level of detail. Do not copy the example's wording or claims. Keep the scientific content of the source abstract, but express it in new wording. Return only the new abstract, with no title, labels, notes, or explanation.

{examples}

TASK
Topic: {topic}
Source abstract:
{source}

Abstract:
```

**Context budget (pre-generation estimate):** under the Llama-3.1 tokenizer, rewrite prompts span 771–1,104 tokens. The worst case plus 256 output tokens is 1,360 ≤ 2,048. The per-model Modal context preflight (`inspect_context_budget`) remains authoritative and runs before any GPU work.

## Execution gates (V5-style; every failure is an absolute stop)

1. **Smoke:** 3 sources × 4 stages. Every cell must reach 3/3 accepted sole attempts of exactly 256 tokens and pass strict validation. Base must pass the hash gate.
2. **Pilot:** 50 sources × 4 stages.
   - Every cell must reach 50/50 under strict per-cell validation.
   - The paired-view gate must pass over all 200 rewrite/human pairs: at least 128 RoBERTa content tokens each, 0 failures.
3. **Billing check:** read actual Modal billing (API) for both accounts. The projected main cost plus spend to date must leave at least **$2 reserve** under the $24 combined ceiling, and account 1 must stay within its $6 working ceiling. Otherwise stop and report.
4. **Main:** 500 sources × 4 stages, launched with `--confirm-full`.
5. **Strict per-cell validation:** every cell 500/500; source order `arxiv2k:0000–0499`; split counts 352/69/79; the rendered rewrite prompt hash matches; model revision; account owner.
6. **Global rewrite gate:** all four cells pass, and the paired-view gate passes over all **2,000** rewrite/human pairs.

At most one Tulu GPU job runs per account at a time.

## Evaluation contract (fixed now, before any rewrite output)

The dated evaluator amendment implementing this contract is committed before rewrite scoring, and its lock check must pass on the amended code set.

**Pairing and split**
- Each rewrite is paired with **its own clean human source**, the same source ID and the same human text, so the human view equals the CLEAN-FG-v5 human view.
- The splits are the source-paired 352/69/79 splits.

**Probe (frozen evaluation v1 protocol)**
- A separate detector per cell; RoBERTa mean-pooled content view.
- Train-only `StandardScaler`.
- L2 logistic regression (`lbfgs`, tolerance 1e-4, max_iter 10,000, seed 730241).
- C from {0.01, 0.1, 1, 10, 100} by validation AUROC, smallest C on ties, no refit.
- A single test AUROC on 79 held-out paired sources.
- d′ = √2·Φ⁻¹(AUROC), with AUROC = 1 right-censored (d′ = null).
- 10,000-replicate paired source bootstrap, seed 730241, with the same resample across all cells in a run.

**Primary test**
- The exact one-sided tie-aware Page test on rewrite detectability, alternative **Base < SFT < DPO < RLVR**, over the 79 test-source blocks of sourcewise AUROC contributions *q* (as in `docs/EVALUATION-v1.md`). d′ is monotone in AUROC, so this is the ordered test on rewrite d′.
- α = 0.05.
- Accompanying contrasts: SFT−Base, DPO−SFT, RLVR−DPO, RLVR−Base.
- The V5 Tulu ceiling audit and `WITHHELD_CEILING` rule apply unchanged.

**Verdict labels**, as in V5: `DIRECTIONAL_SUPPORTED`, `DIRECTIONAL_NOT_SUPPORTED`, `WITHHELD_CEILING`.

**View and headroom rule (inherited from Move 2, adapted to four cells)**
- The primary view is **128 tokens**.
- If **any** of the four rewrite cells has unrounded test AUROC exactly 1 at 128 tokens, also score at 64. If any is still at ceiling at 64, also score at 32. Nothing is ever scored below 32.
- The **rewrite headroom view** is the longest evaluated view at which all four rewrite cells are below ceiling. If none exists, the headroom remedy is recorded as exhausted and no headroom verdict is claimed.
- The 128-token results remain the primary instrument-strength findings; headroom results are the saturation remedy.
- The CLEAN-FG-v5 Tulu comparators at 64 and 32 tokens already exist from Move 2 and are reused.

**Secondary preregistered contrast: rewrite-vs-free-generation dissociation**
- The dissociation is computed at 128 tokens and, if one exists, at the rewrite headroom view.
- For each stage *s* ∈ {Base, SFT, DPO, RLVR}:

  **Δ_s = AUROC_RW(s) − AUROC_FG(s)**, which equals the mean over the 79 test sources of (*q*_RW,s,i − *q*_FG,s,i).

- The FG detectors are the CLEAN-FG-v5 Tulu detectors at the same view. The evaluator refits them deterministically from the locked FG views and asserts that their test AUROCs equal the released values exactly.
- Uncertainty comes from one paired source bootstrap over all eight cells (4 RW + 4 FG): 10,000 replicates, seed 730241, the same source resample for every cell, with human and generated members travelling together.
- **Per-stage decision:** at α = 0.05, a stage dissociates if the 95% percentile CI of Δ_s excludes 0.
  - The sign of Δ_s gives the direction: positive means rewrites are *more* detectable than free generation.
  - If both AUROCs for stage *s* equal 1, Δ_s is `WITHHELD_CEILING`.
  - If exactly one equals 1, Δ_s is reported and flagged `CEILING_CENSORED`, and no dissociation decision is made for that stage.
- **Summary:** the difference-in-differences **Δ_RLVR − Δ_Base**, with its bootstrap 95% CI, under the same decision and ceiling rules. It indicates whether post-training changes rewrite and free-generation detectability differently.

**Reporting**
- Every number is reported exactly as produced, whichever direction it points.
- The reports include:
  - the rewrite staircase table with CIs and ceiling flags;
  - the Page test and contrasts;
  - the dissociation table;
  - all gates, billing per account before and after, commits, and deviations.

## Stop conditions (absolute)

Any of the following stops the work:
- any gate failure;
- a hash mismatch, including in the Base hash gate;
- a changed run contract;
- a wrong account owner;
- any need to alter a frozen or locked file without a dated amendment;
- billing that cannot be read, or a projection that cannot keep a $2 reserve under the $24 ceiling.

## Results pointer (appended after execution; the text above is unchanged)

The results are recorded in `docs/reports/WORKER4-HEADROOM-AND-REWRITE-2026-09-28.md` and `docs/evidence/rewrite-2026-09-28/`.
- Every gate passed.
- **128 tokens:** `WITHHELD_CEILING`.
- **32-token rewrite headroom view:** exact Page p = 3.3835174123521534e-05, `DIRECTIONAL_SUPPORTED`. No stage shows a dissociation; Δ_RLVR − Δ_Base = -0.0019227687870533305, CI [-0.03300753084441588, 0.02868130107354594].
