# CLEAN-FG-v5 evaluator amendment: RoBERTa boundary mask

Date: 2026-09-28

Status: prospective. The user authorized this amendment ("amendment A") on 2026-09-28. It was recorded after the nine-cell global main gate passed, but **before any RoBERTa representation, evaluation lock, detector fit, AUROC, d′, or test score existed**. No outcome was inspected.

## Defect

The first real run of `scripts/build_roberta_views.py` over the nine gated main roots stopped while encoding the human texts, before any array or lock was written. The crash trace is in `docs/evidence/build-roberta-views-failure-2026-09-28.txt`.

- The frozen builder loads `FacebookAI/roberta-base@e2da8e2f…` with `AutoTokenizer(..., use_fast=True)`.
- The frozen `src/aegis_staircases/views.py::add_model_boundaries` called `tokenizer.get_special_tokens_mask(content_ids, already_has_special_tokens=False)`.
- Under the pinned `transformers==4.57.3`, `RobertaTokenizerFast` rejects that call with an `AssertionError`. Only slow tokenizers support `already_has_special_tokens=False`.
- The synthetic tests used a fake tokenizer, so the defect was latent.
- Running the evaluator therefore required changing a locked file. Under the experiment's stop rules, that needed an explicit user decision.

## Change

There is one line in one locked file, `src/aegis_staircases/views.py`:

```diff
-        for value in tokenizer.get_special_tokens_mask(ids, already_has_special_tokens=False)
+        for value in tokenizer.get_special_tokens_mask(model_ids, already_has_special_tokens=True)
```

The mask is now computed on the boundary-added `model_ids`, which the fast tokenizer supports.

Everything else is unchanged:
- the tokenizer and model;
- the content tokenization (`add_special_tokens=False`, first 128 IDs);
- boundary insertion after truncation (`build_inputs_with_special_tokens`);
- pooling over the 128 content positions only;
- the mask-length check and the `recovered == content_ids` guard;
- every other evaluator, validator, configuration, and package pin.

The fake tokenizer in `tests/test_evaluation_contract.py` now models the fast tokenizer. It flags special IDs by membership and rejects `already_has_special_tokens=False`. A new test runs the real pinned tokenizer when its cached snapshot is present.

## Equivalence evidence

The frozen call defined content positions by position: a boundary first, then the content IDs, then a boundary. The amended call defines them by membership in RoBERTa's special IDs `{0, 1, 2, 3, 50264}`. The two definitions agree unless a content ID is itself a special ID. If that happens, the unchanged `recovered == content_ids` guard raises instead of silently changing a view.

Before the amendment was applied, the amended mask was compared with the frozen positional mask using the real pinned fast tokenizer. The comparison covered all 5,000 frozen 128-token content views: 500 human texts plus 9 × 500 generated texts. It used no scores.

- Content special-ID hits: **0**
- Mask mismatches: **0**

A dummy-text check also confirmed that the slow tokenizer's frozen call returns the identical mask.

## Lock consequence

The evaluation lock is built by `scripts/build_roberta_views.py` at view time. It hashes every file in `configs/evaluation.v1.json` → `evaluation_lock.code_files`, so the amended `views.py` hash is bound into the first and only lock. No earlier lock existed. `configs/evaluation.v1.json`, `configs/experiment.v1.json`, and the other eight locked code files are byte-identical to freeze commit `a054ac8`.
