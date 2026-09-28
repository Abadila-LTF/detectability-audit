# CLEAN-FG-v5 preregistration

Status: frozen before any new main-panel generation.

## Question

Under one shared topic-conditioned free-generation treatment, does linear-probe detectability change across the Tulu Base→SFT→DPO→RLVR post-training staircase, and does it change with parameter scale across the five core deduplicated Pythia checkpoints?

Operational failures, fixed-output contract failures, representation-view failures, and detector-ceiling failures are distinct from scientific null results.

## Immutable data contract

- Source CSV SHA-256: `eda175032835cd4d91b118e1d407fd0b29bdea76ad96b9d77e7be8740953fdb0`.
- Human means `generated == 0`; filtering occurs before indexing.
- Topic table SHA-256: `a3529491e4052d46b968d84084e041c657bc6aa03a9d869bc2cc95fdd0dd87a9`.
- Every topic row must match the exact filtered-human text at the same compact index.
- Main identities are fixed at compact-human indices 0–499. Their original train/validation/test assignments are retained.
- Pilot identities are 50 frozen training IDs at or above 500 and can never enter the analysis.
- The single full exemplar is ID 0760. It was selected before any Pythia or main-panel generation from 518 eligible training abstracts outside both panels: retain 260–310-word sources with `primary_common_include=true`, minimize the maximum token count under the pinned representative Tulu and Pythia tokenizers, and break ties by compact index. The committed selection report freezes the tokenizer hashes, candidate count, selected identity/hash, and token counts.
- The machine queue retains source identity/hash, topic identity/hash, and descriptive source-length metadata for later auditing. The rendered generator prompt exposes only the topic and shared exemplar; it never contains target human text or target length.

## Shared intervention

Every Tulu checkpoint and Pythia size receives the exact same plain one-shot prompt serialization with the full selected abstract. No chat template is applied to any primary cell. The prompt requests a new scientific abstract about the frozen topic and contains no target-length field. Every model is capped at the common 2,048-token native context supported by the core Pythia suite; no context extension is allowed.

Sampling is fixed at temperature 1.0, top-p 0.95, unrestricted top-k, repetition penalty 1.0, one sequence, no beam search. Each endpoint seed is a deterministic hash of run ID, suite, cell, source ID, and attempt zero. Models use pinned official commit SHAs, FP16, no quantization, one pinned vLLM/CUDA backend, and an A10 GPU.

Every source gets exactly one generation attempt with `min_tokens=max_tokens=256` and EOS ignored until that fixed native-model-token budget is complete. Acceptance requires a nonempty output with exactly 256 recorded token IDs. There is no length rejection, retry, or model-dependent selection. Raw token IDs, raw text, normalized text, word count, and finish reason are all archived.

## Gates

Infrastructure smoke uses the first three pilot IDs. It must establish exact identities/hashes, successful model loading, finite generation, append-only checkpoint/resume behavior, and a measured cost projection.

The 50-source pilot then runs in every cell. Before main generation, each cell must achieve:

- 50/50 nonempty outputs with exactly 256 native model tokens and one attempt each;
- at least 128 content tokens under pinned `FacebookAI/roberta-base` revision `e2da8e2f811d1448a5b465c236feacd80ffbac7b` for every generated output and its paired human text;
- no source, topic, prompt, revision, dtype, or backend mismatch;
- a full-run projection inside the frozen $24 experiment ceiling.

If a cell fails, the main panel is not opened. Any prompt/decoder amendment gets a new version and reruns the disjoint pilot.

The main suite is evaluable only with 500/500 exact-token outputs in every cell and a valid 128-token paired representation view for every frozen source. Missing or failed endpoints are operational failures and are not handled by post-hoc intersection. Whitespace word counts and natural EOS behavior are descriptive diagnostics, never inclusion criteria.

## Models

Tulu: official Base, SFT, DPO, and final RLVR Llama-3.1-8B checkpoints.

Pythia core: deduplicated 70M, 410M, 1.4B, 2.8B, and 6.9B checkpoints. The 12B checkpoint and any added 160M/1B points require a prospective amendment; they cannot be added after inspecting the five-point result.

## Frozen evaluation direction

Evaluation code will be frozen before opening main test scores. It will symmetrically take the first 128 content tokens under pinned `FacebookAI/roberta-base` revision `e2da8e2f811d1448a5b465c236feacd80ffbac7b`, then use the frozen RoBERTa representation, train-only scaling, validation-selected logistic-regression regularization, source-paired splits, and source-clustered paired bootstrap intervals.

Tulu's primary ordered statistic is a one-sided exact Page test for Base<SFT<DPO<RLVR, accompanied by adjacent and endpoint paired contrasts. Pythia's primary statistic is exact-permutation Spearman correlation between detectability and log10(parameters), with a one-sided negative alternative. AUROC=1 is treated as right-censored rather than converted to a finite exact d-prime. A directional verdict is withheld if admissible ceiling rankings can change it.

## Provenance and crash safety

Each endpoint is keyed by run, suite, cell, cohort, source, and attempt zero. A checkpoint can resume only when the source, human hash, model revision, prompt contract, decoding contract, code commit, and run contract match exactly. Endpoint files are append-only, written atomically, and committed to a new Modal Volume after every generated batch. A crash can require deterministic recomputation of at most the in-flight batch; it cannot mix or silently overwrite accepted history.

## Prospective v1→v5 amendments

V1 exposed Pythia's native 2,048-token context after three valid Tulu-only infrastructure outputs. V2's CPU gate showed that three full exemplars left no safe Pythia output budget and launched no GPU. V3 froze the tokenizer-aware full one-shot exemplar; its word-interval smoke accepted Pythia 3/3 but exhausted Tulu 3/3. V4 raised the token minimum; Tulu accepted 3/3, while Pythia accepted only 2/3 after eight attempts and produced 128–374 words at similar token caps. This demonstrated that word-count rejection sampling itself would create scale-dependent attrition. Before any main-panel generation, V5 therefore replaces word-interval selection with the fixed 256-native-token/128-RoBERTa-token protocol above. Source identities, topic table, exemplar, model list, sampling distribution, dtype, backend, and concurrency are unchanged. V1–V4 outputs are archived and ineligible for V5 analysis or resume.
