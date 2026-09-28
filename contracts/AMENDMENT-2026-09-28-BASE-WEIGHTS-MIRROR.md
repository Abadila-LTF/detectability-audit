# CLEAN-FG-v5 amendment: Tulu Base weights acquisition channel

Date: 2026-09-28

Status: recorded before any mirror weight download, Modal call, or Base generation. The user confirmed Option 1 (section (g)) on 2026-09-28.

Scope: this amendment changes only the **acquisition channel** for the bytes of the Tulu Base cell. The frozen Base identity does not change: it stays `meta-llama/Llama-3.1-8B` at `d04e592bb4f6aa9cfee91e2e20afa771667e1d4b`. `configs/experiment.v1.json` stays byte-identical (SHA-256 `8c34041d5fe3698e8aac97ef22b500924878d117f57efa134f352e3c94606a24`). No file in the evaluation lock changes.

Everything else is also unchanged:
- source IDs, topic table, and exemplar;
- prompt and decoding;
- the 256-native-token single-attempt protocol and 2,048-token context;
- dtype, backend, and GPU;
- account ownership (`account_2_only`) and the `--confirm-gated-access` launch requirement;
- the paired first-128 RoBERTa-token view, split roles, and evaluation rules.

## (a) Official repository remains gated

- Frozen Base identity: `meta-llama/Llama-3.1-8B` at `d04e592bb4f6aa9cfee91e2e20afa771667e1d4b`.
- The access request has been pending since July 2026. The account-2 cache preflight received HTTP 403 before any Base GPU work (`state/STATUS.md`). The user reports that the repository still returns 403.
- On 2026-09-28, an unauthenticated `resolve` of `config.json` at the pinned revision returned HTTP 401. `/api/models/meta-llama/Llama-3.1-8B/refs` also returned 401 ("Access to model … is restricted"); see `docs/evidence/base-weights-2026-09-28/official-refs-401.json`.
- The Hub **metadata** endpoint for the pinned official revision is publicly readable: `/api/models/meta-llama/Llama-3.1-8B/revision/d04e592…?blobs=true`. It reports the LFS SHA-256 of every shard and the git blob ID of every small file, which makes the direct comparison in (d) possible. The response is saved as `docs/evidence/base-weights-2026-09-28/official-meta-llama-d04e592-metadata.json`. A live re-fetch the same day returned identical sibling entries.

## (b) Acquisition channel

| Field | Value |
|---|---|
| Repository | `NousResearch/Meta-Llama-3.1-8B` |
| Pinned revision | `1f47e50cdbe801ad8a5174156ec3a0655108fb9f` |
| Gated / private | `false` / `false` |
| Last modified | 2024-07-25T20:21:42Z |
| Stored parameters | 8,030,261,248, all BF16 (Hub safetensors metadata; shard-4 header read: 5/5 tensors BF16) |
| Quantization | none |
| Credentials used for download | none (anonymous; no `HF_TOKEN`) |

The repository metadata snapshot is saved as `docs/evidence/base-weights-2026-09-28/nousresearch-1f47e50-metadata.json`.

## (c) Pinned file hashes

- **Shards:** the SHA-256 values are the Hub LFS object IDs at the pinned revision. They are identical to the official LFS SHA-256 values at `d04e592…`.
- **Small files:** these are not stored in LFS. Each was downloaded from `…/resolve/1f47e50…/<file>` and SHA-256'd locally. Its recomputed git blob ID matched the Hub-reported `blobId` at both the channel revision and the official revision. This was re-checked on 2026-09-28 from the saved bytes.

| File | Bytes | SHA-256 | Official git blob ID (SHA-1) |
|---|---:|---|---|
| `model-00001-of-00004.safetensors` | 4,976,698,672 | `f8b9704ab09cdeb097aa4a0a24bca96f906eec36bad63ab495bc21475058601b` | (LFS) |
| `model-00002-of-00004.safetensors` | 4,999,802,720 | `c28b25e7541751056ee126627e007f8d4288319733285e9f7b17b9ff6eb313f0` | (LFS) |
| `model-00003-of-00004.safetensors` | 4,915,916,176 | `d8e9504dd4e4a146d484c52a97584ec14dac92237c46b064934af67a85e7d383` | (LFS) |
| `model-00004-of-00004.safetensors` | 1,168,138,808 | `e4486f35c040f683f7d790354f66c169c109eb9fa0954a4a35d7c458a108405d` | (LFS) |
| `config.json` | 826 | `54acfad3cffe057640904ca8a1e83525e6551c70c7a04c641f5a9eda0bbf64bd` | `cccf055d6f8f210387a248c91dc40e0c7a4bafab` |
| `tokenizer.json` | 9,085,658 | `76e48799b099d43365bd24ccd8ecc5aedac831718da780552f03b0a6eb4412aa` | `f916e71031fa08f3c6ef1680a590c15b52d3cdd9` |
| `tokenizer_config.json` | 50,500 | `8004530facf809ac432114de2a4dcc65fcb632da5ec16d666091aeb6a2ee444a` | `cb9ec25536e44d86778b10509d3e5bdca459a5cf` |
| `special_tokens_map.json` | 73 | `462d91939dbc37178aa5a3eae7068d1990ccc92e09f288cc71f42cdf139d69cc` | `d8cd5076496dbe4be2320312abc10adc43097b81` |
| `generation_config.json` | 185 | `e645194d2dd27c86ed34a5a23f306c1a0fd79a42123c26551b7251c700a3379d` | `fc9506438b7b55383dc04c0816561442324846c3` |
| `model.safetensors.index.json` | 23,950 | `146776fce3f6db1103aa6f249e65ee5544c5923ce6f971b092eee79aa6e5d37b` | `0fd8120f1c6acddc268ebc2583058efaf699a771` |

The frozen `snapshot_download` ignore patterns exclude `original/*` and `*.pth`, so those files are not part of the cached snapshot and are never fetched through this channel.

## (d) Byte-identity cross-checks

Shard LFS SHA-256 at the three pinned revisions:

| Shard | `NousResearch/Meta-Llama-3.1-8B@1f47e50…` | `unsloth/Meta-Llama-3.1-8B@e9a141a2091ea561b96483212645a2a05e6f99fc` | Official `meta-llama/Llama-3.1-8B@d04e592…` (public metadata) |
|---|---|---|---|
| 1/4 | `f8b9704ab09cdeb097aa4a0a24bca96f906eec36bad63ab495bc21475058601b` | `f8b9704ab09cdeb097aa4a0a24bca96f906eec36bad63ab495bc21475058601b` | `f8b9704ab09cdeb097aa4a0a24bca96f906eec36bad63ab495bc21475058601b` |
| 2/4 | `c28b25e7541751056ee126627e007f8d4288319733285e9f7b17b9ff6eb313f0` | `c28b25e7541751056ee126627e007f8d4288319733285e9f7b17b9ff6eb313f0` | `c28b25e7541751056ee126627e007f8d4288319733285e9f7b17b9ff6eb313f0` |
| 3/4 | `d8e9504dd4e4a146d484c52a97584ec14dac92237c46b064934af67a85e7d383` | `d8e9504dd4e4a146d484c52a97584ec14dac92237c46b064934af67a85e7d383` | `d8e9504dd4e4a146d484c52a97584ec14dac92237c46b064934af67a85e7d383` |
| 4/4 | `e4486f35c040f683f7d790354f66c169c109eb9fa0954a4a35d7c458a108405d` | `e4486f35c040f683f7d790354f66c169c109eb9fa0954a4a35d7c458a108405d` | `e4486f35c040f683f7d790354f66c169c109eb9fa0954a4a35d7c458a108405d` |

Shard byte sizes are also identical across all three repositories.

Git blob IDs of the small files the runner loads:

| File | Nous (`1f47e50…`) | Official (`d04e592…`) | unsloth |
|---|---|---|---|
| `config.json` | `cccf055d6f8f210387a248c91dc40e0c7a4bafab` | identical | differs (`2a9e131b…`) |
| `tokenizer.json` | `f916e71031fa08f3c6ef1680a590c15b52d3cdd9` | identical | differs (LFS, 17,209,920 B) |
| `tokenizer_config.json` | `cb9ec25536e44d86778b10509d3e5bdca459a5cf` | identical | differs |
| `special_tokens_map.json` | `d8cd5076496dbe4be2320312abc10adc43097b81` | identical | differs |
| `generation_config.json` | `fc9506438b7b55383dc04c0816561442324846c3` | identical | differs |
| `model.safetensors.index.json` | `0fd8120f1c6acddc268ebc2583058efaf699a771` | identical | identical |

Conclusion:
- The Nous snapshot at `1f47e50…` is content-identical to the frozen official snapshot at `d04e592…` for every file the runner loads. The weights match by SHA-256; the tokenizer, config, and index match by git blob ID. The two repositories differ only in `README.md`, which the runner does not load and the channel does not download.
- **unsloth is disqualified.** It independently confirms the weights, but it ships a different tokenizer and config. Only the Nous repository is eligible as the channel.

## (e) Runtime identical to the eight completed cells

The Base cell uses the unchanged `runtime` block:
- `vllm==0.21.0` on `nvidia/cuda:12.9.0-devel-ubuntu22.04`, Python 3.12, one A10;
- `dtype=float16`, `quantization=null`;
- `max_model_len=2048`, `gpu_memory_utilization=0.9`, batch 8;
- `enforce_eager=True`, `trust_remote_code=False`.

This is the same BF16-stored→FP16-loaded path used for Tulu SFT, DPO, and RLVR. For example, Tulu SFT `f2a0b46…` is stored as BF16 with 8,030,326,784 parameters. Quantized or otherwise reduced-precision variants are forbidden.

## (f) Standing commitment

When Meta grants access to `meta-llama/Llama-3.1-8B`:
1. Download the pinned official revision's four shards and six small files with an authenticated token, directly from the official repository, into a **fresh directory outside the model cache**. The cache already holds blobs keyed by the same identifiers, so a cache-mediated download would skip the bytes.
2. Recompute SHA-256 (and, for the small files, git blob SHA-1) over the downloaded bytes and compare them against section (c).
3. Append the verification result to this file.

A mismatch would invalidate the channel-produced Base cell.

## (g) Confirmed design (Option 1)

### Identity and records

- `configs/experiment.v1.json` is not edited. The Base contract, every Base endpoint record, and the validators continue to use `model_hf_id = meta-llama/Llama-3.1-8B` and `model_revision = d04e592…`.
- This identity is content-true. The loaded bytes are verified at cache time against the official repository's own metadata at that revision.
- This amendment and the committed hash-gate logs are the provenance record for the acquisition channel.

### Code scope

- Only `modal_app.py` changes, plus a new non-locked helper module and new tests. No file in the evaluation-lock `code_files` changes.
- The dispatcher routes the gated Base model to a new account-2-only CPU function that caches through the verified channel.
  - The function has **no secret attached** and downloads anonymously (`token=False`). It refuses to run if `HF_TOKEN` is present in its environment.
  - The existing `aegis-hf-read` secret function stays defined and is still attached only to the official gated-cache path.
- The launch-time requirements are unchanged: `--confirm-gated-access` and `account_2_only`.
- The GPU worker receives no token and still reads only the committed model Volume through `snapshot_download(..., local_files_only=True)`.

### Hash gate

The gate runs at cache time, before any GPU work, on every Base launch (pilot and main).

1. Fetch the official metadata at `d04e592…` from the public API without credentials. Require that:
   - the reported `sha` equals the pinned revision;
   - the official `lfs.sha256`, `blobId`, and `size` of all ten files equal the pinned values in (c).
2. Download exactly the ten files in (c) from `NousResearch/Meta-Llama-3.1-8B@1f47e50…` into a staging directory on the model Volume. Download nothing else.
3. Verify every staged file:
   - Shards: byte size and SHA-256 must equal the official `lfs.sha256`.
   - Small files: byte size must match. The recomputed git blob SHA-1 must equal the official `blobId`, and the SHA-256 must equal the value in (c).
4. Hard-fail on any mismatch, missing file, or extra file.
5. Only after the gate passes, materialize the verified files in the standard Hugging Face cache layout:
   - each file becomes `models--meta-llama--Llama-3.1-8B/blobs/<official etag>`, where the etag is the LFS SHA-256 for shards and the git blob ID for small files;
   - each is linked from `snapshots/d04e592…/<file>`.

   This is where `snapshot_download(repo_id="meta-llama/Llama-3.1-8B", revision="d04e592…", cache_dir="/models", local_files_only=True)` resolves.
6. Re-hash all ten files through the snapshot path.
   - The snapshot directory may contain only these ten files plus `README.md` and `LICENSE`. Those two are the official, ungated, non-loaded files downloaded directly from `meta-llama/Llama-3.1-8B` by the July preflight.
   - `README.md` and `LICENSE` must match their official git blob IDs (`e70490e45ef7a6a61087776f8b109b5541f62922` and `a7c3ca16cee30425ed6ad841a809590f2bcbf290`).
   - Any other entry is a hard failure.
   - `snapshot_download(..., local_files_only=True)` must resolve to that directory.
7. Write the full verification log (pass or fail) to the account-2 results Volume at `clean-fg-v5/tulu/base/_weights_verification/<phase>-<unix-seconds>.json`, beside the Base run roots. Download it through the account-2 wrapper and commit it under `docs/evidence/`.

The dispatcher launches no GPU worker unless the gate reports `PASSED`. Any mismatch is an absolute stop.

## Implementation status

The previous session was blocked at the config-update step on 2026-09-28, before any Modal call. The finding below is kept as history. It is what led to the Option-1 decision in (g).

### Blocking finding (historical): experiment-config hash binding

`scripts/validate_results.py` requires, for every cell, `contract["config_sha256"] == sha256_file("configs/experiment.v1.json")`. It is one of the evaluation-lock `code_files`, and the global gate (`scripts/validate_main_panel.py`) calls it for all nine cells.

All eight completed main contracts bind `config_sha256 = 8c34041d5fe3698e8aac97ef22b500924878d117f57efa134f352e3c94606a24`, which is the current file. The same validator also requires Base records' `model_hf_id`/`model_revision` to equal the config entry.

On a scratch copy, changing only the Base `hf_id`/`revision` made validation of the completed Tulu SFT main cell fail with `RuntimeError: Config hash mismatch`. The same outcome follows for all eight cells.

This left two paths, and neither was covered by the originally planned config-only change:
- Editing the config as planned makes the preregistered nine-cell gate unpassable unless a locked validation file is changed.
- Keeping the config unchanged leaves the official repository identity in the Base contract and records.

Execution stopped for a user decision (`docs/WORKER4-STOP-REPORT-2026-09-28.md`). The user chose Option 1: keep the config and official identity, and change only the acquisition channel under the hash gate in (g).
