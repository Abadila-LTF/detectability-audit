"""Verified acquisition channel for gated model weights (2026-09-28 Base amendment).

The frozen model identity is never changed. Bytes are fetched anonymously from a
content-identical public repository, verified against the official repository's
public metadata at the pinned revision, and only then materialized in the standard
Hugging Face cache layout under the official identity. See
docs/AMENDMENT-2026-09-28-BASE-WEIGHTS-MIRROR.md.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

SCHEMA_VERSION = "aegis.clean-staircase.weights-channel-gate.v1"
HUB_ENDPOINT = "https://huggingface.co"
CHUNK_BYTES = 16 * 1024 * 1024
TOKEN_ENV_KEYS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN")

# Every file the runner loads. `lfs_sha256` marks a shard; `blob_id` marks a small file.
LLAMA_31_8B_FILES: Dict[str, Dict[str, Any]] = {
    "model-00001-of-00004.safetensors": {
        "size": 4976698672,
        "sha256": "f8b9704ab09cdeb097aa4a0a24bca96f906eec36bad63ab495bc21475058601b",
        "lfs": True,
    },
    "model-00002-of-00004.safetensors": {
        "size": 4999802720,
        "sha256": "c28b25e7541751056ee126627e007f8d4288319733285e9f7b17b9ff6eb313f0",
        "lfs": True,
    },
    "model-00003-of-00004.safetensors": {
        "size": 4915916176,
        "sha256": "d8e9504dd4e4a146d484c52a97584ec14dac92237c46b064934af67a85e7d383",
        "lfs": True,
    },
    "model-00004-of-00004.safetensors": {
        "size": 1168138808,
        "sha256": "e4486f35c040f683f7d790354f66c169c109eb9fa0954a4a35d7c458a108405d",
        "lfs": True,
    },
    "config.json": {
        "size": 826,
        "sha256": "54acfad3cffe057640904ca8a1e83525e6551c70c7a04c641f5a9eda0bbf64bd",
        "blob_id": "cccf055d6f8f210387a248c91dc40e0c7a4bafab",
        "lfs": False,
    },
    "tokenizer.json": {
        "size": 9085658,
        "sha256": "76e48799b099d43365bd24ccd8ecc5aedac831718da780552f03b0a6eb4412aa",
        "blob_id": "f916e71031fa08f3c6ef1680a590c15b52d3cdd9",
        "lfs": False,
    },
    "tokenizer_config.json": {
        "size": 50500,
        "sha256": "8004530facf809ac432114de2a4dcc65fcb632da5ec16d666091aeb6a2ee444a",
        "blob_id": "cb9ec25536e44d86778b10509d3e5bdca459a5cf",
        "lfs": False,
    },
    "special_tokens_map.json": {
        "size": 73,
        "sha256": "462d91939dbc37178aa5a3eae7068d1990ccc92e09f288cc71f42cdf139d69cc",
        "blob_id": "d8cd5076496dbe4be2320312abc10adc43097b81",
        "lfs": False,
    },
    "generation_config.json": {
        "size": 185,
        "sha256": "e645194d2dd27c86ed34a5a23f306c1a0fd79a42123c26551b7251c700a3379d",
        "blob_id": "fc9506438b7b55383dc04c0816561442324846c3",
        "lfs": False,
    },
    "model.safetensors.index.json": {
        "size": 23950,
        "sha256": "146776fce3f6db1103aa6f249e65ee5544c5923ce6f971b092eee79aa6e5d37b",
        "blob_id": "0fd8120f1c6acddc268ebc2583058efaf699a771",
        "lfs": False,
    },
}

# Official, ungated, non-loaded files already cached by the July authenticated preflight.
LLAMA_31_8B_PREEXISTING_OFFICIAL: Dict[str, Dict[str, Any]] = {
    "README.md": {"size": 40883, "blob_id": "e70490e45ef7a6a61087776f8b109b5541f62922"},
    "LICENSE": {"size": 7627, "blob_id": "a7c3ca16cee30425ed6ad841a809590f2bcbf290"},
}

VERIFIED_CHANNELS: Dict[tuple, Dict[str, Any]] = {
    ("meta-llama/Llama-3.1-8B", "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"): {
        "channel_hf_id": "NousResearch/Meta-Llama-3.1-8B",
        "channel_revision": "1f47e50cdbe801ad8a5174156ec3a0655108fb9f",
        "amendment": "docs/AMENDMENT-2026-09-28-BASE-WEIGHTS-MIRROR.md",
        "files": LLAMA_31_8B_FILES,
        "preexisting_official": LLAMA_31_8B_PREEXISTING_OFFICIAL,
    }
}


class HashGateError(RuntimeError):
    """Raised on any gate failure; carries the verification log written so far."""

    def __init__(self, message: str, log: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.log = log or {}


def verified_channel_for(model: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    return VERIFIED_CHANNELS.get((model.get("hf_id"), model.get("revision")))


def official_metadata_url(hf_id: str, revision: str) -> str:
    return f"{HUB_ENDPOINT}/api/models/{hf_id}/revision/{revision}?blobs=true"


def fetch_official_metadata(hf_id: str, revision: str, timeout: int = 120) -> bytes:
    """Fetch public Hub metadata with no Authorization header."""
    request = urllib.request.Request(
        official_metadata_url(hf_id, revision),
        headers={"User-Agent": "aegis-clean-staircases-hash-gate"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def git_blob_sha1(path: Path) -> str:
    digest = hashlib.sha1(b"blob %d\0" % path.stat().st_size)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_stream(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def etag_for(spec: Mapping[str, Any]) -> str:
    """Hugging Face cache blob name: LFS SHA-256 for shards, git blob ID otherwise."""
    return spec["sha256"] if spec["lfs"] else spec["blob_id"]


def check_official_metadata(
    metadata: Mapping[str, Any], revision: str, channel: Mapping[str, Any]
) -> list:
    """Return failures comparing public official metadata with the pinned table."""
    failures = []
    if metadata.get("sha") != revision:
        failures.append(f"official metadata sha {metadata.get('sha')!r} != pinned {revision}")
    siblings = {item.get("rfilename"): item for item in metadata.get("siblings", [])}
    expected = dict(channel["files"])
    expected.update(
        {name: {**spec, "lfs": False} for name, spec in channel["preexisting_official"].items()}
    )
    for name, spec in expected.items():
        sibling = siblings.get(name)
        if sibling is None:
            failures.append(f"{name}: absent from official metadata")
            continue
        if sibling.get("size") != spec["size"]:
            failures.append(f"{name}: official size {sibling.get('size')} != pinned {spec['size']}")
        lfs = sibling.get("lfs") or {}
        if spec["lfs"]:
            if lfs.get("sha256") != spec["sha256"]:
                failures.append(f"{name}: official lfs.sha256 {lfs.get('sha256')} != pinned {spec['sha256']}")
        else:
            if lfs:
                failures.append(f"{name}: official file is LFS but pinned as a git blob")
            if sibling.get("blobId") != spec["blob_id"]:
                failures.append(f"{name}: official blobId {sibling.get('blobId')} != pinned {spec['blob_id']}")
    return failures


def _entry_names(directory: Path, ignore: frozenset) -> list:
    return sorted(entry.name for entry in directory.iterdir() if entry.name not in ignore)


def verify_directory(
    directory: Path,
    channel: Mapping[str, Any],
    *,
    allow_preexisting_official: bool,
    ignore: frozenset = frozenset(),
    blobs_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Hash every pinned file in `directory` and require an exact file set."""
    files = channel["files"]
    allowed_extra = channel["preexisting_official"] if allow_preexisting_official else {}
    failures = []
    entries = []
    present = _entry_names(directory, ignore) if directory.is_dir() else []
    for name in sorted(set(files) - set(present)):
        failures.append(f"{name}: missing")
    for name in present:
        if name not in files and name not in allowed_extra:
            failures.append(f"{name}: unexpected extra entry")
    for name in present:
        spec = files.get(name) or allowed_extra.get(name)
        if spec is None:
            continue
        path = directory / name
        entry: Dict[str, Any] = {"file": name, "is_symlink": path.is_symlink()}
        if path.is_symlink():
            target = os.readlink(path)
            entry["symlink_target"] = target
            if blobs_dir is not None and path.resolve().parent != blobs_dir.resolve():
                failures.append(f"{name}: symlink resolves outside the repository blobs directory")
        if not path.is_file():
            failures.append(f"{name}: not a regular file")
            entries.append(entry)
            continue
        entry["size"] = path.stat().st_size
        if entry["size"] != spec["size"]:
            failures.append(f"{name}: size {entry['size']} != {spec['size']}")
        if name in files and spec["lfs"]:
            entry["sha256"] = sha256_stream(path)
            entry["expected_sha256"] = spec["sha256"]
            entry["match"] = entry["sha256"] == spec["sha256"] and entry["size"] == spec["size"]
        elif name in files:
            entry["sha256"] = sha256_stream(path)
            entry["git_blob_sha1"] = git_blob_sha1(path)
            entry["expected_sha256"] = spec["sha256"]
            entry["expected_git_blob_sha1"] = spec["blob_id"]
            entry["match"] = (
                entry["sha256"] == spec["sha256"]
                and entry["git_blob_sha1"] == spec["blob_id"]
                and entry["size"] == spec["size"]
            )
        else:
            entry["role"] = "preexisting_official_not_loaded"
            entry["git_blob_sha1"] = git_blob_sha1(path)
            entry["expected_git_blob_sha1"] = spec["blob_id"]
            entry["match"] = entry["git_blob_sha1"] == spec["blob_id"] and entry["size"] == spec["size"]
        if not entry["match"]:
            failures.append(f"{name}: hash mismatch")
        entries.append(entry)
    return {
        "directory": str(directory),
        "entries_present": present,
        "ignored_entries": sorted(ignore),
        "files": entries,
        "failures": failures,
        "passed": not failures,
    }


def _link_snapshot_file(snapshot_dir: Path, name: str, etag: str) -> str:
    link = snapshot_dir / name
    target = f"../../blobs/{etag}"
    if link.is_symlink():
        if os.readlink(link) != target:
            raise RuntimeError(f"Existing snapshot link {link} points to {os.readlink(link)}, not {target}")
        return "kept_existing_link"
    if link.exists():
        raise RuntimeError(f"Refusing to replace existing non-link snapshot entry {link}")
    os.symlink(target, link)
    return "linked"


def materialize(staging_dir: Path, storage_folder: Path, revision: str, channel: Mapping[str, Any]) -> list:
    """Move verified staged files into blobs/<etag> and link them from snapshots/<revision>."""
    blobs_dir = storage_folder / "blobs"
    snapshot_dir = storage_folder / "snapshots" / revision
    blobs_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    actions = []
    for name, spec in sorted(channel["files"].items()):
        etag = etag_for(spec)
        blob = blobs_dir / etag
        if blob.exists():
            blob_action = "kept_existing_blob"
        else:
            os.replace(staging_dir / name, blob)
            blob_action = "moved"
        actions.append(
            {"file": name, "blob": f"blobs/{etag}", "blob_action": blob_action,
             "link_action": _link_snapshot_file(snapshot_dir, name, etag)}
        )
    return actions


def run_hash_gate(
    *,
    model: Mapping[str, Any],
    cache_dir: Path,
    staging_root: Path,
    fetch_metadata: Callable[[str, str], bytes],
    download: Callable[[str, str, str, Path], Any],
    resolve_snapshot: Callable[[], str],
    environ: Mapping[str, str],
    context: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Run the cache-time hash gate. Returns a PASSED log or raises HashGateError."""
    started = time.time()
    hf_id, revision = model["hf_id"], model["revision"]
    channel = verified_channel_for(model)
    log: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "FAILED",
        "context": dict(context or {}),
        "official_hf_id": hf_id,
        "official_revision": revision,
        "started_unix_seconds": started,
    }

    def fail(message: str) -> None:
        log["failure"] = message
        log["elapsed_seconds"] = round(time.time() - started, 3)
        raise HashGateError(message, log)

    if channel is None:
        fail(f"No verified acquisition channel is pinned for {hf_id}@{revision}")
    log.update(
        {
            "channel_hf_id": channel["channel_hf_id"],
            "channel_revision": channel["channel_revision"],
            "channel_credentials": "none",
            "amendment": channel["amendment"],
            "pinned_files": channel["files"],
            "preexisting_official_allowed": channel["preexisting_official"],
        }
    )
    present_tokens = sorted(key for key in TOKEN_ENV_KEYS if environ.get(key))
    if present_tokens:
        fail(f"Hugging Face token variables must be absent from the channel cache: {present_tokens}")

    try:
        raw = fetch_metadata(hf_id, revision)
        metadata = json.loads(raw.decode("utf-8"))
    except Exception as error:  # noqa: BLE001 - every fetch failure is a gate failure
        fail(f"Official metadata fetch failed: {error!r}")
    metadata_failures = check_official_metadata(metadata, revision, channel)
    log["official_metadata"] = {
        "url": official_metadata_url(hf_id, revision),
        "response_sha256": hashlib.sha256(raw).hexdigest(),
        "sha": metadata.get("sha"),
        "gated": metadata.get("gated"),
        "siblings": sorted(
            (
                {
                    "rfilename": item.get("rfilename"),
                    "size": item.get("size"),
                    "blobId": item.get("blobId"),
                    "lfs_sha256": (item.get("lfs") or {}).get("sha256"),
                }
                for item in metadata.get("siblings", [])
            ),
            key=lambda item: item["rfilename"],
        ),
        "failures": metadata_failures,
    }
    if metadata_failures:
        fail("Official metadata does not match the pinned table")

    storage_folder = cache_dir / ("models--" + hf_id.replace("/", "--"))
    snapshot_dir = storage_folder / "snapshots" / revision
    already = snapshot_dir.is_dir() and all(
        (snapshot_dir / name).is_symlink() or (snapshot_dir / name).exists() for name in channel["files"]
    )
    if already:
        log["staging"] = {"action": "skipped_snapshot_already_materialized"}
        log["materialization"] = []
    else:
        staging_dir = staging_root / channel["channel_revision"]
        staging_dir.mkdir(parents=True, exist_ok=True)
        download_seconds = time.time()
        for name in sorted(channel["files"]):
            try:
                download(channel["channel_hf_id"], channel["channel_revision"], name, staging_dir)
            except Exception as error:  # noqa: BLE001 - every download failure is a gate failure
                fail(f"Channel download failed for {name}: {error!r}")
        staged = verify_directory(
            staging_dir, channel, allow_preexisting_official=False, ignore=frozenset({".cache"})
        )
        staged["download_seconds"] = round(time.time() - download_seconds, 3)
        log["staging"] = staged
        if not staged["passed"]:
            fail("Staged channel files failed the hash gate; nothing was materialized")
        try:
            log["materialization"] = materialize(staging_dir, storage_folder, revision, channel)
        except Exception as error:  # noqa: BLE001 - never proceed after a partial materialization
            fail(f"Materialization failed: {error!r}")
        shutil.rmtree(staging_dir)

    final = verify_directory(
        snapshot_dir,
        channel,
        allow_preexisting_official=True,
        blobs_dir=storage_folder / "blobs",
    )
    log["snapshot"] = final
    if not final["passed"]:
        fail("Materialized snapshot failed the hash gate")
    try:
        resolved = resolve_snapshot()
    except Exception as error:  # noqa: BLE001
        fail(f"snapshot_download(local_files_only=True) failed: {error!r}")
    log["resolved_snapshot_path"] = resolved
    if Path(resolved).resolve() != snapshot_dir.resolve():
        fail(f"snapshot_download(local_files_only=True) resolved {resolved}, not {snapshot_dir}")
    log["status"] = "PASSED"
    log["elapsed_seconds"] = round(time.time() - started, 3)
    return log
