#!/usr/bin/env python3
"""Install only HF_TOKEN from the private env file as the account-2 Modal secret."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ENV = ROOT.parent / "AEGIS-Huminizer" / ".env"
MODAL_WRAPPER = ROOT / "scripts" / "modal_account2.py"
SECRET_NAME = "aegis-hf-read"


def read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def main() -> int:
    token = read_env(SOURCE_ENV).get("HF_TOKEN", "")
    if not token:
        raise SystemExit("HF_TOKEN is missing from AEGIS-Huminizer/.env")

    file_descriptor, raw_path = tempfile.mkstemp(prefix="aegis-hf-read-", suffix=".json")
    payload_path = Path(raw_path)
    try:
        os.fchmod(file_descriptor, 0o600)
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            json.dump({"HF_TOKEN": token}, handle)
            handle.write("\n")
        command = [
            sys.executable,
            str(MODAL_WRAPPER),
            "secret",
            "create",
            "--force",
            "--from-json",
            str(payload_path),
            SECRET_NAME,
        ]
        completed = subprocess.run(command, cwd=ROOT, check=False)
        if completed.returncode == 0:
            print(f"Configured Modal secret {SECRET_NAME}; HF_TOKEN was not printed.")
        return completed.returncode
    finally:
        payload_path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
