#!/usr/bin/env python3
"""Run Modal using only the user's first-account credentials without printing them."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ENV = ROOT.parent / "AEGIS-Huminizer" / ".env"
MODAL_PYTHON = ROOT.parent / "AEGIS-Huminizer" / ".venv" / "bin" / "python"


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
    values = read_env(SOURCE_ENV)
    token_id = values.get("MODAL_TOKEN_ID", "")
    token_secret = values.get("MODAL_TOKEN_SECRET", "")
    if not token_id or not token_secret:
        raise SystemExit("First-account Modal credentials are missing from AEGIS-Huminizer/.env")
    if not MODAL_PYTHON.exists():
        raise SystemExit(f"Modal Python runtime not found: {MODAL_PYTHON}")
    env = os.environ.copy()
    env["MODAL_TOKEN_ID"] = token_id
    env["MODAL_TOKEN_SECRET"] = token_secret
    env["AEGIS_MODAL_ACCOUNT_ALIAS"] = "account_1"
    env["MODAL_ENVIRONMENT"] = "main"
    env.pop("MODAL_TOKEN_ID2", None)
    env.pop("MODAL_TOKEN_SECRET2", None)
    command = [str(MODAL_PYTHON), "-m", "modal", *sys.argv[1:]]
    return subprocess.run(command, cwd=ROOT, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
