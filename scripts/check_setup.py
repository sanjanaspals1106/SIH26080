#!/usr/bin/env python
"""Setup check, PRD section 22.2 step 6.

Checks: imports (xarray, cfgrib, xgboost, sklearn, fastapi), xgboost >= 2.0,
a PostgreSQL connection using DATABASE_URL, and node >= 20.
Prints PASS or FAIL for each item. Exits with 1 if any item failed.

Run from the repository root: python scripts/check_setup.py
"""

from __future__ import annotations

import importlib
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
results: list[bool] = []


def report(ok: bool, item: str, detail: str = "") -> None:
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {item}" + (f"  ({detail})" if detail else ""))


def major_minor(version: str) -> tuple[int, int]:
    m = re.match(r"\D*(\d+)\.(\d+)", version)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def read_dotenv_value(name: str) -> str | None:
    """Value from the environment, or from .env in the repository root (simple KEY=VALUE lines)."""
    if os.environ.get(name):
        return os.environ[name]
    env_file = ROOT / ".env"
    if env_file.is_file():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                if key.strip() == name and value.strip():
                    return value.strip().strip("\"'")
    return None


def check_imports() -> None:
    for name in ("xarray", "cfgrib", "xgboost", "sklearn", "fastapi"):
        try:
            module = importlib.import_module(name)
            report(True, f"import {name}", getattr(module, "__version__", ""))
        except Exception as exc:  # any import problem counts as a failure
            report(False, f"import {name}", f"{type(exc).__name__}: {exc}")


def check_xgboost_version() -> None:
    try:
        import xgboost

        version = xgboost.__version__
        report(major_minor(version) >= (2, 0), "xgboost >= 2.0", f"found {version}")
    except Exception as exc:
        report(False, "xgboost >= 2.0", f"xgboost not importable: {type(exc).__name__}")


def check_postgres() -> None:
    url = read_dotenv_value("DATABASE_URL")
    if not url:
        report(False, "PostgreSQL connection", "DATABASE_URL is not set (environment or .env)")
        return
    # SQLAlchemy-style "postgresql+psycopg://" is not understood by psycopg itself.
    url = re.sub(r"^(postgres(?:ql)?)\+\w+://", r"\1://", url)
    try:
        import psycopg

        with psycopg.connect(url, connect_timeout=5) as conn:
            row = conn.execute("SELECT version()").fetchone()
        report(True, "PostgreSQL connection", (row[0] if row else "").split(",")[0])
    except Exception as exc:
        # Do not print the URL: it contains the password.
        report(False, "PostgreSQL connection", f"{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}")


def check_node() -> None:
    try:
        out = subprocess.run(["node", "--version"], capture_output=True, text=True, timeout=10, check=True)
        version = out.stdout.strip()
        report(major_minor(version)[0] >= 20, "node >= 20", f"found {version}")
    except FileNotFoundError:
        report(False, "node >= 20", "node not found on PATH")
    except Exception as exc:
        report(False, "node >= 20", f"{type(exc).__name__}: {exc}")


def main() -> int:
    check_imports()
    check_xgboost_version()
    check_postgres()
    check_node()
    failed = results.count(False)
    print(f"\n{len(results) - failed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
