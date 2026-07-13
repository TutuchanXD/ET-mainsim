from __future__ import annotations

import importlib.metadata
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def _git_output(repo: Path, *args: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def git_provenance(repo: Path | str) -> dict[str, Any]:
    root = Path(repo).resolve()
    commit = _git_output(root, "rev-parse", "HEAD")
    branch = _git_output(root, "branch", "--show-current")
    status = _git_output(root, "status", "--porcelain")
    return {
        "root": str(root),
        "commit": commit,
        "branch": branch or None,
        "dirty": None if status is None else bool(status),
    }


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def collect_provenance(repo_root: Path | str) -> dict[str, Any]:
    import photsim7

    photsim_root = Path(photsim7.__file__).resolve().parents[1]
    return {
        "et_mainsim": {
            **git_provenance(repo_root),
            "version": _distribution_version("et-mainsim") or "0.1.0",
        },
        "photsim7": {
            **git_provenance(photsim_root),
            "version": _distribution_version("photsim7"),
        },
        "runtime": {
            "python": sys.version.split()[0],
            "executable": sys.executable,
            "platform": platform.platform(),
            "hostname": platform.node(),
        },
    }


__all__ = ["collect_provenance", "git_provenance"]
