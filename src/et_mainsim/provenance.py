from __future__ import annotations

import base64
import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


_INSTALLED_DISTRIBUTION_IDENTITY_SCHEMA_ID = (
    "et_mainsim.installed_distribution_identity.v1"
)
_CANONICAL_DIST_INFO_FILES = frozenset(
    {
        "METADATA",
        "WHEEL",
        "entry_points.txt",
        "top_level.txt",
    }
)


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


def _normalized_distribution_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _is_generated_cache_file(path: Path) -> bool:
    return "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}


def _record_sha256(path: Path) -> tuple[str, str, int] | None:
    if not path.is_file() or path.is_symlink():
        return None
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        size = path.stat().st_size
    except OSError:
        return None
    encoded = base64.urlsafe_b64encode(digest.digest()).rstrip(b"=").decode("ascii")
    return encoded, digest.hexdigest(), int(size)


def _installed_distribution_identity_v1(
    distribution_name: str,
    package_root: Path | str,
) -> dict[str, Any] | None:
    """Return a verified, location-independent identity for one wheel install.

    Python cache files, the mutable installation receipt, and generated console
    wrappers are deliberately outside the identity.  Every importable package
    file plus the immutable wheel metadata must be present in ``RECORD``, carry
    a SHA-256 entry, and still match that entry byte for byte.
    """

    expected_name = _normalized_distribution_name(distribution_name)
    root = Path(package_root).expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        return None
    try:
        distribution = importlib.metadata.distribution(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return None
    metadata_name = distribution.metadata.get("Name")
    version = distribution.version
    if (
        not isinstance(metadata_name, str)
        or _normalized_distribution_name(metadata_name) != expected_name
        or not isinstance(version, str)
        or not version
    ):
        return None

    direct_url_text = distribution.read_text("direct_url.json")
    if direct_url_text is not None:
        try:
            direct_url = json.loads(direct_url_text)
        except json.JSONDecodeError:
            return None
        if not isinstance(direct_url, dict):
            return None
        directory_info = direct_url.get("dir_info")
        if directory_info is not None:
            if not isinstance(directory_info, dict):
                return None
            editable = directory_info.get("editable", False)
            if type(editable) is not bool or editable:
                return None

    try:
        actual_package_files: set[str] = set()
        for candidate in root.rglob("*"):
            if candidate.is_symlink():
                return None
            if not candidate.is_file():
                continue
            relative = candidate.relative_to(root)
            if _is_generated_cache_file(relative):
                continue
            actual_package_files.add(relative.as_posix())
    except OSError:
        return None
    if not actual_package_files:
        return None

    files = distribution.files
    if files is None:
        return None
    record_entries: dict[str, Any] = {}
    package_record_files: set[str] = set()
    for entry in files:
        located = Path(entry.locate())
        try:
            relative = located.resolve().relative_to(root)
        except (OSError, ValueError):
            relative = None
        if relative is not None and not _is_generated_cache_file(relative):
            label = relative.as_posix()
            package_record_files.add(label)
        else:
            entry_path = Path(str(entry))
            if (
                entry_path.name not in _CANONICAL_DIST_INFO_FILES
                or not any(part.endswith(".dist-info") for part in entry_path.parts)
            ):
                continue
            label = f".dist-info/{entry_path.name}"
        if label in record_entries:
            return None
        record_entries[label] = entry

    if package_record_files != actual_package_files or not {
        ".dist-info/METADATA",
        ".dist-info/WHEEL",
    }.issubset(record_entries):
        return None

    canonical_rows: list[dict[str, Any]] = []
    for label, entry in sorted(record_entries.items()):
        recorded_hash = entry.hash
        recorded_size = entry.size
        if (
            recorded_hash is None
            or recorded_hash.mode != "sha256"
            or not isinstance(recorded_hash.value, str)
            or not recorded_hash.value
            or isinstance(recorded_size, bool)
            or not isinstance(recorded_size, int)
            or recorded_size < 0
        ):
            return None
        calculated = _record_sha256(Path(entry.locate()))
        if calculated is None:
            return None
        encoded_hash, hexadecimal_hash, actual_size = calculated
        if encoded_hash != recorded_hash.value or actual_size != recorded_size:
            return None
        canonical_rows.append(
            {
                "path": label,
                "sha256": hexadecimal_hash,
                "size_bytes": actual_size,
            }
        )

    canonical = json.dumps(
        canonical_rows,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return {
        "schema_id": _INSTALLED_DISTRIBUTION_IDENTITY_SCHEMA_ID,
        "name": expected_name,
        "version": version,
        "record_entry_count": len(canonical_rows),
        "record_tree_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def _package_provenance(
    *,
    git_root: Path | str,
    package_root: Path,
    distribution_name: str,
    fallback_version: str | None = None,
) -> dict[str, Any]:
    git = git_provenance(git_root)
    version = _distribution_version(distribution_name) or fallback_version
    result = {**git, "version": version}
    if git["commit"] is None and git["dirty"] is None:
        identity = _installed_distribution_identity_v1(
            distribution_name,
            package_root,
        )
        if identity is not None:
            result["distribution_identity"] = identity
    return result


def collect_provenance(repo_root: Path | str) -> dict[str, Any]:
    import photsim7

    et_mainsim_package_root = Path(__file__).resolve().parent
    photsim_package_root = Path(photsim7.__file__).resolve().parent
    photsim_root = Path(photsim7.__file__).resolve().parents[1]
    return {
        "et_mainsim": _package_provenance(
            git_root=repo_root,
            package_root=et_mainsim_package_root,
            distribution_name="et-mainsim",
            fallback_version="0.1.0",
        ),
        "photsim7": _package_provenance(
            git_root=photsim_root,
            package_root=photsim_package_root,
            distribution_name="photsim7",
        ),
        "runtime": {
            "python": sys.version.split()[0],
            "executable": sys.executable,
            "platform": platform.platform(),
            "hostname": platform.node(),
        },
    }


__all__ = ["collect_provenance", "git_provenance"]
