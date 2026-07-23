"""Atomically publish one complete scratch-rendered independent stamp shard.

The renderer writes HDF5 incrementally.  On a shared filesystem that pattern can
be much slower than writing the complete shard on node-local storage and copying
the five finished HDF5 members once.  This module keeps that optimization out of
the scientific renderer: it accepts only a completed, independently validated
scratch shard, verifies its metadata and byte identity after copying, then makes
the formal shard directory visible with one same-filesystem rename.

It never overwrites a formal product and never changes the HDF5 content produced
by the renderer.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import ctypes
from dataclasses import dataclass
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any
from uuid import uuid4

import numpy as np

from .independent_stamp_production import _read_staged_bundle_coverage
from .time_shards import ContinuousTimeShard, ContinuousTimeShardPlan


class StagedStampShardPublishError(RuntimeError):
    """A scratch shard cannot safely become a formal delivery shard."""


_STAGED_LOCAL_SCRATCH_DELIVERY_EXECUTION_MODE = "staged_local_scratch_v1"
_LEGACY_GALAXY_PRODUCTION_SCHEMA_ID = "et_mainsim.galaxy_stamp_production.v1"
_LEGACY_GALAXY_PRODUCTION_SCHEMA_VERSION = 2
_GENERIC_MANIFEST_PATH_KEY = "production_manifest"
_GENERIC_MANIFEST_IDENTITY_KEY = "production_manifest_identity"
_LEGACY_GALAXY_MANIFEST_PATH_KEY = "galaxy_production_manifest"
_LEGACY_GALAXY_MANIFEST_IDENTITY_KEY = "galaxy_production_manifest_identity"
STAMP_SHARD_PUBLICATION_RECEIPT_FILENAME = "publication_receipt.json"
STAMP_SHARD_PUBLICATION_RECEIPT_SCHEMA_ID = (
    "et_mainsim.stamp_shard_publication_receipt.v1"
)
STAMP_SHARD_PUBLICATION_RECEIPT_SCHEMA_VERSION = 1


_UNSUPPORTED_DIRECTORY_FSYNC_ERRNOS = frozenset(
    {
        errno.EINVAL,
        errno.ENOSYS,
        errno.ENOTSUP,
        errno.EOPNOTSUPP,
    }
)


def _strict_source_id(value: Any) -> int:
    if isinstance(value, bool):
        raise StagedStampShardPublishError("target_source_id must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise StagedStampShardPublishError(
            "target_source_id must be an integer"
        ) from error
    if result < 0 or result > int(np.iinfo(np.int64).max):
        raise StagedStampShardPublishError(
            "target_source_id must be a non-negative signed int64 integer"
        )
    return result


def _normalise_case(value: str) -> str:
    case = str(value).strip()
    if case not in {"injected", "static"}:
        raise StagedStampShardPublishError("case must be injected or static")
    return case


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> bool:
    """Persist a rename, tolerating only filesystems without directory fsync."""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        try:
            os.fsync(descriptor)
        except OSError as error:
            if error.errno in _UNSUPPORTED_DIRECTORY_FSYNC_ERRNOS:
                return False
            raise
    finally:
        os.close(descriptor)
    return True


def _atomic_publish_directory_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish a directory while refusing every existing target."""

    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is None:
        raise StagedStampShardPublishError(
            "atomic no-replace directory publication is unavailable"
        )
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    at_fdcwd = -100
    rename_noreplace = 1
    status = renameat2(
        at_fdcwd,
        os.fsencode(source),
        at_fdcwd,
        os.fsencode(destination),
        rename_noreplace,
    )
    if status == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            error_number,
            f"formal shard delivery already exists: {destination}",
            str(destination),
        )
    if error_number in {
        errno.EINVAL,
        errno.ENOSYS,
        errno.ENOTSUP,
        errno.EOPNOTSUPP,
    }:
        raise StagedStampShardPublishError(
            "filesystem does not support atomic no-replace directory publication"
        )
    raise OSError(
        error_number,
        os.strerror(error_number),
        str(destination),
    )


def _require_no_symlink_directory_components(path: Path, *, run_root: Path) -> None:
    """Reject an existing non-directory or symlink in a formal parent path."""

    try:
        relative = path.relative_to(run_root)
    except ValueError as error:
        raise StagedStampShardPublishError(
            "formal shard parent escapes the production run root"
        ) from error
    current = run_root
    for component in relative.parts:
        current = current / component
        if current.is_symlink():
            raise StagedStampShardPublishError(
                f"formal shard parent contains a symbolic link: {current}"
            )
        if current.exists() and not current.is_dir():
            raise StagedStampShardPublishError(
                f"formal shard parent component is not a directory: {current}"
            )


@dataclass(frozen=True)
class StagedStampShardPublishRequest:
    """Immutable identity for one scratch-to-formal delivery publication."""

    staged_case_root: Path | str
    formal_case_root: Path | str
    production_manifest_path: Path | str
    target_source_id: int
    shard: ContinuousTimeShard
    case: str = "injected"

    def __post_init__(self) -> None:
        if not isinstance(self.shard, ContinuousTimeShard):
            raise TypeError("shard must be a ContinuousTimeShard")
        staged_case_root = Path(self.staged_case_root).expanduser().resolve()
        formal_case_root = Path(self.formal_case_root).expanduser().resolve()
        production_manifest_path = Path(self.production_manifest_path).expanduser().resolve()
        if not staged_case_root.is_dir() or staged_case_root.is_symlink():
            raise StagedStampShardPublishError(
                "staged_case_root must be an existing non-symlink directory"
            )
        if not production_manifest_path.is_file() or production_manifest_path.is_symlink():
            raise StagedStampShardPublishError(
                "production_manifest_path must be an existing non-symlink file"
            )
        object.__setattr__(self, "staged_case_root", staged_case_root)
        object.__setattr__(self, "formal_case_root", formal_case_root)
        object.__setattr__(self, "production_manifest_path", production_manifest_path)
        object.__setattr__(self, "target_source_id", _strict_source_id(self.target_source_id))
        object.__setattr__(self, "case", _normalise_case(self.case))

    @property
    def staged_shard_root(self) -> Path:
        return (
            Path(self.staged_case_root)
            / "stamps"
            / f"target_{self.target_source_id}"
            / "delivery"
            / f"shard_{self.shard.shard_id:05d}"
        )

    @property
    def final_shard_root(self) -> Path:
        return (
            Path(self.formal_case_root)
            / "stamps"
            / f"target_{self.target_source_id}"
            / "delivery"
            / f"shard_{self.shard.shard_id:05d}"
        )


@dataclass(frozen=True)
class StagedStampShardPublishResult:
    """Published formal path and content identities for one shard."""

    final_shard_root: Path
    member_sha256: Mapping[str, str]
    parent_directory_fsync: str


@dataclass(frozen=True)
class _FrozenProductionManifest:
    """One immutable production-manifest byte snapshot for a publication."""

    path: Path
    payload: Mapping[str, Any]
    content_identity: Mapping[str, Any]


@dataclass(frozen=True)
class _FrozenTimeShard:
    """One immutable time-plan byte snapshot and the selected canonical shard."""

    path: Path
    content_identity: Mapping[str, Any]
    shard: ContinuousTimeShard


def _expected_members(shard: ContinuousTimeShard) -> tuple[tuple[str, str, int], ...]:
    return (
        ("raw.h5", "raw", 1),
        *(
            (f"coadd_{factor * 10:d}s.h5", "coadd", int(factor))
            for factor in shard.coadd_sizes
        ),
    )


def _content_identity_from_bytes(raw_bytes: bytes) -> dict[str, Any]:
    """Return the portable file-content receipt for one exact byte snapshot."""

    return {
        "sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "size_bytes": len(raw_bytes),
    }


def _read_frozen_file_bytes(path: Path, *, label: str) -> bytes:
    """Read one file once, so parsing and hashing share the same bytes."""

    try:
        return path.read_bytes()
    except OSError as error:
        raise StagedStampShardPublishError(f"cannot read {label}") from error


def _json_object_from_bytes(raw_bytes: bytes, *, label: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(raw_bytes)
    except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StagedStampShardPublishError(
            f"{label} must contain a JSON object"
        ) from error
    if not isinstance(payload, Mapping):
        raise StagedStampShardPublishError(
            f"{label} must contain a JSON object"
        )
    return payload


def _freeze_production_manifest(path: Path) -> _FrozenProductionManifest:
    raw_bytes = _read_frozen_file_bytes(path, label="production manifest")
    return _FrozenProductionManifest(
        path=path,
        payload=_json_object_from_bytes(raw_bytes, label="production_manifest_path"),
        content_identity=_content_identity_from_bytes(raw_bytes),
    )


def _run_id_from_manifest(payload: Mapping[str, Any]) -> str:
    run_id = payload.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise StagedStampShardPublishError("production manifest has no non-empty run_id")
    return run_id


def _same_file_content_identity(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> bool:
    """Compare the portable content fields of two file-identity receipts."""

    return (
        actual.get("sha256") == expected.get("sha256")
        and actual.get("size_bytes") == expected.get("size_bytes")
    )


def _delivery_execution_mode_from_manifest(payload: Mapping[str, Any]) -> str:
    """Read the generic frozen delivery mode without importing a campaign."""

    delivery = payload.get("delivery")
    if not isinstance(delivery, Mapping):
        raise ValueError("production manifest delivery must be an object")
    execution_mode = delivery.get("execution_mode", "direct_shared_filesystem")
    if not isinstance(execution_mode, str) or execution_mode not in {
        "direct_shared_filesystem",
        _STAGED_LOCAL_SCRATCH_DELIVERY_EXECUTION_MODE,
    }:
        raise ValueError("production manifest delivery.execution_mode is invalid")
    return execution_mode


def _is_legacy_galaxy_production_manifest(payload: Mapping[str, Any]) -> bool:
    """Return whether the manifest may use the retired Galaxy-only keys."""

    return (
        payload.get("schema_id") == _LEGACY_GALAXY_PRODUCTION_SCHEMA_ID
        and payload.get("schema_version")
        == _LEGACY_GALAXY_PRODUCTION_SCHEMA_VERSION
    )


def _caller_production_manifest_reference(
    caller: Mapping[str, Any],
    *,
    production_manifest_payload: Mapping[str, Any],
    member_name: str,
) -> tuple[Any, Any]:
    """Select generic provenance or the exact historical Galaxy fallback.

    A partially populated generic reference is never allowed to fall back.  It
    would otherwise let an HDF5 member silently change provenance dialect when
    one key is missing.  Likewise, Galaxy-named fields cannot authenticate a
    generic campaign.
    """

    generic_path_present = _GENERIC_MANIFEST_PATH_KEY in caller
    generic_identity_present = _GENERIC_MANIFEST_IDENTITY_KEY in caller
    legacy_path_present = _LEGACY_GALAXY_MANIFEST_PATH_KEY in caller
    legacy_identity_present = _LEGACY_GALAXY_MANIFEST_IDENTITY_KEY in caller
    generic_present = generic_path_present or generic_identity_present
    legacy_present = legacy_path_present or legacy_identity_present
    legacy_galaxy_manifest = _is_legacy_galaxy_production_manifest(
        production_manifest_payload
    )

    if generic_present:
        if not generic_path_present or not generic_identity_present:
            raise StagedStampShardPublishError(
                f"staged delivery member {member_name} has an incomplete generic "
                "production manifest reference"
            )
        if legacy_present:
            if not legacy_galaxy_manifest:
                raise StagedStampShardPublishError(
                    f"staged delivery member {member_name} uses Galaxy provenance "
                    "on a generic production manifest"
                )
            if not legacy_path_present or not legacy_identity_present:
                raise StagedStampShardPublishError(
                    f"staged delivery member {member_name} has an incomplete legacy "
                    "Galaxy production manifest reference"
                )
            if (
                caller[_GENERIC_MANIFEST_PATH_KEY]
                != caller[_LEGACY_GALAXY_MANIFEST_PATH_KEY]
                or caller[_GENERIC_MANIFEST_IDENTITY_KEY]
                != caller[_LEGACY_GALAXY_MANIFEST_IDENTITY_KEY]
            ):
                raise StagedStampShardPublishError(
                    f"staged delivery member {member_name} has conflicting generic "
                    "and legacy Galaxy production manifest references"
                )
        return (
            caller[_GENERIC_MANIFEST_PATH_KEY],
            caller[_GENERIC_MANIFEST_IDENTITY_KEY],
        )

    if legacy_present:
        if not legacy_galaxy_manifest:
            raise StagedStampShardPublishError(
                f"staged delivery member {member_name} generic production requires "
                "caller_manifest.production_manifest"
            )
        if not legacy_path_present or not legacy_identity_present:
            raise StagedStampShardPublishError(
                f"staged delivery member {member_name} has an incomplete legacy "
                "Galaxy production manifest reference"
            )
        return (
            caller[_LEGACY_GALAXY_MANIFEST_PATH_KEY],
            caller[_LEGACY_GALAXY_MANIFEST_IDENTITY_KEY],
        )

    raise StagedStampShardPublishError(
        f"staged delivery member {member_name} has no production manifest reference"
    )


def _require_staged_manifest_and_canonical_formal_root(
    request: StagedStampShardPublishRequest,
    payload: Mapping[str, Any],
) -> None:
    """Reject mixed writer modes and cross-run publication before I/O begins."""

    try:
        execution_mode = _delivery_execution_mode_from_manifest(payload)
    except (TypeError, ValueError) as error:
        raise StagedStampShardPublishError(
            "production manifest delivery.execution_mode is invalid"
        ) from error
    if execution_mode != _STAGED_LOCAL_SCRATCH_DELIVERY_EXECUTION_MODE:
        raise StagedStampShardPublishError(
            "staged publisher requires "
            "delivery.execution_mode='staged_local_scratch_v1'"
        )
    expected_formal_case_root = (
        Path(request.production_manifest_path).parent / "cases" / request.case
    ).resolve()
    if Path(request.formal_case_root).resolve() != expected_formal_case_root:
        raise StagedStampShardPublishError(
            "formal_case_root must equal the canonical production "
            "manifest cases/<case> root"
        )


def _frozen_time_shard_from_manifest(
    production_manifest_path: Path,
    payload: Mapping[str, Any],
    *,
    shard_id: int,
) -> _FrozenTimeShard:
    """Load one canonical shard from a single identity-checked plan snapshot."""

    delivery = payload.get("delivery")
    if not isinstance(delivery, Mapping):
        raise StagedStampShardPublishError(
            "production manifest has no delivery time-shard plan"
        )
    relative_path = delivery.get("time_plan_relative_path")
    if not isinstance(relative_path, str) or not relative_path.strip():
        raise StagedStampShardPublishError(
            "production manifest has no delivery.time_plan_relative_path"
        )
    run_root = production_manifest_path.parent.resolve()
    time_plan_path = (run_root / relative_path).resolve()
    try:
        time_plan_path.relative_to(run_root)
    except ValueError as error:
        raise StagedStampShardPublishError(
            "production time-shard plan must stay below the production manifest root"
        ) from error
    time_plan_bytes = _read_frozen_file_bytes(
        time_plan_path,
        label="the frozen production time-shard plan",
    )
    actual_identity = _content_identity_from_bytes(time_plan_bytes)
    expected_identity = delivery.get("time_plan_identity")
    if not isinstance(expected_identity, Mapping) or not _same_file_content_identity(
        actual_identity, expected_identity
    ):
        raise StagedStampShardPublishError(
            "time shard plan identity changed after production preparation"
        )
    try:
        plan = ContinuousTimeShardPlan.from_manifest_dict(
            _json_object_from_bytes(time_plan_bytes, label="frozen production time-shard plan")
        )
    except (TypeError, ValueError) as error:
        raise StagedStampShardPublishError(
            "cannot read the frozen production time-shard plan"
        ) from error
    for shard in plan.shards:
        if shard.shard_id == _strict_source_id(shard_id):
            return _FrozenTimeShard(
                path=time_plan_path,
                content_identity=actual_identity,
                shard=shard,
            )
    raise StagedStampShardPublishError(
        f"production time-shard plan has no shard_id={int(shard_id)}"
    )


def _load_frozen_time_shard(
    production_manifest_path: Path,
    *,
    shard_id: int,
) -> ContinuousTimeShard:
    frozen_manifest = _freeze_production_manifest(production_manifest_path)
    return _frozen_time_shard_from_manifest(
        production_manifest_path,
        frozen_manifest.payload,
        shard_id=shard_id,
    ).shard


def _require_frozen_inputs_unchanged(
    *,
    frozen_manifest: _FrozenProductionManifest,
    frozen_time_shard: _FrozenTimeShard,
) -> None:
    """Reject a mutable input drift before making a formal shard visible."""

    current_manifest_identity = _content_identity_from_bytes(
        _read_frozen_file_bytes(
            frozen_manifest.path,
            label="production manifest",
        )
    )
    current_time_plan_identity = _content_identity_from_bytes(
        _read_frozen_file_bytes(
            frozen_time_shard.path,
            label="the frozen production time-shard plan",
        )
    )
    if not (
        _same_file_content_identity(
            current_manifest_identity,
            frozen_manifest.content_identity,
        )
        and _same_file_content_identity(
            current_time_plan_identity,
            frozen_time_shard.content_identity,
        )
    ):
        raise StagedStampShardPublishError(
            "frozen publication inputs changed before formal publication"
        )


def _require_exact_members(root: Path, *, shard: ContinuousTimeShard) -> None:
    if not root.is_dir() or root.is_symlink():
        raise StagedStampShardPublishError(
            f"completed staged shard directory is unavailable: {root}"
        )
    expected = {name for name, _, _ in _expected_members(shard)}
    actual = {entry.name for entry in root.iterdir()}
    if actual != expected:
        raise StagedStampShardPublishError(
            "staged shard members differ from the exact raw/coadd delivery set"
        )
    for name in expected:
        member = root / name
        if not member.is_file() or member.is_symlink():
            raise StagedStampShardPublishError(
                f"staged shard member is not a regular file: {member}"
            )


def _validate_bundle_coverage(
    path: Path,
    *,
    shard: ContinuousTimeShard,
    product_kind: str,
    coadd_factor: int,
) -> Mapping[str, Any]:
    try:
        (
            actual_product_kind,
            actual_coadd_factor,
            manifest,
            raw_starts,
            raw_stops,
            time_starts,
            exposure_seconds,
        ) = _read_staged_bundle_coverage(path)
    except RuntimeError as error:
        raise StagedStampShardPublishError(
            f"cannot validate staged delivery member {path.name}"
        ) from error
    if actual_product_kind != product_kind or actual_coadd_factor != coadd_factor:
        raise StagedStampShardPublishError(
            f"staged delivery member {path.name} has the wrong product identity"
        )
    if manifest.get("time_shard") != shard.to_manifest_dict():
        raise StagedStampShardPublishError(
            f"staged delivery member {path.name} declares the wrong time shard"
        )
    expected_starts = np.arange(
        shard.raw_start_index,
        shard.raw_stop_index,
        coadd_factor,
        dtype=np.int64,
    )
    expected_stops = expected_starts + coadd_factor
    expected_times = expected_starts.astype(np.float64) * shard.raw_exposure_seconds
    expected_exposures = np.full(
        expected_starts.shape,
        shard.raw_exposure_seconds * coadd_factor,
        dtype=np.float64,
    )
    if (
        not np.array_equal(raw_starts, expected_starts)
        or not np.array_equal(raw_stops, expected_stops)
        or not np.array_equal(time_starts, expected_times)
        or not np.array_equal(exposure_seconds, expected_exposures)
    ):
        raise StagedStampShardPublishError(
            f"staged delivery member {path.name} does not exactly cover its time shard"
        )
    return manifest


def _validate_shard_contract(
    root: Path,
    *,
    request: StagedStampShardPublishRequest,
    run_id: str,
    production_manifest_payload: Mapping[str, Any],
    production_manifest_identity: Mapping[str, Any],
) -> None:
    _require_exact_members(root, shard=request.shard)
    canonical_caller: Mapping[str, Any] | None = None
    for name, product_kind, coadd_factor in _expected_members(request.shard):
        manifest = _validate_bundle_coverage(
            root / name,
            shard=request.shard,
            product_kind=product_kind,
            coadd_factor=coadd_factor,
        )
        if int(manifest.get("target_source_id_int64", -1)) != request.target_source_id:
            raise StagedStampShardPublishError(
                f"staged delivery member {name} has the wrong target source"
            )
        caller = manifest.get("caller_manifest")
        if not isinstance(caller, Mapping):
            raise StagedStampShardPublishError(
                f"staged delivery member {name} has no caller manifest"
            )
        if caller.get("run_id") != run_id:
            raise StagedStampShardPublishError(
                f"staged delivery member {name} run_id does not match production manifest"
            )
        if caller.get("case") != request.case:
            raise StagedStampShardPublishError(
                f"staged delivery member {name} case does not match publication request"
            )
        caller_manifest, caller_manifest_identity = (
            _caller_production_manifest_reference(
                caller,
                production_manifest_payload=production_manifest_payload,
                member_name=name,
            )
        )
        if (
            not isinstance(caller_manifest, str)
            or Path(caller_manifest).expanduser().resolve()
            != request.production_manifest_path
        ):
            raise StagedStampShardPublishError(
                f"staged delivery member {name} does not cite the canonical production manifest"
            )
        if (
            not isinstance(caller_manifest_identity, Mapping)
            or dict(caller_manifest_identity) != dict(production_manifest_identity)
        ):
            raise StagedStampShardPublishError(
                f"staged delivery member {name} production manifest content identity "
                "does not match publication input"
            )
        caller_dict = dict(caller)
        if canonical_caller is None:
            canonical_caller = caller_dict
        elif caller_dict != canonical_caller:
            raise StagedStampShardPublishError(
                "raw and coadd members do not have the same caller manifest"
            )


def _copy_members_and_verify(
    source_root: Path,
    destination_root: Path,
    *,
    shard: ContinuousTimeShard,
) -> dict[str, str]:
    identities: dict[str, str] = {}
    for name, _, _ in _expected_members(shard):
        source = source_root / name
        destination = destination_root / name
        shutil.copy2(source, destination)
        _fsync_file(destination)
        source_hash = _sha256(source)
        destination_hash = _sha256(destination)
        if source_hash != destination_hash:
            raise StagedStampShardPublishError(
                f"byte identity mismatch after copying {name} to formal storage"
            )
        # The formal receipt records the destination digest.  Equality above
        # separately proves that it is byte-identical to the scratch source.
        identities[name] = destination_hash
    return identities


def _full_shard_identity(shard: ContinuousTimeShard) -> dict[str, Any]:
    """Serialize every field that determines one independent time shard."""

    return {
        "shard_id": int(shard.shard_id),
        "raw_start_index": int(shard.raw_start_index),
        "raw_stop_index": int(shard.raw_stop_index),
        "coadd_sizes": [int(value) for value in shard.coadd_sizes],
        "raw_exposure_seconds": float(shard.raw_exposure_seconds),
    }


def _sha256_hex(value: str, *, label: str) -> str:
    prefix = "sha256:"
    if not isinstance(value, str) or not value.startswith(prefix):
        raise StagedStampShardPublishError(f"{label} has no SHA-256 identity")
    digest = value[len(prefix) :]
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise StagedStampShardPublishError(f"{label} has an invalid SHA-256 identity")
    return digest


def _publication_receipt_payload(
    *,
    request: StagedStampShardPublishRequest,
    staging_root: Path,
    run_id: str,
    production_manifest_identity: Mapping[str, Any],
    member_sha256: Mapping[str, str],
) -> dict[str, Any]:
    """Build one relocation-safe receipt for the future final shard paths."""

    run_root = Path(request.production_manifest_path).parent.resolve()
    expected_names = tuple(name for name, _, _ in _expected_members(request.shard))
    if set(member_sha256) != set(expected_names):
        raise StagedStampShardPublishError(
            "copied member identities differ from the exact raw/coadd delivery set"
        )
    manifest_sha256 = production_manifest_identity.get("sha256")
    manifest_size = production_manifest_identity.get("size_bytes")
    if (
        not isinstance(manifest_sha256, str)
        or len(manifest_sha256) != 64
        or any(character not in "0123456789abcdef" for character in manifest_sha256)
        or isinstance(manifest_size, bool)
        or not isinstance(manifest_size, int)
        or manifest_size < 0
    ):
        raise StagedStampShardPublishError(
            "production manifest has an invalid frozen content identity"
        )

    members: dict[str, dict[str, Any]] = {}
    for name in expected_names:
        staged_member = staging_root / name
        if not staged_member.is_file() or staged_member.is_symlink():
            raise StagedStampShardPublishError(
                f"copied shard member is not a regular file: {staged_member}"
            )
        final_member = request.final_shard_root / name
        try:
            relative_path = final_member.relative_to(run_root).as_posix()
        except ValueError as error:
            raise StagedStampShardPublishError(
                "formal shard member path escapes the production run root"
            ) from error
        members[name] = {
            "path_relative_to_run_root": relative_path,
            "size_bytes": int(staged_member.stat().st_size),
            "sha256": _sha256_hex(
                member_sha256[name],
                label=f"copied shard member {name}",
            ),
        }

    manifest_relative_path = Path(request.production_manifest_path).relative_to(
        run_root
    ).as_posix()
    return {
        "schema_id": STAMP_SHARD_PUBLICATION_RECEIPT_SCHEMA_ID,
        "schema_version": STAMP_SHARD_PUBLICATION_RECEIPT_SCHEMA_VERSION,
        "complete": True,
        "run_id": run_id,
        "case": request.case,
        "target_source_id_int64": request.target_source_id,
        "shard": _full_shard_identity(request.shard),
        "production_manifest": {
            "path_relative_to_run_root": manifest_relative_path,
            "size_bytes": manifest_size,
            "sha256": manifest_sha256,
        },
        "members": members,
    }


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(payload),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _validate_publication_receipt_readback(
    path: Path,
    *,
    expected_payload: Mapping[str, Any],
) -> None:
    """Require the durable receipt bytes to equal the canonical intended bytes."""

    if not path.is_file() or path.is_symlink():
        raise StagedStampShardPublishError(
            "publication receipt is not a regular non-symlink file"
        )
    actual_bytes = _read_frozen_file_bytes(path, label="publication receipt")
    expected_bytes = _canonical_json_bytes(expected_payload)
    if actual_bytes != expected_bytes:
        raise StagedStampShardPublishError(
            "publication receipt readback differs from the intended receipt"
        )
    parsed = _json_object_from_bytes(actual_bytes, label="publication receipt")
    if dict(parsed) != dict(expected_payload):
        raise StagedStampShardPublishError(
            "publication receipt readback has the wrong payload"
        )


def _write_and_validate_publication_receipt(
    staging_root: Path,
    *,
    payload: Mapping[str, Any],
) -> Path:
    """Durably write and strictly read back a receipt inside hidden staging."""

    path = staging_root / STAMP_SHARD_PUBLICATION_RECEIPT_FILENAME
    raw_bytes = _canonical_json_bytes(payload)
    try:
        with path.open("xb") as stream:
            stream.write(raw_bytes)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as error:
        raise StagedStampShardPublishError(
            "publication receipt unexpectedly already exists in staging"
        ) from error
    _validate_publication_receipt_readback(
        path,
        expected_payload=payload,
    )
    return path


def publish_staged_independent_stamp_shard(
    request: StagedStampShardPublishRequest,
) -> StagedStampShardPublishResult:
    """Copy, verify and atomically publish one completed local scratch shard."""

    if not isinstance(request, StagedStampShardPublishRequest):
        raise TypeError("request must be a StagedStampShardPublishRequest")
    production_manifest_path = Path(request.production_manifest_path)
    frozen_manifest = _freeze_production_manifest(production_manifest_path)
    _require_staged_manifest_and_canonical_formal_root(
        request,
        frozen_manifest.payload,
    )
    frozen_time_shard = _frozen_time_shard_from_manifest(
        production_manifest_path,
        frozen_manifest.payload,
        shard_id=request.shard.shard_id,
    )
    if request.shard != frozen_time_shard.shard:
        raise StagedStampShardPublishError(
            "request shard does not match the frozen production time-shard plan"
        )
    run_id = _run_id_from_manifest(frozen_manifest.payload)
    production_manifest_identity = frozen_manifest.content_identity
    source_root = request.staged_shard_root
    _validate_shard_contract(
        source_root,
        request=request,
        run_id=run_id,
        production_manifest_payload=frozen_manifest.payload,
        production_manifest_identity=production_manifest_identity,
    )

    final_root = request.final_shard_root
    parent = final_root.parent
    run_root = production_manifest_path.parent.resolve()
    _require_no_symlink_directory_components(parent, run_root=run_root)
    parent.mkdir(parents=True, exist_ok=True)
    _require_no_symlink_directory_components(parent, run_root=run_root)
    if final_root.exists() or final_root.is_symlink():
        raise FileExistsError(f"formal shard delivery already exists: {final_root}")
    lock = parent / f".{final_root.name}.staged-publish.lock"
    try:
        lock.mkdir()
    except FileExistsError as error:
        raise StagedStampShardPublishError(
            f"formal shard publication is already locked: {final_root}"
        ) from error

    staging_root = parent / f".{final_root.name}.{uuid4().hex}.incoming"
    published = False
    try:
        if final_root.exists() or final_root.is_symlink():
            raise FileExistsError(f"formal shard delivery already exists: {final_root}")
        staging_root.mkdir()
        identities = _copy_members_and_verify(
            source_root,
            staging_root,
            shard=request.shard,
        )
        _validate_shard_contract(
            staging_root,
            request=request,
            run_id=run_id,
            production_manifest_payload=frozen_manifest.payload,
            production_manifest_identity=production_manifest_identity,
        )
        receipt_payload = _publication_receipt_payload(
            request=request,
            staging_root=staging_root,
            run_id=run_id,
            production_manifest_identity=production_manifest_identity,
            member_sha256=identities,
        )
        _write_and_validate_publication_receipt(
            staging_root,
            payload=receipt_payload,
        )
        _fsync_directory(staging_root)
        _require_frozen_inputs_unchanged(
            frozen_manifest=frozen_manifest,
            frozen_time_shard=frozen_time_shard,
        )
        _require_no_symlink_directory_components(parent, run_root=run_root)
        _atomic_publish_directory_noreplace(staging_root, final_root)
        published = True
        parent_directory_fsync = (
            "completed" if _fsync_directory(parent) else "unsupported"
        )
    except BaseException:
        if not published:
            if staging_root.exists() or staging_root.is_symlink():
                shutil.rmtree(staging_root)
            lock.rmdir()
        raise
    lock.rmdir()
    return StagedStampShardPublishResult(
        final_shard_root=final_root,
        member_sha256=dict(identities),
        parent_directory_fsync=parent_directory_fsync,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    publish = subparsers.add_parser(
        "publish",
        help="copy one completed scratch shard and atomically publish it",
    )
    publish.add_argument("--staged-case-root", required=True)
    publish.add_argument("--formal-case-root", required=True)
    publish.add_argument("--production-manifest", required=True)
    publish.add_argument("--source-id", type=int, required=True)
    publish.add_argument("--shard-id", type=int, required=True)
    publish.add_argument("--case", choices=("injected", "static"), required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the fail-closed scratch publication command-line interface."""

    args = _parser().parse_args(argv)
    production_manifest = Path(args.production_manifest).expanduser().resolve()
    shard = _load_frozen_time_shard(
        production_manifest,
        shard_id=args.shard_id,
    )
    result = publish_staged_independent_stamp_shard(
        StagedStampShardPublishRequest(
            staged_case_root=args.staged_case_root,
            formal_case_root=args.formal_case_root,
            production_manifest_path=production_manifest,
            target_source_id=args.source_id,
            shard=shard,
            case=args.case,
        )
    )
    print(
        json.dumps(
            {
                "final_shard_root": str(result.final_shard_root),
                "member_sha256": dict(result.member_sha256),
                "parent_directory_fsync": result.parent_directory_fsync,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "StagedStampShardPublishError",
    "StagedStampShardPublishRequest",
    "StagedStampShardPublishResult",
    "main",
    "publish_staged_independent_stamp_shard",
]


if __name__ == "__main__":  # pragma: no cover - CLI invocation guard.
    raise SystemExit(main())
