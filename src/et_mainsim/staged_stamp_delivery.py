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
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any
from uuid import uuid4

import numpy as np

from .galaxy_stamp_production import (
    STAGED_LOCAL_SCRATCH_DELIVERY_EXECUTION_MODE,
    delivery_execution_mode_from_manifest,
)
from .independent_stamp_production import _read_staged_bundle_coverage
from .stamp_inputs import file_identity
from .time_shards import ContinuousTimeShard, ContinuousTimeShardPlan


class StagedStampShardPublishError(RuntimeError):
    """A scratch shard cannot safely become a formal delivery shard."""


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


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


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


def _expected_members(shard: ContinuousTimeShard) -> tuple[tuple[str, str, int], ...]:
    return (
        ("raw.h5", "raw", 1),
        *(
            (f"coadd_{factor * 10:d}s.h5", "coadd", int(factor))
            for factor in shard.coadd_sizes
        ),
    )


def _load_production_manifest(path: Path) -> Mapping[str, Any]:
    try:
        with path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise StagedStampShardPublishError(
            "production_manifest_path must contain a JSON object"
        ) from error
    if not isinstance(payload, Mapping):
        raise StagedStampShardPublishError(
            "production_manifest_path must contain a JSON object"
        )
    return payload


def _load_run_id(path: Path) -> str:
    payload = _load_production_manifest(path)
    run_id = payload.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise StagedStampShardPublishError("production manifest has no non-empty run_id")
    return run_id


def _production_manifest_content_identity(path: Path) -> dict[str, Any]:
    """Return the exact relocatable content receipt carried by scratch HDF5."""

    identity = file_identity(path)
    return {
        "sha256": identity["sha256"],
        "size_bytes": identity["size_bytes"],
    }


def _require_staged_manifest_and_canonical_formal_root(
    request: StagedStampShardPublishRequest,
) -> Mapping[str, Any]:
    """Reject mixed writer modes and cross-run publication before I/O begins."""

    payload = _load_production_manifest(Path(request.production_manifest_path))
    try:
        execution_mode = delivery_execution_mode_from_manifest(payload)
    except ValueError as error:
        raise StagedStampShardPublishError(
            "production manifest delivery.execution_mode is invalid"
        ) from error
    if execution_mode != STAGED_LOCAL_SCRATCH_DELIVERY_EXECUTION_MODE:
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
    return payload


def _load_frozen_time_shard(
    production_manifest_path: Path,
    *,
    shard_id: int,
) -> ContinuousTimeShard:
    payload = _load_production_manifest(production_manifest_path)
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
    try:
        with time_plan_path.open(encoding="utf-8") as stream:
            time_plan_payload = json.load(stream)
        if not isinstance(time_plan_payload, Mapping):
            raise ValueError("time plan is not an object")
        plan = ContinuousTimeShardPlan.from_manifest_dict(time_plan_payload)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise StagedStampShardPublishError(
            "cannot read the frozen production time-shard plan"
        ) from error
    for shard in plan.shards:
        if shard.shard_id == _strict_source_id(shard_id):
            return shard
    raise StagedStampShardPublishError(
        f"production time-shard plan has no shard_id={int(shard_id)}"
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
        caller_manifest = caller.get("galaxy_production_manifest")
        if not isinstance(caller_manifest, str) or Path(caller_manifest).expanduser().resolve() != request.production_manifest_path:
            raise StagedStampShardPublishError(
                f"staged delivery member {name} does not cite the canonical production manifest"
            )
        caller_manifest_identity = caller.get("galaxy_production_manifest_identity")
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
        identities[name] = source_hash
    return identities


def publish_staged_independent_stamp_shard(
    request: StagedStampShardPublishRequest,
) -> StagedStampShardPublishResult:
    """Copy, verify and atomically publish one completed local scratch shard."""

    if not isinstance(request, StagedStampShardPublishRequest):
        raise TypeError("request must be a StagedStampShardPublishRequest")
    _require_staged_manifest_and_canonical_formal_root(request)
    run_id = _load_run_id(Path(request.production_manifest_path))
    production_manifest_identity = _production_manifest_content_identity(
        Path(request.production_manifest_path)
    )
    source_root = request.staged_shard_root
    _validate_shard_contract(
        source_root,
        request=request,
        run_id=run_id,
        production_manifest_identity=production_manifest_identity,
    )

    final_root = request.final_shard_root
    parent = final_root.parent
    parent.mkdir(parents=True, exist_ok=True)
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
            production_manifest_identity=production_manifest_identity,
        )
        _fsync_directory(staging_root)
        os.replace(staging_root, final_root)
        published = True
        _fsync_directory(parent)
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
