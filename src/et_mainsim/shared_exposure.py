"""Fail-closed contracts for one parent exposure shared by many stamps.

The objects in this module are deliberately small JSON control-plane
artifacts.  They do not claim scientific lineage for an upstream parent
frame: file hashes here are storage/resume guards only.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
from threading import RLock
from typing import Any, Callable

import numpy as np

from photsim7.simulation_services import FullFrameSourcePixelGeometry
from photsim7.stamp_products import (
    STAMP_DETECTOR_COORDINATE_CONVENTION,
    StampWindow,
)


TARGET_PLAN_SCHEMA_ID = "et_mainsim.shared_exposure_target_plan.v1"
TARGET_PLAN_SCHEMA_VERSION = 1
FRAME_COMPLETION_SCHEMA_ID = "et_mainsim.shared_exposure_frame_completion.v1"
FRAME_COMPLETION_SCHEMA_VERSION = 1

_SOURCE_GEOMETRY_SCHEMA_ID = "photsim7.full_frame_source_pixel_geometry.v1"
_SOURCE_GEOMETRY_SCHEMA_VERSION = 1
_POSITION_BASIS = "static_base_renderer_positions"
_POSITION_CONTRACTS = {
    (
        "Detector Xpix Shifted",
        "Detector Ypix Shifted",
    ): "direct_shifted_frame_grid",
    ("x0", "y0"): "centered_offsets_to_frame_grid",
}
_GEOMETRY_TRUTH_MODES = frozenset(
    {"physical_et_focalplane", "reference_field_nonphysical"}
)
_PARENT_GUARD_SCOPE = "storage_resume_guard_only"
_NOT_SCIENTIFIC_LINEAGE = "not_scientific_lineage"
_UPSTREAM_NEGATIVE_CLAIMS = {
    "independent_stamp_simulation": False,
    "lineage_claimed": False,
    "parent_content_hash_status": "not_available",
    "parent_content_identity_status": "not_available",
    "target_association_status": "not_verified_against_parent_truth",
    "truth_transfer_status": "not_transferred_source_axis",
    "zero_new_rng_draws": True,
}


class SharedExposureContractError(ValueError):
    """Raised when a shared-exposure control artifact is malformed."""


class SharedExposurePublicationError(RuntimeError):
    """Raised when immutable publication would overwrite different bytes."""


class SharedExposureReferenceDriftError(RuntimeError):
    """Raised when a marker's referenced storage is missing or has drifted."""


class SharedExposureArrayMismatchError(AssertionError):
    """Raised when two arrays are not exactly identical in the contract sense."""


_StorageGuardIdentity = tuple[int, int, int, int, int]


def _storage_guard_identity(stat_result: os.stat_result) -> _StorageGuardIdentity:
    return (
        stat_result.st_dev,
        stat_result.st_ino,
        stat_result.st_size,
        stat_result.st_mtime_ns,
        stat_result.st_ctime_ns,
    )


class SharedExposureStorageGuardCache:
    """Caller-owned cache for immutable-file storage-guard hashes.

    Entries are keyed by device, inode, size, modification time, and change
    time.  Every lookup restats the path before using an entry, so replacement
    or mutation invalidates the cached digest.  The cache deliberately has no
    process-global instance: callers choose the exact batch lifetime.
    """

    def __init__(self) -> None:
        self._entries: dict[_StorageGuardIdentity, tuple[int, str]] = {}
        self._lock = RLock()

    def clear(self) -> None:
        """Discard all cached storage guards."""

        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def _guard_for(self, path: Path, *, field_name: str) -> tuple[int, str]:
        try:
            before = path.stat()
        except FileNotFoundError as exc:
            raise SharedExposureReferenceDriftError(
                f"{field_name} reference is missing: {path}"
            ) from exc
        except OSError as exc:
            raise SharedExposureReferenceDriftError(
                f"{field_name} reference could not be inspected and may have "
                f"drifted: {path}"
            ) from exc
        identity = _storage_guard_identity(before)

        # Hash misses while holding the lock.  This makes a shared caller-owned
        # cache deduplicate even concurrent requests for the same large shard.
        with self._lock:
            cached = self._entries.get(identity)
            if cached is not None:
                try:
                    after = path.stat()
                except FileNotFoundError as exc:
                    raise SharedExposureReferenceDriftError(
                        f"{field_name} reference is missing: {path}"
                    ) from exc
                except OSError as exc:
                    raise SharedExposureReferenceDriftError(
                        f"{field_name} reference could not be inspected and may "
                        f"have drifted: {path}"
                    ) from exc
                if _storage_guard_identity(after) == identity:
                    return cached

            hashed_identity, size, digest = _hash_file_storage_guard_uncached(
                path,
                field_name=field_name,
            )
            result = (size, digest)
            self._entries[hashed_identity] = result
            return result


def _require_exact_keys(
    value: Any,
    expected: set[str] | frozenset[str],
    *,
    field_name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SharedExposureContractError(f"{field_name} must be an object")
    keys = set(value)
    if keys != set(expected) or not all(isinstance(key, str) for key in value):
        missing = sorted(set(expected) - keys)
        extra = sorted(keys - set(expected), key=repr)
        raise SharedExposureContractError(
            f"{field_name} fields disagree; missing={missing!r}, extra={extra!r}"
        )
    return value


def _strict_int(
    value: Any,
    *,
    field_name: str,
    minimum: int | None = None,
) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise SharedExposureContractError(f"{field_name} must be an integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise SharedExposureContractError(f"{field_name} must be at least {minimum}")
    return result


def _strict_source_id(value: Any, *, field_name: str) -> int:
    try:
        result = _strict_int(value, field_name=field_name)
    except SharedExposureContractError as exc:
        raise SharedExposureContractError(
            f"{field_name} must be a signed 64-bit integer"
        ) from exc
    limits = np.iinfo(np.int64)
    if result < limits.min or result > limits.max:
        raise SharedExposureContractError(
            f"{field_name} must be a signed 64-bit integer"
        )
    return result


def _strict_text(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise SharedExposureContractError(
            f"{field_name} must be a non-empty canonical string"
        )
    return value


def _strict_float(value: Any, *, field_name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (float, np.floating)
    ):
        raise SharedExposureContractError(
            f"{field_name} must be a finite floating-point number"
        )
    result = float(value)
    if not np.isfinite(result):
        raise SharedExposureContractError(
            f"{field_name} must be a finite floating-point number"
        )
    return result


def _strict_bool(value: Any, *, field_name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise SharedExposureContractError(f"{field_name} must be boolean")
    return bool(value)


def _strict_sha256(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        raise SharedExposureContractError(
            f"{field_name} must contain 64 lowercase hexadecimal characters"
        )
    try:
        int(value, 16)
    except ValueError as exc:
        raise SharedExposureContractError(
            f"{field_name} must contain 64 lowercase hexadecimal characters"
        ) from exc
    return value


def _strict_shape(value: Any, *, field_name: str) -> tuple[int, int]:
    if isinstance(value, (str, bytes)):
        raise SharedExposureContractError(
            f"{field_name} must contain exactly two positive integers"
        )
    try:
        items = tuple(value)
    except TypeError as exc:
        raise SharedExposureContractError(
            f"{field_name} must contain exactly two positive integers"
        ) from exc
    if len(items) != 2:
        raise SharedExposureContractError(
            f"{field_name} must contain exactly two positive integers"
        )
    if any(
        isinstance(item, (bool, np.bool_))
        or not isinstance(item, (int, np.integer))
        or int(item) <= 0
        for item in items
    ):
        raise SharedExposureContractError(
            f"{field_name} must contain exactly two positive integers"
        )
    return int(items[0]), int(items[1])


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SharedExposureContractError(
            "shared-exposure artifact must be canonical JSON data"
        ) from exc


def _content_digest(payload: Mapping[str, Any]) -> str:
    without_hash = dict(payload)
    without_hash.pop("content_sha256", None)
    return sha256(_canonical_json_bytes(without_hash)).hexdigest()


def _attach_content_digest(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["content_sha256"] = _content_digest(result)
    return json.loads(_canonical_json_bytes(result))


def _validate_content_digest(payload: Mapping[str, Any]) -> None:
    recorded = _strict_sha256(
        payload.get("content_sha256"), field_name="content_sha256"
    )
    expected = _content_digest(payload)
    if recorded != expected:
        raise SharedExposureContractError(
            "content_sha256 does not match canonical artifact content"
        )


def _canonical_copy(payload: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(_canonical_json_bytes(payload))


def _read_json_object(
    path: Path, *, artifact_name: str
) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
    except FileNotFoundError as exc:
        raise SharedExposureContractError(
            f"{artifact_name} is missing: {path}"
        ) from exc
    except OSError as exc:
        raise SharedExposureContractError(
            f"failed to read {artifact_name}: {path}"
        ) from exc
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SharedExposureContractError(
            f"{artifact_name} must contain one UTF-8 JSON object"
        ) from exc
    if not isinstance(payload, dict):
        raise SharedExposureContractError(f"{artifact_name} must be a JSON object")
    return payload, raw


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_canonical_no_replace(
    path: Path,
    payload: Mapping[str, Any],
    *,
    validate: Callable[[Mapping[str, Any]], dict[str, Any]],
) -> Path:
    """Publish canonical bytes without ever replacing an existing inode."""

    path = Path(path)
    validated = validate(payload)
    canonical_bytes = _canonical_json_bytes(validated)
    proposed_bytes = _canonical_json_bytes(payload)
    if proposed_bytes != canonical_bytes:
        raise SharedExposureContractError(
            "artifact is not in its canonical validated representation"
        )

    try:
        existing_bytes = path.read_bytes()
    except FileNotFoundError:
        existing_bytes = None
    except OSError as exc:
        raise SharedExposurePublicationError(
            f"failed to inspect publication destination {path}"
        ) from exc
    if existing_bytes is not None and existing_bytes != proposed_bytes:
        raise SharedExposurePublicationError(
            f"immutable publication conflict at {path}"
        )
    if existing_bytes is not None:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError:
            try:
                winner_bytes = path.read_bytes()
            except OSError as exc:
                raise SharedExposurePublicationError(
                    f"publication race left unreadable destination {path}"
                ) from exc
            if winner_bytes != canonical_bytes:
                raise SharedExposurePublicationError(
                    f"immutable publication conflict at {path}"
                )
        except OSError as exc:
            raise SharedExposurePublicationError(
                f"failed to publish immutable artifact at {path}"
            ) from exc
        _fsync_directory(path.parent)
        return path
    finally:
        temporary_path.unlink(missing_ok=True)
        try:
            _fsync_directory(path.parent)
        except OSError:
            # The actual publication result has already been determined.  A
            # cleanup-directory fsync failure cannot safely be repaired here.
            pass


def _normalize_requested_source_ids(value: Any) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)):
        values = (value,)
    else:
        try:
            values = tuple(value)
        except TypeError as exc:
            raise SharedExposureContractError(
                "target_source_ids must be an iterable of signed 64-bit integers"
            ) from exc
    if not values:
        raise SharedExposureContractError("target_source_ids must not be empty")
    result = tuple(
        _strict_source_id(item, field_name="target_source_ids item") for item in values
    )
    if len(set(result)) != len(result):
        raise SharedExposureContractError(
            "target_source_ids must not contain duplicate source IDs"
        )
    return result


def build_shared_exposure_target_plan(
    geometry: FullFrameSourcePixelGeometry,
    target_source_ids: Iterable[int],
    *,
    detector_shape: tuple[int, int],
    stamp_shape: tuple[int, int],
) -> dict[str, Any]:
    """Build the immutable target/window plan from renderer base positions."""

    if not isinstance(geometry, FullFrameSourcePixelGeometry):
        raise SharedExposureContractError(
            "geometry must be a FullFrameSourcePixelGeometry"
        )
    detector_shape = _strict_shape(detector_shape, field_name="detector_shape")
    stamp_shape = _strict_shape(stamp_shape, field_name="stamp_shape")
    requested = _normalize_requested_source_ids(target_source_ids)

    detector_ids = tuple(str(value) for value in geometry.detector_ids.tolist())
    unique_detector_ids = set(detector_ids)
    if len(unique_detector_ids) != 1:
        raise SharedExposureContractError(
            "shared-exposure geometry must describe exactly one detector"
        )
    detector_id = _strict_text(
        next(iter(unique_detector_ids)), field_name="geometry detector_id"
    )

    source_lookup = {
        int(source_id): index
        for index, source_id in enumerate(geometry.source_ids.tolist())
    }
    missing = [source_id for source_id in requested if source_id not in source_lookup]
    if missing:
        raise SharedExposureContractError(
            f"requested source IDs are missing from renderer geometry: {missing!r}"
        )

    targets: list[dict[str, Any]] = []
    for request_index, source_id in enumerate(requested):
        geometry_index = source_lookup[source_id]
        target_detector_id = detector_ids[geometry_index]
        if target_detector_id != detector_id:
            raise SharedExposureContractError(
                "target detector identity disagrees with shared detector"
            )
        x_frame_pix = float(geometry.x_frame_pix[geometry_index])
        y_frame_pix = float(geometry.y_frame_pix[geometry_index])
        if not np.isfinite(x_frame_pix) or not np.isfinite(y_frame_pix):
            raise SharedExposureContractError(
                "renderer target positions must be finite"
            )
        window = StampWindow.centered_on(
            target_x_detector_pix=x_frame_pix,
            target_y_detector_pix=y_frame_pix,
            shape=stamp_shape,
            detector_shape=detector_shape,
        )
        if any(size <= 0 for size in window.clipped_shape):
            raise SharedExposureContractError(
                f"target source_id={source_id} has no detector overlap"
            )
        targets.append(
            {
                "request_index": request_index,
                "source_id": source_id,
                "detector_id": detector_id,
                "x_frame_pix": x_frame_pix,
                "y_frame_pix": y_frame_pix,
                "window": window.to_schema(),
            }
        )

    payload = {
        "schema_id": TARGET_PLAN_SCHEMA_ID,
        "schema_version": TARGET_PLAN_SCHEMA_VERSION,
        "source_geometry": {
            "schema_id": geometry.schema_id,
            "schema_version": geometry.schema_version,
            "geometry_truth_mode": geometry.geometry_truth_mode,
            "geometry_truth_content_sha256": (geometry.geometry_truth_content_sha256),
            "position_basis": _POSITION_BASIS,
            "position_columns": list(geometry.position_columns),
            "position_transform": geometry.position_transform,
            "coordinate_convention": geometry.coordinate_convention,
        },
        "detector": {
            "detector_id": detector_id,
            "shape": list(detector_shape),
        },
        "stamp_shape": list(stamp_shape),
        "targets": targets,
    }
    plan = _attach_content_digest(payload)
    return validate_shared_exposure_target_plan(plan)


_WINDOW_KEYS = frozenset(
    {
        "shape",
        "x_start_detector_pix",
        "y_start_detector_pix",
        "x_stop_detector_pix_exclusive",
        "y_stop_detector_pix_exclusive",
        "clipped_detector_bounds",
        "clipped_shape",
        "clipped_by_detector",
        "target_x_detector_pix",
        "target_y_detector_pix",
        "detector_shape",
        "coordinate_convention",
    }
)


def _validate_window_schema(
    value: Any,
    *,
    detector_shape: tuple[int, int],
    stamp_shape: tuple[int, int],
    target_x: float,
    target_y: float,
) -> StampWindow:
    window = _require_exact_keys(value, _WINDOW_KEYS, field_name="target.window")
    shape = _strict_shape(window["shape"], field_name="target.window.shape")
    recorded_detector_shape = _strict_shape(
        window["detector_shape"], field_name="target.window.detector_shape"
    )
    if shape != stamp_shape or recorded_detector_shape != detector_shape:
        raise SharedExposureContractError(
            "target window shape or detector_shape disagrees with target plan"
        )
    x_start = _strict_int(
        window["x_start_detector_pix"],
        field_name="target.window.x_start_detector_pix",
    )
    y_start = _strict_int(
        window["y_start_detector_pix"],
        field_name="target.window.y_start_detector_pix",
    )
    _strict_int(
        window["x_stop_detector_pix_exclusive"],
        field_name="target.window.x_stop_detector_pix_exclusive",
    )
    _strict_int(
        window["y_stop_detector_pix_exclusive"],
        field_name="target.window.y_stop_detector_pix_exclusive",
    )
    _strict_shape(window["clipped_shape"], field_name="target.window.clipped_shape")
    bounds = _require_exact_keys(
        window["clipped_detector_bounds"],
        {
            "x_start_pix",
            "x_stop_pix_exclusive",
            "y_start_pix",
            "y_stop_pix_exclusive",
        },
        field_name="target.window.clipped_detector_bounds",
    )
    for name in bounds:
        _strict_int(
            bounds[name],
            field_name=f"target.window.clipped_detector_bounds.{name}",
            minimum=0,
        )
    _strict_bool(
        window["clipped_by_detector"],
        field_name="target.window.clipped_by_detector",
    )
    recorded_x = _strict_float(
        window["target_x_detector_pix"],
        field_name="target.window.target_x_detector_pix",
    )
    recorded_y = _strict_float(
        window["target_y_detector_pix"],
        field_name="target.window.target_y_detector_pix",
    )
    if recorded_x != target_x or recorded_y != target_y:
        raise SharedExposureContractError(
            "target window center disagrees with renderer target position"
        )
    if window["coordinate_convention"] != STAMP_DETECTOR_COORDINATE_CONVENTION:
        raise SharedExposureContractError(
            "target window coordinate_convention is unsupported"
        )
    result = StampWindow(
        x_start_detector_pix=x_start,
        y_start_detector_pix=y_start,
        shape=shape,
        detector_shape=recorded_detector_shape,
        target_x_detector_pix=recorded_x,
        target_y_detector_pix=recorded_y,
    )
    if window != result.to_schema():
        raise SharedExposureContractError(
            "target window derived bounds disagree with canonical StampWindow"
        )
    if any(size <= 0 for size in result.clipped_shape):
        raise SharedExposureContractError("target window has no detector overlap")
    return result


def validate_shared_exposure_target_plan(
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate plan schema, derived windows, and intrinsic content digest."""

    plan = _require_exact_keys(
        plan,
        {
            "schema_id",
            "schema_version",
            "content_sha256",
            "source_geometry",
            "detector",
            "stamp_shape",
            "targets",
        },
        field_name="target plan",
    )
    if plan["schema_id"] != TARGET_PLAN_SCHEMA_ID:
        raise SharedExposureContractError("unsupported target plan schema_id")
    if (
        _strict_int(plan["schema_version"], field_name="target plan schema_version")
        != TARGET_PLAN_SCHEMA_VERSION
    ):
        raise SharedExposureContractError("unsupported target plan schema_version")

    source_geometry = _require_exact_keys(
        plan["source_geometry"],
        {
            "schema_id",
            "schema_version",
            "geometry_truth_mode",
            "geometry_truth_content_sha256",
            "position_basis",
            "position_columns",
            "position_transform",
            "coordinate_convention",
        },
        field_name="source_geometry",
    )
    if source_geometry["schema_id"] != _SOURCE_GEOMETRY_SCHEMA_ID:
        raise SharedExposureContractError("unsupported source geometry schema_id")
    if (
        _strict_int(
            source_geometry["schema_version"],
            field_name="source_geometry.schema_version",
        )
        != _SOURCE_GEOMETRY_SCHEMA_VERSION
    ):
        raise SharedExposureContractError("unsupported source geometry schema_version")
    if source_geometry["geometry_truth_mode"] not in _GEOMETRY_TRUTH_MODES:
        raise SharedExposureContractError("unsupported geometry_truth_mode")
    _strict_sha256(
        source_geometry["geometry_truth_content_sha256"],
        field_name="source_geometry.geometry_truth_content_sha256",
    )
    if source_geometry["position_basis"] != _POSITION_BASIS:
        raise SharedExposureContractError(
            "source geometry must use static_base_renderer_positions"
        )
    columns_value = source_geometry["position_columns"]
    if not isinstance(columns_value, list) or len(columns_value) != 2:
        raise SharedExposureContractError(
            "source_geometry.position_columns must contain two names"
        )
    columns = tuple(
        _strict_text(item, field_name="source_geometry.position_columns item")
        for item in columns_value
    )
    if _POSITION_CONTRACTS.get(columns) != source_geometry["position_transform"]:
        raise SharedExposureContractError(
            "source geometry position columns and transform disagree"
        )
    if (
        source_geometry["coordinate_convention"]
        != FullFrameSourcePixelGeometry.coordinate_convention
    ):
        raise SharedExposureContractError(
            "source geometry coordinate_convention is unsupported"
        )

    detector = _require_exact_keys(
        plan["detector"], {"detector_id", "shape"}, field_name="detector"
    )
    detector_id = _strict_text(
        detector["detector_id"], field_name="detector.detector_id"
    )
    detector_shape = _strict_shape(detector["shape"], field_name="detector.shape")
    stamp_shape = _strict_shape(plan["stamp_shape"], field_name="stamp_shape")

    targets = plan["targets"]
    if not isinstance(targets, list) or not targets:
        raise SharedExposureContractError("targets must be a non-empty array")
    seen_ids: set[int] = set()
    for request_index, raw_target in enumerate(targets):
        target = _require_exact_keys(
            raw_target,
            {
                "request_index",
                "source_id",
                "detector_id",
                "x_frame_pix",
                "y_frame_pix",
                "window",
            },
            field_name=f"targets[{request_index}]",
        )
        recorded_index = _strict_int(
            target["request_index"],
            field_name=f"targets[{request_index}].request_index",
            minimum=0,
        )
        if recorded_index != request_index:
            raise SharedExposureContractError(
                "target request_index must match requested target order"
            )
        source_id = _strict_source_id(
            target["source_id"],
            field_name=f"targets[{request_index}].source_id",
        )
        if source_id in seen_ids:
            raise SharedExposureContractError("targets contain duplicate source IDs")
        seen_ids.add(source_id)
        if target["detector_id"] != detector_id:
            raise SharedExposureContractError(
                "target detector_id disagrees with target plan detector"
            )
        x_frame_pix = _strict_float(
            target["x_frame_pix"],
            field_name=f"targets[{request_index}].x_frame_pix",
        )
        y_frame_pix = _strict_float(
            target["y_frame_pix"],
            field_name=f"targets[{request_index}].y_frame_pix",
        )
        _validate_window_schema(
            target["window"],
            detector_shape=detector_shape,
            stamp_shape=stamp_shape,
            target_x=x_frame_pix,
            target_y=y_frame_pix,
        )

    _validate_content_digest(plan)
    return _canonical_copy(plan)


def publish_shared_exposure_target_plan(
    path: Path | str,
    plan: Mapping[str, Any],
) -> Path:
    """Atomically publish an immutable canonical target plan."""

    return _publish_canonical_no_replace(
        Path(path),
        plan,
        validate=validate_shared_exposure_target_plan,
    )


def read_shared_exposure_target_plan(path: Path | str) -> dict[str, Any]:
    """Read a target plan and require its on-disk bytes to be canonical."""

    path = Path(path)
    payload, raw = _read_json_object(path, artifact_name="target plan")
    validated = validate_shared_exposure_target_plan(payload)
    if raw != _canonical_json_bytes(validated):
        raise SharedExposureContractError(
            f"target plan is not encoded as canonical JSON: {path}"
        )
    return validated


def shared_exposure_product_shard_path(
    root: Path | str,
    *,
    plan_content_sha256: str,
    product_key: str,
) -> Path:
    """Return a path-safe shard name derived only from cryptographic hashes."""

    plan_digest = _strict_sha256(plan_content_sha256, field_name="plan_content_sha256")
    product_key = _strict_text(product_key, field_name="product_key")
    product_digest = sha256(product_key.encode("utf-8")).hexdigest()
    resolved_root = Path(root).resolve()
    return (
        resolved_root
        / "shared-exposure-products"
        / plan_digest
        / f"{product_digest}.h5"
    )


def _resolved_reference_root(reference_root: Path | str) -> Path:
    root = Path(reference_root).resolve()
    if not root.is_dir():
        raise SharedExposureContractError(
            f"reference_root must be an existing directory: {root}"
        )
    return root


def _reference_path_for_build(
    value: Path | str,
    *,
    root: Path,
    field_name: str,
) -> tuple[Path, str]:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise SharedExposureReferenceDriftError(
            f"{field_name} reference is missing: {candidate}"
        ) from exc
    if not resolved.is_file():
        raise SharedExposureReferenceDriftError(
            f"{field_name} reference is missing or not a file: {candidate}"
        )
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise SharedExposureContractError(
            f"{field_name} reference must remain within reference_root"
        ) from exc
    return resolved, relative.as_posix()


def _validated_relative_path(value: Any, *, field_name: str) -> PurePosixPath:
    text = _strict_text(value, field_name=field_name)
    relative = PurePosixPath(text)
    if (
        relative.is_absolute()
        or not relative.parts
        or text != relative.as_posix()
        or any(part in {"", ".", ".."} for part in relative.parts)
        or "\\" in text
    ):
        raise SharedExposureContractError(
            f"{field_name} must be a canonical relative path within reference_root"
        )
    return relative


def _reference_path_for_validation(
    value: Any,
    *,
    root: Path,
    field_name: str,
) -> Path:
    relative = _validated_relative_path(value, field_name=field_name)
    candidate = root.joinpath(*relative.parts)
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise SharedExposureReferenceDriftError(
            f"{field_name} reference is missing: {candidate}"
        ) from exc
    except OSError as exc:
        raise SharedExposureReferenceDriftError(
            f"{field_name} reference cannot be resolved and may have drifted"
        ) from exc
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise SharedExposureReferenceDriftError(
            f"{field_name} reference drifted outside reference_root"
        ) from exc
    if not resolved.is_file():
        raise SharedExposureReferenceDriftError(
            f"{field_name} reference is missing or not a file: {candidate}"
        )
    return resolved


def _hash_file_storage_guard_uncached(
    path: Path,
    *,
    field_name: str,
) -> tuple[_StorageGuardIdentity, int, str]:
    try:
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            digest = sha256()
            size = 0
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                digest.update(chunk)
            after = os.fstat(handle.fileno())
        current = path.stat()
    except FileNotFoundError as exc:
        raise SharedExposureReferenceDriftError(
            f"{field_name} reference is missing: {path}"
        ) from exc
    except OSError as exc:
        raise SharedExposureReferenceDriftError(
            f"{field_name} reference could not be hashed and may have drifted: {path}"
        ) from exc
    identity_before = _storage_guard_identity(before)
    identity_after = _storage_guard_identity(after)
    path_identity = _storage_guard_identity(current)
    if (
        identity_before != identity_after
        or identity_after != path_identity
        or size != after.st_size
    ):
        raise SharedExposureReferenceDriftError(
            f"{field_name} reference drifted while it was being hashed: {path}"
        )
    return identity_after, size, digest.hexdigest()


def _validated_storage_guard_cache(
    value: SharedExposureStorageGuardCache | None,
) -> SharedExposureStorageGuardCache | None:
    if value is not None and not isinstance(value, SharedExposureStorageGuardCache):
        raise TypeError(
            "storage_guard_cache must be a SharedExposureStorageGuardCache or None"
        )
    return value


def _file_storage_guard(
    path: Path,
    *,
    field_name: str,
    storage_guard_cache: SharedExposureStorageGuardCache | None = None,
) -> tuple[int, str]:
    cache = _validated_storage_guard_cache(storage_guard_cache)
    if cache is not None:
        return cache._guard_for(path, field_name=field_name)
    _, size, digest = _hash_file_storage_guard_uncached(
        path,
        field_name=field_name,
    )
    return size, digest


def _guard_record_for_build(
    value: Path | str,
    *,
    root: Path,
    field_name: str,
    storage_guard_cache: SharedExposureStorageGuardCache | None = None,
) -> tuple[Path, dict[str, Any]]:
    resolved, relative = _reference_path_for_build(
        value, root=root, field_name=field_name
    )
    size, digest = _file_storage_guard(
        resolved,
        field_name=field_name,
        storage_guard_cache=storage_guard_cache,
    )
    return resolved, {
        "path": relative,
        "size_bytes": size,
        "storage_guard_sha256": digest,
    }


def _validate_guard_record(
    value: Any,
    *,
    expected_keys: set[str],
    field_name: str,
) -> Mapping[str, Any]:
    record = _require_exact_keys(value, expected_keys, field_name=field_name)
    _validated_relative_path(record["path"], field_name=f"{field_name}.path")
    _strict_int(record["size_bytes"], field_name=f"{field_name}.size_bytes", minimum=0)
    _strict_sha256(
        record["storage_guard_sha256"],
        field_name=f"{field_name}.storage_guard_sha256",
    )
    return record


def _assert_guard_unchanged(
    path: Path,
    record: Mapping[str, Any],
    *,
    field_name: str,
    storage_guard_cache: SharedExposureStorageGuardCache | None = None,
) -> None:
    size, digest = _file_storage_guard(
        path,
        field_name=field_name,
        storage_guard_cache=storage_guard_cache,
    )
    if size != record["size_bytes"] or digest != record["storage_guard_sha256"]:
        raise SharedExposureReferenceDriftError(
            f"{field_name} reference storage drift detected: {path}"
        )


def build_shared_exposure_frame_completion(
    *,
    frame_index: int,
    detector_id: str,
    mode: str,
    reference_root: Path | str,
    parent_path: Path | str,
    plan_path: Path | str,
    product_shards: Mapping[str, Path | str],
    storage_guard_cache: SharedExposureStorageGuardCache | None = None,
) -> dict[str, Any]:
    """Build a completion marker whose hashes are storage guards only."""

    storage_guard_cache = _validated_storage_guard_cache(storage_guard_cache)
    frame_index = _strict_int(frame_index, field_name="frame_index", minimum=0)
    detector_id = _strict_text(detector_id, field_name="detector_id")
    mode = _strict_text(mode, field_name="mode")
    root = _resolved_reference_root(reference_root)

    _, parent_guard = _guard_record_for_build(
        parent_path,
        root=root,
        field_name="parent",
        storage_guard_cache=storage_guard_cache,
    )
    parent_record = {
        "scope": _PARENT_GUARD_SCOPE,
        "scientific_lineage": _NOT_SCIENTIFIC_LINEAGE,
        **parent_guard,
    }

    resolved_plan, plan_guard = _guard_record_for_build(
        plan_path,
        root=root,
        field_name="plan",
        storage_guard_cache=storage_guard_cache,
    )
    try:
        plan = read_shared_exposure_target_plan(resolved_plan)
    except SharedExposureContractError as exc:
        raise SharedExposureContractError(
            f"plan reference is not a valid canonical target plan: {resolved_plan}"
        ) from exc
    if plan["detector"]["detector_id"] != detector_id:
        raise SharedExposureContractError(
            "completion detector_id disagrees with target plan detector_id"
        )
    plan_record = {
        "path": plan_guard["path"],
        "schema_id": TARGET_PLAN_SCHEMA_ID,
        "content_sha256": plan["content_sha256"],
        "size_bytes": plan_guard["size_bytes"],
        "storage_guard_sha256": plan_guard["storage_guard_sha256"],
    }

    if not isinstance(product_shards, Mapping) or not product_shards:
        raise SharedExposureContractError(
            "product_shards must be a non-empty product-key mapping"
        )
    shard_records: list[dict[str, Any]] = []
    product_keys = [
        _strict_text(product_key, field_name="product shard key")
        for product_key in product_shards
    ]
    for product_key in sorted(product_keys):
        resolved_shard, shard_guard = _guard_record_for_build(
            product_shards[product_key],
            root=root,
            field_name=f"product shard {product_key!r}",
            storage_guard_cache=storage_guard_cache,
        )
        expected_path = shared_exposure_product_shard_path(
            resolved_shard.parent,
            plan_content_sha256=plan["content_sha256"],
            product_key=product_key,
        )
        if resolved_shard.name != expected_path.name:
            raise SharedExposureContractError(
                f"product shard {product_key!r} does not use its hashed shard name"
            )
        shard_records.append({"product_key": product_key, **shard_guard})

    payload = {
        "schema_id": FRAME_COMPLETION_SCHEMA_ID,
        "schema_version": FRAME_COMPLETION_SCHEMA_VERSION,
        "frame": {"detector_id": detector_id, "frame_index": frame_index},
        "mode": mode,
        "parent_storage_guard": parent_record,
        "plan": plan_record,
        "shards": shard_records,
        "upstream_negative_claims": dict(_UPSTREAM_NEGATIVE_CLAIMS),
    }
    marker = _attach_content_digest(payload)
    return validate_shared_exposure_frame_completion(
        marker,
        reference_root=root,
        storage_guard_cache=storage_guard_cache,
    )


def _validate_negative_claims(value: Any) -> None:
    claims = _require_exact_keys(
        value,
        set(_UPSTREAM_NEGATIVE_CLAIMS),
        field_name="upstream_negative_claims",
    )
    for name, expected in _UPSTREAM_NEGATIVE_CLAIMS.items():
        actual = claims[name]
        if isinstance(expected, bool):
            if not isinstance(actual, bool) or actual is not expected:
                raise SharedExposureContractError(
                    f"upstream_negative_claims.{name} must remain {expected!r}"
                )
        elif not isinstance(actual, str) or actual != expected:
            raise SharedExposureContractError(
                f"upstream_negative_claims.{name} must remain {expected!r}"
            )


def validate_shared_exposure_frame_completion(
    marker: Mapping[str, Any],
    *,
    reference_root: Path | str,
    storage_guard_cache: SharedExposureStorageGuardCache | None = None,
) -> dict[str, Any]:
    """Validate a marker and verify every referenced storage guard.

    With no cache, every call rehashes every reference.  A caller-owned cache
    may reuse a digest only while the file's full stat identity is unchanged.
    """

    storage_guard_cache = _validated_storage_guard_cache(storage_guard_cache)
    marker = _require_exact_keys(
        marker,
        {
            "schema_id",
            "schema_version",
            "content_sha256",
            "frame",
            "mode",
            "parent_storage_guard",
            "plan",
            "shards",
            "upstream_negative_claims",
        },
        field_name="frame completion marker",
    )
    if marker["schema_id"] != FRAME_COMPLETION_SCHEMA_ID:
        raise SharedExposureContractError("unsupported frame completion schema_id")
    if (
        _strict_int(
            marker["schema_version"],
            field_name="frame completion schema_version",
        )
        != FRAME_COMPLETION_SCHEMA_VERSION
    ):
        raise SharedExposureContractError("unsupported frame completion schema_version")
    frame = _require_exact_keys(
        marker["frame"], {"detector_id", "frame_index"}, field_name="frame"
    )
    detector_id = _strict_text(frame["detector_id"], field_name="frame.detector_id")
    _strict_int(frame["frame_index"], field_name="frame_index", minimum=0)
    _strict_text(marker["mode"], field_name="mode")
    root = _resolved_reference_root(reference_root)

    parent_record = _validate_guard_record(
        marker["parent_storage_guard"],
        expected_keys={
            "scope",
            "scientific_lineage",
            "path",
            "size_bytes",
            "storage_guard_sha256",
        },
        field_name="parent_storage_guard",
    )
    if parent_record["scope"] != _PARENT_GUARD_SCOPE:
        raise SharedExposureContractError(
            "parent_storage_guard.scope must remain storage_resume_guard_only"
        )
    if parent_record["scientific_lineage"] != _NOT_SCIENTIFIC_LINEAGE:
        raise SharedExposureContractError(
            "parent_storage_guard.scientific_lineage must remain not_scientific_lineage"
        )

    plan_record = _validate_guard_record(
        marker["plan"],
        expected_keys={
            "path",
            "schema_id",
            "content_sha256",
            "size_bytes",
            "storage_guard_sha256",
        },
        field_name="plan",
    )
    if plan_record["schema_id"] != TARGET_PLAN_SCHEMA_ID:
        raise SharedExposureContractError("completion plan schema_id is unsupported")
    plan_digest = _strict_sha256(
        plan_record["content_sha256"], field_name="plan.content_sha256"
    )

    shards = marker["shards"]
    if not isinstance(shards, list) or not shards:
        raise SharedExposureContractError("shards must be a non-empty array")
    shard_records: list[Mapping[str, Any]] = []
    product_keys: list[str] = []
    recorded_paths: set[str] = set()
    for index, item in enumerate(shards):
        record = _validate_guard_record(
            item,
            expected_keys={
                "product_key",
                "path",
                "size_bytes",
                "storage_guard_sha256",
            },
            field_name=f"shards[{index}]",
        )
        product_key = _strict_text(
            record["product_key"], field_name=f"shards[{index}].product_key"
        )
        product_keys.append(product_key)
        if record["path"] in recorded_paths:
            raise SharedExposureContractError("shards contain duplicate paths")
        recorded_paths.add(record["path"])
        recorded_relative_path = _validated_relative_path(
            record["path"], field_name=f"shards[{index}].path"
        )
        expected_name = shared_exposure_product_shard_path(
            root,
            plan_content_sha256=plan_digest,
            product_key=product_key,
        ).name
        if recorded_relative_path.name != expected_name:
            raise SharedExposureContractError(
                f"shard {product_key!r} does not use its hashed shard name"
            )
        shard_records.append(record)
    if product_keys != sorted(product_keys) or len(set(product_keys)) != len(
        product_keys
    ):
        raise SharedExposureContractError(
            "shards must have unique product keys in sorted order"
        )

    _validate_negative_claims(marker["upstream_negative_claims"])
    _validate_content_digest(marker)

    # The marker's intrinsic contract is valid.  Only now consult mutable
    # storage, and translate all referenced-plan damage into drift semantics.
    parent_path = _reference_path_for_validation(
        parent_record["path"], root=root, field_name="parent_storage_guard"
    )
    plan_path = _reference_path_for_validation(
        plan_record["path"], root=root, field_name="plan"
    )
    shard_paths_and_records = [
        (
            _reference_path_for_validation(
                record["path"], root=root, field_name=f"shards[{index}]"
            ),
            record,
        )
        for index, record in enumerate(shard_records)
    ]
    _assert_guard_unchanged(
        parent_path,
        parent_record,
        field_name="parent_storage_guard",
        storage_guard_cache=storage_guard_cache,
    )
    _assert_guard_unchanged(
        plan_path,
        plan_record,
        field_name="plan",
        storage_guard_cache=storage_guard_cache,
    )
    try:
        plan = read_shared_exposure_target_plan(plan_path)
    except SharedExposureContractError as exc:
        raise SharedExposureReferenceDriftError(
            f"plan reference content drift detected: {plan_path}"
        ) from exc
    if plan["content_sha256"] != plan_digest:
        raise SharedExposureReferenceDriftError(
            "plan reference intrinsic content_sha256 drift detected"
        )
    if plan["detector"]["detector_id"] != detector_id:
        raise SharedExposureContractError(
            "completion detector_id disagrees with referenced target plan"
        )
    for index, (path, record) in enumerate(shard_paths_and_records):
        _assert_guard_unchanged(
            path,
            record,
            field_name=f"shards[{index}]",
            storage_guard_cache=storage_guard_cache,
        )

    return _canonical_copy(marker)


def publish_shared_exposure_frame_completion(
    path: Path | str,
    marker: Mapping[str, Any],
    *,
    reference_root: Path | str,
    storage_guard_cache: SharedExposureStorageGuardCache | None = None,
) -> Path:
    """Atomically publish an immutable canonical frame-completion marker."""

    storage_guard_cache = _validated_storage_guard_cache(storage_guard_cache)
    return _publish_canonical_no_replace(
        Path(path),
        marker,
        validate=lambda candidate: validate_shared_exposure_frame_completion(
            candidate,
            reference_root=reference_root,
            storage_guard_cache=storage_guard_cache,
        ),
    )


def read_shared_exposure_frame_completion(
    path: Path | str,
    *,
    reference_root: Path | str,
    storage_guard_cache: SharedExposureStorageGuardCache | None = None,
) -> dict[str, Any]:
    """Read a completion marker and revalidate every storage guard."""

    storage_guard_cache = _validated_storage_guard_cache(storage_guard_cache)
    path = Path(path)
    payload, raw = _read_json_object(path, artifact_name="frame completion marker")
    validated = validate_shared_exposure_frame_completion(
        payload,
        reference_root=reference_root,
        storage_guard_cache=storage_guard_cache,
    )
    if raw != _canonical_json_bytes(validated):
        raise SharedExposureContractError(
            f"frame completion marker is not encoded as canonical JSON: {path}"
        )
    return validated


def _as_numpy_array(value: Any, *, field_name: str) -> np.ndarray:
    candidate = value
    if hasattr(candidate, "detach") and callable(candidate.detach):
        candidate = candidate.detach()
    if hasattr(candidate, "cpu") and callable(candidate.cpu):
        candidate = candidate.cpu()
    if hasattr(candidate, "numpy") and callable(candidate.numpy):
        try:
            candidate = candidate.numpy()
        except (TypeError, RuntimeError, ValueError):
            pass
    try:
        array = np.asarray(candidate)
    except (TypeError, ValueError) as exc:
        raise SharedExposureContractError(
            f"{field_name} must be convertible to a NumPy array"
        ) from exc
    if array.dtype.hasobject:
        raise SharedExposureContractError(
            f"{field_name} must not have an object-containing dtype"
        )
    return array


def array_c_order_fingerprint(array: Any) -> dict[str, Any]:
    """Fingerprint exact shape, dtype identity, and logical C-order bytes."""

    normalized = _as_numpy_array(array, field_name="array")
    contiguous = np.ascontiguousarray(normalized)
    content = contiguous.tobytes(order="C")
    return {
        "shape": [int(size) for size in normalized.shape],
        "dtype": np.lib.format.dtype_to_descr(normalized.dtype),
        "nbytes": len(content),
        "content_sha256": sha256(content).hexdigest(),
    }


def assert_exact_array_match(expected: Any, actual: Any) -> None:
    """Require equal shape, equal dtype, and equal logical C-order bytes."""

    expected_array = _as_numpy_array(expected, field_name="expected")
    actual_array = _as_numpy_array(actual, field_name="actual")
    if expected_array.shape != actual_array.shape:
        raise SharedExposureArrayMismatchError(
            f"array shape mismatch: expected {expected_array.shape}, "
            f"got {actual_array.shape}"
        )
    if expected_array.dtype != actual_array.dtype:
        raise SharedExposureArrayMismatchError(
            f"array dtype mismatch: expected {expected_array.dtype}, "
            f"got {actual_array.dtype}"
        )
    expected_bytes = np.ascontiguousarray(expected_array).tobytes(order="C")
    actual_bytes = np.ascontiguousarray(actual_array).tobytes(order="C")
    if expected_bytes != actual_bytes:
        raise SharedExposureArrayMismatchError("array C-order bytes mismatch")


def assert_exact_parent_crop(parent: Any, crop: Any, window: StampWindow) -> None:
    """Require ``crop`` to be the exact zero-padded crop of ``parent``."""

    if not isinstance(window, StampWindow):
        raise SharedExposureContractError("window must be a StampWindow")
    parent_array = _as_numpy_array(parent, field_name="parent")
    if parent_array.ndim != 2:
        raise SharedExposureArrayMismatchError("parent shape must be exactly 2-D")
    if parent_array.shape != window.detector_shape:
        raise SharedExposureArrayMismatchError(
            "parent shape must match window.detector_shape"
        )
    expected = np.zeros(window.shape, dtype=parent_array.dtype)
    y0, y1, x0, x1 = window.clipped_bounds
    if y1 > y0 and x1 > x0:
        expected[window.insertion_slices] = parent_array[y0:y1, x0:x1]
    assert_exact_array_match(expected, crop)


__all__ = [
    "FRAME_COMPLETION_SCHEMA_ID",
    "FRAME_COMPLETION_SCHEMA_VERSION",
    "SharedExposureArrayMismatchError",
    "SharedExposureContractError",
    "SharedExposurePublicationError",
    "SharedExposureReferenceDriftError",
    "SharedExposureStorageGuardCache",
    "TARGET_PLAN_SCHEMA_ID",
    "TARGET_PLAN_SCHEMA_VERSION",
    "array_c_order_fingerprint",
    "assert_exact_array_match",
    "assert_exact_parent_crop",
    "build_shared_exposure_frame_completion",
    "build_shared_exposure_target_plan",
    "publish_shared_exposure_frame_completion",
    "publish_shared_exposure_target_plan",
    "read_shared_exposure_frame_completion",
    "read_shared_exposure_target_plan",
    "shared_exposure_product_shard_path",
    "validate_shared_exposure_frame_completion",
    "validate_shared_exposure_target_plan",
]
