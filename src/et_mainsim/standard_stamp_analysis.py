"""Formal standard light-curve analysis for independent Galaxy stamp delivery.

This module is deliberately a *post-processing* layer over
``et_mainsim.stamp_delivery_bundle.v1``.  It never reads historical pickle
products, calls legacy PCA/SG filtering, or manufactures a new observation:
``final_dn`` remains the only detector observation.  The derived light curve
uses :mod:`et_mainsim.reference_photometry`, which streams the formal HDF5
bundles through its documented central 13x13 reference-aperture contract.

The production-manifest discovery is intentionally fail-closed.  Every shard
listed in the frozen time plan must have published its final HDF5 member; a
staging/partial directory is never considered a scientific input.  For an
``injected`` Galaxy case, the frozen factor snapshot is identity-checked and
used to emit the physically through-origin model residual and residual CDPP.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
import csv
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Literal

import numpy as np

from .galaxy_lightcurves import read_galaxy_factor_snapshot
from .galaxy_stamp_production import (
    GALAXY_STAMP_PRODUCTION_SCHEMA_ID,
    GALAXY_STAMP_PRODUCTION_SCHEMA_VERSION,
)
from .reference_photometry import (
    STANDARD_CDPP_WINDOWS_MINUTES,
    InjectedModelResidualResult,
    ReferencePhotometryResult,
    compute_injected_model_residual_v1,
    reduce_stamp_delivery_series_v1,
)
from .stamp_inputs import file_identity
from .stamp_delivery import validate_stamp_delivery_bundle
from .time_shards import ContinuousTimeShardPlan


STANDARD_STAMP_ANALYSIS_SCHEMA_ID = "et_mainsim.standard_stamp_analysis.v1"
STANDARD_STAMP_ANALYSIS_SCHEMA_VERSION = 1
STANDARD_STAMP_ANALYSIS_LIGHTCURVE_SCHEMA_ID = (
    "et_mainsim.standard_stamp_reference_lightcurve.v1"
)

AnalysisCase = Literal["static", "injected"]


class StandardStampAnalysisError(ValueError):
    """Raised when a formal production cannot meet the analysis contract."""


class StandardStampAnalysisNotReadyError(StandardStampAnalysisError):
    """Raised when a frozen manifest names a shard that has not been published."""


@dataclass(frozen=True)
class _DeliveryBundleReceipt:
    """Content identity and schema validation evidence for one HDF5 input."""

    path: Path
    size_bytes: int
    sha256: str
    validation: Mapping[str, Any]


def _strict_source_id(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise StandardStampAnalysisError(f"{name} must be a non-negative signed int64")
    try:
        source_id = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise StandardStampAnalysisError(
            f"{name} must be a non-negative signed int64"
        ) from error
    if source_id < 0 or source_id > int(np.iinfo(np.int64).max):
        raise StandardStampAnalysisError(f"{name} must be a non-negative signed int64")
    return source_id


def _finite_positive(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise StandardStampAnalysisError(f"{name} must be finite and positive")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise StandardStampAnalysisError(
            f"{name} must be finite and positive"
        ) from error
    if not math.isfinite(result) or result <= 0.0:
        raise StandardStampAnalysisError(f"{name} must be finite and positive")
    return result


def _normalise_case(value: object) -> AnalysisCase:
    if not isinstance(value, str) or value not in {"static", "injected"}:
        raise StandardStampAnalysisError("case must be exactly 'static' or 'injected'")
    return value  # type: ignore[return-value]


def _normalise_cdpp_windows(values: Iterable[int]) -> tuple[int, ...]:
    windows = tuple(int(value) for value in values)
    if not windows or any(value <= 0 for value in windows):
        raise StandardStampAnalysisError(
            "CDPP windows must be non-empty positive minutes"
        )
    if len(set(windows)) != len(windows):
        raise StandardStampAnalysisError("CDPP windows must not repeat")
    return windows


@dataclass(frozen=True)
class StandardStampAnalysisRequest:
    """A fully explicit request for one formal target/case/cadence analysis."""

    production_manifest_path: Path | str
    source_id: int
    case: AnalysisCase | str
    cadence_seconds: float
    output_dir: Path | str
    cdpp_windows_minutes: tuple[int, ...] = STANDARD_CDPP_WINDOWS_MINUTES
    bin_origin_seconds: float = 0.0
    batch_frames: int = 4_096
    overwrite: bool = False

    def __post_init__(self) -> None:
        manifest_path = Path(self.production_manifest_path).expanduser().resolve()
        source_id = _strict_source_id(self.source_id, name="source_id")
        case = _normalise_case(self.case)
        cadence_seconds = _finite_positive(
            self.cadence_seconds,
            name="cadence_seconds",
        )
        output_dir = Path(self.output_dir).expanduser().resolve()
        windows = _normalise_cdpp_windows(self.cdpp_windows_minutes)
        try:
            origin = float(self.bin_origin_seconds)
        except (TypeError, ValueError, OverflowError) as error:
            raise StandardStampAnalysisError(
                "bin_origin_seconds must be finite"
            ) from error
        if not math.isfinite(origin):
            raise StandardStampAnalysisError("bin_origin_seconds must be finite")
        if (
            isinstance(self.batch_frames, (bool, np.bool_))
            or int(self.batch_frames) <= 0
        ):
            raise StandardStampAnalysisError("batch_frames must be positive")
        if not isinstance(self.overwrite, bool):
            raise StandardStampAnalysisError("overwrite must be a boolean")
        object.__setattr__(self, "production_manifest_path", manifest_path)
        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(self, "case", case)
        object.__setattr__(self, "cadence_seconds", cadence_seconds)
        object.__setattr__(self, "output_dir", output_dir)
        object.__setattr__(self, "cdpp_windows_minutes", windows)
        object.__setattr__(self, "bin_origin_seconds", origin)
        object.__setattr__(self, "batch_frames", int(self.batch_frames))


@dataclass(frozen=True)
class StandardStampAnalysisInput:
    """Resolved immutable inputs for one standard analysis invocation."""

    production_manifest_path: Path
    run_root: Path
    production_manifest: Mapping[str, Any]
    target: Mapping[str, Any]
    time_plan: ContinuousTimeShardPlan
    source_id: int
    case: AnalysisCase
    cadence_seconds: float
    product_filename: str
    bundle_paths: tuple[Path, ...]
    factor_snapshot_path: Path | None
    factor_snapshot_identity: Mapping[str, Any] | None


@dataclass(frozen=True)
class StandardStampAnalysisResult:
    """Paths and compact summary from a completed standard analysis."""

    analysis_manifest_path: Path
    reference_lightcurve_path: Path
    source_id: int
    case: AnalysisCase
    cadence_seconds: float
    cadence_count: int
    valid_cadence_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis_manifest_path": str(self.analysis_manifest_path),
            "reference_lightcurve_path": str(self.reference_lightcurve_path),
            "source_id": str(self.source_id),
            "case": self.case,
            "cadence_seconds": self.cadence_seconds,
            "cadence_count": self.cadence_count,
            "valid_cadence_count": self.valid_cadence_count,
        }


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as error:
        raise StandardStampAnalysisError(
            f"{label} is not valid JSON: {path}"
        ) from error
    if not isinstance(payload, dict):
        raise StandardStampAnalysisError(f"{label} must be a JSON object: {path}")
    return payload


def _resolve_relative_resource(run_root: Path, value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise StandardStampAnalysisError(f"{label} must have a non-empty relative path")
    relative = Path(value)
    if relative.is_absolute():
        raise StandardStampAnalysisError(f"{label} must use a manifest-relative path")
    candidate = (run_root / relative).resolve()
    try:
        candidate.relative_to(run_root)
    except ValueError as error:
        raise StandardStampAnalysisError(
            f"{label} path escapes the production manifest root"
        ) from error
    return candidate


def _same_file_content_identity(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> bool:
    try:
        return int(actual["size_bytes"]) == int(expected["size_bytes"]) and str(
            actual["sha256"]
        ) == str(expected["sha256"])
    except (KeyError, TypeError, ValueError):
        return False


def _load_galaxy_manifest(path: Path) -> tuple[Path, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"production manifest does not exist: {path}")
    manifest = _read_json_object(path, label="production manifest")
    if manifest.get("schema_id") != GALAXY_STAMP_PRODUCTION_SCHEMA_ID:
        raise StandardStampAnalysisError(
            "standard_stamp_analysis_v1 currently requires a formal Galaxy "
            "production manifest"
        )
    if int(manifest.get("schema_version", 0)) != GALAXY_STAMP_PRODUCTION_SCHEMA_VERSION:
        raise StandardStampAnalysisError(
            "unsupported Galaxy production manifest version"
        )
    if manifest.get("observation_product") != "final_dn":
        raise StandardStampAnalysisError(
            "production manifest observation_product must be final_dn"
        )
    if manifest.get("background_realization_delivered") is not False:
        raise StandardStampAnalysisError(
            "production manifest must not expose a background realization"
        )
    return path, manifest


def _target_from_manifest(
    manifest: Mapping[str, Any], source_id: int
) -> Mapping[str, Any]:
    targets = manifest.get("targets")
    if not isinstance(targets, list):
        raise StandardStampAnalysisError("production manifest targets must be a list")
    for candidate in targets:
        if not isinstance(candidate, Mapping):
            raise StandardStampAnalysisError(
                "production manifest target entries must be objects"
            )
        if (
            _strict_source_id(candidate.get("source_id_int64"), name="target.source_id")
            == source_id
        ):
            return candidate
    raise StandardStampAnalysisError(
        f"production manifest has no source_id={source_id}"
    )


def _time_plan_from_manifest(
    run_root: Path,
    manifest: Mapping[str, Any],
) -> ContinuousTimeShardPlan:
    delivery = manifest.get("delivery")
    if not isinstance(delivery, Mapping):
        raise StandardStampAnalysisError(
            "production manifest delivery must be an object"
        )
    time_plan_path = _resolve_relative_resource(
        run_root,
        delivery.get("time_plan_relative_path"),
        label="time shard plan",
    )
    if not time_plan_path.is_file():
        raise FileNotFoundError(f"time shard plan does not exist: {time_plan_path}")
    expected_identity = delivery.get("time_plan_identity")
    if not isinstance(expected_identity, Mapping) or not _same_file_content_identity(
        file_identity(time_plan_path),
        expected_identity,
    ):
        raise StandardStampAnalysisError(
            "time shard plan identity changed after production preparation"
        )
    try:
        plan = ContinuousTimeShardPlan.from_manifest_dict(
            _read_json_object(time_plan_path, label="time shard plan")
        )
    except ValueError as error:
        raise StandardStampAnalysisError("time shard plan is invalid") from error
    raw_exposure = _finite_positive(
        delivery.get("raw_exposure_seconds"),
        name="delivery.raw_exposure_seconds",
    )
    if not math.isclose(
        plan.raw_exposure_seconds,
        raw_exposure,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise StandardStampAnalysisError(
            "time shard plan raw exposure differs from the production manifest"
        )
    return plan


def _filename_for_cadence(
    *,
    time_plan: ContinuousTimeShardPlan,
    cadence_seconds: float,
) -> str:
    raw_exposure = time_plan.raw_exposure_seconds
    ratio = cadence_seconds / raw_exposure
    coadd_factor = round(ratio)
    if coadd_factor <= 0 or not math.isclose(
        ratio,
        float(coadd_factor),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise StandardStampAnalysisError(
            "cadence_seconds must be an exact integral multiple of raw exposure"
        )
    if coadd_factor == 1:
        return "raw.h5"
    if coadd_factor not in time_plan.coadd_sizes:
        raise StandardStampAnalysisError(
            "requested cadence is not part of the frozen delivery coadd set"
        )
    filename_seconds = coadd_factor * raw_exposure
    if not math.isclose(filename_seconds, round(filename_seconds), abs_tol=1e-12):
        raise StandardStampAnalysisError(
            "formal delivery filenames require an integral cadence in seconds"
        )
    return f"coadd_{int(round(filename_seconds))}s.h5"


def _expected_bundle_paths(
    *,
    run_root: Path,
    source_id: int,
    case: AnalysisCase,
    product_filename: str,
    time_plan: ContinuousTimeShardPlan,
) -> tuple[Path, ...]:
    delivery_root = (
        run_root / "cases" / case / "stamps" / f"target_{source_id}" / "delivery"
    )
    paths = tuple(
        delivery_root / f"shard_{shard.shard_id:05d}" / product_filename
        for shard in time_plan.shards
    )
    missing = tuple(path for path in paths if not path.is_file())
    if missing:
        examples = ", ".join(str(path) for path in missing[:3])
        raise StandardStampAnalysisNotReadyError(
            "missing formal delivery shard(s); analysis only accepts published "
            f"final HDF5 members ({len(missing)} missing): {examples}"
        )
    return paths


def _bundle_stat_fingerprints(
    bundle_paths: Sequence[Path],
) -> dict[Path, tuple[int, int, int, int]]:
    """Capture cheap file-state evidence around validation/reduction.

    Formal final members are immutable by production contract.  Comparing the
    device/inode/size/mtime tuple before and after analysis detects a normal
    filesystem replacement or mutation during the small header/streaming
    time-of-check-to-time-of-use interval, while full SHA-256 receipts bind the
    content that was accepted before the reduction begins.
    """

    fingerprints: dict[Path, tuple[int, int, int, int]] = {}
    for path in bundle_paths:
        stat = path.stat()
        fingerprints[path] = (
            int(stat.st_dev),
            int(stat.st_ino),
            int(stat.st_size),
            int(stat.st_mtime_ns),
        )
    return fingerprints


def _validate_and_identify_bundles(
    bundle_paths: Sequence[Path],
) -> tuple[_DeliveryBundleReceipt, ...]:
    """Fully validate and hash every HDF5 input selected for this analysis."""

    receipts: list[_DeliveryBundleReceipt] = []
    for path in bundle_paths:
        identity = file_identity(path)
        validation = validate_stamp_delivery_bundle(path)
        receipts.append(
            _DeliveryBundleReceipt(
                path=path,
                size_bytes=int(identity["size_bytes"]),
                sha256=str(identity["sha256"]),
                validation={
                    "complete": bool(validation.complete),
                    "product_kind": validation.product_kind,
                    "coadd_factor": int(validation.coadd_factor),
                    "frame_count": int(validation.frame_count),
                    "stamp_shape": list(validation.stamp_shape),
                    "final_dn_dtype": validation.final_dn_dtype,
                    "observation_product": validation.observation_product,
                },
            )
        )
    return tuple(receipts)


def _assert_bundle_stat_fingerprints_unchanged(
    expected: Mapping[Path, tuple[int, int, int, int]],
) -> None:
    """Fail rather than publishing an analysis whose delivery inputs moved."""

    for path, before in expected.items():
        stat = path.stat()
        after = (
            int(stat.st_dev),
            int(stat.st_ino),
            int(stat.st_size),
            int(stat.st_mtime_ns),
        )
        if after != before:
            raise StandardStampAnalysisError(
                f"formal delivery bundle changed during validation or reduction: {path}"
            )


def _verify_bundle_context(
    *,
    bundle_paths: Sequence[Path],
    time_plan: ContinuousTimeShardPlan,
    source_id: int,
    case: AnalysisCase,
    run_id: str,
    expected_factor_snapshot_identity: Mapping[str, Any] | None,
) -> None:
    """Check small per-shard manifest JSON without materialising image cubes."""

    try:
        import h5py
    except ImportError as error:  # pragma: no cover - packaging guard
        raise RuntimeError("h5py is required for formal stamp analysis") from error

    if len(bundle_paths) != len(time_plan.shards):
        raise StandardStampAnalysisError(
            "bundle path count does not match the frozen plan"
        )
    for path, shard in zip(bundle_paths, time_plan.shards, strict=True):
        with h5py.File(path, "r") as handle:
            try:
                raw_json = handle["manifest_json"][()]
            except KeyError as error:
                raise StandardStampAnalysisError(
                    f"formal delivery lacks manifest_json: {path}"
                ) from error
        if isinstance(raw_json, np.ndarray):
            raw_json = raw_json.reshape(()).item()
        if isinstance(raw_json, (bytes, np.bytes_)):
            raw_json = bytes(raw_json).decode("utf-8")
        if not isinstance(raw_json, str):
            raise StandardStampAnalysisError(
                f"formal delivery manifest_json is invalid: {path}"
            )
        try:
            delivery_manifest = json.loads(raw_json)
        except json.JSONDecodeError as error:
            raise StandardStampAnalysisError(
                f"formal delivery manifest_json is invalid: {path}"
            ) from error
        if not isinstance(delivery_manifest, Mapping):
            raise StandardStampAnalysisError(
                f"formal delivery manifest must be an object: {path}"
            )
        if (
            _strict_source_id(
                delivery_manifest.get("target_source_id_int64"),
                name="delivery.target_source_id",
            )
            != source_id
        ):
            raise StandardStampAnalysisError(
                "formal delivery target does not match request"
            )
        caller = delivery_manifest.get("caller_manifest")
        if not isinstance(caller, Mapping):
            raise StandardStampAnalysisError(
                "formal delivery caller_manifest is missing"
            )
        if caller.get("case") != case or str(caller.get("run_id")) != run_id:
            raise StandardStampAnalysisError(
                "formal delivery caller case/run identity does not match request"
            )
        if expected_factor_snapshot_identity is not None:
            source_truth = caller.get("target_input_truth")
            variability = (
                source_truth.get("variability")
                if isinstance(source_truth, Mapping)
                else None
            )
            if (
                not isinstance(variability, Mapping)
                or variability.get("enabled") is not True
                or variability.get("case") != "injected"
            ):
                raise StandardStampAnalysisError(
                    "injected formal delivery lacks frozen variability provenance"
                )
            delivered_identity = variability.get("source_factor_snapshot_identity")
            if not isinstance(
                delivered_identity, Mapping
            ) or not _same_file_content_identity(
                delivered_identity,
                expected_factor_snapshot_identity,
            ):
                raise StandardStampAnalysisError(
                    "formal delivery factor snapshot identity does not match the "
                    "frozen production input"
                )
        time_shard = delivery_manifest.get("time_shard")
        interval = (
            time_shard.get("raw_frame_interval")
            if isinstance(time_shard, Mapping)
            else None
        )
        if not isinstance(interval, Mapping):
            raise StandardStampAnalysisError(
                "formal delivery time shard metadata is missing"
            )
        if (
            _strict_source_id(
                interval.get("start_index"), name="time_shard.start_index"
            )
            != shard.raw_start_index
            or _strict_source_id(
                interval.get("stop_index"), name="time_shard.stop_index"
            )
            != shard.raw_stop_index
        ):
            raise StandardStampAnalysisError(
                "formal delivery time-shard identity does not match the frozen plan"
            )


def discover_standard_stamp_analysis_input(
    request: StandardStampAnalysisRequest,
) -> StandardStampAnalysisInput:
    """Resolve formal Galaxy bundle paths from the frozen production manifest.

    This is intentionally public so notebooks can show ``PENDING`` versus
    ready status before running the potentially long stream reduction.
    """

    if not isinstance(request, StandardStampAnalysisRequest):
        raise TypeError("request must be a StandardStampAnalysisRequest")
    manifest_path, manifest = _load_galaxy_manifest(request.production_manifest_path)
    run_root = manifest_path.parent.resolve()
    target = _target_from_manifest(manifest, request.source_id)
    time_plan = _time_plan_from_manifest(run_root, manifest)
    product_filename = _filename_for_cadence(
        time_plan=time_plan,
        cadence_seconds=request.cadence_seconds,
    )
    snapshot_path: Path | None = None
    snapshot_identity: Mapping[str, Any] | None = None
    if request.case == "injected":
        snapshot_path = _resolve_relative_resource(
            run_root,
            target.get("factor_snapshot_relative_path"),
            label="frozen factor snapshot",
        )
        if not snapshot_path.is_file():
            raise FileNotFoundError(
                f"frozen factor snapshot does not exist: {snapshot_path}"
            )
        expected_identity = target.get("factor_snapshot")
        if not isinstance(
            expected_identity, Mapping
        ) or not _same_file_content_identity(
            file_identity(snapshot_path),
            expected_identity,
        ):
            raise StandardStampAnalysisError(
                "frozen factor snapshot identity changed after production preparation"
            )
        snapshot = read_galaxy_factor_snapshot(snapshot_path)
        if snapshot.source_id != request.source_id:
            raise StandardStampAnalysisError(
                "frozen factor snapshot source ID does not match request"
            )
        if snapshot.factors.size != time_plan.raw_stop_index:
            raise StandardStampAnalysisError(
                "frozen factor snapshot length does not match the production raw axis"
            )
        snapshot_identity = dict(expected_identity)
    bundle_paths = _expected_bundle_paths(
        run_root=run_root,
        source_id=request.source_id,
        case=request.case,
        product_filename=product_filename,
        time_plan=time_plan,
    )
    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise StandardStampAnalysisError("production manifest run_id must be non-empty")
    _verify_bundle_context(
        bundle_paths=bundle_paths,
        time_plan=time_plan,
        source_id=request.source_id,
        case=request.case,
        run_id=run_id,
        expected_factor_snapshot_identity=snapshot_identity,
    )
    return StandardStampAnalysisInput(
        production_manifest_path=manifest_path,
        run_root=run_root,
        production_manifest=manifest,
        target=target,
        time_plan=time_plan,
        source_id=request.source_id,
        case=request.case,
        cadence_seconds=request.cadence_seconds,
        product_filename=product_filename,
        bundle_paths=bundle_paths,
        factor_snapshot_path=snapshot_path,
        factor_snapshot_identity=snapshot_identity,
    )


def _json_safe(value: Any) -> Any:
    """Convert NumPy values and non-finite diagnostics to portable JSON."""

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                _json_safe(dict(payload)),
                stream,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _csv_value(value: Any) -> str | int | float:
    if isinstance(value, (np.bool_, bool)):
        return int(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return "" if not math.isfinite(numeric) else numeric
    return str(value)


def _atomic_reference_lightcurve_csv(
    path: Path,
    *,
    reference: ReferencePhotometryResult,
    residual: InjectedModelResidualResult | None,
) -> Path:
    if reference.exposure_seconds is None:
        raise StandardStampAnalysisError(
            "formal reference photometry must retain exposure_seconds"
        )
    if (
        reference.raw_frame_start_index is None
        or reference.raw_frame_stop_index_exclusive is None
    ):
        raise StandardStampAnalysisError(
            "formal reference photometry must retain raw-frame intervals"
        )
    fields = [
        "time_start_seconds",
        "exposure_seconds",
        "raw_frame_start_index",
        "raw_frame_stop_index_exclusive",
        "flux_derived_e",
        "aperture_valid",
        "aperture_usable_pixel_count",
        "aperture_invalid_pixel_count",
    ]
    if residual is not None:
        fields.extend(
            (
                "injected_raw_factor_sum",
                "model_flux_e",
                "model_residual_e",
                "model_residual_ppm",
                "injected_model_valid",
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            for index in range(reference.time_seconds.size):
                row: dict[str, str | int | float] = {
                    "time_start_seconds": _csv_value(reference.time_seconds[index]),
                    "exposure_seconds": _csv_value(reference.exposure_seconds[index]),
                    "raw_frame_start_index": _csv_value(
                        reference.raw_frame_start_index[index]
                    ),
                    "raw_frame_stop_index_exclusive": _csv_value(
                        reference.raw_frame_stop_index_exclusive[index]
                    ),
                    "flux_derived_e": _csv_value(reference.flux_e[index]),
                    "aperture_valid": _csv_value(reference.aperture_valid[index]),
                    "aperture_usable_pixel_count": _csv_value(
                        reference.aperture_usable_pixel_count[index]
                    ),
                    "aperture_invalid_pixel_count": int(
                        reference.aperture_pixel_count
                        - reference.aperture_usable_pixel_count[index]
                    ),
                }
                if residual is not None:
                    row.update(
                        {
                            "injected_raw_factor_sum": _csv_value(
                                residual.raw_factor_sum[index]
                            ),
                            "model_flux_e": _csv_value(residual.fitted_flux_e[index]),
                            "model_residual_e": _csv_value(residual.residual_e[index]),
                            "model_residual_ppm": _csv_value(
                                residual.residual_ppm[index]
                            ),
                            "injected_model_valid": _csv_value(
                                residual.valid_mask[index]
                            ),
                        }
                    )
                writer.writerow(row)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _quality_summary(reference: ReferencePhotometryResult) -> dict[str, Any]:
    valid = np.asarray(reference.aperture_valid, dtype=bool)
    usable = np.asarray(reference.aperture_usable_pixel_count, dtype=np.int64)
    return {
        "cadence_count": int(valid.size),
        "valid_cadence_count": int(np.count_nonzero(valid)),
        "invalid_cadence_count": int(np.count_nonzero(~valid)),
        "aperture_pixel_count": int(reference.aperture_pixel_count),
        "minimum_usable_pixel_count": int(np.min(usable)),
        "maximum_usable_pixel_count": int(np.max(usable)),
        "quality_policy": "invalidate_whole_fixed_aperture_cadence",
    }


def _analysis_manifest(
    *,
    resolved: StandardStampAnalysisInput,
    request: StandardStampAnalysisRequest,
    reference: ReferencePhotometryResult,
    residual: InjectedModelResidualResult | None,
    bundle_receipts: Sequence[_DeliveryBundleReceipt],
    lightcurve_path: Path,
) -> dict[str, Any]:
    delivery = resolved.production_manifest["delivery"]
    assert isinstance(delivery, Mapping)  # guaranteed during discovery
    ordinary_label = (
        "undetrended_astrophysical_plus_instrument_legacy_compatible_diagnostic"
        if resolved.case == "injected"
        else "undetrended_static_source_legacy_compatible_diagnostic"
    )
    residual_label = (
        None
        if residual is None
        else "undetrended_injected_model_residual_legacy_compatible_diagnostic"
    )
    return {
        "schema_id": STANDARD_STAMP_ANALYSIS_SCHEMA_ID,
        "schema_version": STANDARD_STAMP_ANALYSIS_SCHEMA_VERSION,
        "complete": True,
        "production_manifest_path": str(resolved.production_manifest_path),
        "production_manifest_identity": file_identity(
            resolved.production_manifest_path
        ),
        "run_id": str(resolved.production_manifest["run_id"]),
        "source_id": str(resolved.source_id),
        "source_id_int64": resolved.source_id,
        "case": resolved.case,
        "observation_product": "final_dn",
        "background_realization_used": False,
        "reference_lightcurve": {
            "schema_id": STANDARD_STAMP_ANALYSIS_LIGHTCURVE_SCHEMA_ID,
            "path": lightcurve_path.name,
            "format": "csv",
            "columns": [
                "time_start_seconds",
                "exposure_seconds",
                "raw_frame_start_index",
                "raw_frame_stop_index_exclusive",
                "flux_derived_e",
                "aperture_valid",
                "aperture_usable_pixel_count",
                "aperture_invalid_pixel_count",
                *(
                    []
                    if residual is None
                    else [
                        "injected_raw_factor_sum",
                        "model_flux_e",
                        "model_residual_e",
                        "model_residual_ppm",
                        "injected_model_valid",
                    ]
                ),
            ],
            "electron_semantics": (
                "derived=((final_dn-bias_level_sum_dn-column_noise_sum_dn_by_x)"
                "*gain_e_per_dn)-background_expectation_e"
            ),
        },
        "delivery": {
            "product_filename": resolved.product_filename,
            "cadence_seconds": resolved.cadence_seconds,
            "raw_exposure_seconds": float(delivery["raw_exposure_seconds"]),
            "bundle_count": len(resolved.bundle_paths),
            "bundle_receipt_policy": (
                "full_file_sha256_and_schema_validation_before_reduction; "
                "file_state_unchanged_through_reduction"
            ),
            "bundle_paths_relative_to_run_root": [
                path.relative_to(resolved.run_root).as_posix()
                for path in resolved.bundle_paths
            ],
            "bundle_receipts": [
                {
                    "path_relative_to_run_root": receipt.path.relative_to(
                        resolved.run_root
                    ).as_posix(),
                    "size_bytes": receipt.size_bytes,
                    "sha256": receipt.sha256,
                    "validation": dict(receipt.validation),
                }
                for receipt in bundle_receipts
            ],
        },
        "quality": _quality_summary(reference),
        "ordinary_cdpp_label": ordinary_label,
        "injected_model_residual_cdpp_label": residual_label,
        "observed_cdpp": {
            str(window): metric.to_dict()
            for window, metric in reference.cdpp_by_window_minutes.items()
        },
        "reference_photometry": reference.to_dict(),
        "injected_model_residual": (None if residual is None else residual.to_dict()),
        "frozen_variability": (
            None
            if resolved.factor_snapshot_path is None
            else {
                "path_relative_to_run_root": resolved.factor_snapshot_path.relative_to(
                    resolved.run_root
                ).as_posix(),
                "identity": dict(resolved.factor_snapshot_identity or {}),
                "time_alignment": "simulation_raw_frame_index",
            }
        ),
        "cdpp": {
            "windows_minutes": list(request.cdpp_windows_minutes),
            "bin_origin_seconds": request.bin_origin_seconds,
            "detrending": "not_applied",
            "legacy_standard_pca_sg_pipeline": "not_applied",
            "observed_metric": "legacy_mean_absolute_deviation_times_1.4826",
            "residual_metric": "injected_model_residual_mad_times_1.4826",
        },
        "legacy_compatibility": {
            "legacy_pickle_pca_sg_used": False,
            "mad_statistic_only": True,
            "note": (
                "The MAD normalization is legacy-compatible, but this formal "
                "path does not call legacy bin_lcs, pickle, PCA, or SG routines."
            ),
        },
    }


def run_standard_stamp_analysis_v1(
    request: StandardStampAnalysisRequest,
) -> StandardStampAnalysisResult:
    """Write the formal standard reference curve and JSON analysis manifest.

    A manifest is written only after the reference CSV has been atomically
    published.  Existing analysis files are immutable unless the caller
    explicitly asks for ``overwrite=True``.
    """

    resolved = discover_standard_stamp_analysis_input(request)
    output_dir = Path(request.output_dir)
    lightcurve_path = output_dir / "reference_lightcurve.csv"
    analysis_manifest_path = output_dir / "analysis_manifest.json"
    existing = tuple(
        path for path in (lightcurve_path, analysis_manifest_path) if path.exists()
    )
    if existing and not request.overwrite:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            "standard analysis output already exists; use a distinct output_dir "
            f"or overwrite=True: {names}"
        )

    bundle_stat_fingerprints = _bundle_stat_fingerprints(resolved.bundle_paths)
    bundle_receipts = _validate_and_identify_bundles(resolved.bundle_paths)
    reference = reduce_stamp_delivery_series_v1(
        resolved.bundle_paths,
        cdpp_windows_minutes=request.cdpp_windows_minutes,
        bin_origin_seconds=request.bin_origin_seconds,
        batch_frames=request.batch_frames,
    )
    _assert_bundle_stat_fingerprints_unchanged(bundle_stat_fingerprints)
    residual: InjectedModelResidualResult | None = None
    if resolved.case == "injected":
        assert resolved.factor_snapshot_path is not None  # enforced by discovery
        if not _same_file_content_identity(
            file_identity(resolved.factor_snapshot_path),
            resolved.factor_snapshot_identity or {},
        ):
            raise StandardStampAnalysisError(
                "frozen factor snapshot identity changed during analysis"
            )
        snapshot = read_galaxy_factor_snapshot(resolved.factor_snapshot_path)
        residual = compute_injected_model_residual_v1(
            reference,
            raw_frame_factors=snapshot.factors,
            raw_exposure_seconds=resolved.time_plan.raw_exposure_seconds,
            windows_minutes=request.cdpp_windows_minutes,
            bin_origin_seconds=request.bin_origin_seconds,
        )

    _atomic_reference_lightcurve_csv(
        lightcurve_path,
        reference=reference,
        residual=residual,
    )
    manifest = _analysis_manifest(
        resolved=resolved,
        request=request,
        reference=reference,
        residual=residual,
        bundle_receipts=bundle_receipts,
        lightcurve_path=lightcurve_path,
    )
    _atomic_json(analysis_manifest_path, manifest)
    return StandardStampAnalysisResult(
        analysis_manifest_path=analysis_manifest_path,
        reference_lightcurve_path=lightcurve_path,
        source_id=resolved.source_id,
        case=resolved.case,
        cadence_seconds=resolved.cadence_seconds,
        cadence_count=int(reference.aperture_valid.size),
        valid_cadence_count=int(np.count_nonzero(reference.aperture_valid)),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-manifest", required=True)
    parser.add_argument("--source-id", required=True, type=int)
    parser.add_argument("--case", required=True, choices=("static", "injected"))
    parser.add_argument("--cadence-seconds", required=True, type=float)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--cdpp-window-minutes",
        nargs="+",
        type=int,
        default=STANDARD_CDPP_WINDOWS_MINUTES,
    )
    parser.add_argument("--bin-origin-seconds", type=float, default=0.0)
    parser.add_argument("--batch-frames", type=int, default=4_096)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the maintained CLI, printing only a compact JSON completion record."""

    args = _parser().parse_args(None if argv is None else list(argv))
    result = run_standard_stamp_analysis_v1(
        StandardStampAnalysisRequest(
            production_manifest_path=args.production_manifest,
            source_id=args.source_id,
            case=args.case,
            cadence_seconds=args.cadence_seconds,
            output_dir=args.output_dir,
            cdpp_windows_minutes=tuple(args.cdpp_window_minutes),
            bin_origin_seconds=args.bin_origin_seconds,
            batch_frames=args.batch_frames,
            overwrite=args.overwrite,
        )
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by CLI invocation.
    raise SystemExit(main())


__all__ = [
    "STANDARD_STAMP_ANALYSIS_LIGHTCURVE_SCHEMA_ID",
    "STANDARD_STAMP_ANALYSIS_SCHEMA_ID",
    "STANDARD_STAMP_ANALYSIS_SCHEMA_VERSION",
    "StandardStampAnalysisError",
    "StandardStampAnalysisInput",
    "StandardStampAnalysisNotReadyError",
    "StandardStampAnalysisRequest",
    "StandardStampAnalysisResult",
    "discover_standard_stamp_analysis_input",
    "main",
    "run_standard_stamp_analysis_v1",
]
