"""Metadata-level completion gate for formal Galaxy stamp campaigns.

The HDF5 writer already performs a bounded, full-payload validation before it
atomically publishes each bundle.  Re-reading every image plane for a whole
campaign would therefore duplicate multiple terabytes of I/O.  This module
adds the complementary campaign gate: it derives the complete target × shard
× product matrix from the frozen production manifest, then verifies every
published member's immutable delivery header, compact time coordinates, and
caller provenance against that matrix.

The resulting receipt is deliberately useful in two states.  During running
production it reports missing final members without treating in-progress
``.partial`` files as science input.  At hand-off, ``ready=True`` proves that
the complete final matrix has no missing, malformed, unexpected, or partial
members.  It is not a substitute for the per-bundle streaming validation or
the SHA-256 input receipts recorded by standard 60-second photometry.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from typing import Any, Literal
import uuid

import numpy as np

from .galaxy_stamp_production import (
    GALAXY_STAMP_PRODUCTION_SCHEMA_ID,
    GALAXY_STAMP_PRODUCTION_SCHEMA_VERSION,
)
from .stamp_inputs import file_identity
from .stamp_delivery import (
    STAMP_DELIVERY_OBSERVATION_PRODUCT,
    STAMP_DELIVERY_SCHEMA_ID,
    STAMP_DELIVERY_SCHEMA_VERSION,
)
from .time_shards import ContinuousTimeShard, ContinuousTimeShardPlan


GALAXY_CAMPAIGN_DELIVERY_QC_SCHEMA_ID = "et_mainsim.galaxy_campaign_delivery_qc.v1"
GALAXY_CAMPAIGN_DELIVERY_QC_SCHEMA_VERSION = 1

CampaignCase = Literal["static", "injected"]


class GalaxyCampaignDeliveryQCError(ValueError):
    """Raised when the frozen campaign manifest itself is invalid."""


def _decode_scalar(value: Any) -> Any:
    """Convert scalar HDF5/NumPy values into ordinary JSON-like values."""

    if isinstance(value, np.ndarray):
        if value.shape != ():
            return value
        value = value.reshape(()).item()
    elif isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8")
    return value


def _strict_source_id(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise GalaxyCampaignDeliveryQCError(
            f"{name} must be a non-negative signed int64"
        )
    try:
        source_id = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise GalaxyCampaignDeliveryQCError(
            f"{name} must be a non-negative signed int64"
        ) from error
    if source_id < 0 or source_id > int(np.iinfo(np.int64).max):
        raise GalaxyCampaignDeliveryQCError(
            f"{name} must be a non-negative signed int64"
        )
    return source_id


def _normalise_case(value: object) -> CampaignCase:
    if value not in {"static", "injected"}:
        raise GalaxyCampaignDeliveryQCError(
            "case must be exactly 'static' or 'injected'"
        )
    return value  # type: ignore[return-value]


def _positive_float(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise GalaxyCampaignDeliveryQCError(f"{name} must be finite and positive")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise GalaxyCampaignDeliveryQCError(
            f"{name} must be finite and positive"
        ) from error
    if not math.isfinite(result) or result <= 0.0:
        raise GalaxyCampaignDeliveryQCError(f"{name} must be finite and positive")
    return result


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise GalaxyCampaignDeliveryQCError(f"{name} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise GalaxyCampaignDeliveryQCError(
            f"{name} must be a positive integer"
        ) from error
    if result <= 0 or result != value:
        raise GalaxyCampaignDeliveryQCError(f"{name} must be a positive integer")
    return result


def _json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise GalaxyCampaignDeliveryQCError(
            f"{label} does not exist: {path}"
        ) from error
    except json.JSONDecodeError as error:
        raise GalaxyCampaignDeliveryQCError(
            f"{label} is not valid JSON: {path}"
        ) from error
    if not isinstance(payload, dict):
        raise GalaxyCampaignDeliveryQCError(f"{label} must be a JSON object: {path}")
    return payload


def _resolve_relative_resource(
    run_root: Path,
    relative_path: Any,
    *,
    label: str,
) -> Path:
    if not isinstance(relative_path, str) or not relative_path.strip():
        raise GalaxyCampaignDeliveryQCError(f"{label} requires a relative path")
    candidate = Path(relative_path)
    if candidate.is_absolute():
        raise GalaxyCampaignDeliveryQCError(f"{label} path must be relative")
    resolved_root = run_root.resolve()
    resolved = (resolved_root / candidate).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise GalaxyCampaignDeliveryQCError(
            f"{label} relative path escapes prepared run root"
        ) from error
    return resolved


def _same_file_identity(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    """Compare a frozen content identity while allowing a relocated path."""

    return actual.get("sha256") == expected.get("sha256") and actual.get(
        "size_bytes"
    ) == expected.get("size_bytes")


@dataclass(frozen=True)
class GalaxyCampaignDeliveryQCRequest:
    """Explicit metadata audit request for one frozen Galaxy delivery case."""

    production_manifest_path: Path | str
    case: CampaignCase | str = "injected"

    def __post_init__(self) -> None:
        manifest = Path(self.production_manifest_path).expanduser().resolve()
        object.__setattr__(self, "production_manifest_path", manifest)
        object.__setattr__(self, "case", _normalise_case(self.case))


@dataclass(frozen=True)
class _ExpectedBundle:
    source_id: int
    shard: ContinuousTimeShard
    product_name: str
    product_kind: Literal["raw", "coadd"]
    coadd_factor: int
    filename: str
    path: Path

    @property
    def frame_count(self) -> int:
        if self.coadd_factor == 1:
            return self.shard.raw_frame_count
        return self.shard.coadd_count(self.coadd_factor)

    def record(self) -> dict[str, Any]:
        return {
            "source_id": str(self.source_id),
            "shard_id": int(self.shard.shard_id),
            "product": self.product_name,
            "path": str(self.path),
        }


@dataclass(frozen=True)
class GalaxyCampaignDeliveryQCResult:
    """Compact, serializable result of a full delivery-matrix metadata sweep."""

    production_manifest_path: Path
    run_root: Path
    run_id: str
    case: CampaignCase
    target_count: int
    shard_count: int
    expected_bundle_count: int
    valid_bundle_count: int
    missing_bundles: tuple[Mapping[str, Any], ...]
    invalid_bundles: tuple[Mapping[str, Any], ...]
    unexpected_final_bundles: tuple[str, ...]
    partial_artifacts: tuple[str, ...]
    product_summaries: Mapping[str, Mapping[str, int]]
    time_plan: ContinuousTimeShardPlan
    manifest_identity: Mapping[str, Any]
    time_plan_identity: Mapping[str, Any]

    @property
    def missing_bundle_count(self) -> int:
        return len(self.missing_bundles)

    @property
    def invalid_bundle_count(self) -> int:
        return len(self.invalid_bundles)

    @property
    def ready(self) -> bool:
        return (
            self.missing_bundle_count == 0
            and self.invalid_bundle_count == 0
            and not self.unexpected_final_bundles
            and not self.partial_artifacts
            and self.valid_bundle_count == self.expected_bundle_count
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_id": GALAXY_CAMPAIGN_DELIVERY_QC_SCHEMA_ID,
            "schema_version": GALAXY_CAMPAIGN_DELIVERY_QC_SCHEMA_VERSION,
            "status": "ready" if self.ready else "incomplete_or_invalid",
            "ready": self.ready,
            "production_manifest_path": str(self.production_manifest_path),
            "run_root": str(self.run_root),
            "run_id": self.run_id,
            "case": self.case,
            "inspection_mode": (
                "metadata_schema_time_axis_and_provenance_only; "
                "per_bundle_full_payload_validation_occurs_before_atomic_publish"
            ),
            "manifest_identity": dict(self.manifest_identity),
            "time_plan_identity": dict(self.time_plan_identity),
            "coverage": {
                "target_count": self.target_count,
                "shard_count": self.shard_count,
                "raw_frame_interval": {
                    "start_index": self.time_plan.raw_start_index,
                    "stop_index": self.time_plan.raw_stop_index,
                },
                "accepted_raw_frame_interval": {
                    "start_index": self.time_plan.accepted_raw_start_index,
                    "stop_index": self.time_plan.accepted_raw_stop_index,
                },
                "accepted_raw_frame_count_per_target": (
                    self.time_plan.accepted_raw_frame_count
                ),
                "raw_exposure_seconds": self.time_plan.raw_exposure_seconds,
                "coadd_sizes": list(self.time_plan.coadd_sizes),
            },
            "expected_bundle_count": self.expected_bundle_count,
            "valid_bundle_count": self.valid_bundle_count,
            "missing_bundle_count": self.missing_bundle_count,
            "invalid_bundle_count": self.invalid_bundle_count,
            "unexpected_final_bundle_count": len(self.unexpected_final_bundles),
            "partial_artifact_count": len(self.partial_artifacts),
            "products": {
                name: dict(summary)
                for name, summary in sorted(self.product_summaries.items())
            },
            "missing_bundles": [dict(record) for record in self.missing_bundles],
            "invalid_bundles": [dict(record) for record in self.invalid_bundles],
            "unexpected_final_bundles": list(self.unexpected_final_bundles),
            "partial_artifacts": list(self.partial_artifacts),
        }


def _load_campaign(
    manifest_path: Path,
) -> tuple[
    Path, dict[str, Any], ContinuousTimeShardPlan, tuple[int, ...], tuple[int, int]
]:
    manifest = _json_object(manifest_path, label="production manifest")
    if manifest.get("schema_id") != GALAXY_STAMP_PRODUCTION_SCHEMA_ID:
        raise GalaxyCampaignDeliveryQCError("unsupported Galaxy production manifest")
    if int(manifest.get("schema_version", 0)) != GALAXY_STAMP_PRODUCTION_SCHEMA_VERSION:
        raise GalaxyCampaignDeliveryQCError(
            "unsupported Galaxy production manifest version"
        )
    run_root = manifest_path.parent.resolve()
    if manifest.get("observation_product") != STAMP_DELIVERY_OBSERVATION_PRODUCT:
        raise GalaxyCampaignDeliveryQCError(
            "production manifest observation_product must be final_dn"
        )
    if manifest.get("background_realization_delivered") is not False:
        raise GalaxyCampaignDeliveryQCError(
            "production manifest must not claim a background realization delivery"
        )
    delivery = manifest.get("delivery")
    if not isinstance(delivery, Mapping):
        raise GalaxyCampaignDeliveryQCError(
            "production manifest delivery must be an object"
        )
    time_plan_path = _resolve_relative_resource(
        run_root,
        delivery.get("time_plan_relative_path"),
        label="time shard plan",
    )
    expected_time_identity = delivery.get("time_plan_identity")
    if not isinstance(expected_time_identity, Mapping) or not _same_file_identity(
        file_identity(time_plan_path), expected_time_identity
    ):
        raise GalaxyCampaignDeliveryQCError(
            "time shard plan identity changed after production preparation"
        )
    try:
        time_plan = ContinuousTimeShardPlan.from_manifest_dict(
            _json_object(time_plan_path, label="time shard plan")
        )
    except ValueError as error:
        raise GalaxyCampaignDeliveryQCError("invalid frozen time shard plan") from error
    raw_seconds = _positive_float(
        delivery.get("raw_exposure_seconds"), name="delivery.raw_exposure_seconds"
    )
    if not math.isclose(raw_seconds, time_plan.raw_exposure_seconds, abs_tol=1e-12):
        raise GalaxyCampaignDeliveryQCError(
            "delivery raw_exposure_seconds conflicts with frozen time shard plan"
        )
    coadd_sizes_value = delivery.get("coadd_sizes")
    if not isinstance(coadd_sizes_value, list):
        raise GalaxyCampaignDeliveryQCError("delivery.coadd_sizes must be a list")
    coadd_sizes = tuple(
        sorted(
            _positive_int(value, name="delivery.coadd_sizes")
            for value in coadd_sizes_value
        )
    )
    if (
        len(set(coadd_sizes)) != len(coadd_sizes)
        or coadd_sizes != time_plan.coadd_sizes
    ):
        raise GalaxyCampaignDeliveryQCError(
            "delivery coadd_sizes conflict with frozen time shard plan"
        )
    cadence_seconds = delivery.get("cadence_seconds")
    if not isinstance(cadence_seconds, list):
        raise GalaxyCampaignDeliveryQCError("delivery.cadence_seconds must be a list")
    expected_cadence_seconds = [
        size * time_plan.raw_exposure_seconds for size in time_plan.coadd_sizes
    ]
    if len(cadence_seconds) != len(expected_cadence_seconds) or any(
        not math.isclose(
            _positive_float(value, name="delivery.cadence_seconds"),
            expected,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for value, expected in zip(
            cadence_seconds, expected_cadence_seconds, strict=True
        )
    ):
        raise GalaxyCampaignDeliveryQCError(
            "delivery cadence_seconds conflict with frozen time shard plan"
        )
    stamp_shape_value = delivery.get("stamp_shape")
    if not isinstance(stamp_shape_value, list) or len(stamp_shape_value) != 2:
        raise GalaxyCampaignDeliveryQCError(
            "delivery.stamp_shape must have two dimensions"
        )
    stamp_shape = tuple(
        _positive_int(value, name="delivery.stamp_shape") for value in stamp_shape_value
    )
    return run_root, manifest, time_plan, coadd_sizes, stamp_shape


def _target_ids(manifest: Mapping[str, Any]) -> tuple[int, ...]:
    targets = manifest.get("targets")
    if not isinstance(targets, list) or not targets:
        raise GalaxyCampaignDeliveryQCError(
            "production manifest targets must be non-empty"
        )
    values: list[int] = []
    for target in targets:
        if not isinstance(target, Mapping):
            raise GalaxyCampaignDeliveryQCError(
                "production manifest target must be an object"
            )
        values.append(
            _strict_source_id(target.get("source_id_int64"), name="target.source_id")
        )
    if len(set(values)) != len(values):
        raise GalaxyCampaignDeliveryQCError(
            "production manifest target IDs must be unique"
        )
    return tuple(values)


def _product_specs(
    *,
    time_plan: ContinuousTimeShardPlan,
) -> tuple[tuple[str, Literal["raw", "coadd"], int, str], ...]:
    products: list[tuple[str, Literal["raw", "coadd"], int, str]] = [
        ("raw", "raw", 1, "raw.h5")
    ]
    for factor in time_plan.coadd_sizes:
        cadence_seconds = factor * time_plan.raw_exposure_seconds
        if not math.isclose(cadence_seconds, round(cadence_seconds), abs_tol=1e-12):
            raise GalaxyCampaignDeliveryQCError(
                "formal delivery filenames require integral cadence seconds"
            )
        products.append(
            (
                f"coadd_{int(round(cadence_seconds))}s",
                "coadd",
                factor,
                f"coadd_{int(round(cadence_seconds))}s.h5",
            )
        )
    return tuple(products)


def _expected_bundles(
    *,
    run_root: Path,
    case: CampaignCase,
    source_ids: tuple[int, ...],
    time_plan: ContinuousTimeShardPlan,
) -> tuple[_ExpectedBundle, ...]:
    bundles: list[_ExpectedBundle] = []
    for source_id in source_ids:
        delivery_root = (
            run_root / "cases" / case / "stamps" / f"target_{source_id}" / "delivery"
        )
        for shard in time_plan.shards:
            for product_name, product_kind, factor, filename in _product_specs(
                time_plan=time_plan
            ):
                bundles.append(
                    _ExpectedBundle(
                        source_id=source_id,
                        shard=shard,
                        product_name=product_name,
                        product_kind=product_kind,
                        coadd_factor=factor,
                        filename=filename,
                        path=delivery_root / f"shard_{shard.shard_id:05d}" / filename,
                    )
                )
    return tuple(bundles)


def _h5_attribute(handle: Any, name: str) -> Any:
    if name not in handle.attrs:
        raise GalaxyCampaignDeliveryQCError(
            f"delivery root attribute is missing: {name}"
        )
    return _decode_scalar(handle.attrs[name])


def _load_json_dataset(handle: Any, name: str) -> Mapping[str, Any]:
    if name not in handle:
        raise GalaxyCampaignDeliveryQCError(f"delivery dataset is missing: {name}")
    raw = _decode_scalar(handle[name][()])
    if not isinstance(raw, str):
        raise GalaxyCampaignDeliveryQCError(f"delivery {name} must be JSON text")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise GalaxyCampaignDeliveryQCError(
            f"delivery {name} is not valid JSON"
        ) from error
    if not isinstance(payload, Mapping):
        raise GalaxyCampaignDeliveryQCError(f"delivery {name} must be a JSON object")
    return payload


def _expect_array(handle: Any, name: str, *, shape: tuple[int, ...]) -> np.ndarray:
    if name not in handle:
        raise GalaxyCampaignDeliveryQCError(f"delivery dataset is missing: {name}")
    dataset = handle[name]
    if tuple(int(size) for size in dataset.shape) != shape:
        raise GalaxyCampaignDeliveryQCError(
            f"delivery {name} has shape {tuple(dataset.shape)}, expected {shape}"
        )
    return np.asarray(dataset)


def _validate_delivery_header(
    path: Path,
    *,
    expected: _ExpectedBundle,
    run_id: str,
    case: CampaignCase,
    stamp_shape: tuple[int, int],
    raw_exposure_seconds: float,
) -> None:
    """Inspect only compact metadata and timing vectors for one final member."""

    try:
        import h5py
    except ImportError as error:  # pragma: no cover - packaging guard
        raise RuntimeError("h5py is required for formal campaign QC") from error

    expected_starts = np.arange(
        expected.shard.raw_start_index,
        expected.shard.raw_stop_index,
        expected.coadd_factor,
        dtype=np.int64,
    )
    expected_stops = expected_starts + expected.coadd_factor
    expected_times = expected_starts.astype(np.float64) * raw_exposure_seconds
    expected_exposure = np.full(
        expected.frame_count,
        expected.coadd_factor * raw_exposure_seconds,
        dtype=np.float64,
    )
    with h5py.File(path, "r") as handle:
        if _h5_attribute(handle, "schema_id") != STAMP_DELIVERY_SCHEMA_ID:
            raise GalaxyCampaignDeliveryQCError("delivery schema_id is unsupported")
        if (
            int(_h5_attribute(handle, "schema_version"))
            != STAMP_DELIVERY_SCHEMA_VERSION
        ):
            raise GalaxyCampaignDeliveryQCError(
                "delivery schema_version is unsupported"
            )
        if _h5_attribute(handle, "complete") is not True:
            raise GalaxyCampaignDeliveryQCError("delivery bundle is not complete")
        if (
            _h5_attribute(handle, "observation_product")
            != STAMP_DELIVERY_OBSERVATION_PRODUCT
        ):
            raise GalaxyCampaignDeliveryQCError(
                "delivery observation_product must be final_dn"
            )
        if _h5_attribute(handle, "background_realization_used") is not False:
            raise GalaxyCampaignDeliveryQCError(
                "delivery background_realization_used must be false"
            )
        if _h5_attribute(handle, "product_kind") != expected.product_kind:
            raise GalaxyCampaignDeliveryQCError(
                "delivery product_kind conflicts with path"
            )
        if int(_h5_attribute(handle, "coadd_factor")) != expected.coadd_factor:
            raise GalaxyCampaignDeliveryQCError(
                "delivery coadd_factor conflicts with path"
            )

        if "final_dn" not in handle:
            raise GalaxyCampaignDeliveryQCError("delivery dataset is missing: final_dn")
        final_dn = handle["final_dn"]
        expected_shape = (expected.frame_count, *stamp_shape)
        if tuple(int(size) for size in final_dn.shape) != expected_shape:
            raise GalaxyCampaignDeliveryQCError(
                f"delivery final_dn has shape {tuple(final_dn.shape)}, expected {expected_shape}"
            )
        final_dtype = np.dtype(final_dn.dtype)
        if final_dtype.kind != "u":
            raise GalaxyCampaignDeliveryQCError(
                "delivery final_dn must use unsigned DN"
            )
        if expected.product_kind == "coadd" and final_dtype != np.dtype(np.uint64):
            raise GalaxyCampaignDeliveryQCError(
                "delivery coadd final_dn must use uint64"
            )

        starts = _expect_array(
            handle,
            "raw_frame_start_index",
            shape=(expected.frame_count,),
        )
        stops = _expect_array(
            handle,
            "raw_frame_stop_index_exclusive",
            shape=(expected.frame_count,),
        )
        times = _expect_array(
            handle,
            "time_start_seconds",
            shape=(expected.frame_count,),
        )
        exposures = _expect_array(
            handle,
            "exposure_seconds",
            shape=(expected.frame_count,),
        )
        if not np.array_equal(starts, expected_starts):
            raise GalaxyCampaignDeliveryQCError(
                "delivery raw_frame_start_index conflicts with frozen time plan"
            )
        if not np.array_equal(stops, expected_stops):
            raise GalaxyCampaignDeliveryQCError(
                "delivery raw_frame_stop_index_exclusive conflicts with frozen time plan"
            )
        if not np.array_equal(times, expected_times):
            raise GalaxyCampaignDeliveryQCError(
                "delivery time_start_seconds conflicts with frozen time plan"
            )
        if not np.array_equal(exposures, expected_exposure):
            raise GalaxyCampaignDeliveryQCError(
                "delivery exposure_seconds conflicts with frozen time plan"
            )

        delivery_manifest = _load_json_dataset(handle, "manifest_json")
        if (
            _strict_source_id(
                delivery_manifest.get("target_source_id_int64"),
                name="delivery.target_source_id",
            )
            != expected.source_id
        ):
            raise GalaxyCampaignDeliveryQCError(
                "delivery target source conflicts with path"
            )
        if delivery_manifest.get("product_kind") != expected.product_kind:
            raise GalaxyCampaignDeliveryQCError(
                "delivery manifest product_kind conflicts with path"
            )
        if int(delivery_manifest.get("coadd_factor", -1)) != expected.coadd_factor:
            raise GalaxyCampaignDeliveryQCError(
                "delivery manifest coadd_factor conflicts with path"
            )
        delivered_shape = delivery_manifest.get("stamp_shape")
        if delivered_shape != list(stamp_shape):
            raise GalaxyCampaignDeliveryQCError(
                "delivery manifest stamp_shape conflicts with frozen manifest"
            )
        caller = delivery_manifest.get("caller_manifest")
        if not isinstance(caller, Mapping):
            raise GalaxyCampaignDeliveryQCError("delivery caller_manifest is missing")
        if str(caller.get("run_id")) != run_id:
            raise GalaxyCampaignDeliveryQCError("delivery caller run_id conflicts")
        if caller.get("case") != case:
            raise GalaxyCampaignDeliveryQCError("delivery caller case conflicts")
        provenance = _load_json_dataset(handle, "provenance_json")
        if provenance.get("observation_product") != STAMP_DELIVERY_OBSERVATION_PRODUCT:
            raise GalaxyCampaignDeliveryQCError(
                "delivery provenance observation_product must be final_dn"
            )
        if provenance.get("background_realization_used") is not False:
            raise GalaxyCampaignDeliveryQCError(
                "delivery provenance background_realization_used must be false"
            )
        time_shard = delivery_manifest.get("time_shard")
        interval = (
            time_shard.get("raw_frame_interval")
            if isinstance(time_shard, Mapping)
            else None
        )
        if not isinstance(interval, Mapping):
            raise GalaxyCampaignDeliveryQCError(
                "delivery time_shard metadata is missing"
            )
        if (
            _strict_source_id(interval.get("start_index"), name="time_shard.start")
            != expected.shard.raw_start_index
            or _strict_source_id(interval.get("stop_index"), name="time_shard.stop")
            != expected.shard.raw_stop_index
        ):
            raise GalaxyCampaignDeliveryQCError(
                "delivery time_shard conflicts with frozen time plan"
            )


def _unexpected_files(
    *,
    run_root: Path,
    case: CampaignCase,
    expected_paths: set[Path],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    stamps_root = run_root / "cases" / case / "stamps"
    if not stamps_root.exists():
        return (), ()
    final_members = tuple(
        sorted(
            str(path)
            for path in stamps_root.rglob("*.h5")
            if path.resolve() not in expected_paths
        )
    )
    partial_members = tuple(
        sorted(str(path) for path in stamps_root.rglob("*.partial"))
    )
    return final_members, partial_members


def audit_galaxy_campaign_delivery_v1(
    request: GalaxyCampaignDeliveryQCRequest,
) -> GalaxyCampaignDeliveryQCResult:
    """Audit all expected final Galaxy delivery members without image-cube I/O.

    Missing files are represented in the result rather than raised, allowing a
    running campaign to expose progress safely.  A malformed manifest or an
    existing member whose metadata cannot meet the frozen contract is recorded
    as an invalid delivery member.
    """

    if not isinstance(request, GalaxyCampaignDeliveryQCRequest):
        raise TypeError("request must be GalaxyCampaignDeliveryQCRequest")
    run_root, manifest, time_plan, _coadd_sizes, stamp_shape = _load_campaign(
        request.production_manifest_path
    )
    run_id_value = manifest.get("run_id")
    if not isinstance(run_id_value, str) or not run_id_value.strip():
        raise GalaxyCampaignDeliveryQCError(
            "production manifest run_id must be non-empty"
        )
    run_id = run_id_value
    source_ids = _target_ids(manifest)
    expected = _expected_bundles(
        run_root=run_root,
        case=request.case,
        source_ids=source_ids,
        time_plan=time_plan,
    )
    product_summaries: dict[str, dict[str, int]] = {}
    for product_name, _kind, _factor, _filename in _product_specs(time_plan=time_plan):
        product_summaries[product_name] = {
            "expected_bundle_count": 0,
            "valid_bundle_count": 0,
            "missing_bundle_count": 0,
            "invalid_bundle_count": 0,
        }

    valid_count = 0
    missing: list[Mapping[str, Any]] = []
    invalid: list[Mapping[str, Any]] = []
    for bundle in expected:
        summary = product_summaries[bundle.product_name]
        summary["expected_bundle_count"] += 1
        if not bundle.path.is_file():
            summary["missing_bundle_count"] += 1
            missing.append(bundle.record())
            continue
        try:
            _validate_delivery_header(
                bundle.path,
                expected=bundle,
                run_id=run_id,
                case=request.case,
                stamp_shape=stamp_shape,
                raw_exposure_seconds=time_plan.raw_exposure_seconds,
            )
        except (OSError, TypeError, ValueError) as error:
            summary["invalid_bundle_count"] += 1
            record = bundle.record()
            record["error"] = str(error)
            invalid.append(record)
            continue
        summary["valid_bundle_count"] += 1
        valid_count += 1

    unexpected, partials = _unexpected_files(
        run_root=run_root,
        case=request.case,
        expected_paths={bundle.path.resolve() for bundle in expected},
    )
    return GalaxyCampaignDeliveryQCResult(
        production_manifest_path=request.production_manifest_path,
        run_root=run_root,
        run_id=run_id,
        case=request.case,
        target_count=len(source_ids),
        shard_count=len(time_plan.shards),
        expected_bundle_count=len(expected),
        valid_bundle_count=valid_count,
        missing_bundles=tuple(missing),
        invalid_bundles=tuple(invalid),
        unexpected_final_bundles=unexpected,
        partial_artifacts=partials,
        product_summaries=product_summaries,
        time_plan=time_plan,
        manifest_identity=file_identity(request.production_manifest_path),
        time_plan_identity=file_identity(
            _resolve_relative_resource(
                run_root,
                manifest["delivery"]["time_plan_relative_path"],
                label="time shard plan",
            )
        ),
    )


def write_galaxy_campaign_delivery_qc_json(
    result: GalaxyCampaignDeliveryQCResult,
    path: Path | str,
) -> Path:
    """Atomically persist a machine-readable campaign-QC receipt."""

    if not isinstance(result, GalaxyCampaignDeliveryQCResult):
        raise TypeError("result must be GalaxyCampaignDeliveryQCResult")
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(
                result.to_dict(), stream, ensure_ascii=False, indent=2, sort_keys=True
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


__all__ = [
    "GALAXY_CAMPAIGN_DELIVERY_QC_SCHEMA_ID",
    "GALAXY_CAMPAIGN_DELIVERY_QC_SCHEMA_VERSION",
    "GalaxyCampaignDeliveryQCError",
    "GalaxyCampaignDeliveryQCRequest",
    "GalaxyCampaignDeliveryQCResult",
    "audit_galaxy_campaign_delivery_v1",
    "write_galaxy_campaign_delivery_qc_json",
]
