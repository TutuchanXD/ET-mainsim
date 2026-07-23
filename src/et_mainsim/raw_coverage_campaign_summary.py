"""Fail-closed ten-source summary for formal raw-coverage Galaxy analyses.

This is deliberately a metadata and small-CSV reducer.  It never reopens the
multi-terabyte ``final_dn`` delivery cubes: the campaign QC receipt and strict
raw analysis already bind those immutable inputs.  Before publishing a
summary, it proves that every expected target has one matching strict analysis
and one matching raw-coverage v3 analysis under the same frozen policy.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Literal

import numpy as np

from .galaxy_stamp_production import (
    GALAXY_STAMP_PRODUCTION_SCHEMA_ID,
    GALAXY_STAMP_PRODUCTION_SCHEMA_VERSION,
)
from .raw_coverage_policy import (
    FrozenRawCoveragePolicy,
    FrozenRawCoveragePolicyError,
    RAW_COVERAGE_ANALYSIS_PROFILE,
    RAW_COVERAGE_POLICY_SCHEMA_ID,
    load_frozen_raw_coverage_policy_v1,
    validate_frozen_raw_coverage_policy_for_run_v1,
)
from .stamp_inputs import file_identity
from .standard_stamp_analysis import STANDARD_STAMP_ANALYSIS_SCHEMA_ID


GALAXY_RAW_COVERAGE_CAMPAIGN_SUMMARY_SCHEMA_ID = (
    "et_mainsim.galaxy_raw_coverage_v2_campaign_summary.v1"
)
GALAXY_RAW_COVERAGE_CAMPAIGN_SUMMARY_SCHEMA_VERSION = 1
_COVERAGE_SCHEMA_ID = "et_mainsim.raw_coverage_aware_stamp_analysis.v3"
_COVERAGE_SCHEMA_VERSION = 3
_CAMPAIGN_QC_SCHEMA_ID = "et_mainsim.galaxy_campaign_delivery_qc.v1"
_TIME_TOLERANCE = 1e-8


class GalaxyRawCoverageCampaignSummaryError(ValueError):
    """Raised when a formal campaign contract is malformed or inconsistent."""


class GalaxyRawCoverageCampaignSummaryNotReadyError(
    GalaxyRawCoverageCampaignSummaryError
):
    """Raised when one or more expected immutable source products are absent."""


def _json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise GalaxyRawCoverageCampaignSummaryError(
            f"{label} does not exist: {path}"
        ) from error
    except json.JSONDecodeError as error:
        raise GalaxyRawCoverageCampaignSummaryError(
            f"{label} is not valid JSON: {path}"
        ) from error
    if not isinstance(payload, dict):
        raise GalaxyRawCoverageCampaignSummaryError(f"{label} must be a JSON object")
    return payload


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GalaxyRawCoverageCampaignSummaryError(f"{label} must be an object")
    return value


def _text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GalaxyRawCoverageCampaignSummaryError(f"{label} must be a non-empty string")
    return value.strip()


def _int(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise GalaxyRawCoverageCampaignSummaryError(f"{label} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise GalaxyRawCoverageCampaignSummaryError(f"{label} must be an integer") from error
    if result != value or result < minimum:
        raise GalaxyRawCoverageCampaignSummaryError(
            f"{label} must be an integer at least {minimum}"
        )
    return result


def _source_id(value: Any, *, label: str) -> int:
    result = _int(value, label=label, minimum=0)
    if result > 2**63 - 1:
        raise GalaxyRawCoverageCampaignSummaryError(f"{label} must fit signed int64")
    return result


def _finite(value: Any, *, label: str, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise GalaxyRawCoverageCampaignSummaryError(f"{label} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise GalaxyRawCoverageCampaignSummaryError(f"{label} must be finite") from error
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise GalaxyRawCoverageCampaignSummaryError(f"{label} must be finite")
    return result


def _identity(value: Any, *, label: str) -> dict[str, Any]:
    mapping = _mapping(value, label=label)
    sha256 = mapping.get("sha256")
    if not isinstance(sha256, str) or len(sha256) != 64:
        raise GalaxyRawCoverageCampaignSummaryError(
            f"{label}.sha256 must be a SHA-256 hex string"
        )
    try:
        int(sha256, 16)
    except ValueError as error:
        raise GalaxyRawCoverageCampaignSummaryError(
            f"{label}.sha256 must be a SHA-256 hex string"
        ) from error
    return {
        "sha256": sha256,
        "size_bytes": _int(mapping.get("size_bytes"), label=f"{label}.size_bytes", minimum=1),
    }


def _same_identity(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    return (
        actual.get("sha256") == expected.get("sha256")
        and actual.get("size_bytes") == expected.get("size_bytes")
    )


def _file_identity(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise GalaxyRawCoverageCampaignSummaryError(
            f"{label} must be a real regular file: {path}"
        )
    return _identity(file_identity(path), label=label)


def _relative_path(run_root: Path, path: Path, *, label: str, file: bool = False) -> str:
    if path.is_symlink():
        raise GalaxyRawCoverageCampaignSummaryError(f"{label} must not be a symlink")
    if file and not path.is_file():
        raise GalaxyRawCoverageCampaignSummaryError(f"{label} must be a regular file")
    try:
        return path.resolve().relative_to(run_root.resolve()).as_posix()
    except ValueError as error:
        raise GalaxyRawCoverageCampaignSummaryError(
            f"{label} must be located within the formal run root"
        ) from error


def _resolve_relative_file(run_root: Path, relative: Any, *, label: str) -> Path:
    text = _text(relative, label=label)
    candidate = Path(text)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise GalaxyRawCoverageCampaignSummaryError(f"{label} must be a safe relative path")
    path = (run_root / candidate).resolve()
    _relative_path(run_root, path, label=label, file=True)
    return path


def _json_fingerprint(value: Any) -> str:
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class GalaxyRawCoverageCampaignSummaryRequest:
    """One explicit immutable ten-source campaign summary request."""

    production_manifest_path: Path | str
    campaign_qc_path: Path | str
    coverage_policy_path: Path | str
    output_dir: Path | str
    case: Literal["injected"] | str = "injected"

    def __post_init__(self) -> None:
        if self.case != "injected":
            raise GalaxyRawCoverageCampaignSummaryError(
                "raw coverage campaign summary only supports injected"
            )
        object.__setattr__(
            self,
            "production_manifest_path",
            Path(self.production_manifest_path).expanduser().resolve(),
        )
        object.__setattr__(
            self,
            "campaign_qc_path",
            Path(self.campaign_qc_path).expanduser().resolve(),
        )
        object.__setattr__(
            self,
            "coverage_policy_path",
            Path(self.coverage_policy_path).expanduser().resolve(),
        )
        object.__setattr__(self, "output_dir", Path(self.output_dir).expanduser().resolve())
        object.__setattr__(self, "case", "injected")


@dataclass(frozen=True)
class _Target:
    source_order: int
    source_id: int
    gaia_g_mag: float
    source_class: str
    detector_id: str
    detector_xpix: float | None
    detector_ypix: float | None
    field_angle_deg: float | None
    factor_path_relative_to_run_root: str
    factor_identity: Mapping[str, Any]


@dataclass(frozen=True)
class _SourceAudit:
    target: _Target
    strict_manifest_path: Path
    strict_manifest_identity: Mapping[str, Any]
    strict_curve_path: Path
    strict_curve_identity: Mapping[str, Any]
    strict_quality: Mapping[str, Any]
    coverage_manifest_path: Path
    coverage_manifest_identity: Mapping[str, Any]
    binned_lightcurve_path: Path
    binned_lightcurve_identity: Mapping[str, Any]
    module_identity: Mapping[str, Any]
    metrics: Mapping[int, Mapping[str, Any]]


@dataclass(frozen=True)
class GalaxyRawCoverageCampaignAuditResult:
    """Read-only all-source audit; an incomplete campaign is not publishable."""

    request: GalaxyRawCoverageCampaignSummaryRequest
    run_root: Path
    run_id: str
    production_manifest_identity: Mapping[str, Any]
    production_manifest_relative_to_run_root: str
    campaign_qc_identity: Mapping[str, Any]
    campaign_qc_relative_to_run_root: str
    policy: FrozenRawCoveragePolicy
    policy_relative_to_run_root: str
    source_audits: tuple[_SourceAudit, ...]
    errors_by_source: Mapping[str, str]

    @property
    def ready(self) -> bool:
        return not self.errors_by_source and len(self.source_audits) == 10

    @property
    def fingerprint(self) -> str:
        return _json_fingerprint(
            {
                "production_manifest": self.production_manifest_identity,
                "campaign_qc": self.campaign_qc_identity,
                "policy": self.policy.content_identity,
                "sources": [
                    {
                        "source_id": source.target.source_id,
                        "strict_manifest": source.strict_manifest_identity,
                        "strict_curve": source.strict_curve_identity,
                        "coverage_manifest": source.coverage_manifest_identity,
                        "binned_lightcurve": source.binned_lightcurve_identity,
                        "module": source.module_identity,
                    }
                    for source in self.source_audits
                ],
            }
        )


@dataclass(frozen=True)
class GalaxyRawCoverageCampaignSummaryResult:
    output_dir: Path
    summary_manifest_path: Path
    source_summary_path: Path
    source_window_metrics_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_dir": str(self.output_dir),
            "summary_manifest_path": str(self.summary_manifest_path),
            "source_summary_path": str(self.source_summary_path),
            "source_window_metrics_path": str(self.source_window_metrics_path),
        }


def _load_targets(manifest: Mapping[str, Any], *, run_root: Path) -> tuple[_Target, ...]:
    targets = manifest.get("targets")
    if not isinstance(targets, list) or len(targets) != 10:
        raise GalaxyRawCoverageCampaignSummaryError(
            "formal raw coverage summary requires exactly 10 targets"
        )
    result: list[_Target] = []
    source_ids: set[int] = set()
    for source_order, raw in enumerate(targets):
        target = _mapping(raw, label="production target")
        source_id = _source_id(target.get("source_id_int64"), label="target.source_id_int64")
        if source_id in source_ids:
            raise GalaxyRawCoverageCampaignSummaryError(
                "production manifest target IDs must be unique"
            )
        source_ids.add(source_id)
        mapping = _mapping(target.get("focalplane_mapping"), label="target.focalplane_mapping")
        factor_path = _text(
            target.get("factor_snapshot_relative_path"),
            label="target.factor_snapshot_relative_path",
        )
        actual_factor = _resolve_relative_file(
            run_root,
            factor_path,
            label="target factor snapshot",
        )
        factor_identity = _identity(target.get("factor_snapshot"), label="target.factor_snapshot")
        if not _same_identity(_file_identity(actual_factor, label="target factor snapshot"), factor_identity):
            raise GalaxyRawCoverageCampaignSummaryError(
                "target factor snapshot identity changed after production preparation"
            )
        def optional_float(name: str) -> float | None:
            value = mapping.get(name)
            return (
                None
                if value is None
                else _finite(value, label=f"target.focalplane_mapping.{name}")
            )
        result.append(
            _Target(
                source_order=source_order,
                source_id=source_id,
                gaia_g_mag=_finite(target.get("gaia_g_mag"), label="target.gaia_g_mag"),
                source_class=_text(target.get("source_class"), label="target.source_class"),
                detector_id=_text(mapping.get("detector_id"), label="target.detector_id"),
                detector_xpix=optional_float("detector_xpix"),
                detector_ypix=optional_float("detector_ypix"),
                field_angle_deg=optional_float("field_angle_deg"),
                factor_path_relative_to_run_root=factor_path,
                factor_identity=factor_identity,
            )
        )
    return tuple(result)


def _load_campaign(
    request: GalaxyRawCoverageCampaignSummaryRequest,
) -> tuple[
    Path,
    Mapping[str, Any],
    str,
    Mapping[str, Any],
    str,
    tuple[_Target, ...],
    FrozenRawCoveragePolicy,
    str,
    Mapping[str, Any],
    str,
    int,
    int,
]:
    manifest_path = Path(request.production_manifest_path)
    manifest = _json_object(manifest_path, label="production manifest")
    if manifest.get("schema_id") != GALAXY_STAMP_PRODUCTION_SCHEMA_ID:
        raise GalaxyRawCoverageCampaignSummaryError("unsupported Galaxy production manifest")
    if int(manifest.get("schema_version", 0)) != GALAXY_STAMP_PRODUCTION_SCHEMA_VERSION:
        raise GalaxyRawCoverageCampaignSummaryError(
            "unsupported Galaxy production manifest version"
        )
    if manifest.get("observation_product") != "final_dn":
        raise GalaxyRawCoverageCampaignSummaryError(
            "production manifest observation_product must be final_dn"
        )
    if manifest.get("background_realization_delivered") is not False:
        raise GalaxyRawCoverageCampaignSummaryError(
            "production manifest must not deliver a background realization"
        )
    run_root = manifest_path.parent.resolve()
    manifest_relative = _relative_path(
        run_root,
        manifest_path,
        label="production manifest",
        file=True,
    )
    manifest_identity = _file_identity(manifest_path, label="production manifest")
    run_id = _text(manifest.get("run_id"), label="production manifest run_id")
    delivery = _mapping(manifest.get("delivery"), label="production manifest delivery")
    if not math.isclose(
        _finite(delivery.get("raw_exposure_seconds"), label="delivery.raw_exposure_seconds", positive=True),
        10.0,
        abs_tol=1e-12,
    ):
        raise GalaxyRawCoverageCampaignSummaryError(
            "formal raw coverage summary requires a 10-s raw exposure"
        )
    targets = _load_targets(manifest, run_root=run_root)
    try:
        policy = load_frozen_raw_coverage_policy_v1(request.coverage_policy_path)
        policy_relative, validated_manifest_relative = (
            validate_frozen_raw_coverage_policy_for_run_v1(
                policy,
                production_manifest_path=manifest_path,
                case=request.case,
            )
        )
    except FrozenRawCoveragePolicyError as error:
        raise GalaxyRawCoverageCampaignSummaryError(
            f"invalid frozen coverage policy: {error}"
        ) from error
    if validated_manifest_relative != manifest_relative:
        raise GalaxyRawCoverageCampaignSummaryError(
            "frozen coverage policy resolves a different production-manifest path"
        )
    qc_path = Path(request.campaign_qc_path)
    qc_relative = _relative_path(
        run_root,
        qc_path,
        label="campaign QC receipt",
        file=True,
    )
    qc = _json_object(qc_path, label="campaign QC receipt")
    if qc.get("schema_id") != _CAMPAIGN_QC_SCHEMA_ID or int(qc.get("schema_version", 0)) != 1:
        raise GalaxyRawCoverageCampaignSummaryError("campaign QC receipt has an unsupported schema")
    if qc.get("ready") is not True or qc.get("case") != "injected":
        raise GalaxyRawCoverageCampaignSummaryError(
            "campaign QC receipt is not ready for injected delivery"
        )
    if qc.get("run_id") != run_id:
        raise GalaxyRawCoverageCampaignSummaryError(
            "campaign QC receipt run_id conflicts with production manifest"
        )
    qc_manifest_identity = _identity(qc.get("manifest_identity"), label="campaign QC manifest_identity")
    if not _same_identity(qc_manifest_identity, manifest_identity):
        raise GalaxyRawCoverageCampaignSummaryError(
            "campaign QC receipt binds a different production manifest"
        )
    coverage = _mapping(qc.get("coverage"), label="campaign QC coverage")
    if _int(coverage.get("target_count"), label="campaign QC target_count", minimum=1) != 10:
        raise GalaxyRawCoverageCampaignSummaryError(
            "campaign QC receipt must prove exactly 10 targets"
        )
    shard_count = _int(coverage.get("shard_count"), label="campaign QC shard_count", minimum=1)
    cadence_count = _int(
        coverage.get("accepted_raw_frame_count_per_target"),
        label="campaign QC accepted_raw_frame_count_per_target",
        minimum=1,
    )
    return (
        run_root,
        manifest,
        run_id,
        manifest_identity,
        manifest_relative,
        targets,
        policy,
        policy_relative,
        _file_identity(qc_path, label="campaign QC receipt"),
        qc_relative,
        shard_count,
        cadence_count,
    )


def _manifest_identity_matches(
    value: Any,
    expected: Mapping[str, Any],
    *,
    label: str,
) -> None:
    actual = _identity(value, label=label)
    if not _same_identity(actual, expected):
        raise GalaxyRawCoverageCampaignSummaryError(f"{label} conflicts with frozen run")


def _require_source_header(
    payload: Mapping[str, Any],
    *,
    target: _Target,
    run_id: str,
    label: str,
) -> None:
    if payload.get("complete") is not True:
        raise GalaxyRawCoverageCampaignSummaryError(f"{label} is not complete")
    if payload.get("run_id") != run_id:
        raise GalaxyRawCoverageCampaignSummaryError(f"{label} run_id conflicts")
    if payload.get("case") != "injected":
        raise GalaxyRawCoverageCampaignSummaryError(f"{label} case must be injected")
    if _source_id(payload.get("source_id_int64"), label=f"{label}.source_id_int64") != target.source_id:
        raise GalaxyRawCoverageCampaignSummaryError(f"{label} source ID conflicts")
    if payload.get("observation_product") != "final_dn":
        raise GalaxyRawCoverageCampaignSummaryError(f"{label} must derive from final_dn")
    if payload.get("background_realization_used") is not False:
        raise GalaxyRawCoverageCampaignSummaryError(
            f"{label} must not use a background realization"
        )


def _validate_strict_source(
    *,
    run_root: Path,
    target: _Target,
    run_id: str,
    production_manifest_relative: str,
    production_manifest_identity: Mapping[str, Any],
    shard_count: int,
    cadence_count: int,
) -> tuple[Path, Mapping[str, Any], Path, Mapping[str, Any], Mapping[str, Any]]:
    strict_dir = (
        run_root
        / "analysis"
        / f"source_{target.source_id}"
        / "injected"
        / "raw_10s_strict"
    )
    if not strict_dir.is_dir() or strict_dir.is_symlink():
        raise GalaxyRawCoverageCampaignSummaryError("strict raw analysis directory is missing")
    strict_manifest_path = strict_dir / "analysis_manifest.json"
    strict_manifest_identity = _file_identity(strict_manifest_path, label="strict analysis manifest")
    strict = _json_object(strict_manifest_path, label="strict analysis manifest")
    if strict.get("schema_id") != STANDARD_STAMP_ANALYSIS_SCHEMA_ID or int(strict.get("schema_version", 0)) != 1:
        raise GalaxyRawCoverageCampaignSummaryError("strict analysis manifest has an unsupported schema")
    _require_source_header(strict, target=target, run_id=run_id, label="strict analysis")
    if strict.get("production_manifest_relative_to_run_root") != production_manifest_relative:
        raise GalaxyRawCoverageCampaignSummaryError(
            "strict analysis production manifest path conflicts"
        )
    _manifest_identity_matches(
        strict.get("production_manifest_identity"),
        production_manifest_identity,
        label="strict analysis production manifest identity",
    )
    delivery = _mapping(strict.get("delivery"), label="strict analysis delivery")
    if delivery.get("product_filename") != "raw.h5":
        raise GalaxyRawCoverageCampaignSummaryError("strict analysis must consume raw.h5")
    if not all(
        math.isclose(_finite(delivery.get(field), label=f"strict delivery.{field}"), 10.0, abs_tol=1e-12)
        for field in ("cadence_seconds", "raw_exposure_seconds")
    ):
        raise GalaxyRawCoverageCampaignSummaryError("strict analysis must use raw 10-s cadence")
    if _int(delivery.get("bundle_count"), label="strict delivery.bundle_count", minimum=1) != shard_count:
        raise GalaxyRawCoverageCampaignSummaryError("strict analysis bundle count conflicts with campaign QC")
    quality = _mapping(strict.get("quality"), label="strict analysis quality")
    valid = _int(quality.get("valid_cadence_count"), label="strict quality valid", minimum=0)
    invalid = _int(quality.get("invalid_cadence_count"), label="strict quality invalid", minimum=0)
    if _int(quality.get("cadence_count"), label="strict quality cadence", minimum=1) != valid + invalid:
        raise GalaxyRawCoverageCampaignSummaryError("strict quality cadence counts do not close")
    if valid + invalid != cadence_count:
        raise GalaxyRawCoverageCampaignSummaryError("strict cadence count conflicts with campaign QC")
    if quality.get("quality_policy") != "invalidate_whole_fixed_aperture_cadence":
        raise GalaxyRawCoverageCampaignSummaryError("strict quality policy is unsupported")
    frozen = _mapping(strict.get("frozen_variability"), label="strict frozen_variability")
    if frozen.get("path_relative_to_run_root") != target.factor_path_relative_to_run_root:
        raise GalaxyRawCoverageCampaignSummaryError("strict factor snapshot path conflicts")
    _manifest_identity_matches(
        frozen.get("identity"), target.factor_identity, label="strict factor snapshot identity"
    )
    if frozen.get("time_alignment") != "simulation_raw_frame_index":
        raise GalaxyRawCoverageCampaignSummaryError("strict factor time alignment is unsupported")
    reference = _mapping(strict.get("reference_lightcurve"), label="strict reference_lightcurve")
    if reference.get("schema_id") != "et_mainsim.standard_stamp_reference_lightcurve.v1" or reference.get("format") != "csv":
        raise GalaxyRawCoverageCampaignSummaryError("strict reference light curve is unsupported")
    strict_curve_path = _resolve_relative_file(
        strict_dir,
        reference.get("path"),
        label="strict reference light curve path",
    )
    strict_curve_identity = _file_identity(strict_curve_path, label="strict reference light curve")
    return (
        strict_manifest_path,
        strict_manifest_identity,
        strict_curve_path,
        strict_curve_identity,
        quality,
    )


def _csv_float(value: Any, *, label: str, allow_empty: bool = False) -> float | None:
    if value is None or str(value).strip() == "":
        if allow_empty:
            return None
        raise GalaxyRawCoverageCampaignSummaryError(f"{label} is missing")
    return _finite(str(value), label=label)


def _csv_int(value: Any, *, label: str, minimum: int = 0) -> int:
    if value is None or str(value).strip() == "":
        raise GalaxyRawCoverageCampaignSummaryError(f"{label} is missing")
    text = str(value).strip()
    try:
        result = int(text, 10)
    except ValueError as error:
        raise GalaxyRawCoverageCampaignSummaryError(f"{label} must be an integer") from error
    if str(result) != text or result < minimum:
        raise GalaxyRawCoverageCampaignSummaryError(
            f"{label} must be an integer at least {minimum}"
        )
    return result


def _csv_bool(value: Any, *, label: str) -> bool:
    text = str(value).strip().lower() if value is not None else ""
    if text in {"1", "true"}:
        return True
    if text in {"0", "false"}:
        return False
    raise GalaxyRawCoverageCampaignSummaryError(f"{label} must be exactly 0/1")


def _mad_cdpp(values: list[float], *, divide_by_center: bool) -> float | None:
    if len(values) < 2 or not all(math.isfinite(item) for item in values):
        return None
    samples = np.asarray(values, dtype=np.float64)
    center = float(np.median(samples))
    mad = float(np.mean(np.abs(samples - center)))
    if divide_by_center:
        if center <= 0.0:
            return None
        return float(1.4826 * mad / center * 1_000_000.0)
    return float(1.4826 * mad)


def _metric_float(value: Any, *, label: str) -> float | None:
    if value is None:
        return None
    return _finite(value, label=label)


def _require_close(actual: float | None, expected: float | None, *, label: str) -> None:
    if actual is None or expected is None:
        if actual is expected:
            return
        raise GalaxyRawCoverageCampaignSummaryError(f"{label} null status conflicts")
    if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-9):
        raise GalaxyRawCoverageCampaignSummaryError(f"{label} conflicts with binned CSV")


def _validate_binned_metrics(
    *,
    path: Path,
    policy: FrozenRawCoveragePolicy,
    manifest_metrics: Any,
) -> Mapping[int, Mapping[str, Any]]:
    if not isinstance(manifest_metrics, Mapping):
        raise GalaxyRawCoverageCampaignSummaryError("coverage metrics must be an object")
    required_columns = {
        "window_minutes",
        "bin_id",
        "coverage_fraction",
        "valid_cadence_count",
        "accepted",
        "observed_flux_rate_e_per_s",
        "residual_fraction_ppm",
    }
    grouped: dict[int, list[dict[str, Any]]] = {window: [] for window in policy.windows_minutes}
    seen: set[tuple[int, int]] = set()
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or ())
        missing = sorted(required_columns - fields)
        if missing:
            raise GalaxyRawCoverageCampaignSummaryError(
                "coverage binned light curve lacks columns: " + ", ".join(missing)
            )
        for row_index, row in enumerate(reader, start=2):
            window = _csv_int(row.get("window_minutes"), label=f"window_minutes row {row_index}", minimum=1)
            if window not in grouped:
                raise GalaxyRawCoverageCampaignSummaryError("coverage binned light curve has an unexpected window")
            bin_id = _csv_int(row.get("bin_id"), label=f"bin_id row {row_index}", minimum=0)
            if (window, bin_id) in seen:
                raise GalaxyRawCoverageCampaignSummaryError("coverage binned light curve repeats a bin")
            seen.add((window, bin_id))
            coverage = _csv_float(row.get("coverage_fraction"), label=f"coverage_fraction row {row_index}")
            assert coverage is not None
            if coverage < -_TIME_TOLERANCE or coverage > 1.0 + _TIME_TOLERANCE:
                raise GalaxyRawCoverageCampaignSummaryError("coverage fraction is outside [0, 1]")
            accepted = _csv_bool(row.get("accepted"), label=f"accepted row {row_index}")
            expected_accepted = coverage + _TIME_TOLERANCE >= policy.minimum_coverage_fraction
            if accepted != expected_accepted:
                raise GalaxyRawCoverageCampaignSummaryError(
                    "coverage accepted flag conflicts with frozen threshold"
                )
            valid_cadence_count = _csv_int(
                row.get("valid_cadence_count"),
                label=f"valid_cadence_count row {row_index}",
                minimum=0,
            )
            observed = _csv_float(
                row.get("observed_flux_rate_e_per_s"),
                label=f"observed_flux_rate_e_per_s row {row_index}",
                allow_empty=not accepted,
            )
            residual = _csv_float(
                row.get("residual_fraction_ppm"),
                label=f"residual_fraction_ppm row {row_index}",
                allow_empty=not accepted,
            )
            if accepted and (observed is None or observed <= 0.0 or residual is None):
                raise GalaxyRawCoverageCampaignSummaryError(
                    "accepted coverage bins require finite observed and residual values"
                )
            grouped[window].append(
                {
                    "bin_id": bin_id,
                    "accepted": accepted,
                    "valid_cadence_count": valid_cadence_count,
                    "observed": observed,
                    "residual": residual,
                }
            )
    result: dict[int, Mapping[str, Any]] = {}
    for window in policy.windows_minutes:
        rows = grouped[window]
        if not rows:
            raise GalaxyRawCoverageCampaignSummaryError(
                "coverage binned light curve has no rows for a policy window"
            )
        metric = _mapping(manifest_metrics.get(str(window)), label=f"coverage metric {window}")
        if _int(metric.get("window_minutes"), label="coverage metric window", minimum=1) != window:
            raise GalaxyRawCoverageCampaignSummaryError("coverage metric window conflicts")
        accepted = [row for row in rows if row["accepted"]]
        total_count = len(rows)
        accepted_count = len(accepted)
        rejected_count = total_count - accepted_count
        sample_count = sum(int(row["valid_cadence_count"]) for row in accepted)
        expected_observed = (
            None
            if accepted_count < policy.minimum_accepted_bins
            else _mad_cdpp([float(row["observed"]) for row in accepted], divide_by_center=True)
        )
        expected_residual = (
            None
            if accepted_count < policy.minimum_accepted_bins
            else _mad_cdpp([float(row["residual"]) for row in accepted], divide_by_center=False)
        )
        for key, expected in (
            ("total_bin_count", total_count),
            ("accepted_bin_count", accepted_count),
            ("rejected_bin_count", rejected_count),
            ("accepted_sample_count", sample_count),
            ("minimum_accepted_bins", policy.minimum_accepted_bins),
        ):
            if _int(metric.get(key), label=f"coverage metric {key}", minimum=0) != expected:
                raise GalaxyRawCoverageCampaignSummaryError(
                    f"coverage metric {key} conflicts with binned CSV"
                )
        if not math.isclose(
            _finite(metric.get("minimum_coverage_fraction"), label="coverage metric threshold"),
            policy.minimum_coverage_fraction,
            abs_tol=1e-12,
        ):
            raise GalaxyRawCoverageCampaignSummaryError(
                "coverage metric threshold conflicts with frozen policy"
            )
        _require_close(
            _metric_float(metric.get("observed_cdpp_ppm"), label="observed CDPP"),
            expected_observed,
            label="observed CDPP",
        )
        _require_close(
            _metric_float(metric.get("residual_cdpp_ppm"), label="residual CDPP"),
            expected_residual,
            label="residual CDPP",
        )
        result[window] = metric
    return result


def _validate_coverage_source(
    *,
    run_root: Path,
    target: _Target,
    run_id: str,
    production_manifest_relative: str,
    production_manifest_identity: Mapping[str, Any],
    campaign_qc_relative: str,
    campaign_qc_identity: Mapping[str, Any],
    policy: FrozenRawCoveragePolicy,
    policy_relative: str,
    strict_manifest_path: Path,
    strict_manifest_identity: Mapping[str, Any],
    strict_curve_path: Path,
    strict_curve_identity: Mapping[str, Any],
) -> tuple[Path, Mapping[str, Any], Path, Mapping[str, Any], Mapping[str, Any], Mapping[int, Mapping[str, Any]]]:
    coverage_dir = strict_manifest_path.parent.parent / RAW_COVERAGE_ANALYSIS_PROFILE
    if not coverage_dir.is_dir() or coverage_dir.is_symlink():
        raise GalaxyRawCoverageCampaignSummaryError("coverage analysis directory is missing")
    parent = coverage_dir.parent
    if (parent / f".{coverage_dir.name}.lock").exists():
        raise GalaxyRawCoverageCampaignSummaryError("coverage analysis publication lock is active")
    if any(parent.glob(f".{coverage_dir.name}.staging-*")):
        raise GalaxyRawCoverageCampaignSummaryError("coverage analysis staging directory exists")
    coverage_manifest_path = coverage_dir / "coverage_aware_analysis_manifest.json"
    coverage_manifest_identity = _file_identity(
        coverage_manifest_path, label="coverage analysis manifest"
    )
    coverage = _json_object(coverage_manifest_path, label="coverage analysis manifest")
    if coverage.get("schema_id") != _COVERAGE_SCHEMA_ID or int(coverage.get("schema_version", 0)) != _COVERAGE_SCHEMA_VERSION:
        raise GalaxyRawCoverageCampaignSummaryError("coverage analysis manifest has an unsupported schema")
    _require_source_header(coverage, target=target, run_id=run_id, label="coverage analysis")
    production = _mapping(coverage.get("production_manifest"), label="coverage production_manifest")
    if production.get("path_relative_to_run_root") != production_manifest_relative:
        raise GalaxyRawCoverageCampaignSummaryError("coverage production manifest path conflicts")
    _manifest_identity_matches(
        production.get("identity"), production_manifest_identity, label="coverage production identity"
    )
    qc = _mapping(coverage.get("campaign_qc"), label="coverage campaign_qc")
    if qc.get("path_relative_to_run_root") != campaign_qc_relative:
        raise GalaxyRawCoverageCampaignSummaryError("coverage campaign QC path conflicts")
    _manifest_identity_matches(qc.get("identity"), campaign_qc_identity, label="coverage campaign QC identity")
    frozen_policy = _mapping(coverage.get("frozen_coverage_policy"), label="coverage frozen policy")
    if frozen_policy.get("schema_id") != RAW_COVERAGE_POLICY_SCHEMA_ID:
        raise GalaxyRawCoverageCampaignSummaryError("coverage frozen policy schema conflicts")
    if frozen_policy.get("path_relative_to_run_root") != policy_relative:
        raise GalaxyRawCoverageCampaignSummaryError("coverage frozen policy path conflicts")
    _manifest_identity_matches(
        frozen_policy.get("identity"), policy.content_identity, label="coverage frozen policy identity"
    )
    if coverage.get("coverage_policy") != policy.coverage_policy_record():
        raise GalaxyRawCoverageCampaignSummaryError("coverage policy record conflicts with frozen policy")
    raw = _mapping(coverage.get("input_raw_delivery"), label="coverage raw delivery")
    if raw.get("product_filename") != "raw.h5" or not all(
        math.isclose(_finite(raw.get(field), label=f"coverage raw delivery.{field}"), 10.0, abs_tol=1e-12)
        for field in ("cadence_seconds", "raw_exposure_seconds")
    ):
        raise GalaxyRawCoverageCampaignSummaryError("coverage analysis must consume raw 10-s delivery")
    reference = _mapping(coverage.get("input_reference_analysis"), label="coverage strict reference")
    if reference.get("path_relative_to_run_root") != strict_manifest_path.parent.relative_to(run_root).as_posix():
        raise GalaxyRawCoverageCampaignSummaryError("coverage strict reference path conflicts")
    _manifest_identity_matches(
        reference.get("analysis_manifest"), strict_manifest_identity, label="coverage strict manifest identity"
    )
    curve = _mapping(reference.get("reference_lightcurve"), label="coverage strict curve")
    if curve.get("path_relative_to_run_root") != strict_curve_path.relative_to(run_root).as_posix():
        raise GalaxyRawCoverageCampaignSummaryError("coverage strict curve path conflicts")
    _manifest_identity_matches(curve.get("identity"), strict_curve_identity, label="coverage strict curve identity")
    implementation = _mapping(coverage.get("analysis_implementation"), label="coverage implementation")
    if implementation.get("module") != "et_mainsim.coverage_aware_stamp_analysis":
        raise GalaxyRawCoverageCampaignSummaryError("coverage implementation module conflicts")
    module_identity = _identity(implementation.get("module_identity"), label="coverage module identity")
    binned = _mapping(coverage.get("binned_lightcurve"), label="coverage binned light curve")
    if binned.get("path") != "coverage_aware_binned_lightcurve.csv" or binned.get("format") != "csv":
        raise GalaxyRawCoverageCampaignSummaryError("coverage binned light curve contract conflicts")
    binned_path = coverage_dir / str(binned["path"])
    binned_identity = _file_identity(binned_path, label="coverage binned light curve")
    _manifest_identity_matches(binned.get("identity"), binned_identity, label="coverage binned light curve identity")
    metrics = _validate_binned_metrics(
        path=binned_path,
        policy=policy,
        manifest_metrics=coverage.get("metrics"),
    )
    return (
        coverage_manifest_path,
        coverage_manifest_identity,
        binned_path,
        binned_identity,
        module_identity,
        metrics,
    )


def audit_galaxy_raw_coverage_campaign_v1(
    request: GalaxyRawCoverageCampaignSummaryRequest,
) -> GalaxyRawCoverageCampaignAuditResult:
    """Audit exactly ten policy-consistent source analyses without publishing."""

    if not isinstance(request, GalaxyRawCoverageCampaignSummaryRequest):
        raise TypeError("request must be GalaxyRawCoverageCampaignSummaryRequest")
    (
        run_root,
        _manifest,
        run_id,
        manifest_identity,
        manifest_relative,
        targets,
        policy,
        policy_relative,
        qc_identity,
        qc_relative,
        shard_count,
        cadence_count,
    ) = _load_campaign(request)
    _relative_path(run_root, Path(request.output_dir), label="summary output directory")
    audits: list[_SourceAudit] = []
    errors: dict[str, str] = {}
    expected_module_identity: Mapping[str, Any] | None = None
    for target in targets:
        try:
            (
                strict_manifest_path,
                strict_manifest_identity,
                strict_curve_path,
                strict_curve_identity,
                strict_quality,
            ) = _validate_strict_source(
                run_root=run_root,
                target=target,
                run_id=run_id,
                production_manifest_relative=manifest_relative,
                production_manifest_identity=manifest_identity,
                shard_count=shard_count,
                cadence_count=cadence_count,
            )
            (
                coverage_manifest_path,
                coverage_manifest_identity,
                binned_path,
                binned_identity,
                module_identity,
                metrics,
            ) = _validate_coverage_source(
                run_root=run_root,
                target=target,
                run_id=run_id,
                production_manifest_relative=manifest_relative,
                production_manifest_identity=manifest_identity,
                campaign_qc_relative=qc_relative,
                campaign_qc_identity=qc_identity,
                policy=policy,
                policy_relative=policy_relative,
                strict_manifest_path=strict_manifest_path,
                strict_manifest_identity=strict_manifest_identity,
                strict_curve_path=strict_curve_path,
                strict_curve_identity=strict_curve_identity,
            )
            if expected_module_identity is None:
                expected_module_identity = module_identity
            elif not _same_identity(module_identity, expected_module_identity):
                raise GalaxyRawCoverageCampaignSummaryError(
                    "coverage implementation identity differs across sources"
                )
            audits.append(
                _SourceAudit(
                    target=target,
                    strict_manifest_path=strict_manifest_path,
                    strict_manifest_identity=strict_manifest_identity,
                    strict_curve_path=strict_curve_path,
                    strict_curve_identity=strict_curve_identity,
                    strict_quality=strict_quality,
                    coverage_manifest_path=coverage_manifest_path,
                    coverage_manifest_identity=coverage_manifest_identity,
                    binned_lightcurve_path=binned_path,
                    binned_lightcurve_identity=binned_identity,
                    module_identity=module_identity,
                    metrics=metrics,
                )
            )
        except (OSError, ValueError) as error:
            errors[str(target.source_id)] = str(error)
    return GalaxyRawCoverageCampaignAuditResult(
        request=request,
        run_root=run_root,
        run_id=run_id,
        production_manifest_identity=manifest_identity,
        production_manifest_relative_to_run_root=manifest_relative,
        campaign_qc_identity=qc_identity,
        campaign_qc_relative_to_run_root=qc_relative,
        policy=policy,
        policy_relative_to_run_root=policy_relative,
        source_audits=tuple(audits),
        errors_by_source=errors,
    )


def _csv_value(value: Any) -> str | int | float:
    if value is None:
        return ""
    if isinstance(value, float):
        return "" if not math.isfinite(value) else value
    return value


def _write_csv(path: Path, *, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})
        stream.flush()
        os.fsync(stream.fileno())
    return path


def _source_summary_rows(audit: GalaxyRawCoverageCampaignAuditResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in audit.source_audits:
        target = source.target
        quality = source.strict_quality
        rows.append(
            {
                "run_id": audit.run_id,
                "source_order": target.source_order,
                "source_id": str(target.source_id),
                "case": "injected",
                "analysis_profile": RAW_COVERAGE_ANALYSIS_PROFILE,
                "gaia_g_vega_mag": target.gaia_g_mag,
                "source_class": target.source_class,
                "detector_id": target.detector_id,
                "detector_xpix": target.detector_xpix,
                "detector_ypix": target.detector_ypix,
                "field_angle_deg": target.field_angle_deg,
                "factor_snapshot_sha256": target.factor_identity["sha256"],
                "strict_manifest_sha256": source.strict_manifest_identity["sha256"],
                "coverage_manifest_sha256": source.coverage_manifest_identity["sha256"],
                "strict_cadence_count": quality["cadence_count"],
                "strict_valid_cadence_count": quality["valid_cadence_count"],
                "strict_invalid_cadence_count": quality["invalid_cadence_count"],
                "observation_product": "final_dn",
                "background_realization_used": False,
            }
        )
    return rows


def _source_metric_rows(audit: GalaxyRawCoverageCampaignAuditResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in audit.source_audits:
        target = source.target
        for window in audit.policy.windows_minutes:
            metric = source.metrics[window]
            rows.append(
                {
                    "run_id": audit.run_id,
                    "source_order": target.source_order,
                    "source_id": str(target.source_id),
                    "case": "injected",
                    "analysis_profile": RAW_COVERAGE_ANALYSIS_PROFILE,
                    "gaia_g_vega_mag": target.gaia_g_mag,
                    "source_class": target.source_class,
                    "detector_id": target.detector_id,
                    "window_minutes": window,
                    "policy_sha256": audit.policy.content_identity["sha256"],
                    "total_bin_count": metric["total_bin_count"],
                    "accepted_bin_count": metric["accepted_bin_count"],
                    "rejected_bin_count": metric["rejected_bin_count"],
                    "accepted_sample_count": metric["accepted_sample_count"],
                    "minimum_coverage_fraction": metric["minimum_coverage_fraction"],
                    "minimum_accepted_bins": metric["minimum_accepted_bins"],
                    "observed_cdpp_ppm": metric["observed_cdpp_ppm"],
                    "residual_cdpp_ppm": metric["residual_cdpp_ppm"],
                    "observed_estimator": metric.get("observed_estimator"),
                    "residual_estimator": metric.get("residual_estimator"),
                    "aggregation": metric.get("aggregation"),
                    "strict_manifest_sha256": source.strict_manifest_identity["sha256"],
                    "coverage_manifest_sha256": source.coverage_manifest_identity["sha256"],
                    "binned_lightcurve_sha256": source.binned_lightcurve_identity["sha256"],
                }
            )
    return rows


def _summary_manifest(
    *,
    audit: GalaxyRawCoverageCampaignAuditResult,
    source_summary_path: Path,
    source_metrics_path: Path,
) -> dict[str, Any]:
    return {
        "schema_id": GALAXY_RAW_COVERAGE_CAMPAIGN_SUMMARY_SCHEMA_ID,
        "schema_version": GALAXY_RAW_COVERAGE_CAMPAIGN_SUMMARY_SCHEMA_VERSION,
        "complete": True,
        "ready": True,
        "run_id": audit.run_id,
        "case": "injected",
        "analysis_profile": RAW_COVERAGE_ANALYSIS_PROFILE,
        "source_count": len(audit.source_audits),
        "observation_product": "final_dn",
        "background_realization_used": False,
        "derived_electron_formula": (
            "((final_dn-bias_level_sum_dn-column_noise_sum_dn_by_x)"
            "*gain_e_per_dn)-background_expectation_e"
        ),
        "production_manifest": {
            "path_relative_to_run_root": audit.production_manifest_relative_to_run_root,
            "identity": dict(audit.production_manifest_identity),
        },
        "campaign_qc": {
            "path_relative_to_run_root": audit.campaign_qc_relative_to_run_root,
            "identity": dict(audit.campaign_qc_identity),
        },
        "frozen_coverage_policy": {
            "schema_id": audit.policy.schema_id,
            "schema_version": audit.policy.schema_version,
            "path_relative_to_run_root": audit.policy_relative_to_run_root,
            "identity": audit.policy.content_identity,
            "record": audit.policy.coverage_policy_record(),
        },
        "analysis_implementation": {
            "module": "et_mainsim.raw_coverage_campaign_summary",
            "module_identity": _file_identity(Path(__file__), label="summary module"),
            "coverage_module_identity": dict(audit.source_audits[0].module_identity),
        },
        "source_artifacts": [
            {
                "source_order": source.target.source_order,
                "source_id": str(source.target.source_id),
                "strict_manifest": {
                    "path_relative_to_run_root": source.strict_manifest_path.relative_to(audit.run_root).as_posix(),
                    "identity": dict(source.strict_manifest_identity),
                },
                "strict_reference_lightcurve": {
                    "path_relative_to_run_root": source.strict_curve_path.relative_to(audit.run_root).as_posix(),
                    "identity": dict(source.strict_curve_identity),
                },
                "coverage_manifest": {
                    "path_relative_to_run_root": source.coverage_manifest_path.relative_to(audit.run_root).as_posix(),
                    "identity": dict(source.coverage_manifest_identity),
                },
                "coverage_binned_lightcurve": {
                    "path_relative_to_run_root": source.binned_lightcurve_path.relative_to(audit.run_root).as_posix(),
                    "identity": dict(source.binned_lightcurve_identity),
                },
            }
            for source in audit.source_audits
        ],
        "tables": {
            "source_summary": {
                "path": source_summary_path.name,
                "format": "csv",
                "identity": _file_identity(source_summary_path, label="source summary CSV"),
            },
            "source_window_metrics": {
                "path": source_metrics_path.name,
                "format": "csv",
                "identity": _file_identity(source_metrics_path, label="source window metrics CSV"),
                "row_count": len(audit.source_audits) * len(audit.policy.windows_minutes),
            },
        },
        "interpretation": {
            "observed_cdpp": "undetrended astrophysical variation plus instrument",
            "residual_cdpp": "known injected q(t) model residual diagnostic",
            "legacy_status": "legacy-MAD-compatible only; no legacy pickle, PCA, or Savitzky-Golay workflow",
            "cross_source_aggregation": "not a single campaign instrument CDPP; values remain per-source diagnostics",
        },
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return path


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_galaxy_raw_coverage_campaign_summary_v1(
    request: GalaxyRawCoverageCampaignSummaryRequest,
) -> GalaxyRawCoverageCampaignSummaryResult:
    """Atomically publish the ten-source summary only after a complete audit."""

    audit = audit_galaxy_raw_coverage_campaign_v1(request)
    if not audit.ready:
        details = "; ".join(
            f"source_{source_id}: {reason}"
            for source_id, reason in sorted(audit.errors_by_source.items())
        )
        raise GalaxyRawCoverageCampaignSummaryNotReadyError(
            f"raw coverage campaign is not ready: {details}"
        )
    output_dir = Path(request.output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"raw coverage campaign summary already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir.parent / f".{output_dir.name}.lock"
    try:
        lock_path.mkdir()
    except FileExistsError as error:
        raise FileExistsError(
            f"raw coverage campaign summary publication is already in progress: {output_dir}"
        ) from error
    staging_dir: Path | None = None
    try:
        if output_dir.exists() or output_dir.is_symlink():
            raise FileExistsError(f"raw coverage campaign summary already exists: {output_dir}")
        staging_dir = Path(
            tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
        )
        source_summary_path = _write_csv(
            staging_dir / "source_summary.csv",
            fields=(
                "run_id", "source_order", "source_id", "case", "analysis_profile",
                "gaia_g_vega_mag", "source_class", "detector_id", "detector_xpix",
                "detector_ypix", "field_angle_deg", "factor_snapshot_sha256",
                "strict_manifest_sha256", "coverage_manifest_sha256", "strict_cadence_count",
                "strict_valid_cadence_count", "strict_invalid_cadence_count",
                "observation_product", "background_realization_used",
            ),
            rows=_source_summary_rows(audit),
        )
        source_metrics_path = _write_csv(
            staging_dir / "source_window_metrics.csv",
            fields=(
                "run_id", "source_order", "source_id", "case", "analysis_profile",
                "gaia_g_vega_mag", "source_class", "detector_id", "window_minutes",
                "policy_sha256", "total_bin_count", "accepted_bin_count", "rejected_bin_count",
                "accepted_sample_count", "minimum_coverage_fraction", "minimum_accepted_bins",
                "observed_cdpp_ppm", "residual_cdpp_ppm", "observed_estimator",
                "residual_estimator", "aggregation", "strict_manifest_sha256",
                "coverage_manifest_sha256", "binned_lightcurve_sha256",
            ),
            rows=_source_metric_rows(audit),
        )
        summary_manifest_path = _write_json(
            staging_dir / "campaign_summary_manifest.json",
            _summary_manifest(
                audit=audit,
                source_summary_path=source_summary_path,
                source_metrics_path=source_metrics_path,
            ),
        )
        fresh_audit = audit_galaxy_raw_coverage_campaign_v1(request)
        if not fresh_audit.ready or fresh_audit.fingerprint != audit.fingerprint:
            raise GalaxyRawCoverageCampaignSummaryError(
                "raw coverage campaign inputs changed during summary publication"
            )
        _fsync_directory(staging_dir)
        os.replace(staging_dir, output_dir)
        _fsync_directory(output_dir.parent)
        staging_dir = None
    finally:
        if staging_dir is not None:
            shutil.rmtree(staging_dir, ignore_errors=True)
        lock_path.rmdir()
    return GalaxyRawCoverageCampaignSummaryResult(
        output_dir=output_dir,
        summary_manifest_path=output_dir / summary_manifest_path.name,
        source_summary_path=output_dir / source_summary_path.name,
        source_window_metrics_path=output_dir / source_metrics_path.name,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-manifest", required=True)
    parser.add_argument("--campaign-qc", required=True)
    parser.add_argument("--coverage-policy", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Publish the summary, or return 2 without writes when it is incomplete."""

    args = _parser().parse_args(None if argv is None else list(argv))
    request = GalaxyRawCoverageCampaignSummaryRequest(
        production_manifest_path=args.production_manifest,
        campaign_qc_path=args.campaign_qc,
        coverage_policy_path=args.coverage_policy,
        output_dir=args.output_dir,
    )
    try:
        result = publish_galaxy_raw_coverage_campaign_summary_v1(request)
    except GalaxyRawCoverageCampaignSummaryNotReadyError as error:
        print(json.dumps({"ready": False, "reason": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps({"ready": True, **result.to_dict()}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI invocation only.
    raise SystemExit(main())


__all__ = [
    "GALAXY_RAW_COVERAGE_CAMPAIGN_SUMMARY_SCHEMA_ID",
    "GALAXY_RAW_COVERAGE_CAMPAIGN_SUMMARY_SCHEMA_VERSION",
    "GalaxyRawCoverageCampaignAuditResult",
    "GalaxyRawCoverageCampaignSummaryError",
    "GalaxyRawCoverageCampaignSummaryNotReadyError",
    "GalaxyRawCoverageCampaignSummaryRequest",
    "GalaxyRawCoverageCampaignSummaryResult",
    "audit_galaxy_raw_coverage_campaign_v1",
    "main",
    "publish_galaxy_raw_coverage_campaign_summary_v1",
]
