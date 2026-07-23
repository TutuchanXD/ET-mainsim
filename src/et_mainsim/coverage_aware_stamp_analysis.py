"""Raw-10-s coverage-aware, non-imputing CDPP reduction for formal stamps.

This layer is deliberately separate from the strict reference-photometry
contract.  The strict reducer remains the QA truth: an aperture cadence is
invalid whenever a pixel in the fixed aperture is invalid, saturated, or
cosmic-affected.  Here those invalid *cadences* are omitted from a physical
time bin; no image pixel or flux sample is reconstructed.  An accepted bin is
normalised by its actual usable exposure and records its coverage fraction.

The resulting statistic follows the legacy analysis family's robust MAD form
(``1.4826 * mean(abs(x - median(x)))``), while retaining an explicit coverage
contract that the historical pickle workflow did not record.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import argparse
import csv
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .stamp_inputs import file_identity


COVERAGE_AWARE_STAMP_ANALYSIS_SCHEMA_ID = (
    "et_mainsim.raw_coverage_aware_stamp_analysis.v2"
)
COVERAGE_AWARE_STAMP_ANALYSIS_SCHEMA_VERSION = 2

_TIME_TOLERANCE_SECONDS = 1e-8
_RAW_CADENCE_SECONDS = 10.0


class CoverageAwareAnalysisError(ValueError):
    """Raised when a coverage-aware reduction would hide a data-contract error."""


def _finite_float_vector(
    value: ArrayLike, *, name: str, allow_nan: bool
) -> NDArray[np.float64]:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise CoverageAwareAnalysisError(f"{name} must be numeric") from error
    if result.ndim != 1 or result.size == 0:
        raise CoverageAwareAnalysisError(
            f"{name} must be a non-empty one-dimensional array"
        )
    if np.any(np.isinf(result)) or (not allow_nan and not np.all(np.isfinite(result))):
        raise CoverageAwareAnalysisError(f"{name} must contain only finite values")
    return result


def _binary_mask(
    value: ArrayLike, *, name: str, shape: tuple[int, ...]
) -> NDArray[np.bool_]:
    raw = np.asarray(value)
    if raw.shape != shape:
        raise CoverageAwareAnalysisError(f"{name} must match the cadence axis")
    if raw.dtype.kind == "b":
        return raw.astype(bool, copy=False)
    if raw.dtype.kind not in {"i", "u"} or not np.all((raw == 0) | (raw == 1)):
        raise CoverageAwareAnalysisError(f"{name} values must be exactly 0 or 1")
    return raw.astype(bool, copy=False)


def _optional_model_vector(
    value: ArrayLike | None,
    *,
    name: str,
    shape: tuple[int, ...],
) -> NDArray[np.float64] | None:
    if value is None:
        return None
    result = _finite_float_vector(value, name=name, allow_nan=True)
    if result.shape != shape:
        raise CoverageAwareAnalysisError(f"{name} must match the cadence axis")
    return result


@dataclass(frozen=True)
class CoverageAwareLightCurve:
    """A cadence-level strict-aperture light curve with optional injected model."""

    time_seconds: ArrayLike
    exposure_seconds: ArrayLike
    flux_e: ArrayLike
    aperture_valid: ArrayLike
    model_flux_e: ArrayLike | None = None
    residual_e: ArrayLike | None = None

    def __post_init__(self) -> None:
        time = _finite_float_vector(
            self.time_seconds, name="time_seconds", allow_nan=False
        )
        exposure = _finite_float_vector(
            self.exposure_seconds,
            name="exposure_seconds",
            allow_nan=False,
        )
        flux = _finite_float_vector(self.flux_e, name="flux_e", allow_nan=True)
        if exposure.shape != time.shape or flux.shape != time.shape:
            raise CoverageAwareAnalysisError(
                "time_seconds, exposure_seconds, and flux_e must match"
            )
        if np.any(exposure <= 0.0):
            raise CoverageAwareAnalysisError("exposure_seconds must be positive")
        if time.size > 1 and not np.allclose(
            time[1:],
            time[:-1] + exposure[:-1],
            rtol=0.0,
            atol=_TIME_TOLERANCE_SECONDS,
        ):
            raise CoverageAwareAnalysisError(
                "cadence intervals must be globally contiguous"
            )
        valid = _binary_mask(
            self.aperture_valid, name="aperture_valid", shape=time.shape
        )
        model = _optional_model_vector(
            self.model_flux_e,
            name="model_flux_e",
            shape=time.shape,
        )
        residual = _optional_model_vector(
            self.residual_e,
            name="residual_e",
            shape=time.shape,
        )
        if (model is None) != (residual is None):
            raise CoverageAwareAnalysisError(
                "model_flux_e and residual_e must be provided together"
            )
        usable = valid & np.isfinite(flux)
        if np.any(valid & ~np.isfinite(flux)):
            raise CoverageAwareAnalysisError("valid cadences require finite flux_e")
        if model is not None and residual is not None:
            if np.any(
                usable & (~np.isfinite(model) | ~np.isfinite(residual) | (model <= 0.0))
            ):
                raise CoverageAwareAnalysisError(
                    "valid injected-model cadences require finite positive model_flux_e "
                    "and finite residual_e"
                )
        object.__setattr__(self, "time_seconds", time)
        object.__setattr__(self, "exposure_seconds", exposure)
        object.__setattr__(self, "flux_e", flux)
        object.__setattr__(self, "aperture_valid", valid)
        object.__setattr__(self, "model_flux_e", model)
        object.__setattr__(self, "residual_e", residual)


@dataclass(frozen=True)
class CoverageAwareBinnedRow:
    """One physical CDPP window and its explicit usable-exposure accounting."""

    window_minutes: int
    bin_id: int
    time_start_seconds: float
    time_stop_seconds: float
    expected_exposure_seconds: float
    effective_exposure_seconds: float
    coverage_fraction: float
    expected_cadence_count: int
    valid_cadence_count: int
    accepted: bool
    observed_flux_rate_e_per_s: float
    model_flux_rate_e_per_s: float | None
    residual_fraction_ppm: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_minutes": self.window_minutes,
            "bin_id": self.bin_id,
            "time_start_seconds": self.time_start_seconds,
            "time_stop_seconds": self.time_stop_seconds,
            "expected_exposure_seconds": self.expected_exposure_seconds,
            "effective_exposure_seconds": self.effective_exposure_seconds,
            "coverage_fraction": self.coverage_fraction,
            "expected_cadence_count": self.expected_cadence_count,
            "valid_cadence_count": self.valid_cadence_count,
            "accepted": self.accepted,
            "observed_flux_rate_e_per_s": self.observed_flux_rate_e_per_s,
            "model_flux_rate_e_per_s": self.model_flux_rate_e_per_s,
            "residual_fraction_ppm": self.residual_fraction_ppm,
        }


@dataclass(frozen=True)
class CoverageAwareCDPPMetric:
    """One window-size CDPP metric plus accepted-bin accounting."""

    window_minutes: int
    total_bin_count: int
    accepted_bin_count: int
    rejected_bin_count: int
    accepted_sample_count: int
    minimum_coverage_fraction: float
    minimum_accepted_bins: int
    observed_cdpp_ppm: float
    residual_cdpp_ppm: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_minutes": self.window_minutes,
            "total_bin_count": self.total_bin_count,
            "accepted_bin_count": self.accepted_bin_count,
            "rejected_bin_count": self.rejected_bin_count,
            "accepted_sample_count": self.accepted_sample_count,
            "minimum_coverage_fraction": self.minimum_coverage_fraction,
            "minimum_accepted_bins": self.minimum_accepted_bins,
            "observed_cdpp_ppm": self.observed_cdpp_ppm,
            "residual_cdpp_ppm": self.residual_cdpp_ppm,
            "observed_estimator": "legacy_median_centered_mean_absolute_deviation_times_1.4826",
            "residual_estimator": "legacy_median_centered_mean_absolute_deviation_times_1.4826",
            "aggregation": "valid_cadence_counts_normalized_by_actual_effective_exposure",
        }


@dataclass(frozen=True)
class CoverageAwareCDPPResult:
    """All requested physical-window rows and their coverage-aware CDPP values."""

    binned_rows: tuple[CoverageAwareBinnedRow, ...]
    metrics_by_window_minutes: Mapping[int, CoverageAwareCDPPMetric]


@dataclass(frozen=True)
class CoverageAwareStampAnalysisRequest:
    """One immutable strict-reference input and a separate derived output path."""

    reference_analysis_dir: Path | str
    output_dir: Path | str
    windows_minutes: tuple[int, ...] = (30, 90, 390)
    minimum_coverage_fraction: float | None = None
    minimum_accepted_bins: int = 10
    bin_origin_seconds: float = 0.0

    def __post_init__(self) -> None:
        reference_dir = Path(self.reference_analysis_dir).expanduser().resolve()
        output_dir = Path(self.output_dir).expanduser().resolve()
        windows = _normalise_windows(self.windows_minutes)
        if self.minimum_coverage_fraction is None:
            raise CoverageAwareAnalysisError(
                "minimum_coverage_fraction must be explicit for formal science analysis"
            )
        coverage = _validate_fraction(self.minimum_coverage_fraction)
        minimum_bins = _validate_minimum_bins(self.minimum_accepted_bins)
        try:
            origin = float(self.bin_origin_seconds)
        except (TypeError, ValueError, OverflowError) as error:
            raise CoverageAwareAnalysisError(
                "bin_origin_seconds must be finite"
            ) from error
        if not math.isfinite(origin):
            raise CoverageAwareAnalysisError("bin_origin_seconds must be finite")
        object.__setattr__(self, "reference_analysis_dir", reference_dir)
        object.__setattr__(self, "output_dir", output_dir)
        object.__setattr__(self, "windows_minutes", windows)
        object.__setattr__(self, "minimum_coverage_fraction", coverage)
        object.__setattr__(self, "minimum_accepted_bins", minimum_bins)
        object.__setattr__(self, "bin_origin_seconds", origin)


@dataclass(frozen=True)
class CoverageAwareStampAnalysisResult:
    """Published paths and compact science-independent analysis identity."""

    output_dir: Path
    analysis_manifest_path: Path
    binned_lightcurve_path: Path
    source_id: int
    case: str

    def to_dict(self) -> dict[str, Any]:
        """Return a compact, JSON-safe immutable-publication receipt."""

        return {
            "output_dir": str(self.output_dir),
            "analysis_manifest_path": str(self.analysis_manifest_path),
            "binned_lightcurve_path": str(self.binned_lightcurve_path),
            "source_id_int64": self.source_id,
            "case": self.case,
        }


@dataclass(frozen=True)
class _ReferenceAnalysisInput:
    reference_dir: Path
    analysis_manifest_path: Path
    analysis_manifest: Mapping[str, Any]
    lightcurve_path: Path
    source_id: int
    case: str
    cadence_seconds: float
    raw_exposure_seconds: float
    curve: CoverageAwareLightCurve


def _normalise_windows(windows_minutes: Iterable[int]) -> tuple[int, ...]:
    windows = tuple(int(value) for value in windows_minutes)
    if not windows or any(value <= 0 for value in windows):
        raise CoverageAwareAnalysisError(
            "windows_minutes must be non-empty positive integers"
        )
    if len(set(windows)) != len(windows):
        raise CoverageAwareAnalysisError("windows_minutes must not repeat")
    return windows


def _validate_fraction(value: float) -> float:
    try:
        fraction = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise CoverageAwareAnalysisError(
            "minimum_coverage_fraction must be finite and in (0, 1]"
        ) from error
    if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise CoverageAwareAnalysisError(
            "minimum_coverage_fraction must be finite and in (0, 1]"
        )
    return fraction


def _validate_minimum_bins(value: int) -> int:
    if isinstance(value, (bool, np.bool_)) or int(value) < 2:
        raise CoverageAwareAnalysisError("minimum_accepted_bins must be at least two")
    return int(value)


def _legacy_mad_cdpp_ppm(values: list[float], *, divide_by_center: bool) -> float:
    samples = np.asarray(values, dtype=np.float64)
    if samples.size == 0 or not np.all(np.isfinite(samples)):
        return float("nan")
    center = float(np.median(samples))
    mad = float(np.mean(np.abs(samples - center)))
    if divide_by_center:
        if center <= 0.0:
            return float("nan")
        return float(1.4826 * mad / center * 1_000_000.0)
    return float(1.4826 * mad)


def _require_bin_edges_align_with_cadence(
    curve: CoverageAwareLightCurve,
    *,
    lower: float,
    upper: float,
) -> None:
    starts = np.asarray(curve.time_seconds, dtype=np.float64)
    stops = starts + np.asarray(curve.exposure_seconds, dtype=np.float64)
    intersects_lower = (starts < lower - _TIME_TOLERANCE_SECONDS) & (
        stops > lower + _TIME_TOLERANCE_SECONDS
    )
    intersects_upper = (starts < upper - _TIME_TOLERANCE_SECONDS) & (
        stops > upper + _TIME_TOLERANCE_SECONDS
    )
    if np.any(intersects_lower | intersects_upper):
        raise CoverageAwareAnalysisError(
            "CDPP bin boundaries must align with whole physical cadence intervals"
        )


def compute_coverage_aware_cdpp_v1(
    curve: CoverageAwareLightCurve,
    *,
    windows_minutes: Iterable[int],
    minimum_coverage_fraction: float,
    minimum_accepted_bins: int = 10,
    bin_origin_seconds: float = 0.0,
) -> CoverageAwareCDPPResult:
    """Aggregate strict-valid cadence samples without repairing missing exposure.

    Each accepted bin contains only whole, clean cadence intervals.  Its flux
    is expressed as a rate over the usable exposure rather than being scaled
    to a fictitious full window.  The metric records the coverage threshold,
    accepted bin count, and sample count so it cannot be confused with a
    complete-exposure CDPP value.
    """

    if not isinstance(curve, CoverageAwareLightCurve):
        raise TypeError("curve must be a CoverageAwareLightCurve")
    windows = _normalise_windows(windows_minutes)
    minimum_coverage = _validate_fraction(minimum_coverage_fraction)
    minimum_bins = _validate_minimum_bins(minimum_accepted_bins)
    try:
        origin = float(bin_origin_seconds)
    except (TypeError, ValueError, OverflowError) as error:
        raise CoverageAwareAnalysisError("bin_origin_seconds must be finite") from error
    if not math.isfinite(origin):
        raise CoverageAwareAnalysisError("bin_origin_seconds must be finite")

    time = np.asarray(curve.time_seconds, dtype=np.float64)
    exposure = np.asarray(curve.exposure_seconds, dtype=np.float64)
    flux = np.asarray(curve.flux_e, dtype=np.float64)
    valid = np.asarray(curve.aperture_valid, dtype=bool) & np.isfinite(flux)
    model = (
        None
        if curve.model_flux_e is None
        else np.asarray(curve.model_flux_e, dtype=np.float64)
    )
    residual = (
        None
        if curve.residual_e is None
        else np.asarray(curve.residual_e, dtype=np.float64)
    )
    if model is not None and residual is not None:
        valid &= np.isfinite(model) & (model > 0.0) & np.isfinite(residual)

    coverage_start = float(time[0])
    coverage_end = float(time[-1] + exposure[-1])
    rows: list[CoverageAwareBinnedRow] = []
    metrics: dict[int, CoverageAwareCDPPMetric] = {}
    for minutes in windows:
        window_seconds = float(minutes * 60)
        first_bin = int(
            np.ceil(
                (coverage_start - origin) / window_seconds - _TIME_TOLERANCE_SECONDS
            )
        )
        last_bin_exclusive = int(
            np.floor((coverage_end - origin) / window_seconds + _TIME_TOLERANCE_SECONDS)
        )
        observed_rates: list[float] = []
        residual_fractions: list[float] = []
        accepted_sample_count = 0
        total = 0
        rejected = 0
        for bin_id in range(first_bin, last_bin_exclusive):
            lower = origin + bin_id * window_seconds
            upper = lower + window_seconds
            _require_bin_edges_align_with_cadence(curve, lower=lower, upper=upper)
            contained = (time >= lower - _TIME_TOLERANCE_SECONDS) & (
                time + exposure <= upper + _TIME_TOLERANCE_SECONDS
            )
            if not np.any(contained):
                continue
            total += 1
            usable = contained & valid
            expected_count = int(np.count_nonzero(contained))
            usable_count = int(np.count_nonzero(usable))
            effective = float(np.sum(exposure[usable], dtype=np.float64))
            coverage = effective / window_seconds
            accepted = coverage + _TIME_TOLERANCE_SECONDS >= minimum_coverage
            observed_rate = float("nan")
            model_rate: float | None = None
            residual_ppm: float | None = None
            if usable_count:
                observed_rate = float(
                    np.sum(flux[usable], dtype=np.float64) / effective
                )
                if model is not None and residual is not None:
                    model_sum = float(np.sum(model[usable], dtype=np.float64))
                    if not math.isfinite(model_sum) or model_sum <= 0.0:
                        accepted = False
                    else:
                        model_rate = model_sum / effective
                        residual_ppm = float(
                            np.sum(residual[usable], dtype=np.float64)
                            / model_sum
                            * 1_000_000.0
                        )
            if accepted and not math.isfinite(observed_rate):
                accepted = False
            if accepted:
                observed_rates.append(observed_rate)
                accepted_sample_count += usable_count
                if residual_ppm is not None:
                    residual_fractions.append(residual_ppm)
            else:
                rejected += 1
            rows.append(
                CoverageAwareBinnedRow(
                    window_minutes=minutes,
                    bin_id=bin_id,
                    time_start_seconds=lower,
                    time_stop_seconds=upper,
                    expected_exposure_seconds=window_seconds,
                    effective_exposure_seconds=effective,
                    coverage_fraction=coverage,
                    expected_cadence_count=expected_count,
                    valid_cadence_count=usable_count,
                    accepted=accepted,
                    observed_flux_rate_e_per_s=observed_rate,
                    model_flux_rate_e_per_s=model_rate,
                    residual_fraction_ppm=residual_ppm,
                )
            )
        observed_cdpp = (
            _legacy_mad_cdpp_ppm(observed_rates, divide_by_center=True)
            if len(observed_rates) >= minimum_bins
            else float("nan")
        )
        residual_cdpp = (
            _legacy_mad_cdpp_ppm(residual_fractions, divide_by_center=False)
            if model is not None and len(residual_fractions) >= minimum_bins
            else float("nan")
        )
        metrics[minutes] = CoverageAwareCDPPMetric(
            window_minutes=minutes,
            total_bin_count=total,
            accepted_bin_count=len(observed_rates),
            rejected_bin_count=rejected,
            accepted_sample_count=accepted_sample_count,
            minimum_coverage_fraction=minimum_coverage,
            minimum_accepted_bins=minimum_bins,
            observed_cdpp_ppm=observed_cdpp,
            residual_cdpp_ppm=residual_cdpp,
        )
    return CoverageAwareCDPPResult(
        binned_rows=tuple(rows),
        metrics_by_window_minutes=metrics,
    )


def _strict_int(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise CoverageAwareAnalysisError(f"{name} must be a non-negative integer")
    try:
        result = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as error:
        raise CoverageAwareAnalysisError(
            f"{name} must be a non-negative integer"
        ) from error
    if result < 0 or str(result) != str(value).strip():
        raise CoverageAwareAnalysisError(f"{name} must be a non-negative integer")
    return result


def _strict_finite_float(value: object, *, name: str) -> float:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as error:
        raise CoverageAwareAnalysisError(f"{name} must be finite") from error
    if not math.isfinite(result):
        raise CoverageAwareAnalysisError(f"{name} must be finite")
    return result


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise CoverageAwareAnalysisError(f"{label} does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise CoverageAwareAnalysisError(
            f"{label} is not valid JSON: {path}"
        ) from error
    if not isinstance(payload, dict):
        raise CoverageAwareAnalysisError(f"{label} must be a JSON object")
    return payload


def _resolve_child(root: Path, relative: object, *, label: str) -> Path:
    if not isinstance(relative, str) or not relative.strip():
        raise CoverageAwareAnalysisError(f"{label} must be a non-empty relative path")
    candidate = Path(relative)
    if candidate.is_absolute():
        raise CoverageAwareAnalysisError(f"{label} must be relative")
    resolved_root = root.resolve()
    resolved = (resolved_root / candidate).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise CoverageAwareAnalysisError(
            f"{label} escapes reference_analysis_dir"
        ) from error
    return resolved


def _csv_float(value: object, *, name: str) -> float:
    if value is None or str(value).strip() == "":
        return float("nan")
    try:
        result = float(str(value))
    except (TypeError, ValueError) as error:
        raise CoverageAwareAnalysisError(f"{name} must be numeric or empty") from error
    if math.isinf(result):
        raise CoverageAwareAnalysisError(f"{name} must not be infinite")
    return result


def _csv_binary(value: object, *, name: str) -> bool:
    if value is None:
        raise CoverageAwareAnalysisError(f"{name} is missing")
    normalized = str(value).strip().lower()
    if normalized in {"1", "true"}:
        return True
    if normalized in {"0", "false"}:
        return False
    raise CoverageAwareAnalysisError(f"{name} must be exactly 0/1 or true/false")


def _read_reference_lightcurve_csv(
    path: Path,
    *,
    case: str,
) -> CoverageAwareLightCurve:
    required = {
        "time_start_seconds",
        "exposure_seconds",
        "raw_frame_start_index",
        "raw_frame_stop_index_exclusive",
        "flux_derived_e",
        "aperture_valid",
    }
    try:
        stream = path.open("r", encoding="utf-8", newline="")
    except FileNotFoundError as error:
        raise CoverageAwareAnalysisError(
            f"reference light curve does not exist: {path}"
        ) from error
    with stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or ())
        missing = sorted(required - fields)
        if missing:
            raise CoverageAwareAnalysisError(
                "reference light curve lacks required columns: " + ", ".join(missing)
            )
        model_fields = {"model_flux_e", "model_residual_e", "injected_model_valid"}
        if case == "injected" and not model_fields.issubset(fields):
            raise CoverageAwareAnalysisError(
                "injected reference light curve lacks model residual columns"
            )
        if model_fields & fields and not model_fields.issubset(fields):
            raise CoverageAwareAnalysisError(
                "reference light curve model columns must be all present or all absent"
            )
        values: dict[str, list[float] | list[bool]] = {
            "time": [],
            "exposure": [],
            "flux": [],
            "valid": [],
            "model": [],
            "residual": [],
        }
        has_model = model_fields.issubset(fields)
        previous_raw_stop: int | None = None
        for row_index, row in enumerate(reader, start=2):
            raw_start = _strict_int(
                row.get("raw_frame_start_index"),
                name=f"raw_frame_start_index row {row_index}",
            )
            raw_stop = _strict_int(
                row.get("raw_frame_stop_index_exclusive"),
                name=f"raw_frame_stop_index_exclusive row {row_index}",
            )
            if raw_stop != raw_start + 1:
                raise CoverageAwareAnalysisError(
                    "raw 10-s reference cadence must contain exactly one raw frame"
                )
            if previous_raw_stop is not None and raw_start != previous_raw_stop:
                raise CoverageAwareAnalysisError(
                    "raw 10-s reference raw-frame indices must be globally continuous"
                )
            previous_raw_stop = raw_stop
            values["time"].append(
                _csv_float(
                    row.get("time_start_seconds"),
                    name=f"time_start_seconds row {row_index}",
                )
            )
            values["exposure"].append(
                _csv_float(
                    row.get("exposure_seconds"),
                    name=f"exposure_seconds row {row_index}",
                )
            )
            values["flux"].append(
                _csv_float(
                    row.get("flux_derived_e"), name=f"flux_derived_e row {row_index}"
                )
            )
            values["valid"].append(
                _csv_binary(
                    row.get("aperture_valid"), name=f"aperture_valid row {row_index}"
                )
            )
            if has_model:
                model_valid = _csv_binary(
                    row.get("injected_model_valid"),
                    name=f"injected_model_valid row {row_index}",
                )
                aperture_valid = bool(values["valid"][-1])
                if model_valid != aperture_valid:
                    raise CoverageAwareAnalysisError(
                        "injected_model_valid must equal aperture_valid"
                    )
                values["model"].append(
                    _csv_float(
                        row.get("model_flux_e"), name=f"model_flux_e row {row_index}"
                    )
                )
                values["residual"].append(
                    _csv_float(
                        row.get("model_residual_e"),
                        name=f"model_residual_e row {row_index}",
                    )
                )
    return CoverageAwareLightCurve(
        time_seconds=np.asarray(values["time"], dtype=np.float64),
        exposure_seconds=np.asarray(values["exposure"], dtype=np.float64),
        flux_e=np.asarray(values["flux"], dtype=np.float64),
        aperture_valid=np.asarray(values["valid"], dtype=bool),
        model_flux_e=(
            None
            if not values["model"]
            else np.asarray(values["model"], dtype=np.float64)
        ),
        residual_e=(
            None
            if not values["residual"]
            else np.asarray(values["residual"], dtype=np.float64)
        ),
    )


def _discover_reference_analysis(
    request: CoverageAwareStampAnalysisRequest,
) -> _ReferenceAnalysisInput:
    reference_dir = Path(request.reference_analysis_dir)
    if not reference_dir.is_dir() or reference_dir.is_symlink():
        raise CoverageAwareAnalysisError(
            "reference_analysis_dir must be a real published analysis directory"
        )
    manifest_path = reference_dir / "analysis_manifest.json"
    manifest = _read_json_object(manifest_path, label="reference analysis manifest")
    if manifest.get("schema_id") != "et_mainsim.standard_stamp_analysis.v1":
        raise CoverageAwareAnalysisError("unsupported reference analysis schema")
    if (
        int(manifest.get("schema_version", -1)) != 1
        or manifest.get("complete") is not True
    ):
        raise CoverageAwareAnalysisError("reference analysis is not complete")
    if manifest.get("observation_product") != "final_dn":
        raise CoverageAwareAnalysisError("reference analysis must derive from final_dn")
    if manifest.get("background_realization_used") is not False:
        raise CoverageAwareAnalysisError(
            "reference analysis must not use a background realization"
        )
    delivery = manifest.get("delivery")
    if not isinstance(delivery, Mapping):
        raise CoverageAwareAnalysisError("reference analysis lacks delivery metadata")
    cadence_seconds = _strict_finite_float(
        delivery.get("cadence_seconds"),
        name="delivery.cadence_seconds",
    )
    raw_exposure_seconds = _strict_finite_float(
        delivery.get("raw_exposure_seconds"),
        name="delivery.raw_exposure_seconds",
    )
    if (
        delivery.get("product_filename") != "raw.h5"
        or not math.isclose(
            cadence_seconds,
            _RAW_CADENCE_SECONDS,
            rel_tol=0.0,
            abs_tol=_TIME_TOLERANCE_SECONDS,
        )
        or not math.isclose(
            raw_exposure_seconds,
            _RAW_CADENCE_SECONDS,
            rel_tol=0.0,
            abs_tol=_TIME_TOLERANCE_SECONDS,
        )
    ):
        raise CoverageAwareAnalysisError(
            "coverage-aware science analysis requires a raw 10-s reference input"
        )
    source_id = _strict_int(manifest.get("source_id_int64"), name="source_id_int64")
    case = manifest.get("case")
    if case not in {"static", "injected"}:
        raise CoverageAwareAnalysisError(
            "reference analysis case must be static or injected"
        )
    reference = manifest.get("reference_lightcurve")
    if not isinstance(reference, Mapping):
        raise CoverageAwareAnalysisError(
            "reference analysis lacks reference_lightcurve"
        )
    if (
        reference.get("schema_id")
        != "et_mainsim.standard_stamp_reference_lightcurve.v1"
    ):
        raise CoverageAwareAnalysisError("unsupported reference light-curve schema")
    if reference.get("format") != "csv":
        raise CoverageAwareAnalysisError("reference light curve must be CSV")
    lightcurve_path = _resolve_child(
        reference_dir,
        reference.get("path"),
        label="reference light curve path",
    )
    curve = _read_reference_lightcurve_csv(lightcurve_path, case=str(case))
    if not np.allclose(
        np.asarray(curve.exposure_seconds, dtype=np.float64),
        raw_exposure_seconds,
        rtol=0.0,
        atol=_TIME_TOLERANCE_SECONDS,
    ):
        raise CoverageAwareAnalysisError(
            "raw 10-s reference rows must each have the declared 10-s exposure"
        )
    return _ReferenceAnalysisInput(
        reference_dir=reference_dir,
        analysis_manifest_path=manifest_path,
        analysis_manifest=manifest,
        lightcurve_path=lightcurve_path,
        source_id=source_id,
        case=str(case),
        cadence_seconds=cadence_seconds,
        raw_exposure_seconds=raw_exposure_seconds,
        curve=curve,
    )


def _csv_value(value: object) -> str | int | float:
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        return "" if not math.isfinite(numeric) else numeric
    return str(value)


def _write_binned_lightcurve(
    path: Path, rows: Iterable[CoverageAwareBinnedRow]
) -> Path:
    fields = [
        "window_minutes",
        "bin_id",
        "time_start_seconds",
        "time_stop_seconds",
        "expected_exposure_seconds",
        "effective_exposure_seconds",
        "coverage_fraction",
        "expected_cadence_count",
        "valid_cadence_count",
        "accepted",
        "observed_flux_rate_e_per_s",
        "model_flux_rate_e_per_s",
        "residual_fraction_ppm",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {key: _csv_value(value) for key, value in row.to_dict().items()}
            )
        stream.flush()
        os.fsync(stream.fileno())
    return path


def _json_safe(value: Any) -> Any:
    """Convert NumPy/non-finite diagnostics to portable strict JSON values."""

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(
            _json_safe(payload),
            stream,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return path


def _fsync_directory(path: Path) -> None:
    """Persist a staged or published directory entry before returning success."""

    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _output_manifest(
    *,
    resolved: _ReferenceAnalysisInput,
    request: CoverageAwareStampAnalysisRequest,
    result: CoverageAwareCDPPResult,
    binned_lightcurve_path: Path,
) -> dict[str, Any]:
    return {
        "schema_id": COVERAGE_AWARE_STAMP_ANALYSIS_SCHEMA_ID,
        "schema_version": COVERAGE_AWARE_STAMP_ANALYSIS_SCHEMA_VERSION,
        "complete": True,
        "source_id": str(resolved.source_id),
        "source_id_int64": resolved.source_id,
        "case": resolved.case,
        "observation_product": "final_dn",
        "background_realization_used": False,
        "analysis_implementation": {
            "module": "et_mainsim.coverage_aware_stamp_analysis",
            "module_identity": file_identity(Path(__file__)),
        },
        "input_reference_analysis": {
            "path": str(resolved.reference_dir),
            "analysis_manifest": file_identity(resolved.analysis_manifest_path),
            "reference_lightcurve": file_identity(resolved.lightcurve_path),
            "source_id_int64": resolved.source_id,
            "case": resolved.case,
            "strict_quality_policy": "invalidate_whole_fixed_aperture_cadence",
        },
        "input_raw_delivery": {
            "product_filename": "raw.h5",
            "cadence_seconds": resolved.cadence_seconds,
            "raw_exposure_seconds": resolved.raw_exposure_seconds,
            "raw_frame_policy": "one_contiguous_raw_frame_per_reference_cadence",
        },
        "coverage_policy": {
            "minimum_coverage_fraction": request.minimum_coverage_fraction,
            "minimum_accepted_bins": request.minimum_accepted_bins,
            "bin_origin_seconds": request.bin_origin_seconds,
            "invalid_cadence_handling": "omit_whole_invalid_cadences_without_pixel_or_flux_imputation",
            "accepted_bin_normalization": "actual_effective_exposure_only",
        },
        "legacy_compatibility": {
            "pca_used": False,
            "savgol_detrending_used": False,
            "pickle_adapter_used": False,
            "statistic": "median_centered_mean_absolute_deviation_times_1.4826",
            "note": (
                "This is a coverage-recording, no-imputation adapter for formal "
                "independent stamps; it is not the legacy pickle/PCA/SG workflow."
            ),
        },
        "binned_lightcurve": {
            "path": binned_lightcurve_path.name,
            "format": "csv",
            "identity": file_identity(binned_lightcurve_path),
        },
        "metrics": {
            str(window): metric.to_dict()
            for window, metric in result.metrics_by_window_minutes.items()
        },
    }


def run_coverage_aware_stamp_analysis_v1(
    request: CoverageAwareStampAnalysisRequest,
) -> CoverageAwareStampAnalysisResult:
    """Publish a separate coverage-aware analysis without replacing strict QA."""

    if not isinstance(request, CoverageAwareStampAnalysisRequest):
        raise TypeError("request must be a CoverageAwareStampAnalysisRequest")
    resolved = _discover_reference_analysis(request)
    output_dir = Path(request.output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(
            f"coverage-aware analysis output already exists: {output_dir}"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir.parent / f".{output_dir.name}.lock"
    try:
        lock_path.mkdir()
    except FileExistsError as error:
        raise FileExistsError(
            f"coverage-aware analysis publication is already in progress: {output_dir}"
        ) from error
    staging_dir: Path | None = None
    try:
        if output_dir.exists() or output_dir.is_symlink():
            raise FileExistsError(
                f"coverage-aware analysis output already exists: {output_dir}"
            )
        staging_dir = Path(
            tempfile.mkdtemp(
                prefix=f".{output_dir.name}.staging-",
                dir=output_dir.parent,
            )
        )
        result = compute_coverage_aware_cdpp_v1(
            resolved.curve,
            windows_minutes=request.windows_minutes,
            minimum_coverage_fraction=request.minimum_coverage_fraction,
            minimum_accepted_bins=request.minimum_accepted_bins,
            bin_origin_seconds=request.bin_origin_seconds,
        )
        binned_path = _write_binned_lightcurve(
            staging_dir / "coverage_aware_binned_lightcurve.csv",
            result.binned_rows,
        )
        manifest_path = _write_json(
            staging_dir / "coverage_aware_analysis_manifest.json",
            _output_manifest(
                resolved=resolved,
                request=request,
                result=result,
                binned_lightcurve_path=binned_path,
            ),
        )
        _fsync_directory(staging_dir)
        os.replace(staging_dir, output_dir)
        _fsync_directory(output_dir.parent)
        staging_dir = None
    finally:
        if staging_dir is not None:
            shutil.rmtree(staging_dir, ignore_errors=True)
        lock_path.rmdir()
    return CoverageAwareStampAnalysisResult(
        output_dir=output_dir,
        analysis_manifest_path=output_dir / manifest_path.name,
        binned_lightcurve_path=output_dir / binned_path.name,
        source_id=resolved.source_id,
        case=resolved.case,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-analysis-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--windows-minutes",
        nargs="+",
        type=int,
        default=(30, 90, 390),
    )
    parser.add_argument(
        "--minimum-coverage-fraction",
        required=True,
        type=float,
        help=(
            "Required scientific policy: accept a physical bin only when its "
            "usable exposure fraction reaches this value."
        ),
    )
    parser.add_argument("--minimum-accepted-bins", type=int, default=10)
    parser.add_argument("--bin-origin-seconds", type=float, default=0.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the no-imputation adapter and print a compact JSON receipt."""

    args = _parser().parse_args(None if argv is None else list(argv))
    request = CoverageAwareStampAnalysisRequest(
        reference_analysis_dir=args.reference_analysis_dir,
        output_dir=args.output_dir,
        windows_minutes=tuple(args.windows_minutes),
        minimum_coverage_fraction=args.minimum_coverage_fraction,
        minimum_accepted_bins=args.minimum_accepted_bins,
        bin_origin_seconds=args.bin_origin_seconds,
    )
    result = run_coverage_aware_stamp_analysis_v1(request)
    completion = result.to_dict()
    completion.update(
        {
            "windows_minutes": list(request.windows_minutes),
            "minimum_coverage_fraction": request.minimum_coverage_fraction,
            "minimum_accepted_bins": request.minimum_accepted_bins,
            "bin_origin_seconds": request.bin_origin_seconds,
        }
    )
    print(json.dumps(completion, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by CLI invocation.
    raise SystemExit(main())


__all__ = [
    "COVERAGE_AWARE_STAMP_ANALYSIS_SCHEMA_ID",
    "COVERAGE_AWARE_STAMP_ANALYSIS_SCHEMA_VERSION",
    "CoverageAwareAnalysisError",
    "CoverageAwareBinnedRow",
    "CoverageAwareCDPPMetric",
    "CoverageAwareCDPPResult",
    "CoverageAwareLightCurve",
    "CoverageAwareStampAnalysisRequest",
    "CoverageAwareStampAnalysisResult",
    "compute_coverage_aware_cdpp_v1",
    "main",
    "run_coverage_aware_stamp_analysis_v1",
]
