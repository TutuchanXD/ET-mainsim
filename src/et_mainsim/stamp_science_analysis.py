"""Streaming, atomic science analysis for formal independent stamp series.

The formal HDF5 ``final_dn`` plane remains the only detector observation.
This module derives calibrated electron light curves without materialising a
long-duration image cube: one deterministic, chunk-aligned training pass freezes the
legacy-compatible optimal aperture, then one sequential raw pass emits the
10/30/60/120/300-s cadence products.  The summed cadences all reuse that same
aperture and are derived from the raw planes, while bounded samples from the
formal coadd products provide an independent parity check.

Publication is directory-atomic.  A caller either sees a complete, hashed,
read-back-validated product directory or no final directory at all.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
from typing import Any, Iterator, Literal
import uuid

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .reference_photometry import (
    ReferencePhotometryContractError,
    ReferencePhotometryInput,
    _FormalDeliveryHeader,
    _formal_identity_json,
    _formal_json_dataset,
    _formal_series_identity_json,
    _formal_time_intervals_are_contiguous,
    _read_formal_delivery_header,
)
from .provenance import collect_provenance
from .galaxy_lightcurves import (
    GALAXY_FACTOR_SNAPSHOT_SCHEMA_ID,
    read_galaxy_factor_snapshot,
)
from .galaxy_stamp_production import delivery_execution_mode_from_manifest
from .stamp_science_inputs import (
    SCIENCE_FACTOR_SNAPSHOT_SCHEMA_ID,
    read_science_factor_snapshot,
)
from .staged_stamp_delivery import (
    _atomic_publish_directory_noreplace,
    _fsync_directory as _strict_fsync_directory,
)
from .time_shards import ContinuousTimeShardPlan
from .stamp_science_photometry import (
    SCIENCE_PHOTOMETRY_SCHEMA_ID,
    SCIENCE_PHOTOMETRY_SCHEMA_VERSION,
    ScienceApertureDefinition,
    ScienceFluxUncertaintyModelResult,
    SciencePhotometryContractError,
    SciencePhotometryResult,
    ScienceVariabilityModelResult,
    StampSciencePhotometryPolicy,
    build_local_background_mask_v1,
    build_reference_fixed13_aperture_v1,
    build_science_optimal_aperture_v1,
    compute_science_cdpp_v1,
    compute_science_flux_uncertainty_model_v1,
    fit_science_variability_model_v1,
    reduce_science_photometry_v1,
)


STAMP_SCIENCE_ANALYSIS_SCHEMA_ID = "et_mainsim.stamp_science_analysis.v2"
STAMP_SCIENCE_ANALYSIS_SCHEMA_VERSION = 2
STAMP_SCIENCE_ANALYSIS_PUBLICATION_SCHEMA_ID = (
    "et_mainsim.stamp_science_analysis_publication.v2"
)
STAMP_SCIENCE_ANALYSIS_PUBLICATION_SCHEMA_VERSION = 2
STAMP_SCIENCE_ANALYSIS_PRODUCT_SET_SCHEMA_ID = (
    "et_mainsim.stamp_science_analysis_product_set.v2"
)
STAMP_SCIENCE_ANALYSIS_PRODUCT_SET_SCHEMA_VERSION = 2
_TIME_ATOL_SECONDS = 1e-8
_INT64_MAX = int(np.iinfo(np.int64).max)
_REQUIRED_COADD_FACTORS = (1, 3, 6, 12, 30)
_REFERENCE_LIGHTCURVE_SCHEMA_ID = (
    "et_mainsim.stamp_science_reference_lightcurve.v2"
)
_REFERENCE_LIGHTCURVE_SCHEMA_VERSION = 2
_PHOTOMETRY_TABLE_SCHEMA_ID = "et_mainsim.stamp_science_photometry_table.v2"
_PHOTOMETRY_TABLE_SCHEMA_VERSION = 2
_FORMAL_NOISE_MODEL_SCHEMA_ID = "et_mainsim.formal_stamp_noise_parameters.v1"
_FORMAL_NOISE_MODEL_SOURCE = "production_manifest.simulation_spec_base.readout"
_FORMAL_NOISE_MODEL_QUANTIZATION_FORMULA = (
    "gain_electrons_per_adu/sqrt(12) when ADC rounding is enabled"
)
_NOISE_MODEL_FIELDS = {
    "schema_id",
    "source",
    "read_noise_e_per_raw_pixel",
    "quantization_noise_e_per_raw_pixel",
    "quantization_formula",
}
_PHOTOMETRY_TABLE_COLUMNS = (
    "cadence_seconds",
    "time_start_seconds",
    "exposure_seconds",
    "raw_frame_start_index",
    "raw_frame_stop_index_exclusive",
    "flux_expectation_bgsub_e",
    "flux_local_bgsub_e",
    "local_background_e_per_pixel",
    "centroid_x",
    "centroid_y",
    "aperture_valid",
    "local_background_valid",
    "quality_bitmask",
    "raw_factor_sum",
    "fitted_flux_expectation_e",
    "residual_expectation_e",
    "residual_expectation_ppm",
    "fitted_flux_local_e",
    "residual_local_e",
    "residual_local_ppm",
    "background_expectation_aperture_e",
    "captured_flux_fraction",
    "captured_flux_denominator_e",
    "captured_flux_qa_pass",
    "flux_uncertainty_e",
    "source_variance_e2",
    "background_variance_e2",
    "read_variance_e2",
    "quantization_variance_e2",
    "uncertainty_valid",
    "flux_expectation_bgsub_e_per_s",
    "flux_local_bgsub_e_per_s",
    "fitted_flux_expectation_e_per_s",
    "fitted_flux_local_e_per_s",
    "model_flux_uncertainty_e",
    "model_flux_uncertainty_e_per_s",
)
_PHOTOMETRY_TABLE_HDF_DATASETS = {
    column: ("local_model_valid" if column == "local_background_valid" else column)
    for column in _PHOTOMETRY_TABLE_COLUMNS
    if column != "cadence_seconds"
}
_PHOTOMETRY_TABLE_BOOLEAN_COLUMNS = frozenset(
    {
        "aperture_valid",
        "local_background_valid",
        "captured_flux_qa_pass",
        "uncertainty_valid",
    }
)
_CENTROID_QUALITY_SCHEMA_ID = "et_mainsim.stamp_science_centroid_quality.v1"
_CENTROID_QUALITY_COLUMNS = (
    "cadence_seconds",
    "time_start_seconds",
    "exposure_seconds",
    "centroid_x_stamp_pixel_0based",
    "centroid_y_stamp_pixel_0based",
    "aperture_valid",
    "aperture_usable_pixel_count",
    "aperture_invalid_pixel_count",
    "saturated_pixel_count",
    "cosmic_pixel_count",
    "background_usable_pixel_count",
    "quality_bitmask",
)
_CENTROID_QUALITY_HDF_DATASETS = {
    column: {
        "centroid_x_stamp_pixel_0based": "centroid_x",
        "centroid_y_stamp_pixel_0based": "centroid_y",
    }.get(column, column)
    for column in _CENTROID_QUALITY_COLUMNS
    if column != "cadence_seconds"
}
_CENTROID_QUALITY_BOOLEAN_COLUMNS = frozenset({"aperture_valid"})
_CDPP_SCHEMA_ID = "et_mainsim.stamp_science_cdpp.v1"
_CDPP_TABLE_SCHEMA_ID = "et_mainsim.stamp_science_cdpp_table.v1"
_CDPP_ESTIMATOR = (
    "legacy_median_centered_mean_absolute_deviation_times_1.4826"
)
_CDPP_TABLE_COLUMNS = (
    "cadence_seconds",
    "background_estimator",
    "window_minutes",
    "total_bin_count",
    "accepted_bin_count",
    "rejected_bin_count",
    "accepted_sample_count",
    "minimum_coverage_fraction",
    "minimum_accepted_bins",
    "observed_cdpp_ppm",
    "residual_cdpp_ppm",
)
_CDPP_BINNED_ROW_FIELDS = {
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
}
_APERTURE_DEFINITION_SCHEMA_ID = (
    "et_mainsim.science_optimal_aperture_definition.v2"
)
_APERTURE_DEFINITION_SCHEMA_VERSION = 2
_APERTURE_DEFINITION_FIELDS = frozenset(
    {
        "schema_id",
        "schema_version",
        "algorithm",
        "maximum_cumulative_snr",
        "signal_template_shape",
        "target_peak_yx",
        "training_raw_frame_indices",
        "aperture_pixel_count",
        "background_pixel_count",
        "files",
        "metadata",
    }
)
_QUALITY_SUMMARY_SCHEMA_ID = "et_mainsim.stamp_science_quality_summary.v2"
_QUALITY_SUMMARY_SCHEMA_VERSION = 2
_QUALITY_FLAG_BITS = {
    "aperture_invalid": 1 << 0,
    "aperture_saturated": 1 << 1,
    "aperture_cosmic": 1 << 2,
    "insufficient_background": 1 << 3,
    "centroid_unavailable": 1 << 4,
}
_QUALITY_SUMMARY_COORDINATE_SYSTEM = "zero_based_stamp_local_x_column_y_row"
_REPRESENTATIVE_FRAMES_SCHEMA_ID = (
    "et_mainsim.representative_calibrated_stamp_frames.v2"
)
_REPRESENTATIVE_FRAMES_SCHEMA_VERSION = 2
_BASE_ANALYSIS_ARTIFACT_NAMES = (
    "photometry.h5",
    "photometry.ecsv",
    "reference_lightcurve.ecsv",
    "centroid_quality.ecsv",
    "aperture_definition.json",
    "cdpp.json",
    "cdpp.ecsv",
    "quality_summary.json",
    "aperture_mask.npy",
    "background_mask.npy",
    "representative_calibrated_frames.h5",
    "figures/lightcurve_overview.png",
    "figures/cdpp_summary.png",
    "figures/representative_frames.png",
)
_SCIENCE_TEMPLATE_ARTIFACT_NAMES = (
    "signal_template_e.npy",
    "noise_template_e.npy",
)
_REPRESENTATIVE_FRAME_DATASET_NAMES = {
    "final_dn",
    "calibrated_e",
    "calibrated_bgsub_e",
    "background_expectation_e",
    "captured_flux_fraction",
    "captured_flux_denominator_e",
    "captured_flux_qa_pass",
    "valid_mask",
    "saturated_mask",
    "cosmic_mask",
    "bias_level_sum_dn",
    "column_noise_sum_dn_by_x",
    "gain_e_per_dn",
    "time_start_seconds",
    "exposure_seconds",
    "raw_frame_start_index",
    "raw_frame_stop_index_exclusive",
    "selection_role",
    "input_shard_path",
    "input_shard_semantic_sha256",
}
_REPRESENTATIVE_FRAME_SELECTION_POLICY = (
    "first_clean_then_first_clean_at_or_after_midpoint_then_last_clean_v1"
)
_REPRESENTATIVE_FRAME_CLEAN_DEFINITION = (
    "all_science_aperture_pixels_valid_not_saturated_not_cosmic"
)
_REPRESENTATIVE_FRAME_SELECTION_ROLES = (
    "first_clean",
    "middle_clean",
    "last_clean",
)
_CAPTURED_FLUX_QA_DEFINITION = (
    "pass_requires_no_detector_edge_or_requested_window_truncation"
)
_CAPTURED_FLUX_QA_DENOMINATOR = "source_effective_photon_count_electron"
_DIRECT_COADD_PARITY_COMPARISON = (
    "all_formal_planes_exact_except_float_rtol_1e-12_atol_1e-8"
)
_DIRECT_COADD_PARITY_FIELDS = {
    "required",
    "passed",
    "comparison",
    "sample_count",
    "samples",
}
_DIRECT_COADD_PARITY_SAMPLE_FIELDS = {
    "factor",
    "path",
    "local_frame_index",
    "raw_frame_start_index",
}
_REPRESENTATIVE_FRAME_CONTRACT_FIELDS = {
    "selection_role",
    "raw_frame_start_index",
    "input_shard_index",
    "input_shard_path",
    "input_shard_semantic_sha256",
}
_EXPECTATION_ONLY_BACKGROUND_STRATEGY = "delivered_expectation_only"
_LOCAL_DIAGNOSTIC_BACKGROUND_STRATEGY = (
    "delivered_expectation_plus_local_diagnostic"
)
_REFERENCE_LIGHTCURVE_REQUIRED_COLUMNS = tuple(
    sorted(
        (
            "cadence_seconds",
            "time_start_seconds",
            "exposure_seconds",
            "raw_frame_start_index",
            "raw_frame_stop_index_exclusive",
            "raw_relative_flux_mean",
            "raw_relative_flux_sum",
            "flux_expectation_bgsub_e",
            "flux_expectation_bgsub_e_per_s",
            "aperture_valid",
            "quality_bitmask",
            "captured_flux_fraction",
            "captured_flux_denominator_e",
            "captured_flux_qa_pass",
            "fitted_flux_expectation_e",
            "fitted_flux_expectation_e_per_s",
            "residual_expectation_e",
            "residual_expectation_ppm",
        )
    )
)


class StampScienceAnalysisContractError(ValueError):
    """Raised when formal inputs or a publication violate the v2 wire contract."""


def _reference_lightcurve_contract_v1() -> dict[str, Any]:
    return {
        "artifact": "reference_lightcurve.ecsv",
        "schema_id": _REFERENCE_LIGHTCURVE_SCHEMA_ID,
        "schema_version": _REFERENCE_LIGHTCURVE_SCHEMA_VERSION,
        "measured_flux_column": "flux_expectation_bgsub_e",
        "measured_rate_column": "flux_expectation_bgsub_e_per_s",
        "validity_column": "aperture_valid",
        "quality_column": "quality_bitmask",
        "fitted_flux_column": "fitted_flux_expectation_e",
        "residual_columns": [
            "residual_expectation_e",
            "residual_expectation_ppm",
        ],
        "required_columns": list(_REFERENCE_LIGHTCURVE_REQUIRED_COLUMNS),
    }


def _photometry_table_metadata_v2() -> dict[str, Any]:
    return {
        "schema_id": _PHOTOMETRY_TABLE_SCHEMA_ID,
        "schema_version": _PHOTOMETRY_TABLE_SCHEMA_VERSION,
        "observation_product": "final_dn",
        "background_realization_used": False,
    }


def _centroid_quality_metadata_v1() -> dict[str, Any]:
    return {
        "schema_id": _CENTROID_QUALITY_SCHEMA_ID,
        "centroid_coordinate_system": "zero_based_stamp_local_x_column_y_row",
    }


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise StampScienceAnalysisContractError(f"{name} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise StampScienceAnalysisContractError(
            f"{name} must be a positive integer"
        ) from error
    if result <= 0 or result != value:
        raise StampScienceAnalysisContractError(f"{name} must be a positive integer")
    return result


def _finite_nonnegative(value: object, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise StampScienceAnalysisContractError(
            f"{name} must be finite and non-negative"
        ) from error
    if not math.isfinite(result) or result < 0.0:
        raise StampScienceAnalysisContractError(
            f"{name} must be finite and non-negative"
        )
    return result


def _json_mapping(value: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise StampScienceAnalysisContractError(f"{name} must be a JSON object")
    try:
        encoded = json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise StampScienceAnalysisContractError(
            f"{name} must be JSON serializable without NaN"
        ) from error
    if not isinstance(decoded, dict):  # pragma: no cover - mapping preserves object.
        raise StampScienceAnalysisContractError(f"{name} must be a JSON object")
    return decoded


def _is_native_schema_version(value: object, expected: int) -> bool:
    return type(value) is int and value == expected


def _is_hdf_integer_schema_version(value: object, expected: int) -> bool:
    return bool(
        not isinstance(value, (bool, np.bool_))
        and isinstance(value, (int, np.integer))
        and value == expected
    )


def _is_hdf_boolean(value: object, expected: bool) -> bool:
    return bool(
        type(value) in {bool, np.bool_}
        and bool(value) is expected
    )


def _canonical_json_text(value: object, *, name: str) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise StampScienceAnalysisContractError(
            f"{name} must be canonical finite JSON"
        ) from error


def _analysis_artifact_names_v1(analysis_product: object) -> tuple[str, ...]:
    if analysis_product == "reference_fixed13_v1":
        return _BASE_ANALYSIS_ARTIFACT_NAMES
    if analysis_product == "science_optimal_aperture_v1":
        return (
            *_BASE_ANALYSIS_ARTIFACT_NAMES,
            *_SCIENCE_TEMPLATE_ARTIFACT_NAMES,
        )
    raise StampScienceAnalysisContractError(
        "analysis contract analysis_product is invalid"
    )


def _positive_factor_vector(value: ArrayLike) -> NDArray[np.float64]:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise StampScienceAnalysisContractError(
            "raw_relative_flux must be a finite positive vector"
        ) from error
    if (
        result.ndim != 1
        or result.size == 0
        or not np.all(np.isfinite(result))
        or np.any(result <= 0.0)
    ):
        raise StampScienceAnalysisContractError(
            "raw_relative_flux must be a finite positive vector"
        )
    return result


@dataclass(frozen=True)
class StampScienceAnalysisPolicy:
    """Frozen streaming, parity, and numerical policy for one target."""

    coadd_factors: tuple[int, ...] = _REQUIRED_COADD_FACTORS
    raw_exposure_seconds: float = 10.0
    stream_batch_frames: int = 64
    direct_coadd_samples_per_shard: int = 3
    require_direct_coadd_parity: bool = True
    photometry: StampSciencePhotometryPolicy = field(
        default_factory=StampSciencePhotometryPolicy
    )

    def __post_init__(self) -> None:
        factors = tuple(
            _positive_integer(value, name="coadd factor")
            for value in self.coadd_factors
        )
        if not factors or factors[0] != 1 or tuple(sorted(set(factors))) != factors:
            raise StampScienceAnalysisContractError(
                "coadd_factors must be unique, increasing, and begin with 1"
            )
        if any(value > np.iinfo(np.uint16).max for value in factors):
            raise StampScienceAnalysisContractError(
                "coadd_factors exceed formal uint16 quality-count capacity"
            )
        raw_seconds = _finite_nonnegative(
            self.raw_exposure_seconds,
            name="raw_exposure_seconds",
        )
        if raw_seconds <= 0.0:
            raise StampScienceAnalysisContractError(
                "raw_exposure_seconds must be positive"
            )
        object.__setattr__(self, "coadd_factors", factors)
        object.__setattr__(self, "raw_exposure_seconds", raw_seconds)
        object.__setattr__(
            self,
            "stream_batch_frames",
            _positive_integer(self.stream_batch_frames, name="stream_batch_frames"),
        )
        object.__setattr__(
            self,
            "direct_coadd_samples_per_shard",
            _positive_integer(
                self.direct_coadd_samples_per_shard,
                name="direct_coadd_samples_per_shard",
            ),
        )
        if not isinstance(self.require_direct_coadd_parity, (bool, np.bool_)):
            raise StampScienceAnalysisContractError(
                "require_direct_coadd_parity must be boolean"
            )
        object.__setattr__(
            self,
            "require_direct_coadd_parity",
            bool(self.require_direct_coadd_parity),
        )
        if not isinstance(self.photometry, StampSciencePhotometryPolicy):
            raise TypeError("photometry must be a StampSciencePhotometryPolicy")
        if self.photometry.minimum_accepted_bins < 2:
            raise StampScienceAnalysisContractError(
                "photometry.minimum_accepted_bins must be at least two"
            )

    def to_dict(self) -> dict[str, Any]:
        photometry = {
            name: getattr(self.photometry, name)
            for name in self.photometry.__dataclass_fields__
        }
        photometry["cdpp_windows_minutes"] = list(
            self.photometry.cdpp_windows_minutes
        )
        return {
            "coadd_factors": list(self.coadd_factors),
            "raw_exposure_seconds": self.raw_exposure_seconds,
            "stream_batch_frames": self.stream_batch_frames,
            "direct_coadd_samples_per_shard": (
                self.direct_coadd_samples_per_shard
            ),
            "require_direct_coadd_parity": self.require_direct_coadd_parity,
            "photometry": photometry,
        }


def _valid_training_raw_frame_indices(value: object) -> bool:
    try:
        training = np.asarray(value)
    except (TypeError, ValueError):
        return False
    return bool(
        training.ndim == 1
        and training.size > 0
        and training.dtype.kind in {"i", "u"}
        and not np.any(training < 0)
        and not np.any(training > np.iinfo(np.int64).max)
        and (
            training.size == 1
            or not np.any(training[1:] <= training[:-1])
        )
    )


def _validate_frozen_aperture_definition(
    value: ScienceApertureDefinition,
) -> ScienceApertureDefinition:
    if not isinstance(value, ScienceApertureDefinition):
        raise StampScienceAnalysisContractError(
            "frozen_aperture must be a ScienceApertureDefinition"
        )
    aperture = np.asarray(value.aperture_mask)
    background = (
        None if value.background_mask is None else np.asarray(value.background_mask)
    )
    signal = (
        None if value.signal_template_e is None else np.asarray(value.signal_template_e)
    )
    noise = (
        None if value.noise_template_e is None else np.asarray(value.noise_template_e)
    )
    training = (
        None
        if value.training_raw_frame_indices is None
        else np.asarray(value.training_raw_frame_indices)
    )
    metadata = value.metadata if isinstance(value.metadata, Mapping) else {}
    background_strategy = metadata.get("background_strategy")
    if background_strategy == _EXPECTATION_ONLY_BACKGROUND_STRATEGY:
        background_invalid = background is not None
    else:
        background_invalid = bool(
            background is None
            or background.shape != aperture.shape
            or background.dtype.kind not in {"b", "i", "u"}
            or not np.all((background == 0) | (background == 1))
            or not np.any(background)
            or np.any(
                np.asarray(aperture, dtype=bool)
                & np.asarray(background, dtype=bool)
            )
        )
    if (
        aperture.ndim != 2
        or aperture.dtype.kind not in {"b", "i", "u"}
        or not np.all((aperture == 0) | (aperture == 1))
        or not np.any(aperture)
        or background_invalid
        or signal is None
        or noise is None
        or signal.shape != aperture.shape
        or noise.shape != aperture.shape
        or not np.all(np.isfinite(signal))
        or np.any(signal < 0.0)
        or not np.all(np.isfinite(noise))
        or np.any(noise <= 0.0)
        or tuple(value.signal_template_shape) != aperture.shape
        or training is None
        or not _valid_training_raw_frame_indices(training)
        or not isinstance(value.algorithm, str)
        or not value.algorithm
        or not math.isfinite(float(value.maximum_cumulative_snr))
        or float(value.maximum_cumulative_snr) <= 0.0
    ):
        raise StampScienceAnalysisContractError(
            "frozen_aperture violates the complete mask/template contract"
        )
    return value


@dataclass(frozen=True)
class StampScienceAnalysisRequest:
    """Immutable request for one target/case formal raw-shard series."""

    raw_bundle_paths: Sequence[Path | str]
    output_dir: Path | str
    raw_relative_flux: ArrayLike
    raw_relative_flux_identity: Mapping[str, Any]
    code_identity: Mapping[str, Any]
    analysis_context: Mapping[str, Any]
    read_noise_e_per_pixel: float
    quantization_noise_e_per_pixel: float
    direct_coadd_bundle_paths: Mapping[int, Sequence[Path | str]] = field(
        default_factory=dict
    )
    policy: StampScienceAnalysisPolicy = field(
        default_factory=StampScienceAnalysisPolicy
    )
    aperture_mode: Literal["train", "reuse_published"] | str = "train"
    frozen_aperture: ScienceApertureDefinition | None = None
    aperture_source_identity: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        raw_paths = tuple(
            Path(path).expanduser().resolve() for path in self.raw_bundle_paths
        )
        if not raw_paths or len(set(raw_paths)) != len(raw_paths):
            raise StampScienceAnalysisContractError(
                "raw_bundle_paths must contain unique formal bundle paths"
            )
        direct: dict[int, tuple[Path, ...]] = {}
        for factor, paths in self.direct_coadd_bundle_paths.items():
            resolved_factor = _positive_integer(factor, name="direct coadd factor")
            resolved_paths = tuple(
                Path(path).expanduser().resolve() for path in paths
            )
            if not resolved_paths or len(set(resolved_paths)) != len(resolved_paths):
                raise StampScienceAnalysisContractError(
                    f"direct coadd factor {resolved_factor} paths must be unique"
                )
            direct[resolved_factor] = resolved_paths
        output = Path(self.output_dir).expanduser().resolve()
        if output.name in {"", ".", ".."}:
            raise StampScienceAnalysisContractError(
                "output_dir must name a final product directory"
            )
        if not isinstance(self.policy, StampScienceAnalysisPolicy):
            raise TypeError("policy must be a StampScienceAnalysisPolicy")
        expected_direct = set(self.policy.coadd_factors) - {1}
        if self.policy.require_direct_coadd_parity and set(direct) != expected_direct:
            raise StampScienceAnalysisContractError(
                "direct_coadd_bundle_paths must contain every configured non-raw factor"
            )
        if not set(direct).issubset(expected_direct):
            raise StampScienceAnalysisContractError(
                "direct_coadd_bundle_paths contains an unconfigured factor"
            )
        object.__setattr__(self, "raw_bundle_paths", raw_paths)
        object.__setattr__(self, "direct_coadd_bundle_paths", direct)
        object.__setattr__(self, "output_dir", output)
        object.__setattr__(
            self,
            "raw_relative_flux",
            _positive_factor_vector(self.raw_relative_flux),
        )
        object.__setattr__(
            self,
            "raw_relative_flux_identity",
            _json_mapping(
                self.raw_relative_flux_identity,
                name="raw_relative_flux_identity",
            ),
        )
        object.__setattr__(
            self,
            "code_identity",
            _json_mapping(self.code_identity, name="code_identity"),
        )
        context = _json_mapping(self.analysis_context, name="analysis_context")
        case = context.get("case")
        source_id = context.get("source_id")
        if case not in {"static", "injected"}:
            raise StampScienceAnalysisContractError(
                "analysis_context.case must be 'static' or 'injected'"
            )
        if not isinstance(source_id, str) or not source_id:
            raise StampScienceAnalysisContractError(
                "analysis_context.source_id must be a non-empty string"
            )
        object.__setattr__(self, "analysis_context", context)
        mode = self.aperture_mode
        if mode not in {"train", "reuse_published"}:
            raise StampScienceAnalysisContractError(
                "aperture_mode must be 'train' or 'reuse_published'"
            )
        source_identity = _json_mapping(
            self.aperture_source_identity,
            name="aperture_source_identity",
        )
        if mode == "train":
            if self.frozen_aperture is not None or source_identity:
                raise StampScienceAnalysisContractError(
                    "aperture_mode='train' forbids a frozen aperture/source identity"
                )
        else:
            if not source_identity:
                raise StampScienceAnalysisContractError(
                    "aperture_mode='reuse_published' requires aperture_source_identity"
                )
            _validate_frozen_aperture_definition(self.frozen_aperture)  # type: ignore[arg-type]
        q = np.asarray(self.raw_relative_flux, dtype=np.float64)
        if case == "static" and mode != "reuse_published":
            raise StampScienceAnalysisContractError(
                "static analysis requires aperture_mode='reuse_published'"
            )
        if case == "static" and not np.array_equal(q, np.ones(q.shape)):
            raise StampScienceAnalysisContractError(
                "static analysis requires an all-unity raw_relative_flux"
            )
        object.__setattr__(self, "aperture_mode", mode)
        object.__setattr__(self, "aperture_source_identity", source_identity)
        object.__setattr__(
            self,
            "read_noise_e_per_pixel",
            _finite_nonnegative(
                self.read_noise_e_per_pixel,
                name="read_noise_e_per_pixel",
            ),
        )
        object.__setattr__(
            self,
            "quantization_noise_e_per_pixel",
            _finite_nonnegative(
                self.quantization_noise_e_per_pixel,
                name="quantization_noise_e_per_pixel",
            ),
        )


@dataclass(frozen=True)
class StampScienceAnalysisPublication:
    output_dir: Path
    hdf5_path: Path
    ecsv_path: Path
    manifest_path: Path
    aperture_definition_path: Path
    cdpp_path: Path
    aperture_mask_path: Path
    background_mask_path: Path
    representative_frames_path: Path


@dataclass(frozen=True)
class StampScienceAnalysisProductSetPublication:
    """Atomic two-product publication for one source/case analysis root."""

    output_dir: Path
    manifest_path: Path
    reference_fixed13: StampScienceAnalysisPublication
    science_optimal_aperture: StampScienceAnalysisPublication


@dataclass(frozen=True)
class StampScienceAnalysisValidation:
    output_dir: Path
    complete: bool
    cadence_seconds: tuple[int, ...]
    raw_frame_count: int
    aperture_pixel_count: int


@dataclass(frozen=True)
class StampScienceAnalysisProductSetValidation:
    output_dir: Path
    manifest_path: Path
    complete: bool
    products: Mapping[str, StampScienceAnalysisValidation]


@dataclass(frozen=True)
class _FileStat:
    device: int
    inode: int
    size_bytes: int
    mtime_ns: int

    @classmethod
    def from_path(cls, path: Path) -> "_FileStat":
        value = path.stat()
        return cls(
            device=int(value.st_dev),
            inode=int(value.st_ino),
            size_bytes=int(value.st_size),
            mtime_ns=int(value.st_mtime_ns),
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "device": self.device,
            "inode": self.inode,
            "size_bytes": self.size_bytes,
            "mtime_ns": self.mtime_ns,
        }


@dataclass(frozen=True)
class _InputHeader:
    formal: _FormalDeliveryHeader
    chunk_frames: int
    initial_stat: _FileStat
    cross_product_manifest_identity: str
    cross_product_provenance_identity: str
    exact_manifest_identity: str
    exact_provenance_identity: str
    target_source_id: str
    case: str
    run_id: str
    production_manifest_reference: str
    production_manifest_content_identity: Mapping[str, Any] | None


def _canonical_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_series_manifest_json(manifest: Mapping[str, Any]) -> str:
    """Remove only producer-declared shard/product execution audit fields.

    The physical RNG block deliberately retains the seed, canonical context,
    target-spec identity, formula, and every other science-defining value.
    Only the selected shard's absolute offset/interval audit is omitted.
    """

    candidate = _json_mapping(manifest, name="formal delivery manifest_json")
    for name in ("time_shard", "product_kind", "coadd_factor"):
        candidate.pop(name, None)
    return _formal_series_identity_json(
        candidate,
        caller_key="caller_manifest",
        omit_top_level_time_shard=True,
    )


def _canonical_series_provenance_json(
    provenance: Mapping[str, Any],
) -> str:
    candidate = _json_mapping(
        provenance,
        name="formal delivery provenance_json",
    )
    for name in ("product_kind", "coadd_factor"):
        candidate.pop(name, None)
    return _formal_series_identity_json(
        candidate,
        caller_key="caller_provenance",
        omit_top_level_time_shard=False,
    )


def _require_input_capture_qa_all_true(
    handle: Any,
    *,
    path: Path,
    frame_count: int,
) -> None:
    """Scan the tiny cadence gate before any expensive analysis work starts."""

    if "captured_flux_qa_pass" not in handle:
        raise StampScienceAnalysisContractError(
            "formal input lacks captured_flux_qa_pass"
        )
    dataset = handle["captured_flux_qa_pass"]
    if dataset.shape != (frame_count,):
        raise StampScienceAnalysisContractError(
            "formal input captured_flux_qa_pass axis differs"
        )
    chunk_frames = (
        int(dataset.chunks[0])
        if dataset.chunks is not None and int(dataset.chunks[0]) > 0
        else min(frame_count, 65_536)
    )
    for start in range(0, frame_count, chunk_frames):
        values = np.asarray(
            dataset[start : min(frame_count, start + chunk_frames)]
        )
        if values.dtype.kind not in {"b", "i", "u"} or not np.all(
            (values == 0) | (values == 1)
        ):
            raise StampScienceAnalysisContractError(
                "formal input captured_flux_qa_pass is not binary"
            )
        if not np.all(values == 1):
            raise StampScienceAnalysisContractError(
                "formal input captured_flux_qa_pass contains false: "
                f"{path}"
            )


def _read_input_header(path: Path) -> _InputHeader:
    """Centralized adapter around reference_photometry's formal helpers."""

    try:
        formal = _read_formal_delivery_header(path)
        import h5py

        with h5py.File(formal.path, "r") as handle:
            chunks = handle["final_dn"].chunks
            if chunks is None or len(chunks) != 3 or int(chunks[0]) <= 0:
                raise StampScienceAnalysisContractError(
                    "formal final_dn must use bounded frame-axis chunks"
                )
            manifest = _formal_json_dataset(handle, "manifest_json")
            provenance = _formal_json_dataset(handle, "provenance_json")
            _require_input_capture_qa_all_true(
                handle,
                path=formal.path,
                frame_count=formal.frame_count,
            )
        cross_manifest = _canonical_series_manifest_json(manifest)
        cross_provenance = _canonical_series_provenance_json(provenance)
        target_value = manifest.get(
            "target_source_id", manifest.get("target_source_id_int64")
        )
        caller = manifest.get("caller_manifest")
        if not isinstance(caller, Mapping):
            raise StampScienceAnalysisContractError(
                "formal delivery caller_manifest must be an object"
            )
        case = caller.get("case")
        run_id = caller.get("run_id")
        production_reference = caller.get(
            "production_manifest", caller.get("galaxy_production_manifest")
        )
        production_identity = caller.get(
            "production_manifest_identity",
            caller.get("galaxy_production_manifest_identity"),
        )
        if (
            target_value is None
            or case not in {"static", "injected"}
            or not isinstance(run_id, str)
            or not run_id
            or not isinstance(production_reference, str)
            or not production_reference
            or (
                production_identity is not None
                and not isinstance(production_identity, Mapping)
            )
        ):
            raise StampScienceAnalysisContractError(
                "formal delivery lacks target/case/run/production-manifest binding"
            )
    except StampScienceAnalysisContractError:
        raise
    except (ReferencePhotometryContractError, OSError, ValueError) as error:
        raise StampScienceAnalysisContractError(str(error)) from error
    return _InputHeader(
        formal=formal,
        chunk_frames=int(chunks[0]),
        initial_stat=_FileStat.from_path(formal.path),
        cross_product_manifest_identity=_canonical_digest(cross_manifest),
        cross_product_provenance_identity=_canonical_digest(cross_provenance),
        exact_manifest_identity=_formal_identity_json(
            manifest,
            omit=frozenset(),
        ),
        exact_provenance_identity=_formal_identity_json(
            provenance,
            omit=frozenset(),
        ),
        target_source_id=str(target_value),
        case=str(case),
        run_id=run_id,
        production_manifest_reference=production_reference,
        production_manifest_content_identity=(
            None if production_identity is None else dict(production_identity)
        ),
    )


def _static_gain_matches(left: _InputHeader, right: _InputHeader) -> bool:
    left_gain = left.formal.static_gain_e_per_dn
    right_gain = right.formal.static_gain_e_per_dn
    return (
        left.formal.gain_mode == right.formal.gain_mode
        and left_gain is not None
        and right_gain is not None
        and np.array_equal(left_gain, right_gain)
    )


def _read_series_headers(
    paths: Sequence[Path],
    *,
    product_kind: Literal["raw", "coadd"],
    coadd_factor: int,
) -> tuple[_InputHeader, ...]:
    headers = tuple(_read_input_header(path) for path in paths)
    headers = tuple(sorted(headers, key=lambda item: item.formal.first_raw_frame_start))
    if not headers:  # Request validation normally prevents this.
        raise StampScienceAnalysisContractError("formal series must not be empty")
    first = headers[0]
    previous_raw_stop: int | None = None
    previous_time_end: float | None = None
    for header in headers:
        formal = header.formal
        if formal.gain_mode == "per_frame_stamp_map":
            raise StampScienceAnalysisContractError(
                "formal science analysis does not support per-frame gain maps"
            )
        if (
            formal.product_kind != product_kind
            or formal.coadd_factor != coadd_factor
        ):
            raise StampScienceAnalysisContractError(
                "formal series product kind/coadd factor is incompatible"
            )
        if (
            formal.stamp_shape != first.formal.stamp_shape
            or header.cross_product_manifest_identity
            != first.cross_product_manifest_identity
            or header.cross_product_provenance_identity
            != first.cross_product_provenance_identity
            or not _static_gain_matches(header, first)
        ):
            raise StampScienceAnalysisContractError(
                "formal series contains incompatible shard identities"
            )
        if previous_raw_stop is not None and (
            formal.first_raw_frame_start != previous_raw_stop
            or not math.isclose(
                formal.first_time_start_seconds,
                float(previous_time_end),
                rel_tol=0.0,
                abs_tol=_TIME_ATOL_SECONDS,
            )
        ):
            raise StampScienceAnalysisContractError(
                "formal series shards are not globally continuous"
            )
        previous_raw_stop = formal.last_raw_frame_stop
        previous_time_end = formal.last_time_end_seconds
    return headers


def _validate_cross_product_headers(
    raw: tuple[_InputHeader, ...],
    direct: Mapping[int, tuple[_InputHeader, ...]],
) -> None:
    first = raw[0]
    raw_start = first.formal.first_raw_frame_start
    raw_stop = raw[-1].formal.last_raw_frame_stop
    raw_time_start = first.formal.first_time_start_seconds
    raw_time_stop = raw[-1].formal.last_time_end_seconds
    for factor, headers in direct.items():
        candidate = headers[0]
        if (
            candidate.formal.stamp_shape != first.formal.stamp_shape
            or not _static_gain_matches(candidate, first)
            or candidate.cross_product_manifest_identity
            != first.cross_product_manifest_identity
            or candidate.cross_product_provenance_identity
            != first.cross_product_provenance_identity
        ):
            raise StampScienceAnalysisContractError(
                f"direct coadd factor {factor} has incompatible science identity"
            )
        if (
            candidate.formal.first_raw_frame_start != raw_start
            or headers[-1].formal.last_raw_frame_stop != raw_stop
            or not math.isclose(
                candidate.formal.first_time_start_seconds,
                raw_time_start,
                rel_tol=0.0,
                abs_tol=_TIME_ATOL_SECONDS,
            )
            or not math.isclose(
                headers[-1].formal.last_time_end_seconds,
                raw_time_stop,
                rel_tol=0.0,
                abs_tol=_TIME_ATOL_SECONDS,
            )
        ):
            raise StampScienceAnalysisContractError(
                f"direct coadd factor {factor} does not cover the raw series"
            )


@dataclass(frozen=True)
class _DeliveryBatch:
    final_dn: NDArray[np.unsignedinteger]
    background_expectation_e: NDArray[np.float64]
    captured_flux_fraction: NDArray[np.float64]
    captured_flux_denominator_e: NDArray[np.float64]
    captured_flux_qa_pass: NDArray[np.bool_]
    bias_level_sum_dn: NDArray[np.float64]
    column_noise_sum_dn_by_x: NDArray[np.float64]
    valid_mask: NDArray[np.bool_]
    fullwell_count: NDArray[np.uint16]
    adc_low_count: NDArray[np.uint16]
    adc_high_count: NDArray[np.uint16]
    cosmic_count: NDArray[np.uint16]
    time_start_seconds: NDArray[np.float64]
    exposure_seconds: NDArray[np.float64]
    raw_frame_start_index: NDArray[np.int64]
    raw_frame_stop_index_exclusive: NDArray[np.int64]
    gain_e_per_dn: NDArray[np.float64]

    @property
    def frame_count(self) -> int:
        return int(self.final_dn.shape[0])

    @property
    def saturated_mask(self) -> NDArray[np.bool_]:
        return (
            (self.fullwell_count > 0)
            | (self.adc_low_count > 0)
            | (self.adc_high_count > 0)
        )

    @property
    def cosmic_mask(self) -> NDArray[np.bool_]:
        return self.cosmic_count > 0

    def to_photometry_input(self) -> ReferencePhotometryInput:
        return ReferencePhotometryInput.from_arrays(
            final_dn=self.final_dn,
            background_expectation_e=self.background_expectation_e,
            bias_level_sum_dn=self.bias_level_sum_dn,
            column_noise_sum_dn_by_x=self.column_noise_sum_dn_by_x,
            valid_mask=self.valid_mask,
            saturated_mask=self.saturated_mask,
            cosmic_mask=self.cosmic_mask,
            time_index=self.time_start_seconds,
            gain_e_per_dn=self.gain_e_per_dn,
            time_index_unit="seconds",
            exposure_seconds=self.exposure_seconds,
        )


def _binary_batch_mask(value: Any, *, name: str, shape: tuple[int, ...]) -> NDArray[np.bool_]:
    result = np.asarray(value)
    if (
        result.shape != shape
        or result.dtype.kind not in {"b", "i", "u"}
        or not np.all((result == 0) | (result == 1))
    ):
        raise StampScienceAnalysisContractError(
            f"formal delivery {name} is not a binary mask"
        )
    return np.asarray(result, dtype=bool)


def _read_delivery_batch(
    handle: Any,
    header: _InputHeader,
    frame_slice: slice,
) -> _DeliveryBatch:
    """Read and validate one explicitly bounded full-stamp frame slice."""

    start = frame_slice.start
    stop = frame_slice.stop
    if (
        frame_slice.step not in (None, 1)
        or start is None
        or stop is None
        or not isinstance(start, int)
        or not isinstance(stop, int)
        or start < 0
        or stop <= start
        or stop > header.formal.frame_count
    ):
        raise StampScienceAnalysisContractError(
            "formal delivery reads require a positive bounded frame slice"
        )
    slc = slice(start, stop)
    final = np.asarray(handle["final_dn"][slc])
    n_frames, ny, nx = final.shape
    shape = (n_frames, ny, nx)
    if shape[1:] != header.formal.stamp_shape or final.dtype.kind != "u":
        raise StampScienceAnalysisContractError(
            "formal delivery final_dn batch violates its header"
        )
    background = np.asarray(
        handle["background_expectation_e"][slc], dtype=np.float64
    )
    captured_fraction = np.asarray(
        handle["captured_flux_fraction"][slc], dtype=np.float64
    )
    captured_denominator = np.asarray(
        handle["captured_flux_denominator_e"][slc], dtype=np.float64
    )
    captured_qa = _binary_batch_mask(
        handle["captured_flux_qa_pass"][slc],
        name="captured_flux_qa_pass",
        shape=(n_frames,),
    )
    if not np.all(captured_qa):
        raise StampScienceAnalysisContractError(
            "formal input captured_flux_qa_pass contains false"
        )
    bias = np.asarray(handle["bias_level_sum_dn"][slc], dtype=np.float64)
    column = np.asarray(
        handle["column_noise_sum_dn_by_x"][slc], dtype=np.float64
    )
    valid = _binary_batch_mask(
        handle["valid_mask"][slc], name="valid_mask", shape=shape
    )
    counts: dict[str, NDArray[np.uint16]] = {}
    for name in (
        "fullwell_count",
        "adc_low_count",
        "adc_high_count",
        "cosmic_count",
    ):
        raw_count = np.asarray(handle[name][slc])
        if (
            raw_count.shape != shape
            or raw_count.dtype.kind not in {"i", "u"}
            or np.any(raw_count < 0)
            or np.any(raw_count > header.formal.coadd_factor)
        ):
            raise StampScienceAnalysisContractError(
                f"formal delivery {name} violates the quality-count contract"
            )
        counts[name] = np.asarray(raw_count, dtype=np.uint16)
    saturated = _binary_batch_mask(
        handle["saturated_mask"][slc], name="saturated_mask", shape=shape
    )
    cosmic = _binary_batch_mask(
        handle["cosmic_mask"][slc], name="cosmic_mask", shape=shape
    )
    time = np.asarray(handle["time_start_seconds"][slc], dtype=np.float64)
    exposure = np.asarray(handle["exposure_seconds"][slc], dtype=np.float64)
    raw_start = np.asarray(
        handle["raw_frame_start_index"][slc], dtype=np.int64
    )
    raw_stop = np.asarray(
        handle["raw_frame_stop_index_exclusive"][slc], dtype=np.int64
    )
    if (
        background.shape != shape
        or captured_fraction.shape != (n_frames,)
        or captured_denominator.shape != (n_frames,)
        or bias.shape != (n_frames,)
        or column.shape != (n_frames, nx)
        or not np.all(np.isfinite(background))
        or np.any(background < 0.0)
        or not np.all(np.isfinite(captured_fraction))
        or np.any(captured_fraction < 0.0)
        or np.any(captured_fraction > 1.0 + 1.0e-6)
        or not np.all(np.isfinite(captured_denominator))
        or np.any(captured_denominator <= 0.0)
        or not np.all(np.isfinite(bias))
        or not np.all(np.isfinite(column))
        or not np.all(np.isfinite(time))
        or not np.all(np.isfinite(exposure))
        or np.any(exposure <= 0.0)
        or np.any(raw_stop - raw_start != header.formal.coadd_factor)
        or (n_frames > 1 and not _formal_time_intervals_are_contiguous(time, exposure))
        or (n_frames > 1 and not np.all(raw_start[1:] == raw_stop[:-1]))
        or not np.array_equal(
            saturated,
            (counts["fullwell_count"] > 0)
            | (counts["adc_low_count"] > 0)
            | (counts["adc_high_count"] > 0),
        )
        or not np.array_equal(cosmic, counts["cosmic_count"] > 0)
    ):
        raise StampScienceAnalysisContractError(
            "formal delivery batch violates plane/time/quality contracts"
        )
    gain = header.formal.static_gain_e_per_dn
    if gain is None:
        raise StampScienceAnalysisContractError(
            "formal science analysis does not support per-frame gain maps"
        )
    return _DeliveryBatch(
        final_dn=final,
        background_expectation_e=background,
        captured_flux_fraction=captured_fraction,
        captured_flux_denominator_e=captured_denominator,
        captured_flux_qa_pass=captured_qa,
        bias_level_sum_dn=bias,
        column_noise_sum_dn_by_x=column,
        valid_mask=valid,
        fullwell_count=counts["fullwell_count"],
        adc_low_count=counts["adc_low_count"],
        adc_high_count=counts["adc_high_count"],
        cosmic_count=counts["cosmic_count"],
        time_start_seconds=time,
        exposure_seconds=exposure,
        raw_frame_start_index=raw_start,
        raw_frame_stop_index_exclusive=raw_stop,
        gain_e_per_dn=np.asarray(gain, dtype=np.float64),
    )


def _slice_batch(batch: _DeliveryBatch, selection: slice | NDArray[np.int64]) -> _DeliveryBatch:
    return _DeliveryBatch(
        final_dn=batch.final_dn[selection],
        background_expectation_e=batch.background_expectation_e[selection],
        captured_flux_fraction=batch.captured_flux_fraction[selection],
        captured_flux_denominator_e=batch.captured_flux_denominator_e[selection],
        captured_flux_qa_pass=batch.captured_flux_qa_pass[selection],
        bias_level_sum_dn=batch.bias_level_sum_dn[selection],
        column_noise_sum_dn_by_x=batch.column_noise_sum_dn_by_x[selection],
        valid_mask=batch.valid_mask[selection],
        fullwell_count=batch.fullwell_count[selection],
        adc_low_count=batch.adc_low_count[selection],
        adc_high_count=batch.adc_high_count[selection],
        cosmic_count=batch.cosmic_count[selection],
        time_start_seconds=batch.time_start_seconds[selection],
        exposure_seconds=batch.exposure_seconds[selection],
        raw_frame_start_index=batch.raw_frame_start_index[selection],
        raw_frame_stop_index_exclusive=batch.raw_frame_stop_index_exclusive[selection],
        gain_e_per_dn=batch.gain_e_per_dn,
    )


def _concatenate_batches(
    batches: Sequence[_DeliveryBatch],
    *,
    require_contiguous: bool,
) -> _DeliveryBatch:
    if not batches:
        raise StampScienceAnalysisContractError("cannot concatenate no delivery batches")
    first = batches[0]
    for left, right in zip(batches, batches[1:], strict=False):
        if (
            left.final_dn.shape[1:] != right.final_dn.shape[1:]
            or not np.array_equal(left.gain_e_per_dn, right.gain_e_per_dn)
        ):
            raise StampScienceAnalysisContractError(
                "delivery batches have incompatible stamp/gain contracts"
            )
        if require_contiguous and (
            int(right.raw_frame_start_index[0])
            != int(left.raw_frame_stop_index_exclusive[-1])
            or not math.isclose(
                float(right.time_start_seconds[0]),
                float(left.time_start_seconds[-1] + left.exposure_seconds[-1]),
                rel_tol=0.0,
                abs_tol=_TIME_ATOL_SECONDS,
            )
        ):
            raise StampScienceAnalysisContractError(
                "delivery batches are not globally continuous"
            )

    def concat(name: str) -> NDArray[Any]:
        return np.concatenate([getattr(item, name) for item in batches], axis=0)

    return _DeliveryBatch(
        final_dn=concat("final_dn"),
        background_expectation_e=concat("background_expectation_e"),
        captured_flux_fraction=concat("captured_flux_fraction"),
        captured_flux_denominator_e=concat("captured_flux_denominator_e"),
        captured_flux_qa_pass=concat("captured_flux_qa_pass"),
        bias_level_sum_dn=concat("bias_level_sum_dn"),
        column_noise_sum_dn_by_x=concat("column_noise_sum_dn_by_x"),
        valid_mask=concat("valid_mask"),
        fullwell_count=concat("fullwell_count"),
        adc_low_count=concat("adc_low_count"),
        adc_high_count=concat("adc_high_count"),
        cosmic_count=concat("cosmic_count"),
        time_start_seconds=concat("time_start_seconds"),
        exposure_seconds=concat("exposure_seconds"),
        raw_frame_start_index=concat("raw_frame_start_index"),
        raw_frame_stop_index_exclusive=concat("raw_frame_stop_index_exclusive"),
        gain_e_per_dn=first.gain_e_per_dn,
    )


def _grouped_sum(value: NDArray[Any], *, groups: int, factor: int) -> NDArray[Any]:
    return np.sum(
        value.reshape(groups, factor, *value.shape[1:]),
        axis=1,
        dtype=np.float64 if value.dtype.kind == "f" else None,
    )


def _coadd_batch(batch: _DeliveryBatch, *, factor: int) -> _DeliveryBatch:
    if batch.frame_count % factor:
        raise StampScienceAnalysisContractError(
            "internal raw batch is not divisible by its coadd factor"
        )
    groups = batch.frame_count // factor
    ny, nx = batch.final_dn.shape[1:]

    def quality_sum(name: str) -> NDArray[np.uint16]:
        summed = np.sum(
            getattr(batch, name).reshape(groups, factor, ny, nx),
            axis=1,
            dtype=np.uint32,
        )
        if np.any(summed > np.iinfo(np.uint16).max):
            raise StampScienceAnalysisContractError(
                "raw-derived quality count exceeds uint16 capacity"
            )
        return np.asarray(summed, dtype=np.uint16)

    final = np.sum(
        batch.final_dn.reshape(groups, factor, ny, nx),
        axis=1,
        dtype=np.uint64,
    )
    valid = np.all(
        batch.valid_mask.reshape(groups, factor, ny, nx),
        axis=1,
    )
    capture_denominator = np.sum(
        batch.captured_flux_denominator_e.reshape(groups, factor),
        axis=1,
        dtype=np.float64,
    )
    capture_fraction = np.sum(
        (
            batch.captured_flux_fraction
            * batch.captured_flux_denominator_e
        ).reshape(groups, factor),
        axis=1,
        dtype=np.float64,
    ) / capture_denominator
    return _DeliveryBatch(
        final_dn=final,
        background_expectation_e=_grouped_sum(
            batch.background_expectation_e,
            groups=groups,
            factor=factor,
        ),
        captured_flux_fraction=capture_fraction,
        captured_flux_denominator_e=capture_denominator,
        captured_flux_qa_pass=np.all(
            batch.captured_flux_qa_pass.reshape(groups, factor),
            axis=1,
        ),
        bias_level_sum_dn=_grouped_sum(
            batch.bias_level_sum_dn,
            groups=groups,
            factor=factor,
        ),
        column_noise_sum_dn_by_x=_grouped_sum(
            batch.column_noise_sum_dn_by_x,
            groups=groups,
            factor=factor,
        ),
        valid_mask=valid,
        fullwell_count=quality_sum("fullwell_count"),
        adc_low_count=quality_sum("adc_low_count"),
        adc_high_count=quality_sum("adc_high_count"),
        cosmic_count=quality_sum("cosmic_count"),
        time_start_seconds=batch.time_start_seconds.reshape(groups, factor)[:, 0],
        exposure_seconds=np.sum(
            batch.exposure_seconds.reshape(groups, factor),
            axis=1,
            dtype=np.float64,
        ),
        raw_frame_start_index=batch.raw_frame_start_index.reshape(
            groups, factor
        )[:, 0],
        raw_frame_stop_index_exclusive=batch.raw_frame_stop_index_exclusive.reshape(
            groups, factor
        )[:, -1],
        gain_e_per_dn=batch.gain_e_per_dn,
    )


class _FactorAccumulator:
    """Carry fewer than ``factor`` raw frames across bounded read batches."""

    def __init__(self, factor: int) -> None:
        self.factor = factor
        self.carry: _DeliveryBatch | None = None

    def push(self, batch: _DeliveryBatch) -> _DeliveryBatch | None:
        combined = batch
        if self.carry is not None:
            combined = _concatenate_batches(
                (self.carry, batch),
                require_contiguous=True,
            )
        complete_count = (combined.frame_count // self.factor) * self.factor
        if complete_count == 0:
            self.carry = combined
            return None
        complete = _slice_batch(combined, slice(0, complete_count))
        if complete_count < combined.frame_count:
            self.carry = _slice_batch(combined, slice(complete_count, combined.frame_count))
        else:
            self.carry = None
        if self.factor == 1:
            return complete
        return _coadd_batch(complete, factor=self.factor)

    def finish(self) -> None:
        if self.carry is not None:
            raise StampScienceAnalysisContractError(
                f"raw frame count is not divisible by coadd factor {self.factor}"
            )


def _selected_chunk_numbers(n_chunks: int, count: int) -> tuple[int, ...]:
    selected_count = min(n_chunks, count)
    if selected_count == 1:
        return (0,)
    values = np.rint(
        np.linspace(0, n_chunks - 1, num=selected_count)
    ).astype(np.int64)
    unique = tuple(int(value) for value in np.unique(values))
    if len(unique) != selected_count:  # pragma: no cover - defensive rounding guard.
        raise StampScienceAnalysisContractError(
            "could not select unique deterministic training chunks"
        )
    return unique


def _training_slices(
    header: _InputHeader,
    *,
    policy: StampSciencePhotometryPolicy,
) -> tuple[slice, ...]:
    if header.chunk_frames != policy.training_block_frames:
        raise StampScienceAnalysisContractError(
            "formal final_dn chunk size conflicts with training_block_frames"
        )
    n_chunks = int(math.ceil(header.formal.frame_count / header.chunk_frames))
    slices: list[slice] = []
    for chunk_number in _selected_chunk_numbers(
        n_chunks,
        policy.training_blocks_per_shard,
    ):
        start = chunk_number * header.chunk_frames
        stop = min(start + header.chunk_frames, header.formal.frame_count)
        slices.append(slice(start, stop))
    return tuple(slices)


def _assert_raw_batch_cadence(
    batch: _DeliveryBatch,
    *,
    raw_exposure_seconds: float,
) -> None:
    if (
        np.any(batch.raw_frame_stop_index_exclusive - batch.raw_frame_start_index != 1)
        or not np.allclose(
            batch.exposure_seconds,
            raw_exposure_seconds,
            rtol=0.0,
            atol=_TIME_ATOL_SECONDS,
        )
    ):
        raise StampScienceAnalysisContractError(
            "formal raw series conflicts with the frozen raw exposure cadence"
        )


def _train_aperture(
    headers: tuple[_InputHeader, ...],
    *,
    raw_relative_flux: NDArray[np.float64],
    first_raw_index: int,
    request: StampScienceAnalysisRequest,
) -> ScienceApertureDefinition:
    import h5py

    ny, nx = headers[0].formal.stamp_shape
    usable_count = np.zeros((ny, nx), dtype=np.int64)
    denominator = np.zeros((ny, nx), dtype=np.float64)
    numerator = np.zeros((ny, nx), dtype=np.float64)
    background_sum = np.zeros((ny, nx), dtype=np.float64)
    absolute_indices: list[NDArray[np.int64]] = []
    training_frame_count = 0
    excluded_training_sample_count = 0
    for header in headers:
        if _FileStat.from_path(header.formal.path) != header.initial_stat:
            raise StampScienceAnalysisContractError(
                "formal raw shard changed before aperture training"
            )
        with h5py.File(header.formal.path, "r") as handle:
            for frame_slice in _training_slices(
                header,
                policy=request.policy.photometry,
            ):
                batch = _read_delivery_batch(handle, header, frame_slice)
                _assert_raw_batch_cadence(
                    batch,
                    raw_exposure_seconds=request.policy.raw_exposure_seconds,
                )
                absolute_indices.append(batch.raw_frame_start_index)
                q_indices = batch.raw_frame_start_index - first_raw_index
                if np.any(q_indices < 0) or np.any(
                    q_indices >= raw_relative_flux.size
                ):
                    raise StampScienceAnalysisContractError(
                        "training raw-frame indices are outside raw_relative_flux"
                    )
                q = raw_relative_flux[q_indices]
                usable = (
                    batch.valid_mask
                    & ~batch.saturated_mask
                    & ~batch.cosmic_mask
                )
                calibrated = (
                    (
                        batch.final_dn.astype(np.float64)
                        - batch.bias_level_sum_dn[:, None, None]
                        - batch.column_noise_sum_dn_by_x[:, None, :]
                    )
                    * batch.gain_e_per_dn
                    - batch.background_expectation_e
                )
                q_cube = q[:, None, None]
                usable_count += np.count_nonzero(usable, axis=0)
                denominator += np.sum(
                    np.where(usable, q_cube * q_cube, 0.0),
                    axis=0,
                    dtype=np.float64,
                )
                numerator += np.sum(
                    np.where(usable, q_cube * calibrated, 0.0),
                    axis=0,
                    dtype=np.float64,
                )
                background_sum += np.sum(
                    np.where(usable, batch.background_expectation_e, 0.0),
                    axis=0,
                    dtype=np.float64,
                )
                training_frame_count += batch.frame_count
                excluded_training_sample_count += int(
                    usable.size - np.count_nonzero(usable)
                )
    absolute = np.concatenate(absolute_indices)
    if training_frame_count != absolute.size:
        raise StampScienceAnalysisContractError(
            "online training frame/index accounting differs"
        )
    minimum_valid_count = int(
        math.ceil(
            training_frame_count
            * request.policy.photometry.minimum_training_valid_fraction
        )
    )
    permanent_valid = usable_count >= minimum_valid_count
    fitted = permanent_valid & (denominator > 0.0)
    if not np.any(fitted):
        raise StampScienceAnalysisContractError(
            "no pixel satisfies the online training-validity contract"
        )
    signal = np.zeros((ny, nx), dtype=np.float64)
    signal[fitted] = numerator[fitted] / denominator[fitted]
    np.maximum(signal, 0.0, out=signal)
    background_mean = np.zeros((ny, nx), dtype=np.float64)
    background_mean[fitted] = background_sum[fitted] / usable_count[fitted]
    noise = np.sqrt(
        signal
        + background_mean
        + request.read_noise_e_per_pixel**2
        + request.quantization_noise_e_per_pixel**2
    )
    if np.any(noise[permanent_valid] <= 0.0):
        raise StampScienceAnalysisContractError(
            "online training noise is not positive on valid pixels"
        )
    noise[~permanent_valid] = 1.0
    selected = build_science_optimal_aperture_v1(
        signal_template_e=signal,
        noise_template_e=noise,
        permanent_valid_mask=permanent_valid,
    )
    background_mask: NDArray[np.bool_] | None = None
    background_pixel_count = 0
    if request.policy.photometry.local_background_enabled:
        background_mask = build_local_background_mask_v1(
            selected.aperture_mask,
            exclusion_radius_pixels=request.policy.photometry.background_guard_pixels,
            border_pixels=request.policy.photometry.background_border_pixels,
            permanent_valid_mask=permanent_valid,
        )
        background_pixel_count = int(np.count_nonzero(background_mask))
        if background_pixel_count < request.policy.photometry.minimum_background_pixels:
            raise StampScienceAnalysisContractError(
                "online-trained background mask has too few pixels"
            )
    peak_flat = int(np.argmax(np.where(permanent_valid, signal, -np.inf)))
    peak_yx = tuple(int(value) for value in np.unravel_index(peak_flat, signal.shape))
    return ScienceApertureDefinition(
        aperture_mask=selected.aperture_mask,
        maximum_cumulative_snr=selected.maximum_cumulative_snr,
        algorithm=selected.algorithm,
        signal_template_shape=selected.signal_template_shape,
        background_mask=background_mask,
        signal_template_e=signal,
        noise_template_e=noise,
        target_peak_yx=peak_yx,
        training_raw_frame_indices=absolute,
        metadata={
            "template_fit": "through_origin_q_weighted_v1",
            "background_strategy": request.policy.photometry.background_strategy,
            "local_background_enabled": (
                request.policy.photometry.local_background_enabled
            ),
            "training_accumulator": (
                "online_per_pixel_sufficient_statistics_v1"
            ),
            "minimum_training_valid_fraction": (
                request.policy.photometry.minimum_training_valid_fraction
            ),
            "minimum_training_valid_count": minimum_valid_count,
            "training_frame_count": training_frame_count,
            "excluded_training_sample_count": excluded_training_sample_count,
            "background_pixel_count": background_pixel_count,
            "read_noise_e_per_pixel": request.read_noise_e_per_pixel,
            "quantization_noise_e_per_pixel": (
                request.quantization_noise_e_per_pixel
            ),
            "persistent_accumulator_bytes": int(
                usable_count.nbytes
                + denominator.nbytes
                + numerator.nbytes
                + background_sum.nbytes
            ),
        },
    )


@dataclass
class _CadenceCollector:
    factor: int
    photometry_parts: list[SciencePhotometryResult] = field(default_factory=list)
    raw_start_parts: list[NDArray[np.int64]] = field(default_factory=list)
    raw_stop_parts: list[NDArray[np.int64]] = field(default_factory=list)
    background_expectation_aperture_parts: list[NDArray[np.float64]] = field(
        default_factory=list
    )
    captured_flux_fraction_parts: list[NDArray[np.float64]] = field(
        default_factory=list
    )
    captured_flux_denominator_parts: list[NDArray[np.float64]] = field(
        default_factory=list
    )
    captured_flux_qa_parts: list[NDArray[np.bool_]] = field(
        default_factory=list
    )

    def add(
        self,
        batch: _DeliveryBatch,
        result: SciencePhotometryResult,
        *,
        aperture_mask: NDArray[np.bool_],
    ) -> None:
        if result.time_seconds.shape != batch.time_start_seconds.shape:
            raise StampScienceAnalysisContractError(
                "photometry result cadence axis differs from delivery batch"
            )
        self.photometry_parts.append(result)
        self.raw_start_parts.append(batch.raw_frame_start_index)
        self.raw_stop_parts.append(batch.raw_frame_stop_index_exclusive)
        self.captured_flux_fraction_parts.append(batch.captured_flux_fraction)
        self.captured_flux_denominator_parts.append(
            batch.captured_flux_denominator_e
        )
        self.captured_flux_qa_parts.append(batch.captured_flux_qa_pass)
        self.background_expectation_aperture_parts.append(
            np.sum(
                batch.background_expectation_e[:, aperture_mask],
                axis=1,
                dtype=np.float64,
            )
        )


@dataclass(frozen=True)
class _CadenceAnalysis:
    factor: int
    time_seconds: NDArray[np.float64]
    exposure_seconds: NDArray[np.float64]
    raw_start: NDArray[np.int64]
    raw_stop: NDArray[np.int64]
    photometry: Mapping[str, NDArray[Any]]
    expectation_model: ScienceVariabilityModelResult
    local_model: ScienceVariabilityModelResult
    uncertainty: ScienceFluxUncertaintyModelResult
    background_expectation_aperture_e: NDArray[np.float64]
    captured_flux_fraction: NDArray[np.float64]
    captured_flux_denominator_e: NDArray[np.float64]
    captured_flux_qa_pass: NDArray[np.bool_]
    expectation_cdpp: Any
    local_cdpp: Any


_PHOTOMETRY_ARRAY_FIELDS = (
    "flux_expectation_bgsub_e",
    "flux_local_bgsub_e",
    "local_background_e_per_pixel",
    "centroid_x",
    "centroid_y",
    "aperture_valid",
    "aperture_usable_pixel_count",
    "aperture_invalid_pixel_count",
    "saturated_pixel_count",
    "cosmic_pixel_count",
    "background_usable_pixel_count",
    "quality_bitmask",
)
_CADENCE_DATASET_NAMES = {
    "time_start_seconds",
    "exposure_seconds",
    "raw_frame_start_index",
    "raw_frame_stop_index_exclusive",
    *_PHOTOMETRY_ARRAY_FIELDS,
    "raw_factor_sum",
    "fitted_flux_expectation_e",
    "residual_expectation_e",
    "residual_expectation_ppm",
    "fitted_flux_local_e",
    "residual_local_e",
    "residual_local_ppm",
    "local_model_valid",
    "background_expectation_aperture_e",
    "captured_flux_fraction",
    "captured_flux_denominator_e",
    "captured_flux_qa_pass",
    "flux_uncertainty_e",
    "source_variance_e2",
    "background_variance_e2",
    "read_variance_e2",
    "quantization_variance_e2",
    "uncertainty_valid",
    "uncertainty_coadd_factor",
    "flux_expectation_bgsub_e_per_s",
    "flux_local_bgsub_e_per_s",
    "fitted_flux_expectation_e_per_s",
    "fitted_flux_local_e_per_s",
    "model_flux_uncertainty_e",
    "model_flux_uncertainty_e_per_s",
}


def _unavailable_local_variability_model(
    expectation_model: ScienceVariabilityModelResult,
) -> ScienceVariabilityModelResult:
    """Return an explicit unavailable diagnostic without fabricating flux."""

    shape = expectation_model.raw_factor_sum.shape
    unavailable = np.full(shape, np.nan, dtype=np.float64)
    return ScienceVariabilityModelResult(
        raw_factor_sum=np.array(expectation_model.raw_factor_sum, copy=True),
        fitted_flux_e=unavailable.copy(),
        residual_e=unavailable.copy(),
        residual_ppm=unavailable.copy(),
        fit_scale_e_per_raw_factor=float("nan"),
        fit_intercept_e=float("nan"),
        valid_mask=np.zeros(shape, dtype=bool),
    )


def _finish_cadence(
    collector: _CadenceCollector,
    *,
    raw_relative_flux: NDArray[np.float64],
    first_raw_index: int,
    policy: StampSciencePhotometryPolicy,
    aperture_mask: NDArray[np.bool_],
    read_noise_e_per_pixel: float,
    quantization_noise_e_per_pixel: float,
) -> _CadenceAnalysis:
    if not collector.photometry_parts:
        raise StampScienceAnalysisContractError(
            f"coadd factor {collector.factor} emitted no cadence"
        )
    time = np.concatenate(
        [np.asarray(item.time_seconds, dtype=np.float64) for item in collector.photometry_parts]
    )
    exposure_parts = [item.exposure_seconds for item in collector.photometry_parts]
    if any(item is None for item in exposure_parts):
        raise StampScienceAnalysisContractError(
            "science photometry result lacks exposure_seconds"
        )
    exposure = np.concatenate(
        [np.asarray(item, dtype=np.float64) for item in exposure_parts if item is not None]
    )
    raw_start = np.concatenate(collector.raw_start_parts)
    raw_stop = np.concatenate(collector.raw_stop_parts)
    background_expectation_aperture = np.concatenate(
        collector.background_expectation_aperture_parts
    )
    captured_flux_fraction = np.concatenate(
        collector.captured_flux_fraction_parts
    )
    captured_flux_denominator = np.concatenate(
        collector.captured_flux_denominator_parts
    )
    captured_flux_qa = np.concatenate(collector.captured_flux_qa_parts)
    if not np.all(captured_flux_qa):
        raise StampScienceAnalysisContractError(
            "reconstructed cadence captured_flux_qa_pass contains false"
        )
    arrays = {
        name: np.concatenate(
            [np.asarray(getattr(item, name)) for item in collector.photometry_parts]
        )
        for name in _PHOTOMETRY_ARRAY_FIELDS
    }
    local_valid = np.asarray(arrays["aperture_valid"], dtype=bool) & np.isfinite(
        arrays["flux_local_bgsub_e"]
    )
    local_start = raw_start - first_raw_index
    local_stop = raw_stop - first_raw_index
    expectation_model = fit_science_variability_model_v1(
        flux_e=arrays["flux_expectation_bgsub_e"],
        aperture_valid=arrays["aperture_valid"],
        raw_relative_flux=raw_relative_flux,
        raw_frame_start_index=local_start,
        raw_frame_stop_index_exclusive=local_stop,
    )
    local_background_enabled = all(
        bool(item.product_semantics.get("local_background_enabled", False))
        for item in collector.photometry_parts
    )
    if local_background_enabled:
        local_model = fit_science_variability_model_v1(
            flux_e=arrays["flux_local_bgsub_e"],
            aperture_valid=local_valid,
            raw_relative_flux=raw_relative_flux,
            raw_frame_start_index=local_start,
            raw_frame_stop_index_exclusive=local_stop,
        )
    else:
        local_model = _unavailable_local_variability_model(expectation_model)
    uncertainty = compute_science_flux_uncertainty_model_v1(
        fitted_source_expectation_e=expectation_model.fitted_flux_e,
        aperture_mask=aperture_mask,
        read_noise_e_per_raw_pixel=read_noise_e_per_pixel,
        quantization_noise_e_per_raw_pixel=quantization_noise_e_per_pixel,
        coadd_factor=collector.factor,
        cadence_valid=expectation_model.valid_mask,
        background_expectation_aperture_e=background_expectation_aperture,
    )
    expectation_cdpp = compute_science_cdpp_v1(
        time_seconds=time,
        exposure_seconds=exposure,
        flux_e=arrays["flux_expectation_bgsub_e"],
        aperture_valid=arrays["aperture_valid"],
        model_flux_e=expectation_model.fitted_flux_e,
        residual_e=expectation_model.residual_e,
        windows_minutes=policy.cdpp_windows_minutes,
        minimum_coverage_fraction=policy.minimum_coverage_fraction,
        minimum_accepted_bins=policy.minimum_accepted_bins,
        bin_origin_seconds=policy.bin_origin_seconds,
    )
    local_cdpp = None
    if local_background_enabled:
        local_cdpp = compute_science_cdpp_v1(
            time_seconds=time,
            exposure_seconds=exposure,
            flux_e=arrays["flux_local_bgsub_e"],
            aperture_valid=local_valid,
            model_flux_e=local_model.fitted_flux_e,
            residual_e=local_model.residual_e,
            windows_minutes=policy.cdpp_windows_minutes,
            minimum_coverage_fraction=policy.minimum_coverage_fraction,
            minimum_accepted_bins=policy.minimum_accepted_bins,
            bin_origin_seconds=policy.bin_origin_seconds,
        )
    return _CadenceAnalysis(
        factor=collector.factor,
        time_seconds=time,
        exposure_seconds=exposure,
        raw_start=raw_start,
        raw_stop=raw_stop,
        photometry=arrays,
        expectation_model=expectation_model,
        local_model=local_model,
        uncertainty=uncertainty,
        background_expectation_aperture_e=(
            background_expectation_aperture
        ),
        captured_flux_fraction=captured_flux_fraction,
        captured_flux_denominator_e=captured_flux_denominator,
        captured_flux_qa_pass=captured_flux_qa,
        expectation_cdpp=expectation_cdpp,
        local_cdpp=local_cdpp,
    )


_SEMANTIC_DATASET_NAMES = (
    "final_dn",
    "background_expectation_e",
    "captured_flux_fraction",
    "captured_flux_denominator_e",
    "captured_flux_qa_pass",
    "bias_level_sum_dn",
    "column_noise_sum_dn_by_x",
    "valid_mask",
    "fullwell_count",
    "adc_low_count",
    "adc_high_count",
    "cosmic_count",
    "saturated_mask",
    "cosmic_mask",
    "time_start_seconds",
    "exposure_seconds",
    "raw_frame_start_index",
    "raw_frame_stop_index_exclusive",
)
_RAW_SHARD_IDENTITY_FIELDS = {
    "path",
    "file_stat",
    "product_kind",
    "coadd_factor",
    "frame_count",
    "stamp_shape",
    "gain_mode",
    "manifest_identity_sha256",
    "provenance_identity_sha256",
    "series_manifest_identity_sha256",
    "series_provenance_identity_sha256",
    "first_raw_frame_start",
    "last_raw_frame_stop",
    "first_time_start_seconds",
    "last_time_end_seconds",
    "target_source_id",
    "case",
    "run_id",
    "production_manifest_reference",
    "production_manifest_content_identity",
    "identity_mode",
    "semantic_sha256",
    "datasets",
    "gain_e_per_dn",
    "byte_identity",
}
_INPUT_HEADER_IDENTITY_FIELDS = _RAW_SHARD_IDENTITY_FIELDS - {
    "identity_mode",
    "semantic_sha256",
    "datasets",
    "gain_e_per_dn",
    "byte_identity",
}


class _SemanticShardHasher:
    """Batch-size-independent hashes of every formal raw semantic plane."""

    def __init__(self, handle: Any, header: _InputHeader) -> None:
        self._hashers: dict[str, Any] = {}
        self._descriptors: dict[str, dict[str, Any]] = {}
        for name in _SEMANTIC_DATASET_NAMES:
            dataset = handle[name]
            digest = hashlib.sha256()
            descriptor = {
                "dtype": np.dtype(dataset.dtype).str,
                "shape": [int(value) for value in dataset.shape],
            }
            digest.update(
                json.dumps(
                    descriptor,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            self._hashers[name] = digest
            self._descriptors[name] = descriptor
        gain = np.ascontiguousarray(header.formal.static_gain_e_per_dn)
        gain_descriptor = {
            "dtype": gain.dtype.str,
            "shape": [int(value) for value in gain.shape],
        }
        gain_digest = hashlib.sha256()
        gain_digest.update(
            json.dumps(
                gain_descriptor,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        gain_digest.update(gain.tobytes(order="C"))
        self._gain_identity = {
            **gain_descriptor,
            "sha256": gain_digest.hexdigest(),
        }

    def update(self, batch: _DeliveryBatch) -> None:
        arrays: dict[str, NDArray[Any]] = {
            "final_dn": batch.final_dn,
            "background_expectation_e": batch.background_expectation_e,
            "captured_flux_fraction": batch.captured_flux_fraction,
            "captured_flux_denominator_e": batch.captured_flux_denominator_e,
            "captured_flux_qa_pass": batch.captured_flux_qa_pass,
            "bias_level_sum_dn": batch.bias_level_sum_dn,
            "column_noise_sum_dn_by_x": batch.column_noise_sum_dn_by_x,
            "valid_mask": batch.valid_mask,
            "fullwell_count": batch.fullwell_count,
            "adc_low_count": batch.adc_low_count,
            "adc_high_count": batch.adc_high_count,
            "cosmic_count": batch.cosmic_count,
            "saturated_mask": batch.saturated_mask,
            "cosmic_mask": batch.cosmic_mask,
            "time_start_seconds": batch.time_start_seconds,
            "exposure_seconds": batch.exposure_seconds,
            "raw_frame_start_index": batch.raw_frame_start_index,
            "raw_frame_stop_index_exclusive": batch.raw_frame_stop_index_exclusive,
        }
        for name, array in arrays.items():
            self._hashers[name].update(np.ascontiguousarray(array).tobytes(order="C"))

    def identity(self) -> dict[str, Any]:
        datasets = {
            name: {
                **self._descriptors[name],
                "sha256": digest.hexdigest(),
            }
            for name, digest in self._hashers.items()
        }
        aggregate = hashlib.sha256(
            json.dumps(
                {"datasets": datasets, "gain_e_per_dn": self._gain_identity},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return {
            "identity_mode": "canonical_semantic_planes_sha256_v1",
            "semantic_sha256": aggregate,
            "datasets": datasets,
            "gain_e_per_dn": self._gain_identity,
        }


def _headers_equivalent(left: _InputHeader, right: _InputHeader) -> bool:
    a = left.formal
    b = right.formal
    return bool(
        a.path == b.path
        and a.product_kind == b.product_kind
        and a.coadd_factor == b.coadd_factor
        and a.frame_count == b.frame_count
        and a.stamp_shape == b.stamp_shape
        and a.gain_mode == b.gain_mode
        and a.static_gain_e_per_dn is not None
        and b.static_gain_e_per_dn is not None
        and np.array_equal(a.static_gain_e_per_dn, b.static_gain_e_per_dn)
        and left.exact_manifest_identity == right.exact_manifest_identity
        and left.exact_provenance_identity == right.exact_provenance_identity
        and a.first_raw_frame_start == b.first_raw_frame_start
        and a.last_raw_frame_stop == b.last_raw_frame_stop
        and math.isclose(
            a.first_time_start_seconds,
            b.first_time_start_seconds,
            rel_tol=0.0,
            abs_tol=_TIME_ATOL_SECONDS,
        )
        and math.isclose(
            a.last_time_end_seconds,
            b.last_time_end_seconds,
            rel_tol=0.0,
            abs_tol=_TIME_ATOL_SECONDS,
        )
        and left.chunk_frames == right.chunk_frames
        and left.cross_product_manifest_identity
        == right.cross_product_manifest_identity
        and left.cross_product_provenance_identity
        == right.cross_product_provenance_identity
        and left.target_source_id == right.target_source_id
        and left.case == right.case
        and left.run_id == right.run_id
        and left.production_manifest_reference
        == right.production_manifest_reference
        and left.production_manifest_content_identity
        == right.production_manifest_content_identity
    )


@dataclass(frozen=True)
class _DirectSample:
    factor: int
    header: _InputHeader
    local_frame_index: int
    raw_start: int


@dataclass(frozen=True)
class _RepresentativeFrame:
    selection_role: str
    batch: _DeliveryBatch
    input_shard_index: int


def _direct_sample_local_indices(
    frame_count: int,
    *,
    samples_per_shard: int,
) -> tuple[int, ...]:
    count = min(samples_per_shard, frame_count)
    if count == 1:
        return (0,)
    return tuple(
        int(value)
        for value in np.rint(
            np.linspace(0, frame_count - 1, num=count)
        ).astype(np.int64)
    )


def _direct_samples(
    headers: Mapping[int, tuple[_InputHeader, ...]],
    *,
    samples_per_shard: int,
) -> tuple[_DirectSample, ...]:
    result: list[_DirectSample] = []
    for factor, series in sorted(headers.items()):
        for header in series:
            for index in _direct_sample_local_indices(
                header.formal.frame_count,
                samples_per_shard=samples_per_shard,
            ):
                result.append(
                    _DirectSample(
                        factor=factor,
                        header=header,
                        local_frame_index=index,
                        raw_start=(
                            header.formal.first_raw_frame_start + index * factor
                        ),
                    )
                )
    return tuple(result)


def _capture_parity_frames(
    batch: _DeliveryBatch,
    *,
    requested_raw_starts: set[int],
    destination: dict[int, _DeliveryBatch],
) -> None:
    for index, raw_start in enumerate(batch.raw_frame_start_index):
        resolved = int(raw_start)
        if resolved in requested_raw_starts:
            if resolved in destination:
                raise StampScienceAnalysisContractError(
                    "raw-derived parity frame was emitted more than once"
                )
            destination[resolved] = _slice_batch(
                batch,
                np.asarray([index], dtype=np.int64),
            )


def _compare_parity_batch(
    derived: _DeliveryBatch,
    direct: _DeliveryBatch,
) -> None:
    exact_names = (
        "final_dn",
        "valid_mask",
        "fullwell_count",
        "adc_low_count",
        "adc_high_count",
        "cosmic_count",
        "captured_flux_qa_pass",
        "raw_frame_start_index",
        "raw_frame_stop_index_exclusive",
    )
    float_names = (
        "background_expectation_e",
        "captured_flux_fraction",
        "captured_flux_denominator_e",
        "bias_level_sum_dn",
        "column_noise_sum_dn_by_x",
        "time_start_seconds",
        "exposure_seconds",
        "gain_e_per_dn",
    )
    if any(
        not np.array_equal(getattr(derived, name), getattr(direct, name))
        for name in exact_names
    ) or any(
        not np.allclose(
            getattr(derived, name),
            getattr(direct, name),
            rtol=1e-12,
            atol=1e-8,
        )
        for name in float_names
    ):
        raise StampScienceAnalysisContractError(
            "raw-derived/direct-coadd parity check failed"
        )


def _validate_direct_parity(
    samples: tuple[_DirectSample, ...],
    captures: Mapping[int, Mapping[int, _DeliveryBatch]],
) -> dict[str, Any]:
    import h5py

    rows: list[dict[str, Any]] = []
    for sample in samples:
        derived = captures.get(sample.factor, {}).get(sample.raw_start)
        if derived is None:
            raise StampScienceAnalysisContractError(
                "raw-derived/direct-coadd parity sample was not emitted"
            )
        if _FileStat.from_path(sample.header.formal.path) != sample.header.initial_stat:
            raise StampScienceAnalysisContractError(
                "direct coadd shard changed before parity validation"
            )
        with h5py.File(sample.header.formal.path, "r") as handle:
            direct = _read_delivery_batch(
                handle,
                sample.header,
                slice(sample.local_frame_index, sample.local_frame_index + 1),
            )
        _compare_parity_batch(derived, direct)
        if _FileStat.from_path(sample.header.formal.path) != sample.header.initial_stat:
            raise StampScienceAnalysisContractError(
                "direct coadd shard changed during parity validation"
            )
        rows.append(
            {
                "factor": sample.factor,
                "path": str(sample.header.formal.path),
                "local_frame_index": sample.local_frame_index,
                "raw_frame_start_index": sample.raw_start,
            }
        )
    return {
        "required": bool(samples),
        "passed": True,
        "comparison": _DIRECT_COADD_PARITY_COMPARISON,
        "sample_count": len(rows),
        "samples": rows,
    }


def _input_header_identity(header: _InputHeader) -> dict[str, Any]:
    return {
        "path": str(header.formal.path),
        "file_stat": header.initial_stat.to_dict(),
        "product_kind": header.formal.product_kind,
        "coadd_factor": header.formal.coadd_factor,
        "frame_count": header.formal.frame_count,
        "stamp_shape": list(header.formal.stamp_shape),
        "gain_mode": header.formal.gain_mode,
        "manifest_identity_sha256": _canonical_digest(
            header.exact_manifest_identity
        ),
        "provenance_identity_sha256": _canonical_digest(
            header.exact_provenance_identity
        ),
        "series_manifest_identity_sha256": (
            header.cross_product_manifest_identity
        ),
        "series_provenance_identity_sha256": (
            header.cross_product_provenance_identity
        ),
        "first_raw_frame_start": header.formal.first_raw_frame_start,
        "last_raw_frame_stop": header.formal.last_raw_frame_stop,
        "first_time_start_seconds": header.formal.first_time_start_seconds,
        "last_time_end_seconds": header.formal.last_time_end_seconds,
        "target_source_id": header.target_source_id,
        "case": header.case,
        "run_id": header.run_id,
        "production_manifest_reference": header.production_manifest_reference,
        "production_manifest_content_identity": (
            None
            if header.production_manifest_content_identity is None
            else dict(header.production_manifest_content_identity)
        ),
    }


def _resolve_input_byte_identity(header: _InputHeader) -> dict[str, Any]:
    """Bind exact HDF5 bytes via a trusted publisher receipt or full SHA256."""

    source = header.formal.path
    current_stat = _FileStat.from_path(source)
    if current_stat != header.initial_stat:
        raise StampScienceAnalysisContractError(
            "formal bundle stat changed before byte-identity preflight"
        )
    receipt_path = source.parent / "publication_receipt.json"
    if receipt_path.exists():
        if not receipt_path.is_file() or receipt_path.is_symlink():
            raise StampScienceAnalysisContractError(
                "publication receipt must be a real regular file"
            )
        receipt_stat = _FileStat.from_path(receipt_path)
        try:
            receipt_bytes = receipt_path.read_bytes()
            receipt = json.loads(receipt_bytes.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise StampScienceAnalysisContractError(
                "shard publication receipt is not valid UTF-8 JSON"
            ) from error
        if (
            not isinstance(receipt, dict)
            or _FileStat.from_path(receipt_path) != receipt_stat
        ):
            raise StampScienceAnalysisContractError(
                "shard publication receipt changed during identity preflight"
            )
        if (
            receipt.get("schema_id")
            != "et_mainsim.stamp_shard_publication_receipt.v1"
            or receipt.get("schema_version") != 1
            or receipt.get("complete") is not True
            or receipt.get("run_id") != header.run_id
            or receipt.get("case") != header.case
            or str(receipt.get("target_source_id_int64"))
            != header.target_source_id
        ):
            raise StampScienceAnalysisContractError(
                "publication receipt identity differs from the formal bundle"
            )
        members = receipt.get("members")
        member = members.get(source.name) if isinstance(members, Mapping) else None
        if not isinstance(member, Mapping):
            raise StampScienceAnalysisContractError(
                "publication receipt does not bind the formal bundle member"
            )
        sha256 = member.get("sha256")
        size = member.get("size_bytes")
        relative = member.get("path_relative_to_run_root")
        if (
            not isinstance(sha256, str)
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size != current_stat.size_bytes
            or not isinstance(relative, str)
            or Path(relative).name != source.name
            or not source.as_posix().endswith(f"/{relative}")
        ):
            raise StampScienceAnalysisContractError(
                "publication receipt member content/stat identity is invalid"
            )
        receipt_production = receipt.get("production_manifest")
        production_relative = (
            receipt_production.get("path_relative_to_run_root")
            if isinstance(receipt_production, Mapping)
            else None
        )
        production_size = (
            receipt_production.get("size_bytes")
            if isinstance(receipt_production, Mapping)
            else None
        )
        production_sha256 = (
            receipt_production.get("sha256")
            if isinstance(receipt_production, Mapping)
            else None
        )
        if (
            not isinstance(receipt_production, Mapping)
            or not isinstance(production_relative, str)
            or Path(production_relative).name
            != Path(header.production_manifest_reference).name
            or isinstance(production_size, bool)
            or not isinstance(production_size, int)
            or production_size < 0
            or not isinstance(production_sha256, str)
            or len(production_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in production_sha256
            )
        ):
            raise StampScienceAnalysisContractError(
                "publication receipt lacks a valid production-manifest identity"
            )
        recorded_production = header.production_manifest_content_identity
        if recorded_production is not None and (
            receipt_production.get("size_bytes")
            != recorded_production.get("size_bytes")
            or receipt_production.get("sha256")
            != recorded_production.get("sha256")
        ):
            raise StampScienceAnalysisContractError(
                "publication receipt and formal bundle bind different production manifests"
            )
        current_identity = _file_identity(source)
        if (
            current_identity["size_bytes"] != size
            or current_identity["sha256"] != sha256
        ):
            raise StampScienceAnalysisContractError(
                "publication receipt digest differs from the current bundle bytes"
            )
        if _FileStat.from_path(source) != current_stat:
            raise StampScienceAnalysisContractError(
                "formal bundle stat changed during receipt identity preflight"
            )
        return {
            "trust_scope": "publisher_receipt_plus_stat_and_formal_header_v1",
            "size_bytes": size,
            "sha256": sha256,
            "publication_receipt": {
                "path": str(receipt_path.resolve()),
                "identity": {
                    "size_bytes": len(receipt_bytes),
                    "sha256": hashlib.sha256(receipt_bytes).hexdigest(),
                },
            },
            "runtime_path": str(source),
            "runtime_hostname": os.uname().nodename,
        }
    full = _file_identity(source)
    if (
        full["size_bytes"] != current_stat.size_bytes
        or _FileStat.from_path(source) != current_stat
    ):
        raise StampScienceAnalysisContractError(
            "formal bundle stat changed during full-byte SHA256 preflight"
        )
    return {
        "trust_scope": "locally_computed_full_file_sha256_v1",
        **full,
        "runtime_path": str(source),
        "runtime_hostname": os.uname().nodename,
    }


def _build_reference_aperture(
    stamp_shape: tuple[int, int],
    *,
    policy: StampSciencePhotometryPolicy,
) -> ScienceApertureDefinition:
    base = build_reference_fixed13_aperture_v1(stamp_shape)
    background_mask: NDArray[np.bool_] | None = None
    background_pixel_count = 0
    if policy.local_background_enabled:
        permanent_valid = np.ones(stamp_shape, dtype=bool)
        background_mask = build_local_background_mask_v1(
            base.aperture_mask,
            exclusion_radius_pixels=policy.background_guard_pixels,
            border_pixels=policy.background_border_pixels,
            permanent_valid_mask=permanent_valid,
        )
        background_pixel_count = int(np.count_nonzero(background_mask))
        if background_pixel_count < policy.minimum_background_pixels:
            raise StampScienceAnalysisContractError(
                "reference fixed13 background mask has too few pixels"
            )
    return ScienceApertureDefinition(
        aperture_mask=base.aperture_mask,
        background_mask=background_mask,
        maximum_cumulative_snr=base.maximum_cumulative_snr,
        algorithm=base.algorithm,
        signal_template_shape=base.signal_template_shape,
        target_peak_yx=base.target_peak_yx,
        metadata={
            **dict(base.metadata),
            "background_strategy": policy.background_strategy,
            "local_background_enabled": policy.local_background_enabled,
            "background_mask_algorithm": (
                "et_mainsim.local_background_mask_rectangular_v1"
            ),
            "background_pixel_count": background_pixel_count,
            "background_guard_pixels": policy.background_guard_pixels,
            "background_border_pixels": policy.background_border_pixels,
        },
    )


def _bind_aperture_background_policy_v1(
    aperture: ScienceApertureDefinition,
    *,
    policy: StampSciencePhotometryPolicy,
) -> ScienceApertureDefinition:
    validated = _validate_frozen_aperture_definition(aperture)
    metadata = dict(validated.metadata)
    recorded_strategy = metadata.get("background_strategy")
    if (
        recorded_strategy is not None
        and recorded_strategy != policy.background_strategy
    ):
        raise StampScienceAnalysisContractError(
            "frozen aperture background metadata differs from analysis policy"
        )
    metadata["background_strategy"] = policy.background_strategy
    return ScienceApertureDefinition(
        aperture_mask=validated.aperture_mask,
        background_mask=validated.background_mask,
        signal_template_e=validated.signal_template_e,
        noise_template_e=validated.noise_template_e,
        maximum_cumulative_snr=validated.maximum_cumulative_snr,
        algorithm=validated.algorithm,
        signal_template_shape=validated.signal_template_shape,
        target_peak_yx=validated.target_peak_yx,
        training_raw_frame_indices=validated.training_raw_frame_indices,
        metadata=metadata,
    )


def _stream_raw_product_analyses(
    headers: tuple[_InputHeader, ...],
    *,
    apertures: Mapping[str, ScienceApertureDefinition],
    representative_aperture_name: str | None,
    request: StampScienceAnalysisRequest,
    direct_samples: tuple[_DirectSample, ...],
) -> tuple[
    dict[str, dict[int, _CadenceCollector]],
    list[dict[str, Any]],
    dict[int, dict[int, _DeliveryBatch]],
    dict[str, tuple[_RepresentativeFrame, ...]],
]:
    import h5py

    if not apertures or (
        representative_aperture_name is not None
        and representative_aperture_name not in apertures
    ):
        raise StampScienceAnalysisContractError(
            "raw product analysis requires a representative aperture"
        )
    accumulators = {
        factor: _FactorAccumulator(factor) for factor in request.policy.coadd_factors
    }
    collectors = {
        name: {
            factor: _CadenceCollector(factor)
            for factor in request.policy.coadd_factors
        }
        for name in apertures
    }
    requested_starts: dict[int, set[int]] = {
        factor: {
            sample.raw_start for sample in direct_samples if sample.factor == factor
        }
        for factor in request.policy.coadd_factors
        if factor != 1
    }
    captures: dict[int, dict[int, _DeliveryBatch]] = {
        factor: {} for factor in requested_starts
    }
    identities: list[dict[str, Any]] = []
    representative_state: dict[
        str, dict[str, _RepresentativeFrame | None]
    ] = {
        name: {"first_clean": None, "middle_clean": None, "last_clean": None}
        for name in apertures
    }
    middle_raw_index = (
        headers[0].formal.first_raw_frame_start
        + (
            headers[-1].formal.last_raw_frame_stop
            - headers[0].formal.first_raw_frame_start
        )
        // 2
    )
    expected_raw_start: int | None = None
    expected_time_start: float | None = None
    for shard_index, header in enumerate(headers):
        if _FileStat.from_path(header.formal.path) != header.initial_stat:
            raise StampScienceAnalysisContractError(
                "formal raw shard changed between training and sequential reduction"
            )
        with h5py.File(header.formal.path, "r") as handle:
            hasher = _SemanticShardHasher(handle, header)
            for offset in range(0, header.formal.frame_count, request.policy.stream_batch_frames):
                stop = min(
                    offset + request.policy.stream_batch_frames,
                    header.formal.frame_count,
                )
                batch = _read_delivery_batch(handle, header, slice(offset, stop))
                _assert_raw_batch_cadence(
                    batch,
                    raw_exposure_seconds=request.policy.raw_exposure_seconds,
                )
                if expected_raw_start is not None and (
                    int(batch.raw_frame_start_index[0]) != expected_raw_start
                    or not math.isclose(
                        float(batch.time_start_seconds[0]),
                        float(expected_time_start),
                        rel_tol=0.0,
                        abs_tol=_TIME_ATOL_SECONDS,
                    )
                ):
                    raise StampScienceAnalysisContractError(
                        "formal raw frames are not globally continuous"
                    )
                expected_raw_start = int(batch.raw_frame_stop_index_exclusive[-1])
                expected_time_start = float(
                    batch.time_start_seconds[-1] + batch.exposure_seconds[-1]
                )
                hasher.update(batch)
                for aperture_name, aperture in apertures.items():
                    state = representative_state[aperture_name]
                    aperture_usable = (
                        batch.valid_mask[:, aperture.aperture_mask]
                        & ~batch.saturated_mask[:, aperture.aperture_mask]
                        & ~batch.cosmic_mask[:, aperture.aperture_mask]
                    )
                    clean_indices = np.flatnonzero(np.all(aperture_usable, axis=1))
                    if not clean_indices.size:
                        continue
                    if state["first_clean"] is None:
                        state["first_clean"] = _RepresentativeFrame(
                            selection_role="first_clean",
                            batch=_slice_batch(
                                batch,
                                np.asarray([int(clean_indices[0])], dtype=np.int64),
                            ),
                            input_shard_index=shard_index,
                        )
                    if state["middle_clean"] is None:
                        middle_candidates = clean_indices[
                            batch.raw_frame_start_index[clean_indices]
                            >= middle_raw_index
                        ]
                        if middle_candidates.size:
                            state["middle_clean"] = _RepresentativeFrame(
                                selection_role="middle_clean",
                                batch=_slice_batch(
                                    batch,
                                    np.asarray(
                                        [int(middle_candidates[0])], dtype=np.int64
                                    ),
                                ),
                                input_shard_index=shard_index,
                            )
                    state["last_clean"] = _RepresentativeFrame(
                        selection_role="last_clean",
                        batch=_slice_batch(
                            batch,
                            np.asarray([int(clean_indices[-1])], dtype=np.int64),
                        ),
                        input_shard_index=shard_index,
                    )
                for factor, accumulator in accumulators.items():
                    emitted = accumulator.push(batch)
                    if emitted is None:
                        continue
                    photometry_input = emitted.to_photometry_input()
                    for name, aperture in apertures.items():
                        result = reduce_science_photometry_v1(
                            photometry_input,
                            aperture_mask=aperture.aperture_mask,
                            background_mask=aperture.background_mask,
                            minimum_background_pixels=(
                                request.policy.photometry.minimum_background_pixels
                            ),
                        )
                        collectors[name][factor].add(
                            emitted,
                            result,
                            aperture_mask=np.asarray(
                                aperture.aperture_mask,
                                dtype=bool,
                            ),
                        )
                    if factor in requested_starts:
                        _capture_parity_frames(
                            emitted,
                            requested_raw_starts=requested_starts[factor],
                            destination=captures[factor],
                        )
            identity = _input_header_identity(header)
            identity.update(hasher.identity())
            identities.append(identity)
        current = _read_input_header(header.formal.path)
        if (
            _FileStat.from_path(header.formal.path) != header.initial_stat
            or not _headers_equivalent(current, header)
        ):
            raise StampScienceAnalysisContractError(
                "formal raw shard changed during sequential reduction"
            )
    for accumulator in accumulators.values():
        accumulator.finish()
    representative_frames: dict[str, tuple[_RepresentativeFrame, ...]] = {}
    for name, state in representative_state.items():
        values = (
            state["first_clean"],
            state["middle_clean"],
            state["last_clean"],
        )
        if any(item is None for item in values):
            raise StampScienceAnalysisContractError(
                f"could not select first/middle/last clean representative raw frames for {name}"
            )
        representative_frames[name] = values  # type: ignore[assignment]
    return (
        collectors,
        identities,
        captures,
        representative_frames,
    )


def _stream_raw_analysis(
    headers: tuple[_InputHeader, ...],
    *,
    aperture: ScienceApertureDefinition,
    request: StampScienceAnalysisRequest,
    direct_samples: tuple[_DirectSample, ...],
) -> tuple[
    dict[int, _CadenceCollector],
    list[dict[str, Any]],
    dict[int, dict[int, _DeliveryBatch]],
    tuple[_RepresentativeFrame, ...],
]:
    """Backward-compatible single-aperture adapter around the shared pass."""

    collectors, identities, captures, frames_by_product = _stream_raw_product_analyses(
        headers,
        apertures={"science_optimal_aperture_v1": aperture},
        representative_aperture_name="science_optimal_aperture_v1",
        request=request,
        direct_samples=direct_samples,
    )
    return (
        collectors["science_optimal_aperture_v1"],
        identities,
        captures,
        frames_by_product["science_optimal_aperture_v1"],
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        resolved = float(value)
        return resolved if math.isfinite(resolved) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _cdpp_payload(result: Any) -> dict[str, Any]:
    if result is None:
        return {
            "status": "not_computed",
            "reason": "background_strategy_delivered_expectation_only",
        }
    return {
        "metrics_by_window_minutes": {
            str(minutes): metric.to_dict()
            for minutes, metric in sorted(result.metrics_by_window_minutes.items())
        },
        "binned_rows": [row.to_dict() for row in result.binned_rows],
    }


def _all_cdpp_payload(analyses: Mapping[int, _CadenceAnalysis]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_id": _CDPP_SCHEMA_ID,
        "estimator": _CDPP_ESTIMATOR,
        "cadences": {},
    }
    cadences = payload["cadences"]
    assert isinstance(cadences, dict)
    for factor, analysis in sorted(analyses.items()):
        cadence_seconds = int(round(float(analysis.exposure_seconds[0])))
        cadences[f"{cadence_seconds}s"] = {
            "coadd_factor": factor,
            "expectation_background": _cdpp_payload(analysis.expectation_cdpp),
            "local_background": _cdpp_payload(analysis.local_cdpp),
        }
    return _json_safe(payload)


def _aperture_payload(aperture: ScienceApertureDefinition) -> dict[str, Any]:
    files = {
        "aperture_mask": "aperture_mask.npy",
        "background_mask": "background_mask.npy",
    }
    if aperture.signal_template_e is not None:
        files["signal_template_e"] = "signal_template_e.npy"
    if aperture.noise_template_e is not None:
        files["noise_template_e"] = "noise_template_e.npy"
    return _json_safe(
        {
            "schema_id": _APERTURE_DEFINITION_SCHEMA_ID,
            "schema_version": _APERTURE_DEFINITION_SCHEMA_VERSION,
            "algorithm": aperture.algorithm,
            "maximum_cumulative_snr": aperture.maximum_cumulative_snr,
            "signal_template_shape": list(aperture.signal_template_shape),
            "target_peak_yx": aperture.target_peak_yx,
            "training_raw_frame_indices": aperture.training_raw_frame_indices,
            "aperture_pixel_count": int(np.count_nonzero(aperture.aperture_mask)),
            "background_pixel_count": (
                0
                if aperture.background_mask is None
                else int(np.count_nonzero(aperture.background_mask))
            ),
            "files": files,
            "metadata": aperture.metadata,
        }
    )


def _q_identity(q: NDArray[np.float64]) -> dict[str, Any]:
    normalized = np.ascontiguousarray(q.astype("<f8", copy=False))
    return {
        "dtype": "<f8",
        "count": int(q.size),
        "minimum": float(np.min(q)),
        "maximum": float(np.max(q)),
        "sha256": hashlib.sha256(normalized.tobytes(order="C")).hexdigest(),
    }


def _direct_input_identities(
    headers: Mapping[int, tuple[_InputHeader, ...]],
    byte_identities: Mapping[str, Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    return {
        str(factor): [
            {
                **_input_header_identity(header),
                "identity_mode": "validated_header_plus_bounded_parity_samples_v1",
                "byte_identity": dict(byte_identities[str(header.formal.path)]),
            }
            for header in series
        ]
        for factor, series in sorted(headers.items())
    }


def _flux_uncertainty_model_contract_v1() -> dict[str, str]:
    return {
        "schema_id": "et_mainsim.science_flux_uncertainty_model.v1",
        "semantic_role": "analytic_model_standard_deviation_not_empirical_scatter",
        "authoritative_dataset": "model_flux_uncertainty_e",
        "compatibility_alias": "flux_uncertainty_e",
        "rate_dataset": "model_flux_uncertainty_e_per_s",
        "source_variance": "fitted_source_expectation_e",
        "background_variance": "streamed_background_expectation_aperture_e",
        "read_and_quantization_noise_scale": (
            "aperture_pixel_count_times_raw_coadd_factor"
        ),
        "quality_invalid_uncertainty": "NaN_components_retained",
    }


def _captured_flux_qa_contract_v1(
    cadence_records: Mapping[str, tuple[bool, float]],
) -> dict[str, Any]:
    return {
        "definition": _CAPTURED_FLUX_QA_DEFINITION,
        "fraction_denominator": _CAPTURED_FLUX_QA_DENOMINATOR,
        "post_crop_renormalization": False,
        "cadences": {
            name: {
                "all_pass": bool(all_pass),
                "minimum_fraction": float(minimum_fraction),
            }
            for name, (all_pass, minimum_fraction) in sorted(
                cadence_records.items()
            )
        },
    }


def _build_contract(
    *,
    request: StampScienceAnalysisRequest,
    raw_headers: tuple[_InputHeader, ...],
    raw_identities: list[dict[str, Any]],
    direct_headers: Mapping[int, tuple[_InputHeader, ...]],
    parity: Mapping[str, Any],
    aperture: ScienceApertureDefinition,
    analyses: Mapping[int, _CadenceAnalysis],
    representative_frames: tuple[_RepresentativeFrame, ...],
    raw_identities_for_frames: Sequence[Mapping[str, Any]],
    direct_byte_identities: Mapping[str, Mapping[str, Any]],
    execution_code_identity: Mapping[str, Any],
    analysis_product: str = "science_optimal_aperture_v1",
    aperture_mode: str | None = None,
    aperture_source_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    first_raw = raw_headers[0].formal.first_raw_frame_start
    last_raw = raw_headers[-1].formal.last_raw_frame_stop
    q_identity = _q_identity(np.asarray(request.raw_relative_flux, dtype=np.float64))
    q_identity["source_identity"] = dict(request.raw_relative_flux_identity)
    return _json_safe(
        {
            "schema_id": STAMP_SCIENCE_ANALYSIS_SCHEMA_ID,
            "schema_version": STAMP_SCIENCE_ANALYSIS_SCHEMA_VERSION,
            "complete": True,
            "analysis_product": analysis_product,
            "observation_product": "final_dn",
            "calibrated_electron_products_are_derived": True,
            "background_realization_used": False,
            "background_products": (
                [
                    "expectation_background_subtracted",
                    "local_background_diagnostic",
                ]
                if request.policy.photometry.local_background_enabled
                else ["expectation_background_subtracted"]
            ),
            "default_background_product": "background_expectation_e",
            "background_strategy": (
                request.policy.photometry.background_strategy
            ),
            "captured_flux_qa": _captured_flux_qa_contract_v1(
                {
                    f"{int(round(float(analysis.exposure_seconds[0])))}s": (
                        bool(np.all(analysis.captured_flux_qa_pass)),
                        float(np.min(analysis.captured_flux_fraction)),
                    )
                    for _, analysis in sorted(analyses.items())
                }
            ),
            "source_model": "through_origin_integrated_raw_relative_flux_v1",
            "flux_uncertainty_model": _flux_uncertainty_model_contract_v1(),
            "rate_products": {
                "definition": "integrated_electrons_divided_by_exposure_seconds",
                "unit": "electron / s",
                "datasets": [
                    "flux_expectation_bgsub_e_per_s",
                    "flux_local_bgsub_e_per_s",
                    "fitted_flux_expectation_e_per_s",
                    "fitted_flux_local_e_per_s",
                    "model_flux_uncertainty_e_per_s",
                ],
            },
            "reference_lightcurve": _reference_lightcurve_contract_v1(),
            "science_photometry_schema_id": SCIENCE_PHOTOMETRY_SCHEMA_ID,
            "science_photometry_schema_version": (
                SCIENCE_PHOTOMETRY_SCHEMA_VERSION
            ),
            "raw_frame_interval": {
                "start_index": first_raw,
                "stop_index_exclusive": last_raw,
                "count": last_raw - first_raw,
            },
            "cadence_seconds": [
                int(round(float(analysis.exposure_seconds[0])))
                for _, analysis in sorted(analyses.items())
            ],
            "input_raw_shards": raw_identities,
            "input_direct_coadd_shards": _direct_input_identities(
                direct_headers,
                direct_byte_identities,
            ),
            "raw_relative_flux": q_identity,
            "direct_coadd_parity": dict(parity),
            "policy": request.policy.to_dict(),
            "aperture": _aperture_payload(aperture),
            "code_identity": dict(execution_code_identity),
            "request_code_identity": dict(request.code_identity),
            "execution_code_identity": dict(execution_code_identity),
            "analysis_context": dict(request.analysis_context),
            "aperture_mode": (
                request.aperture_mode if aperture_mode is None else aperture_mode
            ),
            "aperture_source_identity": dict(
                request.aperture_source_identity
                if aperture_source_identity is None
                else aperture_source_identity
            ),
            "streaming": {
                "aperture_training_pass": (
                    "deterministic_chunk_aligned_bounded_samples"
                ),
                "reduction_pass": "one_sequential_raw_pass",
                "image_cube_materialized": False,
                "summed_cadences": "raw_plane_accumulation_before_calibration",
                "fixed_aperture_reused_for_all_cadences": True,
            },
            "input_hdf5_byte_identity_policy": {
                "preferred": "trusted_staged_publication_receipt_v1",
                "receipt_absent_fallback": "full_file_sha256_preflight_v1",
                "raw_semantic_hash_in_sequential_pass": True,
            },
            "representative_calibrated_frames": {
                "artifact": "representative_calibrated_frames.h5",
                "selection_policy": _REPRESENTATIVE_FRAME_SELECTION_POLICY,
                "clean_definition": _REPRESENTATIVE_FRAME_CLEAN_DEFINITION,
                "frames": [
                    {
                        "selection_role": item.selection_role,
                        "raw_frame_start_index": int(
                            item.batch.raw_frame_start_index[0]
                        ),
                        "input_shard_index": item.input_shard_index,
                        "input_shard_path": raw_identities_for_frames[
                            item.input_shard_index
                        ]["path"],
                        "input_shard_semantic_sha256": raw_identities_for_frames[
                            item.input_shard_index
                        ]["semantic_sha256"],
                    }
                    for item in representative_frames
                ],
            },
        }
    )


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(
            _json_safe(value),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _create_vector_dataset(group: Any, name: str, value: ArrayLike) -> None:
    array = np.asarray(value)
    chunks = (min(max(1, int(array.shape[0])), 65_536),)
    group.create_dataset(name, data=array, chunks=chunks)


def _write_authoritative_hdf5(
    path: Path,
    *,
    contract: Mapping[str, Any],
    q: NDArray[np.float64],
    aperture: ScienceApertureDefinition,
    analyses: Mapping[int, _CadenceAnalysis],
    cdpp_payload: Mapping[str, Any],
) -> None:
    import h5py

    persisted_background_mask = (
        np.zeros(aperture.aperture_mask.shape, dtype=bool)
        if aperture.background_mask is None
        else np.asarray(aperture.background_mask, dtype=bool)
    )
    with h5py.File(path, "w") as handle:
        handle.attrs["schema_id"] = STAMP_SCIENCE_ANALYSIS_SCHEMA_ID
        handle.attrs["schema_version"] = STAMP_SCIENCE_ANALYSIS_SCHEMA_VERSION
        handle.attrs["complete"] = False
        handle.attrs["observation_product"] = "final_dn"
        handle.attrs["background_realization_used"] = False
        handle.create_dataset(
            "analysis_contract_json",
            data=np.bytes_(
                json.dumps(
                    contract,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            ),
        )
        handle.create_dataset(
            "cdpp_json",
            data=np.bytes_(
                json.dumps(
                    cdpp_payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            ),
        )
        _create_vector_dataset(handle, "raw_relative_flux", q)
        aperture_group = handle.create_group("aperture")
        aperture_group.create_dataset(
            "aperture_mask", data=np.asarray(aperture.aperture_mask, dtype=bool)
        )
        aperture_group.create_dataset(
            "background_mask",
            data=persisted_background_mask,
        )
        if aperture.signal_template_e is not None:
            aperture_group.create_dataset(
                "signal_template_e", data=aperture.signal_template_e
            )
        if aperture.noise_template_e is not None:
            aperture_group.create_dataset(
                "noise_template_e", data=aperture.noise_template_e
            )
        if aperture.training_raw_frame_indices is not None:
            _create_vector_dataset(
                aperture_group,
                "training_raw_frame_indices",
                aperture.training_raw_frame_indices,
            )
        cadences = handle.create_group("cadences")
        for _, analysis in sorted(analyses.items()):
            cadence_seconds = int(round(float(analysis.exposure_seconds[0])))
            group = cadences.create_group(f"{cadence_seconds}s")
            group.attrs["coadd_factor"] = analysis.factor
            exposure = np.asarray(analysis.exposure_seconds, dtype=np.float64)
            values: dict[str, ArrayLike] = {
                "time_start_seconds": analysis.time_seconds,
                "exposure_seconds": analysis.exposure_seconds,
                "raw_frame_start_index": analysis.raw_start,
                "raw_frame_stop_index_exclusive": analysis.raw_stop,
                **analysis.photometry,
                "raw_factor_sum": analysis.expectation_model.raw_factor_sum,
                "fitted_flux_expectation_e": (
                    analysis.expectation_model.fitted_flux_e
                ),
                "residual_expectation_e": analysis.expectation_model.residual_e,
                "residual_expectation_ppm": (
                    analysis.expectation_model.residual_ppm
                ),
                "fitted_flux_local_e": analysis.local_model.fitted_flux_e,
                "residual_local_e": analysis.local_model.residual_e,
                "residual_local_ppm": analysis.local_model.residual_ppm,
                "local_model_valid": analysis.local_model.valid_mask,
                "background_expectation_aperture_e": (
                    analysis.background_expectation_aperture_e
                ),
                "captured_flux_fraction": analysis.captured_flux_fraction,
                "captured_flux_denominator_e": (
                    analysis.captured_flux_denominator_e
                ),
                "captured_flux_qa_pass": analysis.captured_flux_qa_pass,
                "flux_uncertainty_e": analysis.uncertainty.uncertainty_e,
                "source_variance_e2": analysis.uncertainty.source_variance_e2,
                "background_variance_e2": (
                    analysis.uncertainty.background_variance_e2
                ),
                "read_variance_e2": analysis.uncertainty.read_variance_e2,
                "quantization_variance_e2": (
                    analysis.uncertainty.quantization_variance_e2
                ),
                "uncertainty_valid": analysis.uncertainty.valid_mask,
                "uncertainty_coadd_factor": analysis.uncertainty.coadd_factor,
                "flux_expectation_bgsub_e_per_s": (
                    analysis.photometry["flux_expectation_bgsub_e"] / exposure
                ),
                "flux_local_bgsub_e_per_s": (
                    analysis.photometry["flux_local_bgsub_e"] / exposure
                ),
                "fitted_flux_expectation_e_per_s": (
                    analysis.expectation_model.fitted_flux_e / exposure
                ),
                "fitted_flux_local_e_per_s": (
                    analysis.local_model.fitted_flux_e / exposure
                ),
                "model_flux_uncertainty_e": (
                    analysis.uncertainty.uncertainty_e
                ),
                "model_flux_uncertainty_e_per_s": (
                    analysis.uncertainty.uncertainty_e / exposure
                ),
            }
            if set(values) != _CADENCE_DATASET_NAMES:  # pragma: no cover
                raise StampScienceAnalysisContractError(
                    "writer cadence dataset set differs from the frozen contract"
                )
            for name, value in values.items():
                _create_vector_dataset(group, name, value)
            group.attrs["fit_scale_expectation_e_per_raw_factor"] = (
                analysis.expectation_model.fit_scale_e_per_raw_factor
            )
            group.attrs["fit_scale_local_e_per_raw_factor"] = (
                analysis.local_model.fit_scale_e_per_raw_factor
            )
            group.attrs["fit_intercept_e"] = 0.0
            group.attrs["uncertainty_model_json"] = json.dumps(
                _json_safe(analysis.uncertainty.metadata),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        handle.flush()
        handle.attrs["complete"] = True
        handle.flush()
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _write_portable_ecsv_v1(
    path: Path,
    *,
    analyses: Mapping[int, _CadenceAnalysis],
) -> None:
    from astropy.table import Table, vstack

    tables: list[Table] = []
    for _, analysis in sorted(analyses.items()):
        cadence_seconds = int(round(float(analysis.exposure_seconds[0])))
        local_valid = np.asarray(analysis.local_model.valid_mask, dtype=bool)
        table = Table(
            {
                "cadence_seconds": np.full(
                    analysis.time_seconds.shape, cadence_seconds, dtype=np.int32
                ),
                "time_start_seconds": analysis.time_seconds,
                "exposure_seconds": analysis.exposure_seconds,
                "raw_frame_start_index": analysis.raw_start,
                "raw_frame_stop_index_exclusive": analysis.raw_stop,
                "flux_expectation_bgsub_e": analysis.photometry[
                    "flux_expectation_bgsub_e"
                ],
                "flux_local_bgsub_e": analysis.photometry["flux_local_bgsub_e"],
                "local_background_e_per_pixel": analysis.photometry[
                    "local_background_e_per_pixel"
                ],
                "centroid_x": analysis.photometry["centroid_x"],
                "centroid_y": analysis.photometry["centroid_y"],
                "aperture_valid": analysis.photometry["aperture_valid"],
                "local_background_valid": local_valid,
                "quality_bitmask": analysis.photometry["quality_bitmask"],
                "raw_factor_sum": analysis.expectation_model.raw_factor_sum,
                "fitted_flux_expectation_e": (
                    analysis.expectation_model.fitted_flux_e
                ),
                "residual_expectation_e": analysis.expectation_model.residual_e,
                "residual_expectation_ppm": analysis.expectation_model.residual_ppm,
                "fitted_flux_local_e": analysis.local_model.fitted_flux_e,
                "residual_local_e": analysis.local_model.residual_e,
                "residual_local_ppm": analysis.local_model.residual_ppm,
                "background_expectation_aperture_e": (
                    analysis.background_expectation_aperture_e
                ),
                "captured_flux_fraction": analysis.captured_flux_fraction,
                "captured_flux_denominator_e": (
                    analysis.captured_flux_denominator_e
                ),
                "captured_flux_qa_pass": analysis.captured_flux_qa_pass,
                "flux_uncertainty_e": analysis.uncertainty.uncertainty_e,
                "source_variance_e2": analysis.uncertainty.source_variance_e2,
                "background_variance_e2": (
                    analysis.uncertainty.background_variance_e2
                ),
                "read_variance_e2": analysis.uncertainty.read_variance_e2,
                "quantization_variance_e2": (
                    analysis.uncertainty.quantization_variance_e2
                ),
                "uncertainty_valid": analysis.uncertainty.valid_mask,
                "flux_expectation_bgsub_e_per_s": (
                    analysis.photometry["flux_expectation_bgsub_e"]
                    / analysis.exposure_seconds
                ),
                "flux_local_bgsub_e_per_s": (
                    analysis.photometry["flux_local_bgsub_e"]
                    / analysis.exposure_seconds
                ),
                "fitted_flux_expectation_e_per_s": (
                    analysis.expectation_model.fitted_flux_e
                    / analysis.exposure_seconds
                ),
                "fitted_flux_local_e_per_s": (
                    analysis.local_model.fitted_flux_e
                    / analysis.exposure_seconds
                ),
                "model_flux_uncertainty_e": (
                    analysis.uncertainty.uncertainty_e
                ),
                "model_flux_uncertainty_e_per_s": (
                    analysis.uncertainty.uncertainty_e
                    / analysis.exposure_seconds
                ),
            }
        )
        tables.append(table)
    combined = vstack(tables, metadata_conflicts="error")
    combined.meta = _photometry_table_metadata_v2()
    combined.write(path, format="ascii.ecsv", overwrite=False)
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _write_reference_lightcurve_ecsv_v1(
    path: Path,
    *,
    analyses: Mapping[int, _CadenceAnalysis],
) -> None:
    from astropy.table import Table, vstack

    tables = []
    for factor, analysis in sorted(analyses.items()):
        cadence_seconds = int(round(float(analysis.exposure_seconds[0])))
        tables.append(
            Table(
                {
                    "cadence_seconds": np.full(
                        analysis.time_seconds.shape,
                        cadence_seconds,
                        dtype=np.int32,
                    ),
                    "time_start_seconds": analysis.time_seconds,
                    "exposure_seconds": analysis.exposure_seconds,
                    "raw_frame_start_index": analysis.raw_start,
                    "raw_frame_stop_index_exclusive": analysis.raw_stop,
                    "raw_relative_flux_mean": (
                        analysis.expectation_model.raw_factor_sum / factor
                    ),
                    "raw_relative_flux_sum": (
                        analysis.expectation_model.raw_factor_sum
                    ),
                    "flux_expectation_bgsub_e": analysis.photometry[
                        "flux_expectation_bgsub_e"
                    ],
                    "flux_expectation_bgsub_e_per_s": (
                        analysis.photometry["flux_expectation_bgsub_e"]
                        / analysis.exposure_seconds
                    ),
                    "aperture_valid": analysis.photometry["aperture_valid"],
                    "quality_bitmask": analysis.photometry["quality_bitmask"],
                    "captured_flux_fraction": analysis.captured_flux_fraction,
                    "captured_flux_denominator_e": (
                        analysis.captured_flux_denominator_e
                    ),
                    "captured_flux_qa_pass": analysis.captured_flux_qa_pass,
                    "fitted_flux_expectation_e": (
                        analysis.expectation_model.fitted_flux_e
                    ),
                    "fitted_flux_expectation_e_per_s": (
                        analysis.expectation_model.fitted_flux_e
                        / analysis.exposure_seconds
                    ),
                    "residual_expectation_e": (
                        analysis.expectation_model.residual_e
                    ),
                    "residual_expectation_ppm": (
                        analysis.expectation_model.residual_ppm
                    ),
                }
            )
        )
    combined = vstack(tables, metadata_conflicts="error")
    combined.meta = {
        "schema_id": _REFERENCE_LIGHTCURVE_SCHEMA_ID,
        "schema_version": _REFERENCE_LIGHTCURVE_SCHEMA_VERSION,
        "time_alignment": "simulation_raw_frame_index",
        "flux_factor_semantics": "dimensionless_relative_flux",
        "measured_flux_semantics": (
            "final_dn_calibrated_to_electrons_minus_expectation_background"
        ),
    }
    combined.write(path, format="ascii.ecsv", overwrite=False)
    _fsync_file(path)


def _write_centroid_quality_ecsv_v1(
    path: Path,
    *,
    analyses: Mapping[int, _CadenceAnalysis],
) -> None:
    from astropy.table import Table, vstack

    tables = []
    for _, analysis in sorted(analyses.items()):
        cadence_seconds = int(round(float(analysis.exposure_seconds[0])))
        tables.append(
            Table(
                {
                    "cadence_seconds": np.full(
                        analysis.time_seconds.shape,
                        cadence_seconds,
                        dtype=np.int32,
                    ),
                    "time_start_seconds": analysis.time_seconds,
                    "exposure_seconds": analysis.exposure_seconds,
                    "centroid_x_stamp_pixel_0based": analysis.photometry[
                        "centroid_x"
                    ],
                    "centroid_y_stamp_pixel_0based": analysis.photometry[
                        "centroid_y"
                    ],
                    "aperture_valid": analysis.photometry["aperture_valid"],
                    "aperture_usable_pixel_count": analysis.photometry[
                        "aperture_usable_pixel_count"
                    ],
                    "aperture_invalid_pixel_count": analysis.photometry[
                        "aperture_invalid_pixel_count"
                    ],
                    "saturated_pixel_count": analysis.photometry[
                        "saturated_pixel_count"
                    ],
                    "cosmic_pixel_count": analysis.photometry[
                        "cosmic_pixel_count"
                    ],
                    "background_usable_pixel_count": analysis.photometry[
                        "background_usable_pixel_count"
                    ],
                    "quality_bitmask": analysis.photometry["quality_bitmask"],
                }
            )
        )
    combined = vstack(tables, metadata_conflicts="error")
    combined.meta = _centroid_quality_metadata_v1()
    combined.write(path, format="ascii.ecsv", overwrite=False)
    _fsync_file(path)


def _write_cdpp_ecsv_v1(
    path: Path,
    *,
    analyses: Mapping[int, _CadenceAnalysis],
) -> None:
    from astropy.table import Table

    rows: list[tuple[Any, ...]] = []
    for _, analysis in sorted(analyses.items()):
        cadence_seconds = int(round(float(analysis.exposure_seconds[0])))
        for background_name, result in (
            ("expectation_background", analysis.expectation_cdpp),
            ("local_background", analysis.local_cdpp),
        ):
            if result is None:
                continue
            for window, metric in sorted(result.metrics_by_window_minutes.items()):
                rows.append(
                    (
                        cadence_seconds,
                        background_name,
                        int(window),
                        metric.total_bin_count,
                        metric.accepted_bin_count,
                        metric.rejected_bin_count,
                        metric.accepted_sample_count,
                        metric.minimum_coverage_fraction,
                        metric.minimum_accepted_bins,
                        metric.observed_cdpp_ppm,
                        metric.residual_cdpp_ppm,
                    )
                )
    table = Table(
        rows=rows,
        names=_CDPP_TABLE_COLUMNS,
    )
    table.meta = {
        "schema_id": _CDPP_TABLE_SCHEMA_ID,
        "estimator": _CDPP_ESTIMATOR,
    }
    table.write(path, format="ascii.ecsv", overwrite=False)
    _fsync_file(path)


def _quality_cadence_summary_v1(
    *,
    quality_bitmask: object,
    aperture_valid: object,
    uncertainty_valid: object,
    captured_flux_qa_pass: object,
    captured_flux_fraction: object,
) -> dict[str, Any]:
    quality = np.asarray(quality_bitmask, dtype=np.uint16)
    valid = np.asarray(aperture_valid, dtype=bool)
    uncertainty = np.asarray(uncertainty_valid, dtype=bool)
    captured_qa = np.asarray(captured_flux_qa_pass, dtype=bool)
    captured_fraction = np.asarray(captured_flux_fraction, dtype=np.float64)
    return {
        "cadence_count": int(quality.size),
        "aperture_valid_count": int(np.count_nonzero(valid)),
        "aperture_invalid_count": int(np.count_nonzero(~valid)),
        "quality_flag_counts": {
            name: int(np.count_nonzero(quality & bit))
            for name, bit in _QUALITY_FLAG_BITS.items()
        },
        "model_uncertainty_valid_count": int(np.count_nonzero(uncertainty)),
        "captured_flux_qa_pass_count": int(np.count_nonzero(captured_qa)),
        "captured_flux_qa_fail_count": int(np.count_nonzero(~captured_qa)),
        "minimum_captured_flux_fraction": float(np.min(captured_fraction)),
    }


def _quality_summary_v1(
    analyses: Mapping[int, _CadenceAnalysis],
) -> dict[str, Any]:
    cadences: dict[str, Any] = {}
    for _, analysis in sorted(analyses.items()):
        seconds = int(round(float(analysis.exposure_seconds[0])))
        cadences[f"{seconds}s"] = _quality_cadence_summary_v1(
            quality_bitmask=analysis.photometry["quality_bitmask"],
            aperture_valid=analysis.photometry["aperture_valid"],
            uncertainty_valid=analysis.uncertainty.valid_mask,
            captured_flux_qa_pass=analysis.captured_flux_qa_pass,
            captured_flux_fraction=analysis.captured_flux_fraction,
        )
    return {
        "schema_id": _QUALITY_SUMMARY_SCHEMA_ID,
        "schema_version": _QUALITY_SUMMARY_SCHEMA_VERSION,
        "centroid_coordinate_system": _QUALITY_SUMMARY_COORDINATE_SYSTEM,
        "cadences": cadences,
    }


def _write_science_figures_v1(
    staging: Path,
    *,
    analyses: Mapping[int, _CadenceAnalysis],
    representative_frames: tuple[_RepresentativeFrame, ...],
    aperture: ScienceApertureDefinition,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_root = staging / "figures"
    figures_root.mkdir()

    raw = analyses[min(analyses)]
    stride = max(1, int(math.ceil(raw.time_seconds.size / 20_000)))
    selection = slice(None, None, stride)
    days = raw.time_seconds[selection] / 86_400.0
    observed = raw.photometry["flux_expectation_bgsub_e"][selection]
    model = raw.expectation_model.fitted_flux_e[selection]
    figure, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(days, observed, color="#1f77b4", linewidth=0.7, label="Measured")
    axes[0].plot(days, model, color="#d62728", linewidth=0.7, label="Injected model")
    axes[0].set_ylabel("Flux (electrons)")
    axes[0].legend(loc="best")
    axes[0].set_title("Stamp science light curve")
    axes[1].plot(
        days,
        raw.expectation_model.residual_ppm[selection],
        color="#2f4f4f",
        linewidth=0.7,
    )
    axes[1].axhline(0.0, color="black", linewidth=0.5)
    axes[1].set_xlabel("Simulation time (days)")
    axes[1].set_ylabel("Residual (ppm)")
    figure.tight_layout()
    figure.savefig(figures_root / "lightcurve_overview.png", dpi=150)
    plt.close(figure)

    rows = []
    for _, analysis in sorted(analyses.items()):
        cadence = int(round(float(analysis.exposure_seconds[0])))
        for window, metric in sorted(
            analysis.expectation_cdpp.metrics_by_window_minutes.items()
        ):
            rows.append((f"{cadence}s/{window}m", metric.residual_cdpp_ppm))
    figure, axis = plt.subplots(figsize=(max(8, len(rows) * 0.45), 4.5))
    axis.bar(
        np.arange(len(rows)),
        [item[1] for item in rows],
        color="#4c78a8",
    )
    axis.set_xticks(np.arange(len(rows)), [item[0] for item in rows], rotation=60)
    axis.set_ylabel("Residual CDPP (ppm)")
    axis.set_xlabel("Cadence / window")
    axis.set_title("Coverage-aware CDPP summary")
    figure.tight_layout()
    figure.savefig(figures_root / "cdpp_summary.png", dpi=150)
    plt.close(figure)

    calibrated = []
    for item in representative_frames:
        batch = item.batch
        image = (
            (
                batch.final_dn.astype(np.float64)
                - batch.bias_level_sum_dn[:, None, None]
                - batch.column_noise_sum_dn_by_x[:, None, :]
            )
            * batch.gain_e_per_dn
            - batch.background_expectation_e
        )[0]
        calibrated.append(image)
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, image, item in zip(axes, calibrated, representative_frames, strict=True):
        shown = axis.imshow(image, origin="lower", cmap="viridis")
        y, x = np.nonzero(aperture.aperture_mask)
        axis.scatter(x, y, s=4, facecolors="none", edgecolors="white", linewidths=0.3)
        axis.set_title(item.selection_role.replace("_", " ").title())
        axis.set_xlabel("Stamp x (pixel)")
        axis.set_ylabel("Stamp y (pixel)")
        figure.colorbar(shown, ax=axis, fraction=0.046, pad=0.04, label="Electrons")
    figure.tight_layout()
    figure.savefig(figures_root / "representative_frames.png", dpi=150)
    plt.close(figure)
    for name in (
        "lightcurve_overview.png",
        "cdpp_summary.png",
        "representative_frames.png",
    ):
        _fsync_file(figures_root / name)
    _fsync_directory(figures_root)


def _write_representative_frames_hdf5(
    path: Path,
    *,
    frames: tuple[_RepresentativeFrame, ...],
    raw_identities: Sequence[Mapping[str, Any]],
) -> None:
    import h5py

    batches = [item.batch for item in frames]
    final = np.concatenate([item.final_dn for item in batches], axis=0)
    background = np.concatenate(
        [item.background_expectation_e for item in batches], axis=0
    )
    bias = np.concatenate([item.bias_level_sum_dn for item in batches])
    column = np.concatenate(
        [item.column_noise_sum_dn_by_x for item in batches], axis=0
    )
    gain = batches[0].gain_e_per_dn
    calibrated = (
        (final.astype(np.float64) - bias[:, None, None] - column[:, None, :])
        * gain
    )
    calibrated_bgsub = calibrated - background
    with h5py.File(path, "w") as handle:
        handle.attrs["schema_id"] = _REPRESENTATIVE_FRAMES_SCHEMA_ID
        handle.attrs["schema_version"] = _REPRESENTATIVE_FRAMES_SCHEMA_VERSION
        handle.attrs["complete"] = False
        handle.attrs["observation_product"] = "final_dn"
        handle.attrs["background_realization_used"] = False
        handle.create_dataset("final_dn", data=final)
        handle.create_dataset("calibrated_e", data=calibrated)
        handle.create_dataset("calibrated_bgsub_e", data=calibrated_bgsub)
        handle.create_dataset("background_expectation_e", data=background)
        handle.create_dataset(
            "captured_flux_fraction",
            data=np.concatenate(
                [item.captured_flux_fraction for item in batches]
            ),
        )
        handle.create_dataset(
            "captured_flux_denominator_e",
            data=np.concatenate(
                [item.captured_flux_denominator_e for item in batches]
            ),
        )
        handle.create_dataset(
            "captured_flux_qa_pass",
            data=np.concatenate(
                [item.captured_flux_qa_pass for item in batches]
            ),
        )
        handle.create_dataset(
            "valid_mask",
            data=np.concatenate([item.valid_mask for item in batches], axis=0),
        )
        handle.create_dataset(
            "saturated_mask",
            data=np.concatenate([item.saturated_mask for item in batches], axis=0),
        )
        handle.create_dataset(
            "cosmic_mask",
            data=np.concatenate([item.cosmic_mask for item in batches], axis=0),
        )
        handle.create_dataset("bias_level_sum_dn", data=bias)
        handle.create_dataset("column_noise_sum_dn_by_x", data=column)
        handle.create_dataset("gain_e_per_dn", data=gain)
        handle.create_dataset(
            "time_start_seconds",
            data=np.concatenate([item.time_start_seconds for item in batches]),
        )
        handle.create_dataset(
            "exposure_seconds",
            data=np.concatenate([item.exposure_seconds for item in batches]),
        )
        handle.create_dataset(
            "raw_frame_start_index",
            data=np.concatenate([item.raw_frame_start_index for item in batches]),
        )
        handle.create_dataset(
            "raw_frame_stop_index_exclusive",
            data=np.concatenate(
                [item.raw_frame_stop_index_exclusive for item in batches]
            ),
        )
        string_dtype = h5py.string_dtype(encoding="utf-8")
        handle.create_dataset(
            "selection_role",
            data=np.asarray([item.selection_role for item in frames], dtype=object),
            dtype=string_dtype,
        )
        handle.create_dataset(
            "input_shard_path",
            data=np.asarray(
                [
                    raw_identities[item.input_shard_index]["path"]
                    for item in frames
                ],
                dtype=object,
            ),
            dtype=string_dtype,
        )
        handle.create_dataset(
            "input_shard_semantic_sha256",
            data=np.asarray(
                [
                    raw_identities[item.input_shard_index]["semantic_sha256"]
                    for item in frames
                ],
                dtype=object,
            ),
            dtype=string_dtype,
        )
        handle.flush()
        handle.attrs["complete"] = True
        handle.flush()
    _fsync_file(path)


def _file_identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "size_bytes": int(path.stat().st_size),
        "sha256": digest.hexdigest(),
    }


def _write_analysis_artifacts(
    staging: Path,
    *,
    contract: Mapping[str, Any],
    q: NDArray[np.float64],
    aperture: ScienceApertureDefinition,
    analyses: Mapping[int, _CadenceAnalysis],
    representative_frames: tuple[_RepresentativeFrame, ...],
    raw_identities: Sequence[Mapping[str, Any]],
) -> None:
    analysis_product = contract.get("analysis_product")
    artifact_names = _analysis_artifact_names_v1(analysis_product)
    if analysis_product == "science_optimal_aperture_v1" and (
        aperture.signal_template_e is None or aperture.noise_template_e is None
    ):
        raise StampScienceAnalysisContractError(
            "science-optimal analysis requires signal/noise template artifacts"
        )
    if analysis_product == "reference_fixed13_v1" and (
        aperture.signal_template_e is not None or aperture.noise_template_e is not None
    ):
        raise StampScienceAnalysisContractError(
            "reference analysis forbids signal/noise template artifacts"
        )
    cdpp = _all_cdpp_payload(analyses)
    _write_authoritative_hdf5(
        staging / "photometry.h5",
        contract=contract,
        q=q,
        aperture=aperture,
        analyses=analyses,
        cdpp_payload=cdpp,
    )
    _write_portable_ecsv_v1(staging / "photometry.ecsv", analyses=analyses)
    _write_reference_lightcurve_ecsv_v1(
        staging / "reference_lightcurve.ecsv",
        analyses=analyses,
    )
    _write_centroid_quality_ecsv_v1(
        staging / "centroid_quality.ecsv",
        analyses=analyses,
    )
    _write_cdpp_ecsv_v1(staging / "cdpp.ecsv", analyses=analyses)
    _write_representative_frames_hdf5(
        staging / "representative_calibrated_frames.h5",
        frames=representative_frames,
        raw_identities=raw_identities,
    )
    _write_json(staging / "aperture_definition.json", _aperture_payload(aperture))
    _write_json(staging / "cdpp.json", cdpp)
    _write_json(
        staging / "quality_summary.json",
        _quality_summary_v1(analyses),
    )
    _write_science_figures_v1(
        staging,
        analyses=analyses,
        representative_frames=representative_frames,
        aperture=aperture,
    )
    np.save(staging / "aperture_mask.npy", aperture.aperture_mask, allow_pickle=False)
    np.save(
        staging / "background_mask.npy",
        (
            np.zeros(aperture.aperture_mask.shape, dtype=bool)
            if aperture.background_mask is None
            else aperture.background_mask
        ),
        allow_pickle=False,
    )
    if analysis_product == "science_optimal_aperture_v1":
        assert aperture.signal_template_e is not None
        assert aperture.noise_template_e is not None
        np.save(
            staging / "signal_template_e.npy",
            aperture.signal_template_e,
            allow_pickle=False,
        )
        np.save(
            staging / "noise_template_e.npy",
            aperture.noise_template_e,
            allow_pickle=False,
        )
    for name in (
        "aperture_mask.npy",
        "background_mask.npy",
        *(
            _SCIENCE_TEMPLATE_ARTIFACT_NAMES
            if analysis_product == "science_optimal_aperture_v1"
            else ()
        ),
    ):
        _fsync_file(staging / name)
    publication_manifest = {
        "schema_id": STAMP_SCIENCE_ANALYSIS_PUBLICATION_SCHEMA_ID,
        "schema_version": STAMP_SCIENCE_ANALYSIS_PUBLICATION_SCHEMA_VERSION,
        "complete": True,
        "ready": True,
        "authoritative_product": "photometry.h5",
        "contract": contract,
        "artifacts": {
            name: _file_identity(staging / name) for name in artifact_names
        },
    }
    _write_json(staging / "analysis_manifest.json", publication_manifest)


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StampScienceAnalysisContractError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise StampScienceAnalysisContractError(f"{label} must be a JSON object")
    return value


def _decode_hdf_json(value: Any, *, name: str) -> dict[str, Any]:
    if isinstance(value, np.ndarray):
        if value.shape != ():
            raise StampScienceAnalysisContractError(f"{name} must be scalar")
        value = value.item()
    if isinstance(value, (bytes, np.bytes_)):
        value = bytes(value).decode("utf-8")
    if not isinstance(value, str):
        raise StampScienceAnalysisContractError(f"{name} must be UTF-8 JSON")
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as error:
        raise StampScienceAnalysisContractError(f"{name} contains invalid JSON") from error
    if not isinstance(decoded, dict):
        raise StampScienceAnalysisContractError(f"{name} must encode an object")
    return decoded


def _validate_reference_lightcurve_ecsv_v1(
    path: Path,
    *,
    authoritative_hdf5_path: Path,
    contract: Mapping[str, Any],
) -> None:
    import h5py
    from astropy.table import Table

    expected_contract = _reference_lightcurve_contract_v1()
    reference_contract = contract.get("reference_lightcurve")
    if (
        not isinstance(reference_contract, Mapping)
        or not _is_native_schema_version(
            reference_contract.get("schema_version"),
            _REFERENCE_LIGHTCURVE_SCHEMA_VERSION,
        )
        or dict(reference_contract) != expected_contract
    ):
        raise StampScienceAnalysisContractError(
            "analysis contract lacks the v2 reference-lightcurve contract"
        )
    try:
        table = Table.read(path, format="ascii.ecsv")
    except (OSError, UnicodeError, TypeError, ValueError) as error:
        raise StampScienceAnalysisContractError(
            "reference-lightcurve ECSV cannot be read"
        ) from error
    if (
        table.meta.get("schema_id") != _REFERENCE_LIGHTCURVE_SCHEMA_ID
        or not _is_native_schema_version(
            table.meta.get("schema_version"),
            _REFERENCE_LIGHTCURVE_SCHEMA_VERSION,
        )
    ):
        raise StampScienceAnalysisContractError(
            "reference-lightcurve ECSV schema is invalid"
        )
    if not set(_REFERENCE_LIGHTCURVE_REQUIRED_COLUMNS).issubset(table.colnames):
        raise StampScienceAnalysisContractError(
            "reference-lightcurve ECSV lacks required columns"
        )

    table_cadence = np.asarray(table["cadence_seconds"])
    if table_cadence.dtype.kind != "i":
        raise StampScienceAnalysisContractError(
            "reference-lightcurve ECSV column dtype differs: cadence_seconds"
        )
    total_expected_rows = 0
    with h5py.File(authoritative_hdf5_path, "r") as handle:
        cadence_groups = handle["cadences"]
        expected_cadences = {
            int(name.removesuffix("s")) for name in cadence_groups
        }
        if set(np.unique(table_cadence).tolist()) != expected_cadences:
            raise StampScienceAnalysisContractError(
                "reference-lightcurve ECSV cadence set differs from HDF5"
            )
        for seconds in sorted(expected_cadences):
            group = cadence_groups[f"{seconds}s"]
            selected = table_cadence == seconds
            count = int(group["time_start_seconds"].shape[0])
            total_expected_rows += count
            if int(np.count_nonzero(selected)) != count:
                raise StampScienceAnalysisContractError(
                    f"reference-lightcurve ECSV cadence {seconds}s axis differs"
                )

            float_columns = {
                "time_start_seconds": np.asarray(
                    group["time_start_seconds"], dtype=np.float64
                ),
                "exposure_seconds": np.asarray(
                    group["exposure_seconds"], dtype=np.float64
                ),
                "raw_relative_flux_mean": (
                    np.asarray(group["raw_factor_sum"], dtype=np.float64)
                    / int(group.attrs["coadd_factor"])
                ),
                "raw_relative_flux_sum": np.asarray(
                    group["raw_factor_sum"], dtype=np.float64
                ),
                "flux_expectation_bgsub_e": np.asarray(
                    group["flux_expectation_bgsub_e"], dtype=np.float64
                ),
                "flux_expectation_bgsub_e_per_s": np.asarray(
                    group["flux_expectation_bgsub_e_per_s"], dtype=np.float64
                ),
                "fitted_flux_expectation_e": np.asarray(
                    group["fitted_flux_expectation_e"], dtype=np.float64
                ),
                "fitted_flux_expectation_e_per_s": np.asarray(
                    group["fitted_flux_expectation_e_per_s"], dtype=np.float64
                ),
                "residual_expectation_e": np.asarray(
                    group["residual_expectation_e"], dtype=np.float64
                ),
                "residual_expectation_ppm": np.asarray(
                    group["residual_expectation_ppm"], dtype=np.float64
                ),
                "captured_flux_fraction": np.asarray(
                    group["captured_flux_fraction"], dtype=np.float64
                ),
                "captured_flux_denominator_e": np.asarray(
                    group["captured_flux_denominator_e"], dtype=np.float64
                ),
            }
            for column, expected in float_columns.items():
                actual_values = np.asarray(table[column])
                if actual_values.dtype.kind != "f":
                    raise StampScienceAnalysisContractError(
                        "reference-lightcurve ECSV column dtype differs: "
                        f"{column}"
                    )
                actual = np.asarray(actual_values, dtype=np.float64)[selected]
                if not np.allclose(
                    actual,
                    expected,
                    rtol=1e-12,
                    atol=1e-12,
                    equal_nan=True,
                ):
                    raise StampScienceAnalysisContractError(
                        "reference-lightcurve ECSV column differs from HDF5: "
                        f"{column}"
                    )
            integer_columns = {
                "raw_frame_start_index": np.asarray(
                    group["raw_frame_start_index"], dtype=np.int64
                ),
                "raw_frame_stop_index_exclusive": np.asarray(
                    group["raw_frame_stop_index_exclusive"], dtype=np.int64
                ),
                "quality_bitmask": np.asarray(
                    group["quality_bitmask"], dtype=np.uint16
                ),
            }
            for column, expected in integer_columns.items():
                actual_values = np.asarray(table[column])
                if actual_values.dtype.kind != expected.dtype.kind:
                    raise StampScienceAnalysisContractError(
                        "reference-lightcurve ECSV column dtype differs: "
                        f"{column}"
                    )
                actual = actual_values[selected]
                if not np.array_equal(actual, expected):
                    raise StampScienceAnalysisContractError(
                        "reference-lightcurve ECSV column differs from HDF5: "
                        f"{column}"
                    )
            aperture_valid_values = np.asarray(table["aperture_valid"])
            if aperture_valid_values.dtype.kind != "b":
                raise StampScienceAnalysisContractError(
                    "reference-lightcurve ECSV column dtype differs: aperture_valid"
                )
            aperture_valid = aperture_valid_values[selected]
            if not np.array_equal(
                aperture_valid,
                np.asarray(group["aperture_valid"], dtype=bool),
            ):
                raise StampScienceAnalysisContractError(
                    "reference-lightcurve ECSV column differs from HDF5: "
                    "aperture_valid"
                )
            captured_qa_values = np.asarray(table["captured_flux_qa_pass"])
            if captured_qa_values.dtype.kind != "b":
                raise StampScienceAnalysisContractError(
                    "reference-lightcurve ECSV column dtype differs: "
                    "captured_flux_qa_pass"
                )
            captured_qa = captured_qa_values[selected]
            if not np.array_equal(
                captured_qa,
                np.asarray(group["captured_flux_qa_pass"], dtype=bool),
            ):
                raise StampScienceAnalysisContractError(
                    "reference-lightcurve ECSV column differs from HDF5: "
                    "captured_flux_qa_pass"
                )
    if len(table) != total_expected_rows:
        raise StampScienceAnalysisContractError(
            "reference-lightcurve ECSV row count differs from HDF5"
        )


def _metadata_matches_exactly(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> bool:
    return set(actual) == set(expected) and all(
        type(actual[name]) is type(value) and actual[name] == value
        for name, value in expected.items()
    )


def _validate_portable_cadence_table_parity_v1(
    path: Path,
    *,
    authoritative_hdf5_path: Path,
    label: str,
    expected_metadata: Mapping[str, Any],
    expected_columns: tuple[str, ...],
    hdf_dataset_by_column: Mapping[str, str],
    boolean_columns: frozenset[str],
) -> None:
    """Bind one portable cadence table exactly to authoritative HDF5 rows."""

    import h5py
    from astropy.table import Table

    try:
        table = Table.read(path, format="ascii.ecsv")
    except (OSError, UnicodeError, TypeError, ValueError) as error:
        raise StampScienceAnalysisContractError(f"{label} cannot be read") from error
    if not _metadata_matches_exactly(table.meta, expected_metadata) or tuple(
        table.colnames
    ) != expected_columns:
        raise StampScienceAnalysisContractError(f"{label} schema is invalid")

    expected_parts: dict[str, list[NDArray[Any]]] = {
        name: [] for name in expected_columns
    }
    with h5py.File(authoritative_hdf5_path, "r") as handle:
        cadence_groups = handle["cadences"]
        cadence_names = sorted(
            cadence_groups,
            key=lambda item: int(item.removesuffix("s")),
        )
        for cadence_name in cadence_names:
            seconds = int(cadence_name.removesuffix("s"))
            group = cadence_groups[cadence_name]
            count = int(group["time_start_seconds"].shape[0])
            expected_parts["cadence_seconds"].append(
                np.full(count, seconds, dtype=np.int32)
            )
            for column, dataset_name in hdf_dataset_by_column.items():
                if dataset_name not in group:
                    raise StampScienceAnalysisContractError(
                        f"{label} authoritative HDF5 dataset is missing: "
                        f"{dataset_name}"
                    )
                expected = np.asarray(group[dataset_name])
                if expected.ndim != 1 or expected.shape != (count,):
                    raise StampScienceAnalysisContractError(
                        f"{label} authoritative HDF5 cadence axis is invalid: "
                        f"{dataset_name}"
                    )
                expected_parts[column].append(expected)

    expected_values = {
        name: np.concatenate(parts) if parts else np.asarray([], dtype=np.float64)
        for name, parts in expected_parts.items()
    }
    expected_row_count = int(expected_values["cadence_seconds"].shape[0])
    if expected_row_count <= 0 or len(table) != expected_row_count:
        raise StampScienceAnalysisContractError(
            f"{label} row count differs from authoritative HDF5"
        )
    for column in expected_columns:
        table_column = table[column]
        mask = getattr(table_column, "mask", None)
        actual = np.asarray(table_column)
        expected = expected_values[column]
        if (
            actual.ndim != 1
            or actual.shape != expected.shape
            or (mask is not None and np.any(mask))
            or actual.dtype.kind != expected.dtype.kind
            or (
                column in boolean_columns
                and (actual.dtype.kind != "b" or expected.dtype.kind != "b")
            )
        ):
            raise StampScienceAnalysisContractError(
                f"{label} column dtype/axis differs from authoritative HDF5: "
                f"{column}"
            )
        if not np.array_equal(actual, expected, equal_nan=True):
            raise StampScienceAnalysisContractError(
                f"{label} column differs from authoritative HDF5: {column}"
            )


def _validate_portable_photometry_ecsv_v2(
    path: Path,
    *,
    authoritative_hdf5_path: Path,
) -> None:
    """Bind the exact v2 portable table to all authoritative cadence rows."""

    _validate_portable_cadence_table_parity_v1(
        path,
        authoritative_hdf5_path=authoritative_hdf5_path,
        label="portable photometry ECSV v2",
        expected_metadata=_photometry_table_metadata_v2(),
        expected_columns=_PHOTOMETRY_TABLE_COLUMNS,
        hdf_dataset_by_column=_PHOTOMETRY_TABLE_HDF_DATASETS,
        boolean_columns=_PHOTOMETRY_TABLE_BOOLEAN_COLUMNS,
    )


def _validate_centroid_quality_ecsv_v1(
    path: Path,
    *,
    authoritative_hdf5_path: Path,
) -> None:
    _validate_portable_cadence_table_parity_v1(
        path,
        authoritative_hdf5_path=authoritative_hdf5_path,
        label="centroid-quality ECSV",
        expected_metadata=_centroid_quality_metadata_v1(),
        expected_columns=_CENTROID_QUALITY_COLUMNS,
        hdf_dataset_by_column=_CENTROID_QUALITY_HDF_DATASETS,
        boolean_columns=_CENTROID_QUALITY_BOOLEAN_COLUMNS,
    )


def _cdpp_table_rows_from_payload_v1(
    payload: Mapping[str, Any],
    *,
    background_strategy: object,
) -> list[dict[str, Any]]:
    if (
        set(payload) != {"schema_id", "estimator", "cadences"}
        or payload.get("schema_id") != _CDPP_SCHEMA_ID
        or payload.get("estimator") != _CDPP_ESTIMATOR
        or not isinstance(payload.get("cadences"), Mapping)
        or not payload["cadences"]
        or background_strategy
        not in {
            _EXPECTATION_ONLY_BACKGROUND_STRATEGY,
            _LOCAL_DIAGNOSTIC_BACKGROUND_STRATEGY,
        }
    ):
        raise StampScienceAnalysisContractError(
            "authoritative HDF5 CDPP schema is invalid"
        )
    cadence_records: list[tuple[int, Mapping[str, Any]]] = []
    for cadence_name, cadence_payload in payload["cadences"].items():
        if (
            not isinstance(cadence_name, str)
            or not cadence_name.endswith("s")
            or not cadence_name[:-1].isdigit()
            or int(cadence_name[:-1]) <= 0
            or cadence_name != f"{int(cadence_name[:-1])}s"
            or not isinstance(cadence_payload, Mapping)
            or set(cadence_payload)
            != {
                "coadd_factor",
                "expectation_background",
                "local_background",
            }
            or type(cadence_payload.get("coadd_factor")) is not int
            or cadence_payload["coadd_factor"] <= 0
        ):
            raise StampScienceAnalysisContractError(
                "authoritative HDF5 CDPP cadence schema is invalid"
            )
        cadence_records.append((int(cadence_name[:-1]), cadence_payload))

    metric_fields = set(_CDPP_TABLE_COLUMNS[2:]) | {
        "observed_estimator",
        "residual_estimator",
        "aggregation",
    }
    nonnegative_integer_fields = {
        "total_bin_count",
        "accepted_bin_count",
        "rejected_bin_count",
        "accepted_sample_count",
    }
    expected_rows: list[dict[str, Any]] = []
    for cadence_seconds, cadence_payload in sorted(cadence_records):
        for background_name in (
            "expectation_background",
            "local_background",
        ):
            background_payload = cadence_payload[background_name]
            if not isinstance(background_payload, Mapping):
                raise StampScienceAnalysisContractError(
                    "authoritative HDF5 CDPP background schema is invalid"
                )
            if "status" in background_payload:
                if (
                    background_name != "local_background"
                    or background_strategy
                    != _EXPECTATION_ONLY_BACKGROUND_STRATEGY
                    or dict(background_payload)
                    != {
                        "status": "not_computed",
                        "reason": (
                            "background_strategy_delivered_expectation_only"
                        ),
                    }
                ):
                    raise StampScienceAnalysisContractError(
                        "authoritative HDF5 CDPP background role is invalid"
                    )
                continue
            if (
                background_name == "local_background"
                and background_strategy == _EXPECTATION_ONLY_BACKGROUND_STRATEGY
            ):
                raise StampScienceAnalysisContractError(
                    "authoritative HDF5 CDPP background role is invalid"
                )
            if (
                set(background_payload) != {"metrics_by_window_minutes", "binned_rows"}
                or not isinstance(
                    background_payload.get("metrics_by_window_minutes"), Mapping
                )
                or not isinstance(background_payload.get("binned_rows"), list)
            ):
                raise StampScienceAnalysisContractError(
                    "authoritative HDF5 CDPP background schema is invalid"
                )
            metrics: list[tuple[int, Mapping[str, Any]]] = []
            for window_name, metric in background_payload[
                "metrics_by_window_minutes"
            ].items():
                if (
                    not isinstance(window_name, str)
                    or not window_name.isdigit()
                    or int(window_name) <= 0
                    or window_name != str(int(window_name))
                    or not isinstance(metric, Mapping)
                    or set(metric) != metric_fields
                ):
                    raise StampScienceAnalysisContractError(
                        "authoritative HDF5 CDPP metric schema is invalid"
                    )
                window = int(window_name)
                if (
                    type(metric.get("window_minutes")) is not int
                    or metric["window_minutes"] != window
                    or any(
                        type(metric.get(name)) is not int
                        or metric[name] < 0
                        for name in nonnegative_integer_fields
                    )
                    or type(metric.get("minimum_accepted_bins")) is not int
                    or metric["minimum_accepted_bins"] <= 0
                    or metric["total_bin_count"]
                    != metric["accepted_bin_count"] + metric["rejected_bin_count"]
                    or type(metric.get("minimum_coverage_fraction")) is not float
                    or not math.isfinite(metric["minimum_coverage_fraction"])
                    or not 0.0 < float(metric["minimum_coverage_fraction"]) <= 1.0
                    or metric.get("observed_estimator") != _CDPP_ESTIMATOR
                    or metric.get("residual_estimator") != _CDPP_ESTIMATOR
                    or metric.get("aggregation")
                    != "valid_cadence_counts_normalized_by_actual_effective_exposure"
                ):
                    raise StampScienceAnalysisContractError(
                        "authoritative HDF5 CDPP metric values are invalid"
                    )
                for name in ("observed_cdpp_ppm", "residual_cdpp_ppm"):
                    value = metric[name]
                    if value is not None and (
                        type(value) is not float
                        or not math.isfinite(value)
                        or value < 0.0
                    ):
                        raise StampScienceAnalysisContractError(
                            "authoritative HDF5 CDPP metric values are invalid"
                        )
                metrics.append((window, metric))

            metrics_by_window = dict(metrics)
            grouped_rows: dict[int, list[Mapping[str, Any]]] = {
                window: [] for window in metrics_by_window
            }
            seen_bins: set[tuple[int, int]] = set()
            for row in background_payload["binned_rows"]:
                if not isinstance(row, Mapping) or set(row) != _CDPP_BINNED_ROW_FIELDS:
                    raise StampScienceAnalysisContractError(
                        "authoritative HDF5 CDPP binned-row schema is invalid"
                    )
                window = row.get("window_minutes")
                bin_id = row.get("bin_id")
                expected_count = row.get("expected_cadence_count")
                valid_count = row.get("valid_cadence_count")
                accepted = row.get("accepted")
                if (
                    type(window) is not int
                    or window not in metrics_by_window
                    or type(bin_id) is not int
                    or (window, bin_id) in seen_bins
                    or type(expected_count) is not int
                    or expected_count <= 0
                    or type(valid_count) is not int
                    or not 0 <= valid_count <= expected_count
                    or type(accepted) is not bool
                ):
                    raise StampScienceAnalysisContractError(
                        "authoritative HDF5 CDPP binned-row values are invalid"
                    )
                float_fields = (
                    "time_start_seconds",
                    "time_stop_seconds",
                    "expected_exposure_seconds",
                    "effective_exposure_seconds",
                    "coverage_fraction",
                )
                if any(
                    type(row.get(name)) is not float
                    or not math.isfinite(row[name])
                    for name in float_fields
                ):
                    raise StampScienceAnalysisContractError(
                        "authoritative HDF5 CDPP binned-row values are invalid"
                    )
                time_start = row["time_start_seconds"]
                time_stop = row["time_stop_seconds"]
                expected_exposure = row["expected_exposure_seconds"]
                effective_exposure = row["effective_exposure_seconds"]
                coverage = row["coverage_fraction"]
                if (
                    time_stop <= time_start
                    or expected_exposure <= 0.0
                    or not math.isclose(
                        time_stop - time_start,
                        expected_exposure,
                        rel_tol=0.0,
                        abs_tol=_TIME_ATOL_SECONDS,
                    )
                    or not math.isclose(
                        expected_exposure,
                        float(window * 60),
                        rel_tol=0.0,
                        abs_tol=_TIME_ATOL_SECONDS,
                    )
                    or not math.isclose(
                        expected_exposure,
                        float(expected_count * cadence_seconds),
                        rel_tol=0.0,
                        abs_tol=_TIME_ATOL_SECONDS,
                    )
                    or effective_exposure < 0.0
                    or effective_exposure > expected_exposure + _TIME_ATOL_SECONDS
                    or not math.isclose(
                        effective_exposure,
                        float(valid_count * cadence_seconds),
                        rel_tol=0.0,
                        abs_tol=_TIME_ATOL_SECONDS,
                    )
                    or not 0.0 <= coverage <= 1.0
                    or not math.isclose(
                        coverage,
                        effective_exposure / expected_exposure,
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                ):
                    raise StampScienceAnalysisContractError(
                        "authoritative HDF5 CDPP binned-row values are invalid"
                    )
                optional_float_fields = (
                    "observed_flux_rate_e_per_s",
                    "model_flux_rate_e_per_s",
                    "residual_fraction_ppm",
                )
                if any(
                    value is not None
                    and (type(value) is not float or not math.isfinite(value))
                    for value in (row[name] for name in optional_float_fields)
                ):
                    raise StampScienceAnalysisContractError(
                        "authoritative HDF5 CDPP binned-row values are invalid"
                    )
                observed = row["observed_flux_rate_e_per_s"]
                model = row["model_flux_rate_e_per_s"]
                residual = row["residual_fraction_ppm"]
                threshold = metrics_by_window[window][
                    "minimum_coverage_fraction"
                ]
                if (
                    accepted != (coverage + _TIME_ATOL_SECONDS >= threshold)
                    or (observed is None) != (valid_count == 0)
                    or (model is None) != (valid_count == 0)
                    or (residual is None) != (valid_count == 0)
                    or (model is not None and model <= 0.0)
                ):
                    raise StampScienceAnalysisContractError(
                        "authoritative HDF5 CDPP binned-row values are invalid"
                    )
                seen_bins.add((window, bin_id))
                grouped_rows[window].append(row)

            for window, metric in metrics:
                rows = grouped_rows[window]
                accepted_rows = [row for row in rows if row["accepted"]]
                if (
                    metric["total_bin_count"] != len(rows)
                    or metric["accepted_bin_count"] != len(accepted_rows)
                    or metric["rejected_bin_count"]
                    != len(rows) - len(accepted_rows)
                    or metric["accepted_sample_count"]
                    != sum(row["valid_cadence_count"] for row in accepted_rows)
                ):
                    raise StampScienceAnalysisContractError(
                        "authoritative HDF5 CDPP metric conflicts with binned rows"
                    )

                def expected_cdpp(
                    field: str,
                    *,
                    divide_by_center: bool,
                ) -> float | None:
                    if len(accepted_rows) < metric["minimum_accepted_bins"]:
                        return None
                    samples = np.asarray(
                        [row[field] for row in accepted_rows],
                        dtype=np.float64,
                    )
                    center = float(np.median(samples))
                    if divide_by_center and center <= 0.0:
                        return None
                    value = float(
                        1.4826 * np.mean(np.abs(samples - center), dtype=np.float64)
                    )
                    if divide_by_center:
                        value = value / center * 1_000_000.0
                    return value

                for metric_name, row_name, divide_by_center in (
                    (
                        "observed_cdpp_ppm",
                        "observed_flux_rate_e_per_s",
                        True,
                    ),
                    ("residual_cdpp_ppm", "residual_fraction_ppm", False),
                ):
                    expected = expected_cdpp(
                        row_name,
                        divide_by_center=divide_by_center,
                    )
                    actual = metric[metric_name]
                    if (actual is None) != (expected is None) or (
                        actual is not None
                        and expected is not None
                        and not math.isclose(
                            actual,
                            expected,
                            rel_tol=1e-12,
                            abs_tol=1e-9,
                        )
                    ):
                        raise StampScienceAnalysisContractError(
                            "authoritative HDF5 CDPP metric conflicts with binned rows"
                        )
            for _, metric in sorted(metrics):
                expected_rows.append(
                    {
                        "cadence_seconds": cadence_seconds,
                        "background_estimator": background_name,
                        **{
                            name: metric[name]
                            for name in _CDPP_TABLE_COLUMNS[2:]
                        },
                    }
                )
    return expected_rows


def _validate_cdpp_portable_products_v1(
    root: Path,
    *,
    authoritative_payload: Mapping[str, Any],
    background_strategy: object,
) -> None:
    from astropy.table import Table

    expected_rows = _cdpp_table_rows_from_payload_v1(
        authoritative_payload,
        background_strategy=background_strategy,
    )
    portable_json = _read_json_object(root / "cdpp.json", label="CDPP product")
    if _canonical_json_text(
        portable_json,
        name="portable CDPP JSON",
    ) != _canonical_json_text(
        dict(authoritative_payload),
        name="authoritative HDF5 CDPP",
    ):
        raise StampScienceAnalysisContractError(
            "portable CDPP JSON differs from authoritative HDF5 CDPP"
        )
    try:
        table = Table.read(root / "cdpp.ecsv", format="ascii.ecsv")
    except (OSError, UnicodeError, TypeError, ValueError) as error:
        raise StampScienceAnalysisContractError(
            "portable CDPP ECSV cannot be read"
        ) from error
    if (
        dict(table.meta)
        != {
            "schema_id": _CDPP_TABLE_SCHEMA_ID,
            "estimator": _CDPP_ESTIMATOR,
        }
        or tuple(table.colnames) != _CDPP_TABLE_COLUMNS
    ):
        raise StampScienceAnalysisContractError(
            "portable CDPP ECSV schema is invalid"
        )
    integer_columns = {
        "cadence_seconds",
        "window_minutes",
        "total_bin_count",
        "accepted_bin_count",
        "rejected_bin_count",
        "accepted_sample_count",
        "minimum_accepted_bins",
    }
    float_columns = {
        "minimum_coverage_fraction",
        "observed_cdpp_ppm",
        "residual_cdpp_ppm",
    }
    if (
        any(table[name].dtype.kind != "i" for name in integer_columns)
        or table["background_estimator"].dtype.kind not in {"S", "U"}
        or any(table[name].dtype.kind != "f" for name in float_columns)
    ):
        raise StampScienceAnalysisContractError(
            "portable CDPP ECSV column types are invalid"
        )

    def normalized(value: Any) -> Any:
        if np.ma.is_masked(value):
            return None
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    actual_rows = [
        {name: normalized(row[name]) for name in _CDPP_TABLE_COLUMNS}
        for row in table
    ]
    if actual_rows != expected_rows:
        raise StampScienceAnalysisContractError(
            "portable CDPP ECSV values differ from authoritative HDF5 CDPP"
        )


def _publication_cadence_contract_v1(
    contract: Mapping[str, Any],
) -> tuple[
    tuple[int, ...],
    tuple[int, ...],
    float,
    frozenset[str],
    dict[str, int],
]:
    policy = contract.get("policy")
    photometry_policy = (
        policy.get("photometry") if isinstance(policy, Mapping) else None
    )
    factors_value = policy.get("coadd_factors") if isinstance(policy, Mapping) else None
    raw_exposure_value = (
        policy.get("raw_exposure_seconds") if isinstance(policy, Mapping) else None
    )
    factors = (
        tuple(factors_value) if isinstance(factors_value, list) else ()
    )
    if (
        not factors
        or any(type(value) is not int or value <= 0 for value in factors)
        or factors[0] != 1
        or tuple(sorted(set(factors))) != factors
        or isinstance(raw_exposure_value, bool)
        or not isinstance(raw_exposure_value, (int, float))
        or not math.isfinite(float(raw_exposure_value))
        or float(raw_exposure_value) <= 0.0
        or not isinstance(photometry_policy, Mapping)
    ):
        raise StampScienceAnalysisContractError(
            "analysis cadence policy is invalid"
        )
    raw_exposure = float(raw_exposure_value)
    expected_seconds = tuple(
        int(round(raw_exposure * factor)) for factor in factors
    )
    cadence_value = contract.get("cadence_seconds")
    cadence_seconds = (
        tuple(cadence_value) if isinstance(cadence_value, list) else ()
    )
    expected_names = frozenset(f"{seconds}s" for seconds in expected_seconds)
    captured = contract.get("captured_flux_qa")
    captured_cadences = (
        captured.get("cadences") if isinstance(captured, Mapping) else None
    )
    background_strategy = contract.get("background_strategy")
    if (
        any(seconds <= 0 for seconds in expected_seconds)
        or len(expected_names) != len(expected_seconds)
        or any(type(value) is not int for value in cadence_seconds)
        or cadence_seconds != expected_seconds
        or not isinstance(captured_cadences, Mapping)
        or set(captured_cadences) != expected_names
        or background_strategy
        not in (
            _EXPECTATION_ONLY_BACKGROUND_STRATEGY,
            _LOCAL_DIAGNOSTIC_BACKGROUND_STRATEGY,
        )
        or photometry_policy.get("background_strategy") != background_strategy
    ):
        raise StampScienceAnalysisContractError(
            "analysis cadence/background strategy contract is inconsistent"
        )
    raw_interval_value = contract.get("raw_frame_interval")
    if not isinstance(raw_interval_value, Mapping) or set(raw_interval_value) != {
        "start_index",
        "stop_index_exclusive",
        "count",
    }:
        raise StampScienceAnalysisContractError(
            "analysis cadence raw-frame interval is invalid"
        )
    if any(
        type(raw_interval_value.get(name)) is not int
        for name in ("start_index", "stop_index_exclusive", "count")
    ):
        raise StampScienceAnalysisContractError(
            "analysis cadence raw-frame interval is invalid"
        )
    raw_interval = {
        name: raw_interval_value[name]
        for name in ("start_index", "stop_index_exclusive", "count")
    }
    if (
        not all(
            _is_native_int64_index(raw_interval[name])
            for name in ("start_index", "stop_index_exclusive", "count")
        )
        or raw_interval["stop_index_exclusive"] <= raw_interval["start_index"]
        or raw_interval["count"]
        != raw_interval["stop_index_exclusive"] - raw_interval["start_index"]
        or any(raw_interval["count"] % factor != 0 for factor in factors)
    ):
        raise StampScienceAnalysisContractError(
            "analysis cadence raw-frame interval is invalid"
        )
    return (
        factors,
        expected_seconds,
        raw_exposure,
        expected_names,
        raw_interval,
    )


def _recorded_content_identity_is_valid(value: Any) -> bool:
    return bool(
        isinstance(value, Mapping)
        and set(value) == {"size_bytes", "sha256"}
        and type(value.get("size_bytes")) is int
        and value["size_bytes"] >= 0
        and _is_lowercase_sha256(value.get("sha256"))
    )


def _semantic_descriptor_is_valid(
    value: Any,
    *,
    expected_shape: list[int],
) -> bool:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"dtype", "shape", "sha256"}
        or type(value.get("dtype")) is not str
        or value.get("shape") != expected_shape
        or not _is_lowercase_sha256(value.get("sha256"))
    ):
        return False
    try:
        return np.dtype(value["dtype"]).str == value["dtype"]
    except (TypeError, ValueError):
        return False


def _validate_writer_raw_shard_identities_v1(
    contract: Mapping[str, Any],
    raw_shards: list[Any],
) -> None:
    error_message = "writer raw-shard identity schema/frozen fields are invalid"
    context = contract.get("analysis_context")
    if (
        not isinstance(context, Mapping)
        or type(context.get("source_id")) is not str
        or not context["source_id"]
        or context.get("case") not in {"static", "injected"}
    ):
        raise StampScienceAnalysisContractError(error_message)

    frozen_identity: dict[str, Any] | None = None
    seen_paths: set[str] = set()
    for shard in raw_shards:
        if not isinstance(shard, Mapping) or set(shard) != _RAW_SHARD_IDENTITY_FIELDS:
            raise StampScienceAnalysisContractError(error_message)
        path = shard.get("path")
        file_stat = shard.get("file_stat")
        frame_count = shard.get("frame_count")
        stamp_shape = shard.get("stamp_shape")
        first_time = shard.get("first_time_start_seconds")
        last_time = shard.get("last_time_end_seconds")
        production_identity = shard.get("production_manifest_content_identity")
        if (
            not _is_canonical_absolute_path_text(path)
            or path in seen_paths
            or not isinstance(file_stat, Mapping)
            or set(file_stat) != {"device", "inode", "size_bytes", "mtime_ns"}
            or any(
                type(file_stat.get(name)) is not int or file_stat[name] < 0
                for name in ("device", "inode", "size_bytes", "mtime_ns")
            )
            or file_stat["size_bytes"] <= 0
            or shard.get("product_kind") != "raw"
            or type(shard.get("coadd_factor")) is not int
            or shard["coadd_factor"] != 1
            or type(frame_count) is not int
            or frame_count <= 0
            or not isinstance(stamp_shape, list)
            or len(stamp_shape) != 2
            or any(type(value) is not int or value <= 0 for value in stamp_shape)
            or shard.get("gain_mode") not in {"scalar", "stamp_map"}
            or any(
                not _is_lowercase_sha256(shard.get(name))
                for name in (
                    "manifest_identity_sha256",
                    "provenance_identity_sha256",
                    "series_manifest_identity_sha256",
                    "series_provenance_identity_sha256",
                    "semantic_sha256",
                )
            )
            or type(first_time) is not float
            or type(last_time) is not float
            or not math.isfinite(first_time)
            or not math.isfinite(last_time)
            or type(shard.get("target_source_id")) is not str
            or not shard["target_source_id"]
            or shard["target_source_id"] != context["source_id"]
            or shard.get("case") != context["case"]
            or type(shard.get("run_id")) is not str
            or not shard["run_id"]
            or type(shard.get("production_manifest_reference")) is not str
            or not shard["production_manifest_reference"]
            or shard["production_manifest_reference"].strip()
            != shard["production_manifest_reference"]
            or (
                production_identity is not None
                and not _recorded_content_identity_is_valid(production_identity)
            )
            or shard.get("identity_mode")
            != "canonical_semantic_planes_sha256_v1"
        ):
            raise StampScienceAnalysisContractError(error_message)
        seen_paths.add(path)

        datasets = shard.get("datasets")
        gain_identity = shard.get("gain_e_per_dn")
        if not isinstance(datasets, Mapping) or set(datasets) != set(
            _SEMANTIC_DATASET_NAMES
        ):
            raise StampScienceAnalysisContractError(error_message)
        cube_shape = [frame_count, *stamp_shape]
        vector_shape = [frame_count]
        column_shape = [frame_count, stamp_shape[1]]
        cube_names = {
            "final_dn",
            "background_expectation_e",
            "valid_mask",
            "fullwell_count",
            "adc_low_count",
            "adc_high_count",
            "cosmic_count",
            "saturated_mask",
            "cosmic_mask",
        }
        vector_names = {
            "captured_flux_fraction",
            "captured_flux_denominator_e",
            "captured_flux_qa_pass",
            "bias_level_sum_dn",
            "time_start_seconds",
            "exposure_seconds",
            "raw_frame_start_index",
            "raw_frame_stop_index_exclusive",
        }
        if any(
            not _semantic_descriptor_is_valid(
                datasets[name],
                expected_shape=(
                    cube_shape
                    if name in cube_names
                    else vector_shape
                    if name in vector_names
                    else column_shape
                ),
            )
            for name in _SEMANTIC_DATASET_NAMES
        ):
            raise StampScienceAnalysisContractError(error_message)
        gain_shape = [1] if shard["gain_mode"] == "scalar" else stamp_shape
        if not _semantic_descriptor_is_valid(
            gain_identity,
            expected_shape=gain_shape,
        ):
            raise StampScienceAnalysisContractError(error_message)
        recomputed_semantic = hashlib.sha256(
            json.dumps(
                {
                    "datasets": dict(datasets),
                    "gain_e_per_dn": dict(gain_identity),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        if recomputed_semantic != shard["semantic_sha256"]:
            raise StampScienceAnalysisContractError(error_message)

        byte_identity = shard.get("byte_identity")
        trust_scope = (
            byte_identity.get("trust_scope")
            if isinstance(byte_identity, Mapping)
            else None
        )
        common_byte_fields = {
            "trust_scope",
            "size_bytes",
            "sha256",
            "runtime_path",
            "runtime_hostname",
        }
        expected_byte_fields = (
            common_byte_fields | {"publication_receipt"}
            if trust_scope
            == "publisher_receipt_plus_stat_and_formal_header_v1"
            else common_byte_fields
        )
        if (
            not isinstance(byte_identity, Mapping)
            or trust_scope
            not in {
                "locally_computed_full_file_sha256_v1",
                "publisher_receipt_plus_stat_and_formal_header_v1",
            }
            or set(byte_identity) != expected_byte_fields
            or type(byte_identity.get("size_bytes")) is not int
            or byte_identity["size_bytes"] <= 0
            or byte_identity["size_bytes"] != file_stat["size_bytes"]
            or not _is_lowercase_sha256(byte_identity.get("sha256"))
            or byte_identity.get("runtime_path") != path
            or type(byte_identity.get("runtime_hostname")) is not str
            or not byte_identity["runtime_hostname"]
        ):
            raise StampScienceAnalysisContractError(error_message)
        if trust_scope == "publisher_receipt_plus_stat_and_formal_header_v1":
            receipt = byte_identity.get("publication_receipt")
            receipt_path = receipt.get("path") if isinstance(receipt, Mapping) else None
            if (
                not isinstance(receipt, Mapping)
                or set(receipt) != {"path", "identity"}
                or not _is_canonical_absolute_path_text(receipt_path)
                or Path(receipt_path).name != "publication_receipt.json"
                or Path(receipt_path).parent != Path(path).parent
                or not _recorded_content_identity_is_valid(receipt.get("identity"))
            ):
                raise StampScienceAnalysisContractError(error_message)

        current_frozen = {
            name: shard[name]
            for name in (
                "stamp_shape",
                "gain_mode",
                "series_manifest_identity_sha256",
                "series_provenance_identity_sha256",
                "target_source_id",
                "case",
                "run_id",
                "production_manifest_reference",
                "production_manifest_content_identity",
                "gain_e_per_dn",
            )
        }
        if frozen_identity is None:
            frozen_identity = current_frozen
        elif _canonical_json_text(
            current_frozen,
            name="raw-shard frozen identity",
        ) != _canonical_json_text(
            frozen_identity,
            name="first raw-shard frozen identity",
        ):
            raise StampScienceAnalysisContractError(error_message)

    if context.get("formal_profile_id") == STAMP_SCIENCE_FORMAL_PROFILE_ID:
        raw_bindings = context.get("raw_bundles")
        if (
            not isinstance(raw_bindings, list)
            or len(raw_bindings) != len(raw_shards)
        ):
            raise StampScienceAnalysisContractError(error_message)
        for binding, shard in zip(raw_bindings, raw_shards, strict=True):
            header_identity = (
                binding.get("identity") if isinstance(binding, Mapping) else None
            )
            recorded_header = {
                name: shard[name] for name in _INPUT_HEADER_IDENTITY_FIELDS
            }
            if (
                not isinstance(binding, Mapping)
                or set(binding) != {"path", "identity"}
                or binding.get("path") != shard["path"]
                or not isinstance(header_identity, Mapping)
                or set(header_identity) != _INPUT_HEADER_IDENTITY_FIELDS
                or _canonical_json_text(
                    header_identity,
                    name="request raw-bundle header identity",
                )
                != _canonical_json_text(
                    recorded_header,
                    name="published raw-shard header identity",
                )
            ):
                raise StampScienceAnalysisContractError(error_message)


@dataclass(frozen=True)
class _RawCadenceAnchor:
    first_time_start_seconds: float


def _raw_cadence_anchor_contract_v1(
    contract: Mapping[str, Any],
    *,
    raw_frame_interval: Mapping[str, int],
    raw_exposure_seconds: float,
) -> _RawCadenceAnchor:
    raw_shards = contract.get("input_raw_shards")
    if not isinstance(raw_shards, list) or not raw_shards:
        raise StampScienceAnalysisContractError(
            "authoritative analysis semantics raw-shard coverage is invalid "
            "for representative calibrated-frame provenance"
        )
    _validate_writer_raw_shard_identities_v1(contract, raw_shards)
    expected_start = raw_frame_interval["start_index"]
    expected_time: float | None = None
    first_time: float | None = None
    for shard in raw_shards:
        if not isinstance(shard, Mapping):
            raise StampScienceAnalysisContractError(
                "authoritative analysis semantics raw-shard coverage is invalid "
                "for representative calibrated-frame provenance"
            )
        start = shard.get("first_raw_frame_start")
        stop = shard.get("last_raw_frame_stop")
        frame_count = shard.get("frame_count")
        time_start = shard.get("first_time_start_seconds")
        time_stop = shard.get("last_time_end_seconds")
        if (
            not _is_native_int64_index(start)
            or not _is_native_int64_index(stop)
            or not _is_native_int64_index(frame_count)
            or frame_count <= 0
            or stop <= start
            or stop - start != frame_count
            or type(time_start) not in {int, float}
            or type(time_stop) not in {int, float}
            or not math.isfinite(float(time_start))
            or not math.isfinite(float(time_stop))
            or start != expected_start
            or (
                expected_time is not None
                and not math.isclose(
                    float(time_start),
                    expected_time,
                    rel_tol=0.0,
                    abs_tol=_TIME_ATOL_SECONDS,
                )
            )
            or not math.isclose(
                float(time_stop) - float(time_start),
                frame_count * raw_exposure_seconds,
                rel_tol=0.0,
                abs_tol=_TIME_ATOL_SECONDS,
            )
        ):
            raise StampScienceAnalysisContractError(
                "authoritative analysis semantics raw-shard coverage is invalid "
                "for representative calibrated-frame provenance"
            )
        if first_time is None:
            first_time = float(time_start)
        expected_start = stop
        expected_time = float(time_stop)
    if (
        expected_start != raw_frame_interval["stop_index_exclusive"]
        or first_time is None
    ):
        raise StampScienceAnalysisContractError(
            "authoritative analysis semantics raw-shard coverage is invalid "
            "for representative calibrated-frame provenance"
        )
    return _RawCadenceAnchor(first_time_start_seconds=first_time)


def _publication_analysis_policy_v1(
    contract: Mapping[str, Any],
) -> StampScienceAnalysisPolicy:
    policy_value = contract.get("policy")
    try:
        policy = _analysis_policy_from_json(policy_value)
        encoded = _canonical_json_text(
            policy_value,
            name="authoritative analysis semantics policy",
        )
        roundtrip = _canonical_json_text(
            policy.to_dict(),
            name="authoritative analysis semantics policy round trip",
        )
    except (SciencePhotometryContractError, TypeError, ValueError) as error:
        raise StampScienceAnalysisContractError(
            "authoritative analysis semantics policy is invalid"
        ) from error
    if encoded != roundtrip:
        raise StampScienceAnalysisContractError(
            "authoritative analysis semantics policy is not type-strict"
        )
    context = contract.get("analysis_context")
    if (
        isinstance(context, Mapping)
        and context.get("formal_profile_id") == STAMP_SCIENCE_FORMAL_PROFILE_ID
        and encoded
        != _canonical_json_text(
            _formal_analysis_policy_v1().to_dict(),
            name="frozen formal analysis policy",
        )
    ):
        raise StampScienceAnalysisContractError(
            "authoritative formal analysis policy differs from the frozen profile"
        )
    return policy


def _validate_publication_direct_parity_v1(
    contract: Mapping[str, Any],
    *,
    policy: StampScienceAnalysisPolicy,
    raw_frame_interval: Mapping[str, int],
) -> None:
    parity = contract.get("direct_coadd_parity")
    samples = parity.get("samples") if isinstance(parity, Mapping) else None
    if (
        not isinstance(parity, Mapping)
        or set(parity) != _DIRECT_COADD_PARITY_FIELDS
        or type(parity.get("required")) is not bool
        or parity.get("passed") is not True
        or parity.get("comparison") != _DIRECT_COADD_PARITY_COMPARISON
        or type(parity.get("sample_count")) is not int
        or not isinstance(samples, list)
        or parity["sample_count"] != len(samples)
        or parity["required"] is not bool(samples)
        or (policy.require_direct_coadd_parity and not samples)
    ):
        raise StampScienceAnalysisContractError(
            "analysis direct-coadd parity contract is invalid"
        )

    direct_shards = contract.get("input_direct_coadd_shards")
    if not isinstance(direct_shards, Mapping):
        raise StampScienceAnalysisContractError(
            "analysis direct-coadd parity contract is invalid"
        )
    configured_factors = set(policy.coadd_factors) - {1}
    parsed_shards: dict[int, list[Any]] = {}
    for factor_text, records in direct_shards.items():
        if (
            not isinstance(factor_text, str)
            or not factor_text.isdecimal()
            or str(int(factor_text)) != factor_text
            or not isinstance(records, list)
            or not records
        ):
            raise StampScienceAnalysisContractError(
                "analysis direct-coadd parity contract is invalid"
            )
        factor = int(factor_text)
        if factor not in configured_factors or factor in parsed_shards:
            raise StampScienceAnalysisContractError(
                "analysis direct-coadd parity contract is invalid"
            )
        parsed_shards[factor] = records

    actual_factors = set(parsed_shards)
    if (
        policy.require_direct_coadd_parity
        and actual_factors != configured_factors
    ) or not actual_factors.issubset(configured_factors):
        raise StampScienceAnalysisContractError(
            "analysis direct-coadd parity contract is invalid"
        )

    expected_samples: list[dict[str, Any]] = []
    bound_paths: set[tuple[int, str]] = set()
    for factor, records in sorted(parsed_shards.items()):
        expected_start = raw_frame_interval["start_index"]
        for record in records:
            path = record.get("path") if isinstance(record, Mapping) else None
            frame_count = (
                record.get("frame_count") if isinstance(record, Mapping) else None
            )
            first_start = (
                record.get("first_raw_frame_start")
                if isinstance(record, Mapping)
                else None
            )
            last_stop = (
                record.get("last_raw_frame_stop")
                if isinstance(record, Mapping)
                else None
            )
            if (
                not isinstance(record, Mapping)
                or type(record.get("coadd_factor")) is not int
                or record["coadd_factor"] != factor
                or not isinstance(path, str)
                or not _is_canonical_absolute_path_text(path)
                or (factor, path) in bound_paths
                or type(frame_count) is not int
                or not _is_native_int64_index(frame_count)
                or frame_count <= 0
                or type(first_start) is not int
                or not _is_native_int64_index(first_start)
                or type(last_stop) is not int
                or not _is_native_int64_index(last_stop)
                or first_start != expected_start
                or last_stop <= first_start
                or last_stop - first_start != frame_count * factor
                or last_stop > raw_frame_interval["stop_index_exclusive"]
            ):
                raise StampScienceAnalysisContractError(
                    "analysis direct-coadd parity contract is invalid"
                )
            bound_paths.add((factor, path))
            expected_samples.extend(
                {
                    "factor": factor,
                    "path": path,
                    "local_frame_index": local_index,
                    "raw_frame_start_index": first_start + local_index * factor,
                }
                for local_index in _direct_sample_local_indices(
                    frame_count,
                    samples_per_shard=policy.direct_coadd_samples_per_shard,
                )
            )
            expected_start = last_stop
        if expected_start != raw_frame_interval["stop_index_exclusive"]:
            raise StampScienceAnalysisContractError(
                "analysis direct-coadd parity contract is invalid"
            )

    if _canonical_json_text(
        samples,
        name="recorded direct-coadd parity samples",
    ) != _canonical_json_text(
        expected_samples,
        name="recomputed direct-coadd parity samples",
    ):
        raise StampScienceAnalysisContractError(
            "analysis direct-coadd parity samples differ from the deterministic "
            "factor/shard sample set"
        )


def _validate_publication_code_provenance_v1(
    contract: Mapping[str, Any],
) -> None:
    context = contract.get("analysis_context")
    if (
        not isinstance(context, Mapping)
        or context.get("formal_profile_id") != STAMP_SCIENCE_FORMAL_PROFILE_ID
    ):
        return

    request_identity = contract.get("request_code_identity")
    execution_identity = contract.get("execution_code_identity")
    code_identity = contract.get("code_identity")
    base_fields = {
        "schema_id",
        "schema_version",
        "provenance",
        "analysis_dependencies",
    }
    if (
        not isinstance(request_identity, Mapping)
        or not isinstance(execution_identity, Mapping)
        or not isinstance(code_identity, Mapping)
        or set(request_identity) != base_fields
        or set(execution_identity) != base_fields | {"execution_hardware"}
        or _canonical_json_text(
            code_identity,
            name="analysis code identity",
        )
        != _canonical_json_text(
            execution_identity,
            name="analysis execution code identity",
        )
    ):
        raise StampScienceAnalysisContractError(
            "formal analysis code provenance contract is invalid"
        )
    hardware = execution_identity.get("execution_hardware")
    hardware_fields = {
        "schema_id",
        "analysis_compute_device",
        "cpu_count",
        "machine",
        "cuda_available",
        "cuda_device_names",
    }
    cpu_count = hardware.get("cpu_count") if isinstance(hardware, Mapping) else None
    cuda_names = (
        hardware.get("cuda_device_names")
        if isinstance(hardware, Mapping)
        else None
    )
    if (
        not isinstance(hardware, Mapping)
        or set(hardware) != hardware_fields
        or hardware.get("schema_id")
        != "et_mainsim.analysis_execution_hardware.v1"
        or hardware.get("analysis_compute_device") != "cpu"
        or (
            cpu_count is not None
            and (type(cpu_count) is not int or cpu_count <= 0)
        )
        or not isinstance(hardware.get("machine"), str)
        or type(hardware.get("cuda_available")) is not bool
        or not isinstance(cuda_names, list)
        or any(not isinstance(name, str) or not name for name in cuda_names)
        or (not hardware["cuda_available"] and cuda_names)
    ):
        raise StampScienceAnalysisContractError(
            "formal analysis code provenance execution hardware is invalid"
        )
    execution_base = {
        name: execution_identity[name] for name in sorted(base_fields)
    }
    current_identity = collect_formal_analysis_code_identity_v1()
    if (
        not _formal_code_identity_matches_execution_v1(
            request_identity,
            execution_base,
        )
        or not _formal_code_identity_matches_execution_v1(
            execution_base,
            current_identity,
        )
    ):
        raise StampScienceAnalysisContractError(
            "formal analysis code provenance differs from request/current execution"
        )


def _analysis_noise_parameters_v1(
    contract: Mapping[str, Any],
) -> tuple[float, float]:
    context = contract.get("analysis_context")
    noise_model = (
        context.get("noise_model") if isinstance(context, Mapping) else None
    )
    if (
        not isinstance(noise_model, Mapping)
        or set(noise_model) != _NOISE_MODEL_FIELDS
        or noise_model.get("schema_id") != _FORMAL_NOISE_MODEL_SCHEMA_ID
        or not isinstance(noise_model.get("source"), str)
        or not noise_model["source"]
        or not isinstance(noise_model.get("quantization_formula"), str)
        or not noise_model["quantization_formula"]
    ):
        raise StampScienceAnalysisContractError(
            "authoritative analysis semantics noise model is invalid"
        )
    values = tuple(
        noise_model.get(name)
        for name in (
            "read_noise_e_per_raw_pixel",
            "quantization_noise_e_per_raw_pixel",
        )
    )
    if any(
        type(value) is not float
        or not math.isfinite(value)
        or value < 0.0
        for value in values
    ):
        raise StampScienceAnalysisContractError(
            "authoritative analysis semantics noise model is invalid"
        )
    if (
        isinstance(context, Mapping)
        and context.get("formal_profile_id") == STAMP_SCIENCE_FORMAL_PROFILE_ID
        and (
            noise_model["source"] != _FORMAL_NOISE_MODEL_SOURCE
            or noise_model["quantization_formula"]
            != _FORMAL_NOISE_MODEL_QUANTIZATION_FORMULA
        )
    ):
        raise StampScienceAnalysisContractError(
            "authoritative formal analysis noise model differs from the frozen profile"
        )
    return values[0], values[1]


def _arrays_match_exactly(actual: Any, expected: Any) -> bool:
    try:
        actual_array = np.asarray(actual)
        expected_array = np.asarray(expected)
        return bool(
            actual_array.dtype == expected_array.dtype
            and np.array_equal(
                actual_array,
                expected_array,
                equal_nan=True,
            )
        )
    except (TypeError, ValueError):
        return False


def _numeric_scalar_matches_exactly(actual: Any, expected: float) -> bool:
    if isinstance(actual, (bool, np.bool_)) or not isinstance(
        actual,
        (int, float, np.integer, np.floating),
    ):
        return False
    resolved = float(actual)
    return bool(
        resolved == expected
        or (math.isnan(resolved) and math.isnan(expected))
    )


def _validate_recomputed_cadence_semantics_v1(
    group: Any,
    *,
    cadence_name: str,
    factor: int,
    time_seconds: NDArray[np.float64],
    exposure_seconds: NDArray[np.float64],
    raw_frame_start_index: NDArray[np.int64],
    raw_frame_stop_index_exclusive: NDArray[np.int64],
    raw_frame_interval: Mapping[str, int],
    raw_relative_flux: NDArray[np.float64],
    photometry_policy: StampSciencePhotometryPolicy,
    aperture_mask: NDArray[Any],
    read_noise_e_per_raw_pixel: float,
    quantization_noise_e_per_raw_pixel: float,
    authoritative_cdpp: Mapping[str, Any],
) -> None:
    local_start = (
        raw_frame_start_index - raw_frame_interval["start_index"]
    )
    local_stop = (
        raw_frame_stop_index_exclusive - raw_frame_interval["start_index"]
    )
    aperture_valid = np.asarray(group["aperture_valid"])
    expectation_flux = np.asarray(
        group["flux_expectation_bgsub_e"],
        dtype=np.float64,
    )
    local_flux = np.asarray(group["flux_local_bgsub_e"], dtype=np.float64)
    local_valid = np.asarray(aperture_valid, dtype=bool) & np.isfinite(
        local_flux
    )
    try:
        expectation_model = fit_science_variability_model_v1(
            flux_e=expectation_flux,
            aperture_valid=aperture_valid,
            raw_relative_flux=raw_relative_flux,
            raw_frame_start_index=local_start,
            raw_frame_stop_index_exclusive=local_stop,
        )
        if photometry_policy.local_background_enabled:
            local_model = fit_science_variability_model_v1(
                flux_e=local_flux,
                aperture_valid=local_valid,
                raw_relative_flux=raw_relative_flux,
                raw_frame_start_index=local_start,
                raw_frame_stop_index_exclusive=local_stop,
            )
        else:
            local_model = _unavailable_local_variability_model(
                expectation_model
            )
        uncertainty = compute_science_flux_uncertainty_model_v1(
            fitted_source_expectation_e=expectation_model.fitted_flux_e,
            aperture_mask=aperture_mask,
            read_noise_e_per_raw_pixel=read_noise_e_per_raw_pixel,
            quantization_noise_e_per_raw_pixel=(
                quantization_noise_e_per_raw_pixel
            ),
            coadd_factor=factor,
            cadence_valid=expectation_model.valid_mask,
            background_expectation_aperture_e=np.asarray(
                group["background_expectation_aperture_e"],
                dtype=np.float64,
            ),
        )
        expectation_cdpp = compute_science_cdpp_v1(
            time_seconds=time_seconds,
            exposure_seconds=exposure_seconds,
            flux_e=expectation_flux,
            aperture_valid=aperture_valid,
            model_flux_e=expectation_model.fitted_flux_e,
            residual_e=expectation_model.residual_e,
            windows_minutes=photometry_policy.cdpp_windows_minutes,
            minimum_coverage_fraction=(
                photometry_policy.minimum_coverage_fraction
            ),
            minimum_accepted_bins=photometry_policy.minimum_accepted_bins,
            bin_origin_seconds=photometry_policy.bin_origin_seconds,
        )
        local_cdpp = None
        if photometry_policy.local_background_enabled:
            local_cdpp = compute_science_cdpp_v1(
                time_seconds=time_seconds,
                exposure_seconds=exposure_seconds,
                flux_e=local_flux,
                aperture_valid=local_valid,
                model_flux_e=local_model.fitted_flux_e,
                residual_e=local_model.residual_e,
                windows_minutes=photometry_policy.cdpp_windows_minutes,
                minimum_coverage_fraction=(
                    photometry_policy.minimum_coverage_fraction
                ),
                minimum_accepted_bins=(
                    photometry_policy.minimum_accepted_bins
                ),
                bin_origin_seconds=photometry_policy.bin_origin_seconds,
            )
    except (SciencePhotometryContractError, TypeError, ValueError) as error:
        raise StampScienceAnalysisContractError(
            f"authoritative analysis semantics for cadence {cadence_name} "
            "cannot be recomputed"
        ) from error

    expected_arrays = {
        "raw_factor_sum": expectation_model.raw_factor_sum,
        "fitted_flux_expectation_e": expectation_model.fitted_flux_e,
        "residual_expectation_e": expectation_model.residual_e,
        "residual_expectation_ppm": expectation_model.residual_ppm,
        "fitted_flux_local_e": local_model.fitted_flux_e,
        "residual_local_e": local_model.residual_e,
        "residual_local_ppm": local_model.residual_ppm,
        "local_model_valid": local_model.valid_mask,
        "flux_uncertainty_e": uncertainty.uncertainty_e,
        "source_variance_e2": uncertainty.source_variance_e2,
        "background_variance_e2": uncertainty.background_variance_e2,
        "read_variance_e2": uncertainty.read_variance_e2,
        "quantization_variance_e2": uncertainty.quantization_variance_e2,
        "uncertainty_valid": uncertainty.valid_mask,
        "uncertainty_coadd_factor": uncertainty.coadd_factor,
        "flux_expectation_bgsub_e_per_s": expectation_flux
        / exposure_seconds,
        "flux_local_bgsub_e_per_s": local_flux / exposure_seconds,
        "fitted_flux_expectation_e_per_s": expectation_model.fitted_flux_e
        / exposure_seconds,
        "fitted_flux_local_e_per_s": local_model.fitted_flux_e
        / exposure_seconds,
        "model_flux_uncertainty_e": uncertainty.uncertainty_e,
        "model_flux_uncertainty_e_per_s": uncertainty.uncertainty_e
        / exposure_seconds,
    }
    if any(
        not _arrays_match_exactly(group[name], expected)
        for name, expected in expected_arrays.items()
    ):
        raise StampScienceAnalysisContractError(
            f"authoritative analysis semantics for cadence {cadence_name} "
            "differ from recomputed model/uncertainty values"
        )
    if (
        not _arrays_match_exactly(
            local_model.raw_factor_sum,
            expectation_model.raw_factor_sum,
        )
        or not _numeric_scalar_matches_exactly(
            group.attrs.get("fit_scale_expectation_e_per_raw_factor"),
            expectation_model.fit_scale_e_per_raw_factor,
        )
        or not _numeric_scalar_matches_exactly(
            group.attrs.get("fit_scale_local_e_per_raw_factor"),
            local_model.fit_scale_e_per_raw_factor,
        )
        or not _numeric_scalar_matches_exactly(
            group.attrs.get("fit_intercept_e"),
            expectation_model.fit_intercept_e,
        )
    ):
        raise StampScienceAnalysisContractError(
            f"authoritative analysis semantics for cadence {cadence_name} "
            "fit attributes differ"
        )
    try:
        uncertainty_metadata = _decode_hdf_json(
            group.attrs.get("uncertainty_model_json"),
            name=f"cadence {cadence_name} uncertainty_model_json",
        )
    except StampScienceAnalysisContractError as error:
        raise StampScienceAnalysisContractError(
            f"authoritative analysis semantics for cadence {cadence_name} "
            "uncertainty metadata is invalid"
        ) from error
    if _canonical_json_text(
        uncertainty_metadata,
        name=f"cadence {cadence_name} uncertainty metadata",
    ) != _canonical_json_text(
        _json_safe(uncertainty.metadata),
        name=f"cadence {cadence_name} recomputed uncertainty metadata",
    ):
        raise StampScienceAnalysisContractError(
            f"authoritative analysis semantics for cadence {cadence_name} "
            "uncertainty metadata differs"
        )
    expected_cdpp = _json_safe(
        {
            "coadd_factor": factor,
            "expectation_background": _cdpp_payload(expectation_cdpp),
            "local_background": _cdpp_payload(local_cdpp),
        }
    )
    cadence_payload = authoritative_cdpp.get("cadences")
    actual_cdpp = (
        cadence_payload.get(cadence_name)
        if isinstance(cadence_payload, Mapping)
        else None
    )
    if _canonical_json_text(
        actual_cdpp,
        name=f"authoritative analysis semantics cadence {cadence_name} CDPP",
    ) != _canonical_json_text(
        expected_cdpp,
        name=(
            f"authoritative analysis semantics recomputed cadence "
            f"{cadence_name} CDPP"
        ),
    ):
        raise StampScienceAnalysisContractError(
            f"authoritative analysis semantics for cadence {cadence_name} "
            "CDPP background role/value differs from recomputation"
        )


@dataclass(frozen=True)
class _RepresentativeFrameProvenance:
    selection_roles: tuple[str, ...]
    raw_frame_start_indices: tuple[int, ...]
    input_shard_paths: tuple[str, ...]
    input_shard_semantic_sha256: tuple[str, ...]


def _is_lowercase_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_canonical_absolute_path_text(value: Any) -> bool:
    if type(value) is not str or not value or value.strip() != value:
        return False
    path = Path(value)
    return bool(
        path.is_absolute()
        and path.name
        and os.path.normpath(value) == value
        and "." not in path.parts
        and ".." not in path.parts
    )


def _is_native_int64_index(value: Any) -> bool:
    return type(value) is int and 0 <= value <= _INT64_MAX


def _representative_frame_provenance_contract_v1(
    contract: Mapping[str, Any],
    *,
    raw_frame_interval: Mapping[str, int],
) -> _RepresentativeFrameProvenance:
    representative = contract.get("representative_calibrated_frames")
    raw_shards = contract.get("input_raw_shards")
    if (
        not isinstance(representative, Mapping)
        or set(representative)
        != {"artifact", "selection_policy", "clean_definition", "frames"}
        or representative.get("artifact")
        != "representative_calibrated_frames.h5"
        or representative.get("selection_policy")
        != _REPRESENTATIVE_FRAME_SELECTION_POLICY
        or representative.get("clean_definition")
        != _REPRESENTATIVE_FRAME_CLEAN_DEFINITION
        or not isinstance(representative.get("frames"), list)
        or len(representative["frames"])
        != len(_REPRESENTATIVE_FRAME_SELECTION_ROLES)
        or not isinstance(raw_shards, list)
        or not raw_shards
    ):
        raise StampScienceAnalysisContractError(
            "representative calibrated-frame provenance contract is invalid"
        )

    raw_interval_start = raw_frame_interval["start_index"]
    raw_interval_stop = raw_frame_interval["stop_index_exclusive"]
    raw_shard_bindings: list[tuple[str, str, int, int]] = []
    raw_shard_paths: set[str] = set()
    for shard in raw_shards:
        if not isinstance(shard, Mapping):
            raise StampScienceAnalysisContractError(
                "representative calibrated-frame provenance differs from bound "
                "raw semantic identities/policy"
            )
        shard_path = shard.get("path")
        shard_semantic_hash = shard.get("semantic_sha256")
        shard_start = shard.get("first_raw_frame_start")
        shard_stop = shard.get("last_raw_frame_stop")
        if (
            not _is_canonical_absolute_path_text(shard_path)
            or shard_path in raw_shard_paths
            or not _is_lowercase_sha256(shard_semantic_hash)
            or not _is_native_int64_index(shard_start)
            or not _is_native_int64_index(shard_stop)
            or shard_stop <= shard_start
            or not raw_interval_start <= shard_start
            or shard_stop > raw_interval_stop
        ):
            raise StampScienceAnalysisContractError(
                "representative calibrated-frame provenance differs from bound "
                "raw semantic identities/policy"
            )
        raw_shard_paths.add(shard_path)
        raw_shard_bindings.append(
            (shard_path, shard_semantic_hash, shard_start, shard_stop)
        )

    starts: list[int] = []
    paths: list[str] = []
    semantic_hashes: list[str] = []
    for expected_role, frame in zip(
        _REPRESENTATIVE_FRAME_SELECTION_ROLES,
        representative["frames"],
        strict=True,
    ):
        if not isinstance(frame, Mapping) or set(frame) != (
            _REPRESENTATIVE_FRAME_CONTRACT_FIELDS
        ):
            raise StampScienceAnalysisContractError(
                "representative calibrated-frame provenance contract is invalid"
            )
        role = frame.get("selection_role")
        raw_start = frame.get("raw_frame_start_index")
        shard_index = frame.get("input_shard_index")
        path = frame.get("input_shard_path")
        semantic_hash = frame.get("input_shard_semantic_sha256")
        if (
            type(role) is not str
            or role != expected_role
            or not _is_native_int64_index(raw_start)
            or not _is_native_int64_index(shard_index)
            or shard_index >= len(raw_shards)
            or not _is_canonical_absolute_path_text(path)
            or not _is_lowercase_sha256(semantic_hash)
        ):
            raise StampScienceAnalysisContractError(
                "representative calibrated-frame provenance contract is invalid"
            )
        shard_path, shard_semantic_hash, shard_start, shard_stop = (
            raw_shard_bindings[shard_index]
        )
        if (
            not shard_start <= raw_start < shard_stop
            or path != shard_path
            or semantic_hash != shard_semantic_hash
        ):
            raise StampScienceAnalysisContractError(
                "representative calibrated-frame provenance differs from bound "
                "raw semantic identities/policy"
            )
        starts.append(raw_start)
        paths.append(path)
        semantic_hashes.append(semantic_hash)
    return _RepresentativeFrameProvenance(
        selection_roles=_REPRESENTATIVE_FRAME_SELECTION_ROLES,
        raw_frame_start_indices=tuple(starts),
        input_shard_paths=tuple(paths),
        input_shard_semantic_sha256=tuple(semantic_hashes),
    )


def _validate_analysis_dir(path: Path) -> StampScienceAnalysisValidation:
    import h5py

    input_root = path.expanduser()
    if input_root.is_symlink():
        raise StampScienceAnalysisContractError(
            "science analysis publication must not be a symbolic link"
        )
    root = input_root.resolve()
    if not root.is_dir():
        raise StampScienceAnalysisContractError(
            "science analysis publication must be a real directory"
        )
    manifest_path = root / "analysis_manifest.json"
    manifest = _read_json_object(manifest_path, label="analysis manifest")
    if (
        manifest.get("schema_id")
        != STAMP_SCIENCE_ANALYSIS_PUBLICATION_SCHEMA_ID
        or not _is_native_schema_version(
            manifest.get("schema_version"),
            STAMP_SCIENCE_ANALYSIS_PUBLICATION_SCHEMA_VERSION,
        )
        or manifest.get("complete") is not True
        or manifest.get("ready") is not True
        or manifest.get("authoritative_product") != "photometry.h5"
    ):
        raise StampScienceAnalysisContractError(
            "analysis manifest schema/completeness is invalid"
        )
    contract = manifest.get("contract")
    artifacts = manifest.get("artifacts")
    if not isinstance(contract, dict) or not isinstance(artifacts, dict):
        raise StampScienceAnalysisContractError(
            "analysis manifest lacks contract/artifact identities"
        )
    analysis_product = contract.get("analysis_product")
    if analysis_product not in {
        "reference_fixed13_v1",
        "science_optimal_aperture_v1",
    }:
        raise StampScienceAnalysisContractError(
            "analysis contract analysis_product is invalid"
        )
    expected_artifact_names = set(_analysis_artifact_names_v1(analysis_product))
    if set(artifacts) != expected_artifact_names:
        raise StampScienceAnalysisContractError(
            "analysis artifact set differs from the product contract"
        )
    captured_contract = contract.get("captured_flux_qa")
    captured_cadences = (
        captured_contract.get("cadences")
        if isinstance(captured_contract, Mapping)
        else None
    )
    if (
        contract.get("schema_id") != STAMP_SCIENCE_ANALYSIS_SCHEMA_ID
        or not _is_native_schema_version(
            contract.get("schema_version"),
            STAMP_SCIENCE_ANALYSIS_SCHEMA_VERSION,
        )
        or contract.get("science_photometry_schema_id")
        != SCIENCE_PHOTOMETRY_SCHEMA_ID
        or not _is_native_schema_version(
            contract.get("science_photometry_schema_version"),
            SCIENCE_PHOTOMETRY_SCHEMA_VERSION,
        )
        or not isinstance(captured_cadences, Mapping)
        or not captured_cadences
        or any(
            not isinstance(record, Mapping) or record.get("all_pass") is not True
            for record in captured_cadences.values()
        )
    ):
        raise StampScienceAnalysisContractError(
            "analysis contract schema or captured-flux gate is invalid"
        )
    (
        expected_factors,
        expected_cadence_seconds,
        raw_exposure_seconds,
        expected_cadence_names,
        raw_frame_interval,
    ) = _publication_cadence_contract_v1(contract)
    expected_background_products = (
        [
            "expectation_background_subtracted",
            "local_background_diagnostic",
        ]
        if contract.get("background_strategy")
        == _LOCAL_DIAGNOSTIC_BACKGROUND_STRATEGY
        else ["expectation_background_subtracted"]
    )
    if (
        contract.get("complete") is not True
        or contract.get("observation_product") != "final_dn"
        or contract.get("calibrated_electron_products_are_derived") is not True
        or contract.get("background_realization_used") is not False
        or contract.get("background_products") != expected_background_products
        or contract.get("default_background_product")
        != "background_expectation_e"
    ):
        raise StampScienceAnalysisContractError(
            "analysis contract observation/background semantics are invalid"
        )
    recorded_uncertainty_contract = contract.get("flux_uncertainty_model")
    if (
        not isinstance(recorded_uncertainty_contract, Mapping)
        or _canonical_json_text(
            recorded_uncertainty_contract,
            name="analysis contract flux uncertainty model",
        )
        != _canonical_json_text(
            _flux_uncertainty_model_contract_v1(),
            name="frozen flux uncertainty model",
        )
    ):
        raise StampScienceAnalysisContractError(
            "analysis contract flux uncertainty model is invalid"
        )
    analysis_policy = _publication_analysis_policy_v1(contract)
    raw_cadence_anchor = _raw_cadence_anchor_contract_v1(
        contract,
        raw_frame_interval=raw_frame_interval,
        raw_exposure_seconds=raw_exposure_seconds,
    )
    _validate_publication_direct_parity_v1(
        contract,
        policy=analysis_policy,
        raw_frame_interval=raw_frame_interval,
    )
    _validate_publication_code_provenance_v1(contract)
    (
        read_noise_e_per_raw_pixel,
        quantization_noise_e_per_raw_pixel,
    ) = _analysis_noise_parameters_v1(contract)
    representative_provenance = _representative_frame_provenance_contract_v1(
        contract,
        raw_frame_interval=raw_frame_interval,
    )
    for name, expected in artifacts.items():
        relative = Path(name) if isinstance(name, str) else Path()
        if (
            not isinstance(name, str)
            or not name
            or relative.is_absolute()
            or ".." in relative.parts
            or len(relative.parts) > 2
            or (len(relative.parts) == 2 and relative.parts[0] != "figures")
            or not isinstance(expected, dict)
        ):
            raise StampScienceAnalysisContractError(
                "analysis artifact identity is invalid"
            )
        artifact = root / relative
        if (
            not artifact.is_file()
            or artifact.is_symlink()
            or (artifact.parent != root and artifact.parent.is_symlink())
        ):
            raise StampScienceAnalysisContractError(
                f"analysis artifact is missing or unsafe: {name}"
            )
        if _file_identity(artifact) != expected:
            raise StampScienceAnalysisContractError(
                f"analysis artifact hash/readback mismatch: {name}"
            )
    hdf_path = root / "photometry.h5"
    science_training_indices: NDArray[Any] | None = None
    signal_template: NDArray[np.float64] | None = None
    noise_template: NDArray[np.float64] | None = None
    authoritative_cdpp: dict[str, Any]
    representative_time_start_seconds: tuple[float, ...] | None = None
    representative_exposure_seconds: tuple[float, ...] | None = None
    representative_factor_one: dict[str, NDArray[Any]] | None = None
    expected_quality_cadences: dict[str, Any] = {}
    expected_captured_records: dict[str, tuple[bool, float]] = {}
    with h5py.File(hdf_path, "r") as handle:
        schema_id = handle.attrs.get("schema_id")
        if isinstance(schema_id, bytes):
            schema_id = schema_id.decode("utf-8")
        if (
            schema_id != STAMP_SCIENCE_ANALYSIS_SCHEMA_ID
            or not _is_hdf_integer_schema_version(
                handle.attrs.get("schema_version"),
                STAMP_SCIENCE_ANALYSIS_SCHEMA_VERSION,
            )
            or not _is_hdf_boolean(handle.attrs.get("complete"), True)
            or handle.attrs.get("observation_product") != "final_dn"
            or not _is_hdf_boolean(
                handle.attrs.get("background_realization_used"),
                False,
            )
        ):
            raise StampScienceAnalysisContractError(
                "authoritative HDF5 schema/completeness is invalid"
            )
        hdf_contract = _decode_hdf_json(
            handle["analysis_contract_json"][()],
            name="analysis_contract_json",
        )
        if _canonical_json_text(
            hdf_contract,
            name="authoritative HDF5 analysis contract",
        ) != _canonical_json_text(
            contract,
            name="publication manifest analysis contract",
        ):
            raise StampScienceAnalysisContractError(
                "HDF5 and publication manifest contracts differ"
            )
        if (
            "raw_relative_flux" not in handle
            or "cadences" not in handle
            or "cdpp_json" not in handle
        ):
            raise StampScienceAnalysisContractError(
                "authoritative HDF5 lacks q/cadence/CDPP products"
            )
        authoritative_cdpp = _decode_hdf_json(
            handle["cdpp_json"][()],
            name="authoritative HDF5 cdpp_json",
        )
        cdpp_cadences = authoritative_cdpp.get("cadences")
        if (
            not isinstance(cdpp_cadences, Mapping)
            or set(cdpp_cadences) != expected_cadence_names
        ):
            raise StampScienceAnalysisContractError(
                "authoritative HDF5 CDPP cadence set differs from contract"
            )
        q = np.asarray(handle["raw_relative_flux"], dtype=np.float64)
        if (
            q.ndim != 1
            or q.size != raw_frame_interval["count"]
            or not np.all(np.isfinite(q))
            or np.any(q <= 0.0)
        ):
            raise StampScienceAnalysisContractError(
                "authoritative HDF5 raw_relative_flux is invalid"
            )
        recorded_q_identity = contract.get("raw_relative_flux")
        source_identity = (
            recorded_q_identity.get("source_identity")
            if isinstance(recorded_q_identity, Mapping)
            else None
        )
        expected_q_identity = _q_identity(q)
        if isinstance(source_identity, Mapping):
            expected_q_identity["source_identity"] = dict(source_identity)
        if (
            not isinstance(recorded_q_identity, Mapping)
            or not isinstance(source_identity, Mapping)
            or _canonical_json_text(
                recorded_q_identity,
                name="recorded raw_relative_flux identity",
            )
            != _canonical_json_text(
                expected_q_identity,
                name="recomputed raw_relative_flux identity",
            )
        ):
            raise StampScienceAnalysisContractError(
                "authoritative HDF5 raw_relative_flux identity differs from contract"
            )
        cadence_seconds: list[int] = []
        raw_frame_count = 0
        cadence_group = handle.get("cadences")
        semantic_aperture_group = handle.get("aperture")
        if (
            not isinstance(cadence_group, h5py.Group)
            or set(cadence_group) != expected_cadence_names
        ):
            raise StampScienceAnalysisContractError(
                "authoritative HDF5 cadence group set differs from contract"
            )
        if (
            not isinstance(semantic_aperture_group, h5py.Group)
            or "aperture_mask" not in semantic_aperture_group
        ):
            raise StampScienceAnalysisContractError(
                "authoritative HDF5 aperture product is incomplete"
            )
        semantic_aperture = np.asarray(
            semantic_aperture_group["aperture_mask"]
        )
        for factor, seconds in zip(
            expected_factors,
            expected_cadence_seconds,
            strict=True,
        ):
            name = f"{seconds}s"
            group = cadence_group[name]
            factor_attribute = group.attrs.get("coadd_factor")
            if (
                not isinstance(group, h5py.Group)
                or set(group) != _CADENCE_DATASET_NAMES
                or isinstance(factor_attribute, (bool, np.bool_))
                or not isinstance(factor_attribute, (int, np.integer))
                or int(factor_attribute) != factor
            ):
                raise StampScienceAnalysisContractError(
                    f"authoritative HDF5 cadence {name} contract is invalid"
                )
            count = int(group["time_start_seconds"].shape[0])
            if count <= 0 or any(
                tuple(group[item].shape) != (count,)
                for item in _CADENCE_DATASET_NAMES
            ):
                raise StampScienceAnalysisContractError(
                    f"authoritative HDF5 cadence {name} axes differ"
                )
            time = np.asarray(group["time_start_seconds"], dtype=np.float64)
            exposure = np.asarray(group["exposure_seconds"], dtype=np.float64)
            start_dataset = group["raw_frame_start_index"]
            stop_dataset = group["raw_frame_stop_index_exclusive"]
            raw_starts = np.asarray(start_dataset)
            raw_stops = np.asarray(stop_dataset)
            expected_exposure = raw_exposure_seconds * factor
            expected_count = raw_frame_interval["count"] // factor
            expected_starts = (
                raw_frame_interval["start_index"]
                + np.arange(expected_count, dtype=np.int64) * factor
            )
            expected_stops = expected_starts + factor
            expected_time = (
                raw_cadence_anchor.first_time_start_seconds
                + (
                    expected_starts - raw_frame_interval["start_index"]
                )
                * raw_exposure_seconds
            )
            if (
                not np.all(np.isfinite(time))
                or not np.all(np.isfinite(exposure))
                or not np.allclose(
                    exposure,
                    expected_exposure,
                    rtol=0.0,
                    atol=_TIME_ATOL_SECONDS,
                )
                or start_dataset.dtype.kind not in {"i", "u"}
                or stop_dataset.dtype.kind not in {"i", "u"}
                or count != expected_count
                or not np.array_equal(raw_starts, expected_starts)
                or not np.array_equal(raw_stops, expected_stops)
                or not np.allclose(
                    time,
                    expected_time,
                    rtol=0.0,
                    atol=_TIME_ATOL_SECONDS,
                )
                or (count > 1 and not _formal_time_intervals_are_contiguous(time, exposure))
            ):
                raise StampScienceAnalysisContractError(
                    f"authoritative analysis semantics cadence {name} "
                    "intervals differ from raw-shard coverage"
                )
            starts = np.asarray(raw_starts, dtype=np.int64)
            stops = np.asarray(raw_stops, dtype=np.int64)
            uncertainty_valid_raw = np.asarray(group["uncertainty_valid"])
            uncertainty_factor = np.asarray(group["uncertainty_coadd_factor"])
            if (
                uncertainty_valid_raw.dtype.kind not in {"b", "i", "u"}
                or not np.all(
                    (uncertainty_valid_raw == 0) | (uncertainty_valid_raw == 1)
                )
                or uncertainty_factor.dtype.kind not in {"i", "u"}
                or not np.all(uncertainty_factor == factor)
            ):
                raise StampScienceAnalysisContractError(
                    f"authoritative HDF5 cadence {name} uncertainty factor differs"
                )
            uncertainty_valid = np.asarray(uncertainty_valid_raw, dtype=bool)
            quality_bitmask = np.asarray(group["quality_bitmask"])
            aperture_valid_raw = np.asarray(group["aperture_valid"])
            if (
                quality_bitmask.dtype.kind not in {"i", "u"}
                or np.any(quality_bitmask < 0)
                or np.any(quality_bitmask > np.iinfo(np.uint16).max)
                or aperture_valid_raw.dtype.kind not in {"b", "i", "u"}
                or not np.all(
                    (aperture_valid_raw == 0) | (aperture_valid_raw == 1)
                )
            ):
                raise StampScienceAnalysisContractError(
                    f"authoritative HDF5 cadence {name} quality fields are invalid"
                )
            captured_fraction = np.asarray(
                group["captured_flux_fraction"], dtype=np.float64
            )
            captured_denominator = np.asarray(
                group["captured_flux_denominator_e"], dtype=np.float64
            )
            captured_qa = np.asarray(group["captured_flux_qa_pass"])
            if (
                not np.all(np.isfinite(captured_fraction))
                or np.any(captured_fraction < 0.0)
                or np.any(captured_fraction > 1.0 + 1.0e-6)
                or not np.all(np.isfinite(captured_denominator))
                or np.any(captured_denominator <= 0.0)
                or captured_qa.dtype.kind not in {"b", "i", "u"}
                or not np.all((captured_qa == 0) | (captured_qa == 1))
                or not np.all(captured_qa == 1)
            ):
                raise StampScienceAnalysisContractError(
                    f"authoritative HDF5 cadence {name} capture QA did not pass"
                )
            expected_quality_cadences[name] = _quality_cadence_summary_v1(
                quality_bitmask=quality_bitmask,
                aperture_valid=aperture_valid_raw,
                uncertainty_valid=uncertainty_valid,
                captured_flux_qa_pass=captured_qa,
                captured_flux_fraction=captured_fraction,
            )
            expected_captured_records[name] = (
                bool(np.all(captured_qa)),
                float(np.min(captured_fraction)),
            )
            uncertainty = np.asarray(group["flux_uncertainty_e"], dtype=np.float64)
            variance_components = tuple(
                np.asarray(group[item], dtype=np.float64)
                for item in (
                    "source_variance_e2",
                    "background_variance_e2",
                    "read_variance_e2",
                    "quantization_variance_e2",
                )
            )
            if any(
                not np.all(np.isfinite(component)) or np.any(component < 0.0)
                for component in variance_components
            ):
                raise StampScienceAnalysisContractError(
                    f"authoritative HDF5 cadence {name} variance components are invalid"
                )
            component_total = sum(variance_components)
            if (
                np.any(component_total < 0.0)
                or not np.all(np.isfinite(component_total))
                or not np.allclose(
                    uncertainty[uncertainty_valid],
                    np.sqrt(component_total[uncertainty_valid]),
                    rtol=1e-12,
                    atol=1e-12,
                )
                or np.any(np.isfinite(uncertainty[~uncertainty_valid]))
            ):
                raise StampScienceAnalysisContractError(
                    f"authoritative HDF5 cadence {name} uncertainty differs"
                )
            if (
                not np.allclose(
                    np.asarray(group["model_flux_uncertainty_e"]),
                    uncertainty,
                    equal_nan=True,
                )
                or not np.allclose(
                    np.asarray(group["flux_expectation_bgsub_e_per_s"]),
                    np.asarray(group["flux_expectation_bgsub_e"])
                    / exposure,
                    equal_nan=True,
                )
                or not np.allclose(
                    np.asarray(group["model_flux_uncertainty_e_per_s"]),
                    uncertainty / exposure,
                    equal_nan=True,
                )
            ):
                raise StampScienceAnalysisContractError(
                    f"authoritative HDF5 cadence {name} rate/model uncertainty differs"
                )
            _validate_recomputed_cadence_semantics_v1(
                group,
                cadence_name=name,
                factor=factor,
                time_seconds=time,
                exposure_seconds=exposure,
                raw_frame_start_index=starts,
                raw_frame_stop_index_exclusive=stops,
                raw_frame_interval=raw_frame_interval,
                raw_relative_flux=q,
                photometry_policy=analysis_policy.photometry,
                aperture_mask=semantic_aperture,
                read_noise_e_per_raw_pixel=read_noise_e_per_raw_pixel,
                quantization_noise_e_per_raw_pixel=(
                    quantization_noise_e_per_raw_pixel
                ),
                authoritative_cdpp=authoritative_cdpp,
            )
            cadence_seconds.append(seconds)
            if factor == 1:
                raw_frame_count = count
                aperture_valid = aperture_valid_raw
                saturated_count = np.asarray(group["saturated_pixel_count"])
                cosmic_count = np.asarray(group["cosmic_pixel_count"])
                if (
                    aperture_valid.dtype.kind not in {"b", "i", "u"}
                    or not np.all((aperture_valid == 0) | (aperture_valid == 1))
                    or saturated_count.dtype.kind not in {"i", "u"}
                    or cosmic_count.dtype.kind not in {"i", "u"}
                    or np.any(saturated_count < 0)
                    or np.any(cosmic_count < 0)
                ):
                    raise StampScienceAnalysisContractError(
                        "representative calibrated-frame provenance cadence "
                        "quality is invalid"
                    )
                clean = (
                    (aperture_valid == 1)
                    & (saturated_count == 0)
                    & (cosmic_count == 0)
                )
                clean_positions = np.flatnonzero(clean)
                midpoint = (
                    raw_frame_interval["start_index"]
                    + raw_frame_interval["count"] // 2
                )
                middle_positions = clean_positions[
                    starts[clean_positions] >= midpoint
                ]
                if clean_positions.size == 0 or middle_positions.size == 0:
                    raise StampScienceAnalysisContractError(
                        "representative calibrated-frame provenance selection "
                        "is unavailable"
                    )
                selected_positions = np.asarray(
                    [
                        int(clean_positions[0]),
                        int(middle_positions[0]),
                        int(clean_positions[-1]),
                    ],
                    dtype=np.int64,
                )
                selected_starts = tuple(
                    int(value) for value in starts[selected_positions]
                )
                if (
                    selected_starts
                    != representative_provenance.raw_frame_start_indices
                ):
                    raise StampScienceAnalysisContractError(
                        "representative calibrated-frame provenance selection "
                        "differs from authoritative factor-1 cadence"
                    )
                representative_time_start_seconds = tuple(
                    float(value) for value in time[selected_positions]
                )
                representative_exposure_seconds = tuple(
                    float(value) for value in exposure[selected_positions]
                )
                representative_factor_one = {
                    field: np.asarray(group[field])[selected_positions]
                    for field in (
                        "flux_expectation_bgsub_e",
                        "flux_local_bgsub_e",
                        "local_background_e_per_pixel",
                        "centroid_x",
                        "centroid_y",
                        "aperture_valid",
                        "aperture_usable_pixel_count",
                        "aperture_invalid_pixel_count",
                        "saturated_pixel_count",
                        "cosmic_pixel_count",
                        "background_usable_pixel_count",
                        "quality_bitmask",
                        "background_expectation_aperture_e",
                        "captured_flux_fraction",
                        "captured_flux_denominator_e",
                        "captured_flux_qa_pass",
                    )
                }
        aperture_group = handle.get("aperture")
        required_aperture_datasets = {
            "aperture_mask",
            "background_mask",
        }
        if analysis_product == "science_optimal_aperture_v1":
            required_aperture_datasets.update(
                {
                    "signal_template_e",
                    "noise_template_e",
                    "training_raw_frame_indices",
                }
            )
        if (
            not isinstance(aperture_group, h5py.Group)
            or set(aperture_group) != required_aperture_datasets
        ):
            raise StampScienceAnalysisContractError(
                "authoritative HDF5 aperture product is incomplete"
            )
        aperture = np.asarray(aperture_group["aperture_mask"], dtype=bool)
        background = np.asarray(aperture_group["background_mask"], dtype=bool)
        if (
            aperture.ndim != 2
            or background.shape != aperture.shape
            or not np.any(aperture)
            or np.any(aperture & background)
        ):
            raise StampScienceAnalysisContractError(
                "authoritative HDF5 aperture masks are invalid"
            )
        if analysis_product == "science_optimal_aperture_v1":
            signal_template = np.asarray(
                aperture_group["signal_template_e"], dtype=np.float64
            )
            noise_template = np.asarray(
                aperture_group["noise_template_e"], dtype=np.float64
            )
            science_training_indices = np.asarray(
                aperture_group["training_raw_frame_indices"]
            )
            if (
                signal_template.shape != aperture.shape
                or noise_template.shape != aperture.shape
                or not np.all(np.isfinite(signal_template))
                or np.any(signal_template < 0.0)
                or not np.all(np.isfinite(noise_template))
                or np.any(noise_template <= 0.0)
                or not _valid_training_raw_frame_indices(
                    science_training_indices
                )
            ):
                raise StampScienceAnalysisContractError(
                    "authoritative HDF5 aperture training products are invalid"
                )
    expected_captured_contract = _captured_flux_qa_contract_v1(
        expected_captured_records
    )
    if _canonical_json_text(
        captured_contract,
        name="recorded captured-flux contract",
    ) != _canonical_json_text(
        expected_captured_contract,
        name="recomputed captured-flux contract",
    ):
        raise StampScienceAnalysisContractError(
            "captured-flux contract differs from authoritative HDF5 cadences"
        )
    aperture_npy = np.load(root / "aperture_mask.npy", allow_pickle=False)
    background_npy = np.load(root / "background_mask.npy", allow_pickle=False)
    if not np.array_equal(aperture_npy, aperture) or not np.array_equal(
        background_npy, background
    ):
        raise StampScienceAnalysisContractError(
            "portable and authoritative aperture masks differ"
        )
    if analysis_product == "science_optimal_aperture_v1":
        portable_signal = np.load(
            root / "signal_template_e.npy",
            allow_pickle=False,
        )
        portable_noise = np.load(
            root / "noise_template_e.npy",
            allow_pickle=False,
        )
        if (
            signal_template is None
            or noise_template is None
            or portable_signal.shape != aperture.shape
            or portable_noise.shape != aperture.shape
            or not np.array_equal(portable_signal, signal_template)
            or not np.array_equal(portable_noise, noise_template)
        ):
            raise StampScienceAnalysisContractError(
                "portable and authoritative signal/noise templates differ"
            )
    aperture_definition = _read_json_object(
        root / "aperture_definition.json", label="aperture definition"
    )
    contract_aperture = contract.get("aperture")
    if (
        not isinstance(contract_aperture, Mapping)
        or set(contract_aperture) != _APERTURE_DEFINITION_FIELDS
        or set(aperture_definition) != _APERTURE_DEFINITION_FIELDS
        or _canonical_json_text(
            aperture_definition,
            name="portable aperture definition",
        )
        != _canonical_json_text(
            dict(contract_aperture),
            name="analysis contract aperture definition",
        )
        or aperture_definition.get("schema_id")
        != _APERTURE_DEFINITION_SCHEMA_ID
        or not _is_native_schema_version(
            aperture_definition.get("schema_version"),
            _APERTURE_DEFINITION_SCHEMA_VERSION,
        )
    ):
        raise StampScienceAnalysisContractError(
            "aperture definition schema is invalid"
        )
    maximum_cumulative_snr = aperture_definition["maximum_cumulative_snr"]
    algorithm = aperture_definition["algorithm"]
    signal_template_shape = aperture_definition["signal_template_shape"]
    recorded_training_indices = aperture_definition["training_raw_frame_indices"]
    target_peak_yx = aperture_definition["target_peak_yx"]
    aperture_pixel_count = aperture_definition["aperture_pixel_count"]
    background_pixel_count = aperture_definition["background_pixel_count"]
    expected_aperture_files = {
        "aperture_mask": "aperture_mask.npy",
        "background_mask": "background_mask.npy",
    }
    if analysis_product == "science_optimal_aperture_v1":
        expected_aperture_files.update(
            {
                "signal_template_e": "signal_template_e.npy",
                "noise_template_e": "noise_template_e.npy",
            }
        )
    if (
        not isinstance(algorithm, str)
        or not algorithm
        or not isinstance(signal_template_shape, list)
        or len(signal_template_shape) != 2
        or any(type(item) is not int for item in signal_template_shape)
        or tuple(signal_template_shape) != aperture.shape
        or (
            target_peak_yx is not None
            and (
                not isinstance(target_peak_yx, list)
                or len(target_peak_yx) != 2
                or any(type(item) is not int for item in target_peak_yx)
            )
        )
        or type(aperture_pixel_count) is not int
        or aperture_pixel_count != int(np.count_nonzero(aperture))
        or type(background_pixel_count) is not int
        or background_pixel_count != int(np.count_nonzero(background))
        or aperture_definition["files"] != expected_aperture_files
        or not isinstance(aperture_definition["metadata"], Mapping)
    ):
        raise StampScienceAnalysisContractError(
            "aperture definition fields differ from authoritative HDF5"
        )
    if analysis_product == "science_optimal_aperture_v1":
        if (
            isinstance(maximum_cumulative_snr, bool)
            or not isinstance(maximum_cumulative_snr, (int, float))
            or not math.isfinite(float(maximum_cumulative_snr))
            or float(maximum_cumulative_snr) <= 0.0
            or not isinstance(recorded_training_indices, list)
            or any(type(item) is not int for item in recorded_training_indices)
            or not _valid_training_raw_frame_indices(recorded_training_indices)
            or science_training_indices is None
            or not np.array_equal(
                np.asarray(recorded_training_indices, dtype=np.int64),
                science_training_indices,
            )
        ):
            raise StampScienceAnalysisContractError(
                "aperture definition science-optimal training fields are invalid"
            )
    elif maximum_cumulative_snr is not None or recorded_training_indices is not None:
        raise StampScienceAnalysisContractError(
            "aperture definition reference training fields are invalid"
        )
    aperture_metadata = aperture_definition["metadata"]
    if (
        aperture_metadata.get("background_strategy")
        != contract.get("background_strategy")
    ):
        raise StampScienceAnalysisContractError(
            "aperture metadata background strategy differs from analysis contract"
        )
    _validate_cdpp_portable_products_v1(
        root,
        authoritative_payload=authoritative_cdpp,
        background_strategy=contract.get("background_strategy"),
    )
    quality_summary = _read_json_object(
        root / "quality_summary.json", label="quality summary"
    )
    expected_quality_summary = {
        "schema_id": _QUALITY_SUMMARY_SCHEMA_ID,
        "schema_version": _QUALITY_SUMMARY_SCHEMA_VERSION,
        "centroid_coordinate_system": _QUALITY_SUMMARY_COORDINATE_SYSTEM,
        "cadences": expected_quality_cadences,
    }
    if _canonical_json_text(
        quality_summary,
        name="portable quality summary",
    ) != _canonical_json_text(
        expected_quality_summary,
        name="authoritative HDF5 quality summary",
    ):
        raise StampScienceAnalysisContractError(
            "quality summary schema/values differ from authoritative HDF5 "
            "cadence products"
        )
    for name in (
        "photometry.ecsv",
        "reference_lightcurve.ecsv",
        "centroid_quality.ecsv",
    ):
        if (root / name).stat().st_size <= 0:
            raise StampScienceAnalysisContractError(f"portable ECSV is empty: {name}")
    _validate_portable_photometry_ecsv_v2(
        root / "photometry.ecsv",
        authoritative_hdf5_path=hdf_path,
    )
    _validate_reference_lightcurve_ecsv_v1(
        root / "reference_lightcurve.ecsv",
        authoritative_hdf5_path=hdf_path,
        contract=contract,
    )
    _validate_centroid_quality_ecsv_v1(
        root / "centroid_quality.ecsv",
        authoritative_hdf5_path=hdf_path,
    )
    for name in (
        "figures/lightcurve_overview.png",
        "figures/cdpp_summary.png",
        "figures/representative_frames.png",
    ):
        if (root / name).stat().st_size <= 0:
            raise StampScienceAnalysisContractError(f"science figure is empty: {name}")
    if (
        representative_time_start_seconds is None
        or representative_exposure_seconds is None
        or representative_factor_one is None
    ):
        raise StampScienceAnalysisContractError(
            "representative calibrated-frame provenance lacks an authoritative "
            "factor-1 cadence binding"
        )
    with h5py.File(root / "representative_calibrated_frames.h5", "r") as handle:
        rep_schema = handle.attrs.get("schema_id")
        if isinstance(rep_schema, bytes):
            rep_schema = rep_schema.decode("utf-8")
        if (
            rep_schema != _REPRESENTATIVE_FRAMES_SCHEMA_ID
            or not _is_hdf_integer_schema_version(
                handle.attrs.get("schema_version"),
                _REPRESENTATIVE_FRAMES_SCHEMA_VERSION,
            )
            or not _is_hdf_boolean(handle.attrs.get("complete"), True)
            or handle.attrs.get("observation_product") != "final_dn"
            or not _is_hdf_boolean(
                handle.attrs.get("background_realization_used"),
                False,
            )
            or set(handle) != _REPRESENTATIVE_FRAME_DATASET_NAMES
        ):
            raise StampScienceAnalysisContractError(
                "representative calibrated-frame product is invalid"
            )

        frame_shape = (3, *aperture.shape)
        vector_shape = (3,)
        column_shape = (3, aperture.shape[1])
        cube_names = {
            "final_dn",
            "calibrated_e",
            "calibrated_bgsub_e",
            "background_expectation_e",
            "valid_mask",
            "saturated_mask",
            "cosmic_mask",
        }
        vector_names = {
            "captured_flux_fraction",
            "captured_flux_denominator_e",
            "captured_flux_qa_pass",
            "bias_level_sum_dn",
            "time_start_seconds",
            "exposure_seconds",
            "raw_frame_start_index",
            "raw_frame_stop_index_exclusive",
            "selection_role",
            "input_shard_path",
            "input_shard_semantic_sha256",
        }
        gain = np.asarray(handle["gain_e_per_dn"])
        if (
            any(tuple(handle[name].shape) != frame_shape for name in cube_names)
            or any(
                tuple(handle[name].shape) != vector_shape for name in vector_names
            )
            or tuple(handle["column_noise_sum_dn_by_x"].shape) != column_shape
            or gain.shape not in {(), aperture.shape}
            or gain.dtype.kind not in {"f", "i", "u"}
            or not np.all(np.isfinite(gain))
            or np.any(gain <= 0.0)
            or handle["final_dn"].dtype.kind != "u"
        ):
            raise StampScienceAnalysisContractError(
                "representative calibrated-frame product is invalid"
            )

        def binary_dataset(name: str, shape: tuple[int, ...]) -> NDArray[Any]:
            value = np.asarray(handle[name])
            if (
                value.shape != shape
                or value.dtype.kind not in {"b", "i", "u"}
                or not np.all((value == 0) | (value == 1))
            ):
                raise StampScienceAnalysisContractError(
                    "representative calibrated-frame product is invalid"
                )
            return value

        def string_vector(name: str) -> tuple[str, ...]:
            dataset = handle[name]
            if h5py.check_string_dtype(dataset.dtype) is None:
                raise StampScienceAnalysisContractError(
                    "representative calibrated-frame provenance is invalid"
                )
            try:
                values = np.asarray(dataset.asstr()[...]).tolist()
            except (TypeError, UnicodeDecodeError, ValueError) as error:
                raise StampScienceAnalysisContractError(
                    "representative calibrated-frame provenance is invalid"
                ) from error
            if (
                not isinstance(values, list)
                or len(values) != len(_REPRESENTATIVE_FRAME_SELECTION_ROLES)
                or any(type(value) is not str for value in values)
            ):
                raise StampScienceAnalysisContractError(
                    "representative calibrated-frame provenance is invalid"
                )
            return tuple(values)

        captured_qa = binary_dataset("captured_flux_qa_pass", vector_shape)
        valid = binary_dataset("valid_mask", frame_shape).astype(bool)
        saturated = binary_dataset("saturated_mask", frame_shape).astype(bool)
        cosmic = binary_dataset("cosmic_mask", frame_shape).astype(bool)
        final = np.asarray(handle["final_dn"])
        calibrated = np.asarray(handle["calibrated_e"], dtype=np.float64)
        calibrated_bgsub = np.asarray(
            handle["calibrated_bgsub_e"],
            dtype=np.float64,
        )
        background_expectation = np.asarray(
            handle["background_expectation_e"],
            dtype=np.float64,
        )
        captured_fraction = np.asarray(
            handle["captured_flux_fraction"],
            dtype=np.float64,
        )
        captured_denominator = np.asarray(
            handle["captured_flux_denominator_e"],
            dtype=np.float64,
        )
        bias = np.asarray(handle["bias_level_sum_dn"], dtype=np.float64)
        column = np.asarray(
            handle["column_noise_sum_dn_by_x"],
            dtype=np.float64,
        )
        time = np.asarray(handle["time_start_seconds"], dtype=np.float64)
        exposure = np.asarray(handle["exposure_seconds"], dtype=np.float64)
        raw_start_dataset = handle["raw_frame_start_index"]
        raw_stop_dataset = handle["raw_frame_stop_index_exclusive"]
        raw_start = np.asarray(raw_start_dataset)
        raw_stop = np.asarray(raw_stop_dataset)
        selection_roles = string_vector("selection_role")
        input_shard_paths = string_vector("input_shard_path")
        input_shard_semantic_sha256 = string_vector(
            "input_shard_semantic_sha256"
        )
        if (
            raw_start_dataset.dtype.kind != "i"
            or raw_start_dataset.dtype.itemsize != np.dtype(np.int64).itemsize
            or raw_stop_dataset.dtype.kind != "i"
            or raw_stop_dataset.dtype.itemsize != np.dtype(np.int64).itemsize
            or tuple(raw_start.tolist())
            != representative_provenance.raw_frame_start_indices
            or tuple(raw_stop.tolist())
            != tuple(
                value + 1
                for value in representative_provenance.raw_frame_start_indices
            )
            or selection_roles != representative_provenance.selection_roles
            or input_shard_paths != representative_provenance.input_shard_paths
            or input_shard_semantic_sha256
            != representative_provenance.input_shard_semantic_sha256
            or not np.allclose(
                time,
                representative_time_start_seconds,
                rtol=0.0,
                atol=_TIME_ATOL_SECONDS,
            )
            or not np.allclose(
                exposure,
                representative_exposure_seconds,
                rtol=0.0,
                atol=_TIME_ATOL_SECONDS,
            )
        ):
            raise StampScienceAnalysisContractError(
                "representative calibrated-frame provenance differs from contract"
            )
        expected_calibrated = (
            final.astype(np.float64) - bias[:, None, None] - column[:, None, :]
        ) * gain
        if (
            not np.all(np.isfinite(calibrated))
            or not np.all(np.isfinite(calibrated_bgsub))
            or not np.all(np.isfinite(background_expectation))
            or np.any(background_expectation < 0.0)
            or not np.all(np.isfinite(captured_fraction))
            or np.any(captured_fraction < 0.0)
            or np.any(captured_fraction > 1.0 + 1.0e-6)
            or not np.all(np.isfinite(captured_denominator))
            or np.any(captured_denominator <= 0.0)
            or not np.all(captured_qa == 1)
            or not np.all(np.isfinite(bias))
            or not np.all(np.isfinite(column))
            or not np.all(np.isfinite(time))
            or not np.all(np.isfinite(exposure))
            or np.any(exposure <= 0.0)
            or not np.allclose(
                calibrated,
                expected_calibrated,
                rtol=0.0,
                atol=1e-8,
            )
            or not np.allclose(
                calibrated_bgsub,
                calibrated - background_expectation,
                rtol=0.0,
                atol=1e-8,
            )
        ):
            raise StampScienceAnalysisContractError(
                "representative calibrated-frame electron semantics differ"
            )
        if (
            np.any(saturated[:, aperture])
            or np.any(cosmic[:, aperture])
            or not np.all(valid[:, aperture])
        ):
            raise StampScienceAnalysisContractError(
                "representative frames are not clean for their product aperture"
            )
        try:
            representative_photometry = reduce_science_photometry_v1(
                ReferencePhotometryInput.from_arrays(
                    final_dn=final,
                    background_expectation_e=background_expectation,
                    bias_level_sum_dn=bias,
                    column_noise_sum_dn_by_x=column,
                    valid_mask=valid,
                    saturated_mask=saturated,
                    cosmic_mask=cosmic,
                    time_index=np.arange(final.shape[0], dtype=np.float64),
                    gain_e_per_dn=gain,
                    time_index_unit="seconds",
                    exposure_seconds=exposure,
                ),
                aperture_mask=aperture,
                background_mask=(
                    background
                    if analysis_policy.photometry.local_background_enabled
                    else None
                ),
                minimum_background_pixels=(
                    analysis_policy.photometry.minimum_background_pixels
                ),
            )
        except (
            ReferencePhotometryContractError,
            SciencePhotometryContractError,
        ) as error:
            raise StampScienceAnalysisContractError(
                "representative calibrated-frame photometry is invalid"
            ) from error
        recomputed_fields = {
            "flux_expectation_bgsub_e": (
                representative_photometry.flux_expectation_bgsub_e
            ),
            "flux_local_bgsub_e": representative_photometry.flux_local_bgsub_e,
            "local_background_e_per_pixel": (
                representative_photometry.local_background_e_per_pixel
            ),
            "centroid_x": representative_photometry.centroid_x,
            "centroid_y": representative_photometry.centroid_y,
            "aperture_valid": representative_photometry.aperture_valid,
            "aperture_usable_pixel_count": (
                representative_photometry.aperture_usable_pixel_count
            ),
            "aperture_invalid_pixel_count": (
                representative_photometry.aperture_invalid_pixel_count
            ),
            "saturated_pixel_count": (
                representative_photometry.saturated_pixel_count
            ),
            "cosmic_pixel_count": representative_photometry.cosmic_pixel_count,
            "background_usable_pixel_count": (
                representative_photometry.background_usable_pixel_count
            ),
            "quality_bitmask": representative_photometry.quality_bitmask,
            "background_expectation_aperture_e": np.sum(
                background_expectation[:, aperture],
                axis=1,
                dtype=np.float64,
            ),
            "captured_flux_fraction": captured_fraction,
            "captured_flux_denominator_e": captured_denominator,
            "captured_flux_qa_pass": captured_qa,
        }
        if any(
            not _arrays_match_exactly(
                recomputed_fields[name],
                representative_factor_one[name],
            )
            for name in representative_factor_one
        ):
            raise StampScienceAnalysisContractError(
                "representative calibrated-frame photometry differs from the "
                "authoritative factor-1 cadence"
            )
    return StampScienceAnalysisValidation(
        output_dir=root,
        complete=True,
        cadence_seconds=tuple(cadence_seconds),
        raw_frame_count=raw_frame_count,
        aperture_pixel_count=int(np.count_nonzero(aperture)),
    )


def _validate_staged_analysis_v1(path: Path) -> StampScienceAnalysisValidation:
    """Read back every staged artifact before the directory becomes visible."""

    return _validate_analysis_dir(path)


def validate_stamp_science_analysis_v1(
    path: Path | str,
) -> StampScienceAnalysisValidation:
    """Validate a complete, already-published science analysis directory."""

    return _validate_analysis_dir(Path(path))


def validate_stamp_science_analysis_product_set_v1(
    path: Path | str,
) -> StampScienceAnalysisProductSetValidation:
    """Validate the atomic two-aperture product-set publication and bindings."""

    input_root = Path(path).expanduser()
    if input_root.is_symlink():
        raise StampScienceAnalysisContractError(
            "science analysis product set must not be a symbolic link"
        )
    root = input_root.resolve()
    if not root.is_dir():
        raise StampScienceAnalysisContractError(
            "science analysis product set must be a real directory"
        )
    manifest_path = root / "product_set_manifest.json"
    manifest = _read_json_object(manifest_path, label="product-set manifest")
    expected_names = {
        "reference_fixed13_v1",
        "science_optimal_aperture_v1",
    }
    products = manifest.get("products")
    if (
        manifest.get("schema_id")
        != STAMP_SCIENCE_ANALYSIS_PRODUCT_SET_SCHEMA_ID
        or not _is_native_schema_version(
            manifest.get("schema_version"),
            STAMP_SCIENCE_ANALYSIS_PRODUCT_SET_SCHEMA_VERSION,
        )
        or manifest.get("complete") is not True
        or manifest.get("ready") is not True
        or not isinstance(products, Mapping)
        or set(products) != expected_names
    ):
        raise StampScienceAnalysisContractError(
            "analysis product-set manifest schema/completeness is invalid"
        )
    validations: dict[str, StampScienceAnalysisValidation] = {}
    common_context: Mapping[str, Any] | None = None
    common_raw_policy_binding: Mapping[str, Any] | None = None
    for name in sorted(expected_names):
        record = products[name]
        if (
            not isinstance(record, Mapping)
            or set(record) != {"relative_path", "analysis_manifest"}
            or record.get("relative_path") != name
            or not isinstance(record.get("analysis_manifest"), Mapping)
        ):
            raise StampScienceAnalysisContractError(
                f"analysis product-set record is invalid: {name}"
            )
        product_root = root / name
        validation = _validate_analysis_dir(product_root)
        if _file_identity(product_root / "analysis_manifest.json") != dict(
            record["analysis_manifest"]
        ):
            raise StampScienceAnalysisContractError(
                f"analysis product-set manifest identity differs: {name}"
            )
        child_manifest = _read_json_object(
            product_root / "analysis_manifest.json",
            label=f"{name} analysis manifest",
        )
        contract = child_manifest.get("contract")
        if (
            not isinstance(contract, Mapping)
            or contract.get("analysis_product") != name
        ):
            raise StampScienceAnalysisContractError(
                f"analysis product does not match its directory role: {name}"
            )
        context = contract.get("analysis_context")
        if not isinstance(context, Mapping):
            raise StampScienceAnalysisContractError(
                f"analysis product lacks bound analysis context: {name}"
            )
        raw_shards = contract.get("input_raw_shards")
        raw_relative_flux = contract.get("raw_relative_flux")
        policy = contract.get("policy")
        if (
            not isinstance(raw_shards, list)
            or not raw_shards
            or any(
                not isinstance(item, Mapping)
                or not isinstance(item.get("semantic_sha256"), str)
                or len(item["semantic_sha256"]) != 64
                for item in raw_shards
            )
            or not isinstance(raw_relative_flux, Mapping)
            or not isinstance(raw_relative_flux.get("sha256"), str)
            or len(raw_relative_flux["sha256"]) != 64
            or not isinstance(policy, Mapping)
        ):
            raise StampScienceAnalysisContractError(
                f"analysis product lacks bound raw semantic identities/policy: {name}"
            )
        raw_policy_binding = {
            "input_raw_shards": raw_shards,
            "raw_relative_flux": raw_relative_flux,
            "policy": policy,
        }
        if common_context is None:
            common_context = dict(context)
        elif _canonical_json_text(
            dict(context),
            name=f"{name} analysis context",
        ) != _canonical_json_text(
            dict(common_context),
            name="common child analysis context",
        ):
            raise StampScienceAnalysisContractError(
                "analysis products bind different source/production contexts"
            )
        if common_raw_policy_binding is None:
            common_raw_policy_binding = raw_policy_binding
        elif _canonical_json_text(
            raw_policy_binding,
            name=f"{name} raw semantic identities/policy",
        ) != _canonical_json_text(
            common_raw_policy_binding,
            name="common child raw semantic identities/policy",
        ):
            raise StampScienceAnalysisContractError(
                "analysis products bind different raw semantic identities/policy"
            )
        validations[name] = validation
    if _canonical_json_text(
        manifest.get("analysis_context", {}),
        name="product-set analysis context",
    ) != _canonical_json_text(
        dict(common_context or {}),
        name="common child analysis context",
    ):
        raise StampScienceAnalysisContractError(
            "product-set and child analysis contexts differ"
        )
    formal_profile_id = manifest.get("formal_profile_id")
    child_formal_profile_id = (
        common_context.get("formal_profile_id")
        if isinstance(common_context, Mapping)
        else None
    )
    if (
        child_formal_profile_id != formal_profile_id
        or (
            formal_profile_id is not None
            and formal_profile_id != STAMP_SCIENCE_FORMAL_PROFILE_ID
        )
    ):
        raise StampScienceAnalysisContractError(
            "product-set formal profile is unsupported or differs from its children"
        )
    return StampScienceAnalysisProductSetValidation(
        output_dir=root,
        manifest_path=manifest_path,
        complete=True,
        products=validations,
    )


@contextmanager
def _exclusive_output_lock(target: Path) -> Iterator[None]:
    target.parent.mkdir(parents=True, exist_ok=True)
    lock = target.with_name(f".{target.name}.lock")
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise FileExistsError(f"science analysis output is locked: {lock}") from error
    try:
        os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        yield
    finally:
        os.close(descriptor)
        try:
            lock.unlink()
        except FileNotFoundError:  # pragma: no cover - defensive cleanup.
            pass


def _fsync_directory(path: Path) -> None:
    """Persist directory entries and propagate real storage failures."""

    _strict_fsync_directory(path)


def _publication_from_root(root: Path) -> StampScienceAnalysisPublication:
    return StampScienceAnalysisPublication(
        output_dir=root,
        hdf5_path=root / "photometry.h5",
        ecsv_path=root / "photometry.ecsv",
        manifest_path=root / "analysis_manifest.json",
        aperture_definition_path=root / "aperture_definition.json",
        cdpp_path=root / "cdpp.json",
        aperture_mask_path=root / "aperture_mask.npy",
        background_mask_path=root / "background_mask.npy",
        representative_frames_path=root / "representative_calibrated_frames.h5",
    )


def _collect_analysis_execution_hardware_v1() -> dict[str, Any]:
    """Record the host resources visible to the CPU analysis process."""

    import torch

    cuda_available = bool(torch.cuda.is_available())
    cuda_names = (
        [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
        if cuda_available
        else []
    )
    return {
        "schema_id": "et_mainsim.analysis_execution_hardware.v1",
        "analysis_compute_device": "cpu",
        "cpu_count": os.cpu_count(),
        "machine": platform.machine(),
        "cuda_available": cuda_available,
        "cuda_device_names": cuda_names,
    }


def _collect_analysis_execution_identity_v1(
    request: StampScienceAnalysisRequest,
) -> dict[str, Any]:
    """Collect the identity of the process that actually publishes analysis."""

    if (
        request.analysis_context.get("formal_profile_id")
        == STAMP_SCIENCE_FORMAL_PROFILE_ID
    ):
        validate_stamp_science_analysis_request_ready_v1(request)
        identity = collect_formal_analysis_code_identity_v1()
        return {
            **identity,
            "execution_hardware": _collect_analysis_execution_hardware_v1(),
        }
    return dict(request.code_identity)


def analyze_stamp_science_series_v1(
    request: StampScienceAnalysisRequest,
) -> StampScienceAnalysisPublication:
    """Analyze one formal target/case series and publish it atomically."""

    if not isinstance(request, StampScienceAnalysisRequest):
        raise TypeError("request must be a StampScienceAnalysisRequest")
    execution_code_identity = _collect_analysis_execution_identity_v1(request)
    target = Path(request.output_dir)
    with _exclusive_output_lock(target):
        if target.exists():
            raise FileExistsError(
                f"science analysis output already exists: {target}; complete products "
                "are immutable"
            )
        raw_headers = _read_series_headers(
            request.raw_bundle_paths,
            product_kind="raw",
            coadd_factor=1,
        )
        direct_headers = {
            factor: _read_series_headers(
                paths,
                product_kind="coadd",
                coadd_factor=factor,
            )
            for factor, paths in request.direct_coadd_bundle_paths.items()
        }
        _validate_cross_product_headers(raw_headers, direct_headers)
        all_headers = raw_headers + tuple(
            header
            for _, headers in sorted(direct_headers.items())
            for header in headers
        )
        byte_identities = {
            str(header.formal.path): _resolve_input_byte_identity(header)
            for header in all_headers
        }
        first_raw = raw_headers[0].formal.first_raw_frame_start
        last_raw = raw_headers[-1].formal.last_raw_frame_stop
        raw_count = last_raw - first_raw
        q = np.asarray(request.raw_relative_flux, dtype=np.float64)
        if q.size != raw_count:
            raise StampScienceAnalysisContractError(
                "raw_relative_flux length must equal the formal raw-frame interval"
            )
        if request.aperture_mode == "train":
            aperture = _train_aperture(
                raw_headers,
                raw_relative_flux=q,
                first_raw_index=first_raw,
                request=request,
            )
        else:
            assert request.frozen_aperture is not None
            aperture = _validate_frozen_aperture_definition(
                request.frozen_aperture
            )
        aperture = _bind_aperture_background_policy_v1(
            aperture,
            policy=request.policy.photometry,
        )
        samples = _direct_samples(
            direct_headers,
            samples_per_shard=request.policy.direct_coadd_samples_per_shard,
        )
        collectors, raw_identities, captures, representative_frames = (
            _stream_raw_analysis(
                raw_headers,
                aperture=aperture,
                request=request,
                direct_samples=samples,
            )
        )
        for identity in raw_identities:
            identity["byte_identity"] = dict(
                byte_identities[str(identity["path"])]
            )
        analyses = {
            factor: _finish_cadence(
                collector,
                raw_relative_flux=q,
                first_raw_index=first_raw,
                policy=request.policy.photometry,
                aperture_mask=np.asarray(aperture.aperture_mask, dtype=bool),
                read_noise_e_per_pixel=request.read_noise_e_per_pixel,
                quantization_noise_e_per_pixel=(
                    request.quantization_noise_e_per_pixel
                ),
            )
            for factor, collector in collectors.items()
        }
        parity = _validate_direct_parity(samples, captures)
        if request.policy.require_direct_coadd_parity and not parity["passed"]:
            raise StampScienceAnalysisContractError(
                "raw-derived/direct-coadd parity is required"
            )
        contract = _build_contract(
            request=request,
            raw_headers=raw_headers,
            raw_identities=raw_identities,
            direct_headers=direct_headers,
            parity=parity,
            aperture=aperture,
            analyses=analyses,
            representative_frames=representative_frames,
            raw_identities_for_frames=raw_identities,
            direct_byte_identities=byte_identities,
            execution_code_identity=execution_code_identity,
        )
        partial = target.with_name(f".{target.name}.{uuid.uuid4().hex}.partial")
        try:
            partial.mkdir()
            _write_analysis_artifacts(
                partial,
                contract=contract,
                q=q,
                aperture=aperture,
                analyses=analyses,
                representative_frames=representative_frames,
                raw_identities=raw_identities,
            )
            _validate_staged_analysis_v1(partial)
            _atomic_publish_directory_noreplace(partial, target)
            _fsync_directory(target.parent)
        except BaseException:
            shutil.rmtree(partial, ignore_errors=True)
            raise
    return _publication_from_root(target)


def analyze_stamp_science_product_set_v1(
    request: StampScienceAnalysisRequest,
) -> StampScienceAnalysisProductSetPublication:
    """Publish reference-fixed13 and science-OA products from one raw pass."""

    if not isinstance(request, StampScienceAnalysisRequest):
        raise TypeError("request must be a StampScienceAnalysisRequest")
    execution_code_identity = _collect_analysis_execution_identity_v1(request)
    target = Path(request.output_dir)
    with _exclusive_output_lock(target):
        if target.exists():
            raise FileExistsError(
                f"science analysis output already exists: {target}; complete products "
                "are immutable"
            )
        raw_headers = _read_series_headers(
            request.raw_bundle_paths,
            product_kind="raw",
            coadd_factor=1,
        )
        direct_headers = {
            factor: _read_series_headers(
                paths,
                product_kind="coadd",
                coadd_factor=factor,
            )
            for factor, paths in request.direct_coadd_bundle_paths.items()
        }
        _validate_cross_product_headers(raw_headers, direct_headers)
        all_headers = raw_headers + tuple(
            header
            for _, headers in sorted(direct_headers.items())
            for header in headers
        )
        byte_identities = {
            str(header.formal.path): _resolve_input_byte_identity(header)
            for header in all_headers
        }
        first_raw = raw_headers[0].formal.first_raw_frame_start
        last_raw = raw_headers[-1].formal.last_raw_frame_stop
        q = np.asarray(request.raw_relative_flux, dtype=np.float64)
        if q.size != last_raw - first_raw:
            raise StampScienceAnalysisContractError(
                "raw_relative_flux length must equal the formal raw-frame interval"
            )
        if request.aperture_mode == "train":
            science_aperture = _train_aperture(
                raw_headers,
                raw_relative_flux=q,
                first_raw_index=first_raw,
                request=request,
            )
        else:
            assert request.frozen_aperture is not None
            science_aperture = _validate_frozen_aperture_definition(
                request.frozen_aperture
            )
        science_aperture = _bind_aperture_background_policy_v1(
            science_aperture,
            policy=request.policy.photometry,
        )
        reference_aperture = _build_reference_aperture(
            raw_headers[0].formal.stamp_shape,
            policy=request.policy.photometry,
        )
        apertures = {
            "reference_fixed13_v1": reference_aperture,
            "science_optimal_aperture_v1": science_aperture,
        }
        samples = _direct_samples(
            direct_headers,
            samples_per_shard=request.policy.direct_coadd_samples_per_shard,
        )
        collectors_by_product, raw_identities, captures, representative_frames = (
            _stream_raw_product_analyses(
                raw_headers,
                apertures=apertures,
                representative_aperture_name=None,
                request=request,
                direct_samples=samples,
            )
        )
        for identity in raw_identities:
            identity["byte_identity"] = dict(
                byte_identities[str(identity["path"])]
            )
        analyses_by_product = {
            product_name: {
                factor: _finish_cadence(
                    collector,
                    raw_relative_flux=q,
                    first_raw_index=first_raw,
                    policy=request.policy.photometry,
                    aperture_mask=np.asarray(
                        apertures[product_name].aperture_mask,
                        dtype=bool,
                    ),
                    read_noise_e_per_pixel=request.read_noise_e_per_pixel,
                    quantization_noise_e_per_pixel=(
                        request.quantization_noise_e_per_pixel
                    ),
                )
                for factor, collector in collectors.items()
            }
            for product_name, collectors in collectors_by_product.items()
        }
        parity = _validate_direct_parity(samples, captures)
        if request.policy.require_direct_coadd_parity and not parity["passed"]:
            raise StampScienceAnalysisContractError(
                "raw-derived/direct-coadd parity is required"
            )
        contracts = {
            product_name: _build_contract(
                request=request,
                raw_headers=raw_headers,
                raw_identities=raw_identities,
                direct_headers=direct_headers,
                parity=parity,
                aperture=aperture,
                analyses=analyses_by_product[product_name],
                representative_frames=representative_frames[product_name],
                raw_identities_for_frames=raw_identities,
                direct_byte_identities=byte_identities,
                execution_code_identity=execution_code_identity,
                analysis_product=product_name,
                aperture_mode=(
                    "fixed_reference_13x13"
                    if product_name == "reference_fixed13_v1"
                    else request.aperture_mode
                ),
                aperture_source_identity=(
                    {
                        "mode": "legacy_floor_centered_fixed13_v1",
                        "aperture_pixel_count": 169,
                    }
                    if product_name == "reference_fixed13_v1"
                    else request.aperture_source_identity
                ),
            )
            for product_name, aperture in apertures.items()
        }
        partial = target.with_name(f".{target.name}.{uuid.uuid4().hex}.partial")
        try:
            partial.mkdir()
            for product_name, aperture in apertures.items():
                product_root = partial / product_name
                product_root.mkdir()
                _write_analysis_artifacts(
                    product_root,
                    contract=contracts[product_name],
                    q=q,
                    aperture=aperture,
                    analyses=analyses_by_product[product_name],
                    representative_frames=representative_frames[product_name],
                    raw_identities=raw_identities,
                )
                _validate_staged_analysis_v1(product_root)
            product_set_manifest = {
                "schema_id": STAMP_SCIENCE_ANALYSIS_PRODUCT_SET_SCHEMA_ID,
                "schema_version": STAMP_SCIENCE_ANALYSIS_PRODUCT_SET_SCHEMA_VERSION,
                "complete": True,
                "ready": True,
                "formal_profile_id": request.analysis_context.get(
                    "formal_profile_id"
                ),
                "analysis_context": dict(request.analysis_context),
                "products": {
                    product_name: {
                        "relative_path": product_name,
                        "analysis_manifest": _file_identity(
                            partial / product_name / "analysis_manifest.json"
                        ),
                    }
                    for product_name in apertures
                },
            }
            _write_json(
                partial / "product_set_manifest.json",
                product_set_manifest,
            )
            validate_stamp_science_analysis_product_set_v1(partial)
            _atomic_publish_directory_noreplace(partial, target)
            _fsync_directory(target.parent)
        except BaseException:
            shutil.rmtree(partial, ignore_errors=True)
            raise
    return StampScienceAnalysisProductSetPublication(
        output_dir=target,
        manifest_path=target / "product_set_manifest.json",
        reference_fixed13=_publication_from_root(
            target / "reference_fixed13_v1"
        ),
        science_optimal_aperture=_publication_from_root(
            target / "science_optimal_aperture_v1"
        ),
    )


STAMP_SCIENCE_ANALYSIS_REQUEST_SCHEMA_ID = (
    "et_mainsim.stamp_science_analysis_request.v2"
)
STAMP_SCIENCE_ANALYSIS_REQUEST_SCHEMA_VERSION = 2
STAMP_SCIENCE_FORMAL_PROFILE_ID = "et_stamp_science_formal_10s_v2"


def _formal_package_reproducibility_identity_v1(
    record: object,
    *,
    distribution_name: str,
) -> tuple[Any, ...] | None:
    """Normalize either a clean checkout or a verified installed wheel."""

    if not isinstance(record, Mapping):
        return None
    version = record.get("version")
    if not isinstance(version, str) or not version:
        return None
    commit = record.get("commit")
    dirty = record.get("dirty")
    if (
        isinstance(commit, str)
        and len(commit) == 40
        and all(character in "0123456789abcdef" for character in commit.lower())
        and dirty is False
    ):
        return ("clean_git_commit_v1", commit.lower(), version)
    if commit is not None or dirty is not None:
        return None

    identity = record.get("distribution_identity")
    if (
        not isinstance(identity, Mapping)
        or set(identity)
        != {
            "schema_id",
            "name",
            "version",
            "record_entry_count",
            "record_tree_sha256",
        }
        or identity.get("schema_id")
        != "et_mainsim.installed_distribution_identity.v1"
        or identity.get("name") != distribution_name
        or identity.get("version") != version
        or type(identity.get("record_entry_count")) is not int
        or identity["record_entry_count"] <= 0
    ):
        return None
    record_digest = identity.get("record_tree_sha256")
    if (
        not isinstance(record_digest, str)
        or len(record_digest) != 64
        or record_digest != record_digest.lower()
        or any(character not in "0123456789abcdef" for character in record_digest)
    ):
        return None
    return (
        "verified_installed_distribution_v1",
        distribution_name,
        version,
        identity["record_entry_count"],
        record_digest,
    )


def collect_formal_analysis_code_identity_v1(
    repo_root: Path | str | None = None,
) -> dict[str, Any]:
    """Collect trusted provenance from a checkout or installed wheel."""

    root = (
        Path(__file__).resolve().parents[2]
        if repo_root is None
        else Path(repo_root).expanduser().resolve()
    )
    provenance = collect_provenance(root)
    for provenance_name, distribution_name in (
        ("et_mainsim", "et-mainsim"),
        ("photsim7", "photsim7"),
    ):
        if (
            _formal_package_reproducibility_identity_v1(
                provenance.get(provenance_name),
                distribution_name=distribution_name,
            )
            is None
        ):
            raise StampScienceAnalysisContractError(
                "formal analysis requires clean Git commits or verified installed "
                "distributions for ET-mainsim and Photsim7"
            )
    dependencies = {}
    for distribution in ("numpy", "h5py", "astropy", "matplotlib", "torch"):
        try:
            dependencies[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as error:
            raise StampScienceAnalysisContractError(
                f"formal analysis dependency is not installed: {distribution}"
            ) from error
    return {
        "schema_id": "et_mainsim.formal_analysis_code_identity.v1",
        "schema_version": 1,
        "provenance": provenance,
        "analysis_dependencies": dependencies,
    }


@dataclass(frozen=True)
class _ResolvedProductionSource:
    manifest_path: Path
    manifest_binding: Mapping[str, Any]
    manifest: Mapping[str, Any]
    run_id: str
    source_identity: Mapping[str, str]
    target: Mapping[str, Any]
    factor_snapshot_path: Path
    factor_snapshot_binding: Mapping[str, Any]
    read_noise_e_per_raw_pixel: float
    quantization_noise_e_per_raw_pixel: float


@dataclass(frozen=True)
class StampScienceAnalysisBundleDiscovery:
    raw_bundle_paths: tuple[Path, ...]
    direct_coadd_bundle_paths: Mapping[int, tuple[Path, ...]]
    shard_ids: tuple[int, ...]
    time_plan_identity: Mapping[str, Any]
    static_task_list_binding: Mapping[str, Any] | None = None
    gate_task_list_binding: Mapping[str, Any] | None = None


def _production_requires_publication_receipts_v1(
    production: _ResolvedProductionSource,
) -> bool:
    track = production.source_identity.get("production_track")
    delivery = production.manifest.get("delivery")
    if not isinstance(delivery, Mapping):
        raise StampScienceAnalysisContractError(
            "formal production delivery contract is invalid"
        )
    if track in {"aster", "varlc", "wdlc"}:
        if delivery.get("execution_mode") != "staged_local_scratch_v1":
            raise StampScienceAnalysisContractError(
                "formal science production requires staged_local_scratch_v1 delivery"
            )
        return True
    if track != "galaxy":
        raise StampScienceAnalysisContractError(
            "formal production track is unsupported"
        )
    try:
        execution_mode = delivery_execution_mode_from_manifest(
            production.manifest
        )
    except ValueError as error:
        raise StampScienceAnalysisContractError(
            "formal Galaxy production delivery execution_mode is invalid"
        ) from error
    return execution_mode == "staged_local_scratch_v1"


def _same_content_identity(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return bool(
        left.get("size_bytes") == right.get("size_bytes")
        and left.get("sha256") == right.get("sha256")
    )


def _resolved_child_file(root: Path, relative: object, *, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise StampScienceAnalysisContractError(f"{label} relative path is invalid")
    candidate_relative = Path(relative)
    if candidate_relative.is_absolute() or ".." in candidate_relative.parts:
        raise StampScienceAnalysisContractError(f"{label} must remain under run root")
    candidate = (root / candidate_relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise StampScienceAnalysisContractError(
            f"{label} escapes the production run root"
        ) from error
    if not candidate.is_file() or candidate.is_symlink():
        raise StampScienceAnalysisContractError(
            f"{label} must be a real regular file"
        )
    return candidate


def _quantity_value(
    value: object,
    *,
    label: str,
    unit: str,
) -> float:
    if not isinstance(value, Mapping) or set(value) != {"unit", "value"}:
        raise StampScienceAnalysisContractError(
            f"production manifest {label} quantity is invalid"
        )
    if value.get("unit") != unit:
        raise StampScienceAnalysisContractError(
            f"production manifest {label} unit must be {unit!r}"
        )
    return _finite_nonnegative(value.get("value"), name=label)


def _resolve_production_source_v1(
    production_manifest: Path,
    *,
    source_id: str,
    expected_binding: Mapping[str, Any] | None = None,
) -> _ResolvedProductionSource:
    manifest_path = production_manifest.expanduser().resolve()
    binding = _cli_file_binding(manifest_path)
    if expected_binding is not None and dict(binding) != dict(expected_binding):
        raise StampScienceAnalysisContractError(
            "production_manifest identity/path drift detected"
        )
    initial_stat = _FileStat.from_path(manifest_path)
    try:
        manifest_bytes = manifest_path.read_bytes()
        production = json.loads(manifest_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StampScienceAnalysisContractError(
            "production manifest is not valid bound UTF-8 JSON"
        ) from error
    if (
        not isinstance(production, dict)
        or _FileStat.from_path(manifest_path) != initial_stat
        or len(manifest_bytes) != binding["identity"]["size_bytes"]
        or hashlib.sha256(manifest_bytes).hexdigest()
        != binding["identity"]["sha256"]
    ):
        raise StampScienceAnalysisContractError(
            "production manifest changed during byte-bound parsing"
        )
    schema_version = production.get("schema_version")
    if type(schema_version) is not int:
        raise StampScienceAnalysisContractError(
            "production manifest schema/version is unsupported"
        )
    schema = (production.get("schema_id"), schema_version)
    if schema == ("et_mainsim.science_stamp_production.v1", 1):
        track_value = production.get("production_track")
        if track_value not in {"aster", "varlc", "wdlc"}:
            raise StampScienceAnalysisContractError(
                "science production manifest has an unsupported production_track"
            )
        production_track = str(track_value)
    elif schema in {
        ("et_mainsim.galaxy_stamp_production.v1", 2),
        ("et_mainsim.galaxy_stamp_production.v1", 3),
    }:
        production_track = "galaxy"
    else:
        raise StampScienceAnalysisContractError(
            "production manifest schema/version is unsupported"
        )
    run_id = production.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise StampScienceAnalysisContractError("production manifest run_id is invalid")
    targets = production.get("targets")
    if not isinstance(targets, list):
        raise StampScienceAnalysisContractError("production manifest targets are invalid")
    matches = [
        item
        for item in targets
        if isinstance(item, Mapping)
        and str(item.get("source_id", item.get("source_id_int64"))) == source_id
    ]
    if len(matches) != 1:
        raise StampScienceAnalysisContractError(
            "production manifest must contain exactly one requested source"
        )
    target = dict(matches[0])
    if production_track == "galaxy":
        source_identity = {
            "production_track": "galaxy",
            "namespace": "gaia_dr3",
            "external_source_id": source_id,
            "source_id": source_id,
        }
    else:
        namespace = target.get("source_id_namespace")
        external = target.get("external_source_id")
        if (
            not isinstance(namespace, str)
            or not namespace
            or not isinstance(external, str)
            or not external
        ):
            raise StampScienceAnalysisContractError(
                "science target lacks its namespace/external source identity"
            )
        source_identity = {
            "production_track": production_track,
            "namespace": namespace,
            "external_source_id": external,
            "source_id": source_id,
        }
    snapshot_path = _resolved_child_file(
        manifest_path.parent,
        target.get("factor_snapshot_relative_path"),
        label="factor snapshot",
    )
    snapshot_binding = _cli_file_binding(snapshot_path)
    recorded_snapshot = target.get("factor_snapshot")
    if not isinstance(recorded_snapshot, Mapping) or not _same_content_identity(
        snapshot_binding["identity"], recorded_snapshot
    ):
        raise StampScienceAnalysisContractError(
            "factor snapshot identity differs from the production manifest"
        )
    delivery = production.get("delivery")
    spec = production.get("simulation_spec_base")
    if not isinstance(delivery, Mapping) or not isinstance(spec, Mapping):
        raise StampScienceAnalysisContractError(
            "production manifest lacks frozen delivery/simulation spec"
        )
    if (
        production_track in {"aster", "varlc", "wdlc"}
        and delivery.get("execution_mode") != "staged_local_scratch_v1"
    ):
        raise StampScienceAnalysisContractError(
            "formal science production requires staged_local_scratch_v1 delivery"
        )
    raw_seconds = _finite_nonnegative(
        delivery.get("raw_exposure_seconds"),
        name="delivery.raw_exposure_seconds",
    )
    cadence_seconds = delivery.get("cadence_seconds")
    observation = spec.get("observation")
    readout = spec.get("readout")
    if not isinstance(observation, Mapping) or not isinstance(readout, Mapping):
        raise StampScienceAnalysisContractError(
            "production simulation spec lacks observation/readout"
        )
    exposure = _quantity_value(
        observation.get("exposure_duration"),
        label="observation.exposure_duration",
        unit="s",
    )
    if (
        not math.isclose(raw_seconds, 10.0, rel_tol=0.0, abs_tol=1e-12)
        or not math.isclose(exposure, 10.0, rel_tol=0.0, abs_tol=1e-12)
        or cadence_seconds != [30.0, 60.0, 120.0, 300.0]
    ):
        raise StampScienceAnalysisContractError(
            "production manifest conflicts with the formal 10-second cadence profile"
        )
    read_noise = _quantity_value(
        readout.get("readout_noise"),
        label="readout.readout_noise",
        unit="electron / pix",
    )
    adc_enabled = readout.get("enable_adc_digitization")
    adc_rounded = readout.get("adc_round_values")
    if not isinstance(adc_enabled, (bool, np.bool_)) or not isinstance(
        adc_rounded, (bool, np.bool_)
    ):
        raise StampScienceAnalysisContractError(
            "production readout ADC switches must be boolean"
        )
    if bool(adc_enabled) and bool(adc_rounded):
        gain = _quantity_value(
            readout.get("gain_electrons_per_adu"),
            label="readout.gain_electrons_per_adu",
            unit="electron / adu",
        )
        quantization_noise = gain / math.sqrt(12.0)
    else:
        quantization_noise = 0.0
    return _ResolvedProductionSource(
        manifest_path=manifest_path,
        manifest_binding=binding,
        manifest=production,
        run_id=run_id,
        source_identity=source_identity,
        target=target,
        factor_snapshot_path=snapshot_path,
        factor_snapshot_binding=snapshot_binding,
        read_noise_e_per_raw_pixel=read_noise,
        quantization_noise_e_per_raw_pixel=quantization_noise,
    )


def _load_bound_time_plan_v1(
    production: _ResolvedProductionSource,
) -> tuple[ContinuousTimeShardPlan, dict[str, Any]]:
    delivery = production.manifest.get("delivery")
    if not isinstance(delivery, Mapping):
        raise StampScienceAnalysisContractError(
            "production manifest has no delivery time plan"
        )
    time_plan_path = _resolved_child_file(
        production.manifest_path.parent,
        delivery.get("time_plan_relative_path"),
        label="time shard plan",
    )
    expected_identity = delivery.get("time_plan_identity")
    actual_identity = _file_identity(time_plan_path)
    if not isinstance(expected_identity, Mapping) or not _same_content_identity(
        actual_identity,
        expected_identity,
    ):
        raise StampScienceAnalysisContractError(
            "time shard plan identity differs from the production manifest"
        )
    initial_stat = _FileStat.from_path(time_plan_path)
    try:
        raw = time_plan_path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
        plan = ContinuousTimeShardPlan.from_manifest_dict(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise StampScienceAnalysisContractError(
            "could not parse the frozen production time shard plan"
        ) from error
    if (
        _FileStat.from_path(time_plan_path) != initial_stat
        or len(raw) != actual_identity["size_bytes"]
        or hashlib.sha256(raw).hexdigest() != actual_identity["sha256"]
        or plan.raw_exposure_seconds != 10.0
        or plan.coadd_sizes != (3, 6, 12, 30)
    ):
        raise StampScienceAnalysisContractError(
            "time shard plan changed or conflicts with the formal profile"
        )
    return plan, actual_identity


def _load_bound_analysis_task_list_v1(
    path: Path,
    *,
    expected_path: Path,
    expected_case: str,
    production: _ResolvedProductionSource,
    label: str,
) -> tuple[dict[str, Any], tuple[tuple[str, int], ...]]:
    resolved = path.expanduser().resolve()
    if resolved != expected_path.resolve():
        expected_relative = expected_path.resolve().relative_to(
            production.manifest_path.parent.resolve()
        )
        raise StampScienceAnalysisContractError(
            f"formal {label} task list must be {expected_relative.as_posix()}"
        )
    if not resolved.is_file() or resolved.is_symlink():
        raise StampScienceAnalysisContractError(
            f"formal {label} task list is missing or unsafe"
        )
    binding = _cli_file_binding(resolved)
    task_stat = _FileStat.from_path(resolved)
    try:
        task_bytes = resolved.read_bytes()
        payload = json.loads(task_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StampScienceAnalysisContractError(
            f"formal {label} task list is not valid UTF-8 JSON"
        ) from error
    if (
        not isinstance(payload, Mapping)
        or _FileStat.from_path(resolved) != task_stat
        or len(task_bytes) != binding["identity"]["size_bytes"]
        or hashlib.sha256(task_bytes).hexdigest()
        != binding["identity"]["sha256"]
        or payload.get("schema_id")
        != "et_mainsim.science_stamp_task_list.v1"
        or type(payload.get("schema_version")) is not int
        or payload.get("schema_version") != 1
        or payload.get("case") != expected_case
        or not isinstance(payload.get("production_manifest_identity"), Mapping)
        or not _same_content_identity(
            payload["production_manifest_identity"],
            production.manifest_binding["identity"],
        )
        or not isinstance(payload.get("tasks"), list)
    ):
        raise StampScienceAnalysisContractError(
            f"formal {label} task list schema/production identity is invalid"
        )
    tasks: list[tuple[str, int]] = []
    for task in payload["tasks"]:
        if (
            not isinstance(task, Mapping)
            or set(task) != {"source_id", "shard_id"}
            or isinstance(task.get("source_id"), bool)
            or isinstance(task.get("shard_id"), bool)
            or not isinstance(task.get("source_id"), int)
            or not isinstance(task.get("shard_id"), int)
        ):
            raise StampScienceAnalysisContractError(
                f"formal {label} task list contains an invalid task"
            )
        tasks.append((str(task["source_id"]), int(task["shard_id"])))
    return binding, tuple(tasks)


def discover_stamp_science_analysis_bundles_v1(
    production_manifest: Path | str,
    *,
    source_id: str,
    case: Literal["static", "injected"] | str,
    shard_ids: Sequence[int] | None = None,
    static_task_list: Path | str | None = None,
    gate_task_list: Path | str | None = None,
) -> StampScienceAnalysisBundleDiscovery:
    """Discover the exact formal raw/coadd matrix from the frozen run layout."""

    source_text = str(source_id).strip()
    case_text = str(case).strip().lower()
    if not source_text or not source_text.isdigit() or case_text not in {
        "static",
        "injected",
    }:
        raise StampScienceAnalysisContractError(
            "formal bundle discovery source_id/case is invalid"
        )
    production = _resolve_production_source_v1(
        Path(production_manifest),
        source_id=source_text,
    )
    plan, time_plan_identity = _load_bound_time_plan_v1(production)
    by_id = {item.shard_id: item for item in plan.shards}
    delivery_root = (
        production.manifest_path.parent
        / "cases"
        / case_text
        / "stamps"
        / f"target_{source_text}"
        / "delivery"
    )
    if not delivery_root.is_dir() or delivery_root.is_symlink():
        raise StampScienceAnalysisContractError(
            "formal target delivery directory is missing or unsafe"
        )
    if case_text == "injected":
        if static_task_list is not None:
            raise StampScienceAnalysisContractError(
                "formal injected discovery forbids a static task list"
            )
        task_list_binding = None
        if gate_task_list is None:
            if shard_ids is not None:
                raise StampScienceAnalysisContractError(
                    "formal full injected discovery requires the complete "
                    "time-shard plan"
                )
            selected_ids = tuple(sorted(by_id))
            gate_task_list_binding = None
        else:
            if (
                shard_ids is None
                or any(isinstance(value, (bool, np.bool_)) for value in shard_ids)
                or tuple(int(value) for value in shard_ids) != tuple(range(6))
            ):
                raise StampScienceAnalysisContractError(
                    "formal injected gate analysis requires explicit shards 0..5"
                )
            expected_gate_path = (
                production.manifest_path.parent
                / "inputs"
                / "task_lists"
                / "injected_gate.json"
            )
            gate_task_list_binding, tasks = _load_bound_analysis_task_list_v1(
                Path(gate_task_list),
                expected_path=expected_gate_path,
                expected_case="injected",
                production=production,
                label="injected gate",
            )
            tail_shard_id = max(by_id)
            representative_shard_ids = (*range(6), tail_shard_id)
            expected_tasks = {
                (source_text, shard_id) for shard_id in representative_shard_ids
            }
            if (
                len(tasks) != len(expected_tasks)
                or set(tasks) != expected_tasks
                or any(value not in by_id for value in representative_shard_ids)
            ):
                raise StampScienceAnalysisContractError(
                    "formal injected gate task list must declare exactly "
                    f"shards 0..5 and tail shard {tail_shard_id} for only the "
                    "requested representative source"
                )
            selected_ids = tuple(range(6))
    else:
        if gate_task_list is not None:
            raise StampScienceAnalysisContractError(
                "formal static discovery forbids an injected gate task list"
            )
        task_list_path = (
            production.manifest_path.parent
            / "inputs"
            / "task_lists"
            / "static_representative.json"
            if static_task_list is None
            else Path(static_task_list).expanduser().resolve()
        )
        expected_task_list_path = (
            production.manifest_path.parent
            / "inputs"
            / "task_lists"
            / "static_representative.json"
        ).resolve()
        task_list_binding, tasks = _load_bound_analysis_task_list_v1(
            task_list_path,
            expected_path=expected_task_list_path,
            expected_case="static",
            production=production,
            label="static",
        )
        gate_task_list_binding = None
        selected_values = [
            shard_id for task_source, shard_id in tasks if task_source == source_text
        ]
        selected_ids = tuple(sorted(selected_values))
        if (
            not selected_ids
            or len(set(selected_ids)) != len(selected_ids)
            or any(value not in by_id for value in selected_ids)
            or (
                shard_ids is not None
                and tuple(sorted(int(value) for value in shard_ids))
                != selected_ids
            )
        ):
            raise StampScienceAnalysisContractError(
                "formal static task list has no exact declared source/shard subset"
            )
    raw_paths: list[Path] = []
    coadd_paths: dict[int, list[Path]] = {factor: [] for factor in (3, 6, 12, 30)}
    requires_receipt = _production_requires_publication_receipts_v1(production)
    for shard_id in selected_ids:
        shard = by_id[shard_id]
        shard_root = delivery_root / f"shard_{shard_id:05d}"
        if not shard_root.is_dir() or shard_root.is_symlink():
            raise StampScienceAnalysisContractError(
                f"formal delivery shard is missing or unsafe: {shard_id}"
            )
        receipt_path = shard_root / "publication_receipt.json"
        if requires_receipt and (
            not receipt_path.is_file() or receipt_path.is_symlink()
        ):
            raise StampScienceAnalysisContractError(
                f"staged formal science shard {shard_id} requires a publication receipt"
            )
        expected_names = {
            "raw.h5",
            "coadd_30s.h5",
            "coadd_60s.h5",
            "coadd_120s.h5",
            "coadd_300s.h5",
        }
        actual_names = {
            item.name
            for item in shard_root.iterdir()
            if item.name != "publication_receipt.json"
        }
        if actual_names != expected_names:
            raise StampScienceAnalysisContractError(
                f"formal shard {shard_id} does not contain the exact raw/coadd matrix"
            )
        member_paths = {
            1: shard_root / "raw.h5",
            3: shard_root / "coadd_30s.h5",
            6: shard_root / "coadd_60s.h5",
            12: shard_root / "coadd_120s.h5",
            30: shard_root / "coadd_300s.h5",
        }
        for factor, member in member_paths.items():
            if not member.is_file() or member.is_symlink():
                raise StampScienceAnalysisContractError(
                    f"formal shard {shard_id} member is missing or unsafe"
                )
            header = _read_input_header(member)
            expected_kind = "raw" if factor == 1 else "coadd"
            if (
                header.formal.product_kind != expected_kind
                or header.formal.coadd_factor != factor
                or header.formal.first_raw_frame_start != shard.raw_start_index
                or header.formal.last_raw_frame_stop != shard.raw_stop_index
                or header.formal.frame_count != shard.raw_frame_count // factor
                or header.target_source_id != source_text
                or header.case != case_text
                or header.run_id != production.run_id
                or not math.isclose(
                    header.formal.first_time_start_seconds,
                    shard.raw_start_index * 10.0,
                    rel_tol=0.0,
                    abs_tol=_TIME_ATOL_SECONDS,
                )
                or not math.isclose(
                    header.formal.last_time_end_seconds,
                    shard.raw_stop_index * 10.0,
                    rel_tol=0.0,
                    abs_tol=_TIME_ATOL_SECONDS,
                )
            ):
                raise StampScienceAnalysisContractError(
                    f"formal shard {shard_id} member conflicts with its time/source identity"
                )
            recorded = header.production_manifest_content_identity
            if requires_receipt and recorded is None:
                raise StampScienceAnalysisContractError(
                    f"staged formal science shard {shard_id} HDF5 lacks production byte identity"
                )
            if recorded is not None and not _same_content_identity(
                recorded,
                production.manifest_binding["identity"],
            ):
                raise StampScienceAnalysisContractError(
                    f"formal shard {shard_id} binds different production bytes"
                )
            if requires_receipt and _resolve_input_byte_identity(header).get(
                "trust_scope"
            ) != "publisher_receipt_plus_stat_and_formal_header_v1":
                raise StampScienceAnalysisContractError(
                    f"staged formal science shard {shard_id} lacks trusted "
                    "publication receipt identity"
                )
        raw_paths.append(member_paths[1])
        for factor in coadd_paths:
            coadd_paths[factor].append(member_paths[factor])
    return StampScienceAnalysisBundleDiscovery(
        raw_bundle_paths=tuple(raw_paths),
        direct_coadd_bundle_paths={
            factor: tuple(paths) for factor, paths in coadd_paths.items()
        },
        shard_ids=selected_ids,
        time_plan_identity=time_plan_identity,
        static_task_list_binding=task_list_binding,
        gate_task_list_binding=gate_task_list_binding,
    )


def _formal_analysis_policy_v1() -> StampScienceAnalysisPolicy:
    return StampScienceAnalysisPolicy()


def _formal_code_identity_matches_execution_v1(
    recorded: Mapping[str, Any],
    current: Mapping[str, Any],
) -> bool:
    """Compare reproducibility invariants while allowing a different run host."""

    required_fields = {
        "schema_id",
        "schema_version",
        "provenance",
        "analysis_dependencies",
    }
    if (
        set(recorded) != required_fields
        or set(current) != required_fields
        or recorded.get("schema_id")
        != "et_mainsim.formal_analysis_code_identity.v1"
        or current.get("schema_id")
        != "et_mainsim.formal_analysis_code_identity.v1"
        or type(recorded.get("schema_version")) is not int
        or recorded["schema_version"] != 1
        or type(current.get("schema_version")) is not int
        or current["schema_version"] != 1
        or recorded.get("analysis_dependencies")
        != current.get("analysis_dependencies")
    ):
        return False
    recorded_provenance = recorded.get("provenance")
    current_provenance = current.get("provenance")
    if not isinstance(recorded_provenance, Mapping) or not isinstance(
        current_provenance, Mapping
    ):
        return False
    for provenance_name, distribution_name in (
        ("et_mainsim", "et-mainsim"),
        ("photsim7", "photsim7"),
    ):
        recorded_identity = _formal_package_reproducibility_identity_v1(
            recorded_provenance.get(provenance_name),
            distribution_name=distribution_name,
        )
        current_identity = _formal_package_reproducibility_identity_v1(
            current_provenance.get(provenance_name),
            distribution_name=distribution_name,
        )
        if recorded_identity is None or recorded_identity != current_identity:
            return False
    recorded_runtime = recorded_provenance.get("runtime")
    current_runtime = current_provenance.get("runtime")
    return bool(
        isinstance(recorded_runtime, Mapping)
        and isinstance(current_runtime, Mapping)
        and recorded_runtime.get("python") == current_runtime.get("python")
    )


def validate_stamp_science_analysis_request_ready_v1(
    request: StampScienceAnalysisRequest,
) -> StampScienceAnalysisRequest:
    """Fail closed unless a request matches the frozen formal production profile."""

    if not isinstance(request, StampScienceAnalysisRequest):
        raise TypeError("request must be a StampScienceAnalysisRequest")
    expected = _formal_analysis_policy_v1().to_dict()
    if request.policy.to_dict() != expected:
        raise StampScienceAnalysisContractError(
            "request differs from the frozen formal analysis profile"
        )
    context = request.analysis_context
    if (
        context.get("formal_profile_id") != STAMP_SCIENCE_FORMAL_PROFILE_ID
        or not isinstance(context.get("production_track"), str)
        or not isinstance(context.get("source_identity"), Mapping)
        or not isinstance(context.get("production_manifest"), Mapping)
        or not isinstance(context.get("noise_model"), Mapping)
    ):
        raise StampScienceAnalysisContractError(
            "request lacks the identity-bound formal analysis profile"
        )
    production_binding = context["production_manifest"]
    if not isinstance(production_binding, Mapping):  # narrowed above.
        raise StampScienceAnalysisContractError("formal production binding is invalid")
    production_path_text = production_binding.get("path")
    source_identity = context["source_identity"]
    if (
        not isinstance(production_path_text, str)
        or not production_path_text
        or not isinstance(source_identity, Mapping)
        or context.get("source_id") != source_identity.get("source_id")
    ):
        raise StampScienceAnalysisContractError(
            "formal production/source identity is invalid"
        )
    production = _resolve_production_source_v1(
        Path(production_path_text),
        source_id=str(context["source_id"]),
        expected_binding=production_binding,
    )
    expected_noise = context["noise_model"]
    if (
        dict(source_identity) != dict(production.source_identity)
        or context.get("production_track")
        != production.source_identity["production_track"]
        or not isinstance(expected_noise, Mapping)
        or expected_noise.get("read_noise_e_per_raw_pixel")
        != production.read_noise_e_per_raw_pixel
        or expected_noise.get("quantization_noise_e_per_raw_pixel")
        != production.quantization_noise_e_per_raw_pixel
        or request.read_noise_e_per_pixel
        != production.read_noise_e_per_raw_pixel
        or request.quantization_noise_e_per_pixel
        != production.quantization_noise_e_per_raw_pixel
    ):
        raise StampScienceAnalysisContractError(
            "formal request identity/noise differs from current production bytes"
        )
    raw_headers = _read_series_headers(
        request.raw_bundle_paths,
        product_kind="raw",
        coadd_factor=1,
    )
    _validate_production_binding_for_headers(
        production.manifest_path,
        source_id=str(context["source_id"]),
        case=str(context.get("case")),
        headers=raw_headers,
    )
    if _production_requires_publication_receipts_v1(production):
        all_formal_headers = list(raw_headers)
        for factor, paths in request.direct_coadd_bundle_paths.items():
            all_formal_headers.extend(
                _read_series_headers(
                    paths,
                    product_kind="coadd",
                    coadd_factor=factor,
                )
            )
        if any(
            _resolve_input_byte_identity(header).get("trust_scope")
            != "publisher_receipt_plus_stat_and_formal_header_v1"
            for header in all_formal_headers
        ):
            raise StampScienceAnalysisContractError(
                "formal staged science requests require publication receipts for every HDF5 member"
            )
    discovery_context = context.get("input_discovery")
    if not isinstance(discovery_context, Mapping):
        raise StampScienceAnalysisContractError(
            "formal request input discovery audit is invalid"
        )
    if discovery_context.get("mode") != "canonical_production_layout_v1":
        raise StampScienceAnalysisContractError(
            "formal requests must use canonical production layout discovery"
        )
    if discovery_context.get("mode") == "canonical_production_layout_v1":
        recorded_shards = discovery_context.get("shard_ids")
        if not isinstance(recorded_shards, list):
            raise StampScienceAnalysisContractError(
                "formal canonical discovery lacks shard_ids"
            )
        recorded_task_list = discovery_context.get("static_task_list")
        recorded_gate_task_list = discovery_context.get("gate_task_list")
        if context.get("case") == "static":
            if (
                not isinstance(recorded_task_list, Mapping)
                or not isinstance(recorded_task_list.get("path"), str)
                or recorded_gate_task_list is not None
            ):
                raise StampScienceAnalysisContractError(
                    "formal static discovery lacks its bound task list"
                )
            task_list_path: Path | None = Path(recorded_task_list["path"])
            gate_task_list_path: Path | None = None
        else:
            if recorded_task_list is not None:
                raise StampScienceAnalysisContractError(
                    "formal injected discovery must not bind a static task list"
                )
            task_list_path = None
            if recorded_gate_task_list is None:
                gate_task_list_path = None
            elif (
                isinstance(recorded_gate_task_list, Mapping)
                and isinstance(recorded_gate_task_list.get("path"), str)
            ):
                gate_task_list_path = Path(recorded_gate_task_list["path"])
            else:
                raise StampScienceAnalysisContractError(
                    "formal injected gate task-list binding is invalid"
                )
        rediscovered = discover_stamp_science_analysis_bundles_v1(
            production.manifest_path,
            source_id=str(context["source_id"]),
            case=str(context.get("case")),
            shard_ids=(
                None
                if (
                    context.get("case") == "injected"
                    and recorded_gate_task_list is None
                )
                else tuple(recorded_shards)
            ),
            static_task_list=task_list_path,
            gate_task_list=gate_task_list_path,
        )
        if (
            list(rediscovered.shard_ids) != recorded_shards
            or tuple(request.raw_bundle_paths) != rediscovered.raw_bundle_paths
            or {
                factor: tuple(paths)
                for factor, paths in request.direct_coadd_bundle_paths.items()
            }
            != dict(rediscovered.direct_coadd_bundle_paths)
            or not _same_content_identity(
                rediscovered.time_plan_identity,
                discovery_context.get("time_plan_identity", {}),
            )
            or (
                context.get("case") == "static"
                and dict(rediscovered.static_task_list_binding or {})
                != dict(recorded_task_list)
            )
            or (
                context.get("case") == "injected"
                and (
                    None
                    if rediscovered.gate_task_list_binding is None
                    else dict(rediscovered.gate_task_list_binding)
                )
                != (
                    None
                    if recorded_gate_task_list is None
                    else dict(recorded_gate_task_list)
                )
            )
        ):
            raise StampScienceAnalysisContractError(
                "formal canonical bundle discovery differs from the frozen request"
            )
    if request.analysis_context.get("case") == "injected":
        q_source = request.raw_relative_flux_identity.get("source_identity")
        if not isinstance(q_source, Mapping) or any(
            q_source.get(name) != value
            for name, value in production.source_identity.items()
        ):
            raise StampScienceAnalysisContractError(
                "formal injected q lacks the complete production source identity"
            )
    elif request.analysis_context.get("case") == "static":
        if request.aperture_mode != "reuse_published":
            raise StampScienceAnalysisContractError(
                "formal static request must reuse the paired injected aperture"
            )
    else:
        raise StampScienceAnalysisContractError("formal request case is invalid")
    recorded_code = request.code_identity
    current_code = collect_formal_analysis_code_identity_v1()
    if not _formal_code_identity_matches_execution_v1(
        recorded_code,
        current_code,
    ):
        raise StampScienceAnalysisContractError(
            "formal request code identity differs from the executing checkout"
        )
    return request


def _write_bound_json_noreplace(path: Path, payload: Mapping[str, Any]) -> Path:
    target = path.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"analysis request already exists: {target}")
    partial = target.with_name(f".{target.name}.{uuid.uuid4().hex}.partial")
    try:
        with partial.open("xb") as stream:
            stream.write(
                (
                    json.dumps(
                        _json_safe(payload),
                        ensure_ascii=False,
                        sort_keys=True,
                        indent=2,
                        allow_nan=False,
                    )
                    + "\n"
                ).encode("utf-8")
            )
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(partial, target)
        except FileExistsError as error:
            raise FileExistsError(
                f"analysis request already exists: {target}"
            ) from error
        _fsync_directory(target.parent)
    finally:
        try:
            partial.unlink()
        except FileNotFoundError:
            pass
    return target


def write_stamp_science_analysis_request_v1(
    path: Path | str,
    *,
    production_manifest: Path | str,
    source_id: str,
    case: Literal["static", "injected"] | str,
    shard_ids: Sequence[int] | None = None,
    static_task_list: Path | str | None = None,
    gate_task_list: Path | str | None = None,
    output_dir: Path | str,
    aperture_analysis_manifest: Path | str | None = None,
) -> Path:
    """Write one strict, manifest-derived formal analysis request."""

    source_text = str(source_id).strip()
    case_text = str(case).strip().lower()
    if not source_text or case_text not in {"static", "injected"}:
        raise StampScienceAnalysisContractError("source_id/case is invalid")
    production = _resolve_production_source_v1(
        Path(production_manifest),
        source_id=source_text,
    )
    discovery = discover_stamp_science_analysis_bundles_v1(
        production.manifest_path,
        source_id=source_text,
        case=case_text,
        shard_ids=shard_ids,
        static_task_list=static_task_list,
        gate_task_list=gate_task_list,
    )
    raw_bundle_paths = discovery.raw_bundle_paths
    direct_coadd_bundle_paths = discovery.direct_coadd_bundle_paths
    input_discovery = {
        "mode": "canonical_production_layout_v1",
        "shard_ids": list(discovery.shard_ids),
        "time_plan_identity": dict(discovery.time_plan_identity),
        "static_task_list": (
            None
            if discovery.static_task_list_binding is None
            else dict(discovery.static_task_list_binding)
        ),
        "gate_task_list": (
            None
            if discovery.gate_task_list_binding is None
            else dict(discovery.gate_task_list_binding)
        ),
    }
    raw_bindings = [_cli_bundle_binding(item) for item in raw_bundle_paths]
    coadd_bindings = {
        str(int(factor)): [_cli_bundle_binding(item) for item in paths]
        for factor, paths in sorted(direct_coadd_bundle_paths.items())
    }
    policy = _formal_analysis_policy_v1()
    if set(int(item) for item in coadd_bindings) != set(policy.coadd_factors) - {1}:
        raise StampScienceAnalysisContractError(
            "formal analysis request must bind coadd factors 3/6/12/30"
        )
    raw_headers = _read_series_headers(
        tuple(Path(item["path"]) for item in raw_bindings),
        product_kind="raw",
        coadd_factor=1,
    )
    _validate_production_binding_for_headers(
        production.manifest_path,
        source_id=source_text,
        case=case_text,
        headers=raw_headers,
    )
    q: Mapping[str, Any]
    if case_text == "static":
        q = {"mode": "unity"}
        if aperture_analysis_manifest is None:
            raise StampScienceAnalysisContractError(
                "static formal request requires an injected aperture analysis manifest"
            )
        aperture = {
            "mode": "reuse_published",
            "analysis_manifest": _cli_file_binding(aperture_analysis_manifest),
        }
    else:
        q = {
            "mode": "factor_snapshot_npz",
            "snapshot": dict(production.factor_snapshot_binding),
        }
        if aperture_analysis_manifest is None:
            aperture = {"mode": "train"}
        else:
            aperture = {
                "mode": "reuse_published",
                "analysis_manifest": _cli_file_binding(aperture_analysis_manifest),
            }
    payload = {
        "schema_id": STAMP_SCIENCE_ANALYSIS_REQUEST_SCHEMA_ID,
        "schema_version": STAMP_SCIENCE_ANALYSIS_REQUEST_SCHEMA_VERSION,
        "formal_profile_id": STAMP_SCIENCE_FORMAL_PROFILE_ID,
        "production_manifest": dict(production.manifest_binding),
        "source_identity": dict(production.source_identity),
        "source_id": source_text,
        "case": case_text,
        "input_discovery": input_discovery,
        "raw_bundles": raw_bindings,
        "coadd_bundles": coadd_bindings,
        "q": q,
        "aperture": aperture,
        "output_dir": str(Path(output_dir).expanduser().resolve()),
        "read_noise_e_per_pixel": production.read_noise_e_per_raw_pixel,
        "quantization_noise_e_per_pixel": (
            production.quantization_noise_e_per_raw_pixel
        ),
        "noise_model": {
            "schema_id": _FORMAL_NOISE_MODEL_SCHEMA_ID,
            "source": _FORMAL_NOISE_MODEL_SOURCE,
            "read_noise_e_per_raw_pixel": production.read_noise_e_per_raw_pixel,
            "quantization_noise_e_per_raw_pixel": (
                production.quantization_noise_e_per_raw_pixel
            ),
            "quantization_formula": _FORMAL_NOISE_MODEL_QUANTIZATION_FORMULA,
        },
        "policy": policy.to_dict(),
        "code_identity": collect_formal_analysis_code_identity_v1(),
    }
    return _write_bound_json_noreplace(Path(path), payload)


def _require_exact_keys(
    value: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str] = frozenset(),
    label: str,
) -> None:
    keys = set(value)
    missing = required - keys
    unknown = keys - required - optional
    if missing or unknown:
        raise StampScienceAnalysisContractError(
            f"{label} keys differ; missing={sorted(missing)}, unknown={sorted(unknown)}"
        )


def _cli_file_binding(path: Path | str) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.is_file() or source.is_symlink():
        raise StampScienceAnalysisContractError(
            f"bound input must be a real regular file: {source}"
        )
    return {"path": str(source), "identity": _file_identity(source)}


def _cli_bundle_binding(path: Path | str) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.is_file() or source.is_symlink():
        raise StampScienceAnalysisContractError(
            f"bound formal bundle must be a real regular file: {source}"
        )
    header = _read_input_header(source)
    return {"path": str(source), "identity": _input_header_identity(header)}


def _verify_cli_binding(
    value: Any,
    *,
    label: str,
    bundle: bool,
) -> Path:
    if not isinstance(value, Mapping):
        raise StampScienceAnalysisContractError(f"{label} binding must be an object")
    _require_exact_keys(
        value,
        required={"path", "identity"},
        label=f"{label} binding",
    )
    path_text = value.get("path")
    identity = value.get("identity")
    if not isinstance(path_text, str) or not path_text or not isinstance(identity, Mapping):
        raise StampScienceAnalysisContractError(f"{label} binding is invalid")
    actual = _cli_bundle_binding(path_text) if bundle else _cli_file_binding(path_text)
    if actual != {"path": str(Path(path_text).expanduser().resolve()), "identity": dict(identity)}:
        raise StampScienceAnalysisContractError(f"{label} identity/path drift detected")
    return Path(actual["path"])


def _analysis_policy_from_json(value: Any) -> StampScienceAnalysisPolicy:
    if not isinstance(value, Mapping):
        raise StampScienceAnalysisContractError("policy must be an object")
    _require_exact_keys(
        value,
        required={
            "coadd_factors",
            "raw_exposure_seconds",
            "stream_batch_frames",
            "direct_coadd_samples_per_shard",
            "require_direct_coadd_parity",
            "photometry",
        },
        label="policy",
    )
    photometry_value = value["photometry"]
    if not isinstance(photometry_value, Mapping):
        raise StampScienceAnalysisContractError("policy.photometry must be an object")
    expected_photometry = set(StampSciencePhotometryPolicy.__dataclass_fields__)
    _require_exact_keys(
        photometry_value,
        required=expected_photometry,
        label="policy.photometry",
    )
    photometry_payload = dict(photometry_value)
    windows = photometry_payload.get("cdpp_windows_minutes")
    if not isinstance(windows, list):
        raise StampScienceAnalysisContractError(
            "policy.photometry.cdpp_windows_minutes must be a list"
        )
    photometry_payload["cdpp_windows_minutes"] = tuple(windows)
    photometry = StampSciencePhotometryPolicy(**photometry_payload)
    factors = value["coadd_factors"]
    if not isinstance(factors, list):
        raise StampScienceAnalysisContractError("policy.coadd_factors must be a list")
    return StampScienceAnalysisPolicy(
        coadd_factors=tuple(factors),
        raw_exposure_seconds=value["raw_exposure_seconds"],
        stream_batch_frames=value["stream_batch_frames"],
        direct_coadd_samples_per_shard=value[
            "direct_coadd_samples_per_shard"
        ],
        require_direct_coadd_parity=value["require_direct_coadd_parity"],
        photometry=photometry,
    )


def _validate_production_binding_for_headers(
    production_manifest: Path,
    *,
    source_id: str,
    case: str,
    headers: tuple[_InputHeader, ...],
) -> None:
    production = _read_json_object(
        production_manifest,
        label="production manifest",
    )
    schema_version = production.get("schema_version")
    schema = (production.get("schema_id"), schema_version)
    supported = {
        ("et_mainsim.science_stamp_production.v1", 1),
        ("et_mainsim.galaxy_stamp_production.v1", 2),
        ("et_mainsim.galaxy_stamp_production.v1", 3),
    }
    run_id = production.get("run_id")
    if (
        type(schema_version) is not int
        or schema not in supported
        or not isinstance(run_id, str)
        or not run_id
    ):
        raise StampScienceAnalysisContractError(
            "production manifest schema/version/run_id is unsupported"
        )
    actual_identity = _file_identity(production_manifest)
    for header in headers:
        if (
            header.target_source_id != source_id
            or header.case != case
            or header.run_id != run_id
        ):
            raise StampScienceAnalysisContractError(
                "formal bundle target/case/run differs from the production request"
            )
        recorded = Path(header.production_manifest_reference)
        if recorded.name != production_manifest.name:
            raise StampScienceAnalysisContractError(
                "formal bundle production-manifest filename differs from the request"
            )
        if recorded.parent.name not in {"", "."} and (
            recorded.parent.name != production_manifest.parent.name
        ):
            raise StampScienceAnalysisContractError(
                "formal bundle production-manifest run directory differs from the request"
            )
        recorded_identity = header.production_manifest_content_identity
        if recorded_identity is not None:
            if (
                recorded_identity.get("size_bytes")
                != actual_identity["size_bytes"]
                or recorded_identity.get("sha256") != actual_identity["sha256"]
            ):
                raise StampScienceAnalysisContractError(
                    "formal bundle binds a different production-manifest content identity"
                )


def _load_published_aperture_v1(
    analysis_dir: Path,
) -> tuple[ScienceApertureDefinition, dict[str, Any], dict[str, Any]]:
    import h5py

    validate_stamp_science_analysis_v1(analysis_dir)
    manifest_path = analysis_dir / "analysis_manifest.json"
    manifest = _read_json_object(manifest_path, label="published aperture manifest")
    contract = manifest.get("contract")
    if not isinstance(contract, dict):
        raise StampScienceAnalysisContractError(
            "published aperture manifest lacks its analysis contract"
        )
    if contract.get("analysis_product") != "science_optimal_aperture_v1":
        raise StampScienceAnalysisContractError(
            "published aperture analysis_product must be "
            "science_optimal_aperture_v1"
        )
    definition_payload = _read_json_object(
        analysis_dir / "aperture_definition.json",
        label="published aperture definition",
    )
    metadata = definition_payload.get("metadata")
    policy = contract.get("policy")
    photometry_policy = (
        policy.get("photometry") if isinstance(policy, Mapping) else None
    )
    contract_strategy = contract.get("background_strategy")
    metadata_strategy = (
        metadata.get("background_strategy") if isinstance(metadata, Mapping) else None
    )
    policy_strategy = (
        photometry_policy.get("background_strategy")
        if isinstance(photometry_policy, Mapping)
        else None
    )
    if (
        contract_strategy
        not in {
            _EXPECTATION_ONLY_BACKGROUND_STRATEGY,
            _LOCAL_DIAGNOSTIC_BACKGROUND_STRATEGY,
        }
        or metadata_strategy != contract_strategy
        or policy_strategy != contract_strategy
    ):
        raise StampScienceAnalysisContractError(
            "published aperture background strategy metadata/contract differs"
        )
    with h5py.File(analysis_dir / "photometry.h5", "r") as handle:
        group = handle["aperture"]
        persisted_background = np.asarray(group["background_mask"], dtype=bool)
        background_pixel_count = definition_payload.get("background_pixel_count")
        if (
            isinstance(background_pixel_count, bool)
            or not isinstance(background_pixel_count, int)
            or background_pixel_count != int(np.count_nonzero(persisted_background))
        ):
            raise StampScienceAnalysisContractError(
                "published aperture background mask/count differs"
            )
        if contract_strategy == _EXPECTATION_ONLY_BACKGROUND_STRATEGY:
            if np.any(persisted_background):
                raise StampScienceAnalysisContractError(
                    "expectation-only published aperture has local background pixels"
                )
            restored_background: NDArray[np.bool_] | None = None
        else:
            restored_background = persisted_background
        definition = ScienceApertureDefinition(
            aperture_mask=np.asarray(group["aperture_mask"], dtype=bool),
            background_mask=restored_background,
            signal_template_e=np.asarray(group["signal_template_e"], dtype=np.float64),
            noise_template_e=np.asarray(group["noise_template_e"], dtype=np.float64),
            training_raw_frame_indices=np.asarray(
                group["training_raw_frame_indices"], dtype=np.int64
            ),
            maximum_cumulative_snr=float(
                definition_payload["maximum_cumulative_snr"]
            ),
            algorithm=str(definition_payload["algorithm"]),
            signal_template_shape=tuple(
                int(item) for item in definition_payload["signal_template_shape"]
            ),
            target_peak_yx=(
                None
                if definition_payload.get("target_peak_yx") is None
                else tuple(
                    int(item) for item in definition_payload["target_peak_yx"]
                )
            ),
            metadata=(metadata if isinstance(metadata, Mapping) else {}),
        )
    source_identity = {
        "mode": "validated_published_injected_analysis_v1",
        "analysis_dir": str(analysis_dir.resolve()),
        "analysis_manifest": _cli_file_binding(manifest_path),
    }
    return _validate_frozen_aperture_definition(definition), source_identity, contract


def _load_factor_snapshot(
    path: Path,
    *,
    expected_source_identity: Mapping[str, Any],
    expected_file_identity: Mapping[str, Any] | None = None,
    first_raw_index: int,
    last_raw_index: int,
) -> tuple[NDArray[np.float64], dict[str, str]]:
    expected = _json_mapping(
        expected_source_identity,
        name="expected factor snapshot source identity",
    )
    required_identity_keys = {
        "production_track",
        "namespace",
        "external_source_id",
        "source_id",
    }
    if set(expected) != required_identity_keys or any(
        not isinstance(expected[name], str) or not expected[name]
        for name in required_identity_keys
    ):
        raise StampScienceAnalysisContractError(
            "expected factor snapshot source identity is incomplete"
        )
    initial_stat = _FileStat.from_path(path)
    if expected_file_identity is not None and not _same_content_identity(
        _file_identity(path), expected_file_identity
    ):
        raise StampScienceAnalysisContractError(
            "factor snapshot byte identity differs from the production manifest"
        )
    try:
        track = expected["production_track"]
        if track == "galaxy":
            snapshot = read_galaxy_factor_snapshot(path)
            actual_identity = {
                "production_track": "galaxy",
                "namespace": "gaia_dr3",
                "external_source_id": str(snapshot.source_id),
                "source_id": str(snapshot.source_id),
                "snapshot_schema_id": GALAXY_FACTOR_SNAPSHOT_SCHEMA_ID,
            }
        elif track in {"aster", "varlc", "wdlc"}:
            snapshot = read_science_factor_snapshot(path)
            actual_identity = {
                "production_track": str(snapshot.metadata.get("track")),
                "namespace": snapshot.namespace,
                "external_source_id": snapshot.external_source_id,
                "source_id": str(snapshot.source_id_int64),
                "snapshot_schema_id": SCIENCE_FACTOR_SNAPSHOT_SCHEMA_ID,
            }
        else:
            raise StampScienceAnalysisContractError(
                "factor snapshot production_track is unsupported"
            )
        factors = _positive_factor_vector(snapshot.factors)
    except (OSError, ValueError) as error:
        raise StampScienceAnalysisContractError(
            f"could not read frozen factor snapshot: {path}"
        ) from error
    if _FileStat.from_path(path) != initial_stat or (
        expected_file_identity is not None
        and not _same_content_identity(_file_identity(path), expected_file_identity)
    ):
        raise StampScienceAnalysisContractError(
            "factor snapshot changed during byte-bound parsing"
        )
    if any(
        actual_identity[name] != expected[name]
        for name in required_identity_keys
    ):
        raise StampScienceAnalysisContractError(
            "factor snapshot source identity differs from the production manifest"
        )
    if factors.size < last_raw_index:
        raise StampScienceAnalysisContractError(
            "factor snapshot does not cover the requested absolute raw interval"
        )
    return (
        np.asarray(factors[first_raw_index:last_raw_index], dtype=np.float64),
        actual_identity,
    )


def load_stamp_science_analysis_request_v1(
    path: Path | str,
) -> StampScienceAnalysisRequest:
    """Load one strictly identity-bound request JSON for the module CLI."""

    request_path = Path(path).expanduser().resolve()
    if not request_path.is_file() or request_path.is_symlink():
        raise StampScienceAnalysisContractError(
            "analysis request must be a real regular JSON file"
        )
    payload = _read_json_object(request_path, label="analysis request")
    _require_exact_keys(
        payload,
        required={
            "schema_id",
            "schema_version",
            "formal_profile_id",
            "production_manifest",
            "source_identity",
            "source_id",
            "case",
            "input_discovery",
            "raw_bundles",
            "coadd_bundles",
            "q",
            "aperture",
            "output_dir",
            "read_noise_e_per_pixel",
            "quantization_noise_e_per_pixel",
            "noise_model",
            "policy",
            "code_identity",
        },
        label="analysis request",
    )
    if (
        payload["schema_id"] != STAMP_SCIENCE_ANALYSIS_REQUEST_SCHEMA_ID
        or payload["schema_version"]
        != STAMP_SCIENCE_ANALYSIS_REQUEST_SCHEMA_VERSION
    ):
        raise StampScienceAnalysisContractError(
            "unsupported analysis request schema/version"
        )
    source_id = payload["source_id"]
    case = payload["case"]
    if not isinstance(source_id, str) or not source_id or case not in {"static", "injected"}:
        raise StampScienceAnalysisContractError(
            "request source_id/case is invalid"
        )
    if not isinstance(payload["input_discovery"], Mapping):
        raise StampScienceAnalysisContractError(
            "request input_discovery must be an object"
        )
    production_manifest = _verify_cli_binding(
        payload["production_manifest"],
        label="production_manifest",
        bundle=False,
    )
    production = _resolve_production_source_v1(
        production_manifest,
        source_id=source_id,
        expected_binding=payload["production_manifest"],
    )
    if (
        payload["formal_profile_id"] != STAMP_SCIENCE_FORMAL_PROFILE_ID
        or payload["source_identity"] != production.source_identity
    ):
        raise StampScienceAnalysisContractError(
            "request formal profile/source identity differs from production"
        )
    noise_model = payload["noise_model"]
    expected_noise_model = {
        "schema_id": _FORMAL_NOISE_MODEL_SCHEMA_ID,
        "source": _FORMAL_NOISE_MODEL_SOURCE,
        "read_noise_e_per_raw_pixel": production.read_noise_e_per_raw_pixel,
        "quantization_noise_e_per_raw_pixel": (
            production.quantization_noise_e_per_raw_pixel
        ),
        "quantization_formula": _FORMAL_NOISE_MODEL_QUANTIZATION_FORMULA,
    }
    if (
        noise_model != expected_noise_model
        or payload["read_noise_e_per_pixel"]
        != production.read_noise_e_per_raw_pixel
        or payload["quantization_noise_e_per_pixel"]
        != production.quantization_noise_e_per_raw_pixel
    ):
        raise StampScienceAnalysisContractError(
            "request noise parameters differ from the frozen production manifest"
        )
    raw_value = payload["raw_bundles"]
    if not isinstance(raw_value, list) or not raw_value:
        raise StampScienceAnalysisContractError("raw_bundles must be a non-empty list")
    raw_paths = tuple(
        _verify_cli_binding(item, label="raw_bundle", bundle=True)
        for item in raw_value
    )
    coadd_value = payload["coadd_bundles"]
    if not isinstance(coadd_value, Mapping):
        raise StampScienceAnalysisContractError("coadd_bundles must be an object")
    coadd_paths: dict[int, tuple[Path, ...]] = {}
    for factor_text, bindings in coadd_value.items():
        if not isinstance(factor_text, str) or not factor_text.isdigit():
            raise StampScienceAnalysisContractError(
                "coadd_bundles keys must be decimal factor strings"
            )
        if not isinstance(bindings, list) or not bindings:
            raise StampScienceAnalysisContractError(
                f"coadd factor {factor_text} must bind a non-empty list"
            )
        coadd_paths[int(factor_text)] = tuple(
            _verify_cli_binding(item, label=f"coadd_{factor_text}", bundle=True)
            for item in bindings
        )
    policy = _analysis_policy_from_json(payload["policy"])
    raw_headers = _read_series_headers(
        raw_paths,
        product_kind="raw",
        coadd_factor=1,
    )
    _validate_production_binding_for_headers(
        production_manifest,
        source_id=source_id,
        case=case,
        headers=raw_headers,
    )
    first_raw = raw_headers[0].formal.first_raw_frame_start
    last_raw = raw_headers[-1].formal.last_raw_frame_stop
    q_value = payload["q"]
    if not isinstance(q_value, Mapping) or "mode" not in q_value:
        raise StampScienceAnalysisContractError("q must be a mode-tagged object")
    if case == "static":
        _require_exact_keys(q_value, required={"mode"}, label="static q")
        if q_value["mode"] != "unity":
            raise StampScienceAnalysisContractError(
                "static request q.mode must be 'unity'"
            )
        q = np.ones(last_raw - first_raw, dtype=np.float64)
        q_identity: dict[str, Any] = {
            "mode": "unity",
            "absolute_raw_frame_interval": [first_raw, last_raw],
        }
    else:
        _require_exact_keys(
            q_value,
            required={"mode", "snapshot"},
            label="injected q",
        )
        if q_value["mode"] != "factor_snapshot_npz":
            raise StampScienceAnalysisContractError(
                "injected request q.mode must be 'factor_snapshot_npz'"
            )
        snapshot = _verify_cli_binding(
            q_value["snapshot"],
            label="factor_snapshot",
            bundle=False,
        )
        if dict(q_value["snapshot"]) != dict(production.factor_snapshot_binding):
            raise StampScienceAnalysisContractError(
                "request factor snapshot differs from the production target"
            )
        q, snapshot_source_identity = _load_factor_snapshot(
            snapshot,
            expected_source_identity=production.source_identity,
            expected_file_identity=production.factor_snapshot_binding["identity"],
            first_raw_index=first_raw,
            last_raw_index=last_raw,
        )
        q_identity = {
            "mode": "factor_snapshot_npz",
            "snapshot": dict(q_value["snapshot"]),
            "source_identity": snapshot_source_identity,
            "absolute_raw_frame_interval": [first_raw, last_raw],
        }
    aperture_value = payload["aperture"]
    if not isinstance(aperture_value, Mapping) or "mode" not in aperture_value:
        raise StampScienceAnalysisContractError(
            "aperture must be a mode-tagged object"
        )
    frozen_aperture: ScienceApertureDefinition | None = None
    aperture_source: dict[str, Any] = {}
    aperture_mode = aperture_value["mode"]
    if aperture_mode == "train":
        _require_exact_keys(aperture_value, required={"mode"}, label="train aperture")
        if case == "static":
            raise StampScienceAnalysisContractError(
                "static request must reuse an injected published aperture"
            )
    elif aperture_mode == "reuse_published":
        _require_exact_keys(
            aperture_value,
            required={"mode", "analysis_manifest"},
            label="reused aperture",
        )
        manifest_path = _verify_cli_binding(
            aperture_value["analysis_manifest"],
            label="aperture analysis_manifest",
            bundle=False,
        )
        analysis_dir = manifest_path.parent
        frozen_aperture, aperture_source, aperture_contract = (
            _load_published_aperture_v1(analysis_dir)
        )
        context = aperture_contract.get("analysis_context")
        paired_production = (
            context.get("production_manifest")
            if isinstance(context, Mapping)
            else None
        )
        paired_source = (
            context.get("source_identity")
            if isinstance(context, Mapping)
            else None
        )
        if (
            not isinstance(context, Mapping)
            or context.get("case") != "injected"
            or context.get("source_id") != source_id
            or context.get("production_track")
            != production.source_identity["production_track"]
            or paired_source != production.source_identity
            or paired_production != production.manifest_binding
        ):
            raise StampScienceAnalysisContractError(
                "reused aperture must come from the same production manifest "
                "and paired injected source identity"
            )
    else:
        raise StampScienceAnalysisContractError(
            "aperture.mode must be 'train' or 'reuse_published'"
        )
    output_text = payload["output_dir"]
    if not isinstance(output_text, str) or not output_text:
        raise StampScienceAnalysisContractError("output_dir must be a non-empty path")
    code_identity = payload["code_identity"]
    if not isinstance(code_identity, Mapping):
        raise StampScienceAnalysisContractError("code_identity must be an object")
    analysis_context = {
        "formal_profile_id": STAMP_SCIENCE_FORMAL_PROFILE_ID,
        "request_manifest": _cli_file_binding(request_path),
        "production_manifest": dict(production.manifest_binding),
        "production_track": production.source_identity["production_track"],
        "source_identity": dict(production.source_identity),
        "noise_model": dict(expected_noise_model),
        "source_id": source_id,
        "case": case,
        "input_discovery": dict(payload["input_discovery"]),
        "raw_bundles": list(raw_value),
        "coadd_bundles": dict(coadd_value),
    }
    return StampScienceAnalysisRequest(
        raw_bundle_paths=raw_paths,
        direct_coadd_bundle_paths=coadd_paths,
        output_dir=output_text,
        raw_relative_flux=q,
        raw_relative_flux_identity=q_identity,
        code_identity=code_identity,
        analysis_context=analysis_context,
        read_noise_e_per_pixel=production.read_noise_e_per_raw_pixel,
        quantization_noise_e_per_pixel=(
            production.quantization_noise_e_per_raw_pixel
        ),
        policy=policy,
        aperture_mode=aperture_mode,
        frozen_aperture=frozen_aperture,
        aperture_source_identity=aperture_source,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m et_mainsim.stamp_science_analysis"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="run one identity-bound request")
    run_parser.add_argument("--request", type=Path, required=True)
    validate_parser = subparsers.add_parser(
        "validate-request",
        help="validate the frozen formal request without running analysis",
    )
    validate_parser.add_argument("--request", type=Path, required=True)
    write_parser = subparsers.add_parser(
        "write-request",
        help="derive and atomically freeze one formal request",
    )
    write_parser.add_argument("--output", type=Path, required=True)
    write_parser.add_argument("--production-manifest", type=Path, required=True)
    write_parser.add_argument("--source-id", required=True)
    write_parser.add_argument("--case", choices=("static", "injected"), required=True)
    write_parser.add_argument(
        "--shard-id",
        type=int,
        action="append",
        help="static-only canonical shard selection; injected always discovers all",
    )
    write_parser.add_argument(
        "--static-task-list",
        type=Path,
        help=(
            "identity-bound et_mainsim.science_stamp_task_list.v1; defaults to "
            "inputs/task_lists/static_representative.json for static"
        ),
    )
    write_parser.add_argument(
        "--gate-task-list",
        type=Path,
        help=(
            "injected-only admission gate bound to "
            "inputs/task_lists/injected_gate.json; admits exactly shards 0..5"
        ),
    )
    write_parser.add_argument("--analysis-output", type=Path, required=True)
    write_parser.add_argument("--aperture-analysis-manifest", type=Path)
    arguments = parser.parse_args(argv)
    if arguments.command == "write-request":
        output = write_stamp_science_analysis_request_v1(
            arguments.output,
            production_manifest=arguments.production_manifest,
            source_id=arguments.source_id,
            case=arguments.case,
            shard_ids=arguments.shard_id,
            static_task_list=arguments.static_task_list,
            gate_task_list=arguments.gate_task_list,
            output_dir=arguments.analysis_output,
            aperture_analysis_manifest=arguments.aperture_analysis_manifest,
        )
        request = load_stamp_science_analysis_request_v1(output)
        validate_stamp_science_analysis_request_ready_v1(request)
        print(output)
        return 0
    request = load_stamp_science_analysis_request_v1(arguments.request)
    validate_stamp_science_analysis_request_ready_v1(request)
    if arguments.command == "validate-request":
        print(arguments.request.expanduser().resolve())
        return 0
    if arguments.command != "run":  # pragma: no cover - argparse enforces this.
        parser.error("unsupported command")
    publication = analyze_stamp_science_product_set_v1(request)
    print(publication.output_dir)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main().
    raise SystemExit(main())
