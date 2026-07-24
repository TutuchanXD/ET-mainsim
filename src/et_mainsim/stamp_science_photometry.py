"""Common science-aperture photometry for formal independent stamp products.

This module is the rectangular-HDF5 adapter around a deliberately small set of
legacy-validated numerical kernels.  It does not instantiate the historical
Analysis pickle workflow.  final_dn remains the only detector observation;
electron images and both background-subtracted light curves are derived
products.  ``background_expectation_e`` is the complete sky, scattered-light,
and dark-current expectation; dark current must never be added to it again.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntFlag
import math
from typing import Any, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .reference_photometry import ReferencePhotometryInput


SCIENCE_PHOTOMETRY_SCHEMA_ID = "et_mainsim.stamp_science_photometry.v2"
SCIENCE_PHOTOMETRY_SCHEMA_VERSION = 2
_LOCAL_BACKGROUND_POLICY_VERSION = 1
_LOCAL_BACKGROUND_ESTIMATOR = "per_frame_median"
_LOCAL_BACKGROUND_SIGMA_CLIPPING = "none"
_EXPECTATION_BACKGROUND_STRATEGY = "delivered_expectation_only"
_LOCAL_DIAGNOSTIC_BACKGROUND_STRATEGY = (
    "delivered_expectation_plus_local_diagnostic"
)
_BACKGROUND_STRATEGIES = {
    _EXPECTATION_BACKGROUND_STRATEGY,
    _LOCAL_DIAGNOSTIC_BACKGROUND_STRATEGY,
}


class SciencePhotometryContractError(ValueError):
    """Raised when a science-photometry input violates the frozen contract."""


class ScienceQualityFlag(IntFlag):
    """Cadence-level quality bits for the common science product."""

    OK = 0
    APERTURE_INVALID = 1 << 0
    APERTURE_SATURATED = 1 << 1
    APERTURE_COSMIC = 1 << 2
    INSUFFICIENT_BACKGROUND = 1 << 3
    CENTROID_UNAVAILABLE = 1 << 4


@dataclass(frozen=True)
class StampSciencePhotometryPolicy:
    """Frozen numerical policy shared by all four science tracks."""

    cdpp_windows_minutes: tuple[int, ...] = (30, 90, 390)
    minimum_coverage_fraction: float = 0.95
    minimum_accepted_bins: int = 10
    bin_origin_seconds: float = 0.0
    training_blocks_per_shard: int = 4
    training_block_frames: int = 64
    minimum_training_valid_fraction: float = 0.8
    background_strategy: str = _EXPECTATION_BACKGROUND_STRATEGY
    background_guard_pixels: int = 8
    background_border_pixels: int = 1
    minimum_background_pixels: int = 1_024
    local_background_policy_version: int = _LOCAL_BACKGROUND_POLICY_VERSION
    local_background_estimator: str = _LOCAL_BACKGROUND_ESTIMATOR
    local_background_sigma_clipping: str = _LOCAL_BACKGROUND_SIGMA_CLIPPING

    def __post_init__(self) -> None:
        windows = tuple(
            _positive_integer(value, name="cdpp window")
            for value in self.cdpp_windows_minutes
        )
        if not windows or len(set(windows)) != len(windows):
            raise SciencePhotometryContractError(
                "cdpp_windows_minutes must be unique positive integers"
            )
        try:
            coverage = float(self.minimum_coverage_fraction)
            training_fraction = float(self.minimum_training_valid_fraction)
            origin = float(self.bin_origin_seconds)
        except (TypeError, ValueError, OverflowError) as error:
            raise SciencePhotometryContractError(
                "photometry policy fractions and origin must be finite"
            ) from error
        if not math.isfinite(origin):
            raise SciencePhotometryContractError(
                "bin_origin_seconds must be finite"
            )
        if not math.isfinite(coverage) or not 0.0 < coverage <= 1.0:
            raise SciencePhotometryContractError(
                "minimum_coverage_fraction must be in (0, 1]"
            )
        if not math.isfinite(training_fraction) or not 0.0 < training_fraction <= 1.0:
            raise SciencePhotometryContractError(
                "minimum_training_valid_fraction must be in (0, 1]"
            )
        object.__setattr__(self, "cdpp_windows_minutes", windows)
        object.__setattr__(self, "minimum_coverage_fraction", coverage)
        object.__setattr__(self, "minimum_training_valid_fraction", training_fraction)
        object.__setattr__(self, "bin_origin_seconds", origin)
        if self.background_strategy not in _BACKGROUND_STRATEGIES:
            raise SciencePhotometryContractError(
                "background_strategy must be delivered_expectation_only or "
                "delivered_expectation_plus_local_diagnostic"
            )
        for name in (
            "minimum_accepted_bins",
            "training_blocks_per_shard",
            "training_block_frames",
            "minimum_background_pixels",
        ):
            object.__setattr__(
                self,
                name,
                _positive_integer(getattr(self, name), name=name),
            )
        for name in ("background_guard_pixels", "background_border_pixels"):
            object.__setattr__(
                self,
                name,
                _nonnegative_integer(getattr(self, name), name=name),
            )
        background_policy_version = _positive_integer(
            self.local_background_policy_version,
            name="local_background_policy_version",
        )
        if background_policy_version != _LOCAL_BACKGROUND_POLICY_VERSION:
            raise SciencePhotometryContractError(
                "local_background_policy_version must be 1"
            )
        if self.local_background_estimator != _LOCAL_BACKGROUND_ESTIMATOR:
            raise SciencePhotometryContractError(
                "local_background_estimator must be per_frame_median"
            )
        if (
            self.local_background_sigma_clipping
            != _LOCAL_BACKGROUND_SIGMA_CLIPPING
        ):
            raise SciencePhotometryContractError(
                "local_background_sigma_clipping must be none"
            )
        object.__setattr__(
            self,
            "local_background_policy_version",
            background_policy_version,
        )

    @property
    def local_background_enabled(self) -> bool:
        """Return whether the replaceable stamp-local diagnostic was requested."""

        return self.background_strategy == _LOCAL_DIAGNOSTIC_BACKGROUND_STRATEGY


@dataclass(frozen=True)
class ScienceApertureDefinition:
    """One frozen rectangular aperture selected by the legacy SNR kernel."""

    aperture_mask: NDArray[np.bool_]
    maximum_cumulative_snr: float
    algorithm: str
    signal_template_shape: tuple[int, int]
    background_mask: NDArray[np.bool_] | None = None
    signal_template_e: NDArray[np.float64] | None = None
    noise_template_e: NDArray[np.float64] | None = None
    target_peak_yx: tuple[int, int] | None = None
    training_raw_frame_indices: NDArray[np.int64] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScienceVariabilityModelResult:
    """Through-origin source model evaluated over delivered cadence intervals."""

    raw_factor_sum: NDArray[np.float64]
    fitted_flux_e: NDArray[np.float64]
    residual_e: NDArray[np.float64]
    residual_ppm: NDArray[np.float64]
    fit_scale_e_per_raw_factor: float
    fit_intercept_e: float
    valid_mask: NDArray[np.bool_]


@dataclass(frozen=True)
class ScienceFluxUncertaintyModelResult:
    """Auditable per-cadence variance components and gated uncertainty."""

    uncertainty_e: NDArray[np.float64]
    source_variance_e2: NDArray[np.float64]
    background_variance_e2: NDArray[np.float64]
    read_variance_e2: NDArray[np.float64]
    quantization_variance_e2: NDArray[np.float64]
    valid_mask: NDArray[np.bool_]
    coadd_factor: NDArray[np.int64]
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class SciencePhotometryResult:
    """Dual-background electron light curves and transparent quality arrays."""

    time_seconds: NDArray[np.float64]
    exposure_seconds: NDArray[np.float64] | None
    flux_expectation_bgsub_e: NDArray[np.float64]
    flux_local_bgsub_e: NDArray[np.float64]
    local_background_e_per_pixel: NDArray[np.float64]
    centroid_x: NDArray[np.float64]
    centroid_y: NDArray[np.float64]
    aperture_valid: NDArray[np.bool_]
    aperture_usable_pixel_count: NDArray[np.int64]
    aperture_invalid_pixel_count: NDArray[np.int64]
    saturated_pixel_count: NDArray[np.int64]
    cosmic_pixel_count: NDArray[np.int64]
    background_usable_pixel_count: NDArray[np.int64]
    quality_bitmask: NDArray[np.uint16]
    aperture_mask: NDArray[np.bool_]
    background_mask: NDArray[np.bool_]
    product_semantics: Mapping[str, Any]

    @property
    def aperture_pixel_count(self) -> int:
        return int(np.count_nonzero(self.aperture_mask))


def _finite_2d_float(
    value: ArrayLike,
    *,
    name: str,
    nonnegative: bool,
    positive: bool = False,
) -> NDArray[np.float64]:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise SciencePhotometryContractError(
            f"{name} must be a finite 2-D numeric array"
        ) from error
    if array.ndim != 2 or array.size == 0 or not np.all(np.isfinite(array)):
        raise SciencePhotometryContractError(
            f"{name} must be a finite non-empty 2-D numeric array"
        )
    if nonnegative and np.any(array < 0.0):
        raise SciencePhotometryContractError(f"{name} must be non-negative")
    if positive and np.any(array <= 0.0):
        raise SciencePhotometryContractError(f"{name} must be positive")
    return array


def _bool_2d(
    value: ArrayLike,
    *,
    name: str,
    shape: tuple[int, int] | None = None,
    require_any: bool = True,
) -> NDArray[np.bool_]:
    array = np.asarray(value)
    if array.ndim != 2 or array.size == 0:
        raise SciencePhotometryContractError(
            f"{name} must be a non-empty 2-D mask"
        )
    if shape is not None and array.shape != shape:
        raise SciencePhotometryContractError(
            f"{name} must have shape {shape}, got {array.shape}"
        )
    if array.dtype.kind not in {"b", "i", "u"} or not np.all(
        (array == 0) | (array == 1)
    ):
        raise SciencePhotometryContractError(
            f"{name} must contain only boolean/0/1 values"
        )
    result = np.asarray(array, dtype=bool)
    if require_any and not np.any(result):
        raise SciencePhotometryContractError(f"{name} must select at least one pixel")
    return result


def _nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise SciencePhotometryContractError(
            f"{name} must be a non-negative integer"
        )
    try:
        integer = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise SciencePhotometryContractError(
            f"{name} must be a non-negative integer"
        ) from error
    if integer < 0 or integer != value:
        raise SciencePhotometryContractError(
            f"{name} must be a non-negative integer"
        )
    return integer


def build_reference_fixed13_aperture_v1(
    stamp_shape: tuple[int, int],
    target_center_yx: tuple[int, int] | None = None,
) -> ScienceApertureDefinition:
    """Build the frozen 13x13 reference-QA aperture on a full stamp.

    This mask is deliberately not a science-optimal aperture.  With no
    explicit target center, it exactly follows the historical reference
    reducer convention ``(ny // 2, nx // 2)``.  An explicit center is useful
    for a known off-center target but must keep all 169 pixels inside the
    delivered stamp.
    """

    try:
        shape_values = tuple(stamp_shape)
    except TypeError as error:
        raise SciencePhotometryContractError(
            "stamp_shape must contain two positive integers"
        ) from error
    if len(shape_values) != 2:
        raise SciencePhotometryContractError(
            "stamp_shape must contain two positive integers"
        )
    ny = _positive_integer(shape_values[0], name="stamp_shape[0]")
    nx = _positive_integer(shape_values[1], name="stamp_shape[1]")
    half_width = 13 // 2
    if ny < 13 or nx < 13:
        raise SciencePhotometryContractError(
            f"13x13 aperture does not fit stamp shape {(ny, nx)}"
        )

    if target_center_yx is None:
        center_y, center_x = ny // 2, nx // 2
        center_policy = "stamp_floor_center_yx"
    else:
        try:
            center_values = tuple(target_center_yx)
        except TypeError as error:
            raise SciencePhotometryContractError(
                "target_center_yx must contain two non-negative integers"
            ) from error
        if len(center_values) != 2:
            raise SciencePhotometryContractError(
                "target_center_yx must contain two non-negative integers"
            )
        center_y = _nonnegative_integer(
            center_values[0],
            name="target_center_yx[0]",
        )
        center_x = _nonnegative_integer(
            center_values[1],
            name="target_center_yx[1]",
        )
        center_policy = "explicit_integer_target_center_yx"

    if (
        center_y - half_width < 0
        or center_y + half_width >= ny
        or center_x - half_width < 0
        or center_x + half_width >= nx
    ):
        raise SciencePhotometryContractError(
            "fixed 13x13 aperture crosses the stamp boundary"
        )
    aperture = np.zeros((ny, nx), dtype=bool)
    aperture[
        center_y - half_width : center_y + half_width + 1,
        center_x - half_width : center_x + half_width + 1,
    ] = True
    return ScienceApertureDefinition(
        aperture_mask=aperture,
        maximum_cumulative_snr=float("nan"),
        algorithm="et_mainsim.reference_fixed13_aperture_v1",
        signal_template_shape=(ny, nx),
        target_peak_yx=(center_y, center_x),
        metadata={
            "aperture_role": "reference_qa_not_science_optimal",
            "aperture_shape": [13, 13],
            "target_center_yx": [center_y, center_x],
            "target_center_policy": center_policy,
            "maximum_cumulative_snr_applicable": False,
        },
    )


def build_science_optimal_aperture_v1(
    *,
    signal_template_e: ArrayLike,
    noise_template_e: ArrayLike,
    permanent_valid_mask: ArrayLike | None = None,
) -> ScienceApertureDefinition:
    """Select a rectangular science aperture with the validated legacy kernel.

    noise_template_e is a per-pixel standard deviation, not a variance.
    Invalid pixels are retained in the rectangular tensor only as zero-signal,
    effectively infinite-noise entries so the legacy function sees the exact
    physical shape without padding it to a square.
    """

    signal = _finite_2d_float(
        signal_template_e,
        name="signal_template_e",
        nonnegative=True,
    )
    noise = _finite_2d_float(
        noise_template_e,
        name="noise_template_e",
        nonnegative=True,
        positive=True,
    )
    if noise.shape != signal.shape:
        raise SciencePhotometryContractError(
            "noise_template_e must have the same shape as signal_template_e"
        )
    if permanent_valid_mask is None:
        valid = np.ones(signal.shape, dtype=bool)
    else:
        valid = _bool_2d(
            permanent_valid_mask,
            name="permanent_valid_mask",
            shape=signal.shape,
        )
    if not np.any(signal[valid] > 0.0):
        raise SciencePhotometryContractError(
            "signal_template_e has no positive target signal on valid pixels"
        )

    bounded_signal = np.where(valid, signal, 0.0)
    bounded_noise = np.where(valid, noise, np.inf)

    try:
        import torch
        from photsim7.aperture import maximize_cumulative_snr
    except ImportError as error:  # pragma: no cover - packaging guard
        raise RuntimeError(
            "Photsim7 and Torch are required for legacy-compatible optimal aperture"
        ) from error

    selected, maximum_snr = maximize_cumulative_snr(
        torch.as_tensor(bounded_signal, dtype=torch.float64, device="cpu"),
        torch.as_tensor(bounded_noise, dtype=torch.float64, device="cpu"),
        plot=False,
    )
    if not isinstance(selected, torch.Tensor):
        raise SciencePhotometryContractError(
            "legacy optimal-aperture kernel returned a non-tensor mask"
        )
    mask = np.asarray(selected.detach().cpu().numpy(), dtype=bool)
    if mask.shape != signal.shape or not np.any(mask):
        raise SciencePhotometryContractError(
            "legacy optimal-aperture kernel returned an invalid mask"
        )
    if np.any(mask & ~valid):
        raise SciencePhotometryContractError(
            "legacy optimal-aperture kernel selected a permanently invalid pixel"
        )
    peak_flat_index = int(np.argmax(np.where(valid, signal, -np.inf)))
    peak_index = np.unravel_index(peak_flat_index, signal.shape)
    if not bool(mask[peak_index]):
        raise SciencePhotometryContractError(
            "legacy optimal-aperture mask does not contain the target-signal peak"
        )
    resolved_snr = float(maximum_snr)
    if not math.isfinite(resolved_snr) or resolved_snr <= 0.0:
        raise SciencePhotometryContractError(
            "legacy optimal-aperture kernel returned an invalid maximum SNR"
        )
    return ScienceApertureDefinition(
        aperture_mask=mask,
        maximum_cumulative_snr=resolved_snr,
        algorithm="photsim7.aperture.maximize_cumulative_snr",
        signal_template_shape=tuple(int(size) for size in signal.shape),
    )


def build_local_background_mask_v1(
    aperture_mask: ArrayLike,
    *,
    exclusion_radius_pixels: int = 3,
    border_pixels: int = 1,
    permanent_valid_mask: ArrayLike | None = None,
) -> NDArray[np.bool_]:
    """Build a source-excluded rectangular background mask.

    Dilation uses Chebyshev distance so its behavior is explicit and has no
    SciPy/photutils dependency.  Individual cadence masks are applied later;
    this function represents the frozen spatial candidate set.
    """

    aperture = _bool_2d(aperture_mask, name="aperture_mask")
    radius = _nonnegative_integer(
        exclusion_radius_pixels,
        name="exclusion_radius_pixels",
    )
    border = _nonnegative_integer(border_pixels, name="border_pixels")
    ny, nx = aperture.shape
    if border * 2 >= ny or border * 2 >= nx:
        raise SciencePhotometryContractError(
            "border_pixels removes the entire stamp"
        )
    if permanent_valid_mask is None:
        permanent_valid = np.ones(aperture.shape, dtype=bool)
    else:
        permanent_valid = _bool_2d(
            permanent_valid_mask,
            name="permanent_valid_mask",
            shape=aperture.shape,
        )

    dilated = _dilate_mask(aperture, radius=radius)
    background = ~dilated & permanent_valid
    if border:
        background[:border, :] = False
        background[-border:, :] = False
        background[:, :border] = False
        background[:, -border:] = False
    if not np.any(background):
        raise SciencePhotometryContractError(
            "background-mask policy leaves no candidate pixels"
        )
    return background


def _dilate_mask(
    mask: NDArray[np.bool_],
    *,
    radius: int,
) -> NDArray[np.bool_]:
    """Return a Chebyshev-radius dilation without an optional SciPy dependency."""

    ny, nx = mask.shape
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    dilated = np.zeros_like(mask)
    for y_offset in range(2 * radius + 1):
        for x_offset in range(2 * radius + 1):
            dilated |= padded[
                y_offset : y_offset + ny,
                x_offset : x_offset + nx,
            ]
    return dilated


def _positive_integer(value: object, *, name: str) -> int:
    integer = _nonnegative_integer(value, name=name)
    if integer <= 0:
        raise SciencePhotometryContractError(f"{name} must be positive")
    return integer


def _finite_nonnegative_scalar(value: object, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise SciencePhotometryContractError(
            f"{name} must be finite and non-negative"
        ) from error
    if not math.isfinite(result) or result < 0.0:
        raise SciencePhotometryContractError(
            f"{name} must be finite and non-negative"
        )
    return result


def compute_science_flux_uncertainty_model_v1(
    *,
    fitted_source_expectation_e: ArrayLike,
    aperture_mask: ArrayLike,
    read_noise_e_per_raw_pixel: float,
    quantization_noise_e_per_raw_pixel: float,
    coadd_factor: int | ArrayLike,
    cadence_valid: ArrayLike | None = None,
    background_expectation_e: ArrayLike | None = None,
    background_expectation_aperture_e: ArrayLike | None = None,
) -> ScienceFluxUncertaintyModelResult:
    """Evaluate the frozen source/background/read/quantization noise model.

    Exactly one background representation is required.  A bounded in-memory
    caller may pass the full cadence-by-stamp ``background_expectation_e``
    cube; a streaming caller may instead pass its already-computed 1-D
    ``background_expectation_aperture_e`` sums.  Both represent sky, scattered
    light, and dark current, so no separate dark-current term is accepted or
    added.  Read and quantization variances are specified per raw exposure and
    are multiplied by both the aperture pixel count and raw coadd factor.

    Quality-invalid cadences retain all finite physical variance components
    for auditing, while only ``uncertainty_e`` is set to NaN.
    """

    try:
        source = np.asarray(fitted_source_expectation_e, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise SciencePhotometryContractError(
            "fitted_source_expectation_e must be a finite non-negative vector"
        ) from error
    if (
        source.ndim != 1
        or source.size == 0
        or not np.all(np.isfinite(source))
        or np.any(source < 0.0)
    ):
        raise SciencePhotometryContractError(
            "fitted_source_expectation_e must be a finite non-negative vector"
        )
    cadence_count = int(source.size)
    aperture = _bool_2d(aperture_mask, name="aperture_mask")
    cube_provided = background_expectation_e is not None
    aperture_sum_provided = background_expectation_aperture_e is not None
    if cube_provided == aperture_sum_provided:
        raise SciencePhotometryContractError(
            "exactly one of background_expectation_e or "
            "background_expectation_aperture_e must be provided"
        )
    if cube_provided:
        try:
            background = np.asarray(
                background_expectation_e,
                dtype=np.float64,
            )
        except (TypeError, ValueError) as error:
            raise SciencePhotometryContractError(
                "background_expectation_e must be a finite non-negative "
                "cadence-by-stamp cube"
            ) from error
        expected_background_shape = (cadence_count, *aperture.shape)
        if (
            background.shape != expected_background_shape
            or not np.all(np.isfinite(background))
            or np.any(background < 0.0)
        ):
            raise SciencePhotometryContractError(
                "background_expectation_e must be a finite non-negative cube "
                f"with shape {expected_background_shape}"
            )
        background_variance = np.sum(
            background[:, aperture],
            axis=1,
            dtype=np.float64,
        )
        background_product = "background_expectation_e"
        background_input_representation = "cadence_stamp_cube"
    else:
        try:
            background_variance = np.asarray(
                background_expectation_aperture_e,
                dtype=np.float64,
            )
        except (TypeError, ValueError) as error:
            raise SciencePhotometryContractError(
                "background_expectation_aperture_e must be a finite "
                "non-negative cadence vector"
            ) from error
        if (
            background_variance.shape != (cadence_count,)
            or not np.all(np.isfinite(background_variance))
            or np.any(background_variance < 0.0)
        ):
            raise SciencePhotometryContractError(
                "background_expectation_aperture_e must be a finite "
                f"non-negative vector with shape {(cadence_count,)}"
            )
        background_variance = background_variance.copy()
        background_product = "background_expectation_aperture_e"
        background_input_representation = "cadence_aperture_sum_vector"

    read_noise = _finite_nonnegative_scalar(
        read_noise_e_per_raw_pixel,
        name="read_noise_e_per_raw_pixel",
    )
    quantization_noise = _finite_nonnegative_scalar(
        quantization_noise_e_per_raw_pixel,
        name="quantization_noise_e_per_raw_pixel",
    )
    raw_coadd = np.asarray(coadd_factor)
    if raw_coadd.ndim == 0:
        resolved_factor = _positive_integer(
            raw_coadd.item(),
            name="coadd_factor",
        )
        factors = np.full(cadence_count, resolved_factor, dtype=np.int64)
    elif raw_coadd.shape == (cadence_count,):
        factors = np.asarray(
            [
                _positive_integer(value, name="coadd_factor")
                for value in raw_coadd.tolist()
            ],
            dtype=np.int64,
        )
    else:
        raise SciencePhotometryContractError(
            "coadd_factor must be one positive integer or a cadence vector"
        )

    if cadence_valid is None:
        valid = np.ones(cadence_count, dtype=bool)
    else:
        valid_raw = np.asarray(cadence_valid)
        if (
            valid_raw.shape != (cadence_count,)
            or valid_raw.dtype.kind not in {"b", "i", "u"}
            or not np.all((valid_raw == 0) | (valid_raw == 1))
        ):
            raise SciencePhotometryContractError(
                "cadence_valid must be a binary cadence vector"
            )
        valid = np.asarray(valid_raw, dtype=bool)

    aperture_pixel_count = int(np.count_nonzero(aperture))
    source_variance = source.copy()
    factor_float = np.asarray(factors, dtype=np.float64)
    read_variance = (
        aperture_pixel_count * factor_float * read_noise**2
    )
    quantization_variance = (
        aperture_pixel_count * factor_float * quantization_noise**2
    )
    total_variance = (
        source_variance
        + background_variance
        + read_variance
        + quantization_variance
    )
    if not np.all(np.isfinite(total_variance)) or np.any(total_variance < 0.0):
        raise SciencePhotometryContractError(
            "flux uncertainty variance is non-finite or negative"
        )
    uncertainty = np.sqrt(total_variance)
    uncertainty[~valid] = np.nan
    return ScienceFluxUncertaintyModelResult(
        uncertainty_e=uncertainty,
        source_variance_e2=source_variance,
        background_variance_e2=background_variance,
        read_variance_e2=read_variance,
        quantization_variance_e2=quantization_variance,
        valid_mask=valid,
        coadd_factor=factors,
        metadata={
            "schema_id": "et_mainsim.science_flux_uncertainty_model.v1",
            "schema_version": 1,
            "algorithm": (
                "poisson_source_background_plus_raw_read_quantization_v1"
            ),
            "background_product": background_product,
            "background_input_representation": (
                background_input_representation
            ),
            "background_components": (
                "sky+scattered_light+dark_current"
            ),
            "dark_current_counted_once_via_background_expectation": True,
            "read_variance_scaling": (
                "aperture_pixel_count*raw_coadd_factor*"
                "read_noise_e_per_raw_pixel**2"
            ),
            "quantization_variance_scaling": (
                "aperture_pixel_count*raw_coadd_factor*"
                "quantization_noise_e_per_raw_pixel**2"
            ),
            "aperture_pixel_count": aperture_pixel_count,
            "invalid_cadence_policy": (
                "uncertainty_nan_components_retained"
            ),
        },
    )


def _calibrated_expectation_bgsub_e(
    delivery: ReferencePhotometryInput,
) -> NDArray[np.float64]:
    calibrated = (
        (
            delivery.final_dn
            - delivery.bias_level_sum_dn
            - delivery.column_noise_sum_dn_by_x
        )
        * delivery.gain_e_per_dn
        - delivery.background_expectation_e
    )
    if not np.all(np.isfinite(calibrated)):
        raise SciencePhotometryContractError(
            "derived expectation-background electron cube is non-finite"
        )
    return np.asarray(calibrated, dtype=np.float64)


def _positive_factor_vector(
    value: ArrayLike,
    *,
    name: str,
    expected_count: int | None = None,
) -> NDArray[np.float64]:
    try:
        factors = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise SciencePhotometryContractError(
            f"{name} must be a finite positive 1-D vector"
        ) from error
    if (
        factors.ndim != 1
        or factors.size == 0
        or not np.all(np.isfinite(factors))
        or np.any(factors <= 0.0)
    ):
        raise SciencePhotometryContractError(
            f"{name} must be a finite positive 1-D vector"
        )
    if expected_count is not None and factors.size != expected_count:
        raise SciencePhotometryContractError(
            f"{name} must contain exactly {expected_count} samples"
        )
    return factors


def _training_index_vector(
    value: ArrayLike,
    *,
    expected_count: int,
) -> NDArray[np.int64]:
    raw = np.asarray(value)
    if raw.shape != (expected_count,) or raw.dtype.kind not in {"i", "u"}:
        raise SciencePhotometryContractError(
            "training_raw_frame_indices must be an integer cadence vector"
        )
    indices = np.asarray(raw, dtype=np.int64)
    if (
        np.any(indices < 0)
        or (indices.size > 1 and not np.all(np.diff(indices) > 0))
    ):
        raise SciencePhotometryContractError(
            "training_raw_frame_indices must be non-negative and strictly increasing"
        )
    return indices


def train_science_optimal_aperture_v1(
    delivery: ReferencePhotometryInput,
    *,
    raw_relative_flux: ArrayLike,
    training_raw_frame_indices: ArrayLike,
    read_noise_e_per_pixel: float,
    quantization_noise_e_per_pixel: float,
    policy: StampSciencePhotometryPolicy | None = None,
) -> ScienceApertureDefinition:
    """Train one fixed science aperture from deterministic raw-frame samples.

    The baseline source template is a per-pixel regression through the origin,
    ``sum(q * image) / sum(q**2)``.  Invalid, saturated, and cosmic-affected
    samples are excluded independently per pixel instead of being repaired.
    """

    if not isinstance(delivery, ReferencePhotometryInput):
        raise TypeError("delivery must be a ReferencePhotometryInput")
    resolved_policy = policy or StampSciencePhotometryPolicy()
    if not isinstance(resolved_policy, StampSciencePhotometryPolicy):
        raise TypeError("policy must be a StampSciencePhotometryPolicy")
    n_frames, ny, nx = delivery.shape
    q = _positive_factor_vector(
        raw_relative_flux,
        name="raw_relative_flux",
        expected_count=n_frames,
    )
    training_indices = _training_index_vector(
        training_raw_frame_indices,
        expected_count=n_frames,
    )
    read_noise = _finite_nonnegative_scalar(
        read_noise_e_per_pixel,
        name="read_noise_e_per_pixel",
    )
    quantization_noise = _finite_nonnegative_scalar(
        quantization_noise_e_per_pixel,
        name="quantization_noise_e_per_pixel",
    )
    calibrated = _calibrated_expectation_bgsub_e(delivery)
    usable = (
        delivery.valid_mask
        & ~delivery.saturated_mask
        & ~delivery.cosmic_mask
    )
    usable_count = np.count_nonzero(usable, axis=0)
    minimum_valid_count = int(
        math.ceil(n_frames * resolved_policy.minimum_training_valid_fraction)
    )
    permanent_valid = usable_count >= minimum_valid_count
    if not np.any(permanent_valid):
        raise SciencePhotometryContractError(
            "no pixel satisfies minimum_training_valid_fraction"
        )

    q_cube = q[:, None, None]
    weighted_q = np.where(usable, q_cube, 0.0)
    denominator = np.sum(weighted_q * weighted_q, axis=0, dtype=np.float64)
    numerator = np.sum(
        np.where(usable, q_cube * calibrated, 0.0),
        axis=0,
        dtype=np.float64,
    )
    signal = np.zeros((ny, nx), dtype=np.float64)
    fitted = permanent_valid & (denominator > 0.0)
    signal[fitted] = numerator[fitted] / denominator[fitted]
    np.maximum(signal, 0.0, out=signal)

    background_sum = np.sum(
        np.where(usable, delivery.background_expectation_e, 0.0),
        axis=0,
        dtype=np.float64,
    )
    background_mean = np.zeros((ny, nx), dtype=np.float64)
    background_mean[fitted] = background_sum[fitted] / usable_count[fitted]
    noise = np.sqrt(
        signal
        + background_mean
        + read_noise**2
        + quantization_noise**2
    )
    if np.any(noise[permanent_valid] <= 0.0):
        raise SciencePhotometryContractError(
            "training noise template is not positive on all valid pixels"
        )
    # The low-level selector receives finite positive values everywhere; its
    # permanent-valid argument ensures invalid pixels cannot enter the mask.
    noise[~permanent_valid] = 1.0
    selection = build_science_optimal_aperture_v1(
        signal_template_e=signal,
        noise_template_e=noise,
        permanent_valid_mask=permanent_valid,
    )
    background_mask: NDArray[np.bool_] | None = None
    background_pixel_count = 0
    if resolved_policy.local_background_enabled:
        background_mask = build_local_background_mask_v1(
            selection.aperture_mask,
            exclusion_radius_pixels=resolved_policy.background_guard_pixels,
            border_pixels=resolved_policy.background_border_pixels,
            permanent_valid_mask=permanent_valid,
        )
        background_pixel_count = int(np.count_nonzero(background_mask))
        if background_pixel_count < resolved_policy.minimum_background_pixels:
            raise SciencePhotometryContractError(
                "trained background mask has fewer pixels than minimum_background_pixels"
            )
    peak_flat = int(np.argmax(np.where(permanent_valid, signal, -np.inf)))
    peak_yx = tuple(
        int(value) for value in np.unravel_index(peak_flat, signal.shape)
    )
    return ScienceApertureDefinition(
        aperture_mask=selection.aperture_mask,
        maximum_cumulative_snr=selection.maximum_cumulative_snr,
        algorithm=selection.algorithm,
        signal_template_shape=selection.signal_template_shape,
        background_mask=background_mask,
        signal_template_e=signal,
        noise_template_e=noise,
        target_peak_yx=peak_yx,
        training_raw_frame_indices=training_indices,
        metadata={
            "template_fit": "through_origin_q_weighted_v1",
            "background_strategy": resolved_policy.background_strategy,
            "local_background_enabled": resolved_policy.local_background_enabled,
            "minimum_training_valid_fraction": (
                resolved_policy.minimum_training_valid_fraction
            ),
            "minimum_training_valid_count": minimum_valid_count,
            "excluded_training_sample_count": int(
                usable.size - np.count_nonzero(usable)
            ),
            "background_pixel_count": background_pixel_count,
            "read_noise_e_per_pixel": read_noise,
            "quantization_noise_e_per_pixel": quantization_noise,
        },
    )


def coadd_reference_photometry_input_v1(
    delivery: ReferencePhotometryInput,
    *,
    factor: int,
) -> ReferencePhotometryInput:
    """Accumulate raw delivery planes without inventing calibrated images.

    This helper encodes the scalar/static-gain linearity used by the streaming
    backend.  Per-frame gain changes fail closed because summing DN before
    calibration would then be physically ambiguous.
    """

    if not isinstance(delivery, ReferencePhotometryInput):
        raise TypeError("delivery must be a ReferencePhotometryInput")
    resolved_factor = _positive_integer(factor, name="factor")
    n_frames, ny, nx = delivery.shape
    if n_frames % resolved_factor:
        raise SciencePhotometryContractError(
            "raw cadence count must be divisible by factor"
        )
    if delivery.exposure_seconds is None:
        raise SciencePhotometryContractError(
            "raw exposure_seconds are required for coadd accumulation"
        )
    time = np.asarray(delivery.time_seconds, dtype=np.float64)
    exposure = np.asarray(delivery.exposure_seconds, dtype=np.float64)
    if n_frames > 1 and not np.allclose(
        time[1:],
        time[:-1] + exposure[:-1],
        rtol=0.0,
        atol=1e-8,
    ):
        raise SciencePhotometryContractError(
            "raw cadence intervals must be globally contiguous"
        )
    static_gain = np.asarray(delivery.gain_e_per_dn[0], dtype=np.float64)
    if not np.all(delivery.gain_e_per_dn == static_gain[None, :, :]):
        raise SciencePhotometryContractError(
            "raw accumulation does not support per-frame gain maps"
        )
    n_coadds = n_frames // resolved_factor

    def grouped_sum(value: NDArray[np.generic]) -> NDArray[Any]:
        return np.sum(
            value.reshape(n_coadds, resolved_factor, *value.shape[1:]),
            axis=1,
        )

    def grouped_any(value: NDArray[np.bool_]) -> NDArray[np.bool_]:
        return np.any(
            value.reshape(n_coadds, resolved_factor, ny, nx),
            axis=1,
        )

    return ReferencePhotometryInput.from_arrays(
        final_dn=grouped_sum(delivery.final_dn),
        background_expectation_e=grouped_sum(
            delivery.background_expectation_e
        ),
        bias_level_sum_dn=grouped_sum(
            delivery.bias_level_sum_dn[:, 0, 0]
        ),
        column_noise_sum_dn_by_x=grouped_sum(
            delivery.column_noise_sum_dn_by_x[:, 0, :]
        ),
        valid_mask=np.all(
            delivery.valid_mask.reshape(
                n_coadds, resolved_factor, ny, nx
            ),
            axis=1,
        ),
        saturated_mask=grouped_any(delivery.saturated_mask),
        cosmic_mask=grouped_any(delivery.cosmic_mask),
        time_index=time.reshape(n_coadds, resolved_factor)[:, 0],
        gain_e_per_dn=static_gain,
        time_index_unit="seconds",
        exposure_seconds=grouped_sum(exposure),
    )


def fit_science_variability_model_v1(
    *,
    flux_e: ArrayLike,
    aperture_valid: ArrayLike,
    raw_relative_flux: ArrayLike,
    raw_frame_start_index: ArrayLike,
    raw_frame_stop_index_exclusive: ArrayLike,
) -> ScienceVariabilityModelResult:
    """Fit the known integrated source factor with zero additive intercept."""

    flux = np.asarray(flux_e, dtype=np.float64)
    valid_raw = np.asarray(aperture_valid)
    raw_start = np.asarray(raw_frame_start_index)
    raw_stop = np.asarray(raw_frame_stop_index_exclusive)
    if flux.ndim != 1 or flux.size == 0:
        raise SciencePhotometryContractError("flux_e must be a non-empty vector")
    if (
        valid_raw.shape != flux.shape
        or valid_raw.dtype.kind not in {"b", "i", "u"}
        or not np.all((valid_raw == 0) | (valid_raw == 1))
    ):
        raise SciencePhotometryContractError(
            "aperture_valid must be a binary cadence vector"
        )
    if (
        raw_start.shape != flux.shape
        or raw_stop.shape != flux.shape
        or raw_start.dtype.kind not in {"i", "u"}
        or raw_stop.dtype.kind not in {"i", "u"}
    ):
        raise SciencePhotometryContractError(
            "raw frame intervals must be integer cadence vectors"
        )
    start = np.asarray(raw_start, dtype=np.int64)
    stop = np.asarray(raw_stop, dtype=np.int64)
    if (
        np.any(start < 0)
        or np.any(stop <= start)
        or (start.size > 1 and not np.all(start[1:] == stop[:-1]))
    ):
        raise SciencePhotometryContractError(
            "raw frame intervals must be positive-width and contiguous"
        )
    q = _positive_factor_vector(
        raw_relative_flux,
        name="raw-relative-flux vector",
    )
    if int(stop[-1]) > q.size:
        raise SciencePhotometryContractError(
            "raw-relative-flux vector does not cover all raw frame intervals"
        )
    prefix = np.concatenate(([0.0], np.cumsum(q, dtype=np.float64)))
    factor_sum = prefix[stop] - prefix[start]
    valid = np.asarray(valid_raw, dtype=bool) & np.isfinite(flux)
    if int(np.count_nonzero(valid)) < 2:
        raise SciencePhotometryContractError(
            "at least two valid cadences are required for variability fitting"
        )
    denominator = float(np.dot(factor_sum[valid], factor_sum[valid]))
    scale = float(np.dot(factor_sum[valid], flux[valid]) / denominator)
    if not math.isfinite(scale) or scale <= 0.0:
        raise SciencePhotometryContractError(
            "variability fit produced a non-positive scale"
        )
    fitted_flux = scale * factor_sum
    if not np.all(np.isfinite(fitted_flux)) or np.any(fitted_flux <= 0.0):
        raise SciencePhotometryContractError(
            "variability fitted flux is invalid"
        )
    residual = np.full(flux.shape, np.nan, dtype=np.float64)
    residual[valid] = flux[valid] - fitted_flux[valid]
    residual_ppm = np.full(flux.shape, np.nan, dtype=np.float64)
    residual_ppm[valid] = residual[valid] / fitted_flux[valid] * 1_000_000.0
    return ScienceVariabilityModelResult(
        raw_factor_sum=factor_sum,
        fitted_flux_e=fitted_flux,
        residual_e=residual,
        residual_ppm=residual_ppm,
        fit_scale_e_per_raw_factor=scale,
        fit_intercept_e=0.0,
        valid_mask=valid,
    )


def compute_science_cdpp_v1(
    *,
    time_seconds: ArrayLike,
    exposure_seconds: ArrayLike,
    flux_e: ArrayLike,
    aperture_valid: ArrayLike,
    model_flux_e: ArrayLike | None = None,
    residual_e: ArrayLike | None = None,
    windows_minutes: tuple[int, ...] = (30, 90, 390),
    minimum_coverage_fraction: float = 0.95,
    minimum_accepted_bins: int = 10,
    bin_origin_seconds: float = 0.0,
) -> Any:
    """Compute the frozen median-centered legacy mean-MAD CDPP product."""

    from .coverage_aware_stamp_analysis import (
        CoverageAwareLightCurve,
        compute_coverage_aware_cdpp_v1,
    )

    curve = CoverageAwareLightCurve(
        time_seconds=time_seconds,
        exposure_seconds=exposure_seconds,
        flux_e=flux_e,
        aperture_valid=aperture_valid,
        model_flux_e=model_flux_e,
        residual_e=residual_e,
    )
    return compute_coverage_aware_cdpp_v1(
        curve,
        windows_minutes=windows_minutes,
        minimum_coverage_fraction=minimum_coverage_fraction,
        minimum_accepted_bins=minimum_accepted_bins,
        bin_origin_seconds=bin_origin_seconds,
    )


def reduce_science_photometry_v1(
    delivery: ReferencePhotometryInput,
    *,
    aperture_mask: ArrayLike,
    background_mask: ArrayLike | None = None,
    minimum_background_pixels: int = 32,
    centroid_support_radius_pixels: int = 1,
) -> SciencePhotometryResult:
    """Reduce one in-memory delivery using expectation and local backgrounds."""

    if not isinstance(delivery, ReferencePhotometryInput):
        raise TypeError("delivery must be a ReferencePhotometryInput")
    n_frames, ny, nx = delivery.shape
    aperture = _bool_2d(
        aperture_mask,
        name="aperture_mask",
        shape=(ny, nx),
    )
    local_background_enabled = background_mask is not None
    if local_background_enabled:
        background = _bool_2d(
            background_mask,
            name="background_mask",
            shape=(ny, nx),
        )
        if np.any(aperture & background):
            raise SciencePhotometryContractError(
                "aperture_mask and background_mask must not overlap"
            )
        minimum_background = _positive_integer(
            minimum_background_pixels,
            name="minimum_background_pixels",
        )
        if minimum_background > int(np.count_nonzero(background)):
            raise SciencePhotometryContractError(
                "minimum_background_pixels exceeds the frozen background mask"
            )
    else:
        background = np.zeros((ny, nx), dtype=bool)
        minimum_background = 0
    centroid_radius = _nonnegative_integer(
        centroid_support_radius_pixels,
        name="centroid_support_radius_pixels",
    )

    calibrated_bgsub = _calibrated_expectation_bgsub_e(delivery)

    usable = (
        delivery.valid_mask
        & ~delivery.saturated_mask
        & ~delivery.cosmic_mask
    )
    aperture_pixel_count = int(np.count_nonzero(aperture))
    aperture_usable = np.count_nonzero(
        usable[:, aperture],
        axis=1,
    ).astype(np.int64, copy=False)
    aperture_valid = aperture_usable == aperture_pixel_count
    invalid_counts = np.count_nonzero(
        ~delivery.valid_mask[:, aperture],
        axis=1,
    ).astype(np.int64, copy=False)
    saturated_counts = np.count_nonzero(
        delivery.saturated_mask[:, aperture],
        axis=1,
    ).astype(np.int64, copy=False)
    cosmic_counts = np.count_nonzero(
        delivery.cosmic_mask[:, aperture],
        axis=1,
    ).astype(np.int64, copy=False)

    expectation_flux = np.full(n_frames, np.nan, dtype=np.float64)
    if np.any(aperture_valid):
        expectation_flux[aperture_valid] = np.sum(
            calibrated_bgsub[aperture_valid][:, aperture],
            axis=1,
            dtype=np.float64,
        )

    background_usable = usable[:, background]
    background_counts = np.zeros(n_frames, dtype=np.int64)
    local_background = np.full(n_frames, np.nan, dtype=np.float64)
    if local_background_enabled:
        background_counts = np.count_nonzero(
            background_usable,
            axis=1,
        ).astype(np.int64, copy=False)
        for frame_index in range(n_frames):
            if background_counts[frame_index] < minimum_background:
                continue
            values = calibrated_bgsub[frame_index, background][
                background_usable[frame_index]
            ]
            local_background[frame_index] = float(np.median(values))

    local_flux = np.full(n_frames, np.nan, dtype=np.float64)
    local_valid = aperture_valid & np.isfinite(local_background)
    local_flux[local_valid] = (
        expectation_flux[local_valid]
        - aperture_pixel_count * local_background[local_valid]
    )

    quality = np.zeros(n_frames, dtype=np.uint16)
    quality[invalid_counts > 0] |= int(ScienceQualityFlag.APERTURE_INVALID)
    quality[saturated_counts > 0] |= int(
        ScienceQualityFlag.APERTURE_SATURATED
    )
    quality[cosmic_counts > 0] |= int(ScienceQualityFlag.APERTURE_COSMIC)
    if local_background_enabled:
        quality[~np.isfinite(local_background)] |= int(
            ScienceQualityFlag.INSUFFICIENT_BACKGROUND
        )

    centroid_support = _dilate_mask(aperture, radius=centroid_radius)
    centroid_frames = np.zeros_like(calibrated_bgsub)
    finite_local = np.isfinite(local_background)
    if local_background_enabled:
        centroid_frames[finite_local] = (
            calibrated_bgsub[finite_local]
            - local_background[finite_local, None, None]
        )
    else:
        centroid_frames[:] = calibrated_bgsub
    centroid_frames[~usable | ~centroid_support[None, :, :]] = 0.0
    np.maximum(centroid_frames, 0.0, out=centroid_frames)
    centroid_totals = np.sum(centroid_frames, axis=(1, 2), dtype=np.float64)
    y_grid, x_grid = np.indices((ny, nx), dtype=np.float64)
    centroid_x = np.full(n_frames, np.nan, dtype=np.float64)
    centroid_y = np.full(n_frames, np.nan, dtype=np.float64)
    positive_total = centroid_totals > 0.0
    centroid_x[positive_total] = np.sum(
        centroid_frames[positive_total] * x_grid[None, :, :],
        axis=(1, 2),
        dtype=np.float64,
    ) / centroid_totals[positive_total]
    centroid_y[positive_total] = np.sum(
        centroid_frames[positive_total] * y_grid[None, :, :],
        axis=(1, 2),
        dtype=np.float64,
    ) / centroid_totals[positive_total]
    centroid_invalid = (
        (centroid_totals <= 0.0)
        | ~np.isfinite(centroid_x)
        | ~np.isfinite(centroid_y)
    )
    centroid_x[centroid_invalid] = np.nan
    centroid_y[centroid_invalid] = np.nan
    quality[centroid_invalid] |= int(ScienceQualityFlag.CENTROID_UNAVAILABLE)

    return SciencePhotometryResult(
        time_seconds=np.asarray(delivery.time_seconds, dtype=np.float64),
        exposure_seconds=(
            None
            if delivery.exposure_seconds is None
            else np.asarray(delivery.exposure_seconds, dtype=np.float64)
        ),
        flux_expectation_bgsub_e=expectation_flux,
        flux_local_bgsub_e=local_flux,
        local_background_e_per_pixel=local_background,
        centroid_x=centroid_x,
        centroid_y=centroid_y,
        aperture_valid=aperture_valid,
        aperture_usable_pixel_count=aperture_usable,
        aperture_invalid_pixel_count=invalid_counts,
        saturated_pixel_count=saturated_counts,
        cosmic_pixel_count=cosmic_counts,
        background_usable_pixel_count=background_counts,
        quality_bitmask=quality,
        aperture_mask=aperture,
        background_mask=background,
        product_semantics={
            "schema_id": SCIENCE_PHOTOMETRY_SCHEMA_ID,
            "schema_version": SCIENCE_PHOTOMETRY_SCHEMA_VERSION,
            "observation_product": "final_dn",
            "calibrated_electron_product": "derived",
            "expectation_background_product": "background_expectation_e",
            "default_background_product": "background_expectation_e",
            "background_strategy": (
                _LOCAL_DIAGNOSTIC_BACKGROUND_STRATEGY
                if local_background_enabled
                else _EXPECTATION_BACKGROUND_STRATEGY
            ),
            "local_background_enabled": local_background_enabled,
            "local_background_role": "replaceable_diagnostic_estimator",
            "local_background_policy_version": _LOCAL_BACKGROUND_POLICY_VERSION,
            "local_background_estimator": _LOCAL_BACKGROUND_ESTIMATOR,
            "local_background_sigma_clipping": (
                _LOCAL_BACKGROUND_SIGMA_CLIPPING
            ),
            "background_realization_used": False,
            "centroid_algorithm": (
                "legacy_center_of_mass_math_on_aperture_support_v1"
            ),
            "centroid_support_radius_pixels": centroid_radius,
            "quality_policy": (
                "invalidate_whole_science_aperture_cadence"
            ),
        },
    )


__all__ = [
    "SCIENCE_PHOTOMETRY_SCHEMA_ID",
    "SCIENCE_PHOTOMETRY_SCHEMA_VERSION",
    "ScienceApertureDefinition",
    "ScienceFluxUncertaintyModelResult",
    "SciencePhotometryContractError",
    "SciencePhotometryResult",
    "ScienceQualityFlag",
    "ScienceVariabilityModelResult",
    "StampSciencePhotometryPolicy",
    "build_local_background_mask_v1",
    "build_reference_fixed13_aperture_v1",
    "build_science_optimal_aperture_v1",
    "coadd_reference_photometry_input_v1",
    "compute_science_cdpp_v1",
    "compute_science_flux_uncertainty_model_v1",
    "fit_science_variability_model_v1",
    "reduce_science_photometry_v1",
    "train_science_optimal_aperture_v1",
]
