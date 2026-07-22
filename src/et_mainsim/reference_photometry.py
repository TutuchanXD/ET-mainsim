"""Reference fixed-aperture photometry for stamp delivery bundles.

``final_dn`` is the only detector-observation product in this contract.  The
electron-domain light curve made here is deliberately a *derived calibration
product*:

``((final_dn - bias_level_sum_dn - column_noise_sum_dn_by_x) * gain_e_per_dn)
   - background_expectation_e``.

In particular, a background-realization plane is never read or subtracted.
Subtracting it would incorrectly remove the Poisson realization from an
observation.  This module is intentionally independent of the legacy
``bin_lcs`` helper: it bins on the supplied physical time axis on CPU and
requires complete exposure coverage for each CDPP window.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray


REFERENCE_PHOTOMETRY_SCHEMA_ID = "et_mainsim.reference_photometry.v1"
REFERENCE_PHOTOMETRY_SCHEMA_VERSION = 1
FIXED_APERTURE_SHAPE = (13, 13)
STANDARD_CDPP_WINDOWS_MINUTES = (30, 90, 390)

TimeIndexUnit = Literal["frame_index", "seconds"]


class ReferencePhotometryContractError(ValueError):
    """Raised when a delivery bundle cannot support the v1 reduction contract."""


def _as_array(value: ArrayLike, *, field_name: str) -> NDArray[np.generic]:
    array = np.asarray(value)
    if array.size == 0:
        raise ReferencePhotometryContractError(f"{field_name} must not be empty")
    return array


def _as_finite_float_array(
    value: ArrayLike,
    *,
    field_name: str,
) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ReferencePhotometryContractError(
            f"{field_name} must contain only finite numeric values"
        )
    return array


def _as_bool_mask(
    value: ArrayLike,
    *,
    field_name: str,
    shape: tuple[int, int, int],
) -> NDArray[np.bool_]:
    array = _as_array(value, field_name=field_name)
    if array.shape != shape:
        raise ReferencePhotometryContractError(
            f"{field_name} must have shape {shape}, got {array.shape}"
        )
    if array.dtype.kind not in {"b", "i", "u"}:
        raise ReferencePhotometryContractError(
            f"{field_name} must be a boolean or integer mask, got {array.dtype}"
        )
    return np.asarray(array, dtype=bool)


def _broadcast_bias(
    value: ArrayLike,
    *,
    n_frames: int,
) -> NDArray[np.float64]:
    array = _as_finite_float_array(value, field_name="bias_level_sum_dn")
    if array.shape == ():
        return np.broadcast_to(array, (n_frames, 1, 1))
    if array.shape == (n_frames,):
        return array[:, None, None]
    if array.shape == (n_frames, 1, 1):
        return array
    raise ReferencePhotometryContractError(
        "bias_level_sum_dn must be scalar, (n_frames,), or (n_frames, 1, 1); "
        f"got {array.shape}"
    )


def _broadcast_column_noise(
    value: ArrayLike,
    *,
    n_frames: int,
    nx: int,
) -> NDArray[np.float64]:
    array = _as_finite_float_array(value, field_name="column_noise_sum_dn_by_x")
    if array.shape == ():
        return np.broadcast_to(array, (n_frames, 1, nx))
    if array.shape == (nx,):
        return np.broadcast_to(array[None, None, :], (n_frames, 1, nx))
    if array.shape == (n_frames, nx):
        return array[:, None, :]
    if array.shape == (n_frames, 1, nx):
        return array
    raise ReferencePhotometryContractError(
        "column_noise_sum_dn_by_x must be scalar, (nx,), (n_frames, nx), or "
        f"(n_frames, 1, nx); got {array.shape}"
    )


def _broadcast_gain(
    value: ArrayLike,
    *,
    shape: tuple[int, int, int],
) -> NDArray[np.float64]:
    array = _as_finite_float_array(value, field_name="gain_e_per_dn")
    if np.any(array <= 0.0):
        raise ReferencePhotometryContractError("gain_e_per_dn must be positive")
    n_frames, ny, nx = shape
    if array.shape == ():
        return np.broadcast_to(array, shape)
    if array.shape == (ny, nx):
        return np.broadcast_to(array[None, :, :], shape)
    if array.shape == shape:
        return array
    raise ReferencePhotometryContractError(
        "gain_e_per_dn must be scalar, (ny, nx), or (n_frames, ny, nx); "
        f"got {array.shape}"
    )


def _normalise_time_index_unit(value: object) -> TimeIndexUnit:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str):
        raise ReferencePhotometryContractError(
            "time_index_unit must be either 'frame_index' or 'seconds'"
        )
    normalised = value.strip().lower()
    if normalised not in {"frame_index", "seconds"}:
        raise ReferencePhotometryContractError(
            "time_index_unit must be either 'frame_index' or 'seconds'"
        )
    return normalised  # type: ignore[return-value]


def _normalise_optional_exposure_seconds(
    value: ArrayLike | None,
    *,
    n_frames: int,
) -> NDArray[np.float64] | None:
    if value is None:
        return None
    array = _as_finite_float_array(value, field_name="exposure_seconds")
    if np.any(array <= 0.0):
        raise ReferencePhotometryContractError("exposure_seconds must be positive")
    if array.shape == ():
        return np.broadcast_to(array, (n_frames,))
    if array.shape == (n_frames,):
        return array
    raise ReferencePhotometryContractError(
        "exposure_seconds must be scalar or (n_frames,), "
        f"got {array.shape}"
    )


@dataclass(frozen=True)
class ReferencePhotometryInput:
    """Validated inputs from one composite HDF5 delivery-bundle product.

    ``time_index`` is either an absolute raw/coadd frame index or an absolute
    start time in seconds.  If it is a frame index, ``raw_frame_seconds`` is
    required so no cadence is inferred silently at input time.
    """

    final_dn: NDArray[np.float64]
    background_expectation_e: NDArray[np.float64]
    bias_level_sum_dn: NDArray[np.float64]
    column_noise_sum_dn_by_x: NDArray[np.float64]
    valid_mask: NDArray[np.bool_]
    saturated_mask: NDArray[np.bool_]
    cosmic_mask: NDArray[np.bool_]
    time_index: NDArray[np.float64]
    gain_e_per_dn: NDArray[np.float64]
    time_index_unit: TimeIndexUnit
    raw_frame_seconds: float | None
    exposure_seconds: NDArray[np.float64] | None

    @classmethod
    def from_arrays(
        cls,
        *,
        final_dn: ArrayLike,
        background_expectation_e: ArrayLike,
        bias_level_sum_dn: ArrayLike,
        column_noise_sum_dn_by_x: ArrayLike,
        valid_mask: ArrayLike,
        saturated_mask: ArrayLike,
        cosmic_mask: ArrayLike,
        time_index: ArrayLike,
        gain_e_per_dn: ArrayLike,
        time_index_unit: TimeIndexUnit | str,
        raw_frame_seconds: float | None = None,
        exposure_seconds: ArrayLike | None = None,
    ) -> "ReferencePhotometryInput":
        final = _as_finite_float_array(final_dn, field_name="final_dn")
        if final.ndim != 3:
            raise ReferencePhotometryContractError(
                "final_dn must have shape (n_frames, ny, nx)"
            )
        n_frames, ny, nx = final.shape
        if n_frames <= 0 or ny <= 0 or nx <= 0:
            raise ReferencePhotometryContractError("final_dn dimensions must be positive")

        background = _as_finite_float_array(
            background_expectation_e,
            field_name="background_expectation_e",
        )
        if background.shape != final.shape:
            raise ReferencePhotometryContractError(
                "background_expectation_e must have the same shape as final_dn; "
                f"got {background.shape} and {final.shape}"
            )

        normalised_time = _as_finite_float_array(time_index, field_name="time_index")
        if normalised_time.shape != (n_frames,):
            raise ReferencePhotometryContractError(
                "time_index must have shape (n_frames,), "
                f"got {normalised_time.shape}"
            )
        if n_frames > 1 and not np.all(np.diff(normalised_time) > 0.0):
            raise ReferencePhotometryContractError(
                "time_index must be strictly increasing"
            )

        unit = _normalise_time_index_unit(time_index_unit)
        frame_seconds: float | None
        if raw_frame_seconds is None:
            frame_seconds = None
        else:
            frame_seconds = float(raw_frame_seconds)
            if not np.isfinite(frame_seconds) or frame_seconds <= 0.0:
                raise ReferencePhotometryContractError(
                    "raw_frame_seconds must be a finite positive value"
                )
        if unit == "frame_index":
            if not np.all(np.equal(normalised_time, np.floor(normalised_time))):
                raise ReferencePhotometryContractError(
                    "frame_index time_index values must be integers"
                )
            if frame_seconds is None:
                raise ReferencePhotometryContractError(
                    "raw_frame_seconds is required when time_index_unit='frame_index'"
                )

        shape = (n_frames, ny, nx)
        return cls(
            final_dn=final,
            background_expectation_e=background,
            bias_level_sum_dn=_broadcast_bias(
                bias_level_sum_dn,
                n_frames=n_frames,
            ),
            column_noise_sum_dn_by_x=_broadcast_column_noise(
                column_noise_sum_dn_by_x,
                n_frames=n_frames,
                nx=nx,
            ),
            valid_mask=_as_bool_mask(valid_mask, field_name="valid_mask", shape=shape),
            saturated_mask=_as_bool_mask(
                saturated_mask,
                field_name="saturated_mask",
                shape=shape,
            ),
            cosmic_mask=_as_bool_mask(cosmic_mask, field_name="cosmic_mask", shape=shape),
            time_index=normalised_time,
            gain_e_per_dn=_broadcast_gain(gain_e_per_dn, shape=shape),
            time_index_unit=unit,
            raw_frame_seconds=frame_seconds,
            exposure_seconds=_normalise_optional_exposure_seconds(
                exposure_seconds,
                n_frames=n_frames,
            ),
        )

    @property
    def time_seconds(self) -> NDArray[np.float64]:
        """Absolute start times in seconds for cadence-aware aggregation."""

        if self.time_index_unit == "seconds":
            return self.time_index
        assert self.raw_frame_seconds is not None  # guaranteed by from_arrays
        return self.time_index * self.raw_frame_seconds

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(size) for size in self.final_dn.shape)  # type: ignore[return-value]


@dataclass(frozen=True)
class CadenceCDPP:
    """One complete-window, legacy-compatible MAD CDPP measurement."""

    window_minutes: int
    cdpp_ppm: float
    complete_bin_count: int
    rejected_bin_count: int
    accepted_sample_count: int
    estimator: str = "legacy_mean_absolute_deviation_times_1.4826"
    aggregation: str = "sum_complete_exposure_electrons"

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_minutes": self.window_minutes,
            "cdpp_ppm": self.cdpp_ppm,
            "complete_bin_count": self.complete_bin_count,
            "rejected_bin_count": self.rejected_bin_count,
            "accepted_sample_count": self.accepted_sample_count,
            "estimator": self.estimator,
            "aggregation": self.aggregation,
        }


@dataclass(frozen=True)
class ReferencePhotometryResult:
    """The v1 derived electron light curve and transparent quality metadata."""

    time_seconds: NDArray[np.float64]
    flux_e: NDArray[np.float64]
    aperture_valid: NDArray[np.bool_]
    aperture_usable_pixel_count: NDArray[np.int64]
    aperture_mask: NDArray[np.bool_]
    aperture_shape: tuple[int, int]
    aperture_pixel_count: int
    exposure_seconds: NDArray[np.float64] | None
    cdpp_by_window_minutes: Mapping[int, CadenceCDPP]
    product_semantics: Mapping[str, str | bool]
    raw_frame_start_index: NDArray[np.int64] | None = None
    raw_frame_stop_index_exclusive: NDArray[np.int64] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_id": REFERENCE_PHOTOMETRY_SCHEMA_ID,
            "schema_version": REFERENCE_PHOTOMETRY_SCHEMA_VERSION,
            "aperture_shape": list(self.aperture_shape),
            "aperture_pixel_count": self.aperture_pixel_count,
            "valid_cadence_count": int(np.count_nonzero(self.aperture_valid)),
            "cadence_count": int(self.aperture_valid.size),
            "has_raw_frame_intervals": self.raw_frame_start_index is not None,
            "cdpp_by_window_minutes": {
                str(window): metric.to_dict()
                for window, metric in self.cdpp_by_window_minutes.items()
            },
            "product_semantics": dict(self.product_semantics),
        }


@dataclass(frozen=True)
class InjectedModelResidualResult:
    """A delivery light curve after removing its known injected source model."""

    raw_factor_sum: NDArray[np.float64]
    fitted_flux_e: NDArray[np.float64]
    residual_e: NDArray[np.float64]
    residual_ppm: NDArray[np.float64]
    fit_scale_e_per_raw_factor: float
    fit_intercept_e: float
    valid_mask: NDArray[np.bool_]
    cdpp_by_window_minutes: Mapping[int, CadenceCDPP]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_id": "et_mainsim.injected_model_residual.v1",
            "fit_scale_e_per_raw_factor": self.fit_scale_e_per_raw_factor,
            "fit_intercept_e": self.fit_intercept_e,
            "valid_cadence_count": int(np.count_nonzero(self.valid_mask)),
            "cdpp_by_window_minutes": {
                str(window): metric.to_dict()
                for window, metric in self.cdpp_by_window_minutes.items()
            },
        }


def _group_for_read(handle: Any, group: str | None) -> Any:
    if group is None or group in {"", "/"}:
        return handle
    try:
        return handle[group]
    except KeyError as error:
        raise ReferencePhotometryContractError(
            f"delivery bundle does not contain group {group!r}"
        ) from error


def _required_dataset(group: Any, name: str) -> NDArray[np.generic]:
    if name not in group:
        raise ReferencePhotometryContractError(
            f"delivery bundle is missing required dataset {name!r}"
        )
    return np.asarray(group[name])


def _first_metadata_value(
    group: Any,
    root: Any,
    name: str,
) -> object | None:
    if name in group:
        value = np.asarray(group[name])
        if value.size != 1:
            raise ReferencePhotometryContractError(
                f"metadata dataset {name!r} must be scalar"
            )
        return value.reshape(()).item()
    if name in group.attrs:
        return group.attrs[name]
    if group is not root and name in root.attrs:
        return root.attrs[name]
    return None


def load_reference_photometry_input(
    bundle_path: Path | str,
    *,
    group: str | None = None,
    gain_e_per_dn: ArrayLike | None = None,
    time_index_unit: TimeIndexUnit | str | None = None,
    raw_frame_seconds: float | None = None,
    exposure_seconds: ArrayLike | None = None,
) -> ReferencePhotometryInput:
    """Read the v1-required product planes from a composite HDF5 bundle.

    The bundle must explicitly carry an electronic gain and time-unit metadata
    unless the caller provides them.  This fails closed rather than guessing
    whether a numeric time index is raw frames or seconds.
    """

    try:
        import h5py
    except ImportError as error:  # pragma: no cover - packaging guard
        raise RuntimeError(
            "h5py is required to read a reference-photometry delivery bundle"
        ) from error

    path = Path(bundle_path)
    with h5py.File(path, "r") as handle:
        selected_group = _group_for_read(handle, group)
        payload: dict[str, ArrayLike] = {
            name: _required_dataset(selected_group, name)
            for name in (
                "final_dn",
                "background_expectation_e",
                "bias_level_sum_dn",
                "column_noise_sum_dn_by_x",
                "valid_mask",
                "saturated_mask",
                "cosmic_mask",
                "time_index",
            )
        }

        resolved_gain = gain_e_per_dn
        if resolved_gain is None:
            resolved_gain = _first_metadata_value(
                selected_group,
                handle,
                "gain_e_per_dn",
            )
        if resolved_gain is None:
            raise ReferencePhotometryContractError(
                "delivery bundle needs gain_e_per_dn metadata or an explicit argument"
            )

        resolved_unit = time_index_unit
        if resolved_unit is None:
            resolved_unit = _first_metadata_value(
                selected_group,
                handle,
                "time_index_unit",
            )
        if resolved_unit is None:
            raise ReferencePhotometryContractError(
                "delivery bundle needs time_index_unit metadata or an explicit argument"
            )

        resolved_raw_frame_seconds = raw_frame_seconds
        if resolved_raw_frame_seconds is None:
            metadata_value = _first_metadata_value(
                selected_group,
                handle,
                "raw_frame_seconds",
            )
            if metadata_value is not None:
                resolved_raw_frame_seconds = float(metadata_value)

        resolved_exposure_seconds = exposure_seconds
        if resolved_exposure_seconds is None:
            for name in ("exposure_seconds", "coadd_exposure_seconds"):
                metadata_value = _first_metadata_value(selected_group, handle, name)
                if metadata_value is not None:
                    resolved_exposure_seconds = metadata_value
                    break

    return ReferencePhotometryInput.from_arrays(
        **payload,
        gain_e_per_dn=resolved_gain,
        time_index_unit=resolved_unit,
        raw_frame_seconds=resolved_raw_frame_seconds,
        exposure_seconds=resolved_exposure_seconds,
    )


def _infer_exposure_seconds(time_seconds: NDArray[np.float64]) -> NDArray[np.float64] | None:
    if time_seconds.size < 2:
        return None
    intervals = np.diff(time_seconds)
    positive_intervals = intervals[intervals > 0.0]
    if positive_intervals.size == 0:
        return None
    cadence_seconds = float(np.median(positive_intervals))
    if not np.isfinite(cadence_seconds) or cadence_seconds <= 0.0:
        return None
    return np.full(time_seconds.shape, cadence_seconds, dtype=np.float64)


def _fixed_central_aperture_mask(
    *,
    ny: int,
    nx: int,
    aperture_shape: tuple[int, int] = FIXED_APERTURE_SHAPE,
) -> NDArray[np.bool_]:
    aperture_ny, aperture_nx = aperture_shape
    if aperture_ny <= 0 or aperture_nx <= 0:
        raise ReferencePhotometryContractError("aperture dimensions must be positive")
    if aperture_ny % 2 == 0 or aperture_nx % 2 == 0:
        raise ReferencePhotometryContractError("fixed aperture dimensions must be odd")
    if aperture_ny > ny or aperture_nx > nx:
        raise ReferencePhotometryContractError(
            f"fixed aperture {aperture_shape} does not fit stamp shape {(ny, nx)}"
        )
    center_y, center_x = ny // 2, nx // 2
    half_y, half_x = aperture_ny // 2, aperture_nx // 2
    mask = np.zeros((ny, nx), dtype=bool)
    mask[
        center_y - half_y : center_y + half_y + 1,
        center_x - half_x : center_x + half_x + 1,
    ] = True
    return mask


def _validate_cdpp_inputs(
    *,
    time_seconds: ArrayLike,
    flux_e: ArrayLike,
    aperture_valid: ArrayLike,
    exposure_seconds: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_], NDArray[np.float64]]:
    time = _as_finite_float_array(time_seconds, field_name="time_seconds")
    flux = _as_finite_float_array(flux_e, field_name="flux_e")
    valid = _as_array(aperture_valid, field_name="aperture_valid")
    exposure = _as_finite_float_array(
        exposure_seconds,
        field_name="exposure_seconds",
    )
    if time.ndim != 1 or flux.ndim != 1 or valid.ndim != 1:
        raise ReferencePhotometryContractError(
            "time_seconds, flux_e, and aperture_valid must each be one-dimensional"
        )
    if time.shape != flux.shape or time.shape != valid.shape:
        raise ReferencePhotometryContractError(
            "time_seconds, flux_e, and aperture_valid must have the same shape"
        )
    if valid.dtype.kind not in {"b", "i", "u"}:
        raise ReferencePhotometryContractError(
            "aperture_valid must be a boolean or integer mask"
        )
    if exposure.shape == ():
        exposure = np.broadcast_to(exposure, time.shape)
    if exposure.shape != time.shape:
        raise ReferencePhotometryContractError(
            "exposure_seconds must be scalar or have the same shape as time_seconds"
        )
    if np.any(exposure <= 0.0):
        raise ReferencePhotometryContractError("exposure_seconds must be positive")
    if time.size > 1 and not np.all(np.diff(time) > 0.0):
        raise ReferencePhotometryContractError("time_seconds must be strictly increasing")
    return time, flux, np.asarray(valid, dtype=bool), exposure


def _covered_duration_seconds(
    starts: NDArray[np.float64],
    exposures: NDArray[np.float64],
    *,
    lower: float,
    upper: float,
) -> float:
    """Return the duration of the union of sample intervals inside a time bin."""

    if starts.size == 0:
        return 0.0
    intervals = sorted(
        (
            max(lower, float(start)),
            min(upper, float(start + exposure)),
        )
        for start, exposure in zip(starts, exposures, strict=True)
        if start < upper and start + exposure > lower
    )
    if not intervals:
        return 0.0
    covered = 0.0
    current_start, current_end = intervals[0]
    for start, end in intervals[1:]:
        if end <= start:
            continue
        if start > current_end:
            covered += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    return covered + max(0.0, current_end - current_start)


def _coerce_windows_minutes(windows_minutes: Iterable[int]) -> tuple[int, ...]:
    windows = tuple(int(window) for window in windows_minutes)
    if not windows:
        raise ReferencePhotometryContractError("at least one CDPP window is required")
    if any(window <= 0 for window in windows):
        raise ReferencePhotometryContractError("CDPP windows must be positive minutes")
    if len(set(windows)) != len(windows):
        raise ReferencePhotometryContractError("CDPP windows must not repeat")
    return windows


def compute_cadence_aware_cdpp(
    *,
    time_seconds: ArrayLike,
    flux_e: ArrayLike,
    aperture_valid: ArrayLike,
    exposure_seconds: ArrayLike,
    windows_minutes: Iterable[int] = STANDARD_CDPP_WINDOWS_MINUTES,
    bin_origin_seconds: float = 0.0,
) -> dict[int, CadenceCDPP]:
    """Calculate complete-window MAD CDPP on the supplied physical cadence.

    Each sample is interpreted as a count accumulated over the interval
    ``[time_seconds, time_seconds + exposure_seconds)``.  A bin is accepted
    only if usable samples cover its full duration; unfilled leading/trailing
    partial bins are ignored, while internal gaps and masked samples are
    counted as rejected bins.  This prevents an output cadence or missing data
    from being treated as a synthetic regularly sampled light curve.
    """

    time, flux, valid, exposure = _validate_cdpp_inputs(
        time_seconds=time_seconds,
        flux_e=flux_e,
        aperture_valid=aperture_valid,
        exposure_seconds=exposure_seconds,
    )
    windows = _coerce_windows_minutes(windows_minutes)
    origin = float(bin_origin_seconds)
    if not np.isfinite(origin):
        raise ReferencePhotometryContractError("bin_origin_seconds must be finite")

    coverage_start = float(time[0])
    coverage_end = float(np.max(time + exposure))
    metrics: dict[int, CadenceCDPP] = {}
    tolerance = 1e-8

    for window_minutes in windows:
        window_seconds = float(window_minutes * 60)
        first_full_bin = int(np.ceil((coverage_start - origin) / window_seconds - tolerance))
        last_full_bin_exclusive = int(
            np.floor((coverage_end - origin) / window_seconds + tolerance)
        )
        binned_fluxes: list[float] = []
        accepted_samples = 0
        rejected_bins = 0
        for bin_id in range(first_full_bin, last_full_bin_exclusive):
            lower = origin + bin_id * window_seconds
            upper = lower + window_seconds
            selection = (time >= lower) & (time < upper)
            usable = selection & valid
            covered = _covered_duration_seconds(
                time[usable],
                exposure[usable],
                lower=lower,
                upper=upper,
            )
            if covered < window_seconds - tolerance:
                rejected_bins += 1
                continue
            binned_fluxes.append(float(np.sum(flux[usable], dtype=np.float64)))
            accepted_samples += int(np.count_nonzero(usable))

        binned = np.asarray(binned_fluxes, dtype=np.float64)
        if binned.size < 2 or not np.all(np.isfinite(binned)):
            cdpp_ppm = float("nan")
        else:
            mean_flux = float(np.mean(binned))
            if not np.isfinite(mean_flux) or mean_flux <= 0.0:
                cdpp_ppm = float("nan")
            else:
                legacy_mad = float(np.mean(np.abs(binned - mean_flux)))
                cdpp_ppm = float(legacy_mad * 1.4826 / mean_flux * 1_000_000.0)
        metrics[window_minutes] = CadenceCDPP(
            window_minutes=window_minutes,
            cdpp_ppm=cdpp_ppm,
            complete_bin_count=int(binned.size),
            rejected_bin_count=rejected_bins,
            accepted_sample_count=accepted_samples,
        )
    return metrics


def _compute_model_residual_cdpp(
    *,
    time_seconds: ArrayLike,
    residual_e: ArrayLike,
    fitted_flux_e: ArrayLike,
    valid_mask: ArrayLike,
    exposure_seconds: ArrayLike,
    windows_minutes: Iterable[int],
    bin_origin_seconds: float,
    minimum_complete_bins: int,
) -> dict[int, CadenceCDPP]:
    """CDPP of residual/model fractions after complete exposure aggregation."""

    time, residual, valid, exposure = _validate_cdpp_inputs(
        time_seconds=time_seconds,
        flux_e=residual_e,
        aperture_valid=valid_mask,
        exposure_seconds=exposure_seconds,
    )
    fitted = _as_finite_float_array(fitted_flux_e, field_name="fitted_flux_e")
    if fitted.shape != time.shape or np.any(fitted <= 0.0):
        raise ReferencePhotometryContractError(
            "fitted_flux_e must be positive and match time_seconds"
        )
    if isinstance(minimum_complete_bins, (bool, np.bool_)) or int(
        minimum_complete_bins
    ) < 2:
        raise ReferencePhotometryContractError("minimum_complete_bins must be at least two")
    windows = _coerce_windows_minutes(windows_minutes)
    origin = float(bin_origin_seconds)
    if not np.isfinite(origin):
        raise ReferencePhotometryContractError("bin_origin_seconds must be finite")
    coverage_start = float(time[0])
    coverage_end = float(np.max(time + exposure))
    tolerance = 1e-8
    metrics: dict[int, CadenceCDPP] = {}
    for window_minutes in windows:
        window_seconds = float(window_minutes * 60)
        first_full_bin = int(np.ceil((coverage_start - origin) / window_seconds - tolerance))
        last_full_bin_exclusive = int(
            np.floor((coverage_end - origin) / window_seconds + tolerance)
        )
        fractions: list[float] = []
        accepted_samples = 0
        rejected_bins = 0
        for bin_id in range(first_full_bin, last_full_bin_exclusive):
            lower = origin + bin_id * window_seconds
            upper = lower + window_seconds
            selection = (time >= lower) & (time < upper)
            usable = selection & valid
            covered = _covered_duration_seconds(
                time[usable],
                exposure[usable],
                lower=lower,
                upper=upper,
            )
            if covered < window_seconds - tolerance:
                rejected_bins += 1
                continue
            denominator = float(np.sum(fitted[usable], dtype=np.float64))
            if not np.isfinite(denominator) or denominator <= 0.0:
                rejected_bins += 1
                continue
            fractions.append(float(np.sum(residual[usable], dtype=np.float64)) / denominator)
            accepted_samples += int(np.count_nonzero(usable))
        binned = np.asarray(fractions, dtype=np.float64)
        if binned.size < int(minimum_complete_bins):
            cdpp_ppm = float("nan")
        else:
            centered = binned - float(np.mean(binned))
            cdpp_ppm = float(1.4826 * np.mean(np.abs(centered)) * 1_000_000.0)
        metrics[window_minutes] = CadenceCDPP(
            window_minutes=window_minutes,
            cdpp_ppm=cdpp_ppm,
            complete_bin_count=int(binned.size),
            rejected_bin_count=rejected_bins,
            accepted_sample_count=accepted_samples,
            estimator="injected_model_residual_mad_times_1.4826",
            aggregation="sum_complete_exposure_residual_over_fitted_model",
        )
    return metrics


def compute_injected_model_residual_v1(
    reference: ReferencePhotometryResult,
    *,
    raw_frame_factors: ArrayLike,
    raw_exposure_seconds: float,
    windows_minutes: Iterable[int] = STANDARD_CDPP_WINDOWS_MINUTES,
    bin_origin_seconds: float = 0.0,
    minimum_complete_bins: int = 10,
) -> InjectedModelResidualResult:
    """Fit and remove a known injected ``q(t)`` model before reporting CDPP.

    This is appropriate for variable sources.  The ordinary fixed-aperture
    CDPP remains a useful observed-light-curve statistic, but it includes the
    astrophysical variation by construction and must not be labelled as an
    instrumental-noise metric.  ``raw_frame_factors`` are the frozen 10-s
    exposure-averaged factors, not native FITS node samples.
    """

    if not isinstance(reference, ReferencePhotometryResult):
        raise TypeError("reference must be a ReferencePhotometryResult")
    if (
        reference.raw_frame_start_index is None
        or reference.raw_frame_stop_index_exclusive is None
        or reference.exposure_seconds is None
    ):
        raise ReferencePhotometryContractError(
            "injected-model residuals require formal raw-frame intervals and exposure_seconds"
        )
    try:
        raw_exposure = float(raw_exposure_seconds)
    except (TypeError, ValueError, OverflowError) as error:
        raise ReferencePhotometryContractError(
            "raw_exposure_seconds must be finite and positive"
        ) from error
    if not np.isfinite(raw_exposure) or raw_exposure <= 0.0:
        raise ReferencePhotometryContractError(
            "raw_exposure_seconds must be finite and positive"
        )
    factors = _as_finite_float_array(raw_frame_factors, field_name="raw_frame_factors")
    if factors.ndim != 1 or np.any(factors <= 0.0):
        raise ReferencePhotometryContractError(
            "raw_frame_factors must be a one-dimensional positive array"
        )
    start = np.asarray(reference.raw_frame_start_index, dtype=np.int64)
    stop = np.asarray(reference.raw_frame_stop_index_exclusive, dtype=np.int64)
    if start.shape != reference.time_seconds.shape or stop.shape != start.shape:
        raise ReferencePhotometryContractError(
            "reference raw-frame intervals must match the light-curve cadence axis"
        )
    if (
        np.any(start < 0)
        or np.any(stop <= start)
        or np.any(stop > factors.size)
        or (start.size > 1 and not np.all(start[1:] == stop[:-1]))
    ):
        raise ReferencePhotometryContractError("reference raw-frame intervals are invalid")
    exposure = np.asarray(reference.exposure_seconds, dtype=np.float64)
    raw_width = stop - start
    if not np.allclose(
        exposure,
        raw_width.astype(np.float64) * raw_exposure,
        rtol=0.0,
        atol=1e-8,
    ):
        raise ReferencePhotometryContractError(
            "reference exposure_seconds conflict with raw-frame intervals"
        )
    prefix = np.concatenate(([0.0], np.cumsum(factors, dtype=np.float64)))
    raw_factor_sum = prefix[stop] - prefix[start]
    valid = np.asarray(reference.aperture_valid, dtype=bool) & np.isfinite(reference.flux_e)
    if int(np.count_nonzero(valid)) < 2:
        raise ReferencePhotometryContractError(
            "at least two valid cadences are required for injected-model fitting"
        )
    # The formal calibration has already subtracted the background expectation,
    # so the injected target model is physically proportional to q(t) and has
    # no free additive flux term.  A two-parameter intercept fit is nearly
    # singular for low-amplitude variables (q is close to one) and would let a
    # regression absorb meaningful detector residuals.
    model_valid = raw_factor_sum[valid]
    flux_valid = np.asarray(reference.flux_e, dtype=np.float64)[valid]
    denominator = float(np.dot(model_valid, model_valid))
    scale = float(np.dot(model_valid, flux_valid) / denominator)
    intercept = 0.0
    if not np.isfinite(scale) or scale <= 0.0:
        raise ReferencePhotometryContractError("injected-model fit produced a non-positive scale")
    fitted = scale * raw_factor_sum + intercept
    if np.any(fitted <= 0.0) or not np.all(np.isfinite(fitted)):
        raise ReferencePhotometryContractError("injected-model fitted flux is invalid")
    residual_e = np.full(reference.flux_e.shape, np.nan, dtype=np.float64)
    residual_e[valid] = np.asarray(reference.flux_e, dtype=np.float64)[valid] - fitted[valid]
    residual_ppm = np.full(reference.flux_e.shape, np.nan, dtype=np.float64)
    residual_ppm[valid] = residual_e[valid] / fitted[valid] * 1_000_000.0
    cdpp = _compute_model_residual_cdpp(
        time_seconds=reference.time_seconds,
        residual_e=np.nan_to_num(residual_e, nan=0.0),
        fitted_flux_e=fitted,
        valid_mask=valid,
        exposure_seconds=exposure,
        windows_minutes=windows_minutes,
        bin_origin_seconds=bin_origin_seconds,
        minimum_complete_bins=minimum_complete_bins,
    )
    return InjectedModelResidualResult(
        raw_factor_sum=raw_factor_sum,
        fitted_flux_e=fitted,
        residual_e=residual_e,
        residual_ppm=residual_ppm,
        fit_scale_e_per_raw_factor=float(scale),
        fit_intercept_e=float(intercept),
        valid_mask=valid,
        cdpp_by_window_minutes=cdpp,
    )


def reduce_reference_photometry_v1(
    delivery: ReferencePhotometryInput,
    *,
    cdpp_windows_minutes: Iterable[int] = STANDARD_CDPP_WINDOWS_MINUTES,
    bin_origin_seconds: float = 0.0,
) -> ReferencePhotometryResult:
    """Produce a fixed-aperture derived-electron light curve from ``final_dn``.

    The v1 mask policy is intentionally conservative: a cadence is invalid if
    *any* 13x13 aperture pixel is invalid, saturated, or cosmic-affected.  It
    does not rescale a partial aperture, because doing so would fabricate a
    flux measurement for a detector observation with unavailable pixels.
    """

    if not isinstance(delivery, ReferencePhotometryInput):
        raise TypeError("delivery must be a ReferencePhotometryInput")

    n_frames, ny, nx = delivery.shape
    aperture_mask = _fixed_central_aperture_mask(
        ny=ny,
        nx=nx,
        aperture_shape=FIXED_APERTURE_SHAPE,
    )
    pixel_count = int(np.count_nonzero(aperture_mask))
    usable = (
        delivery.valid_mask
        & ~delivery.saturated_mask
        & ~delivery.cosmic_mask
    )
    aperture_usable_pixel_count = np.count_nonzero(
        usable[:, aperture_mask],
        axis=1,
    ).astype(np.int64, copy=False)
    aperture_valid = aperture_usable_pixel_count == pixel_count

    calibrated_e = (
        (
            delivery.final_dn
            - delivery.bias_level_sum_dn
            - delivery.column_noise_sum_dn_by_x
        )
        * delivery.gain_e_per_dn
        - delivery.background_expectation_e
    )
    flux_e = np.full(n_frames, np.nan, dtype=np.float64)
    if np.any(aperture_valid):
        flux_e[aperture_valid] = np.sum(
            calibrated_e[aperture_valid][:, aperture_mask],
            axis=1,
            dtype=np.float64,
        )

    resolved_exposure_seconds = delivery.exposure_seconds
    if resolved_exposure_seconds is None:
        resolved_exposure_seconds = _infer_exposure_seconds(delivery.time_seconds)
    if resolved_exposure_seconds is None:
        cdpp_by_window_minutes = {
            minutes: CadenceCDPP(
                window_minutes=minutes,
                cdpp_ppm=float("nan"),
                complete_bin_count=0,
                rejected_bin_count=0,
                accepted_sample_count=0,
            )
            for minutes in _coerce_windows_minutes(cdpp_windows_minutes)
        }
    else:
        cdpp_by_window_minutes = compute_cadence_aware_cdpp(
            time_seconds=delivery.time_seconds,
            flux_e=np.nan_to_num(flux_e, nan=0.0),
            aperture_valid=aperture_valid,
            exposure_seconds=resolved_exposure_seconds,
            windows_minutes=cdpp_windows_minutes,
            bin_origin_seconds=bin_origin_seconds,
        )

    return ReferencePhotometryResult(
        time_seconds=delivery.time_seconds,
        flux_e=flux_e,
        aperture_valid=aperture_valid,
        aperture_usable_pixel_count=aperture_usable_pixel_count,
        aperture_mask=aperture_mask,
        aperture_shape=FIXED_APERTURE_SHAPE,
        aperture_pixel_count=pixel_count,
        exposure_seconds=resolved_exposure_seconds,
        cdpp_by_window_minutes=cdpp_by_window_minutes,
        product_semantics={
            "observation_product": "final_dn",
            "calibrated_electron_product": "derived",
            "background_subtraction": "background_expectation_e_only",
            "background_realization_used": False,
            "mask_policy": "invalidate_whole_fixed_aperture_cadence",
        },
    )


def reduce_reference_photometry_bundle_v1(
    bundle_path: Path | str,
    *,
    group: str | None = None,
    gain_e_per_dn: ArrayLike | None = None,
    time_index_unit: TimeIndexUnit | str | None = None,
    raw_frame_seconds: float | None = None,
    exposure_seconds: ArrayLike | None = None,
    cdpp_windows_minutes: Iterable[int] = STANDARD_CDPP_WINDOWS_MINUTES,
    bin_origin_seconds: float = 0.0,
) -> ReferencePhotometryResult:
    """Load and reduce one composite HDF5 delivery bundle in one call."""

    return reduce_reference_photometry_v1(
        load_reference_photometry_input(
            bundle_path,
            group=group,
            gain_e_per_dn=gain_e_per_dn,
            time_index_unit=time_index_unit,
            raw_frame_seconds=raw_frame_seconds,
            exposure_seconds=exposure_seconds,
        ),
        cdpp_windows_minutes=cdpp_windows_minutes,
        bin_origin_seconds=bin_origin_seconds,
    )


def reduce_stamp_delivery_bundle_v1(
    bundle_path: Path | str,
    *,
    cdpp_windows_minutes: Iterable[int] = STANDARD_CDPP_WINDOWS_MINUTES,
    bin_origin_seconds: float = 0.0,
) -> ReferencePhotometryResult:
    """Reduce one formal ``StampDeliveryBundle`` without legacy schema guesses.

    The formal delivery schema stores physical interval starts as
    ``time_start_seconds`` rather than the historical composite-bundle
    ``time_index`` field.  Going through its supported adapter preserves that
    distinction and deliberately exposes no sampled background realization.
    """

    from .stamp_delivery import read_stamp_delivery_bundle

    bundle = read_stamp_delivery_bundle(bundle_path)
    delivery = ReferencePhotometryInput.from_arrays(
        **bundle.to_reference_photometry_payload()
    )
    return reduce_reference_photometry_v1(
        delivery,
        cdpp_windows_minutes=cdpp_windows_minutes,
        bin_origin_seconds=bin_origin_seconds,
    )


@dataclass(frozen=True)
class _FormalDeliveryHeader:
    """Small formal-bundle identity needed before streaming a series."""

    path: Path
    product_kind: str
    coadd_factor: int
    frame_count: int
    stamp_shape: tuple[int, int]
    gain_e_per_dn: NDArray[np.float64]
    manifest_identity: str
    provenance_identity: str
    first_raw_frame_start: int
    last_raw_frame_stop: int
    first_time_start_seconds: float
    last_time_end_seconds: float


_FORMAL_REQUIRED_DATASETS = (
    "final_dn",
    "background_expectation_e",
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
    "manifest_json",
    "provenance_json",
)


def _h5_scalar_string(value: Any, *, name: str) -> str:
    if isinstance(value, np.ndarray):
        if value.shape != ():
            raise ReferencePhotometryContractError(f"{name} must be scalar")
        value = value.item()
    if isinstance(value, (bytes, np.bytes_)):
        value = bytes(value).decode("utf-8")
    if not isinstance(value, str):
        raise ReferencePhotometryContractError(f"{name} must be a UTF-8 string")
    return value


def _formal_json_dataset(handle: Any, name: str) -> dict[str, Any]:
    try:
        value = _h5_scalar_string(handle[name][()], name=name)
        decoded = json.loads(value)
    except (KeyError, json.JSONDecodeError) as error:
        raise ReferencePhotometryContractError(
            f"formal delivery bundle has invalid {name}"
        ) from error
    if not isinstance(decoded, dict):
        raise ReferencePhotometryContractError(f"formal delivery {name} must be an object")
    return decoded


def _formal_identity_json(value: Mapping[str, Any], *, omit: frozenset[str]) -> str:
    candidate = {key: item for key, item in value.items() if key not in omit}
    try:
        return json.dumps(
            candidate,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ReferencePhotometryContractError(
            "formal delivery manifest/provenance must be JSON serializable"
        ) from error


def _read_formal_delivery_header(path: Path | str) -> _FormalDeliveryHeader:
    """Read only header/small vectors; never materialize delivery image cubes."""

    try:
        import h5py
    except ImportError as error:  # pragma: no cover - packaging guard.
        raise RuntimeError("h5py is required to read formal stamp delivery bundles") from error

    from .stamp_delivery import (
        STAMP_DELIVERY_OBSERVATION_PRODUCT,
        STAMP_DELIVERY_SCHEMA_ID,
        STAMP_DELIVERY_SCHEMA_VERSION,
    )

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"formal delivery bundle does not exist: {source}")
    with h5py.File(source, "r") as handle:
        schema_id = _h5_scalar_string(handle.attrs.get("schema_id"), name="schema_id")
        if schema_id != STAMP_DELIVERY_SCHEMA_ID:
            raise ReferencePhotometryContractError("unsupported formal delivery schema")
        if int(handle.attrs.get("schema_version", -1)) != STAMP_DELIVERY_SCHEMA_VERSION:
            raise ReferencePhotometryContractError("unsupported formal delivery schema version")
        if bool(handle.attrs.get("complete", False)) is not True:
            raise ReferencePhotometryContractError("formal delivery bundle is not complete")
        product_kind = _h5_scalar_string(
            handle.attrs.get("product_kind"),
            name="product_kind",
        )
        if product_kind not in {"raw", "coadd"}:
            raise ReferencePhotometryContractError("formal delivery product_kind is invalid")
        coadd_factor = int(handle.attrs.get("coadd_factor", 0))
        if coadd_factor <= 0 or (product_kind == "raw" and coadd_factor != 1):
            raise ReferencePhotometryContractError("formal delivery coadd_factor is invalid")
        if product_kind == "coadd" and coadd_factor <= 1:
            raise ReferencePhotometryContractError("formal delivery coadd_factor is invalid")
        if (
            _h5_scalar_string(
                handle.attrs.get("observation_product"),
                name="observation_product",
            )
            != STAMP_DELIVERY_OBSERVATION_PRODUCT
        ):
            raise ReferencePhotometryContractError(
                "formal delivery observation_product must be final_dn"
            )
        if bool(handle.attrs.get("background_realization_used", True)):
            raise ReferencePhotometryContractError(
                "formal delivery must not expose a background realization"
            )
        missing = [name for name in _FORMAL_REQUIRED_DATASETS if name not in handle]
        if missing:
            raise ReferencePhotometryContractError(
                f"formal delivery bundle is missing required datasets: {missing}"
            )
        final = handle["final_dn"]
        if len(final.shape) != 3 or any(int(size) <= 0 for size in final.shape):
            raise ReferencePhotometryContractError("formal final_dn shape is invalid")
        if final.dtype.kind != "u":
            raise ReferencePhotometryContractError("formal final_dn must use unsigned DN")
        if product_kind == "coadd" and final.dtype != np.dtype(np.uint64):
            raise ReferencePhotometryContractError("formal coadd final_dn must use uint64")
        n_frames, ny, nx = (int(size) for size in final.shape)
        expected_shapes = {
            "background_expectation_e": (n_frames, ny, nx),
            "valid_mask": (n_frames, ny, nx),
            "fullwell_count": (n_frames, ny, nx),
            "adc_low_count": (n_frames, ny, nx),
            "adc_high_count": (n_frames, ny, nx),
            "cosmic_count": (n_frames, ny, nx),
            "saturated_mask": (n_frames, ny, nx),
            "cosmic_mask": (n_frames, ny, nx),
            "bias_level_sum_dn": (n_frames,),
            "column_noise_sum_dn_by_x": (n_frames, nx),
            "time_start_seconds": (n_frames,),
            "exposure_seconds": (n_frames,),
            "raw_frame_start_index": (n_frames,),
            "raw_frame_stop_index_exclusive": (n_frames,),
        }
        for name, expected_shape in expected_shapes.items():
            if tuple(handle[name].shape) != expected_shape:
                raise ReferencePhotometryContractError(
                    f"formal delivery {name} shape differs from final_dn"
                )
        if "gain_e_per_dn" in handle and "gain_e_per_dn" in handle.attrs:
            raise ReferencePhotometryContractError(
                "formal delivery gain_e_per_dn is stored twice"
            )
        if "gain_e_per_dn" in handle:
            gain = np.asarray(handle["gain_e_per_dn"], dtype=np.float64)
        elif "gain_e_per_dn" in handle.attrs:
            gain = np.asarray(handle.attrs["gain_e_per_dn"], dtype=np.float64)
        else:
            raise ReferencePhotometryContractError("formal delivery lacks gain_e_per_dn")
        if gain.shape not in {(), (ny, nx)} or not np.all(np.isfinite(gain)) or np.any(gain <= 0.0):
            raise ReferencePhotometryContractError(
                "streamed formal delivery gain must be positive scalar or stamp map"
            )
        starts = np.asarray(handle["time_start_seconds"], dtype=np.float64)
        exposure = np.asarray(handle["exposure_seconds"], dtype=np.float64)
        raw_start = np.asarray(handle["raw_frame_start_index"], dtype=np.int64)
        raw_stop = np.asarray(handle["raw_frame_stop_index_exclusive"], dtype=np.int64)
        if (
            not np.all(np.isfinite(starts))
            or not np.all(np.isfinite(exposure))
            or np.any(exposure <= 0.0)
            or np.any(raw_start < 0)
            or np.any(raw_stop - raw_start != coadd_factor)
            or (n_frames > 1 and not np.all(np.diff(starts) > 0.0))
            or (n_frames > 1 and not np.all(raw_start[1:] == raw_stop[:-1]))
        ):
            raise ReferencePhotometryContractError(
                "formal delivery has invalid frame intervals"
            )
        manifest = _formal_json_dataset(handle, "manifest_json")
        provenance = _formal_json_dataset(handle, "provenance_json")
    return _FormalDeliveryHeader(
        path=source,
        product_kind=product_kind,
        coadd_factor=coadd_factor,
        frame_count=n_frames,
        stamp_shape=(ny, nx),
        gain_e_per_dn=gain,
        manifest_identity=_formal_identity_json(manifest, omit=frozenset({"time_shard"})),
        provenance_identity=_formal_identity_json(provenance, omit=frozenset()),
        first_raw_frame_start=int(raw_start[0]),
        last_raw_frame_stop=int(raw_stop[-1]),
        first_time_start_seconds=float(starts[0]),
        last_time_end_seconds=float(starts[-1] + exposure[-1]),
    )


def reduce_stamp_delivery_series_v1(
    bundle_paths: Iterable[Path | str],
    *,
    cdpp_windows_minutes: Iterable[int] = STANDARD_CDPP_WINDOWS_MINUTES,
    bin_origin_seconds: float = 0.0,
    batch_frames: int = 4_096,
) -> ReferencePhotometryResult:
    """Stream a target/product series across contiguous formal time shards.

    This is the production reduction path: it reads only the central 13x13
    aperture from each HDF5 image cube, verifies global raw-frame continuity,
    and computes CDPP once over the entire series.  It never averages
    shard-level CDPP values, which would be mathematically invalid for MAD.
    """

    if isinstance(batch_frames, (bool, np.bool_)) or int(batch_frames) <= 0:
        raise ReferencePhotometryContractError("batch_frames must be positive")
    headers = tuple(_read_formal_delivery_header(path) for path in bundle_paths)
    if not headers:
        raise ReferencePhotometryContractError("at least one formal delivery bundle is required")
    headers = tuple(sorted(headers, key=lambda item: item.first_raw_frame_start))
    first = headers[0]
    ny, nx = first.stamp_shape
    aperture_mask = _fixed_central_aperture_mask(ny=ny, nx=nx)
    y_indices, x_indices = np.nonzero(aperture_mask)
    y0, y1 = int(np.min(y_indices)), int(np.max(y_indices)) + 1
    x0, x1 = int(np.min(x_indices)), int(np.max(x_indices)) + 1
    pixel_count = int(np.count_nonzero(aperture_mask))
    previous_raw_stop: int | None = None
    previous_time_end: float | None = None
    for header in headers:
        if (
            header.product_kind != first.product_kind
            or header.coadd_factor != first.coadd_factor
            or header.stamp_shape != first.stamp_shape
            or not np.array_equal(header.gain_e_per_dn, first.gain_e_per_dn)
            or header.manifest_identity != first.manifest_identity
            or header.provenance_identity != first.provenance_identity
        ):
            raise ReferencePhotometryContractError(
                "formal delivery series contains incompatible shard identities"
            )
        if previous_raw_stop is not None and (
            header.first_raw_frame_start != previous_raw_stop
            or not math.isclose(
                header.first_time_start_seconds,
                float(previous_time_end),
                rel_tol=0.0,
                abs_tol=1e-8,
            )
        ):
            raise ReferencePhotometryContractError(
                "formal delivery shards are not globally continuous"
            )
        previous_raw_stop = header.last_raw_frame_stop
        previous_time_end = header.last_time_end_seconds

    try:
        import h5py
    except ImportError as error:  # pragma: no cover - packaging guard.
        raise RuntimeError("h5py is required to stream formal delivery bundles") from error

    time_parts: list[NDArray[np.float64]] = []
    exposure_parts: list[NDArray[np.float64]] = []
    flux_parts: list[NDArray[np.float64]] = []
    valid_parts: list[NDArray[np.bool_]] = []
    usable_count_parts: list[NDArray[np.int64]] = []
    raw_start_parts: list[NDArray[np.int64]] = []
    raw_stop_parts: list[NDArray[np.int64]] = []
    expected_raw_start: int | None = None
    expected_time_start: float | None = None
    for header in headers:
        with h5py.File(header.path, "r") as handle:
            if header.gain_e_per_dn.shape == ():
                gain_crop: float | NDArray[np.float64] = float(header.gain_e_per_dn)
            else:
                gain_crop = header.gain_e_per_dn[y0:y1, x0:x1]
            for offset in range(0, header.frame_count, int(batch_frames)):
                stop = min(offset + int(batch_frames), header.frame_count)
                frame_slice = slice(offset, stop)
                final = np.asarray(
                    handle["final_dn"][frame_slice, y0:y1, x0:x1],
                    dtype=np.float64,
                )
                background = np.asarray(
                    handle["background_expectation_e"][frame_slice, y0:y1, x0:x1],
                    dtype=np.float64,
                )
                bias = np.asarray(handle["bias_level_sum_dn"][frame_slice], dtype=np.float64)
                column = np.asarray(
                    handle["column_noise_sum_dn_by_x"][frame_slice, x0:x1],
                    dtype=np.float64,
                )
                valid = np.asarray(
                    handle["valid_mask"][frame_slice, y0:y1, x0:x1],
                    dtype=bool,
                )
                fullwell = np.asarray(
                    handle["fullwell_count"][frame_slice, y0:y1, x0:x1],
                    dtype=np.uint16,
                )
                adc_low = np.asarray(
                    handle["adc_low_count"][frame_slice, y0:y1, x0:x1],
                    dtype=np.uint16,
                )
                adc_high = np.asarray(
                    handle["adc_high_count"][frame_slice, y0:y1, x0:x1],
                    dtype=np.uint16,
                )
                cosmic_count = np.asarray(
                    handle["cosmic_count"][frame_slice, y0:y1, x0:x1],
                    dtype=np.uint16,
                )
                saturated = np.asarray(
                    handle["saturated_mask"][frame_slice, y0:y1, x0:x1],
                    dtype=bool,
                )
                cosmic = np.asarray(
                    handle["cosmic_mask"][frame_slice, y0:y1, x0:x1],
                    dtype=bool,
                )
                time = np.asarray(handle["time_start_seconds"][frame_slice], dtype=np.float64)
                exposure = np.asarray(handle["exposure_seconds"][frame_slice], dtype=np.float64)
                raw_start = np.asarray(
                    handle["raw_frame_start_index"][frame_slice], dtype=np.int64
                )
                raw_stop = np.asarray(
                    handle["raw_frame_stop_index_exclusive"][frame_slice], dtype=np.int64
                )
                if (
                    not np.all(np.isfinite(background))
                    or np.any(background < 0.0)
                    or not np.all(np.isfinite(bias))
                    or not np.all(np.isfinite(column))
                    or not np.all(np.isfinite(time))
                    or not np.all(np.isfinite(exposure))
                    or np.any(exposure <= 0.0)
                    or np.any(raw_stop - raw_start != header.coadd_factor)
                    or np.any(fullwell > header.coadd_factor)
                    or np.any(adc_low > header.coadd_factor)
                    or np.any(adc_high > header.coadd_factor)
                    or np.any(cosmic_count > header.coadd_factor)
                    or not np.array_equal(
                        saturated,
                        (fullwell > 0) | (adc_low > 0) | (adc_high > 0),
                    )
                    or not np.array_equal(cosmic, cosmic_count > 0)
                ):
                    raise ReferencePhotometryContractError(
                        "formal delivery aperture planes violate the delivery contract"
                    )
                if expected_raw_start is not None and (
                    int(raw_start[0]) != expected_raw_start
                    or not math.isclose(
                        float(time[0]),
                        float(expected_time_start),
                        rel_tol=0.0,
                        abs_tol=1e-8,
                    )
                ):
                    raise ReferencePhotometryContractError(
                        "formal delivery frames are not globally continuous"
                    )
                if time.size > 1 and (
                    not np.all(np.diff(time) > 0.0)
                    or not np.all(raw_start[1:] == raw_stop[:-1])
                ):
                    raise ReferencePhotometryContractError(
                        "formal delivery frames are not globally continuous"
                    )
                expected_raw_start = int(raw_stop[-1])
                expected_time_start = float(time[-1] + exposure[-1])
                usable = valid & ~saturated & ~cosmic
                usable_counts = np.count_nonzero(
                    usable.reshape(usable.shape[0], -1),
                    axis=1,
                ).astype(np.int64, copy=False)
                aperture_valid = usable_counts == pixel_count
                calibrated = (
                    (final - bias[:, None, None] - column[:, None, :]) * gain_crop
                    - background
                )
                flux = np.full(time.shape, np.nan, dtype=np.float64)
                if np.any(aperture_valid):
                    flux[aperture_valid] = np.sum(
                        calibrated[aperture_valid],
                        axis=(1, 2),
                        dtype=np.float64,
                    )
                time_parts.append(time)
                exposure_parts.append(exposure)
                flux_parts.append(flux)
                valid_parts.append(aperture_valid)
                usable_count_parts.append(usable_counts)
                raw_start_parts.append(raw_start)
                raw_stop_parts.append(raw_stop)

    time_seconds = np.concatenate(time_parts)
    exposure_seconds = np.concatenate(exposure_parts)
    flux_e = np.concatenate(flux_parts)
    aperture_valid = np.concatenate(valid_parts)
    usable_counts = np.concatenate(usable_count_parts)
    raw_frame_start_index = np.concatenate(raw_start_parts)
    raw_frame_stop_index_exclusive = np.concatenate(raw_stop_parts)
    cdpp_by_window_minutes = compute_cadence_aware_cdpp(
        time_seconds=time_seconds,
        flux_e=np.nan_to_num(flux_e, nan=0.0),
        aperture_valid=aperture_valid,
        exposure_seconds=exposure_seconds,
        windows_minutes=cdpp_windows_minutes,
        bin_origin_seconds=bin_origin_seconds,
    )
    return ReferencePhotometryResult(
        time_seconds=time_seconds,
        flux_e=flux_e,
        aperture_valid=aperture_valid,
        aperture_usable_pixel_count=usable_counts,
        aperture_mask=aperture_mask,
        aperture_shape=FIXED_APERTURE_SHAPE,
        aperture_pixel_count=pixel_count,
        exposure_seconds=exposure_seconds,
        cdpp_by_window_minutes=cdpp_by_window_minutes,
        product_semantics={
            "observation_product": "final_dn",
            "calibrated_electron_product": "derived",
            "background_subtraction": "background_expectation_e_only",
            "background_realization_used": False,
            "mask_policy": "invalidate_whole_fixed_aperture_cadence",
            "input_mode": "streamed_formal_delivery_shards",
        },
        raw_frame_start_index=raw_frame_start_index,
        raw_frame_stop_index_exclusive=raw_frame_stop_index_exclusive,
    )
