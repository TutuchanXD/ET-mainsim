"""Galaxy-team FITS light-curve ingestion for independent stamp production.

The submitted FITS stores ``relative_flux = Delta F / F_ref`` on a native
time grid.  This module deliberately ignores the absolute native epoch: the
first finite native sample is aligned to simulation raw-frame zero.  Native
time *intervals* are retained solely to construct a piecewise-linear clean
flux curve and calculate a mean factor over every 10 s detector exposure.

The resulting factor is ``q = 1 + Delta F / F_ref`` and is suitable for the
Photsim7 source-variability injection point before PSF rendering and photon
noise.  It is not a noisy light curve and it never copies a 120 s sample into
multiple 10 s frames without exposure averaging.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .stamp_inputs import file_identity


GALAXY_LIGHTCURVE_SCHEMA_ID = "et_mainsim.galaxy_lightcurve_input.v1"
GALAXY_FACTOR_SNAPSHOT_SCHEMA_ID = "et_mainsim.galaxy_factor_snapshot.v1"


def _strict_source_id(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a signed int64 Gaia source ID")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a signed int64 Gaia source ID") from error
    if result < 0 or result > int(np.iinfo(np.int64).max):
        raise ValueError(f"{name} must be a non-negative signed int64 Gaia source ID")
    return result


def _finite_scalar(value: Any, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _finite_vector(value: ArrayLike, *, name: str) -> NDArray[np.float64]:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if result.ndim != 1 or result.size < 2:
        raise ValueError(f"{name} must be a one-dimensional array with at least 2 rows")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    result = np.array(result, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _normalise_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("input_identity must be a mapping")
    candidate = dict(value)
    try:
        json.dumps(candidate, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as error:
        raise ValueError("input_identity must be JSON serializable") from error
    return candidate


@dataclass(frozen=True)
class GalaxyLightCurve:
    """One finite Galaxy-team clean curve after FITS padding removal."""

    source_id: int
    gaia_g_mag: float
    ra_deg: float
    dec_deg: float
    source_class: str
    native_time_seconds: NDArray[np.float64]
    clean_flux_factor: NDArray[np.float64]
    input_identity: Mapping[str, Any]

    def __post_init__(self) -> None:
        source_id = _strict_source_id(self.source_id, name="source_id")
        gaia_g_mag = _finite_scalar(self.gaia_g_mag, name="gaia_g_mag")
        ra_deg = _finite_scalar(self.ra_deg, name="ra_deg")
        dec_deg = _finite_scalar(self.dec_deg, name="dec_deg")
        source_class = str(self.source_class).strip()
        if not source_class:
            raise ValueError("source_class must be non-empty")
        time = _finite_vector(self.native_time_seconds, name="native_time_seconds")
        factors = _finite_vector(self.clean_flux_factor, name="clean_flux_factor")
        if factors.shape != time.shape:
            raise ValueError("clean_flux_factor must match native_time_seconds")
        if np.any(np.diff(time) <= 0.0):
            raise ValueError("native_time_seconds must be strictly increasing")
        if not np.isclose(time[0], 0.0, rtol=0.0, atol=1e-12):
            raise ValueError("native_time_seconds must be normalized to start at zero")
        if np.any(factors <= 0.0):
            raise ValueError("clean_flux_factor must be strictly positive")
        factors.setflags(write=False)
        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(self, "gaia_g_mag", gaia_g_mag)
        object.__setattr__(self, "ra_deg", ra_deg)
        object.__setattr__(self, "dec_deg", dec_deg)
        object.__setattr__(self, "source_class", source_class)
        object.__setattr__(self, "native_time_seconds", time)
        object.__setattr__(self, "clean_flux_factor", factors)
        object.__setattr__(self, "input_identity", _normalise_identity(self.input_identity))

    def to_metadata(self) -> dict[str, Any]:
        return {
            "schema_id": GALAXY_LIGHTCURVE_SCHEMA_ID,
            "source_id": str(self.source_id),
            "source_id_int64": self.source_id,
            "gaia_g_mag": self.gaia_g_mag,
            "magnitude_system": "Gaia_G_Vega",
            "ra_deg": self.ra_deg,
            "dec_deg": self.dec_deg,
            "source_class": self.source_class,
            "native_curve": {
                "sample_count": int(self.native_time_seconds.size),
                "start_time_seconds_ignored_for_simulation": True,
                "relative_time_origin_seconds": 0.0,
            },
            "q_definition": "1_plus_delta_f_over_f_ref",
            "clean_flux_selection": "relative_flux_column_without_noise",
            "input_identity": dict(self.input_identity),
        }


@dataclass(frozen=True)
class GalaxyFactorSnapshot:
    """A frozen all-raw-frame factor vector ready for one target run."""

    source_id: int
    factors: NDArray[np.float64]
    metadata: Mapping[str, Any]
    metadata_json: str

    def __post_init__(self) -> None:
        source_id = _strict_source_id(self.source_id, name="source_id")
        factors = _finite_vector(self.factors, name="factors")
        if np.any(factors <= 0.0):
            raise ValueError("factors must be strictly positive")
        metadata = _normalise_identity(self.metadata)
        try:
            decoded = json.loads(str(self.metadata_json))
        except json.JSONDecodeError as error:
            raise ValueError("metadata_json must be valid JSON") from error
        if not isinstance(decoded, dict) or decoded != metadata:
            raise ValueError("metadata_json must encode exactly metadata")
        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(self, "factors", factors)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "metadata_json", str(self.metadata_json))


def _piecewise_linear_integral(
    *,
    time_seconds: NDArray[np.float64],
    factors: NDArray[np.float64],
    query_seconds: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Integrate q(t) from zero to each in-range query time exactly."""

    if np.any(query_seconds < 0.0) or np.any(query_seconds > time_seconds[-1]):
        raise ValueError("query time is outside the native clean-flux coverage")
    widths = np.diff(time_seconds)
    segment_areas = widths * (factors[:-1] + factors[1:]) / 2.0
    prefix = np.concatenate(([0.0], np.cumsum(segment_areas, dtype=np.float64)))
    indices = np.searchsorted(time_seconds, query_seconds, side="right") - 1
    indices = np.clip(indices, 0, time_seconds.size - 2)
    local_time = query_seconds - time_seconds[indices]
    slopes = (factors[indices + 1] - factors[indices]) / widths[indices]
    result = prefix[indices] + factors[indices] * local_time + 0.5 * slopes * local_time**2
    at_end = np.isclose(query_seconds, time_seconds[-1], rtol=0.0, atol=1e-10)
    result[at_end] = prefix[-1]
    return result


def exposure_averaged_factors(
    *,
    native_time_seconds: ArrayLike,
    clean_flux_factor: ArrayLike,
    n_raw_frames: int,
    raw_exposure_seconds: float,
) -> NDArray[np.float64]:
    """Sample a clean piecewise-linear curve as exact raw-exposure means.

    Simulation frame zero corresponds to the first finite native sample.  The
    native absolute epoch is intentionally absent from this calculation.
    """

    time = _finite_vector(native_time_seconds, name="native_time_seconds")
    factors = _finite_vector(clean_flux_factor, name="clean_flux_factor")
    if time.shape != factors.shape:
        raise ValueError("clean_flux_factor must match native_time_seconds")
    if not np.isclose(time[0], 0.0, rtol=0.0, atol=1e-12):
        raise ValueError("native_time_seconds must start at zero")
    if np.any(np.diff(time) <= 0.0):
        raise ValueError("native_time_seconds must be strictly increasing")
    if np.any(factors <= 0.0):
        raise ValueError("clean_flux_factor must be strictly positive")
    if isinstance(n_raw_frames, (bool, np.bool_)) or int(n_raw_frames) <= 0:
        raise ValueError("n_raw_frames must be a positive integer")
    n_frames = int(n_raw_frames)
    exposure = _finite_scalar(raw_exposure_seconds, name="raw_exposure_seconds")
    if exposure <= 0.0:
        raise ValueError("raw_exposure_seconds must be positive")
    starts = np.arange(n_frames, dtype=np.float64) * exposure
    stops = starts + exposure
    tolerance = max(1e-9, abs(float(time[-1])) * 1e-12)
    if stops[-1] > time[-1] + tolerance:
        raise ValueError(
            "native clean-flux coverage does not cover all requested raw exposures"
        )
    stops = np.minimum(stops, time[-1])
    integrated = _piecewise_linear_integral(
        time_seconds=time,
        factors=factors,
        query_seconds=stops,
    ) - _piecewise_linear_integral(
        time_seconds=time,
        factors=factors,
        query_seconds=starts,
    )
    result = integrated / exposure
    if not np.all(np.isfinite(result)) or np.any(result <= 0.0):
        raise ValueError("exposure-averaged factors must be finite and strictly positive")
    result.setflags(write=False)
    return result


def _selected_rows(
    path: Path,
    *,
    requested: tuple[int, ...] | None,
) -> dict[int, dict[str, Any]]:
    """Copy only selected variable-length FITS rows before closing the file."""

    from astropy.io import fits

    with fits.open(path, memmap=True) as handle:
        for hdu in handle:
            data = getattr(hdu, "data", None)
            names = getattr(data, "names", None)
            if names and {"Source", "Gmag", "RAJ2000", "DEJ2000", "time", "relative_flux"}.issubset(names):
                selected_set = None if requested is None else set(requested)
                rows: dict[int, dict[str, Any]] = {}
                for row in data:
                    source_id = _strict_source_id(row["Source"], name="FITS Source")
                    if selected_set is not None and source_id not in selected_set:
                        continue
                    if source_id in rows:
                        raise ValueError(f"Galaxy FITS has duplicate Source={source_id}")
                    rows[source_id] = {
                        "source_id": source_id,
                        "gaia_g_mag": row["Gmag"].item(),
                        "ra_deg": row["RAJ2000"].item(),
                        "dec_deg": row["DEJ2000"].item(),
                        "source_class": str(row["class"]),
                        # Variable-length FITS columns point into a heap.  A
                        # structured-array copy preserves only heap offsets,
                        # so copy each referenced vector while the HDU is live.
                        "native_days": np.array(row["time"], dtype=np.float64, copy=True),
                        "delta_flux": np.array(
                            row["relative_flux"], dtype=np.float64, copy=True
                        ),
                    }
                return rows
    raise ValueError("Galaxy FITS has no table with the required source/light-curve columns")


def load_galaxy_lightcurves(
    path: Path | str,
    *,
    source_ids: Iterable[int] | None = None,
) -> dict[int, GalaxyLightCurve]:
    """Load selected Galaxy FITS curves, stripping terminal NaN padding.

    FITS ``time`` is interpreted as day offsets only to retain the native
    spacing; its absolute origin is reset to zero before any simulation use.
    """

    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Galaxy FITS does not exist: {source_path}")
    requested = (
        None
        if source_ids is None
        else tuple(_strict_source_id(value, name="source_ids") for value in source_ids)
    )
    if requested is not None and len(set(requested)) != len(requested):
        raise ValueError("source_ids must not contain duplicates")
    rows = _selected_rows(source_path, requested=requested)
    identity = file_identity(source_path)
    by_source: dict[int, Mapping[str, Any]] = rows
    selected_ids = tuple(by_source) if requested is None else requested
    missing = sorted(set(selected_ids) - set(by_source))
    if missing:
        raise ValueError(f"Galaxy FITS is missing requested source IDs: {missing}")

    result: dict[int, GalaxyLightCurve] = {}
    for source_id in selected_ids:
        row = by_source[source_id]
        native_days = np.asarray(row["native_days"], dtype=np.float64)
        delta_flux = np.asarray(row["delta_flux"], dtype=np.float64)
        finite = np.isfinite(native_days) & np.isfinite(delta_flux)
        native_days = native_days[finite]
        delta_flux = delta_flux[finite]
        if native_days.size < 2:
            raise ValueError(f"Galaxy source {source_id} has fewer than two finite nodes")
        native_seconds = (native_days - native_days[0]) * 86400.0
        clean_factor = 1.0 + delta_flux
        result[source_id] = GalaxyLightCurve(
            source_id=source_id,
            gaia_g_mag=row["gaia_g_mag"],
            ra_deg=row["ra_deg"],
            dec_deg=row["dec_deg"],
            source_class=row["source_class"],
            native_time_seconds=native_seconds,
            clean_flux_factor=clean_factor,
            input_identity=identity,
        )
    return result


def write_galaxy_factor_snapshot(
    path: Path | str,
    *,
    curve: GalaxyLightCurve,
    factors: ArrayLike,
    raw_exposure_seconds: float,
) -> dict[str, Any]:
    """Write a compact, immutable factor snapshot and return its identity."""

    if not isinstance(curve, GalaxyLightCurve):
        raise TypeError("curve must be a GalaxyLightCurve")
    factors_array = _finite_vector(factors, name="factors")
    if np.any(factors_array <= 0.0):
        raise ValueError("factors must be strictly positive")
    exposure = _finite_scalar(raw_exposure_seconds, name="raw_exposure_seconds")
    if exposure <= 0.0:
        raise ValueError("raw_exposure_seconds must be positive")
    metadata = {
        **curve.to_metadata(),
        "schema_id": GALAXY_FACTOR_SNAPSHOT_SCHEMA_ID,
        "raw_exposure_seconds": exposure,
        "n_raw_frames": int(factors_array.size),
        "time_alignment": "simulation_raw_frame_index",
        "interpolation": "piecewise_linear_clean_flux",
        "exposure_sampling": "exact_interval_mean",
        "extrapolation": "forbidden",
    }
    metadata_json = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        source_id=np.asarray(curve.source_id, dtype=np.int64),
        factors=np.asarray(factors_array, dtype=np.float64),
        metadata_json=np.asarray(metadata_json),
    )
    return file_identity(target)


def read_galaxy_factor_snapshot(path: Path | str) -> GalaxyFactorSnapshot:
    """Read one factor snapshot and reject malformed semantic metadata."""

    source = Path(path).expanduser().resolve()
    with np.load(source, allow_pickle=False) as payload:
        required = {"source_id", "factors", "metadata_json"}
        missing = required - set(payload.files)
        if missing:
            raise ValueError(f"Galaxy factor snapshot is missing arrays: {sorted(missing)}")
        source_id = _strict_source_id(payload["source_id"].reshape(()).item(), name="source_id")
        factors = np.asarray(payload["factors"], dtype=np.float64)
        raw_metadata = payload["metadata_json"].reshape(()).item()
    metadata_json = str(raw_metadata)
    try:
        metadata = json.loads(metadata_json)
    except json.JSONDecodeError as error:
        raise ValueError("Galaxy factor snapshot metadata_json is invalid") from error
    if not isinstance(metadata, dict):
        raise ValueError("Galaxy factor snapshot metadata must be an object")
    if metadata.get("schema_id") != GALAXY_FACTOR_SNAPSHOT_SCHEMA_ID:
        raise ValueError("unsupported Galaxy factor snapshot schema")
    if _strict_source_id(metadata.get("source_id_int64"), name="metadata.source_id_int64") != source_id:
        raise ValueError("Galaxy factor snapshot source_id disagrees with metadata")
    if metadata.get("q_definition") != "1_plus_delta_f_over_f_ref":
        raise ValueError("Galaxy factor snapshot q_definition is unsupported")
    if metadata.get("time_alignment") != "simulation_raw_frame_index":
        raise ValueError("Galaxy factor snapshot time_alignment is unsupported")
    if int(metadata.get("n_raw_frames", -1)) != int(factors.size):
        raise ValueError("Galaxy factor snapshot n_raw_frames disagrees with factors")
    return GalaxyFactorSnapshot(
        source_id=source_id,
        factors=factors,
        metadata=metadata,
        metadata_json=metadata_json,
    )


__all__ = [
    "GALAXY_FACTOR_SNAPSHOT_SCHEMA_ID",
    "GALAXY_LIGHTCURVE_SCHEMA_ID",
    "GalaxyFactorSnapshot",
    "GalaxyLightCurve",
    "exposure_averaged_factors",
    "load_galaxy_lightcurves",
    "read_galaxy_factor_snapshot",
    "write_galaxy_factor_snapshot",
]
