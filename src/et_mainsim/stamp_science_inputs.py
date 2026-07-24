"""Frozen Aster, varlc, and wdlc inputs for formal stamp production.

The science-team files carry external string identifiers and relative source
signals, while Photsim7 requires signed-int64 catalog identities and one
dimensionless factor per simulation raw frame.  This module is the common,
fail-closed boundary between those two contracts.  It deliberately does not
interpret input epochs as simulation time: input row zero always maps to raw
frame zero.

All three tracks use the approved no-coordinate policy: a non-physical
``main_rd`` reference position, explicit 12-degree PSF node (ID 6), and DVA
disabled.  The returned factors are read-only and can be frozen into the same
snapshot format for a team-independent production worker.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Literal
import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .stamp_inputs import file_identity


SCIENCE_INPUT_CURVE_SCHEMA_ID = "et_mainsim.stamp_science_input_curve.v1"
SCIENCE_FACTOR_SNAPSHOT_SCHEMA_ID = (
    "et_mainsim.stamp_science_factor_snapshot.v1"
)
NAMESPACED_SOURCE_ID_SCHEMA_ID = (
    "et_mainsim.namespaced_source_id.frozen_explicit_mapping.v1"
)

RAW_EXPOSURE_SECONDS = 10.0
RAW_FRAMES_90D = 777_600
WDLC_NORMALIZATION_RAW_FRAMES = 3_153_600
DEFAULT_GAIA_G_VEGA = 11.5
EXPLICIT_PSF_ID = 6
PSF_NODE_ANGLE_DEG = 12.0
REFERENCE_DETECTOR_ID = "main_rd"
LOCATION_MODE = "reference_field_nonphysical"
NO_COORDINATE_POLICY = "disable"

# The submitted mode tables round frequency, amplitude, and phase to six
# decimal places.  Over the approved 180-day campaign that representation
# differs from the team's derived WD time series by 11.7717 ppm at the single
# worst sample while retaining a 1.689 ppm RMS.  The 15-ppm maximum budget is
# therefore specific to the approved 180-day product; the independent 2-ppm
# RMS guard remains unchanged, and these tolerances are not a 270/365-day
# accuracy claim.
WDLC_MODE_GATE_MAX_ABS = 1.5e-5
WDLC_MODE_GATE_RMS = 2.0e-6
WDLC_ELECTRON_GATE_MAX_ABS = 5.1e-9
WDLC_ELECTRON_GATE_RMS = 3.0e-9

Track = Literal["aster", "varlc", "wdlc"]


ASTER_PRECISION_SOURCES = (
    ("0000000473", "F_dwarf", 1500.0, 1500.0),
    ("0000000599", "G_dwarf", 3000.0, 1500.0),
    ("0000000036", "K_dwarf", 4500.0, 1500.0),
    ("0000000086", "subgiant", 6000.0, 1500.0),
    ("0000000622", "red_giant", 7500.0, 1500.0),
)

VARLC_SOURCES = (
    ("KIC003331147", 2000.0, 4500.0),
    ("KIC011145123", 4450.0, 4500.0),
    ("TIC260161111", 6900.0, 4500.0),
)

WDLC_SOURCES = (
    ("wd", "WD", 1_000_000.0, 3000.0, 7500.0),
    ("sdb", "sdB", 750_000.0, 6000.0, 7500.0),
)

_FROZEN_INTERNAL_SOURCE_IDS = {
    ("aster", "0000000473"): 473,
    ("aster", "0000000599"): 599,
    ("aster", "0000000036"): 36,
    ("aster", "0000000086"): 86,
    ("aster", "0000000622"): 622,
    ("varlc", "KIC003331147"): 3_331_147,
    ("varlc", "KIC011145123"): 11_145_123,
    ("varlc", "TIC260161111"): 260_161_111,
    ("wdlc", "WD"): 1,
    ("wdlc", "sdB"): 2,
}


def _nonempty_text(value: object, *, name: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} must be non-empty")
    return result


def _finite(value: object, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_frame_count(value: object, *, name: str = "raw_frame_count") -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if result <= 0 or result != value:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _json_object(value: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    try:
        encoded = json.dumps(
            dict(value),
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be JSON serializable") from error
    if not isinstance(decoded, dict):
        raise ValueError(f"{name} must encode a JSON object")
    return decoded


def _readonly_positive_factors(value: ArrayLike) -> NDArray[np.float64]:
    try:
        factors = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("factors must contain numeric values") from error
    if factors.ndim != 1 or factors.size == 0:
        raise ValueError("factors must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(factors)):
        raise ValueError("factors must contain only finite values")
    if np.any(factors <= 0.0):
        raise ValueError("factors must be strictly positive")
    result = np.array(factors, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def stable_namespaced_source_id(namespace: str, external_source_id: str) -> int:
    """Return the explicitly frozen internal int64 for one external identity.

    The integer alone is not globally unique and must never replace the full
    ``(namespace, external_source_id)`` identity.  Unknown pairs fail closed;
    production is not allowed to invent a new mapping implicitly.
    """

    canonical_namespace = _nonempty_text(namespace, name="namespace").lower()
    external = _nonempty_text(
        external_source_id,
        name="external_source_id",
    )
    try:
        return _FROZEN_INTERNAL_SOURCE_IDS[(canonical_namespace, external)]
    except KeyError as error:
        raise ValueError(
            "source identity has no frozen internal int64 mapping: "
            f"namespace={canonical_namespace!r}, external_source_id={external!r}"
        ) from error


def _source_identity(namespace: str, external_source_id: str) -> dict[str, Any]:
    canonical_namespace = _nonempty_text(namespace, name="namespace").lower()
    external = _nonempty_text(external_source_id, name="external_source_id")
    return {
        "schema_id": NAMESPACED_SOURCE_ID_SCHEMA_ID,
        "namespace": canonical_namespace,
        "external_source_id": external,
        "source_id_int64": stable_namespaced_source_id(
            canonical_namespace,
            external,
        ),
        "mapping": "frozen_explicit_mapping_v1",
        "global_identity_fields": ["namespace", "external_source_id"],
    }


@dataclass(frozen=True)
class ScienceInputCurve:
    """One production-ready, namespaced source-factor sequence."""

    track: Track | str
    namespace: str
    external_source_id: str
    source_id_int64: int
    source_class: str
    gaia_g_mag: float
    detector_xpix: float
    detector_ypix: float
    factors: NDArray[np.float64]
    metadata: Mapping[str, Any]
    detector_id: str = REFERENCE_DETECTOR_ID
    psf_id: int = EXPLICIT_PSF_ID
    psf_node_angle_deg: float = PSF_NODE_ANGLE_DEG
    location_mode: str = LOCATION_MODE
    dva_enabled: bool = False

    def __post_init__(self) -> None:
        track = _nonempty_text(self.track, name="track").lower()
        if track not in {"aster", "varlc", "wdlc"}:
            raise ValueError("track must be one of 'aster', 'varlc', or 'wdlc'")
        namespace = _nonempty_text(self.namespace, name="namespace").lower()
        if namespace != track:
            raise ValueError(
                "namespace must equal track for the global science-source identity"
            )
        external = _nonempty_text(
            self.external_source_id,
            name="external_source_id",
        )
        expected_source_id = stable_namespaced_source_id(namespace, external)
        if isinstance(self.source_id_int64, (bool, np.bool_)):
            raise ValueError("source_id_int64 must match the namespaced identity")
        try:
            source_id = int(self.source_id_int64)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                "source_id_int64 must match the namespaced identity"
            ) from error
        if source_id != expected_source_id:
            raise ValueError("source_id_int64 must match the namespaced identity")
        source_class = _nonempty_text(self.source_class, name="source_class")
        gaia_g_mag = _finite(self.gaia_g_mag, name="gaia_g_mag")
        detector_xpix = _finite(self.detector_xpix, name="detector_xpix")
        detector_ypix = _finite(self.detector_ypix, name="detector_ypix")
        detector_id = _nonempty_text(self.detector_id, name="detector_id")
        if detector_id != REFERENCE_DETECTOR_ID:
            raise ValueError("no-coordinate science inputs require detector_id='main_rd'")
        if isinstance(self.psf_id, (bool, np.bool_)) or int(self.psf_id) != 6:
            raise ValueError("no-coordinate science inputs require PSF ID 6")
        psf_angle = _finite(self.psf_node_angle_deg, name="psf_node_angle_deg")
        if not math.isclose(psf_angle, 12.0, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("no-coordinate science inputs require the 12-degree PSF")
        if str(self.location_mode) != LOCATION_MODE:
            raise ValueError(
                "no-coordinate science inputs require reference_field_nonphysical"
            )
        if self.dva_enabled is not False:
            raise ValueError("DVA must be disabled for no-coordinate science inputs")

        object.__setattr__(self, "track", track)
        object.__setattr__(self, "namespace", namespace)
        object.__setattr__(self, "external_source_id", external)
        object.__setattr__(self, "source_id_int64", source_id)
        object.__setattr__(self, "source_class", source_class)
        object.__setattr__(self, "gaia_g_mag", gaia_g_mag)
        object.__setattr__(self, "detector_xpix", detector_xpix)
        object.__setattr__(self, "detector_ypix", detector_ypix)
        object.__setattr__(self, "detector_id", detector_id)
        object.__setattr__(self, "psf_id", int(self.psf_id))
        object.__setattr__(self, "psf_node_angle_deg", psf_angle)
        object.__setattr__(self, "factors", _readonly_positive_factors(self.factors))
        object.__setattr__(
            self,
            "metadata",
            _json_object(self.metadata, name="metadata"),
        )

    @property
    def source_identity(self) -> dict[str, Any]:
        return _source_identity(self.namespace, self.external_source_id)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "schema_id": SCIENCE_INPUT_CURVE_SCHEMA_ID,
            "schema_version": 1,
            "track": self.track,
            "source_identity": self.source_identity,
            "source_class": self.source_class,
            "gaia_g_mag": self.gaia_g_mag,
            "magnitude_system": "Gaia_G_Vega",
            "geometry": {
                "detector_id": self.detector_id,
                "detector_xpix": self.detector_xpix,
                "detector_ypix": self.detector_ypix,
                "location_mode": self.location_mode,
                "physical_geometry_claim": False,
                "sky_coordinate": None,
            },
            "psf": {
                "chosen_psf_id": self.psf_id,
                "node_angle_deg": self.psf_node_angle_deg,
                "selection": "explicit_not_nearest_from_reference_position",
            },
            "dva": {
                "enabled": self.dva_enabled,
                "no_coordinate_policy": NO_COORDINATE_POLICY,
            },
            "n_raw_frames": int(self.factors.size),
            "raw_exposure_seconds": RAW_EXPOSURE_SECONDS,
            "time_alignment": "simulation_raw_frame_index",
            "adapter_metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ScienceFactorSnapshot:
    """Read-only factor snapshot consumed by a team-independent worker."""

    source_id_int64: int
    namespace: str
    external_source_id: str
    factors: NDArray[np.float64]
    metadata: Mapping[str, Any]
    metadata_json: str

    def __post_init__(self) -> None:
        namespace = _nonempty_text(self.namespace, name="namespace").lower()
        external = _nonempty_text(
            self.external_source_id,
            name="external_source_id",
        )
        expected = stable_namespaced_source_id(namespace, external)
        if int(self.source_id_int64) != expected:
            raise ValueError("snapshot source_id_int64 conflicts with namespaced identity")
        metadata = _json_object(self.metadata, name="metadata")
        try:
            decoded = json.loads(str(self.metadata_json))
        except json.JSONDecodeError as error:
            raise ValueError("metadata_json must be valid JSON") from error
        if decoded != metadata:
            raise ValueError("metadata_json must encode exactly metadata")
        object.__setattr__(self, "source_id_int64", expected)
        object.__setattr__(self, "namespace", namespace)
        object.__setattr__(self, "external_source_id", external)
        object.__setattr__(self, "factors", _readonly_positive_factors(self.factors))
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "metadata_json", str(self.metadata_json))


def _track_root(input_root: Path | str, directory_name: str) -> Path:
    root = Path(input_root).expanduser().resolve()
    candidate = root / directory_name
    if candidate.is_dir():
        return candidate
    if root.name.lower() == directory_name.lower() and root.is_dir():
        return root
    return candidate


def _load_text_columns(
    path: Path,
    *,
    raw_frame_count: int,
    usecols: Sequence[int],
    skiprows: int = 0,
) -> NDArray[np.float64]:
    if not path.is_file():
        raise FileNotFoundError(f"science light curve does not exist: {path}")
    try:
        with warnings.catch_warnings():
            # NumPy 1.23+ warns that comment-only header lines no longer count
            # toward max_rows.  Data-row semantics are exactly what this
            # bounded production read requires, so silence only that notice.
            warnings.filterwarnings(
                "ignore",
                message=r"Input line .*contained no data.*max_rows=.*",
                category=UserWarning,
            )
            values = np.loadtxt(
                path,
                dtype=np.float64,
                comments="#",
                skiprows=skiprows,
                usecols=tuple(usecols),
                max_rows=raw_frame_count,
                ndmin=2,
            )
    except ValueError as error:
        raise ValueError(f"failed to parse science light curve {path}") from error
    if values.shape != (raw_frame_count, len(tuple(usecols))):
        raise ValueError(
            f"science light curve {path} has fewer than {raw_frame_count} data rows"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"science light curve {path} contains non-finite values")
    return values


def _validate_native_cadence(
    time_seconds: NDArray[np.float64],
    *,
    label: str,
) -> None:
    if time_seconds.size <= 1:
        return
    cadence = np.diff(time_seconds)
    if not np.allclose(
        cadence,
        RAW_EXPOSURE_SECONDS,
        rtol=0.0,
        atol=1.0e-5,
    ):
        raise ValueError(f"{label} native cadence must be uniformly 10 seconds")


def _integer_value_counts(values: NDArray[np.float64], *, label: str) -> dict[str, int]:
    rounded = np.rint(values)
    if not np.array_equal(values, rounded):
        raise ValueError(f"{label} must contain integer values")
    unique, counts = np.unique(rounded.astype(np.int64), return_counts=True)
    return {
        str(int(value)): int(count)
        for value, count in zip(unique, counts, strict=True)
    }


def _aster_generator_metadata(path: Path) -> dict[str, Any]:
    """Record, but do not adopt, the PSLS light-curve generator magnitude."""

    if not path.is_file():
        raise FileNotFoundError(f"Aster generator log does not exist: {path}")
    text = path.read_text(encoding="utf-8", errors="strict")
    magnitude_match = re.search(
        r"\bmagnitude\s*=\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))",
        text,
        flags=re.IGNORECASE,
    )
    sampling_match = re.search(
        r"\bsampling\s*=\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))",
        text,
        flags=re.IGNORECASE,
    )
    if magnitude_match is None or sampling_match is None:
        raise ValueError(
            f"Aster generator log lacks magnitude or sampling metadata: {path}"
        )
    magnitude = _finite(
        magnitude_match.group(1),
        name="Aster generator magnitude",
    )
    sampling = _finite(
        sampling_match.group(1),
        name="Aster generator sampling",
    )
    if not math.isclose(
        sampling,
        RAW_EXPOSURE_SECONDS,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("Aster generator log sampling must be 10 seconds")
    return {
        "magnitude": magnitude,
        "sampling_seconds": sampling,
        "magnitude_role": "not_adopted_for_precision_absolute_flux",
        "precision_gaia_g_vega": DEFAULT_GAIA_G_VEGA,
        "g6_role": "separate_saturation_validation",
    }


def _common_curve(
    *,
    track: Track,
    namespace: str,
    external_source_id: str,
    source_class: str,
    detector_xpix: float,
    detector_ypix: float,
    factors: ArrayLike,
    metadata: Mapping[str, Any],
) -> ScienceInputCurve:
    return ScienceInputCurve(
        track=track,
        namespace=namespace,
        external_source_id=external_source_id,
        source_id_int64=stable_namespaced_source_id(
            namespace,
            external_source_id,
        ),
        source_class=source_class,
        gaia_g_mag=DEFAULT_GAIA_G_VEGA,
        detector_xpix=detector_xpix,
        detector_ypix=detector_ypix,
        factors=np.asarray(factors, dtype=np.float64),
        metadata=metadata,
    )


def _reject_source_id_collisions(curves: Sequence[ScienceInputCurve]) -> None:
    ids = [curve.source_id_int64 for curve in curves]
    if len(ids) != len(set(ids)):
        raise ValueError("namespaced source IDs collide within the selected track")


def load_aster_precision_inputs(
    input_root: Path | str,
    *,
    raw_frame_count: int = RAW_FRAMES_90D,
) -> tuple[ScienceInputCurve, ...]:
    """Load the frozen five-source Aster precision sample."""

    n_frames = _positive_frame_count(raw_frame_count)
    source_root = _track_root(input_root, "Aster") / "lightcurves_test10"
    curves: list[ScienceInputCurve] = []
    for external_id, source_class, detector_x, detector_y in ASTER_PRECISION_SOURCES:
        path = source_root / f"{external_id}.dat"
        log_path = source_root / f"{external_id}.txt"
        values = _load_text_columns(
            path,
            raw_frame_count=n_frames,
            usecols=(0, 1, 2),
        )
        native_time = values[:, 0]
        ppm = values[:, 1]
        flags = values[:, 2]
        _validate_native_cadence(native_time, label=f"Aster {external_id}")
        value_counts = _integer_value_counts(
            flags,
            label=f"Aster {external_id} Flag",
        )
        factors = 1.0 + ppm * 1.0e-6
        curves.append(
            _common_curve(
                track="aster",
                namespace="aster",
                external_source_id=external_id,
                source_class=source_class,
                detector_xpix=detector_x,
                detector_ypix=detector_y,
                factors=factors,
                metadata={
                    "input_provider": "PSLS",
                    "magnitude_origin": (
                        "precision_override_of_generator_magnitude_6"
                    ),
                    "q_definition": "1_plus_ppm_times_1e_minus_6",
                    "input_file": file_identity(path),
                    "input_log_file": file_identity(log_path),
                    "input_generator_metadata": _aster_generator_metadata(
                        log_path
                    ),
                    "input_columns": {
                        "time": "Time [s]",
                        "signal": "Flux variation [ppm]",
                        "quality": "Flag",
                    },
                    "input_time": {
                        "native_start_seconds": float(native_time[0]),
                        "native_cadence_seconds": RAW_EXPOSURE_SECONDS,
                        "absolute_origin_ignored": True,
                        "first_valid_sample_maps_to_raw_frame_zero": True,
                    },
                    "input_flags": {
                        "policy": "recorded_not_applied",
                        "nonzero_count": int(np.count_nonzero(flags)),
                        "value_counts": value_counts,
                    },
                    "native_sample_interpretation": "10_second_exposure_mean",
                },
            )
        )
    _reject_source_id_collisions(curves)
    return tuple(curves)


def load_varlc_inputs(
    input_root: Path | str,
    *,
    raw_frame_count: int = RAW_FRAMES_90D,
) -> tuple[ScienceInputCurve, ...]:
    """Load all three submitted normalized-flux varlc curves."""

    n_frames = _positive_frame_count(raw_frame_count)
    source_root = _track_root(input_root, "varlc")
    curves: list[ScienceInputCurve] = []
    for external_id, detector_x, detector_y in VARLC_SOURCES:
        path = source_root / f"{external_id}_simulated_light_curve.dat"
        values = _load_text_columns(
            path,
            raw_frame_count=n_frames,
            usecols=(0, 1),
        )
        native_time_days = values[:, 0]
        factors = values[:, 1]
        _validate_native_cadence(
            native_time_days * 86_400.0,
            label=f"varlc {external_id}",
        )
        curves.append(
            _common_curve(
                track="varlc",
                namespace="varlc",
                external_source_id=external_id,
                source_class="pulsating_variable",
                detector_xpix=detector_x,
                detector_ypix=detector_y,
                factors=factors,
                metadata={
                    "magnitude_origin": "project_default_missing_input",
                    "q_definition": "normalised_flux",
                    "input_file": file_identity(path),
                    "input_columns": {
                        "time": "time_day",
                        "signal": "normalised_flux",
                    },
                    "input_time": {
                        "native_start_days": float(native_time_days[0]),
                        "native_cadence_seconds": RAW_EXPOSURE_SECONDS,
                        "absolute_origin_ignored": True,
                        "first_sample_maps_to_raw_frame_zero": True,
                        "used_for": "cadence_and_order_validation_only",
                    },
                    "native_sample_interpretation": "10_second_exposure_mean",
                },
            )
        )
    _reject_source_id_collisions(curves)
    return tuple(curves)


@dataclass(frozen=True)
class _WdlcComponents:
    frequency_hz: NDArray[np.float64]
    amplitude_fraction: NDArray[np.float64]
    phase_rad: NDArray[np.float64]
    identity: Mapping[str, Any]


def _load_wdlc_components(path: Path, *, expected_star_type: str) -> _WdlcComponents:
    if not path.is_file():
        raise FileNotFoundError(f"wdlc component table does not exist: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "Star type",
            "Injected f (uHz)",
            "Amplitude (ppt)",
            "Phase (rad)",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"wdlc component table has unsupported columns: {path}")
        rows = list(reader)
    if len(rows) != 300:
        raise ValueError(f"wdlc component table must contain exactly 300 rows: {path}")
    frequency: list[float] = []
    amplitude: list[float] = []
    phase: list[float] = []
    for row_index, row in enumerate(rows):
        if str(row["Star type"]).strip().casefold() != expected_star_type.casefold():
            raise ValueError(
                f"wdlc component row {row_index} has the wrong Star type"
            )
        frequency_value = _finite(
            row["Injected f (uHz)"],
            name=f"wdlc component row {row_index} frequency",
        )
        amplitude_value = _finite(
            row["Amplitude (ppt)"],
            name=f"wdlc component row {row_index} amplitude",
        )
        phase_value = _finite(
            row["Phase (rad)"],
            name=f"wdlc component row {row_index} phase",
        )
        if frequency_value <= 0.0 or amplitude_value < 0.0:
            raise ValueError("wdlc frequencies must be positive and amplitudes non-negative")
        frequency.append(frequency_value * 1.0e-6)
        amplitude.append(amplitude_value * 1.0e-3)
        phase.append(phase_value)
    arrays = [
        np.asarray(values, dtype=np.float64)
        for values in (frequency, amplitude, phase)
    ]
    for array in arrays:
        array.setflags(write=False)
    return _WdlcComponents(
        frequency_hz=arrays[0],
        amplitude_fraction=arrays[1],
        phase_rad=arrays[2],
        identity=file_identity(path),
    )


def _sum_sinusoids(
    time_seconds: NDArray[np.float64],
    *,
    components: _WdlcComponents,
    amplitude_scale: NDArray[np.float64] | None = None,
    block_size: int = 16_384,
) -> NDArray[np.float64]:
    times = np.asarray(time_seconds, dtype=np.float64)
    if times.ndim != 1 or not np.all(np.isfinite(times)):
        raise ValueError("time_seconds must be a finite one-dimensional array")
    amplitudes = (
        components.amplitude_fraction
        if amplitude_scale is None
        else components.amplitude_fraction * np.asarray(amplitude_scale)
    )
    active = amplitudes != 0.0
    frequency = components.frequency_hz[active]
    phase = components.phase_rad[active]
    amplitudes = amplitudes[active]
    result = np.empty(times.size, dtype=np.float64)
    if amplitudes.size == 0:
        result.fill(0.0)
        return result
    angular_frequency = 2.0 * np.pi * frequency
    for start in range(0, times.size, block_size):
        stop = min(start + block_size, times.size)
        angles = times[start:stop, None] * angular_frequency[None, :]
        angles += phase[None, :]
        result[start:stop] = np.sin(angles) @ amplitudes
    return result


def _load_wdlc_out(
    path: Path,
    *,
    raw_frame_count: int,
    label: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    values = _load_text_columns(
        path,
        raw_frame_count=raw_frame_count,
        usecols=(0, 1, 2),
        skiprows=3,
    )
    time_days = values[:, 0]
    expected_days = np.arange(raw_frame_count, dtype=np.float64) * (
        RAW_EXPOSURE_SECONDS / 86_400.0
    )
    # The carrier prints BJD to five decimals, so its timestamp quantization is
    # much coarser than the underlying exact 10-s row index.
    if not np.allclose(time_days, expected_days, rtol=0.0, atol=5.1e-6):
        raise ValueError(f"{label} time rows are not ordered on the 10-second grid")
    status_counts = _integer_value_counts(values[:, 2], label=f"{label} Status")
    if status_counts != {"1": raw_frame_count}:
        raise ValueError(f"{label} requires Status=1 for every selected row")
    return values[:, 1], time_days


def _gate_metrics(actual: NDArray[np.float64], expected: NDArray[np.float64]) -> dict[str, Any]:
    difference = np.asarray(actual, dtype=np.float64) - np.asarray(
        expected,
        dtype=np.float64,
    )
    return {
        "sample_count": int(difference.size),
        "max_abs": float(np.max(np.abs(difference))),
        "rms": float(np.sqrt(np.mean(np.square(difference), dtype=np.float64))),
    }


def _wdlc_mode_gate_metrics(
    actual: NDArray[np.float64],
    expected: NDArray[np.float64],
) -> dict[str, Any]:
    metrics = _gate_metrics(actual, expected)
    metrics.update(
        {
            "max_abs_tolerance": WDLC_MODE_GATE_MAX_ABS,
            "rms_tolerance": WDLC_MODE_GATE_RMS,
        }
    )
    metrics["passed"] = bool(
        metrics["max_abs"] <= WDLC_MODE_GATE_MAX_ABS
        and metrics["rms"] <= WDLC_MODE_GATE_RMS
    )
    return metrics


def _load_one_wdlc(
    root: Path,
    *,
    short_name: str,
    external_id: str,
    reference_electron_rate: float,
    detector_xpix: float,
    detector_ypix: float,
    raw_frame_count: int,
) -> ScienceInputCurve:
    components_path = root / "input_models" / f"{short_name}_components.csv"
    fractional_path = root / "lightcurve" / f"{short_name}_light_curve.out"
    electron_path = (
        root
        / "lightcurve"
        / f"{short_name}_light_curve_electron_rate.out"
    )
    components = _load_wdlc_components(
        components_path,
        expected_star_type=external_id,
    )
    normalization_frame_count = _positive_frame_count(
        WDLC_NORMALIZATION_RAW_FRAMES,
        name="WDLC_NORMALIZATION_RAW_FRAMES",
    )
    normalization_time = (
        np.arange(normalization_frame_count, dtype=np.float64)
        * RAW_EXPOSURE_SECONDS
    )
    instantaneous_full = _sum_sinusoids(
        normalization_time,
        components=components,
    )
    normalization_median = float(np.median(instantaneous_full))

    selected_time = (
        np.arange(raw_frame_count, dtype=np.float64) * RAW_EXPOSURE_SECONDS
    )
    reconstructed_instantaneous = (
        _sum_sinusoids(selected_time, components=components)
        - normalization_median
    )
    fractional_out, fractional_time = _load_wdlc_out(
        fractional_path,
        raw_frame_count=raw_frame_count,
        label=f"wdlc {external_id} fractional .out",
    )
    mode_gate = _wdlc_mode_gate_metrics(
        reconstructed_instantaneous,
        fractional_out,
    )
    if not mode_gate["passed"]:
        raise ValueError(
            f"wdlc {external_id} mode reconstruction gate failed: "
            f"max_abs={mode_gate['max_abs']:.6g}, rms={mode_gate['rms']:.6g}"
        )

    electron_rate, electron_time = _load_wdlc_out(
        electron_path,
        raw_frame_count=raw_frame_count,
        label=f"wdlc {external_id} electron-rate .out",
    )
    if not np.array_equal(fractional_time, electron_time):
        raise ValueError(f"wdlc {external_id} .out time carriers disagree")
    electron_fractional = electron_rate / reference_electron_rate - 1.0
    electron_gate = _gate_metrics(electron_fractional, fractional_out)
    electron_gate.update(
        {
            "reference_electron_rate_per_second": reference_electron_rate,
            "max_abs_tolerance": WDLC_ELECTRON_GATE_MAX_ABS,
            "rms_tolerance": WDLC_ELECTRON_GATE_RMS,
        }
    )
    electron_gate["passed"] = bool(
        electron_gate["max_abs"] <= WDLC_ELECTRON_GATE_MAX_ABS
        and electron_gate["rms"] <= WDLC_ELECTRON_GATE_RMS
    )
    if not electron_gate["passed"]:
        raise ValueError(
            f"wdlc {external_id} electron-rate consistency gate failed: "
            f"max_abs={electron_gate['max_abs']:.6g}, "
            f"rms={electron_gate['rms']:.6g}"
        )

    exposure_midpoint = selected_time + RAW_EXPOSURE_SECONDS / 2.0
    sinc_attenuation = np.sinc(
        components.frequency_hz * RAW_EXPOSURE_SECONDS
    )
    exposure_fractional = (
        _sum_sinusoids(
            exposure_midpoint,
            components=components,
            amplitude_scale=sinc_attenuation,
        )
        - normalization_median
    )
    factors = 1.0 + exposure_fractional
    return _common_curve(
        track="wdlc",
        namespace="wdlc",
        external_source_id=external_id,
        source_class=(
            "white_dwarf" if external_id == "WD" else "hot_subdwarf_B"
        ),
        detector_xpix=detector_xpix,
        detector_ypix=detector_ypix,
        factors=factors,
        metadata={
            "magnitude_origin": "project_default_missing_input",
            "q_definition": "1_plus_fractional_exposure_average",
            "mode_component_file": components.identity,
            "fractional_out_file": file_identity(fractional_path),
            "electron_rate_out_file": file_identity(electron_path),
            "mode_formula": "sum_A_fraction_sin_2pi_frequency_time_plus_phase",
            "component_count": int(components.frequency_hz.size),
            "instantaneous_normalization": {
                "operation": "subtract_median",
                "source": "reconstructed_full_365d_10s_sequence",
                "frame_count": normalization_frame_count,
                "median_fractional_flux": normalization_median,
            },
            "validation": {
                "mode_reconstruction_vs_fractional_out": mode_gate,
                "electron_rate_vs_fractional_out": electron_gate,
            },
            "electron_rate_role": "consistency_only_not_ET_absolute_flux",
            "input_time": {
                "absolute_origin_ignored": True,
                "row_zero_maps_to_raw_frame_zero": True,
                "printed_BJD_used_for": "row_order_validation_only",
            },
            "exposure_sampling": "analytic_sinc_interval_mean",
            "raw_exposure_seconds": RAW_EXPOSURE_SECONDS,
        },
    )


def load_wdlc_inputs(
    input_root: Path | str,
    *,
    raw_frame_count: int = RAW_FRAMES_90D,
) -> tuple[ScienceInputCurve, ...]:
    """Load WD and sdB factors reconstructed from their 300-mode tables."""

    n_frames = _positive_frame_count(raw_frame_count)
    if n_frames > WDLC_NORMALIZATION_RAW_FRAMES:
        raise ValueError("wdlc requested duration exceeds the submitted 365-day sequence")
    source_root = (
        _track_root(input_root, "wdlc") / "lightcurve_test_for_ET2.0"
    )
    curves = tuple(
        _load_one_wdlc(
            source_root,
            short_name=short_name,
            external_id=external_id,
            reference_electron_rate=reference_rate,
            detector_xpix=detector_x,
            detector_ypix=detector_y,
            raw_frame_count=n_frames,
        )
        for short_name, external_id, reference_rate, detector_x, detector_y in WDLC_SOURCES
    )
    _reject_source_id_collisions(curves)
    return curves


def load_science_track_inputs(
    track: str,
    input_root: Path | str,
    duration_days: float = 90.0,
    raw_exposure_seconds: float = RAW_EXPOSURE_SECONDS,
) -> tuple[ScienceInputCurve, ...]:
    """Dispatch one frozen science track without team branches downstream."""

    normalized_track = _nonempty_text(track, name="track").lower()
    duration = _finite(duration_days, name="duration_days")
    exposure = _finite(raw_exposure_seconds, name="raw_exposure_seconds")
    if duration <= 0.0:
        raise ValueError("duration_days must be positive")
    if not math.isclose(
        exposure,
        RAW_EXPOSURE_SECONDS,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("formal science adapters require 10-second raw exposures")
    frame_count_float = duration * 86_400.0 / exposure
    frame_count = round(frame_count_float)
    if frame_count <= 0 or not math.isclose(
        frame_count_float,
        float(frame_count),
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ):
        raise ValueError("duration_days must contain an integral number of raw exposures")
    loaders = {
        "aster": load_aster_precision_inputs,
        "varlc": load_varlc_inputs,
        "wdlc": load_wdlc_inputs,
    }
    try:
        loader = loaders[normalized_track]
    except KeyError as error:
        raise ValueError("track must be one of 'aster', 'varlc', or 'wdlc'") from error
    return loader(input_root, raw_frame_count=frame_count)


def write_science_factor_snapshot(
    path: Path | str,
    *,
    curve: ScienceInputCurve,
) -> dict[str, Any]:
    """Write one immutable common factor snapshot and return its file identity."""

    if not isinstance(curve, ScienceInputCurve):
        raise TypeError("curve must be a ScienceInputCurve")
    metadata = {
        "schema_id": SCIENCE_FACTOR_SNAPSHOT_SCHEMA_ID,
        "schema_version": 1,
        "source_identity": curve.source_identity,
        "track": curve.track,
        "n_raw_frames": int(curve.factors.size),
        "raw_exposure_seconds": RAW_EXPOSURE_SECONDS,
        "time_alignment": "simulation_raw_frame_index",
        "input_curve": curve.to_metadata(),
    }
    metadata_json = json.dumps(
        metadata,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as handle:
        np.savez_compressed(
            handle,
            source_id_int64=np.asarray(curve.source_id_int64, dtype=np.int64),
            namespace=np.asarray(curve.namespace),
            external_source_id=np.asarray(curve.external_source_id),
            factors=np.asarray(curve.factors, dtype=np.float64),
            metadata_json=np.asarray(metadata_json),
        )
    return file_identity(target)


def read_science_factor_snapshot(path: Path | str) -> ScienceFactorSnapshot:
    """Read and semantically validate one common factor snapshot."""

    source = Path(path).expanduser().resolve()
    with np.load(source, allow_pickle=False) as payload:
        required = {
            "source_id_int64",
            "namespace",
            "external_source_id",
            "factors",
            "metadata_json",
        }
        missing = required - set(payload.files)
        if missing:
            raise ValueError(
                f"science factor snapshot is missing arrays: {sorted(missing)}"
            )
        source_id = int(payload["source_id_int64"].reshape(()).item())
        namespace = str(payload["namespace"].reshape(()).item())
        external = str(payload["external_source_id"].reshape(()).item())
        factors = np.asarray(payload["factors"], dtype=np.float64)
        metadata_json = str(payload["metadata_json"].reshape(()).item())
    try:
        metadata = json.loads(metadata_json)
    except json.JSONDecodeError as error:
        raise ValueError("science factor snapshot metadata_json is invalid") from error
    if not isinstance(metadata, dict):
        raise ValueError("science factor snapshot metadata must be an object")
    if metadata.get("schema_id") != SCIENCE_FACTOR_SNAPSHOT_SCHEMA_ID:
        raise ValueError("unsupported science factor snapshot schema")
    identity = metadata.get("source_identity")
    if not isinstance(identity, Mapping):
        raise ValueError("science factor snapshot has no source identity")
    if (
        identity.get("namespace") != namespace
        or identity.get("external_source_id") != external
        or int(identity.get("source_id_int64", -1)) != source_id
    ):
        raise ValueError("science factor snapshot identity arrays disagree with metadata")
    if int(metadata.get("n_raw_frames", -1)) != int(factors.size):
        raise ValueError("science factor snapshot frame count disagrees with factors")
    if metadata.get("time_alignment") != "simulation_raw_frame_index":
        raise ValueError("science factor snapshot time alignment is unsupported")
    return ScienceFactorSnapshot(
        source_id_int64=source_id,
        namespace=namespace,
        external_source_id=external,
        factors=factors,
        metadata=metadata,
        metadata_json=metadata_json,
    )


__all__ = [
    "ASTER_PRECISION_SOURCES",
    "EXPLICIT_PSF_ID",
    "LOCATION_MODE",
    "NAMESPACED_SOURCE_ID_SCHEMA_ID",
    "PSF_NODE_ANGLE_DEG",
    "RAW_EXPOSURE_SECONDS",
    "RAW_FRAMES_90D",
    "SCIENCE_FACTOR_SNAPSHOT_SCHEMA_ID",
    "SCIENCE_INPUT_CURVE_SCHEMA_ID",
    "ScienceFactorSnapshot",
    "ScienceInputCurve",
    "VARLC_SOURCES",
    "WDLC_SOURCES",
    "load_aster_precision_inputs",
    "load_science_track_inputs",
    "load_varlc_inputs",
    "load_wdlc_inputs",
    "read_science_factor_snapshot",
    "stable_namespaced_source_id",
    "write_science_factor_snapshot",
]
