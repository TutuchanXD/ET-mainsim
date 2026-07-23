"""Formal G=6 Aster saturation validation with calibrated stamp delivery.

This is deliberately a saturation-response validation, not a precision
photometry production.  The source has no sky coordinate, so an explicit
PSF field ID is used with the stamp workflow's non-physical reference-field
geometry declaration.  The only observation product remains ``final_dn``;
the delivery companions preserve the full-well/ADC masks needed to prove the
expected saturation response.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Literal

import numpy as np

from .science_stamp_production import build_science_independent_production_spec
from .independent_stamp_production import (
    IndependentStampShardRequest,
    raw_stamp_delivery_frame_from_photsim7,
    run_independent_stamp_time_shard,
)
from .provenance import collect_provenance
from .stamp_inputs import file_identity
from .time_shards import (
    ContinuousTimeShardPlan,
    coadd_sizes_for_cadences,
    plan_continuous_time_shards,
)


ASTER_G6_SATURATION_SCHEMA_ID = "et_mainsim.aster_g6_saturation_validation.v1"
ASTER_G6_SATURATION_SCHEMA_VERSION = 1
DEFAULT_ASTER_G6_SOURCE_ID = 9000000000000000622
DEFAULT_ASTER_G6_MAG = 6.0
DEFAULT_ASTER_G6_PSF_ID = 6
DEFAULT_ASTER_G6_PSF_NODE_ANGLE_DEG = 12.0
DEFAULT_ASTER_G6_RAW_EXPOSURE_SECONDS = 10.0
DEFAULT_ASTER_G6_N_RAW_FRAMES = 360
DEFAULT_ASTER_G6_CADENCE_SECONDS = (30.0, 60.0, 120.0, 300.0)
DEFAULT_ASTER_G6_STAMP_SHAPE = (100, 300)

AsterSaturationCase = Literal["static", "injected"]


def _positive_float(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite and positive")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be finite and positive") from error
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _source_id(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("source_id must be a non-negative signed int64 integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("source_id must be a non-negative signed int64 integer") from error
    if result < 0 or result > int(np.iinfo(np.int64).max):
        raise ValueError("source_id must be a non-negative signed int64 integer")
    return result


def _canonical_json_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        dict(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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
                dict(payload),
                stream,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _resource_record(path: Path, *, run_root: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "relative_path": path.relative_to(run_root).as_posix(),
        "file_identity": file_identity(path),
    }


def _same_file_content_identity(
    actual: Mapping[str, Any], expected: Mapping[str, Any]
) -> bool:
    try:
        return (
            int(actual["size_bytes"]) == int(expected["size_bytes"])
            and str(actual["sha256"]) == str(expected["sha256"])
        )
    except (KeyError, TypeError, ValueError):
        return False


def _resolve_manifest_resource(
    run_root: Path,
    record: Mapping[str, Any],
    *,
    label: str,
) -> Path:
    relative_text = record.get("relative_path")
    if not isinstance(relative_text, str) or not relative_text.strip():
        raise ValueError(f"{label} requires a relative_path")
    relative = Path(relative_text)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} relative_path escapes the prepared run root")
    root = run_root.resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} relative_path escapes the prepared run root") from error
    if not candidate.is_file():
        raise FileNotFoundError(f"{label} does not exist: {candidate}")
    expected_identity = record.get("file_identity")
    if not isinstance(expected_identity, Mapping) or not _same_file_content_identity(
        file_identity(candidate), expected_identity
    ):
        raise ValueError(f"{label} identity changed after preparation")
    return candidate


def _normalise_case(value: str) -> AsterSaturationCase:
    normalised = str(value).strip().lower()
    if normalised not in {"static", "injected"}:
        raise ValueError("case must be either 'static' or 'injected'")
    return normalised  # type: ignore[return-value]


def _parse_native_metadata(path: Path) -> tuple[float, float]:
    text = path.read_text(encoding="utf-8", errors="replace")
    magnitude = re.search(
        r"\bmagnitude\s*=\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))",
        text,
        flags=re.IGNORECASE,
    )
    sampling = re.search(
        r"\bsampling\s*=\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))",
        text,
        flags=re.IGNORECASE,
    )
    if magnitude is None:
        raise ValueError("Aster source log lacks magnitude metadata")
    if sampling is None:
        raise ValueError("Aster source log lacks sampling metadata")
    return float(magnitude.group(1)), float(sampling.group(1))


@dataclass(frozen=True)
class AsterG6SaturationValidationConfig:
    """Frozen request for one G=6, PSF=6 saturation response validation."""

    source_dat: Path | str
    source_log: Path | str
    variability_ecsv: Path | str
    output_root: Path | str
    run_id: str
    data_root: Path | str
    source_id: int = DEFAULT_ASTER_G6_SOURCE_ID
    gaia_g_mag: float = DEFAULT_ASTER_G6_MAG
    psf_id: int = DEFAULT_ASTER_G6_PSF_ID
    psf_node_angle_deg: float = DEFAULT_ASTER_G6_PSF_NODE_ANGLE_DEG
    n_raw_frames: int = DEFAULT_ASTER_G6_N_RAW_FRAMES
    raw_exposure_seconds: float = DEFAULT_ASTER_G6_RAW_EXPOSURE_SECONDS
    cadence_seconds: tuple[float, ...] = DEFAULT_ASTER_G6_CADENCE_SECONDS
    stamp_shape: tuple[int, int] = DEFAULT_ASTER_G6_STAMP_SHAPE
    max_raw_frames_per_shard: int = DEFAULT_ASTER_G6_N_RAW_FRAMES
    device: str = "cuda"
    run_seed: int = 20260714

    def __post_init__(self) -> None:
        source_dat = Path(self.source_dat).expanduser().resolve()
        source_log = Path(self.source_log).expanduser().resolve()
        variability = Path(self.variability_ecsv).expanduser().resolve()
        output_root = Path(self.output_root).expanduser().resolve()
        data_root = Path(self.data_root).expanduser().resolve()
        run_id = str(self.run_id).strip()
        if not run_id:
            raise ValueError("run_id must be non-empty")
        source_id = _source_id(self.source_id)
        gaia_g_mag = _positive_float(self.gaia_g_mag, name="gaia_g_mag")
        if not math.isclose(gaia_g_mag, DEFAULT_ASTER_G6_MAG, abs_tol=0.0):
            raise ValueError("Aster saturation validation requires Gaia G magnitude 6")
        psf_id = _positive_int(self.psf_id, name="psf_id")
        if psf_id != DEFAULT_ASTER_G6_PSF_ID:
            raise ValueError("Aster saturation validation requires explicit PSF ID 6")
        node_angle = _positive_float(
            self.psf_node_angle_deg, name="psf_node_angle_deg"
        )
        if not math.isclose(node_angle, DEFAULT_ASTER_G6_PSF_NODE_ANGLE_DEG, abs_tol=0.0):
            raise ValueError("Aster saturation validation requires the 12 degree PSF node")
        n_raw_frames = _positive_int(self.n_raw_frames, name="n_raw_frames")
        raw_exposure = _positive_float(
            self.raw_exposure_seconds, name="raw_exposure_seconds"
        )
        if not math.isclose(raw_exposure, DEFAULT_ASTER_G6_RAW_EXPOSURE_SECONDS, abs_tol=0.0):
            raise ValueError("Aster saturation validation requires 10 s raw exposure")
        cadence_seconds = tuple(
            _positive_float(value, name="cadence_seconds")
            for value in self.cadence_seconds
        )
        if not cadence_seconds or len(set(cadence_seconds)) != len(cadence_seconds):
            raise ValueError("cadence_seconds must be a non-empty unique sequence")
        try:
            stamp_shape = tuple(int(value) for value in self.stamp_shape)
        except (TypeError, ValueError) as error:
            raise ValueError("stamp_shape must contain two positive integers") from error
        if len(stamp_shape) != 2 or any(value <= 0 for value in stamp_shape):
            raise ValueError("stamp_shape must contain two positive integers")
        if stamp_shape != DEFAULT_ASTER_G6_STAMP_SHAPE:
            raise ValueError("Aster saturation validation requires a 100x300 stamp")
        max_raw_frames = _positive_int(
            self.max_raw_frames_per_shard, name="max_raw_frames_per_shard"
        )
        if max_raw_frames != n_raw_frames:
            raise ValueError(
                "Aster saturation validation requires one non-resumable shard containing all raw frames"
            )
        device = str(self.device).strip().lower()
        if device not in {"cpu", "cuda"}:
            raise ValueError("device must be 'cpu' or 'cuda'")
        if isinstance(self.run_seed, (bool, np.bool_)):
            raise ValueError("run_seed must be an integer")
        for label, path in (
            ("source_dat", source_dat),
            ("source_log", source_log),
            ("variability_ecsv", variability),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"{label} does not exist: {path}")
        if not data_root.is_dir():
            raise FileNotFoundError(f"Photsim7 data root does not exist: {data_root}")
        coadd_sizes_for_cadences(
            raw_exposure_seconds=raw_exposure,
            cadence_seconds=cadence_seconds,
        )

        object.__setattr__(self, "source_dat", source_dat)
        object.__setattr__(self, "source_log", source_log)
        object.__setattr__(self, "variability_ecsv", variability)
        object.__setattr__(self, "output_root", output_root)
        object.__setattr__(self, "data_root", data_root)
        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(self, "gaia_g_mag", gaia_g_mag)
        object.__setattr__(self, "psf_id", psf_id)
        object.__setattr__(self, "psf_node_angle_deg", node_angle)
        object.__setattr__(self, "n_raw_frames", n_raw_frames)
        object.__setattr__(self, "raw_exposure_seconds", raw_exposure)
        object.__setattr__(self, "cadence_seconds", cadence_seconds)
        object.__setattr__(self, "stamp_shape", stamp_shape)
        object.__setattr__(self, "max_raw_frames_per_shard", max_raw_frames)
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "run_seed", int(self.run_seed))

    @property
    def coadd_sizes(self) -> tuple[int, ...]:
        return coadd_sizes_for_cadences(
            raw_exposure_seconds=self.raw_exposure_seconds,
            cadence_seconds=self.cadence_seconds,
        )

    @property
    def run_root(self) -> Path:
        return Path(self.output_root) / self.run_id


@dataclass(frozen=True)
class AsterG6SaturationPreparation:
    """Immutable inputs and time plan for the Aster saturation worker."""

    run_root: Path
    manifest_path: Path
    time_plan_path: Path
    time_plan: ContinuousTimeShardPlan


def _freeze_variability(
    source: Path,
    destination: Path,
    *,
    n_raw_frames: int,
    raw_exposure_seconds: float,
) -> None:
    from astropy.table import Table

    table = Table.read(source)
    required = {"curve_id", "frame_index", "relative_flux"}
    missing = sorted(required - set(table.colnames))
    if missing:
        raise ValueError(
            "Aster variability ECSV lacks required columns: " + ", ".join(missing)
        )
    frame_index = np.asarray(table["frame_index"], dtype=np.int64)
    relative_flux = np.asarray(table["relative_flux"], dtype=np.float64)
    if frame_index.ndim != 1 or relative_flux.shape != frame_index.shape:
        raise ValueError("Aster variability ECSV has invalid frame columns")
    if not np.all(np.isfinite(relative_flux)) or np.any(relative_flux <= 0.0):
        raise ValueError("Aster variability ECSV relative_flux must be finite and positive")
    expected = np.arange(n_raw_frames, dtype=np.int64)
    selected = frame_index < n_raw_frames
    selected_indices = frame_index[selected]
    if selected_indices.size != n_raw_frames or not np.array_equal(
        np.sort(selected_indices), expected
    ):
        raise ValueError(
            "Aster variability ECSV must contain exactly frame_index 0.."
            f"{n_raw_frames - 1} for the frozen validation window"
        )
    ordered_flux = np.empty(n_raw_frames, dtype=np.float64)
    for index, value in zip(frame_index[selected], relative_flux[selected], strict=True):
        if not 0 <= int(index) < n_raw_frames:
            raise RuntimeError("selected Aster frame index is invalid")
        ordered_flux[int(index)] = float(value)
    if not np.all(np.isfinite(ordered_flux)) or np.any(ordered_flux <= 0.0):
        raise ValueError("Aster frozen variability factors are invalid")
    frozen = Table()
    frozen["curve_id"] = ["aster_psls_0000000622_g6_1h"] * n_raw_frames
    frozen["frame_index"] = expected
    frozen["relative_flux"] = ordered_flux
    frozen["simulation_time_s"] = expected.astype(np.float64) * raw_exposure_seconds
    frozen.meta = {
        "schema_id": "et_mainsim.stamp_variability_table",
        "schema_version": 1,
        "source_file_id": "0000000622",
        "source_curve_semantics": "q = 1 + ppm * 1e-6",
        "flux_semantics": "dimensionless_q_per_raw_frame",
        "time_alignment": "simulation_raw_frame_index",
        "resampling": "identity_native_10s_no_interpolation",
        "raw_exposure_s": raw_exposure_seconds,
        "n_raw_frames": n_raw_frames,
        "original_source_identity": file_identity(source),
    }
    frozen.write(destination, format="ascii.ecsv", overwrite=False)


def prepare_aster_g6_saturation_validation(
    config: AsterG6SaturationValidationConfig,
) -> AsterG6SaturationPreparation:
    """Freeze G=6 Aster inputs for one-hour saturation response validation."""

    if not isinstance(config, AsterG6SaturationValidationConfig):
        raise TypeError("config must be an AsterG6SaturationValidationConfig")
    native_magnitude, native_sampling = _parse_native_metadata(Path(config.source_log))
    if not math.isclose(native_magnitude, DEFAULT_ASTER_G6_MAG, abs_tol=0.0):
        raise ValueError("Aster source log must declare magnitude=6")
    if not math.isclose(
        native_sampling, DEFAULT_ASTER_G6_RAW_EXPOSURE_SECONDS, abs_tol=0.0
    ):
        raise ValueError("Aster source log must declare sampling=10")
    run_root = config.run_root
    if run_root.exists():
        raise FileExistsError(
            f"saturation validation run root already exists: {run_root}; formal runs are not resumed"
        )
    time_plan = plan_continuous_time_shards(
        raw_start_index=0,
        raw_stop_index=config.n_raw_frames,
        coadd_sizes=config.coadd_sizes,
        raw_exposure_seconds=config.raw_exposure_seconds,
        max_raw_frames_per_shard=config.max_raw_frames_per_shard,
    )
    if len(time_plan.shards) != 1:
        raise RuntimeError("Aster saturation validation must have one time shard")
    base_spec = build_science_independent_production_spec(
        n_raw_frames=config.n_raw_frames,
        raw_exposure_seconds=config.raw_exposure_seconds,
        device=config.device,
        run_seed=config.run_seed,
    )
    from astropy.table import Table

    inputs_root = run_root / "inputs"
    run_root.mkdir(parents=True, exist_ok=False)
    inputs_root.mkdir(parents=True, exist_ok=False)
    static_target_path = inputs_root / "aster_g6_static_target.ecsv"
    injected_target_path = inputs_root / "aster_g6_injected_target.ecsv"
    variability_path = inputs_root / "aster_psls_0000000622_g6_1h_10s_variability.ecsv"
    static_target = Table(
        rows=[(np.int64(config.source_id), float(config.gaia_g_mag), int(config.psf_id))],
        names=("source_id", "gaia_g_mag", "psf_id"),
    )
    static_target.meta = {
        "schema_id": ASTER_G6_SATURATION_SCHEMA_ID,
        "magnitude_system": "Gaia_G_Vega",
        "coordinate_mode": "explicit_psf_no_sky_coordinate",
        "scene_policy": "one_independent_target_no_neighbors",
        "purpose": "saturation_response_validation",
    }
    static_target.write(static_target_path, format="ascii.ecsv", overwrite=False)
    injected_target = static_target.copy(copy_data=True)
    injected_target["curve_id"] = ["aster_psls_0000000622_g6_1h"]
    injected_target.write(injected_target_path, format="ascii.ecsv", overwrite=False)
    _freeze_variability(
        Path(config.variability_ecsv),
        variability_path,
        n_raw_frames=config.n_raw_frames,
        raw_exposure_seconds=config.raw_exposure_seconds,
    )
    time_plan_path = time_plan.write_manifest(inputs_root / "time_shards.json")
    spec_json = base_spec.to_json_dict()
    manifest = {
        "schema_id": ASTER_G6_SATURATION_SCHEMA_ID,
        "schema_version": ASTER_G6_SATURATION_SCHEMA_VERSION,
        "run_id": config.run_id,
        "run_root": str(run_root),
        "observation_product": "final_dn",
        "background_realization_delivered": False,
        "scientific_scope": {
            "purpose": "saturation_response_validation",
            "not_precision_photometry": True,
            "not_cdpp_eligible": True,
            "scene_policy": "independent_target_only_no_neighbors",
            "coordinate_policy": "explicit_psf_no_sky_coordinate_nonphysical_reference_field",
        },
        "target": {
            "source_id": int(config.source_id),
            "source_file_id": "0000000622",
            "gaia_g_mag": float(config.gaia_g_mag),
            "magnitude_system": "Gaia_G_Vega",
            "psf_id": int(config.psf_id),
            "psf_node_angle_deg": float(config.psf_node_angle_deg),
            "coordinate_mode": "explicit_psf_no_sky_coordinate",
            "detector_placement": "physical_detector_center_reference_field",
        },
        "inputs": {
            "source_dat": file_identity(Path(config.source_dat)),
            "source_log": file_identity(Path(config.source_log)),
            "native_metadata": {
                "magnitude": native_magnitude,
                "sampling_seconds": native_sampling,
            },
            "source_variability_ecsv": file_identity(Path(config.variability_ecsv)),
            "static_target": _resource_record(static_target_path, run_root=run_root),
            "injected_target": _resource_record(injected_target_path, run_root=run_root),
            "frozen_variability": _resource_record(variability_path, run_root=run_root),
        },
        "runtime_defaults": {
            "data_root": str(config.data_root),
            "device": config.device,
        },
        "delivery": {
            "stamp_shape": list(config.stamp_shape),
            "raw_exposure_seconds": config.raw_exposure_seconds,
            "cadence_seconds": list(config.cadence_seconds),
            "coadd_sizes": list(config.coadd_sizes),
            "time_plan_path": str(time_plan_path),
            "time_plan_relative_path": time_plan_path.relative_to(run_root).as_posix(),
            "time_plan_identity": file_identity(time_plan_path),
            "tail_policy": "reject_incomplete_global_coadd_tail",
        },
        "simulation_spec_base": spec_json,
        "simulation_spec_base_sha256": _canonical_json_sha256(spec_json),
        "software_provenance_at_prepare": collect_provenance(
            Path(__file__).resolve().parents[2]
        ),
    }
    manifest_path = _atomic_json(run_root / "production_manifest.json", manifest)
    return AsterG6SaturationPreparation(
        run_root=run_root,
        manifest_path=manifest_path,
        time_plan_path=time_plan_path,
        time_plan=time_plan,
    )


def _load_manifest(path: Path | str) -> tuple[Path, dict[str, Any]]:
    manifest_path = Path(path).expanduser().resolve()
    with manifest_path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if payload.get("schema_id") != ASTER_G6_SATURATION_SCHEMA_ID:
        raise ValueError("unsupported Aster saturation validation manifest")
    if int(payload.get("schema_version", 0)) != ASTER_G6_SATURATION_SCHEMA_VERSION:
        raise ValueError("unsupported Aster saturation validation manifest version")
    return manifest_path, payload


def _load_time_plan(run_root: Path, manifest: Mapping[str, Any]) -> ContinuousTimeShardPlan:
    delivery = manifest.get("delivery")
    if not isinstance(delivery, Mapping):
        raise ValueError("Aster saturation manifest lacks delivery")
    path = _resolve_manifest_resource(
        run_root,
        {
            "relative_path": delivery.get("time_plan_relative_path"),
            "file_identity": delivery.get("time_plan_identity"),
        },
        label="time shard plan",
    )
    with path.open(encoding="utf-8") as stream:
        return ContinuousTimeShardPlan.from_manifest_dict(json.load(stream))


def run_aster_g6_saturation_validation(
    manifest_path: Path | str,
    *,
    case: AsterSaturationCase,
    data_root: Path | str | None = None,
    device: Literal["cpu", "cuda"] | None = None,
    batch_size: int = 64,
) -> tuple[dict[str, Any], ...]:
    """Render either the static or injected one-hour Aster validation case."""

    selected_case = _normalise_case(case)
    resolved_manifest_path, manifest = _load_manifest(manifest_path)
    run_root = resolved_manifest_path.parent
    target = manifest.get("target")
    if not isinstance(target, Mapping):
        raise ValueError("Aster saturation manifest lacks target metadata")
    source_id = _source_id(target.get("source_id"))
    if float(target.get("gaia_g_mag", float("nan"))) != DEFAULT_ASTER_G6_MAG:
        raise ValueError("Aster saturation manifest magnitude is not G=6")
    if int(target.get("psf_id", -1)) != DEFAULT_ASTER_G6_PSF_ID:
        raise ValueError("Aster saturation manifest PSF ID is not 6")
    inputs = manifest.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("Aster saturation manifest lacks frozen inputs")
    target_key = "static_target" if selected_case == "static" else "injected_target"
    target_record = inputs.get(target_key)
    if not isinstance(target_record, Mapping):
        raise ValueError(f"Aster saturation manifest lacks {target_key}")
    target_table_path = _resolve_manifest_resource(
        run_root, target_record, label=f"{selected_case} target table"
    )
    variability_path: Path | None = None
    if selected_case == "injected":
        variability_record = inputs.get("frozen_variability")
        if not isinstance(variability_record, Mapping):
            raise ValueError("Aster saturation manifest lacks frozen variability")
        variability_path = _resolve_manifest_resource(
            run_root, variability_record, label="frozen variability"
        )
    runtime_defaults = manifest.get("runtime_defaults")
    if not isinstance(runtime_defaults, Mapping):
        raise ValueError("Aster saturation manifest lacks runtime defaults")
    resolved_data_root = Path(
        data_root if data_root is not None else str(runtime_defaults.get("data_root", ""))
    ).expanduser().resolve()
    if not resolved_data_root.is_dir():
        raise FileNotFoundError(f"Photsim7 data root does not exist: {resolved_data_root}")
    spec_payload = manifest.get("simulation_spec_base")
    if not isinstance(spec_payload, Mapping):
        raise ValueError("Aster saturation manifest lacks simulation_spec_base")

    from photsim7.data_registry import DataRegistry
    from photsim7.simulation_services import (
        build_simulation_context,
        build_stamp_services,
    )
    from photsim7.specs import SimulationSpec
    from photsim7.stamp_pipeline import run_single_cadence_stamp

    from .config import ExecutionConfig, RunConfig, RunPaths, StampWorkload
    from .workflows.stamp import (
        _prepare_table_inputs,
        _science_api,
        _table_catalog,
        _target_spec,
        build_run_plan,
    )

    base_spec = SimulationSpec.from_json_dict(dict(spec_payload))
    compute_device = str(device).strip().lower() if device is not None else base_spec.psf.compute_device
    if compute_device not in {"cpu", "cuda"}:
        raise ValueError("device must be 'cpu' or 'cuda'")
    delivery = manifest["delivery"]
    if not isinstance(delivery, Mapping):
        raise ValueError("Aster saturation manifest lacks delivery")
    stamp_shape = tuple(int(value) for value in delivery["stamp_shape"])
    if stamp_shape != DEFAULT_ASTER_G6_STAMP_SHAPE:
        raise ValueError("Aster saturation delivery stamp shape is invalid")
    execution = ExecutionConfig(
        backend="local-subprocess" if compute_device == "cuda" else "in-process",
        device=compute_device,
        gpu_ids=("0",) if compute_device == "cuda" else (),
        workers_per_device=1,
        resume=False,
        overwrite=False,
        progress=False,
    )
    run_config = RunConfig(
        schema_id="et_mainsim.execution_config",
        schema_version=1,
        workflow="et-stamp",
        run_id=str(manifest["run_id"]),
        paths=RunPaths(output_root=str(run_root), data_root=str(resolved_data_root)),
        execution=execution,
        workload=StampWorkload(
            input_mode="table",
            input_table=str(target_table_path),
            variability_table="" if variability_path is None else str(variability_path),
            target_source_ids=(source_id,),
            stamp_rows=stamp_shape[0],
            stamp_cols=stamp_shape[1],
            include_neighbors=False,
            save_raw=True,
            save_coadd=True,
            write_batch_size=int(batch_size),
        ),
    )
    spec = replace(base_spec, psf=replace(base_spec.psf, compute_device=compute_device))
    repo_root = Path(__file__).resolve().parents[2]
    plan = build_run_plan(
        preset_name="aster-g6-saturation-validation",
        run_config=run_config,
        spec=spec,
        repo_root=repo_root,
        cwd=repo_root,
    )
    api = _science_api()
    prepared = _prepare_table_inputs(plan, api, requested_target_ids=(source_id,))
    prepared_target = prepared.targets[source_id]
    if prepared.psf_ids[source_id] != DEFAULT_ASTER_G6_PSF_ID or not math.isclose(
        float(prepared_target.psf_node_angle_deg),
        DEFAULT_ASTER_G6_PSF_NODE_ANGLE_DEG,
        abs_tol=0.0,
    ):
        raise ValueError("Aster saturation runtime did not resolve explicit PSF 6 at 12 degrees")
    source_variability = prepared.source_variability[source_id]
    if selected_case == "static" and source_variability is not None:
        raise ValueError("static Aster saturation case unexpectedly has variability")
    if selected_case == "injected":
        if source_variability is None or source_variability.relative_flux.shape != (
            1,
            int(base_spec.observation.resolved_n_frames),
        ):
            raise ValueError("injected Aster saturation variability does not match raw frames")
    source_truth = dict(prepared.source_input_truth[source_id])
    source_truth["saturation_validation"] = {
        "purpose": "saturation_response_validation",
        "not_precision_photometry": True,
        "not_cdpp_eligible": True,
        "source_file_id": "0000000622",
        "gaia_g_vega_mag": DEFAULT_ASTER_G6_MAG,
        "explicit_psf_id": DEFAULT_ASTER_G6_PSF_ID,
        "psf_node_angle_deg": DEFAULT_ASTER_G6_PSF_NODE_ANGLE_DEG,
    }
    catalog = _table_catalog(plan, prepared_target, api, source_input_truth=source_truth)
    target_spec = _target_spec(
        plan,
        target=prepared_target,
        psf_id=prepared.psf_ids[source_id],
        source_input_truth=source_truth,
    )
    context = build_simulation_context(
        target_spec,
        data_registry=DataRegistry(data_root=resolved_data_root),
        spacecraft_id="et",
        absolute_raw_frame_start_index=0,
    )
    services = build_stamp_services(context, catalog=catalog)
    time_plan = _load_time_plan(run_root, manifest)
    if len(time_plan.shards) != 1:
        raise ValueError("Aster saturation validation must contain exactly one time shard")
    target_spec_json = target_spec.to_json_dict()
    target_spec_sha256 = _canonical_json_sha256(target_spec_json)
    runtime_provenance = collect_provenance(repo_root)
    renderer_options = {
        "enable_stellar_photon_noise": True,
        "enable_background_light": True,
        "enable_scattered_light": True,
        "enable_dark_current": True,
    }

    def render_raw(global_frame_index: int) -> Any:
        return run_single_cadence_stamp(
            target_spec,
            target_source_id=source_id,
            stamp_shape=stamp_shape,
            services=services,
            frame_index=global_frame_index,
            source_variability=source_variability,
            include_neighbors=False,
            renderer_options=renderer_options,
            rng_trace_scope={
                "workflow": "aster-g6-saturation-validation",
                "run_id": str(manifest["run_id"]),
                "science_realization_id": "aster-g6-psf6-paired-v1",
            },
        )

    request = IndependentStampShardRequest(
        output_root=run_root / "cases" / selected_case,
        target_source_id=source_id,
        stamp_shape=stamp_shape,
        shard=time_plan.shards[0],
        gain_e_per_dn=float(target_spec.readout.gain_electrons_per_adu.value),
        manifest={
            "run_id": str(manifest["run_id"]),
            "case": selected_case,
            "saturation_validation_manifest": str(resolved_manifest_path),
            "target_input_truth": source_truth,
            "simulation_spec_sha256": target_spec_sha256,
            "renderer_options": renderer_options,
        },
        provenance={
            "observation_product": "final_dn",
            "background_realization_used": False,
            "delivery_calibration": "bias_and_column_are_pre_adc_adu_codes",
            "software": runtime_provenance,
            "simulation_spec": target_spec_json,
        },
        batch_size=int(batch_size),
        overwrite=False,
    )
    report = run_independent_stamp_time_shard(
        request,
        render_raw=render_raw,
        adapt_raw=raw_stamp_delivery_frame_from_photsim7,
    )
    return (
        {
            "source_id": str(source_id),
            "case": selected_case,
            "shard_id": report.shard_id,
            "raw_frame_count": report.raw_frame_count,
            "raw_path": str(report.raw_path),
            "coadd_paths": {
                str(key): str(value) for key, value in report.coadd_paths.items()
            },
        },
    )


__all__ = [
    "ASTER_G6_SATURATION_SCHEMA_ID",
    "ASTER_G6_SATURATION_SCHEMA_VERSION",
    "AsterG6SaturationPreparation",
    "AsterG6SaturationValidationConfig",
    "prepare_aster_g6_saturation_validation",
    "run_aster_g6_saturation_validation",
]
