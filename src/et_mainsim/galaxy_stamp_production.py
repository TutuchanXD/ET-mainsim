"""Formal independent Galaxy-stamp production preparation and workers.

This workflow is intentionally limited to the Galaxy-team FITS input whose
Gaia G Vega magnitude, ICRS coordinates, and clean ``Delta F / F_ref`` curves
are all available.  It prepares immutable 10-s exposure-averaged factor
snapshots and then renders one target over globally indexed time shards.

The only detector observation written by the worker is ``final_dn``.  It uses
the Stage-9 detector chain and SD-24 expectation companions through
``run_independent_stamp_time_shard``; it does not manufacture a background
realization product or re-anchor time at a shard boundary.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Literal

import numpy as np

from .galaxy_lightcurves import (
    exposure_averaged_factors,
    load_galaxy_lightcurves,
    read_galaxy_factor_snapshot,
    write_galaxy_factor_snapshot,
)
from .independent_stamp_production import (
    IndependentStampShardRequest,
    raw_stamp_delivery_frame_from_photsim7,
    run_independent_stamp_time_shard,
)
from .provenance import collect_provenance
from .stamp_inputs import file_identity
from .time_shards import (
    ContinuousTimeShard,
    ContinuousTimeShardPlan,
    coadd_sizes_for_cadences,
    plan_continuous_time_shards,
)


GALAXY_STAMP_PRODUCTION_SCHEMA_ID = "et_mainsim.galaxy_stamp_production.v1"
GALAXY_STAMP_PRODUCTION_SCHEMA_VERSION = 2
DEFAULT_GALAXY_PRODUCTION_SOURCE_IDS = (
    2104827888243104128,
    2107035368292387840,
    2119800835728275584,
    2075356720322484736,
    2125709679939965952,
    2125773997068553600,
    2102899692744736512,
    2129135070975727104,
    2129135620732022656,
    2080632520701306880,
)
DEFAULT_RAW_EXPOSURE_SECONDS = 10.0
DEFAULT_DURATION_DAYS = 90.0
DEFAULT_CADENCE_SECONDS = (30.0, 60.0, 120.0, 300.0)
DEFAULT_MAX_RAW_FRAMES_PER_SHARD = 8_640  # one day at 10 s
DEFAULT_STAMP_SHAPE = (27, 27)
FORMAL_STAMP_CENTERING_POLICY = "nearest_integer_np_rint"
FORMAL_INTER_PRV_RMS_PERCENT = 1.0
FORMAL_INTRA_PRV_RMS_PERCENT = 1.0
FORMAL_READOUT_NOISE_E_PER_PIXEL = 5.0
FORMAL_COLUMN_NOISE_SIGMA_ADU = 0.0
FORMAL_PIXEL_PHASE_PROFILE_PATH = (
    "detector/pixel_response_profile_teff5500_feh-0.1_logg4.4_"
    "pfc_v240423.npy"
)

ProductionCase = Literal["static", "injected"]
DeliveryExecutionMode = Literal[
    "direct_shared_filesystem",
    "staged_local_scratch_v1",
]

DIRECT_SHARED_FILESYSTEM_DELIVERY_EXECUTION_MODE: DeliveryExecutionMode = (
    "direct_shared_filesystem"
)
STAGED_LOCAL_SCRATCH_DELIVERY_EXECUTION_MODE: DeliveryExecutionMode = (
    "staged_local_scratch_v1"
)


def _normalise_delivery_execution_mode(value: object) -> DeliveryExecutionMode:
    """Return one frozen writer mode; mixed-mode manifests are invalid."""

    normalised = str(value).strip()
    if normalised == DIRECT_SHARED_FILESYSTEM_DELIVERY_EXECUTION_MODE:
        return DIRECT_SHARED_FILESYSTEM_DELIVERY_EXECUTION_MODE
    if normalised == STAGED_LOCAL_SCRATCH_DELIVERY_EXECUTION_MODE:
        return STAGED_LOCAL_SCRATCH_DELIVERY_EXECUTION_MODE
    raise ValueError(
        "delivery.execution_mode must be one of "
        f"{DIRECT_SHARED_FILESYSTEM_DELIVERY_EXECUTION_MODE!r}, "
        f"{STAGED_LOCAL_SCRATCH_DELIVERY_EXECUTION_MODE!r}"
    )


def delivery_execution_mode_from_manifest(
    manifest: Mapping[str, Any],
) -> DeliveryExecutionMode:
    """Read the frozen writer mode with v2 direct-mode compatibility.

    Historical v2 manifests predate writer-mode freezing and were all rendered
    directly to their formal shared filesystem roots. They therefore retain
    that exact interpretation, while every newly prepared manifest writes an
    explicit mode.
    """

    delivery = manifest.get("delivery")
    if not isinstance(delivery, Mapping):
        raise ValueError("production manifest delivery must be an object")
    return _normalise_delivery_execution_mode(
        delivery.get(
            "execution_mode",
            DIRECT_SHARED_FILESYSTEM_DELIVERY_EXECUTION_MODE,
        )
    )


def _strict_source_id(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a non-negative signed int64 source ID")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a non-negative signed int64 source ID") from error
    if result < 0 or result > int(np.iinfo(np.int64).max):
        raise ValueError(f"{name} must be a non-negative signed int64 source ID")
    return result


def _galaxy_physical_rng_pairing_metadata(
    *,
    context: Any,
    source_id: int,
    shard: ContinuousTimeShard,
    target_spec_sha256: str,
) -> dict[str, Any]:
    """Record the actual physical RNG coordinates of one delivery shard.

    ``rng_trace_scope`` is intentionally not an input: Photsim7 treats it as
    an execution/provenance label and explicitly removes it before physical
    detector seed derivation.  The authoritative physical coordinates are
    instead the canonical ``SimulationContext.detector_rng_scope`` values.

    Source ID is retained solely as a human-readable comparison label.  It is
    *not* added to the physical RNG identity: detector random fields are
    addressed by their absolute detector coordinates, so same-detector stamps
    share a common physical realization where their detector coordinates
    overlap.
    """

    normalized_source_id = _strict_source_id(source_id, name="source_id")
    if shard.raw_frame_count <= 0:
        raise ValueError("physical RNG audit requires a non-empty time shard")
    normalized_spec_sha256 = str(target_spec_sha256).strip().lower()
    if len(normalized_spec_sha256) != 64:
        raise ValueError("target_spec_sha256 must contain 64 hexadecimal characters")
    try:
        int(normalized_spec_sha256, 16)
    except ValueError as error:
        raise ValueError(
            "target_spec_sha256 must contain 64 hexadecimal characters"
        ) from error

    first_scope = context.detector_rng_scope(
        local_frame_index=shard.raw_start_index
    )
    last_scope = context.detector_rng_scope(
        local_frame_index=shard.raw_stop_index - 1
    )
    if not isinstance(first_scope, Mapping) or not isinstance(last_scope, Mapping):
        raise TypeError("SimulationContext detector_rng_scope must return mappings")
    required_scope_keys = (
        "science_realization_id",
        "spacecraft_id",
        "absolute_raw_frame_index",
        "detector_id",
        "scope_id",
    )
    try:
        for key in required_scope_keys:
            first_scope[key]
            last_scope[key]
    except KeyError as error:
        raise ValueError(
            "SimulationContext detector_rng_scope lacks a canonical physical key"
        ) from error
    canonical_scope = {
        "science_realization_id": int(first_scope["science_realization_id"]),
        "spacecraft_id": str(first_scope["spacecraft_id"]),
        "detector_id": str(first_scope["detector_id"]),
        "scope_id": int(first_scope["scope_id"]),
    }
    if not canonical_scope["spacecraft_id"] or not canonical_scope["detector_id"]:
        raise ValueError("SimulationContext physical RNG scope has an empty identifier")
    for key in canonical_scope:
        if first_scope[key] != last_scope[key]:
            raise ValueError(
                "a Galaxy time shard cannot change canonical physical RNG scope"
            )
    absolute_start = int(first_scope["absolute_raw_frame_index"])
    absolute_stop = int(last_scope["absolute_raw_frame_index"]) + 1
    if absolute_stop - absolute_start != shard.raw_frame_count:
        raise ValueError(
            "SimulationContext absolute raw-frame interval does not match time shard"
        )
    context_absolute_start = int(context.absolute_raw_frame_start_index)
    if absolute_start != context_absolute_start + shard.raw_start_index:
        raise ValueError(
            "SimulationContext absolute raw-frame formula does not match time shard"
        )

    return {
        "schema_id": "et_mainsim.galaxy_physical_rng_pairing.v1",
        "schema_version": 1,
        "seed_tree_run_seed": int(context.seed_tree.run_seed),
        "canonical_context_scope": canonical_scope,
        "absolute_raw_frame_index": {
            "formula": "absolute_raw_frame_start_index + local_frame_index",
            "absolute_raw_frame_start_index": context_absolute_start,
            "selected_shard_absolute_frame_interval": {
                "start_index": absolute_start,
                "stop_index": absolute_stop,
            },
        },
        "selected_time_shard": shard.to_manifest_dict(),
        "target_spec_sha256": normalized_spec_sha256,
        "source_id_comparison_label": normalized_source_id,
        "source_id_in_physical_rng_identity": False,
        "case_not_in_physical_rng_identity": True,
        "rng_trace_scope_role": "execution_label_only",
    }


def _finite_positive(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite and positive")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be finite and positive") from error
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _canonical_json_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                dict(payload),
                handle,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _same_file_content_identity(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> bool:
    """Compare immutable file content without treating its host path as data.

    Formal inputs are prepared below the run root and may then be accessed via
    a different mount point on H100.  ``file_identity`` deliberately records
    its preparation path for provenance, but path equality is neither a
    content-integrity test nor portable across machines.
    """

    if not isinstance(actual, Mapping) or not isinstance(expected, Mapping):
        return False
    try:
        return (
            int(actual["size_bytes"]) == int(expected["size_bytes"])
            and str(actual["sha256"]) == str(expected["sha256"])
        )
    except (KeyError, TypeError, ValueError):
        return False


def _file_content_identity(path: Path | str) -> dict[str, Any]:
    """Return only the relocatable content portion of a file identity."""

    identity = file_identity(path)
    return {
        "sha256": identity["sha256"],
        "size_bytes": identity["size_bytes"],
    }


def _registry_identity_without_local_locator(
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a semantic registry envelope without host-local location fields."""

    if not isinstance(identity, Mapping):
        raise TypeError("focal-plane registry identity must be an object")
    projected = dict(identity)
    # Keep every scientific/governance field, including the content SHA and
    # owner attestation.  These two keys are the sole host-local locators in
    # et_coord's semantic registry identity contract.
    projected.pop("registry_data_dir", None)
    projected.pop("path", None)
    try:
        json.dumps(projected, ensure_ascii=False, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("focal-plane registry identity is not JSON-safe") from error
    return projected


def _same_semantic_registry_identity(
    prepared: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> bool:
    """Compare all frozen registry semantics while allowing a moved data root."""

    try:
        return _registry_identity_without_local_locator(
            prepared
        ) == _registry_identity_without_local_locator(runtime)
    except (TypeError, ValueError):
        return False


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _validate_formal_registry_gate(
    prepared: Mapping[str, Any],
    runtime: Mapping[str, Any],
    attestation_verification: Mapping[str, Any],
) -> None:
    """Fail closed unless H100 uses the same owner-frozen ET registry.

    A portable root locator is allowed to change, but CSV content, coordinate
    algorithm, governance state, and the owner-attestation verification are
    part of the scientific input.  This mirrors Photsim7's semantic-registry
    comparison without importing its private helper across repositories.
    """

    schema_id = "et_coord.semantic_registry_identity.v1"
    for label, identity in (("prepared", prepared), ("runtime", runtime)):
        if not isinstance(identity, Mapping):
            raise ValueError(f"{label} focal-plane registry identity must be an object")
        if identity.get("schema_id") != schema_id or identity.get("schema_version") != 1:
            raise ValueError(f"{label} focal-plane registry identity schema is unsupported")
        if not _is_sha256(identity.get("sha256")) or not _is_sha256(
            identity.get("semantic_content_sha256")
        ):
            raise ValueError(f"{label} focal-plane registry identity lacks SHA-256 fields")
        if identity.get("freeze_status") != "owner_frozen":
            raise ValueError(
                f"{label} focal-plane registry must have freeze_status='owner_frozen'"
            )
        if identity.get("owner_approval_required") is not False:
            raise ValueError(
                f"{label} focal-plane registry requires outstanding owner approval"
            )
    if not _same_semantic_registry_identity(prepared, runtime):
        raise ValueError(
            "runtime focal-plane registry identity differs from the frozen "
            "production preparation identity"
        )
    if not isinstance(prepared.get("owner_attestation"), Mapping):
        raise ValueError("prepared focal-plane registry lacks owner attestation")
    if not isinstance(attestation_verification, Mapping):
        raise ValueError("runtime focal-plane registry attestation verification is invalid")
    errors = attestation_verification.get("errors")
    if (
        attestation_verification.get("verified") is not True
        or not isinstance(errors, list)
        or errors
    ):
        raise ValueError("runtime focal-plane registry attestation verification failed")


def _resolve_manifest_resource(
    run_root: Path | str,
    record: Mapping[str, Any],
    *,
    label: str,
) -> Path:
    """Resolve a prepared resource relative to the manifest's actual root.

    Formal v2 manifests carry a relative path so a complete run root can move
    from a workstation mount to H100 unchanged.  The absolute preparation path
    remains provenance only and is never a runtime fallback.  A relative path
    may never be absolute, contain ``..``, or escape the run root after symlink
    resolution.
    """

    if not isinstance(record, Mapping):
        raise ValueError(f"{label} record must be an object")
    root = Path(run_root).expanduser().resolve()
    relative_path = record.get("relative_path")
    if not isinstance(relative_path, str) or not relative_path.strip():
        raise ValueError(f"{label} requires relative_path in formal manifest v2")
    relative = Path(relative_path)
    if relative.is_absolute():
        raise ValueError(f"{label} relative_path must be relative")
    if ".." in relative.parts:
        raise ValueError(f"{label} relative_path escapes prepared run root")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} relative_path escapes prepared run root") from error
    if not candidate.is_file():
        raise FileNotFoundError(f"{label} does not exist: {candidate}")
    return candidate


def _normalise_case(value: str) -> ProductionCase:
    normalised = str(value).strip().lower()
    if normalised not in {"static", "injected"}:
        raise ValueError("case must be either 'static' or 'injected'")
    return normalised  # type: ignore[return-value]


@dataclass(frozen=True)
class GalaxyStampProductionConfig:
    """Frozen request used to prepare one no-resume Galaxy production run."""

    input_fits: Path | str
    output_root: Path | str
    run_id: str
    data_root: Path | str
    focalplane_registry: Path | str
    source_ids: tuple[int, ...] = DEFAULT_GALAXY_PRODUCTION_SOURCE_IDS
    duration_days: float = DEFAULT_DURATION_DAYS
    raw_exposure_seconds: float = DEFAULT_RAW_EXPOSURE_SECONDS
    cadence_seconds: tuple[float, ...] = DEFAULT_CADENCE_SECONDS
    max_raw_frames_per_shard: int = DEFAULT_MAX_RAW_FRAMES_PER_SHARD
    stamp_shape: tuple[int, int] = DEFAULT_STAMP_SHAPE
    device: str = "cuda"
    run_seed: int = 20260714
    delivery_execution_mode: DeliveryExecutionMode | str = (
        DIRECT_SHARED_FILESYSTEM_DELIVERY_EXECUTION_MODE
    )

    def __post_init__(self) -> None:
        input_fits = Path(self.input_fits).expanduser().resolve()
        output_root = Path(self.output_root).expanduser().resolve()
        data_root = Path(self.data_root).expanduser().resolve()
        focalplane_registry = Path(self.focalplane_registry).expanduser().resolve()
        run_id = str(self.run_id).strip()
        if not run_id:
            raise ValueError("run_id must be non-empty")
        source_ids = tuple(
            _strict_source_id(value, name="source_ids") for value in self.source_ids
        )
        if not source_ids or len(set(source_ids)) != len(source_ids):
            raise ValueError("source_ids must be a non-empty unique sequence")
        duration_days = _finite_positive(self.duration_days, name="duration_days")
        raw_exposure_seconds = _finite_positive(
            self.raw_exposure_seconds,
            name="raw_exposure_seconds",
        )
        cadence_seconds = tuple(
            _finite_positive(value, name="cadence_seconds")
            for value in self.cadence_seconds
        )
        if not cadence_seconds or len(set(cadence_seconds)) != len(cadence_seconds):
            raise ValueError("cadence_seconds must be a non-empty unique sequence")
        raw_frames_float = duration_days * 86_400.0 / raw_exposure_seconds
        raw_frames = round(raw_frames_float)
        if raw_frames <= 0 or not math.isclose(
            raw_frames_float,
            float(raw_frames),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "duration_days must contain an integral number of raw exposures"
            )
        if isinstance(self.max_raw_frames_per_shard, (bool, np.bool_)) or int(
            self.max_raw_frames_per_shard
        ) <= 0:
            raise ValueError("max_raw_frames_per_shard must be a positive integer")
        try:
            ny, nx = (int(value) for value in self.stamp_shape)
        except (TypeError, ValueError) as error:
            raise ValueError("stamp_shape must contain two positive integers") from error
        if ny <= 0 or nx <= 0:
            raise ValueError("stamp_shape must contain two positive integers")
        if (ny, nx) != DEFAULT_STAMP_SHAPE:
            raise ValueError("formal Galaxy production freezes stamp_shape at 27x27")
        device = str(self.device).strip().lower()
        if device not in {"cpu", "cuda"}:
            raise ValueError("device must be 'cpu' or 'cuda'")
        if isinstance(self.run_seed, (bool, np.bool_)):
            raise ValueError("run_seed must be an integer")
        delivery_execution_mode = _normalise_delivery_execution_mode(
            self.delivery_execution_mode
        )

        object.__setattr__(self, "input_fits", input_fits)
        object.__setattr__(self, "output_root", output_root)
        object.__setattr__(self, "data_root", data_root)
        object.__setattr__(self, "focalplane_registry", focalplane_registry)
        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(self, "source_ids", source_ids)
        object.__setattr__(self, "duration_days", duration_days)
        object.__setattr__(self, "raw_exposure_seconds", raw_exposure_seconds)
        object.__setattr__(self, "cadence_seconds", cadence_seconds)
        object.__setattr__(
            self,
            "max_raw_frames_per_shard",
            int(self.max_raw_frames_per_shard),
        )
        object.__setattr__(self, "stamp_shape", (ny, nx))
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "run_seed", int(self.run_seed))
        object.__setattr__(
            self,
            "delivery_execution_mode",
            delivery_execution_mode,
        )

    @property
    def n_raw_frames(self) -> int:
        return round(self.duration_days * 86_400.0 / self.raw_exposure_seconds)

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
class GalaxyStampProductionPreparation:
    """Published immutable plan and inputs for target-time workers."""

    run_root: Path
    manifest_path: Path
    time_plan_path: Path
    time_plan: ContinuousTimeShardPlan


def build_galaxy_independent_production_spec(
    *,
    n_raw_frames: int,
    raw_exposure_seconds: float,
    device: str,
    run_seed: int,
) -> Any:
    """Create the approved independent-stamp scientific configuration.

    The preset supplies the ET physical defaults and temperature-driven legacy
    breathing.  This function freezes the approved detector-response policy,
    the 10-s global time axis, and the SD-24 expectation companion.
    """

    if isinstance(n_raw_frames, (bool, np.bool_)) or int(n_raw_frames) <= 0:
        raise ValueError("n_raw_frames must be a positive integer")
    exposure = _finite_positive(raw_exposure_seconds, name="raw_exposure_seconds")
    normalized_device = str(device).strip().lower()
    if normalized_device not in {"cpu", "cuda"}:
        raise ValueError("device must be 'cpu' or 'cuda'")
    if isinstance(run_seed, (bool, np.bool_)):
        raise ValueError("run_seed must be an integer")

    from astropy import units as u
    from photsim7.specs import BackgroundOutputPolicy

    from .presets import load_preset

    base = load_preset("et-stamp-production").simulation_spec
    response = replace(
        base.detector_response,
        enable_inter_pixel_response=True,
        inter_prv_rms=FORMAL_INTER_PRV_RMS_PERCENT * u.percent,
        inter_prv_nominal=100.0 * u.percent,
        enable_intra_pixel_response=True,
        intra_prv_rms=FORMAL_INTRA_PRV_RMS_PERCENT * u.percent,
        enable_pixel_phase_response=True,
        pixel_response_profile_mod="flux conserved",
        pixel_phase_profile_path=FORMAL_PIXEL_PHASE_PROFILE_PATH,
        scripted_sensitivity_enabled=False,
        whole_pixel_gain_normal_enabled=False,
        whole_pixel_gain_sinusoidal_enabled=False,
        enable_flat_field_correction=False,
    )
    readout = replace(
        base.readout,
        readout_noise=(
            FORMAL_READOUT_NOISE_E_PER_PIXEL * base.readout.readout_noise.unit
        ),
        column_noise_sigma_adu=(
            FORMAL_COLUMN_NOISE_SIGMA_ADU
            * base.readout.column_noise_sigma_adu.unit
        ),
    )
    return replace(
        base,
        observation=replace(
            base.observation,
            observing_duration=int(n_raw_frames) * exposure * u.s,
            exposure_duration=exposure * u.s,
            n_frames=int(n_raw_frames),
            n_raw_frames_per_coadd=1,
            frame_start_s=None,
        ),
        detector_response=response,
        readout=readout,
        artifacts=replace(
            base.artifacts,
            background_output_policy=BackgroundOutputPolicy.EXPECTATION,
        ),
        psf=replace(base.psf, compute_device=normalized_device, mode="stamp"),
        rng=replace(base.rng, run_seed=int(run_seed)),
    )


def _map_curve_to_detector(
    *,
    focalplane_registry: Path,
    registry_sha256: str,
    ra_deg: float,
    dec_deg: float,
) -> dict[str, Any]:
    from .stamp_inputs import (
        _focalplane_detector_pixel_shape,
        _sky_to_focal,
        _validate_physical_detector_coordinates,
    )

    mapped = _sky_to_focal(
        focalplane_registry,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        registry_sha256=registry_sha256,
    )
    if str(getattr(mapped, "status", "")) != "ok":
        raise ValueError(
            "Galaxy target is outside the fixed ET field of view: "
            f"ra_deg={ra_deg}, dec_deg={dec_deg}"
        )
    detector_id = str(getattr(mapped, "detector_id", "")).strip()
    if not detector_id:
        raise ValueError("focal-plane mapping returned no detector_id")
    values = {
        "detector_id": detector_id,
        "detector_xpix": float(getattr(mapped, "xpix")),
        "detector_ypix": float(getattr(mapped, "ypix")),
        "field_x_deg": float(getattr(mapped, "field_x_deg")),
        "field_y_deg": float(getattr(mapped, "field_y_deg")),
        "focalplane_residual_arcsec": float(getattr(mapped, "residual_arcsec")),
    }
    if not all(math.isfinite(value) for key, value in values.items() if key != "detector_id"):
        raise ValueError("focal-plane mapping returned a non-finite coordinate")
    physical_pixel_width, physical_pixel_height = _focalplane_detector_pixel_shape(
        focalplane_registry,
        detector_id=detector_id,
        registry_sha256=registry_sha256,
    )
    _validate_physical_detector_coordinates(
        detector_id=detector_id,
        detector_xpix=float(values["detector_xpix"]),
        detector_ypix=float(values["detector_ypix"]),
        pixel_width=physical_pixel_width,
        pixel_height=physical_pixel_height,
        context="Galaxy target mapping",
    )
    # Preserve enough geometry with the immutable mapping to make a later
    # audit independent of the renderer's row/column storage orientation.
    values["physical_detector_pixel_width"] = physical_pixel_width
    values["physical_detector_pixel_height"] = physical_pixel_height
    values["field_angle_deg"] = float(
        math.hypot(values["field_x_deg"], values["field_y_deg"])
    )
    return values


def _target_table_filename(detector_id: str) -> str:
    safe = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in detector_id
    )
    return f"targets_{safe}.ecsv"


def prepare_galaxy_independent_production(
    config: GalaxyStampProductionConfig,
) -> GalaxyStampProductionPreparation:
    """Freeze long-duration Galaxy inputs and a globally aligned shard plan.

    This performs no image rendering.  It deliberately writes the factor
    snapshots inside the output run root, so the formal worker never needs to
    reopen the original FITS or infer a time alignment from its native epoch.
    """

    if not isinstance(config, GalaxyStampProductionConfig):
        raise TypeError("config must be a GalaxyStampProductionConfig")
    if not Path(config.input_fits).is_file():
        raise FileNotFoundError(f"Galaxy FITS does not exist: {config.input_fits}")
    if not Path(config.data_root).is_dir():
        raise FileNotFoundError(f"Photsim7 data root does not exist: {config.data_root}")
    if not Path(config.focalplane_registry).is_dir():
        raise FileNotFoundError(
            "focalplane_registry does not exist: "
            f"{config.focalplane_registry}"
        )
    run_root = config.run_root
    if run_root.exists():
        raise FileExistsError(
            f"production run root already exists: {run_root}; formal runs are not resumed"
        )

    from astropy.table import Table
    from .stamp_inputs import focalplane_registry_identity

    curves = load_galaxy_lightcurves(config.input_fits, source_ids=config.source_ids)
    registry_identity = focalplane_registry_identity(config.focalplane_registry)
    time_plan = plan_continuous_time_shards(
        raw_start_index=0,
        raw_stop_index=config.n_raw_frames,
        coadd_sizes=config.coadd_sizes,
        raw_exposure_seconds=config.raw_exposure_seconds,
        max_raw_frames_per_shard=config.max_raw_frames_per_shard,
    )
    base_spec = build_galaxy_independent_production_spec(
        n_raw_frames=config.n_raw_frames,
        raw_exposure_seconds=config.raw_exposure_seconds,
        device=config.device,
        run_seed=config.run_seed,
    )

    # Validate every coordinate against the frozen physical focal-plane
    # geometry before creating the no-resume run root or writing any prepared
    # inputs.  A bad mapping must remain a retryable configuration error, not
    # leave an unusable partial formal run behind.
    mapped_curves = {
        source_id: _map_curve_to_detector(
            focalplane_registry=Path(config.focalplane_registry),
            registry_sha256=str(registry_identity["sha256"]),
            ra_deg=curve.ra_deg,
            dec_deg=curve.dec_deg,
        )
        for source_id, curve in curves.items()
    }

    inputs_root = run_root / "inputs"
    factors_root = inputs_root / "galaxy_factor_snapshots"
    tables_root = inputs_root / "target_tables"
    run_root.mkdir(parents=True, exist_ok=False)
    factors_root.mkdir(parents=True, exist_ok=False)
    tables_root.mkdir(parents=True, exist_ok=False)

    records: list[dict[str, Any]] = []
    grouped_rows: dict[str, list[tuple[int, float, float, float]]] = {}
    for source_id in config.source_ids:
        curve = curves[source_id]
        factors = exposure_averaged_factors(
            native_time_seconds=curve.native_time_seconds,
            clean_flux_factor=curve.clean_flux_factor,
            n_raw_frames=config.n_raw_frames,
            raw_exposure_seconds=config.raw_exposure_seconds,
        )
        snapshot_path = factors_root / f"source_{source_id}.npz"
        snapshot_identity = write_galaxy_factor_snapshot(
            snapshot_path,
            curve=curve,
            factors=factors,
            raw_exposure_seconds=config.raw_exposure_seconds,
        )
        mapping = dict(mapped_curves[source_id])
        detector_id = str(mapping["detector_id"])
        grouped_rows.setdefault(detector_id, []).append(
            (source_id, curve.gaia_g_mag, curve.ra_deg, curve.dec_deg)
        )
        records.append(
            {
                "source_id": str(source_id),
                "source_id_int64": source_id,
                "gaia_g_mag": curve.gaia_g_mag,
                "magnitude_system": "Gaia_G_Vega",
                "source_class": curve.source_class,
                "ra_deg": curve.ra_deg,
                "dec_deg": curve.dec_deg,
                "focalplane_mapping": mapping,
                "factor_snapshot": snapshot_identity,
                "factor_snapshot_path": str(snapshot_path),
                "factor_snapshot_relative_path": snapshot_path.relative_to(
                    run_root
                ).as_posix(),
                "factor_min": float(np.min(factors)),
                "factor_max": float(np.max(factors)),
                "factor_count": int(factors.size),
                "input_curve": curve.to_metadata(),
            }
        )

    table_identities: dict[str, dict[str, Any]] = {}
    for detector_id, rows in sorted(grouped_rows.items()):
        table_path = tables_root / _target_table_filename(detector_id)
        table = Table(
            rows=[
                (np.int64(source_id), g_mag, ra_deg, dec_deg)
                for source_id, g_mag, ra_deg, dec_deg in rows
            ],
            names=("source_id", "gaia_g_mag", "ra_deg", "dec_deg"),
        )
        table.meta = {
            "schema_id": GALAXY_STAMP_PRODUCTION_SCHEMA_ID,
            "coordinate_system": "ICRS_J2000",
            "magnitude_system": "Gaia_G_Vega",
            "scene_policy": "independent_target",
            "focalplane_registry_identity": dict(registry_identity),
        }
        table.write(table_path, format="ascii.ecsv", overwrite=False)
        table_identities[detector_id] = {
            "path": str(table_path),
            "relative_path": table_path.relative_to(run_root).as_posix(),
            "file_identity": file_identity(table_path),
        }
    for record in records:
        detector_id = str(record["focalplane_mapping"]["detector_id"])
        record["target_table"] = dict(table_identities[detector_id])

    time_plan_path = time_plan.write_manifest(inputs_root / "time_shards.json")
    spec_json = base_spec.to_json_dict()
    manifest = {
        "schema_id": GALAXY_STAMP_PRODUCTION_SCHEMA_ID,
        "schema_version": GALAXY_STAMP_PRODUCTION_SCHEMA_VERSION,
        "run_id": config.run_id,
        "run_root": str(run_root),
        "scientific_scope": "independent_target_only_no_neighbors",
        "observation_product": "final_dn",
        "background_realization_delivered": False,
        "input": {
            "galaxy_fits": file_identity(config.input_fits),
            "q_definition": "1_plus_delta_f_over_f_ref",
            "time_alignment": "simulation_raw_frame_index",
            "interpolation": "piecewise_linear_clean_flux",
            "exposure_sampling": "exact_interval_mean",
            "native_absolute_time_used": False,
            "focalplane_registry": dict(registry_identity),
        },
        "runtime_defaults": {
            "data_root": str(config.data_root),
            "focalplane_registry": str(config.focalplane_registry),
            "device": config.device,
        },
        "delivery": {
            "execution_mode": config.delivery_execution_mode,
            "stamp_shape": list(config.stamp_shape),
            "stamp_centering_policy": FORMAL_STAMP_CENTERING_POLICY,
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
        "targets": records,
        "target_tables": table_identities,
        "software_provenance_at_prepare": collect_provenance(
            Path(__file__).resolve().parents[2]
        ),
    }
    manifest_path = _atomic_json(run_root / "production_manifest.json", manifest)
    return GalaxyStampProductionPreparation(
        run_root=run_root,
        manifest_path=manifest_path,
        time_plan_path=time_plan_path,
        time_plan=time_plan,
    )


def _load_manifest(path: Path | str) -> tuple[Path, dict[str, Any]]:
    manifest_path = Path(path).expanduser().resolve()
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_id") != GALAXY_STAMP_PRODUCTION_SCHEMA_ID:
        raise ValueError("unsupported Galaxy stamp production manifest")
    if int(payload.get("schema_version", 0)) != GALAXY_STAMP_PRODUCTION_SCHEMA_VERSION:
        raise ValueError("unsupported Galaxy stamp production manifest version")
    return manifest_path, payload


def _manifest_target(payload: Mapping[str, Any], source_id: int) -> Mapping[str, Any]:
    targets = payload.get("targets")
    if not isinstance(targets, list):
        raise ValueError("production manifest targets must be a list")
    for candidate in targets:
        if not isinstance(candidate, Mapping):
            raise ValueError("production manifest target entries must be objects")
        if _strict_source_id(candidate.get("source_id_int64"), name="target.source_id") == source_id:
            return candidate
    raise ValueError(f"production manifest has no source_id={source_id}")


def _load_time_plan(
    run_root: Path,
    payload: Mapping[str, Any],
) -> ContinuousTimeShardPlan:
    delivery = payload.get("delivery")
    if not isinstance(delivery, Mapping):
        raise ValueError("production manifest delivery must be an object")
    path = _resolve_manifest_resource(
        run_root,
        {
            "relative_path": delivery.get("time_plan_relative_path"),
        },
        label="time shard plan",
    )
    expected_identity = delivery.get("time_plan_identity")
    if not isinstance(expected_identity, Mapping) or not _same_file_content_identity(
        file_identity(path),
        expected_identity,
    ):
        raise ValueError("time shard plan identity changed after production preparation")
    with path.open("r", encoding="utf-8") as handle:
        return ContinuousTimeShardPlan.from_manifest_dict(json.load(handle))


def _runtime_paths(
    payload: Mapping[str, Any],
    *,
    data_root: Path | str | None,
    focalplane_registry: Path | str | None,
) -> tuple[Path, Path, dict[str, Any], dict[str, Any]]:
    defaults = payload.get("runtime_defaults")
    if not isinstance(defaults, Mapping):
        raise ValueError("production manifest runtime_defaults must be an object")
    resolved_data_root = Path(
        data_root if data_root is not None else str(defaults.get("data_root", ""))
    ).expanduser().resolve()
    resolved_registry = Path(
        focalplane_registry
        if focalplane_registry is not None
        else str(defaults.get("focalplane_registry", ""))
    ).expanduser().resolve()
    if not resolved_data_root.is_dir():
        raise FileNotFoundError(f"Photsim7 data root does not exist: {resolved_data_root}")
    if not resolved_registry.is_dir():
        raise FileNotFoundError(
            f"focalplane registry does not exist: {resolved_registry}"
        )
    input_payload = payload.get("input")
    if not isinstance(input_payload, Mapping):
        raise ValueError("production manifest input must be an object")
    prepared_registry_identity = input_payload.get("focalplane_registry")
    if not isinstance(prepared_registry_identity, Mapping):
        raise ValueError("production manifest lacks frozen focal-plane registry identity")
    from .stamp_inputs import focalplane_registry_identity

    runtime_registry_identity = focalplane_registry_identity(resolved_registry)
    try:
        from et_coord import verify_semantic_registry_owner_attestation
    except (ImportError, AttributeError) as error:
        raise RuntimeError(
            "formal Galaxy production requires the et_coord owner-attestation verifier"
        ) from error
    try:
        verification = verify_semantic_registry_owner_attestation(
            resolved_registry,
            attestation=prepared_registry_identity.get("owner_attestation"),
        )
    except (OSError, TypeError, ValueError) as error:
        raise ValueError(
            "runtime focal-plane registry owner-attestation verification failed"
        ) from error
    _validate_formal_registry_gate(
        prepared_registry_identity,
        runtime_registry_identity,
        verification,
    )
    return (
        resolved_data_root,
        resolved_registry,
        dict(runtime_registry_identity),
        dict(verification),
    )


def run_galaxy_independent_target(
    manifest_path: Path | str,
    *,
    source_id: int,
    case: ProductionCase | str = "injected",
    shard_ids: Iterable[int] | None = None,
    data_root: Path | str | None = None,
    focalplane_registry: Path | str | None = None,
    device: str | None = None,
    batch_size: int = 32,
    output_root: Path | str | None = None,
) -> tuple[dict[str, Any], ...]:
    """Render one target for complete globally aligned shards without resume.

    A worker builds its physical services exactly once for the whole requested
    duration.  It then passes absolute raw frame indices unchanged into every
    shard callback.  Re-running an already-published target/shard is rejected
    by the atomic delivery writer rather than silently resuming it.  When
    ``output_root`` is supplied it is a case root (for example a node-local
    ``.../injected`` scratch directory), while the immutable production
    manifest and all HDF5 caller-manifest provenance remain canonical.
    """

    source_id = _strict_source_id(source_id, name="source_id")
    case = _normalise_case(str(case))
    if isinstance(batch_size, (bool, np.bool_)) or int(batch_size) <= 0:
        raise ValueError("batch_size must be a positive integer")
    resolved_manifest_path, manifest = _load_manifest(manifest_path)
    run_root = resolved_manifest_path.parent
    delivery_execution_mode = delivery_execution_mode_from_manifest(manifest)
    if (
        delivery_execution_mode
        == DIRECT_SHARED_FILESYSTEM_DELIVERY_EXECUTION_MODE
        and output_root is not None
    ):
        raise ValueError(
            "delivery.execution_mode='direct_shared_filesystem' requires "
            "output_root to be omitted"
        )
    if (
        delivery_execution_mode
        == STAGED_LOCAL_SCRATCH_DELIVERY_EXECUTION_MODE
        and output_root is None
    ):
        raise ValueError(
            "delivery.execution_mode='staged_local_scratch_v1' requires "
            "output_root to name the node-local case root"
        )
    production_manifest_identity = _file_content_identity(resolved_manifest_path)
    delivery_output_root = (
        run_root / "cases" / case
        if output_root is None
        else Path(output_root).expanduser().resolve()
    )
    if delivery_output_root.exists() and (
        not delivery_output_root.is_dir() or delivery_output_root.is_symlink()
    ):
        raise ValueError("output_root must be a directory path when it exists")
    target_record = _manifest_target(manifest, source_id)
    time_plan = _load_time_plan(run_root, manifest)
    (
        resolved_data_root,
        resolved_registry,
        runtime_registry_identity,
        runtime_registry_verification,
    ) = _runtime_paths(
        manifest,
        data_root=data_root,
        focalplane_registry=focalplane_registry,
    )
    requested_shards = None if shard_ids is None else {int(value) for value in shard_ids}
    shards = tuple(
        shard
        for shard in time_plan.shards
        if requested_shards is None or shard.shard_id in requested_shards
    )
    if not shards:
        raise ValueError("selected no time shards")
    if requested_shards is not None and {item.shard_id for item in shards} != requested_shards:
        raise ValueError("one or more requested shard_ids are absent from the plan")

    from photsim7.data_registry import DataRegistry
    from photsim7.simulation_services import (
        build_simulation_context,
        build_stamp_services,
    )
    from photsim7.source_variability import SourceVariability
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

    spec_payload = manifest.get("simulation_spec_base")
    if not isinstance(spec_payload, Mapping):
        raise ValueError("production manifest simulation_spec_base must be an object")
    base_spec = SimulationSpec.from_json_dict(dict(spec_payload))
    compute_device = str(device).strip().lower() if device is not None else base_spec.psf.compute_device
    if compute_device not in {"cpu", "cuda"}:
        raise ValueError("device must be 'cpu' or 'cuda'")
    detector_mapping = target_record.get("focalplane_mapping")
    if not isinstance(detector_mapping, Mapping):
        raise ValueError("target focalplane_mapping must be an object")
    detector_id = str(detector_mapping.get("detector_id", "")).strip()
    if not detector_id:
        raise ValueError("target focalplane_mapping has no detector_id")
    target_table = target_record.get("target_table")
    if not isinstance(target_table, Mapping):
        raise ValueError("target target_table must be an object")
    target_table_path = _resolve_manifest_resource(
        run_root,
        target_table,
        label="target table",
    )
    expected_target_table_identity = target_table.get("file_identity")
    if (
        not isinstance(expected_target_table_identity, Mapping)
        or not _same_file_content_identity(
            file_identity(target_table_path),
            expected_target_table_identity,
        )
    ):
        raise ValueError("target table identity changed after production preparation")

    spec = replace(
        base_spec,
        detector=replace(base_spec.detector, detector_id=detector_id),
        psf=replace(base_spec.psf, compute_device=compute_device),
    )
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
        paths=RunPaths(
            output_root=str(run_root),
            data_root=str(resolved_data_root),
            focalplane_registry=str(resolved_registry),
        ),
        execution=execution,
        workload=StampWorkload(
            input_mode="table",
            input_table=str(target_table_path),
            target_source_ids=(source_id,),
            stamp_rows=int(manifest["delivery"]["stamp_shape"][0]),
            stamp_cols=int(manifest["delivery"]["stamp_shape"][1]),
            include_neighbors=False,
            save_raw=True,
            save_coadd=True,
            write_batch_size=int(batch_size),
        ),
    )
    plan = build_run_plan(
        preset_name="galaxy-independent-stamp-production",
        run_config=run_config,
        spec=spec,
        repo_root=Path(__file__).resolve().parents[2],
        cwd=Path(__file__).resolve().parents[2],
    )
    api = _science_api()
    prepared = _prepare_table_inputs(plan, api, requested_target_ids=(source_id,))
    target = prepared.targets[source_id]
    snapshot_path = _resolve_manifest_resource(
        run_root,
        {
            "relative_path": target_record.get("factor_snapshot_relative_path"),
        },
        label="factor snapshot",
    )
    snapshot = read_galaxy_factor_snapshot(snapshot_path)
    if snapshot.source_id != source_id or snapshot.factors.size != int(
        spec.observation.resolved_n_frames
    ):
        raise ValueError("factor snapshot does not match the frozen source/time axis")
    expected_snapshot_identity = target_record.get("factor_snapshot")
    if not isinstance(expected_snapshot_identity, Mapping) or not _same_file_content_identity(
        file_identity(snapshot_path),
        expected_snapshot_identity,
    ):
        raise ValueError("factor snapshot identity changed after production preparation")

    source_truth = dict(prepared.source_input_truth[source_id])
    source_truth["runtime_focalplane_registry_identity"] = dict(
        runtime_registry_identity
    )
    source_truth["runtime_focalplane_registry_attestation_verification"] = dict(
        runtime_registry_verification
    )
    source_truth["variability"] = {
        "enabled": case == "injected",
        "case": case,
        "source_factor_snapshot": dict(snapshot.metadata),
        "source_factor_snapshot_identity": dict(expected_snapshot_identity),
        "semantics": "dimensionless_relative_flux_per_raw_frame",
        "time_alignment": "simulation_raw_frame_index",
    }
    catalog = _table_catalog(plan, target, api, source_input_truth=source_truth)
    target_spec = _target_spec(
        plan,
        target=target,
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
    source_variability = (
        None
        if case == "static"
        else SourceVariability(
            source_ids=np.asarray([source_id], dtype=np.int64),
            relative_flux=np.asarray(snapshot.factors[None, :], dtype=np.float64),
        )
    )
    renderer_options = {
        "enable_stellar_photon_noise": True,
        "enable_background_light": True,
        "enable_scattered_light": True,
        "enable_dark_current": True,
    }
    rng_trace_scope = {
        "workflow": "galaxy-independent-stamp-production",
        "run_id": str(manifest["run_id"]),
        "case": case,
    }
    target_spec_json = target_spec.to_json_dict()
    target_spec_sha256 = _canonical_json_sha256(target_spec_json)
    runtime_provenance = collect_provenance(Path(__file__).resolve().parents[2])

    def render_raw(global_frame_index: int) -> Any:
        return run_single_cadence_stamp(
            target_spec,
            target_source_id=source_id,
            stamp_shape=run_config.workload.stamp_shape,
            services=services,
            frame_index=global_frame_index,
            source_variability=source_variability,
            include_neighbors=False,
            renderer_options=renderer_options,
            rng_trace_scope=rng_trace_scope,
        )

    reports: list[dict[str, Any]] = []
    for shard in shards:
        physical_rng_pairing = _galaxy_physical_rng_pairing_metadata(
            context=context,
            source_id=source_id,
            shard=shard,
            target_spec_sha256=target_spec_sha256,
        )
        request = IndependentStampShardRequest(
            output_root=delivery_output_root,
            target_source_id=source_id,
            stamp_shape=run_config.workload.stamp_shape,
            shard=shard,
            gain_e_per_dn=float(target_spec.readout.gain_electrons_per_adu.value),
            manifest={
                "run_id": str(manifest["run_id"]),
                "case": case,
                "stamp_centering_policy": FORMAL_STAMP_CENTERING_POLICY,
                "rng_trace_scope": dict(rng_trace_scope),
                "physical_rng_pairing": dict(physical_rng_pairing),
                "production_manifest": str(resolved_manifest_path),
                "production_manifest_identity": dict(production_manifest_identity),
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
                "factor_snapshot_identity": dict(expected_snapshot_identity),
                "runtime_focalplane_registry_identity": dict(
                    runtime_registry_identity
                ),
                "runtime_focalplane_registry_attestation_verification": dict(
                    runtime_registry_verification
                ),
                "physical_rng_pairing": dict(physical_rng_pairing),
            },
            batch_size=int(batch_size),
            overwrite=False,
        )
        report = run_independent_stamp_time_shard(
            request,
            render_raw=render_raw,
            adapt_raw=raw_stamp_delivery_frame_from_photsim7,
        )
        reports.append(
            {
                "source_id": str(source_id),
                "case": case,
                "shard_id": report.shard_id,
                "raw_frame_count": report.raw_frame_count,
                "raw_path": str(report.raw_path),
                "coadd_paths": {str(key): str(value) for key, value in report.coadd_paths.items()},
            }
        )
    return tuple(reports)


__all__ = [
    "DIRECT_SHARED_FILESYSTEM_DELIVERY_EXECUTION_MODE",
    "DEFAULT_CADENCE_SECONDS",
    "DEFAULT_DURATION_DAYS",
    "DEFAULT_GALAXY_PRODUCTION_SOURCE_IDS",
    "DEFAULT_MAX_RAW_FRAMES_PER_SHARD",
    "DEFAULT_RAW_EXPOSURE_SECONDS",
    "DEFAULT_STAMP_SHAPE",
    "FORMAL_COLUMN_NOISE_SIGMA_ADU",
    "FORMAL_INTER_PRV_RMS_PERCENT",
    "FORMAL_INTRA_PRV_RMS_PERCENT",
    "FORMAL_PIXEL_PHASE_PROFILE_PATH",
    "FORMAL_READOUT_NOISE_E_PER_PIXEL",
    "FORMAL_STAMP_CENTERING_POLICY",
    "GALAXY_STAMP_PRODUCTION_SCHEMA_ID",
    "GALAXY_STAMP_PRODUCTION_SCHEMA_VERSION",
    "GalaxyStampProductionConfig",
    "GalaxyStampProductionPreparation",
    "STAGED_LOCAL_SCRATCH_DELIVERY_EXECUTION_MODE",
    "build_galaxy_independent_production_spec",
    "delivery_execution_mode_from_manifest",
    "prepare_galaxy_independent_production",
    "run_galaxy_independent_target",
]
