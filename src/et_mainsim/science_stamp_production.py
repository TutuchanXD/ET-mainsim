"""Common formal independent-stamp production for non-sky light curves.

The Aster, varlc, and wdlc tracks share this producer.  Their input adapters
freeze dimensionless 10-second source factors, while this module owns the
common 90-day time plan, explicit 12-degree PSF reference geometry, DVA-off
science spec, target table, and immutable production manifest.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
import errno
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
    DIRECT_SHARED_FILESYSTEM_DELIVERY_EXECUTION_MODE,
    DEFAULT_CADENCE_SECONDS,
    DEFAULT_DURATION_DAYS,
    DEFAULT_MAX_RAW_FRAMES_PER_SHARD,
    DEFAULT_RAW_EXPOSURE_SECONDS,
    DEFAULT_STAMP_SHAPE,
    GALAXY_STAMP_PRODUCTION_SCHEMA_ID,
    STAGED_LOCAL_SCRATCH_DELIVERY_EXECUTION_MODE,
    _atomic_json,
    _canonical_json_sha256,
    _file_content_identity,
    _finite_positive,
    _galaxy_physical_rng_pairing_metadata,
    _load_time_plan,
    _normalise_case,
    _normalise_delivery_execution_mode,
    _resolve_manifest_resource,
    _runtime_paths,
    _same_file_content_identity,
    _strict_source_id,
    build_galaxy_independent_production_spec,
    delivery_execution_mode_from_manifest,
)
from .independent_stamp_production import (
    IndependentStampShardRequest,
    raw_stamp_delivery_frame_from_photsim7,
    run_independent_stamp_time_shard,
)
from .provenance import collect_provenance
from .stamp_inputs import file_identity, focalplane_registry_identity
from .stamp_science_inputs import (
    ScienceInputCurve,
    load_science_track_inputs,
    read_science_factor_snapshot,
    write_science_factor_snapshot,
)
from .time_shards import (
    ContinuousTimeShardPlan,
    coadd_sizes_for_cadences,
    plan_continuous_time_shards,
)


SCIENCE_STAMP_PRODUCTION_SCHEMA_ID = "et_mainsim.science_stamp_production.v1"
SCIENCE_STAMP_PRODUCTION_SCHEMA_VERSION = 1
SCIENCE_STAMP_TASK_LIST_SCHEMA_ID = "et_mainsim.science_stamp_task_list.v1"
SCIENCE_STAMP_TASK_LIST_SCHEMA_VERSION = 1
_SUPPORTED_GALAXY_TASK_LIST_MANIFEST_VERSIONS = frozenset({2, 3})
SCIENCE_PRODUCTION_TRACKS = ("aster", "varlc", "wdlc")
REFERENCE_DETECTOR_ID = "main_rd"
REFERENCE_PSF_ID = 6
REFERENCE_PSF_NODE_ANGLE_DEG = 12.0
REFERENCE_DETECTOR_ROWS = 9_120
REFERENCE_DETECTOR_COLS = 8_900
_UNSUPPORTED_DIRECTORY_FSYNC_ERRNOS = frozenset(
    {
        errno.EINVAL,
        errno.ENOSYS,
        errno.ENOTSUP,
        errno.EOPNOTSUPP,
    }
)

ScienceProductionTrack = Literal["aster", "varlc", "wdlc"]
ScienceProductionCase = Literal["static", "injected"]


def _normalise_track(value: object) -> ScienceProductionTrack:
    track = str(value).strip().lower()
    if track not in SCIENCE_PRODUCTION_TRACKS:
        raise ValueError(
            "track must be one of " + ", ".join(SCIENCE_PRODUCTION_TRACKS)
        )
    return track  # type: ignore[return-value]


def _strict_positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if result <= 0 or result != value:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _strict_task_integer(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < 0 or result > int(np.iinfo(np.int64).max):
        raise ValueError(f"{name} must be a non-negative signed int64 integer")
    return result


@dataclass(frozen=True)
class ScienceStampProductionConfig:
    """Frozen request for one Aster, varlc, or wdlc no-resume campaign."""

    track: ScienceProductionTrack | str
    input_root: Path | str
    output_root: Path | str
    run_id: str
    data_root: Path | str
    focalplane_registry: Path | str
    external_source_ids: tuple[str, ...] | None = None
    duration_days: float = DEFAULT_DURATION_DAYS
    raw_exposure_seconds: float = DEFAULT_RAW_EXPOSURE_SECONDS
    cadence_seconds: tuple[float, ...] = DEFAULT_CADENCE_SECONDS
    max_raw_frames_per_shard: int = DEFAULT_MAX_RAW_FRAMES_PER_SHARD
    stamp_shape: tuple[int, int] = DEFAULT_STAMP_SHAPE
    device: str = "cuda"
    run_seed: int = 20260714
    delivery_execution_mode: str = STAGED_LOCAL_SCRATCH_DELIVERY_EXECUTION_MODE

    def __post_init__(self) -> None:
        track = _normalise_track(self.track)
        run_id = str(self.run_id).strip()
        if not run_id:
            raise ValueError("run_id must be non-empty")
        duration = _finite_positive(self.duration_days, name="duration_days")
        exposure = _finite_positive(
            self.raw_exposure_seconds,
            name="raw_exposure_seconds",
        )
        if not math.isclose(exposure, 10.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("science stamp production requires 10-second raw exposures")
        raw_frames_float = duration * 86_400.0 / exposure
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
        cadences = tuple(
            _finite_positive(value, name="cadence_seconds")
            for value in self.cadence_seconds
        )
        if not cadences or len(set(cadences)) != len(cadences):
            raise ValueError("cadence_seconds must be a non-empty unique sequence")
        try:
            rows, cols = (int(value) for value in self.stamp_shape)
        except (TypeError, ValueError) as error:
            raise ValueError("stamp_shape must contain two positive integers") from error
        if (rows, cols) != DEFAULT_STAMP_SHAPE:
            raise ValueError("formal science production freezes stamp_shape at 100x300")
        device = str(self.device).strip().lower()
        if device not in {"cpu", "cuda"}:
            raise ValueError("device must be 'cpu' or 'cuda'")
        if isinstance(self.run_seed, (bool, np.bool_)):
            raise ValueError("run_seed must be an integer")
        source_ids = self.external_source_ids
        if source_ids is not None:
            source_ids = tuple(str(value).strip() for value in source_ids)
            if (
                not source_ids
                or any(not value for value in source_ids)
                or len(set(source_ids)) != len(source_ids)
            ):
                raise ValueError(
                    "external_source_ids must be a non-empty unique sequence"
                )
        execution_mode = _normalise_delivery_execution_mode(
            self.delivery_execution_mode
        )
        object.__setattr__(self, "track", track)
        for name in (
            "input_root",
            "output_root",
            "data_root",
            "focalplane_registry",
        ):
            object.__setattr__(
                self,
                name,
                Path(getattr(self, name)).expanduser().resolve(),
            )
        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(self, "external_source_ids", source_ids)
        object.__setattr__(self, "duration_days", duration)
        object.__setattr__(self, "raw_exposure_seconds", exposure)
        object.__setattr__(self, "cadence_seconds", cadences)
        object.__setattr__(
            self,
            "max_raw_frames_per_shard",
            _strict_positive_integer(
                self.max_raw_frames_per_shard,
                name="max_raw_frames_per_shard",
            ),
        )
        object.__setattr__(self, "stamp_shape", (rows, cols))
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "run_seed", int(self.run_seed))
        object.__setattr__(self, "delivery_execution_mode", execution_mode)

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
class ScienceStampProductionPreparation:
    """Immutable manifest and time plan published by prepare."""

    run_root: Path
    manifest_path: Path
    time_plan_path: Path
    time_plan: ContinuousTimeShardPlan


@dataclass(frozen=True)
class ScienceStampTaskListWriteResult:
    """Identity of one immutable, case-bound formal array task list."""

    path: Path
    case: ScienceProductionCase
    task_count: int
    identity: Mapping[str, Any]


def _atomic_json_no_overwrite(path: Path, payload: Mapping[str, Any]) -> bytes:
    """Publish JSON atomically while preserving any pre-existing path."""

    destination = path.expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"task-list output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            dict(payload),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        # A hard-link is a same-directory, no-replace atomic publication.  A
        # racing creator makes this fail with FileExistsError instead of being
        # overwritten as it would be by os.replace().
        os.link(temporary, destination)
        try:
            parent_descriptor = os.open(
                destination.parent,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
            try:
                os.fsync(parent_descriptor)
            finally:
                os.close(parent_descriptor)
        except OSError as error:
            # The bytes are already durable and visible; some network file
            # systems do not support directory fsync.
            if error.errno not in _UNSUPPORTED_DIRECTORY_FSYNC_ERRNOS:
                raise
    finally:
        temporary.unlink(missing_ok=True)
    return encoded


def write_science_stamp_task_list(
    manifest_path: Path | str,
    *,
    case: ScienceProductionCase | str,
    tasks: Iterable[tuple[int, int]],
    output_path: Path | str,
) -> ScienceStampTaskListWriteResult:
    """Freeze an exact source/shard selection for one formal array case.

    The task list is bound to the byte identity of the production manifest.
    The staged launcher rechecks that identity before both rendering and
    publication, so a later manifest mutation fails closed.
    """

    resolved_case = _normalise_case(str(case))
    manifest_input = Path(manifest_path).expanduser()
    if manifest_input.is_symlink():
        raise ValueError("production manifest must not be a symbolic link")
    resolved_manifest = manifest_input.resolve()
    manifest_raw = resolved_manifest.read_bytes()
    try:
        manifest = json.loads(manifest_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("science production manifest is not valid JSON") from error
    if not isinstance(manifest, dict):
        raise ValueError("science production manifest must be an object")
    schema_id = manifest.get("schema_id")
    schema_version = manifest.get("schema_version")
    is_science_manifest = (
        schema_id == SCIENCE_STAMP_PRODUCTION_SCHEMA_ID
        and type(schema_version) is int
        and schema_version == SCIENCE_STAMP_PRODUCTION_SCHEMA_VERSION
    )
    is_galaxy_manifest = (
        schema_id == GALAXY_STAMP_PRODUCTION_SCHEMA_ID
        and type(schema_version) is int
        and schema_version in _SUPPORTED_GALAXY_TASK_LIST_MANIFEST_VERSIONS
    )
    if not is_science_manifest and not is_galaxy_manifest:
        raise ValueError("unsupported science stamp production manifest")
    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("production manifest run_id must be non-empty")
    if is_science_manifest:
        _normalise_track(manifest.get("production_track"))

    target_entries = manifest.get("targets")
    if not isinstance(target_entries, list) or not target_entries:
        raise ValueError("production manifest has no formal targets")
    source_ids: set[int] = set()
    namespaced_source_ids: set[tuple[str, str]] = set()
    for index, target in enumerate(target_entries):
        if not isinstance(target, Mapping):
            raise ValueError(f"target {index} must be an object")
        source_id = _strict_task_integer(
            target.get("source_id_int64"),
            name=f"target {index} source_id_int64",
        )
        if source_id in source_ids:
            raise ValueError("production manifest contains duplicate source identities")
        if target.get("source_id") != str(source_id):
            raise ValueError(
                f"target {index} source_id differs from source_id_int64"
            )
        if is_science_manifest:
            namespace = target.get("source_id_namespace")
            external_source_id = target.get("external_source_id")
            if (
                not isinstance(namespace, str)
                or not namespace.strip()
                or not isinstance(external_source_id, str)
                or not external_source_id.strip()
            ):
                raise ValueError(
                    f"target {index} lacks a complete namespaced source identity"
                )
            namespaced_identity = (namespace, external_source_id)
            if namespaced_identity in namespaced_source_ids:
                raise ValueError(
                    "production manifest contains duplicate namespaced source identities"
                )
            namespaced_source_ids.add(namespaced_identity)
        source_ids.add(source_id)

    time_plan = _load_time_plan(resolved_manifest.parent, manifest)
    shard_ids = {shard.shard_id for shard in time_plan.shards}
    normalised_tasks: list[dict[str, int]] = []
    seen: set[tuple[int, int]] = set()
    for index, task in enumerate(tasks):
        if (
            isinstance(task, (str, bytes))
            or not isinstance(task, Iterable)
        ):
            raise ValueError(f"task {index} must contain source_id and shard_id")
        values = tuple(task)
        if len(values) != 2:
            raise ValueError(f"task {index} must contain source_id and shard_id")
        source_id = _strict_task_integer(
            values[0],
            name=f"task {index} source_id",
        )
        shard_id = _strict_task_integer(
            values[1],
            name=f"task {index} shard_id",
        )
        pair = (source_id, shard_id)
        if pair in seen:
            raise ValueError(f"duplicate task source_id={source_id} shard_id={shard_id}")
        if source_id not in source_ids:
            raise ValueError(f"unknown source_id={source_id}")
        if shard_id not in shard_ids:
            raise ValueError(f"unknown shard_id={shard_id}")
        seen.add(pair)
        normalised_tasks.append({"source_id": source_id, "shard_id": shard_id})
    if not normalised_tasks:
        raise ValueError("task list requires at least one task")

    manifest_identity = {
        "sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "size_bytes": len(manifest_raw),
    }
    payload = {
        "schema_id": SCIENCE_STAMP_TASK_LIST_SCHEMA_ID,
        "schema_version": SCIENCE_STAMP_TASK_LIST_SCHEMA_VERSION,
        "case": resolved_case,
        "production_manifest_identity": manifest_identity,
        "tasks": normalised_tasks,
    }
    output_input = Path(output_path).expanduser()
    if output_input.is_symlink():
        raise FileExistsError(f"task-list output already exists: {output_input}")
    destination = output_input.resolve()
    encoded = _atomic_json_no_overwrite(destination, payload)
    return ScienceStampTaskListWriteResult(
        path=destination,
        case=resolved_case,  # type: ignore[arg-type]
        task_count=len(normalised_tasks),
        identity={
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "size_bytes": len(encoded),
        },
    )


def build_science_independent_production_spec(
    *,
    n_raw_frames: int,
    raw_exposure_seconds: float,
    device: str,
    run_seed: int,
) -> Any:
    """Build the validated ET stamp spec with DVA disabled for no-sky inputs."""

    base = build_galaxy_independent_production_spec(
        n_raw_frames=n_raw_frames,
        raw_exposure_seconds=raw_exposure_seconds,
        device=device,
        run_seed=run_seed,
    )
    return replace(
        base,
        dynamic_effects=replace(
            base.dynamic_effects,
            dva=replace(base.dynamic_effects.dva, enabled=False),
        ),
    )


def _validate_reference_position(
    curve: ScienceInputCurve,
    *,
    stamp_shape: tuple[int, int],
) -> None:
    rows, cols = stamp_shape
    half_x = (cols - 1) / 2.0
    half_y = (rows - 1) / 2.0
    if not (
        half_x <= curve.detector_xpix <= REFERENCE_DETECTOR_COLS - 1 - half_x
        and half_y
        <= curve.detector_ypix
        <= REFERENCE_DETECTOR_ROWS - 1 - half_y
    ):
        raise ValueError(
            "reference detector position must keep the full 100x300 stamp "
            "inside main_rd"
        )


def _selected_curves(
    config: ScienceStampProductionConfig,
) -> tuple[ScienceInputCurve, ...]:
    curves = tuple(
        load_science_track_inputs(
            config.track,
            config.input_root,
            duration_days=config.duration_days,
            raw_exposure_seconds=config.raw_exposure_seconds,
        )
    )
    if config.external_source_ids is not None:
        requested = set(config.external_source_ids)
        curves = tuple(
            curve for curve in curves if curve.external_source_id in requested
        )
        if {curve.external_source_id for curve in curves} != requested:
            missing = sorted(
                requested - {curve.external_source_id for curve in curves}
            )
            raise ValueError(f"input adapter did not provide requested sources: {missing}")
    if not curves:
        raise ValueError("science production requires at least one input curve")
    identities: set[tuple[str, str]] = set()
    source_ids: set[int] = set()
    for curve in curves:
        if not isinstance(curve, ScienceInputCurve):
            raise TypeError("input adapter must return ScienceInputCurve values")
        if curve.track != config.track:
            raise ValueError("input curve track differs from production track")
        identity = (curve.namespace, curve.external_source_id)
        if identity in identities or curve.source_id_int64 in source_ids:
            raise ValueError("input curves contain duplicate formal identities")
        identities.add(identity)
        source_ids.add(curve.source_id_int64)
        if curve.factors.shape != (config.n_raw_frames,):
            raise ValueError("input curve factor count differs from production duration")
        magnitude_origin = curve.metadata.get("magnitude_origin")
        if magnitude_origin not in {
            "project_default_missing_input",
            "precision_override_of_generator_magnitude_6",
        }:
            raise ValueError("input curve lacks a supported magnitude_origin")
        if curve.psf_id != REFERENCE_PSF_ID or not math.isclose(
            curve.psf_node_angle_deg,
            REFERENCE_PSF_NODE_ANGLE_DEG,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("input curve must use the frozen 12-degree PSF node")
        if curve.dva_enabled:
            raise ValueError("reference-field input curve must disable DVA")
        _validate_reference_position(curve, stamp_shape=config.stamp_shape)
    return curves


def prepare_science_independent_production(
    config: ScienceStampProductionConfig,
) -> ScienceStampProductionPreparation:
    """Freeze common input snapshots and one globally aligned time plan."""

    if not isinstance(config, ScienceStampProductionConfig):
        raise TypeError("config must be a ScienceStampProductionConfig")
    for path, label in (
        (Path(config.input_root), "input_root"),
        (Path(config.data_root), "Photsim7 data_root"),
        (Path(config.focalplane_registry), "focalplane_registry"),
    ):
        if not path.is_dir():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    if config.run_root.exists():
        raise FileExistsError(
            f"production run root already exists: {config.run_root}; "
            "formal runs are not resumed"
        )

    curves = _selected_curves(config)
    registry_identity = focalplane_registry_identity(config.focalplane_registry)
    time_plan = plan_continuous_time_shards(
        raw_start_index=0,
        raw_stop_index=config.n_raw_frames,
        coadd_sizes=config.coadd_sizes,
        raw_exposure_seconds=config.raw_exposure_seconds,
        max_raw_frames_per_shard=config.max_raw_frames_per_shard,
    )
    base_spec = build_science_independent_production_spec(
        n_raw_frames=config.n_raw_frames,
        raw_exposure_seconds=config.raw_exposure_seconds,
        device=config.device,
        run_seed=config.run_seed,
    )

    run_root = config.run_root
    try:
        factors_root = run_root / "inputs" / "factor_snapshots"
        tables_root = run_root / "inputs" / "target_tables"
        factors_root.mkdir(parents=True, exist_ok=False)
        tables_root.mkdir(parents=True, exist_ok=False)
        records: list[dict[str, Any]] = []
        table_rows: list[tuple[int, float, int, float, float]] = []
        for curve in curves:
            snapshot_path = factors_root / f"source_{curve.source_id_int64}.npz"
            snapshot_identity = write_science_factor_snapshot(
                snapshot_path,
                curve=curve,
            )
            records.append(
                {
                    "source_id": str(curve.source_id_int64),
                    "source_id_int64": curve.source_id_int64,
                    "external_source_id": curve.external_source_id,
                    "source_id_namespace": curve.namespace,
                    "source_class": curve.source_class,
                    "gaia_g_mag": curve.gaia_g_mag,
                    "magnitude_system": "Gaia_G_Vega",
                    "magnitude_origin": str(curve.metadata["magnitude_origin"]),
                    "factor_snapshot": snapshot_identity,
                    "factor_snapshot_relative_path": snapshot_path.relative_to(
                        run_root
                    ).as_posix(),
                    "factor_min": float(np.min(curve.factors)),
                    "factor_max": float(np.max(curve.factors)),
                    "factor_count": int(curve.factors.size),
                    "input_curve": dict(curve.metadata),
                    "detector_placement": {
                        "detector_id": REFERENCE_DETECTOR_ID,
                        "detector_xpix": curve.detector_xpix,
                        "detector_ypix": curve.detector_ypix,
                        "location_mode": "reference_field_nonphysical",
                    },
                    "psf": {
                        "selection": "explicit_field_id",
                        "psf_id": curve.psf_id,
                        "node_angle_deg": curve.psf_node_angle_deg,
                    },
                    "dva_policy": "disabled_no_sky_coordinate",
                }
            )
            table_rows.append(
                (
                    curve.source_id_int64,
                    curve.gaia_g_mag,
                    curve.psf_id,
                    curve.detector_xpix,
                    curve.detector_ypix,
                )
            )

        from astropy.table import Table

        table_path = tables_root / "targets_main_rd.ecsv"
        table = Table(
            rows=table_rows,
            names=(
                "source_id",
                "gaia_g_mag",
                "psf_id",
                "detector_xpix",
                "detector_ypix",
            ),
        )
        table.meta = {
            "schema_id": SCIENCE_STAMP_PRODUCTION_SCHEMA_ID,
            "production_track": config.track,
            "magnitude_system": "Gaia_G_Vega",
            "scene_policy": "independent_target",
            "geometry": "reference_field_nonphysical",
            "dva_enabled": False,
        }
        table.write(table_path, format="ascii.ecsv", overwrite=False)
        target_table = {
            "relative_path": table_path.relative_to(run_root).as_posix(),
            "file_identity": file_identity(table_path),
        }
        for record in records:
            record["target_table"] = dict(target_table)

        time_plan_path = time_plan.write_manifest(
            run_root / "inputs" / "time_shards.json"
        )
        spec_json = base_spec.to_json_dict()
        manifest = {
            "schema_id": SCIENCE_STAMP_PRODUCTION_SCHEMA_ID,
            "schema_version": SCIENCE_STAMP_PRODUCTION_SCHEMA_VERSION,
            "run_id": config.run_id,
            "production_track": config.track,
            "scientific_scope": "independent_target_only_no_neighbors",
            "observation_product": "final_dn",
            "background_realization_delivered": False,
            "input": {
                "input_root": str(config.input_root),
                "adapter_schema_id": "et_mainsim.stamp_science_inputs.v1",
                "q_semantics": "dimensionless_relative_flux_per_raw_frame",
                "time_alignment": "simulation_raw_frame_index",
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
                "raw_exposure_seconds": config.raw_exposure_seconds,
                "cadence_seconds": list(config.cadence_seconds),
                "coadd_sizes": list(config.coadd_sizes),
                "time_plan_relative_path": time_plan_path.relative_to(
                    run_root
                ).as_posix(),
                "time_plan_identity": _file_content_identity(time_plan_path),
                "tail_policy": "reject_incomplete_global_coadd_tail",
            },
            "simulation_spec_base": spec_json,
            "simulation_spec_base_sha256": _canonical_json_sha256(spec_json),
            "targets": records,
            "target_tables": {REFERENCE_DETECTOR_ID: target_table},
            "software_provenance_at_prepare": collect_provenance(
                Path(__file__).resolve().parents[2]
            ),
        }
        manifest_path = _atomic_json(run_root / "production_manifest.json", manifest)
    except BaseException:
        if run_root.exists():
            shutil.rmtree(run_root)
        raise
    return ScienceStampProductionPreparation(
        run_root=run_root,
        manifest_path=manifest_path,
        time_plan_path=time_plan_path,
        time_plan=time_plan,
    )


def _load_science_manifest(path: Path | str) -> tuple[Path, dict[str, Any]]:
    manifest_path = Path(path).expanduser().resolve()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as error:
        raise ValueError("science production manifest is not valid JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("science production manifest must be an object")
    if payload.get("schema_id") != SCIENCE_STAMP_PRODUCTION_SCHEMA_ID or int(
        payload.get("schema_version", 0)
    ) != SCIENCE_STAMP_PRODUCTION_SCHEMA_VERSION:
        raise ValueError("unsupported science stamp production manifest")
    return manifest_path, payload


def _manifest_target(
    payload: Mapping[str, Any],
    *,
    source_id: int,
) -> Mapping[str, Any]:
    targets = payload.get("targets")
    if not isinstance(targets, list):
        raise ValueError("production manifest targets must be a list")
    for target in targets:
        if not isinstance(target, Mapping):
            raise ValueError("production manifest target entries must be objects")
        if _strict_source_id(
            target.get("source_id_int64"),
            name="target source_id_int64",
        ) == source_id:
            return target
    raise ValueError(f"production manifest has no source_id={source_id}")


def _science_physical_rng_pairing_metadata(
    *,
    context: Any,
    source_id: int,
    shard: Any,
    target_spec_sha256: str,
) -> dict[str, Any]:
    payload = _galaxy_physical_rng_pairing_metadata(
        context=context,
        source_id=source_id,
        shard=shard,
        target_spec_sha256=target_spec_sha256,
    )
    payload["schema_id"] = "et_mainsim.science_physical_rng_pairing.v1"
    return payload


def run_science_independent_target(
    manifest_path: Path | str,
    *,
    source_id: int,
    case: ScienceProductionCase | str = "injected",
    shard_ids: Iterable[int] | None = None,
    data_root: Path | str | None = None,
    focalplane_registry: Path | str | None = None,
    device: str | None = None,
    batch_size: int = 32,
    output_root: Path | str | None = None,
) -> tuple[dict[str, Any], ...]:
    """Render selected no-resume shards for one prepared science source."""

    resolved_source_id = _strict_source_id(source_id, name="source_id")
    resolved_case = _normalise_case(str(case))
    resolved_batch_size = _strict_positive_integer(batch_size, name="batch_size")
    resolved_manifest_path, manifest = _load_science_manifest(manifest_path)
    execution_mode = delivery_execution_mode_from_manifest(manifest)
    if (
        execution_mode == DIRECT_SHARED_FILESYSTEM_DELIVERY_EXECUTION_MODE
        and output_root is not None
    ):
        raise ValueError(
            "delivery.execution_mode='direct_shared_filesystem' requires "
            "output_root to be omitted"
        )
    if (
        execution_mode == STAGED_LOCAL_SCRATCH_DELIVERY_EXECUTION_MODE
        and output_root is None
    ):
        raise ValueError(
            "delivery.execution_mode='staged_local_scratch_v1' requires "
            "output_root to name the node-local case root"
        )

    run_root = resolved_manifest_path.parent
    delivery_output_root = (
        run_root / "cases" / resolved_case
        if output_root is None
        else Path(output_root).expanduser().resolve()
    )
    if delivery_output_root.exists() and (
        not delivery_output_root.is_dir() or delivery_output_root.is_symlink()
    ):
        raise ValueError("output_root must be a directory path when it exists")
    production_manifest_identity = _file_content_identity(resolved_manifest_path)
    target_record = _manifest_target(manifest, source_id=resolved_source_id)
    time_plan = _load_time_plan(run_root, manifest)
    requested_shards = (
        None if shard_ids is None else {int(value) for value in shard_ids}
    )
    shards = tuple(
        shard
        for shard in time_plan.shards
        if requested_shards is None or shard.shard_id in requested_shards
    )
    if not shards:
        raise ValueError("selected no time shards")
    if requested_shards is not None and {
        item.shard_id for item in shards
    } != requested_shards:
        raise ValueError("one or more requested shard_ids are absent from the plan")

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

    snapshot_path = _resolve_manifest_resource(
        run_root,
        {"relative_path": target_record.get("factor_snapshot_relative_path")},
        label="factor snapshot",
    )
    expected_snapshot_identity = target_record.get("factor_snapshot")
    if (
        not isinstance(expected_snapshot_identity, Mapping)
        or not _same_file_content_identity(
            file_identity(snapshot_path),
            expected_snapshot_identity,
        )
    ):
        raise ValueError("factor snapshot identity changed after production preparation")
    snapshot = read_science_factor_snapshot(snapshot_path)
    if (
        snapshot.source_id_int64 != resolved_source_id
        or snapshot.external_source_id
        != str(target_record.get("external_source_id", ""))
        or snapshot.namespace != str(target_record.get("source_id_namespace", ""))
    ):
        raise ValueError("factor snapshot source identity differs from manifest")

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
    spec_payload = manifest.get("simulation_spec_base")
    if not isinstance(spec_payload, Mapping):
        raise ValueError("production manifest simulation_spec_base must be an object")

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

    base_spec = SimulationSpec.from_json_dict(dict(spec_payload))
    compute_device = (
        str(device).strip().lower()
        if device is not None
        else base_spec.psf.compute_device
    )
    if compute_device not in {"cpu", "cuda"}:
        raise ValueError("device must be 'cpu' or 'cuda'")
    placement = target_record.get("detector_placement")
    if not isinstance(placement, Mapping) or placement.get("detector_id") != (
        REFERENCE_DETECTOR_ID
    ):
        raise ValueError("target detector placement must use main_rd")
    spec = replace(
        base_spec,
        detector=replace(base_spec.detector, detector_id=REFERENCE_DETECTOR_ID),
        psf=replace(base_spec.psf, compute_device=compute_device),
    )
    if spec.dynamic_effects.dva.enabled:
        raise ValueError("reference-field production spec must disable DVA")
    if snapshot.factors.size != int(spec.observation.resolved_n_frames):
        raise ValueError("factor snapshot does not match the frozen source/time axis")

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
            target_source_ids=(resolved_source_id,),
            stamp_rows=int(manifest["delivery"]["stamp_shape"][0]),
            stamp_cols=int(manifest["delivery"]["stamp_shape"][1]),
            include_neighbors=False,
            save_raw=True,
            save_coadd=True,
            write_batch_size=resolved_batch_size,
        ),
    )
    plan = build_run_plan(
        preset_name="science-independent-stamp-production",
        run_config=run_config,
        spec=spec,
        repo_root=Path(__file__).resolve().parents[2],
        cwd=Path(__file__).resolve().parents[2],
    )
    api = _science_api()
    prepared = _prepare_table_inputs(
        plan,
        api,
        requested_target_ids=(resolved_source_id,),
    )
    target = prepared.targets[resolved_source_id]
    source_truth = dict(prepared.source_input_truth[resolved_source_id])
    source_truth["runtime_focalplane_registry_identity"] = dict(
        runtime_registry_identity
    )
    source_truth["runtime_focalplane_registry_attestation_verification"] = dict(
        runtime_registry_verification
    )
    source_truth["variability"] = {
        "enabled": resolved_case == "injected",
        "case": resolved_case,
        "source_factor_snapshot": dict(snapshot.metadata),
        "source_factor_snapshot_identity": dict(expected_snapshot_identity),
        "semantics": "dimensionless_relative_flux_per_raw_frame",
        "time_alignment": "simulation_raw_frame_index",
    }
    catalog = _table_catalog(plan, target, api, source_input_truth=source_truth)
    target_spec = _target_spec(
        plan,
        target=target,
        psf_id=prepared.psf_ids[resolved_source_id],
        source_input_truth=source_truth,
    )
    if target_spec.dynamic_effects.dva.enabled:
        raise ValueError("runtime target spec unexpectedly enabled DVA")
    context = build_simulation_context(
        target_spec,
        data_registry=DataRegistry(data_root=resolved_data_root),
        spacecraft_id="et",
        absolute_raw_frame_start_index=0,
    )
    services = build_stamp_services(context, catalog=catalog)
    source_variability = (
        None
        if resolved_case == "static"
        else SourceVariability(
            source_ids=np.asarray([resolved_source_id], dtype=np.int64),
            relative_flux=np.asarray(
                snapshot.factors[None, :],
                dtype=np.float64,
            ),
        )
    )
    renderer_options = {
        "enable_stellar_photon_noise": True,
        "enable_background_light": True,
        "enable_scattered_light": True,
        "enable_dark_current": True,
    }
    rng_trace_scope = {
        "workflow": "science-independent-stamp-production",
        "production_track": str(manifest["production_track"]),
        "run_id": str(manifest["run_id"]),
        "case": resolved_case,
    }
    target_spec_json = target_spec.to_json_dict()
    target_spec_sha256 = _canonical_json_sha256(target_spec_json)
    runtime_provenance = collect_provenance(Path(__file__).resolve().parents[2])

    def render_raw(global_frame_index: int) -> Any:
        return run_single_cadence_stamp(
            target_spec,
            target_source_id=resolved_source_id,
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
        physical_rng_pairing = _science_physical_rng_pairing_metadata(
            context=context,
            source_id=resolved_source_id,
            shard=shard,
            target_spec_sha256=target_spec_sha256,
        )
        request = IndependentStampShardRequest(
            output_root=delivery_output_root,
            target_source_id=resolved_source_id,
            stamp_shape=run_config.workload.stamp_shape,
            shard=shard,
            gain_e_per_dn=float(
                target_spec.readout.gain_electrons_per_adu.value
            ),
            manifest={
                "run_id": str(manifest["run_id"]),
                "production_track": str(manifest["production_track"]),
                "case": resolved_case,
                "rng_trace_scope": dict(rng_trace_scope),
                "physical_rng_pairing": dict(physical_rng_pairing),
                "production_manifest": str(resolved_manifest_path),
                "production_manifest_identity": dict(
                    production_manifest_identity
                ),
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
            batch_size=resolved_batch_size,
            overwrite=False,
        )
        report = run_independent_stamp_time_shard(
            request,
            render_raw=render_raw,
            adapt_raw=raw_stamp_delivery_frame_from_photsim7,
        )
        reports.append(
            {
                "source_id": str(resolved_source_id),
                "external_source_id": snapshot.external_source_id,
                "production_track": str(manifest["production_track"]),
                "case": resolved_case,
                "shard_id": report.shard_id,
                "raw_frame_count": report.raw_frame_count,
                "raw_path": str(report.raw_path),
                "coadd_paths": {
                    str(key): str(value)
                    for key, value in report.coadd_paths.items()
                },
            }
        )
    return tuple(reports)


__all__ = [
    "REFERENCE_DETECTOR_ID",
    "REFERENCE_PSF_ID",
    "REFERENCE_PSF_NODE_ANGLE_DEG",
    "SCIENCE_PRODUCTION_TRACKS",
    "SCIENCE_STAMP_PRODUCTION_SCHEMA_ID",
    "SCIENCE_STAMP_PRODUCTION_SCHEMA_VERSION",
    "SCIENCE_STAMP_TASK_LIST_SCHEMA_ID",
    "SCIENCE_STAMP_TASK_LIST_SCHEMA_VERSION",
    "ScienceStampProductionConfig",
    "ScienceStampProductionPreparation",
    "ScienceStampTaskListWriteResult",
    "build_science_independent_production_spec",
    "prepare_science_independent_production",
    "run_science_independent_target",
    "write_science_stamp_task_list",
]
