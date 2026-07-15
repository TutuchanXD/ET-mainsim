from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
from astropy.table import Table

from et_mainsim.config import (
    ResolvedRunPaths,
    RunConfig,
    StampWorkload,
    worker_assignments,
)
from et_mainsim.manifest import RunManifestStore
from et_mainsim.presets import resource_path
from et_mainsim.provenance import collect_provenance
from et_mainsim.stamp_inputs import (
    StampTarget,
    file_identity,
    load_stamp_target_table,
    load_stamp_variability_table,
)


@dataclass(frozen=True)
class StampRunPlan:
    preset_name: str
    run_config: RunConfig
    paths: ResolvedRunPaths
    spec: Any
    run_dir: Path
    catalog_cache: Path
    input_table_path: Path | None
    variability_table_path: Path | None
    repo_root: Path

    @property
    def workload(self) -> StampWorkload:
        workload = self.run_config.workload
        if not isinstance(workload, StampWorkload):
            raise TypeError("stamp run plan requires StampWorkload")
        return workload

    def to_dict(self, *, dry_run: bool) -> dict[str, Any]:
        return {
            "dry_run": bool(dry_run),
            "workflow": "et-stamp",
            "preset": self.preset_name,
            "run_id": self.run_config.run_id,
            "run_dir": str(self.run_dir),
            "catalog_cache": str(self.catalog_cache),
            "input_table": (
                None if self.input_table_path is None else str(self.input_table_path)
            ),
            "variability_table": (
                None
                if self.variability_table_path is None
                else str(self.variability_table_path)
            ),
            "paths": self.paths.to_dict(),
            "execution": self.run_config.execution.to_dict(),
            "workload": self.workload.to_dict(),
            "frame_plan": _frame_plan(self.spec, self.workload),
            "simulation_spec": self.spec.to_json_dict(),
        }


@dataclass(frozen=True)
class PreparedStampInputs:
    target_ids: tuple[int, ...]
    catalogs: Mapping[int, Any]
    psf_ids: Mapping[int, int]
    targets: Mapping[int, StampTarget]
    source_variability: Mapping[int, Any | None]
    source_input_truth: Mapping[int, Mapping[str, Any]]
    shared_catalog: Any | None
    provenance: Mapping[str, Any]
    input_identities: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StampWorkerRequest:
    plan: StampRunPlan
    target_ids: tuple[int, ...]
    rank: int = 0
    world_size: int = 1
    input_identities: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        target_ids = tuple(int(value) for value in self.target_ids)
        rank = int(self.rank)
        world_size = int(self.world_size)
        if rank < 0 or world_size <= 0 or rank >= world_size:
            raise ValueError("rank must be smaller than positive world_size")
        if self.plan.workload.coadd_shard_count > 1 and world_size != 1:
            raise ValueError(
                "stamp coadd sharding currently requires a single worker"
            )
        input_identities = dict(self.input_identities)
        if self.plan.workload.input_mode == "table" and not input_identities:
            raise ValueError(
                "stamp table worker requests require verified input identities"
            )
        object.__setattr__(self, "target_ids", target_ids)
        object.__setattr__(self, "rank", rank)
        object.__setattr__(self, "world_size", world_size)
        object.__setattr__(self, "input_identities", input_identities)

    @classmethod
    def from_plan(
        cls,
        plan: StampRunPlan,
        *,
        target_ids: tuple[int, ...],
        rank: int = 0,
        world_size: int = 1,
        input_identities: Mapping[str, Any] | None = None,
    ) -> "StampWorkerRequest":
        return cls(
            plan=plan,
            target_ids=target_ids,
            rank=rank,
            world_size=world_size,
            input_identities=dict(input_identities or {}),
        )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "schema_id": "et_mainsim.stamp_worker_request",
            "schema_version": 2,
            "preset_name": self.plan.preset_name,
            "run_config": {
                "schema_id": self.plan.run_config.schema_id,
                "schema_version": self.plan.run_config.schema_version,
                "workflow": self.plan.run_config.workflow,
                "run_id": self.plan.run_config.run_id,
                "paths": {
                    name: "" if value is None else str(value)
                    for name, value in self.plan.paths.to_dict().items()
                },
                "execution": self.plan.run_config.execution.to_dict(),
                "workload": self.plan.workload.to_dict(),
            },
            "resolved_paths": self.plan.paths.to_dict(),
            "simulation_spec": self.plan.spec.to_json_dict(),
            "run_dir": str(self.plan.run_dir),
            "catalog_cache": str(self.plan.catalog_cache),
            "input_table_path": (
                None
                if self.plan.input_table_path is None
                else str(self.plan.input_table_path)
            ),
            "variability_table_path": (
                None
                if self.plan.variability_table_path is None
                else str(self.plan.variability_table_path)
            ),
            "repo_root": str(self.plan.repo_root),
            "target_ids": list(self.target_ids),
            "rank": self.rank,
            "world_size": self.world_size,
            "input_identities": dict(self.input_identities),
        }

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, Any]) -> "StampWorkerRequest":
        if payload.get("schema_id") != "et_mainsim.stamp_worker_request":
            raise ValueError("Unsupported stamp worker request")
        if int(payload.get("schema_version", 0)) != 2:
            raise ValueError("Unsupported stamp worker request version")
        from photsim7.specs import SimulationSpec

        config = RunConfig.from_mapping(
            payload["run_config"],
            source="stamp-worker-request",
        )
        resolved = dict(payload["resolved_paths"])
        paths = ResolvedRunPaths(
            output_root=Path(resolved["output_root"]),
            data_root=(
                None if resolved.get("data_root") is None else Path(resolved["data_root"])
            ),
            catalog_path=(
                None
                if resolved.get("catalog_path") is None
                else Path(resolved["catalog_path"])
            ),
            focalplane_registry=(
                None
                if resolved.get("focalplane_registry") is None
                else Path(resolved["focalplane_registry"])
            ),
            catalog_cache=(
                None
                if resolved.get("catalog_cache") is None
                else Path(resolved["catalog_cache"])
            ),
        )
        plan = StampRunPlan(
            preset_name=str(payload["preset_name"]),
            run_config=config,
            paths=paths,
            spec=SimulationSpec.from_json_dict(payload["simulation_spec"]),
            run_dir=Path(payload["run_dir"]),
            catalog_cache=Path(payload["catalog_cache"]),
            input_table_path=(
                None
                if payload.get("input_table_path") is None
                else Path(payload["input_table_path"])
            ),
            variability_table_path=(
                None
                if payload.get("variability_table_path") is None
                else Path(payload["variability_table_path"])
            ),
            repo_root=Path(payload["repo_root"]),
        )
        return cls(
            plan=plan,
            target_ids=tuple(payload["target_ids"]),
            rank=int(payload["rank"]),
            world_size=int(payload["world_size"]),
            input_identities=dict(payload.get("input_identities", {})),
        )


def _science_api() -> SimpleNamespace:
    from photsim7.artifacts import ItemStatus, StampShardReader, StampShardWriter
    from photsim7.catalog_sources import PreparedStarCatalog, StarCatalogCache
    from photsim7.data_registry import DataRegistry
    from photsim7.simulation_services import (
        build_catalog_from_spec,
        build_stamp_services,
    )
    from photsim7.source_variability import SourceVariability
    from photsim7.psf.model import load_psf_bundle
    from photsim7.psf_bundle_paths import resolve_psf_bundle_filename
    from photsim7.stamp_pipeline import run_stamp_coadd
    from photsim7.stamp_products import write_stamp_product_schema

    return SimpleNamespace(
        PreparedStarCatalog=PreparedStarCatalog,
        StarCatalogCache=StarCatalogCache,
        DataRegistry=DataRegistry,
        ItemStatus=ItemStatus,
        StampShardReader=StampShardReader,
        StampShardWriter=StampShardWriter,
        build_catalog_from_spec=build_catalog_from_spec,
        build_stamp_services=build_stamp_services,
        SourceVariability=SourceVariability,
        load_psf_bundle=load_psf_bundle,
        resolve_psf_bundle_filename=resolve_psf_bundle_filename,
        run_stamp_coadd=run_stamp_coadd,
        write_stamp_product_schema=write_stamp_product_schema,
    )


def _resolve_table_path(value: str, *, cwd: Path) -> Path:
    prefix = "package://"
    if value.startswith(prefix):
        name = value[len(prefix) :]
        if not name or "/" in name or "\\" in name:
            raise ValueError(f"Invalid packaged table reference {value!r}")
        return Path(resource_path(name)).resolve()
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = cwd / path
    return path.resolve(strict=False)


def _table_spec(spec: Any) -> Any:
    reference_options = {
        name: value
        for name, value in spec.catalog.query_options.items()
        if name
        in {"reference_field_angle_deg", "reference_field_polar_angle_rad"}
    }
    return replace(
        spec,
        catalog=replace(
            spec.catalog,
            source_type="prepared",
            source_path="",
            registry_data_dir="",
            cache_path="",
            query_options=reference_options,
            input_magnitude_system="Gaia_G",
            photon_magnitude_system="ET",
            magnitude_conversion="gaia_g_vega_equals_et_ab_g2v_approx",
        ),
    )


def build_run_plan(
    *,
    preset_name: str,
    run_config: RunConfig,
    spec: Any,
    repo_root: Path | str,
    env: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    frames: int | None = None,
    target_epoch_jyear: float | None = None,
    run_seed: int | None = None,
) -> StampRunPlan:
    if run_config.workflow != "et-stamp" or not isinstance(
        run_config.workload, StampWorkload
    ):
        raise ValueError("stamp plan requires workflow='et-stamp'")
    from et_mainsim.workflows.full_frame import resolve_simulation_spec

    base = Path.cwd() if cwd is None else Path(cwd)
    paths = run_config.resolve_paths(env=env, cwd=base)
    logical_run_dir = paths.output_root / run_config.run_id
    workload = run_config.workload
    shard_relative_path = _coadd_shard_relative_path(workload)
    run_dir = (
        logical_run_dir
        if shard_relative_path == "."
        else logical_run_dir / shard_relative_path
    )
    catalog_cache = paths.catalog_cache or logical_run_dir / "cache" / "stars.npz"
    resolved_spec = resolve_simulation_spec(
        spec,
        paths=paths,
        catalog_cache=catalog_cache,
        frames=frames,
        target_epoch_jyear=target_epoch_jyear,
        run_seed=run_seed,
        device=run_config.execution.device,
    )
    if resolved_spec.psf.mode != "stamp":
        raise ValueError("et-stamp requires a SimulationSpec with psf.mode='stamp'")
    input_table_path = None
    variability_table_path = None
    if run_config.workload.input_mode == "table":
        input_table_path = _resolve_table_path(
            run_config.workload.input_table,
            cwd=base,
        )
        resolved_spec = _table_spec(resolved_spec)
        if run_config.workload.variability_table:
            variability_table_path = _resolve_table_path(
                run_config.workload.variability_table,
                cwd=base,
            )
    _frame_plan(resolved_spec, workload)
    return StampRunPlan(
        preset_name=preset_name,
        run_config=run_config,
        paths=paths,
        spec=resolved_spec,
        run_dir=run_dir,
        catalog_cache=catalog_cache,
        input_table_path=input_table_path,
        variability_table_path=variability_table_path,
        repo_root=Path(repo_root).resolve(),
    )


def _coadd_shard_relative_path(workload: StampWorkload) -> str:
    if workload.coadd_shard_count == 1:
        return "."
    return (
        f"coadd_shard_{workload.coadd_shard_index:04d}_of_"
        f"{workload.coadd_shard_count:04d}"
    )


def _coadd_shard_geometry(
    spec: Any,
    workload: StampWorkload | None = None,
) -> tuple[int, int, int, int, int, int]:
    global_raw_count = int(spec.observation.resolved_n_frames)
    per_coadd = int(spec.observation.n_raw_frames_per_coadd)
    if global_raw_count % per_coadd:
        raise ValueError("stamp raw frame count must be divisible by coadd size")
    global_coadd_count = global_raw_count // per_coadd
    shard_index = 0 if workload is None else workload.coadd_shard_index
    shard_count = 1 if workload is None else workload.coadd_shard_count
    if shard_index >= global_coadd_count:
        raise ValueError(
            f"coadd shard {shard_index}/{shard_count} selects no global coadds "
            f"from {global_coadd_count}"
        )
    selected_coadd_count = (
        (global_coadd_count - 1 - shard_index) // shard_count + 1
    )
    return (
        global_raw_count,
        global_coadd_count,
        per_coadd,
        shard_index,
        shard_count,
        selected_coadd_count,
    )


def _frame_plan(
    spec: Any,
    workload: StampWorkload | None = None,
) -> dict[str, Any]:
    (
        global_raw_count,
        global_coadd_count,
        per_coadd,
        shard_index,
        shard_count,
        selected_coadd_count,
    ) = _coadd_shard_geometry(spec, workload)
    coadd_indices = tuple(range(shard_index, global_coadd_count, shard_count))
    raw_frame_indices = tuple(
        frame_index
        for coadd_index in coadd_indices
        for frame_index in range(
            coadd_index * per_coadd,
            (coadd_index + 1) * per_coadd,
        )
    )
    return {
        "global_raw_frame_count": global_raw_count,
        "global_coadd_count": global_coadd_count,
        "n_raw_frames_per_coadd": per_coadd,
        "coadd_shard_index": shard_index,
        "coadd_shard_count": shard_count,
        "raw_frame_count": len(raw_frame_indices),
        "raw_frame_indices": list(raw_frame_indices),
        "coadd_count": selected_coadd_count,
        "coadd_indices": list(coadd_indices),
    }


def preflight(plan: StampRunPlan) -> None:
    _frame_plan(plan.spec, plan.workload)
    if plan.paths.data_root is None:
        raise ValueError("ET_DATA_DIR or paths.data_root is required to run")
    if not plan.paths.data_root.is_dir():
        raise FileNotFoundError(
            f"Photsim7 data root does not exist: {plan.paths.data_root}"
        )
    if plan.workload.input_mode == "table":
        if plan.input_table_path is None or not plan.input_table_path.is_file():
            raise FileNotFoundError(
                f"stamp target table does not exist: {plan.input_table_path}"
            )
        if (
            plan.variability_table_path is not None
            and not plan.variability_table_path.is_file()
        ):
            raise FileNotFoundError(
                "stamp variability table does not exist: "
                f"{plan.variability_table_path}"
            )
        return
    cache_available = (
        plan.catalog_cache.is_file()
        and not plan.run_config.execution.force_catalog_cache
    )
    catalog = plan.spec.catalog
    if catalog.source_type == "et_focalplane_query":
        if not catalog.registry_data_dir or not Path(
            catalog.registry_data_dir
        ).is_dir():
            raise FileNotFoundError(
                "ET_FOCALPLANE_ROOT or paths.focalplane_registry must reference focal-plane data"
            )
        if cache_available:
            return
        if not catalog.source_path or not Path(catalog.source_path).is_dir():
            raise FileNotFoundError(
                "GAIA_CATALOG_DIR or paths.catalog_path must reference a catalog directory"
            )
        focalplane_src = Path(catalog.query_options["et_focalplane_src"])
        if not focalplane_src.is_dir():
            raise FileNotFoundError(
                f"ET focal-plane source does not exist: {focalplane_src}"
            )
    elif cache_available:
        return
    elif catalog.source_type != "prepared" and not Path(catalog.source_path).is_file():
        raise FileNotFoundError(f"Catalog source does not exist: {catalog.source_path}")


def _table_catalog(
    plan: StampRunPlan,
    target: StampTarget,
    api: Any,
    *,
    source_input_truth: Mapping[str, Any] | None = None,
) -> Any:
    rows, cols = (int(value) for value in plan.spec.detector.shape)
    center_x = (cols - 1) / 2.0
    center_y = (rows - 1) / 2.0
    star_data: dict[str, Any] = {
        "x0": np.asarray([0.0], dtype=np.float64),
        "y0": np.asarray([0.0], dtype=np.float64),
        "frame_xpix": np.asarray([center_x], dtype=np.float64),
        "frame_ypix": np.asarray([center_y], dtype=np.float64),
        "ra": np.asarray(
            [
                target.ra_deg
                if target.ra_deg is not None
                else plan.spec.catalog.target_ra_deg or 0.0
            ]
        ),
        "dec": np.asarray(
            [
                target.dec_deg
                if target.dec_deg is not None
                else plan.spec.catalog.target_dec_deg or 0.0
            ]
        ),
        "source_id": np.asarray([target.source_id], dtype=np.int64),
        "gaia_g_mag": np.asarray([target.gaia_g_mag], dtype=np.float64),
        "detector_xpix": np.asarray([target.detector_xpix], dtype=np.float64),
        "detector_ypix": np.asarray([target.detector_ypix], dtype=np.float64),
        "detector_xpix_shifted": np.asarray([center_x], dtype=np.float64),
        "detector_ypix_shifted": np.asarray([center_y], dtype=np.float64),
        "detector_id": str(plan.spec.detector.detector_id),
    }
    if target.field_x_deg is not None:
        star_data["field_x_deg"] = np.asarray(
            [target.field_x_deg], dtype=np.float64
        )
        star_data["field_y_deg"] = np.asarray(
            [target.field_y_deg], dtype=np.float64
        )
        star_data["field_angle_deg"] = np.asarray(
            [target.field_angle_deg], dtype=np.float64
        )
    return api.PreparedStarCatalog(
        star_data=star_data,
        metadata={
            "source": {
                "type": "et_mainsim_stamp_target_table",
                "row_source_id": target.source_id,
                "n_sources": 1,
            },
            "magnitude": {
                "input_column": "gaia_g_mag",
                "input_system": "Gaia_G",
                "photon_system": "ET",
                "conversion": "gaia_g_vega_equals_et_ab_g2v_approx",
            },
            "scene_policy": "independent_target_only_no_neighbors",
            "source_input_truth": dict(source_input_truth or {}),
        },
    )


@dataclass(frozen=True)
class _PsfBundleIndex:
    node_angles_deg: Mapping[int, float]
    provenance: Mapping[str, Any]


def _psf_bundle_asset_identity(plan: StampRunPlan) -> dict[str, Any]:
    if plan.paths.data_root is None:
        raise ValueError("data_root is required")
    bundle_name = str(plan.spec.psf.bundle_name)
    if bundle_name.lower().startswith("kp_"):
        raise ValueError(
            "stamp table input requires a deterministic ET PSF bundle; "
            "kp_N randomly reselects backing interpolants and is not supported"
        )
    from photsim7.psf_bundle_paths import resolve_psf_bundle_filename

    bundle_path = Path(
        resolve_psf_bundle_filename(bundle_name, plan.paths.data_root)
    )
    return file_identity(bundle_path)


def _load_psf_bundle_index(plan: StampRunPlan, api: Any) -> _PsfBundleIndex:
    if plan.paths.data_root is None:
        raise ValueError("data_root is required")
    bundle_name = str(plan.spec.psf.bundle_name)
    if bundle_name.lower().startswith("kp_"):
        raise ValueError(
            "stamp table input requires a deterministic ET PSF bundle; "
            "kp_N randomly reselects backing interpolants and is not supported"
        )
    bundle = api.load_psf_bundle(
        bundle_name,
        n_rows=int(plan.spec.detector.shape[0]),
        n_cols=int(plan.spec.detector.shape[1]),
        n_subpixels=int(plan.spec.detector.n_subpixels),
        pad_to_detector_shape=False,
        base_data_dir=str(plan.paths.data_root),
    )
    images = bundle.get("images")
    angles = np.asarray(bundle.get("angles"), dtype=np.float64)
    if not isinstance(images, Mapping) or not images:
        raise ValueError(f"PSF bundle {bundle_name!r} has no field IDs")
    node_angles: dict[int, float] = {}
    for raw_field_id in images:
        field_id = int(raw_field_id)
        if field_id < 0 or field_id >= angles.size:
            raise ValueError(
                f"PSF bundle field ID {field_id} has no matching angle node"
            )
        angle = float(angles[field_id])
        if not np.isfinite(angle):
            raise ValueError(f"PSF bundle field ID {field_id} angle is not finite")
        node_angles[field_id] = angle
    provenance: dict[str, Any] = {
        "bundle_name": bundle_name,
        "available_field_ids": sorted(node_angles),
        "node_angles_deg": {
            str(key): value for key, value in sorted(node_angles.items())
        },
    }
    try:
        provenance["file_identity"] = _psf_bundle_asset_identity(plan)
    except FileNotFoundError:
        if hasattr(api, "resolve_psf_bundle_filename"):
            raise
    return _PsfBundleIndex(
        node_angles_deg=node_angles,
        provenance=provenance,
    )


def _resolve_target_psf(
    target: StampTarget,
    *,
    bundle: _PsfBundleIndex,
) -> StampTarget:
    if target.location_mode == "explicit_psf":
        requested = int(target.psf_id)  # validated by the target-table loader
        if requested not in bundle.node_angles_deg:
            raise ValueError(
                f"target {target.source_id} requests unavailable PSF ID {requested}; "
                f"available={sorted(bundle.node_angles_deg)}"
            )
        return replace(
            target,
            psf_node_angle_deg=float(bundle.node_angles_deg[requested]),
        )
    if target.field_angle_deg is None:
        raise ValueError(
            f"coordinate target {target.source_id} is missing field_angle_deg"
        )
    available_ids = np.asarray(sorted(bundle.node_angles_deg), dtype=np.int64)
    available_angles = np.asarray(
        [bundle.node_angles_deg[int(value)] for value in available_ids],
        dtype=np.float64,
    )
    offset = int(
        np.argmin(np.abs(available_angles - float(target.field_angle_deg)))
    )
    chosen_id = int(available_ids[offset])
    node_angle = float(available_angles[offset])
    return replace(
        target,
        psf_id=chosen_id,
        psf_node_angle_deg=node_angle,
        psf_angle_delta_deg=abs(float(target.field_angle_deg) - node_angle),
    )


def _source_input_truth(
    target: StampTarget,
    *,
    target_provenance: Mapping[str, Any],
    variability_provenance: Mapping[str, Any] | None,
    psf_bundle_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    location = {
        "mode": target.location_mode,
        "coordinate_frame": (
            "ICRS_J2000" if target.location_mode == "sky_icrs_j2000" else None
        ),
        "ra_deg": target.ra_deg,
        "dec_deg": target.dec_deg,
        "detector_xpix": target.detector_xpix,
        "detector_ypix": target.detector_ypix,
        "field_x_deg": target.field_x_deg,
        "field_y_deg": target.field_y_deg,
        "field_angle_deg": target.field_angle_deg,
        "focalplane_residual_arcsec": target.focalplane_residual_arcsec,
    }
    psf = {
        "selection_policy": (
            "nearest_radial_field_angle"
            if target.location_mode == "sky_icrs_j2000"
            else "explicit_field_id"
        ),
        "chosen_psf_id": target.psf_id,
        "node_angle_deg": target.psf_node_angle_deg,
        "angle_delta_deg": target.psf_angle_delta_deg,
        "bundle": dict(psf_bundle_provenance),
    }
    variability_identity = (
        None
        if variability_provenance is None
        else variability_provenance.get("file_identity")
    )
    return {
        "schema_id": "et_mainsim.stamp_source_input_truth",
        "schema_version": 1,
        "source_id": target.source_id,
        "gaia_g_mag": target.gaia_g_mag,
        "magnitude_system": "Gaia_G_Vega",
        "target_table_identity": target_provenance["file_identity"],
        "target_table_meta": target_provenance.get("table_meta", {}),
        "focalplane_registry_identity": target_provenance.get(
            "focalplane_registry_identity"
        ),
        "location": location,
        "psf": psf,
        "variability": {
            "enabled": target.curve_id is not None,
            "curve_id": target.curve_id,
            "semantics": "dimensionless_relative_flux_per_raw_frame",
            "time_alignment": "simulation_raw_frame_index",
            "variability_table_identity": variability_identity,
            "variability_table_meta": (
                {}
                if variability_provenance is None
                else variability_provenance.get("table_meta", {})
            ),
        },
    }


def _validate_input_identities(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any] | None,
) -> None:
    if not expected:
        return
    if dict(actual) != dict(expected):
        raise ValueError(
            "stamp worker input identity mismatch; a target, variability, "
            "focalplane registry, or PSF bundle input changed after worker planning"
        )


def _prepare_table_inputs(
    plan: StampRunPlan,
    api: Any,
    *,
    requested_target_ids: tuple[int, ...] | None = None,
    apply_workload_selection: bool = True,
    expected_identities: Mapping[str, Any] | None = None,
) -> PreparedStampInputs:
    if plan.input_table_path is None:
        raise ValueError("table input path is required")
    loaded = load_stamp_target_table(
        plan.input_table_path,
        detector_shape=tuple(plan.spec.detector.shape),
        detector_id=str(plan.spec.detector.detector_id),
        focalplane_registry=plan.paths.focalplane_registry,
    )
    targets = list(loaded.targets)
    requested = (
        set(requested_target_ids)
        if requested_target_ids is not None
        else set(plan.workload.target_source_ids)
    )
    if requested:
        by_available_id = {target.source_id: target for target in targets}
        missing = sorted(requested - set(by_available_id))
        if missing:
            raise ValueError(f"stamp target rows are absent: {missing}")
        targets = [
            by_available_id[target_id]
            for target_id in (
                requested_target_ids
                if requested_target_ids is not None
                else plan.workload.target_source_ids
            )
        ]
    if apply_workload_selection and plan.workload.target_limit:
        targets = targets[: plan.workload.target_limit]
    if not targets:
        raise ValueError("stamp target table selected no rows")

    variability = None
    if plan.variability_table_path is not None:
        variability = load_stamp_variability_table(
            plan.variability_table_path,
            raw_frame_count=int(plan.spec.observation.resolved_n_frames),
        )
    referenced_curve_ids = sorted(
        {target.curve_id for target in targets if target.curve_id is not None}
    )
    if referenced_curve_ids and variability is None:
        raise ValueError(
            "stamp targets reference curve_id values but no variability_table was provided"
        )
    available_curve_ids = set() if variability is None else set(variability.curves)
    missing_curves = sorted(set(referenced_curve_ids) - available_curve_ids)
    if missing_curves:
        raise ValueError(
            f"stamp targets reference absent variability curves: {missing_curves}"
        )

    bundle = _load_psf_bundle_index(plan, api)
    targets = [_resolve_target_psf(target, bundle=bundle) for target in targets]
    input_identities: dict[str, Any] = {
        "target_table": loaded.provenance["file_identity"],
    }
    if variability is not None:
        input_identities["variability_table"] = variability.provenance[
            "file_identity"
        ]
    registry_identity = loaded.provenance.get("focalplane_registry_identity")
    if registry_identity is not None:
        input_identities["focalplane_registry"] = registry_identity
    if bundle.provenance.get("file_identity") is not None:
        input_identities["psf_bundle"] = bundle.provenance["file_identity"]
    _validate_input_identities(input_identities, expected_identities)

    source_variability: dict[int, Any | None] = {}
    input_truth: dict[int, Mapping[str, Any]] = {}
    catalogs: dict[int, Any] = {}
    for target in targets:
        curve = (
            None
            if target.curve_id is None or variability is None
            else variability.curves[target.curve_id]
        )
        source_variability[target.source_id] = (
            None
            if curve is None
            else api.SourceVariability(
                source_ids=np.asarray([target.source_id], dtype=np.int64),
                relative_flux=np.asarray([curve], dtype=np.float64),
            )
        )
        truth = _source_input_truth(
            target,
            target_provenance=loaded.provenance,
            variability_provenance=(
                None if variability is None else variability.provenance
            ),
            psf_bundle_provenance=bundle.provenance,
        )
        input_truth[target.source_id] = truth
        catalogs[target.source_id] = _table_catalog(
            plan,
            target,
            api,
            source_input_truth=truth,
        )

    unreferenced = sorted(available_curve_ids - set(referenced_curve_ids))
    provenance = {
        "schema_id": "et_mainsim.prepared_stamp_table_inputs",
        "schema_version": 1,
        "scene_policy": "one_independent_target_per_row_no_neighbors",
        "target_table": dict(loaded.provenance),
        "variability_table": (
            None if variability is None else dict(variability.provenance)
        ),
        "variability_selection": {
            "referenced_curve_ids": referenced_curve_ids,
            "referenced_curve_count": len(referenced_curve_ids),
            "unreferenced_curve_ids": unreferenced,
            "unreferenced_curve_count": len(unreferenced),
            "static_target_count": sum(
                target.curve_id is None for target in targets
            ),
            "variable_target_count": sum(
                target.curve_id is not None for target in targets
            ),
        },
        "psf_bundle": dict(bundle.provenance),
        "targets": [input_truth[target.source_id] for target in targets],
    }
    return PreparedStampInputs(
        target_ids=tuple(target.source_id for target in targets),
        catalogs=catalogs,
        psf_ids={target.source_id: int(target.psf_id) for target in targets},
        targets={target.source_id: target for target in targets},
        source_variability=source_variability,
        source_input_truth=input_truth,
        shared_catalog=None,
        provenance=provenance,
        input_identities=input_identities,
    )


def _select_target_ids(catalog: Any, workload: StampWorkload) -> tuple[int, ...]:
    if "source_id" not in catalog.star_data:
        raise ValueError("stamp catalog must provide source_id")
    available = tuple(int(value) for value in catalog.star_data["source_id"])
    requested = workload.target_source_ids or available
    missing = sorted(set(requested) - set(available))
    if missing:
        raise ValueError(f"stamp target source IDs are absent from catalog: {missing}")
    selected = tuple(requested)
    if workload.target_limit:
        selected = selected[: workload.target_limit]
    if not selected:
        raise ValueError("stamp workload selected no target sources")
    return selected


def prepare_stamp_inputs(
    plan: StampRunPlan,
    *,
    science_api: Any | None = None,
) -> PreparedStampInputs:
    api = _science_api() if science_api is None else science_api
    workload = plan.workload
    if workload.input_mode == "table":
        return _prepare_table_inputs(plan, api)

    if plan.paths.data_root is None:
        raise ValueError("data_root is required")
    if plan.run_config.execution.force_catalog_cache:
        plan.catalog_cache.unlink(missing_ok=True)
    registry = api.DataRegistry(data_root=plan.paths.data_root)
    catalog = api.build_catalog_from_spec(plan.spec, data_registry=registry)
    target_ids = _select_target_ids(catalog, workload)
    return PreparedStampInputs(
        target_ids=target_ids,
        catalogs={target_id: catalog for target_id in target_ids},
        psf_ids={},
        targets={},
        source_variability={target_id: None for target_id in target_ids},
        source_input_truth={target_id: {} for target_id in target_ids},
        shared_catalog=catalog,
        provenance={
            "schema_id": "et_mainsim.stamp_catalog_selection",
            "schema_version": 1,
            "catalog_cache": str(plan.catalog_cache),
            "n_sources": int(catalog.n_sources),
            "target_ids": list(target_ids),
            "include_neighbors": workload.include_neighbors,
        },
        input_identities={},
    )


def _target_dir(plan: StampRunPlan, target_id: int) -> Path:
    return plan.run_dir / "stamps" / f"target_{int(target_id)}"


def _target_artifact_manifest_path(plan: StampRunPlan, target_id: int) -> Path:
    return _target_dir(plan, target_id) / "target_artifacts.json"


def _source_variability_truth_path(plan: StampRunPlan, target_id: int) -> Path:
    return _target_dir(plan, target_id) / "source_variability_truth.ecsv"


def _artifact_policy(plan: StampRunPlan) -> dict[str, Any]:
    detailed = plan.workload.artifact_profile == "detailed"
    return {
        "profile": plan.workload.artifact_profile,
        "raw_schema_sidecars": bool(plan.workload.save_raw and detailed),
        "coadd_schema_sidecars": bool(plan.workload.save_coadd and detailed),
        "electron_component_sidecars": bool(
            plan.workload.save_electron_components
        ),
        "write_batch_size": plan.workload.write_batch_size,
        "write_strategy": "batch_preferred_with_single_write_fallback",
    }


def _coadd_shard_provenance(plan: StampRunPlan) -> dict[str, Any]:
    (
        global_raw_count,
        global_coadd_count,
        per_coadd,
        shard_index,
        shard_count,
        selected_coadd_count,
    ) = _coadd_shard_geometry(plan.spec, plan.workload)
    return {
        "logical_run_id": plan.run_config.run_id,
        "output_relative_path": _coadd_shard_relative_path(plan.workload),
        "global_raw_frame_count": global_raw_count,
        "global_coadd_count": global_coadd_count,
        "n_raw_frames_per_coadd": per_coadd,
        "coadd_shard_index": shard_index,
        "coadd_shard_count": shard_count,
        "selected_raw_frame_count": selected_coadd_count * per_coadd,
        "selected_coadd_count": selected_coadd_count,
        "selection_rule": {
            "kind": "strided_global_coadds",
            "start": shard_index,
            "stop_exclusive": global_coadd_count,
            "step": shard_count,
            "raw_frame_mapping": "contiguous_blocks_by_coadd_index",
        },
    }


def _read_target_artifacts(
    plan: StampRunPlan,
    target_id: int,
) -> dict[str, Any] | None:
    path = _target_artifact_manifest_path(plan, target_id)
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("schema_id") != "et_mainsim.stamp_target_artifacts":
            return None
        if int(payload.get("schema_version", 0)) != 1:
            return None
        if int(payload.get("target_source_id", -1)) != int(target_id):
            return None
        expected = payload.get("source_variability_truth_identity")
        truth_path = _source_variability_truth_path(plan, target_id)
        if not isinstance(expected, Mapping) or not truth_path.is_file():
            return None
        if file_identity(truth_path) != dict(expected):
            return None
        table = Table.read(truth_path, format="ascii.ecsv")
        raw_ids, _ = _expected_ids(plan)
        if len(table) != len(raw_ids):
            return None
        required_columns = {
            "frame_index",
            "source_id",
            "curve_id",
            "relative_flux",
            "baseline_photon_count_electron",
            "effective_photon_count_electron",
            "psf_field_id",
        }
        if not required_columns.issubset(table.colnames):
            return None
        if tuple(int(value) for value in table["frame_index"]) != raw_ids:
            return None
        if any(int(value) != int(target_id) for value in table["source_id"]):
            return None
        relative_flux = np.asarray(table["relative_flux"], dtype=np.float64)
        baseline = np.asarray(
            table["baseline_photon_count_electron"], dtype=np.float64
        )
        effective = np.asarray(
            table["effective_photon_count_electron"], dtype=np.float64
        )
        if not all(
            np.all(np.isfinite(values))
            for values in (relative_flux, baseline, effective)
        ):
            return None
        if np.any(relative_flux < 0.0) or np.any(baseline < 0.0):
            return None
        if not np.allclose(
            effective,
            baseline * relative_flux,
            rtol=1e-12,
            atol=1e-9,
        ):
            return None
        if any(int(value) < 0 for value in table["psf_field_id"]):
            return None
        return payload
    except (KeyError, OSError, TypeError, ValueError):
        return None


def _expected_ids(plan: StampRunPlan) -> tuple[tuple[int, ...], tuple[int, ...]]:
    frame_plan = _frame_plan(plan.spec, plan.workload)
    return (
        tuple(frame_plan["raw_frame_indices"]),
        tuple(frame_plan["coadd_indices"]),
    )


def _shard_complete(
    path: Path,
    *,
    target_id: int,
    frame_ids: tuple[int, ...],
    api: Any,
) -> bool:
    if not path.is_file():
        return False
    try:
        with api.StampShardReader(path) as reader:
            return (
                reader.is_complete
                and reader.star_ids == (int(target_id),)
                and reader.frame_ids == frame_ids
            )
    except (OSError, RuntimeError, TypeError, ValueError):
        return False


def target_is_complete(plan: StampRunPlan, target_id: int, *, api: Any) -> bool:
    raw_ids, coadd_ids = _expected_ids(plan)
    target_dir = _target_dir(plan, target_id)
    checks = []
    if plan.workload.save_raw:
        checks.append(
            _shard_complete(
                target_dir / "raw.h5",
                target_id=target_id,
                frame_ids=raw_ids,
                api=api,
            )
        )
        if plan.workload.artifact_profile == "detailed":
            checks.append(
                all(
                    (
                        target_dir
                        / "schemas"
                        / "raw"
                        / f"frame_{frame_id:06d}.json"
                    ).is_file()
                    for frame_id in raw_ids
                )
            )
    if plan.workload.save_coadd:
        checks.append(
            _shard_complete(
                target_dir / "coadd.h5",
                target_id=target_id,
                frame_ids=coadd_ids,
                api=api,
            )
        )
        if plan.workload.artifact_profile == "detailed":
            checks.append(
                all(
                    (
                        target_dir
                        / "schemas"
                        / "coadd"
                        / f"coadd_{coadd_id:06d}.json"
                    ).is_file()
                    for coadd_id in coadd_ids
                )
            )
    checks.append(_read_target_artifacts(plan, target_id) is not None)
    return bool(checks) and all(checks)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _save_electron_components(path: Path, products: Any) -> None:
    arrays: dict[str, np.ndarray] = {}
    if products.electron_stamp is not None:
        arrays["electron_stamp"] = _to_numpy(products.electron_stamp.array)
    for name, product in products.electron_components.items():
        arrays[str(name)] = _to_numpy(product.array)
    if not arrays:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _truth_value_for_target(
    payload: Mapping[str, Any],
    name: str,
    *,
    target_id: int,
) -> float | int:
    if "source_id" not in payload or name not in payload:
        raise RuntimeError(f"stamp truth payload is missing {name!r}")
    source_ids = _to_numpy(payload["source_id"]).reshape(-1)
    offsets = np.flatnonzero(source_ids == int(target_id))
    if offsets.size != 1:
        raise RuntimeError(
            f"stamp truth must contain target {target_id} exactly once"
        )
    values = _to_numpy(payload[name]).reshape(-1)
    if values.size != source_ids.size:
        raise RuntimeError(
            f"stamp truth field {name!r} does not align with source_id"
        )
    value = values[int(offsets[0])]
    if np.issubdtype(values.dtype, np.integer):
        return int(value)
    converted = float(value)
    if not np.isfinite(converted):
        raise RuntimeError(f"stamp truth field {name!r} is not finite")
    return converted


class _SourceVariabilityTruthAccumulator:
    __slots__ = (
        "raw_frame_count",
        "frame_indices",
        "target_id",
        "curve_id",
        "coadd_shard",
        "relative_flux",
        "baseline_photon_count_electron",
        "effective_photon_count_electron",
        "psf_field_id",
        "_seen",
    )

    def __init__(
        self,
        *,
        raw_frame_count: int | None = None,
        frame_indices: tuple[int, ...] | None = None,
        target_id: int,
        curve_id: str | None,
        coadd_shard: Mapping[str, Any] | None = None,
    ) -> None:
        if frame_indices is None:
            if raw_frame_count is None:
                raise ValueError("raw_frame_count or frame_indices is required")
            count = int(raw_frame_count)
            if count <= 0:
                raise ValueError("raw_frame_count must be positive")
            indices = np.arange(count, dtype=np.int64)
        else:
            if raw_frame_count is not None:
                raise ValueError(
                    "raw_frame_count and frame_indices are mutually exclusive"
                )
            indices = np.asarray(
                tuple(int(value) for value in frame_indices),
                dtype=np.int64,
            )
            if indices.size == 0:
                raise ValueError("frame_indices must not be empty")
            if np.any(indices < 0) or np.any(np.diff(indices) <= 0):
                raise ValueError(
                    "frame_indices must be sorted unique non-negative integers"
                )
        self.raw_frame_count = int(indices.size)
        self.frame_indices = indices
        self.target_id = int(target_id)
        self.curve_id = "" if curve_id is None else str(curve_id)
        self.coadd_shard = (
            None if coadd_shard is None else dict(coadd_shard)
        )
        self.relative_flux = np.empty(self.raw_frame_count, dtype=np.float64)
        self.baseline_photon_count_electron = np.empty(
            self.raw_frame_count,
            dtype=np.float64,
        )
        self.effective_photon_count_electron = np.empty(
            self.raw_frame_count,
            dtype=np.float64,
        )
        self.psf_field_id = np.empty(self.raw_frame_count, dtype=np.int64)
        self._seen = np.zeros(self.raw_frame_count, dtype=bool)

    def add(self, products: Any) -> None:
        frame_index = int(products.frame_index)
        offset = int(np.searchsorted(self.frame_indices, frame_index))
        if (
            offset >= self.raw_frame_count
            or int(self.frame_indices[offset]) != frame_index
        ):
            raise RuntimeError(
                f"raw frame {frame_index} is not selected for this coadd shard"
            )
        if self._seen[offset]:
            raise RuntimeError(f"duplicate raw frame {frame_index} in truth output")
        truth = getattr(products, "truth", None)
        payload = None if truth is None else getattr(truth, "payload", None)
        if not isinstance(payload, Mapping):
            raise RuntimeError(
                "stamp products must expose the numeric truth payload"
            )
        self.relative_flux[offset] = float(
            _truth_value_for_target(
                payload,
                "source_relative_flux_factor",
                target_id=self.target_id,
            )
        )
        self.baseline_photon_count_electron[offset] = float(
            _truth_value_for_target(
                payload,
                "source_baseline_photon_count_electron",
                target_id=self.target_id,
            )
        )
        self.effective_photon_count_electron[offset] = float(
            _truth_value_for_target(
                payload,
                "source_effective_photon_count_electron",
                target_id=self.target_id,
            )
        )
        self.psf_field_id[offset] = int(
            _truth_value_for_target(
                payload,
                "source_psf_field_index",
                target_id=self.target_id,
            )
        )
        self._seen[offset] = True

    def to_table(
        self,
        *,
        source_input_truth: Mapping[str, Any],
    ) -> Table:
        missing = np.flatnonzero(~self._seen)
        if missing.size:
            preview = self.frame_indices[missing[:10]].tolist()
            raise RuntimeError(
                "source variability truth is missing raw frames: "
                f"{preview}{'...' if missing.size > len(preview) else ''}"
            )
        meta = {
            "schema_id": "et_mainsim.source_variability_truth",
            "schema_version": 1,
            "target_source_id": self.target_id,
            "time_alignment": "simulation_raw_frame_index",
            "source_input_truth": dict(source_input_truth),
        }
        if self.coadd_shard is not None:
            meta["coadd_shard"] = dict(self.coadd_shard)
        return Table(
            {
                "frame_index": self.frame_indices,
                "source_id": np.full(
                    self.raw_frame_count,
                    self.target_id,
                    dtype=np.int64,
                ),
                "curve_id": np.full(self.raw_frame_count, self.curve_id),
                "relative_flux": self.relative_flux,
                "baseline_photon_count_electron": (
                    self.baseline_photon_count_electron
                ),
                "effective_photon_count_electron": (
                    self.effective_photon_count_electron
                ),
                "psf_field_id": self.psf_field_id,
            },
            meta=meta,
            copy=False,
        )


def _write_source_variability_truth(
    path: Path,
    *,
    accumulator: _SourceVariabilityTruthAccumulator,
    source_input_truth: Mapping[str, Any],
) -> dict[str, Any]:
    table = accumulator.to_table(source_input_truth=source_input_truth)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.write(path, format="ascii.ecsv", overwrite=True)
    return file_identity(path)


def _target_spec(
    plan: StampRunPlan,
    *,
    target: StampTarget | None,
    psf_id: int | None,
) -> Any:
    if target is not None and target.location_mode == "sky_icrs_j2000":
        return replace(
            plan.spec,
            psf=replace(
                plan.spec.psf,
                field_id="nearest",
                field_id_policy="nearest",
            ),
        )
    if psf_id is None:
        return plan.spec
    return replace(
        plan.spec,
        psf=replace(
            plan.spec.psf,
            field_id=int(psf_id),
            field_id_policy=None,
        ),
    )


class _StampWriteBuffer:
    __slots__ = ("writer", "batch_size", "_items")

    def __init__(self, writer: Any, *, batch_size: int) -> None:
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.writer = writer
        self.batch_size = batch_size
        self._items: list[tuple[int, int, np.ndarray, int]] = []

    @property
    def pending_count(self) -> int:
        return len(self._items)

    def add(
        self,
        *,
        star_id: int,
        frame_id: int,
        stamp: np.ndarray,
        seed: int,
    ) -> None:
        self._items.append((star_id, frame_id, stamp, seed))
        if len(self._items) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        if not self._items:
            return
        write_stamps = getattr(self.writer, "write_stamps", None)
        if callable(write_stamps):
            write_stamps(tuple(self._items))
        else:
            for star_id, frame_id, stamp, seed in self._items:
                self.writer.write_stamp(
                    star_id,
                    frame_id,
                    stamp,
                    seed=seed,
                )
        self._items.clear()


def _open_writer(
    *,
    api: Any,
    path: Path,
    plan: StampRunPlan,
    target_id: int,
    case_id: str,
    frame_ids: tuple[int, ...],
    array: np.ndarray,
    unit: str,
    domain: str,
    product_schema: Mapping[str, Any] | None,
    source_input_truth: Mapping[str, Any],
    coadd_shard: Mapping[str, Any],
) -> Any | None:
    if path.is_file():
        if _shard_complete(
            path,
            target_id=target_id,
            frame_ids=frame_ids,
            api=api,
        ):
            return None
        raise RuntimeError(f"existing stamp shard failed readback validation: {path}")
    return api.StampShardWriter(
        path,
        run_id=plan.run_config.run_id,
        case_id=case_id,
        star_ids=(target_id,),
        frame_ids=frame_ids,
        stamp_shape=plan.workload.stamp_shape,
        dtype=array.dtype,
        unit=unit,
        domain=domain,
        provenance={
            "workflow": "et-stamp",
            "preset": plan.preset_name,
            "target_source_id": target_id,
            "input_mode": plan.workload.input_mode,
            "source_input_truth": dict(source_input_truth),
            "artifact_policy": _artifact_policy(plan),
            "coadd_shard": dict(coadd_shard),
        },
        product_schema=product_schema,
        resume=plan.run_config.execution.resume,
    )


def _render_target(
    plan: StampRunPlan,
    *,
    target_id: int,
    catalog: Any,
    psf_id: int | None,
    target: StampTarget | None,
    source_variability: Any | None,
    source_input_truth: Mapping[str, Any],
    api: Any,
    worker_rank: int = 0,
) -> dict[str, Any]:
    target_dir = _target_dir(plan, target_id)
    if target_is_complete(plan, target_id, api=api):
        if plan.run_config.execution.resume:
            artifacts = _read_target_artifacts(plan, target_id)
            return {
                "target_id": target_id,
                "status": "skipped",
                "artifacts": artifacts,
            }
        if not plan.run_config.execution.overwrite:
            raise FileExistsError(
                f"target {target_id} already has complete artifacts; use resume or overwrite"
            )
    if plan.run_config.execution.overwrite and target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    if plan.paths.data_root is None:
        raise ValueError("data_root is required")

    spec = _target_spec(plan, target=target, psf_id=psf_id)
    registry = api.DataRegistry(data_root=plan.paths.data_root)
    services = api.build_stamp_services(
        spec,
        catalog=catalog,
        data_registry=registry,
    )
    raw_ids, coadd_ids = _expected_ids(plan)
    coadd_shard = _coadd_shard_provenance(plan)
    raw_writer = None
    coadd_writer = None
    raw_write_buffer = None
    coadd_write_buffer = None
    raw_path = target_dir / "raw.h5"
    coadd_path = target_dir / "coadd.h5"
    truth_accumulator = _SourceVariabilityTruthAccumulator(
        frame_indices=raw_ids,
        target_id=target_id,
        curve_id=(None if target is None else target.curve_id),
        coadd_shard=coadd_shard,
    )
    try:
        for coadd_index in coadd_ids:
            result = api.run_stamp_coadd(
                spec,
                target_source_id=target_id,
                stamp_shape=plan.workload.stamp_shape,
                coadd_index=coadd_index,
                services=services,
                source_variability=source_variability,
                include_neighbors=plan.workload.include_neighbors,
                worker_rank=worker_rank,
                rng_trace_scope={
                    "run_id": plan.run_config.run_id,
                    "workflow": "et-stamp",
                },
            )
            first_raw = result.raw_results[0].stamp_products
            first_raw_array = _to_numpy(first_raw.final_stamp.array)
            coadd_products = result.coadd_products
            coadd_array = _to_numpy(coadd_products.coadd_stamp.array)
            if plan.workload.save_raw and raw_writer is None and not raw_path.is_file():
                raw_writer = _open_writer(
                    api=api,
                    path=raw_path,
                    plan=plan,
                    target_id=target_id,
                    case_id="raw",
                    frame_ids=raw_ids,
                    array=first_raw_array,
                    unit=first_raw.final_stamp.unit,
                    domain=first_raw.final_stamp.domain,
                    product_schema=first_raw.to_schema_dict(),
                    source_input_truth=source_input_truth,
                    coadd_shard=coadd_shard,
                )
                if raw_writer is not None:
                    raw_write_buffer = _StampWriteBuffer(
                        raw_writer,
                        batch_size=plan.workload.write_batch_size,
                    )
            if (
                plan.workload.save_coadd
                and coadd_writer is None
                and not coadd_path.is_file()
            ):
                coadd_writer = _open_writer(
                    api=api,
                    path=coadd_path,
                    plan=plan,
                    target_id=target_id,
                    case_id="coadd",
                    frame_ids=coadd_ids,
                    array=coadd_array,
                    unit=coadd_products.coadd_stamp.unit,
                    domain=coadd_products.coadd_stamp.domain,
                    product_schema=None,
                    source_input_truth=source_input_truth,
                    coadd_shard=coadd_shard,
                )
                if coadd_writer is not None:
                    coadd_write_buffer = _StampWriteBuffer(
                        coadd_writer,
                        batch_size=plan.workload.write_batch_size,
                    )

            for raw_result in result.raw_results:
                products = raw_result.stamp_products
                frame_id = int(products.frame_index)
                truth_accumulator.add(products)
                if (
                    plan.workload.save_raw
                    and plan.workload.artifact_profile == "detailed"
                ):
                    schema = products.to_schema_dict()
                    schema["source_input_truth"] = dict(source_input_truth)
                    schema["coadd_shard"] = dict(coadd_shard)
                    api.write_stamp_product_schema(
                        target_dir
                        / "schemas"
                        / "raw"
                        / f"frame_{frame_id:06d}.json",
                        schema,
                    )
                if raw_writer is not None and raw_writer.item_status(
                    target_id, frame_id
                ) != api.ItemStatus.COMPLETE:
                    raw_write_buffer.add(
                        star_id=target_id,
                        frame_id=frame_id,
                        stamp=_to_numpy(products.final_stamp.array),
                        seed=spec.rng.run_seed,
                    )
                if plan.workload.save_electron_components:
                    _save_electron_components(
                        target_dir
                        / "electron_components"
                        / f"frame_{frame_id:06d}.npz",
                        products,
                    )
            if (
                plan.workload.save_coadd
                and plan.workload.artifact_profile == "detailed"
            ):
                coadd_schema = coadd_products.to_schema_dict()
                coadd_schema["source_input_truth"] = dict(source_input_truth)
                coadd_schema["coadd_shard"] = dict(coadd_shard)
                api.write_stamp_product_schema(
                    target_dir
                    / "schemas"
                    / "coadd"
                    / f"coadd_{coadd_index:06d}.json",
                    coadd_schema,
                )
            if coadd_writer is not None and coadd_writer.item_status(
                target_id, coadd_index
            ) != api.ItemStatus.COMPLETE:
                coadd_write_buffer.add(
                    star_id=target_id,
                    frame_id=coadd_index,
                    stamp=coadd_array,
                    seed=spec.rng.run_seed,
                )
        if raw_writer is not None:
            raw_write_buffer.flush()
            raw_writer.finalize()
            raw_writer = None
        if coadd_writer is not None:
            coadd_write_buffer.flush()
            coadd_writer.finalize()
            coadd_writer = None
        truth_identity = _write_source_variability_truth(
            _source_variability_truth_path(plan, target_id),
            accumulator=truth_accumulator,
            source_input_truth=source_input_truth,
        )
        selected_field_ids = np.asarray(
            services.psf_result.psf_field_ids, dtype=np.int64
        )
        if psf_id is not None and (
            selected_field_ids.size != 1 or int(selected_field_ids[0]) != int(psf_id)
        ):
            raise RuntimeError(
                f"resolved PSF ID {psf_id} disagrees with runtime selection "
                f"{selected_field_ids.tolist()}"
            )
        from et_mainsim.manifest import _atomic_write_json

        target_artifacts = {
            "schema_id": "et_mainsim.stamp_target_artifacts",
            "schema_version": 1,
            "target_source_id": target_id,
            "source_input_truth": dict(source_input_truth),
            "source_variability_truth_identity": truth_identity,
            "artifact_policy": _artifact_policy(plan),
            "coadd_shard": dict(coadd_shard),
            "runtime_psf": {
                "selected_field_ids": selected_field_ids.tolist(),
                "provenance": dict(services.psf_result.provenance),
            },
        }
        _atomic_write_json(
            _target_artifact_manifest_path(plan, target_id),
            target_artifacts,
        )
    finally:
        if raw_writer is not None:
            raw_writer.close()
        if coadd_writer is not None:
            coadd_writer.close()
    if not target_is_complete(plan, target_id, api=api):
        raise RuntimeError(
            f"stamp artifacts for target {target_id} failed readback validation"
        )
    return {
        "target_id": target_id,
        "status": "rendered",
        "artifacts": target_artifacts,
    }


def _worker_inputs(request: StampWorkerRequest, api: Any) -> PreparedStampInputs:
    plan = request.plan
    if plan.workload.input_mode == "table":
        return _prepare_table_inputs(
            plan,
            api,
            requested_target_ids=request.target_ids,
            apply_workload_selection=False,
            expected_identities=request.input_identities,
        )
    catalog = api.StarCatalogCache.read(plan.catalog_cache)
    return PreparedStampInputs(
        target_ids=request.target_ids,
        catalogs={target_id: catalog for target_id in request.target_ids},
        psf_ids={},
        targets={},
        source_variability={target_id: None for target_id in request.target_ids},
        source_input_truth={target_id: {} for target_id in request.target_ids},
        shared_catalog=catalog,
        provenance={"catalog_cache": str(plan.catalog_cache)},
        input_identities={},
    )


def run_stamp_worker(
    request: StampWorkerRequest,
    *,
    science_api: Any | None = None,
) -> list[dict[str, Any]]:
    api = _science_api() if science_api is None else science_api
    prepared = _worker_inputs(request, api)
    assigned = prepared.target_ids[request.rank :: request.world_size]
    results = [
        _render_target(
            request.plan,
            target_id=target_id,
            catalog=prepared.catalogs[target_id],
            psf_id=prepared.psf_ids.get(target_id),
            target=prepared.targets.get(target_id),
            source_variability=prepared.source_variability.get(target_id),
            source_input_truth=prepared.source_input_truth.get(target_id, {}),
            api=api,
            worker_rank=request.rank,
        )
        for target_id in assigned
    ]
    from et_mainsim.manifest import _atomic_write_json

    _atomic_write_json(
        request.plan.run_dir / f"stamp_worker_{request.rank:02d}_done.json",
        {
            "schema_id": "et_mainsim.stamp_worker_result",
            "schema_version": 1,
            "rank": request.rank,
            "world_size": request.world_size,
            "pid": os.getpid(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "targets": results,
        },
    )
    return results


def run_stamp_worker_request_file(path: Path | str) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        request = StampWorkerRequest.from_json_dict(json.load(handle))
    return run_stamp_worker(request)


def _launch_subprocess_workers(
    plan: StampRunPlan,
    target_ids: tuple[int, ...],
    *,
    input_identities: Mapping[str, Any],
) -> list[dict[str, Any]]:
    from et_mainsim.manifest import _atomic_write_json

    assignments = worker_assignments(plan.run_config.execution)
    request_dir = plan.run_dir / "worker_requests"
    log_dir = plan.run_dir / "logs"
    request_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    processes = []
    for assignment in assignments:
        request = StampWorkerRequest.from_plan(
            plan,
            target_ids=target_ids,
            rank=assignment.rank,
            world_size=assignment.world_size,
            input_identities=input_identities,
        )
        request_path = request_dir / f"stamp_worker_{assignment.rank:02d}.json"
        _atomic_write_json(request_path, request.to_json_dict())
        log_path = log_dir / f"stamp_worker_{assignment.rank:02d}.log"
        log_handle = log_path.open("w", encoding="utf-8")
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        source_root = str(Path(__file__).resolve().parents[2])
        existing_pythonpath = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = (
            source_root
            if not existing_pythonpath
            else os.pathsep.join((source_root, existing_pythonpath))
        )
        if assignment.visible_device is not None:
            environment["CUDA_VISIBLE_DEVICES"] = assignment.visible_device
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "et_mainsim",
                "_worker",
                "--request",
                str(request_path),
            ],
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append((assignment, process, log_handle, log_path))
    failures = []
    for assignment, process, log_handle, log_path in processes:
        return_code = process.wait()
        log_handle.close()
        if return_code != 0:
            failures.append(
                f"rank {assignment.rank} exited {return_code}; see {log_path}"
            )
    if failures:
        raise RuntimeError("Stamp worker failures: " + "; ".join(failures))
    results = []
    for assignment in assignments:
        with (
            plan.run_dir / f"stamp_worker_{assignment.rank:02d}_done.json"
        ).open("r", encoding="utf-8") as handle:
            results.extend(json.load(handle)["targets"])
    return results


def _catalog_manifest(prepared: PreparedStampInputs, plan: StampRunPlan) -> dict[str, Any]:
    return {
        "input_mode": plan.workload.input_mode,
        "target_ids": list(prepared.target_ids),
        "n_targets": len(prepared.target_ids),
        "shared_catalog_n_sources": (
            None
            if prepared.shared_catalog is None
            else int(prepared.shared_catalog.n_sources)
        ),
        "metadata": dict(prepared.provenance),
    }


def _workload_identity(plan: StampRunPlan) -> dict[str, Any]:
    payload = plan.workload.to_dict()
    if plan.workload.input_mode != "table":
        return payload
    if plan.input_table_path is None:
        raise ValueError("table input path is required")
    payload["input_table_identity"] = file_identity(plan.input_table_path)
    if plan.variability_table_path is not None:
        payload["variability_table_identity"] = file_identity(
            plan.variability_table_path
        )
    target_table = load_stamp_target_table(
        plan.input_table_path,
        detector_shape=tuple(plan.spec.detector.shape),
        detector_id=str(plan.spec.detector.detector_id),
        focalplane_registry=plan.paths.focalplane_registry,
    )
    registry_identity = target_table.provenance.get(
        "focalplane_registry_identity"
    )
    if registry_identity is not None:
        payload["focalplane_registry_identity"] = registry_identity
    payload["psf_bundle_identity"] = _psf_bundle_asset_identity(plan)
    return payload


def _upgrade_stamp_artifact_manifest(
    store: RunManifestStore,
    *,
    workload: Mapping[str, Any],
    artifact_policy: Mapping[str, Any],
    frame_plan: Mapping[str, Any],
    coadd_shard: Mapping[str, Any],
) -> None:
    payload = store.load()
    stored_workload = dict(payload.get("workload", {}))
    normalized_workload = dict(stored_workload)
    normalized_workload.setdefault("artifact_profile", "detailed")
    normalized_workload.setdefault("write_batch_size", 32)
    normalized_workload.setdefault("coadd_shard_index", 0)
    normalized_workload.setdefault("coadd_shard_count", 1)
    if normalized_workload != dict(workload):
        return
    artifacts = dict(payload.get("artifacts", {}))
    changed = stored_workload != normalized_workload
    if "artifact_policy" not in artifacts:
        artifacts["artifact_policy"] = dict(artifact_policy)
        changed = True
    if "coadd_shard" not in artifacts:
        artifacts["coadd_shard"] = dict(coadd_shard)
        changed = True
    provenance = dict(payload.get("provenance", {}))
    if "coadd_shard" not in provenance:
        provenance["coadd_shard"] = dict(coadd_shard)
        changed = True
    if payload.get("frame_plan") != dict(frame_plan):
        payload["frame_plan"] = dict(frame_plan)
        changed = True
    if not changed:
        return
    payload["workload"] = normalized_workload
    payload["artifacts"] = artifacts
    payload["provenance"] = provenance
    from et_mainsim.manifest import _atomic_write_json

    _atomic_write_json(store.path, payload)


def run_stamp(
    plan: StampRunPlan,
    *,
    science_api: Any | None = None,
) -> dict[str, Any]:
    preflight(plan)
    api = _science_api() if science_api is None else science_api
    store = RunManifestStore(plan.run_dir / "run_manifest.json")
    if (
        plan.run_dir.exists()
        and not store.path.exists()
        and any(plan.run_dir.iterdir())
    ):
        raise FileExistsError(
            f"Existing nonempty run directory {plan.run_dir} does not contain "
            "run_manifest.json; use a new run id"
        )
    plan.run_dir.mkdir(parents=True, exist_ok=True)
    execution_payload = {
        **plan.run_config.execution.to_dict(),
        "paths": plan.paths.to_dict(),
    }
    workload_payload = _workload_identity(plan)
    spec_payload = plan.spec.to_json_dict()
    frame_plan = _frame_plan(plan.spec, plan.workload)
    coadd_shard = _coadd_shard_provenance(plan)
    if store.path.exists():
        _upgrade_stamp_artifact_manifest(
            store,
            workload=workload_payload,
            artifact_policy=_artifact_policy(plan),
            frame_plan=frame_plan,
            coadd_shard=coadd_shard,
        )
        store.ensure_identity(
            workflow="et-stamp",
            run_id=plan.run_config.run_id,
            simulation_spec=spec_payload,
            execution=execution_payload,
            workload=workload_payload,
        )
    else:
        store.create(
            workflow="et-stamp",
            preset=plan.preset_name,
            run_id=plan.run_config.run_id,
            simulation_spec=spec_payload,
            execution=execution_payload,
            workload=workload_payload,
            frame_plan=frame_plan,
            provenance={
                **collect_provenance(plan.repo_root),
                "coadd_shard": coadd_shard,
            },
            artifacts={
                "run_manifest": str(store.path),
                "stamp_root": str(plan.run_dir / "stamps"),
                "raw_shard_name": "raw.h5",
                "coadd_shard_name": "coadd.h5",
                "schema_root_name": (
                    "schemas"
                    if plan.workload.artifact_profile == "detailed"
                    else None
                ),
                "source_variability_truth_name": "source_variability_truth.ecsv",
                "target_artifact_manifest_name": "target_artifacts.json",
                "artifact_policy": _artifact_policy(plan),
                "coadd_shard": coadd_shard,
            },
        )
    try:
        store.start_attempt(
            control={
                "resume": plan.run_config.execution.resume,
                "overwrite": plan.run_config.execution.overwrite,
                "force_catalog_cache": (
                    plan.run_config.execution.force_catalog_cache
                ),
            }
        )
        prepared = prepare_stamp_inputs(plan, science_api=api)
        planned_input_identities: dict[str, Any] = {}
        if plan.workload.input_mode == "table":
            planned_input_identities["target_table"] = workload_payload[
                "input_table_identity"
            ]
            if "variability_table_identity" in workload_payload:
                planned_input_identities["variability_table"] = workload_payload[
                    "variability_table_identity"
                ]
            if "focalplane_registry_identity" in workload_payload:
                planned_input_identities["focalplane_registry"] = workload_payload[
                    "focalplane_registry_identity"
                ]
            planned_input_identities["psf_bundle"] = workload_payload[
                "psf_bundle_identity"
            ]
            _validate_input_identities(
                prepared.input_identities,
                planned_input_identities,
            )
        store.update(catalog=_catalog_manifest(prepared, plan))
        if plan.run_config.execution.backend == "in-process":
            results = [
                _render_target(
                    plan,
                    target_id=target_id,
                    catalog=prepared.catalogs[target_id],
                    psf_id=prepared.psf_ids.get(target_id),
                    target=prepared.targets.get(target_id),
                    source_variability=prepared.source_variability.get(target_id),
                    source_input_truth=prepared.source_input_truth.get(
                        target_id, {}
                    ),
                    api=api,
                )
                for target_id in prepared.target_ids
            ]
        else:
            results = _launch_subprocess_workers(
                plan,
                prepared.target_ids,
                input_identities=prepared.input_identities,
            )
        rendered = sum(result["status"] == "rendered" for result in results)
        skipped = sum(result["status"] == "skipped" for result in results)
        return store.transition(
            "completed",
            completion={
                "requested_targets": len(results),
                "completed_targets": len(results),
                "rendered_targets": rendered,
                "skipped_targets": skipped,
                "targets": results,
            },
        )
    except BaseException as error:
        if store.load()["status"] == "running":
            store.fail(error)
        raise


__all__ = [
    "PreparedStampInputs",
    "StampRunPlan",
    "StampWorkerRequest",
    "build_run_plan",
    "preflight",
    "prepare_stamp_inputs",
    "run_stamp",
    "run_stamp_worker",
    "run_stamp_worker_request_file",
    "target_is_complete",
]
