from __future__ import annotations

import hashlib
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


_TARGET_ARTIFACT_SCHEMA_VERSION = 2
_SELECTION_ARTIFACT_SCHEMA_ID = (
    "et_mainsim.stamp_selection_truth_artifacts.v1"
)
_SELECTION_INDEX_SCHEMA_ID = "et_mainsim.selection_truth_index.v1"
_SELECTION_TRUTH_SCOPE = (
    "geometry_psf_and_jitter_selection_truth_only"
)
_ET_STAMP_SPACECRAFT_ID = "et"
_ET_STAMP_ABSOLUTE_RAW_FRAME_START_INDEX = 0


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
            "frame_plan": _frame_plan(self.spec),
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
    from photsim7.selection_artifacts import (
        cadence_selection_truth_relative_path,
        read_cadence_selection_truth,
    )

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
        cadence_selection_truth_relative_path=(
            cadence_selection_truth_relative_path
        ),
        read_cadence_selection_truth=read_cadence_selection_truth,
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
    target_epoch_jyear = float(spec.catalog.target_epoch_jyear)
    if not np.isclose(target_epoch_jyear, 2000.0, rtol=0.0, atol=1.0e-12):
        raise ValueError(
            "stamp table mode requires canonical ICRS/J2000 epoch 2000.0; "
            f"found {target_epoch_jyear!r}"
        )
    reference_options = {
        "reference_field_angle_deg": 12.0,
        "reference_field_polar_angle_rad": float(np.pi / 4.0),
        "reference_pixel_scale_arcsec_per_pix": float(
            spec.detector.pixel_scale.to_value("arcsec / pix")
        ),
        "reference_x_axis_sign": 1.0,
        "reference_y_axis_sign": 1.0,
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
    run_dir = paths.output_root / run_config.run_id
    catalog_cache = paths.catalog_cache or run_dir / "cache" / "stars.npz"
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


def _frame_plan(spec: Any) -> dict[str, Any]:
    raw_count = int(spec.observation.resolved_n_frames)
    per_coadd = int(spec.observation.n_raw_frames_per_coadd)
    if raw_count % per_coadd:
        raise ValueError("stamp raw frame count must be divisible by coadd size")
    return {
        "raw_frame_count": raw_count,
        "raw_frame_indices": list(range(raw_count)),
        "n_raw_frames_per_coadd": per_coadd,
        "coadd_count": raw_count // per_coadd,
        "coadd_indices": list(range(raw_count // per_coadd)),
    }


def preflight(plan: StampRunPlan) -> None:
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


def _target_reference_field_options(
    plan: StampRunPlan,
    target: StampTarget,
) -> dict[str, float]:
    if target.location_mode != "explicit_psf":
        raise ValueError(
            "reference-field options are only valid for explicit-PSF targets"
        )
    if target.psf_node_angle_deg is None:
        raise ValueError(
            f"explicit-PSF target {target.source_id} is missing its PSF node angle"
        )
    return {
        "reference_field_angle_deg": float(target.psf_node_angle_deg),
        "reference_field_polar_angle_rad": float(np.pi / 4.0),
        "reference_pixel_scale_arcsec_per_pix": float(
            plan.spec.detector.pixel_scale.to_value("arcsec / pix")
        ),
        "reference_x_axis_sign": 1.0,
        "reference_y_axis_sign": 1.0,
    }


def _table_catalog(
    plan: StampRunPlan,
    target: StampTarget,
    api: Any,
    *,
    source_input_truth: Mapping[str, Any] | None = None,
) -> Any:
    from photsim7.geometry_truth import (
        physical_et_focalplane_declaration,
        reference_field_nonphysical_declaration,
    )

    rows, cols = (int(value) for value in plan.spec.detector.shape)
    center_x = (cols - 1) / 2.0
    center_y = (rows - 1) / 2.0
    star_data: dict[str, Any] = {
        "x0": np.asarray([0.0], dtype=np.float64),
        "y0": np.asarray([0.0], dtype=np.float64),
        "frame_xpix": np.asarray([center_x], dtype=np.float64),
        "frame_ypix": np.asarray([center_y], dtype=np.float64),
        # The legacy star-table carrier requires RA/Dec even for explicitly
        # non-physical reference-field rows.  Only the versioned geometry
        # declaration below is authoritative for projector selection.
        "ra": np.asarray(
            [
                target.ra_deg
                if target.ra_deg is not None
                else plan.spec.catalog.target_ra_deg or 0.0
            ],
            dtype=np.float64,
        ),
        "dec": np.asarray(
            [
                target.dec_deg
                if target.dec_deg is not None
                else plan.spec.catalog.target_dec_deg or 0.0
            ],
            dtype=np.float64,
        ),
        "source_id": np.asarray([target.source_id], dtype=np.int64),
        "gaia_g_mag": np.asarray([target.gaia_g_mag], dtype=np.float64),
        "detector_xpix": np.asarray([target.detector_xpix], dtype=np.float64),
        "detector_ypix": np.asarray([target.detector_ypix], dtype=np.float64),
        "detector_xpix_shifted": np.asarray([center_x], dtype=np.float64),
        "detector_ypix_shifted": np.asarray([center_y], dtype=np.float64),
        "detector_id": str(plan.spec.detector.detector_id),
    }
    if target.location_mode == "sky_icrs_j2000":
        if target.ra_deg is None or target.dec_deg is None:
            raise ValueError(
                f"coordinate target {target.source_id} is missing canonical ICRS coordinates"
            )
        registry_identity = (
            None
            if source_input_truth is None
            else source_input_truth.get("focalplane_registry_identity")
        )
        if plan.paths.focalplane_registry is None or registry_identity is None:
            raise ValueError(
                f"coordinate target {target.source_id} is missing frozen focal-plane registry identity"
            )
        coordinate_epoch_jyear = float(plan.spec.catalog.target_epoch_jyear)
        star_data.update(
            {
                "ra": np.asarray([target.ra_deg], dtype=np.float64),
                "dec": np.asarray([target.dec_deg], dtype=np.float64),
                "icrs_ra_deg": np.asarray([target.ra_deg], dtype=np.float64),
                "icrs_dec_deg": np.asarray([target.dec_deg], dtype=np.float64),
                "target_epoch": np.asarray(
                    [coordinate_epoch_jyear],
                    dtype=np.float64,
                ),
            }
        )
        geometry_declaration = physical_et_focalplane_declaration(
            coordinate_frame="icrs",
            coordinate_epoch_jyear=coordinate_epoch_jyear,
            registry_data_dir=plan.paths.focalplane_registry,
            focalplane_registry_identity=registry_identity,
        )
    else:
        reference_options = _target_reference_field_options(plan, target)
        geometry_declaration = reference_field_nonphysical_declaration(
            **reference_options
        )
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
            "geometry": geometry_declaration,
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
        expected_sha256=plan.spec.psf.bundle_sha256,
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
        "expected_sha256": plan.spec.psf.bundle_sha256,
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
        "coordinate_epoch_jyear": (
            2000.0 if target.location_mode == "sky_icrs_j2000" else None
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
        "schema_id": "et_mainsim.stamp_source_input_truth.v2",
        "schema_version": 2,
        "source_id": target.source_id,
        "gaia_g_mag": target.gaia_g_mag,
        "magnitude_system": "Gaia_G_Vega",
        "target_table_identity": target_provenance["file_identity"],
        "target_table_meta": target_provenance.get("table_meta", {}),
        "focalplane_registry_identity": (
            target_provenance.get("focalplane_registry_identity")
            if target.location_mode == "sky_icrs_j2000"
            else None
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


def _selection_index_record(
    *,
    local_frame_index: int,
    absolute_raw_frame_index: int,
    content_sha256: str,
) -> bytes:
    payload = {
        "absolute_raw_frame_index": int(absolute_raw_frame_index),
        "content_sha256": str(content_sha256),
        "local_frame_index": int(local_frame_index),
    }
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _selection_identity_payload(identity: Any, *, target_dir: Path) -> dict[str, Any]:
    relative = Path(str(identity.relative_path))
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError("selection artifact path must stay below target_dir")
    expected = (target_dir / relative).resolve(strict=False)
    if Path(identity.path).resolve(strict=False) != expected:
        raise RuntimeError(
            "selection artifact identity is not rooted at the target directory"
        )
    content_sha256 = str(identity.content_sha256)
    if len(content_sha256) != 64 or any(
        value not in "0123456789abcdef" for value in content_sha256
    ):
        raise RuntimeError("selection artifact content_sha256 is invalid")
    return {
        "schema_id": str(identity.schema_id),
        "schema_version": int(identity.schema_version),
        "relative_path": relative.as_posix(),
        "content_sha256": content_sha256,
    }


class _SelectionTruthAccumulator:
    def __init__(
        self,
        *,
        target_dir: Path,
        frame_indices: tuple[int, ...],
        spacecraft_id: str,
        science_realization_id: int,
        requested_science_profile_id: str,
        absolute_raw_frame_start_index: int,
    ) -> None:
        if not frame_indices:
            raise ValueError("selection truth requires at least one raw frame")
        self.target_dir = Path(target_dir)
        self.frame_indices = tuple(int(value) for value in frame_indices)
        self.expected_spacecraft_id = str(spacecraft_id)
        self.expected_science_realization_id = int(science_realization_id)
        self.expected_requested_science_profile_id = str(
            requested_science_profile_id
        )
        if not self.expected_requested_science_profile_id:
            raise ValueError("requested science profile ID must not be empty")
        self.expected_absolute_raw_frame_start_index = int(
            absolute_raw_frame_start_index
        )
        self.count = 0
        self.absolute_raw_frame_start_index: int | None = None
        self.geometry: dict[str, Any] | None = None
        self.psf: dict[str, Any] | None = None
        self.cadence_schema_id: str | None = None
        self.cadence_schema_version: int | None = None
        self.science_conformance_claim: bool | None = None
        self.unavailable_marker: dict[str, Any] | None = None
        self.digest = hashlib.sha256()

    def add(self, raw_result: Any) -> None:
        if self.count >= len(self.frame_indices):
            raise RuntimeError("received more selection sidecars than raw frames")
        expected_local = self.frame_indices[self.count]
        product_frame_index = int(
            raw_result.stamp_products.frame_index
        )
        if product_frame_index != expected_local:
            raise RuntimeError(
                "selection results are not ordered by the raw frame plan"
            )
        truth = getattr(raw_result, "selection_truth", None)
        artifacts = getattr(raw_result, "selection_artifacts", None)
        if truth is None or artifacts is None:
            if truth is not None or artifacts is not None:
                raise RuntimeError(
                    "selection truth and durable artifacts must be present "
                    "together"
                )
            marker = getattr(
                getattr(raw_result, "stamp_products", None),
                "selection_truth",
                None,
            )
            if (
                not isinstance(marker, Mapping)
                or marker.get("verification_status") != "unavailable"
                or marker.get("science_conformance_claim") is not False
            ):
                raise RuntimeError(
                    "missing selection truth must carry an explicit "
                    "non-conformant unavailable marker"
                )
            if self.geometry is not None:
                raise RuntimeError(
                    "selection truth availability changed by cadence"
                )
            normalized = {
                "missing_components": sorted(
                    str(value)
                    for value in marker.get("missing_components", ())
                ),
                "requested_science_profile_id": str(
                    marker.get("requested_science_profile_id", "")
                ),
                "science_conformance_claim_scope": str(
                    marker.get("science_conformance_claim_scope", "")
                ),
            }
            if self.unavailable_marker is None:
                if (
                    normalized["requested_science_profile_id"]
                    != self.expected_requested_science_profile_id
                    or normalized["science_conformance_claim_scope"]
                    != _SELECTION_TRUTH_SCOPE
                ):
                    raise RuntimeError(
                        "unavailable selection marker conflicts with plan"
                    )
                self.unavailable_marker = normalized
            elif self.unavailable_marker != normalized:
                raise RuntimeError(
                    "unavailable selection marker changed by cadence"
                )
            self.count += 1
            return
        if self.unavailable_marker is not None:
            raise RuntimeError("selection truth availability changed by cadence")
        local_frame_index = int(truth.local_frame_index)
        if local_frame_index != expected_local:
            raise RuntimeError(
                "selection sidecars are not ordered by the raw frame plan"
            )
        absolute_raw_frame_index = int(truth.absolute_raw_frame_index)
        raw_start = absolute_raw_frame_index - local_frame_index
        if str(truth.spacecraft_id) != self.expected_spacecraft_id:
            raise RuntimeError("selection truth spacecraft conflicts with plan")
        if (
            int(truth.science_realization_id)
            != self.expected_science_realization_id
        ):
            raise RuntimeError(
                "selection truth science realization conflicts with plan"
            )
        if raw_start != self.expected_absolute_raw_frame_start_index:
            raise RuntimeError(
                "selection truth absolute raw-frame origin conflicts with plan"
            )
        if self.absolute_raw_frame_start_index is None:
            self.absolute_raw_frame_start_index = raw_start
        elif self.absolute_raw_frame_start_index != raw_start:
            raise RuntimeError(
                "selection sidecars use inconsistent absolute frame origins"
            )

        geometry = _selection_identity_payload(
            artifacts.geometry,
            target_dir=self.target_dir,
        )
        psf = _selection_identity_payload(
            artifacts.psf,
            target_dir=self.target_dir,
        )
        cadence = _selection_identity_payload(
            artifacts.cadence,
            target_dir=self.target_dir,
        )
        if geometry["content_sha256"] != (
            truth.source_geometry_truth.content_sha256
        ):
            raise RuntimeError("geometry sidecar identity conflicts with truth")
        if psf["content_sha256"] != truth.psf_selection_truth.content_sha256:
            raise RuntimeError("PSF sidecar identity conflicts with truth")
        if cadence["content_sha256"] != truth.content_sha256:
            raise RuntimeError("cadence sidecar identity conflicts with truth")
        if self.geometry is None:
            self.geometry = geometry
            self.psf = psf
            self.cadence_schema_id = cadence["schema_id"]
            self.cadence_schema_version = cadence["schema_version"]
            self.science_conformance_claim = bool(
                truth.science_conformance_claim
            )
        elif self.geometry != geometry or self.psf != psf:
            raise RuntimeError(
                "static geometry or PSF selection identity changed by cadence"
            )
        elif (
            self.cadence_schema_id != cadence["schema_id"]
            or self.cadence_schema_version != cadence["schema_version"]
            or self.science_conformance_claim
            is not bool(truth.science_conformance_claim)
        ):
            raise RuntimeError("cadence selection contract changed by cadence")

        expected_relative = (
            Path("selection_truth")
            / "cadence"
            / f"frame_{absolute_raw_frame_index:09d}.json"
        ).as_posix()
        if cadence["relative_path"] != expected_relative:
            raise RuntimeError("cadence sidecar path conflicts with frame identity")
        self.digest.update(
            _selection_index_record(
                local_frame_index=local_frame_index,
                absolute_raw_frame_index=absolute_raw_frame_index,
                content_sha256=truth.content_sha256,
            )
        )
        self.count += 1

    def to_manifest(self) -> dict[str, Any]:
        if self.count != len(self.frame_indices):
            raise RuntimeError("selection truth is missing raw frames")
        if self.unavailable_marker is not None:
            return {
                "schema_id": _SELECTION_ARTIFACT_SCHEMA_ID,
                "schema_version": 1,
                "verification_status": "unavailable",
                "science_conformance_claim": False,
                "artifact_root": None,
                "cadence_count": self.count,
                **dict(self.unavailable_marker),
            }
        if (
            self.geometry is None
            or self.psf is None
            or self.absolute_raw_frame_start_index is None
            or self.cadence_schema_id is None
            or self.cadence_schema_version is None
            or self.science_conformance_claim is None
        ):
            raise RuntimeError("selection truth accumulator is incomplete")
        return {
            "schema_id": _SELECTION_ARTIFACT_SCHEMA_ID,
            "schema_version": 1,
            "verification_status": "persisted_and_verified",
            "science_conformance_claim": self.science_conformance_claim,
            "science_conformance_claim_scope": _SELECTION_TRUTH_SCOPE,
            "requested_science_profile_id": (
                self.expected_requested_science_profile_id
            ),
            "missing_components": [],
            "artifact_root": ".",
            "source_geometry_truth": dict(self.geometry),
            "psf_selection_truth": dict(self.psf),
            "cadence_selection_truth": {
                "schema_id": self.cadence_schema_id,
                "schema_version": self.cadence_schema_version,
                "relative_directory": "selection_truth/cadence",
                "filename_template": (
                    "frame_{absolute_raw_frame_index:09d}.json"
                ),
                "count": self.count,
                "absolute_raw_frame_start_index": (
                    self.absolute_raw_frame_start_index
                ),
                "spacecraft_id": self.expected_spacecraft_id,
                "science_realization_id": (
                    self.expected_science_realization_id
                ),
                "index_digest_schema_id": _SELECTION_INDEX_SCHEMA_ID,
                "index_content_sha256": self.digest.hexdigest(),
            },
        }


def _selection_identity_from_manifest(
    value: Any,
    *,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} identity must be a mapping")
    result = dict(value)
    expected_keys = {
        "schema_id",
        "schema_version",
        "relative_path",
        "content_sha256",
    }
    if set(result) != expected_keys:
        raise ValueError(f"{label} identity keys are invalid")
    relative = Path(str(result["relative_path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} path must stay below target_dir")
    digest = str(result["content_sha256"])
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{label} content_sha256 is invalid")
    result["relative_path"] = relative.as_posix()
    result["content_sha256"] = digest
    result["schema_version"] = int(result["schema_version"])
    return result


def _validate_selection_sidecars(
    plan: StampRunPlan,
    target_id: int,
    payload: Mapping[str, Any],
    *,
    api: Any,
) -> bool:
    selection = payload.get("selection_truth")
    if not isinstance(selection, Mapping):
        return False
    if selection.get("schema_id") != _SELECTION_ARTIFACT_SCHEMA_ID:
        return False
    if int(selection.get("schema_version", 0)) != 1:
        return False
    raw_ids, _ = _expected_ids(plan)
    verification_status = selection.get("verification_status")
    if verification_status == "unavailable":
        if set(selection) != {
            "schema_id",
            "schema_version",
            "verification_status",
            "science_conformance_claim",
            "science_conformance_claim_scope",
            "requested_science_profile_id",
            "missing_components",
            "artifact_root",
            "cadence_count",
        }:
            return False
        return bool(
            selection.get("science_conformance_claim") is False
            and selection.get("artifact_root") is None
            and int(selection.get("cadence_count", -1)) == len(raw_ids)
            and selection.get("missing_components")
            == ["jitter_model_selection_truth"]
            and not bool(plan.spec.psf.use_jitter_integrated_psf)
            and selection.get("requested_science_profile_id")
            == plan.spec.science_profile.profile_id
            and selection.get("science_conformance_claim_scope")
            == _SELECTION_TRUTH_SCOPE
        )
    if verification_status != "persisted_and_verified":
        return False
    if set(selection) != {
        "schema_id",
        "schema_version",
        "verification_status",
        "science_conformance_claim",
        "science_conformance_claim_scope",
        "requested_science_profile_id",
        "missing_components",
        "artifact_root",
        "source_geometry_truth",
        "psf_selection_truth",
        "cadence_selection_truth",
    }:
        return False
    if not isinstance(selection.get("science_conformance_claim"), bool):
        return False
    if (
        selection.get("science_conformance_claim_scope")
        != _SELECTION_TRUTH_SCOPE
        or selection.get("requested_science_profile_id")
        != plan.spec.science_profile.profile_id
    ):
        return False
    if selection.get("missing_components") != []:
        return False
    if selection.get("artifact_root") != ".":
        return False
    geometry = _selection_identity_from_manifest(
        selection["source_geometry_truth"],
        label="source geometry truth",
    )
    psf = _selection_identity_from_manifest(
        selection["psf_selection_truth"],
        label="PSF selection truth",
    )
    cadence = selection["cadence_selection_truth"]
    if not isinstance(cadence, Mapping):
        return False
    if set(cadence) != {
        "schema_id",
        "schema_version",
        "relative_directory",
        "filename_template",
        "count",
        "absolute_raw_frame_start_index",
        "spacecraft_id",
        "science_realization_id",
        "index_digest_schema_id",
        "index_content_sha256",
    }:
        return False
    if cadence.get("schema_id") != "photsim7.cadence_selection_truth.v1":
        return False
    if int(cadence.get("schema_version", 0)) != 1:
        return False
    if cadence.get("relative_directory") != "selection_truth/cadence":
        return False
    if cadence.get("filename_template") != (
        "frame_{absolute_raw_frame_index:09d}.json"
    ):
        return False
    if cadence.get("index_digest_schema_id") != _SELECTION_INDEX_SCHEMA_ID:
        return False
    if int(cadence.get("count", -1)) != len(raw_ids):
        return False
    raw_start = int(cadence["absolute_raw_frame_start_index"])
    if raw_start != _ET_STAMP_ABSOLUTE_RAW_FRAME_START_INDEX:
        return False
    spacecraft_id = str(cadence["spacecraft_id"])
    if spacecraft_id != _ET_STAMP_SPACECRAFT_ID:
        return False
    science_realization_id = int(cadence["science_realization_id"])
    if science_realization_id != (
        plan.spec.science_profile.science_realization_id
    ):
        return False
    expected_digest = str(cadence["index_content_sha256"])
    if len(expected_digest) != 64 or any(
        character not in "0123456789abcdef"
        for character in expected_digest
    ):
        return False

    target_dir = _target_dir(plan, target_id)
    digest = hashlib.sha256()
    expected_seed_tree = plan.spec.rng.to_seed_tree()
    for local_frame_index in raw_ids:
        absolute_raw_frame_index = raw_start + local_frame_index
        relative = api.cadence_selection_truth_relative_path(
            absolute_raw_frame_index
        )
        truth = api.read_cadence_selection_truth(
            target_dir / relative,
            artifact_root=target_dir,
        )
        if truth.local_frame_index != local_frame_index:
            return False
        if truth.absolute_raw_frame_index != absolute_raw_frame_index:
            return False
        if truth.detector_id != str(plan.spec.detector.detector_id):
            return False
        if truth.spacecraft_id != spacecraft_id:
            return False
        if truth.science_realization_id != science_realization_id:
            return False
        try:
            truth.jitter_model_selection_truth.rng_trace_payload(
                expected_seed_tree
            )
        except (AttributeError, TypeError, ValueError):
            return False
        if (
            truth.science_conformance_claim
            is not selection["science_conformance_claim"]
        ):
            return False
        geometry_reference = truth.geometry_reference
        psf_reference = truth.psf_reference
        for field_name in (
            "schema_id",
            "schema_version",
            "relative_path",
            "content_sha256",
        ):
            if geometry[field_name] != geometry_reference[field_name]:
                return False
            if psf[field_name] != psf_reference[field_name]:
                return False
        digest.update(
            _selection_index_record(
                local_frame_index=local_frame_index,
                absolute_raw_frame_index=absolute_raw_frame_index,
                content_sha256=truth.content_sha256,
            )
        )
    return digest.hexdigest() == expected_digest


def _read_target_artifacts(
    plan: StampRunPlan,
    target_id: int,
    *,
    api: Any,
) -> dict[str, Any] | None:
    path = _target_artifact_manifest_path(plan, target_id)
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("schema_id") != "et_mainsim.stamp_target_artifacts":
            return None
        if int(payload.get("schema_version", 0)) != (
            _TARGET_ARTIFACT_SCHEMA_VERSION
        ):
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
        if len(table) != int(plan.spec.observation.resolved_n_frames):
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
        if tuple(int(value) for value in table["frame_index"]) != tuple(
            range(int(plan.spec.observation.resolved_n_frames))
        ):
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
        if not _validate_selection_sidecars(
            plan,
            target_id,
            payload,
            api=api,
        ):
            return None
        return payload
    except (KeyError, OSError, TypeError, ValueError):
        return None


def _expected_ids(plan: StampRunPlan) -> tuple[tuple[int, ...], tuple[int, ...]]:
    frame_plan = _frame_plan(plan.spec)
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
    checks.append(
        _read_target_artifacts(plan, target_id, api=api) is not None
    )
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


def _variability_truth_row(
    products: Any,
    *,
    target_id: int,
    curve_id: str | None,
) -> dict[str, Any]:
    truth = getattr(products, "truth", None)
    payload = None if truth is None else getattr(truth, "payload", None)
    if not isinstance(payload, Mapping):
        raise RuntimeError("stamp products must expose the numeric truth payload")
    return {
        "frame_index": int(products.frame_index),
        "source_id": int(target_id),
        "curve_id": "" if curve_id is None else str(curve_id),
        "relative_flux": float(
            _truth_value_for_target(
                payload,
                "source_relative_flux_factor",
                target_id=target_id,
            )
        ),
        "baseline_photon_count_electron": float(
            _truth_value_for_target(
                payload,
                "source_baseline_photon_count_electron",
                target_id=target_id,
            )
        ),
        "effective_photon_count_electron": float(
            _truth_value_for_target(
                payload,
                "source_effective_photon_count_electron",
                target_id=target_id,
            )
        ),
        "psf_field_id": int(
            _truth_value_for_target(
                payload,
                "source_psf_field_index",
                target_id=target_id,
            )
        ),
    }


def _write_source_variability_truth(
    path: Path,
    *,
    rows: list[Mapping[str, Any]],
    target_id: int,
    source_input_truth: Mapping[str, Any],
) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: int(row["frame_index"]))
    frame_indices = tuple(int(row["frame_index"]) for row in ordered)
    if frame_indices != tuple(range(len(ordered))):
        raise RuntimeError(
            "source variability truth rows must contain every raw frame exactly once"
        )
    table = Table(
        {
            "frame_index": [int(row["frame_index"]) for row in ordered],
            "source_id": [int(row["source_id"]) for row in ordered],
            "curve_id": [str(row["curve_id"]) for row in ordered],
            "relative_flux": [float(row["relative_flux"]) for row in ordered],
            "baseline_photon_count_electron": [
                float(row["baseline_photon_count_electron"]) for row in ordered
            ],
            "effective_photon_count_electron": [
                float(row["effective_photon_count_electron"]) for row in ordered
            ],
            "psf_field_id": [int(row["psf_field_id"]) for row in ordered],
        },
        meta={
            "schema_id": "et_mainsim.source_variability_truth",
            "schema_version": 1,
            "target_source_id": int(target_id),
            "time_alignment": "simulation_raw_frame_index",
            "source_input_truth": dict(source_input_truth),
        },
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    table.write(path, format="ascii.ecsv", overwrite=True)
    return file_identity(path)


def _target_spec(
    plan: StampRunPlan,
    *,
    target: StampTarget | None,
    psf_id: int | None,
    source_input_truth: Mapping[str, Any] | None = None,
) -> Any:
    actual_bundle_sha256 = _source_input_psf_bundle_sha256(source_input_truth)
    bundle_sha256 = plan.spec.psf.bundle_sha256
    if (
        bundle_sha256 is not None
        and actual_bundle_sha256 is not None
        and actual_bundle_sha256 != bundle_sha256
    ):
        raise ValueError(
            "PSF bundle sha256 differs from the accepted SimulationSpec "
            f"identity: expected {bundle_sha256}, got {actual_bundle_sha256}"
        )
    if target is not None and target.location_mode == "sky_icrs_j2000":
        return replace(
            plan.spec,
            psf=replace(
                plan.spec.psf,
                bundle_sha256=bundle_sha256,
                field_id="nearest",
                field_id_policy="nearest",
            ),
        )
    if psf_id is None:
        return plan.spec
    if target is None:
        raise ValueError("table explicit-PSF selection requires target metadata")
    reference_options = _target_reference_field_options(plan, target)
    return replace(
        plan.spec,
        catalog=replace(
            plan.spec.catalog,
            query_options=reference_options,
        ),
        psf=replace(
            plan.spec.psf,
            bundle_sha256=bundle_sha256,
            field_id=int(psf_id),
            field_id_policy=None,
        ),
    )


def _source_input_psf_bundle_sha256(
    source_input_truth: Mapping[str, Any] | None,
) -> str | None:
    if not source_input_truth:
        return None
    psf = source_input_truth.get("psf")
    bundle = psf.get("bundle") if isinstance(psf, Mapping) else None
    identity = (
        bundle.get("file_identity")
        if isinstance(bundle, Mapping)
        else None
    )
    if identity is None:
        return None
    if not isinstance(identity, Mapping):
        raise ValueError("PSF bundle file_identity must be a mapping")
    raw_sha256 = identity.get("sha256")
    sha256 = str(raw_sha256).strip().lower()
    if len(sha256) != 64:
        raise ValueError("PSF bundle sha256 must contain 64 hexadecimal characters")
    try:
        int(sha256, 16)
    except ValueError as exc:
        raise ValueError(
            "PSF bundle sha256 must contain 64 hexadecimal characters"
        ) from exc
    return sha256


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
            artifacts = _read_target_artifacts(plan, target_id, api=api)
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

    spec = _target_spec(
        plan,
        target=target,
        psf_id=psf_id,
        source_input_truth=source_input_truth,
    )
    registry = api.DataRegistry(data_root=plan.paths.data_root)
    services = api.build_stamp_services(
        spec,
        catalog=catalog,
        data_registry=registry,
    )
    raw_ids, coadd_ids = _expected_ids(plan)
    raw_writer = None
    coadd_writer = None
    raw_path = target_dir / "raw.h5"
    coadd_path = target_dir / "coadd.h5"
    truth_rows: list[dict[str, Any]] = []
    selection_truth = _SelectionTruthAccumulator(
        target_dir=target_dir,
        frame_indices=raw_ids,
        spacecraft_id=_ET_STAMP_SPACECRAFT_ID,
        science_realization_id=(
            spec.science_profile.science_realization_id
        ),
        requested_science_profile_id=(
            spec.science_profile.profile_id
        ),
        absolute_raw_frame_start_index=(
            _ET_STAMP_ABSOLUTE_RAW_FRAME_START_INDEX
        ),
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
                run_dir=target_dir,
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
                )

            for raw_result in result.raw_results:
                selection_truth.add(raw_result)
                products = raw_result.stamp_products
                frame_id = int(products.frame_index)
                truth_rows.append(
                    _variability_truth_row(
                        products,
                        target_id=target_id,
                        curve_id=(None if target is None else target.curve_id),
                    )
                )
                if plan.workload.save_raw:
                    schema = products.to_schema_dict()
                    schema["source_input_truth"] = dict(source_input_truth)
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
                    raw_writer.write_stamp(
                        target_id,
                        frame_id,
                        _to_numpy(products.final_stamp.array),
                        seed=spec.rng.run_seed,
                    )
                if plan.workload.save_electron_components:
                    _save_electron_components(
                        target_dir
                        / "electron_components"
                        / f"frame_{frame_id:06d}.npz",
                        products,
                    )
            if plan.workload.save_coadd:
                coadd_schema = coadd_products.to_schema_dict()
                coadd_schema["source_input_truth"] = dict(source_input_truth)
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
                coadd_writer.write_stamp(
                    target_id,
                    coadd_index,
                    coadd_array,
                    seed=spec.rng.run_seed,
                )
        if raw_writer is not None:
            raw_writer.finalize()
            raw_writer = None
        if coadd_writer is not None:
            coadd_writer.finalize()
            coadd_writer = None
        truth_identity = _write_source_variability_truth(
            _source_variability_truth_path(plan, target_id),
            rows=truth_rows,
            target_id=target_id,
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
            "schema_version": _TARGET_ARTIFACT_SCHEMA_VERSION,
            "target_source_id": target_id,
            "source_input_truth": dict(source_input_truth),
            "source_variability_truth_identity": truth_identity,
            "selection_truth": selection_truth.to_manifest(),
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


def _stamp_product_contract() -> dict[str, Any]:
    return {
        "target_artifact_schema_id": "et_mainsim.stamp_target_artifacts",
        "target_artifact_schema_version": _TARGET_ARTIFACT_SCHEMA_VERSION,
        "selection_artifact_schema_id": _SELECTION_ARTIFACT_SCHEMA_ID,
        "selection_artifact_schema_version": 1,
        "selection_index_schema_id": _SELECTION_INDEX_SCHEMA_ID,
        "source_geometry_truth_schema_id": (
            "photsim7.source_geometry_truth.v1"
        ),
        "psf_selection_truth_schema_id": (
            "photsim7.psf_selection_truth.v2"
        ),
        "cadence_selection_truth_schema_id": (
            "photsim7.cadence_selection_truth.v1"
        ),
    }


def _workload_identity(plan: StampRunPlan) -> dict[str, Any]:
    payload = plan.workload.to_dict()
    payload["product_contract"] = _stamp_product_contract()
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
    if store.path.exists():
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
            frame_plan=_frame_plan(plan.spec),
            provenance=collect_provenance(plan.repo_root),
            artifacts={
                "run_manifest": str(store.path),
                "stamp_root": str(plan.run_dir / "stamps"),
                "raw_shard_name": "raw.h5",
                "coadd_shard_name": "coadd.h5",
                "schema_root_name": "schemas",
                "source_variability_truth_name": "source_variability_truth.ecsv",
                "target_artifact_manifest_name": "target_artifacts.json",
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
