from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np

from et_mainsim.config import (
    ResolvedRunPaths,
    RunConfig,
    StampWorkload,
    worker_assignments,
)
from et_mainsim.manifest import RunManifestStore
from et_mainsim.presets import resource_path
from et_mainsim.provenance import collect_provenance
from et_mainsim.stamp_inputs import StampTarget, load_stamp_target_table


@dataclass(frozen=True)
class StampRunPlan:
    preset_name: str
    run_config: RunConfig
    paths: ResolvedRunPaths
    spec: Any
    run_dir: Path
    catalog_cache: Path
    input_table_path: Path | None
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
    shared_catalog: Any | None
    provenance: Mapping[str, Any]


@dataclass(frozen=True)
class StampWorkerRequest:
    plan: StampRunPlan
    target_ids: tuple[int, ...]
    rank: int = 0
    world_size: int = 1

    def __post_init__(self) -> None:
        target_ids = tuple(int(value) for value in self.target_ids)
        rank = int(self.rank)
        world_size = int(self.world_size)
        if rank < 0 or world_size <= 0 or rank >= world_size:
            raise ValueError("rank must be smaller than positive world_size")
        object.__setattr__(self, "target_ids", target_ids)
        object.__setattr__(self, "rank", rank)
        object.__setattr__(self, "world_size", world_size)

    @classmethod
    def from_plan(
        cls,
        plan: StampRunPlan,
        *,
        target_ids: tuple[int, ...],
        rank: int = 0,
        world_size: int = 1,
    ) -> "StampWorkerRequest":
        return cls(
            plan=plan,
            target_ids=target_ids,
            rank=rank,
            world_size=world_size,
        )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "schema_id": "et_mainsim.stamp_worker_request",
            "schema_version": 1,
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
            "repo_root": str(self.plan.repo_root),
            "target_ids": list(self.target_ids),
            "rank": self.rank,
            "world_size": self.world_size,
        }

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, Any]) -> "StampWorkerRequest":
        if payload.get("schema_id") != "et_mainsim.stamp_worker_request":
            raise ValueError("Unsupported stamp worker request")
        if int(payload.get("schema_version", 0)) != 1:
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
            repo_root=Path(payload["repo_root"]),
        )
        return cls(
            plan=plan,
            target_ids=tuple(payload["target_ids"]),
            rank=int(payload["rank"]),
            world_size=int(payload["world_size"]),
        )


def _science_api() -> SimpleNamespace:
    from photsim7.artifacts import ItemStatus, StampShardReader, StampShardWriter
    from photsim7.catalog_sources import PreparedStarCatalog, StarCatalogCache
    from photsim7.data_registry import DataRegistry
    from photsim7.simulation_services import (
        build_catalog_from_spec,
        build_stamp_services,
    )
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
    if run_config.workload.input_mode == "table":
        input_table_path = _resolve_table_path(
            run_config.workload.input_table,
            cwd=base,
        )
        resolved_spec = _table_spec(resolved_spec)
    return StampRunPlan(
        preset_name=preset_name,
        run_config=run_config,
        paths=paths,
        spec=resolved_spec,
        run_dir=run_dir,
        catalog_cache=catalog_cache,
        input_table_path=input_table_path,
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
        return
    catalog = plan.spec.catalog
    if catalog.source_type == "et_focalplane_query":
        if not catalog.source_path or not Path(catalog.source_path).is_dir():
            raise FileNotFoundError(
                "GAIA_CATALOG_DIR or paths.catalog_path must reference a catalog directory"
            )
        if not catalog.registry_data_dir or not Path(
            catalog.registry_data_dir
        ).is_dir():
            raise FileNotFoundError(
                "ET_FOCALPLANE_ROOT or paths.focalplane_registry must reference focal-plane data"
            )
        focalplane_src = Path(catalog.query_options["et_focalplane_src"])
        if not focalplane_src.is_dir():
            raise FileNotFoundError(
                f"ET focal-plane source does not exist: {focalplane_src}"
            )
    elif catalog.source_type != "prepared" and not Path(catalog.source_path).is_file():
        raise FileNotFoundError(f"Catalog source does not exist: {catalog.source_path}")


def _table_catalog(plan: StampRunPlan, target: StampTarget, api: Any) -> Any:
    rows, cols = (int(value) for value in plan.spec.detector.shape)
    center_x = (cols - 1) / 2.0
    center_y = (rows - 1) / 2.0
    return api.PreparedStarCatalog(
        star_data={
            "x0": np.asarray([0.0], dtype=np.float64),
            "y0": np.asarray([0.0], dtype=np.float64),
            "frame_xpix": np.asarray([center_x], dtype=np.float64),
            "frame_ypix": np.asarray([center_y], dtype=np.float64),
            "ra": np.asarray([plan.spec.catalog.target_ra_deg or 0.0]),
            "dec": np.asarray([plan.spec.catalog.target_dec_deg or 0.0]),
            "source_id": np.asarray([target.source_id], dtype=np.int64),
            "gaia_g_mag": np.asarray([target.gaia_g_mag], dtype=np.float64),
            "detector_xpix": np.asarray([target.detector_xpix], dtype=np.float64),
            "detector_ypix": np.asarray([target.detector_ypix], dtype=np.float64),
            "detector_xpix_shifted": np.asarray([center_x], dtype=np.float64),
            "detector_ypix_shifted": np.asarray([center_y], dtype=np.float64),
            "detector_id": str(plan.spec.detector.detector_id),
        },
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
        },
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
        if plan.input_table_path is None:
            raise ValueError("table input path is required")
        loaded = load_stamp_target_table(
            plan.input_table_path,
            detector_shape=tuple(plan.spec.detector.shape),
        )
        targets = list(loaded.targets)
        if workload.target_source_ids:
            requested = set(workload.target_source_ids)
            targets = [target for target in targets if target.source_id in requested]
            missing = sorted(requested - {target.source_id for target in targets})
            if missing:
                raise ValueError(f"stamp target rows are absent: {missing}")
        if workload.target_limit:
            targets = targets[: workload.target_limit]
        if not targets:
            raise ValueError("stamp target table selected no rows")
        catalogs = {
            target.source_id: _table_catalog(plan, target, api) for target in targets
        }
        return PreparedStampInputs(
            target_ids=tuple(target.source_id for target in targets),
            catalogs=catalogs,
            psf_ids={target.source_id: target.psf_id for target in targets},
            shared_catalog=None,
            provenance=dict(loaded.provenance),
        )

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
        shared_catalog=catalog,
        provenance={
            "schema_id": "et_mainsim.stamp_catalog_selection",
            "schema_version": 1,
            "catalog_cache": str(plan.catalog_cache),
            "n_sources": int(catalog.n_sources),
            "target_ids": list(target_ids),
            "include_neighbors": workload.include_neighbors,
        },
    )


def _target_dir(plan: StampRunPlan, target_id: int) -> Path:
    return plan.run_dir / "stamps" / f"target_{int(target_id)}"


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


def _target_spec(plan: StampRunPlan, psf_id: int | None) -> Any:
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
    api: Any,
    worker_rank: int = 0,
) -> dict[str, Any]:
    target_dir = _target_dir(plan, target_id)
    if target_is_complete(plan, target_id, api=api):
        if plan.run_config.execution.resume:
            return {"target_id": target_id, "status": "skipped"}
        if not plan.run_config.execution.overwrite:
            raise FileExistsError(
                f"target {target_id} already has complete artifacts; use resume or overwrite"
            )
    if plan.run_config.execution.overwrite and target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    if plan.paths.data_root is None:
        raise ValueError("data_root is required")

    spec = _target_spec(plan, psf_id)
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
    try:
        for coadd_index in coadd_ids:
            result = api.run_stamp_coadd(
                spec,
                target_source_id=target_id,
                stamp_shape=plan.workload.stamp_shape,
                coadd_index=coadd_index,
                services=services,
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
                )

            for raw_result in result.raw_results:
                products = raw_result.stamp_products
                frame_id = int(products.frame_index)
                if plan.workload.save_raw:
                    api.write_stamp_product_schema(
                        target_dir
                        / "schemas"
                        / "raw"
                        / f"frame_{frame_id:06d}.json",
                        products,
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
                api.write_stamp_product_schema(
                    target_dir
                    / "schemas"
                    / "coadd"
                    / f"coadd_{coadd_index:06d}.json",
                    coadd_products,
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
    finally:
        if raw_writer is not None:
            raw_writer.close()
        if coadd_writer is not None:
            coadd_writer.close()
    if not target_is_complete(plan, target_id, api=api):
        raise RuntimeError(
            f"stamp artifacts for target {target_id} failed readback validation"
        )
    return {"target_id": target_id, "status": "rendered"}


def _worker_inputs(request: StampWorkerRequest, api: Any) -> PreparedStampInputs:
    plan = request.plan
    if plan.workload.input_mode == "table":
        if plan.input_table_path is None:
            raise ValueError("table input path is required")
        loaded = load_stamp_target_table(
            plan.input_table_path,
            detector_shape=tuple(plan.spec.detector.shape),
        )
        by_id = {target.source_id: target for target in loaded.targets}
        missing = sorted(set(request.target_ids) - set(by_id))
        if missing:
            raise ValueError(f"stamp worker target rows are absent: {missing}")
        targets = [by_id[target_id] for target_id in request.target_ids]
        return PreparedStampInputs(
            target_ids=request.target_ids,
            catalogs={
                target.source_id: _table_catalog(plan, target, api)
                for target in targets
            },
            psf_ids={target.source_id: target.psf_id for target in targets},
            shared_catalog=None,
            provenance=dict(loaded.provenance),
        )
    catalog = api.StarCatalogCache.read(plan.catalog_cache)
    return PreparedStampInputs(
        target_ids=request.target_ids,
        catalogs={target_id: catalog for target_id in request.target_ids},
        psf_ids={},
        shared_catalog=catalog,
        provenance={"catalog_cache": str(plan.catalog_cache)},
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
        )
        request_path = request_dir / f"stamp_worker_{assignment.rank:02d}.json"
        _atomic_write_json(request_path, request.to_json_dict())
        log_path = log_dir / f"stamp_worker_{assignment.rank:02d}.log"
        log_handle = log_path.open("w", encoding="utf-8")
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
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
    stat = plan.input_table_path.stat()
    payload["input_table_identity"] = {
        "path": str(plan.input_table_path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
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
        store.update(catalog=_catalog_manifest(prepared, plan))
        if plan.run_config.execution.backend == "in-process":
            results = [
                _render_target(
                    plan,
                    target_id=target_id,
                    catalog=prepared.catalogs[target_id],
                    psf_id=prepared.psf_ids.get(target_id),
                    api=api,
                )
                for target_id in prepared.target_ids
            ]
        else:
            results = _launch_subprocess_workers(plan, prepared.target_ids)
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
