from __future__ import annotations

import json
import os
import pickle
import shutil
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, Mapping

import numpy as np

from et_mainsim.config import LegacyWorkload, ResolvedRunPaths, RunConfig
from et_mainsim.manifest import RunManifestStore, _atomic_write_json
from et_mainsim.provenance import collect_provenance


_BASE_REQUIRED_FILES = frozenset(
    {
        "time.pkl",
        "psf_field_ids.pkl",
        "light_curves.pkl",
        "centroids.pkl",
        "apertures.pkl",
        "variants_settings.pkl",
        "stars_metadata_df.pkl",
        "sim_config.pkl",
        "time_manager.pkl",
        "telescope_xy_offsets.pkl",
        "dynamic_param_data.pkl",
        "dynamic_param_config.pkl",
        "legacy_effect_manifest.json",
    }
)


@dataclass(frozen=True)
class LegacyRunPlan:
    preset_name: str
    run_config: RunConfig
    paths: ResolvedRunPaths
    contract: Any
    run_dir: Path
    legacy_root: Path
    repo_root: Path

    @property
    def workload(self) -> LegacyWorkload:
        workload = self.run_config.workload
        if not isinstance(workload, LegacyWorkload):
            raise TypeError("legacy run plan requires LegacyWorkload")
        return workload

    @property
    def spec(self) -> Any:
        return self.contract.spec

    def to_dict(self, *, dry_run: bool) -> dict[str, Any]:
        return {
            "dry_run": bool(dry_run),
            "workflow": "legacy-sim",
            "preset": self.preset_name,
            "run_id": self.run_config.run_id,
            "run_dir": str(self.run_dir),
            "legacy_root": str(self.legacy_root),
            "paths": self.paths.to_dict(),
            "execution": self.run_config.execution.to_dict(),
            "workload": self.workload.to_dict(),
            "run_plan": {
                "run_count": self.workload.run_count,
                "stars_per_run": self.workload.stars_per_run,
                "total_targets": (
                    self.workload.run_count * self.workload.stars_per_run
                ),
                "raw_frame_count": self.spec.observation.resolved_n_frames,
            },
            "simulation_spec": self.spec.to_json_dict(),
            "effect_contract": self.contract.to_metadata(),
        }


def _science_api() -> SimpleNamespace:
    from photsim7.data_registry import DataRegistry
    from photsim7.legacy_workflow import (
        build_et_legacy_full_effect_runtime,
        read_legacy_effect_manifest,
    )

    return SimpleNamespace(
        DataRegistry=DataRegistry,
        build_runtime=build_et_legacy_full_effect_runtime,
        read_effect_manifest=read_legacy_effect_manifest,
    )


def build_run_plan(
    *,
    preset_name: str,
    run_config: RunConfig,
    contract: Any,
    repo_root: Path | str,
    env: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
) -> LegacyRunPlan:
    from photsim7.legacy_workflow import LegacyFullEffectContract

    if run_config.workflow != "legacy-sim" or not isinstance(
        run_config.workload, LegacyWorkload
    ):
        raise ValueError("legacy plan requires workflow='legacy-sim'")
    if not isinstance(contract, LegacyFullEffectContract):
        raise TypeError("legacy preset must provide LegacyFullEffectContract")
    if contract.spec.psf.compute_device != run_config.execution.device:
        raise ValueError(
            "legacy execution device must match the full-effect contract device"
        )
    paths = run_config.resolve_paths(env=env, cwd=cwd)
    run_dir = paths.output_root / run_config.run_id
    return LegacyRunPlan(
        preset_name=preset_name,
        run_config=run_config,
        paths=paths,
        contract=contract,
        run_dir=run_dir,
        legacy_root=run_dir / "legacy",
        repo_root=Path(repo_root).resolve(),
    )


def rebuild_contract(
    contract: Any,
    *,
    frames: int | None = None,
    run_seed: int | None = None,
    compute_device: str | None = None,
) -> Any:
    from photsim7.legacy_workflow import (
        LegacyFullEffectContract,
        make_et_legacy_full_effect_contract,
    )

    if not isinstance(contract, LegacyFullEffectContract):
        raise TypeError("contract must be LegacyFullEffectContract")
    spec = contract.spec
    observing_duration = spec.observation.observing_duration
    if frames is not None:
        frames = int(frames)
        if frames <= 0:
            raise ValueError("frames must be positive")
        observing_duration = frames * spec.observation.sampling_interval
    return make_et_legacy_full_effect_contract(
        shape=tuple(spec.detector.shape),
        detector_id=spec.detector.detector_id,
        run_seed=spec.rng.run_seed if run_seed is None else int(run_seed),
        compute_device=(
            spec.psf.compute_device if compute_device is None else compute_device
        ),
        float_precision=spec.psf.float_precision,
        observing_duration=observing_duration,
        n_jitter_integrated_psf_models=(spec.psf.n_jitter_integrated_psf_models),
        n_jitter_frames_per_model=spec.psf.n_jitter_frames_per_model,
        n_raw_frames_per_coadd=contract.n_raw_frames_per_coadd,
        telescope_count=spec.instrument.telescope_count,
        background_stars_max_mag=spec.catalog.background_stars_max_mag,
    )


def preflight(plan: LegacyRunPlan) -> None:
    if plan.paths.data_root is None:
        raise ValueError("ET_DATA_DIR or paths.data_root is required to run")
    if not plan.paths.data_root.is_dir():
        raise FileNotFoundError(
            f"Photsim7 data root does not exist: {plan.paths.data_root}"
        )


def _required_files(workload: LegacyWorkload) -> frozenset[str]:
    if workload.store_images:
        return _BASE_REQUIRED_FILES | {"images.pkl"}
    return _BASE_REQUIRED_FILES


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _metadata_column(metadata: Any, *names: str) -> np.ndarray | None:
    available = getattr(metadata, "colnames", None)
    if available is None:
        available = getattr(metadata, "columns", None)
    if available is None and isinstance(metadata, Mapping):
        available = metadata.keys()
    if available is None:
        return None
    available_names = {str(name) for name in available}
    for name in names:
        if name in available_names:
            return np.asarray(metadata[name])
    return None


def _target_et_magnitudes(
    metadata: Any,
    *,
    expected_targets: int,
) -> np.ndarray | None:
    magnitudes = _metadata_column(metadata, "ET Mag", "et_mag")
    star_ids = _metadata_column(metadata, "Star ID", "star_id")
    if magnitudes is None or star_ids is None:
        return None
    try:
        magnitudes = np.asarray(magnitudes, dtype=float).reshape(-1)
        star_ids = np.asarray(star_ids, dtype=int).reshape(-1)
    except (TypeError, ValueError):
        return None
    if magnitudes.size != star_ids.size or not np.all(np.isfinite(magnitudes)):
        return None

    first_by_star_id: dict[int, float] = {}
    for star_id, magnitude in zip(star_ids, magnitudes, strict=True):
        first_by_star_id.setdefault(int(star_id), float(magnitude))
    expected_ids = set(range(expected_targets))
    if set(first_by_star_id) != expected_ids:
        return None
    return np.asarray(
        [first_by_star_id[star_id] for star_id in range(expected_targets)],
        dtype=float,
    )


def _canonical_effect_inventory(effects: Any) -> dict[str, str] | None:
    if not isinstance(effects, (list, tuple)):
        return None
    inventory: dict[str, str] = {}
    for effect in effects:
        if not isinstance(effect, Mapping):
            return None
        effect_id = effect.get("effect_id")
        if not isinstance(effect_id, str) or not effect_id.strip():
            return None
        if effect_id in inventory:
            return None
        try:
            canonical_payload = json.dumps(
                effect,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
        except (TypeError, ValueError):
            return None
        inventory[effect_id] = canonical_payload
    return inventory


def _validate_run(
    run_dir: Path,
    *,
    workload: LegacyWorkload,
    api: Any,
    expected_effects: Any,
) -> dict[str, Any] | None:
    required = _required_files(workload)
    if not run_dir.is_dir() or not required.issubset(
        {path.name for path in run_dir.iterdir()}
    ):
        return None
    try:
        values = {
            name: _load_pickle(run_dir / name)
            for name in required
            if name.endswith(".pkl")
        }
        manifest = api.read_effect_manifest(run_dir / "legacy_effect_manifest.json")
        actual_inventory = _canonical_effect_inventory(manifest.get("effects"))
        expected_inventory = _canonical_effect_inventory(expected_effects)
        if (
            actual_inventory is None
            or expected_inventory is None
            or actual_inventory != expected_inventory
        ):
            return None
        enabled = sum(bool(item["enabled"]) for item in manifest["effects"])
        disabled = sum(not bool(item["enabled"]) for item in manifest["effects"])
        light_curves = np.asarray(values["light_curves.pkl"])
        centroids = np.asarray(values["centroids.pkl"])
        apertures = np.asarray(values["apertures.pkl"])
        target_et_magnitudes = _target_et_magnitudes(
            values["stars_metadata_df.pkl"],
            expected_targets=workload.stars_per_run,
        )
        if light_curves.ndim != 4 or light_curves.shape[2] != workload.stars_per_run:
            return None
        if centroids.ndim != 5 or centroids.shape[2] != workload.stars_per_run:
            return None
        if apertures.ndim != 5 or apertures.shape[2] != workload.stars_per_run:
            return None
        if target_et_magnitudes is None or not np.all(np.isfinite(light_curves)):
            return None
        if np.any(target_et_magnitudes < workload.et_mag_min) or np.any(
            target_et_magnitudes > workload.et_mag_max
        ):
            return None
    except (OSError, ValueError, TypeError, KeyError, pickle.UnpicklingError):
        return None
    return {
        "run_dir": str(run_dir),
        "file_count": len(tuple(run_dir.iterdir())),
        "enabled_effects": enabled,
        "disabled_effects": disabled,
        "light_curves_shape": list(light_curves.shape),
        "centroids_shape": list(centroids.shape),
        "apertures_shape": list(apertures.shape),
        "target_et_mag_min": float(np.min(target_et_magnitudes)),
        "target_et_mag_max": float(np.max(target_et_magnitudes)),
    }


def _validate_all_runs(plan: LegacyRunPlan, api: Any) -> list[dict[str, Any]] | None:
    summaries = []
    expected_effects = plan.contract.to_metadata().get("effects")
    for run_index in range(plan.workload.run_count):
        summary = _validate_run(
            plan.legacy_root / f"run_{run_index}",
            workload=plan.workload,
            api=api,
            expected_effects=expected_effects,
        )
        if summary is None:
            return None
        summaries.append(summary)
    expected = {f"run_{index}" for index in range(plan.workload.run_count)}
    actual = (
        {path.name for path in plan.legacy_root.iterdir() if path.is_dir()}
        if plan.legacy_root.is_dir()
        else set()
    )
    if actual != expected:
        return None
    return summaries


def _target_rows(plan: LegacyRunPlan) -> list[dict[str, float]]:
    workload = plan.workload
    rng = np.random.default_rng(plan.spec.rng.run_seed)
    magnitudes = rng.uniform(
        workload.et_mag_min,
        workload.et_mag_max,
        size=workload.stars_per_run,
    )
    return [
        {"x0": 0.0, "y0": 0.0, "et_mag": float(magnitude)} for magnitude in magnitudes
    ]


@contextmanager
def _runtime_environment(plan: LegacyRunPlan) -> Iterator[None]:
    updates = {"ET_DATA_DIR": str(plan.paths.data_root)}
    if plan.run_config.execution.gpu_ids:
        updates["CUDA_VISIBLE_DEVICES"] = ",".join(plan.run_config.execution.gpu_ids)
    previous = {name: os.environ.get(name) for name in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def run_legacy(
    plan: LegacyRunPlan,
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
    workload_payload = plan.workload.to_dict()
    spec_payload = plan.spec.to_json_dict()
    if store.path.exists():
        store.ensure_identity(
            workflow="legacy-sim",
            run_id=plan.run_config.run_id,
            simulation_spec=spec_payload,
            execution=execution_payload,
            workload=workload_payload,
        )
    else:
        store.create(
            workflow="legacy-sim",
            preset=plan.preset_name,
            run_id=plan.run_config.run_id,
            simulation_spec=spec_payload,
            execution=execution_payload,
            workload=workload_payload,
            frame_plan={
                "run_count": plan.workload.run_count,
                "stars_per_run": plan.workload.stars_per_run,
                "raw_frame_count": plan.spec.observation.resolved_n_frames,
            },
            provenance=collect_provenance(plan.repo_root),
            artifacts={
                "run_manifest": str(store.path),
                "legacy_root": str(plan.legacy_root),
                "required_files": sorted(_required_files(plan.workload)),
            },
        )
    try:
        store.start_attempt(
            control={
                "resume": plan.run_config.execution.resume,
                "overwrite": plan.run_config.execution.overwrite,
            }
        )
        completed = _validate_all_runs(plan, api)
        if completed is not None and plan.run_config.execution.resume:
            return store.transition(
                "completed",
                completion={
                    "requested_runs": plan.workload.run_count,
                    "completed_runs": plan.workload.run_count,
                    "rendered_runs": 0,
                    "skipped_runs": plan.workload.run_count,
                    "runs": completed,
                },
            )
        if plan.legacy_root.exists():
            if plan.run_config.execution.overwrite:
                shutil.rmtree(plan.legacy_root)
            elif any(plan.legacy_root.iterdir()):
                raise RuntimeError(
                    "partial legacy output cannot be resumed; use overwrite or a new run id"
                )
        plan.legacy_root.mkdir(parents=True, exist_ok=True)
        if plan.paths.data_root is None:
            raise ValueError("data_root is required")
        with _runtime_environment(plan):
            registry = api.DataRegistry(data_root=plan.paths.data_root)
            runtime = api.build_runtime(plan.contract, data_registry=registry)
            execution = plan.run_config.execution
            with runtime.build_simulator(
                ray_actor_count=execution.ray_actor_count,
                ray_num_cpus=int(execution.ray_num_cpus),
                ray_num_gpus=int(execution.ray_num_gpus),
                start_dashboard=False,
                verbose=False,
                store_images=plan.workload.store_images,
                mag_range=[
                    plan.workload.et_mag_min,
                    plan.workload.et_mag_max,
                ],
            ) as simulator:
                simulator.run(
                    run_count=plan.workload.run_count,
                    n_stars_per_run=plan.workload.stars_per_run,
                    sim_save_dir=plan.legacy_root,
                    resume=False,
                    is_full_save_dir=True,
                    user_star_field=_target_rows(plan),
                )
        summaries = _validate_all_runs(plan, api)
        if summaries is None:
            raise RuntimeError("legacy output failed artifact readback validation")
        _atomic_write_json(
            plan.run_dir / "legacy_output_summary.json",
            {
                "schema_id": "et_mainsim.legacy_output_summary",
                "schema_version": 1,
                "runs": summaries,
            },
        )
        return store.transition(
            "completed",
            completion={
                "requested_runs": plan.workload.run_count,
                "completed_runs": plan.workload.run_count,
                "rendered_runs": plan.workload.run_count,
                "skipped_runs": 0,
                "runs": summaries,
            },
        )
    except BaseException as error:
        if store.load()["status"] == "running":
            store.fail(error)
        raise


__all__ = [
    "LegacyRunPlan",
    "build_run_plan",
    "preflight",
    "rebuild_contract",
    "run_legacy",
]
