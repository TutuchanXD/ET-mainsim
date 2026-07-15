from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from et_mainsim.config import RunConfig
from et_mainsim.presets import (
    canonical_preset_name,
    list_presets,
    load_preset,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="et-mainsim",
        description="ET-mainsim reference workflows powered by Photsim7.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    presets_parser = subparsers.add_parser(
        "presets",
        help="List shipped workflow presets.",
    )
    presets_parser.add_argument("--workflow", default=None)

    show_parser = subparsers.add_parser(
        "show",
        help="Show a validated preset and canonical scientific spec.",
    )
    show_parser.add_argument("preset")
    show_parser.add_argument("--format", choices=("json", "summary"), default="summary")

    run_parser = subparsers.add_parser("run", help="Run a maintained workflow.")
    run_subparsers = run_parser.add_subparsers(dest="workflow", required=True)
    full_frame = run_subparsers.add_parser(
        "et-full-frame",
        help="Run one physical ET main detector.",
    )
    full_frame.add_argument("--preset", default="smoke")
    full_frame.add_argument("--config", type=Path)
    full_frame.add_argument("--spec", type=Path)
    full_frame.add_argument("--run-id")
    full_frame.add_argument("--output-root", type=Path)
    full_frame.add_argument("--data-root", type=Path)
    full_frame.add_argument("--catalog-path", type=Path)
    full_frame.add_argument("--focalplane-registry", type=Path)
    full_frame.add_argument("--catalog-cache", type=Path)
    full_frame.add_argument("--frames", type=int)
    full_frame.add_argument("--frame-indices")
    full_frame.add_argument("--target-epoch-jyear", type=float)
    full_frame.add_argument("--seed", type=int)
    full_frame.add_argument(
        "--backend",
        choices=("in-process", "local-subprocess"),
    )
    full_frame.add_argument("--device", choices=("cpu", "cuda"))
    full_frame.add_argument("--gpus", help="Comma-separated visible GPU ids.")
    full_frame.add_argument("--workers-per-device", type=int)
    full_frame.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    full_frame.add_argument("--overwrite", action="store_true")
    full_frame.add_argument("--force-catalog-cache", action="store_true")
    full_frame.add_argument("--preview-count", type=int)
    full_frame.add_argument("--max-stars", type=int)
    full_frame.add_argument("--progress", action="store_true")
    full_frame.add_argument("--save-cosmic-mask", action="store_true")
    full_frame.add_argument("--save-stellar-mean", action="store_true")
    full_frame.add_argument("--prepare-catalog-only", action="store_true")
    full_frame.add_argument("--dry-run", action="store_true")

    stamp = run_subparsers.add_parser(
        "et-stamp",
        help="Run physical ET target stamps or an independent target table.",
    )
    stamp.add_argument("--preset", default="smoke")
    stamp.add_argument("--config", type=Path)
    stamp.add_argument("--spec", type=Path)
    stamp.add_argument("--run-id")
    stamp.add_argument("--output-root", type=Path)
    stamp.add_argument("--data-root", type=Path)
    stamp.add_argument("--catalog-path", type=Path)
    stamp.add_argument("--focalplane-registry", type=Path)
    stamp.add_argument("--catalog-cache", type=Path)
    stamp.add_argument("--frames", type=int)
    stamp.add_argument("--target-epoch-jyear", type=float)
    stamp.add_argument("--seed", type=int)
    stamp.add_argument(
        "--backend",
        choices=("in-process", "local-subprocess"),
    )
    stamp.add_argument("--device", choices=("cpu", "cuda"))
    stamp.add_argument("--gpus")
    stamp.add_argument("--workers-per-device", type=int)
    stamp.add_argument("--resume", action=argparse.BooleanOptionalAction, default=None)
    stamp.add_argument("--overwrite", action="store_true")
    stamp.add_argument("--force-catalog-cache", action="store_true")
    stamp.add_argument("--preview-count", type=int)
    stamp.add_argument("--max-stars", type=int)
    stamp.add_argument("--progress", action="store_true")
    stamp.add_argument("--save-cosmic-mask", action="store_true")
    stamp.add_argument("--save-stellar-mean", action="store_true")
    stamp.add_argument("--input-table")
    stamp.add_argument("--variability-table")
    stamp.add_argument("--target-source-id", type=int, action="append")
    stamp.add_argument("--target-limit", type=int)
    stamp.add_argument("--stamp-rows", type=int)
    stamp.add_argument("--stamp-cols", type=int)
    stamp.add_argument(
        "--include-neighbors",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    stamp.add_argument("--save-raw", action=argparse.BooleanOptionalAction, default=None)
    stamp.add_argument(
        "--save-coadd",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    stamp.add_argument("--save-electron-components", action="store_true")
    stamp.add_argument(
        "--artifact-profile",
        choices=("detailed", "compact"),
    )
    stamp.add_argument("--write-batch-size", type=int)
    stamp.add_argument("--coadd-shard-index", type=int)
    stamp.add_argument("--coadd-shard-count", type=int)
    stamp.add_argument("--dry-run", action="store_true")

    legacy = run_subparsers.add_parser(
        "legacy-sim",
        help="Run the et_sim_100_det full-effect compatibility workflow.",
    )
    legacy.add_argument("--preset", default="full-effects-smoke")
    legacy.add_argument("--config", type=Path)
    legacy.add_argument("--spec", type=Path)
    legacy.add_argument("--run-id")
    legacy.add_argument("--output-root", type=Path)
    legacy.add_argument("--data-root", type=Path)
    legacy.add_argument("--catalog-path", type=Path)
    legacy.add_argument("--focalplane-registry", type=Path)
    legacy.add_argument("--catalog-cache", type=Path)
    legacy.add_argument("--frames", type=int)
    legacy.add_argument("--seed", type=int)
    legacy.add_argument("--backend", choices=("local-ray",))
    legacy.add_argument("--device", choices=("cpu", "cuda"))
    legacy.add_argument("--gpus")
    legacy.add_argument("--workers-per-device", type=int)
    legacy.add_argument("--resume", action=argparse.BooleanOptionalAction, default=None)
    legacy.add_argument("--overwrite", action="store_true")
    legacy.add_argument("--force-catalog-cache", action="store_true")
    legacy.add_argument("--preview-count", type=int)
    legacy.add_argument("--max-stars", type=int)
    legacy.add_argument("--progress", action="store_true")
    legacy.add_argument("--save-cosmic-mask", action="store_true")
    legacy.add_argument("--save-stellar-mean", action="store_true")
    legacy.add_argument("--ray-actor-count", type=int)
    legacy.add_argument("--ray-num-cpus", type=int)
    legacy.add_argument("--ray-num-gpus", type=int)
    legacy.add_argument("--run-count", type=int)
    legacy.add_argument("--stars-per-run", type=int)
    legacy.add_argument(
        "--store-images",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    legacy.add_argument("--et-mag-min", type=float)
    legacy.add_argument("--et-mag-max", type=float)
    legacy.add_argument("--dry-run", action="store_true")

    worker = subparsers.add_parser("_worker", help=argparse.SUPPRESS)
    worker.add_argument("--request", type=Path, required=True)
    return parser


def _json_print(payload) -> None:
    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


def _run_config_from_args(args, loaded, *, workflow: str) -> RunConfig:
    if args.config is None:
        config = loaded.run_config
    else:
        config = RunConfig.from_toml(
            args.config.read_text(encoding="utf-8"),
            source=str(args.config.resolve()),
        )
    if config.workflow != workflow:
        raise ValueError(
            f"Run config workflow must be {workflow!r}, got {config.workflow!r}"
        )

    path_updates = {}
    for argument, field_name in (
        (args.output_root, "output_root"),
        (args.data_root, "data_root"),
        (args.catalog_path, "catalog_path"),
        (args.focalplane_registry, "focalplane_registry"),
        (args.catalog_cache, "catalog_cache"),
    ):
        if argument is not None:
            path_updates[field_name] = str(argument)
    paths = replace(config.paths, **path_updates) if path_updates else config.paths

    execution_updates = {}
    if args.backend is not None:
        execution_updates["backend"] = args.backend
    if args.device is not None:
        execution_updates["device"] = args.device
    if args.gpus is not None:
        execution_updates["gpu_ids"] = tuple(
            value.strip() for value in args.gpus.split(",") if value.strip()
        )
    if args.workers_per_device is not None:
        execution_updates["workers_per_device"] = args.workers_per_device
    if args.resume is not None:
        execution_updates["resume"] = args.resume
    if args.overwrite:
        execution_updates.update(overwrite=True, resume=False)
    if args.force_catalog_cache:
        execution_updates["force_catalog_cache"] = True
    if args.preview_count is not None:
        execution_updates["preview_count"] = args.preview_count
    if args.max_stars is not None:
        execution_updates["max_stars"] = args.max_stars
    if args.progress:
        execution_updates["progress"] = True
    if args.save_cosmic_mask:
        execution_updates["save_cosmic_mask"] = True
    if args.save_stellar_mean:
        execution_updates["save_stellar_mean"] = True
    if getattr(args, "ray_actor_count", None) is not None:
        execution_updates["ray_actor_count"] = args.ray_actor_count
    if getattr(args, "ray_num_cpus", None) is not None:
        execution_updates["ray_num_cpus"] = args.ray_num_cpus
    if getattr(args, "ray_num_gpus", None) is not None:
        execution_updates["ray_num_gpus"] = args.ray_num_gpus
    execution = (
        replace(config.execution, **execution_updates)
        if execution_updates
        else config.execution
    )
    return replace(
        config,
        run_id=config.run_id if args.run_id is None else args.run_id,
        paths=paths,
        execution=execution,
    )


def _spec_from_args(args, loaded):
    if args.spec is None:
        return loaded.simulation_spec
    from photsim7.specs import SimulationSpec

    return SimulationSpec.from_json(args.spec.read_text(encoding="utf-8"))


def _run_full_frame_command(args) -> int:
    from et_mainsim.workflows.full_frame import build_run_plan, run_full_frame

    preset_name = canonical_preset_name("et-full-frame", args.preset)
    loaded = load_preset(preset_name)
    config = _run_config_from_args(args, loaded, workflow="et-full-frame")
    spec = _spec_from_args(args, loaded)
    repo_root = Path(
        os.environ.get("ET_MAINSIM_ROOT", Path(__file__).resolve().parents[2])
    )
    plan = build_run_plan(
        preset_name=preset_name,
        run_config=config,
        spec=spec,
        repo_root=repo_root,
        frames=args.frames,
        frame_indices=args.frame_indices,
        target_epoch_jyear=args.target_epoch_jyear,
        run_seed=args.seed,
    )
    if args.dry_run:
        _json_print(plan.to_dict(dry_run=True))
        return 0
    manifest = run_full_frame(
        plan,
        prepare_catalog_only=args.prepare_catalog_only,
    )
    _json_print(
        {
            "run_dir": str(plan.run_dir),
            "manifest": str(plan.run_dir / "run_manifest.json"),
            "status": manifest["status"],
            "completion": manifest["completion"],
        }
    )
    return 0


def _stamp_config_from_args(args, loaded) -> RunConfig:
    config = _run_config_from_args(args, loaded, workflow="et-stamp")
    workload = config.workload
    updates = {}
    if args.input_table is not None:
        updates.update(
            input_mode="table",
            input_table=str(args.input_table),
            target_source_ids=(),
            target_limit=0,
            include_neighbors=False,
        )
    if args.variability_table is not None:
        updates["variability_table"] = str(args.variability_table)
    if args.target_source_id is not None:
        updates["target_source_ids"] = tuple(args.target_source_id)
    if args.target_limit is not None:
        updates["target_limit"] = args.target_limit
    if args.stamp_rows is not None:
        updates["stamp_rows"] = args.stamp_rows
    if args.stamp_cols is not None:
        updates["stamp_cols"] = args.stamp_cols
    if args.include_neighbors is not None:
        updates["include_neighbors"] = args.include_neighbors
    if args.save_raw is not None:
        updates["save_raw"] = args.save_raw
    if args.save_coadd is not None:
        updates["save_coadd"] = args.save_coadd
    if args.save_electron_components:
        updates["save_electron_components"] = True
    if args.artifact_profile is not None:
        updates["artifact_profile"] = args.artifact_profile
    if args.write_batch_size is not None:
        updates["write_batch_size"] = args.write_batch_size
    if args.coadd_shard_index is not None:
        updates["coadd_shard_index"] = args.coadd_shard_index
    if args.coadd_shard_count is not None:
        updates["coadd_shard_count"] = args.coadd_shard_count
    if updates:
        workload = replace(workload, **updates)
    return replace(config, workload=workload)


def _run_stamp_command(args) -> int:
    from et_mainsim.workflows.stamp import build_run_plan, run_stamp

    preset_name = canonical_preset_name("et-stamp", args.preset)
    loaded = load_preset(preset_name)
    config = _stamp_config_from_args(args, loaded)
    spec = _spec_from_args(args, loaded)
    repo_root = Path(
        os.environ.get("ET_MAINSIM_ROOT", Path(__file__).resolve().parents[2])
    )
    plan = build_run_plan(
        preset_name=preset_name,
        run_config=config,
        spec=spec,
        repo_root=repo_root,
        frames=args.frames,
        target_epoch_jyear=args.target_epoch_jyear,
        run_seed=args.seed,
    )
    if args.dry_run:
        _json_print(plan.to_dict(dry_run=True))
        return 0
    manifest = run_stamp(plan)
    _json_print(
        {
            "run_dir": str(plan.run_dir),
            "manifest": str(plan.run_dir / "run_manifest.json"),
            "status": manifest["status"],
            "completion": manifest["completion"],
        }
    )
    return 0


def _legacy_config_from_args(args, loaded) -> RunConfig:
    config = _run_config_from_args(args, loaded, workflow="legacy-sim")
    updates = {}
    for argument, field_name in (
        (args.run_count, "run_count"),
        (args.stars_per_run, "stars_per_run"),
        (args.store_images, "store_images"),
        (args.et_mag_min, "et_mag_min"),
        (args.et_mag_max, "et_mag_max"),
    ):
        if argument is not None:
            updates[field_name] = argument
    workload = replace(config.workload, **updates) if updates else config.workload
    return replace(config, workload=workload)


def _run_legacy_command(args) -> int:
    from et_mainsim.workflows.legacy import (
        build_run_plan,
        rebuild_contract,
        run_legacy,
    )

    preset_name = canonical_preset_name("legacy-sim", args.preset)
    loaded = load_preset(preset_name)
    config = _legacy_config_from_args(args, loaded)
    contract = rebuild_contract(
        loaded.science_contract,
        frames=args.frames,
        run_seed=args.seed,
        compute_device=config.execution.device,
    )
    if args.spec is not None:
        from photsim7.specs import SimulationSpec

        requested = SimulationSpec.from_json(args.spec.read_text(encoding="utf-8"))
        if requested.to_json() != contract.spec.to_json():
            raise ValueError(
                "legacy custom spec must exactly match the full-effect contract; "
                "use typed CLI overrides for frames, seed, and device"
            )
    repo_root = Path(
        os.environ.get("ET_MAINSIM_ROOT", Path(__file__).resolve().parents[2])
    )
    plan = build_run_plan(
        preset_name=preset_name,
        run_config=config,
        contract=contract,
        repo_root=repo_root,
    )
    if args.dry_run:
        _json_print(plan.to_dict(dry_run=True))
        return 0
    manifest = run_legacy(plan)
    _json_print(
        {
            "run_dir": str(plan.run_dir),
            "manifest": str(plan.run_dir / "run_manifest.json"),
            "status": manifest["status"],
            "completion": manifest["completion"],
        }
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(None if argv is None else list(argv))
    if args.command == "presets":
        descriptors = list_presets(workflow=args.workflow)
        for descriptor in descriptors:
            print(f"{descriptor.name}\t{descriptor.workflow}\t{descriptor.description}")
        return 0
    if args.command == "show":
        loaded = load_preset(args.preset)
        payload = {
            "preset": loaded.descriptor.to_dict(),
            "simulation_spec": loaded.simulation_spec.to_json_dict(),
            "run_config": loaded.run_config.to_dict(),
        }
        if args.format == "json":
            _json_print(payload)
        else:
            descriptor = loaded.descriptor
            spec = loaded.simulation_spec
            print(f"Preset: {descriptor.name}")
            print(f"Workflow: {descriptor.workflow}")
            print(f"Detector: {spec.detector.detector_id} {spec.detector.shape}")
            print(f"Frames: {spec.observation.resolved_n_frames}")
            print(f"Execution: {loaded.run_config.execution.backend}")
        return 0
    if args.command == "run" and args.workflow == "et-full-frame":
        try:
            return _run_full_frame_command(args)
        except (KeyError, OSError, RuntimeError, ValueError) as error:
            parser.error(str(error))
    if args.command == "run" and args.workflow == "et-stamp":
        try:
            return _run_stamp_command(args)
        except (KeyError, OSError, RuntimeError, ValueError) as error:
            parser.error(str(error))
    if args.command == "run" and args.workflow == "legacy-sim":
        try:
            return _run_legacy_command(args)
        except (KeyError, OSError, RuntimeError, ValueError) as error:
            parser.error(str(error))
    if args.command == "_worker":
        with args.request.open("r", encoding="utf-8") as handle:
            schema_id = json.load(handle).get("schema_id")
        if schema_id == "et_mainsim.full_frame_worker_request":
            from et_mainsim.workflows.full_frame import run_worker_request_file

            result = run_worker_request_file(args.request)
            _json_print(result.to_dict())
        elif schema_id == "et_mainsim.stamp_worker_request":
            from et_mainsim.workflows.stamp import run_stamp_worker_request_file

            result = run_stamp_worker_request_file(args.request)
            _json_print({"targets": result})
        else:
            parser.error(f"Unsupported worker request schema {schema_id!r}")
        return 0
    parser.error("Unsupported command")
    return 2


__all__ = ["main"]
