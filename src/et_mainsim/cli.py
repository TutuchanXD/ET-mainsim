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


def _run_config_from_args(args, loaded) -> RunConfig:
    if args.config is None:
        config = loaded.run_config
    else:
        config = RunConfig.from_toml(
            args.config.read_text(encoding="utf-8"),
            source=str(args.config.resolve()),
        )
    if config.workflow != "et-full-frame":
        raise ValueError(
            f"Run config workflow must be 'et-full-frame', got {config.workflow!r}"
        )

    path_updates = {}
    for argument, field_name in (
        (args.output_root, "output_root"),
        (args.data_root, "data_root"),
        (args.catalog_path, "catalog_path"),
        (args.focalplane_registry, "focalplane_registry"),
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
    config = _run_config_from_args(args, loaded)
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
    if args.command == "_worker":
        from et_mainsim.workflows.full_frame import run_worker_request_file

        result = run_worker_request_file(args.request)
        _json_print(result.to_dict())
        return 0
    parser.error("Unsupported command")
    return 2


__all__ = ["main"]
