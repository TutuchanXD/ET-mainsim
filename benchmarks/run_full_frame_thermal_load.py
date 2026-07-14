#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

from et_mainsim.config import RunPaths
from et_mainsim.presets import load_preset
from et_mainsim.workflows.full_frame import build_run_plan, run_full_frame


def build_benchmark_spec(
    base: Any,
    *,
    frames: int,
    mag_limit: float,
    jitter_models: int,
    run_seed: int,
) -> Any:
    frames = int(frames)
    jitter_models = int(jitter_models)
    if frames <= 0 or jitter_models <= 0:
        raise ValueError("frames and jitter_models must be positive")
    query_options = dict(base.catalog.query_options)
    query_options["mag_lim"] = float(mag_limit)
    return replace(
        base,
        observation=replace(
            base.observation,
            observing_duration=frames * base.observation.sampling_interval,
            n_frames=frames,
            frame_start_s=None,
        ),
        catalog=replace(
            base.catalog,
            background_stars_max_mag=float(mag_limit),
            query_options=query_options,
        ),
        psf=replace(
            base.psf,
            n_jitter_integrated_psf_models=jitter_models,
            compute_device="cuda",
        ),
        rng=replace(base.rng, run_seed=int(run_seed)),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one maintained full-frame thermal-load benchmark worker."
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--catalog-cache", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=1)
    parser.add_argument("--mag-limit", type=float, default=17.0)
    parser.add_argument("--max-stars", type=int, default=100000)
    parser.add_argument("--jitter-models", type=int, default=100)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--gpu", default="0")
    return parser


def main() -> None:
    args = _parser().parse_args()
    loaded = load_preset("et-full-frame-production")
    spec = build_benchmark_spec(
        loaded.simulation_spec,
        frames=args.frames,
        mag_limit=args.mag_limit,
        jitter_models=args.jitter_models,
        run_seed=args.seed,
    )
    config = replace(
        loaded.run_config,
        run_id=args.run_id,
        paths=RunPaths(
            output_root=str(args.output_root),
            data_root=os.environ.get("ET_DATA_DIR", ""),
            catalog_path=os.environ.get("GAIA_CATALOG_DIR", ""),
            focalplane_registry=(
                str(Path(os.environ["ET_FOCALPLANE_ROOT"]) / "data")
                if os.environ.get("ET_FOCALPLANE_ROOT")
                else ""
            ),
            catalog_cache=str(args.catalog_cache),
        ),
        execution=replace(
            loaded.run_config.execution,
            backend="local-subprocess",
            device="cuda",
            gpu_ids=(str(args.gpu),),
            workers_per_device=1,
            frame_indices=tuple(range(int(args.frames))),
            resume=False,
            overwrite=False,
            preview_count=0,
            max_stars=args.max_stars,
        ),
    )
    repo_root = Path(
        os.environ.get("ET_MAINSIM_ROOT", Path(__file__).resolve().parents[1])
    )
    plan = build_run_plan(
        preset_name="benchmark-full-frame-thermal-load",
        run_config=config,
        spec=spec,
        repo_root=repo_root,
    )
    manifest = run_full_frame(plan)
    print(plan.run_dir)
    print(manifest["completion"])


if __name__ == "__main__":
    main()
