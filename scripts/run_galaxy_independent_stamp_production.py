#!/usr/bin/env python3
"""Prepare or execute the formal independent Galaxy stamp production workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from et_mainsim.galaxy_stamp_production import (
    DEFAULT_CADENCE_SECONDS,
    DEFAULT_DURATION_DAYS,
    DEFAULT_GALAXY_PRODUCTION_SOURCE_IDS,
    DEFAULT_MAX_RAW_FRAMES_PER_SHARD,
    DEFAULT_RAW_EXPOSURE_SECONDS,
    DEFAULT_STAMP_SHAPE,
    GalaxyStampProductionConfig,
    prepare_galaxy_independent_production,
    run_galaxy_independent_target,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare",
        help="freeze Galaxy FITS inputs, 10-s factor snapshots, and time shards",
    )
    prepare.add_argument("--input-fits", required=True)
    prepare.add_argument("--output-root", required=True)
    prepare.add_argument("--run-id", required=True)
    prepare.add_argument("--data-root", required=True)
    prepare.add_argument("--focalplane-registry", required=True)
    prepare.add_argument(
        "--source-id",
        action="append",
        type=int,
        dest="source_ids",
        help="repeat to override the approved ten-source Galaxy set",
    )
    prepare.add_argument("--duration-days", type=float, default=DEFAULT_DURATION_DAYS)
    prepare.add_argument(
        "--raw-exposure-seconds",
        type=float,
        default=DEFAULT_RAW_EXPOSURE_SECONDS,
    )
    prepare.add_argument(
        "--cadence-seconds",
        nargs="+",
        type=float,
        default=DEFAULT_CADENCE_SECONDS,
    )
    prepare.add_argument(
        "--max-raw-frames-per-shard",
        type=int,
        default=DEFAULT_MAX_RAW_FRAMES_PER_SHARD,
    )
    prepare.add_argument(
        "--stamp-shape",
        nargs=2,
        type=int,
        default=DEFAULT_STAMP_SHAPE,
        metavar=("ROWS", "COLS"),
    )
    prepare.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    prepare.add_argument("--run-seed", type=int, default=20260714)

    run = subparsers.add_parser(
        "run-target",
        help="render one prepared target for all or selected non-resumable shards",
    )
    run.add_argument("--manifest", required=True)
    run.add_argument("--source-id", required=True, type=int)
    run.add_argument("--case", choices=("static", "injected"), default="injected")
    run.add_argument(
        "--shard-id",
        action="append",
        type=int,
        dest="shard_ids",
        help="repeat to run selected shards; omit to run every planned shard",
    )
    run.add_argument("--data-root")
    run.add_argument("--focalplane-registry")
    run.add_argument("--device", choices=("cpu", "cuda"))
    run.add_argument("--batch-size", type=int, default=32)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        preparation = prepare_galaxy_independent_production(
            GalaxyStampProductionConfig(
                input_fits=Path(args.input_fits),
                output_root=Path(args.output_root),
                run_id=args.run_id,
                data_root=Path(args.data_root),
                focalplane_registry=Path(args.focalplane_registry),
                source_ids=(
                    DEFAULT_GALAXY_PRODUCTION_SOURCE_IDS
                    if args.source_ids is None
                    else tuple(args.source_ids)
                ),
                duration_days=args.duration_days,
                raw_exposure_seconds=args.raw_exposure_seconds,
                cadence_seconds=tuple(args.cadence_seconds),
                max_raw_frames_per_shard=args.max_raw_frames_per_shard,
                stamp_shape=tuple(args.stamp_shape),
                device=args.device,
                run_seed=args.run_seed,
            )
        )
        print(
            json.dumps(
                {
                    "manifest_path": str(preparation.manifest_path),
                    "time_plan_path": str(preparation.time_plan_path),
                    "shard_count": len(preparation.time_plan.shards),
                    "accepted_raw_frame_count": preparation.time_plan.accepted_raw_frame_count,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    reports = run_galaxy_independent_target(
        args.manifest,
        source_id=args.source_id,
        case=args.case,
        shard_ids=args.shard_ids,
        data_root=args.data_root,
        focalplane_registry=args.focalplane_registry,
        device=args.device,
        batch_size=args.batch_size,
    )
    print(json.dumps(list(reports), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by production launcher.
    raise SystemExit(main())
