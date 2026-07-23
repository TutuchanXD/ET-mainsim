#!/usr/bin/env python3
"""Prepare or execute common independent Aster/varlc/wdlc stamp production."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from et_mainsim.science_stamp_production import (
    DEFAULT_CADENCE_SECONDS,
    DEFAULT_DURATION_DAYS,
    DEFAULT_MAX_RAW_FRAMES_PER_SHARD,
    DEFAULT_RAW_EXPOSURE_SECONDS,
    DEFAULT_STAMP_SHAPE,
    DIRECT_SHARED_FILESYSTEM_DELIVERY_EXECUTION_MODE,
    SCIENCE_PRODUCTION_TRACKS,
    STAGED_LOCAL_SCRATCH_DELIVERY_EXECUTION_MODE,
    ScienceStampProductionConfig,
    prepare_science_independent_production,
    run_science_independent_target,
    write_science_stamp_task_list,
)


def _task_pair(value: str) -> tuple[int, int]:
    source_text, separator, shard_text = str(value).partition(":")
    if (
        separator != ":"
        or not source_text.isdecimal()
        or not shard_text.isdecimal()
    ):
        raise argparse.ArgumentTypeError("task must be SOURCE_ID:SHARD_ID")
    return int(source_text), int(shard_text)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare",
        help="freeze science inputs, 10-second factors, targets, and time shards",
    )
    prepare.add_argument("--track", choices=SCIENCE_PRODUCTION_TRACKS, required=True)
    prepare.add_argument("--input-root", required=True)
    prepare.add_argument("--output-root", required=True)
    prepare.add_argument("--run-id", required=True)
    prepare.add_argument("--data-root", required=True)
    prepare.add_argument("--focalplane-registry", required=True)
    prepare.add_argument(
        "--external-source-id",
        action="append",
        dest="external_source_ids",
        help=(
            "repeat to select adapter source identities; values remain strings "
            "so zero-padded Aster identifiers are preserved"
        ),
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
    prepare.add_argument(
        "--delivery-execution-mode",
        choices=(
            DIRECT_SHARED_FILESYSTEM_DELIVERY_EXECUTION_MODE,
            STAGED_LOCAL_SCRATCH_DELIVERY_EXECUTION_MODE,
        ),
        default=STAGED_LOCAL_SCRATCH_DELIVERY_EXECUTION_MODE,
        help=(
            "freeze direct shared writes or node-local staged publication; "
            "never use both modes for one formal run"
        ),
    )

    run = subparsers.add_parser(
        "run-target",
        help="render one prepared target for all or selected no-resume shards",
    )
    run.add_argument("--manifest", required=True)
    run.add_argument("--source-id", required=True, type=int)
    run.add_argument("--case", choices=("static", "injected"), default="injected")
    run.add_argument(
        "--shard-id",
        action="append",
        type=int,
        dest="shard_ids",
        help="repeat to render selected shards; omit to render the full plan",
    )
    run.add_argument("--data-root")
    run.add_argument("--focalplane-registry")
    run.add_argument("--device", choices=("cpu", "cuda"))
    run.add_argument("--batch-size", type=int, default=32)
    run.add_argument(
        "--output-root",
        help=(
            "node-local case root required by staged_local_scratch_v1 and "
            "forbidden by direct_shared_filesystem"
        ),
    )

    task_list = subparsers.add_parser(
        "write-task-list",
        help="freeze an exact case-bound source/shard selection for Slurm",
    )
    task_list.add_argument("--manifest", required=True)
    task_list.add_argument("--case", choices=("static", "injected"), required=True)
    task_list.add_argument(
        "--task",
        action="append",
        type=_task_pair,
        required=True,
        dest="tasks",
        metavar="SOURCE_ID:SHARD_ID",
    )
    task_list.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the common production CLI and emit one machine-readable JSON value."""

    args = _parser().parse_args(argv)
    if args.command == "prepare":
        preparation = prepare_science_independent_production(
            ScienceStampProductionConfig(
                track=args.track,
                input_root=Path(args.input_root),
                output_root=Path(args.output_root),
                run_id=args.run_id,
                data_root=Path(args.data_root),
                focalplane_registry=Path(args.focalplane_registry),
                external_source_ids=(
                    None
                    if args.external_source_ids is None
                    else tuple(args.external_source_ids)
                ),
                duration_days=args.duration_days,
                raw_exposure_seconds=args.raw_exposure_seconds,
                cadence_seconds=tuple(args.cadence_seconds),
                max_raw_frames_per_shard=args.max_raw_frames_per_shard,
                stamp_shape=tuple(args.stamp_shape),
                device=args.device,
                run_seed=args.run_seed,
                delivery_execution_mode=args.delivery_execution_mode,
            )
        )
        print(
            json.dumps(
                {
                    "production_track": args.track,
                    "manifest_path": str(preparation.manifest_path),
                    "time_plan_path": str(preparation.time_plan_path),
                    "shard_count": len(preparation.time_plan.shards),
                    "accepted_raw_frame_count": (
                        preparation.time_plan.accepted_raw_frame_count
                    ),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0

    if args.command == "write-task-list":
        result = write_science_stamp_task_list(
            args.manifest,
            case=args.case,
            tasks=tuple(args.tasks),
            output_path=args.output,
        )
        print(
            json.dumps(
                {
                    "path": str(result.path),
                    "case": result.case,
                    "task_count": result.task_count,
                    "sha256": result.identity["sha256"],
                    "size_bytes": result.identity["size_bytes"],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0

    reports = run_science_independent_target(
        args.manifest,
        source_id=args.source_id,
        case=args.case,
        shard_ids=args.shard_ids,
        data_root=args.data_root,
        focalplane_registry=args.focalplane_registry,
        device=args.device,
        batch_size=args.batch_size,
        output_root=args.output_root,
    )
    print(json.dumps(list(reports), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by formal launcher.
    raise SystemExit(main())
