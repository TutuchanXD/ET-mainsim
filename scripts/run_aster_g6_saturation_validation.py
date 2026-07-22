#!/usr/bin/env python3
"""Prepare or render the approved one-hour Aster G=6 saturation validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from et_mainsim.aster_saturation_validation import (
    AsterG6SaturationValidationConfig,
    prepare_aster_g6_saturation_validation,
    run_aster_g6_saturation_validation,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="freeze G=6 Aster inputs")
    prepare.add_argument("--source-dat", required=True)
    prepare.add_argument("--source-log", required=True)
    prepare.add_argument("--variability-ecsv", required=True)
    prepare.add_argument("--output-root", required=True)
    prepare.add_argument("--run-id", required=True)
    prepare.add_argument("--data-root", required=True)
    prepare.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    prepare.add_argument("--run-seed", type=int, default=20260714)

    run = subparsers.add_parser("run", help="render one frozen validation case")
    run.add_argument("--manifest", required=True)
    run.add_argument("--case", choices=("static", "injected"), required=True)
    run.add_argument("--data-root")
    run.add_argument("--device", choices=("cpu", "cuda"))
    run.add_argument("--batch-size", type=int, default=64)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        prepared = prepare_aster_g6_saturation_validation(
            AsterG6SaturationValidationConfig(
                source_dat=Path(args.source_dat),
                source_log=Path(args.source_log),
                variability_ecsv=Path(args.variability_ecsv),
                output_root=Path(args.output_root),
                run_id=args.run_id,
                data_root=Path(args.data_root),
                device=args.device,
                run_seed=args.run_seed,
            )
        )
        print(
            json.dumps(
                {
                    "manifest_path": str(prepared.manifest_path),
                    "time_plan_path": str(prepared.time_plan_path),
                    "raw_frame_count": prepared.time_plan.accepted_raw_frame_count,
                    "shard_count": len(prepared.time_plan.shards),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    reports = run_aster_g6_saturation_validation(
        args.manifest,
        case=args.case,
        data_root=args.data_root,
        device=args.device,
        batch_size=args.batch_size,
    )
    print(json.dumps(list(reports), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by command-line use.
    raise SystemExit(main())
