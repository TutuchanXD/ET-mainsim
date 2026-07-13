#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean, median
from typing import Any


MAG_RE = re.compile(r"_g_lt_(?P<tag>[0-9]+(?:p[0-9]+)?)$")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int | None = None) -> int | None:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    position = (len(ordered) - 1) * float(fraction)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _stats(values: list[float], prefix: str) -> dict[str, float | None]:
    if not values:
        return {
            f"{prefix}_min": None,
            f"{prefix}_p50": None,
            f"{prefix}_mean": None,
            f"{prefix}_p90": None,
            f"{prefix}_max": None,
        }
    return {
        f"{prefix}_min": min(values),
        f"{prefix}_p50": median(values),
        f"{prefix}_mean": mean(values),
        f"{prefix}_p90": _percentile(values, 0.9),
        f"{prefix}_max": max(values),
    }


def _mag_limit_from_name(run_name: str) -> float | None:
    match = MAG_RE.search(run_name)
    if match is None:
        return None
    return float(match.group("tag").replace("p", "."))


def _expected_frames(run_metadata: dict[str, Any]) -> int | None:
    frame_plan = run_metadata.get("frame_plan")
    if isinstance(frame_plan, dict):
        requested = frame_plan.get("requested")
        if isinstance(requested, list):
            return len(requested)
        count = _as_int(frame_plan.get("count"))
        if count is not None:
            return count
    selected = run_metadata.get("selected_frame_indices")
    if isinstance(selected, list):
        return len(selected)
    args = (
        run_metadata.get("args") if isinstance(run_metadata.get("args"), dict) else {}
    )
    return _as_int(args.get("frames"))


def _worker_count(run_metadata: dict[str, Any]) -> int | None:
    execution = run_metadata.get("execution")
    if isinstance(execution, dict):
        if execution.get("backend") == "in-process":
            return 1
        workers_per_device = _as_int(execution.get("workers_per_device"), 1)
        if execution.get("device") == "cuda":
            gpu_ids = execution.get("gpu_ids")
            if isinstance(gpu_ids, list) and workers_per_device is not None:
                return len(gpu_ids) * workers_per_device
        return workers_per_device
    worker_gpu_ids = run_metadata.get("worker_gpu_ids")
    if isinstance(worker_gpu_ids, list):
        return len(worker_gpu_ids)
    args = (
        run_metadata.get("args") if isinstance(run_metadata.get("args"), dict) else {}
    )
    gpus = [
        item.strip() for item in str(args.get("gpus", "")).split(",") if item.strip()
    ]
    workers_per_gpu = _as_int(args.get("workers_per_gpu"))
    if gpus and workers_per_gpu is not None:
        return len(gpus) * workers_per_gpu
    return None


def _mag_limit(run_name: str, run_metadata: dict[str, Any]) -> float | None:
    simulation_spec = run_metadata.get("simulation_spec")
    if isinstance(simulation_spec, dict):
        catalog = simulation_spec.get("catalog")
        if isinstance(catalog, dict):
            value = _as_float(catalog.get("background_stars_max_mag"))
            if value is not None:
                return value
    args = (
        run_metadata.get("args") if isinstance(run_metadata.get("args"), dict) else {}
    )
    return _as_float(args.get("mag_limit"), _mag_limit_from_name(run_name))


def _frame_rows(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary_path in sorted((run_dir / "frame_summaries").glob("frame_*.json")):
        if summary_path.name.endswith("_schema.json"):
            continue
        payload = _read_json(summary_path)
        app_metrics = (
            payload.get("et_mainsim")
            if isinstance(payload.get("et_mainsim"), dict)
            else {}
        )
        rows.append(
            {
                "run_name": run_dir.name,
                "frame_index": _as_int(payload.get("frame_index")),
                "rank": _as_int(app_metrics.get("rank"), _as_int(payload.get("rank"))),
                "n_stars": _as_int(
                    app_metrics.get("n_stars"),
                    _as_int(payload.get("n_stars")),
                ),
                "render_elapsed_s": _as_float(
                    app_metrics.get("pipeline_elapsed_s"),
                    _as_float(payload.get("render_elapsed_s")),
                ),
                "electronics_elapsed_s": _as_float(
                    app_metrics.get("electronics_elapsed_s"),
                    _as_float(payload.get("electronics_elapsed_s")),
                ),
                "total_elapsed_s": _as_float(
                    app_metrics.get("total_elapsed_s"),
                    _as_float(payload.get("total_elapsed_s")),
                ),
                "actual_cosmic_events": _as_int(payload.get("actual_cosmic_events"), 0),
                "saturated_pixels": _as_int(
                    payload.get("saturated_pixels"),
                    _as_int(payload.get("saturation_count"), 0),
                ),
                "peak_cuda_allocated_mb": _as_float(
                    app_metrics.get("peak_cuda_allocated_mb"),
                    _as_float(payload.get("peak_cuda_allocated_mb")),
                ),
                "peak_cuda_reserved_mb": _as_float(
                    app_metrics.get("peak_cuda_reserved_mb"),
                    _as_float(payload.get("peak_cuda_reserved_mb")),
                ),
                "frame_path": payload.get("frame_path")
                or str(
                    run_dir
                    / "frames"
                    / f"frame_{int(payload.get('frame_index', -1)):06d}.npy"
                ),
                "summary_path": str(summary_path),
            }
        )
    return rows


def summarize_run(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run_manifest_path = run_dir / "run_manifest.json"
    run_config_path = run_dir / "run_config.json"
    if run_manifest_path.exists():
        run_metadata = _read_json(run_manifest_path)
    elif run_config_path.exists():
        run_metadata = _read_json(run_config_path)
    else:
        run_metadata = {}
    frames = _frame_rows(run_dir)
    total_times = [
        row["total_elapsed_s"] for row in frames if row["total_elapsed_s"] is not None
    ]
    render_times = [
        row["render_elapsed_s"] for row in frames if row["render_elapsed_s"] is not None
    ]
    electronics_times = [
        row["electronics_elapsed_s"]
        for row in frames
        if row["electronics_elapsed_s"] is not None
    ]
    n_stars_values = [row["n_stars"] for row in frames if row["n_stars"] is not None]
    worker_count = _worker_count(run_metadata)
    completed = len(frames)

    summary: dict[str, Any] = {
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "mag_limit": _mag_limit(run_dir.name, run_metadata),
        "frames_expected": _expected_frames(run_metadata),
        "frames_completed": completed,
        "worker_count": worker_count,
        "frames_per_worker": None
        if worker_count in (None, 0)
        else completed / float(worker_count),
        "n_stars": None if not n_stars_values else int(median(n_stars_values)),
        "actual_cosmic_events_sum": sum(
            int(row["actual_cosmic_events"] or 0) for row in frames
        ),
        "saturated_pixels_max": None
        if not frames
        else max(int(row["saturated_pixels"] or 0) for row in frames),
        "peak_cuda_allocated_mb_max": None
        if not frames
        else max(
            (
                row["peak_cuda_allocated_mb"]
                for row in frames
                if row["peak_cuda_allocated_mb"] is not None
            ),
            default=None,
        ),
        "peak_cuda_reserved_mb_max": None
        if not frames
        else max(
            (
                row["peak_cuda_reserved_mb"]
                for row in frames
                if row["peak_cuda_reserved_mb"] is not None
            ),
            default=None,
        ),
    }
    summary.update(_stats(total_times, "total_elapsed_s"))
    summary.update(_stats(render_times, "render_elapsed_s"))
    summary.update(_stats(electronics_times, "electronics_elapsed_s"))
    return summary, frames


def _run_dirs(root: Path) -> list[Path]:
    if (
        (root / "run_manifest.json").exists()
        or (root / "run_config.json").exists()
        or (root / "frame_summaries").exists()
    ):
        return [root]
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir()
        and (
            (path / "run_manifest.json").exists()
            or (path / "run_config.json").exists()
            or (path / "frame_summaries").exists()
        )
    )


def collect_benchmark(root: Path | str) -> dict[str, Any]:
    root = Path(root).expanduser()
    runs: list[dict[str, Any]] = []
    frames: list[dict[str, Any]] = []
    for run_dir in _run_dirs(root):
        run_summary, frame_rows = summarize_run(run_dir)
        runs.append(run_summary)
        frames.extend(frame_rows)
    runs.sort(
        key=lambda row: (
            row["mag_limit"] is None,
            row["mag_limit"] or 0.0,
            row["run_name"],
        )
    )
    frames.sort(
        key=lambda row: (
            row["run_name"],
            row["frame_index"] if row["frame_index"] is not None else -1,
        )
    )
    return {
        "root": str(root),
        "runs": runs,
        "frames": frames,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_reports(report: dict[str, Any], output_prefix: Path | str) -> dict[str, Path]:
    output_prefix = Path(output_prefix).expanduser()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_json = output_prefix.with_name(output_prefix.name + "_summary.json")
    run_csv = output_prefix.with_name(output_prefix.name + "_runs.csv")
    frame_csv = output_prefix.with_name(output_prefix.name + "_frames.csv")
    summary_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_csv(run_csv, report["runs"])
    _write_csv(frame_csv, report["frames"])
    return {
        "summary_json": summary_json,
        "run_csv": run_csv,
        "frame_csv": frame_csv,
    }


def _print_table(report: dict[str, Any]) -> None:
    header = (
        "mag_limit frames completed n_stars workers total_mean_s "
        "render_mean_s electronics_mean_s"
    )
    print(header)
    for row in report["runs"]:
        print(
            f"{row['mag_limit']} {row['frames_expected']} {row['frames_completed']} "
            f"{row['n_stars']} {row['worker_count']} "
            f"{row['total_elapsed_s_mean']} {row['render_elapsed_s_mean']} "
            f"{row['electronics_elapsed_s_mean']}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize ET-mainsim main_rd full-frame benchmark outputs.",
    )
    parser.add_argument(
        "root", type=Path, help="Run directory or parent containing run dirs."
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help="Output prefix for JSON/CSV reports. Defaults to <root>/benchmark.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.root).expanduser()
    output_prefix = (
        Path(args.output_prefix).expanduser()
        if args.output_prefix is not None
        else root / "benchmark"
    )
    report = collect_benchmark(root)
    outputs = write_reports(report, output_prefix)
    _print_table(report)
    print("wrote:")
    for path in outputs.values():
        print(path)


if __name__ == "__main__":
    main()
