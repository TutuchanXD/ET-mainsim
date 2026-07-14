from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


MODULE_DIR = Path(__file__).resolve().parents[1] / "benchmarks"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import evaluate_main_rd_benchmark as bench  # noqa: E402


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_run(root: Path, name: str, *, mag_limit: float, totals: list[float]) -> Path:
    run_dir = root / name
    _write_json(
        run_dir / "run_config.json",
        {
            "args": {
                "frames": 4,
                "mag_limit": mag_limit,
                "gpus": "0,1,2",
                "workers_per_gpu": 8,
            },
            "selected_frame_indices": [0, 1, 2, 3],
            "worker_gpu_ids": ["0"] * 8 + ["1"] * 8 + ["2"] * 8,
        },
    )
    for index, total in enumerate(totals):
        _write_json(
            run_dir / "frame_summaries" / f"frame_{index:06d}.json",
            {
                "frame_index": index,
                "rank": index % 2,
                "n_stars": 100 + int(mag_limit * 10),
                "render_elapsed_s": total - 1.0,
                "electronics_elapsed_s": 1.0,
                "total_elapsed_s": total,
                "actual_cosmic_events": index + 1,
                "saturated_pixels": index * 2,
                "peak_cuda_allocated_mb": 2048.0 + index,
                "peak_cuda_reserved_mb": 3072.0 + index,
            },
        )
    return run_dir


def test_collect_benchmark_summarizes_frame_times_and_star_counts(tmp_path):
    _make_run(
        tmp_path, "main_rd_8900x9120_g_lt_16", mag_limit=16.0, totals=[10.0, 14.0]
    )
    _make_run(
        tmp_path,
        "main_rd_8900x9120_g_lt_16p5",
        mag_limit=16.5,
        totals=[20.0, 24.0, 28.0],
    )

    report = bench.collect_benchmark(tmp_path)

    assert [row["mag_limit"] for row in report["runs"]] == [16.0, 16.5]
    first = report["runs"][0]
    assert first["run_name"] == "main_rd_8900x9120_g_lt_16"
    assert first["frames_expected"] == 4
    assert first["frames_completed"] == 2
    assert first["n_stars"] == 260
    assert first["worker_count"] == 24
    assert first["total_elapsed_s_mean"] == pytest.approx(12.0)
    assert first["total_elapsed_s_p50"] == pytest.approx(12.0)
    assert first["total_elapsed_s_max"] == pytest.approx(14.0)
    assert first["frames_per_worker"] == pytest.approx(2 / 24)
    assert first["actual_cosmic_events_sum"] == 3
    assert first["saturated_pixels_max"] == 2


def test_write_reports_creates_summary_json_and_csvs(tmp_path):
    _make_run(tmp_path, "main_rd_8900x9120_g_lt_17", mag_limit=17.0, totals=[30.0])
    report = bench.collect_benchmark(tmp_path)

    outputs = bench.write_reports(report, tmp_path / "benchmark")

    summary_json = outputs["summary_json"]
    run_csv = outputs["run_csv"]
    frame_csv = outputs["frame_csv"]
    assert summary_json.exists()
    assert run_csv.exists()
    assert frame_csv.exists()
    assert (
        json.loads(summary_json.read_text(encoding="utf-8"))["runs"][0]["mag_limit"]
        == 17.0
    )
    assert "total_elapsed_s_mean" in run_csv.read_text(encoding="utf-8")
    assert "frame_index" in frame_csv.read_text(encoding="utf-8")


def test_collect_benchmark_reads_unified_run_manifest_and_package_summary(tmp_path):
    run_dir = tmp_path / "et-full-frame-production"
    _write_json(
        run_dir / "run_manifest.json",
        {
            "schema_id": "et_mainsim.run_manifest",
            "schema_version": 1,
            "workflow": "et-full-frame",
            "run_id": run_dir.name,
            "simulation_spec": {
                "catalog": {"background_stars_max_mag": 18.0},
            },
            "execution": {
                "backend": "local-subprocess",
                "device": "cuda",
                "gpu_ids": ["0", "1"],
                "workers_per_device": 2,
            },
            "frame_plan": {"requested": [0, 1, 2]},
        },
    )
    _write_json(
        run_dir / "frame_summaries" / "frame_000000.json",
        {
            "artifact_schema_version": 1,
            "frame_index": 0,
            "actual_cosmic_events": 5,
            "saturation_count": 7,
            "et_mainsim": {
                "rank": 1,
                "n_stars": 900,
                "pipeline_elapsed_s": 12.0,
                "total_elapsed_s": 12.5,
                "peak_cuda_allocated_mb": 2048.0,
                "peak_cuda_reserved_mb": 3072.0,
            },
        },
    )
    _write_json(
        run_dir / "frame_summaries" / "frame_000000_schema.json",
        {"schema_id": "photsim7.single_cadence_frame_products.v1"},
    )

    report = bench.collect_benchmark(run_dir)

    summary = report["runs"][0]
    assert summary["mag_limit"] == 18.0
    assert summary["frames_expected"] == 3
    assert summary["frames_completed"] == 1
    assert summary["worker_count"] == 4
    assert summary["n_stars"] == 900
    assert summary["total_elapsed_s_mean"] == pytest.approx(12.5)
    assert summary["render_elapsed_s_mean"] == pytest.approx(12.0)
    assert summary["actual_cosmic_events_sum"] == 5
    assert summary["saturated_pixels_max"] == 7


def test_collect_benchmark_tolerates_null_frame_index(tmp_path):
    run_dir = tmp_path / "partial-run"
    _write_json(
        run_dir / "frame_summaries" / "frame_partial.json",
        {"frame_index": None},
    )

    report = bench.collect_benchmark(run_dir)

    frame = report["frames"][0]
    assert frame["frame_index"] is None
    assert frame["frame_path"].endswith("frames/frame_-00001.npy")
