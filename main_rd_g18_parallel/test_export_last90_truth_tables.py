from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import export_last90_truth_tables as exporter


def _write_minimal_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "main_rd_test_last90"
    cache_dir = run_dir / "cache"
    cache_dir.mkdir(parents=True)
    spec = {
        "frame_rows": 10,
        "frame_cols": 20,
        "run_label": run_dir.name,
        "exposure_s": 10.0,
        "scattered_light_e_s_pix": 0.2,
        "scattered_light_step_start_frame": 2,
        "scattered_light_step_e_pix_frame": 5.0,
    }
    run_config = {
        "spec": spec,
        "args": {"frames": 3},
        "selected_frame_indices": [0, 1, 2],
    }
    (run_dir / "run_config.json").write_text(json.dumps(run_config), encoding="utf-8")
    np.savez_compressed(
        cache_dir / f"stars_{run_dir.name}.npz",
        source_id=np.array([101, 102], dtype=np.int64),
        x0=np.array([-1.0, 2.0]),
        y0=np.array([3.0, -4.0]),
        detector_xpix=np.array([8.5, 11.5]),
        detector_ypix=np.array([6.0, 1.0]),
        detector_xpix_shifted=np.array([8.5, 11.5]),
        detector_ypix_shifted=np.array([6.0, 1.0]),
        et_mag=np.array([20.0, 21.0]),
        gmag=np.array([20.0, 21.0]),
        ra=np.array([304.0, 305.0]),
        dec=np.array([51.0, 52.0]),
        field_angle_deg=np.array([12.0, 12.0]),
    )
    np.savez_compressed(
        run_dir / "effects_timeseries.npz",
        time_s=np.array([0.0, 10.0, 20.0]),
        total_motion_pix=np.array([[0.0, 0.0], [0.5, -1.0], [2.0, 3.0]]),
        psd_drift_pix=np.array([[0.0, 0.0], [0.1, 0.2], [0.3, 0.4]]),
        dva_drift_pix=np.array([[0.0, 0.0], [0.01, 0.02], [0.03, 0.04]]),
        thermal_drift_pix=np.array([[0.0, 0.0], [0.001, 0.002], [0.003, 0.004]]),
        momentum_dump_pix=np.array([[0.0, 0.0], [1.0, 2.0], [3.0, 4.0]]),
        psf_scale=np.array([1.0, 0.99, 1.01]),
    )
    return run_dir


def test_frame_truth_dataframe_reconstructs_positions_flux_and_step_light(tmp_path):
    run_dir = _write_minimal_run(tmp_path)
    context = exporter.load_run_context(run_dir)

    frame_1 = exporter.build_frame_truth_dataframe(context, 1)
    frame_2 = exporter.build_frame_truth_dataframe(context, 2)

    expected_rate = exporter.et_mag_to_photon_rate_e_s(np.array([20.0]))[0]
    expected_count = expected_rate * 10.0

    assert list(frame_1["source_id"]) == [101, 102]
    assert frame_1.loc[0, "x_detector_truth_pix"] == 9.0
    assert frame_1.loc[0, "y_detector_truth_pix"] == 5.0
    assert frame_1.loc[0, "x0_truth_centered_pix"] == -0.5
    assert frame_1.loc[0, "y0_truth_centered_pix"] == 2.0
    assert math.isclose(frame_1.loc[0, "photon_rate_e_s"], expected_rate)
    assert math.isclose(frame_1.loc[0, "photon_count_e_frame"], expected_count)
    assert math.isclose(frame_1.loc[0, "ideal_photon_snr"], math.sqrt(expected_count))
    assert frame_1.loc[0, "scattered_light_e_s_pix"] == 0.2
    assert frame_1.loc[0, "scattered_light_e_pix_frame"] == 2.0
    assert frame_2.loc[0, "scattered_light_e_s_pix"] == 0.7
    assert frame_2.loc[0, "scattered_light_e_pix_frame"] == 7.0


def test_export_run_writes_one_csv_per_requested_frame(tmp_path):
    run_dir = _write_minimal_run(tmp_path)
    output_dir = run_dir / "truth_tables"

    written = exporter.export_run_truth_tables(
        run_dir,
        output_dir=output_dir,
        frame_indices=[1, 2],
        overwrite=False,
    )

    assert [path.name for path in written] == ["frame_000001.csv", "frame_000002.csv"]
    first = pd.read_csv(output_dir / "frame_000001.csv")
    assert len(first) == 2
    assert {
        "run_name",
        "frame_index",
        "source_id",
        "x_detector_truth_pix",
        "photon_count_e_frame",
        "ideal_photon_snr",
    }.issubset(first.columns)
