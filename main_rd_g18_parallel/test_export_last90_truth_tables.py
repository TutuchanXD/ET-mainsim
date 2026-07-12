from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from astropy import units as u

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


def _write_package_timeline_run(tmp_path: Path) -> Path:
    from photsim7.catalog_sources import PreparedStarCatalog, StarCatalogCache
    from photsim7.dynamic_effect_models import build_frame_timing
    from photsim7.dynamic_effects import (
        EffectComponent,
        EffectSourceGeometry,
        EffectTimeseries,
        ReferenceFieldProjector,
    )
    from photsim7.spec_factories import make_et_main_detector_spec
    from photsim7.specs import CatalogSpec

    run_dir = tmp_path / "main_rd_package_timeline"
    cache_path = run_dir / "cache" / "stars_main_rd_package_timeline.npz"
    base = make_et_main_detector_spec(shape=(10, 20), run_seed=20260712)
    spec = replace(
        base,
        observation=replace(
            base.observation,
            exposure_duration=10 * u.s,
            readout_duration=0 * u.s,
            observing_duration=20 * u.s,
            n_frames=2,
        ),
        catalog=CatalogSpec(
            source_type="detector_xy_csv",
            source_path="unused.csv",
            source_id_column="source_id",
            magnitude_column="gmag",
            x_column="x0",
            y_column="y0",
            input_magnitude_system="Gaia_G",
            photon_magnitude_system="ET",
            background_stars_max_mag=17.0,
            target_ra_deg=10.0,
            target_dec_deg=20.0,
            query_options={
                "reference_field_angle_deg": 8.0,
                "reference_field_polar_angle_rad": 0.5,
            },
        ),
    )
    catalog = PreparedStarCatalog(
        star_data={
            "source_id": np.array([101, 102], dtype=np.int64),
            "x0": np.array([-1.0, 2.0]),
            "y0": np.array([3.0, -4.0]),
            "frame_xpix": np.array([8.5, 11.5]),
            "frame_ypix": np.array([7.5, 0.5]),
            "ra": np.array([10.0, 10.1]),
            "dec": np.array([20.0, 20.1]),
            "gaia_g_mag": np.array([12.0, 13.0]),
        },
        metadata={
            "source": {"type": "detector_xy_csv", "n_sources": 2},
            "default_field_angle_deg": 8.0,
        },
    )
    StarCatalogCache.write(cache_path, catalog)
    source_geometry = EffectSourceGeometry.from_mapping(catalog.star_data)
    projector = ReferenceFieldProjector(
        field_angle_deg=8.0,
        field_polar_angle_rad=0.5,
        pixel_scale_arcsec_per_pix=4.83,
    )
    timeline = EffectTimeseries(
        timing=build_frame_timing(n_frames=2, integration_s=10.0),
        components=(
            EffectComponent(
                name="momentum_dump",
                values=np.array([[0.0, 0.0], [0.5, -1.0]]),
                unit="pix",
                coordinate_frame="renderer_frame_pixel_xy",
                scope="global",
                axes=("frame", "xy"),
                model_id="test",
            ),
            EffectComponent(
                name="psf_breathing",
                values=np.array([1.0, 0.99]),
                unit="dimensionless",
                coordinate_frame="dimensionless_psf_scale",
                scope="global",
                axes=("frame",),
                model_id="test",
            ),
        ),
        source_geometry=source_geometry,
        projector=projector,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_config.json").write_text(
        json.dumps(
            {
                "spec": {
                    "frame_rows": 10,
                    "frame_cols": 20,
                    "run_label": run_dir.name,
                    "exposure_s": 10.0,
                },
                "simulation_spec": spec.to_json_dict(),
                "star_cache": str(cache_path),
                "selected_frame_indices": [0, 1],
            }
        ),
        encoding="utf-8",
    )
    np.savez_compressed(run_dir / "effects_timeseries.npz", **timeline.to_arrays())
    (run_dir / "effects_timeseries.metadata.json").write_text(
        json.dumps(timeline.to_metadata()),
        encoding="utf-8",
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


def test_package_timeline_projects_per_source_truth_and_uses_typed_photometry(
    tmp_path,
):
    from photsim7.photometry import et_mag_to_detected_electron_rate

    run_dir = _write_package_timeline_run(tmp_path)

    context = exporter.load_run_context(run_dir)
    frame = exporter.build_frame_truth_dataframe(context, 1)

    np.testing.assert_allclose(frame["motion_offset_x_pix"], [0.5, 0.5])
    np.testing.assert_allclose(frame["motion_offset_y_pix"], [-1.0, -1.0])
    np.testing.assert_allclose(frame["x_detector_truth_pix"], [9.0, 12.0])
    np.testing.assert_allclose(frame["y_detector_truth_pix"], [6.5, -0.5])
    np.testing.assert_allclose(frame["psf_scale"], [0.99, 0.99])
    expected_rate = et_mag_to_detected_electron_rate(np.array([12.0, 13.0])).value
    np.testing.assert_allclose(frame["photon_rate_e_s"], expected_rate)
    assert context.effect_schema_id == "photsim7.effect_timeseries.v1"
