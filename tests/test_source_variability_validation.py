from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from astropy.table import Table


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "validate_source_variability_stamp.py"
)
SPEC = importlib.util.spec_from_file_location(
    "validate_source_variability_stamp",
    SCRIPT_PATH,
)
assert SPEC is not None and SPEC.loader is not None
validation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validation)


def _write_sn_csv(path: Path) -> None:
    Table(
        {
            "mag_clean": [20.0, 19.0, 18.0, 18.5],
            "time": [-3.0, -2.0, -1.0, 0.0],
            "observer_time": [-3.3, -2.2, -1.1, 0.0],
            "band": ["gaia_g_3260_9290"] * 4,
            "zpsys": ["ab"] * 4,
        }
    ).write(path, format="ascii.csv")


def _option_value(command: tuple[str, ...], option: str) -> str:
    index = command.index(option)
    return command[index + 1]


def test_sn_magnitude_conversion_uses_selected_rows_and_ignores_input_times(
    tmp_path: Path,
) -> None:
    source = tmp_path / "sn.csv"
    _write_sn_csv(source)

    curve = validation.load_sn_lightcurve(source, frame_count=3)

    assert curve.baseline_gaia_g_mag == 18.0
    np.testing.assert_allclose(
        curve.relative_flux,
        10.0 ** (-0.4 * (np.array([20.0, 19.0, 18.0]) - 18.0)),
    )
    assert curve.frame_indices == (0, 1, 2)
    assert curve.selected_source_rows == (0, 1, 2)
    assert curve.ignored_input_time_columns == ("time", "observer_time")
    assert curve.source_sha256 == hashlib.sha256(source.read_bytes()).hexdigest()


def test_validation_ecsv_contract_and_provenance_are_explicit(tmp_path: Path) -> None:
    source = tmp_path / "sn.csv"
    _write_sn_csv(source)
    curve = validation.load_sn_lightcurve(source, frame_count=3)

    paths = validation.write_validation_inputs(curve, tmp_path / "inputs")

    injected = Table.read(paths.injected_target, format="ascii.ecsv")
    static = Table.read(paths.static_target, format="ascii.ecsv")
    variability = Table.read(paths.variability, format="ascii.ecsv")
    assert injected.colnames == [
        "source_id",
        "gaia_g_mag",
        "curve_id",
        "ra_deg",
        "dec_deg",
    ]
    assert static.colnames == ["source_id", "gaia_g_mag", "ra_deg", "dec_deg"]
    assert variability.colnames == ["curve_id", "frame_index", "relative_flux"]
    assert int(injected["source_id"][0]) == validation.SOURCE_ID
    assert float(injected["ra_deg"][0]) == validation.TARGET_RA_DEG
    assert float(injected["dec_deg"][0]) == validation.TARGET_DEC_DEG
    assert float(injected["gaia_g_mag"][0]) == curve.baseline_gaia_g_mag
    assert str(injected["curve_id"][0]) == validation.CURVE_ID
    assert "curve_id" not in static.colnames
    np.testing.assert_allclose(variability["relative_flux"], curve.relative_flux)
    np.testing.assert_array_equal(variability["frame_index"], [0, 1, 2])

    for table in (injected, static, variability):
        assert table.meta["source_sha256"] == curve.source_sha256
        assert table.meta["source_band"] == "gaia_g_3260_9290"
        assert table.meta["source_zpsys"] == "ab"
        assert table.meta["magnitude_semantics"] == "Gaia_G_Vega"
        assert table.meta["magnitude_semantics_note"] == (
            "truncated_gaia_g_ab_treated_as_gaia_g_vega_engineering_proxy"
        )
        assert table.meta["selected_source_rows"] == [0, 1, 2]
        assert table.meta["relative_flux_formula"] == (
            "10**(-0.4*(mag_clean-min(selected_mag_clean)))"
        )
        assert table.meta["time_alignment"] == "simulation_raw_frame_index"
        assert table.meta["ignored_input_time_columns"] == [
            "time",
            "observer_time",
        ]

    from et_mainsim.stamp_inputs import load_stamp_variability_table

    loaded = load_stamp_variability_table(paths.variability, raw_frame_count=3)
    np.testing.assert_allclose(loaded.curves[validation.CURVE_ID], curve.relative_flux)


def test_short_spec_changes_only_production_observation_length(tmp_path: Path) -> None:
    from astropy import units as u
    from photsim7.specs import SimulationSpec

    from et_mainsim.presets import load_preset

    destination = validation.write_short_stamp_spec(
        tmp_path / "validation.spec.json",
        frame_count=22,
        coadd_size=2,
    )
    short = SimulationSpec.from_json(destination.read_text(encoding="utf-8"))
    production = load_preset("et-stamp-production").simulation_spec

    assert short.observation.n_frames == 22
    assert short.observation.resolved_n_frames == 22
    assert short.observation.observing_duration.to_value(u.s) == 220.0
    assert short.observation.n_raw_frames_per_coadd == 2
    short_payload = short.to_json_dict()
    production_payload = production.to_json_dict()
    short_payload.pop("observation")
    production_payload.pop("observation")
    assert short_payload == production_payload


def test_aperture_photometry_and_paired_metrics_recover_known_signal() -> None:
    shape = (15, 15)
    yy, xx = np.indices(shape, dtype=np.float64)
    rr = np.hypot(xx - 7.0, yy - 7.0)
    image = np.full(shape, 10.0)
    image[rr <= 4.0] += 2.5

    measurement = validation.fixed_aperture_photometry(image)

    assert measurement.aperture_pixel_count == int(np.count_nonzero(rr <= 4.0))
    assert measurement.background_median == 10.0
    assert measurement.flux == 2.5 * measurement.aperture_pixel_count
    assert measurement.nan_pixel_count == 0
    assert measurement.saturated_pixel_count == 0

    factor = np.array([0.2, 0.6, 1.0, 0.8], dtype=np.float64)
    static_final = np.array([1000.0, 1100.0, 900.0, 1000.0])
    injected_final = static_final + 1000.0 * (factor - 1.0)
    static_stellar = np.full(4, 750.0)
    injected_stellar = static_stellar * factor
    metrics, series = validation.compute_paired_metrics(
        relative_flux=factor,
        static_final_flux=static_final,
        injected_final_flux=injected_final,
        static_stellar_flux=static_stellar,
        injected_stellar_flux=injected_stellar,
        static_nan_pixel_count=np.zeros(4, dtype=np.int64),
        injected_nan_pixel_count=np.zeros(4, dtype=np.int64),
        static_saturated_pixel_count=np.zeros(4, dtype=np.int64),
        injected_saturated_pixel_count=np.zeros(4, dtype=np.int64),
    )

    np.testing.assert_allclose(series["paired_residual"], factor - 1.0)
    np.testing.assert_allclose(series["residual_error"], 0.0, atol=1e-14)
    assert metrics["stellar_mean_ratio_max_abs_error"] < 1e-14
    assert metrics["final_paired_pearson_r"] == 1.0
    assert abs(metrics["final_paired_slope"] - 1.0) < 1e-14
    assert metrics["final_paired_rmse"] < 1e-14
    assert metrics["has_nan"] is False
    assert metrics["has_saturation"] is False
    assert metrics["acceptance"]["stellar_mean_ratio_error_le_1e-4"] is True
    assert metrics["acceptance"]["final_paired_pearson_r_ge_0_95"] is True
    assert metrics["acceptance"]["no_nan"] is True
    assert metrics["acceptance"]["no_saturation"] is True


def test_analysis_records_the_requested_curve_id(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "sn.csv"
    _write_sn_csv(source)
    curve = validation.load_sn_lightcurve(source, frame_count=4)
    factor = curve.relative_flux
    static = validation.RunPhotometry(
        frame_indices=curve.frame_indices,
        final_flux=np.full(4, 1000.0),
        stellar_flux=np.full(4, 750.0),
        nan_pixel_count=np.zeros(4, dtype=np.int64),
        saturated_pixel_count=np.zeros(4, dtype=np.int64),
    )
    injected = validation.RunPhotometry(
        frame_indices=curve.frame_indices,
        final_flux=1000.0 * factor,
        stellar_flux=750.0 * factor,
        nan_pixel_count=np.zeros(4, dtype=np.int64),
        saturated_pixel_count=np.zeros(4, dtype=np.int64),
    )

    def fake_read(run_dir, **_kwargs):
        return injected if Path(run_dir).name == "injected" else static

    def fake_plot(_series, path):
        path.write_bytes(b"test plot")

    monkeypatch.setattr(validation, "read_run_photometry", fake_read)
    monkeypatch.setattr(validation, "_write_plot", fake_plot)
    products = validation.analyze_paired_runs(
        static_run_dir=tmp_path / "static",
        injected_run_dir=tmp_path / "injected",
        curve=curve,
        curve_id="science_team_curve",
        output_dir=tmp_path / "products",
    )

    metrics = json.loads(products.metrics_json.read_text(encoding="utf-8"))
    assert metrics["curve_id"] == "science_team_curve"


def test_commands_share_run_identity_seed_spec_and_stamp_chain(tmp_path: Path) -> None:
    inputs = validation.ValidationInputPaths(
        static_target=tmp_path / "static.ecsv",
        injected_target=tmp_path / "injected.ecsv",
        variability=tmp_path / "variability.ecsv",
        simulation_spec=tmp_path / "short.spec.json",
    )
    commands = validation.build_run_commands(
        inputs=inputs,
        output_root=tmp_path / "runs",
        data_root=tmp_path / "photsim-data",
        focalplane_registry=tmp_path / "et-focalplane",
        frame_count=22,
        run_id="paired-sn",
        seed=20260715,
        backend="local-subprocess",
        device="cuda",
        gpu_ids=("0",),
        workers_per_device=1,
    )

    for command in (commands.static, commands.injected):
        assert command[:5] == (
            validation.sys.executable,
            "-m",
            "et_mainsim",
            "run",
            "et-stamp",
        )
        assert _option_value(command, "--run-id") == "paired-sn"
        assert _option_value(command, "--seed") == "20260715"
        assert _option_value(command, "--spec") == str(inputs.simulation_spec)
        assert _option_value(command, "--frames") == "22"
        assert _option_value(command, "--stamp-rows") == "15"
        assert _option_value(command, "--stamp-cols") == "15"
        assert "--no-include-neighbors" in command
        assert "--save-raw" in command
        assert "--save-coadd" in command
        assert "--save-electron-components" in command
        assert "--overwrite" in command

    assert _option_value(commands.static, "--output-root") != _option_value(
        commands.injected,
        "--output-root",
    )
    assert _option_value(commands.static, "--input-table") == str(
        inputs.static_target
    )
    assert _option_value(commands.injected, "--input-table") == str(
        inputs.injected_target
    )
    assert "--variability-table" not in commands.static
    assert _option_value(commands.injected, "--variability-table") == str(
        inputs.variability
    )


def test_prepare_only_runs_directly_from_a_source_checkout(tmp_path: Path) -> None:
    source = tmp_path / "sn.csv"
    _write_sn_csv(source)
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--sn-csv",
            str(source),
            "--output-root",
            str(tmp_path / "output"),
            "--data-root",
            str(tmp_path / "photsim-data"),
            "--focalplane-registry",
            str(tmp_path / "et-focalplane"),
            "--frame-count",
            "4",
            "--coadd-size",
            "2",
            "--prepare-only",
        ],
        cwd=SCRIPT_PATH.parents[1],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert (tmp_path / "output" / "validation_inputs" / "commands.json").is_file()
