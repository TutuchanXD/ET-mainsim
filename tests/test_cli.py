from __future__ import annotations

import json

import pytest


def test_cli_lists_and_shows_presets(capsys) -> None:
    from et_mainsim.cli import main

    assert main(["presets"]) == 0
    listed = capsys.readouterr().out
    assert "et-full-frame-smoke" in listed
    assert "et-full-frame-production" in listed
    assert "et-stamp-production" in listed
    assert "legacy-sim-full-effects-production" in listed

    assert main(["show", "et-full-frame-smoke", "--format", "json"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["preset"]["name"] == "et-full-frame-smoke"
    assert shown["simulation_spec"]["detector"]["shape"] == [64, 64]
    assert shown["run_config"]["workflow"] == "et-full-frame"


def test_cli_dry_run_is_read_only_and_resolves_epoch(tmp_path, capsys) -> None:
    from et_mainsim.cli import main

    output_root = tmp_path / "must-not-exist"
    code = main(
        [
            "run",
            "et-full-frame",
            "--preset",
            "smoke",
            "--output-root",
            str(output_root),
            "--frames",
            "2",
            "--target-epoch-jyear",
            "2028.5",
            "--dry-run",
        ]
    )

    assert code == 0
    assert not output_root.exists()
    plan = json.loads(capsys.readouterr().out)
    assert plan["dry_run"] is True
    assert plan["frame_plan"]["requested"] == [0, 1]
    assert plan["simulation_spec"]["catalog"]["target_epoch_jyear"] == 2028.5


def test_module_entrypoint_help_is_available_without_runtime_assets(capsys) -> None:
    from et_mainsim.cli import main

    try:
        main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0

    assert "ET-mainsim reference workflows" in capsys.readouterr().out


def test_cli_reports_preflight_errors_without_traceback(tmp_path, capsys) -> None:
    from et_mainsim.cli import main

    with pytest.raises(SystemExit) as raised:
        main(
            [
                "run",
                "et-full-frame",
                "--preset",
                "smoke",
                "--data-root",
                str(tmp_path / "missing-data"),
                "--output-root",
                str(tmp_path / "output"),
            ]
        )

    assert raised.value.code == 2
    error = capsys.readouterr().err
    assert "Photsim7 data root does not exist" in error
    assert "Traceback" not in error


def test_cli_stamp_table_dry_run_is_read_only_and_query_independent(
    tmp_path, capsys
) -> None:
    from et_mainsim.cli import main

    table = tmp_path / "targets.csv"
    table.write_text("gaia_g_mag,psf_id,curve_id\n12.0,0,sn\n", encoding="utf-8")
    curves = tmp_path / "curves.csv"
    curves.write_text(
        "curve_id,frame_index,relative_flux\nsn,0,1\nsn,1,2\n",
        encoding="utf-8",
    )
    output_root = tmp_path / "must-not-exist"

    code = main(
        [
            "run",
            "et-stamp",
            "--preset",
            "smoke",
            "--input-table",
            str(table),
            "--variability-table",
            str(curves),
            "--output-root",
            str(output_root),
            "--write-batch-size",
            "7",
            "--dry-run",
        ]
    )

    assert code == 0
    assert not output_root.exists()
    plan = json.loads(capsys.readouterr().out)
    assert plan["workflow"] == "et-stamp"
    assert plan["workload"]["input_mode"] == "table"
    assert plan["workload"]["include_neighbors"] is False
    assert plan["workload"]["write_batch_size"] == 7
    assert plan["simulation_spec"]["catalog"]["source_type"] == "prepared"
    assert plan["input_table"] == str(table)
    assert plan["variability_table"] == str(curves)


def test_cli_legacy_dry_run_exposes_full_effect_workload(tmp_path, capsys) -> None:
    from et_mainsim.cli import main

    output_root = tmp_path / "must-not-exist"
    code = main(
        [
            "run",
            "legacy-sim",
            "--preset",
            "full-effects-smoke",
            "--output-root",
            str(output_root),
            "--stars-per-run",
            "3",
            "--dry-run",
        ]
    )

    assert code == 0
    assert not output_root.exists()
    plan = json.loads(capsys.readouterr().out)
    assert plan["workflow"] == "legacy-sim"
    assert plan["workload"]["stars_per_run"] == 3
    effects = plan["effect_contract"]["effects"]
    effects_by_id = {item["effect_id"]: item for item in effects}
    expected_enabled = {
        "scene.target_star",
        "scene.background_stars",
        "noise.stellar_photon",
        "noise.background",
        "noise.scattered",
        "noise.dark_current",
        "noise.readout",
        "gain.scripted",
        "gain.whole_pixel_normal",
        "gain.whole_pixel_sinusoidal",
        "motion.et_psd_low_frequency",
        "motion.dva",
        "motion.thermal",
        "motion.momentum_dump",
        "psf.jitter_integrated_high_frequency",
        "psf.breathing",
        "pixel_response.inter_pixel",
        "pixel_response.intra_pixel",
        "pixel_response.pixel_phase",
        "reduction.coadding",
        "reduction.kepler_optimal_aperture",
        "reduction.oa_helper_variants",
    }
    expected_disabled = {
        "motion.tess_roll_low_frequency",
        "calibration.flat_field",
        "filter.pixel_flux",
        "scene.transit_injection",
        "detector.cosmic_rays",
    }
    assert {
        effect_id for effect_id, effect in effects_by_id.items() if effect["enabled"]
    } == expected_enabled
    assert {
        effect_id
        for effect_id, effect in effects_by_id.items()
        if not effect["enabled"]
    } == expected_disabled
    assert set(effects_by_id) == expected_enabled | expected_disabled
    tess_roll = effects_by_id["motion.tess_roll_low_frequency"]
    assert tess_roll["parameters"] == {
        "asset": "",
        "profile": "et_attitude_xyz",
        "reason": "profile 'et_attitude_xyz' does not consume a TESS PSD",
        "xy_amplitude_multiplier": 1.0,
    }
