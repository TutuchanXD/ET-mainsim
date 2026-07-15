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
    table.write_text(
        "gaia_g_mag,psf_id,curve_id\n12.0,0,sn\n", encoding="utf-8"
    )
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
            "--artifact-profile",
            "compact",
            "--write-batch-size",
            "64",
            "--frames",
            "4",
            "--coadd-shard-index",
            "1",
            "--coadd-shard-count",
            "2",
            "--dry-run",
        ]
    )

    assert code == 0
    assert not output_root.exists()
    plan = json.loads(capsys.readouterr().out)
    assert plan["workflow"] == "et-stamp"
    assert plan["workload"]["input_mode"] == "table"
    assert plan["workload"]["include_neighbors"] is False
    assert plan["workload"]["artifact_profile"] == "compact"
    assert plan["workload"]["write_batch_size"] == 64
    assert plan["workload"]["coadd_shard_index"] == 1
    assert plan["workload"]["coadd_shard_count"] == 2
    assert plan["frame_plan"]["global_coadd_count"] == 2
    assert plan["frame_plan"]["coadd_indices"] == [1]
    assert plan["frame_plan"]["raw_frame_indices"] == [2, 3]
    assert plan["run_dir"].endswith(
        "/et-stamp-smoke/coadd_shard_0001_of_0002"
    )
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
    assert sum(item["enabled"] for item in effects) == 23
    assert sum(not item["enabled"] for item in effects) == 4
