from __future__ import annotations

import json

import pytest


def test_cli_lists_and_shows_presets(capsys) -> None:
    from et_mainsim.cli import main

    assert main(["presets"]) == 0
    listed = capsys.readouterr().out
    assert "et-full-frame-smoke" in listed
    assert "et-full-frame-production" in listed

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
