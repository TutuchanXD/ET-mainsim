import json
from dataclasses import replace

import pytest
from astropy import units as u
from openpyxl import Workbook

from et_mainsim.presets import load_preset
from et_mainsim.workflows.full_frame import build_run_plan


def test_cli_workbook_seed_precedence_preserves_stamp_scene(tmp_path, capsys):
    from et_mainsim.cli import main

    path = tmp_path / "params.xlsx"
    book = Workbook()
    book.active.append(["Group", "Parameter", "Value", "Unit"])
    for name, value in [
        ("Workbook Schema Version", 4),
        ("Run Seed", "17"),
        ("RNG Seed readout.gaussian", "0"),
    ]:
        book.active.append(["RNG", name, value, None])
    book.save(path)
    book.close()
    assert (
        main(
            [
                "run",
                "et-stamp",
                "--preset",
                "smoke",
                "--spec",
                str(path),
                "--seed",
                "19",
                "--output-root",
                str(tmp_path / "out"),
                "--dry-run",
            ]
        )
        == 0
    )
    spec = json.loads(capsys.readouterr().out)["simulation_spec"]
    assert spec["psf"]["mode"] == "stamp"
    assert spec["catalog"]["source_type"] == "detector_xy_csv"
    assert spec["rng"]["run_seed"] == 19
    assert spec["rng"]["stream_seeds"]["readout.gaussian"] == 0
    assert not (tmp_path / "out").exists()


def test_planning_preserves_explicit_observation_and_user_device(tmp_path):
    loaded = load_preset("et-full-frame-smoke")
    spec = replace(
        loaded.simulation_spec,
        observation=replace(
            loaded.simulation_spec.observation,
            n_frames=2,
            observing_duration=20 * u.s,
            frame_start_s=(0.0, 9.0),
        ),
        psf=replace(loaded.simulation_spec.psf, compute_device="cuda"),
    )
    # Time selection is an execution control, not a new observation.
    plan = build_run_plan(
        preset_name=loaded.descriptor.name,
        run_config=loaded.run_config,
        spec=spec,
        repo_root=tmp_path,
        cwd=tmp_path,
        frame_indices="1",
    )
    assert plan.spec.observation == spec.observation
    assert plan.frame_indices == (1,)


@pytest.mark.parametrize(
    "name",
    [
        "et-full-frame-smoke",
        "et-full-frame-production",
        "et-stamp-smoke",
        "et-stamp-production",
    ],
)
def test_presets_accept_mutable_assets_and_inherit_cosmic_seeds(name):
    spec = load_preset(name).simulation_spec
    assert spec.psf.bundle_sha256 is None
    psd = spec.dynamic_effects.psd_motion
    assert not psd.native_jitter_bank_path
    assert psd.native_jitter_bank_sha256 is None
    assert spec.cosmic_rays.seed is None
    assert spec.rng.determinism_mode == "bitwise"


def test_cli_partial_execution_config_inherits_preset_and_explicit_device_wins(
    tmp_path, capsys
):
    from et_mainsim.cli import main

    preset = load_preset("et-full-frame-smoke")
    spec = replace(
        preset.simulation_spec,
        psf=replace(preset.simulation_spec.psf, compute_device="cuda"),
    )
    path = tmp_path / "science.json"
    path.write_text(spec.to_json())
    config = tmp_path / "run.toml"
    config.write_text(
        '[execution]\npreview_count = 3\nbackend = "local-subprocess"\ngpu_ids = ["0"]\n'
    )
    args = [
        "run",
        "et-full-frame",
        "--spec",
        str(path),
        "--config",
        str(config),
        "--dry-run",
    ]
    assert main(args) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["execution"]["preview_count"] == 3
    assert plan["simulation_spec"]["psf"]["compute_device"] == "cuda"
    assert plan["execution"]["device"] == "cuda"
    assert main([*args, "--device", "cpu"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["simulation_spec"]["psf"]["compute_device"] == "cpu"


def test_stamp_workbook_can_explicitly_choose_a_coadd_size_for_500_frames(
    tmp_path, capsys
):
    from et_mainsim.cli import main

    path = tmp_path / "params.xlsx"
    book = Workbook()
    book.active.append(["Group", "Parameter", "Value", "Unit"])
    book.active.append(["Observation", "Workbook Schema Version", 4, None])
    book.active.append(["Observation", "Observing Duration", 5000, "s"])
    book.save(path)
    book.close()
    assert (
        main(
            [
                "run",
                "et-stamp",
                "--preset",
                "production",
                "--spec",
                str(path),
                "--coadd-size",
                "10",
                "--dry-run",
            ]
        )
        == 0
    )
    plan = json.loads(capsys.readouterr().out)
    assert plan["frame_plan"]["raw_frame_count"] == 500
    assert plan["frame_plan"]["coadd_count"] == 50
    assert plan["simulation_spec"]["observation"]["n_raw_frames_per_coadd"] == 10
