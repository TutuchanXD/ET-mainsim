"""Exercise the application against real public upstream rendering/reduction."""

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from et_mainsim.legacy_expectation import (
    CONFIG_SCHEMA, LegacyExpectationConfig, build_legacy_expectation_plan,
    run_legacy_expectation, verify_legacy_expectation_run, read_legacy_expectation_group,
)


def make_inputs(root, kind="catalog-stamp", output_kind="pixels", scopes=6):
    from test_equivalent_coadd_workflow import make_inputs as equivalent_inputs
    from photsim7.specs import ScienceProfileSpec
    _, request, _ = equivalent_inputs(root, cadence=30)
    spec = request.derivation.source_spec
    spec = replace(spec,
        observation=replace(spec.observation, n_raw_frames_per_coadd=3),
        instrument=replace(spec.instrument, telescope_count=scopes),
        science_profile=ScienceProfileSpec(profile_id="legacy_expectation_electrons_v1", science_realization_id=17),
        sky=replace(spec.sky, subtract_nonstellar_mean=True),
    )
    (root / "spec.json").write_text(json.dumps(spec.to_json_dict()))
    (root / "source.json").write_text(json.dumps({"source_ids": [101], "relative_flux": [[1., 1., .5, .5, 1., 1.]]}))
    payload = dict(schema_id=CONFIG_SCHEMA, spec_path="spec.json", aperture_path="apertures.npz",
                   data_root="assets", output_root="results", run_id="expectation", kind=kind,
                   target_ids=[101], stamp_shape=[17, 17], output_kind=output_kind, cpu_threads=1,
                   absolute_raw_frame_start_index=60)
    if kind == "table-stamp":
        (root / "targets.csv").write_text("source_id,gaia_g_mag,psf_id,curve_id\n101,17,0,variable\n")
        (root / "variability.csv").write_text("curve_id,frame_index,relative_flux\n" + "".join(
            f"variable,{i},{v}\n" for i, v in enumerate([1., 1., .5, .5, 1., 1.])))
        payload.update(input_table_path="targets.csv", variability_table_path="variability.csv")
    else:
        payload.update(catalog_path="catalog.json", variability_path="source.json")
    shape = (65, 65) if kind == "full-frame" else (17, 17)
    weights = np.zeros(shape)
    y, x = (n // 2 for n in shape)
    weights[y-4:y+5, x-4:x+5] = 1
    np.savez(root / "apertures.npz", **{"101": weights})
    return LegacyExpectationConfig.from_mapping(payload, base=root)


def products(config):
    return [read_legacy_expectation_group(path) for path in sorted(
        (config.output_root / config.run_id / "groups").glob("group_*"))]


@pytest.mark.parametrize("kind", ["catalog-stamp", "full-frame", "table-stamp"])
@pytest.mark.parametrize("output_kind", ["pixels", "lightcurve"])
def test_real_three_entries_six_scopes_and_readback(tmp_path, kind, output_kind):
    config = make_inputs(tmp_path, kind, output_kind)
    plan = build_legacy_expectation_plan(config)
    assert plan["scope_count"] == 6
    assert plan["group_count"] == 2
    assert not (config.output_root / config.run_id).exists()
    result = run_legacy_expectation(config)
    assert result["status"] == "completed"
    assert verify_legacy_expectation_run(config) == result
    assert run_legacy_expectation(config) == result
    for index, product in enumerate(products(config)):
        arrays = product.arrays
        combined = "target_101/combined__"
        np.testing.assert_array_equal(arrays[combined + "contributing_scope_count"], [6])
        np.testing.assert_array_equal(arrays[combined + "frame_start_s"], [600. + 30. * index])
        np.testing.assert_array_equal(arrays[combined + "effective_integration_s"], [180.])
        np.testing.assert_array_equal(arrays[combined + "flux_electrons"],
            arrays[combined + "scope_flux_electrons"].sum(0))
        assert arrays[combined + "quality"].dtype == np.uint32
        assert np.isnan(arrays[combined + "normalized_flux"]).all()
        single = next(m for k, m in product.manifest["measurements"].items() if "/scope_" in k)
        coadd = single["provenance"]["coadd_manifests"][0]
        assert coadd["unit"] == "electron"
        assert coadd["absolute_raw_frame_indices"] == list(range(60+index*3, 63+index*3))
        assert coadd["nonstellar_mean_policy"] == "subtract_sky_scattered_dark"
        assert coadd["raw_expectations_saved"] is False
        assert product.manifest["remeasurement_supported"] == (output_kind == "pixels")
        assert any("coadds/" in k for k in product.manifest["files"]) == (output_kind == "pixels")


def test_shards_reverse_order_and_resume_equal_serial(tmp_path):
    config = make_inputs(tmp_path, output_kind="lightcurve")
    run_legacy_expectation(config)
    other = replace(config, run_id="sharded")
    assert run_legacy_expectation(other, shard_index=1, shard_count=2)["status"] == "partial"
    with pytest.raises(ValueError, match="incomplete"):
        verify_legacy_expectation_run(other)
    assert run_legacy_expectation(other, group_indices=[1, 0])["status"] == "completed"
    for first, second in zip(products(config), products(other)):
        assert first.arrays.keys() == second.arrays.keys()
        for name in first.arrays:
            np.testing.assert_array_equal(first.arrays[name], second.arrays[name], err_msg=name)


def test_mid_group_failure_is_not_published_and_can_resume(tmp_path, monkeypatch):
    import et_mainsim.legacy_expectation as workflow
    config = make_inputs(tmp_path, output_kind="lightcurve", scopes=1)
    original = workflow._produce_group
    def fail_second(prepared, index, directory, services):
        original(prepared, index, directory, services)
        if index == 1:
            raise RuntimeError("simulated interruption before commit")
    monkeypatch.setattr(workflow, "_produce_group", fail_second)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_legacy_expectation(config)
    assert len(products(config)) == 1
    assert not list((config.output_root / config.run_id / ".staging").iterdir())
    monkeypatch.setattr(workflow, "_produce_group", original)
    assert run_legacy_expectation(config)["status"] == "completed"


@pytest.mark.parametrize("damage", ["missing", "corrupt", "extra", "manifest", "input", "symlink", "group_manifest", "whole_group"])
def test_resume_rejects_missing_corrupt_or_conflicting_data(tmp_path, damage):
    config = make_inputs(tmp_path, output_kind="lightcurve", scopes=1)
    run_legacy_expectation(config)
    run = config.output_root / config.run_id
    group = run / "groups/group_00000000"
    if damage == "missing":
        (group / "light_curves.npz").unlink()
    elif damage == "corrupt":
        (group / "light_curves.npz").write_bytes(b"corrupt")
    elif damage == "extra":
        (group / "extra").write_text("untracked")
    elif damage == "manifest":
        payload = json.loads((run / "run_manifest.json").read_text())
        payload["profile_id"] = "final_dn"
        (run / "run_manifest.json").write_text(json.dumps(payload))
    elif damage == "input":
        config.variability_path.write_text(json.dumps({"source_ids": [101], "relative_flux": [[1.]*6]}))
    elif damage == "group_manifest":
        payload = json.loads((group / "group.json").read_text())
        payload["profile_id"] = "final_dn"
        (group / "group.json").write_text(json.dumps(payload))
    elif damage == "whole_group":
        import shutil
        shutil.rmtree(group)
    else:
        original = group / "light_curves.npz"
        backup = tmp_path / "backup.npz"
        original.rename(backup)
        original.symlink_to(backup)
    with pytest.raises(ValueError):
        run_legacy_expectation(config)


def test_raw_expectations_and_public_remeasurement(tmp_path):
    from photsim7.analysis import read_reference_product, measure_reference_photometry, ReferenceAperture
    config = replace(make_inputs(tmp_path, scopes=1), save_raw_expectations=True)
    run_legacy_expectation(config, group_indices=[0])
    group = config.output_root / config.run_id / "groups/group_00000000"
    coadd_path = next((group / "coadds").glob("*/scope_*/arrays.npz"))
    with np.load(coadd_path) as arrays:
        assert any(k.startswith("raw_000002_") for k in arrays.files)
    directory = next((group / "photometry/target_101").glob("scope_*"))
    reference = read_reference_product(directory)
    a, m = reference.arrays, reference.manifest
    measured = measure_reference_photometry(a["images_electrons"], aperture=ReferenceAperture(a["aperture_weights"]),
        frame_start_s=a["frame_start_s"], frame_stop_s=a["frame_stop_s"], integration_s=a["integration_s"],
        time_origin=m["time_origin"], scope_id=m["scope_id"], target_id=m["target_id"],
        valid_mask=a["valid_pixel_mask"], pixel_origin_xy=m["pixel_origin_xy"])
    np.testing.assert_array_equal(measured.arrays["flux_electrons"], a["flux_electrons"])


@pytest.mark.parametrize("changes", [
    {"kind": "unknown"}, {"save_raw_expectations": 1}, {"target_ids": [True]},
    {"stamp_shape": [1]}, {"cpu_threads": 0}, {"run_id": "../escape"},
    {"run_id": ".run_locks"}, {"output_kind": "dn"}, {"unknown": 1},
    {"output_kind": "lightcurve", "save_raw_expectations": True},
])
def test_invalid_config(tmp_path, changes):
    config = make_inputs(tmp_path)
    with pytest.raises((ValueError, TypeError)):
        LegacyExpectationConfig.from_mapping({**config.to_dict(), **changes})


def test_cli_dry_run(tmp_path, capsys):
    from et_mainsim.cli import main
    config = make_inputs(tmp_path)
    path = tmp_path / "run.json"
    path.write_text(json.dumps(config.to_dict()))
    assert main(["run", "et-legacy-expectation", "--config", str(path), "--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)["profile_id"] == "legacy_expectation_electrons_v1"


def test_full_frame_and_catalog_stamp_share_pixels_and_measurements(tmp_path):
    catalog = make_inputs(tmp_path, scopes=6)
    full_aperture = tmp_path / "full_aperture.npz"
    weights = np.zeros((65, 65))
    weights[28:37, 28:37] = 1
    np.savez(full_aperture, **{"101": weights})
    full = replace(catalog, kind="full-frame", run_id="full", aperture_path=full_aperture)
    run_legacy_expectation(catalog, group_indices=[0])
    run_legacy_expectation(full, group_indices=[0])
    first, second = products(catalog)[0], products(full)[0]
    for name in first.arrays:
        # Summation over differently sized zero-padded images can differ at
        # roundoff, while the actual common image pixels must match exactly.
        np.testing.assert_allclose(first.arrays[name], second.arrays[name], rtol=1e-14, atol=1e-10, err_msg=name)
    for scope in range(6):
        stamp_dir = catalog.output_root / catalog.run_id / f"groups/group_00000000/coadds/target_101/scope_{scope}"
        full_dir = full.output_root / full.run_id / f"groups/group_00000000/coadds/target_None/scope_{scope}"
        with np.load(stamp_dir / "arrays.npz") as a, np.load(full_dir / "arrays.npz") as b:
            for name in a.files:
                np.testing.assert_array_equal(a[name], b[name][24:41, 24:41], err_msg=name)


def test_table_reordering_keeps_per_target_realization(tmp_path):
    config = make_inputs(tmp_path, kind="table-stamp", output_kind="lightcurve", scopes=2)
    config.input_table_path.write_text("source_id,gaia_g_mag,psf_id,curve_id\n101,17,0,variable\n202,18,0,\n")
    with np.load(config.aperture_path) as bundle:
        weights = bundle["101"]
    np.savez(config.aperture_path, **{"101": weights, "202": weights})
    config = replace(config, target_ids=(101, 202))
    run_legacy_expectation(config)
    original = products(config)
    config.input_table_path.write_text("source_id,gaia_g_mag,psf_id,curve_id\n202,18,0,\n101,17,0,variable\n")
    other = replace(config, run_id="reordered", target_ids=(202, 101))
    run_legacy_expectation(other, group_indices=[1, 0])
    for first, second in zip(original, products(other)):
        for name in first.arrays:
            np.testing.assert_array_equal(first.arrays[name], second.arrays[name], err_msg=name)
    single = original[0].manifest["measurements"]
    a = single["target_101/scope_0"]["provenance"]["coadd_manifests"][0]
    b = single["target_202/scope_0"]["provenance"]["coadd_manifests"][0]
    assert a["rng_trace"]["scope"]["science_realization_id"] != b["rng_trace"]["scope"]["science_realization_id"]


def test_disjoint_process_shards_match_serial(tmp_path):
    import os
    import subprocess
    import sys
    config = make_inputs(tmp_path, output_kind="lightcurve", scopes=2)
    run_legacy_expectation(config)
    other = replace(config, run_id="process-shards")
    path = tmp_path / "parallel.json"
    path.write_text(json.dumps(other.to_dict()))
    environment = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
    processes = [subprocess.Popen([sys.executable, "-m", "et_mainsim", "run", "et-legacy-expectation",
        "--config", str(path), "--shard-count", "2", "--shard-index", str(i)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=environment) for i in range(2)]
    for process in processes:
        stdout, stderr = process.communicate(timeout=60)
        assert process.returncode == 0, stderr
        assert json.loads(stdout)["status"] in {"partial", "completed"}
    assert verify_legacy_expectation_run(other)["status"] == "completed"
    for first, second in zip(products(config), products(other)):
        for name in first.arrays:
            np.testing.assert_array_equal(first.arrays[name], second.arrays[name], err_msg=name)


@pytest.mark.parametrize("failure", ["implicit_id", "raw_grid", "cosmic", "profile", "budget", "background"])
def test_preflight_rejects_unsupported_contracts(tmp_path, failure):
    config = make_inputs(tmp_path, kind="table-stamp" if failure == "implicit_id" else "catalog-stamp")
    if failure == "implicit_id":
        config.input_table_path.write_text("gaia_g_mag,psf_id\n17,0\n")
    elif failure == "raw_grid":
        config.variability_path.write_text(json.dumps({"source_ids": [101], "relative_flux": [[1.]]}))
    elif failure == "budget":
        config = replace(config, max_group_pixels=1)
    else:
        spec = json.loads(config.spec_path.read_text())
        if failure == "cosmic":
            spec["cosmic_rays"]["enabled"] = True
        elif failure == "profile":
            spec["science_profile"]["profile_id"] = "unclaimed"
        else:
            spec["sky"]["subtract_nonstellar_mean"] = False
        config.spec_path.write_text(json.dumps(spec))
    with pytest.raises(ValueError):
        build_legacy_expectation_plan(config)
    assert not (config.output_root / config.run_id).exists()
