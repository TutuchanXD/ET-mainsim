from __future__ import annotations

import json
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
from astropy import units as u


REPO_ROOT = Path(__file__).resolve().parents[1]
TESS_TEMPERATURE_PROFILE = "thermal/TESS/tess_temperatures_241209.pkl"
ET_THERMAL_MOTION_TABLE = (
    "thermal/ET/et_lens_temperature_to_centroid_motion_df_241209.pkl"
)


def _assert_temperature_driven_production_dynamics(spec) -> None:
    thermal = spec.dynamic_effects.thermal_drift
    breathing = spec.dynamic_effects.psf_breathing

    assert thermal.enabled is True
    assert thermal.profile == "et_temperature_table"
    assert thermal.temperature_profile_path == TESS_TEMPERATURE_PROFILE
    assert thermal.motion_table_path == ET_THERMAL_MOTION_TABLE
    assert thermal.time_policy == "normalized_observation_phase"
    assert thermal.temperature_values_are_et_lens_c is False

    assert breathing.enabled is True
    assert breathing.profile == "et_temperature_table"
    assert breathing.temperature_profile_path == TESS_TEMPERATURE_PROFILE
    assert breathing.time_policy == "normalized_observation_phase"
    assert breathing.temperature_values_are_et_lens_c is False
    assert breathing.reference_temperature_c == pytest.approx(-15.0)
    assert breathing.scale_per_c == pytest.approx(0.1)

    assert "main_rd_reference" not in {thermal.profile, breathing.profile}


def test_package_import_is_lightweight() -> None:
    script = f"""
import sys
sys.path.insert(0, {str(REPO_ROOT / "src")!r})
import et_mainsim
assert et_mainsim.__version__
assert 'torch' not in sys.modules
assert 'ray' not in sys.modules
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_project_requires_shared_exposure_photsim7_release() -> None:
    payload = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert "photsim7>=0.2.2,<0.3" in payload["project"]["dependencies"]


def test_required_photsim7_runtime_capabilities_are_importable() -> None:
    from et_mainsim.workflows.full_frame import _science_api

    api = _science_api()
    required = {
        "ItemStatus",
        "SharedExposureShardReader",
        "SharedExposureShardWriter",
        "SharedExposureTargetIdentity",
        "StampWindow",
        "cadence_selection_truth_relative_path",
        "read_cadence_selection_truth",
        "resolve_full_frame_source_pixel_geometry",
        "run_single_cadence_full_frame",
        "shared_exposure_crop_v1",
    }

    assert required.issubset(vars(api))


def test_documented_shared_exposure_run_config_is_loadable() -> None:
    from et_mainsim.config import RunConfig

    path = REPO_ROOT / "docs/examples/et_full_frame_shared_exposure.run.toml"
    config = RunConfig.from_toml(path.read_text(encoding="utf-8"), source=str(path))

    shared = config.workload.shared_exposure_stamps
    assert shared.enabled is True
    assert shared.target_source_ids == (1,)
    assert shared.stamp_shape == (100, 300)
    assert shared.product_keys == ("final_stamp", "electron_stamp")


def test_shipped_full_frame_presets_are_typed_and_complete() -> None:
    from et_mainsim.config import FullFrameWorkload, SharedExposureStampsConfig
    from et_mainsim.presets import list_presets, load_preset

    descriptors = list_presets(workflow="et-full-frame")

    assert [item.name for item in descriptors] == [
        "et-full-frame-production",
        "et-full-frame-smoke",
    ]

    smoke = load_preset("et-full-frame-smoke")
    production = load_preset("et-full-frame-production")

    assert smoke.simulation_spec.detector.shape == (64, 64)
    assert smoke.simulation_spec.observation.resolved_n_frames == 1
    assert smoke.simulation_spec.psf.field_id == 0
    assert smoke.simulation_spec.psf.field_id_policy == "explicit"
    assert smoke.simulation_spec.catalog.query_options[
        "reference_field_angle_deg"
    ] == pytest.approx(0.0)
    assert smoke.run_config.execution.backend == "in-process"
    assert smoke.run_config.execution.device == "cpu"
    for loaded in (smoke, production):
        assert isinstance(loaded.run_config.workload, FullFrameWorkload)
        shared = loaded.run_config.workload.shared_exposure_stamps
        assert isinstance(shared, SharedExposureStampsConfig)
        assert shared.to_dict() == {
            "enabled": False,
            "target_source_ids": [],
            "stamp_rows": 100,
            "stamp_cols": 300,
            "frames_per_shard": 32,
            "product_keys": ["final_stamp"],
        }

    spec = production.simulation_spec
    assert spec.detector.shape == (9120, 8900)
    assert spec.observation.resolved_n_frames == 180
    assert spec.observation.sampling_interval == 10 * u.s
    assert spec.instrument.optical_efficiency.to_value(u.percent) == pytest.approx(58.0)
    assert spec.instrument.quantum_efficiency.to_value(u.percent) == pytest.approx(80.0)
    assert spec.catalog.source_type == "et_focalplane_query"
    assert spec.catalog.input_magnitude_system == "Gaia_G"
    assert spec.catalog.photon_magnitude_system == "ET"
    assert spec.catalog.target_epoch_jyear == pytest.approx(2000.0)
    assert spec.dynamic_effects.psd_motion.native_jitter_bank_path == (
        "jitter/et/native/legacy_science_v1_et_attitude_xyz_100x3x300_v1.npy"
    )
    assert spec.dynamic_effects.psd_motion.native_jitter_bank_manifest_path == (
        "jitter/et/native/legacy_science_v1_et_attitude_xyz_100x3x300_v1.manifest.json"
    )
    assert spec.dynamic_effects.psd_motion.native_jitter_bank_sha256 == (
        "696a986c82902ad18f136f284a30b2ce506998d3e900ea2601a3e6af001cc4d0"
    )
    assert (
        spec.dynamic_effects.psd_motion.native_jitter_bank_manifest_sha256
        == "267453c0cc5355f7edfaff76164c56ea38052a866bb967bb124c920394bf7274"
    )
    assert production.run_config.execution.backend == "local-subprocess"
    assert production.run_config.execution.device == "cuda"


def test_shipped_stamp_presets_fix_local_query_and_coadd_contract() -> None:
    from et_mainsim.config import StampWorkload
    from et_mainsim.presets import list_presets, load_preset

    assert [item.name for item in list_presets(workflow="et-stamp")] == [
        "et-stamp-production",
        "et-stamp-smoke",
    ]

    smoke = load_preset("et-stamp-smoke")
    production = load_preset("et-stamp-production")

    assert smoke.simulation_spec.psf.mode == "stamp"
    assert smoke.simulation_spec.observation.resolved_n_frames == 2
    assert smoke.simulation_spec.observation.n_raw_frames_per_coadd == 2
    assert isinstance(smoke.run_config.workload, StampWorkload)
    assert smoke.run_config.workload.stamp_shape == (15, 15)
    assert smoke.run_config.workload.write_batch_size == 32

    spec = production.simulation_spec
    assert spec.detector.shape == (9120, 8900)
    assert spec.detector.n_subpixels == 7
    assert spec.observation.resolved_n_frames == 360
    assert spec.observation.n_raw_frames_per_coadd == 30
    assert spec.catalog.query_options["query_radius_deg"] == pytest.approx(0.07)
    assert spec.psf.mode == "stamp"
    assert production.run_config.workload.include_neighbors is True
    assert production.run_config.workload.write_batch_size == 32


def test_production_presets_select_temperature_driven_dynamics() -> None:
    from et_mainsim.presets import load_preset, resource_path

    canonical_payload = json.loads(
        resource_path("et_full_frame_production.spec.json").read_text(encoding="utf-8")
    )
    assert canonical_payload["schema_version"] == 3
    canonical_thermal = canonical_payload["dynamic_effects"]["thermal_drift"]
    canonical_breathing = canonical_payload["dynamic_effects"]["psf_breathing"]
    assert canonical_thermal["profile"] == "et_temperature_table"
    assert canonical_thermal["temperature_profile_path"] == TESS_TEMPERATURE_PROFILE
    assert canonical_thermal["motion_table_path"] == ET_THERMAL_MOTION_TABLE
    assert canonical_thermal["time_policy"] == "normalized_observation_phase"
    assert canonical_thermal["temperature_values_are_et_lens_c"] is False
    assert canonical_breathing["profile"] == "et_temperature_table"
    assert canonical_breathing["temperature_profile_path"] == TESS_TEMPERATURE_PROFILE
    assert canonical_breathing["time_policy"] == "normalized_observation_phase"
    assert canonical_breathing["temperature_values_are_et_lens_c"] is False
    assert canonical_breathing["reference_temperature_c"] == pytest.approx(-15.0)
    assert canonical_breathing["scale_per_c"] == pytest.approx(0.1)
    assert canonical_payload["science_profile"] == {
        "profile_id": "unclaimed",
        "composition_id": "unclaimed",
        "science_realization_id": 0,
    }

    full_frame = load_preset("et-full-frame-production").simulation_spec
    stamp = load_preset("et-stamp-production").simulation_spec

    _assert_temperature_driven_production_dynamics(full_frame)
    _assert_temperature_driven_production_dynamics(stamp)
    assert full_frame.science_profile.profile_id == "unclaimed"
    assert full_frame.science_profile.composition_id == "unclaimed"
    assert full_frame.science_profile.science_realization_id == 0
    assert stamp.science_profile.profile_id == "unclaimed"
    assert stamp.science_profile.composition_id == "unclaimed"
    assert stamp.science_profile.science_realization_id == 0

    assert (
        full_frame.dynamic_effects.thermal_drift.temperature_profile_path
        == full_frame.dynamic_effects.psf_breathing.temperature_profile_path
        == stamp.dynamic_effects.thermal_drift.temperature_profile_path
        == stamp.dynamic_effects.psf_breathing.temperature_profile_path
        == TESS_TEMPERATURE_PROFILE
    )
    assert (
        full_frame.dynamic_effects.thermal_drift.motion_table_path
        == stamp.dynamic_effects.thermal_drift.motion_table_path
        == ET_THERMAL_MOTION_TABLE
    )


def test_shipped_legacy_presets_are_exact_full_effect_contracts() -> None:
    from et_mainsim.config import LegacyWorkload
    from et_mainsim.presets import list_presets, load_preset

    assert [item.name for item in list_presets(workflow="legacy-sim")] == [
        "legacy-sim-full-effects-production",
        "legacy-sim-full-effects-smoke",
    ]

    smoke = load_preset("legacy-sim-full-effects-smoke")
    production = load_preset("legacy-sim-full-effects-production")

    assert smoke.simulation_spec.detector.shape == (9, 9)
    assert smoke.simulation_spec.psf.n_jitter_integrated_psf_models == 100
    assert smoke.simulation_spec.psf.n_jitter_frames_per_model == 300
    for loaded in (smoke, production):
        psd = loaded.simulation_spec.dynamic_effects.psd_motion
        assert psd.profile == "et_attitude_xyz"
        assert psd.native_jitter_bank_sha256 == (
            "696a986c82902ad18f136f284a30b2ce506998d3e900ea2601a3e6af001cc4d0"
        )
        assert psd.native_jitter_bank_manifest_sha256 == (
            "267453c0cc5355f7edfaff76164c56ea38052a866bb967bb124c920394bf7274"
        )
    assert isinstance(smoke.run_config.workload, LegacyWorkload)
    assert smoke.run_config.execution.backend == "local-ray"

    spec = production.simulation_spec
    assert spec.detector.shape == (101, 101)
    assert spec.observation.resolved_n_frames == 360
    assert spec.psf.n_jitter_integrated_psf_models == 100
    assert spec.psf.n_jitter_frames_per_model == 300
    assert production.run_config.workload.stars_per_run == 100
    assert production.run_config.workload.store_images is False


def test_unknown_preset_is_rejected() -> None:
    from et_mainsim.presets import load_preset

    with pytest.raises(KeyError, match="Unknown preset"):
        load_preset("missing")
