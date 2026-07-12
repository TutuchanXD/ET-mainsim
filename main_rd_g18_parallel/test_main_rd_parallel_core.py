from __future__ import annotations

import argparse
import json
import os
import runpy
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from astropy import units as u


MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import main_rd_parallel_core as core


@pytest.mark.parametrize(
    "script_path",
    sorted(MODULE_DIR.glob("simulate_main_rd_*.py")),
    ids=lambda path: path.name,
)
def test_main_rd_entrypoint_overrides_form_valid_typed_adapter(
    monkeypatch,
    script_path,
):
    captured = {}

    def fake_run_entrypoint(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(core, "run_entrypoint", fake_run_entrypoint)
    monkeypatch.setitem(sys.modules, "main_rd_parallel_core", core)

    runpy.run_path(str(script_path), run_name="__main__")

    overrides = dict(captured.get("spec_overrides") or {})
    spec = core.MainRdRunSpec(
        frame_rows=int(captured["frame_rows"]),
        frame_cols=int(captured["frame_cols"]),
        **overrides,
    )
    assert spec.observing_duration_s == pytest.approx(
        spec.n_frames * spec.exposure_s
    )


def test_path_constants_can_be_overridden_from_environment(tmp_path):
    env = os.environ.copy()
    env.update(
        {
            "ET_ROOT": str(tmp_path / "ET"),
            "PHOTSIM7_ROOT": str(tmp_path / "ET" / "Photosim7"),
            "PHOTSIM7_DATA_DIR": str(tmp_path / "ET" / "Photsim7-data"),
            "ET_FOCALPLANE_ROOT": str(tmp_path / "ET" / "et_focalplane"),
            "GAIA_CATALOG_DIR": str(tmp_path / "gaia_dr3_19mag"),
            "RESULTS_ROOT": str(tmp_path / "results"),
        }
    )
    script = f"""
import sys
sys.path.insert(0, {str(MODULE_DIR)!r})
import main_rd_parallel_core as core
assert str(core.ET_ROOT) == {str(tmp_path / "ET")!r}
assert str(core.PHOTSIM7_ROOT) == {str(tmp_path / "ET" / "Photosim7")!r}
assert str(core.PHOTSIM7_DATA_DIR) == {str(tmp_path / "ET" / "Photsim7-data")!r}
assert str(core.ET_FOCALPLANE_ROOT) == {str(tmp_path / "ET" / "et_focalplane")!r}
assert str(core.GAIA_CATALOG_DIR) == {str(tmp_path / "gaia_dr3_19mag")!r}
assert str(core.RESULTS_ROOT) == {str(tmp_path / "results")!r}
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_module_import_does_not_require_torch():
    script = f"""
import builtins
import sys
sys.path.insert(0, {str(MODULE_DIR)!r})
original_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name == "torch":
        raise ModuleNotFoundError("No module named 'torch'")
    return original_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
import main_rd_parallel_core as core
assert core.MainRdRunSpec(frame_rows=1, frame_cols=1).frame_rows == 1
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_ensure_local_imports_allows_installed_photsim7_when_source_tree_missing(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(core, "PHOTSIM7_ROOT", tmp_path / "missing_photsim7")
    monkeypatch.setattr(
        core.importlib.util,
        "find_spec",
        lambda name: object() if name == "photsim7" else None,
    )

    core.ensure_local_imports()


def test_build_synthetic_mag_distribution_stars_uses_et_mag_and_seeded_positions(tmp_path):
    csv_path = tmp_path / "mag_distribution.csv"
    csv_path.write_text(
        "\n".join(
            [
                "mwmsc_gmag,other",
                "22.9999,a",
                "23.0,b",
                "23.1,c",
                "5.8,d",
            ]
        ),
        encoding="utf-8",
    )

    stars_a = core.build_synthetic_mag_distribution_stars(
        csv_path=csv_path,
        mag_column="mwmsc_gmag",
        mag_limit=23.0,
        frame_rows=500,
        frame_cols=500,
        seed=123,
        psf_field_angle_deg=12.0,
    )
    stars_b = core.build_synthetic_mag_distribution_stars(
        csv_path=csv_path,
        mag_column="mwmsc_gmag",
        mag_limit=23.0,
        frame_rows=500,
        frame_cols=500,
        seed=123,
        psf_field_angle_deg=12.0,
    )

    np.testing.assert_allclose(stars_a["kp_mag"], np.array([22.9999, 23.0, 5.8]))
    np.testing.assert_allclose(stars_a["gaia_g_mag"], stars_a["kp_mag"])
    assert len(stars_a["x0"]) == 3
    expected = core.main_rd_field_geometry_from_frame_offsets(
        stars_a["x0"],
        stars_a["y0"],
    )
    np.testing.assert_allclose(stars_a["field_angle_deg"], expected["field_angle_deg"])
    assert not np.all(stars_a["field_angle_deg"] == 12.0)

    x_abs = stars_a["x0"] + (500 - 1) / 2.0
    y_abs = stars_a["y0"] + (500 - 1) / 2.0
    assert np.all((0.0 <= x_abs) & (x_abs <= 499.0))
    assert np.all((0.0 <= y_abs) & (y_abs <= 499.0))
    np.testing.assert_allclose(stars_a["x0"], stars_b["x0"])
    np.testing.assert_allclose(stars_a["y0"], stars_b["y0"])


def test_synthetic_run_label_is_used_for_cache_path():
    spec = core.MainRdRunSpec(
        frame_rows=500,
        frame_cols=500,
        mag_limit=23.0,
        run_label="main_rd_500x500_magdist_310-50-2420_g_lt_23_colnoise0",
    )

    cache_path = core.star_cache_path(Path("/tmp/results"), spec, 23.0)

    assert cache_path == Path(
        "/tmp/results/main_rd_500x500_magdist_310-50-2420_g_lt_23_colnoise0/"
        "cache/stars_main_rd_500x500_magdist_310-50-2420_g_lt_23_colnoise0.npz"
    )


def test_sample_column_noise_zero_returns_zero_vector():
    torch = pytest.importorskip("torch")

    noise = core.sample_column_noise_adu(
        frame_cols=8,
        sigma_adu=0.0,
        dtype=torch.float32,
        device="cpu",
    )

    assert torch.equal(noise, torch.zeros(8, dtype=torch.float32))


def test_expand_gpu_worker_assignments_repeats_each_gpu():
    assignments = core.expand_gpu_worker_assignments(["0", "1"], workers_per_gpu=3)

    assert assignments == ["0", "0", "0", "1", "1", "1"]


def test_selected_frame_indices_defaults_to_full_range_and_parses_unique_values():
    assert core.selected_frame_indices(None, 4) == [0, 1, 2, 3]
    assert core.selected_frame_indices("0, 180,180", 270) == [0, 180]


def test_selected_frame_indices_rejects_out_of_range_values():
    with pytest.raises(ValueError, match="outside the valid range"):
        core.selected_frame_indices("270", 270)


def test_sim_config_dict_uses_requested_sky_surface_brightness():
    cfg_21 = core.sim_config_dict(
        500,
        500,
        sky_surface_brightness_mag_arcsec2=21.0,
        n_subpixels=3,
    )
    cfg_232 = core.sim_config_dict(
        500,
        500,
        sky_surface_brightness_mag_arcsec2=23.2,
        n_subpixels=5,
    )

    rate_21 = cfg_21["Background Flux"].value
    rate_232 = cfg_232["Background Flux"].value

    assert cfg_232["Sky Background Surface Brightness"] == 23.2
    assert cfg_232["Subpixels Per Pixel Dim"] == 5
    assert cfg_232["Subtract Nonstellar Mean"] is False
    assert rate_232 == pytest.approx(rate_21 * 10 ** (-0.4 * 2.2))


def test_default_run_spec_uses_sky22_and_single_subpixel():
    spec = core.MainRdRunSpec(frame_rows=500, frame_cols=500)

    cfg = core.sim_config_dict(spec.frame_rows, spec.frame_cols, spec=spec)

    assert spec.sky_surface_brightness_mag_arcsec2 == pytest.approx(22.0)
    assert spec.n_subpixels == 1
    assert cfg["Sky Background Mode"] == "surface_brightness"
    assert cfg["Sky Background Surface Brightness"] == pytest.approx(22.0)
    assert cfg["Subpixels Per Pixel Dim"] == 1


@pytest.mark.parametrize(
    ("star_source", "source_type"),
    [
        ("gaia_main_rd", "et_focalplane_query"),
        ("synthetic_mag_distribution", "synthetic_mag_distribution"),
        ("detector_xy_csv", "detector_xy_csv"),
    ],
)
def test_main_rd_run_spec_builds_canonical_simulation_spec(star_source, source_type):
    run_spec = core.MainRdRunSpec(
        frame_rows=50,
        frame_cols=60,
        star_source=star_source,
        exposure_s=3.0,
        n_frames=4,
        observing_duration_s=12.0,
        sky_surface_brightness_mag_arcsec2=23.2,
        readout_noise_e_pix=1.5,
        n_subpixels=5,
        scattered_light_e_s_pix=0.75,
        psf_bundle_name="custom/psf",
    )

    spec = run_spec.to_simulation_spec(run_seed=42, compute_device="cpu")

    assert spec.detector.shape == (50, 60)
    assert spec.observation.exposure_duration == 3 * u.s
    assert spec.observation.observing_duration == 12 * u.s
    assert spec.observation.resolved_n_frames == 4
    assert spec.instrument.optical_efficiency.to_value(u.percent) == pytest.approx(58.0)
    assert spec.instrument.quantum_efficiency.to_value(u.percent) == pytest.approx(80.0)
    assert spec.instrument.telescope_count == 1
    assert spec.catalog.source_type == source_type
    assert spec.catalog.photon_magnitude_system == "ET"
    assert spec.detector.n_subpixels == 5
    assert spec.readout.readout_noise == 1.5 * u.electron / u.pix
    assert spec.sky.scattered_light == 0.75 * u.electron / u.s / u.pix
    assert spec.psf.bundle_name == "psf/et/custom/psf"
    assert spec.psf.field_id_policy == "nearest"
    assert spec.psf.compute_device == "cpu"
    assert spec.dynamic_effects.thermal_drift.profile == "main_rd_reference"
    assert spec.dynamic_effects.momentum_dump.profile == (
        "legacy_random_walk_within_circle"
    )
    assert spec.dynamic_effects.psf_breathing.profile == "main_rd_reference"
    assert spec.rng.run_seed == 42


def test_main_rd_run_spec_rejects_legacy_throughput_and_duration_conflicts():
    with pytest.raises(ValueError, match="optical_efficiency_ratio"):
        core.MainRdRunSpec(
            frame_rows=10,
            frame_cols=10,
            optical_efficiency_ratio=1.01,
        )

    with pytest.raises(ValueError, match="observing_duration_s.*n_frames"):
        core.MainRdRunSpec(
            frame_rows=10,
            frame_cols=10,
            exposure_s=10.0,
            n_frames=2,
            observing_duration_s=30.0,
        )


def test_sim_config_dict_is_derived_from_typed_spec_throughput():
    run_spec = core.MainRdRunSpec(frame_rows=5, frame_cols=7)

    config = core.sim_config_dict(5, 7, spec=run_spec)
    typed = run_spec.to_simulation_spec()

    assert config["Detector Height"] == typed.to_config_dict()["Detector Height"]
    assert config["Detector Width"] == typed.to_config_dict()["Detector Width"]
    assert config["Optical Efficiency Ratio"].to_value(u.percent) == pytest.approx(58.0)
    assert config["ET Quantum Efficiency"].to_value(u.percent) == pytest.approx(80.0)


def test_frame_motion_offsets_uses_exposure_time_as_low_frequency_split(monkeypatch, tmp_path):
    psd_path = tmp_path / "psd.pkl"
    psd_path.write_bytes(b"placeholder")
    calls = []

    monkeypatch.setattr(core, "load_psd_motion", lambda path: {"fake": path})

    def fake_psd_axis_motion(
        psd,
        *,
        axis,
        time_s,
        rng,
        frequency_min_hz,
        frequency_max_hz,
        max_frequency_samples=20000,
    ):
        calls.append(
            {
                "axis": axis,
                "time_s": np.asarray(time_s, dtype=np.float64).copy(),
                "frequency_min_hz": float(frequency_min_hz),
                "frequency_max_hz": None
                if frequency_max_hz is None
                else float(frequency_max_hz),
            }
        )
        return np.zeros_like(time_s, dtype=np.float64)

    monkeypatch.setattr(core, "psd_axis_motion", fake_psd_axis_motion)
    monkeypatch.setattr(
        core,
        "spacecraft_roll_drift_from_angles",
        lambda **kwargs: np.zeros((len(kwargs["theta_x_arcsec"]), 2), dtype=np.float64),
    )

    offsets, metadata = core.frame_motion_offsets(
        n_frames=3,
        seed=123,
        enable_psd_motion=True,
        psd_motion_path=psd_path,
        exposure_s=300.0,
    )

    assert offsets.shape == (3, 2)
    assert metadata["split_hz"] == pytest.approx(1.0 / 300.0)
    assert metadata["exposure_s"] == pytest.approx(300.0)
    assert {call["axis"] for call in calls} == {"x", "y", "z"}
    for call in calls:
        np.testing.assert_allclose(call["time_s"], np.array([0.0, 300.0, 600.0]))
        assert call["frequency_min_hz"] == pytest.approx(0.0)
        assert call["frequency_max_hz"] == pytest.approx(1.0 / 300.0)


def test_jitter_integrated_psf_offsets_uses_exposure_time_as_high_frequency_split(
    monkeypatch,
    tmp_path,
):
    psd_path = tmp_path / "psd.pkl"
    psd_path.write_bytes(b"placeholder")
    calls = []

    monkeypatch.setattr(core, "load_psd_motion", lambda path: {"fake": path})

    def fake_psd_axis_motion(
        psd,
        *,
        axis,
        time_s,
        rng,
        frequency_min_hz,
        frequency_max_hz,
        max_frequency_samples=20000,
    ):
        calls.append(
            {
                "axis": axis,
                "time_s": np.asarray(time_s, dtype=np.float64).copy(),
                "frequency_min_hz": float(frequency_min_hz),
                "frequency_max_hz": None
                if frequency_max_hz is None
                else float(frequency_max_hz),
            }
        )
        return np.linspace(0.0, 1.0, len(time_s), dtype=np.float64)

    monkeypatch.setattr(core, "psd_axis_motion", fake_psd_axis_motion)
    monkeypatch.setattr(
        core,
        "spacecraft_roll_drift_from_angles",
        lambda **kwargs: np.vstack(
            [kwargs["theta_x_arcsec"], kwargs["theta_y_arcsec"]]
        ).T,
    )

    xy_jitter_pix, metadata = core.jitter_integrated_psf_offsets(
        seed=123,
        enable_psd_motion=True,
        enable_jitter_integrated_psf=True,
        psd_motion_path=psd_path,
        n_models=2,
        n_frames_per_model=4,
        exposure_s=300.0,
    )

    assert xy_jitter_pix.shape == (2, 2, 4)
    assert metadata["split_hz"] == pytest.approx(1.0 / 300.0)
    assert metadata["exposure_s"] == pytest.approx(300.0)
    assert len(calls) == 2 * 3
    for call in calls:
        np.testing.assert_allclose(call["time_s"], np.array([0.0, 75.0, 150.0, 225.0]))
        assert call["frequency_min_hz"] == pytest.approx(1.0 / 300.0)
        assert call["frequency_max_hz"] is None


def test_parse_common_args_accepts_script_specific_jitter_psf_default():
    parser = core.parse_common_args(
        "test parser",
        default_frames=1,
        default_mag_limit=17.0,
        default_jitter_psf_models=100,
    )

    args = parser.parse_args([])
    override_args = parser.parse_args(["--jitter-psf-models", "7"])

    assert args.frames == 1
    assert args.mag_limit == pytest.approx(17.0)
    assert args.jitter_psf_models == 100
    assert args.prepare_star_cache_only is False
    assert override_args.jitter_psf_models == 7


def test_parse_common_args_uses_data_registry_relative_psd_path():
    parser = core.parse_common_args("test parser")

    args = parser.parse_args([])

    assert args.psd_motion_path == Path("pds/ET_psd3-2.pkl")


def test_sim_config_dict_uses_run_spec_detector_values():
    spec = core.MainRdRunSpec(
        frame_rows=50,
        frame_cols=60,
        sky_surface_brightness_mag_arcsec2=23.2,
        n_subpixels=5,
        exposure_s=3.0,
        n_frames=4,
        observing_duration_s=12.0,
        dark_current_e_s_pix=0.25,
        scattered_light_e_s_pix=0.75,
        readout_noise_e_pix=1.5,
        full_well_electrons=1234.0,
        gain_electrons_per_adu=2.5,
        adc_bit_depth=12,
        bias_level_adu=42.0,
        column_noise_sigma_adu=0.0,
        cosmic_ray_event_rate_cm2_s=7.0,
        cosmic_ray_library_path="custom_cosmic.npz",
        cosmic_ray_pixel_size_um=8.0,
        psf_bundle_name="custom/psf",
        n_jitter_integrated_psf_models=2,
        n_jitter_frames_per_model=4,
    )

    cfg = core.sim_config_dict(spec.frame_rows, spec.frame_cols, spec=spec)

    assert cfg["Subpixels Per Pixel Dim"] == 5
    assert cfg["Exposure Duration"].value == pytest.approx(3.0)
    assert cfg["Observing Duration"].value == pytest.approx(12.0)
    assert cfg["Dark Current"].value == pytest.approx(0.25)
    assert cfg["Scattered Light"].value == pytest.approx(0.75)
    assert cfg["Readout Noise"].value == pytest.approx(1.5)
    assert cfg["Full Well Electrons"].value == pytest.approx(1234.0)
    assert cfg["Gain Electrons Per ADU"].value == pytest.approx(2.5)
    assert cfg["ADC Bit Depth"] == 12
    assert cfg["Bias Level ADU"].value == pytest.approx(42.0)
    assert cfg["Column Noise Sigma ADU"].value == pytest.approx(0.0)
    assert cfg["Cosmic Ray Event Library Path"] == "custom_cosmic.npz"
    assert cfg["Cosmic Ray Event Library Pixel Size"].value == pytest.approx(8.0)
    assert cfg["Cosmic Ray Event Rate"].value == pytest.approx(7.0)
    assert cfg["PSF Bundle Name"] == "psf/et/custom/psf"
    assert cfg["N Jitter-Integrated PSF Models"] == 2
    assert cfg["N Jitter Frames Per Model"] == 4


def test_apply_detector_chain_uses_run_spec_values(monkeypatch):
    torch = pytest.importorskip("torch")
    calls = {}

    class DummyLibrary:
        @classmethod
        def load(cls, path, *, expected_pixel_size_um):
            calls["library_path"] = path
            calls["expected_pixel_size_um"] = expected_pixel_size_um
            return cls()

    class DummyInjector:
        def __init__(self, library):
            calls["library"] = library

        def inject(self, image_stack, *, mean_events_per_frame, seed, frame_start, allow_partial):
            calls["mean_events_per_frame"] = mean_events_per_frame
            calls["seed"] = seed
            calls["frame_start"] = frame_start
            calls["allow_partial"] = allow_partial
            payload = types.SimpleNamespace(
                events=np.empty((0,), dtype=np.float32),
                mask=np.zeros(tuple(image_stack.shape), dtype=bool),
            )
            return image_stack, payload

    def clip_full_well_electrons(image, *, full_well_electrons):
        calls["full_well_electrons"] = full_well_electrons
        return torch.clamp(image, max=full_well_electrons)

    def electrons_to_adu(image, *, gain_electrons_per_adu):
        calls["gain_electrons_per_adu"] = gain_electrons_per_adu
        return image / gain_electrons_per_adu

    def mean_events_from_rate(*, rate_events_per_cm2_s, n_rows, n_cols, pixel_size_um, exposure_s):
        calls["rate_events_per_cm2_s"] = rate_events_per_cm2_s.value
        calls["pixel_size_um"] = pixel_size_um.value
        calls["exposure_s"] = exposure_s.value
        calls["shape"] = (n_rows, n_cols)
        return 0.0

    def apply_adc_digitization(image, *, enabled, bit_depth, min_value, round_values):
        calls["bit_depth"] = bit_depth
        return torch.clamp(torch.round(image), min=min_value, max=2**bit_depth - 1)

    fake_cosmic_rays = types.SimpleNamespace(
        CosmicRayEventLibrary=DummyLibrary,
        CosmicRayInjector=DummyInjector,
        apply_adc_digitization=apply_adc_digitization,
        clip_full_well_electrons=clip_full_well_electrons,
        electrons_to_adu=electrons_to_adu,
        mean_events_from_rate=mean_events_from_rate,
    )
    monkeypatch.setattr(core, "ensure_local_imports", lambda: None)
    monkeypatch.setitem(sys.modules, "photsim7.cosmic_rays", fake_cosmic_rays)
    spec = core.MainRdRunSpec(
        frame_rows=1,
        frame_cols=1,
        exposure_s=3.0,
        n_frames=1,
        observing_duration_s=3.0,
        readout_noise_e_pix=0.0,
        full_well_electrons=10.0,
        gain_electrons_per_adu=2.0,
        adc_bit_depth=8,
        bias_level_adu=7.0,
        column_noise_sigma_adu=0.0,
        cosmic_ray_event_rate_cm2_s=11.0,
        cosmic_ray_library_path="custom_cosmic.npz",
        cosmic_ray_pixel_size_um=12.0,
    )

    image_dn, _, col_noise, mean_events = core.apply_detector_chain(
        image_electrons=torch.tensor([[20.0]], dtype=torch.float32),
        frame_index=2,
        frame_rows=1,
        frame_cols=1,
        seed=100,
        spec=spec,
    )

    assert image_dn.item() == pytest.approx(12.0)
    assert col_noise.item() == pytest.approx(0.0)
    assert mean_events == pytest.approx(0.0)
    assert calls["library_path"] == "custom_cosmic.npz"
    assert calls["expected_pixel_size_um"] == pytest.approx(12.0)
    assert calls["full_well_electrons"] == pytest.approx(10.0)
    assert calls["gain_electrons_per_adu"] == pytest.approx(2.0)
    assert calls["rate_events_per_cm2_s"] == pytest.approx(11.0)
    assert calls["pixel_size_um"] == pytest.approx(12.0)
    assert calls["exposure_s"] == pytest.approx(3.0)
    assert calls["bit_depth"] == 8


def test_build_detector_xy_stars_uses_gmag_directly_as_et_mag_without_kp_mag(tmp_path):
    csv_path = tmp_path / "detector_xy.csv"
    csv_path.write_text(
        "\n".join(
            [
                "source_id,gmag,x0,y0",
                "101,21.5,201.25,-216.75",
                "102,23.9,-249.97214748526392,249.9937805584423",
            ]
        ),
        encoding="utf-8",
    )

    stars = core.build_detector_xy_stars(
        csv_path=csv_path,
        frame_rows=500,
        frame_cols=500,
        psf_field_angle_deg=12.0,
    )

    assert "kp_mag" not in stars
    np.testing.assert_allclose(stars["et_mag"], np.array([21.5, 23.9]))
    np.testing.assert_allclose(stars["gmag"], stars["et_mag"])
    np.testing.assert_allclose(stars["x0"], np.array([201.25, -249.97214748526392]))
    np.testing.assert_allclose(stars["y0"], np.array([-216.75, 249.9937805584423]))
    np.testing.assert_array_equal(stars["source_id"], np.array([101, 102]))
    np.testing.assert_allclose(stars["detector_xpix_shifted"], stars["x0"] + 249.5)
    np.testing.assert_allclose(stars["detector_ypix_shifted"], stars["y0"] + 249.5)
    expected = core.main_rd_field_geometry_from_frame_offsets(
        stars["x0"],
        stars["y0"],
    )
    np.testing.assert_allclose(stars["field_x_deg"], expected["field_x_deg"])
    np.testing.assert_allclose(stars["field_y_deg"], expected["field_y_deg"])
    np.testing.assert_allclose(stars["field_angle_deg"], expected["field_angle_deg"])
    assert not np.all(stars["field_angle_deg"] == 12.0)


def test_build_synthetic_mag_distribution_stars_computes_field_angles_from_positions(tmp_path):
    csv_path = tmp_path / "mag_distribution.csv"
    csv_path.write_text(
        "\n".join(
            [
                "mwmsc_gmag",
                "20.0",
                "21.0",
                "22.0",
            ]
        ),
        encoding="utf-8",
    )

    stars = core.build_synthetic_mag_distribution_stars(
        csv_path=csv_path,
        mag_column="mwmsc_gmag",
        mag_limit=22.0,
        frame_rows=500,
        frame_cols=500,
        seed=321,
        psf_field_angle_deg=12.0,
    )

    expected = core.main_rd_field_geometry_from_frame_offsets(
        stars["x0"],
        stars["y0"],
    )
    np.testing.assert_allclose(stars["field_x_deg"], expected["field_x_deg"])
    np.testing.assert_allclose(stars["field_y_deg"], expected["field_y_deg"])
    np.testing.assert_allclose(stars["field_angle_deg"], expected["field_angle_deg"])
    assert not np.all(stars["field_angle_deg"] == 12.0)


def test_scattered_light_schedule_adds_10_electrons_per_frame_after_frame_180():
    spec = core.MainRdRunSpec(
        frame_rows=500,
        frame_cols=500,
        scattered_light_e_s_pix=0.0,
        scattered_light_step_start_frame=180,
        scattered_light_step_e_pix_frame=10.0,
    )

    assert core.scattered_light_for_frame(spec, 179).value == pytest.approx(0.0)
    assert core.scattered_light_for_frame(spec, 180).value == pytest.approx(1.0)
    assert core.scattered_light_for_frame(spec, 269).value == pytest.approx(1.0)


def test_launch_dry_run_skips_star_cache_preparation(monkeypatch, tmp_path, capsys):
    spec = core.MainRdRunSpec(frame_rows=10, frame_cols=20)
    args = argparse.Namespace(
        output_root=tmp_path,
        frame_indices=None,
        frames=2,
        mag_limit=18.0,
        dry_run=True,
        prepare_star_cache_only=False,
        star_cache=None,
        gpus="0",
        workers_per_gpu=1,
    )
    monkeypatch.setattr(
        core,
        "prepare_star_cache",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("dry-run should not prepare star cache")
        ),
    )

    core.launch_or_run(args, spec, Path("simulate.py"))

    output = capsys.readouterr().out
    assert "[Dry run] star_cache=" in output
    assert "frame_indices=[0, 1]" in output


def test_prepare_star_cache_only_prepares_cache_without_launching(monkeypatch, tmp_path, capsys):
    spec = core.MainRdRunSpec(frame_rows=10, frame_cols=20)
    cache_path = tmp_path / "stars.npz"
    args = argparse.Namespace(
        output_root=tmp_path,
        frame_indices=None,
        frames=2,
        mag_limit=17.0,
        dry_run=False,
        prepare_star_cache_only=True,
        star_cache=None,
        gpus="0",
        workers_per_gpu=1,
        worker_rank=None,
    )
    monkeypatch.setattr(core, "prepare_star_cache", lambda _args, _spec: cache_path)
    monkeypatch.setattr(
        core,
        "run_worker",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("cache-only mode should not launch workers")
        ),
    )

    core.launch_or_run(args, spec, Path("simulate.py"))

    output = capsys.readouterr().out
    assert f"[Star cache] ready {cache_path}" in output


def test_prepare_star_cache_delegates_to_photsim7_catalog_service(
    monkeypatch,
    tmp_path,
):
    from photsim7.catalog_sources import PreparedStarCatalog, StarCatalogCache
    import photsim7.simulation_services as service_module

    catalog = PreparedStarCatalog(
        star_data={
            "x0": np.array([0.0]),
            "y0": np.array([0.0]),
            "ra": np.array([10.0]),
            "dec": np.array([20.0]),
            "source_id": np.array([1]),
            "gaia_g_mag": np.array([12.0]),
        },
        metadata={"source": {"type": "et_focalplane_query"}},
    )
    captured = {}

    def fake_build(spec, *, data_registry):
        captured["spec"] = spec
        captured["data_registry"] = data_registry
        return catalog

    def fake_write(path, value):
        captured["cache_path"] = Path(path)
        captured["catalog"] = value
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"cache")

    monkeypatch.setattr(service_module, "build_catalog_from_spec", fake_build)
    monkeypatch.setattr(StarCatalogCache, "write", staticmethod(fake_write))
    monkeypatch.setattr(core, "PHOTSIM7_DATA_DIR", tmp_path / "data")
    spec = core.MainRdRunSpec(frame_rows=5, frame_cols=7, n_frames=1, observing_duration_s=10.0)
    args = argparse.Namespace(
        output_root=tmp_path,
        mag_limit=17.0,
        force_star_cache=True,
        catalog_dir=tmp_path / "gaia",
        crop_margin_pix=2.0,
        seed=123,
    )

    cache_path = core.prepare_star_cache(args, spec)

    assert cache_path == captured["cache_path"]
    assert captured["catalog"].metadata["et_mainsim"]["compatibility_adapter"] == (
        "MainRdRunSpec"
    )
    assert captured["spec"].catalog.source_type == "et_focalplane_query"
    assert captured["spec"].catalog.source_path == str(args.catalog_dir)
    assert captured["spec"].catalog.background_stars_max_mag == 17.0
    assert captured["spec"].rng.run_seed == 123


def test_build_main_rd_services_delegates_runtime_overrides(monkeypatch, tmp_path):
    from photsim7.catalog_sources import PreparedStarCatalog
    import photsim7.simulation_services as service_module

    catalog = PreparedStarCatalog(
        star_data={
            "x0": np.array([0.0]),
            "y0": np.array([0.0]),
            "ra": np.array([10.0]),
            "dec": np.array([20.0]),
            "source_id": np.array([1]),
            "gaia_g_mag": np.array([12.0]),
            "detector_id": "main_rd",
            "detector_xpix": np.array([100.0]),
            "detector_ypix": np.array([200.0]),
        },
        metadata={"source": {"type": "et_focalplane_query"}},
    )
    sentinel = object()
    captured = {}

    def fake_build(spec, **kwargs):
        captured["spec"] = spec
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(service_module, "build_full_frame_services", fake_build)
    monkeypatch.setattr(core, "PHOTSIM7_DATA_DIR", tmp_path / "data")
    run_spec = core.MainRdRunSpec(frame_rows=5, frame_cols=7)
    args = argparse.Namespace(
        frames=2,
        mag_limit=17.0,
        seed=456,
        device="cpu",
        catalog_dir=tmp_path / "gaia",
        crop_margin_pix=3.0,
        jitter_integrated_psf=False,
        jitter_psf_models=7,
        jitter_frames_per_model=9,
        enable_psd_motion=False,
        psd_motion_path=tmp_path / "psd.pkl",
        enable_dva_drift=False,
        enable_thermal_drift=False,
        enable_momentum_dump=False,
        enable_psf_breathing=False,
        no_detector_response=True,
    )

    services = core.build_main_rd_services(args, run_spec, catalog)

    typed = captured["spec"]
    assert services is sentinel
    assert captured["catalog"] is catalog
    assert captured["frame_exposure"] == 10 * u.s
    assert typed.observation.resolved_n_frames == 2
    assert typed.observation.observing_duration == 20 * u.s
    assert typed.catalog.source_type == "et_focalplane_query"
    assert typed.catalog.background_stars_max_mag == 17.0
    assert typed.catalog.query_options["reference_field_angle_deg"] == pytest.approx(
        run_spec.target_field_angle_deg
    )
    assert typed.catalog.query_options[
        "reference_field_polar_angle_rad"
    ] == pytest.approx(
        np.arctan2(run_spec.target_field_y_deg, run_spec.target_field_x_deg)
    )
    assert typed.rng.run_seed == 456
    assert typed.psf.compute_device == "cpu"
    assert typed.psf.use_jitter_integrated_psf is False
    assert typed.psf.n_jitter_integrated_psf_models == 7
    assert typed.psf.n_jitter_frames_per_model == 9
    assert typed.dynamic_effects.psd_motion.enabled is False
    assert typed.dynamic_effects.dva.enabled is False
    assert typed.dynamic_effects.thermal_drift.enabled is False
    assert typed.dynamic_effects.momentum_dump.enabled is False
    assert typed.dynamic_effects.psf_breathing.enabled is False
    assert typed.detector_response.enable_inter_pixel_response is False


def test_effect_timeseries_artifacts_encode_disabled_effects():
    typed_spec = core.MainRdRunSpec(
        frame_rows=2,
        frame_cols=3,
        n_frames=2,
        observing_duration_s=20.0,
    ).to_simulation_spec()

    arrays, metadata = core.effect_timeseries_artifacts(None, typed_spec)

    np.testing.assert_allclose(arrays["frame_start_s"], [0.0, 10.0])
    np.testing.assert_allclose(arrays["frame_mid_s"], [5.0, 15.0])
    assert metadata["schema_id"] == "photsim7.effect_timeseries.v1"
    assert metadata["timing"]["n_frames"] == 2
    assert metadata["components"] == []
    assert metadata["metadata"]["all_effects_disabled"] is True


def test_run_worker_uses_package_pipeline_and_preserves_legacy_outputs(
    monkeypatch,
    tmp_path,
):
    from photsim7.catalog_sources import PreparedStarCatalog, StarCatalogCache
    from photsim7.detector_electronics import (
        BiasColumnNoisePayload,
        DetectorElectronicsResult,
    )
    from photsim7.frame_products import (
        BiasColumnNoiseProduct,
        CosmicEventProduct,
        FrameArrayProduct,
        SingleCadenceFrameProducts,
    )
    import photsim7.full_frame_pipeline as pipeline_module

    spec = core.MainRdRunSpec(
        frame_rows=2,
        frame_cols=2,
        run_label="package-worker-contract",
        n_frames=1,
        observing_duration_s=10.0,
    )
    cache_path = tmp_path / "stars.npz"
    catalog = PreparedStarCatalog(
        star_data={
            "x0": np.array([0.0]),
            "y0": np.array([0.0]),
            "ra": np.array([10.0]),
            "dec": np.array([20.0]),
            "source_id": np.array([1]),
            "gaia_g_mag": np.array([12.0]),
            "detector_id": "main_rd",
            "detector_xpix": np.array([4450.0]),
            "detector_ypix": np.array([4560.0]),
        },
        metadata={"source": {"type": "et_focalplane_query"}},
    )
    StarCatalogCache.write(cache_path, catalog)

    class FakeEffects:
        def to_arrays(self):
            return {
                "frame_start_s": np.array([0.0]),
                "frame_mid_s": np.array([5.0]),
                "psd_drift": np.zeros((1, 3)),
            }

        def to_metadata(self):
            return {"schema_id": "photsim7.effect_timeseries.v1"}

    typed_spec = spec.to_simulation_spec(run_seed=321, compute_device="cpu")
    services = types.SimpleNamespace(
        spec=typed_spec,
        catalog=catalog,
        effect_timeseries=FakeEffects(),
        psf_result=types.SimpleNamespace(provenance={"factory": "test"}),
        provenance={"schema_id": "photsim7.full_frame_services.v1"},
    )
    calls = {"service_count": 0, "pipeline": []}

    def fake_build_services(args, run_spec, prepared_catalog):
        calls["service_count"] += 1
        assert run_spec is spec
        assert prepared_catalog.metadata["source"]["type"] == "et_focalplane_query"
        return services

    events = np.array([(0,)], dtype=[("frame_index", "i8")])
    mask = np.array([[False, True], [False, False]])
    column_noise = np.array([0.25, -0.5], dtype=np.float32)
    final_frame = np.array([[11, 12], [13, 14]], dtype=np.uint16)
    stellar_mean = np.full((2, 2), 3.5, dtype=np.float32)

    def fake_run(typed, **kwargs):
        calls["pipeline"].append((typed, kwargs))
        cosmic_payload = types.SimpleNamespace(events=events, mask=mask)
        bias_payload = BiasColumnNoisePayload(
            bias_level_adu=3500.0,
            column_noise_sigma_adu=5.0,
            column_noise_vector_adu=column_noise,
        )
        detector_result = DetectorElectronicsResult(
            image_dn=final_frame,
            image_adu_before_adc=final_frame.astype(np.float32),
            saturation_count=0,
            bias_metadata=bias_payload,
            cosmic_metadata=cosmic_payload,
            domain_transitions=("electrons", "dn"),
            rng_trace={"schema_id": "photsim7.rng_trace.v1", "entries": []},
        )
        products = SingleCadenceFrameProducts(
            frame_index=0,
            detector_id="main_rd",
            final_frame=FrameArrayProduct(
                name="final_frame",
                array=final_frame,
                unit="dn",
                domain="dn",
            ),
            electron_components={
                "stellar_mean": FrameArrayProduct(
                    name="stellar_mean",
                    array=stellar_mean,
                    unit="electron",
                    domain="electrons",
                )
            },
            cosmic_events=CosmicEventProduct(events=events, mask=mask),
            bias_column_noise_adu=BiasColumnNoiseProduct.from_payload(bias_payload),
            frame_summary={
                "frame_index": 0,
                "mean_cosmic_events_per_frame": 0.5,
                "actual_cosmic_events": 1,
                "cosmic_mask_pixels": 1,
            },
            provenance={"pipeline": {"api": "run_single_cadence_full_frame"}},
        )
        writer = kwargs["artifact_writer"]
        paths = writer.write_frame(
            0,
            final_frame,
            summary=dict(products.frame_summary),
            cosmic_events=cosmic_payload,
            column_noise_adu=column_noise,
        )
        paths["frame_product_schema"] = writer.write_frame_product_schema(products)
        return types.SimpleNamespace(
            frame_products=products,
            renderer_components={"stellar_mean": stellar_mean},
            detector_result=detector_result,
            provenance=products.provenance,
            artifact_paths=paths,
        )

    def legacy_builder_called(*_args, **_kwargs):
        raise AssertionError("legacy main-rd physics builder was called")

    monkeypatch.setattr(core, "build_main_rd_services", fake_build_services)
    monkeypatch.setattr(pipeline_module, "run_single_cadence_full_frame", fake_run)
    monkeypatch.setattr(core, "build_full_effect_timeseries", legacy_builder_called)
    monkeypatch.setattr(core, "jitter_integrated_psf_offsets", legacy_builder_called)
    monkeypatch.setattr(core, "build_psf_manager", legacy_builder_called)
    monkeypatch.setattr(core, "build_star_catalog", legacy_builder_called)
    monkeypatch.setattr(core, "build_detector_response_sampler", legacy_builder_called)
    monkeypatch.setattr(core, "make_renderer", legacy_builder_called)
    monkeypatch.setattr(core, "apply_detector_chain", legacy_builder_called)
    monkeypatch.setattr(core, "gpu_memory_snapshot", lambda: "test")
    args = argparse.Namespace(
        frame_indices=None,
        frames=1,
        device="cpu",
        jitter_psf_models=1,
        jitter_frames_per_model=1,
        output_root=tmp_path,
        mag_limit=17.0,
        star_cache=cache_path,
        max_stars=None,
        worker_rank=0,
        worker_world_size=1,
        seed=321,
        overwrite=True,
        preview_count=0,
        save_column_noise=True,
        save_cosmic_mask=True,
        save_stellar_mean=True,
        progress=False,
        catalog_dir=tmp_path,
        crop_margin_pix=2.0,
        jitter_integrated_psf=False,
        enable_psd_motion=False,
        psd_motion_path=tmp_path / "psd.pkl",
        enable_dva_drift=False,
        enable_thermal_drift=False,
        enable_momentum_dump=False,
        enable_psf_breathing=False,
        no_detector_response=True,
    )

    core.run_worker(args, spec)

    run_dir = tmp_path / spec.run_label
    assert calls["service_count"] == 1
    assert len(calls["pipeline"]) == 1
    assert calls["pipeline"][0][0] is typed_spec
    assert calls["pipeline"][0][1]["services"] is services
    assert calls["pipeline"][0][1]["frame_index"] == 0
    assert calls["pipeline"][0][1]["worker_rank"] == 0
    np.testing.assert_array_equal(
        np.load(run_dir / "frames/frame_000000.npy"),
        final_frame,
    )
    np.testing.assert_array_equal(
        np.load(run_dir / "cosmic_events/frame_000000_events.npy"),
        events,
    )
    np.testing.assert_array_equal(
        np.load(run_dir / "cosmic_events/frame_000000_mask.npy"),
        mask,
    )
    np.testing.assert_allclose(
        np.load(run_dir / "bias/frame_000000_column_noise_adu.npy"),
        column_noise,
    )
    np.testing.assert_allclose(
        np.load(run_dir / "frames/frame_000000_stellar_mean_e.npy"),
        stellar_mean,
    )
    with (run_dir / "frame_summaries/frame_000000_schema.json").open(
        encoding="utf-8"
    ) as handle:
        schema = json.load(handle)
    assert schema["schema_id"] == "photsim7.single_cadence_frame_products.v1"
    assert schema["arrays"]["final_frame"]["unit"] == "dn"
    with (run_dir / "frame_summaries/frame_000000.json").open(
        encoding="utf-8"
    ) as handle:
        summary = json.load(handle)
    assert summary["package_frame_summary"]["actual_cosmic_events"] == 1
    assert summary["package_schema_path"].endswith("frame_000000_schema.json")
    with np.load(run_dir / "effects_timeseries.npz") as payload:
        assert set(payload.files) == {"frame_start_s", "frame_mid_s", "psd_drift"}


def test_launch_reuses_existing_star_cache_without_querying_catalog(
    monkeypatch,
    tmp_path,
):
    spec = core.MainRdRunSpec(frame_rows=10, frame_cols=20, run_label="run")
    cache_path = tmp_path / "existing_stars.npz"
    cache_path.write_bytes(b"cache")
    popen_calls = []

    class DummyProc:
        def wait(self):
            return 0

    def fake_popen(cmd, **kwargs):
        popen_calls.append((cmd, kwargs))
        return DummyProc()

    args = argparse.Namespace(
        output_root=tmp_path,
        frame_indices="0",
        frames=1,
        mag_limit=17.0,
        dry_run=False,
        prepare_star_cache_only=False,
        star_cache=cache_path,
        gpus="0",
        workers_per_gpu=1,
        worker_rank=None,
        catalog_dir=tmp_path / "missing_catalog",
        seed=123,
        crop_margin_pix=2.0,
        preview_count=0,
        max_stars=None,
        overwrite=False,
        no_detector_response=False,
        save_column_noise=True,
        save_cosmic_mask=False,
        save_stellar_mean=False,
        progress=False,
        jitter_integrated_psf=True,
        jitter_psf_models=100,
        jitter_frames_per_model=600,
        enable_psd_motion=True,
        psd_motion_path=tmp_path / "psd.pkl",
        enable_dva_drift=True,
        enable_thermal_drift=True,
        enable_momentum_dump=True,
        enable_psf_breathing=True,
        device="cuda",
    )
    monkeypatch.setattr(
        core,
        "prepare_star_cache",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("existing --star-cache should be reused")
        ),
    )
    monkeypatch.setattr(core.subprocess, "Popen", fake_popen)
    expected_git = {
        "et_mainsim": {"commit": "e" * 40, "dirty": False},
        "photsim7": {"commit": "p" * 40, "dirty": False},
    }
    monkeypatch.setattr(
        core,
        "source_git_provenance",
        lambda: expected_git,
        raising=False,
    )

    core.launch_or_run(args, spec, Path("simulate.py"))

    assert len(popen_calls) == 1
    assert "--star-cache" in popen_calls[0][0]
    assert str(cache_path) in popen_calls[0][0]
    with (tmp_path / "run" / "run_config.json").open(encoding="utf-8") as handle:
        run_config = json.load(handle)
    assert run_config["compatibility_adapter"] == "MainRdRunSpec"
    assert run_config["git_provenance"] == expected_git
    assert run_config["simulation_spec"]["schema_id"] == "photsim7.simulation_spec"
    assert run_config["simulation_spec"]["schema_version"] == 1
    assert run_config["simulation_spec"]["observation"]["n_frames"] == 1
    assert run_config["simulation_spec"]["instrument"]["optical_efficiency"] == {
        "value": pytest.approx(58.0),
        "unit": "%",
    }
    assert run_config["simulation_spec"]["instrument"]["quantum_efficiency"] == {
        "value": pytest.approx(80.0),
        "unit": "%",
    }
