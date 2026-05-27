from __future__ import annotations

import argparse
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest


MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import main_rd_parallel_core as core


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
    assert np.all(stars_a["field_angle_deg"] == 12.0)

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


def test_sim_config_dict_uses_run_spec_detector_values():
    spec = core.MainRdRunSpec(
        frame_rows=50,
        frame_cols=60,
        sky_surface_brightness_mag_arcsec2=23.2,
        n_subpixels=5,
        exposure_s=3.0,
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
    assert cfg["PSF Bundle Name"] == "custom/psf"
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
    assert np.all(stars["field_angle_deg"] == 12.0)


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
