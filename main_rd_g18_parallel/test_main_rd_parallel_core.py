from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import main_rd_parallel_core as core


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
