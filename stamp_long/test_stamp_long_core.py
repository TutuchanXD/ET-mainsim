from __future__ import annotations

import csv
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest


MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import stamp_long_core as core


def test_exposure_parameters_scale_direct_long_exposure_values():
    params = core.exposure_parameters(300.0)

    assert params.exposure_s == pytest.approx(300.0)
    assert params.n_coadd_equiv == pytest.approx(30.0)
    assert params.frames_per_day == 288
    assert params.frames_per_year == 105192
    assert params.read_noise_e_pix == pytest.approx(5.0 * np.sqrt(30.0))


def test_exposure_parameters_include_correct_180s_case():
    params = core.exposure_parameters(180.0)

    assert params.exposure_s == pytest.approx(180.0)
    assert params.n_coadd_equiv == pytest.approx(18.0)
    assert params.frames_per_day == 480
    assert params.frames_per_year == 175320
    assert params.read_noise_e_pix == pytest.approx(5.0 * np.sqrt(18.0))


def test_default_render_options_match_reviewed_parameter_set():
    options = core.RenderOptions()

    assert options.background_e_s_pix == pytest.approx(26.0)
    assert options.scattered_light_e_s_pix == pytest.approx(5.0)
    assert options.dark_e_s_pix == pytest.approx(1.0)
    assert options.read_noise_10s_e_pix == pytest.approx(5.0)
    assert options.cosmic_ray_library_path == "cosmic_ray/dark_test_10um/event_library_10um.npz"
    assert options.pixel_size_um == pytest.approx(10.0)
    assert options.psf_bundle_name == "psf/et/241006/D280mm-focus"
    assert options.psf_field_id == 6
    assert options.psf_subpixels == 7
    assert options.jitter_integrated_psf_models == 300
    assert options.jitter_frames_per_model == 600
    assert options.enable_detector_response is True
    assert options.inter_pixel_response_sigma == pytest.approx(0.01)
    assert options.inter_pixel_response_nominal == pytest.approx(1.0)
    assert options.intra_pixel_response_sigma == pytest.approx(0.01)
    assert options.enable_pixel_phase_response is True
    assert options.star_flux_mode == "random_et_mag"
    assert options.et_mag_min == pytest.approx(12.5)
    assert options.et_mag_max == pytest.approx(14.5)


def test_et_mag_to_photon_rate_uses_main_rd_zero_point():
    expected_zero_point = 0.91526 * 615.75 * 1_961_225

    rate = core.et_mag_to_photon_rate_e_s(20.0)

    assert core.ET_PHOTON_RATE_ZEROPOINT_E_S == pytest.approx(expected_zero_point)
    assert rate == pytest.approx(expected_zero_point * 10 ** (-0.4 * 20.0))


def test_random_et_mag_sampling_is_star_stable_and_in_range():
    case = core.BenchmarkCase(
        "T01",
        "test",
        n_stars=10,
        exposure_s=300.0,
        n_frames=3,
        stamp_size=11,
        write_mode="all",
        gpus=0,
        description="ET mag sampling test",
    )
    options = core.RenderOptions(star_flux_mode="random_et_mag")

    frame0 = core._build_render_config(
        case,
        seed=1,
        global_seed=20260617,
        star_id=7,
        frame_id=0,
        device="cpu",
        render_options=options,
    )
    frame1 = core._build_render_config(
        case,
        seed=2,
        global_seed=20260617,
        star_id=7,
        frame_id=1,
        device="cpu",
        render_options=options,
    )
    other_star = core._build_render_config(
        case,
        seed=3,
        global_seed=20260617,
        star_id=8,
        frame_id=0,
        device="cpu",
        render_options=options,
    )

    assert 12.5 <= frame0.et_mag <= 14.5
    assert frame1.et_mag == pytest.approx(frame0.et_mag)
    assert frame1.star_flux_e_s == pytest.approx(frame0.star_flux_e_s)
    assert other_star.et_mag != pytest.approx(frame0.et_mag)
    assert frame0.star_flux_e_s == pytest.approx(core.et_mag_to_photon_rate_e_s(frame0.et_mag))


def test_resolve_data_path_uses_et_data_dir_for_relative_assets(tmp_path):
    relative = "cosmic_ray/dark_test_10um/event_library_10um.npz"

    resolved = core.resolve_data_path(relative, et_data_dir=tmp_path)

    assert resolved == tmp_path / relative


def test_load_photsim7_psf_stamp_uses_requested_field_and_subpixel_grid(tmp_path):
    bundle_dir = tmp_path / "psf" / "et" / "241006" / "D280mm-focus"
    bundle_dir.mkdir(parents=True)
    subpixel_plane = np.zeros((35, 35), dtype=np.float64)
    subpixel_plane[14:21, 14:21] = 1.0
    payload = {
        "images": {6: {7: np.stack([np.zeros_like(subpixel_plane), np.zeros_like(subpixel_plane), subpixel_plane])}},
        "angles": [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
        "filenames": [""] * 7,
        "pixel_diameter": "10.0 um",
    }
    with (bundle_dir / "sim_psf_images.pkl").open("wb") as handle:
        pickle.dump(payload, handle)

    psf = core.load_photsim7_psf_stamp(
        stamp_size=3,
        psf_bundle_name="psf/et/241006/D280mm-focus",
        psf_field_id=6,
        psf_subpixels=7,
        et_data_dir=tmp_path,
    )

    assert psf.shape == (3, 3)
    assert psf.dtype == np.float32
    assert psf.sum(dtype=np.float64) == pytest.approx(1.0)
    assert psf[1, 1] == pytest.approx(1.0)


def test_default_stamp_render_config_uses_auto_device_and_renders():
    config = core.StampRenderConfig(
        exposure_s=1.0,
        stamp_size=3,
        star_flux_e_s=0.0,
        background_e_s_pix=0.0,
        scattered_light_e_s_pix=0.0,
        dark_e_s_pix=0.0,
        read_noise_10s_e_pix=0.0,
        cosmic_ray_event_rate=0.0,
        use_photsim7_psf=False,
        enable_dynamic_effects=False,
    )

    stamp, metadata = core.render_synthetic_stamp(config)

    assert config.device == "auto"
    assert stamp.shape == (3, 3)
    assert metadata["device"] in {"cpu", "cuda:0"}


def test_photsim7_psf_backend_uses_stamp_renderer(monkeypatch):
    calls = {}

    def fake_render_photsim7_stamp(config):
        calls["config"] = config
        image = np.full((config.stamp_size, config.stamp_size), 7.0, dtype=np.float32)
        metadata = {
            "source": "photsim7.stamp_renderer",
            "use_photsim7_psf": True,
            "psf_bundle_name": config.psf_bundle_name,
            "psf_field_id": config.psf_field_id,
            "psf_subpixels": config.psf_subpixels,
            "dynamic_effects": {"enabled": False},
            "jitter_integrated_psf": {"enabled": False},
        }
        return image, metadata

    monkeypatch.setattr(core, "_render_photsim7_stamp", fake_render_photsim7_stamp, raising=False)
    monkeypatch.setattr(
        core,
        "_source_psf",
        lambda config: (
            np.ones((config.stamp_size, config.stamp_size), dtype=np.float32)
            / float(config.stamp_size * config.stamp_size),
            {"source": "legacy"},
        ),
    )
    monkeypatch.setattr(
        core,
        "_inject_cosmic_rays",
        lambda image, *, config, rng: (image, 0),
    )
    config = core.StampRenderConfig(
        exposure_s=1.0,
        stamp_size=3,
        star_flux_e_s=0.0,
        background_e_s_pix=0.0,
        scattered_light_e_s_pix=0.0,
        dark_e_s_pix=0.0,
        read_noise_10s_e_pix=0.0,
        cosmic_ray_event_rate=0.0,
        use_photsim7_psf=True,
        enable_dynamic_effects=False,
    )

    stamp, metadata = core.render_synthetic_stamp(config)

    assert calls["config"] == config
    assert np.all(stamp == 7.0)
    assert metadata["psf"]["source"] == "photsim7.stamp_renderer"


def test_dynamic_effects_use_main_rd_timeseries_with_exposure_split(monkeypatch):
    calls = {}

    def fake_effects_cached(
        *,
        n_frames,
        exposure_s,
        seed,
        enable_psd_motion,
        psd_motion_path_key,
        enable_dva,
        enable_thermal,
        enable_momentum_dump,
        enable_psf_breathing,
    ):
        calls.update(
            {
                "n_frames": n_frames,
                "exposure_s": exposure_s,
                "seed": seed,
                "enable_psd_motion": enable_psd_motion,
                "psd_motion_path_key": psd_motion_path_key,
                "enable_dva": enable_dva,
                "enable_thermal": enable_thermal,
                "enable_momentum_dump": enable_momentum_dump,
                "enable_psf_breathing": enable_psf_breathing,
            }
        )
        arrays = {
            "time_s": np.array([0.0, 300.0, 600.0, 900.0], dtype=np.float64),
            "total_motion_pix": np.array(
                [[0.0, 0.0], [0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
                dtype=np.float32,
            ),
            "psf_scale": np.array([1.0, 1.01, 1.02, 1.03], dtype=np.float32),
        }
        metadata = {
            "motion_split_hz": 1.0 / 300.0,
            "time_step_s": 300.0,
            "components": {"psd_spacecraft_roll_drift": {"enabled": True}},
        }
        return arrays, metadata

    monkeypatch.setattr(
        core,
        "_main_rd_full_effect_timeseries_cached",
        fake_effects_cached,
        raising=False,
    )
    config = core.StampRenderConfig(
        exposure_s=300.0,
        stamp_size=3,
        use_photsim7_psf=False,
        enable_dynamic_effects=True,
        enable_psd_motion=True,
        enable_dva_drift=True,
        enable_thermal_drift=False,
        enable_momentum_dump=False,
        enable_psf_breathing=True,
        n_frames=4,
        frame_id=2,
        global_seed=20260617,
    )

    dynamic = core._dynamic_effects_for_frame(config)

    assert calls["n_frames"] == 4
    assert calls["exposure_s"] == pytest.approx(300.0)
    assert calls["enable_psd_motion"] is True
    assert calls["enable_dva"] is True
    assert calls["enable_thermal"] is False
    assert dynamic["source"] == "main_rd_g18_parallel.build_full_effect_timeseries"
    assert dynamic["motion_split_hz"] == pytest.approx(1.0 / 300.0)
    assert dynamic["total_offset_pix"] == [pytest.approx(0.3), pytest.approx(0.4)]
    assert dynamic["psf_scale"] == pytest.approx(1.02)


def test_render_photsim7_stamp_passes_jitter_model_and_stamp_local_response(monkeypatch):
    calls = {}
    response_sampler = object()

    class FakeRenderer:
        def render_single_cadence(self, **kwargs):
            calls.update(kwargs)
            return {"final_image": np.zeros((3, 3), dtype=np.float32)}

    monkeypatch.setattr(
        core,
        "_dynamic_effects_for_frame",
        lambda config: {
            "enabled": True,
            "total_offset_pix": [0.25, -0.5],
            "psf_scale": 1.03,
            "components": {},
        },
    )
    monkeypatch.setattr(
        core,
        "_build_photsim7_stamp_renderer",
        lambda config, device: FakeRenderer(),
    )
    monkeypatch.setattr(
        core,
        "_build_stamp_local_response_sampler",
        lambda config, device: response_sampler,
        raising=False,
    )
    config = core.StampRenderConfig(
        exposure_s=30.0,
        stamp_size=3,
        star_flux_e_s=0.0,
        background_e_s_pix=0.0,
        scattered_light_e_s_pix=0.0,
        dark_e_s_pix=0.0,
        read_noise_10s_e_pix=0.0,
        cosmic_ray_event_rate=0.0,
        use_photsim7_psf=True,
        enable_dynamic_effects=True,
        jitter_integrated_psf_models=3,
        frame_id=7,
        star_id=42,
        device="cpu",
    )

    image, metadata = core._render_photsim7_stamp(config)

    assert image.shape == (3, 3)
    assert calls["target_x_offset_pix"] == pytest.approx(0.25)
    assert calls["target_y_offset_pix"] == pytest.approx(-0.5)
    assert calls["psf_scale"] == pytest.approx(1.03)
    assert calls["jitter_model_index"] == 1
    assert calls["detector_response_sampler"] is response_sampler
    assert metadata["jitter_integrated_psf"]["enabled"] is True
    assert metadata["jitter_integrated_psf"]["jitter_model_index"] == 1


def test_build_stamp_local_response_sampler_constructs_photsim7_sampler():
    torch = pytest.importorskip("torch")
    config = core.StampRenderConfig(
        exposure_s=30.0,
        stamp_size=3,
        psf_subpixels=3,
        use_photsim7_psf=True,
        enable_detector_response=True,
        enable_pixel_phase_response=False,
        star_id=7,
        global_seed=20260617,
        device="cpu",
    )

    sampler = core._build_stamp_local_response_sampler(config, "cpu")
    patch = sampler.sample_subpixel_patch(
        y_start_pix=0,
        x_start_pix=0,
        n_rows_pix=3,
        n_cols_pix=3,
    )

    assert patch.shape == (9, 9)
    assert patch.dtype == torch.float32
    assert torch.isfinite(patch).all()


def test_explicit_cuda_device_does_not_silently_fallback_to_cpu(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    config = core.StampRenderConfig(
        exposure_s=1.0,
        stamp_size=3,
        star_flux_e_s=0.0,
        background_e_s_pix=0.0,
        scattered_light_e_s_pix=0.0,
        dark_e_s_pix=0.0,
        read_noise_10s_e_pix=0.0,
        cosmic_ray_event_rate=0.0,
        use_photsim7_psf=False,
        enable_dynamic_effects=False,
        device="cuda:0",
    )

    with pytest.raises(RuntimeError, match="CUDA device requested"):
        core.render_synthetic_stamp(config)


def test_derive_seed_is_stable_and_effect_specific():
    seed_a = core.derive_seed(
        20260617,
        exposure_s=30.0,
        frame_id=12,
        star_id=34,
        effect_type="read_noise",
    )
    seed_b = core.derive_seed(
        20260617,
        exposure_s=30.0,
        frame_id=12,
        star_id=34,
        effect_type="read_noise",
    )
    seed_c = core.derive_seed(
        20260617,
        exposure_s=30.0,
        frame_id=12,
        star_id=34,
        effect_type="cosmic_ray",
    )

    assert seed_a == seed_b
    assert seed_a != seed_c
    assert 0 <= seed_a < 2**32


def test_split_ranges_cover_items_without_overlap():
    ranges = core.split_ranges(10, 3)

    assert ranges == [
        core.IndexRange(0, 4),
        core.IndexRange(4, 7),
        core.IndexRange(7, 10),
    ]
    covered = [index for item in ranges for index in range(item.start, item.stop)]
    assert covered == list(range(10))


def test_benchmark_cases_include_expected_stage_cases():
    smoke = core.benchmark_cases("smoke")
    compute = core.benchmark_cases("compute")
    io = core.benchmark_cases("io")
    physics = core.benchmark_cases("physics")

    assert [case.case_id for case in smoke[:4]] == ["F01", "F02", "F03", "F04"]
    assert {case.case_id for case in compute} >= {"C02", "C04", "C05", "C06"}
    assert next(case for case in io if case.case_id == "I04").n_stars == 1680
    assert [case.case_id for case in physics] == [
        "S1D11E030",
        "S1D11E060",
        "S1D11E180",
        "S1D11E300",
        "S1D15E030",
        "S1D15E060",
        "S1D15E180",
        "S1D15E300",
        "L7D11E030",
        "L7D11E060",
        "L7D11E180",
        "L7D11E300",
        "L7D15E030",
        "L7D15E060",
        "L7D15E180",
        "L7D15E300",
    ]
    assert {case.exposure_s for case in physics} == {30.0, 60.0, 180.0, 300.0}
    assert 120.0 not in {case.exposure_s for case in physics}

    s1d11e030 = next(case for case in physics if case.case_id == "S1D11E030")
    assert s1d11e030.n_stars == 1680
    assert s1d11e030.n_frames == 2880
    assert s1d11e030.stamp_size == 11
    assert s1d11e030.gpus == 3

    l7d15e300 = next(case for case in physics if case.case_id == "L7D15E300")
    assert l7d15e300.n_stars == 240
    assert l7d15e300.n_frames == 2016
    assert l7d15e300.stamp_size == 15
    assert l7d15e300.gpus == 3


def test_io_case_dry_run_uses_cpu_worker_count_without_gpu_binding(tmp_path):
    case = core.benchmark_cases("io")[0]
    render_options = core.RenderOptions(
        star_flux_e_s=321.0,
        cosmic_ray_library_path="/tmp/photsim7-cr-events.npz",
    )

    summary = core.run_case(
        case,
        output_root=tmp_path,
        workers_per_gpu=7,
        gpus="0,1,2",
        global_seed=20260617,
        dry_run=True,
        render_options=render_options,
    )

    assert case.gpus == 0
    assert summary["gpu_ids"] == []
    assert summary["world_size"] == 7
    assert summary["render_options"]["star_flux_e_s"] == pytest.approx(321.0)
    assert summary["render_options"]["cosmic_ray_library_path"] == "/tmp/photsim7-cr-events.npz"


def test_dry_run_reports_expected_files_for_stamp_scale_case(tmp_path):
    case = next(case for case in core.benchmark_cases("physics") if case.case_id == "S1D15E180")

    summary = core.run_case(
        case,
        output_root=tmp_path,
        workers_per_gpu=10,
        gpus="0,1,2",
        global_seed=20260617,
        dry_run=True,
        render_options=core.RenderOptions(),
    )

    assert summary["estimated_stamps"] == 806400
    assert summary["expected_files"] == 806400
    assert summary["expected_payload_bytes"] == 806400 * 15 * 15 * 4


def test_run_stage_filters_stamp_scale_group_in_dry_run(tmp_path):
    results = core.run_stage(
        "physics",
        output_root=tmp_path,
        workers_per_gpu=10,
        gpus="0,1,2",
        global_seed=20260617,
        dry_run=True,
        render_options=core.RenderOptions(),
        matrix_preset="stamp_scale_v2",
        scale_groups=["long_low_star"],
    )

    case_ids = [result["case"]["case_id"] for result in results]
    assert case_ids == [
        "L7D11E030",
        "L7D11E060",
        "L7D11E180",
        "L7D11E300",
        "L7D15E030",
        "L7D15E060",
        "L7D15E180",
        "L7D15E300",
    ]


def test_cosmic_ray_library_is_converted_from_adu_to_electrons(tmp_path):
    library_path = tmp_path / "cosmic_ray_events.npz"
    np.savez(library_path, stamps=np.asarray([[[2.0]]], dtype=np.float32))
    config = core.StampRenderConfig(
        exposure_s=1.0,
        stamp_size=3,
        star_flux_e_s=0.0,
        background_e_s_pix=0.0,
        scattered_light_e_s_pix=0.0,
        dark_e_s_pix=0.0,
        read_noise_10s_e_pix=0.0,
        gain_e_per_adu=10.0,
        cosmic_ray_event_rate=1.0,
        cosmic_ray_library_path=str(library_path),
        pixel_size_um=10000.0,
        use_photsim7_psf=False,
        enable_dynamic_effects=False,
        seed=123,
    )

    stamp, metadata = core.render_synthetic_stamp(config)
    positive = stamp[stamp > 0]

    assert metadata["actual_cosmic_events"] > 0
    assert positive.size > 0
    assert np.allclose(np.mod(positive, 20.0), 0.0)


def test_render_synthetic_stamp_is_float32_finite_and_reproducible():
    config = core.StampRenderConfig(
        exposure_s=30.0,
        stamp_size=11,
        star_flux_e_s=100.0,
        background_e_s_pix=0.2,
        scattered_light_e_s_pix=0.0,
        dark_e_s_pix=0.01,
        read_noise_10s_e_pix=6.0,
        gain_e_per_adu=1.4,
        cosmic_ray_event_rate=0.0,
        use_photsim7_psf=False,
        enable_dynamic_effects=False,
        seed=12345,
    )

    stamp_a, meta_a = core.render_synthetic_stamp(config)
    stamp_b, meta_b = core.render_synthetic_stamp(config)

    assert stamp_a.shape == (11, 11)
    assert stamp_a.dtype == np.float32
    assert np.all(np.isfinite(stamp_a))
    assert np.array_equal(stamp_a, stamp_b)
    assert meta_a["n_coadd_equiv"] == pytest.approx(3.0)
    assert meta_a["read_noise_e_pix"] == pytest.approx(6.0 * np.sqrt(3.0))
    assert meta_b["seed"] == meta_a["seed"]


def test_write_stamp_and_manifest_round_trip(tmp_path):
    stamp = np.arange(121, dtype=np.float32).reshape(11, 11)
    record = core.write_stamp_npy(
        output_root=tmp_path,
        case_id="T01",
        exposure_s=30.0,
        stamp=stamp,
        star_id=7,
        frame_id=3,
        stamp_size=11,
        seed=99,
        write_mode="all",
    )
    manifest_path = core.write_manifest(tmp_path / "manifest.csv", [record])

    saved = np.load(record.file_path)
    assert saved.dtype == np.float32
    assert np.array_equal(saved, stamp)
    assert manifest_path.exists()

    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["case_id"] == "T01"
    assert rows[0]["unit"] == "electrons"
    assert rows[0]["dtype"] == "float32"
    assert rows[0]["file_path"] == str(record.file_path)


def test_write_stamp_npy_skips_existing_valid_file_for_resume(tmp_path):
    first_stamp = np.arange(121, dtype=np.float32).reshape(11, 11)
    second_stamp = np.full((11, 11), -99.0, dtype=np.float32)
    first_record = core.write_stamp_npy(
        output_root=tmp_path,
        case_id="T01",
        exposure_s=30.0,
        stamp=first_stamp,
        star_id=7,
        frame_id=3,
        stamp_size=11,
        seed=99,
        write_mode="all",
    )

    second_record = core.write_stamp_npy(
        output_root=tmp_path,
        case_id="T01",
        exposure_s=30.0,
        stamp=second_stamp,
        star_id=7,
        frame_id=3,
        stamp_size=11,
        seed=100,
        write_mode="all",
    )

    saved = np.load(first_record.file_path)
    assert second_record.status == "skipped_existing"
    assert second_record.file_path == first_record.file_path
    assert second_record.file_size_bytes == first_record.file_size_bytes
    assert np.array_equal(saved, first_stamp)
    assert not list(Path(first_record.file_path).parent.glob("*.tmp.*"))


def test_run_case_writes_worker_manifest_shards(tmp_path):
    case = core.BenchmarkCase(
        "T02",
        "test",
        n_stars=2,
        exposure_s=30.0,
        n_frames=2,
        stamp_size=3,
        write_mode="all",
        gpus=0,
        description="manifest shard test",
    )
    render_options = core.RenderOptions(
        star_flux_e_s=0.0,
        background_e_s_pix=0.0,
        scattered_light_e_s_pix=0.0,
        dark_e_s_pix=0.0,
        read_noise_10s_e_pix=0.0,
        cosmic_ray_event_rate=0.0,
        use_photsim7_psf=False,
        enable_dynamic_effects=False,
    )

    summary = core.run_case(
        case,
        output_root=tmp_path,
        workers_per_gpu=2,
        gpus="",
        global_seed=20260617,
        render_options=render_options,
    )

    case_dir = tmp_path / "T02"
    manifest_index = case_dir / "manifest_index.json"
    assert manifest_index.exists()
    assert not (case_dir / "manifest.csv").exists()
    assert summary["manifest_index_path"] == str(manifest_index)
    assert summary["expected_files"] == 4
    assert summary["n_stamps"] == 4
    assert summary["n_written"] == 4
    assert summary["n_failed"] == 0

    manifest_paths = sorted((case_dir / "manifests").glob("manifest.worker*.csv"))
    assert [path.name for path in manifest_paths] == [
        "manifest.worker000.csv",
        "manifest.worker001.csv",
    ]
    rows = []
    for path in manifest_paths:
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    assert len(rows) == 4
    assert {row["status"] for row in rows} == {"completed"}
