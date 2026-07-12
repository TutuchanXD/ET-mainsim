"""ET main_rd full-frame, 120-cadence simulation configuration.

This script is a focused copy of et_sim_10_etpsd3-2.py for evaluating the
full et_focalplane main_rd detector footprint.  It intentionally defaults to a
configuration/validation mode instead of launching a Ray Simulator run because
the legacy Simulator.run path requires square detector frames, while main_rd is
rectangular: 9120 rows x 8900 columns.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from astropy import units as u


def env_path(name: str, default: str | Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser()


def default_photsim7_root(et_root: Path) -> Path:
    for dirname in ("Photsim7", "Photosim7"):
        candidate = et_root / dirname
        if candidate.exists():
            return candidate
    return et_root / "Photsim7"


ET_ROOT = env_path("ET_ROOT", "/home/cxgao/ET")
PHOTSIM7_ROOT = env_path("PHOTSIM7_ROOT", default_photsim7_root(ET_ROOT))
PHOTSIM7_DATA_DIR = env_path(
    "PHOTSIM7_DATA_DIR",
    os.environ.get("ET_DATA_DIR", str(ET_ROOT / "Photsim7-data")),
)
ET_FOCALPLANE_ROOT = env_path("ET_FOCALPLANE_ROOT", ET_ROOT / "et_focalplane")

CONFIG_XLSX_FULL_PATH = PHOTSIM7_DATA_DIR / "et_100_det_inputs_1h.xlsx"
ET_PSD_FULL_PATH = PHOTSIM7_DATA_DIR / "pds" / "ET_psd3-2.pkl"
DVA_MODEL_FULL_PATH = (
    PHOTSIM7_DATA_DIR / "DVA" / "et" / "ET_DVA_effect_models_slim_v231117.pkl"
)

DETECTOR_ID = "main_rd"
DETECTOR_COLS = 8900
DETECTOR_ROWS = 9120
TARGET_RA_DEG = 304.41406499712303
TARGET_DEC_DEG = 51.81987707392268
TARGET_DETECTOR_XPIX = 4450.0
TARGET_DETECTOR_YPIX = 4560.0
TARGET_FIELD_X_DEG = -6.10175
TARGET_FIELD_Y_DEG = -6.23275
TARGET_FIELD_ANGLE_DEG = float(np.hypot(TARGET_FIELD_X_DEG, TARGET_FIELD_Y_DEG))

EXPOSURE_DURATION = 10.0 * u.s
READOUT_DURATION = 0.0 * u.s
OBSERVING_DURATION = 1200.0 * u.s
SIMULATION_CADENCE_MULT = 1
N_RAW_FRAMES_PER_COADD = 1
TELESCOPE_COUNT = 1

MAG_TYPE = "Gaia G"
MAG_LIMIT = 17.0
MAG_RANGE = [0, MAG_LIMIT]
COMPUTE_DEVICE = "cuda"
FLOAT_PRECISION = 32
STORE_IMAGES = True
SIM_SAVE_DIR = "main_rd_full_120x10s_g17_jipsf100_etpsd3_2"

TESS_JITTER_MULT = 1
JITTER_TIME_STEP = 0.05 * u.s
JITTER_RESUME = True
N_JITTER_INTEGRATED_PSF_MODELS = 100
N_STARS_PER_RUN = 1
RUN_COUNT = 1


def ensure_environment() -> None:
    os.environ.setdefault("ET_DATA_DIR", str(PHOTSIM7_DATA_DIR))
    for path in (PHOTSIM7_ROOT, ET_FOCALPLANE_ROOT / "src"):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


ensure_environment()

from photsim7.config import BASE_DATA_DIR  # noqa: E402
from photsim7.configurator import ConfigurationManager, resolve_detector_shape  # noqa: E402
from photsim7.data_generators import generate_tess_centroid_jitter  # noqa: E402
from photsim7.time import AstronomicalTimeManager, resolve_detector_frame_timing  # noqa: E402
from photsim7.utils import calc_relative_aperture_area  # noqa: E402
from photsim7.variants import VariantManager  # noqa: E402


def load_pickle(filepath: str | Path):
    with open(filepath, "rb") as f:
        return pickle.load(f)


def load_sim_config() -> dict:
    config_manager = ConfigurationManager(filepath=str(CONFIG_XLSX_FULL_PATH))
    params = config_manager.parameters

    # Requested overrides for the main_rd full-frame single-cadence setup.
    params["Simulation Name"] = "ET_main_rd_full_120x10s"
    params["Detector Width"] = DETECTOR_COLS * u.pix
    params["Detector Height"] = DETECTOR_ROWS * u.pix
    params["Telescope Count"] = TELESCOPE_COUNT
    params["Readout Duration"] = READOUT_DURATION
    params["Exposure Duration"] = EXPOSURE_DURATION
    params["Simulation Cadence Mult"] = SIMULATION_CADENCE_MULT
    params["Observing Duration"] = OBSERVING_DURATION
    params["Use Jitter-Integrated PSF"] = True
    params["N Jitter-Integrated PSF Models"] = N_JITTER_INTEGRATED_PSF_MODELS
    params["PSF Field ID"] = "nearest"
    params["Cosmic Ray Event Library Path"] = (
        "cosmic_ray/dark_test_10um/event_library_10um.npz"
    )
    params["Cosmic Ray Event Library Pixel Size"] = 10.0 * u.um

    # Preserve the main detector's physical pixel pitch.
    params["Pixel Width"] = 10.0 * u.um

    return params


def build_main_variant_manager() -> VariantManager:
    variant_manager = VariantManager()
    variant_manager.add_variant(
        description="main_only",
        optimal_aperture="none",
        enable_coadding=False,
    )
    if len(variant_manager.variants) != 1 or len(variant_manager.all_variants) != 1:
        raise RuntimeError("Expected exactly one main variant and no OA variants.")
    return variant_manager


def single_frame_duration_s(sim_config: dict) -> float:
    return (
        sim_config["Simulation Cadence Mult"]
        * (sim_config["Readout Duration"] + sim_config["Exposure Duration"])
    ).to(u.s).value


def jitter_cache_path(sim_config: dict) -> Path:
    n_jit_psf = sim_config["N Jitter-Integrated PSF Models"]
    duration_s = single_frame_duration_s(sim_config)
    jitter_name = (
        f"xy_{TESS_JITTER_MULT}X_jitter_pix_N({n_jit_psf})_"
        f"AFJ(Rescaled_ET_PSD)_exp{duration_s:.2f}s+TESS_rotation_jitter.npy"
    )
    return (
        Path(BASE_DATA_DIR)
        / "jitter"
        / "et"
        / "JI-PSF_data"
        / jitter_name
    )


def load_or_build_xy_jitter_pix(sim_config: dict, et_psd: dict, *, build: bool):
    jitter_fp = jitter_cache_path(sim_config)
    if jitter_fp.is_file() and JITTER_RESUME:
        return np.load(jitter_fp)
    if not build:
        return None

    jitter_fp.parent.mkdir(parents=True, exist_ok=True)
    duration_s = single_frame_duration_s(sim_config)
    et_time = np.arange(0, duration_s, JITTER_TIME_STEP.to(u.s).value) * u.s

    xy_jitter_pix = []
    for _ in range(sim_config["N Jitter-Integrated PSF Models"]):
        x_jit_pix, y_jit_pix, _ = generate_tess_centroid_jitter(
            time=et_time,
            tess_psd=et_psd,
            mult=TESS_JITTER_MULT,
            field_angle=10,
            x_axis_angle=45,
            plot=False,
        )
        xy_jitter_pix.append([x_jit_pix, y_jit_pix])

    np.save(jitter_fp, xy_jitter_pix)
    return np.asarray(xy_jitter_pix)


def make_et_xy_drift_func(sim_config: dict, et_psd: dict):
    pixel_scale = sim_config["Pixel Scale"]

    def generate_et_xy_drift(time, field_angle=10.0, x_axis_angle=45):
        from photsim7.data_generators import PixelSpaceSimulator, generate_tess_jitter

        if len(time) < 2:
            zeros = np.zeros(len(time), dtype=float)
            return zeros, zeros, zeros, zeros, zeros

        et_pss = PixelSpaceSimulator(
            plate_scale=pixel_scale,
            x_center=0.0,
            y_center=0.0,
        )
        ztime_s = (time - time[0]).to(u.s).value
        dt = ztime_s[1] - ztime_s[0]
        f_samp = 1 / dt
        duration = ztime_s[-1]

        x_et_freqs, x_et_psd = et_psd["x"]
        y_et_freqs, y_et_psd = et_psd["y"]
        z_et_freqs, z_et_psd = et_psd["z"]
        theta_x_arcsec, _ = generate_tess_jitter(
            x_et_freqs.value,
            x_et_psd.value,
            frequency_min=0,
            frequency_max=1 / 10,
            duration=duration,
            base_f_samp=f_samp,
            supersample_factor=1,
        )
        theta_y_arcsec, _ = generate_tess_jitter(
            y_et_freqs.value,
            y_et_psd.value,
            frequency_min=0,
            frequency_max=1 / 10,
            duration=duration,
            base_f_samp=f_samp,
            supersample_factor=1,
        )
        theta_z_arcsec, _ = generate_tess_jitter(
            z_et_freqs.value,
            z_et_psd.value,
            frequency_min=0,
            frequency_max=1 / 10,
            duration=duration,
            base_f_samp=f_samp,
            supersample_factor=1,
        )

        fa_theta_rad = x_axis_angle / 180 * np.pi
        x_fa = np.cos(fa_theta_rad) * field_angle
        y_fa = np.sin(fa_theta_rad) * field_angle
        x_pix = np.interp(x_fa, [0, 23.5 / 2], [0, 9000])
        y_pix = np.interp(y_fa, [0, 23.5 / 2], [0, 9000])
        x_pix_new, y_pix_new = et_pss.apply_spacecraft_rotations(
            theta_x_arcsec * u.arcsec,
            theta_y_arcsec * u.arcsec,
            theta_z_arcsec * u.arcsec,
            x_pix,
            y_pix,
        )
        return (
            x_pix - x_pix_new,
            y_pix - y_pix_new,
            theta_x_arcsec,
            theta_y_arcsec,
            theta_z_arcsec,
        )

    return generate_et_xy_drift


def make_thermal_drift_func(sim_config: dict):
    pixel_scale = sim_config["Pixel Scale"]

    def generate_thermal_drift_scaled(time):
        t_day = (time - time[0]).to(u.s).value / (24 * 3600)
        days_per_block = 3.0
        cycles_per_block = 4.0
        frequency = cycles_per_block / days_per_block
        amplitude_arcsec = 0.022
        baseline_step_arcsec = 0.03
        baseline = baseline_step_arcsec * (t_day / days_per_block)
        r_drift_arcsec = baseline + amplitude_arcsec * np.sin(
            2 * np.pi * frequency * t_day
        )
        r_drift_pix = r_drift_arcsec / pixel_scale.value
        theta = 12.0 / 180 * np.pi
        return r_drift_pix * np.cos(theta), r_drift_pix * np.sin(theta)

    return generate_thermal_drift_scaled


def generate_weed_psf_breathing(time):
    ztime_s = (time - time[0]).to(u.s).value
    period_s = 3 * 24 * 3600
    amplitude = 0.01
    cycle_time = (ztime_s % period_s) / period_s
    return 1 - amplitude + 2 * amplitude * cycle_time


def build_dynamic_param_config(sim_config: dict, et_psd: dict) -> dict:
    pixel_scale = sim_config["Pixel Scale"]
    dynamic_param_config = {"motion": []}

    if sim_config["Momentum Dump Model"].lower() != "none":
        jump_length = sim_config["Momentum Dump R(68%)"]
        dynamic_param_config["motion"].append(
            dict(
                component_name="momentum_dump_jumps",
                model_name=sim_config["Momentum Dump Model"],
                model_params=dict(
                    period_step=sim_config["Momentum Dump Cycle"],
                    circle_radius=jump_length.to(u.arcsec).value / pixel_scale.value,
                    r_step_avg=jump_length.to(u.arcsec).value / pixel_scale.value,
                    stay_inside=True,
                    random_r_step=True,
                    step_func=True,
                ),
            )
        )

    dynamic_param_config["motion"].append(
        dict(
            component_name="dva_drift",
            model_name="dva_model",
            model_params=dict(
                dva_model=load_pickle(DVA_MODEL_FULL_PATH),
                psf_field_angle=12.0,
                pixel_scale=pixel_scale.value,
                t0=0.0,
            ),
        )
    )

    dynamic_param_config["motion"].append(
        dict(
            component_name="thermal_drift",
            model_name="user_input_xy_function",
            model_params=dict(func=make_thermal_drift_func(sim_config)),
        )
    )

    et_xy_drift_func = make_et_xy_drift_func(sim_config, et_psd)

    def generate_fft_hf_drift(time):
        x_drift, y_drift, *_ = et_xy_drift_func(time)
        return x_drift, y_drift

    dynamic_param_config["motion"].append(
        dict(
            component_name="fft_hf_drift",
            model_name="user_input_xy_function",
            model_params=dict(func=generate_fft_hf_drift),
        )
    )

    dynamic_param_config["psf_scale"] = [
        dict(
            component_name="psf_scale",
            model_name="user_input_r_function",
            model_params=dict(func=generate_weed_psf_breathing),
        )
    ]

    return dynamic_param_config


def build_time_manager(sim_config: dict) -> AstronomicalTimeManager:
    return AstronomicalTimeManager(
        initial_date=sim_config["Observing Start Date"],
        real_readout=sim_config["Readout Duration"],
        real_exposure=sim_config["Exposure Duration"],
        sim_multiplier=sim_config["Simulation Cadence Mult"],
        sim_duration=sim_config["Observing Duration"],
        n_frames_per_coadd=N_RAW_FRAMES_PER_COADD,
    )


def q_to_str(value) -> str:
    if hasattr(value, "to_string"):
        return value.to_string()
    return str(value)


def print_parameter_report(
    sim_config: dict,
    variant_manager: VariantManager,
    dynamic_param_config: dict,
    *,
    jitter_available: bool,
) -> None:
    n_rows, n_cols = resolve_detector_shape(sim_config)
    frame_timing = resolve_detector_frame_timing(sim_config)
    time_manager = build_time_manager(sim_config)
    readout_noise = sim_config["Readout Noise"].to(u.electron / u.pix).value
    sim_readout_noise = (
        readout_noise
        * np.sqrt(sim_config["Simulation Cadence Mult"] * N_RAW_FRAMES_PER_COADD)
    )
    momentum_step_pix = (
        sim_config["Momentum Dump R(68%)"].to(u.arcsec).value
        / sim_config["Pixel Scale"].value
    )

    print("=== ET main_rd full-frame single-cadence configuration ===")
    print(f"config_xlsx: {CONFIG_XLSX_FULL_PATH}")
    print(f"base_data_dir: {BASE_DATA_DIR}")
    print(f"psd_path: {ET_PSD_FULL_PATH}")
    print(f"dva_model_path: {DVA_MODEL_FULL_PATH}")
    print(f"detector_id: {DETECTOR_ID}")
    print(f"detector_shape_rows_cols: {n_rows} x {n_cols}")
    print(f"target_ra_dec_deg: {TARGET_RA_DEG}, {TARGET_DEC_DEG}")
    print(f"target_detector_center_pix_xy: {TARGET_DETECTOR_XPIX}, {TARGET_DETECTOR_YPIX}")
    print(f"target_field_xy_deg: {TARGET_FIELD_X_DEG}, {TARGET_FIELD_Y_DEG}")
    print(f"target_field_angle_deg: {TARGET_FIELD_ANGLE_DEG:.6f}")
    print("")
    print("[sim_config]")
    for key in sorted(sim_config):
        if key == "Sim Readout Noise":
            continue
        print(f"{key}: {q_to_str(sim_config[key])}")
    print(f"Derived Sim Readout Noise: {sim_readout_noise:.6g} electron / pix")
    print(f"Relative Aperture Area vs Kepler: {calc_relative_aperture_area(sim_config['Aperture Diameter']).value:.8f}")
    print("")
    print("[time]")
    print(f"raw_frame_integration_s: {frame_timing['raw_frame_integration_s']}")
    print(f"raw_frame_sampling_interval_s: {frame_timing['raw_frame_sampling_interval_s']}")
    print(f"n_raw_frames_per_coadd: {N_RAW_FRAMES_PER_COADD}")
    print(f"n_sim_frames: {time_manager.n_sim_frames}")
    print(f"n_real_frames: {time_manager.n_real_frames}")
    print(f"coadding_enabled: {N_RAW_FRAMES_PER_COADD > 1}")
    print(f"effective_observing_duration_s: {time_manager.sim_duration.sec}")
    print("")
    print("[variants]")
    print(f"base_variant_count: {len(variant_manager.variants)}")
    print(f"all_variant_count: {len(variant_manager.all_variants)}")
    for variant_id, variant in variant_manager.variants.items():
        print(f"variant_{variant_id}: {variant.settings}")
    print("")
    print("[jitter_psf]")
    print(f"use_jitter_integrated_psf: {sim_config['Use Jitter-Integrated PSF']}")
    print(f"tess_jitter_mult: {TESS_JITTER_MULT}")
    print(f"jitter_time_step_s: {JITTER_TIME_STEP.to(u.s).value}")
    print(f"jitter_cache_path: {jitter_cache_path(sim_config)}")
    print(f"jitter_cache_available: {jitter_available}")
    print("")
    print("[dynamic_param_config]")
    print(f"motion_components: {[c['component_name'] for c in dynamic_param_config['motion']]}")
    print("momentum_dump_step_pix_r68:", f"{momentum_step_pix:.8f}")
    print("dva_psf_field_angle_deg: 12.0")
    print("thermal_amplitude_arcsec: 0.022")
    print("thermal_baseline_step_arcsec_per_3day: 0.03")
    print("fft_drift_psd_frequency_range_hz: [0, 0.1]")
    print("jitter_psd_frequency_range_hz: [0.1, inf]")
    print("psf_breathing: 0.99 -> 1.01 linear ramp over 3 days")
    print("")
    print("[execution]")
    print("legacy_simulator_run_compatible: False")
    print("reason: Simulator.run requires square detector frames; main_rd is 9120 x 8900.")
    print("recommended_renderer: photsim7.full_frame_renderer.SingleCadenceFullFrameRenderer")
    print(f"sim_save_dir: {SIM_SAVE_DIR}")
    print(f"run_count: {RUN_COUNT}")
    print(f"n_stars_per_run: {N_STARS_PER_RUN}")
    print(f"mag_type: {MAG_TYPE}")
    print(f"mag_limit: G<{MAG_LIMIT:g}")
    print(f"mag_range: {MAG_RANGE}")
    print(f"compute_device: {COMPUTE_DEVICE}")
    print(f"float_precision: {FLOAT_PRECISION}")
    print(f"store_images: {STORE_IMAGES}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print the adjusted ET main_rd full-frame single-cadence config."
    )
    parser.add_argument(
        "--build-jitter-cache",
        action="store_true",
        help="Generate the 10 s jitter-integrated PSF cache if it is missing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sim_config = load_sim_config()
    et_psd = load_pickle(ET_PSD_FULL_PATH)
    variant_manager = build_main_variant_manager()
    xy_jitter_pix = load_or_build_xy_jitter_pix(
        sim_config,
        et_psd,
        build=args.build_jitter_cache,
    )
    dynamic_param_config = build_dynamic_param_config(sim_config, et_psd)
    print_parameter_report(
        sim_config,
        variant_manager,
        dynamic_param_config,
        jitter_available=xy_jitter_pix is not None,
    )


if __name__ == "__main__":
    main()
