from photsim7.config import common_import_script

# Import managers
from photsim7.simulator import Simulator
from photsim7.configurator import ConfigurationManager
from photsim7.variants import VariantManager

# Import additional utilities
from photsim7.plot import (
    plot_jitter_integrated_psf_models,
    plot_single_frame_tpf,
    plot_gif_tpfs,
    plot_base_psf_models,
    det_motion_and_component_plot,
    meas_jit_drift_rms,
    plot_psd,
)
from photsim7.data_generators import (
    real_freqs,
    real_mags,
    psd_to_motion,
    gen_psd_motion,
    fft_bins_to_name_str,
    fft_bins_to_fullname_str,
    gen_power_spec,
    load_power_spectrum_config,
    create_power_spec_components,
    generate_jitter,
    et_time_to_thermal_drift,
    psd_to_time_series,
    generate_tess_centroid_jitter,
)
from photsim7.utils import smooth_nday
from photsim7.config import BASE_DATA_DIR
from photsim7.utils import save_pickle, load_pickle


import matplotlib.pyplot as plt
import os
from os.path import join as opj
import numpy as np
import torch
from tqdm import trange
from astropy import units as u
import pickle


def load_pickle(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)


"""
Working with this configuration:
------------------------------------------------------------------------------------------
|                         Last run date: 2025-08-25 07:06 PM EDT                         |
------------------------------------------------------------------------------------------
| Python | Jupyterlab | Numpy | Scipy  | Astropy | Matplotlib | Tensorflow |   Pytorch   |
------------------------------------------------------------------------------------------
| 3.12.9 |   4.3.6    | 2.2.4 | 1.15.2 |  7.0.1  |   3.10.1   |    N/A     | 2.6.0+cu126 |
------------------------------------------------------------------------------------------

Author: Kevin Willis
Email: kevin.w.willis@gmail.com
Date: 2025-08-26
"""

config_xlsx_full_path = "et_inputs_251010_10s_2.xlsx"

config_manager = ConfigurationManager(filepath=config_xlsx_full_path)

variant_manager = VariantManager()  # Initialize the manager

variant_manager.add_variant(
    description="0",
    optimal_aperture=config_manager.parameters["Optimal Aperture Algorithim"],
)

pixel_scale = config_manager.parameters["Pixel Scale"]

et_psd = load_pickle("ET_psd3-2.pkl")

tess_jitter_mult = 1  # Multiply the amplitude of TESS jitter

single_sim_exposure_duration_s = (
    (
        config_manager.parameters["Simulation Cadence Mult"]
        * (
            config_manager.parameters["Readout Duration"]
            + config_manager.parameters["Exposure Duration"]
        )
    )
    .to(u.s)
    .value
)

resume = True

n_jit_psf = config_manager.parameters["N Jitter-Integrated PSF Models"]

jit_fn = f"xy_{tess_jitter_mult}X_jitter_pix_N({n_jit_psf})_AFJ(Rescaled_ET_PSD)_exp{single_sim_exposure_duration_s:.2f}s+TESS_rotation_jitter.npy"

jit_fp = opj(BASE_DATA_DIR, "jitter", "et", "JI-PSF_data", jit_fn)

os.makedirs(os.path.dirname(jit_fp), exist_ok=True)

# -------------------------------------------------------------------

if os.path.isfile(jit_fp) and resume:
    print("Loaded jitter")
    xy_jitter_pix = np.load(jit_fp)

else:
    xy_jitter_pix = []

    for i in trange(
        config_manager.parameters["N Jitter-Integrated PSF Models"]
    ):  # For each JI-PSF

        et_time_step = (0.05 * u.s).to(u.s)
        et_time = np.arange(0, single_sim_exposure_duration_s, et_time_step.value) * u.s

        x_jit_pix, y_jit_pix, xyz_jit_theta_as = generate_tess_centroid_jitter(
            time=et_time,
            tess_psd=et_psd,
            mult=tess_jitter_mult,
            field_angle=10,
            x_axis_angle=45,
            plot=False,
        )

        xy_jitter_pix.append([x_jit_pix, y_jit_pix])

    print("Saved jitter")
    np.save(jit_fp, xy_jitter_pix)

jit_fp


def generate_et_xy_drift(time, et_psd, field_angle=10.0, x_axis_angle=45):

    from photsim7.data_generators import PixelSpaceSimulator

    x_center = 0.0
    y_center = 0.0

    et_pss = PixelSpaceSimulator(
        plate_scale=pixel_scale,
        x_center=x_center,
        y_center=y_center,
    )
    # -----------------------------------------------------------------------------------
    # TESS Roll motion to x, y motion

    ztime_s = (time - time[0]).to(u.s).value

    dt = ztime_s[1] - ztime_s[0]

    f_samp = 1 / dt

    duration = ztime_s[-1]

    from photsim7.data_generators import generate_tess_jitter

    x_et_freqs, x_et_psd = et_psd["x"]
    y_et_freqs, y_et_psd = et_psd["y"]
    z_et_freqs, z_et_psd = et_psd["z"]
    theta_x_arcsec, time2 = generate_tess_jitter(
        x_et_freqs.value,
        x_et_psd.value,
        frequency_min=0,
        frequency_max=1 / 10,
        duration=duration,
        base_f_samp=f_samp,
        supersample_factor=1,
    )
    theta_y_arcsec, time2 = generate_tess_jitter(
        y_et_freqs.value,
        y_et_psd.value,
        frequency_min=0,
        frequency_max=1 / 10,
        duration=duration,
        base_f_samp=f_samp,
        supersample_factor=1,
    )
    theta_z_arcsec, time2 = generate_tess_jitter(
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

    x_et_pix = x_pix - x_pix_new
    y_et_pix = y_pix - y_pix_new

    return x_et_pix, y_et_pix, theta_x_arcsec, theta_y_arcsec, theta_z_arcsec


time_example = np.arange(0, 30 * u.min.to(u.s), 0.05) * u.s


x_drift_et_pix, y_drift_et_pix, theta_x_arcsec, theta_y_arcsec, theta_z_arcsec = (
    generate_et_xy_drift(time_example, et_psd)
)


# Define dynamic param configs (DPC)

# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# Build the main dict that holds all DPC
dynamic_param_config = {}
dynamic_param_config["motion"] = []

# ---------------------------------
# Momentum dump of spacecraft reaction wheels

if (
    config_manager.parameters["Momentum Dump Model"].lower() != "none"
):  # Was MD requested?

    # Get user requested MD settings
    jump_cadence = config_manager.parameters["Momentum Dump Cycle"]
    jump_length = config_manager.parameters["Momentum Dump R(68%)"]
    jump_circ_rad = config_manager.parameters["Momentum Dump R(68%)"]

    # Create the MD DPC item
    dynamic_param_config["motion"] += [  # Its motion, so add to the motion catagory
        dict(
            component_name="momentum_dump_jumps",
            model_name=config_manager.parameters["Momentum Dump Model"],
            model_params=dict(
                period_step=jump_cadence,
                circle_radius=jump_circ_rad.to(u.arcsec).value / pixel_scale.value,
                r_step_avg=jump_length.to(u.arcsec).value / pixel_scale.value,
                stay_inside=True,
                random_r_step=True,
                step_func=True,
            ),
        ),
    ]

# ---------------------------------
# DVA motion

dva_model_data = load_pickle(
    opj(BASE_DATA_DIR, "DVA", "et", "ET_DVA_effect_models_slim_v231117.pkl")
)  # Load DVA model (created from JWST code)

dynamic_param_config["motion"] += [
    dict(
        component_name="dva_drift",
        model_name="dva_model",  # Built-in DVA model
        model_params=dict(
            dva_model=dva_model_data,
            psf_field_angle=12.0,  # Field angle (FA) of 12 deg is where mots stars will be observed in ET
            pixel_scale=pixel_scale.value,
            t0=0.0,
        ),
    ),
]

# ---------------------------------
# Thermal drift due to thermal defocus


def generate_thermal_drift_scaled(time):
    t_day = (time - time[0]).to(u.s).value / (24 * 3600)  # Convert to days
    days_per_block = 3.0
    cycles_per_block = 4.0
    frequency = cycles_per_block / days_per_block
    amplitude = 0.022
    baseline_step = 0.03
    baseline = baseline_step * (t_day / days_per_block)
    r_drift_arcsec = baseline + amplitude * np.sin(2 * np.pi * frequency * t_day)

    r_drift_pix = r_drift_arcsec / pixel_scale.value
    theta = 12.0 / 180 * np.pi
    x_drift = r_drift_pix * np.cos(theta)
    y_drift = r_drift_pix * np.sin(theta)

    return x_drift, y_drift


dynamic_param_config["motion"] += [
    dict(
        component_name="thermal_drift",
        model_name="user_input_xy_function",
        model_params=dict(
            func=generate_thermal_drift_scaled,
        ),
    ),
]


# ---------------------------------
# Other drifts (from power spectrum)
def generate_fft_hf_drift(time):

    x_drift_et_pix, y_drift_et_pix, theta_x_arcsec, theta_y_arcsec, theta_z_arcsec = (
        generate_et_xy_drift(time, et_psd)
    )

    return x_drift_et_pix, y_drift_et_pix


dynamic_param_config["motion"] += [
    dict(
        component_name="fft_hf_drift",
        model_name="user_input_xy_function",  # Tells the generator to use a custom model function
        model_params=dict(
            func=generate_fft_hf_drift,
        ),
    ),
]


# ---------------------------------
# WEED PSF breathing function
def generate_weed_psf_breathing(time):
    ztime_s = (time - time[0]).to(u.s).value
    ztime_days = ztime_s / (24 * 3600)  # Convert to days
    period_days = 3
    period_s = period_days * 24 * 3600  # Convert to seconds
    amplitude = 0.01  # 1% amplitude
    cycle_time = (ztime_s % period_s) / period_s
    breathing_factor = 1 - amplitude + 2 * amplitude * cycle_time
    return breathing_factor


dynamic_param_config["psf_scale"] = []

dynamic_param_config["psf_scale"] += [
    dict(
        component_name="psf_scale",
        model_name="user_input_r_function",
        model_params=dict(
            func=generate_weed_psf_breathing,  # WEED PSF breathing: 3% sinusoidal variation, 1-10 day period
        ),
    ),
]

dynamic_param_config["motion"]

inject_transits = False
# inject_transits = True

if inject_transits:

    n_exop = 20  # How many transits to be injected in a batch of stars.
    n_stars = 500  # How many stars will you simulate per batch? Typically simulate 500 stars per batch.

    transit_data = dict(
        star_ids=np.sort(
            np.random.permutation(n_stars)[:n_exop]
        ),  # In the batch, which star IDs will have transits injected?
        star_mags=np.full(
            n_exop, 13.0
        ),  # What should be the magnitude of each star with transits?
        to_img_gen=[],  # The relative flux profile for each transit. If not provided, it will be autmatically generated for you.
    )

else:
    transit_data = None

transit_data

simulator = Simulator(
    mag_type="ET",  # Which magnitude system do you want to use in the simulation? ['ET' or 'Kepler']
    sim_config=config_manager.parameters,
    dynamic_param_config=dynamic_param_config,
    variant_manager=variant_manager,
    n_raw_frames_per_coadd=180,
    xy_jitter_pix=xy_jitter_pix,
    # compute_device='cpu',
    compute_device="cuda",
    float_precision=32,
    mag_range=[7, 17],
    # mag_range=[12.5, 12.5],
    store_images=True,
    transit_data=transit_data,
    ray_cluster_address=None,
    ray_namespace=None,
    # ray_cluster_address='ray://100.125.174.11:10001',
    # ray_namespace="persistent_actors",
)

simulator.time_manager.summary

# Confirm the readout noise for coadded frames (RN * sqrt(n_raw_frames_per_coadd))
simulator.sim_config["Sim Readout Noise"]


def store_new_dynamic_params(self):

    self.data_generation_manager.store_variant_dynamic_parameters(
        self.time,
        self.n_raw_frames,
        self.dynamic_param_config,
        self.variant_manager,
        self.observatory,
        sim_config=self.sim_config,
        target_star_data=self.star_data,
        will_convole_jitter=self.convole_jitter,
        compute_device=self.compute_device,
        torch_dtype=self.torch_dtype,
    )


store_new_dynamic_params(simulator)

from photsim7.plot import centroid_motion_and_component_plot_with_r95

better_titles = dict(
    total="Total Motion",
    dva_drift="DVA",
    momentum_dump_jumps="Momentum Dumps",
    thermal_drift="Thermal Drift",
    fft_hf_drift="Other Drift",
)

from photsim7.lc_processing import flux_bin_pytorch

et_time_binned, et_x_centroid_binned = flux_bin_pytorch(
    simulator.ztime.to(u.day),
    xy_tot[1][None, :],
    bin_width=30 * u.min,
    device="cpu",
    batch_size=1000,
)
et_x_centroid_binned = et_x_centroid_binned[0]


np.std(et_x_centroid_binned) * pixel_scale.value

# Name the simulation (needs to be a valid folder name)
sim_save_dir = "test"

sim_save_dir

dir0 = opj(BASE_DATA_DIR, "simulations", "ET", sim_save_dir)

os.makedirs(dir0, exist_ok=True)

transit_data_fp = opj(dir0, "transit_data.pkl")

if os.path.isfile(transit_data_fp):
    tmp = load_pickle(opj(dir0, "transit_data.pkl"))
    if tmp is not None:
        simulator.transit_data["to_img_gen"] = tmp["to_img_gen"]
    print("Loaded transit config!")
else:
    save_pickle(transit_data_fp, transit_data)
    print("Saved transit config!")

# Run the simulation

simulator.run(
    n_stars_per_run=500,  # How many stars per batch?
    run_count=10,  # How many batches?
    sim_save_dir=sim_save_dir,  # Save folder name
    resume=False,  # DOES NOT WORK YET! Resume simulation? If stopped, will resume after last completed batch.
)
