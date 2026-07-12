from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import subprocess
import sys
import time
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
GAIA_CATALOG_DIR = env_path("GAIA_CATALOG_DIR", "/home/cxgao/gaia_dr3_19mag")
RESULTS_ROOT = env_path("RESULTS_ROOT", "/home/cxgao/Results/ET-mainsim/main_rd_g18_parallel")
DEFAULT_MAG_DISTRIBUTION_CSV = PHOTSIM7_DATA_DIR / "ET_mag" / "310-50-2420.csv"
DEFAULT_DETECTOR_XY_CSV = (
    PHOTSIM7_DATA_DIR / "ET_mag" / "310-50-2420_square_detector_xy.csv"
)
DEFAULT_DETECTOR_XY_700_CSV = (
    PHOTSIM7_DATA_DIR / "ET_mag" / "310-50-2420_square700pix_glt24_detector_xy.csv"
)

DETECTOR_ID = "main_rd"
TARGET_RA_DEG = 304.41406499712303
TARGET_DEC_DEG = 51.81987707392268
TARGET_FIELD_X_DEG = -6.10175
TARGET_FIELD_Y_DEG = -6.23275
TARGET_FIELD_ANGLE_DEG = float(np.hypot(TARGET_FIELD_X_DEG, TARGET_FIELD_Y_DEG))
TARGET_DETECTOR_XPIX = 4450.0
TARGET_DETECTOR_YPIX = 4560.0

MAG_LIMIT = 18.0
PIXEL_SCALE = 4.83 * u.arcsec / u.pix
PIXEL_WIDTH = 10.0 * u.um
EXPOSURE = 10.0 * u.s
OBSERVING_DURATION = 1800.0 * u.s
N_FRAMES = 180

SKY_SURFACE_BRIGHTNESS = 22.0
DARK_CURRENT = 1.0 * u.electron / u.s / u.pix
SCATTERED_LIGHT = 0.0 * u.electron / u.s / u.pix
READOUT_NOISE = 6.0 * u.electron / u.pix
FULL_WELL_ELECTRONS = 90680.0
GAIN_ELECTRONS_PER_ADU = 1.4
ADC_BIT_DEPTH = 16
BIAS_LEVEL_ADU = 3500.0
COLUMN_NOISE_SIGMA_ADU = 5.0
COSMIC_RAY_EVENT_RATE = 5.0 / (u.cm**2 * u.s)
COSMIC_RAY_LIBRARY_PATH = "cosmic_ray/dark_test_10um/event_library_10um.npz"
COSMIC_RAY_PIXEL_SIZE = 10.0 * u.um

PSF_BUNDLE_NAME = "241006/D280mm-focus"
N_SUBPIXELS = 1
JITTER_INTEGRATED_PSF_MODELS = 300
JITTER_FRAMES_PER_MODEL = 600
MOTION_SPLIT_HZ = 1.0 / EXPOSURE.to(u.s).value
INTER_PIXEL_RESPONSE_SIGMA = 0.01
INTRA_PIXEL_RESPONSE_SIGMA = 0.01
INTER_PIXEL_RESPONSE_NOMINAL = 1.0

REFERENCE_EFFECT_FIELD_ANGLE_DEG = 10.0
REFERENCE_EFFECT_X_AXIS_ANGLE_DEG = 45.0
DVA_FIELD_ANGLE_DEG = 12.0
DVA_THETA_DEG = 12.0
THERMAL_THETA_DEG = 12.0
THERMAL_AMPLITUDE_ARCSEC = 0.022
THERMAL_BASELINE_STEP_ARCSEC_PER_3DAY = 0.03
THERMAL_DAYS_PER_BLOCK = 3.0
THERMAL_CYCLES_PER_BLOCK = 4.0
WEED_PSF_BREATHING_PERIOD_DAY = 3.0
WEED_PSF_BREATHING_AMPLITUDE = 0.01
MOMENTUM_DUMP_MODEL = "random_walk_within_circle"
MOMENTUM_DUMP_CYCLE_DAY = 3.0
MOMENTUM_DUMP_R68_ARCSEC = 0.15


@dataclass(frozen=True)
class MainRdRunSpec:
    frame_rows: int
    frame_cols: int
    run_label: str | None = None
    detector_id: str = DETECTOR_ID
    mag_limit: float = MAG_LIMIT
    star_source: str = "gaia_main_rd"
    mag_distribution_csv: str = str(DEFAULT_MAG_DISTRIBUTION_CSV)
    mag_distribution_column: str = "mwmsc_gmag"
    detector_xy_csv: str = str(DEFAULT_DETECTOR_XY_CSV)
    detector_xy_source_id_column: str = "source_id"
    detector_xy_mag_column: str = "gmag"
    detector_xy_x_column: str = "x0"
    detector_xy_y_column: str = "y0"
    synthetic_psf_field_angle_deg: float = TARGET_FIELD_ANGLE_DEG
    target_ra_deg: float = TARGET_RA_DEG
    target_dec_deg: float = TARGET_DEC_DEG
    target_field_x_deg: float = TARGET_FIELD_X_DEG
    target_field_y_deg: float = TARGET_FIELD_Y_DEG
    target_field_angle_deg: float = TARGET_FIELD_ANGLE_DEG
    target_detector_xpix: float = TARGET_DETECTOR_XPIX
    target_detector_ypix: float = TARGET_DETECTOR_YPIX
    pixel_scale_arcsec_per_pix: float = 4.83
    pixel_width_um: float = 10.0
    exposure_s: float = 10.0
    n_frames: int = N_FRAMES
    observing_duration_s: float = 1800.0
    sky_surface_brightness_mag_arcsec2: float = SKY_SURFACE_BRIGHTNESS
    dark_current_e_s_pix: float = 1.0
    scattered_light_e_s_pix: float = 0.0
    scattered_light_step_start_frame: int | None = None
    scattered_light_step_e_pix_frame: float = 0.0
    readout_noise_e_pix: float = 6.0
    full_well_electrons: float = FULL_WELL_ELECTRONS
    gain_electrons_per_adu: float = GAIN_ELECTRONS_PER_ADU
    adc_bit_depth: int = ADC_BIT_DEPTH
    bias_level_adu: float = BIAS_LEVEL_ADU
    column_noise_sigma_adu: float = COLUMN_NOISE_SIGMA_ADU
    cosmic_ray_event_rate_cm2_s: float = 5.0
    cosmic_ray_library_path: str = COSMIC_RAY_LIBRARY_PATH
    cosmic_ray_pixel_size_um: float = 10.0
    psf_bundle_name: str = PSF_BUNDLE_NAME
    n_subpixels: int = N_SUBPIXELS
    use_jitter_integrated_psf: bool = True
    n_jitter_integrated_psf_models: int = JITTER_INTEGRATED_PSF_MODELS
    n_jitter_frames_per_model: int = JITTER_FRAMES_PER_MODEL
    motion_split_hz: float = MOTION_SPLIT_HZ
    enable_dva_drift: bool = True
    dva_field_angle_deg: float = DVA_FIELD_ANGLE_DEG
    dva_theta_deg: float = DVA_THETA_DEG
    enable_thermal_drift: bool = True
    thermal_theta_deg: float = THERMAL_THETA_DEG
    enable_momentum_dump: bool = True
    momentum_dump_model: str = MOMENTUM_DUMP_MODEL
    momentum_dump_cycle_day: float = MOMENTUM_DUMP_CYCLE_DAY
    momentum_dump_r68_arcsec: float = MOMENTUM_DUMP_R68_ARCSEC
    enable_psf_breathing: bool = True
    psf_breathing_model: str = "weed_linear_3day"
    psf_breathing_amplitude: float = WEED_PSF_BREATHING_AMPLITUDE
    inter_pixel_response_sigma: float = INTER_PIXEL_RESPONSE_SIGMA
    intra_pixel_response_sigma: float = INTRA_PIXEL_RESPONSE_SIGMA
    notes: str = (
        "main_rd center crop. Scattered light and flat-field correction are disabled. "
        "Readout noise is applied after full-well clipping and before gain conversion. "
        "PSD motion is split at the single-frame cadence: slower terms move the frame "
        "centroid, faster terms are integrated into the PSF. DVA drift, thermal "
        "drift, momentum dumps, and WEED PSF breathing follow et_sim_10_etpsd3-2.py."
    )


def ensure_local_imports() -> None:
    os.environ.setdefault("ET_DATA_DIR", str(PHOTSIM7_DATA_DIR))
    photsim7_src = PHOTSIM7_ROOT / "photsim7"
    if not photsim7_src.exists():
        if importlib.util.find_spec("photsim7") is None:
            raise FileNotFoundError(
                f"Photsim7 source not found and no installed photsim7 package is importable: "
                f"{photsim7_src}"
            )
    else:
        existing = sys.modules.get("photsim7")
        if existing is None or not hasattr(existing, "__path__"):
            pkg = types.ModuleType("photsim7")
            pkg.__path__ = [str(photsim7_src)]
            pkg.__package__ = "photsim7"
            sys.modules["photsim7"] = pkg

    import_paths = [ET_FOCALPLANE_ROOT / "src"]
    if photsim7_src.exists():
        import_paths.append(PHOTSIM7_ROOT)
    for path in import_paths:
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def require_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "main_rd_g18_parallel rendering requires torch. Install torch before "
            "running the simulation entrypoints."
        ) from exc
    return torch


def _json_default(value: Any) -> Any:
    if hasattr(value, "to_string"):
        return value.to_string()
    if hasattr(value, "value") and hasattr(value, "unit"):
        return f"{value.value} {value.unit}"
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path | str, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=_json_default)


def sim_config_dict(
    frame_rows: int,
    frame_cols: int,
    *,
    spec: MainRdRunSpec | None = None,
    sky_surface_brightness_mag_arcsec2: float | None = None,
    n_subpixels: int | None = None,
    scattered_light_e_s_pix: float | None = None,
) -> dict[str, Any]:
    ensure_local_imports()
    from photsim7.background import sky_surface_brightness_to_background_flux

    if spec is None:
        pixel_scale = PIXEL_SCALE
        pixel_width = PIXEL_WIDTH
        exposure = EXPOSURE
        observing_duration = OBSERVING_DURATION
        sky_surface_brightness_mag_arcsec2 = (
            SKY_SURFACE_BRIGHTNESS
            if sky_surface_brightness_mag_arcsec2 is None
            else float(sky_surface_brightness_mag_arcsec2)
        )
        n_subpixels = N_SUBPIXELS if n_subpixels is None else int(n_subpixels)
        scattered_light_e_s_pix = (
            0.0 if scattered_light_e_s_pix is None else float(scattered_light_e_s_pix)
        )
        dark_current = DARK_CURRENT
        readout_noise = READOUT_NOISE
        full_well_electrons = FULL_WELL_ELECTRONS
        gain_electrons_per_adu = GAIN_ELECTRONS_PER_ADU
        adc_bit_depth = ADC_BIT_DEPTH
        bias_level_adu = BIAS_LEVEL_ADU
        column_noise_sigma_adu = COLUMN_NOISE_SIGMA_ADU
        cosmic_ray_library_path = COSMIC_RAY_LIBRARY_PATH
        cosmic_ray_pixel_size = COSMIC_RAY_PIXEL_SIZE
        cosmic_ray_event_rate = COSMIC_RAY_EVENT_RATE
        psf_bundle_name = PSF_BUNDLE_NAME
        use_jitter_integrated_psf = True
        n_jitter_integrated_psf_models = JITTER_INTEGRATED_PSF_MODELS
        n_jitter_frames_per_model = JITTER_FRAMES_PER_MODEL
    else:
        pixel_scale = float(spec.pixel_scale_arcsec_per_pix) * u.arcsec / u.pix
        pixel_width = float(spec.pixel_width_um) * u.um
        exposure = float(spec.exposure_s) * u.s
        observing_duration = float(spec.observing_duration_s) * u.s
        sky_surface_brightness_mag_arcsec2 = (
            float(spec.sky_surface_brightness_mag_arcsec2)
            if sky_surface_brightness_mag_arcsec2 is None
            else float(sky_surface_brightness_mag_arcsec2)
        )
        n_subpixels = int(spec.n_subpixels if n_subpixels is None else n_subpixels)
        scattered_light_e_s_pix = (
            float(spec.scattered_light_e_s_pix)
            if scattered_light_e_s_pix is None
            else float(scattered_light_e_s_pix)
        )
        dark_current = float(spec.dark_current_e_s_pix) * u.electron / u.s / u.pix
        readout_noise = float(spec.readout_noise_e_pix) * u.electron / u.pix
        full_well_electrons = float(spec.full_well_electrons)
        gain_electrons_per_adu = float(spec.gain_electrons_per_adu)
        adc_bit_depth = int(spec.adc_bit_depth)
        bias_level_adu = float(spec.bias_level_adu)
        column_noise_sigma_adu = float(spec.column_noise_sigma_adu)
        cosmic_ray_library_path = spec.cosmic_ray_library_path
        cosmic_ray_pixel_size = float(spec.cosmic_ray_pixel_size_um) * u.um
        cosmic_ray_event_rate = float(spec.cosmic_ray_event_rate_cm2_s) / (u.cm**2 * u.s)
        psf_bundle_name = spec.psf_bundle_name
        use_jitter_integrated_psf = bool(spec.use_jitter_integrated_psf)
        n_jitter_integrated_psf_models = int(spec.n_jitter_integrated_psf_models)
        n_jitter_frames_per_model = int(spec.n_jitter_frames_per_model)

    background_flux = sky_surface_brightness_to_background_flux(
        sky_surface_brightness_mag_arcsec2,
        pixel_scale,
        magnitude_system="ET",
    )
    return {
        "Detector Width": int(frame_cols) * u.pix,
        "Detector Height": int(frame_rows) * u.pix,
        "Subpixels Per Pixel Dim": int(n_subpixels),
        "Pixel Scale": pixel_scale,
        "Pixel Width": pixel_width,
        "Exposure Duration": exposure,
        "Observing Duration": observing_duration,
        "Simulation Cadence Mult": 1,
        "Background Flux": background_flux,
        "Sky Background Mode": "surface_brightness",
        "Sky Background Surface Brightness": sky_surface_brightness_mag_arcsec2,
        "Sky Background Magnitude System": "ET",
        "Subtract Nonstellar Mean": False,
        "Dark Current": dark_current,
        "Scattered Light": float(scattered_light_e_s_pix) * u.electron / u.s / u.pix,
        "Readout Noise": readout_noise,
        "Enable ADC Digitization": True,
        "Full Well Electrons": full_well_electrons * u.electron,
        "Gain Electrons Per ADU": gain_electrons_per_adu * u.electron / u.adu,
        "ADC Bit Depth": adc_bit_depth,
        "ADC Min Value": 0.0,
        "ADC Round Values": True,
        "Bias Level ADU": bias_level_adu * u.adu,
        "Column Noise Sigma ADU": column_noise_sigma_adu * u.adu,
        "Save Bias Metadata": True,
        "Enable Cosmic Rays": True,
        "Cosmic Ray Event Library Path": cosmic_ray_library_path,
        "Cosmic Ray Event Library Pixel Size": cosmic_ray_pixel_size,
        "Cosmic Ray Event Rate": cosmic_ray_event_rate,
        "Cosmic Ray Seed": 0,
        "Inter-PRV (RMS)": 1.0 * u.percent,
        "Inter-PRV (Nominal)": 100.0 * u.percent,
        "Intra-PRV (RMS)": 1.0 * u.percent,
        "Pixel Response Profile Mod": "flux conserved",
        "Enable Flat Field Correction": False,
        "Flat Field Uncertainty": 0.0 * u.percent,
        "Enable DVA Drifts": False,
        "Thermal Defocus Model": "none",
        "Thermal Defocus Amplitude": 0.0 * u.percent,
        "Thermal Defocus Offset": 100.0 * u.percent,
        "PSF Bundle Name": psf_bundle_name,
        "PSF Field ID": "nearest",
        "Use Jitter-Integrated PSF": use_jitter_integrated_psf,
        "N Jitter-Integrated PSF Models": n_jitter_integrated_psf_models,
        "N Jitter Frames Per Model": n_jitter_frames_per_model,
        "Telescope Count": 1,
        "Optical Efficiency Ratio": 101.0 * u.percent,
    }


_ET_REGISTRY_CACHE = None


def _et_registry():
    global _ET_REGISTRY_CACHE
    if _ET_REGISTRY_CACHE is None:
        ensure_local_imports()
        from et_coord import load_registry

        _ET_REGISTRY_CACHE = load_registry(ET_FOCALPLANE_ROOT / "data")
    return _ET_REGISTRY_CACHE


def main_rd_field_geometry_from_absolute_detector_xy(
    detector_xpix,
    detector_ypix,
    *,
    detector_id: str = DETECTOR_ID,
) -> dict[str, np.ndarray]:
    """Return field coordinates and field angle for main_rd detector pixels."""
    ensure_local_imports()
    from et_coord.geometry import bilinear_forward_many

    registry = _et_registry()
    detector = registry.get_detector(detector_id)

    xpix = np.asarray(detector_xpix, dtype=float)
    ypix = np.asarray(detector_ypix, dtype=float)
    xpix, ypix = np.broadcast_arrays(xpix, ypix)
    flat_xpix = xpix.ravel()
    flat_ypix = ypix.ravel()

    u = flat_xpix / float(detector.pixel_width)
    v = flat_ypix / float(detector.pixel_height)
    field_xy = bilinear_forward_many(detector.field_corners, u, v).reshape(
        xpix.shape + (2,)
    )
    field_x = field_xy[..., 0]
    field_y = field_xy[..., 1]
    field_angle = np.hypot(field_x, field_y)
    return {
        "field_x_deg": field_x.astype(float, copy=False),
        "field_y_deg": field_y.astype(float, copy=False),
        "field_angle_deg": field_angle.astype(float, copy=False),
        "field_polar_angle_rad": np.arctan2(field_y, field_x).astype(
            float,
            copy=False,
        ),
    }


def main_rd_field_geometry_from_frame_offsets(x0, y0) -> dict[str, np.ndarray]:
    """Map crop/frame-relative star offsets to main_rd field geometry.

    The parallel renderer stores star positions as offsets from the configured
    target detector center.  The et_focalplane transform expects absolute
    detector pixels, so this helper restores the main_rd absolute coordinates
    before evaluating the field angle.
    """
    abs_xpix = TARGET_DETECTOR_XPIX + np.asarray(x0, dtype=float)
    abs_ypix = TARGET_DETECTOR_YPIX + np.asarray(y0, dtype=float)
    return main_rd_field_geometry_from_absolute_detector_xy(abs_xpix, abs_ypix)


def star_summary(star_data: dict[str, Any]) -> dict[str, Any]:
    n_stars = int(len(star_data["x0"]))
    summary: dict[str, Any] = {"n_stars": n_stars}
    if n_stars == 0:
        return summary
    et_mag = star_et_magnitude(star_data)
    x0 = np.asarray(star_data["x0"], dtype=float)
    y0 = np.asarray(star_data["y0"], dtype=float)
    field_angle = np.asarray(star_data.get("field_angle_deg", []), dtype=float)
    summary.update(
        {
            "et_mag_min": float(np.min(et_mag)),
            "et_mag_p50": float(np.percentile(et_mag, 50)),
            "et_mag_p90": float(np.percentile(et_mag, 90)),
            "et_mag_max": float(np.max(et_mag)),
            "x0_min": float(np.min(x0)),
            "x0_max": float(np.max(x0)),
            "y0_min": float(np.min(y0)),
            "y0_max": float(np.max(y0)),
        }
    )
    if "kp_mag" in star_data:
        kp_mag = np.asarray(star_data["kp_mag"], dtype=float)
        summary.update(
            {
                "kp_mag_min": float(np.min(kp_mag)),
                "kp_mag_p50": float(np.percentile(kp_mag, 50)),
                "kp_mag_p90": float(np.percentile(kp_mag, 90)),
                "kp_mag_max": float(np.max(kp_mag)),
            }
        )
    if "gaia_g_mag" in star_data:
        gaia_g = np.asarray(star_data["gaia_g_mag"], dtype=float)
        summary.update(
            {
                "gaia_g_mag_min": float(np.min(gaia_g)),
                "gaia_g_mag_max": float(np.max(gaia_g)),
            }
        )
    if "gmag" in star_data:
        gmag = np.asarray(star_data["gmag"], dtype=float)
        summary.update(
            {
                "gmag_min": float(np.min(gmag)),
                "gmag_max": float(np.max(gmag)),
            }
        )
    if field_angle.size == n_stars:
        summary.update(
            {
                "field_angle_deg_min": float(np.min(field_angle)),
                "field_angle_deg_p50": float(np.percentile(field_angle, 50)),
                "field_angle_deg_p90": float(np.percentile(field_angle, 90)),
                "field_angle_deg_max": float(np.max(field_angle)),
            }
        )
    return summary


def star_et_magnitude(star_data: dict[str, Any]) -> np.ndarray:
    if "et_mag" in star_data:
        return np.asarray(star_data["et_mag"], dtype=float)
    if "kp_mag" in star_data:
        return np.asarray(star_data["kp_mag"], dtype=float)
    raise KeyError("star_data must contain 'et_mag' or legacy 'kp_mag'")


def star_data_for_photsim7_catalog(star_data: dict[str, Any]) -> dict[str, Any]:
    if "kp_mag" in star_data:
        return star_data
    if "et_mag" not in star_data:
        raise KeyError("star_data must contain 'et_mag' or legacy 'kp_mag'")
    adapted = dict(star_data)
    adapted["kp_mag"] = np.asarray(star_data["et_mag"], dtype=float)
    return adapted


def query_main_rd_stars(
    *,
    frame_rows: int,
    frame_cols: int,
    mag_limit: float,
    catalog_dir: Path | str,
    crop_margin_pix: float,
) -> dict[str, Any]:
    ensure_local_imports()
    from photsim7.field import mk_real_field_stars_et_focalplane

    return mk_real_field_stars_et_focalplane(
        target_ra=TARGET_RA_DEG * u.deg,
        target_dec=TARGET_DEC_DEG * u.deg,
        catalog_dir=Path(catalog_dir).expanduser(),
        registry_data_dir=ET_FOCALPLANE_ROOT / "data",
        px_rows=int(frame_rows),
        px_cols=int(frame_cols),
        apply_offset=False,
        mag_lim=float(mag_limit),
        detector_id=DETECTOR_ID,
        crop_to_simulation_frame=True,
        crop_margin_pix=float(crop_margin_pix),
        et_focalplane_src=ET_FOCALPLANE_ROOT / "src",
    )


def build_synthetic_mag_distribution_stars(
    *,
    csv_path: Path | str,
    mag_column: str,
    mag_limit: float,
    frame_rows: int,
    frame_cols: int,
    seed: int,
    psf_field_angle_deg: float,
) -> dict[str, Any]:
    csv_path = Path(csv_path).expanduser()
    if not csv_path.exists():
        raise FileNotFoundError(f"Magnitude distribution CSV not found: {csv_path}")

    magnitudes: list[float] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or mag_column not in reader.fieldnames:
            raise ValueError(
                f"Magnitude column {mag_column!r} not found in {csv_path}; "
                f"available columns: {reader.fieldnames}"
            )
        for line_number, row in enumerate(reader, start=2):
            raw_mag = row.get(mag_column, "")
            try:
                magnitude = float(raw_mag)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid magnitude value at {csv_path}:{line_number}: {raw_mag!r}"
                ) from exc
            if np.isfinite(magnitude) and magnitude <= float(mag_limit):
                magnitudes.append(magnitude)

    if not magnitudes:
        raise ValueError(
            f"No stars with {mag_column} <= {float(mag_limit):g} found in {csv_path}"
        )

    mags = np.asarray(magnitudes, dtype=float)
    rng = np.random.default_rng(int(seed))
    x_abs = rng.uniform(0.0, float(frame_cols - 1), size=len(mags))
    y_abs = rng.uniform(0.0, float(frame_rows - 1), size=len(mags))
    x0 = x_abs - (int(frame_cols) - 1) / 2.0
    y0 = y_abs - (int(frame_rows) - 1) / 2.0
    field_geometry = main_rd_field_geometry_from_frame_offsets(x0, y0)

    return {
        "x0": np.asarray(x0, dtype=float),
        "y0": np.asarray(y0, dtype=float),
        "kp_mag": mags,
        "ra": np.full(len(mags), TARGET_RA_DEG, dtype=float),
        "dec": np.full(len(mags), TARGET_DEC_DEG, dtype=float),
        "source_id": np.arange(len(mags), dtype=np.int64),
        "gaia_g_mag": mags.copy(),
        "detector_xpix": np.asarray(x_abs, dtype=float),
        "detector_ypix": np.asarray(y_abs, dtype=float),
        "detector_xpix_shifted": np.asarray(x_abs, dtype=float),
        "detector_ypix_shifted": np.asarray(y_abs, dtype=float),
        "field_x_deg": field_geometry["field_x_deg"],
        "field_y_deg": field_geometry["field_y_deg"],
        "field_angle_deg": field_geometry["field_angle_deg"],
        "field_polar_angle_rad": field_geometry["field_polar_angle_rad"],
    }


def _read_required_float(row: dict[str, str], column: str, *, csv_path: Path, line_number: int) -> float:
    raw_value = row.get(column, "")
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"Invalid float value in column {column!r} at {csv_path}:{line_number}: "
            f"{raw_value!r}"
        ) from exc
    if not np.isfinite(value):
        raise ValueError(
            f"Non-finite float value in column {column!r} at {csv_path}:{line_number}: "
            f"{raw_value!r}"
        )
    return value


def build_detector_xy_stars(
    *,
    csv_path: Path | str,
    frame_rows: int,
    frame_cols: int,
    psf_field_angle_deg: float,
    source_id_column: str = "source_id",
    mag_column: str = "gmag",
    x_column: str = "x0",
    y_column: str = "y0",
) -> dict[str, Any]:
    csv_path = Path(csv_path).expanduser()
    if not csv_path.exists():
        raise FileNotFoundError(f"Detector-xy CSV not found: {csv_path}")

    required_columns = [source_id_column, mag_column, x_column, y_column]
    source_ids: list[int] = []
    et_mags: list[float] = []
    x0_values: list[float] = []
    y0_values: list[float] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [
            column
            for column in required_columns
            if reader.fieldnames is None or column not in reader.fieldnames
        ]
        if missing:
            raise ValueError(
                f"Detector-xy CSV {csv_path} is missing required columns {missing}; "
                f"available columns: {reader.fieldnames}"
            )
        for line_number, row in enumerate(reader, start=2):
            raw_source_id = row.get(source_id_column, "")
            try:
                source_id = int(raw_source_id)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid source_id value at {csv_path}:{line_number}: "
                    f"{raw_source_id!r}"
                ) from exc
            et_mag = _read_required_float(
                row,
                mag_column,
                csv_path=csv_path,
                line_number=line_number,
            )
            x0 = _read_required_float(
                row,
                x_column,
                csv_path=csv_path,
                line_number=line_number,
            )
            y0 = _read_required_float(
                row,
                y_column,
                csv_path=csv_path,
                line_number=line_number,
            )
            source_ids.append(source_id)
            et_mags.append(et_mag)
            x0_values.append(x0)
            y0_values.append(y0)

    if not et_mags:
        raise ValueError(f"Detector-xy CSV contains no stars: {csv_path}")

    et_mag_arr = np.asarray(et_mags, dtype=float)
    x0_arr = np.asarray(x0_values, dtype=float)
    y0_arr = np.asarray(y0_values, dtype=float)
    x_shifted = x0_arr + (int(frame_cols) - 1) / 2.0
    y_shifted = y0_arr + (int(frame_rows) - 1) / 2.0
    field_geometry = main_rd_field_geometry_from_frame_offsets(x0_arr, y0_arr)
    return {
        "x0": x0_arr,
        "y0": y0_arr,
        "et_mag": et_mag_arr,
        "gmag": et_mag_arr.copy(),
        "ra": np.full(len(et_mag_arr), TARGET_RA_DEG, dtype=float),
        "dec": np.full(len(et_mag_arr), TARGET_DEC_DEG, dtype=float),
        "source_id": np.asarray(source_ids, dtype=np.int64),
        "detector_xpix": np.asarray(x_shifted, dtype=float),
        "detector_ypix": np.asarray(y_shifted, dtype=float),
        "detector_xpix_shifted": np.asarray(x_shifted, dtype=float),
        "detector_ypix_shifted": np.asarray(y_shifted, dtype=float),
        "field_x_deg": field_geometry["field_x_deg"],
        "field_y_deg": field_geometry["field_y_deg"],
        "field_angle_deg": field_geometry["field_angle_deg"],
        "field_polar_angle_rad": field_geometry["field_polar_angle_rad"],
    }


def run_dir_name(spec: MainRdRunSpec, mag_limit: float) -> str:
    if spec.run_label:
        return str(spec.run_label)
    mag_tag = f"{float(mag_limit):g}".replace(".", "p")
    return f"main_rd_{spec.frame_cols}x{spec.frame_rows}_g_lt_{mag_tag}"


def star_cache_path(output_root: Path, spec: MainRdRunSpec, mag_limit: float) -> Path:
    name = run_dir_name(spec, mag_limit)
    return output_root / name / "cache" / f"stars_{name}.npz"


def legacy_star_cache_path(output_root: Path, frame_rows: int, frame_cols: int, mag_limit: float) -> Path:
    mag_tag = f"{float(mag_limit):g}".replace(".", "p")
    return (
        output_root
        / f"main_rd_{frame_cols}x{frame_rows}_g_lt_{mag_tag}"
        / "cache"
        / f"stars_main_rd_{frame_cols}x{frame_rows}_g_lt_{mag_tag}.npz"
    )


def save_star_cache(path: Path, star_data: dict[str, Any], metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n_stars = len(star_data["x0"])
    payload: dict[str, Any] = {}
    for key, value in star_data.items():
        arr = np.asarray(value)
        if arr.ndim == 1 and len(arr) == n_stars:
            payload[key] = arr
    payload["__metadata_json__"] = np.array(json.dumps(metadata, default=_json_default))
    np.savez_compressed(path, **payload)


def load_star_cache(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    with np.load(path, allow_pickle=True) as data:
        metadata = json.loads(str(data["__metadata_json__"].item()))
        star_data = {
            key: np.asarray(data[key])
            for key in data.files
            if key != "__metadata_json__"
        }
    return star_data, metadata


def prepare_star_cache(args: argparse.Namespace, spec: MainRdRunSpec) -> Path:
    output_root = Path(args.output_root).expanduser()
    cache_path = star_cache_path(output_root, spec, args.mag_limit)
    if cache_path.exists() and not args.force_star_cache:
        print(f"[Star cache] reuse {cache_path}")
        return cache_path

    start = time.perf_counter()
    if spec.star_source == "gaia_main_rd":
        print(
            "[Star cache] querying Gaia catalog "
            f"detector={DETECTOR_ID} frame={spec.frame_cols}x{spec.frame_rows} G<{args.mag_limit:g}"
        )
        star_data = query_main_rd_stars(
            frame_rows=spec.frame_rows,
            frame_cols=spec.frame_cols,
            mag_limit=args.mag_limit,
            catalog_dir=args.catalog_dir,
            crop_margin_pix=args.crop_margin_pix,
        )
        source_metadata = {
            "star_source": spec.star_source,
            "catalog_dir": str(Path(args.catalog_dir).expanduser()),
            "crop_margin_pix": float(args.crop_margin_pix),
        }
    elif spec.star_source == "synthetic_mag_distribution":
        print(
            "[Star cache] building synthetic magnitude-distribution catalog "
            f"frame={spec.frame_cols}x{spec.frame_rows} {spec.mag_distribution_column}<={args.mag_limit:g} "
            f"csv={spec.mag_distribution_csv}"
        )
        star_data = build_synthetic_mag_distribution_stars(
            csv_path=spec.mag_distribution_csv,
            mag_column=spec.mag_distribution_column,
            mag_limit=args.mag_limit,
            frame_rows=spec.frame_rows,
            frame_cols=spec.frame_cols,
            seed=int(args.seed),
            psf_field_angle_deg=float(spec.synthetic_psf_field_angle_deg),
        )
        source_metadata = {
            "star_source": spec.star_source,
            "mag_distribution_csv": str(Path(spec.mag_distribution_csv).expanduser()),
            "mag_distribution_column": spec.mag_distribution_column,
            "gaia_g_mag_treated_as_et_mag": True,
            "synthetic_position_seed": int(args.seed),
            "synthetic_position_model": "uniform independent x/y centers within the image bounds",
            "synthetic_psf_field_angle_deg": float(spec.synthetic_psf_field_angle_deg),
        }
    elif spec.star_source == "detector_xy_csv":
        print(
            "[Star cache] building detector-xy catalog "
            f"frame={spec.frame_cols}x{spec.frame_rows} csv={spec.detector_xy_csv}"
        )
        star_data = build_detector_xy_stars(
            csv_path=spec.detector_xy_csv,
            frame_rows=spec.frame_rows,
            frame_cols=spec.frame_cols,
            psf_field_angle_deg=float(spec.synthetic_psf_field_angle_deg),
            source_id_column=spec.detector_xy_source_id_column,
            mag_column=spec.detector_xy_mag_column,
            x_column=spec.detector_xy_x_column,
            y_column=spec.detector_xy_y_column,
        )
        source_metadata = {
            "star_source": spec.star_source,
            "detector_xy_csv": str(Path(spec.detector_xy_csv).expanduser()),
            "detector_xy_source_id_column": spec.detector_xy_source_id_column,
            "detector_xy_mag_column": spec.detector_xy_mag_column,
            "detector_xy_x_column": spec.detector_xy_x_column,
            "detector_xy_y_column": spec.detector_xy_y_column,
            "gmag_used_directly_as_et_mag": True,
            "position_model": (
                "CSV x0/y0 are used directly as detector-centered pixel coordinates"
            ),
            "round_positions": False,
            "reproject_positions": False,
            "synthetic_psf_field_angle_deg": float(spec.synthetic_psf_field_angle_deg),
        }
    else:
        raise ValueError(
            f"Unsupported star_source={spec.star_source!r}; "
            "expected 'gaia_main_rd', 'synthetic_mag_distribution', or 'detector_xy_csv'"
        )
    elapsed = time.perf_counter() - start
    metadata = {
        "spec": asdict(spec),
        "mag_limit": float(args.mag_limit),
        "query_elapsed_s": float(elapsed),
        "summary": star_summary(star_data),
        **source_metadata,
    }
    save_star_cache(cache_path, star_data, metadata)
    write_json(cache_path.with_suffix(".summary.json"), metadata)
    print(
        f"[Star cache] saved {cache_path} "
        f"stars={metadata['summary']['n_stars']} elapsed={elapsed:.2f}s"
    )
    return cache_path


def resolve_or_prepare_star_cache(args: argparse.Namespace, spec: MainRdRunSpec) -> Path:
    if args.star_cache is not None:
        cache_path = Path(args.star_cache).expanduser()
        if not cache_path.exists():
            raise FileNotFoundError(f"--star-cache does not exist: {cache_path}")
        print(f"[Star cache] reuse explicit {cache_path}")
        return cache_path
    return prepare_star_cache(args, spec)


def build_star_catalog(
    *,
    star_data: dict[str, Any],
    frame_rows: int,
    frame_cols: int,
    psf_field_ids: np.ndarray,
    frame_exposure_s: float = EXPOSURE.to(u.s).value,
):
    ensure_local_imports()
    from photsim7.field import Stars

    catalog_star_data = star_data_for_photsim7_catalog(star_data)
    stars = Stars()
    stars.build_catalog(
        catalog_star_data,
        frame_exposure=float(frame_exposure_s) * u.s,
        optical_eff_ratio=1.0,
        aperture_area_ratio=1.0,
        mag_type="ET",
    )
    stars.catalog["Detector Xpix Shifted"] = (
        np.asarray(star_data["x0"], dtype=float) + (int(frame_cols) - 1) / 2.0
    )
    stars.catalog["Detector Ypix Shifted"] = (
        np.asarray(star_data["y0"], dtype=float) + (int(frame_rows) - 1) / 2.0
    )
    stars.catalog["Field ID"] = np.asarray(psf_field_ids, dtype=np.int64)
    return stars.catalog


def build_psf_manager(
    *,
    frame_rows: int,
    frame_cols: int,
    device: str,
    star_data: dict[str, Any],
    n_subpixels: int,
    psf_bundle_name: str = PSF_BUNDLE_NAME,
    pixel_scale_arcsec_per_pix: float = PIXEL_SCALE.to(u.arcsec / u.pix).value,
    integrate_jitter: bool,
    xy_jitter_pix: np.ndarray | None,
    n_jitter_integrated_psf_models: int,
    n_jitter_frames_per_model: int,
) -> tuple[Any, np.ndarray, dict[int, int]]:
    ensure_local_imports()
    from photsim7.psf.model import PSFModelManager

    actor_config = {
        "bundle_name": psf_bundle_name,
        "pixel_scale": float(pixel_scale_arcsec_per_pix) * u.arcsec / u.pix,
        "n_rows": int(frame_rows),
        "n_cols": int(frame_cols),
        "n_subpixels": int(n_subpixels),
        "integrate_jitter": bool(integrate_jitter),
        "n_jitter_integrated_psf_models": int(n_jitter_integrated_psf_models),
        "n_jitter_frames": int(n_jitter_frames_per_model),
        "compute_device": device,
        "float_precision": 32,
    }
    probe = PSFModelManager(
        config=actor_config,
        warp_frame_batch_size=10,
        xy_jitter_pix=None,
        intialize=False,
        build_jit_int_models=False,
        pad_to_detector_shape=False,
    )
    probe.load_bundle_data()
    if "field_angle_deg" in star_data:
        field_angles = np.asarray(star_data["field_angle_deg"], dtype=float)
    else:
        field_angles = np.full(len(star_data["x0"]), TARGET_FIELD_ANGLE_DEG, dtype=float)
    psf_field_ids = probe.map_angles_to_field_ids(field_angles)
    unique_field_ids = np.unique(psf_field_ids).astype(int)
    manager = PSFModelManager(
        config=actor_config,
        warp_frame_batch_size=10,
        xy_jitter_pix=xy_jitter_pix,
        intialize=True,
        build_jit_int_models=bool(integrate_jitter),
        field_ids=unique_field_ids,
        pad_to_detector_shape=False,
    )
    field_id_counts = {
        int(field_id): int(np.count_nonzero(psf_field_ids == field_id))
        for field_id in unique_field_ids
    }
    return manager, psf_field_ids, field_id_counts


def build_detector_response_sampler(
    *,
    frame_rows: int,
    frame_cols: int,
    n_subpixels: int,
    device: str,
    seed: int,
):
    ensure_local_imports()
    from photsim7.full_frame_renderer import LazySubpixelResponseSampler

    return LazySubpixelResponseSampler(
        n_rows=int(frame_rows),
        n_cols=int(frame_cols),
        n_subpixels=int(n_subpixels),
        inter_pixel_response_sigma=INTER_PIXEL_RESPONSE_SIGMA,
        inter_pixel_nominal_response=INTER_PIXEL_RESPONSE_NOMINAL,
        intra_pixel_response_sigma=INTRA_PIXEL_RESPONSE_SIGMA,
        pixel_response_profile_mod="flux conserved",
        enable_inter_pixel_response=True,
        enable_intra_pixel_response=True,
        enable_pixel_phase_response=True,
        random_seed=int(seed),
        compute_device=device,
        float_precision=32,
    )


def make_renderer(
    *,
    sim_config: dict[str, Any],
    frame_exposure_s: float,
    device: str,
    stars,
    psf_model_manager,
    detector_response_sampler,
):
    ensure_local_imports()
    from photsim7.full_frame_renderer import SingleCadenceFullFrameRenderer

    return SingleCadenceFullFrameRenderer(
        sim_config=sim_config,
        stars=stars,
        psf_model_manager=psf_model_manager,
        frame_exposure=float(frame_exposure_s) * u.s,
        detector_response_sampler=detector_response_sampler,
        compute_device=device,
        float_precision=32,
    )


def frame_motion_offsets(
    *,
    n_frames: int,
    seed: int,
    enable_psd_motion: bool,
    psd_motion_path: Path | None,
    exposure_s: float = EXPOSURE.to(u.s).value,
    motion_split_hz: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not enable_psd_motion:
        offsets = np.zeros((int(n_frames), 2), dtype=np.float32)
        return offsets, {"enabled": False, "reason": "disabled"}

    if psd_motion_path is None:
        raise ValueError("--enable-psd-motion requires --psd-motion-path")
    psd_motion_path = Path(psd_motion_path).expanduser()
    if not psd_motion_path.exists():
        raise FileNotFoundError(f"PSD motion file not found: {psd_motion_path}")

    psd = load_psd_motion(psd_motion_path)
    rng = np.random.default_rng(int(seed))
    exposure_s = float(exposure_s)
    if exposure_s <= 0.0:
        raise ValueError(f"exposure_s must be positive, got {exposure_s}")
    time_s = np.arange(int(n_frames), dtype=np.float64) * exposure_s
    split_hz = float(motion_split_hz) if motion_split_hz is not None else 1.0 / exposure_s
    theta_arcsec = {
        axis: psd_axis_motion(
            psd,
            axis=axis,
            time_s=time_s,
            rng=rng,
            frequency_min_hz=0.0,
            frequency_max_hz=split_hz,
        )
        for axis in ("x", "y", "z")
    }
    offsets_pix = spacecraft_roll_drift_from_angles(
        theta_x_arcsec=theta_arcsec["x"],
        theta_y_arcsec=theta_arcsec["y"],
        theta_z_arcsec=theta_arcsec["z"],
        field_angle_deg=REFERENCE_EFFECT_FIELD_ANGLE_DEG,
        x_axis_angle_deg=REFERENCE_EFFECT_X_AXIS_ANGLE_DEG,
    )
    metadata = {
        "enabled": True,
        "path": str(psd_motion_path),
        "seed": int(seed),
        "model": "reference low-frequency ET PSD roll/pitch/yaw drift, frequencies <= split_hz",
        "exposure_s": float(exposure_s),
        "split_hz": float(split_hz),
        "field_angle_deg": REFERENCE_EFFECT_FIELD_ANGLE_DEG,
        "x_axis_angle_deg": REFERENCE_EFFECT_X_AXIS_ANGLE_DEG,
        "theta_x_rms_arcsec": float(np.std(theta_arcsec["x"])),
        "theta_y_rms_arcsec": float(np.std(theta_arcsec["y"])),
        "theta_z_rms_arcsec": float(np.std(theta_arcsec["z"])),
        "rms_x_pix": float(np.std(offsets_pix[:, 0])),
        "rms_y_pix": float(np.std(offsets_pix[:, 1])),
        "max_abs_x_pix": float(np.max(np.abs(offsets_pix[:, 0]))),
        "max_abs_y_pix": float(np.max(np.abs(offsets_pix[:, 1]))),
    }
    return offsets_pix.astype(np.float32), metadata


def legacy_direct_xy_psd_offsets(
    *,
    psd: dict[str, Any],
    rng: np.random.Generator,
    time_s: np.ndarray,
    split_hz: float,
) -> np.ndarray:
    offsets_arcsec = [
        psd_axis_motion(
            psd,
            axis=axis,
            time_s=time_s,
            rng=rng,
            frequency_min_hz=0.0,
            frequency_max_hz=split_hz,
        )
        for axis in ("x", "y")
    ]
    offsets_arcsec_arr = np.vstack(offsets_arcsec).T
    return offsets_arcsec_arr / PIXEL_SCALE.to(u.arcsec / u.pix).value


def load_psd_motion(path: Path | str) -> dict[str, Any]:
    import pickle

    path = Path(path).expanduser()
    with path.open("rb") as handle:
        return pickle.load(handle)


def psd_axis_motion(
    psd: dict[str, Any],
    *,
    axis: str,
    time_s: np.ndarray,
    rng: np.random.Generator,
    frequency_min_hz: float,
    frequency_max_hz: float | None,
    max_frequency_samples: int = 20000,
) -> np.ndarray:
    freqs_q, psd_q = psd[axis]
    freqs_hz = np.asarray(freqs_q.to(u.Hz).value, dtype=np.float64)
    psd_arcsec2_hz = np.asarray(psd_q.to((u.arcsec**2) / u.Hz).value, dtype=np.float64)
    mask = (freqs_hz > float(frequency_min_hz)) & np.isfinite(freqs_hz) & np.isfinite(
        psd_arcsec2_hz
    )
    if frequency_max_hz is not None:
        mask &= freqs_hz <= float(frequency_max_hz)
    freqs_hz = freqs_hz[mask]
    psd_arcsec2_hz = np.clip(psd_arcsec2_hz[mask], 0.0, None)
    if freqs_hz.size == 0:
        return np.zeros_like(time_s, dtype=np.float64)
    if freqs_hz.size > int(max_frequency_samples):
        idx = np.linspace(0, freqs_hz.size - 1, int(max_frequency_samples)).astype(int)
        freqs_hz = freqs_hz[idx]
        psd_arcsec2_hz = psd_arcsec2_hz[idx]
    df = np.gradient(freqs_hz)
    amplitudes = np.sqrt(np.clip(psd_arcsec2_hz * df, 0.0, None))
    phases = rng.uniform(0.0, 2.0 * np.pi, size=freqs_hz.size)
    return np.sum(
        amplitudes[:, None]
        * np.sin(2.0 * np.pi * freqs_hz[:, None] * time_s[None, :] + phases[:, None]),
        axis=0,
    )


def spacecraft_roll_drift_from_angles(
    *,
    theta_x_arcsec: np.ndarray,
    theta_y_arcsec: np.ndarray,
    theta_z_arcsec: np.ndarray,
    field_angle_deg: float,
    x_axis_angle_deg: float,
) -> np.ndarray:
    ensure_local_imports()
    from photsim7.data_generators import PixelSpaceSimulator

    simulator = PixelSpaceSimulator(
        plate_scale=PIXEL_SCALE,
        x_center=0.0,
        y_center=0.0,
    )
    fa_theta_rad = float(x_axis_angle_deg) / 180.0 * np.pi
    x_fa = np.cos(fa_theta_rad) * float(field_angle_deg)
    y_fa = np.sin(fa_theta_rad) * float(field_angle_deg)
    x_pix = np.interp(x_fa, [0.0, 23.5 / 2.0], [0.0, 9000.0])
    y_pix = np.interp(y_fa, [0.0, 23.5 / 2.0], [0.0, 9000.0])
    x_pix_new, y_pix_new = simulator.apply_spacecraft_rotations(
        np.asarray(theta_x_arcsec, dtype=np.float64) * u.arcsec,
        np.asarray(theta_y_arcsec, dtype=np.float64) * u.arcsec,
        np.asarray(theta_z_arcsec, dtype=np.float64) * u.arcsec,
        x_pix,
        y_pix,
    )
    return np.vstack([x_pix - x_pix_new, y_pix - y_pix_new]).T


def generate_dva_drift_offsets(*, time_s: np.ndarray, enabled: bool) -> tuple[np.ndarray, dict[str, Any]]:
    if not enabled:
        return np.zeros((len(time_s), 2), dtype=np.float32), {"enabled": False}
    ensure_local_imports()
    from photsim7.config import BASE_DATA_DIR, opj
    from photsim7.data_generators import DataGenerationManager
    from photsim7.utils import load_pickle

    dva_model_path = opj(BASE_DATA_DIR, "DVA", "et", "ET_DVA_effect_models_slim_v231117.pkl")
    dva_model = load_pickle(dva_model_path)
    dgm = DataGenerationManager()
    time = np.asarray(time_s, dtype=np.float64) * u.s
    dx, dy = dgm.sample_dva_model(
        time,
        dva_model=dva_model,
        psf_field_angle=DVA_FIELD_ANGLE_DEG,
        pixel_scale=PIXEL_SCALE.to(u.arcsec / u.pix).value,
        theta=np.deg2rad(DVA_THETA_DEG),
        t0=0.0,
    )
    offsets = np.vstack([dx, dy]).T.astype(np.float32)
    return offsets, {
        "enabled": True,
        "model": "dva_model",
        "reference": "et_sim_10_etpsd3-2.py",
        "dva_model_path": str(dva_model_path),
        "field_angle_deg": DVA_FIELD_ANGLE_DEG,
        "theta_deg": DVA_THETA_DEG,
        "t0_day": 0.0,
        "rms_x_pix": float(np.std(offsets[:, 0])),
        "rms_y_pix": float(np.std(offsets[:, 1])),
        "max_abs_x_pix": float(np.max(np.abs(offsets[:, 0]))),
        "max_abs_y_pix": float(np.max(np.abs(offsets[:, 1]))),
    }


def generate_thermal_drift_offsets(
    *,
    time_s: np.ndarray,
    enabled: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not enabled:
        return np.zeros((len(time_s), 2), dtype=np.float32), {"enabled": False}
    t_day = (np.asarray(time_s, dtype=np.float64) - float(time_s[0])) / (24.0 * 3600.0)
    frequency = THERMAL_CYCLES_PER_BLOCK / THERMAL_DAYS_PER_BLOCK
    baseline = THERMAL_BASELINE_STEP_ARCSEC_PER_3DAY * (t_day / THERMAL_DAYS_PER_BLOCK)
    r_drift_arcsec = baseline + THERMAL_AMPLITUDE_ARCSEC * np.sin(
        2.0 * np.pi * frequency * t_day
    )
    r_drift_pix = r_drift_arcsec / PIXEL_SCALE.to(u.arcsec / u.pix).value
    theta = np.deg2rad(THERMAL_THETA_DEG)
    offsets = np.vstack([r_drift_pix * np.cos(theta), r_drift_pix * np.sin(theta)]).T
    offsets = offsets.astype(np.float32)
    return offsets, {
        "enabled": True,
        "model": "generate_thermal_drift_scaled",
        "reference": "et_sim_10_etpsd3-2.py",
        "theta_deg": THERMAL_THETA_DEG,
        "amplitude_arcsec": THERMAL_AMPLITUDE_ARCSEC,
        "baseline_step_arcsec_per_3day": THERMAL_BASELINE_STEP_ARCSEC_PER_3DAY,
        "days_per_block": THERMAL_DAYS_PER_BLOCK,
        "cycles_per_block": THERMAL_CYCLES_PER_BLOCK,
        "rms_x_pix": float(np.std(offsets[:, 0])),
        "rms_y_pix": float(np.std(offsets[:, 1])),
        "max_abs_x_pix": float(np.max(np.abs(offsets[:, 0]))),
        "max_abs_y_pix": float(np.max(np.abs(offsets[:, 1]))),
    }


def generate_momentum_dump_offsets(
    *,
    time_s: np.ndarray,
    seed: int,
    enabled: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not enabled:
        return np.zeros((len(time_s), 2), dtype=np.float32), {"enabled": False}
    period_min = MOMENTUM_DUMP_CYCLE_DAY * 24.0 * 60.0
    circle_radius_pix = MOMENTUM_DUMP_R68_ARCSEC / PIXEL_SCALE.to(u.arcsec / u.pix).value
    r_step_avg_pix = circle_radius_pix
    ztime_min = (np.asarray(time_s, dtype=np.float64) - float(time_s[0])) / 60.0
    total_min = float(ztime_min[-1] - ztime_min[0]) if len(ztime_min) else 0.0
    n_walks = max(1, int(np.ceil(total_min / period_min)))
    rng = np.random.default_rng(int(seed))
    x0 = np.zeros(n_walks, dtype=np.float64)
    y0 = np.zeros(n_walks, dtype=np.float64)
    for index in range(1, n_walks):
        while True:
            theta = rng.uniform(0.0, 2.0 * np.pi)
            r_step = rng.normal(0.0, r_step_avg_pix)
            candidate_x = x0[index - 1] + r_step * np.cos(theta)
            candidate_y = y0[index - 1] + r_step * np.sin(theta)
            if np.hypot(candidate_x, candidate_y) < circle_radius_pix:
                x0[index] = candidate_x
                y0[index] = candidate_y
                break
    td = (np.arange(n_walks) * period_min) + period_min / 2.0
    if len(td) == 1:
        x = np.zeros_like(ztime_min)
        y = np.zeros_like(ztime_min)
    else:
        from scipy.interpolate import interp1d

        x = interp1d(td, x0, kind="nearest", bounds_error=False, fill_value="extrapolate")(
            ztime_min
        )
        y = interp1d(td, y0, kind="nearest", bounds_error=False, fill_value="extrapolate")(
            ztime_min
        )
    offsets = np.vstack([x, y]).T.astype(np.float32)
    return offsets, {
        "enabled": True,
        "model": MOMENTUM_DUMP_MODEL,
        "reference": "et_sim_10_etpsd3-2.py",
        "seed": int(seed),
        "period_day": MOMENTUM_DUMP_CYCLE_DAY,
        "r68_arcsec": MOMENTUM_DUMP_R68_ARCSEC,
        "circle_radius_pix": float(circle_radius_pix),
        "r_step_avg_pix": float(r_step_avg_pix),
        "stay_inside": True,
        "random_r_step": True,
        "step_func": True,
        "n_walks": int(n_walks),
        "rms_x_pix": float(np.std(offsets[:, 0])),
        "rms_y_pix": float(np.std(offsets[:, 1])),
        "max_abs_x_pix": float(np.max(np.abs(offsets[:, 0]))),
        "max_abs_y_pix": float(np.max(np.abs(offsets[:, 1]))),
    }


def generate_weed_psf_breathing(
    *,
    time_s: np.ndarray,
    enabled: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not enabled:
        return np.ones(len(time_s), dtype=np.float32), {"enabled": False}
    ztime_s = np.asarray(time_s, dtype=np.float64) - float(time_s[0])
    period_s = WEED_PSF_BREATHING_PERIOD_DAY * 24.0 * 3600.0
    cycle_time = (ztime_s % period_s) / period_s
    scale = 1.0 - WEED_PSF_BREATHING_AMPLITUDE + 2.0 * WEED_PSF_BREATHING_AMPLITUDE * cycle_time
    scale = scale.astype(np.float32)
    return scale, {
        "enabled": True,
        "model": "generate_weed_psf_breathing",
        "reference": "et_sim_10_etpsd3-2.py",
        "period_day": WEED_PSF_BREATHING_PERIOD_DAY,
        "amplitude": WEED_PSF_BREATHING_AMPLITUDE,
        "min_scale": float(np.min(scale)),
        "max_scale": float(np.max(scale)),
    }


def build_full_effect_timeseries(
    *,
    n_frames: int,
    seed: int,
    enable_psd_motion: bool,
    psd_motion_path: Path | None,
    enable_dva: bool,
    enable_thermal: bool,
    enable_momentum_dump: bool,
    enable_psf_breathing: bool,
    exposure_s: float = EXPOSURE.to(u.s).value,
    motion_split_hz: float | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    exposure_s = float(exposure_s)
    if exposure_s <= 0.0:
        raise ValueError(f"exposure_s must be positive, got {exposure_s}")
    split_hz = float(motion_split_hz) if motion_split_hz is not None else 1.0 / exposure_s
    time_s = np.arange(int(n_frames), dtype=np.float64) * exposure_s
    psd_offsets, psd_meta = frame_motion_offsets(
        n_frames=int(n_frames),
        seed=int(seed) + 991,
        enable_psd_motion=bool(enable_psd_motion),
        psd_motion_path=psd_motion_path,
        exposure_s=exposure_s,
        motion_split_hz=split_hz,
    )
    dva_offsets, dva_meta = generate_dva_drift_offsets(time_s=time_s, enabled=enable_dva)
    thermal_offsets, thermal_meta = generate_thermal_drift_offsets(
        time_s=time_s,
        enabled=enable_thermal,
    )
    md_offsets, md_meta = generate_momentum_dump_offsets(
        time_s=time_s,
        seed=int(seed) + 2999,
        enabled=enable_momentum_dump,
    )
    psf_scale, psf_meta = generate_weed_psf_breathing(
        time_s=time_s,
        enabled=enable_psf_breathing,
    )
    total_offsets = psd_offsets + dva_offsets + thermal_offsets + md_offsets
    arrays = {
        "time_s": time_s.astype(np.float64),
        "psd_drift_pix": psd_offsets.astype(np.float32),
        "dva_drift_pix": dva_offsets.astype(np.float32),
        "thermal_drift_pix": thermal_offsets.astype(np.float32),
        "momentum_dump_pix": md_offsets.astype(np.float32),
        "total_motion_pix": total_offsets.astype(np.float32),
        "psf_scale": psf_scale.astype(np.float32),
    }
    metadata = {
        "motion_split_hz": float(split_hz),
        "time_step_s": float(exposure_s),
        "components": {
            "psd_spacecraft_roll_drift": psd_meta,
            "dva_drift": dva_meta,
            "thermal_drift": thermal_meta,
            "momentum_dump_jumps": md_meta,
            "psf_breathing": psf_meta,
        },
        "total_motion": {
            "rms_x_pix": float(np.std(total_offsets[:, 0])),
            "rms_y_pix": float(np.std(total_offsets[:, 1])),
            "max_abs_x_pix": float(np.max(np.abs(total_offsets[:, 0]))),
            "max_abs_y_pix": float(np.max(np.abs(total_offsets[:, 1]))),
        },
    }
    return arrays, metadata


def scattered_light_for_frame(spec: MainRdRunSpec, frame_index: int):
    scattered_rate = float(spec.scattered_light_e_s_pix)
    start_frame = spec.scattered_light_step_start_frame
    if (
        start_frame is not None
        and int(frame_index) >= int(start_frame)
        and float(spec.scattered_light_step_e_pix_frame) != 0.0
    ):
        scattered_rate += (
            float(spec.scattered_light_step_e_pix_frame) / float(spec.exposure_s)
        )
    return scattered_rate * u.electron / u.s / u.pix


def jitter_integrated_psf_offsets(
    *,
    seed: int,
    enable_psd_motion: bool,
    enable_jitter_integrated_psf: bool,
    psd_motion_path: Path | None,
    n_models: int,
    n_frames_per_model: int,
    exposure_s: float = EXPOSURE.to(u.s).value,
    motion_split_hz: float | None = None,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    if not enable_jitter_integrated_psf:
        return None, {"enabled": False, "reason": "disabled"}
    if not enable_psd_motion:
        zeros = np.zeros((1, 2, 1), dtype=np.float32)
        return zeros, {
            "enabled": True,
            "reason": "PSD motion disabled; zero-jitter PSF",
            "exposure_s": float(exposure_s),
        }
    if psd_motion_path is None:
        raise ValueError("Jitter-integrated PSF requires --psd-motion-path")

    psd_motion_path = Path(psd_motion_path).expanduser()
    if not psd_motion_path.exists():
        raise FileNotFoundError(f"PSD motion file not found: {psd_motion_path}")

    psd = load_psd_motion(psd_motion_path)
    exposure_s = float(exposure_s)
    if exposure_s <= 0.0:
        raise ValueError(f"exposure_s must be positive, got {exposure_s}")
    split_hz = float(motion_split_hz) if motion_split_hz is not None else 1.0 / exposure_s
    time_s = np.linspace(0.0, exposure_s, int(n_frames_per_model), endpoint=False)
    pixel_scale_arcsec = PIXEL_SCALE.to(u.arcsec / u.pix).value
    rng = np.random.default_rng(int(seed))
    xy_jitter_pix = np.zeros((int(n_models), 2, int(n_frames_per_model)), dtype=np.float32)
    for model_index in range(int(n_models)):
        theta_x_arcsec = psd_axis_motion(
            psd,
            axis="x",
            time_s=time_s,
            rng=rng,
            frequency_min_hz=split_hz,
            frequency_max_hz=None,
        )
        theta_y_arcsec = psd_axis_motion(
            psd,
            axis="y",
            time_s=time_s,
            rng=rng,
            frequency_min_hz=split_hz,
            frequency_max_hz=None,
        )
        theta_z_arcsec = psd_axis_motion(
            psd,
            axis="z",
            time_s=time_s,
            rng=rng,
            frequency_min_hz=split_hz,
            frequency_max_hz=None,
        )
        offsets_pix = spacecraft_roll_drift_from_angles(
            theta_x_arcsec=theta_x_arcsec,
            theta_y_arcsec=theta_y_arcsec,
            theta_z_arcsec=theta_z_arcsec,
            field_angle_deg=REFERENCE_EFFECT_FIELD_ANGLE_DEG,
            x_axis_angle_deg=REFERENCE_EFFECT_X_AXIS_ANGLE_DEG,
        )
        xy_jitter_pix[model_index, :, :] = offsets_pix.T.astype(np.float32)

    metadata = {
        "enabled": True,
        "path": str(psd_motion_path),
        "seed": int(seed),
        "exposure_s": float(exposure_s),
        "split_hz": float(split_hz),
        "n_models": int(n_models),
        "n_frames_per_model": int(n_frames_per_model),
        "model": "high-frequency x/y/z PSD attitude motion projected to focal-plane x/y and integrated into PSF",
        "field_angle_deg": float(REFERENCE_EFFECT_FIELD_ANGLE_DEG),
        "x_axis_angle_deg": float(REFERENCE_EFFECT_X_AXIS_ANGLE_DEG),
        "rms_x_pix": float(np.std(xy_jitter_pix[:, 0, :])),
        "rms_y_pix": float(np.std(xy_jitter_pix[:, 1, :])),
        "max_abs_x_pix": float(np.max(np.abs(xy_jitter_pix[:, 0, :]))),
        "max_abs_y_pix": float(np.max(np.abs(xy_jitter_pix[:, 1, :]))),
    }
    return xy_jitter_pix, metadata


def sample_column_noise_adu(*, frame_cols: int, sigma_adu: float, dtype, device):
    torch = require_torch()
    sigma_adu = float(sigma_adu)
    if sigma_adu < 0.0:
        raise ValueError(f"column_noise_sigma_adu must be non-negative, got {sigma_adu}")
    if sigma_adu == 0.0:
        return torch.zeros((int(frame_cols),), dtype=dtype, device=device)
    return torch.normal(
        mean=0.0,
        std=sigma_adu,
        size=(int(frame_cols),),
        dtype=dtype,
        device=device,
    )


def load_cosmic_ray_event_library(spec: MainRdRunSpec):
    ensure_local_imports()
    from photsim7.cosmic_rays import CosmicRayEventLibrary

    return CosmicRayEventLibrary.load(
        spec.cosmic_ray_library_path,
        expected_pixel_size_um=float(spec.cosmic_ray_pixel_size_um),
    )


def apply_detector_chain(
    *,
    image_electrons,
    frame_index: int,
    frame_rows: int,
    frame_cols: int,
    seed: int,
    spec: MainRdRunSpec,
    cosmic_ray_library=None,
):
    torch = require_torch()
    ensure_local_imports()
    from photsim7.cosmic_rays import (
        CosmicRayInjector,
        apply_adc_digitization,
        clip_full_well_electrons,
        electrons_to_adu,
        mean_events_from_rate,
    )

    frame_seed = int(seed) + int(frame_index) * 1000003
    torch.manual_seed(frame_seed)
    if torch.cuda.is_available() and image_electrons.is_cuda:
        torch.cuda.manual_seed_all(frame_seed)

    clipped_e = clip_full_well_electrons(
        image_electrons,
        full_well_electrons=float(spec.full_well_electrons),
    )
    readout_sigma = float(spec.readout_noise_e_pix)
    if readout_sigma > 0.0:
        clipped_e = clipped_e + torch.normal(
            mean=0.0,
            std=float(readout_sigma),
            size=clipped_e.shape,
            dtype=clipped_e.dtype,
            device=clipped_e.device,
        )

    image_adu = electrons_to_adu(
        clipped_e,
        gain_electrons_per_adu=float(spec.gain_electrons_per_adu),
    )

    library = (
        load_cosmic_ray_event_library(spec)
        if cosmic_ray_library is None
        else cosmic_ray_library
    )
    mean_events = mean_events_from_rate(
        rate_events_per_cm2_s=float(spec.cosmic_ray_event_rate_cm2_s) / (u.cm**2 * u.s),
        n_rows=int(frame_rows),
        n_cols=int(frame_cols),
        pixel_size_um=float(spec.cosmic_ray_pixel_size_um) * u.um,
        exposure_s=float(spec.exposure_s) * u.s,
    )
    image_adu_stack, cosmic_payload = CosmicRayInjector(library).inject(
        image_adu.unsqueeze(0),
        mean_events_per_frame=mean_events,
        seed=frame_seed + 17,
        frame_start=int(frame_index),
        allow_partial=True,
    )
    image_adu = image_adu_stack[0]

    col_noise = sample_column_noise_adu(
        frame_cols=int(frame_cols),
        sigma_adu=float(spec.column_noise_sigma_adu),
        dtype=image_adu.dtype,
        device=image_adu.device,
    )
    image_adu = image_adu + float(spec.bias_level_adu) + col_noise[None, :]
    image_dn = apply_adc_digitization(
        image_adu,
        enabled=True,
        bit_depth=int(spec.adc_bit_depth),
        min_value=0.0,
        round_values=True,
    )
    return image_dn, cosmic_payload, col_noise, mean_events


def gpu_memory_snapshot() -> str:
    try:
        return subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def plot_preview(image_dn: np.ndarray, preview_path: Path, *, title: str) -> None:
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 8.0), dpi=150)
    vmin, vmax = np.percentile(image_dn, [1.0, 99.7])
    ax.imshow(image_dn, origin="lower", cmap="gray", vmin=vmin, vmax=max(vmax, vmin + 1.0))
    ax.set_title(title)
    ax.set_xlabel("x pix")
    ax.set_ylabel("y pix")
    fig.tight_layout()
    fig.savefig(preview_path)
    plt.close(fig)


def run_worker(args: argparse.Namespace, spec: MainRdRunSpec) -> None:
    ensure_local_imports()
    torch = require_torch()
    frame_indices = selected_frame_indices(args.frame_indices, args.frames)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if int(args.jitter_psf_models) <= 0:
        raise ValueError("--jitter-psf-models must be positive")
    if int(args.jitter_frames_per_model) <= 0:
        raise ValueError("--jitter-frames-per-model must be positive")

    output_root = Path(args.output_root).expanduser()
    run_dir = output_root / run_dir_name(spec, args.mag_limit)
    frames_dir = run_dir / "frames"
    events_dir = run_dir / "cosmic_events"
    bias_dir = run_dir / "bias"
    preview_dir = run_dir / "preview"
    summary_dir = run_dir / "frame_summaries"
    for path in (frames_dir, events_dir, bias_dir, preview_dir, summary_dir):
        path.mkdir(parents=True, exist_ok=True)

    cache_path = Path(args.star_cache).expanduser()
    star_data, cache_metadata = load_star_cache(cache_path)
    if args.max_stars is not None:
        star_data = select_brightest(star_data, args.max_stars)

    if args.device.startswith("cuda"):
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    effect_arrays, effect_metadata = build_full_effect_timeseries(
        n_frames=int(args.frames),
        seed=int(args.seed),
        enable_psd_motion=bool(args.enable_psd_motion),
        psd_motion_path=args.psd_motion_path,
        enable_dva=bool(args.enable_dva_drift),
        enable_thermal=bool(args.enable_thermal_drift),
        enable_momentum_dump=bool(args.enable_momentum_dump),
        enable_psf_breathing=bool(args.enable_psf_breathing),
        exposure_s=float(spec.exposure_s),
    )
    xy_jitter_pix, jitter_metadata = jitter_integrated_psf_offsets(
        seed=int(args.seed) + 1999,
        enable_psd_motion=bool(args.enable_psd_motion),
        enable_jitter_integrated_psf=bool(args.jitter_integrated_psf),
        psd_motion_path=args.psd_motion_path,
        n_models=int(args.jitter_psf_models),
        n_frames_per_model=int(args.jitter_frames_per_model),
        exposure_s=float(spec.exposure_s),
    )

    psf_manager, psf_field_ids, psf_field_id_counts = build_psf_manager(
        frame_rows=spec.frame_rows,
        frame_cols=spec.frame_cols,
        device=args.device,
        star_data=star_data,
        n_subpixels=int(spec.n_subpixels),
        psf_bundle_name=spec.psf_bundle_name,
        pixel_scale_arcsec_per_pix=float(spec.pixel_scale_arcsec_per_pix),
        integrate_jitter=bool(args.jitter_integrated_psf),
        xy_jitter_pix=xy_jitter_pix,
        n_jitter_integrated_psf_models=int(args.jitter_psf_models),
        n_jitter_frames_per_model=int(args.jitter_frames_per_model),
    )
    stars = build_star_catalog(
        star_data=star_data,
        frame_rows=spec.frame_rows,
        frame_cols=spec.frame_cols,
        psf_field_ids=psf_field_ids,
        frame_exposure_s=float(spec.exposure_s),
    )
    detector_response_sampler = None
    if not args.no_detector_response:
        detector_response_sampler = build_detector_response_sampler(
            frame_rows=spec.frame_rows,
            frame_cols=spec.frame_cols,
            n_subpixels=int(spec.n_subpixels),
            device=args.device,
            seed=int(args.seed),
        )
    run_sim_config = sim_config_dict(spec.frame_rows, spec.frame_cols, spec=spec)
    renderer = make_renderer(
        sim_config=run_sim_config,
        frame_exposure_s=float(spec.exposure_s),
        device=args.device,
        stars=stars,
        psf_model_manager=psf_manager,
        detector_response_sampler=detector_response_sampler,
    )
    background_flux_per_pixel = run_sim_config["Background Flux"]
    dark_current_per_pixel = run_sim_config["Dark Current"]
    cosmic_ray_library = load_cosmic_ray_event_library(spec)

    base_x = np.asarray(stars["Detector Xpix Shifted"], dtype=np.float64)
    base_y = np.asarray(stars["Detector Ypix Shifted"], dtype=np.float64)
    if int(args.worker_rank) == 0:
        np.savez_compressed(run_dir / "effects_timeseries.npz", **effect_arrays)

    worker_summary = {
        "spec": asdict(spec),
        "args": vars(args),
        "rank": int(args.worker_rank),
        "world_size": int(args.worker_world_size),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device": args.device,
        "star_cache": str(cache_path),
        "star_cache_metadata": cache_metadata,
        "rendered_star_summary": star_summary(star_data),
        "psf_field_id_counts": psf_field_id_counts,
        "effects": effect_metadata,
        "jitter_integrated_psf": jitter_metadata,
        "gpu_memory_before": gpu_memory_snapshot(),
    }
    write_json(run_dir / f"worker_{args.worker_rank:02d}_start.json", worker_summary)

    assigned_frames = frame_indices[
        int(args.worker_rank) :: int(args.worker_world_size)
    ]
    print(
        f"[Worker {args.worker_rank}] frames={assigned_frames[:5]}"
        f"{'...' if len(assigned_frames) > 5 else ''} count={len(assigned_frames)}"
    )

    for frame_index in assigned_frames:
        frame_path = frames_dir / f"frame_{frame_index:06d}.npy"
        summary_path = summary_dir / f"frame_{frame_index:06d}.json"
        if frame_path.exists() and summary_path.exists() and not args.overwrite:
            print(f"[Worker {args.worker_rank}] skip existing frame {frame_index}")
            continue

        frame_start = time.perf_counter()
        frame_seed = int(args.seed) + frame_index * 1009
        torch.manual_seed(frame_seed)
        if args.device.startswith("cuda"):
            torch.cuda.manual_seed_all(frame_seed)
            torch.cuda.synchronize()

        offset_x, offset_y = effect_arrays["total_motion_pix"][frame_index]
        psf_scale = float(effect_arrays["psf_scale"][frame_index])
        jitter_model_index = (
            int(frame_index) % int(args.jitter_psf_models)
            if bool(args.jitter_integrated_psf)
            else None
        )
        scattered_light_per_pixel = scattered_light_for_frame(spec, frame_index)
        render_start = time.perf_counter()
        components = renderer.render_single_cadence(
            enable_stellar_photon_noise=True,
            enable_background_light=True,
            enable_scattered_light=bool(scattered_light_per_pixel.value != 0.0),
            enable_dark_current=True,
            enable_readout_noise=False,
            background_flux_per_pixel=background_flux_per_pixel,
            scattered_light_per_pixel=scattered_light_per_pixel,
            dark_current_per_pixel=dark_current_per_pixel,
            readout_noise=0.0 * u.electron / u.pix,
            subtract_nonstellar_mean=False,
            star_x_positions_pix=base_x + float(offset_x),
            star_y_positions_pix=base_y + float(offset_y),
            star_psf_scales=np.full(len(stars), psf_scale, dtype=np.float32),
            jitter_model_index=jitter_model_index,
            progress=bool(args.progress),
            return_numpy=False,
        )
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        render_elapsed = time.perf_counter() - render_start

        electronics_start = time.perf_counter()
        image_dn, cosmic_payload, col_noise, mean_events = apply_detector_chain(
            image_electrons=components["final_image"],
            frame_index=frame_index,
            frame_rows=spec.frame_rows,
            frame_cols=spec.frame_cols,
            seed=int(args.seed),
            spec=spec,
            cosmic_ray_library=cosmic_ray_library,
        )
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        electronics_elapsed = time.perf_counter() - electronics_start

        image_np = image_dn.detach().cpu().numpy()
        np.save(frame_path, image_np)
        np.save(events_dir / f"frame_{frame_index:06d}_events.npy", cosmic_payload.events)
        if args.save_column_noise:
            np.save(
                bias_dir / f"frame_{frame_index:06d}_column_noise_adu.npy",
                col_noise.detach().cpu().numpy(),
            )
        if args.save_cosmic_mask:
            np.save(
                events_dir / f"frame_{frame_index:06d}_mask.npy",
                cosmic_payload.mask[0],
            )
        if args.save_stellar_mean:
            np.save(
                frames_dir / f"frame_{frame_index:06d}_stellar_mean_e.npy",
                components["stellar_mean"].detach().cpu().numpy().astype(np.float32),
            )

        if frame_index < int(args.preview_count):
            plot_preview(
                image_np,
                preview_dir / f"frame_{frame_index:06d}.png",
                title=(
                    f"main_rd {spec.frame_cols}x{spec.frame_rows} G<{args.mag_limit:g} "
                    f"frame {frame_index}"
                ),
            )

        peak_allocated_mb = (
            torch.cuda.max_memory_allocated() / 1024**2 if args.device.startswith("cuda") else None
        )
        peak_reserved_mb = (
            torch.cuda.max_memory_reserved() / 1024**2 if args.device.startswith("cuda") else None
        )
        frame_summary = {
            "frame_index": int(frame_index),
            "rank": int(args.worker_rank),
            "device": args.device,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "n_stars": int(len(stars)),
            "render_elapsed_s": float(render_elapsed),
            "electronics_elapsed_s": float(electronics_elapsed),
            "total_elapsed_s": float(time.perf_counter() - frame_start),
            "motion_offset_x_pix": float(offset_x),
            "motion_offset_y_pix": float(offset_y),
            "psf_scale": float(psf_scale),
            "scattered_light_e_s_pix": float(scattered_light_per_pixel.value),
            "scattered_light_e_pix_frame": float(
                scattered_light_per_pixel.value * float(spec.exposure_s)
            ),
            "effect_components": {
                "psd_drift_pix": effect_arrays["psd_drift_pix"][frame_index],
                "dva_drift_pix": effect_arrays["dva_drift_pix"][frame_index],
                "thermal_drift_pix": effect_arrays["thermal_drift_pix"][frame_index],
                "momentum_dump_pix": effect_arrays["momentum_dump_pix"][frame_index],
            },
            "jitter_model_index": None if jitter_model_index is None else int(jitter_model_index),
            "mean_cosmic_events_per_frame": float(mean_events),
            "actual_cosmic_events": int(len(cosmic_payload.events)),
            "cosmic_mask_pixels": int(np.count_nonzero(cosmic_payload.mask)),
            "image_dtype": str(image_np.dtype),
            "image_min": int(np.min(image_np)),
            "image_p50": float(np.percentile(image_np, 50)),
            "image_p99": float(np.percentile(image_np, 99)),
            "image_p999": float(np.percentile(image_np, 99.9)),
            "image_max": int(np.max(image_np)),
            "saturated_pixels": int(
                np.count_nonzero(image_np >= (2 ** int(spec.adc_bit_depth) - 1))
            ),
            "peak_cuda_allocated_mb": peak_allocated_mb,
            "peak_cuda_reserved_mb": peak_reserved_mb,
            "frame_path": str(frame_path),
        }
        write_json(summary_path, frame_summary)
        print(
            f"[Worker {args.worker_rank}] frame={frame_index:06d} "
            f"render={render_elapsed:.2f}s electronics={electronics_elapsed:.2f}s "
            f"events={len(cosmic_payload.events)} sat={frame_summary['saturated_pixels']}"
        )

        del components, image_dn, image_np, cosmic_payload, col_noise
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    worker_summary["gpu_memory_after"] = gpu_memory_snapshot()
    write_json(run_dir / f"worker_{args.worker_rank:02d}_done.json", worker_summary)


def select_brightest(star_data: dict[str, Any], max_stars: int | None) -> dict[str, Any]:
    if max_stars is None:
        return star_data
    max_stars = int(max_stars)
    if max_stars <= 0:
        raise ValueError("max_stars must be positive when provided")
    n_stars = len(star_data["x0"])
    if n_stars <= max_stars:
        return star_data
    order = np.argsort(star_et_magnitude(star_data))[:max_stars]
    selected: dict[str, Any] = {}
    for key, value in star_data.items():
        arr = np.asarray(value)
        if arr.ndim == 1 and len(arr) == n_stars:
            selected[key] = arr[order]
    return selected


def expand_gpu_worker_assignments(gpu_ids: list[str], workers_per_gpu: int) -> list[str]:
    workers_per_gpu = int(workers_per_gpu)
    if workers_per_gpu <= 0:
        raise ValueError("--workers-per-gpu must be a positive integer")
    assignments: list[str] = []
    for gpu_id in gpu_ids:
        assignments.extend([str(gpu_id)] * workers_per_gpu)
    return assignments


def selected_frame_indices(frame_indices: str | None, frames: int) -> list[int]:
    frames = int(frames)
    if frames <= 0:
        raise ValueError("--frames must be a positive integer")
    if frame_indices is None or str(frame_indices).strip() == "":
        return list(range(frames))

    selected: list[int] = []
    seen: set[int] = set()
    for raw_token in str(frame_indices).split(","):
        token = raw_token.strip()
        if not token:
            continue
        try:
            frame_index = int(token)
        except ValueError as exc:
            raise ValueError(
                f"--frame-indices must be a comma-separated list of integers: {frame_indices}"
            ) from exc
        if frame_index < 0 or frame_index >= frames:
            raise ValueError(
                f"Frame index {frame_index} is outside the valid range 0..{frames - 1}"
            )
        if frame_index not in seen:
            selected.append(frame_index)
            seen.add(frame_index)

    if not selected:
        raise ValueError("--frame-indices did not contain any valid frame indices")
    return selected


def parse_common_args(
    description: str,
    *,
    default_frames: int = N_FRAMES,
    default_mag_limit: float = MAG_LIMIT,
    default_jitter_psf_models: int = JITTER_INTEGRATED_PSF_MODELS,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--frames", type=int, default=default_frames)
    parser.add_argument(
        "--frame-indices",
        type=str,
        default=None,
        help=(
            "Optional comma-separated frame indices to render, for example 0,180. "
            "When omitted, render all frames from 0 to --frames-1."
        ),
    )
    parser.add_argument("--mag-limit", type=float, default=default_mag_limit)
    parser.add_argument("--gpus", type=str, default="0,1")
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--output-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--catalog-dir", type=Path, default=GAIA_CATALOG_DIR)
    parser.add_argument("--seed", type=int, default=20260516)
    parser.add_argument("--crop-margin-pix", type=float, default=2.0)
    parser.add_argument("--preview-count", type=int, default=2)
    parser.add_argument("--max-stars", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--force-star-cache", action="store_true")
    parser.add_argument("--no-detector-response", action="store_true")
    parser.add_argument("--save-column-noise", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-cosmic-mask", action="store_true")
    parser.add_argument("--save-stellar-mean", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--prepare-star-cache-only",
        action="store_true",
        help="Build or validate the star cache and exit without launching workers.",
    )
    parser.add_argument(
        "--jitter-integrated-psf",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--jitter-psf-models", type=int, default=default_jitter_psf_models)
    parser.add_argument("--jitter-frames-per-model", type=int, default=JITTER_FRAMES_PER_MODEL)
    parser.add_argument(
        "--psd-motion",
        dest="enable_psd_motion",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--dva-drift",
        dest="enable_dva_drift",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--thermal-drift",
        dest="enable_thermal_drift",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--momentum-dump",
        dest="enable_momentum_dump",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--psf-breathing",
        dest="enable_psf_breathing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--enable-psd-motion",
        dest="enable_psd_motion",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--psd-motion-path",
        type=Path,
        default=Path("/home/cxgao/ET/photsim6_cache/ET_psd3-2.pkl"),
    )
    parser.add_argument("--worker-rank", type=int, default=None)
    parser.add_argument("--worker-world-size", type=int, default=None)
    parser.add_argument("--star-cache", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    return parser


def launch_or_run(args: argparse.Namespace, spec: MainRdRunSpec, script_path: Path) -> None:
    output_root = Path(args.output_root).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    frame_indices = selected_frame_indices(args.frame_indices, args.frames)
    if args.dry_run:
        cache_path = (
            Path(args.star_cache).expanduser()
            if args.star_cache is not None
            else star_cache_path(output_root, spec, args.mag_limit)
        )
        print(f"[Dry run] star_cache={cache_path}")
        print(
            f"[Dry run] frames={args.frames} gpus={args.gpus} "
            f"workers_per_gpu={args.workers_per_gpu} frame_indices={frame_indices}"
        )
        return

    if getattr(args, "prepare_star_cache_only", False):
        cache_path = resolve_or_prepare_star_cache(args, spec)
        print(f"[Star cache] ready {cache_path}")
        return

    cache_path = resolve_or_prepare_star_cache(args, spec)
    if args.worker_rank is not None:
        args.star_cache = cache_path
        run_worker(args, spec)
        return

    gpu_ids = [gpu.strip() for gpu in str(args.gpus).split(",") if gpu.strip()]
    if not gpu_ids:
        raise ValueError("--gpus must contain at least one GPU id, for example 0 or 0,1")
    worker_gpu_ids = expand_gpu_worker_assignments(gpu_ids, args.workers_per_gpu)

    run_dir = output_root / run_dir_name(spec, args.mag_limit)
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    parent_summary = {
        "spec": asdict(spec),
        "args": vars(args),
        "star_cache": str(cache_path),
        "gpu_ids": gpu_ids,
        "worker_gpu_ids": worker_gpu_ids,
        "workers_per_gpu": int(args.workers_per_gpu),
        "selected_frame_indices": frame_indices,
        "script_path": str(script_path),
    }
    write_json(run_dir / "run_config.json", parent_summary)

    processes: list[tuple[int, subprocess.Popen[Any], Any]] = []
    world_size = len(worker_gpu_ids)
    for rank, gpu_id in enumerate(worker_gpu_ids):
        cmd = [
            sys.executable,
            "-u",
            str(script_path),
            "--frames",
            str(args.frames),
            "--mag-limit",
            str(args.mag_limit),
            "--output-root",
            str(output_root),
            "--catalog-dir",
            str(args.catalog_dir),
            "--seed",
            str(args.seed),
            "--crop-margin-pix",
            str(args.crop_margin_pix),
            "--preview-count",
            str(args.preview_count),
            "--frame-indices",
            "" if args.frame_indices is None else str(args.frame_indices),
            "--worker-rank",
            str(rank),
            "--worker-world-size",
            str(world_size),
            "--star-cache",
            str(cache_path),
            "--device",
            args.device,
        ]
        if args.max_stars is not None:
            cmd += ["--max-stars", str(args.max_stars)]
        if args.overwrite:
            cmd.append("--overwrite")
        if args.no_detector_response:
            cmd.append("--no-detector-response")
        if not args.save_column_noise:
            cmd.append("--no-save-column-noise")
        if args.save_cosmic_mask:
            cmd.append("--save-cosmic-mask")
        if args.save_stellar_mean:
            cmd.append("--save-stellar-mean")
        if args.progress:
            cmd.append("--progress")
        if args.jitter_integrated_psf:
            cmd.append("--jitter-integrated-psf")
        else:
            cmd.append("--no-jitter-integrated-psf")
        cmd += ["--jitter-psf-models", str(args.jitter_psf_models)]
        cmd += ["--jitter-frames-per-model", str(args.jitter_frames_per_model)]
        if args.enable_psd_motion:
            cmd.append("--enable-psd-motion")
            cmd += ["--psd-motion-path", str(args.psd_motion_path)]
        else:
            cmd.append("--no-psd-motion")
        cmd.append("--dva-drift" if args.enable_dva_drift else "--no-dva-drift")
        cmd.append("--thermal-drift" if args.enable_thermal_drift else "--no-thermal-drift")
        cmd.append("--momentum-dump" if args.enable_momentum_dump else "--no-momentum-dump")
        cmd.append("--psf-breathing" if args.enable_psf_breathing else "--no-psf-breathing")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        env["PYTHONUNBUFFERED"] = "1"
        log_path = log_dir / f"worker_{rank:02d}_gpu_{gpu_id}.log"
        log_handle = log_path.open("w", encoding="utf-8")
        print(f"[Launcher] rank={rank} gpu={gpu_id} log={log_path}")
        processes.append(
            (
                rank,
                subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                ),
                log_handle,
            )
        )

    failures: list[tuple[int, int]] = []
    for rank, proc, log_handle in processes:
        return_code = proc.wait()
        log_handle.close()
        if return_code != 0:
            failures.append((rank, return_code))
    if failures:
        raise RuntimeError(f"Worker failures: {failures}. See logs in {log_dir}")
    print(f"[Launcher] complete run_dir={run_dir}")


def run_entrypoint(
    *,
    frame_rows: int,
    frame_cols: int,
    description: str,
    script_path: Path,
    spec_overrides: dict[str, Any] | None = None,
) -> None:
    overrides = {} if spec_overrides is None else dict(spec_overrides)
    spec = MainRdRunSpec(frame_rows=int(frame_rows), frame_cols=int(frame_cols), **overrides)
    parser = parse_common_args(
        description,
        default_frames=int(spec.n_frames),
        default_mag_limit=float(spec.mag_limit),
        default_jitter_psf_models=int(spec.n_jitter_integrated_psf_models),
    )
    args = parser.parse_args()
    launch_or_run(args, spec, script_path)
