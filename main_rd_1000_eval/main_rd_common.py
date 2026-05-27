from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from astropy import units as u


ET_ROOT = Path("/home/cxgao/ET")
PHOTSIM7_ROOT = ET_ROOT / "Photsim7"
PHOTSIM7_DATA_DIR = ET_ROOT / "Photsim7-data"
ET_FOCALPLANE_ROOT = ET_ROOT / "et_focalplane"
GAIA_CATALOG_DIR = Path("/home/cxgao/gaia_dr3_19mag")
RESULTS_ROOT = Path("/home/cxgao/Results/ET-mainsim/main_rd_1000_eval")

DETECTOR_ID = "main_rd"
FRAME_ROWS = 1000
FRAME_COLS = 1000
TARGET_RA_DEG = 304.41406499712303
TARGET_DEC_DEG = 51.81987707392268
TARGET_FIELD_X_DEG = -6.10175
TARGET_FIELD_Y_DEG = -6.23275
TARGET_FIELD_ANGLE_DEG = float(np.hypot(TARGET_FIELD_X_DEG, TARGET_FIELD_Y_DEG))
TARGET_DETECTOR_XPIX = 4450.0
TARGET_DETECTOR_YPIX = 4560.0

PIXEL_SCALE = 4.83 * u.arcsec / u.pix
PIXEL_WIDTH = 10.0 * u.um
EXPOSURE = 10.0 * u.s
OBSERVING_DURATION = 1800.0 * u.s
N_FRAMES = 180

SKY_SURFACE_BRIGHTNESS = 21.0
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
N_SUBPIXELS = 3
INTER_PIXEL_RESPONSE_SIGMA = 0.01
INTRA_PIXEL_RESPONSE_SIGMA = 0.01
INTER_PIXEL_RESPONSE_NOMINAL = 1.0


@dataclass(frozen=True)
class ExperimentSpec:
    detector_id: str = DETECTOR_ID
    frame_rows: int = FRAME_ROWS
    frame_cols: int = FRAME_COLS
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
    inter_pixel_response_sigma: float = INTER_PIXEL_RESPONSE_SIGMA
    intra_pixel_response_sigma: float = INTRA_PIXEL_RESPONSE_SIGMA
    notes: str = (
        "main_rd 1000x1000 crop centered on the ET focal-plane main_rd center; "
        "scattered light is disabled for this run by user decision."
    )


def ensure_local_imports() -> None:
    os.environ.setdefault("ET_DATA_DIR", str(PHOTSIM7_DATA_DIR))
    for path in (PHOTSIM7_ROOT, ET_FOCALPLANE_ROOT / "src"):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def ensure_results_root(path: Path | None = None) -> Path:
    root = RESULTS_ROOT if path is None else Path(path).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    return root


def experiment_spec_dict() -> dict[str, Any]:
    return asdict(ExperimentSpec())


def sim_config_dict() -> dict[str, Any]:
    from photsim7.background import sky_surface_brightness_to_background_flux

    background_flux = sky_surface_brightness_to_background_flux(
        SKY_SURFACE_BRIGHTNESS,
        PIXEL_SCALE,
        magnitude_system="ET",
    )
    return {
        "Detector Width": FRAME_COLS * u.pix,
        "Detector Height": FRAME_ROWS * u.pix,
        "Subpixels Per Pixel Dim": N_SUBPIXELS,
        "Pixel Scale": PIXEL_SCALE,
        "Pixel Width": PIXEL_WIDTH,
        "Exposure Duration": EXPOSURE,
        "Observing Duration": OBSERVING_DURATION,
        "Simulation Cadence Mult": 1,
        "Background Flux": background_flux,
        "Sky Background Mode": "surface_brightness",
        "Sky Background Surface Brightness": SKY_SURFACE_BRIGHTNESS,
        "Sky Background Magnitude System": "ET",
        "Subtract Nonstellar Mean": False,
        "Dark Current": DARK_CURRENT,
        "Scattered Light": SCATTERED_LIGHT,
        "Readout Noise": READOUT_NOISE,
        "Enable ADC Digitization": True,
        "Full Well Electrons": FULL_WELL_ELECTRONS * u.electron,
        "Gain Electrons Per ADU": GAIN_ELECTRONS_PER_ADU * u.electron / u.adu,
        "ADC Bit Depth": ADC_BIT_DEPTH,
        "ADC Min Value": 0.0,
        "ADC Round Values": True,
        "Bias Level ADU": BIAS_LEVEL_ADU * u.adu,
        "Column Noise Sigma ADU": COLUMN_NOISE_SIGMA_ADU * u.adu,
        "Save Bias Metadata": True,
        "Enable Cosmic Rays": True,
        "Cosmic Ray Event Library Path": COSMIC_RAY_LIBRARY_PATH,
        "Cosmic Ray Event Library Pixel Size": COSMIC_RAY_PIXEL_SIZE,
        "Cosmic Ray Event Rate": COSMIC_RAY_EVENT_RATE,
        "Cosmic Ray Seed": 0,
        "Inter-PRV (RMS)": 1.0 * u.percent,
        "Inter-PRV (Nominal)": 100.0 * u.percent,
        "Intra-PRV (RMS)": 1.0 * u.percent,
        "Pixel Response Profile Mod": "flux conserved",
        "Enable Flat Field Correction": False,
        "Flat Field Uncertainty": 0.0 * u.percent,
        "Enable DVA Drifts": True,
        "Thermal Defocus Model": "sinusoid",
        "Thermal Defocus Amplitude": 0.0 * u.percent,
        "Thermal Defocus Offset": 100.0 * u.percent,
        "PSF Bundle Name": PSF_BUNDLE_NAME,
        "PSF Field ID": "nearest",
        "Use Jitter-Integrated PSF": False,
        "Telescope Count": 1,
        "Optical Efficiency Ratio": 101.0 * u.percent,
    }


def query_main_rd_stars(
    *,
    mag_limit: float,
    catalog_dir: Path | str = GAIA_CATALOG_DIR,
    crop_margin_pix: float = 2.0,
) -> dict[str, Any]:
    ensure_local_imports()
    from photsim7.field import mk_real_field_stars_et_focalplane

    return mk_real_field_stars_et_focalplane(
        target_ra=TARGET_RA_DEG * u.deg,
        target_dec=TARGET_DEC_DEG * u.deg,
        catalog_dir=Path(catalog_dir).expanduser(),
        registry_data_dir=ET_FOCALPLANE_ROOT / "data",
        px_rows=FRAME_ROWS,
        px_cols=FRAME_COLS,
        apply_offset=False,
        mag_lim=float(mag_limit),
        detector_id=DETECTOR_ID,
        crop_to_simulation_frame=True,
        crop_margin_pix=float(crop_margin_pix),
        et_focalplane_src=ET_FOCALPLANE_ROOT / "src",
    )


def build_stars_table(star_data: dict[str, Any], *, psf_field_id: int):
    ensure_local_imports()
    from photsim7.field import Stars

    stars = Stars()
    stars.build_catalog(
        star_data,
        frame_exposure=EXPOSURE,
        optical_eff_ratio=1.0,
        aperture_area_ratio=1.0,
        mag_type="ET",
    )
    # The et_focalplane query stores absolute detector pixels in Detector X/Ypix
    # and Detector X/Ypix Shifted. The 1000x1000 smoke renderer expects image
    # coordinates for Shifted columns, and it uses them before x0/y0 when present.
    stars.catalog["Detector Xpix Shifted"] = (
        np.asarray(star_data["x0"], dtype=float) + (FRAME_COLS - 1) / 2.0
    )
    stars.catalog["Detector Ypix Shifted"] = (
        np.asarray(star_data["y0"], dtype=float) + (FRAME_ROWS - 1) / 2.0
    )
    stars.catalog["Field ID"] = np.full(len(stars.catalog), int(psf_field_id))
    return stars.catalog


def select_brightest(star_data: dict[str, Any], max_stars: int | None) -> dict[str, Any]:
    if max_stars is None:
        return star_data
    max_stars = int(max_stars)
    if max_stars <= 0:
        raise ValueError("max_stars must be positive when provided")
    n_stars = len(star_data["x0"])
    if n_stars <= max_stars:
        return star_data
    order = np.argsort(np.asarray(star_data["kp_mag"], dtype=float))[:max_stars]
    selected: dict[str, Any] = {}
    for key, value in star_data.items():
        arr = np.asarray(value)
        if arr.ndim == 1 and len(arr) == n_stars:
            selected[key] = arr[order]
        else:
            selected[key] = value
    selected["selection_note"] = f"brightest {max_stars} of {n_stars} stars"
    selected["unselected_star_count"] = int(n_stars - max_stars)
    return selected


def star_summary(star_data: dict[str, Any]) -> dict[str, Any]:
    n_stars = int(len(star_data["x0"]))
    summary: dict[str, Any] = {"n_stars": n_stars}
    if n_stars == 0:
        return summary
    kp_mag = np.asarray(star_data["kp_mag"], dtype=float)
    gaia_g = np.asarray(star_data.get("gaia_g_mag", kp_mag), dtype=float)
    x0 = np.asarray(star_data["x0"], dtype=float)
    y0 = np.asarray(star_data["y0"], dtype=float)
    summary.update(
        {
            "kp_mag_min": float(np.min(kp_mag)),
            "kp_mag_p50": float(np.percentile(kp_mag, 50)),
            "kp_mag_p90": float(np.percentile(kp_mag, 90)),
            "kp_mag_max": float(np.max(kp_mag)),
            "gaia_g_mag_min": float(np.min(gaia_g)),
            "gaia_g_mag_max": float(np.max(gaia_g)),
            "x0_min": float(np.min(x0)),
            "x0_max": float(np.max(x0)),
            "y0_min": float(np.min(y0)),
            "y0_max": float(np.max(y0)),
            "field_angle_deg_median": float(
                np.median(np.asarray(star_data["field_angle_deg"], dtype=float))
            ),
            "field_angle_deg_min": float(
                np.min(np.asarray(star_data["field_angle_deg"], dtype=float))
            ),
            "field_angle_deg_max": float(
                np.max(np.asarray(star_data["field_angle_deg"], dtype=float))
            ),
        }
    )
    return summary


def write_json(path: Path | str, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=_json_default)


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


def stars_to_dataframe(star_data: dict[str, Any]):
    import pandas as pd

    n_stars = len(star_data["x0"])
    columns = {}
    for key, value in star_data.items():
        arr = np.asarray(value)
        if arr.ndim == 1 and len(arr) == n_stars:
            columns[key] = arr
    return pd.DataFrame(columns)
