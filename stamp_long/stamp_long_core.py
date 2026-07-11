from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import multiprocessing as mp
import os
import pickle
import platform
import subprocess
import sys
import time
import types
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


SECONDS_PER_DAY = 24.0 * 3600.0
SECONDS_PER_YEAR = 365.25 * SECONDS_PER_DAY
DEFAULT_OUTPUT_ROOT = Path(
    os.environ.get(
        "RESULTS_ROOT",
        "/home/cxgao/Results/ET-mainsim/stamp_long",
    )
)
DEFAULT_JITTER_SENSITIVITY_OUTPUT_ROOT = Path(
    os.environ.get(
        "JITTER_SENSITIVITY_OUTPUT_ROOT",
        str(DEFAULT_OUTPUT_ROOT.parent / "stamp_long_jitter_sensitivity"),
    )
)
DEFAULT_BASE_EXPOSURE_S = 10.0
DEFAULT_ET_DATA_DIR = Path(
    os.environ.get(
        "ET_DATA_DIR",
        os.environ.get("PHOTSIM7_DATA_DIR", "/home/cxgao/ET/Photsim7-data"),
    )
)


def _default_photsim7_root(et_root: Path) -> Path:
    for dirname in ("Photsim7", "Photosim7"):
        candidate = et_root / dirname
        if candidate.exists():
            return candidate
    return et_root / "Photsim7"


DEFAULT_ET_ROOT = Path(os.environ.get("ET_ROOT", "/home/cxgao/ET")).expanduser()
DEFAULT_PHOTSIM7_ROOT = Path(
    os.environ.get("PHOTSIM7_ROOT", str(_default_photsim7_root(DEFAULT_ET_ROOT)))
).expanduser()
DEFAULT_READ_NOISE_E_PIX = 5.0
DEFAULT_GAIN_E_PER_ADU = 1.4
DEFAULT_PIXEL_SIZE_UM = 10.0
DEFAULT_STAR_FLUX_E_S = 100.0
DEFAULT_STAR_FLUX_MODE = "random_et_mag"
DEFAULT_ET_MAG_MIN = 12.5
DEFAULT_ET_MAG_MAX = 14.5
DEFAULT_BACKGROUND_E_S_PIX = 26.0
DEFAULT_SCATTERED_LIGHT_E_S_PIX = 5.0
DEFAULT_DARK_E_S_PIX = 1.0
DEFAULT_PSF_SIGMA_PIX = 1.25
DEFAULT_PSF_BUNDLE_NAME = "psf/et/241006/D280mm-focus"
DEFAULT_PSF_FIELD_ID = 6
DEFAULT_PSF_SUBPIXELS = 7
DEFAULT_PSD_MOTION_PATH = "pds/ET_psd3-2.pkl"
DEFAULT_DVA_MODEL_PATH = "DVA/et/ET_DVA_effect_models_slim_v231117.pkl"
DEFAULT_JITTER_INTEGRATED_PSF_MODELS = 300
DEFAULT_JITTER_FRAMES_PER_MODEL = 600
DEFAULT_INTER_PIXEL_RESPONSE_SIGMA = 0.01
DEFAULT_INTER_PIXEL_RESPONSE_NOMINAL = 1.0
DEFAULT_INTRA_PIXEL_RESPONSE_SIGMA = 0.01
DEFAULT_RESPONSE_PADDING_PIX = 0
DEFAULT_COSMIC_RAY_EVENT_RATE_CM2_S = 5.0
DEFAULT_COSMIC_RAY_LIBRARY_PATH = "cosmic_ray/dark_test_10um/event_library_10um.npz"
DEFAULT_COSMIC_RAY_PEAK_ADU = 4000.0
PIXEL_SCALE_ARCSEC_PER_PIX = 4.83
REFERENCE_EFFECT_FIELD_ANGLE_DEG = 10.0
REFERENCE_EFFECT_X_AXIS_ANGLE_DEG = 45.0
DVA_FIELD_ANGLE_DEG = 12.0
DVA_THETA_DEG = 12.0
THERMAL_THETA_DEG = 12.0
THERMAL_AMPLITUDE_ARCSEC = 0.022
THERMAL_BASELINE_STEP_ARCSEC_PER_3DAY = 0.03
THERMAL_DAYS_PER_BLOCK = 3.0
THERMAL_CYCLES_PER_BLOCK = 4.0
MOMENTUM_DUMP_CYCLE_DAY = 3.0
MOMENTUM_DUMP_R68_ARCSEC = 0.15
PSF_BREATHING_PERIOD_DAY = 3.0
PSF_BREATHING_AMPLITUDE = 0.01


def ensure_photsim7_imports() -> None:
    os.environ.setdefault("ET_DATA_DIR", str(DEFAULT_ET_DATA_DIR))
    photsim7_src = DEFAULT_PHOTSIM7_ROOT / "photsim7"
    if not photsim7_src.exists():
        if importlib.util.find_spec("photsim7") is None:
            raise FileNotFoundError(
                "Photsim7 source not found and no installed photsim7 package is "
                f"importable: {photsim7_src}"
            )
    else:
        existing = sys.modules.get("photsim7")
        if existing is None or not hasattr(existing, "__path__"):
            pkg = types.ModuleType("photsim7")
            pkg.__path__ = [str(photsim7_src)]
            pkg.__package__ = "photsim7"
            sys.modules["photsim7"] = pkg
        root_str = str(DEFAULT_PHOTSIM7_ROOT)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)


@dataclass(frozen=True)
class ExposureParameters:
    exposure_s: float
    n_coadd_equiv: float
    frames_per_day: int
    frames_per_year: int
    read_noise_e_pix: float


@dataclass(frozen=True)
class IndexRange:
    start: int
    stop: int


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    stage: str
    n_stars: int
    exposure_s: float
    n_frames: int
    stamp_size: int
    write_mode: str
    gpus: int
    description: str


@dataclass(frozen=True)
class StampRenderConfig:
    exposure_s: float
    stamp_size: int
    star_flux_e_s: float = DEFAULT_STAR_FLUX_E_S
    star_flux_mode: str = DEFAULT_STAR_FLUX_MODE
    et_mag: float | None = None
    background_e_s_pix: float = DEFAULT_BACKGROUND_E_S_PIX
    scattered_light_e_s_pix: float = DEFAULT_SCATTERED_LIGHT_E_S_PIX
    dark_e_s_pix: float = DEFAULT_DARK_E_S_PIX
    read_noise_10s_e_pix: float = DEFAULT_READ_NOISE_E_PIX
    gain_e_per_adu: float = DEFAULT_GAIN_E_PER_ADU
    cosmic_ray_event_rate: float = DEFAULT_COSMIC_RAY_EVENT_RATE_CM2_S
    cosmic_ray_library_path: str | None = DEFAULT_COSMIC_RAY_LIBRARY_PATH
    cosmic_ray_peak_adu: float = DEFAULT_COSMIC_RAY_PEAK_ADU
    pixel_size_um: float = DEFAULT_PIXEL_SIZE_UM
    psf_sigma_pix: float = DEFAULT_PSF_SIGMA_PIX
    psf_bundle_name: str = DEFAULT_PSF_BUNDLE_NAME
    psf_field_id: int = DEFAULT_PSF_FIELD_ID
    psf_subpixels: int = DEFAULT_PSF_SUBPIXELS
    use_photsim7_psf: bool = True
    psd_motion_path: str | None = DEFAULT_PSD_MOTION_PATH
    dva_model_path: str | None = DEFAULT_DVA_MODEL_PATH
    jitter_integrated_psf_models: int = DEFAULT_JITTER_INTEGRATED_PSF_MODELS
    jitter_frames_per_model: int = DEFAULT_JITTER_FRAMES_PER_MODEL
    enable_detector_response: bool = True
    response_padding_pix: int = DEFAULT_RESPONSE_PADDING_PIX
    inter_pixel_response_sigma: float = DEFAULT_INTER_PIXEL_RESPONSE_SIGMA
    inter_pixel_response_nominal: float = DEFAULT_INTER_PIXEL_RESPONSE_NOMINAL
    intra_pixel_response_sigma: float = DEFAULT_INTRA_PIXEL_RESPONSE_SIGMA
    enable_inter_pixel_response: bool = True
    enable_intra_pixel_response: bool = True
    enable_pixel_phase_response: bool = True
    enable_dynamic_effects: bool = True
    enable_psd_motion: bool = True
    enable_dva_drift: bool = True
    enable_thermal_drift: bool = True
    enable_momentum_dump: bool = True
    enable_psf_breathing: bool = True
    seed: int = 0
    global_seed: int = 20260617
    star_id: int = 0
    frame_id: int = 0
    n_frames: int = 1
    device: str = "auto"


@dataclass(frozen=True)
class RenderOptions:
    star_flux_e_s: float = DEFAULT_STAR_FLUX_E_S
    star_flux_mode: str = DEFAULT_STAR_FLUX_MODE
    et_mag_min: float = DEFAULT_ET_MAG_MIN
    et_mag_max: float = DEFAULT_ET_MAG_MAX
    background_e_s_pix: float = DEFAULT_BACKGROUND_E_S_PIX
    scattered_light_e_s_pix: float = DEFAULT_SCATTERED_LIGHT_E_S_PIX
    dark_e_s_pix: float = DEFAULT_DARK_E_S_PIX
    read_noise_10s_e_pix: float = DEFAULT_READ_NOISE_E_PIX
    gain_e_per_adu: float = DEFAULT_GAIN_E_PER_ADU
    cosmic_ray_event_rate: float = DEFAULT_COSMIC_RAY_EVENT_RATE_CM2_S
    cosmic_ray_library_path: str | None = DEFAULT_COSMIC_RAY_LIBRARY_PATH
    cosmic_ray_peak_adu: float = DEFAULT_COSMIC_RAY_PEAK_ADU
    pixel_size_um: float = DEFAULT_PIXEL_SIZE_UM
    psf_sigma_pix: float = DEFAULT_PSF_SIGMA_PIX
    psf_bundle_name: str = DEFAULT_PSF_BUNDLE_NAME
    psf_field_id: int = DEFAULT_PSF_FIELD_ID
    psf_subpixels: int = DEFAULT_PSF_SUBPIXELS
    use_photsim7_psf: bool = True
    psd_motion_path: str | None = DEFAULT_PSD_MOTION_PATH
    dva_model_path: str | None = DEFAULT_DVA_MODEL_PATH
    jitter_integrated_psf_models: int = DEFAULT_JITTER_INTEGRATED_PSF_MODELS
    jitter_frames_per_model: int = DEFAULT_JITTER_FRAMES_PER_MODEL
    enable_detector_response: bool = True
    response_padding_pix: int = DEFAULT_RESPONSE_PADDING_PIX
    inter_pixel_response_sigma: float = DEFAULT_INTER_PIXEL_RESPONSE_SIGMA
    inter_pixel_response_nominal: float = DEFAULT_INTER_PIXEL_RESPONSE_NOMINAL
    intra_pixel_response_sigma: float = DEFAULT_INTRA_PIXEL_RESPONSE_SIGMA
    enable_inter_pixel_response: bool = True
    enable_intra_pixel_response: bool = True
    enable_pixel_phase_response: bool = True
    enable_dynamic_effects: bool = True
    enable_psd_motion: bool = True
    enable_dva_drift: bool = True
    enable_thermal_drift: bool = True
    enable_momentum_dump: bool = True
    enable_psf_breathing: bool = True


@dataclass(frozen=True)
class JitterBankVariant:
    n_models: int
    n_frames_per_model: int

    @property
    def variant_id(self) -> str:
        return f"J{int(self.n_models):03d}F{int(self.n_frames_per_model):03d}"


@dataclass(frozen=True)
class JitterSensitivityCase:
    case_id: str
    exposure_s: float
    stamp_size: int
    description: str


@dataclass(frozen=True)
class StampRecord:
    case_id: str
    exposure_time_s: float
    n_coadd_equiv: float
    star_id: int
    frame_id: int
    stamp_size: int
    dtype: str
    unit: str
    seed: int
    file_path: str
    file_size_bytes: int
    status: str


@dataclass(frozen=True)
class WorkerTask:
    case: BenchmarkCase
    output_root: str
    star_range: IndexRange
    worker_rank: int
    world_size: int
    gpu_id: str | None
    global_seed: int
    write_mode: str
    output_format: str
    device_mode: str
    sample_limit: int
    render_options: RenderOptions


@dataclass(frozen=True)
class WorkerResult:
    worker_rank: int
    n_stamps: int
    n_written: int
    n_skipped: int
    n_failed: int
    output_bytes: int
    elapsed_s: float
    manifest_path: str
    n_records: int
    output_format: str = "npy"
    artifact_path: str = ""
    n_shards: int = 0


def _normalize_output_format(output_format: str) -> str:
    normalized = str(output_format).strip().lower()
    if normalized not in {"npy", "hdf5"}:
        raise ValueError(
            f"Unsupported output_format {output_format!r}; expected npy or hdf5"
        )
    return normalized


def exposure_parameters(
    exposure_s: float,
    *,
    base_exposure_s: float = DEFAULT_BASE_EXPOSURE_S,
    read_noise_10s_e_pix: float = DEFAULT_READ_NOISE_E_PIX,
) -> ExposureParameters:
    exposure_s = float(exposure_s)
    if exposure_s <= 0.0:
        raise ValueError(f"exposure_s must be positive, got {exposure_s}")
    n_coadd = exposure_s / float(base_exposure_s)
    return ExposureParameters(
        exposure_s=exposure_s,
        n_coadd_equiv=n_coadd,
        frames_per_day=int(round(SECONDS_PER_DAY / exposure_s)),
        frames_per_year=int(round(SECONDS_PER_YEAR / exposure_s)),
        read_noise_e_pix=float(read_noise_10s_e_pix) * math.sqrt(n_coadd),
    )


def derive_seed(
    global_seed: int,
    *,
    exposure_s: float,
    frame_id: int,
    star_id: int,
    effect_type: str,
) -> int:
    payload = "|".join(
        [
            str(int(global_seed)),
            f"{float(exposure_s):.6f}",
            str(int(frame_id)),
            str(int(star_id)),
            str(effect_type),
        ]
    ).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**32)


def et_mag_to_photon_rate_e_s(et_mag: float | np.ndarray) -> float | np.ndarray:
    ensure_photsim7_imports()
    from astropy import units as u
    from photsim7.photometry import et_mag_to_detected_electron_rate

    rates = np.asarray(
        et_mag_to_detected_electron_rate(et_mag).to_value(u.electron / u.s),
        dtype=np.float64,
    )
    if np.isscalar(et_mag):
        return float(rates)
    return rates


def sample_star_et_mag(
    *,
    global_seed: int,
    star_id: int,
    et_mag_min: float = DEFAULT_ET_MAG_MIN,
    et_mag_max: float = DEFAULT_ET_MAG_MAX,
) -> float:
    et_mag_min = float(et_mag_min)
    et_mag_max = float(et_mag_max)
    if not np.isfinite(et_mag_min) or not np.isfinite(et_mag_max):
        raise ValueError("ET magnitude bounds must be finite")
    if et_mag_min > et_mag_max:
        raise ValueError(
            f"et_mag_min must be <= et_mag_max, got {et_mag_min} > {et_mag_max}"
        )
    seed = derive_seed(
        global_seed,
        exposure_s=0.0,
        frame_id=0,
        star_id=int(star_id),
        effect_type="et_mag",
    )
    rng = np.random.default_rng(seed)
    return float(rng.uniform(et_mag_min, et_mag_max))


def _normalize_star_flux_mode(star_flux_mode: str) -> str:
    normalized = str(star_flux_mode).strip().lower().replace("-", "_")
    if normalized not in {"fixed", "random_et_mag"}:
        raise ValueError(
            f"Unsupported star_flux_mode {star_flux_mode!r}; expected fixed or random_et_mag"
        )
    return normalized


def parse_jitter_bank_variants(raw: str | None) -> list[JitterBankVariant]:
    if raw is None or str(raw).strip() == "":
        return [
            JitterBankVariant(100, 200),
            JitterBankVariant(100, 300),
            JitterBankVariant(200, 400),
            JitterBankVariant(300, 600),
        ]
    variants: list[JitterBankVariant] = []
    for token in str(raw).split(","):
        item = token.strip().lower().replace("*", "x")
        if not item:
            continue
        if "x" not in item:
            raise ValueError(
                f"Invalid jitter bank variant {token!r}; expected '<models>x<frames>'"
            )
        n_models_raw, n_frames_raw = item.split("x", 1)
        variant = JitterBankVariant(int(n_models_raw), int(n_frames_raw))
        if variant.n_models <= 0 or variant.n_frames_per_model <= 0:
            raise ValueError(f"Jitter bank variant values must be positive: {token!r}")
        variants.append(variant)
    if not variants:
        raise ValueError("At least one jitter bank variant is required")
    return variants


def jitter_frame_indices(source_frames: int, target_frames: int) -> list[int]:
    source_frames = int(source_frames)
    target_frames = int(target_frames)
    if source_frames <= 0 or target_frames <= 0:
        raise ValueError("source_frames and target_frames must be positive")
    if target_frames > source_frames:
        raise ValueError(
            f"target_frames must be <= source_frames, got {target_frames} > {source_frames}"
        )
    indices = np.floor(
        np.arange(target_frames, dtype=np.float64)
        * float(source_frames)
        / float(target_frames)
    ).astype(np.int64)
    return [int(index) for index in indices]


def derive_jitter_bank_variant(
    master_xy_jitter_pix: np.ndarray,
    variant: JitterBankVariant,
) -> tuple[np.ndarray, dict[str, Any]]:
    master = np.asarray(master_xy_jitter_pix, dtype=np.float32)
    if master.ndim != 3 or master.shape[1] != 2:
        raise ValueError(
            "master_xy_jitter_pix must have shape (n_models, 2, n_frames_per_model), "
            f"got {master.shape}"
        )
    if int(variant.n_models) > int(master.shape[0]):
        raise ValueError(
            f"variant n_models must be <= master n_models, got {variant.n_models} > {master.shape[0]}"
        )
    if int(variant.n_frames_per_model) > int(master.shape[2]):
        raise ValueError(
            "variant n_frames_per_model must be <= master n_frames_per_model, "
            f"got {variant.n_frames_per_model} > {master.shape[2]}"
        )
    model_indices = list(range(int(variant.n_models)))
    frame_indices = jitter_frame_indices(
        int(master.shape[2]),
        int(variant.n_frames_per_model),
    )
    derived = master[: int(variant.n_models), :, frame_indices].copy()
    metadata = {
        "variant_id": variant.variant_id,
        "source_shape": [int(value) for value in master.shape],
        "variant_shape": [int(value) for value in derived.shape],
        "model_indices": model_indices,
        "frame_indices": frame_indices,
        "model_coverage_fraction": float(variant.n_models) / float(master.shape[0]),
        "frame_coverage_fraction": float(variant.n_frames_per_model) / float(master.shape[2]),
    }
    return derived, metadata


def jitter_comparison_model_indices(n_models: int, *, max_samples: int = 3) -> list[int]:
    n_models = int(n_models)
    max_samples = int(max_samples)
    if n_models <= 0:
        raise ValueError(f"n_models must be positive, got {n_models}")
    if max_samples <= 0:
        raise ValueError(f"max_samples must be positive, got {max_samples}")
    if n_models <= max_samples:
        return list(range(n_models))
    raw = np.linspace(0, n_models - 1, max_samples)
    indices = sorted({int(round(value)) for value in raw})
    if indices[0] != 0:
        indices.insert(0, 0)
    if indices[-1] != n_models - 1:
        indices.append(n_models - 1)
    return indices[:max_samples]


def jitter_sensitivity_cases(raw: str | None) -> list[JitterSensitivityCase]:
    cases = [
        JitterSensitivityCase("J030S11", 30.0, 11, "30 s 11x11 short-exposure representative"),
        JitterSensitivityCase("J300S15", 300.0, 15, "300 s 15x15 long-exposure representative"),
    ]
    if raw is None or str(raw).strip() == "":
        return cases
    wanted = {token.strip() for token in str(raw).split(",") if token.strip()}
    filtered = [case for case in cases if case.case_id in wanted]
    missing = sorted(wanted - {case.case_id for case in filtered})
    if missing:
        raise ValueError(f"Unknown jitter sensitivity case ids: {missing}")
    return filtered


def _master_jitter_variant(variants: Sequence[JitterBankVariant]) -> JitterBankVariant:
    if not variants:
        raise ValueError("At least one jitter bank variant is required")
    return max(
        variants,
        key=lambda variant: (int(variant.n_models), int(variant.n_frames_per_model)),
    )


def _image_moments(image: np.ndarray) -> dict[str, float]:
    weights = np.asarray(image, dtype=np.float64)
    total = float(np.sum(weights))
    if not np.isfinite(total) or np.isclose(total, 0.0):
        return {
            "flux": total,
            "centroid_x": math.nan,
            "centroid_y": math.nan,
            "second_moment_radius_pix": math.nan,
        }
    yy, xx = np.indices(weights.shape, dtype=np.float64)
    centroid_x = float(np.sum(weights * xx) / total)
    centroid_y = float(np.sum(weights * yy) / total)
    radius2 = (xx - centroid_x) ** 2 + (yy - centroid_y) ** 2
    second_moment = float(np.sum(weights * radius2) / total)
    return {
        "flux": total,
        "centroid_x": centroid_x,
        "centroid_y": centroid_y,
        "second_moment_radius_pix": math.sqrt(max(second_moment, 0.0)),
    }


def stamp_difference_metrics(
    candidate: np.ndarray,
    reference: np.ndarray,
) -> dict[str, float]:
    candidate_arr = np.asarray(candidate, dtype=np.float64)
    reference_arr = np.asarray(reference, dtype=np.float64)
    if candidate_arr.shape != reference_arr.shape:
        raise ValueError(
            f"candidate and reference shapes must match, got {candidate_arr.shape} and {reference_arr.shape}"
        )
    diff = candidate_arr - reference_arr
    diff_l2 = float(np.linalg.norm(diff.ravel()))
    reference_l2 = float(np.linalg.norm(reference_arr.ravel()))
    candidate_moments = _image_moments(candidate_arr)
    reference_moments = _image_moments(reference_arr)
    centroid_shift = math.hypot(
        candidate_moments["centroid_x"] - reference_moments["centroid_x"],
        candidate_moments["centroid_y"] - reference_moments["centroid_y"],
    )
    reference_flux = reference_moments["flux"]
    candidate_flux = candidate_moments["flux"]
    flux_delta = candidate_flux - reference_flux
    return {
        "max_abs_diff": float(np.max(np.abs(diff))),
        "mean_abs_diff": float(np.mean(np.abs(diff))),
        "rms_diff": float(math.sqrt(np.mean(diff * diff))),
        "relative_l2": diff_l2 / reference_l2 if reference_l2 > 0.0 else math.nan,
        "reference_flux": float(reference_flux),
        "candidate_flux": float(candidate_flux),
        "flux_delta": float(flux_delta),
        "flux_delta_fraction": (
            flux_delta / reference_flux if not np.isclose(reference_flux, 0.0) else math.nan
        ),
        "reference_centroid_x": reference_moments["centroid_x"],
        "reference_centroid_y": reference_moments["centroid_y"],
        "candidate_centroid_x": candidate_moments["centroid_x"],
        "candidate_centroid_y": candidate_moments["centroid_y"],
        "centroid_shift_pix": float(centroid_shift),
        "reference_second_moment_radius_pix": reference_moments["second_moment_radius_pix"],
        "candidate_second_moment_radius_pix": candidate_moments["second_moment_radius_pix"],
        "second_moment_radius_delta_pix": (
            candidate_moments["second_moment_radius_pix"]
            - reference_moments["second_moment_radius_pix"]
        ),
    }


def split_ranges(total_items: int, n_parts: int) -> list[IndexRange]:
    total_items = int(total_items)
    n_parts = int(n_parts)
    if total_items < 0:
        raise ValueError(f"total_items must be non-negative, got {total_items}")
    if n_parts <= 0:
        raise ValueError(f"n_parts must be positive, got {n_parts}")
    base, remainder = divmod(total_items, n_parts)
    ranges: list[IndexRange] = []
    start = 0
    for part in range(n_parts):
        stop = start + base + (1 if part < remainder else 0)
        ranges.append(IndexRange(start, stop))
        start = stop
    return ranges


def benchmark_cases(stage: str) -> list[BenchmarkCase]:
    stage = str(stage).strip().lower()
    if stage == "smoke":
        return [
            BenchmarkCase("F01", stage, 10, 30.0, 10, 11, "all", 1, "30 s 11x11 smoke"),
            BenchmarkCase("F02", stage, 10, 30.0, 10, 15, "all", 1, "30 s 15x15 smoke"),
            BenchmarkCase("F03", stage, 10, 300.0, 10, 11, "all", 1, "300 s 11x11 smoke"),
            BenchmarkCase("F04", stage, 10, 300.0, 10, 15, "all", 1, "300 s 15x15 smoke"),
            BenchmarkCase("F05", stage, 100, 30.0, 100, 15, "all", 1, "30 s small-file smoke"),
            BenchmarkCase("F06", stage, 100, 300.0, 100, 15, "all", 1, "300 s noise and CR smoke"),
        ]
    if stage == "compute":
        return [
            BenchmarkCase("C01", stage, 1000, 30.0, 100, 11, "none", 1, "1 GPU 30 s 11x11 compute"),
            BenchmarkCase("C02", stage, 1000, 30.0, 100, 15, "none", 1, "1 GPU 30 s 15x15 compute"),
            BenchmarkCase("C03", stage, 1000, 300.0, 100, 11, "none", 1, "1 GPU 300 s 11x11 compute"),
            BenchmarkCase("C04", stage, 1000, 300.0, 100, 15, "none", 1, "1 GPU 300 s 15x15 compute"),
            BenchmarkCase("C05", stage, 1000, 30.0, 100, 15, "none", 3, "3 GPU 30 s compute"),
            BenchmarkCase("C06", stage, 1000, 300.0, 100, 15, "none", 3, "3 GPU 300 s compute"),
        ]
    if stage == "io":
        return [
            BenchmarkCase("I01", stage, 100, 30.0, 100, 15, "all", 0, "small file I/O smoke"),
            BenchmarkCase("I02", stage, 1000, 30.0, 100, 15, "all", 0, "100k-file I/O"),
            BenchmarkCase("I03", stage, 1000, 300.0, 100, 15, "all", 0, "300 s I/O layout"),
            BenchmarkCase("I04", stage, 1680, 30.0, 20, 15, "all", 0, "high-star short-frame I/O"),
        ]
    if stage == "physics":
        return _stamp_scale_v2_cases(stage)
    raise ValueError(f"Unknown benchmark stage {stage!r}")


def _stamp_scale_v2_cases(stage: str) -> list[BenchmarkCase]:
    groups = [
        ("S1D", 1680, 1.0, "short high-star throughput/I/O matrix"),
        ("L7D", 240, 7.0, "long low-star drift/CR/resume stability matrix"),
    ]
    cases: list[BenchmarkCase] = []
    for prefix, n_stars, duration_days, description in groups:
        for stamp_size in (11, 15):
            for exposure_s in (30.0, 60.0, 180.0, 300.0):
                n_frames = int(round(duration_days * SECONDS_PER_DAY / exposure_s))
                case_id = f"{prefix}{stamp_size:02d}E{int(round(exposure_s)):03d}"
                cases.append(
                    BenchmarkCase(
                        case_id,
                        stage,
                        n_stars,
                        exposure_s,
                        n_frames,
                        stamp_size,
                        "all",
                        3,
                        description,
                    )
                )
    return cases


def resolve_data_path(path: Path | str, *, et_data_dir: Path | str | None = None) -> Path:
    data_path = Path(path).expanduser()
    if data_path.is_absolute():
        return data_path
    root = DEFAULT_ET_DATA_DIR if et_data_dir is None else Path(et_data_dir).expanduser()
    return root / data_path


def _optional_resolved_path(path: str | None) -> Path | None:
    if path is None or str(path).strip() == "":
        return None
    return resolve_data_path(str(path))


def _psf_bundle_file(
    psf_bundle_name: str,
    *,
    et_data_dir: Path | str | None = None,
) -> Path:
    bundle = Path(str(psf_bundle_name)).expanduser()
    if bundle.name == "sim_psf_images.pkl":
        psf_path = bundle
    else:
        psf_path = bundle / "sim_psf_images.pkl"
    if psf_path.is_absolute():
        return psf_path
    return resolve_data_path(psf_path, et_data_dir=et_data_dir)


def _center_crop_or_pad(image: np.ndarray, output_size: int) -> np.ndarray:
    output_size = int(output_size)
    if output_size <= 0:
        raise ValueError(f"output_size must be positive, got {output_size}")
    source = np.asarray(image, dtype=np.float32)
    if source.ndim != 2:
        raise ValueError(f"Expected a 2-D image, got shape {source.shape}")
    out = np.zeros((output_size, output_size), dtype=np.float32)
    in_y, in_x = source.shape
    crop_y = min(in_y, output_size)
    crop_x = min(in_x, output_size)
    src_y0 = (in_y - crop_y) // 2
    src_x0 = (in_x - crop_x) // 2
    dst_y0 = (output_size - crop_y) // 2
    dst_x0 = (output_size - crop_x) // 2
    out[dst_y0 : dst_y0 + crop_y, dst_x0 : dst_x0 + crop_x] = source[
        src_y0 : src_y0 + crop_y,
        src_x0 : src_x0 + crop_x,
    ]
    return out


def _downsample_subpixel_psf(image: np.ndarray, subpixels: int) -> np.ndarray:
    subpixels = int(subpixels)
    if subpixels <= 0:
        raise ValueError(f"subpixels must be positive, got {subpixels}")
    plane = np.asarray(image, dtype=np.float32)
    if plane.ndim == 3:
        plane = plane[-1]
    if plane.ndim != 2:
        raise ValueError(f"Expected 2-D or 3-D PSF array, got shape {plane.shape}")
    trim_y = (plane.shape[0] // subpixels) * subpixels
    trim_x = (plane.shape[1] // subpixels) * subpixels
    if trim_y <= 0 or trim_x <= 0:
        raise ValueError(
            f"PSF plane shape {plane.shape} is too small for subpixel grid {subpixels}"
        )
    y0 = (plane.shape[0] - trim_y) // 2
    x0 = (plane.shape[1] - trim_x) // 2
    trimmed = plane[y0 : y0 + trim_y, x0 : x0 + trim_x]
    return trimmed.reshape(
        trim_y // subpixels,
        subpixels,
        trim_x // subpixels,
        subpixels,
    ).sum(axis=(1, 3))


@lru_cache(maxsize=16)
def _load_photsim7_psf_stamp_cached(
    stamp_size: int,
    psf_bundle_name: str,
    psf_field_id: int,
    psf_subpixels: int,
    et_data_dir_key: str,
) -> np.ndarray:
    et_data_dir = None if et_data_dir_key == "" else et_data_dir_key
    psf_path = _psf_bundle_file(psf_bundle_name, et_data_dir=et_data_dir)
    with psf_path.open("rb") as handle:
        payload = pickle.load(handle)
    images = payload["images"]
    try:
        psf_plane = images[int(psf_field_id)][int(psf_subpixels)]
    except KeyError as exc:
        raise KeyError(
            "Requested PSF field/subpixel grid is unavailable: "
            f"field_id={psf_field_id}, subpixels={psf_subpixels}, path={psf_path}"
        ) from exc
    pixel_psf = _downsample_subpixel_psf(psf_plane, int(psf_subpixels))
    stamp = _center_crop_or_pad(pixel_psf, int(stamp_size))
    total = float(np.sum(stamp, dtype=np.float64))
    if total <= 0.0 or not np.isfinite(total):
        raise ValueError(f"PSF normalization failed for {psf_path}")
    return (stamp / total).astype(np.float32, copy=False)


def load_photsim7_psf_stamp(
    *,
    stamp_size: int,
    psf_bundle_name: str = DEFAULT_PSF_BUNDLE_NAME,
    psf_field_id: int = DEFAULT_PSF_FIELD_ID,
    psf_subpixels: int = DEFAULT_PSF_SUBPIXELS,
    et_data_dir: Path | str | None = None,
) -> np.ndarray:
    et_data_dir_key = "" if et_data_dir is None else str(Path(et_data_dir).expanduser())
    return _load_photsim7_psf_stamp_cached(
        int(stamp_size),
        str(psf_bundle_name),
        int(psf_field_id),
        int(psf_subpixels),
        et_data_dir_key,
    ).copy()


def _gaussian_psf(stamp_size: int, sigma_pix: float) -> np.ndarray:
    center = (int(stamp_size) - 1) / 2.0
    axis = np.arange(int(stamp_size), dtype=np.float32) - np.float32(center)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    psf = np.exp(-0.5 * (xx * xx + yy * yy) / float(sigma_pix) ** 2)
    psf_sum = float(np.sum(psf, dtype=np.float64))
    if psf_sum <= 0.0:
        raise ValueError("PSF normalization failed")
    return (psf / psf_sum).astype(np.float32, copy=False)


def _normalize_psf(psf: np.ndarray) -> np.ndarray:
    out = np.asarray(psf, dtype=np.float32)
    total = float(np.sum(out, dtype=np.float64))
    if total <= 0.0 or not np.isfinite(total):
        raise ValueError("PSF normalization failed")
    return (out / total).astype(np.float32, copy=False)


def _bilinear_sample(image: np.ndarray, sample_x: np.ndarray, sample_y: np.ndarray) -> np.ndarray:
    source = np.asarray(image, dtype=np.float32)
    x0 = np.floor(sample_x).astype(np.int64)
    y0 = np.floor(sample_y).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1
    wx = sample_x - x0
    wy = sample_y - y0
    out = np.zeros_like(sample_x, dtype=np.float32)
    for yy, xx, weight in (
        (y0, x0, (1.0 - wy) * (1.0 - wx)),
        (y0, x1, (1.0 - wy) * wx),
        (y1, x0, wy * (1.0 - wx)),
        (y1, x1, wy * wx),
    ):
        mask = (yy >= 0) & (yy < source.shape[0]) & (xx >= 0) & (xx < source.shape[1])
        out[mask] += source[yy[mask], xx[mask]] * weight[mask].astype(np.float32)
    return out


def _shift_image_bilinear(image: np.ndarray, *, dx_pix: float, dy_pix: float) -> np.ndarray:
    yy, xx = np.indices(np.asarray(image).shape, dtype=np.float64)
    shifted = _bilinear_sample(
        np.asarray(image, dtype=np.float32),
        sample_x=xx - float(dx_pix),
        sample_y=yy - float(dy_pix),
    )
    return _normalize_psf(shifted)


def _rescale_psf(image: np.ndarray, *, scale: float) -> np.ndarray:
    scale = float(scale)
    if np.isclose(scale, 1.0, rtol=0.0, atol=1e-6):
        return _normalize_psf(image)
    source = np.asarray(image, dtype=np.float32)
    yy, xx = np.indices(source.shape, dtype=np.float64)
    cy = (source.shape[0] - 1.0) / 2.0
    cx = (source.shape[1] - 1.0) / 2.0
    rescaled = _bilinear_sample(
        source,
        sample_x=cx + (xx - cx) / scale,
        sample_y=cy + (yy - cy) / scale,
    )
    return _normalize_psf(rescaled)


def _frame_elapsed_day(config: StampRenderConfig) -> float:
    return float(config.frame_id) * float(config.exposure_s) / SECONDS_PER_DAY


def _project_radial_to_xy(radial_pix: float, theta_deg: float) -> tuple[float, float]:
    theta = math.radians(float(theta_deg))
    return float(radial_pix) * math.cos(theta), float(radial_pix) * math.sin(theta)


@lru_cache(maxsize=4)
def _load_dva_model_cached(path_key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = resolve_data_path(path_key)
    with path.open("rb") as handle:
        model = pickle.load(handle)
    angle_grid = np.asarray(model["cfov_angles_deg"], dtype=np.float64)
    time_day = np.asarray(model["time_day"], dtype=np.float64)
    time_day = time_day - float(time_day[0])
    radial_profiles_arcsec = np.asarray(model["r_arcsec"], dtype=np.float64)
    return angle_grid, time_day, radial_profiles_arcsec


def _interpolate_dva_profile_for_angle(
    angle_grid: np.ndarray,
    radial_profiles_arcsec: np.ndarray,
    field_angle_deg: float,
) -> np.ndarray:
    field_angle_deg = float(field_angle_deg)
    if field_angle_deg <= float(angle_grid[0]):
        return radial_profiles_arcsec[0]
    if field_angle_deg >= float(angle_grid[-1]):
        return radial_profiles_arcsec[-1]
    upper = int(np.searchsorted(angle_grid, field_angle_deg, side="right"))
    lower = upper - 1
    span = float(angle_grid[upper] - angle_grid[lower])
    weight = 0.0 if span == 0.0 else (field_angle_deg - float(angle_grid[lower])) / span
    return (1.0 - weight) * radial_profiles_arcsec[lower] + weight * radial_profiles_arcsec[upper]


def _dva_drift_offset(config: StampRenderConfig) -> tuple[np.ndarray, dict[str, Any]]:
    if not config.enable_dynamic_effects or not config.enable_dva_drift:
        return np.zeros(2, dtype=np.float32), {"enabled": False}
    path = _optional_resolved_path(config.dva_model_path)
    if path is None:
        return np.zeros(2, dtype=np.float32), {"enabled": False, "reason": "no_dva_model_path"}
    angle_grid, time_day, radial_profiles = _load_dva_model_cached(str(path))
    elapsed_day = max(0.0, _frame_elapsed_day(config))
    model_span_day = float(time_day[-1]) if len(time_day) else 0.0
    wrapped = model_span_day > 0.0 and elapsed_day > model_span_day
    eval_day = elapsed_day % model_span_day if wrapped else elapsed_day
    profile = _interpolate_dva_profile_for_angle(
        angle_grid,
        radial_profiles,
        DVA_FIELD_ANGLE_DEG,
    )
    start_arcsec = float(np.interp(0.0, time_day, profile))
    end_arcsec = float(np.interp(eval_day, time_day, profile))
    radial_pix = (end_arcsec - start_arcsec) / PIXEL_SCALE_ARCSEC_PER_PIX
    dx, dy = _project_radial_to_xy(radial_pix, DVA_THETA_DEG)
    offset = np.asarray([dx, dy], dtype=np.float32)
    return offset, {
        "enabled": True,
        "model": "ET_DVA_effect_models_slim_v231117",
        "path": str(path),
        "field_angle_deg": DVA_FIELD_ANGLE_DEG,
        "theta_deg": DVA_THETA_DEG,
        "elapsed_day": float(elapsed_day),
        "eval_day": float(eval_day),
        "wrapped_to_model_span": bool(wrapped),
        "radial_pix": float(radial_pix),
        "dx_pix": float(offset[0]),
        "dy_pix": float(offset[1]),
    }


def _thermal_drift_offset(config: StampRenderConfig) -> tuple[np.ndarray, dict[str, Any]]:
    if not config.enable_dynamic_effects or not config.enable_thermal_drift:
        return np.zeros(2, dtype=np.float32), {"enabled": False}
    elapsed_day = max(0.0, _frame_elapsed_day(config))
    frequency = THERMAL_CYCLES_PER_BLOCK / THERMAL_DAYS_PER_BLOCK
    baseline = THERMAL_BASELINE_STEP_ARCSEC_PER_3DAY * (elapsed_day / THERMAL_DAYS_PER_BLOCK)
    radial_arcsec = baseline + THERMAL_AMPLITUDE_ARCSEC * math.sin(
        2.0 * math.pi * frequency * elapsed_day
    )
    radial_pix = radial_arcsec / PIXEL_SCALE_ARCSEC_PER_PIX
    dx, dy = _project_radial_to_xy(radial_pix, THERMAL_THETA_DEG)
    offset = np.asarray([dx, dy], dtype=np.float32)
    return offset, {
        "enabled": True,
        "model": "generate_thermal_drift_scaled",
        "theta_deg": THERMAL_THETA_DEG,
        "elapsed_day": float(elapsed_day),
        "amplitude_arcsec": THERMAL_AMPLITUDE_ARCSEC,
        "baseline_step_arcsec_per_3day": THERMAL_BASELINE_STEP_ARCSEC_PER_3DAY,
        "days_per_block": THERMAL_DAYS_PER_BLOCK,
        "cycles_per_block": THERMAL_CYCLES_PER_BLOCK,
        "radial_pix": float(radial_pix),
        "dx_pix": float(offset[0]),
        "dy_pix": float(offset[1]),
    }


def _momentum_dump_offset(config: StampRenderConfig) -> tuple[np.ndarray, dict[str, Any]]:
    if not config.enable_dynamic_effects or not config.enable_momentum_dump:
        return np.zeros(2, dtype=np.float32), {"enabled": False}
    elapsed_day = max(0.0, _frame_elapsed_day(config))
    current_step = int(math.floor(elapsed_day / MOMENTUM_DUMP_CYCLE_DAY))
    circle_radius_pix = MOMENTUM_DUMP_R68_ARCSEC / PIXEL_SCALE_ARCSEC_PER_PIX
    seed = derive_seed(
        config.global_seed,
        exposure_s=0.0,
        frame_id=0,
        star_id=0,
        effect_type="momentum_dump",
    )
    rng = np.random.default_rng(seed)
    x = 0.0
    y = 0.0
    for _ in range(current_step):
        for _attempt in range(1000):
            theta = float(rng.uniform(0.0, 2.0 * math.pi))
            r_step = float(rng.normal(0.0, circle_radius_pix))
            candidate_x = x + r_step * math.cos(theta)
            candidate_y = y + r_step * math.sin(theta)
            if math.hypot(candidate_x, candidate_y) < circle_radius_pix:
                x = candidate_x
                y = candidate_y
                break
    offset = np.asarray([x, y], dtype=np.float32)
    return offset, {
        "enabled": True,
        "model": "random_walk_within_circle",
        "seed": int(seed),
        "elapsed_day": float(elapsed_day),
        "cycle_day": MOMENTUM_DUMP_CYCLE_DAY,
        "r68_arcsec": MOMENTUM_DUMP_R68_ARCSEC,
        "circle_radius_pix": float(circle_radius_pix),
        "current_step": int(current_step),
        "dx_pix": float(offset[0]),
        "dy_pix": float(offset[1]),
    }


def _psf_breathing_scale(config: StampRenderConfig) -> tuple[float, dict[str, Any]]:
    if not config.enable_dynamic_effects or not config.enable_psf_breathing:
        return 1.0, {"enabled": False}
    elapsed_day = max(0.0, _frame_elapsed_day(config))
    cycle_time = (elapsed_day % PSF_BREATHING_PERIOD_DAY) / PSF_BREATHING_PERIOD_DAY
    scale = 1.0 - PSF_BREATHING_AMPLITUDE + 2.0 * PSF_BREATHING_AMPLITUDE * cycle_time
    return float(scale), {
        "enabled": True,
        "model": "weed_linear_3day",
        "elapsed_day": float(elapsed_day),
        "period_day": PSF_BREATHING_PERIOD_DAY,
        "amplitude": PSF_BREATHING_AMPLITUDE,
        "scale": float(scale),
    }


def _psd_motion_offset(config: StampRenderConfig) -> tuple[np.ndarray, dict[str, Any]]:
    if not config.enable_dynamic_effects or not config.enable_psd_motion:
        return np.zeros(2, dtype=np.float32), {"enabled": False}
    path = _optional_resolved_path(config.psd_motion_path)
    return np.zeros(2, dtype=np.float32), {
        "enabled": True,
        "path": None if path is None else str(path),
        "motion_split_hz": float(1.0 / float(config.exposure_s)),
        "reference_field_angle_deg": REFERENCE_EFFECT_FIELD_ANGLE_DEG,
        "reference_x_axis_angle_deg": REFERENCE_EFFECT_X_AXIS_ANGLE_DEG,
        "implementation": "configured for resource accounting; low-frequency PSD drift is not materialized in stamp_long",
        "dx_pix": 0.0,
        "dy_pix": 0.0,
    }


def _main_rd_core_module():
    module_dir = Path(__file__).resolve().parents[1] / "main_rd_g18_parallel"
    module_dir_str = str(module_dir)
    if module_dir_str not in sys.path:
        sys.path.insert(0, module_dir_str)
    import main_rd_parallel_core

    return main_rd_parallel_core


@lru_cache(maxsize=16)
def _main_rd_full_effect_timeseries_cached(
    *,
    n_frames: int,
    exposure_s: float,
    seed: int,
    enable_psd_motion: bool,
    psd_motion_path_key: str,
    enable_dva: bool,
    enable_thermal: bool,
    enable_momentum_dump: bool,
    enable_psf_breathing: bool,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    main_rd_core = _main_rd_core_module()
    psd_motion_path = None if str(psd_motion_path_key) == "" else Path(psd_motion_path_key)
    return main_rd_core.build_full_effect_timeseries(
        n_frames=int(n_frames),
        seed=int(seed),
        enable_psd_motion=bool(enable_psd_motion),
        psd_motion_path=psd_motion_path,
        enable_dva=bool(enable_dva),
        enable_thermal=bool(enable_thermal),
        enable_momentum_dump=bool(enable_momentum_dump),
        enable_psf_breathing=bool(enable_psf_breathing),
        exposure_s=float(exposure_s),
    )


def _dynamic_effects_for_frame(config: StampRenderConfig) -> dict[str, Any]:
    if not config.enable_dynamic_effects:
        return {
            "enabled": False,
            "total_offset_pix": [0.0, 0.0],
            "psf_scale": 1.0,
            "components": {},
        }
    frame_id = int(config.frame_id)
    n_frames = max(int(config.n_frames), frame_id + 1, 1)
    psd_path = _optional_resolved_path(config.psd_motion_path)
    psd_path_key = "" if psd_path is None else str(psd_path)
    arrays, metadata = _main_rd_full_effect_timeseries_cached(
        n_frames=n_frames,
        exposure_s=float(config.exposure_s),
        seed=int(config.global_seed),
        enable_psd_motion=bool(config.enable_psd_motion),
        psd_motion_path_key=psd_path_key,
        enable_dva=bool(config.enable_dva_drift),
        enable_thermal=bool(config.enable_thermal_drift),
        enable_momentum_dump=bool(config.enable_momentum_dump),
        enable_psf_breathing=bool(config.enable_psf_breathing),
    )
    total_motion = np.asarray(arrays["total_motion_pix"], dtype=np.float32)
    psf_scales = np.asarray(arrays["psf_scale"], dtype=np.float32)
    time_s = np.asarray(arrays["time_s"], dtype=np.float64)
    total = total_motion[frame_id].astype(np.float32)
    psf_scale = float(psf_scales[frame_id])
    return {
        "enabled": True,
        "source": "main_rd_g18_parallel.build_full_effect_timeseries",
        "time_s": float(time_s[frame_id]),
        "time_day": float(_frame_elapsed_day(config)),
        "cadence_s": float(config.exposure_s),
        "motion_split_hz": float(metadata.get("motion_split_hz", 1.0 / float(config.exposure_s))),
        "total_offset_pix": [float(total[0]), float(total[1])],
        "psf_scale": float(psf_scale),
        "components": metadata.get("components", {}),
    }


def _source_psf(config: StampRenderConfig) -> tuple[np.ndarray, dict[str, Any]]:
    if config.use_photsim7_psf:
        psf = load_photsim7_psf_stamp(
            stamp_size=int(config.stamp_size),
            psf_bundle_name=str(config.psf_bundle_name),
            psf_field_id=int(config.psf_field_id),
            psf_subpixels=int(config.psf_subpixels),
        )
        source = "photsim7-data"
    else:
        psf = _gaussian_psf(config.stamp_size, config.psf_sigma_pix)
        source = "gaussian"
    dynamic = _dynamic_effects_for_frame(config)
    if dynamic["enabled"]:
        psf = _rescale_psf(psf, scale=float(dynamic["psf_scale"]))
        dx, dy = dynamic["total_offset_pix"]
        if not (np.isclose(dx, 0.0, atol=1e-7) and np.isclose(dy, 0.0, atol=1e-7)):
            psf = _shift_image_bilinear(psf, dx_pix=float(dx), dy_pix=float(dy))
    return _normalize_psf(psf), {
        "source": source,
        "use_photsim7_psf": bool(config.use_photsim7_psf),
        "psf_bundle_name": str(config.psf_bundle_name),
        "psf_field_id": int(config.psf_field_id),
        "psf_subpixels": int(config.psf_subpixels),
        "psf_sigma_pix": float(config.psf_sigma_pix),
        "dynamic_effects": dynamic,
        "jitter_integrated_psf": {
            "enabled": bool(config.use_photsim7_psf),
            "n_models": int(config.jitter_integrated_psf_models),
            "n_frames_per_model": int(config.jitter_frames_per_model),
            "implementation": "precomputed 12-degree PSF plus per-frame offset/breathing in stamp_long",
        },
    }


def _cosmic_ray_mean_events(config: StampRenderConfig) -> float:
    pixel_cm = float(config.pixel_size_um) * 1e-4
    area_cm2 = int(config.stamp_size) * int(config.stamp_size) * pixel_cm * pixel_cm
    return float(config.cosmic_ray_event_rate) * area_cm2 * float(config.exposure_s)


@lru_cache(maxsize=8)
def _load_cosmic_ray_library(path: str | None) -> tuple[np.ndarray, ...] | None:
    if path is None or str(path).strip() == "":
        return None
    library_path = resolve_data_path(str(path))
    if not library_path.exists():
        return None
    try:
        data = np.load(library_path, allow_pickle=True)
        stamps = data["stamps"]
    except Exception:
        return None
    return tuple(np.asarray(stamp, dtype=np.float32) for stamp in stamps)


def _inject_cosmic_rays(
    image: np.ndarray,
    *,
    config: StampRenderConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    mean_events = _cosmic_ray_mean_events(config)
    n_events = int(rng.poisson(max(mean_events, 0.0)))
    if n_events == 0:
        return image, 0

    library = _load_cosmic_ray_library(config.cosmic_ray_library_path)
    out = np.array(image, dtype=np.float32, copy=True)
    for _ in range(n_events):
        if library:
            stamp_adu = library[int(rng.integers(0, len(library)))]
            event = stamp_adu.astype(np.float32, copy=False) * float(config.gain_e_per_adu)
        else:
            event = np.array([[float(config.cosmic_ray_peak_adu) * float(config.gain_e_per_adu)]], dtype=np.float32)
        eh, ew = event.shape
        y0 = int(rng.integers(-eh + 1, int(config.stamp_size)))
        x0 = int(rng.integers(-ew + 1, int(config.stamp_size)))
        y1 = y0 + eh
        x1 = x0 + ew
        cy0 = max(0, y0)
        cx0 = max(0, x0)
        cy1 = min(int(config.stamp_size), y1)
        cx1 = min(int(config.stamp_size), x1)
        if cy1 <= cy0 or cx1 <= cx0:
            continue
        sy0 = cy0 - y0
        sx0 = cx0 - x0
        out[cy0:cy1, cx0:cx1] += event[sy0 : sy0 + (cy1 - cy0), sx0 : sx0 + (cx1 - cx0)]
    return out, n_events


def _render_numpy_stamp(config: StampRenderConfig) -> np.ndarray:
    params = exposure_parameters(
        config.exposure_s,
        read_noise_10s_e_pix=config.read_noise_10s_e_pix,
    )
    rng = np.random.default_rng(int(config.seed))
    psf, _psf_metadata = _source_psf(config)
    source_mean = psf * np.float32(float(config.star_flux_e_s) * float(config.exposure_s))
    background_mean = np.full(
        (int(config.stamp_size), int(config.stamp_size)),
        float(config.background_e_s_pix) * float(config.exposure_s),
        dtype=np.float32,
    )
    scattered_mean = np.full_like(
        background_mean,
        float(config.scattered_light_e_s_pix) * float(config.exposure_s),
    )
    dark_mean = np.full_like(
        background_mean,
        float(config.dark_e_s_pix) * float(config.exposure_s),
    )
    image = rng.poisson(source_mean).astype(np.float32)
    image += rng.poisson(background_mean).astype(np.float32)
    image += rng.poisson(scattered_mean).astype(np.float32)
    image += rng.poisson(dark_mean).astype(np.float32)
    image += rng.normal(
        loc=0.0,
        scale=float(params.read_noise_e_pix),
        size=image.shape,
    ).astype(np.float32)
    return image


def _resolve_requested_device(device: str) -> str:
    requested = str(device)
    if requested != "auto":
        return requested
    try:
        import torch
    except ModuleNotFoundError:
        return "cpu"
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _is_explicit_cuda_device(device: str) -> bool:
    return str(device).startswith("cuda")


def _jitter_integrated_psf_enabled(config: StampRenderConfig) -> bool:
    return bool(
        config.use_photsim7_psf
        and config.enable_dynamic_effects
        and config.enable_psd_motion
        and int(config.jitter_integrated_psf_models) > 0
        and int(config.jitter_frames_per_model) > 0
    )


def _jitter_model_index_for_frame(config: StampRenderConfig) -> int | None:
    if not _jitter_integrated_psf_enabled(config):
        return None
    return int(config.frame_id) % int(config.jitter_integrated_psf_models)


def _photsim7_psf_metadata(
    config: StampRenderConfig,
    *,
    dynamic: dict[str, Any] | None = None,
    jitter_metadata: dict[str, Any] | None = None,
    jitter_model_index: int | None = None,
) -> dict[str, Any]:
    jitter_enabled = _jitter_integrated_psf_enabled(config)
    if jitter_model_index is None and jitter_enabled:
        jitter_model_index = _jitter_model_index_for_frame(config)
    jitter_payload = {
        "enabled": bool(jitter_enabled),
        "n_models": int(config.jitter_integrated_psf_models),
        "n_frames_per_model": int(config.jitter_frames_per_model),
        "jitter_model_index": (
            None if jitter_model_index is None else int(jitter_model_index)
        ),
        "implementation": (
            "Photsim7 PSFModelManager jitter-integrated PSF bank; "
            "high-frequency PSD attitude motion f > 1/exposure_s is integrated "
            "into PSF broadening."
        ),
    }
    if jitter_metadata:
        jitter_payload["metadata"] = jitter_metadata
    return {
        "source": "photsim7.stamp_renderer",
        "use_photsim7_psf": True,
        "psf_bundle_name": str(config.psf_bundle_name),
        "psf_field_id": int(config.psf_field_id),
        "psf_subpixels": int(config.psf_subpixels),
        "psf_sigma_pix": float(config.psf_sigma_pix),
        "dynamic_effects": _dynamic_effects_for_frame(config) if dynamic is None else dynamic,
        "jitter_integrated_psf": jitter_payload,
        "detector_response": {
            "enabled": bool(config.enable_detector_response),
            "response_model": "photsim7.stamp_renderer.StampLocalSubpixelResponseSampler",
            "response_padding_pix": int(config.response_padding_pix),
            "inter_pixel_response_sigma": float(config.inter_pixel_response_sigma),
            "intra_pixel_response_sigma": float(config.intra_pixel_response_sigma),
            "pixel_response_profile": bool(config.enable_pixel_phase_response),
            "stamp_id": int(config.star_id),
        },
    }


def _build_photsim7_stamp_renderer_from_xy_jitter(
    *,
    stamp_size: int,
    exposure_s: float,
    psf_bundle_name: str,
    psf_field_id: int,
    psf_subpixels: int,
    device: str,
    integrate_jitter: bool,
    jitter_integrated_psf_models: int,
    jitter_frames_per_model: int,
    xy_jitter_pix: np.ndarray | None,
    jitter_metadata: dict[str, Any] | None,
):
    ensure_photsim7_imports()
    from astropy import units as u
    from photsim7.psf.model import PSFModelManager
    from photsim7.stamp_renderer import SingleCadenceStampRenderer

    actor_config = {
        "bundle_name": str(psf_bundle_name),
        "pixel_scale": PIXEL_SCALE_ARCSEC_PER_PIX * u.arcsec / u.pix,
        "n_rows": int(stamp_size),
        "n_cols": int(stamp_size),
        "n_subpixels": int(psf_subpixels),
        "integrate_jitter": bool(integrate_jitter),
        "n_jitter_integrated_psf_models": (
            int(jitter_integrated_psf_models) if bool(integrate_jitter) else 1
        ),
        "n_jitter_frames": int(jitter_frames_per_model) if bool(integrate_jitter) else 1,
        "compute_device": str(device),
        "float_precision": 32,
    }
    psf_manager = PSFModelManager(
        config=actor_config,
        warp_frame_batch_size=10,
        xy_jitter_pix=xy_jitter_pix,
        intialize=True,
        build_jit_int_models=True,
        field_ids=np.asarray([int(psf_field_id)], dtype=np.int64),
        pad_to_detector_shape=False,
    )
    sim_config = {
        "Subpixels Per Pixel Dim": int(psf_subpixels),
        "Subtract Nonstellar Mean": False,
    }
    renderer = SingleCadenceStampRenderer(
        sim_config=sim_config,
        psf_model_manager=psf_manager,
        stamp_size_pix=int(stamp_size),
        frame_exposure=float(exposure_s) * u.s,
        detector_response_sampler=None,
        compute_device=str(device),
        float_precision=32,
    )
    renderer.jitter_metadata = {} if jitter_metadata is None else dict(jitter_metadata)
    return renderer


@lru_cache(maxsize=16)
def _build_photsim7_stamp_renderer_cached(
    stamp_size: int,
    exposure_s: float,
    psf_bundle_name: str,
    psf_field_id: int,
    psf_subpixels: int,
    device: str,
    et_data_dir_key: str,
    integrate_jitter: bool,
    jitter_integrated_psf_models: int,
    jitter_frames_per_model: int,
    jitter_seed: int,
    psd_motion_path_key: str,
):
    ensure_photsim7_imports()
    from astropy import units as u
    from photsim7.psf.model import PSFModelManager
    from photsim7.stamp_renderer import SingleCadenceStampRenderer

    xy_jitter_pix = None
    jitter_metadata = {"enabled": False, "reason": "disabled"}
    if bool(integrate_jitter):
        psd_motion_path = (
            None if str(psd_motion_path_key) == "" else Path(psd_motion_path_key)
        )
        main_rd_core = _main_rd_core_module()
        xy_jitter_pix, jitter_metadata = main_rd_core.jitter_integrated_psf_offsets(
            seed=int(jitter_seed),
            enable_psd_motion=True,
            enable_jitter_integrated_psf=True,
            psd_motion_path=psd_motion_path,
            n_models=int(jitter_integrated_psf_models),
            n_frames_per_model=int(jitter_frames_per_model),
            exposure_s=float(exposure_s),
        )

    actor_config = {
        "bundle_name": str(psf_bundle_name),
        "pixel_scale": PIXEL_SCALE_ARCSEC_PER_PIX * u.arcsec / u.pix,
        "n_rows": int(stamp_size),
        "n_cols": int(stamp_size),
        "n_subpixels": int(psf_subpixels),
        "integrate_jitter": bool(integrate_jitter),
        "n_jitter_integrated_psf_models": (
            int(jitter_integrated_psf_models) if bool(integrate_jitter) else 1
        ),
        "n_jitter_frames": int(jitter_frames_per_model) if bool(integrate_jitter) else 1,
        "compute_device": str(device),
        "float_precision": 32,
    }
    psf_manager = PSFModelManager(
        config=actor_config,
        warp_frame_batch_size=10,
        xy_jitter_pix=xy_jitter_pix,
        intialize=True,
        build_jit_int_models=True,
        field_ids=np.asarray([int(psf_field_id)], dtype=np.int64),
        pad_to_detector_shape=False,
    )
    sim_config = {
        "Subpixels Per Pixel Dim": int(psf_subpixels),
        "Subtract Nonstellar Mean": False,
    }
    renderer = SingleCadenceStampRenderer(
        sim_config=sim_config,
        psf_model_manager=psf_manager,
        stamp_size_pix=int(stamp_size),
        frame_exposure=float(exposure_s) * u.s,
        detector_response_sampler=None,
        compute_device=str(device),
        float_precision=32,
    )
    renderer.jitter_metadata = jitter_metadata
    return renderer


def _build_photsim7_stamp_renderer(config: StampRenderConfig, device: str):
    psd_path = _optional_resolved_path(config.psd_motion_path)
    jitter_seed = derive_seed(
        config.global_seed,
        exposure_s=config.exposure_s,
        frame_id=0,
        star_id=0,
        effect_type="jitter_integrated_psf",
    )
    return _build_photsim7_stamp_renderer_cached(
        int(config.stamp_size),
        float(config.exposure_s),
        str(config.psf_bundle_name),
        int(config.psf_field_id),
        int(config.psf_subpixels),
        str(device),
        str(DEFAULT_ET_DATA_DIR),
        bool(_jitter_integrated_psf_enabled(config)),
        int(config.jitter_integrated_psf_models),
        int(config.jitter_frames_per_model),
        int(jitter_seed),
        "" if psd_path is None else str(psd_path),
    )


@lru_cache(maxsize=4096)
def _build_stamp_local_response_sampler_cached(
    stamp_size: int,
    psf_subpixels: int,
    response_padding_pix: int,
    global_seed: int,
    star_id: int,
    device: str,
    inter_pixel_response_sigma: float,
    inter_pixel_response_nominal: float,
    intra_pixel_response_sigma: float,
    enable_inter_pixel_response: bool,
    enable_intra_pixel_response: bool,
    enable_pixel_phase_response: bool,
):
    ensure_photsim7_imports()
    from photsim7.stamp_renderer import StampLocalSubpixelResponseSampler

    return StampLocalSubpixelResponseSampler(
        stamp_size_pix=int(stamp_size),
        n_subpixels=int(psf_subpixels),
        response_padding_pix=int(response_padding_pix),
        inter_pixel_response_sigma=float(inter_pixel_response_sigma),
        inter_pixel_nominal_response=float(inter_pixel_response_nominal),
        intra_pixel_response_sigma=float(intra_pixel_response_sigma),
        pixel_response_profile_mod="flux conserved",
        enable_inter_pixel_response=bool(enable_inter_pixel_response),
        enable_intra_pixel_response=bool(enable_intra_pixel_response),
        enable_pixel_phase_response=bool(enable_pixel_phase_response),
        random_seed=int(global_seed),
        stamp_id=int(star_id),
        compute_device=str(device),
        float_precision=32,
    )


def _build_stamp_local_response_sampler(config: StampRenderConfig, device: str):
    if not config.enable_detector_response:
        return None
    return _build_stamp_local_response_sampler_cached(
        int(config.stamp_size),
        int(config.psf_subpixels),
        int(config.response_padding_pix),
        int(config.global_seed),
        int(config.star_id),
        str(device),
        float(config.inter_pixel_response_sigma),
        float(config.inter_pixel_response_nominal),
        float(config.intra_pixel_response_sigma),
        bool(config.enable_inter_pixel_response),
        bool(config.enable_intra_pixel_response),
        bool(config.enable_pixel_phase_response),
    )


def _render_photsim7_stamp(config: StampRenderConfig) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        import torch
        from astropy import units as u
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Photsim7 stamp rendering requires torch and astropy."
        ) from exc

    requested_device = _resolve_requested_device(config.device)
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device requested ({requested_device}) but torch.cuda.is_available() is False"
        )

    torch.manual_seed(int(config.seed))
    if requested_device.startswith("cuda"):
        torch.cuda.manual_seed_all(int(config.seed))

    params = exposure_parameters(
        config.exposure_s,
        read_noise_10s_e_pix=config.read_noise_10s_e_pix,
    )
    dynamic = _dynamic_effects_for_frame(config)
    dx_pix, dy_pix = (0.0, 0.0)
    psf_scale = 1.0
    if dynamic["enabled"]:
        dx_pix, dy_pix = dynamic["total_offset_pix"]
        psf_scale = float(dynamic["psf_scale"])

    renderer = _build_photsim7_stamp_renderer(config, requested_device)
    jitter_model_index = _jitter_model_index_for_frame(config)
    detector_response_sampler = _build_stamp_local_response_sampler(
        config,
        requested_device,
    )
    components = renderer.render_single_cadence(
        photon_count=float(config.star_flux_e_s) * float(config.exposure_s) * u.electron,
        field_id=int(config.psf_field_id),
        target_x_offset_pix=float(dx_pix),
        target_y_offset_pix=float(dy_pix),
        psf_scale=float(psf_scale),
        jitter_model_index=jitter_model_index,
        detector_response_sampler=detector_response_sampler,
        response_y_start_pix=int(config.response_padding_pix),
        response_x_start_pix=int(config.response_padding_pix),
        enable_stellar_photon_noise=True,
        enable_background_light=True,
        enable_scattered_light=True,
        enable_dark_current=True,
        enable_readout_noise=True,
        background_flux_per_pixel=(
            float(config.background_e_s_pix) * u.electron / u.s / u.pix
        ),
        scattered_light_per_pixel=(
            float(config.scattered_light_e_s_pix) * u.electron / u.s / u.pix
        ),
        dark_current_per_pixel=(
            float(config.dark_e_s_pix) * u.electron / u.s / u.pix
        ),
        readout_noise=float(params.read_noise_e_pix) * u.electron / u.pix,
        gain_ratio=1.0,
        subtract_nonstellar_mean=False,
        return_numpy=True,
    )
    return (
        np.asarray(components["final_image"], dtype=np.float32),
        _photsim7_psf_metadata(
            config,
            dynamic=dynamic,
            jitter_metadata=getattr(renderer, "jitter_metadata", None),
            jitter_model_index=jitter_model_index,
        ),
    )


def _render_expected_photsim7_stamp_with_renderer(
    *,
    renderer: Any,
    config: StampRenderConfig,
    device: str,
    jitter_model_index: int,
) -> np.ndarray:
    from astropy import units as u

    dynamic = _dynamic_effects_for_frame(config)
    dx_pix, dy_pix = (0.0, 0.0)
    psf_scale = 1.0
    if dynamic["enabled"]:
        dx_pix, dy_pix = dynamic["total_offset_pix"]
        psf_scale = float(dynamic["psf_scale"])
    detector_response_sampler = _build_stamp_local_response_sampler(config, device)
    image = renderer.render_expected_stellar_stamp(
        photon_count=float(config.star_flux_e_s) * float(config.exposure_s) * u.electron,
        field_id=int(config.psf_field_id),
        target_x_offset_pix=float(dx_pix),
        target_y_offset_pix=float(dy_pix),
        psf_scale=float(psf_scale),
        jitter_model_index=int(jitter_model_index),
        detector_response_sampler=detector_response_sampler,
        response_y_start_pix=int(config.response_padding_pix),
        response_x_start_pix=int(config.response_padding_pix),
        return_numpy=True,
    )
    return np.asarray(image, dtype=np.float32)


def _render_torch_stamp(config: StampRenderConfig) -> np.ndarray:
    try:
        import torch
    except ModuleNotFoundError as exc:
        if _is_explicit_cuda_device(config.device):
            raise RuntimeError("CUDA device requested but torch is not installed") from exc
        return _render_numpy_stamp(config)

    requested_device = _resolve_requested_device(config.device)
    if requested_device == "cpu":
        return _render_numpy_stamp(config)
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device requested ({requested_device}) but torch.cuda.is_available() is False"
        )

    params = exposure_parameters(
        config.exposure_s,
        read_noise_10s_e_pix=config.read_noise_10s_e_pix,
    )
    device = torch.device(requested_device)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(config.seed))
    psf_np, _psf_metadata = _source_psf(config)
    psf = torch.as_tensor(psf_np, dtype=torch.float32, device=device)
    source_mean = psf * float(config.star_flux_e_s) * float(config.exposure_s)
    background_mean = torch.full_like(
        source_mean,
        float(config.background_e_s_pix) * float(config.exposure_s),
    )
    scattered_mean = torch.full_like(
        source_mean,
        float(config.scattered_light_e_s_pix) * float(config.exposure_s),
    )
    dark_mean = torch.full_like(
        source_mean,
        float(config.dark_e_s_pix) * float(config.exposure_s),
    )
    image = torch.poisson(source_mean, generator=generator)
    image = image + torch.poisson(background_mean, generator=generator)
    image = image + torch.poisson(scattered_mean, generator=generator)
    image = image + torch.poisson(dark_mean, generator=generator)
    image = image + torch.normal(
        mean=0.0,
        std=float(params.read_noise_e_pix),
        size=tuple(image.shape),
        generator=generator,
        dtype=torch.float32,
        device=device,
    )
    return image.detach().cpu().numpy().astype(np.float32, copy=False)


def render_synthetic_stamp(config: StampRenderConfig) -> tuple[np.ndarray, dict[str, Any]]:
    params = exposure_parameters(
        config.exposure_s,
        read_noise_10s_e_pix=config.read_noise_10s_e_pix,
    )
    if config.use_photsim7_psf:
        image, psf_metadata = _render_photsim7_stamp(config)
    else:
        image = _render_torch_stamp(config)
        _psf, psf_metadata = _source_psf(config)
    rng = np.random.default_rng(
        derive_seed(
            config.global_seed,
            exposure_s=config.exposure_s,
            frame_id=config.frame_id,
            star_id=config.star_id,
            effect_type="cosmic_ray",
        )
    )
    image, actual_events = _inject_cosmic_rays(image, config=config, rng=rng)
    image = image.astype(np.float32, copy=False)
    metadata = {
        "exposure_s": float(config.exposure_s),
        "n_coadd_equiv": float(params.n_coadd_equiv),
        "star_flux_e_s": float(config.star_flux_e_s),
        "star_flux_mode": str(config.star_flux_mode),
        "et_mag": None if config.et_mag is None else float(config.et_mag),
        "background_e_s_pix": float(config.background_e_s_pix),
        "scattered_light_e_s_pix": float(config.scattered_light_e_s_pix),
        "dark_e_s_pix": float(config.dark_e_s_pix),
        "read_noise_10s_e_pix": float(config.read_noise_10s_e_pix),
        "read_noise_e_pix": float(params.read_noise_e_pix),
        "cosmic_ray_mean_events": float(_cosmic_ray_mean_events(config)),
        "actual_cosmic_events": int(actual_events),
        "cosmic_ray_library_path": config.cosmic_ray_library_path,
        "cosmic_ray_event_rate_cm2_s": float(config.cosmic_ray_event_rate),
        "gain_e_per_adu": float(config.gain_e_per_adu),
        "pixel_size_um": float(config.pixel_size_um),
        "psf": psf_metadata,
        "seed": int(config.seed),
        "global_seed": int(config.global_seed),
        "star_id": int(config.star_id),
        "frame_id": int(config.frame_id),
        "device": _resolve_requested_device(config.device),
        "requested_device": str(config.device),
    }
    return image, metadata


def write_stamp_npy(
    *,
    output_root: Path | str,
    case_id: str,
    exposure_s: float,
    stamp: np.ndarray,
    star_id: int,
    frame_id: int,
    stamp_size: int,
    seed: int,
    write_mode: str,
) -> StampRecord:
    params = exposure_parameters(exposure_s)
    write_mode = str(write_mode)
    if write_mode not in {"all", "none", "sample"}:
        raise ValueError(f"Unsupported write_mode {write_mode!r}")

    file_path = ""
    file_size = 0
    status = "skipped"
    should_write = write_mode == "all" or (
        write_mode == "sample" and int(star_id) == 0 and int(frame_id) == 0
    )
    if should_write:
        root = Path(output_root).expanduser()
        exp_dir = root / str(case_id) / f"exp{int(round(float(exposure_s))):03d}" / f"star_{int(star_id):06d}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        path = exp_dir / f"frame_{int(frame_id):06d}.npy"
        if _existing_stamp_is_valid(path, stamp_size=int(stamp_size)):
            file_path = str(path)
            file_size = int(path.stat().st_size)
            status = "skipped_existing"
        else:
            tmp_path = path.with_name(
                f".{path.name}.tmp.{os.getpid()}.{int(seed)}"
            )
            try:
                with tmp_path.open("wb") as handle:
                    np.save(handle, np.asarray(stamp, dtype=np.float32))
                os.replace(tmp_path, path)
            finally:
                if tmp_path.exists():
                    tmp_path.unlink()
            file_path = str(path)
            file_size = int(path.stat().st_size)
            status = "completed"

    return StampRecord(
        case_id=str(case_id),
        exposure_time_s=float(exposure_s),
        n_coadd_equiv=float(params.n_coadd_equiv),
        star_id=int(star_id),
        frame_id=int(frame_id),
        stamp_size=int(stamp_size),
        dtype="float32",
        unit="electrons",
        seed=int(seed),
        file_path=file_path,
        file_size_bytes=file_size,
        status=status,
    )


def _existing_stamp_is_valid(path: Path, *, stamp_size: int) -> bool:
    if not path.exists():
        return False
    try:
        existing = np.load(path, mmap_mode="r")
        return existing.shape == (int(stamp_size), int(stamp_size)) and existing.dtype == np.dtype("float32")
    except Exception:
        return False


def write_manifest(path: Path | str, records: Iterable[StampRecord]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(record) for record in records]
    fieldnames = [
        "case_id",
        "exposure_time_s",
        "n_coadd_equiv",
        "star_id",
        "frame_id",
        "stamp_size",
        "dtype",
        "unit",
        "seed",
        "file_path",
        "file_size_bytes",
        "status",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _worker_device(device_mode: str, gpu_id: str | None) -> str:
    if device_mode == "cpu":
        return "cpu"
    if gpu_id is None:
        return "cpu"
    if device_mode == "cuda":
        return f"cuda:{gpu_id}"
    try:
        import torch
    except ModuleNotFoundError:
        return "cpu"
    return f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"


def _build_render_config(
    case: BenchmarkCase,
    *,
    seed: int,
    global_seed: int,
    star_id: int,
    frame_id: int,
    device: str,
    render_options: RenderOptions,
) -> StampRenderConfig:
    star_flux_mode = _normalize_star_flux_mode(render_options.star_flux_mode)
    et_mag = None
    star_flux_e_s = float(render_options.star_flux_e_s)
    if star_flux_mode == "random_et_mag":
        et_mag = sample_star_et_mag(
            global_seed=global_seed,
            star_id=star_id,
            et_mag_min=render_options.et_mag_min,
            et_mag_max=render_options.et_mag_max,
        )
        star_flux_e_s = float(et_mag_to_photon_rate_e_s(et_mag))
    return StampRenderConfig(
        exposure_s=float(case.exposure_s),
        stamp_size=int(case.stamp_size),
        star_flux_e_s=float(star_flux_e_s),
        star_flux_mode=star_flux_mode,
        et_mag=et_mag,
        background_e_s_pix=float(render_options.background_e_s_pix),
        scattered_light_e_s_pix=float(render_options.scattered_light_e_s_pix),
        dark_e_s_pix=float(render_options.dark_e_s_pix),
        read_noise_10s_e_pix=float(render_options.read_noise_10s_e_pix),
        gain_e_per_adu=float(render_options.gain_e_per_adu),
        cosmic_ray_event_rate=float(render_options.cosmic_ray_event_rate),
        cosmic_ray_library_path=render_options.cosmic_ray_library_path,
        cosmic_ray_peak_adu=float(render_options.cosmic_ray_peak_adu),
        pixel_size_um=float(render_options.pixel_size_um),
        psf_sigma_pix=float(render_options.psf_sigma_pix),
        psf_bundle_name=str(render_options.psf_bundle_name),
        psf_field_id=int(render_options.psf_field_id),
        psf_subpixels=int(render_options.psf_subpixels),
        use_photsim7_psf=bool(render_options.use_photsim7_psf),
        psd_motion_path=render_options.psd_motion_path,
        dva_model_path=render_options.dva_model_path,
        jitter_integrated_psf_models=int(render_options.jitter_integrated_psf_models),
        jitter_frames_per_model=int(render_options.jitter_frames_per_model),
        enable_detector_response=bool(render_options.enable_detector_response),
        response_padding_pix=int(render_options.response_padding_pix),
        inter_pixel_response_sigma=float(render_options.inter_pixel_response_sigma),
        inter_pixel_response_nominal=float(render_options.inter_pixel_response_nominal),
        intra_pixel_response_sigma=float(render_options.intra_pixel_response_sigma),
        enable_inter_pixel_response=bool(render_options.enable_inter_pixel_response),
        enable_intra_pixel_response=bool(render_options.enable_intra_pixel_response),
        enable_pixel_phase_response=bool(render_options.enable_pixel_phase_response),
        enable_dynamic_effects=bool(render_options.enable_dynamic_effects),
        enable_psd_motion=bool(render_options.enable_psd_motion),
        enable_dva_drift=bool(render_options.enable_dva_drift),
        enable_thermal_drift=bool(render_options.enable_thermal_drift),
        enable_momentum_dump=bool(render_options.enable_momentum_dump),
        enable_psf_breathing=bool(render_options.enable_psf_breathing),
        seed=int(seed),
        global_seed=int(global_seed),
        star_id=int(star_id),
        frame_id=int(frame_id),
        n_frames=int(case.n_frames),
        device=device,
    )


def _path_metadata(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"path": None, "exists": False}
    return {"path": str(path), "exists": bool(path.exists())}


def _resource_metadata(render_options: RenderOptions) -> dict[str, Any]:
    return {
        "et_data_dir": str(DEFAULT_ET_DATA_DIR),
        "psf_file": _path_metadata(_psf_bundle_file(str(render_options.psf_bundle_name))),
        "cosmic_ray_library": _path_metadata(_optional_resolved_path(render_options.cosmic_ray_library_path)),
        "psd_motion": _path_metadata(_optional_resolved_path(render_options.psd_motion_path)),
        "dva_model": _path_metadata(_optional_resolved_path(render_options.dva_model_path)),
    }


def _case_physics_metadata(
    case: BenchmarkCase,
    *,
    render_options: RenderOptions,
    global_seed: int,
) -> dict[str, Any]:
    seed = derive_seed(
        global_seed,
        exposure_s=case.exposure_s,
        frame_id=0,
        star_id=0,
        effect_type="stamp",
    )
    config = _build_render_config(
        case,
        seed=seed,
        global_seed=global_seed,
        star_id=0,
        frame_id=0,
        device="cpu",
        render_options=render_options,
    )
    params = exposure_parameters(
        case.exposure_s,
        read_noise_10s_e_pix=render_options.read_noise_10s_e_pix,
    )
    if config.use_photsim7_psf:
        psf_metadata = _photsim7_psf_metadata(config)
    else:
        _psf, psf_metadata = _source_psf(config)
    return {
        "exposure_parameters": asdict(params),
        "resource_paths": _resource_metadata(render_options),
        "representative_frame": {
            "star_id": 0,
            "frame_id": 0,
            "seed": int(seed),
            "star_flux_mode": str(config.star_flux_mode),
            "et_mag": None if config.et_mag is None else float(config.et_mag),
            "star_flux_e_s": float(config.star_flux_e_s),
            "psf": psf_metadata,
            "cosmic_ray_mean_events": float(_cosmic_ray_mean_events(config)),
        },
    }


def _stamp_shard_path(task: WorkerTask) -> Path:
    return (
        Path(task.output_root).expanduser()
        / task.case.case_id
        / "shards"
        / f"stamps.worker{int(task.worker_rank):03d}.h5"
    )


def _stamp_shard_provenance(task: WorkerTask) -> dict[str, Any]:
    return {
        "producer": "ET-mainsim.stamp_long",
        "case": asdict(task.case),
        "worker_rank": int(task.worker_rank),
        "world_size": int(task.world_size),
        "global_seed": int(task.global_seed),
        "device_mode": str(task.device_mode),
        "gpu_id": None if task.gpu_id is None else str(task.gpu_id),
        "render_options": asdict(task.render_options),
    }


def _validate_complete_stamp_shard(task: WorkerTask, path: Path) -> int:
    ensure_photsim7_imports()
    from photsim7.artifacts import StampShardReader

    expected_star_ids = tuple(range(task.star_range.start, task.star_range.stop))
    expected_frame_ids = tuple(range(int(task.case.n_frames)))
    expected_chunk = (
        1,
        min(256, len(expected_frame_ids)),
        int(task.case.stamp_size),
        int(task.case.stamp_size),
    )
    with StampShardReader(path) as reader:
        spec = reader.spec
        checks = {
            "run_id": (spec.run_id, str(task.case.case_id)),
            "case_id": (spec.case_id, str(task.case.case_id)),
            "star_ids": (spec.star_ids, expected_star_ids),
            "frame_ids": (spec.frame_ids, expected_frame_ids),
            "image_shape": (
                spec.image_shape,
                (int(task.case.stamp_size), int(task.case.stamp_size)),
            ),
            "chunk_shape": (spec.chunk_shape, expected_chunk),
            "dtype": (spec.dtype_name, "float32"),
            "unit": (spec.unit, "electron"),
            "domain": (spec.domain, "electrons"),
            "provenance": (
                dict(spec.provenance),
                _stamp_shard_provenance(task),
            ),
        }
        mismatches = [name for name, (actual, expected) in checks.items() if actual != expected]
        if mismatches:
            raise ValueError(
                f"Existing final HDF5 shard {path} does not match the worker task: "
                + ", ".join(mismatches)
            )
        return int(spec.item_count)


def _run_worker_task_hdf5(
    task: WorkerTask,
    *,
    start: float,
    device: str,
) -> WorkerResult:
    if str(task.write_mode) != "all":
        raise ValueError("hdf5 output requires write_mode='all'")
    ensure_photsim7_imports()
    from photsim7.artifacts import ItemStatus, StampShardWriter

    shard_path = _stamp_shard_path(task)
    if shard_path.exists():
        item_count = _validate_complete_stamp_shard(task, shard_path)
        return WorkerResult(
            worker_rank=int(task.worker_rank),
            n_stamps=item_count,
            n_written=0,
            n_skipped=item_count,
            n_failed=0,
            output_bytes=int(shard_path.stat().st_size),
            elapsed_s=float(time.perf_counter() - start),
            manifest_path="",
            n_records=0,
            output_format="hdf5",
            artifact_path=str(shard_path),
            n_shards=1,
        )

    writer = StampShardWriter(
        shard_path,
        run_id=str(task.case.case_id),
        case_id=str(task.case.case_id),
        star_ids=range(task.star_range.start, task.star_range.stop),
        frame_ids=range(int(task.case.n_frames)),
        stamp_shape=(int(task.case.stamp_size), int(task.case.stamp_size)),
        dtype="float32",
        unit="electron",
        domain="electrons",
        provenance=_stamp_shard_provenance(task),
        resume=True,
    )
    n_stamps = 0
    n_written = 0
    n_skipped = 0
    try:
        for star_id in range(task.star_range.start, task.star_range.stop):
            for frame_id in range(int(task.case.n_frames)):
                n_stamps += 1
                if writer.item_status(star_id, frame_id) is ItemStatus.COMPLETE:
                    n_skipped += 1
                    continue
                seed = derive_seed(
                    task.global_seed,
                    exposure_s=task.case.exposure_s,
                    frame_id=frame_id,
                    star_id=star_id,
                    effect_type="stamp",
                )
                config = _build_render_config(
                    task.case,
                    seed=seed,
                    global_seed=task.global_seed,
                    star_id=star_id,
                    frame_id=frame_id,
                    device=device,
                    render_options=task.render_options,
                )
                stamp, _metadata = render_synthetic_stamp(config)
                writer.write_stamp(
                    star_id,
                    frame_id,
                    stamp,
                    seed=seed,
                )
                n_written += 1
        final_path = writer.finalize()
    finally:
        writer.close()

    return WorkerResult(
        worker_rank=int(task.worker_rank),
        n_stamps=int(n_stamps),
        n_written=int(n_written),
        n_skipped=int(n_skipped),
        n_failed=0,
        output_bytes=int(final_path.stat().st_size),
        elapsed_s=float(time.perf_counter() - start),
        manifest_path="",
        n_records=0,
        output_format="hdf5",
        artifact_path=str(final_path),
        n_shards=1,
    )


def _run_worker_task(task: WorkerTask) -> WorkerResult:
    start = time.perf_counter()
    output_format = _normalize_output_format(task.output_format)
    device = _worker_device(task.device_mode, task.gpu_id)
    if output_format == "hdf5":
        return _run_worker_task_hdf5(task, start=start, device=device)

    records: list[StampRecord] = []
    n_stamps = 0
    n_written = 0
    n_skipped = 0
    n_failed = 0
    output_bytes = 0
    active_write_mode = task.write_mode
    n_sample_outputs = 0
    for star_id in range(task.star_range.start, task.star_range.stop):
        for frame_id in range(int(task.case.n_frames)):
            seed = derive_seed(
                task.global_seed,
                exposure_s=task.case.exposure_s,
                frame_id=frame_id,
                star_id=star_id,
                effect_type="stamp",
            )
            config = _build_render_config(
                task.case,
                seed=seed,
                global_seed=task.global_seed,
                star_id=star_id,
                frame_id=frame_id,
                device=device,
                render_options=task.render_options,
            )
            stamp, _metadata = render_synthetic_stamp(config)
            record = write_stamp_npy(
                output_root=task.output_root,
                case_id=task.case.case_id,
                exposure_s=task.case.exposure_s,
                stamp=stamp,
                star_id=star_id,
                frame_id=frame_id,
                stamp_size=task.case.stamp_size,
                seed=seed,
                write_mode=active_write_mode,
            )
            records.append(record)
            n_stamps += 1
            if record.status == "completed":
                n_written += 1
                output_bytes += int(record.file_size_bytes)
            elif record.status in {"skipped", "skipped_existing"}:
                n_skipped += 1
            else:
                n_failed += 1
            if active_write_mode == "sample" and record.status in {"completed", "skipped_existing"}:
                n_sample_outputs += 1
                if n_sample_outputs >= int(task.sample_limit):
                    active_write_mode = "none"
    manifest_path = (
        Path(task.output_root).expanduser()
        / task.case.case_id
        / "manifests"
        / f"manifest.worker{int(task.worker_rank):03d}.csv"
    )
    write_manifest(manifest_path, records)
    return WorkerResult(
        worker_rank=int(task.worker_rank),
        n_stamps=int(n_stamps),
        n_written=int(n_written),
        n_skipped=int(n_skipped),
        n_failed=int(n_failed),
        output_bytes=int(output_bytes),
        elapsed_s=float(time.perf_counter() - start),
        manifest_path=str(manifest_path),
        n_records=len(records),
        output_format="npy",
    )


def _git_commit(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def environment_metadata() -> dict[str, Any]:
    metadata = {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "cwd": os.getcwd(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "et_root": os.environ.get("ET_ROOT"),
        "photsim7_root": os.environ.get("PHOTSIM7_ROOT"),
        "et_data_dir": os.environ.get("ET_DATA_DIR"),
        "photsim7_data_dir": os.environ.get("PHOTSIM7_DATA_DIR"),
        "git_commit": _git_commit(Path(__file__).resolve().parents[1]),
    }
    try:
        import torch

        metadata["torch_version"] = torch.__version__
        metadata["torch_cuda_available"] = bool(torch.cuda.is_available())
        metadata["torch_cuda_device_count"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    except Exception as exc:
        metadata["torch_error"] = str(exc)
    return metadata


def write_json(path: Path | str, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return path


def _parse_gpu_ids(gpus: str) -> list[str]:
    return [token.strip() for token in str(gpus).split(",") if token.strip()]


def _process_pool_executor_kwargs(
    *,
    device_mode: str,
    gpu_ids: Sequence[str],
) -> dict[str, Any]:
    if gpu_ids and str(device_mode).strip().lower() != "cpu":
        return {"mp_context": mp.get_context("spawn")}
    return {}


def run_case(
    case: BenchmarkCase,
    *,
    output_root: Path | str,
    workers_per_gpu: int,
    gpus: str,
    global_seed: int,
    write_mode: str | None = None,
    output_format: str = "npy",
    dry_run: bool = False,
    device_mode: str = "auto",
    sample_limit: int = 1,
    render_options: RenderOptions | None = None,
) -> dict[str, Any]:
    output_root = Path(output_root).expanduser()
    case_dir = output_root / case.case_id
    case_write_mode = case.write_mode if write_mode is None else str(write_mode)
    if case_write_mode not in {"all", "none", "sample"}:
        raise ValueError(f"Unsupported write_mode {case_write_mode!r}")
    normalized_output_format = _normalize_output_format(output_format)
    if normalized_output_format == "hdf5" and case_write_mode != "all":
        raise ValueError(
            f"hdf5 output requires write_mode='all', got {case_write_mode!r}"
        )
    render_options = RenderOptions() if render_options is None else render_options
    resource_metadata = _resource_metadata(render_options)
    gpu_ids = _parse_gpu_ids(gpus)
    if case.gpus > 0:
        gpu_ids = gpu_ids[: int(case.gpus)] or ["0"]
        world_size = max(1, len(gpu_ids) * int(workers_per_gpu))
    else:
        gpu_ids = []
        world_size = max(1, int(workers_per_gpu))
    ranges = split_ranges(case.n_stars, world_size)
    expected_files = int(case.n_stars) * int(case.n_frames)
    expected_items = expected_files
    active_worker_count = sum(item.stop > item.start for item in ranges)
    expected_shards = (
        int(active_worker_count) if normalized_output_format == "hdf5" else 0
    )
    expected_output_files = (
        expected_shards + 1
        if normalized_output_format == "hdf5"
        else expected_files
    )
    expected_payload_bytes = expected_files * int(case.stamp_size) * int(case.stamp_size) * 4
    if dry_run:
        return {
            "case": asdict(case),
            "output_root": str(output_root),
            "write_mode": case_write_mode,
            "output_format": normalized_output_format,
            "gpu_ids": gpu_ids,
            "workers_per_gpu": int(workers_per_gpu),
            "world_size": int(world_size),
            "star_ranges": [asdict(item) for item in ranges],
            "estimated_stamps": int(case.n_stars) * int(case.n_frames),
            "expected_files": int(expected_files),
            "expected_items": int(expected_items),
            "expected_shards": int(expected_shards),
            "expected_output_files": int(expected_output_files),
            "expected_payload_bytes": int(expected_payload_bytes),
            "render_options": asdict(render_options),
            "resource_paths": resource_metadata,
        }

    physics_metadata = _case_physics_metadata(
        case,
        render_options=render_options,
        global_seed=int(global_seed),
    )
    write_json(case_dir / "environment.json", environment_metadata())
    write_json(
        case_dir / "case_config.json",
        {
            "case": asdict(case),
            "write_mode": case_write_mode,
            "output_format": normalized_output_format,
            "gpu_ids": gpu_ids,
            "workers_per_gpu": int(workers_per_gpu),
            "global_seed": int(global_seed),
            "device_mode": str(device_mode),
            "render_options": asdict(render_options),
            "resource_paths": resource_metadata,
            "physics_metadata": physics_metadata,
        },
    )

    tasks = [
        WorkerTask(
            case=case,
            output_root=str(output_root),
            star_range=star_range,
            worker_rank=rank,
            world_size=world_size,
            gpu_id=(gpu_ids[rank % len(gpu_ids)] if gpu_ids else None),
            global_seed=int(global_seed),
            write_mode=case_write_mode,
            output_format=normalized_output_format,
            device_mode=str(device_mode),
            sample_limit=int(sample_limit),
            render_options=render_options,
        )
        for rank, star_range in enumerate(ranges)
        if star_range.stop > star_range.start
    ]

    started = time.perf_counter()
    worker_payloads: list[dict[str, Any]] = []
    if len(tasks) == 1:
        results = [_run_worker_task(tasks[0])]
    else:
        executor_kwargs = _process_pool_executor_kwargs(
            device_mode=str(device_mode),
            gpu_ids=gpu_ids,
        )
        with ProcessPoolExecutor(max_workers=len(tasks), **executor_kwargs) as executor:
            futures = [executor.submit(_run_worker_task, task) for task in tasks]
            results = [future.result() for future in as_completed(futures)]
    ordered_results = sorted(results, key=lambda item: item.worker_rank)
    for result in ordered_results:
        worker_payloads.append(asdict(result))

    elapsed = time.perf_counter() - started
    n_stamps = sum(result.n_stamps for result in results)
    n_written = sum(result.n_written for result in results)
    n_skipped = sum(result.n_skipped for result in results)
    n_failed = sum(result.n_failed for result in results)
    n_shards = sum(result.n_shards for result in results)
    output_bytes = sum(result.output_bytes for result in results)
    throughput = float(n_stamps) / elapsed if elapsed > 0 else math.nan
    files_per_s = (
        float(n_shards if normalized_output_format == "hdf5" else n_written) / elapsed
        if elapsed > 0
        else math.nan
    )
    manifest_index_path: Path | None = None
    artifact_index_path: Path | None = None
    manifest_shards: list[dict[str, Any]] = []
    artifact_shards: list[dict[str, Any]] = []
    if normalized_output_format == "hdf5":
        ensure_photsim7_imports()
        from photsim7.artifacts import write_shard_index

        artifact_index_path = case_dir / "shard_index.json"
        write_shard_index(
            artifact_index_path,
            run_id=str(case.case_id),
            shard_paths=[result.artifact_path for result in ordered_results],
            metadata={
                "producer": "ET-mainsim.stamp_long",
                "case_id": str(case.case_id),
                "output_format": "hdf5",
                "expected_items": int(expected_items),
                "worker_count": len(ordered_results),
            },
        )
        artifact_shards = [
            {
                "worker_rank": int(result.worker_rank),
                "path": str(Path(result.artifact_path).relative_to(case_dir)),
                "items": int(result.n_stamps),
                "bytes": int(result.output_bytes),
            }
            for result in ordered_results
        ]
    else:
        manifest_index_path = case_dir / "manifest_index.json"
        manifest_shards = [
            {
                "worker_rank": int(result.worker_rank),
                "path": str(result.manifest_path),
                "records": int(result.n_records),
            }
            for result in ordered_results
        ]
        write_json(
            manifest_index_path,
            {
                "case_id": str(case.case_id),
                "expected_files": int(expected_files),
                "manifest_shards": manifest_shards,
            },
        )
    summary = {
        "case": asdict(case),
        "write_mode": case_write_mode,
        "output_format": normalized_output_format,
        "elapsed_s": float(elapsed),
        "expected_files": int(expected_files),
        "expected_items": int(expected_items),
        "expected_shards": int(expected_shards),
        "expected_output_files": int(expected_output_files),
        "expected_payload_bytes": int(expected_payload_bytes),
        "n_stamps": int(n_stamps),
        "n_written": int(n_written),
        "n_skipped": int(n_skipped),
        "n_failed": int(n_failed),
        "n_shards": int(n_shards),
        "output_bytes": int(output_bytes),
        "output_star_stamp_per_s": float(throughput),
        "files_per_s": float(files_per_s),
        "shards_per_s": (
            float(n_shards) / elapsed if elapsed > 0 else math.nan
        ),
        "pixel_per_s": float(throughput * case.stamp_size * case.stamp_size),
        "manifest_index_path": (
            None if manifest_index_path is None else str(manifest_index_path)
        ),
        "artifact_index_path": (
            None if artifact_index_path is None else str(artifact_index_path)
        ),
        "manifest_shards": manifest_shards,
        "artifact_shards": artifact_shards,
        "workers": worker_payloads,
        "render_options": asdict(render_options),
        "resource_paths": resource_metadata,
        "physics_metadata": physics_metadata,
    }
    write_json(case_dir / "summary.json", summary)
    return summary


_SCALE_GROUP_PREFIXES = {
    "short_high_star": "S1D",
    "long_low_star": "L7D",
}


def _filter_cases(
    cases: Sequence[BenchmarkCase],
    *,
    case_ids: Sequence[str] | None,
    exposures: Sequence[float] | None,
    stamp_sizes: Sequence[int] | None,
    matrix_preset: str | None,
    scale_groups: Sequence[str] | None,
    max_cases: int | None,
) -> list[BenchmarkCase]:
    filtered = list(cases)
    if matrix_preset not in (None, "", "stamp_scale_v2"):
        raise ValueError(f"Unknown matrix_preset {matrix_preset!r}")
    if scale_groups:
        normalized = {str(group).strip() for group in scale_groups if str(group).strip()}
        if "all" not in normalized:
            unknown = sorted(normalized - set(_SCALE_GROUP_PREFIXES))
            if unknown:
                raise ValueError(f"Unknown scale_group values: {unknown}")
            prefixes = tuple(_SCALE_GROUP_PREFIXES[group] for group in sorted(normalized))
            filtered = [case for case in filtered if case.case_id.startswith(prefixes)]
    if case_ids:
        wanted = {str(case_id) for case_id in case_ids}
        filtered = [case for case in filtered if case.case_id in wanted]
    if exposures:
        wanted_exp = {float(value) for value in exposures}
        filtered = [case for case in filtered if float(case.exposure_s) in wanted_exp]
    if stamp_sizes:
        wanted_stamp = {int(value) for value in stamp_sizes}
        filtered = [case for case in filtered if int(case.stamp_size) in wanted_stamp]
    if max_cases is not None:
        filtered = filtered[: int(max_cases)]
    return filtered


def run_stage(
    stage: str,
    *,
    output_root: Path | str,
    workers_per_gpu: int,
    gpus: str,
    global_seed: int,
    write_mode: str | None = None,
    output_format: str = "npy",
    dry_run: bool = False,
    device_mode: str = "auto",
    sample_limit: int = 1,
    render_options: RenderOptions | None = None,
    case_ids: Sequence[str] | None = None,
    exposures: Sequence[float] | None = None,
    stamp_sizes: Sequence[int] | None = None,
    matrix_preset: str | None = None,
    scale_groups: Sequence[str] | None = None,
    max_cases: int | None = None,
) -> list[dict[str, Any]]:
    cases = _filter_cases(
        benchmark_cases(stage),
        case_ids=case_ids,
        exposures=exposures,
        stamp_sizes=stamp_sizes,
        matrix_preset=matrix_preset,
        scale_groups=scale_groups,
        max_cases=max_cases,
    )
    results = []
    for case in cases:
        result = run_case(
            case,
            output_root=output_root,
            workers_per_gpu=workers_per_gpu,
            gpus=gpus,
            global_seed=global_seed,
            write_mode=write_mode,
            output_format=output_format,
            dry_run=dry_run,
            device_mode=device_mode,
            sample_limit=sample_limit,
            render_options=render_options,
        )
        results.append(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return results


def _write_csv_rows(path: Path | str, rows: Sequence[dict[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _jitter_experiment_device(device_mode: str) -> str:
    requested = "cuda:0" if str(device_mode) == "cuda" else str(device_mode)
    resolved = _resolve_requested_device(requested)
    if resolved.startswith("cuda"):
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise RuntimeError("CUDA device requested but torch is not installed") from exc
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device requested ({resolved}) but torch.cuda.is_available() is False"
            )
    return resolved


def _jitter_case_to_benchmark(case: JitterSensitivityCase) -> BenchmarkCase:
    return BenchmarkCase(
        str(case.case_id),
        "jitter_sensitivity",
        1,
        float(case.exposure_s),
        1,
        int(case.stamp_size),
        "none",
        1,
        str(case.description),
    )


def _build_renderer_for_jitter_variant(
    *,
    case: JitterSensitivityCase,
    variant: JitterBankVariant,
    xy_jitter_pix: np.ndarray,
    jitter_metadata: dict[str, Any],
    render_options: RenderOptions,
    device: str,
):
    return _build_photsim7_stamp_renderer_from_xy_jitter(
        stamp_size=int(case.stamp_size),
        exposure_s=float(case.exposure_s),
        psf_bundle_name=str(render_options.psf_bundle_name),
        psf_field_id=int(render_options.psf_field_id),
        psf_subpixels=int(render_options.psf_subpixels),
        device=str(device),
        integrate_jitter=True,
        jitter_integrated_psf_models=int(variant.n_models),
        jitter_frames_per_model=int(variant.n_frames_per_model),
        xy_jitter_pix=xy_jitter_pix,
        jitter_metadata=jitter_metadata,
    )


def run_jitter_sensitivity_case(
    case: JitterSensitivityCase,
    *,
    output_root: Path | str,
    variants: Sequence[JitterBankVariant],
    global_seed: int,
    device_mode: str,
    render_options: RenderOptions | None = None,
    star_id: int = 0,
    frame_id: int = 0,
    model_samples: int = 3,
    save_arrays: bool = True,
) -> dict[str, Any]:
    output_root = Path(output_root).expanduser()
    case_dir = output_root / str(case.case_id)
    case_dir.mkdir(parents=True, exist_ok=True)
    arrays_dir = case_dir / "arrays"
    render_options = RenderOptions() if render_options is None else render_options
    variants = list(variants)
    master_variant = _master_jitter_variant(variants)
    device = _jitter_experiment_device(device_mode)

    benchmark_case = _jitter_case_to_benchmark(case)
    seed = derive_seed(
        int(global_seed),
        exposure_s=float(case.exposure_s),
        frame_id=int(frame_id),
        star_id=int(star_id),
        effect_type="stamp",
    )
    base_config = _build_render_config(
        benchmark_case,
        seed=seed,
        global_seed=int(global_seed),
        star_id=int(star_id),
        frame_id=int(frame_id),
        device=device,
        render_options=render_options,
    )
    reference_config = replace(
        base_config,
        jitter_integrated_psf_models=int(master_variant.n_models),
        jitter_frames_per_model=int(master_variant.n_frames_per_model),
        device=device,
    )
    psd_path = _optional_resolved_path(base_config.psd_motion_path)
    if psd_path is None:
        raise ValueError("Jitter sensitivity experiment requires a PSD motion path")
    jitter_seed = derive_seed(
        int(global_seed),
        exposure_s=float(case.exposure_s),
        frame_id=0,
        star_id=0,
        effect_type="jitter_integrated_psf",
    )

    main_rd_core = _main_rd_core_module()
    master_start = time.perf_counter()
    master_xy_jitter_pix, master_jitter_metadata = main_rd_core.jitter_integrated_psf_offsets(
        seed=int(jitter_seed),
        enable_psd_motion=True,
        enable_jitter_integrated_psf=True,
        psd_motion_path=psd_path,
        n_models=int(master_variant.n_models),
        n_frames_per_model=int(master_variant.n_frames_per_model),
        exposure_s=float(case.exposure_s),
    )
    master_generation_time_s = float(time.perf_counter() - master_start)

    write_json(
        case_dir / "case_config.json",
        {
            "case": asdict(case),
            "variants": [asdict(variant) | {"variant_id": variant.variant_id} for variant in variants],
            "master_variant": asdict(master_variant) | {"variant_id": master_variant.variant_id},
            "global_seed": int(global_seed),
            "stamp_seed": int(seed),
            "jitter_seed": int(jitter_seed),
            "device": str(device),
            "render_options": asdict(render_options),
            "resource_paths": _resource_metadata(render_options),
            "comparison": "deterministic expected stellar stamp; stochastic photon/read/background noise disabled",
        },
    )
    write_json(case_dir / "environment.json", environment_metadata())

    metrics_rows: list[dict[str, Any]] = []
    variant_rows: list[dict[str, Any]] = []
    reference_cache: dict[int, np.ndarray] = {}
    reference_renderer = None
    reference_build_time_s = math.nan

    for variant in variants:
        derived_xy, derive_metadata = derive_jitter_bank_variant(
            master_xy_jitter_pix,
            variant,
        )
        variant_config = replace(
            base_config,
            jitter_integrated_psf_models=int(variant.n_models),
            jitter_frames_per_model=int(variant.n_frames_per_model),
            device=device,
        )
        variant_jitter_metadata = dict(master_jitter_metadata)
        variant_jitter_metadata.update(
            {
                "derived_from_master_variant": master_variant.variant_id,
                "derive_metadata": derive_metadata,
            }
        )
        build_start = time.perf_counter()
        renderer = _build_renderer_for_jitter_variant(
            case=case,
            variant=variant,
            xy_jitter_pix=derived_xy,
            jitter_metadata=variant_jitter_metadata,
            render_options=render_options,
            device=device,
        )
        build_time_s = float(time.perf_counter() - build_start)
        if variant == master_variant:
            reference_renderer = renderer
            reference_build_time_s = build_time_s

        model_indices = jitter_comparison_model_indices(
            int(variant.n_models),
            max_samples=int(model_samples),
        )
        render_time_s = 0.0
        variant_metric_rows: list[dict[str, Any]] = []
        for model_index in model_indices:
            if model_index not in reference_cache:
                if reference_renderer is None:
                    ref_xy, ref_metadata = derive_jitter_bank_variant(
                        master_xy_jitter_pix,
                        master_variant,
                    )
                    ref_metadata["derived_from_master_variant"] = master_variant.variant_id
                    ref_build_start = time.perf_counter()
                    reference_renderer = _build_renderer_for_jitter_variant(
                        case=case,
                        variant=master_variant,
                        xy_jitter_pix=ref_xy,
                        jitter_metadata=dict(master_jitter_metadata) | {"derive_metadata": ref_metadata},
                        render_options=render_options,
                        device=device,
                    )
                    reference_build_time_s = float(time.perf_counter() - ref_build_start)
                reference_cache[model_index] = _render_expected_photsim7_stamp_with_renderer(
                    renderer=reference_renderer,
                    config=reference_config,
                    device=device,
                    jitter_model_index=int(model_index),
                )
            reference_stamp = reference_cache[model_index]
            render_start = time.perf_counter()
            candidate_stamp = _render_expected_photsim7_stamp_with_renderer(
                renderer=renderer,
                config=variant_config,
                device=device,
                jitter_model_index=int(model_index),
            )
            render_time_s += float(time.perf_counter() - render_start)
            row = {
                "case_id": str(case.case_id),
                "variant_id": variant.variant_id,
                "n_models": int(variant.n_models),
                "n_frames_per_model": int(variant.n_frames_per_model),
                "model_index": int(model_index),
                "master_variant_id": master_variant.variant_id,
                "master_generation_time_s": float(master_generation_time_s),
                "build_time_s": float(build_time_s),
                "reference_build_time_s": float(reference_build_time_s),
                "render_time_s": float(render_time_s),
            }
            row.update(stamp_difference_metrics(candidate_stamp, reference_stamp))
            metrics_rows.append(row)
            variant_metric_rows.append(row)
            if save_arrays and model_index == model_indices[0]:
                arrays_dir.mkdir(parents=True, exist_ok=True)
                np.save(
                    arrays_dir / f"reference_{master_variant.variant_id}_model{model_index:03d}.npy",
                    reference_stamp.astype(np.float32, copy=False),
                )
                np.save(
                    arrays_dir / f"{variant.variant_id}_model{model_index:03d}.npy",
                    candidate_stamp.astype(np.float32, copy=False),
                )
                np.save(
                    arrays_dir / f"diff_{variant.variant_id}_model{model_index:03d}.npy",
                    (candidate_stamp - reference_stamp).astype(np.float32, copy=False),
                )

        def _mean(name: str) -> float:
            return float(np.mean([float(row[name]) for row in variant_metric_rows]))

        variant_rows.append(
            {
                "case_id": str(case.case_id),
                "variant_id": variant.variant_id,
                "n_models": int(variant.n_models),
                "n_frames_per_model": int(variant.n_frames_per_model),
                "model_indices": json.dumps(model_indices),
                "master_generation_time_s": float(master_generation_time_s),
                "build_time_s": float(build_time_s),
                "render_time_s": float(render_time_s),
                "mean_relative_l2": _mean("relative_l2"),
                "mean_rms_diff": _mean("rms_diff"),
                "mean_flux_delta_fraction": _mean("flux_delta_fraction"),
                "mean_centroid_shift_pix": _mean("centroid_shift_pix"),
                "mean_second_moment_radius_delta_pix": _mean("second_moment_radius_delta_pix"),
            }
        )

    _write_csv_rows(case_dir / "metrics.csv", metrics_rows)
    _write_csv_rows(case_dir / "variants.csv", variant_rows)
    summary = {
        "case": asdict(case),
        "output_root": str(output_root),
        "case_dir": str(case_dir),
        "device": str(device),
        "global_seed": int(global_seed),
        "stamp_seed": int(seed),
        "jitter_seed": int(jitter_seed),
        "master_variant": asdict(master_variant) | {"variant_id": master_variant.variant_id},
        "master_generation_time_s": float(master_generation_time_s),
        "variants": variant_rows,
        "metrics_csv": str(case_dir / "metrics.csv"),
        "variants_csv": str(case_dir / "variants.csv"),
        "arrays_dir": str(arrays_dir) if save_arrays else None,
    }
    write_json(case_dir / "summary.json", summary)
    return summary


def run_jitter_sensitivity(
    *,
    output_root: Path | str,
    variants: Sequence[JitterBankVariant] | None = None,
    cases: Sequence[JitterSensitivityCase] | None = None,
    global_seed: int,
    device_mode: str,
    render_options: RenderOptions | None = None,
    dry_run: bool = False,
    model_samples: int = 3,
    save_arrays: bool = True,
) -> dict[str, Any]:
    output_root = Path(output_root).expanduser()
    variants = parse_jitter_bank_variants(None) if variants is None else list(variants)
    cases = jitter_sensitivity_cases(None) if cases is None else list(cases)
    master_variant = _master_jitter_variant(variants)
    if dry_run:
        return {
            "dry_run": True,
            "output_root": str(output_root),
            "cases": [asdict(case) for case in cases],
            "variants": [asdict(variant) | {"variant_id": variant.variant_id} for variant in variants],
            "master_variant": asdict(master_variant) | {"variant_id": master_variant.variant_id},
            "global_seed": int(global_seed),
            "device_mode": str(device_mode),
            "model_samples": int(model_samples),
            "comparison": "deterministic expected stellar stamp; variants derived from one master jitter bank",
        }

    case_summaries = [
        run_jitter_sensitivity_case(
            case,
            output_root=output_root,
            variants=variants,
            global_seed=int(global_seed),
            device_mode=str(device_mode),
            render_options=render_options,
            model_samples=int(model_samples),
            save_arrays=bool(save_arrays),
        )
        for case in cases
    ]
    summary = {
        "dry_run": False,
        "output_root": str(output_root),
        "cases": [item["case"] for item in case_summaries],
        "case_summaries": case_summaries,
        "variants": [asdict(variant) | {"variant_id": variant.variant_id} for variant in variants],
        "master_variant": asdict(master_variant) | {"variant_id": master_variant.variant_id},
        "global_seed": int(global_seed),
        "device_mode": str(device_mode),
        "model_samples": int(model_samples),
    }
    write_json(output_root / "summary.json", summary)
    return summary


def _parse_csv_floats(raw: str | None) -> list[float] | None:
    if raw is None or str(raw).strip() == "":
        return None
    return [float(token.strip()) for token in str(raw).split(",") if token.strip()]


def _parse_csv_ints(raw: str | None) -> list[int] | None:
    if raw is None or str(raw).strip() == "":
        return None
    return [int(token.strip()) for token in str(raw).split(",") if token.strip()]


def _parse_csv_strings(raw: str | None) -> list[str] | None:
    if raw is None or str(raw).strip() == "":
        return None
    return [token.strip() for token in str(raw).split(",") if token.strip()]


def build_arg_parser(stage: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Run stamp_long {stage} benchmark cases")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--workers-per-gpu", type=int, default=10)
    parser.add_argument("--gpus", type=str, default="0,1,2")
    parser.add_argument("--seed", type=int, default=20260617)
    parser.add_argument("--write-mode", choices=["all", "none", "sample"], default=None)
    parser.add_argument(
        "--output-format",
        choices=["npy", "hdf5"],
        default=os.environ.get("OUTPUT_FORMAT", "npy"),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--sample-limit", type=int, default=1)
    parser.add_argument("--case-ids", type=str, default=None)
    parser.add_argument("--exposures", type=str, default=None)
    parser.add_argument("--stamp-sizes", type=str, default=None)
    parser.add_argument("--matrix-preset", type=str, default=os.environ.get("MATRIX_PRESET"))
    parser.add_argument("--scale-group", type=str, default=os.environ.get("SCALE_GROUP"))
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--star-flux-e-s", type=float, default=DEFAULT_STAR_FLUX_E_S)
    parser.add_argument(
        "--star-flux-mode",
        choices=["fixed", "random_et_mag", "random-et-mag"],
        default=os.environ.get("STAR_FLUX_MODE", DEFAULT_STAR_FLUX_MODE),
    )
    parser.add_argument("--et-mag-min", type=float, default=DEFAULT_ET_MAG_MIN)
    parser.add_argument("--et-mag-max", type=float, default=DEFAULT_ET_MAG_MAX)
    parser.add_argument("--background-e-s-pix", type=float, default=DEFAULT_BACKGROUND_E_S_PIX)
    parser.add_argument("--scattered-light-e-s-pix", type=float, default=DEFAULT_SCATTERED_LIGHT_E_S_PIX)
    parser.add_argument("--dark-e-s-pix", type=float, default=DEFAULT_DARK_E_S_PIX)
    parser.add_argument("--read-noise-10s-e-pix", type=float, default=DEFAULT_READ_NOISE_E_PIX)
    parser.add_argument("--gain-e-per-adu", type=float, default=DEFAULT_GAIN_E_PER_ADU)
    parser.add_argument("--cosmic-ray-event-rate", type=float, default=DEFAULT_COSMIC_RAY_EVENT_RATE_CM2_S)
    parser.add_argument(
        "--cosmic-ray-library",
        type=str,
        default=(
            os.environ.get("COSMIC_RAY_LIBRARY")
            or os.environ.get("PHOTSIM7_COSMIC_RAY_LIBRARY")
            or DEFAULT_COSMIC_RAY_LIBRARY_PATH
        ),
    )
    parser.add_argument("--cosmic-ray-peak-adu", type=float, default=DEFAULT_COSMIC_RAY_PEAK_ADU)
    parser.add_argument("--pixel-size-um", type=float, default=DEFAULT_PIXEL_SIZE_UM)
    parser.add_argument("--psf-sigma-pix", type=float, default=DEFAULT_PSF_SIGMA_PIX)
    parser.add_argument("--psf-bundle-name", type=str, default=DEFAULT_PSF_BUNDLE_NAME)
    parser.add_argument("--psf-field-id", type=int, default=DEFAULT_PSF_FIELD_ID)
    parser.add_argument("--psf-subpixels", type=int, default=DEFAULT_PSF_SUBPIXELS)
    parser.add_argument("--photsim7-psf", dest="use_photsim7_psf", action="store_true", default=True)
    parser.add_argument("--no-photsim7-psf", dest="use_photsim7_psf", action="store_false")
    parser.add_argument("--psd-motion-path", type=str, default=DEFAULT_PSD_MOTION_PATH)
    parser.add_argument("--dva-model-path", type=str, default=DEFAULT_DVA_MODEL_PATH)
    parser.add_argument("--jitter-psf-models", type=int, default=DEFAULT_JITTER_INTEGRATED_PSF_MODELS)
    parser.add_argument("--jitter-frames-per-model", type=int, default=DEFAULT_JITTER_FRAMES_PER_MODEL)
    parser.add_argument("--detector-response", dest="enable_detector_response", action="store_true", default=True)
    parser.add_argument("--no-detector-response", dest="enable_detector_response", action="store_false")
    parser.add_argument("--response-padding-pix", type=int, default=DEFAULT_RESPONSE_PADDING_PIX)
    parser.add_argument("--inter-pixel-response-sigma", type=float, default=DEFAULT_INTER_PIXEL_RESPONSE_SIGMA)
    parser.add_argument("--inter-pixel-response-nominal", type=float, default=DEFAULT_INTER_PIXEL_RESPONSE_NOMINAL)
    parser.add_argument("--intra-pixel-response-sigma", type=float, default=DEFAULT_INTRA_PIXEL_RESPONSE_SIGMA)
    parser.add_argument("--inter-pixel-response", dest="enable_inter_pixel_response", action="store_true", default=True)
    parser.add_argument("--no-inter-pixel-response", dest="enable_inter_pixel_response", action="store_false")
    parser.add_argument("--intra-pixel-response", dest="enable_intra_pixel_response", action="store_true", default=True)
    parser.add_argument("--no-intra-pixel-response", dest="enable_intra_pixel_response", action="store_false")
    parser.add_argument("--pixel-phase-response", dest="enable_pixel_phase_response", action="store_true", default=True)
    parser.add_argument("--no-pixel-phase-response", dest="enable_pixel_phase_response", action="store_false")
    parser.add_argument("--dynamic-effects", dest="enable_dynamic_effects", action="store_true", default=True)
    parser.add_argument("--no-dynamic-effects", dest="enable_dynamic_effects", action="store_false")
    parser.add_argument("--psd-motion", dest="enable_psd_motion", action="store_true", default=True)
    parser.add_argument("--no-psd-motion", dest="enable_psd_motion", action="store_false")
    parser.add_argument("--dva-drift", dest="enable_dva_drift", action="store_true", default=True)
    parser.add_argument("--no-dva-drift", dest="enable_dva_drift", action="store_false")
    parser.add_argument("--thermal-drift", dest="enable_thermal_drift", action="store_true", default=True)
    parser.add_argument("--no-thermal-drift", dest="enable_thermal_drift", action="store_false")
    parser.add_argument("--momentum-dump", dest="enable_momentum_dump", action="store_true", default=True)
    parser.add_argument("--no-momentum-dump", dest="enable_momentum_dump", action="store_false")
    parser.add_argument("--psf-breathing", dest="enable_psf_breathing", action="store_true", default=True)
    parser.add_argument("--no-psf-breathing", dest="enable_psf_breathing", action="store_false")
    return parser


def _render_options_from_args(args: argparse.Namespace) -> RenderOptions:
    return RenderOptions(
        star_flux_e_s=float(args.star_flux_e_s),
        star_flux_mode=_normalize_star_flux_mode(args.star_flux_mode),
        et_mag_min=float(args.et_mag_min),
        et_mag_max=float(args.et_mag_max),
        background_e_s_pix=float(args.background_e_s_pix),
        scattered_light_e_s_pix=float(args.scattered_light_e_s_pix),
        dark_e_s_pix=float(args.dark_e_s_pix),
        read_noise_10s_e_pix=float(args.read_noise_10s_e_pix),
        gain_e_per_adu=float(args.gain_e_per_adu),
        cosmic_ray_event_rate=float(args.cosmic_ray_event_rate),
        cosmic_ray_library_path=args.cosmic_ray_library,
        cosmic_ray_peak_adu=float(args.cosmic_ray_peak_adu),
        pixel_size_um=float(args.pixel_size_um),
        psf_sigma_pix=float(args.psf_sigma_pix),
        psf_bundle_name=str(args.psf_bundle_name),
        psf_field_id=int(args.psf_field_id),
        psf_subpixels=int(args.psf_subpixels),
        use_photsim7_psf=bool(args.use_photsim7_psf),
        psd_motion_path=args.psd_motion_path,
        dva_model_path=args.dva_model_path,
        jitter_integrated_psf_models=int(args.jitter_psf_models),
        jitter_frames_per_model=int(args.jitter_frames_per_model),
        enable_detector_response=bool(args.enable_detector_response),
        response_padding_pix=int(args.response_padding_pix),
        inter_pixel_response_sigma=float(args.inter_pixel_response_sigma),
        inter_pixel_response_nominal=float(args.inter_pixel_response_nominal),
        intra_pixel_response_sigma=float(args.intra_pixel_response_sigma),
        enable_inter_pixel_response=bool(args.enable_inter_pixel_response),
        enable_intra_pixel_response=bool(args.enable_intra_pixel_response),
        enable_pixel_phase_response=bool(args.enable_pixel_phase_response),
        enable_dynamic_effects=bool(args.enable_dynamic_effects),
        enable_psd_motion=bool(args.enable_psd_motion),
        enable_dva_drift=bool(args.enable_dva_drift),
        enable_thermal_drift=bool(args.enable_thermal_drift),
        enable_momentum_dump=bool(args.enable_momentum_dump),
        enable_psf_breathing=bool(args.enable_psf_breathing),
    )


def build_jitter_sensitivity_arg_parser() -> argparse.ArgumentParser:
    parser = build_arg_parser("jitter_sensitivity")
    parser.description = "Run stamp_long jitter-integrated PSF bank sensitivity experiment"
    parser.set_defaults(output_root=DEFAULT_JITTER_SENSITIVITY_OUTPUT_ROOT)
    parser.add_argument(
        "--variants",
        type=str,
        default=os.environ.get("JITTER_VARIANTS", "100x200,100x300,200x400,300x600"),
        help="Comma-separated jitter bank variants such as 100x200,100x300,200x400,300x600.",
    )
    parser.add_argument(
        "--cases",
        type=str,
        default=os.environ.get("JITTER_CASES"),
        help="Comma-separated jitter sensitivity case ids. Defaults to J030S11,J300S15.",
    )
    parser.add_argument(
        "--model-samples",
        type=int,
        default=int(os.environ.get("JITTER_MODEL_SAMPLES", "3")),
        help="Number of jitter model indices to compare per variant.",
    )
    parser.add_argument(
        "--save-arrays",
        dest="save_arrays",
        action="store_true",
        default=os.environ.get("SAVE_JITTER_ARRAYS", "1") != "0",
        help="Save representative reference/candidate/difference .npy arrays.",
    )
    parser.add_argument(
        "--no-save-arrays",
        dest="save_arrays",
        action="store_false",
        help="Do not save representative .npy arrays.",
    )
    return parser


def run_jitter_sensitivity_cli(argv: Sequence[str] | None = None) -> int:
    parser = build_jitter_sensitivity_arg_parser()
    args = parser.parse_args(argv)
    summary = run_jitter_sensitivity(
        output_root=args.output_root,
        variants=parse_jitter_bank_variants(args.variants),
        cases=jitter_sensitivity_cases(args.cases),
        global_seed=int(args.seed),
        device_mode=str(args.device),
        render_options=_render_options_from_args(args),
        dry_run=bool(args.dry_run),
        model_samples=int(args.model_samples),
        save_arrays=bool(args.save_arrays),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def run_cli(stage: str, argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser(stage)
    args = parser.parse_args(argv)
    render_options = _render_options_from_args(args)
    run_stage(
        stage,
        output_root=args.output_root,
        workers_per_gpu=args.workers_per_gpu,
        gpus=args.gpus,
        global_seed=args.seed,
        write_mode=args.write_mode,
        output_format=args.output_format,
        dry_run=bool(args.dry_run),
        device_mode=args.device,
        sample_limit=int(args.sample_limit),
        render_options=render_options,
        case_ids=_parse_csv_strings(args.case_ids),
        exposures=_parse_csv_floats(args.exposures),
        stamp_sizes=_parse_csv_ints(args.stamp_sizes),
        matrix_preset=args.matrix_preset,
        scale_groups=_parse_csv_strings(args.scale_group),
        max_cases=args.max_cases,
    )
    return 0
