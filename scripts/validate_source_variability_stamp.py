#!/usr/bin/env python3
"""Run a short, paired SN source-variability validation through ``et-stamp``.

The source CSV's ``time`` and ``observer_time`` values are deliberately ignored.
Rows are aligned to the simulator solely by zero-based raw ``frame_index``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, NamedTuple, Sequence

import numpy as np
from astropy.table import Table


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
if SOURCE_ROOT.is_dir() and str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


SOURCE_ID = 900_000_000_000_001
CURVE_ID = "sn_gaia_g_short_validation"
TARGET_RA_DEG = 304.41406499712303
TARGET_DEC_DEG = 51.81987707392268
SOURCE_BAND = "gaia_g_3260_9290"
SOURCE_ZPSYS = "ab"
MAGNITUDE_SEMANTICS_NOTE = (
    "truncated_gaia_g_ab_treated_as_gaia_g_vega_engineering_proxy"
)
RELATIVE_FLUX_FORMULA = (
    "10**(-0.4*(mag_clean-min(selected_mag_clean)))"
)
TIME_ALIGNMENT = "simulation_raw_frame_index"
TIME_COLUMNS = ("time", "observer_time")


class SnLightCurve(NamedTuple):
    source_path: Path
    source_sha256: str
    source_size_bytes: int
    source_band: str
    source_zpsys: str
    magnitudes: np.ndarray
    baseline_gaia_g_mag: float
    relative_flux: np.ndarray
    frame_indices: tuple[int, ...]
    selected_source_rows: tuple[int, ...]
    ignored_input_time_columns: tuple[str, ...]
    total_source_rows: int


class ValidationInputPaths(NamedTuple):
    static_target: Path
    injected_target: Path
    variability: Path
    simulation_spec: Path


class RunCommands(NamedTuple):
    static: tuple[str, ...]
    injected: tuple[str, ...]


class ApertureMeasurement(NamedTuple):
    flux: float
    background_median: float
    aperture_pixel_count: int
    nan_pixel_count: int
    saturated_pixel_count: int


class RunPhotometry(NamedTuple):
    frame_indices: tuple[int, ...]
    final_flux: np.ndarray
    stellar_flux: np.ndarray
    nan_pixel_count: np.ndarray
    saturated_pixel_count: np.ndarray


class ValidationProducts(NamedTuple):
    metrics_json: Path
    per_frame_csv: Path
    plot_png: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _constant_text_column(table: Table, name: str, *, expected: str) -> str:
    if name not in table.colnames:
        raise ValueError(f"SN input requires a {name!r} column")
    values = {str(value).strip() for value in table[name]}
    if values != {expected}:
        raise ValueError(
            f"SN input {name!r} must contain only {expected!r}, got {sorted(values)!r}"
        )
    return expected


def load_sn_lightcurve(
    path: Path | str,
    *,
    frame_count: int = 22,
) -> SnLightCurve:
    """Load the first ``frame_count`` ``mag_clean`` values and derive flux factors.

    The source time axes are inspected only to record their names as ignored
    provenance. Their numeric values never participate in the conversion.
    """

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"SN input CSV does not exist: {source}")
    frame_count = int(frame_count)
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    table = Table.read(source, format="ascii.csv")
    if "mag_clean" not in table.colnames:
        raise ValueError("SN input requires the 'mag_clean' column")
    if len(table) < frame_count:
        raise ValueError(
            f"SN input has {len(table)} rows but frame_count={frame_count}"
        )
    band = _constant_text_column(table, "band", expected=SOURCE_BAND)
    zpsys = _constant_text_column(table, "zpsys", expected=SOURCE_ZPSYS)

    selected = table["mag_clean"][:frame_count]
    if np.ma.isMaskedArray(selected) and np.any(np.ma.getmaskarray(selected)):
        raise ValueError("selected mag_clean values must not be masked")
    try:
        magnitudes = np.asarray(selected, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("selected mag_clean values must be numeric") from exc
    if magnitudes.shape != (frame_count,) or not np.all(np.isfinite(magnitudes)):
        raise ValueError("selected mag_clean values must be finite")

    baseline = float(np.min(magnitudes))
    relative_flux = np.power(10.0, -0.4 * (magnitudes - baseline))
    if not np.all(np.isfinite(relative_flux)) or np.any(relative_flux < 0.0):
        raise ValueError("derived relative_flux values must be finite and non-negative")
    ignored = tuple(name for name in TIME_COLUMNS if name in table.colnames)
    return SnLightCurve(
        source_path=source,
        source_sha256=_sha256(source),
        source_size_bytes=int(source.stat().st_size),
        source_band=band,
        source_zpsys=zpsys,
        magnitudes=magnitudes,
        baseline_gaia_g_mag=baseline,
        relative_flux=relative_flux,
        frame_indices=tuple(range(frame_count)),
        selected_source_rows=tuple(range(frame_count)),
        ignored_input_time_columns=ignored,
        total_source_rows=len(table),
    )


def _validation_meta(curve: SnLightCurve) -> dict[str, Any]:
    frame_count = len(curve.frame_indices)
    return {
        "validation_schema_id": "et_mainsim.source_variability_sn_validation",
        "validation_schema_version": 1,
        "source_path": str(curve.source_path),
        "source_sha256": curve.source_sha256,
        "source_size_bytes": curve.source_size_bytes,
        "source_band": curve.source_band,
        "source_zpsys": curve.source_zpsys,
        "source_magnitude_column": "mag_clean",
        "magnitude_semantics": "Gaia_G_Vega",
        "magnitude_semantics_note": MAGNITUDE_SEMANTICS_NOTE,
        "selected_source_rows": list(curve.selected_source_rows),
        "selected_row_slice": f"0:{frame_count}",
        "source_row_count": curve.total_source_rows,
        "omitted_source_row_count": curve.total_source_rows - frame_count,
        "baseline_definition": "min(selected_mag_clean)",
        "baseline_gaia_g_mag": curve.baseline_gaia_g_mag,
        "relative_flux_formula": RELATIVE_FLUX_FORMULA,
        "time_alignment": TIME_ALIGNMENT,
        "ignored_input_time_columns": list(curve.ignored_input_time_columns),
        "input_time_values_used": False,
    }


def write_short_stamp_spec(
    path: Path | str,
    *,
    frame_count: int,
    coadd_size: int = 2,
) -> Path:
    """Write the production stamp spec with only its observation length shortened."""

    frame_count = int(frame_count)
    coadd_size = int(coadd_size)
    if frame_count <= 0 or coadd_size <= 0:
        raise ValueError("frame_count and coadd_size must be positive")
    if frame_count % coadd_size:
        raise ValueError("frame_count must be divisible by coadd_size")

    from et_mainsim.presets import load_preset

    base = load_preset("et-stamp-production").simulation_spec
    duration = frame_count * base.observation.sampling_interval
    observation = replace(
        base.observation,
        observing_duration=duration,
        n_frames=frame_count,
        n_raw_frames_per_coadd=coadd_size,
        frame_start_s=None,
    )
    spec = replace(base, observation=observation)
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(spec.to_json() + "\n", encoding="utf-8")
    return destination


def write_validation_inputs(
    curve: SnLightCurve,
    output_dir: Path | str,
    *,
    source_id: int = SOURCE_ID,
    curve_id: str = CURVE_ID,
    coadd_size: int | None = None,
) -> ValidationInputPaths:
    """Write paired target tables, the variability table, and one shared spec."""

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    source_id = int(source_id)
    curve_id = str(curve_id).strip()
    if source_id < 0:
        raise ValueError("source_id must be non-negative")
    if not curve_id:
        raise ValueError("curve_id must be non-empty")
    frame_count = len(curve.frame_indices)
    if coadd_size is None:
        coadd_size = 2 if frame_count % 2 == 0 else 1

    common_meta = _validation_meta(curve)
    injected = Table(
        {
            "source_id": np.array([source_id], dtype=np.int64),
            "gaia_g_mag": np.array([curve.baseline_gaia_g_mag]),
            "curve_id": np.array([curve_id]),
            "ra_deg": np.array([TARGET_RA_DEG]),
            "dec_deg": np.array([TARGET_DEC_DEG]),
        }
    )
    injected.meta = {**common_meta, "injection_mode": "intrinsic_relative_flux"}
    static = Table(
        {
            "source_id": np.array([source_id], dtype=np.int64),
            "gaia_g_mag": np.array([curve.baseline_gaia_g_mag]),
            "ra_deg": np.array([TARGET_RA_DEG]),
            "dec_deg": np.array([TARGET_DEC_DEG]),
        }
    )
    static.meta = {**common_meta, "injection_mode": "static_control"}
    variability = Table(
        {
            "curve_id": np.full(frame_count, curve_id),
            "frame_index": np.asarray(curve.frame_indices, dtype=np.int64),
            "relative_flux": np.asarray(curve.relative_flux, dtype=np.float64),
        }
    )
    variability.meta = {
        **common_meta,
        "injection_mode": "intrinsic_relative_flux",
        "curve_id": curve_id,
    }

    static_path = destination / "target_static.ecsv"
    injected_path = destination / "target_injected.ecsv"
    variability_path = destination / "source_variability.ecsv"
    spec_path = destination / "short_stamp.spec.json"
    static.write(static_path, format="ascii.ecsv", overwrite=True)
    injected.write(injected_path, format="ascii.ecsv", overwrite=True)
    variability.write(variability_path, format="ascii.ecsv", overwrite=True)
    write_short_stamp_spec(
        spec_path,
        frame_count=frame_count,
        coadd_size=coadd_size,
    )
    return ValidationInputPaths(
        static_target=static_path,
        injected_target=injected_path,
        variability=variability_path,
        simulation_spec=spec_path,
    )


def _shared_run_command(
    *,
    inputs: ValidationInputPaths,
    data_root: Path | str,
    focalplane_registry: Path | str,
    frame_count: int,
    run_id: str,
    seed: int,
    backend: str,
    device: str,
    gpu_ids: Sequence[str],
    workers_per_device: int,
    stamp_rows: int,
    stamp_cols: int,
) -> tuple[str, ...]:
    command = [
        sys.executable,
        "-m",
        "et_mainsim",
        "run",
        "et-stamp",
        "--preset",
        "production",
        "--run-id",
        str(run_id),
        "--data-root",
        str(data_root),
        "--focalplane-registry",
        str(focalplane_registry),
        "--spec",
        str(inputs.simulation_spec),
        "--frames",
        str(int(frame_count)),
        "--seed",
        str(int(seed)),
        "--backend",
        str(backend),
        "--device",
        str(device),
        "--workers-per-device",
        str(int(workers_per_device)),
        "--stamp-rows",
        str(int(stamp_rows)),
        "--stamp-cols",
        str(int(stamp_cols)),
        "--no-include-neighbors",
        "--save-raw",
        "--save-coadd",
        "--save-electron-components",
        "--overwrite",
    ]
    normalized_gpu_ids = tuple(
        str(value).strip() for value in gpu_ids if str(value).strip()
    )
    if normalized_gpu_ids:
        command.extend(("--gpus", ",".join(normalized_gpu_ids)))
    return tuple(command)


def build_run_commands(
    *,
    inputs: ValidationInputPaths,
    output_root: Path | str,
    data_root: Path | str,
    focalplane_registry: Path | str,
    frame_count: int = 22,
    run_id: str = "source-variability-sn-validation",
    seed: int = 20260715,
    backend: str = "local-subprocess",
    device: str = "cuda",
    gpu_ids: Sequence[str] = ("0",),
    workers_per_device: int = 1,
    stamp_rows: int = 15,
    stamp_cols: int = 15,
) -> RunCommands:
    """Build paired commands differing only in inputs and output roots."""

    root = Path(output_root).expanduser().resolve()
    shared = _shared_run_command(
        inputs=inputs,
        data_root=data_root,
        focalplane_registry=focalplane_registry,
        frame_count=frame_count,
        run_id=run_id,
        seed=seed,
        backend=backend,
        device=device,
        gpu_ids=gpu_ids,
        workers_per_device=workers_per_device,
        stamp_rows=stamp_rows,
        stamp_cols=stamp_cols,
    )
    static = shared + (
        "--output-root",
        str(root / "static"),
        "--input-table",
        str(inputs.static_target),
    )
    injected = shared + (
        "--output-root",
        str(root / "injected"),
        "--input-table",
        str(inputs.injected_target),
        "--variability-table",
        str(inputs.variability),
    )
    return RunCommands(static=static, injected=injected)


def fixed_aperture_photometry(
    image: Any,
    *,
    aperture_radius: float = 4.0,
    annulus_inner: float = 5.5,
    annulus_outer: float = 7.0,
) -> ApertureMeasurement:
    """Measure one centered stamp with a fixed aperture and median annulus."""

    array = np.asarray(image)
    if array.ndim != 2 or min(array.shape) <= 0:
        raise ValueError("image must be a non-empty two-dimensional array")
    aperture_radius = float(aperture_radius)
    annulus_inner = float(annulus_inner)
    annulus_outer = float(annulus_outer)
    if not (0.0 < aperture_radius < annulus_inner < annulus_outer):
        raise ValueError(
            "radii must satisfy 0 < aperture_radius < annulus_inner < annulus_outer"
        )
    values = np.asarray(array, dtype=np.float64)
    yy, xx = np.indices(values.shape, dtype=np.float64)
    center_x = (values.shape[1] - 1) / 2.0
    center_y = (values.shape[0] - 1) / 2.0
    radius = np.hypot(xx - center_x, yy - center_y)
    aperture = radius <= aperture_radius
    annulus = (radius >= annulus_inner) & (radius <= annulus_outer)
    finite_annulus = values[annulus & np.isfinite(values)]
    if not np.any(aperture) or finite_annulus.size == 0:
        raise ValueError("stamp does not contain the requested aperture and annulus")
    background = float(np.median(finite_annulus))
    aperture_values = values[aperture]
    flux = float(np.nansum(aperture_values - background))
    nan_count = int(np.count_nonzero(~np.isfinite(values)))
    saturated_count = 0
    if np.issubdtype(array.dtype, np.integer):
        saturated_count = int(
            np.count_nonzero(array >= np.iinfo(array.dtype).max)
        )
    return ApertureMeasurement(
        flux=flux,
        background_median=background,
        aperture_pixel_count=int(np.count_nonzero(aperture)),
        nan_pixel_count=nan_count,
        saturated_pixel_count=saturated_count,
    )


def _one_dimensional(
    values: Any,
    *,
    name: str,
    length: int | None = None,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if length is not None and array.size != length:
        raise ValueError(f"{name} must have length {length}")
    return array


def compute_paired_metrics(
    *,
    relative_flux: Any,
    static_final_flux: Any,
    injected_final_flux: Any,
    static_stellar_flux: Any,
    injected_stellar_flux: Any,
    static_nan_pixel_count: Any,
    injected_nan_pixel_count: Any,
    static_saturated_pixel_count: Any,
    injected_saturated_pixel_count: Any,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Compute noiseless-ratio and final-DN paired photometric diagnostics."""

    factor = _one_dimensional(relative_flux, name="relative_flux")
    length = factor.size
    static_final = _one_dimensional(
        static_final_flux,
        name="static_final_flux",
        length=length,
    )
    injected_final = _one_dimensional(
        injected_final_flux,
        name="injected_final_flux",
        length=length,
    )
    static_stellar = _one_dimensional(
        static_stellar_flux,
        name="static_stellar_flux",
        length=length,
    )
    injected_stellar = _one_dimensional(
        injected_stellar_flux,
        name="injected_stellar_flux",
        length=length,
    )
    static_nan = np.asarray(static_nan_pixel_count, dtype=np.int64).reshape(-1)
    injected_nan = np.asarray(injected_nan_pixel_count, dtype=np.int64).reshape(-1)
    static_saturated = np.asarray(
        static_saturated_pixel_count,
        dtype=np.int64,
    ).reshape(-1)
    injected_saturated = np.asarray(
        injected_saturated_pixel_count,
        dtype=np.int64,
    ).reshape(-1)
    for name, values in (
        ("static_nan_pixel_count", static_nan),
        ("injected_nan_pixel_count", injected_nan),
        ("static_saturated_pixel_count", static_saturated),
        ("injected_saturated_pixel_count", injected_saturated),
    ):
        if values.size != length or np.any(values < 0):
            raise ValueError(f"{name} must contain {length} non-negative values")

    static_reference = float(np.nanmedian(static_final))
    if not np.isfinite(static_reference) or static_reference == 0.0:
        raise ValueError("static_final_flux must have a finite non-zero median")
    expected_residual = factor - 1.0
    paired_residual = (injected_final - static_final) / static_reference
    with np.errstate(divide="ignore", invalid="ignore"):
        stellar_ratio = injected_stellar / static_stellar
    residual_error = paired_residual - expected_residual
    stellar_ratio_error = stellar_ratio - factor
    static_normalized = static_final / static_reference
    injected_normalized = injected_final / static_reference

    valid = np.isfinite(
        np.column_stack(
            (
                factor,
                paired_residual,
                stellar_ratio,
                static_normalized,
                injected_normalized,
            )
        )
    ).all(axis=1)
    if np.count_nonzero(valid) < 2:
        raise ValueError("at least two finite paired frames are required")
    valid_expected = expected_residual[valid]
    valid_paired = paired_residual[valid]
    if np.ptp(valid_expected) == 0.0 or np.ptp(valid_paired) == 0.0:
        pearson = None
        slope = None
        intercept = None
    else:
        pearson_value = float(np.corrcoef(valid_expected, valid_paired)[0, 1])
        if abs(pearson_value - 1.0) < 1e-15:
            pearson_value = 1.0
        elif abs(pearson_value + 1.0) < 1e-15:
            pearson_value = -1.0
        pearson = pearson_value
        slope_value, intercept_value = np.polyfit(
            valid_expected,
            valid_paired,
            deg=1,
        )
        slope = float(slope_value)
        intercept = float(intercept_value)

    invalid_metric_count = int(length - np.count_nonzero(valid))
    metrics: dict[str, Any] = {
        "frame_count": int(length),
        "valid_metric_frame_count": int(np.count_nonzero(valid)),
        "invalid_metric_frame_count": invalid_metric_count,
        "stellar_mean_ratio_max_abs_error": float(
            np.max(np.abs(stellar_ratio_error[valid]))
        ),
        "final_paired_pearson_r": pearson,
        "final_paired_slope": slope,
        "final_paired_intercept": intercept,
        "final_paired_rmse": float(
            np.sqrt(np.mean(np.square(residual_error[valid])))
        ),
        "static_reference_final_flux": static_reference,
        "static_fractional_rms": float(np.std(static_normalized[valid])),
        "static_nan_pixel_count": int(np.sum(static_nan)),
        "injected_nan_pixel_count": int(np.sum(injected_nan)),
        "static_saturated_pixel_count": int(np.sum(static_saturated)),
        "injected_saturated_pixel_count": int(np.sum(injected_saturated)),
        "has_nan": bool(
            invalid_metric_count
            or np.sum(static_nan) > 0
            or np.sum(injected_nan) > 0
        ),
        "has_saturation": bool(
            np.sum(static_saturated) > 0 or np.sum(injected_saturated) > 0
        ),
    }
    metrics["acceptance"] = {
        "stellar_mean_ratio_error_le_1e-4": bool(
            metrics["stellar_mean_ratio_max_abs_error"] <= 1.0e-4
        ),
        "final_paired_pearson_r_ge_0_95": bool(
            pearson is not None and pearson >= 0.95
        ),
        "no_nan": not metrics["has_nan"],
        "no_saturation": not metrics["has_saturation"],
    }
    series = {
        "relative_flux": factor,
        "expected_paired_residual": expected_residual,
        "static_final_flux": static_final,
        "injected_final_flux": injected_final,
        "static_normalized_flux": static_normalized,
        "injected_normalized_flux": injected_normalized,
        "paired_residual": paired_residual,
        "residual_error": residual_error,
        "static_stellar_flux": static_stellar,
        "injected_stellar_flux": injected_stellar,
        "stellar_mean_ratio": stellar_ratio,
        "stellar_mean_ratio_error": stellar_ratio_error,
        "static_nan_pixel_count": static_nan,
        "injected_nan_pixel_count": injected_nan,
        "static_saturated_pixel_count": static_saturated,
        "injected_saturated_pixel_count": injected_saturated,
    }
    return metrics, series


def read_run_photometry(
    run_dir: Path | str,
    *,
    source_id: int = SOURCE_ID,
    aperture_radius: float = 4.0,
    annulus_inner: float = 5.5,
    annulus_outer: float = 7.0,
) -> RunPhotometry:
    """Read final-DN stamps and saved ``stellar_mean`` electron components."""

    from photsim7.artifacts import StampShardReader

    target_dir = Path(run_dir).expanduser().resolve() / "stamps" / f"target_{source_id}"
    raw_path = target_dir / "raw.h5"
    final_flux: list[float] = []
    stellar_flux: list[float] = []
    nan_counts: list[int] = []
    saturated_counts: list[int] = []
    with StampShardReader(raw_path) as reader:
        if reader.star_ids != (int(source_id),):
            raise ValueError(
                f"raw shard source IDs {reader.star_ids!r} do not match {source_id}"
            )
        frame_indices = tuple(int(value) for value in reader.frame_ids)
        for frame_index in frame_indices:
            image = reader.read_stamp(source_id, frame_index)
            measured = fixed_aperture_photometry(
                image,
                aperture_radius=aperture_radius,
                annulus_inner=annulus_inner,
                annulus_outer=annulus_outer,
            )
            component_path = (
                target_dir
                / "electron_components"
                / f"frame_{frame_index:06d}.npz"
            )
            if not component_path.is_file():
                raise FileNotFoundError(
                    f"saved electron component does not exist: {component_path}"
                )
            with np.load(component_path, allow_pickle=False) as components:
                if "stellar_mean" not in components.files:
                    raise ValueError(
                        f"electron component {component_path} lacks 'stellar_mean'"
                    )
                stellar_image = np.asarray(components["stellar_mean"], dtype=np.float64)
            yy, xx = np.indices(stellar_image.shape, dtype=np.float64)
            center_x = (stellar_image.shape[1] - 1) / 2.0
            center_y = (stellar_image.shape[0] - 1) / 2.0
            aperture = np.hypot(xx - center_x, yy - center_y) <= aperture_radius
            final_flux.append(measured.flux)
            stellar_flux.append(float(np.nansum(stellar_image[aperture])))
            nan_counts.append(
                measured.nan_pixel_count
                + int(np.count_nonzero(~np.isfinite(stellar_image)))
            )
            saturated_counts.append(measured.saturated_pixel_count)
    return RunPhotometry(
        frame_indices=frame_indices,
        final_flux=np.asarray(final_flux, dtype=np.float64),
        stellar_flux=np.asarray(stellar_flux, dtype=np.float64),
        nan_pixel_count=np.asarray(nan_counts, dtype=np.int64),
        saturated_pixel_count=np.asarray(saturated_counts, dtype=np.int64),
    )


def _write_plot(series: Mapping[str, np.ndarray], path: Path) -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib.pyplot as plt

    frame_index = np.arange(len(series["relative_flux"]), dtype=np.int64)
    figure, axes = plt.subplots(3, 1, figsize=(9.0, 10.0), sharex=True)
    axes[0].plot(frame_index, series["relative_flux"], "k-", label="Input factor")
    axes[0].plot(
        frame_index,
        series["static_normalized_flux"],
        "o-",
        label="Static / static median",
    )
    axes[0].plot(
        frame_index,
        series["injected_normalized_flux"],
        "o-",
        label="Injected / static median",
    )
    axes[0].set_ylabel("Normalized flux")
    axes[0].legend(loc="best")
    axes[0].grid(alpha=0.25)

    axes[1].plot(
        frame_index,
        series["expected_paired_residual"],
        "k-",
        label="Expected factor - 1",
    )
    axes[1].plot(
        frame_index,
        series["paired_residual"],
        "o-",
        label="(Injected - static) / static median",
    )
    axes[1].set_ylabel("Paired residual")
    axes[1].legend(loc="best")
    axes[1].grid(alpha=0.25)

    axes[2].axhline(0.0, color="black", linewidth=1.0)
    axes[2].plot(frame_index, series["residual_error"], "o-")
    axes[2].set_xlabel("Simulation raw frame index")
    axes[2].set_ylabel("Residual error")
    axes[2].grid(alpha=0.25)
    figure.suptitle("SN intrinsic variability: paired ET stamp validation")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def analyze_paired_runs(
    *,
    static_run_dir: Path | str,
    injected_run_dir: Path | str,
    curve: SnLightCurve,
    curve_id: str = CURVE_ID,
    output_dir: Path | str,
    source_id: int = SOURCE_ID,
    aperture_radius: float = 4.0,
    annulus_inner: float = 5.5,
    annulus_outer: float = 7.0,
) -> ValidationProducts:
    """Analyze paired runs and write JSON, frame CSV, and a three-panel PNG."""

    curve_id = str(curve_id).strip()
    if not curve_id:
        raise ValueError("curve_id must be non-empty")
    static = read_run_photometry(
        static_run_dir,
        source_id=source_id,
        aperture_radius=aperture_radius,
        annulus_inner=annulus_inner,
        annulus_outer=annulus_outer,
    )
    injected = read_run_photometry(
        injected_run_dir,
        source_id=source_id,
        aperture_radius=aperture_radius,
        annulus_inner=annulus_inner,
        annulus_outer=annulus_outer,
    )
    if static.frame_indices != injected.frame_indices:
        raise ValueError("static and injected raw shards have different frame indices")
    if static.frame_indices != curve.frame_indices:
        raise ValueError("raw shard frame indices do not match the generated curve")
    metrics, series = compute_paired_metrics(
        relative_flux=curve.relative_flux,
        static_final_flux=static.final_flux,
        injected_final_flux=injected.final_flux,
        static_stellar_flux=static.stellar_flux,
        injected_stellar_flux=injected.stellar_flux,
        static_nan_pixel_count=static.nan_pixel_count,
        injected_nan_pixel_count=injected.nan_pixel_count,
        static_saturated_pixel_count=static.saturated_pixel_count,
        injected_saturated_pixel_count=injected.saturated_pixel_count,
    )
    metrics.update(
        {
            "schema_id": "et_mainsim.source_variability_validation_metrics",
            "schema_version": 1,
            "source_id": int(source_id),
            "curve_id": curve_id,
            "source_path": str(curve.source_path),
            "source_sha256": curve.source_sha256,
            "source_band": curve.source_band,
            "source_zpsys": curve.source_zpsys,
            "magnitude_semantics": "Gaia_G_Vega",
            "magnitude_semantics_note": MAGNITUDE_SEMANTICS_NOTE,
            "time_alignment": TIME_ALIGNMENT,
            "ignored_input_time_columns": list(curve.ignored_input_time_columns),
            "static_run_dir": str(Path(static_run_dir).expanduser().resolve()),
            "injected_run_dir": str(Path(injected_run_dir).expanduser().resolve()),
            "aperture_radius_pix": float(aperture_radius),
            "background_annulus_pix": [float(annulus_inner), float(annulus_outer)],
        }
    )

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    metrics_path = destination / "source_variability_validation_metrics.json"
    per_frame_path = destination / "source_variability_validation_per_frame.csv"
    plot_path = destination / "source_variability_validation.png"
    frame_table = Table(
        {
            "frame_index": np.asarray(curve.frame_indices, dtype=np.int64),
            **series,
        }
    )
    frame_table.meta = _validation_meta(curve)
    frame_table.write(per_frame_path, format="ascii.csv", overwrite=True)
    metrics_path.write_text(
        json.dumps(
            metrics,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_plot(series, plot_path)
    return ValidationProducts(
        metrics_json=metrics_path,
        per_frame_csv=per_frame_path,
        plot_png=plot_path,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sn-csv", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--focalplane-registry", type=Path, required=True)
    parser.add_argument("--frame-count", type=int, default=22)
    parser.add_argument("--coadd-size", type=int, default=2)
    parser.add_argument("--source-id", type=int, default=SOURCE_ID)
    parser.add_argument("--curve-id", default=CURVE_ID)
    parser.add_argument("--run-id", default="source-variability-sn-validation")
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument(
        "--backend",
        choices=("in-process", "local-subprocess"),
        default="local-subprocess",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--workers-per-device", type=int, default=1)
    parser.add_argument("--stamp-rows", type=int, default=15)
    parser.add_argument("--stamp-cols", type=int, default=15)
    parser.add_argument("--aperture-radius", type=float, default=4.0)
    parser.add_argument("--annulus-inner", type=float, default=5.5)
    parser.add_argument("--annulus-outer", type=float, default=7.0)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Write inputs and commands without running Photsim7.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(None if argv is None else list(argv))
    if args.frame_count % args.coadd_size:
        raise ValueError("frame_count must be divisible by coadd_size")
    output_root = args.output_root.expanduser().resolve()
    curve = load_sn_lightcurve(args.sn_csv, frame_count=args.frame_count)
    inputs = write_validation_inputs(
        curve,
        output_root / "validation_inputs",
        source_id=args.source_id,
        curve_id=args.curve_id,
        coadd_size=args.coadd_size,
    )
    commands = build_run_commands(
        inputs=inputs,
        output_root=output_root,
        data_root=args.data_root,
        focalplane_registry=args.focalplane_registry,
        frame_count=args.frame_count,
        run_id=args.run_id,
        seed=args.seed,
        backend=args.backend,
        device=args.device,
        gpu_ids=tuple(value.strip() for value in args.gpus.split(",") if value.strip()),
        workers_per_device=args.workers_per_device,
        stamp_rows=args.stamp_rows,
        stamp_cols=args.stamp_cols,
    )
    command_path = output_root / "validation_inputs" / "commands.json"
    command_path.write_text(
        json.dumps(
            {
                "static": list(commands.static),
                "injected": list(commands.injected),
                "static_shell": shlex.join(commands.static),
                "injected_shell": shlex.join(commands.injected),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if args.prepare_only:
        print(command_path)
        return 0

    environment = dict(os.environ)
    environment.setdefault("MPLBACKEND", "Agg")
    existing_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (str(SOURCE_ROOT), existing_pythonpath)
        if value
    )
    subprocess.run(commands.static, cwd=REPO_ROOT, env=environment, check=True)
    subprocess.run(commands.injected, cwd=REPO_ROOT, env=environment, check=True)
    products = analyze_paired_runs(
        static_run_dir=output_root / "static" / args.run_id,
        injected_run_dir=output_root / "injected" / args.run_id,
        curve=curve,
        curve_id=args.curve_id,
        output_dir=output_root / "validation_products",
        source_id=args.source_id,
        aperture_radius=args.aperture_radius,
        annulus_inner=args.annulus_inner,
        annulus_outer=args.annulus_outer,
    )
    print(
        json.dumps(
            {
                "metrics_json": str(products.metrics_json),
                "per_frame_csv": str(products.per_frame_csv),
                "plot_png": str(products.plot_png),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
