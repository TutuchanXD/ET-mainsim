#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from astropy import units as u


RESULTS_ROOT = Path("/home/cxgao/Results/ET-mainsim/main_rd_g18_parallel")
ET_QE_CORRECTION = 0.91526
ET_PHOTON_RATE_ZEROPOINT = ET_QE_CORRECTION * 615.75 * 1_961_225

TRUTH_COLUMNS = [
    "run_name",
    "frame_index",
    "time_s",
    "star_index",
    "source_id",
    "ra_deg",
    "dec_deg",
    "x0_centered_pix",
    "y0_centered_pix",
    "x0_truth_centered_pix",
    "y0_truth_centered_pix",
    "x_detector_static_pix",
    "y_detector_static_pix",
    "x_detector_truth_pix",
    "y_detector_truth_pix",
    "truth_valid_in_frame",
    "et_mag",
    "gmag",
    "field_angle_deg",
    "photon_rate_e_s",
    "photon_count_e_frame",
    "ideal_photon_snr",
    "psf_scale",
    "motion_offset_x_pix",
    "motion_offset_y_pix",
    "psd_dx_pix",
    "psd_dy_pix",
    "dva_dx_pix",
    "dva_dy_pix",
    "thermal_dx_pix",
    "thermal_dy_pix",
    "momentum_dump_dx_pix",
    "momentum_dump_dy_pix",
    "scattered_light_e_s_pix",
    "scattered_light_e_pix_frame",
]


@dataclass(frozen=True)
class RunContext:
    run_dir: Path
    run_name: str
    spec: dict[str, Any]
    frame_indices: list[int]
    frame_rows: int
    frame_cols: int
    exposure_s: float
    star_data: dict[str, np.ndarray]
    effects: dict[str, np.ndarray]
    photon_rate_e_s: np.ndarray
    photon_count_e_frame: np.ndarray
    effect_schema_id: str | None = None
    effect_timeseries: Any = None


def et_mag_to_photon_rate_e_s(et_mag: np.ndarray | float) -> np.ndarray:
    return ET_PHOTON_RATE_ZEROPOINT * np.power(10.0, -0.4 * np.asarray(et_mag, dtype=float))


def discover_last90_runs(results_root: Path | str = RESULTS_ROOT) -> list[Path]:
    root = Path(results_root).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Results root does not exist: {root}")
    return sorted(
        path
        for path in root.glob("*last90")
        if path.is_dir() and (path / "run_config.json").exists()
    )


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_npz_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {
            key: np.asarray(data[key])
            for key in data.files
            if key != "__metadata_json__"
        }


def _find_star_cache(run_dir: Path, run_config: dict[str, Any]) -> Path:
    configured = run_config.get("star_cache")
    if configured:
        path = Path(configured).expanduser()
        if path.exists():
            return path
    matches = sorted((run_dir / "cache").glob("stars_*.npz"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one star cache in {run_dir / 'cache'}, found {len(matches)}"
        )
    return matches[0]


def _required_array(star_data: dict[str, np.ndarray], *names: str) -> np.ndarray:
    for name in names:
        if name in star_data:
            return np.asarray(star_data[name])
    raise KeyError(f"Star cache is missing required columns: {names}")


def _optional_array(
    star_data: dict[str, np.ndarray],
    name: str,
    *,
    length: int,
    default: float,
) -> np.ndarray:
    if name in star_data:
        return np.asarray(star_data[name])
    return np.full(length, default, dtype=float)


def _effect_time_array(effects: dict[str, np.ndarray]) -> np.ndarray:
    if "time_s" in effects:
        return np.asarray(effects["time_s"], dtype=float)
    if "frame_start_s" in effects:
        return np.asarray(effects["frame_start_s"], dtype=float)
    raise KeyError("effects_timeseries is missing time_s or frame_start_s")


def _selected_frame_indices(run_config: dict[str, Any], effects: dict[str, np.ndarray]) -> list[int]:
    selected = run_config.get("selected_frame_indices")
    if selected:
        return [int(frame_index) for frame_index in selected]
    frames = int(
        (run_config.get("args") or {}).get("frames", len(_effect_time_array(effects)))
    )
    return list(range(frames))


def _package_effect_timeseries(
    *,
    run_dir: Path,
    effects: dict[str, np.ndarray],
    spec,
    catalog,
):
    from photsim7.dynamic_effect_models import build_frame_timing
    from photsim7.dynamic_effects import (
        EffectComponent,
        EffectSourceGeometry,
        EffectTimeseries,
    )
    from photsim7.simulation_services import build_projector_from_spec

    metadata_path = run_dir / "effects_timeseries.metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            "Package effects_timeseries requires its metadata sidecar: "
            f"{metadata_path}"
        )
    metadata = _load_json(metadata_path)
    timing_metadata = dict(metadata["timing"])
    timing = build_frame_timing(
        frame_start_s=_effect_time_array(effects),
        integration_s=float(timing_metadata["integration_s"]),
        sampling_interval_s=timing_metadata.get("sampling_interval_s"),
        split_hz=float(timing_metadata["split_hz"]),
    )
    components = []
    for component_metadata in metadata.get("components", []):
        name = str(component_metadata["name"])
        if name not in effects:
            raise KeyError(
                f"effects_timeseries is missing package component {name!r}"
            )
        components.append(
            EffectComponent(
                name=name,
                values=effects[name],
                unit=component_metadata["unit"],
                coordinate_frame=component_metadata["coordinate_frame"],
                scope=component_metadata["scope"],
                axes=tuple(component_metadata["axes"]),
                model_id=component_metadata["model_id"],
                enabled=bool(component_metadata["enabled"]),
                metadata=component_metadata.get("metadata", {}),
            )
        )
    projector = build_projector_from_spec(spec, catalog)
    source_geometry = getattr(projector, "source_geometry", None)
    if source_geometry is None:
        source_values = dict(catalog.star_data)
        source_values.setdefault(
            "source_id",
            np.arange(catalog.n_sources, dtype=np.int64),
        )
        detector_id = source_values.get("detector_id")
        if detector_id is not None and np.asarray(detector_id).ndim == 0:
            source_values["detector_id"] = np.full(
                catalog.n_sources,
                str(detector_id),
                dtype=object,
            )
        source_geometry = EffectSourceGeometry.from_mapping(source_values)
    return EffectTimeseries(
        timing=timing,
        components=tuple(components),
        source_geometry=source_geometry,
        projector=projector,
        jitter_integrated_psf_offsets=effects.get("xy_jitter_pix"),
        metadata=metadata.get("metadata", {}),
        rng_trace=metadata.get("rng_trace"),
    ), metadata


def _select_package_catalog(catalog, max_stars: int | None):
    if max_stars is None:
        return catalog
    from photsim7.catalog_sources import PreparedStarCatalog
    from photsim7.photometry import normalize_magnitude_input

    max_stars = int(max_stars)
    if max_stars <= 0:
        raise ValueError("max_stars must be positive when provided")
    if catalog.n_sources <= max_stars:
        return catalog
    magnitude = normalize_magnitude_input(catalog.star_data, mag_type="ET").magnitude
    selected_indices = np.argsort(magnitude)[:max_stars]
    selected_data: dict[str, Any] = {}
    for key, value in catalog.star_data.items():
        array = np.asarray(value)
        if array.ndim == 1 and len(array) == catalog.n_sources:
            selected_data[key] = array[selected_indices]
        else:
            selected_data[key] = value
    return PreparedStarCatalog(
        star_data=selected_data,
        metadata=dict(catalog.metadata),
        schema_id=catalog.schema_id,
        schema_version=catalog.schema_version,
    )


def load_run_context(run_dir: Path | str) -> RunContext:
    run_dir = Path(run_dir).expanduser()
    run_config = _load_json(run_dir / "run_config.json")
    spec = dict(run_config["spec"])
    effects = _load_npz_arrays(run_dir / "effects_timeseries.npz")
    star_cache = _find_star_cache(run_dir, run_config)
    effect_timeseries = None
    effect_schema_id = None
    if "simulation_spec" in run_config:
        from photsim7.catalog_sources import StarCatalogCache
        from photsim7.simulation_services import build_star_table_from_catalog
        from photsim7.specs import SimulationSpec

        typed_spec = SimulationSpec.from_json_dict(run_config["simulation_spec"])
        catalog = _select_package_catalog(
            StarCatalogCache.read(star_cache),
            (run_config.get("args") or {}).get("max_stars"),
        )
        star_data = dict(catalog.star_data)
        stars = build_star_table_from_catalog(typed_spec, catalog)
        photon_rate = np.asarray(
            stars["Detected Electron Rate"].to_value(u.electron / u.s),
            dtype=float,
        )
        exposure_s = typed_spec.observation.integration_for(
            typed_spec.detector.detector_type
        ).to_value(u.s)
        effect_timeseries, effect_metadata = _package_effect_timeseries(
            run_dir=run_dir,
            effects=effects,
            spec=typed_spec,
            catalog=catalog,
        )
        effect_schema_id = str(effect_metadata["schema_id"])
    else:
        star_data = _load_npz_arrays(star_cache)
        et_mag = _required_array(star_data, "et_mag", "gmag", "kp_mag").astype(float)
        photon_rate = et_mag_to_photon_rate_e_s(et_mag)
        exposure_s = float(spec.get("exposure_s", 10.0))
    photon_count = photon_rate * exposure_s

    return RunContext(
        run_dir=run_dir,
        run_name=str(spec.get("run_label") or run_dir.name),
        spec=spec,
        frame_indices=_selected_frame_indices(run_config, effects),
        frame_rows=int(spec["frame_rows"]),
        frame_cols=int(spec["frame_cols"]),
        exposure_s=exposure_s,
        star_data=star_data,
        effects=effects,
        photon_rate_e_s=photon_rate,
        photon_count_e_frame=photon_count,
        effect_schema_id=effect_schema_id,
        effect_timeseries=effect_timeseries,
    )


def _effect_xy(effects: dict[str, np.ndarray], key: str, frame_index: int) -> tuple[float, float]:
    if key not in effects:
        return 0.0, 0.0
    value = np.asarray(effects[key])[frame_index]
    return float(value[0]), float(value[1])


def _scattered_light_for_frame(context: RunContext, frame_index: int) -> tuple[float, float]:
    rate = float(context.spec.get("scattered_light_e_s_pix", 0.0))
    start_frame = context.spec.get("scattered_light_step_start_frame")
    step_e_pix_frame = float(context.spec.get("scattered_light_step_e_pix_frame", 0.0))
    if start_frame is not None and int(frame_index) >= int(start_frame) and step_e_pix_frame != 0.0:
        rate += step_e_pix_frame / context.exposure_s
    return rate, rate * context.exposure_s


def build_frame_truth_dataframe(context: RunContext, frame_index: int) -> pd.DataFrame:
    frame_index = int(frame_index)
    time_s = _effect_time_array(context.effects)
    if frame_index < 0 or frame_index >= len(time_s):
        raise IndexError(
            f"Frame {frame_index} is outside effects_timeseries range 0.."
            f"{len(time_s) - 1}"
        )

    star_data = context.star_data
    source_id = _required_array(star_data, "source_id")
    n_stars = len(source_id)
    x0 = _required_array(star_data, "x0").astype(float)
    y0 = _required_array(star_data, "y0").astype(float)
    if context.effect_timeseries is None:
        x_static = _required_array(
            star_data,
            "detector_xpix",
            "detector_xpix_shifted",
        ).astype(float)
        y_static = _required_array(
            star_data,
            "detector_ypix",
            "detector_ypix_shifted",
        ).astype(float)
        dx, dy = _effect_xy(context.effects, "total_motion_pix", frame_index)
        total_offsets = np.broadcast_to([dx, dy], (n_stars, 2)).copy()
        psd_offsets = np.broadcast_to(
            _effect_xy(context.effects, "psd_drift_pix", frame_index),
            (n_stars, 2),
        ).copy()
        dva_offsets = np.broadcast_to(
            _effect_xy(context.effects, "dva_drift_pix", frame_index),
            (n_stars, 2),
        ).copy()
        thermal_offsets = np.broadcast_to(
            _effect_xy(context.effects, "thermal_drift_pix", frame_index),
            (n_stars, 2),
        ).copy()
        momentum_offsets = np.broadcast_to(
            _effect_xy(context.effects, "momentum_dump_pix", frame_index),
            (n_stars, 2),
        ).copy()
        psf_scale = np.full(
            n_stars,
            float(context.effects["psf_scale"][frame_index]),
        )
    else:
        x_static = x0 + (float(context.frame_cols) - 1.0) / 2.0
        y_static = y0 + (float(context.frame_rows) - 1.0) / 2.0
        cadence = context.effect_timeseries.for_cadence(
            frame_index,
            base_x_pix=x_static,
            base_y_pix=y_static,
        )
        total_offsets = np.asarray(cadence.source_offsets_pix, dtype=float)
        zeros = np.zeros_like(total_offsets)
        psd_offsets = np.asarray(
            cadence.component_offsets_pix.get("psd_drift", zeros),
            dtype=float,
        )
        dva_offsets = np.asarray(
            cadence.component_offsets_pix.get("dva", zeros),
            dtype=float,
        )
        thermal_offsets = np.asarray(
            cadence.component_offsets_pix.get("thermal", zeros),
            dtype=float,
        )
        momentum_offsets = np.asarray(
            cadence.component_offsets_pix.get("momentum_dump", zeros),
            dtype=float,
        )
        psf_scale = np.asarray(cadence.star_psf_scales, dtype=float)
    et_mag = _required_array(
        star_data,
        "et_mag",
        "gaia_g_mag",
        "gmag",
        "kp_mag",
    ).astype(float)
    gmag = np.asarray(
        star_data.get("gmag", star_data.get("gaia_g_mag", np.full(n_stars, np.nan))),
        dtype=float,
    )
    ra = _optional_array(star_data, "ra", length=n_stars, default=np.nan).astype(float)
    dec = _optional_array(star_data, "dec", length=n_stars, default=np.nan).astype(float)
    field_angle = _optional_array(
        star_data,
        "field_angle_deg",
        length=n_stars,
        default=np.nan,
    ).astype(float)

    scattered_rate, scattered_count = _scattered_light_for_frame(context, frame_index)

    x_truth = x_static + total_offsets[:, 0]
    y_truth = y_static + total_offsets[:, 1]
    valid = (
        (x_truth >= 0.0)
        & (x_truth < float(context.frame_cols))
        & (y_truth >= 0.0)
        & (y_truth < float(context.frame_rows))
    )

    frame = pd.DataFrame(
        {
            "run_name": context.run_name,
            "frame_index": frame_index,
            "time_s": float(time_s[frame_index]),
            "star_index": np.arange(n_stars, dtype=np.int64),
            "source_id": source_id,
            "ra_deg": ra,
            "dec_deg": dec,
            "x0_centered_pix": x0,
            "y0_centered_pix": y0,
            "x0_truth_centered_pix": x0 + total_offsets[:, 0],
            "y0_truth_centered_pix": y0 + total_offsets[:, 1],
            "x_detector_static_pix": x_static,
            "y_detector_static_pix": y_static,
            "x_detector_truth_pix": x_truth,
            "y_detector_truth_pix": y_truth,
            "truth_valid_in_frame": valid,
            "et_mag": et_mag,
            "gmag": gmag,
            "field_angle_deg": field_angle,
            "photon_rate_e_s": context.photon_rate_e_s,
            "photon_count_e_frame": context.photon_count_e_frame,
            "ideal_photon_snr": np.sqrt(np.clip(context.photon_count_e_frame, 0.0, None)),
            "psf_scale": psf_scale,
            "motion_offset_x_pix": total_offsets[:, 0],
            "motion_offset_y_pix": total_offsets[:, 1],
            "psd_dx_pix": psd_offsets[:, 0],
            "psd_dy_pix": psd_offsets[:, 1],
            "dva_dx_pix": dva_offsets[:, 0],
            "dva_dy_pix": dva_offsets[:, 1],
            "thermal_dx_pix": thermal_offsets[:, 0],
            "thermal_dy_pix": thermal_offsets[:, 1],
            "momentum_dump_dx_pix": momentum_offsets[:, 0],
            "momentum_dump_dy_pix": momentum_offsets[:, 1],
            "scattered_light_e_s_pix": scattered_rate,
            "scattered_light_e_pix_frame": scattered_count,
        }
    )
    return frame[TRUTH_COLUMNS]


def _write_manifest(
    *,
    context: RunContext,
    output_dir: Path,
    frame_indices: list[int],
    written_paths: list[Path],
) -> None:
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": context.run_name,
        "run_dir": str(context.run_dir),
        "output_dir": str(output_dir),
        "n_frames": len(frame_indices),
        "frame_indices": frame_indices,
        "n_stars": int(len(context.star_data["source_id"])),
        "total_rows": int(len(frame_indices) * len(context.star_data["source_id"])),
        "columns": TRUTH_COLUMNS,
        "files": [path.name for path in written_paths],
        "snr_definition": "ideal_photon_snr = sqrt(photon_count_e_frame)",
        "position_definition": (
            "x/y_detector_truth_pix = static detector_x/y pix + total_motion_pix for the frame"
        ),
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def export_run_truth_tables(
    run_dir: Path | str,
    *,
    output_dir: Path | str | None = None,
    frame_indices: Iterable[int] | None = None,
    overwrite: bool = False,
    progress_every: int = 25,
) -> list[Path]:
    context = load_run_context(run_dir)
    frames = context.frame_indices if frame_indices is None else [int(index) for index in frame_indices]
    output_path = (
        context.run_dir / "truth_tables"
        if output_dir is None
        else Path(output_dir).expanduser()
    )
    output_path.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for offset, frame_index in enumerate(frames, start=1):
        csv_path = output_path / f"frame_{frame_index:06d}.csv"
        if csv_path.exists() and not overwrite:
            written.append(csv_path)
            continue
        table = build_frame_truth_dataframe(context, frame_index)
        table.to_csv(csv_path, index=False, float_format="%.12g")
        written.append(csv_path)
        if progress_every > 0 and (
            offset == 1 or offset == len(frames) or offset % int(progress_every) == 0
        ):
            print(
                f"[{context.run_name}] wrote {offset}/{len(frames)} "
                f"frame CSV files to {output_path}"
            )

    _write_manifest(
        context=context,
        output_dir=output_path,
        frame_indices=frames,
        written_paths=written,
    )
    return written


def parse_frame_indices(value: str | None) -> list[int] | None:
    if value is None or str(value).strip() == "":
        return None
    indices: list[int] = []
    seen: set[int] = set()
    for token in str(value).split(","):
        token = token.strip()
        if not token:
            continue
        frame_index = int(token)
        if frame_index not in seen:
            indices.append(frame_index)
            seen.add(frame_index)
    return indices


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export per-frame, per-star truth CSV tables from completed last90 "
            "main_rd simulations without rerunning the simulation."
        )
    )
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        default=None,
        help="Specific run directory to export. May be provided multiple times.",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default="truth_tables",
        help="Output subdirectory inside each run directory.",
    )
    parser.add_argument(
        "--frames",
        type=str,
        default=None,
        help="Optional comma-separated frame indices, for example 0,180,269.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_dirs = (
        [Path(path).expanduser() for path in args.run_dir]
        if args.run_dir
        else discover_last90_runs(args.results_root)
    )
    if not run_dirs:
        raise FileNotFoundError(f"No *last90 run directories found under {args.results_root}")

    requested_frames = parse_frame_indices(args.frames)
    for run_dir in run_dirs:
        context = load_run_context(run_dir)
        frames = context.frame_indices if requested_frames is None else requested_frames
        output_dir = context.run_dir / args.output_subdir
        print(
            f"[{context.run_name}] stars={len(context.star_data['source_id'])} "
            f"frames={len(frames)} rows={len(context.star_data['source_id']) * len(frames)}"
        )
        if args.dry_run:
            print(f"[{context.run_name}] dry-run output_dir={output_dir}")
            continue
        export_run_truth_tables(
            context.run_dir,
            output_dir=output_dir,
            frame_indices=frames,
            overwrite=bool(args.overwrite),
            progress_every=int(args.progress_every),
        )


if __name__ == "__main__":
    main()
