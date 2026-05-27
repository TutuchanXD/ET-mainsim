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


def _selected_frame_indices(run_config: dict[str, Any], effects: dict[str, np.ndarray]) -> list[int]:
    selected = run_config.get("selected_frame_indices")
    if selected:
        return [int(frame_index) for frame_index in selected]
    frames = int((run_config.get("args") or {}).get("frames", len(effects["time_s"])))
    return list(range(frames))


def load_run_context(run_dir: Path | str) -> RunContext:
    run_dir = Path(run_dir).expanduser()
    run_config = _load_json(run_dir / "run_config.json")
    spec = dict(run_config["spec"])
    effects = _load_npz_arrays(run_dir / "effects_timeseries.npz")
    star_cache = _find_star_cache(run_dir, run_config)
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
    if frame_index < 0 or frame_index >= len(context.effects["time_s"]):
        raise IndexError(
            f"Frame {frame_index} is outside effects_timeseries range 0.."
            f"{len(context.effects['time_s']) - 1}"
        )

    star_data = context.star_data
    source_id = _required_array(star_data, "source_id")
    n_stars = len(source_id)
    x0 = _required_array(star_data, "x0").astype(float)
    y0 = _required_array(star_data, "y0").astype(float)
    x_static = _required_array(star_data, "detector_xpix", "detector_xpix_shifted").astype(float)
    y_static = _required_array(star_data, "detector_ypix", "detector_ypix_shifted").astype(float)
    et_mag = _required_array(star_data, "et_mag", "gmag", "kp_mag").astype(float)
    gmag = _optional_array(star_data, "gmag", length=n_stars, default=np.nan).astype(float)
    ra = _optional_array(star_data, "ra", length=n_stars, default=np.nan).astype(float)
    dec = _optional_array(star_data, "dec", length=n_stars, default=np.nan).astype(float)
    field_angle = _optional_array(
        star_data,
        "field_angle_deg",
        length=n_stars,
        default=np.nan,
    ).astype(float)

    dx, dy = _effect_xy(context.effects, "total_motion_pix", frame_index)
    psd_dx, psd_dy = _effect_xy(context.effects, "psd_drift_pix", frame_index)
    dva_dx, dva_dy = _effect_xy(context.effects, "dva_drift_pix", frame_index)
    thermal_dx, thermal_dy = _effect_xy(context.effects, "thermal_drift_pix", frame_index)
    momentum_dx, momentum_dy = _effect_xy(context.effects, "momentum_dump_pix", frame_index)
    scattered_rate, scattered_count = _scattered_light_for_frame(context, frame_index)

    x_truth = x_static + dx
    y_truth = y_static + dy
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
            "time_s": float(context.effects["time_s"][frame_index]),
            "star_index": np.arange(n_stars, dtype=np.int64),
            "source_id": source_id,
            "ra_deg": ra,
            "dec_deg": dec,
            "x0_centered_pix": x0,
            "y0_centered_pix": y0,
            "x0_truth_centered_pix": x0 + dx,
            "y0_truth_centered_pix": y0 + dy,
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
            "psf_scale": float(context.effects["psf_scale"][frame_index]),
            "motion_offset_x_pix": dx,
            "motion_offset_y_pix": dy,
            "psd_dx_pix": psd_dx,
            "psd_dy_pix": psd_dy,
            "dva_dx_pix": dva_dx,
            "dva_dy_pix": dva_dy,
            "thermal_dx_pix": thermal_dx,
            "thermal_dy_pix": thermal_dy,
            "momentum_dump_dx_pix": momentum_dx,
            "momentum_dump_dy_pix": momentum_dy,
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
