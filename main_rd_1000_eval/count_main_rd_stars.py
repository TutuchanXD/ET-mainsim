from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from main_rd_common import (
    FRAME_COLS,
    FRAME_ROWS,
    GAIA_CATALOG_DIR,
    RESULTS_ROOT,
    ensure_results_root,
    experiment_spec_dict,
    query_main_rd_stars,
    star_summary,
    stars_to_dataframe,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count local Gaia/et_focalplane stars in the main_rd 1000x1000 crop."
    )
    parser.add_argument(
        "--mag-limits",
        nargs="+",
        type=float,
        default=[14.0, 15.0, 16.0, 17.0],
        help="Gaia G magnitude limits to evaluate.",
    )
    parser.add_argument(
        "--catalog-dir",
        type=Path,
        default=GAIA_CATALOG_DIR,
        help="Gaia healpix catalog directory.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=RESULTS_ROOT,
        help="Output directory under /home/cxgao/Results by default.",
    )
    parser.add_argument(
        "--crop-margin-pix",
        type=float,
        default=2.0,
        help="Extra crop margin used to keep edge PSF contributors.",
    )
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="Save per-limit star CSV files.",
    )
    return parser.parse_args()


def plot_star_map(star_data, *, mag_limit: float, output_path: Path) -> None:
    x = np.asarray(star_data["x0"], dtype=float) + (FRAME_COLS - 1) / 2.0
    y = np.asarray(star_data["y0"], dtype=float) + (FRAME_ROWS - 1) / 2.0
    mag = np.asarray(star_data["kp_mag"], dtype=float)
    fig, ax = plt.subplots(figsize=(7, 7), dpi=140)
    if len(x):
        size = np.clip(30.0 * 10 ** (-0.25 * (mag - np.nanmin(mag))), 1.0, 45.0)
        sc = ax.scatter(x, y, s=size, c=mag, cmap="viridis_r", alpha=0.75, linewidths=0)
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("ET/Kepler-like mag")
    ax.set_xlim(-0.5, FRAME_COLS - 0.5)
    ax.set_ylim(-0.5, FRAME_ROWS - 0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x pix in 1000x1000 crop")
    ax.set_ylabel("y pix in 1000x1000 crop")
    ax.set_title(f"main_rd crop star map, Gaia G < {mag_limit:g}, n={len(x)}")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_root = ensure_results_root(args.output_root)
    summaries = {}

    for mag_limit in args.mag_limits:
        start = time.perf_counter()
        star_data = query_main_rd_stars(
            mag_limit=mag_limit,
            catalog_dir=args.catalog_dir,
            crop_margin_pix=args.crop_margin_pix,
        )
        elapsed_s = time.perf_counter() - start
        summary = star_summary(star_data)
        summary.update(
            {
                "mag_limit": float(mag_limit),
                "elapsed_s": float(elapsed_s),
                "catalog_dir": str(args.catalog_dir),
                "crop_margin_pix": float(args.crop_margin_pix),
            }
        )
        summaries[f"mag_{mag_limit:g}"] = summary
        plot_star_map(
            star_data,
            mag_limit=mag_limit,
            output_path=output_root / f"main_rd_star_map_g_lt_{mag_limit:g}.png",
        )
        if args.save_csv:
            stars_to_dataframe(star_data).to_csv(
                output_root / f"main_rd_stars_g_lt_{mag_limit:g}.csv",
                index=False,
            )
        print(
            f"G<{mag_limit:g}: n={summary['n_stars']} "
            f"elapsed={elapsed_s:.2f}s kp=[{summary.get('kp_mag_min')}, {summary.get('kp_mag_max')}]"
        )

    payload = {
        "experiment": experiment_spec_dict(),
        "summaries": summaries,
    }
    write_json(output_root / "main_rd_star_counts.json", payload)


if __name__ == "__main__":
    main()

