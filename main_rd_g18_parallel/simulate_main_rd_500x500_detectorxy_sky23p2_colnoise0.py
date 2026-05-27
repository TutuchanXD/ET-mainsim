#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

from main_rd_parallel_core import DEFAULT_DETECTOR_XY_CSV, run_entrypoint


if __name__ == "__main__":
    run_entrypoint(
        frame_rows=500,
        frame_cols=500,
        description=(
            "Parallel detector-xy main_rd simulation, 500x500 pixels, "
            "source coordinates from 310-50-2420_square_detector_xy.csv, "
            "sky background 23.2 mag/arcsec^2, column noise disabled."
        ),
        script_path=Path(__file__).resolve(),
        spec_overrides={
            "run_label": "main_rd_500x500_detectorxy_310-50-2420_sky23p2_colnoise0",
            "mag_limit": 24.0,
            "star_source": "detector_xy_csv",
            "detector_xy_csv": str(DEFAULT_DETECTOR_XY_CSV),
            "detector_xy_source_id_column": "source_id",
            "detector_xy_mag_column": "gmag",
            "detector_xy_x_column": "x0",
            "detector_xy_y_column": "y0",
            "synthetic_psf_field_angle_deg": 12.0,
            "sky_surface_brightness_mag_arcsec2": 23.2,
            "column_noise_sigma_adu": 0.0,
            "notes": (
                "Detector-xy 500x500 source field. CSV gmag values are used directly "
                "as ET magnitudes; CSV x0/y0 are used directly as detector-centered "
                "pixel coordinates without rounding, integer casting, reprojection, "
                "or random placement. PSF field angle is fixed at 12 deg; sky "
                "background is 23.2 ET mag/arcsec^2; column noise is disabled. All "
                "other effects match simulate_main_rd_500x500_magdist_g24_sky23p2_colnoise0.py."
            ),
        },
    )
