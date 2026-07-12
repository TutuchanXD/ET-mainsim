#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

from main_rd_parallel_core import DEFAULT_DETECTOR_XY_700_CSV, run_entrypoint


if __name__ == "__main__":
    run_entrypoint(
        frame_rows=700,
        frame_cols=700,
        description=(
            "Parallel detector-xy main_rd simulation, 700x700 pixels, 5x5 "
            "subpixel grid, source coordinates from "
            "310-50-2420_square700pix_glt24_detector_xy.csv, 270 frames; "
            "frames 180-269 add 10 e-/pix/frame scattered light."
        ),
        script_path=Path(__file__).resolve(),
        spec_overrides={
            "run_label": "main_rd_700x700_detectorxy_310-50-2420_sky23p2_colnoise0_subpix5_straylight10_last90",
            "n_frames": 270,
            "observing_duration_s": 2700.0,
            "mag_limit": 24.0,
            "star_source": "detector_xy_csv",
            "detector_xy_csv": str(DEFAULT_DETECTOR_XY_700_CSV),
            "detector_xy_source_id_column": "source_id",
            "detector_xy_mag_column": "gmag",
            "detector_xy_x_column": "x0",
            "detector_xy_y_column": "y0",
            "synthetic_psf_field_angle_deg": 12.0,
            "sky_surface_brightness_mag_arcsec2": 23.2,
            "n_subpixels": 5,
            "column_noise_sigma_adu": 0.0,
            "scattered_light_e_s_pix": 0.0,
            "scattered_light_step_start_frame": 180,
            "scattered_light_step_e_pix_frame": 10.0,
            "notes": (
                "Detector-xy 700x700 branch with 5x5 subpixel grid and 270 frames. "
                "Uses the square700pix G<24 detector-coordinate CSV directly. Frames "
                "0-179 have no added scattered light; frames 180-269 add "
                "10 e-/pix/frame. Other effects match the baseline detector-xy "
                "sky23.2 script."
            ),
        },
    )
