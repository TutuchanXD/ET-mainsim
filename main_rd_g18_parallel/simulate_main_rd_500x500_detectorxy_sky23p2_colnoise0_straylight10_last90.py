#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

from main_rd_parallel_core import DEFAULT_DETECTOR_XY_CSV, run_entrypoint


if __name__ == "__main__":
    run_entrypoint(
        frame_rows=500,
        frame_cols=500,
        description=(
            "Parallel detector-xy main_rd simulation, 500x500 pixels, sky 22 mag/arcsec^2, "
            "1x1 subpixel grid, 270 frames; "
            "frames 0-179 match the baseline, frames 180-269 add 10 e-/pix/frame "
            "scattered light."
        ),
        script_path=Path(__file__).resolve(),
        spec_overrides={
            "run_label": "main_rd_500x500_detectorxy_310-50-2420_sky22_colnoise0_subpix1_straylight10_last90",
            "n_frames": 270,
            "mag_limit": 24.0,
            "star_source": "detector_xy_csv",
            "detector_xy_csv": str(DEFAULT_DETECTOR_XY_CSV),
            "detector_xy_source_id_column": "source_id",
            "detector_xy_mag_column": "gmag",
            "detector_xy_x_column": "x0",
            "detector_xy_y_column": "y0",
            "synthetic_psf_field_angle_deg": 12.0,
            "sky_surface_brightness_mag_arcsec2": 22.0,
            "n_subpixels": 1,
            "column_noise_sigma_adu": 0.0,
            "scattered_light_e_s_pix": 0.0,
            "scattered_light_step_start_frame": 180,
            "scattered_light_step_e_pix_frame": 10.0,
            "notes": (
                "Detector-xy 500x500 baseline with sky background set through the "
                "surface-brightness magnitude path at 22 mag/arcsec^2 and a 1x1 "
                "subpixel grid. Star PSF field IDs are selected from each star's "
                "main_rd detector position. Frames 180-269 add 10 e-/pix/frame "
                "scattered light. All other effects are unchanged."
            ),
        },
    )
