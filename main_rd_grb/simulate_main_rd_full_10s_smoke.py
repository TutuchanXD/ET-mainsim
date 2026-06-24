#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path


ET_ROOT = Path("/home/cxgao/ET")
PARALLEL_DIR = ET_ROOT / "ET-mainsim" / "main_rd_g18_parallel"
if str(PARALLEL_DIR) not in sys.path:
    sys.path.insert(0, str(PARALLEL_DIR))

from main_rd_parallel_core import run_entrypoint


if __name__ == "__main__":
    run_entrypoint(
        frame_rows=9120,
        frame_cols=8900,
        description=(
            "Single-cadence 10 s full main_rd smoke render with "
            "SingleCadenceFullFrameRenderer."
        ),
        script_path=Path(__file__).resolve(),
        spec_overrides={
            "run_label": "main_rd_full_8900x9120_g18_sky22_subpix1_10s_smoke",
            "n_frames": 1,
            "mag_limit": 18.0,
            "star_source": "gaia_main_rd",
            "sky_surface_brightness_mag_arcsec2": 22.0,
            "n_subpixels": 1,
            "exposure_s": 10.0,
            "observing_duration_s": 10.0,
            "scattered_light_e_s_pix": 0.0,
            "scattered_light_step_start_frame": None,
            "scattered_light_step_e_pix_frame": 0.0,
            "notes": (
                "Full main_rd one-frame smoke render. Background uses the "
                "surface-brightness magnitude path at 22 mag/arcsec^2; subpixel "
                "grid is 1x1; per-star PSF field IDs are selected from each "
                "star's main_rd field angle."
            ),
        },
    )
