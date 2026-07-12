#!/usr/bin/env python
from __future__ import annotations

import sys
import os
from pathlib import Path


ET_MAINSIM_ROOT = Path(
    os.environ.get("ET_MAINSIM_ROOT", str(Path(__file__).resolve().parents[1]))
).expanduser()
PARALLEL_DIR = Path(
    os.environ.get("MAIN_RD_PARALLEL_DIR", str(ET_MAINSIM_ROOT / "main_rd_g18_parallel"))
).expanduser()
if str(PARALLEL_DIR) not in sys.path:
    sys.path.insert(0, str(PARALLEL_DIR))

from main_rd_parallel_core import run_entrypoint


if __name__ == "__main__":
    run_entrypoint(
        frame_rows=9120,
        frame_cols=8900,
        description=(
            "Full main_rd production render through Photsim7 "
            "run_single_cadence_full_frame, with Gaia G<17 and 100 JI-PSF models."
        ),
        script_path=Path(__file__).resolve(),
        spec_overrides={
            "run_label": "main_rd_full_8900x9120_g17_sky22_subpix1_jipsf100_120x10s",
            "n_frames": 120,
            "mag_limit": 17.0,
            "star_source": "gaia_main_rd",
            "sky_surface_brightness_mag_arcsec2": 22.0,
            "n_subpixels": 1,
            "exposure_s": 10.0,
            "observing_duration_s": 1200.0,
            "optical_efficiency_ratio": 0.58,
            "quantum_efficiency_ratio": 0.80,
            "scattered_light_e_s_pix": 0.0,
            "scattered_light_step_start_frame": None,
            "scattered_light_step_e_pix_frame": 0.0,
            "n_jitter_integrated_psf_models": 100,
            "notes": (
                "Full main_rd 120-frame 10 s production setup. "
                "Default source cut is Gaia G<17; background uses "
                "22 mag/arcsec^2; subpixel grid is 1x1; default "
                "--jitter-psf-models is 100. Per-star PSF field IDs are "
                "selected from each star's main_rd field angle. The active "
                "cadence path uses the Photsim7 package pipeline and typed spec."
            ),
        },
    )
