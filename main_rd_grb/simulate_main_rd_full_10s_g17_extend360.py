#!/usr/bin/env python
from __future__ import annotations

import os
import sys
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


RUN_LABEL = "main_rd_full_8900x9120_g17_sky22_subpix1_jipsf100_120x10s"
TOTAL_FRAMES = 360
EXPOSURE_S = 10.0


if __name__ == "__main__":
    run_entrypoint(
        frame_rows=9120,
        frame_cols=8900,
        description=(
            "Continuation through Photsim7 run_single_cadence_full_frame for the "
            "existing full main_rd G<17 10 s run. "
            "Use --frames 360 with --frame-indices 120..359 to extend the "
            "existing directory without overwriting frames 0..119."
        ),
        script_path=Path(__file__).resolve(),
        spec_overrides={
            "run_label": RUN_LABEL,
            "n_frames": TOTAL_FRAMES,
            "mag_limit": 17.0,
            "star_source": "gaia_main_rd",
            "sky_surface_brightness_mag_arcsec2": 22.0,
            "n_subpixels": 1,
            "exposure_s": EXPOSURE_S,
            "observing_duration_s": TOTAL_FRAMES * EXPOSURE_S,
            "optical_efficiency_ratio": 0.58,
            "quantum_efficiency_ratio": 0.80,
            "scattered_light_e_s_pix": 0.0,
            "scattered_light_step_start_frame": None,
            "scattered_light_step_e_pix_frame": 0.0,
            "n_jitter_integrated_psf_models": 100,
            "notes": (
                "Continuation of the existing full main_rd production directory. "
                "The physical configuration matches the original G<17, sky 22 "
                "mag/arcsec^2, subpixel 1x1, 100 JI-PSF, ET_psd3-2 setup. "
                "The intended run renders only frames 120..359 while the full "
                "dynamic-effect timeline is evaluated for 360 frames through "
                "the typed Photsim7 package services."
            ),
        },
    )
