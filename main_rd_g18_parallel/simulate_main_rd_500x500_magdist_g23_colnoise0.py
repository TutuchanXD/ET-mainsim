#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

from main_rd_parallel_core import DEFAULT_MAG_DISTRIBUTION_CSV, run_entrypoint


if __name__ == "__main__":
    run_entrypoint(
        frame_rows=500,
        frame_cols=500,
        description=(
            "Parallel synthetic main_rd simulation, 500x500 pixels, "
            "ET-mag distribution from 310-50-2420.csv, mag<=23, column noise disabled."
        ),
        script_path=Path(__file__).resolve(),
        spec_overrides={
            "run_label": "main_rd_500x500_magdist_310-50-2420_g_lt_23_colnoise0",
            "mag_limit": 23.0,
            "star_source": "synthetic_mag_distribution",
            "mag_distribution_csv": str(DEFAULT_MAG_DISTRIBUTION_CSV),
            "mag_distribution_column": "mwmsc_gmag",
            "synthetic_psf_field_angle_deg": 12.0,
            "column_noise_sigma_adu": 0.0,
            "notes": (
                "Synthetic 500x500 dense star field. Gaia G magnitudes from the ET_mag "
                "asset are treated as ET magnitudes; positions are seeded uniform random "
                "image-center coordinates; PSF field angle is fixed at 12 deg; column "
                "noise is disabled."
            ),
        },
    )
