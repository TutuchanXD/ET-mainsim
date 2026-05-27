#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

from main_rd_parallel_core import run_entrypoint


if __name__ == "__main__":
    run_entrypoint(
        frame_rows=1000,
        frame_cols=1000,
        description="Parallel main_rd center-crop simulation, 1000x1000 pixels, G<18.",
        script_path=Path(__file__).resolve(),
    )
