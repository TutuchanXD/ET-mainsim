from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parent


def test_smoke_entrypoint_respects_main_rd_parallel_dir_override(tmp_path):
    parallel_dir = tmp_path / "parallel"
    parallel_dir.mkdir()
    (parallel_dir / "main_rd_parallel_core.py").write_text(
        "def run_entrypoint(**kwargs): pass\n",
        encoding="utf-8",
    )
    script = f"""
import runpy
namespace = runpy.run_path({str(MODULE_DIR / 'simulate_main_rd_full_10s_smoke.py')!r})
assert str(namespace["PARALLEL_DIR"]) == {str(parallel_dir)!r}
"""
    env = os.environ.copy()
    env["MAIN_RD_PARALLEL_DIR"] = str(parallel_dir)

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )

    assert result.returncode == 0, result.stderr
