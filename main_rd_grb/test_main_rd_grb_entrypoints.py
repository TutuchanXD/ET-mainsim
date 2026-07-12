from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parent


def _capture_entrypoint(tmp_path: Path, script_name: str) -> dict:
    parallel_dir = tmp_path / f"parallel-{script_name}"
    parallel_dir.mkdir()
    (parallel_dir / "main_rd_parallel_core.py").write_text(
        """
import json
import os
from pathlib import Path

def run_entrypoint(**kwargs):
    Path(os.environ["ENTRYPOINT_CAPTURE"]).write_text(
        json.dumps(kwargs, default=str),
        encoding="utf-8",
    )
""",
        encoding="utf-8",
    )
    capture_path = tmp_path / f"{script_name}.json"
    env = os.environ.copy()
    env.update(
        {
            "MAIN_RD_PARALLEL_DIR": str(parallel_dir),
            "ENTRYPOINT_CAPTURE": str(capture_path),
        }
    )
    result = subprocess.run(
        [sys.executable, str(MODULE_DIR / script_name)],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(capture_path.read_text(encoding="utf-8"))


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


def test_active_full_frame_entrypoints_pin_package_contract(tmp_path):
    cases = {
        "simulate_main_rd_full_10s_smoke.py": {
            "n_frames": 1,
            "mag_limit": 18.0,
            "observing_duration_s": 10.0,
        },
        "simulate_main_rd_full_10s_g17.py": {
            "n_frames": 120,
            "mag_limit": 17.0,
            "observing_duration_s": 1200.0,
        },
        "simulate_main_rd_full_10s_g17_extend360.py": {
            "n_frames": 360,
            "mag_limit": 17.0,
            "observing_duration_s": 3600.0,
        },
    }

    for script_name, expected in cases.items():
        captured = _capture_entrypoint(tmp_path, script_name)
        overrides = captured["spec_overrides"]
        assert captured["frame_rows"] == 9120
        assert captured["frame_cols"] == 8900
        assert "run_single_cadence_full_frame" in captured["description"]
        assert overrides["star_source"] == "gaia_main_rd"
        assert overrides["n_frames"] == expected["n_frames"]
        assert overrides["mag_limit"] == expected["mag_limit"]
        assert overrides["observing_duration_s"] == expected["observing_duration_s"]
        assert overrides["exposure_s"] == 10.0
        assert overrides["optical_efficiency_ratio"] == 0.58
        assert overrides["quantum_efficiency_ratio"] == 0.80

        source = (MODULE_DIR / script_name).read_text(encoding="utf-8")
        for legacy_builder in (
            "build_psf_manager",
            "build_full_effect_timeseries",
            "apply_detector_chain",
            "make_renderer",
        ):
            assert legacy_builder not in source
