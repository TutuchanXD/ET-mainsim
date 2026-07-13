from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from astropy import units as u


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_package_import_is_lightweight() -> None:
    script = f"""
import sys
sys.path.insert(0, {str(REPO_ROOT / "src")!r})
import et_mainsim
assert et_mainsim.__version__
assert 'torch' not in sys.modules
assert 'ray' not in sys.modules
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_shipped_full_frame_presets_are_typed_and_complete() -> None:
    from et_mainsim.presets import list_presets, load_preset

    descriptors = list_presets(workflow="et-full-frame")

    assert [item.name for item in descriptors] == [
        "et-full-frame-production",
        "et-full-frame-smoke",
    ]

    smoke = load_preset("et-full-frame-smoke")
    production = load_preset("et-full-frame-production")

    assert smoke.simulation_spec.detector.shape == (64, 64)
    assert smoke.simulation_spec.observation.resolved_n_frames == 1
    assert smoke.simulation_spec.psf.field_id == 0
    assert smoke.simulation_spec.psf.field_id_policy == "explicit"
    assert smoke.run_config.execution.backend == "in-process"
    assert smoke.run_config.execution.device == "cpu"

    spec = production.simulation_spec
    assert spec.detector.shape == (9120, 8900)
    assert spec.observation.resolved_n_frames == 180
    assert spec.observation.sampling_interval == 10 * u.s
    assert spec.instrument.optical_efficiency.to_value(u.percent) == pytest.approx(58.0)
    assert spec.instrument.quantum_efficiency.to_value(u.percent) == pytest.approx(80.0)
    assert spec.catalog.source_type == "et_focalplane_query"
    assert spec.catalog.input_magnitude_system == "Gaia_G"
    assert spec.catalog.photon_magnitude_system == "ET"
    assert spec.catalog.target_epoch_jyear == pytest.approx(2000.0)
    assert production.run_config.execution.backend == "local-subprocess"
    assert production.run_config.execution.device == "cuda"


def test_unknown_preset_is_rejected() -> None:
    from et_mainsim.presets import load_preset

    with pytest.raises(KeyError, match="Unknown preset"):
        load_preset("missing")
