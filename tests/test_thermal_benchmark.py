from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_thermal_benchmark_changes_only_explicit_load_dimensions() -> None:
    from benchmarks.run_full_frame_thermal_load import build_benchmark_spec
    from et_mainsim.presets import load_preset

    base = load_preset("et-full-frame-production").simulation_spec
    spec = build_benchmark_spec(
        base,
        frames=1,
        mag_limit=17.0,
        jitter_models=100,
        run_seed=77,
    )

    assert spec.observation.resolved_n_frames == 1
    assert spec.catalog.background_stars_max_mag == 17.0
    assert spec.catalog.query_options["mag_lim"] == 17.0
    assert spec.psf.n_jitter_integrated_psf_models == 100
    assert spec.rng.run_seed == 77
    assert spec.dynamic_effects == base.dynamic_effects
    assert spec.detector_response == base.detector_response
    assert spec.readout == base.readout
