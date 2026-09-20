"""Real workbook/physical ET acceptance for noncanonical raw timing.

Uses the S3 asset setup (ET_DATA_DIR, ET_FOCALPLANE_ROOT, ET_S3_DEVICE),
without modifying the workbook or running a production campaign.
"""

from dataclasses import replace
import json

import numpy as np
import pytest
from astropy import units as u

from test_s3_real_assets import _plan, _arrays


@pytest.mark.parametrize("workflow,table,detector,exposure,readout,starts", [
    ("full_frame", False, "CMOS", 0.25, 0.05, (2, 2.45, 3, 4)),
    ("stamp", False, "CCD", 5, 1, (0, 7, 15, 22)),
    ("stamp", True, "CMOS", 20, 0, (0, 20, 40, 60)),
])
def test_physical_noncanonical_raws_repeat_resume_and_record_time(
    tmp_path, workflow, table, detector, exposure, readout, starts,
):
    def prepare(run_id):
        module, plan = _plan(tmp_path, workflow, run_id, table=table)
        spec = replace(
            plan.spec, detector=replace(plan.spec.detector, detector_type=detector),
            observation=replace(plan.spec.observation, exposure_duration=exposure*u.s,
                                readout_duration=readout*u.s, observing_duration=None,
                                frame_start_s=starts),
        )
        return module, replace(plan, spec=spec)

    module, plan = prepare("first")
    run = module.run_full_frame if workflow == "full_frame" else module.run_stamp
    assert run(plan)["status"] == "completed"
    first = _arrays(plan, workflow)
    assert len(first) == 4
    assert run(plan)["status"] == "completed"
    _, repeat = prepare("repeat")
    assert run(repeat)["status"] == "completed"
    for a, b in zip(first, _arrays(repeat, workflow), strict=True):
        np.testing.assert_array_equal(a, b)
    integration = exposure+readout if detector == "CMOS" else exposure
    if workflow == "stamp":
        schemas = sorted((plan.run_dir / "stamps").glob("target_*/schemas/raw/*.json"))
        timings = [json.loads(path.read_text())["raw_cadence"] for path in schemas]
        assert [timing["frame_start_s"] for timing in timings] == list(starts)
        assert all(timing["integration_s"] == pytest.approx(integration) for timing in timings)
        coadds = sorted((plan.run_dir / "stamps").glob("target_*/schemas/coadd/*.json"))
        for i, path in enumerate(coadds):
            timing = json.loads(path.read_text())["coadd"]
            assert timing["raw_frame_start_s"] == list(starts[i*2:i*2+2])
            assert timing["total_integration_s"] == pytest.approx(2*integration)
            assert timing["readout_count"] == 2
    else:
        _, shard = prepare("shard")
        shard = replace(shard, frame_indices=(1, 3))
        assert run(shard)["status"] == "completed"
        for i, value in zip((1, 3), _arrays(shard, workflow), strict=True):
            np.testing.assert_array_equal(first[i], value)
