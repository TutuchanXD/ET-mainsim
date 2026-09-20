from dataclasses import replace


def test_shared_shard_identity_survives_worker_count_change(tmp_path):
    from test_full_frame_workflow import (
        _selection_ready_worker_request,
        _enable_shared_exposure,
    )
    from et_mainsim.workflows.full_frame import run_worker

    base, api = _selection_ready_worker_request(tmp_path, n_frames=4)
    base = _enable_shared_exposure(base, frames_per_shard=2)
    first = run_worker(replace(base, rank=0, world_size=2), science_api=api)
    assert first.rendered == (0, 1)
    # A failed attempt left the other batch unfinished. A new worker count
    # must reuse the first batch and finish the second without rerendering it.
    results = [
        run_worker(replace(base, rank=rank, world_size=3), science_api=api)
        for rank in range(3)
    ]
    assert sorted(frame for result in results for frame in result.rendered) == [2, 3]
    assert sorted(frame for result in results for frame in result.skipped) == [0, 1]
