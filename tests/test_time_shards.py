from __future__ import annotations

from dataclasses import replace
import json

import pytest


def test_cadence_sizes_and_continuous_shards_reject_only_the_tail() -> None:
    from et_mainsim.time_shards import (
        coadd_sizes_for_cadences,
        plan_continuous_time_shards,
    )

    coadd_sizes = coadd_sizes_for_cadences(
        raw_exposure_seconds=10.0,
        cadence_seconds=(30.0, 60.0, 120.0, 300.0),
    )
    plan = plan_continuous_time_shards(
        raw_start_index=0,
        raw_stop_index=125,
        coadd_sizes=coadd_sizes,
        raw_exposure_seconds=10.0,
        max_raw_frames_per_shard=90,
    )

    assert coadd_sizes == (3, 6, 12, 30)
    assert plan.alignment_raw_frames == 60
    assert plan.accepted_raw_start_index == 0
    assert plan.accepted_raw_stop_index == 120
    assert plan.rejected_tail_raw_interval == (120, 125)
    assert [(shard.raw_start_index, shard.raw_stop_index) for shard in plan.shards] == [
        (0, 60),
        (60, 120),
    ]


def test_coadd_windows_use_global_raw_indices_and_exposure_midpoints() -> None:
    from et_mainsim.time_shards import plan_continuous_time_shards

    plan = plan_continuous_time_shards(
        raw_start_index=60,
        raw_stop_index=180,
        coadd_sizes=(3, 6, 12, 30),
        raw_exposure_seconds=10.0,
        max_raw_frames_per_shard=120,
    )
    windows = tuple(plan.shards[0].iter_coadd_windows(30))

    assert [(item.raw_start_index, item.raw_stop_index) for item in windows] == [
        (60, 90),
        (90, 120),
        (120, 150),
        (150, 180),
    ]
    assert [item.coadd_index for item in windows] == [2, 3, 4, 5]
    assert [item.midpoint_time_seconds for item in windows] == [
        750.0,
        1050.0,
        1350.0,
        1650.0,
    ]
    assert all(item.duration_seconds == 300.0 for item in windows)


def test_global_raw_start_must_align_to_every_requested_cadence() -> None:
    from et_mainsim.time_shards import plan_continuous_time_shards

    with pytest.raises(ValueError, match="raw_start_index must align"):
        plan_continuous_time_shards(
            raw_start_index=1,
            raw_stop_index=121,
            coadd_sizes=(3, 6, 12, 30),
            raw_exposure_seconds=10.0,
            max_raw_frames_per_shard=120,
        )


def test_manifest_round_trip_preserves_tail_and_multi_cadence_partition() -> None:
    from et_mainsim.time_shards import (
        ContinuousTimeShardPlan,
        plan_continuous_time_shards,
    )

    plan = plan_continuous_time_shards(
        raw_start_index=0,
        raw_stop_index=725,
        coadd_sizes=(3, 6, 12, 30),
        raw_exposure_seconds=10.0,
        max_raw_frames_per_shard=180,
    )
    manifest = plan.to_manifest_dict()
    restored = ContinuousTimeShardPlan.from_manifest_dict(manifest)

    assert manifest["schema_id"] == "et_mainsim.continuous_time_shards.v1"
    assert manifest["raw_frame_interval"] == {"start_index": 0, "stop_index": 725}
    assert manifest["accepted_raw_frame_interval"] == {
        "start_index": 0,
        "stop_index": 720,
    }
    assert manifest["rejected_tail_raw_frame_interval"] == {
        "start_index": 720,
        "stop_index": 725,
    }
    assert manifest["coadd_sizes"] == [3, 6, 12, 30]
    assert restored == plan


def test_manifest_cannot_change_the_global_raw_frame_origin() -> None:
    from et_mainsim.time_shards import (
        ContinuousTimeShardPlan,
        plan_continuous_time_shards,
    )

    plan = plan_continuous_time_shards(
        raw_start_index=0,
        raw_stop_index=120,
        coadd_sizes=(3, 6, 12, 30),
        raw_exposure_seconds=10.0,
        max_raw_frames_per_shard=120,
    )
    manifest = plan.to_manifest_dict()
    manifest["time_axis"]["origin_raw_frame_index"] = 60

    with pytest.raises(ValueError, match="global raw frame origin"):
        ContinuousTimeShardPlan.from_manifest_dict(manifest)


def test_manifest_write_is_json_round_trippable(tmp_path) -> None:
    from et_mainsim.time_shards import (
        ContinuousTimeShardPlan,
        plan_continuous_time_shards,
    )

    plan = plan_continuous_time_shards(
        raw_start_index=0,
        raw_stop_index=180,
        coadd_sizes=(3, 6, 12, 30),
        raw_exposure_seconds=10.0,
        max_raw_frames_per_shard=120,
    )
    path = plan.write_manifest(tmp_path / "nested" / "time_shards.json")

    assert path.is_file()
    restored = ContinuousTimeShardPlan.from_manifest_dict(
        json.loads(path.read_text(encoding="utf-8"))
    )
    assert restored == plan


def test_plan_cannot_discard_a_complete_coadd_window_as_tail() -> None:
    from et_mainsim.time_shards import (
        ContinuousTimeShard,
        ContinuousTimeShardPlan,
    )

    with pytest.raises(ValueError, match="maximal complete global coadd interval"):
        ContinuousTimeShardPlan(
            raw_start_index=0,
            raw_stop_index=125,
            accepted_raw_start_index=0,
            accepted_raw_stop_index=60,
            coadd_sizes=(3, 6, 12, 30),
            raw_exposure_seconds=10.0,
            max_raw_frames_per_shard=60,
            shards=(
                ContinuousTimeShard(
                    shard_id=0,
                    raw_start_index=0,
                    raw_stop_index=60,
                    coadd_sizes=(3, 6, 12, 30),
                    raw_exposure_seconds=10.0,
                ),
            ),
        )


def test_coverage_validator_rejects_gaps_and_overlaps() -> None:
    from et_mainsim.time_shards import (
        plan_continuous_time_shards,
        validate_time_shard_coverage,
    )

    plan = plan_continuous_time_shards(
        raw_start_index=0,
        raw_stop_index=240,
        coadd_sizes=(3, 6, 12, 30),
        raw_exposure_seconds=10.0,
        max_raw_frames_per_shard=120,
    )
    first, second = plan.shards

    with pytest.raises(ValueError, match="gap"):
        validate_time_shard_coverage(
            (first, replace(second, raw_start_index=180, raw_stop_index=300)),
            raw_start_index=0,
            raw_stop_index=240,
            coadd_sizes=(3, 6, 12, 30),
        )

    with pytest.raises(ValueError, match="overlap"):
        validate_time_shard_coverage(
            (first, replace(second, raw_start_index=60, raw_stop_index=180)),
            raw_start_index=0,
            raw_stop_index=240,
            coadd_sizes=(3, 6, 12, 30),
        )
