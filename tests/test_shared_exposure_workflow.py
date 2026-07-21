from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest


def _geometry():
    from photsim7.simulation_services import FullFrameSourcePixelGeometry

    return FullFrameSourcePixelGeometry(
        source_ids=np.array([101, 202, 303], dtype=np.int64),
        detector_ids=np.array(["main-1", "main-1", "main-1"]),
        x_frame_pix=np.array([2.25, 7.5, 0.25], dtype=np.float64),
        y_frame_pix=np.array([3.75, 5.5, 0.5], dtype=np.float64),
        position_columns=(
            "Detector Xpix Shifted",
            "Detector Ypix Shifted",
        ),
        position_transform="direct_shifted_frame_grid",
        geometry_truth_mode="reference_field_nonphysical",
        geometry_truth_content_sha256="a" * 64,
    )


def _plan(*, target_source_ids=(303, 101), stamp_shape=(3, 5)):
    from et_mainsim.shared_exposure import build_shared_exposure_target_plan

    return build_shared_exposure_target_plan(
        _geometry(),
        target_source_ids,
        detector_shape=(10, 12),
        stamp_shape=stamp_shape,
    )


def _published_plan(tmp_path: Path):
    from et_mainsim.shared_exposure import publish_shared_exposure_target_plan

    path = tmp_path / "shared-exposure-target-plan.json"
    plan = _plan()
    publish_shared_exposure_target_plan(path, plan)
    return path, plan


def _completion_inputs(tmp_path: Path):
    from et_mainsim.shared_exposure import shared_exposure_product_shard_path

    plan_path, plan = _published_plan(tmp_path)
    parent_path = tmp_path / "parent-frame.npy"
    parent_path.write_bytes(b"parent-frame-storage-bytes")
    shards: dict[str, Path] = {}
    for product_key, payload in (
        ("final_stamp", b"final-shard"),
        ("electron_components.background_mean", b"background-shard"),
    ):
        path = shared_exposure_product_shard_path(
            tmp_path,
            plan_content_sha256=plan["content_sha256"],
            product_key=product_key,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        shards[product_key] = path
    return plan_path, plan, parent_path, shards


def _completion(tmp_path: Path):
    from et_mainsim.shared_exposure import (
        build_shared_exposure_frame_completion,
    )

    plan_path, plan, parent_path, shards = _completion_inputs(tmp_path)
    marker = build_shared_exposure_frame_completion(
        frame_index=17,
        detector_id="main-1",
        mode="single_parent_full_frame_then_shared_crops",
        reference_root=tmp_path,
        parent_path=parent_path,
        plan_path=plan_path,
        product_shards=shards,
    )
    return marker, plan_path, plan, parent_path, shards


def test_target_plan_uses_static_renderer_positions_and_preserves_request_order() -> (
    None
):
    from et_mainsim.shared_exposure import (
        TARGET_PLAN_SCHEMA_ID,
        validate_shared_exposure_target_plan,
    )

    plan = _plan()

    assert plan["schema_id"] == TARGET_PLAN_SCHEMA_ID
    assert len(plan["content_sha256"]) == 64
    assert plan["source_geometry"] == {
        "schema_id": "photsim7.full_frame_source_pixel_geometry.v1",
        "schema_version": 1,
        "geometry_truth_mode": "reference_field_nonphysical",
        "geometry_truth_content_sha256": "a" * 64,
        "position_basis": "static_base_renderer_positions",
        "position_columns": [
            "Detector Xpix Shifted",
            "Detector Ypix Shifted",
        ],
        "position_transform": "direct_shifted_frame_grid",
        "coordinate_convention": _geometry().coordinate_convention,
    }
    assert plan["detector"] == {"detector_id": "main-1", "shape": [10, 12]}
    assert plan["stamp_shape"] == [3, 5]
    assert [target["source_id"] for target in plan["targets"]] == [303, 101]
    assert [target["request_index"] for target in plan["targets"]] == [0, 1]
    assert plan["targets"][0]["x_frame_pix"] == 0.25
    assert plan["targets"][0]["y_frame_pix"] == 0.5
    edge_window = plan["targets"][0]["window"]
    assert edge_window["x_start_detector_pix"] == -2
    assert edge_window["y_start_detector_pix"] == -1
    assert edge_window["clipped_by_detector"] is True
    assert validate_shared_exposure_target_plan(plan) == plan


@pytest.mark.parametrize(
    ("target_source_ids", "message"),
    [
        ((101, 101), "duplicate"),
        ((999,), "missing"),
        ((True,), "signed 64-bit integer"),
        (("101",), "signed 64-bit integer"),
        ((), "must not be empty"),
    ],
)
def test_target_plan_fails_closed_on_invalid_requested_ids(
    target_source_ids,
    message,
) -> None:
    from et_mainsim.shared_exposure import (
        SharedExposureContractError,
        build_shared_exposure_target_plan,
    )

    with pytest.raises(SharedExposureContractError, match=message):
        build_shared_exposure_target_plan(
            _geometry(),
            target_source_ids,
            detector_shape=(10, 12),
            stamp_shape=(3, 5),
        )


def test_target_plan_fails_closed_on_duplicate_geometry_source_ids() -> None:
    from et_mainsim.shared_exposure import (
        SharedExposureContractError,
        build_shared_exposure_target_plan,
    )

    duplicate_geometry = _geometry()
    # Exercise this function's own fail-closed boundary rather than relying on
    # the upstream dataclass constructor to preserve its invariant forever.
    object.__setattr__(
        duplicate_geometry,
        "source_ids",
        np.array([101, 101, 303], dtype=np.int64),
    )

    with pytest.raises(SharedExposureContractError, match="geometry.*duplicate"):
        build_shared_exposure_target_plan(
            duplicate_geometry,
            (101,),
            detector_shape=(10, 12),
            stamp_shape=(3, 5),
        )


@pytest.mark.parametrize(
    ("detector_shape", "stamp_shape"),
    [
        ((True, 12), (3, 5)),
        ((10.0, 12), (3, 5)),
        ((10, 12), (0, 5)),
        ((10, 12), (3, 5.0)),
    ],
)
def test_target_plan_requires_strict_positive_shapes(
    detector_shape,
    stamp_shape,
) -> None:
    from et_mainsim.shared_exposure import (
        SharedExposureContractError,
        build_shared_exposure_target_plan,
    )

    with pytest.raises(SharedExposureContractError, match="positive integers"):
        build_shared_exposure_target_plan(
            _geometry(),
            (101,),
            detector_shape=detector_shape,
            stamp_shape=stamp_shape,
        )


def test_target_plan_rejects_mixed_detector_geometry_and_zero_overlap() -> None:
    from et_mainsim.shared_exposure import (
        SharedExposureContractError,
        build_shared_exposure_target_plan,
    )

    mixed = replace(
        _geometry(),
        detector_ids=np.array(["main-1", "main-2", "main-1"]),
    )
    with pytest.raises(SharedExposureContractError, match="one detector"):
        build_shared_exposure_target_plan(
            mixed,
            (101,),
            detector_shape=(10, 12),
            stamp_shape=(3, 5),
        )

    outside = replace(
        _geometry(),
        x_frame_pix=np.array([2.25, 7.5, -100.0]),
    )
    with pytest.raises(SharedExposureContractError, match="detector overlap"):
        build_shared_exposure_target_plan(
            outside,
            (303,),
            detector_shape=(10, 12),
            stamp_shape=(3, 5),
        )


def test_plan_publication_is_canonical_identical_idempotent_and_conflict_safe(
    tmp_path,
) -> None:
    from et_mainsim.shared_exposure import (
        SharedExposurePublicationError,
        publish_shared_exposure_target_plan,
        read_shared_exposure_target_plan,
    )

    path = tmp_path / "plan.json"
    plan = _plan()
    assert publish_shared_exposure_target_plan(path, plan) == path
    first_bytes = path.read_bytes()
    assert first_bytes == json.dumps(
        plan,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert publish_shared_exposure_target_plan(path, plan) == path
    assert path.read_bytes() == first_bytes
    assert read_shared_exposure_target_plan(path) == plan

    with pytest.raises(SharedExposurePublicationError, match="conflict"):
        publish_shared_exposure_target_plan(
            path,
            _plan(target_source_ids=(101,)),
        )
    assert path.read_bytes() == first_bytes


def test_plan_publication_race_has_one_conflict_safe_winner(tmp_path) -> None:
    from et_mainsim.shared_exposure import (
        SharedExposurePublicationError,
        publish_shared_exposure_target_plan,
        read_shared_exposure_target_plan,
    )

    path = tmp_path / "racing-plan.json"
    plans = (_plan(target_source_ids=(101,)), _plan(target_source_ids=(303,)))

    def _publish(plan):
        try:
            publish_shared_exposure_target_plan(path, plan)
        except SharedExposurePublicationError:
            return "conflict"
        return "published"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = tuple(executor.map(_publish, plans))

    assert sorted(outcomes) == ["conflict", "published"]
    assert read_shared_exposure_target_plan(path) in plans
    assert not tuple(tmp_path.glob("*.tmp"))


def test_target_plan_intrinsic_hash_detects_semantic_drift() -> None:
    from et_mainsim.shared_exposure import (
        SharedExposureContractError,
        validate_shared_exposure_target_plan,
    )

    drifted = json.loads(json.dumps(_plan()))
    drifted["source_geometry"]["geometry_truth_content_sha256"] = "b" * 64

    with pytest.raises(SharedExposureContractError, match="content_sha256"):
        validate_shared_exposure_target_plan(drifted)


def test_product_shard_path_is_content_hashed_and_path_safe(tmp_path) -> None:
    from et_mainsim.shared_exposure import shared_exposure_product_shard_path

    plan_hash = _plan()["content_sha256"]
    first = shared_exposure_product_shard_path(
        tmp_path,
        plan_content_sha256=plan_hash,
        product_key="../../truth/../final_stamp",
    )
    second = shared_exposure_product_shard_path(
        tmp_path,
        plan_content_sha256=plan_hash,
        product_key="final_stamp",
    )

    assert first != second
    assert first.suffix == ".h5"
    assert first.resolve().is_relative_to(tmp_path.resolve())
    assert ".." not in first.relative_to(tmp_path).parts
    assert "truth" not in first.name
    assert "final_stamp" not in first.name
    assert len(first.stem) == 64


def test_frame_completion_records_storage_guards_and_fixed_negative_claims(
    tmp_path,
) -> None:
    from et_mainsim.shared_exposure import (
        FRAME_COMPLETION_SCHEMA_ID,
        validate_shared_exposure_frame_completion,
    )

    marker, _, plan, _, shards = _completion(tmp_path)

    assert marker["schema_id"] == FRAME_COMPLETION_SCHEMA_ID
    assert marker["frame"] == {"detector_id": "main-1", "frame_index": 17}
    assert marker["mode"] == "single_parent_full_frame_then_shared_crops"
    assert marker["parent_storage_guard"]["scope"] == ("storage_resume_guard_only")
    assert marker["parent_storage_guard"]["scientific_lineage"] == (
        "not_scientific_lineage"
    )
    assert len(marker["parent_storage_guard"]["storage_guard_sha256"]) == 64
    assert marker["plan"]["content_sha256"] == plan["content_sha256"]
    assert [item["product_key"] for item in marker["shards"]] == sorted(shards)
    assert marker["upstream_negative_claims"] == {
        "independent_stamp_simulation": False,
        "lineage_claimed": False,
        "parent_content_hash_status": "not_available",
        "parent_content_identity_status": "not_available",
        "target_association_status": "not_verified_against_parent_truth",
        "truth_transfer_status": "not_transferred_source_axis",
        "zero_new_rng_draws": True,
    }
    assert (
        validate_shared_exposure_frame_completion(
            marker,
            reference_root=tmp_path,
        )
        == marker
    )


def test_frame_completion_storage_guard_cache_hashes_shared_files_once_per_batch(
    tmp_path,
    monkeypatch,
) -> None:
    from et_mainsim.shared_exposure import (
        SharedExposureStorageGuardCache,
        build_shared_exposure_frame_completion,
        publish_shared_exposure_frame_completion,
        read_shared_exposure_frame_completion,
    )

    plan_path, _, first_parent_path, shards = _completion_inputs(tmp_path)
    second_parent_path = tmp_path / "parent-frame-2.npy"
    second_parent_path.write_bytes(b"second-parent-frame-storage-bytes")
    tracked_paths = {
        first_parent_path.resolve(),
        second_parent_path.resolve(),
        *(path.resolve() for path in shards.values()),
    }
    read_open_counts = {path: 0 for path in tracked_paths}
    original_open = Path.open

    def _counting_open(path, mode="r", *args, **kwargs):
        resolved = path.resolve()
        if mode == "rb" and resolved in read_open_counts:
            read_open_counts[resolved] += 1
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", _counting_open)

    uncached_marker = build_shared_exposure_frame_completion(
        frame_index=16,
        detector_id="main-1",
        mode="single_parent_full_frame_then_shared_crops",
        reference_root=tmp_path,
        parent_path=first_parent_path,
        plan_path=plan_path,
        product_shards=shards,
    )
    uncached_marker_path = tmp_path / "frame-000016.complete.json"
    publish_shared_exposure_frame_completion(
        uncached_marker_path,
        uncached_marker,
        reference_root=tmp_path,
    )
    assert (
        read_shared_exposure_frame_completion(
            uncached_marker_path,
            reference_root=tmp_path,
        )
        == uncached_marker
    )
    assert read_open_counts[first_parent_path.resolve()] == 4
    assert {read_open_counts[path.resolve()] for path in shards.values()} == {4}
    for path in read_open_counts:
        read_open_counts[path] = 0

    storage_guard_cache = SharedExposureStorageGuardCache()

    for frame_index, parent_path in enumerate(
        (first_parent_path, second_parent_path),
        start=17,
    ):
        marker = build_shared_exposure_frame_completion(
            frame_index=frame_index,
            detector_id="main-1",
            mode="single_parent_full_frame_then_shared_crops",
            reference_root=tmp_path,
            parent_path=parent_path,
            plan_path=plan_path,
            product_shards=shards,
            storage_guard_cache=storage_guard_cache,
        )
        marker_path = tmp_path / f"frame-{frame_index:06d}.complete.json"
        publish_shared_exposure_frame_completion(
            marker_path,
            marker,
            reference_root=tmp_path,
            storage_guard_cache=storage_guard_cache,
        )
        assert (
            read_shared_exposure_frame_completion(
                marker_path,
                reference_root=tmp_path,
                storage_guard_cache=storage_guard_cache,
            )
            == marker
        )

    assert read_open_counts[first_parent_path.resolve()] == 1
    assert read_open_counts[second_parent_path.resolve()] == 1
    assert {read_open_counts[path.resolve()] for path in shards.values()} == {1}


def test_frame_completion_storage_guard_cache_rehashes_stat_changed_file(
    tmp_path,
    monkeypatch,
) -> None:
    from et_mainsim.shared_exposure import (
        SharedExposureReferenceDriftError,
        SharedExposureStorageGuardCache,
        build_shared_exposure_frame_completion,
        validate_shared_exposure_frame_completion,
    )

    plan_path, _, parent_path, shards = _completion_inputs(tmp_path)
    shard_path = next(iter(shards.values()))
    original = shard_path.read_bytes()
    shard_read_opens = 0
    original_open = Path.open

    def _counting_open(path, mode="r", *args, **kwargs):
        nonlocal shard_read_opens
        if mode == "rb" and path.resolve() == shard_path.resolve():
            shard_read_opens += 1
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", _counting_open)
    storage_guard_cache = SharedExposureStorageGuardCache()
    marker = build_shared_exposure_frame_completion(
        frame_index=17,
        detector_id="main-1",
        mode="single_parent_full_frame_then_shared_crops",
        reference_root=tmp_path,
        parent_path=parent_path,
        plan_path=plan_path,
        product_shards=shards,
        storage_guard_cache=storage_guard_cache,
    )
    assert shard_read_opens == 1

    before = shard_path.stat()
    replacement = bytes([original[0] ^ 0xFF]) + original[1:]
    shard_path.write_bytes(replacement)
    os.utime(
        shard_path,
        ns=(before.st_atime_ns, before.st_mtime_ns),
    )
    after = shard_path.stat()
    assert after.st_size == before.st_size
    assert after.st_mtime_ns == before.st_mtime_ns
    assert (after.st_dev, after.st_ino, after.st_ctime_ns) != (
        before.st_dev,
        before.st_ino,
        before.st_ctime_ns,
    )

    with pytest.raises(SharedExposureReferenceDriftError, match="drift"):
        validate_shared_exposure_frame_completion(
            marker,
            reference_root=tmp_path,
            storage_guard_cache=storage_guard_cache,
        )
    assert shard_read_opens == 2


def test_frame_completion_accepts_root_bounded_nested_worker_shards(
    tmp_path,
) -> None:
    from et_mainsim.shared_exposure import (
        build_shared_exposure_frame_completion,
        shared_exposure_product_shard_path,
        validate_shared_exposure_frame_completion,
    )

    plan_path, plan = _published_plan(tmp_path)
    parent_path = tmp_path / "parent-frame.npy"
    parent_path.write_bytes(b"parent-frame-storage-bytes")
    worker_directory = tmp_path / "shared_exposure" / "shards" / "worker_0000"
    worker_directory.mkdir(parents=True)
    product_key = "final_stamp"
    hashed_name = shared_exposure_product_shard_path(
        worker_directory,
        plan_content_sha256=plan["content_sha256"],
        product_key=product_key,
    ).name
    shard_path = worker_directory / hashed_name
    shard_path.write_bytes(b"nested-worker-shard")

    marker = build_shared_exposure_frame_completion(
        frame_index=17,
        detector_id="main-1",
        mode="single_parent_full_frame_then_shared_crops",
        reference_root=tmp_path,
        parent_path=parent_path,
        plan_path=plan_path,
        product_shards={product_key: shard_path},
    )

    assert marker["shards"][0]["path"] == shard_path.relative_to(tmp_path).as_posix()
    assert (
        validate_shared_exposure_frame_completion(
            marker,
            reference_root=tmp_path,
        )
        == marker
    )


def test_frame_completion_rejects_unhashed_and_outside_shard_paths(
    tmp_path,
) -> None:
    from et_mainsim.shared_exposure import (
        SharedExposureContractError,
        build_shared_exposure_frame_completion,
        shared_exposure_product_shard_path,
    )

    plan_path, plan = _published_plan(tmp_path)
    parent_path = tmp_path / "parent-frame.npy"
    parent_path.write_bytes(b"parent-frame-storage-bytes")

    unhashed = tmp_path / "shared_exposure" / "worker_0000" / "final_stamp.h5"
    unhashed.parent.mkdir(parents=True)
    unhashed.write_bytes(b"unhashed")
    with pytest.raises(SharedExposureContractError, match="hashed shard name"):
        build_shared_exposure_frame_completion(
            frame_index=17,
            detector_id="main-1",
            mode="shared",
            reference_root=tmp_path,
            parent_path=parent_path,
            plan_path=plan_path,
            product_shards={"final_stamp": unhashed},
        )

    outside_directory = tmp_path.parent / f"{tmp_path.name}-outside-shards"
    outside_directory.mkdir()
    outside = (
        outside_directory
        / shared_exposure_product_shard_path(
            outside_directory,
            plan_content_sha256=plan["content_sha256"],
            product_key="final_stamp",
        ).name
    )
    outside.write_bytes(b"outside")
    try:
        with pytest.raises(SharedExposureContractError, match="reference_root"):
            build_shared_exposure_frame_completion(
                frame_index=17,
                detector_id="main-1",
                mode="shared",
                reference_root=tmp_path,
                parent_path=parent_path,
                plan_path=plan_path,
                product_shards={"final_stamp": outside},
            )
    finally:
        outside.unlink(missing_ok=True)
        outside_directory.rmdir()


def test_committed_marker_rehashes_every_reference_and_fails_closed_on_drift(
    tmp_path,
) -> None:
    from et_mainsim.shared_exposure import (
        SharedExposureReferenceDriftError,
        publish_shared_exposure_frame_completion,
        read_shared_exposure_frame_completion,
    )

    marker, plan_path, _, parent_path, shards = _completion(tmp_path)
    marker_path = tmp_path / "frame-000017.complete.json"
    publish_shared_exposure_frame_completion(
        marker_path,
        marker,
        reference_root=tmp_path,
    )
    assert (
        read_shared_exposure_frame_completion(
            marker_path,
            reference_root=tmp_path,
        )
        == marker
    )

    for path in (parent_path, plan_path, *shards.values()):
        original = path.read_bytes()
        path.write_bytes(original + b"drift")
        with pytest.raises(SharedExposureReferenceDriftError, match="drift"):
            read_shared_exposure_frame_completion(
                marker_path,
                reference_root=tmp_path,
            )
        path.write_bytes(original)


def test_marker_publication_is_identical_idempotent_and_conflict_safe(tmp_path) -> None:
    from et_mainsim.shared_exposure import (
        SharedExposurePublicationError,
        build_shared_exposure_frame_completion,
        publish_shared_exposure_frame_completion,
    )

    marker, plan_path, _, parent_path, shards = _completion(tmp_path)
    marker_path = tmp_path / "frame.complete.json"
    publish_shared_exposure_frame_completion(
        marker_path,
        marker,
        reference_root=tmp_path,
    )
    first_bytes = marker_path.read_bytes()
    publish_shared_exposure_frame_completion(
        marker_path,
        marker,
        reference_root=tmp_path,
    )
    assert marker_path.read_bytes() == first_bytes

    conflicting = build_shared_exposure_frame_completion(
        frame_index=17,
        detector_id="main-1",
        mode="conflicting-mode",
        reference_root=tmp_path,
        parent_path=parent_path,
        plan_path=plan_path,
        product_shards=shards,
    )
    with pytest.raises(SharedExposurePublicationError, match="conflict"):
        publish_shared_exposure_frame_completion(
            marker_path,
            conflicting,
            reference_root=tmp_path,
        )
    assert marker_path.read_bytes() == first_bytes


def test_marker_publication_race_has_one_conflict_safe_winner(tmp_path) -> None:
    from et_mainsim.shared_exposure import (
        SharedExposurePublicationError,
        build_shared_exposure_frame_completion,
        publish_shared_exposure_frame_completion,
        read_shared_exposure_frame_completion,
    )

    marker, plan_path, _, parent_path, shards = _completion(tmp_path)
    second = build_shared_exposure_frame_completion(
        frame_index=17,
        detector_id="main-1",
        mode="second-valid-mode",
        reference_root=tmp_path,
        parent_path=parent_path,
        plan_path=plan_path,
        product_shards=shards,
    )
    marker_path = tmp_path / "racing-frame.complete.json"

    def _publish(candidate):
        try:
            publish_shared_exposure_frame_completion(
                marker_path,
                candidate,
                reference_root=tmp_path,
            )
        except SharedExposurePublicationError:
            return "conflict"
        return "published"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = tuple(executor.map(_publish, (marker, second)))

    assert sorted(outcomes) == ["conflict", "published"]
    assert read_shared_exposure_frame_completion(
        marker_path,
        reference_root=tmp_path,
    ) in (marker, second)
    assert not tuple(tmp_path.glob("*.tmp"))


def test_marker_validation_rejects_bool_int_missing_and_outside_references(
    tmp_path,
) -> None:
    from et_mainsim.shared_exposure import (
        SharedExposureContractError,
        SharedExposureReferenceDriftError,
        build_shared_exposure_frame_completion,
        validate_shared_exposure_frame_completion,
    )

    marker, plan_path, _, parent_path, shards = _completion(tmp_path)
    mutated = {**marker, "frame": {**marker["frame"], "frame_index": True}}
    with pytest.raises(SharedExposureContractError, match="frame_index"):
        validate_shared_exposure_frame_completion(
            mutated,
            reference_root=tmp_path,
        )

    mutated_size = {
        **marker,
        "parent_storage_guard": {
            **marker["parent_storage_guard"],
            "size_bytes": True,
        },
    }
    with pytest.raises(SharedExposureContractError, match="size_bytes"):
        validate_shared_exposure_frame_completion(
            mutated_size,
            reference_root=tmp_path,
        )

    next(iter(shards.values())).unlink()
    with pytest.raises(SharedExposureReferenceDriftError, match="missing"):
        validate_shared_exposure_frame_completion(
            marker,
            reference_root=tmp_path,
        )

    outside = tmp_path.parent / "outside-parent.bin"
    outside.write_bytes(b"outside")
    try:
        with pytest.raises(SharedExposureContractError, match="reference_root"):
            build_shared_exposure_frame_completion(
                frame_index=17,
                detector_id="main-1",
                mode="shared",
                reference_root=tmp_path,
                parent_path=outside,
                plan_path=plan_path,
                product_shards=shards,
            )
    finally:
        outside.unlink(missing_ok=True)


def test_marker_content_hash_detects_marker_drift(tmp_path) -> None:
    from et_mainsim.shared_exposure import (
        SharedExposureContractError,
        publish_shared_exposure_frame_completion,
        read_shared_exposure_frame_completion,
    )

    marker, _, _, _, _ = _completion(tmp_path)
    marker_path = tmp_path / "drifted-marker.json"
    publish_shared_exposure_frame_completion(
        marker_path,
        marker,
        reference_root=tmp_path,
    )
    payload = json.loads(marker_path.read_text(encoding="utf-8"))
    payload["mode"] = "drifted"
    marker_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SharedExposureContractError, match="content_sha256"):
        read_shared_exposure_frame_completion(
            marker_path,
            reference_root=tmp_path,
        )


def test_exact_array_helpers_compare_shape_dtype_and_c_order_bytes() -> None:
    from photsim7.stamp_products import StampWindow

    from et_mainsim.shared_exposure import (
        SharedExposureArrayMismatchError,
        array_c_order_fingerprint,
        assert_exact_array_match,
        assert_exact_parent_crop,
    )

    expected = np.arange(12, dtype=np.float32).reshape(3, 4)
    fortran = np.asfortranarray(expected)
    assert array_c_order_fingerprint(expected) == array_c_order_fingerprint(fortran)
    assert_exact_array_match(expected, fortran)

    with pytest.raises(SharedExposureArrayMismatchError, match="dtype"):
        assert_exact_array_match(expected, expected.astype(np.float64))
    with pytest.raises(SharedExposureArrayMismatchError, match="shape"):
        assert_exact_array_match(expected, expected[:, :3])
    drifted = expected.copy()
    drifted[1, 2] += np.float32(1)
    with pytest.raises(SharedExposureArrayMismatchError, match="C-order bytes"):
        assert_exact_array_match(expected, drifted)

    parent = np.arange(20, dtype=np.uint16).reshape(4, 5)
    window = StampWindow(
        x_start_detector_pix=-1,
        y_start_detector_pix=1,
        shape=(3, 4),
        detector_shape=parent.shape,
        target_x_detector_pix=0.25,
        target_y_detector_pix=2.0,
    )
    crop = np.zeros(window.shape, dtype=parent.dtype)
    crop[window.insertion_slices] = parent[1:4, 0:3]
    assert_exact_parent_crop(parent, crop, window)
    crop[0, 1] += 1
    with pytest.raises(SharedExposureArrayMismatchError, match="C-order bytes"):
        assert_exact_parent_crop(parent, crop, window)


def test_storage_guard_cache_is_part_of_the_public_module_contract() -> None:
    import et_mainsim.shared_exposure as shared_exposure

    assert "SharedExposureStorageGuardCache" in shared_exposure.__all__
    assert "TARGET_PLAN_SCHEMA_VERSION" in shared_exposure.__all__
    assert "FRAME_COMPLETION_SCHEMA_VERSION" in shared_exposure.__all__
