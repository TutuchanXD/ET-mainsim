from __future__ import annotations

import json

import numpy as np
import pytest


def _render_raw(frame_index: int):
    from et_mainsim.independent_stamp_production import RawStampDeliveryFrame

    shape = (3, 5)
    value = frame_index + 1
    return RawStampDeliveryFrame(
        final_dn=np.full(shape, value, dtype=np.uint16),
        background_expectation_e=np.full(shape, value * 0.5, dtype=np.float64),
        bias_level_dn=float(value),
        column_noise_dn_by_x=np.full(shape[1], value * 0.25, dtype=np.float64),
        valid_mask=np.ones(shape, dtype=bool),
        fullwell_mask=np.zeros(shape, dtype=bool),
        adc_low_mask=np.zeros(shape, dtype=bool),
        adc_high_mask=np.zeros(shape, dtype=bool),
        cosmic_mask=np.zeros(shape, dtype=bool),
    )


def _make_staged_shard(tmp_path):
    from et_mainsim.independent_stamp_production import (
        IndependentStampShardRequest,
        run_independent_stamp_time_shard,
    )
    from et_mainsim.time_shards import ContinuousTimeShard

    staged_case_root = tmp_path / "local-scratch" / "injected"
    shard = ContinuousTimeShard(
        shard_id=0,
        raw_start_index=12,
        raw_stop_index=18,
        coadd_sizes=(3, 6),
        raw_exposure_seconds=10.0,
    )
    from et_mainsim.time_shards import ContinuousTimeShardPlan

    inputs_root = tmp_path / "inputs"
    inputs_root.mkdir()
    time_plan = ContinuousTimeShardPlan(
        raw_start_index=12,
        raw_stop_index=18,
        accepted_raw_start_index=12,
        accepted_raw_stop_index=18,
        coadd_sizes=(3, 6),
        raw_exposure_seconds=10.0,
        max_raw_frames_per_shard=6,
        shards=(shard,),
    )
    time_plan.write_manifest(inputs_root / "time_shards.json")
    production_manifest = tmp_path / "production_manifest.json"
    production_manifest.write_text(
        json.dumps(
            {
                "run_id": "staged-fixture",
                "delivery": {
                    "execution_mode": "staged_local_scratch_v1",
                    "time_plan_relative_path": "inputs/time_shards.json",
                },
            }
        ),
        encoding="utf-8",
    )
    from et_mainsim.stamp_inputs import file_identity

    production_manifest_identity = file_identity(production_manifest)
    request = IndependentStampShardRequest(
        output_root=staged_case_root,
        target_source_id=42,
        stamp_shape=(3, 5),
        shard=shard,
        gain_e_per_dn=4.83,
        manifest={
            "run_id": "staged-fixture",
            "case": "injected",
            "galaxy_production_manifest": str(production_manifest.resolve()),
            "galaxy_production_manifest_identity": {
                "sha256": production_manifest_identity["sha256"],
                "size_bytes": production_manifest_identity["size_bytes"],
            },
        },
        provenance={"code": "test"},
        batch_size=2,
    )
    run_independent_stamp_time_shard(
        request,
        render_raw=_render_raw,
        adapt_raw=lambda frame: frame,
    )
    return production_manifest, staged_case_root, shard


def test_publish_staged_shard_copies_verifies_and_atomically_publishes(tmp_path) -> None:
    from et_mainsim.staged_stamp_delivery import (
        StagedStampShardPublishRequest,
        publish_staged_independent_stamp_shard,
    )
    from et_mainsim.stamp_delivery import read_stamp_delivery_bundle

    production_manifest, staged_case_root, shard = _make_staged_shard(tmp_path)
    formal_case_root = production_manifest.parent / "cases" / "injected"

    result = publish_staged_independent_stamp_shard(
        StagedStampShardPublishRequest(
            staged_case_root=staged_case_root,
            formal_case_root=formal_case_root,
            production_manifest_path=production_manifest,
            target_source_id=42,
            shard=shard,
            case="injected",
        )
    )

    assert result.final_shard_root == (
        formal_case_root / "stamps" / "target_42" / "delivery" / "shard_00000"
    )
    assert result.final_shard_root.is_dir()
    assert (result.final_shard_root / "raw.h5").is_file()
    assert (result.final_shard_root / "coadd_30s.h5").is_file()
    assert (result.final_shard_root / "coadd_60s.h5").is_file()
    assert not list(result.final_shard_root.parent.glob(".shard_00000.*"))
    raw = read_stamp_delivery_bundle(result.final_shard_root / "raw.h5")
    assert raw.manifest["caller_manifest"]["run_id"] == "staged-fixture"
    np.testing.assert_array_equal(raw.raw_frame_start_index, [12, 13, 14, 15, 16, 17])
    assert result.member_sha256["raw.h5"].startswith("sha256:")


def test_publish_staged_shard_refuses_existing_final_without_modifying_it(tmp_path) -> None:
    from et_mainsim.staged_stamp_delivery import (
        StagedStampShardPublishRequest,
        publish_staged_independent_stamp_shard,
    )

    production_manifest, staged_case_root, shard = _make_staged_shard(tmp_path)
    formal_case_root = production_manifest.parent / "cases" / "injected"
    request = StagedStampShardPublishRequest(
        staged_case_root=staged_case_root,
        formal_case_root=formal_case_root,
        production_manifest_path=production_manifest,
        target_source_id=42,
        shard=shard,
        case="injected",
    )
    published = publish_staged_independent_stamp_shard(request)
    before = (published.final_shard_root / "raw.h5").read_bytes()

    with pytest.raises(FileExistsError, match="already exists"):
        publish_staged_independent_stamp_shard(request)

    assert (published.final_shard_root / "raw.h5").read_bytes() == before
    assert not list(published.final_shard_root.parent.glob(".shard_00000.*"))


def test_publish_staged_shard_rejects_a_different_production_manifest(tmp_path) -> None:
    from et_mainsim.staged_stamp_delivery import (
        StagedStampShardPublishRequest,
        StagedStampShardPublishError,
        publish_staged_independent_stamp_shard,
    )

    production_manifest, staged_case_root, shard = _make_staged_shard(tmp_path)
    different_manifest = tmp_path / "different_manifest.json"
    different_manifest.write_text(
        json.dumps(
            {
                "run_id": "different",
                "delivery": {
                    "execution_mode": "staged_local_scratch_v1",
                },
            }
        ),
        encoding="utf-8",
    )
    formal_case_root = production_manifest.parent / "cases" / "injected"

    with pytest.raises(StagedStampShardPublishError, match="run_id"):
        publish_staged_independent_stamp_shard(
            StagedStampShardPublishRequest(
                staged_case_root=staged_case_root,
                formal_case_root=formal_case_root,
                production_manifest_path=different_manifest,
                target_source_id=42,
                shard=shard,
                case="injected",
            )
        )

    assert not (formal_case_root / "stamps" / "target_42" / "delivery" / "shard_00000").exists()


@pytest.mark.parametrize(
    "execution_mode",
    (None, "direct_shared_filesystem"),
)
def test_publish_staged_shard_rejects_nonstaged_manifest_mode(
    tmp_path,
    execution_mode,
) -> None:
    from et_mainsim.staged_stamp_delivery import (
        StagedStampShardPublishError,
        StagedStampShardPublishRequest,
        publish_staged_independent_stamp_shard,
    )

    production_manifest, staged_case_root, shard = _make_staged_shard(tmp_path)
    payload = json.loads(production_manifest.read_text(encoding="utf-8"))
    if execution_mode is None:
        payload["delivery"].pop("execution_mode")
    else:
        payload["delivery"]["execution_mode"] = execution_mode
    production_manifest.write_text(json.dumps(payload), encoding="utf-8")
    formal_case_root = production_manifest.parent / "cases" / "injected"

    with pytest.raises(StagedStampShardPublishError, match="execution_mode"):
        publish_staged_independent_stamp_shard(
            StagedStampShardPublishRequest(
                staged_case_root=staged_case_root,
                formal_case_root=formal_case_root,
                production_manifest_path=production_manifest,
                target_source_id=42,
                shard=shard,
                case="injected",
            )
        )

    assert not (
        formal_case_root / "stamps" / "target_42" / "delivery" / "shard_00000"
    ).exists()


def test_publish_staged_shard_rejects_noncanonical_formal_case_root(tmp_path) -> None:
    from et_mainsim.staged_stamp_delivery import (
        StagedStampShardPublishError,
        StagedStampShardPublishRequest,
        publish_staged_independent_stamp_shard,
    )

    production_manifest, staged_case_root, shard = _make_staged_shard(tmp_path)
    canonical_case_root = production_manifest.parent / "cases" / "injected"
    other_case_root = tmp_path / "other-run" / "cases" / "injected"

    with pytest.raises(StagedStampShardPublishError, match="formal_case_root"):
        publish_staged_independent_stamp_shard(
            StagedStampShardPublishRequest(
                staged_case_root=staged_case_root,
                formal_case_root=other_case_root,
                production_manifest_path=production_manifest,
                target_source_id=42,
                shard=shard,
                case="injected",
            )
        )

    assert not (
        canonical_case_root / "stamps" / "target_42" / "delivery" / "shard_00000"
    ).exists()


def test_publish_staged_shard_rejects_changed_production_manifest_identity(
    tmp_path,
) -> None:
    from et_mainsim.staged_stamp_delivery import (
        StagedStampShardPublishError,
        StagedStampShardPublishRequest,
        publish_staged_independent_stamp_shard,
    )

    production_manifest, staged_case_root, shard = _make_staged_shard(tmp_path)
    payload = json.loads(production_manifest.read_text(encoding="utf-8"))
    payload["post_render_revision"] = "different-content-same-path"
    production_manifest.write_text(json.dumps(payload), encoding="utf-8")
    formal_case_root = production_manifest.parent / "cases" / "injected"

    with pytest.raises(StagedStampShardPublishError, match="content identity"):
        publish_staged_independent_stamp_shard(
            StagedStampShardPublishRequest(
                staged_case_root=staged_case_root,
                formal_case_root=formal_case_root,
                production_manifest_path=production_manifest,
                target_source_id=42,
                shard=shard,
                case="injected",
            )
        )

    assert not (
        formal_case_root / "stamps" / "target_42" / "delivery" / "shard_00000"
    ).exists()


def test_publish_staged_shard_removes_incoming_directory_when_copy_fails(
    tmp_path,
    monkeypatch,
) -> None:
    import et_mainsim.staged_stamp_delivery as staged_delivery

    production_manifest, staged_case_root, shard = _make_staged_shard(tmp_path)
    formal_case_root = production_manifest.parent / "cases" / "injected"
    request = staged_delivery.StagedStampShardPublishRequest(
        staged_case_root=staged_case_root,
        formal_case_root=formal_case_root,
        production_manifest_path=production_manifest,
        target_source_id=42,
        shard=shard,
        case="injected",
    )
    original_copy2 = staged_delivery.shutil.copy2
    call_count = 0

    def _fail_after_first_copy(source, destination, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise OSError("synthetic transfer failure")
        return original_copy2(source, destination, *args, **kwargs)

    monkeypatch.setattr(staged_delivery.shutil, "copy2", _fail_after_first_copy)

    with pytest.raises(OSError, match="synthetic transfer failure"):
        staged_delivery.publish_staged_independent_stamp_shard(request)

    final_parent = formal_case_root / "stamps" / "target_42" / "delivery"
    assert not (final_parent / "shard_00000").exists()
    assert not list(final_parent.glob(".shard_00000.*"))


def test_publish_staged_shard_keeps_a_qc_visible_lock_if_postrename_fsync_fails(
    tmp_path,
    monkeypatch,
) -> None:
    import et_mainsim.staged_stamp_delivery as staged_delivery

    production_manifest, staged_case_root, shard = _make_staged_shard(tmp_path)
    formal_case_root = production_manifest.parent / "cases" / "injected"
    request = staged_delivery.StagedStampShardPublishRequest(
        staged_case_root=staged_case_root,
        formal_case_root=formal_case_root,
        production_manifest_path=production_manifest,
        target_source_id=42,
        shard=shard,
        case="injected",
    )
    final_parent = formal_case_root / "stamps" / "target_42" / "delivery"
    original_fsync_directory = staged_delivery._fsync_directory

    def _fail_after_rename(path):
        if path == final_parent:
            raise OSError("synthetic post-rename directory fsync failure")
        return original_fsync_directory(path)

    monkeypatch.setattr(
        staged_delivery,
        "_fsync_directory",
        _fail_after_rename,
    )

    with pytest.raises(OSError, match="post-rename"):
        staged_delivery.publish_staged_independent_stamp_shard(request)

    assert (final_parent / "shard_00000").is_dir()
    assert (final_parent / ".shard_00000.staged-publish.lock").is_dir()


def test_publish_cli_resolves_the_frozen_time_shard_from_production_manifest(
    tmp_path,
) -> None:
    from et_mainsim.staged_stamp_delivery import main

    production_manifest, staged_case_root, _ = _make_staged_shard(tmp_path)
    formal_case_root = production_manifest.parent / "cases" / "injected"

    assert (
        main(
            [
                "publish",
                "--staged-case-root",
                str(staged_case_root),
                "--formal-case-root",
                str(formal_case_root),
                "--production-manifest",
                str(production_manifest),
                "--source-id",
                "42",
                "--shard-id",
                "0",
                "--case",
                "injected",
            ]
        )
        == 0
    )
    assert (formal_case_root / "stamps" / "target_42" / "delivery" / "shard_00000").is_dir()
