from __future__ import annotations

import errno
import hashlib
import json
import os
from pathlib import Path

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


def _make_staged_shard(
    tmp_path,
    *,
    production_schema_id="et_mainsim.stamp_science_production.v1",
    production_schema_version=1,
    legacy_galaxy_provenance=False,
    partial_generic_provenance=False,
):
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
    time_plan_path = time_plan.write_manifest(inputs_root / "time_shards.json")
    from et_mainsim.stamp_inputs import file_identity

    production_manifest = tmp_path / "production_manifest.json"
    production_manifest.write_text(
        json.dumps(
            {
                "schema_id": production_schema_id,
                "schema_version": production_schema_version,
                "run_id": "staged-fixture",
                "delivery": {
                    "execution_mode": "staged_local_scratch_v1",
                    "time_plan_relative_path": "inputs/time_shards.json",
                    "time_plan_identity": file_identity(time_plan_path),
                },
            }
        ),
        encoding="utf-8",
    )

    production_manifest_identity = file_identity(production_manifest)
    if legacy_galaxy_provenance:
        manifest_reference = {
            "galaxy_production_manifest": str(production_manifest.resolve()),
            "galaxy_production_manifest_identity": {
                "sha256": production_manifest_identity["sha256"],
                "size_bytes": production_manifest_identity["size_bytes"],
            },
        }
    else:
        manifest_reference = {
            "production_manifest": str(production_manifest.resolve()),
            "production_manifest_identity": {
                "sha256": production_manifest_identity["sha256"],
                "size_bytes": production_manifest_identity["size_bytes"],
            },
        }
    if partial_generic_provenance:
        manifest_reference["production_manifest"] = str(
            production_manifest.resolve()
        )
    request = IndependentStampShardRequest(
        output_root=staged_case_root,
        target_source_id=42,
        stamp_shape=(3, 5),
        shard=shard,
        gain_e_per_dn=4.83,
        manifest={
            "run_id": "staged-fixture",
            "case": "injected",
            **manifest_reference,
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


def test_publish_staged_shard_accepts_legacy_provenance_only_for_galaxy_v3(
    tmp_path,
) -> None:
    from et_mainsim.galaxy_stamp_production import (
        GALAXY_STAMP_PRODUCTION_SCHEMA_ID,
        GALAXY_STAMP_PRODUCTION_SCHEMA_VERSION,
    )
    from et_mainsim.staged_stamp_delivery import (
        StagedStampShardPublishRequest,
        publish_staged_independent_stamp_shard,
    )

    production_manifest, staged_case_root, shard = _make_staged_shard(
        tmp_path,
        production_schema_id=GALAXY_STAMP_PRODUCTION_SCHEMA_ID,
        production_schema_version=GALAXY_STAMP_PRODUCTION_SCHEMA_VERSION,
        legacy_galaxy_provenance=True,
    )
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

    assert result.final_shard_root.is_dir()


def test_publish_staged_shard_rejects_galaxy_provenance_on_generic_campaign(
    tmp_path,
) -> None:
    from et_mainsim.staged_stamp_delivery import (
        StagedStampShardPublishError,
        StagedStampShardPublishRequest,
        publish_staged_independent_stamp_shard,
    )

    production_manifest, staged_case_root, shard = _make_staged_shard(
        tmp_path,
        legacy_galaxy_provenance=True,
    )
    formal_case_root = production_manifest.parent / "cases" / "injected"

    with pytest.raises(
        StagedStampShardPublishError,
        match="generic production requires caller_manifest.production_manifest",
    ):
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


def test_publish_staged_shard_does_not_fallback_from_partial_generic_provenance(
    tmp_path,
) -> None:
    from et_mainsim.galaxy_stamp_production import (
        GALAXY_STAMP_PRODUCTION_SCHEMA_ID,
        GALAXY_STAMP_PRODUCTION_SCHEMA_VERSION,
    )
    from et_mainsim.staged_stamp_delivery import (
        StagedStampShardPublishError,
        StagedStampShardPublishRequest,
        publish_staged_independent_stamp_shard,
    )

    production_manifest, staged_case_root, shard = _make_staged_shard(
        tmp_path,
        production_schema_id=GALAXY_STAMP_PRODUCTION_SCHEMA_ID,
        production_schema_version=GALAXY_STAMP_PRODUCTION_SCHEMA_VERSION,
        legacy_galaxy_provenance=True,
        partial_generic_provenance=True,
    )
    formal_case_root = production_manifest.parent / "cases" / "injected"

    with pytest.raises(
        StagedStampShardPublishError,
        match="incomplete generic production manifest reference",
    ):
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
    receipt_path = result.final_shard_root / "publication_receipt.json"
    assert receipt_path.is_file()
    assert not receipt_path.is_symlink()
    assert not (
        staged_case_root
        / "stamps"
        / "target_42"
        / "delivery"
        / "shard_00000"
        / "publication_receipt.json"
    ).exists()
    assert not list(result.final_shard_root.parent.glob(".shard_00000.*"))
    raw = read_stamp_delivery_bundle(result.final_shard_root / "raw.h5")
    assert raw.manifest["caller_manifest"]["run_id"] == "staged-fixture"
    np.testing.assert_array_equal(raw.raw_frame_start_index, [12, 13, 14, 15, 16, 17])
    assert result.member_sha256["raw.h5"].startswith("sha256:")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert set(receipt) == {
        "schema_id",
        "schema_version",
        "complete",
        "run_id",
        "case",
        "target_source_id_int64",
        "shard",
        "production_manifest",
        "members",
    }
    assert receipt["schema_id"] == "et_mainsim.stamp_shard_publication_receipt.v1"
    assert receipt["schema_version"] == 1
    assert receipt["complete"] is True
    assert receipt["run_id"] == "staged-fixture"
    assert receipt["case"] == "injected"
    assert receipt["target_source_id_int64"] == 42
    assert receipt["shard"] == {
        "shard_id": 0,
        "raw_start_index": 12,
        "raw_stop_index": 18,
        "coadd_sizes": [3, 6],
        "raw_exposure_seconds": 10.0,
    }
    manifest_raw = production_manifest.read_bytes()
    assert receipt["production_manifest"] == {
        "path_relative_to_run_root": "production_manifest.json",
        "size_bytes": len(manifest_raw),
        "sha256": hashlib.sha256(manifest_raw).hexdigest(),
    }
    expected_names = {"raw.h5", "coadd_30s.h5", "coadd_60s.h5"}
    assert set(receipt["members"]) == expected_names
    for name in expected_names:
        member = result.final_shard_root / name
        raw_bytes = member.read_bytes()
        assert receipt["members"][name] == {
            "path_relative_to_run_root": (
                f"cases/injected/stamps/target_42/delivery/shard_00000/{name}"
            ),
            "size_bytes": len(raw_bytes),
            "sha256": hashlib.sha256(raw_bytes).hexdigest(),
        }


def test_publish_staged_shard_fails_atomically_if_receipt_readback_fails(
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

    def _fail_readback(*args, **kwargs):
        raise staged_delivery.StagedStampShardPublishError(
            "synthetic publication receipt readback failure"
        )

    monkeypatch.setattr(
        staged_delivery,
        "_validate_publication_receipt_readback",
        _fail_readback,
    )

    with pytest.raises(
        staged_delivery.StagedStampShardPublishError,
        match="synthetic publication receipt readback failure",
    ):
        staged_delivery.publish_staged_independent_stamp_shard(request)

    final_parent = formal_case_root / "stamps" / "target_42" / "delivery"
    assert not (final_parent / "shard_00000").exists()
    assert not list(final_parent.glob(".shard_00000.*"))


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


def test_publish_staged_shard_never_replaces_a_racing_final_path(
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
    competing_target = tmp_path / "competing-final"
    competing_target.mkdir()
    original_publish = staged_delivery._atomic_publish_directory_noreplace

    def _race(source, destination):
        destination.symlink_to(competing_target, target_is_directory=True)
        return original_publish(source, destination)

    monkeypatch.setattr(
        staged_delivery,
        "_atomic_publish_directory_noreplace",
        _race,
    )

    with pytest.raises(FileExistsError, match="already exists"):
        staged_delivery.publish_staged_independent_stamp_shard(request)

    assert request.final_shard_root.is_symlink()
    assert request.final_shard_root.resolve() == competing_target.resolve()
    assert list(competing_target.iterdir()) == []
    assert not list(request.final_shard_root.parent.glob(".shard_00000.*"))


def test_atomic_publish_fails_closed_when_plain_rename_replaces_empty_directory(
    tmp_path,
    monkeypatch,
) -> None:
    import et_mainsim.staged_stamp_delivery as staged_delivery

    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    (source / "complete").write_text("payload", encoding="utf-8")

    def unsupported_noreplace(_source, _destination):
        raise OSError(errno.EINVAL, "RENAME_NOREPLACE unsupported")

    monkeypatch.setattr(
        staged_delivery,
        "_renameat2_noreplace",
        unsupported_noreplace,
    )

    with pytest.raises(
        staged_delivery.StagedStampShardPublishError,
        match="plain directory rename does not provide no-replace semantics",
    ):
        staged_delivery._atomic_publish_directory_noreplace(source, destination)

    assert (source / "complete").read_text(encoding="utf-8") == "payload"
    assert not destination.exists()
    assert not list(tmp_path.glob(".rename-noreplace-probe-*"))


def test_atomic_publish_uses_capability_checked_plain_rename_fallback(
    tmp_path,
    monkeypatch,
) -> None:
    import et_mainsim.staged_stamp_delivery as staged_delivery

    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    (source / "complete").write_text("payload", encoding="utf-8")

    monkeypatch.setattr(
        staged_delivery,
        "_renameat2_noreplace",
        lambda *_args: (_ for _ in ()).throw(
            OSError(errno.EINVAL, "RENAME_NOREPLACE unsupported")
        ),
    )
    monkeypatch.setattr(
        staged_delivery,
        "_plain_directory_rename_is_noreplace",
        lambda _parent: True,
    )

    staged_delivery._atomic_publish_directory_noreplace(source, destination)

    assert not source.exists()
    assert (destination / "complete").read_text(encoding="utf-8") == "payload"


def test_atomic_plain_rename_fallback_preserves_a_racing_destination(
    tmp_path,
    monkeypatch,
) -> None:
    import et_mainsim.staged_stamp_delivery as staged_delivery

    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    (source / "source-marker").write_text("source", encoding="utf-8")
    (destination / "owner-marker").write_text("owner", encoding="utf-8")
    original_rename = os.rename

    monkeypatch.setattr(
        staged_delivery,
        "_renameat2_noreplace",
        lambda *_args: (_ for _ in ()).throw(
            OSError(errno.EINVAL, "RENAME_NOREPLACE unsupported")
        ),
    )
    monkeypatch.setattr(
        staged_delivery,
        "_plain_directory_rename_is_noreplace",
        lambda _parent: True,
    )

    def beegfs_rename(source_path, destination_path):
        if Path(destination_path) == destination:
            raise OSError(errno.EEXIST, "destination exists")
        return original_rename(source_path, destination_path)

    monkeypatch.setattr(staged_delivery.os, "rename", beegfs_rename)

    with pytest.raises(FileExistsError, match="already exists"):
        staged_delivery._atomic_publish_directory_noreplace(source, destination)

    assert (source / "source-marker").read_text(encoding="utf-8") == "source"
    assert (destination / "owner-marker").read_text(encoding="utf-8") == "owner"


def test_publish_staged_shard_rejects_a_symlinked_formal_parent_component(
    tmp_path,
) -> None:
    from et_mainsim.staged_stamp_delivery import (
        StagedStampShardPublishError,
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
    formal_case_root.mkdir(parents=True)
    outside_stamps = tmp_path / "outside-stamps"
    outside_stamps.mkdir()
    (formal_case_root / "stamps").symlink_to(
        outside_stamps,
        target_is_directory=True,
    )

    with pytest.raises(StagedStampShardPublishError, match="symbolic link"):
        publish_staged_independent_stamp_shard(request)

    assert list(outside_stamps.iterdir()) == []


def test_publish_staged_shard_rejects_a_different_production_manifest(tmp_path) -> None:
    from et_mainsim.staged_stamp_delivery import (
        StagedStampShardPublishRequest,
        StagedStampShardPublishError,
        publish_staged_independent_stamp_shard,
    )

    production_manifest, staged_case_root, shard = _make_staged_shard(tmp_path)
    original_payload = json.loads(production_manifest.read_text(encoding="utf-8"))
    different_manifest = tmp_path / "different_manifest.json"
    different_manifest.write_text(
        json.dumps(
            {
                "run_id": "different",
                "delivery": original_payload["delivery"],
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


def test_publish_api_rejects_a_shard_that_differs_from_the_frozen_plan(
    tmp_path,
) -> None:
    from et_mainsim.staged_stamp_delivery import (
        StagedStampShardPublishError,
        StagedStampShardPublishRequest,
        publish_staged_independent_stamp_shard,
    )
    from et_mainsim.time_shards import ContinuousTimeShard

    production_manifest, staged_case_root, _ = _make_staged_shard(tmp_path)
    formal_case_root = production_manifest.parent / "cases" / "injected"
    wrong_shard = ContinuousTimeShard(
        shard_id=0,
        raw_start_index=18,
        raw_stop_index=24,
        coadd_sizes=(3, 6),
        raw_exposure_seconds=10.0,
    )

    with pytest.raises(
        StagedStampShardPublishError,
        match="does not match the frozen production time-shard plan",
    ):
        publish_staged_independent_stamp_shard(
            StagedStampShardPublishRequest(
                staged_case_root=staged_case_root,
                formal_case_root=formal_case_root,
                production_manifest_path=production_manifest,
                target_source_id=42,
                shard=wrong_shard,
                case="injected",
            )
        )

    assert not (
        formal_case_root / "stamps" / "target_42" / "delivery" / "shard_00000"
    ).exists()


def test_publish_api_rejects_changed_frozen_time_plan_identity(tmp_path) -> None:
    from et_mainsim.staged_stamp_delivery import (
        StagedStampShardPublishError,
        StagedStampShardPublishRequest,
        publish_staged_independent_stamp_shard,
    )

    production_manifest, staged_case_root, shard = _make_staged_shard(tmp_path)
    formal_case_root = production_manifest.parent / "cases" / "injected"
    time_plan_path = production_manifest.parent / "inputs" / "time_shards.json"
    time_plan_path.write_bytes(time_plan_path.read_bytes() + b"\n")

    with pytest.raises(
        StagedStampShardPublishError,
        match="time shard plan identity changed after production preparation",
    ):
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


def test_publish_api_checks_time_plan_identity_before_parsing_it(tmp_path) -> None:
    from et_mainsim.staged_stamp_delivery import (
        StagedStampShardPublishError,
        StagedStampShardPublishRequest,
        publish_staged_independent_stamp_shard,
    )

    production_manifest, staged_case_root, shard = _make_staged_shard(tmp_path)
    formal_case_root = production_manifest.parent / "cases" / "injected"
    time_plan_path = production_manifest.parent / "inputs" / "time_shards.json"
    time_plan_path.write_text("not a JSON time plan", encoding="utf-8")

    with pytest.raises(
        StagedStampShardPublishError,
        match="time shard plan identity changed after production preparation",
    ):
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


def test_frozen_time_shard_uses_one_plan_byte_snapshot(tmp_path, monkeypatch) -> None:
    import et_mainsim.staged_stamp_delivery as staged_delivery
    from et_mainsim.time_shards import ContinuousTimeShard, ContinuousTimeShardPlan

    production_manifest, _, shard = _make_staged_shard(tmp_path)
    time_plan_path = production_manifest.parent / "inputs" / "time_shards.json"
    original_time_plan_bytes = time_plan_path.read_bytes()
    replacement_plan = ContinuousTimeShardPlan(
        raw_start_index=18,
        raw_stop_index=24,
        accepted_raw_start_index=18,
        accepted_raw_stop_index=24,
        coadd_sizes=(3, 6),
        raw_exposure_seconds=10.0,
        max_raw_frames_per_shard=6,
        shards=(
            ContinuousTimeShard(
                shard_id=0,
                raw_start_index=18,
                raw_stop_index=24,
                coadd_sizes=(3, 6),
                raw_exposure_seconds=10.0,
            ),
        ),
    )
    replacement_path = replacement_plan.write_manifest(tmp_path / "replacement.json")
    replacement_time_plan_bytes = replacement_path.read_bytes()

    def _return_snapshot_then_replace(path, *, label):
        source = Path(path)
        if source == time_plan_path:
            time_plan_path.write_bytes(replacement_time_plan_bytes)
            return original_time_plan_bytes
        return source.read_bytes()

    monkeypatch.setattr(
        staged_delivery,
        "_read_frozen_file_bytes",
        _return_snapshot_then_replace,
    )

    assert staged_delivery._load_frozen_time_shard(
        production_manifest,
        shard_id=0,
    ) == shard


def test_publish_rejects_frozen_input_drift_after_copy_before_rename(
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
    time_plan_path = production_manifest.parent / "inputs" / "time_shards.json"
    original_copy = staged_delivery._copy_members_and_verify

    def _copy_then_drift(source_root, destination_root, *, shard):
        result = original_copy(source_root, destination_root, shard=shard)
        time_plan_path.write_bytes(time_plan_path.read_bytes() + b"\n")
        return result

    monkeypatch.setattr(
        staged_delivery,
        "_copy_members_and_verify",
        _copy_then_drift,
    )

    with pytest.raises(
        staged_delivery.StagedStampShardPublishError,
        match="frozen publication inputs changed before formal publication",
    ):
        staged_delivery.publish_staged_independent_stamp_shard(request)

    final_parent = formal_case_root / "stamps" / "target_42" / "delivery"
    assert not (final_parent / "shard_00000").exists()
    assert not list(final_parent.glob(".shard_00000.*"))


def test_publish_rejects_frozen_manifest_drift_after_copy_before_rename(
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
    original_copy = staged_delivery._copy_members_and_verify

    def _copy_then_drift(source_root, destination_root, *, shard):
        result = original_copy(source_root, destination_root, shard=shard)
        production_manifest.write_bytes(production_manifest.read_bytes() + b"\n")
        return result

    monkeypatch.setattr(
        staged_delivery,
        "_copy_members_and_verify",
        _copy_then_drift,
    )

    with pytest.raises(
        staged_delivery.StagedStampShardPublishError,
        match="frozen publication inputs changed before formal publication",
    ):
        staged_delivery.publish_staged_independent_stamp_shard(request)

    final_parent = formal_case_root / "stamps" / "target_42" / "delivery"
    assert not (final_parent / "shard_00000").exists()
    assert not list(final_parent.glob(".shard_00000.*"))


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


def test_publish_staged_shard_tolerates_unsupported_parent_directory_fsync(
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
    original_fsync = staged_delivery.os.fsync

    def _reject_parent_directory_fsync(descriptor):
        descriptor_target = Path(os.readlink(f"/proc/self/fd/{descriptor}")).resolve()
        if descriptor_target == final_parent:
            raise OSError(errno.EINVAL, "directory fsync is unsupported")
        return original_fsync(descriptor)

    monkeypatch.setattr(staged_delivery.os, "fsync", _reject_parent_directory_fsync)

    result = staged_delivery.publish_staged_independent_stamp_shard(request)

    assert result.final_shard_root.is_dir()
    assert result.parent_directory_fsync == "unsupported"
    assert not (final_parent / ".shard_00000.staged-publish.lock").exists()


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
