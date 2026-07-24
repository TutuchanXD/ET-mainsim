from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import pytest


RAW_EXPOSURE_SECONDS = 10.0
RUN_ID = "campaign-qc-fixture"
SOURCE_IDS = (41, 42)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_publication_receipt(
    manifest_path: Path,
    *,
    source_id: int,
    shard_id: int,
    overrides: dict[str, object] | None = None,
) -> Path:
    from et_mainsim.time_shards import ContinuousTimeShardPlan

    run_root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    time_plan = ContinuousTimeShardPlan.from_manifest_dict(
        json.loads(
            (run_root / manifest["delivery"]["time_plan_relative_path"]).read_text(
                encoding="utf-8"
            )
        )
    )
    shard = time_plan.shards[shard_id]
    shard_root = (
        run_root
        / "cases"
        / "injected"
        / "stamps"
        / f"target_{source_id}"
        / "delivery"
        / f"shard_{shard_id:05d}"
    )
    members = {}
    for path in sorted(shard_root.glob("*.h5")):
        members[path.name] = {
            "path_relative_to_run_root": path.relative_to(run_root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    payload: dict[str, object] = {
        "schema_id": "et_mainsim.stamp_shard_publication_receipt.v1",
        "schema_version": 1,
        "complete": True,
        "run_id": RUN_ID,
        "case": "injected",
        "target_source_id_int64": source_id,
        "shard": {
            "shard_id": shard.shard_id,
            "raw_start_index": shard.raw_start_index,
            "raw_stop_index": shard.raw_stop_index,
            "coadd_sizes": list(shard.coadd_sizes),
            "raw_exposure_seconds": shard.raw_exposure_seconds,
        },
        "production_manifest": {
            "path_relative_to_run_root": "production_manifest.json",
            "size_bytes": manifest_path.stat().st_size,
            "sha256": _sha256(manifest_path),
        },
        "members": members,
    }
    if overrides:
        payload.update(overrides)
    receipt_path = shard_root / "publication_receipt.json"
    receipt_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return receipt_path


def _write_fixture_run(tmp_path: Path) -> Path:
    """Build a tiny complete raw+coadd Galaxy delivery campaign."""

    from et_mainsim.galaxy_stamp_production import (
        GALAXY_STAMP_PRODUCTION_SCHEMA_ID,
        GALAXY_STAMP_PRODUCTION_SCHEMA_VERSION,
    )
    from et_mainsim.stamp_delivery import (
        StampDeliveryBundle,
        write_stamp_delivery_bundle,
    )
    from et_mainsim.stamp_inputs import file_identity
    from et_mainsim.time_shards import plan_continuous_time_shards

    run_root = tmp_path / "campaign"
    inputs_root = run_root / "inputs"
    inputs_root.mkdir(parents=True)
    plan = plan_continuous_time_shards(
        raw_start_index=0,
        raw_stop_index=12,
        coadd_sizes=(3, 6),
        raw_exposure_seconds=RAW_EXPOSURE_SECONDS,
        max_raw_frames_per_shard=6,
    )
    time_plan_path = plan.write_manifest(inputs_root / "time_shards.json")
    manifest_path = run_root / "production_manifest.json"
    manifest = {
        "schema_id": GALAXY_STAMP_PRODUCTION_SCHEMA_ID,
        "schema_version": GALAXY_STAMP_PRODUCTION_SCHEMA_VERSION,
        "run_id": RUN_ID,
        "run_root": str(run_root),
        "observation_product": "final_dn",
        "background_realization_delivered": False,
        "delivery": {
            "stamp_shape": [3, 4],
            "raw_exposure_seconds": RAW_EXPOSURE_SECONDS,
            "cadence_seconds": [30.0, 60.0],
            "coadd_sizes": [3, 6],
            "time_plan_relative_path": "inputs/time_shards.json",
            "time_plan_identity": file_identity(time_plan_path),
        },
        "targets": [
            {
                "source_id": str(source_id),
                "source_id_int64": source_id,
                "focalplane_mapping": {"detector_id": "main_lu"},
            }
            for source_id in SOURCE_IDS
        ],
    }
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    for source_id in SOURCE_IDS:
        for shard in plan.shards:
            for product_kind, factor, filename in (
                ("raw", 1, "raw.h5"),
                ("coadd", 3, "coadd_30s.h5"),
                ("coadd", 6, "coadd_60s.h5"),
            ):
                starts = np.arange(
                    shard.raw_start_index,
                    shard.raw_stop_index,
                    factor,
                    dtype=np.int64,
                )
                stops = starts + factor
                n_frames = starts.size
                dtype = np.uint16 if factor == 1 else np.uint64
                shape = (n_frames, 3, 4)
                bundle = StampDeliveryBundle.from_arrays(
                    product_kind=product_kind,
                    coadd_factor=factor,
                    final_dn=np.full(shape, 1024, dtype=dtype),
                    background_expectation_e=np.zeros(shape),
                    captured_flux_fraction=np.ones(n_frames),
                    captured_flux_denominator_e=np.full(n_frames, 1_000.0),
                    captured_flux_qa_pass=np.ones(n_frames, dtype=bool),
                    bias_level_sum_dn=np.zeros(n_frames),
                    column_noise_sum_dn_by_x=np.zeros((n_frames, 4)),
                    valid_mask=np.ones(shape, dtype=bool),
                    fullwell_count=np.zeros(shape, dtype=np.uint16),
                    adc_low_count=np.zeros(shape, dtype=np.uint16),
                    adc_high_count=np.zeros(shape, dtype=np.uint16),
                    cosmic_count=np.zeros(shape, dtype=np.uint16),
                    time_start_seconds=starts.astype(float) * RAW_EXPOSURE_SECONDS,
                    exposure_seconds=np.full(n_frames, factor * RAW_EXPOSURE_SECONDS),
                    raw_frame_start_index=starts,
                    raw_frame_stop_index_exclusive=stops,
                    gain_e_per_dn=np.asarray(1.4),
                    manifest={
                        "schema_id": "et_mainsim.independent_stamp_production.v1",
                        "product_kind": product_kind,
                        "coadd_factor": factor,
                        "target_source_id": str(source_id),
                        "target_source_id_int64": source_id,
                        "stamp_shape": [3, 4],
                        "time_shard": {
                            "raw_frame_interval": {
                                "start_index": shard.raw_start_index,
                                "stop_index": shard.raw_stop_index,
                            }
                        },
                        "caller_manifest": {
                            "run_id": RUN_ID,
                            "case": "injected",
                        },
                    },
                    provenance={
                        "observation_product": "final_dn",
                        "background_realization_used": False,
                    },
                )
                destination = (
                    run_root
                    / "cases"
                    / "injected"
                    / "stamps"
                    / f"target_{source_id}"
                    / "delivery"
                    / f"shard_{shard.shard_id:05d}"
                    / filename
                )
                write_stamp_delivery_bundle(destination, bundle)
    return manifest_path


def test_campaign_qc_accepts_complete_manifest_anchored_delivery(
    tmp_path: Path,
) -> None:
    from et_mainsim.galaxy_campaign_qc import (
        GalaxyCampaignDeliveryQCRequest,
        audit_galaxy_campaign_delivery_v1,
    )

    manifest_path = _write_fixture_run(tmp_path)
    result = audit_galaxy_campaign_delivery_v1(
        GalaxyCampaignDeliveryQCRequest(
            production_manifest_path=manifest_path,
            case="injected",
        )
    )

    assert result.ready is True
    assert result.expected_bundle_count == 12
    assert result.valid_bundle_count == 12
    assert result.missing_bundle_count == 0
    assert result.invalid_bundle_count == 0
    payload = result.to_dict()
    assert payload["schema_id"] == "et_mainsim.galaxy_campaign_delivery_qc.v1"
    assert payload["coverage"]["accepted_raw_frame_interval"] == {
        "start_index": 0,
        "stop_index": 12,
    }
    assert payload["products"]["raw"]["expected_bundle_count"] == 4
    assert payload["products"]["coadd_30s"]["expected_bundle_count"] == 4
    assert payload["products"]["coadd_60s"]["valid_bundle_count"] == 4


def test_campaign_qc_strictly_accepts_optional_bound_publication_receipts(
    tmp_path: Path,
) -> None:
    from et_mainsim.galaxy_campaign_qc import (
        GalaxyCampaignDeliveryQCRequest,
        audit_galaxy_campaign_delivery_v1,
    )

    manifest_path = _write_fixture_run(tmp_path)
    for source_id in SOURCE_IDS:
        for shard_id in (0, 1):
            _write_publication_receipt(
                manifest_path,
                source_id=source_id,
                shard_id=shard_id,
            )

    result = audit_galaxy_campaign_delivery_v1(
        GalaxyCampaignDeliveryQCRequest(
            production_manifest_path=manifest_path,
            case="injected",
        )
    )

    assert result.ready is True
    assert result.invalid_bundle_count == 0
    assert result.partial_artifacts == ()


@pytest.mark.parametrize(
    ("overrides", "expected_error"),
    (
        ({"case": "static"}, "case"),
        ({"run_id": "another-run"}, "run_id"),
        ({"members": {}}, "members"),
        (
            {
                "production_manifest": {
                    "path_relative_to_run_root": "production_manifest.json",
                    "size_bytes": 1,
                    "sha256": "0" * 64,
                }
            },
            "production manifest",
        ),
    ),
)
def test_campaign_qc_rejects_a_present_but_unbound_publication_receipt(
    tmp_path: Path,
    overrides: dict[str, object],
    expected_error: str,
) -> None:
    from et_mainsim.galaxy_campaign_qc import (
        GalaxyCampaignDeliveryQCRequest,
        audit_galaxy_campaign_delivery_v1,
    )

    manifest_path = _write_fixture_run(tmp_path)
    receipt_path = _write_publication_receipt(
        manifest_path,
        source_id=41,
        shard_id=0,
        overrides=overrides,
    )

    result = audit_galaxy_campaign_delivery_v1(
        GalaxyCampaignDeliveryQCRequest(
            production_manifest_path=manifest_path,
            case="injected",
        )
    )

    assert result.ready is False
    assert result.invalid_bundle_count == 1
    assert result.invalid_bundles[0]["path"] == str(receipt_path)
    assert result.invalid_bundles[0]["product"] == "publication_receipt"
    assert expected_error in result.invalid_bundles[0]["error"]


def test_campaign_qc_rejects_manifest_drift_during_the_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import et_mainsim.galaxy_campaign_qc as campaign_qc

    manifest_path = _write_fixture_run(tmp_path)
    original_load = campaign_qc._load_campaign

    def _load_then_drift(*args, **kwargs):
        result = original_load(*args, **kwargs)
        manifest_path.write_bytes(manifest_path.read_bytes() + b"\n")
        return result

    monkeypatch.setattr(campaign_qc, "_load_campaign", _load_then_drift)

    with pytest.raises(
        campaign_qc.GalaxyCampaignDeliveryQCError,
        match="production manifest changed during campaign QC",
    ):
        campaign_qc.audit_galaxy_campaign_delivery_v1(
            campaign_qc.GalaxyCampaignDeliveryQCRequest(
                production_manifest_path=manifest_path,
                case="injected",
            )
        )


def test_campaign_qc_reports_missing_and_rejects_noncanonical_time_axis(
    tmp_path: Path,
) -> None:
    from et_mainsim.galaxy_campaign_qc import (
        GalaxyCampaignDeliveryQCRequest,
        audit_galaxy_campaign_delivery_v1,
    )

    manifest_path = _write_fixture_run(tmp_path)
    run_root = manifest_path.parent
    missing = (
        run_root
        / "cases"
        / "injected"
        / "stamps"
        / "target_41"
        / "delivery"
        / "shard_00000"
        / "raw.h5"
    )
    missing.unlink()
    malformed = (
        run_root
        / "cases"
        / "injected"
        / "stamps"
        / "target_42"
        / "delivery"
        / "shard_00001"
        / "coadd_30s.h5"
    )
    with h5py.File(malformed, "r+") as handle:
        handle["raw_frame_start_index"][0] += 1

    result = audit_galaxy_campaign_delivery_v1(
        GalaxyCampaignDeliveryQCRequest(
            production_manifest_path=manifest_path,
            case="injected",
        )
    )

    assert result.ready is False
    assert result.missing_bundle_count == 1
    assert result.invalid_bundle_count == 1
    assert result.missing_bundles[0]["path"] == str(missing)
    assert "raw_frame_start_index" in result.invalid_bundles[0]["error"]


@pytest.mark.parametrize(
    ("artifact_name", "as_directory"),
    (
        (".shard_00000.transfer.incoming", True),
        (".shard_00000.staged-publish.lock", True),
        (".shard_00000.lock", False),
        ("unrecognised_delivery_sibling", True),
    ),
)
def test_campaign_qc_rejects_staged_or_direct_delivery_residue(
    tmp_path: Path,
    artifact_name: str,
    as_directory: bool,
) -> None:
    from et_mainsim.galaxy_campaign_qc import (
        GalaxyCampaignDeliveryQCRequest,
        audit_galaxy_campaign_delivery_v1,
    )

    manifest_path = _write_fixture_run(tmp_path)
    delivery_root = (
        manifest_path.parent
        / "cases"
        / "injected"
        / "stamps"
        / "target_41"
        / "delivery"
    )
    artifact = delivery_root / artifact_name
    if as_directory:
        artifact.mkdir()
    else:
        artifact.write_text("unfinished direct writer\n", encoding="utf-8")

    result = audit_galaxy_campaign_delivery_v1(
        GalaxyCampaignDeliveryQCRequest(
            production_manifest_path=manifest_path,
            case="injected",
        )
    )

    assert result.valid_bundle_count == result.expected_bundle_count == 12
    assert result.missing_bundle_count == 0
    assert result.invalid_bundle_count == 0
    assert result.ready is False
    assert str(artifact) in result.partial_artifacts


def test_campaign_qc_cli_writes_receipt_and_require_complete_is_fail_closed(
    tmp_path: Path,
) -> None:
    manifest_path = _write_fixture_run(tmp_path)
    receipt = tmp_path / "campaign_qc.json"
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "audit_galaxy_campaign_delivery.py"
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--production-manifest",
            str(manifest_path),
            "--case",
            "injected",
            "--output-json",
            str(receipt),
            "--require-complete",
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(receipt.read_text(encoding="utf-8"))["ready"] is True

    (
        manifest_path.parent
        / "cases"
        / "injected"
        / "stamps"
        / "target_41"
        / "delivery"
        / "shard_00000"
        / "raw.h5"
    ).unlink()
    incomplete = subprocess.run(
        [
            sys.executable,
            str(script),
            "--production-manifest",
            str(manifest_path),
            "--case",
            "injected",
            "--output-json",
            str(receipt),
            "--require-complete",
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert incomplete.returncode == 2
    assert json.loads(receipt.read_text(encoding="utf-8"))["ready"] is False
