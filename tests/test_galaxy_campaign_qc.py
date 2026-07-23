from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np


RAW_EXPOSURE_SECONDS = 10.0
RUN_ID = "campaign-qc-fixture"
SOURCE_IDS = (41, 42)


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
