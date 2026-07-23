from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np


RAW_EXPOSURE_SECONDS = 10.0
RUN_ID = "provenance-audit-fixture"
ET_MAINSIM_COMMIT = "a" * 40
PHOTSIM7_COMMIT = "b" * 40


def _write_scalar_json(handle: h5py.File, name: str, value: object) -> None:
    del handle[name]
    handle.create_dataset(name, data=json.dumps(value, sort_keys=True))


def _write_fixture_run(tmp_path: Path) -> Path:
    """Create a compact, complete delivery with immutable header provenance."""

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
    factors_root = inputs_root / "galaxy_factor_snapshots"
    factors_root.mkdir()
    plan = plan_continuous_time_shards(
        raw_start_index=0,
        raw_stop_index=12,
        coadd_sizes=(3,),
        raw_exposure_seconds=RAW_EXPOSURE_SECONDS,
        max_raw_frames_per_shard=6,
    )
    time_plan_path = plan.write_manifest(inputs_root / "time_shards.json")
    source_ids = (41, 42)
    targets = []
    for source_id, field_angle in zip(source_ids, (2.1, 9.1), strict=True):
        factor_path = factors_root / f"source_{source_id}.npz"
        factor_path.write_bytes(b"fixture factor snapshot\n")
        targets.append(
            {
                "source_id": str(source_id),
                "source_id_int64": source_id,
                "gaia_g_mag": 11.0 + source_id / 100.0,
                "magnitude_system": "Gaia_G_Vega",
                "ra_deg": 280.0 + source_id,
                "dec_deg": 40.0 + source_id / 100.0,
                "factor_snapshot": file_identity(factor_path),
                "focalplane_mapping": {
                    "detector_id": "main_lu",
                    "detector_xpix": 1000.0 + source_id,
                    "detector_ypix": 2000.0 + source_id,
                    "field_angle_deg": field_angle,
                },
            }
        )
    manifest_path = run_root / "production_manifest.json"
    manifest = {
        "schema_id": GALAXY_STAMP_PRODUCTION_SCHEMA_ID,
        "schema_version": GALAXY_STAMP_PRODUCTION_SCHEMA_VERSION,
        "run_id": RUN_ID,
        "observation_product": "final_dn",
        "background_realization_delivered": False,
        "delivery": {
            "stamp_shape": [3, 4],
            "raw_exposure_seconds": RAW_EXPOSURE_SECONDS,
            "cadence_seconds": [30.0],
            "coadd_sizes": [3],
            "time_plan_relative_path": "inputs/time_shards.json",
            "time_plan_identity": file_identity(time_plan_path),
        },
        "software_provenance_at_prepare": {
            "et_mainsim": {"commit": ET_MAINSIM_COMMIT, "dirty": False},
            "photsim7": {"commit": PHOTSIM7_COMMIT, "dirty": False},
        },
        "targets": targets,
    }
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    node_angles = {"0": 0.0, "1": 2.0, "2": 4.0, "3": 6.0, "4": 8.0, "5": 10.0}
    for target in targets:
        source_id = target["source_id_int64"]
        field_angle = target["focalplane_mapping"]["field_angle_deg"]
        psf_id = min(node_angles, key=lambda key: abs(node_angles[key] - field_angle))
        target_truth = {
            "schema_id": "et_mainsim.galaxy_target_truth.v1",
            "source_id": source_id,
            "gaia_g_mag": target["gaia_g_mag"],
            "magnitude_system": target["magnitude_system"],
            "location": {
                "mode": "sky_icrs_j2000",
                "ra_deg": target["ra_deg"],
                "dec_deg": target["dec_deg"],
                "detector_id": target["focalplane_mapping"]["detector_id"],
                "detector_xpix": target["focalplane_mapping"]["detector_xpix"],
                "detector_ypix": target["focalplane_mapping"]["detector_ypix"],
                "field_angle_deg": field_angle,
            },
            "psf": {
                "selection_policy": "nearest_radial_field_angle",
                "chosen_psf_id": int(psf_id),
                "node_angle_deg": node_angles[psf_id],
                "bundle": {
                    "expected_sha256": "c" * 64,
                    "file_identity": {"sha256": "c" * 64, "size_bytes": 10},
                    "node_angles_deg": node_angles,
                },
            },
        }
        for shard in plan.shards:
            for product_kind, factor, filename in (
                ("raw", 1, "raw.h5"),
                ("coadd", 3, "coadd_30s.h5"),
            ):
                starts = np.arange(
                    shard.raw_start_index,
                    shard.raw_stop_index,
                    factor,
                    dtype=np.int64,
                )
                frame_count = starts.size
                bundle = StampDeliveryBundle.from_arrays(
                    product_kind=product_kind,
                    coadd_factor=factor,
                    final_dn=np.full(
                        (frame_count, 3, 4),
                        1024,
                        dtype=np.uint16 if factor == 1 else np.uint64,
                    ),
                    background_expectation_e=np.zeros((frame_count, 3, 4)),
                    bias_level_sum_dn=np.zeros(frame_count),
                    column_noise_sum_dn_by_x=np.zeros((frame_count, 4)),
                    valid_mask=np.ones((frame_count, 3, 4), dtype=bool),
                    fullwell_count=np.zeros((frame_count, 3, 4), dtype=np.uint16),
                    adc_low_count=np.zeros((frame_count, 3, 4), dtype=np.uint16),
                    adc_high_count=np.zeros((frame_count, 3, 4), dtype=np.uint16),
                    cosmic_count=np.zeros((frame_count, 3, 4), dtype=np.uint16),
                    time_start_seconds=starts.astype(float) * RAW_EXPOSURE_SECONDS,
                    exposure_seconds=np.full(frame_count, factor * RAW_EXPOSURE_SECONDS),
                    raw_frame_start_index=starts,
                    raw_frame_stop_index_exclusive=starts + factor,
                    gain_e_per_dn=np.asarray(1.4),
                    manifest={
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
                            "target_input_truth": target_truth,
                        },
                    },
                    provenance={
                        "observation_product": "final_dn",
                        "background_realization_used": False,
                        "caller_provenance": {
                            "software": {
                                "et_mainsim": {
                                    "commit": ET_MAINSIM_COMMIT,
                                    "dirty": False,
                                    "version": "0.1.0",
                                },
                                "photsim7": {
                                    "commit": PHOTSIM7_COMMIT,
                                    "dirty": False,
                                    "version": "0.1.0",
                                },
                            },
                            "factor_snapshot_identity": target["factor_snapshot"],
                            "simulation_spec": {
                                "detector": {
                                    "detector_id": target["focalplane_mapping"]["detector_id"]
                                }
                            },
                        },
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


def test_provenance_audit_accepts_full_manifest_anchored_delivery(tmp_path: Path) -> None:
    from et_mainsim.galaxy_delivery_provenance_audit import (
        GalaxyDeliveryProvenanceAuditRequest,
        audit_galaxy_delivery_provenance_v1,
    )

    manifest_path = _write_fixture_run(tmp_path)
    result = audit_galaxy_delivery_provenance_v1(
        GalaxyDeliveryProvenanceAuditRequest(production_manifest_path=manifest_path)
    )

    assert result.ready is True
    assert result.expected_bundle_count == 8
    assert result.valid_bundle_count == 8
    assert result.invalid_bundle_count == 0
    payload = result.to_dict()
    assert payload["source_summaries"]["41"]["chosen_psf_id"] == 1
    assert payload["software"]["observed_versions"]["photsim7"] == {"0.1.0": 8}


def test_provenance_audit_rejects_one_product_with_wrong_psf_or_commit(
    tmp_path: Path,
) -> None:
    from et_mainsim.galaxy_delivery_provenance_audit import (
        GalaxyDeliveryProvenanceAuditRequest,
        audit_galaxy_delivery_provenance_v1,
    )

    manifest_path = _write_fixture_run(tmp_path)
    malformed = (
        manifest_path.parent
        / "cases/injected/stamps/target_42/delivery/shard_00001/coadd_30s.h5"
    )
    with h5py.File(malformed, "r+") as handle:
        delivery_manifest = json.loads(handle["manifest_json"][()].decode())
        delivery_manifest["caller_manifest"]["target_input_truth"]["psf"][
            "chosen_psf_id"
        ] = 0
        _write_scalar_json(handle, "manifest_json", delivery_manifest)
        provenance = json.loads(handle["provenance_json"][()].decode())
        provenance["caller_provenance"]["software"]["photsim7"]["commit"] = "d" * 40
        _write_scalar_json(handle, "provenance_json", provenance)

    result = audit_galaxy_delivery_provenance_v1(
        GalaxyDeliveryProvenanceAuditRequest(production_manifest_path=manifest_path)
    )

    assert result.ready is False
    assert result.invalid_bundle_count == 1
    assert result.valid_bundle_count == 7
    assert "chosen_psf_id" in result.invalid_bundles[0]["error"]


def test_provenance_audit_cli_writes_fail_closed_receipt(tmp_path: Path) -> None:
    from et_mainsim.galaxy_delivery_provenance_audit import main

    manifest_path = _write_fixture_run(tmp_path)
    receipt = tmp_path / "provenance_audit.json"
    assert (
        main(
            [
                "--production-manifest",
                str(manifest_path),
                "--output-json",
                str(receipt),
                "--require-complete",
            ]
        )
        == 0
    )
    assert json.loads(receipt.read_text(encoding="utf-8"))["ready"] is True

    (
        manifest_path.parent
        / "cases/injected/stamps/target_41/delivery/shard_00000/raw.h5"
    ).unlink()
    assert (
        main(
            [
                "--production-manifest",
                str(manifest_path),
                "--output-json",
                str(receipt),
                "--require-complete",
            ]
        )
        == 2
    )
    assert json.loads(receipt.read_text(encoding="utf-8"))["ready"] is False
