from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest


SOURCE_IDS = tuple(range(1_000, 1_010))
WINDOWS = (30, 90, 390)


def _content_identity(path: Path) -> dict[str, object]:
    from et_mainsim.stamp_inputs import file_identity

    identity = file_identity(path)
    return {"sha256": identity["sha256"], "size_bytes": identity["size_bytes"]}


def _cdpp(values: list[float], *, divide_by_center: bool) -> float:
    import numpy as np

    samples = np.asarray(values, dtype=float)
    center = float(np.median(samples))
    mad = float(np.mean(np.abs(samples - center)))
    return (
        1.4826 * mad / center * 1_000_000.0
        if divide_by_center
        else 1.4826 * mad
    )


def _write_complete_campaign_fixture(
    tmp_path: Path,
    *,
    minimum_accepted_bins: int = 2,
) -> tuple[Path, Path, Path, Path]:
    """Create ten policy-consistent, tiny published source analyses."""

    from et_mainsim.galaxy_stamp_production import (
        GALAXY_STAMP_PRODUCTION_SCHEMA_ID,
        GALAXY_STAMP_PRODUCTION_SCHEMA_VERSION,
    )
    from et_mainsim.raw_coverage_policy import (
        FrozenRawCoveragePolicyRequest,
        write_frozen_raw_coverage_policy_v1,
    )
    from et_mainsim.stamp_inputs import file_identity

    run_root = tmp_path / "campaign"
    inputs = run_root / "inputs" / "galaxy_factor_snapshots"
    inputs.mkdir(parents=True)
    factor_identities: dict[int, dict[str, object]] = {}
    for source_id in SOURCE_IDS:
        factor_path = inputs / f"source_{source_id}.npz"
        factor_path.write_bytes(f"factors-{source_id}".encode("ascii"))
        factor_identities[source_id] = file_identity(factor_path)

    manifest_path = run_root / "production_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_id": GALAXY_STAMP_PRODUCTION_SCHEMA_ID,
                "schema_version": GALAXY_STAMP_PRODUCTION_SCHEMA_VERSION,
                "run_id": "summary-fixture",
                "observation_product": "final_dn",
                "background_realization_delivered": False,
                "delivery": {"raw_exposure_seconds": 10.0, "stamp_shape": [100, 300]},
                "targets": [
                    {
                        "source_id": str(source_id),
                        "source_id_int64": source_id,
                        "gaia_g_mag": 11.0 + index / 100.0,
                        "source_class": "rotation" if index == 0 else "subgiant",
                        "focalplane_mapping": {
                            "detector_id": "main_lu" if index < 3 else "main_ld",
                            "detector_xpix": 100.0 + index,
                            "detector_ypix": 200.0 + index,
                            "field_angle_deg": 2.0 + index,
                        },
                        "factor_snapshot_relative_path": (
                            f"inputs/galaxy_factor_snapshots/source_{source_id}.npz"
                        ),
                        "factor_snapshot": factor_identities[source_id],
                    }
                    for index, source_id in enumerate(SOURCE_IDS)
                ],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    policy_path = run_root / "analysis" / "raw_10s_coverage_v2_policy.json"
    policy = write_frozen_raw_coverage_policy_v1(
        FrozenRawCoveragePolicyRequest(
            production_manifest_path=manifest_path,
            output_path=policy_path,
            minimum_coverage_fraction=0.95,
            minimum_accepted_bins=minimum_accepted_bins,
        )
    )
    qc_path = run_root / "quality_control" / "injected_campaign_delivery_qc.json"
    qc_path.parent.mkdir(parents=True)
    qc_path.write_text(
        json.dumps(
            {
                "schema_id": "et_mainsim.galaxy_campaign_delivery_qc.v1",
                "schema_version": 1,
                "ready": True,
                "run_id": "summary-fixture",
                "case": "injected",
                "manifest_identity": file_identity(manifest_path),
                "coverage": {
                    "target_count": len(SOURCE_IDS),
                    "shard_count": 2,
                    "accepted_raw_frame_count_per_target": 12,
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    module_marker = run_root / "analysis" / "coverage_module.py"
    module_marker.write_text("fixture module\n", encoding="utf-8")
    module_identity = _content_identity(module_marker)

    for source_order, source_id in enumerate(SOURCE_IDS):
        strict_dir = (
            run_root
            / "analysis"
            / f"source_{source_id}"
            / "injected"
            / "raw_10s_strict"
        )
        coverage_dir = strict_dir.parent / "raw_10s_coverage_v2"
        strict_dir.mkdir(parents=True)
        coverage_dir.mkdir()
        strict_curve = strict_dir / "reference_lightcurve.csv"
        strict_curve.write_text("fixture strict curve\n", encoding="utf-8")
        strict_manifest = strict_dir / "analysis_manifest.json"
        strict_manifest.write_text(
            json.dumps(
                {
                    "schema_id": "et_mainsim.standard_stamp_analysis.v1",
                    "schema_version": 1,
                    "complete": True,
                    "run_id": "summary-fixture",
                    "production_manifest_relative_to_run_root": "production_manifest.json",
                    "production_manifest_identity": file_identity(manifest_path),
                    "source_id": str(source_id),
                    "source_id_int64": source_id,
                    "case": "injected",
                    "observation_product": "final_dn",
                    "background_realization_used": False,
                    "reference_lightcurve": {
                        "schema_id": "et_mainsim.standard_stamp_reference_lightcurve.v1",
                        "path": strict_curve.name,
                        "format": "csv",
                    },
                    "delivery": {
                        "product_filename": "raw.h5",
                        "cadence_seconds": 10.0,
                        "raw_exposure_seconds": 10.0,
                        "bundle_count": 2,
                    },
                    "quality": {
                        "cadence_count": 12,
                        "valid_cadence_count": 12,
                        "invalid_cadence_count": 0,
                        "aperture_pixel_count": 169,
                        "minimum_usable_pixel_count": 169,
                        "maximum_usable_pixel_count": 169,
                        "quality_policy": "invalidate_whole_fixed_aperture_cadence",
                    },
                    "frozen_variability": {
                        "path_relative_to_run_root": (
                            f"inputs/galaxy_factor_snapshots/source_{source_id}.npz"
                        ),
                        "identity": factor_identities[source_id],
                        "time_alignment": "simulation_raw_frame_index",
                    },
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        binned_path = coverage_dir / "coverage_aware_binned_lightcurve.csv"
        fieldnames = [
            "window_minutes",
            "bin_id",
            "time_start_seconds",
            "time_stop_seconds",
            "expected_exposure_seconds",
            "effective_exposure_seconds",
            "coverage_fraction",
            "expected_cadence_count",
            "valid_cadence_count",
            "accepted",
            "observed_flux_rate_e_per_s",
            "model_flux_rate_e_per_s",
            "residual_fraction_ppm",
        ]
        with binned_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for window in WINDOWS:
                for bin_id, (rate, residual) in enumerate(((100.0, 1.0), (110.0, 3.0))):
                    writer.writerow(
                        {
                            "window_minutes": window,
                            "bin_id": bin_id,
                            "time_start_seconds": bin_id * window * 60,
                            "time_stop_seconds": (bin_id + 1) * window * 60,
                            "expected_exposure_seconds": window * 60,
                            "effective_exposure_seconds": window * 60,
                            "coverage_fraction": 1.0,
                            "expected_cadence_count": 6,
                            "valid_cadence_count": 6,
                            "accepted": 1,
                            "observed_flux_rate_e_per_s": rate,
                            "model_flux_rate_e_per_s": rate,
                            "residual_fraction_ppm": residual,
                        }
                    )
        metrics = {
            str(window): {
                "window_minutes": window,
                "total_bin_count": 2,
                "accepted_bin_count": 2,
                "rejected_bin_count": 0,
                "accepted_sample_count": 12,
                "minimum_coverage_fraction": 0.95,
                "minimum_accepted_bins": minimum_accepted_bins,
                "observed_cdpp_ppm": (
                    _cdpp([100.0, 110.0], divide_by_center=True)
                    if minimum_accepted_bins <= 2
                    else None
                ),
                "residual_cdpp_ppm": (
                    _cdpp([1.0, 3.0], divide_by_center=False)
                    if minimum_accepted_bins <= 2
                    else None
                ),
                "observed_estimator": "legacy_median_centered_mean_absolute_deviation_times_1.4826",
                "residual_estimator": "legacy_median_centered_mean_absolute_deviation_times_1.4826",
                "aggregation": "valid_cadence_counts_normalized_by_actual_effective_exposure",
            }
            for window in WINDOWS
        }
        coverage_manifest = coverage_dir / "coverage_aware_analysis_manifest.json"
        coverage_manifest.write_text(
            json.dumps(
                {
                    "schema_id": "et_mainsim.raw_coverage_aware_stamp_analysis.v3",
                    "schema_version": 3,
                    "complete": True,
                    "run_id": "summary-fixture",
                    "source_id": str(source_id),
                    "source_id_int64": source_id,
                    "case": "injected",
                    "observation_product": "final_dn",
                    "background_realization_used": False,
                    "production_manifest": {
                        "path_relative_to_run_root": "production_manifest.json",
                        "identity": _content_identity(manifest_path),
                    },
                    "campaign_qc": {
                        "path_relative_to_run_root": (
                            "quality_control/injected_campaign_delivery_qc.json"
                        ),
                        "identity": _content_identity(qc_path),
                    },
                    "analysis_implementation": {
                        "module": "et_mainsim.coverage_aware_stamp_analysis",
                        "module_identity": module_identity,
                    },
                    "input_reference_analysis": {
                        "path_relative_to_run_root": strict_dir.relative_to(run_root).as_posix(),
                        "analysis_manifest": _content_identity(strict_manifest),
                        "reference_lightcurve": {
                            "path_relative_to_run_root": strict_curve.relative_to(run_root).as_posix(),
                            "identity": _content_identity(strict_curve),
                        },
                        "source_id_int64": source_id,
                        "case": "injected",
                        "strict_quality_policy": "invalidate_whole_fixed_aperture_cadence",
                    },
                    "input_raw_delivery": {
                        "product_filename": "raw.h5",
                        "cadence_seconds": 10.0,
                        "raw_exposure_seconds": 10.0,
                        "raw_frame_policy": "one_contiguous_raw_frame_per_reference_cadence",
                    },
                    "frozen_coverage_policy": {
                        "schema_id": policy.schema_id,
                        "schema_version": policy.schema_version,
                        "path_relative_to_run_root": policy_path.relative_to(run_root).as_posix(),
                        "identity": policy.content_identity,
                    },
                    "coverage_policy": policy.coverage_policy_record(),
                    "binned_lightcurve": {
                        "path": binned_path.name,
                        "format": "csv",
                        "identity": _content_identity(binned_path),
                    },
                    "metrics": metrics,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return manifest_path, qc_path, policy_path, run_root / "analysis" / "campaign" / "injected" / "raw_10s_coverage_v2_summary"


def test_raw_coverage_campaign_summary_publishes_all_ten_sources_atomically(
    tmp_path: Path,
) -> None:
    from et_mainsim.raw_coverage_campaign_summary import (
        GalaxyRawCoverageCampaignSummaryRequest,
        publish_galaxy_raw_coverage_campaign_summary_v1,
    )

    manifest_path, qc_path, policy_path, output_dir = _write_complete_campaign_fixture(
        tmp_path
    )
    result = publish_galaxy_raw_coverage_campaign_summary_v1(
        GalaxyRawCoverageCampaignSummaryRequest(
            production_manifest_path=manifest_path,
            campaign_qc_path=qc_path,
            coverage_policy_path=policy_path,
            output_dir=output_dir,
        )
    )

    assert result.output_dir == output_dir
    assert result.summary_manifest_path.is_file()
    assert result.source_summary_path.is_file()
    assert result.source_window_metrics_path.is_file()
    summary = json.loads(result.summary_manifest_path.read_text(encoding="utf-8"))
    assert summary["ready"] is True
    assert summary["source_count"] == 10
    assert summary["observation_product"] == "final_dn"
    assert summary["background_realization_used"] is False
    assert summary["frozen_coverage_policy"]["identity"] == _content_identity(policy_path)
    rows = list(csv.DictReader(result.source_window_metrics_path.open(encoding="utf-8")))
    assert len(rows) == 30
    assert {row["source_id"] for row in rows} == {str(value) for value in SOURCE_IDS}
    assert {int(row["window_minutes"]) for row in rows} == set(WINDOWS)
    assert all(row["observed_cdpp_ppm"] for row in rows)


def test_raw_coverage_campaign_summary_is_fail_closed_for_a_missing_source(
    tmp_path: Path,
) -> None:
    from et_mainsim.raw_coverage_campaign_summary import (
        GalaxyRawCoverageCampaignSummaryNotReadyError,
        GalaxyRawCoverageCampaignSummaryRequest,
        audit_galaxy_raw_coverage_campaign_v1,
        publish_galaxy_raw_coverage_campaign_summary_v1,
    )

    manifest_path, qc_path, policy_path, output_dir = _write_complete_campaign_fixture(
        tmp_path
    )
    missing = (
        manifest_path.parent
        / "analysis"
        / f"source_{SOURCE_IDS[-1]}"
        / "injected"
        / "raw_10s_coverage_v2"
        / "coverage_aware_analysis_manifest.json"
    )
    missing.unlink()
    request = GalaxyRawCoverageCampaignSummaryRequest(
        production_manifest_path=manifest_path,
        campaign_qc_path=qc_path,
        coverage_policy_path=policy_path,
        output_dir=output_dir,
    )

    audit = audit_galaxy_raw_coverage_campaign_v1(request)
    assert audit.ready is False
    assert str(SOURCE_IDS[-1]) in audit.errors_by_source
    with pytest.raises(GalaxyRawCoverageCampaignSummaryNotReadyError, match="not ready"):
        publish_galaxy_raw_coverage_campaign_summary_v1(request)
    assert not output_dir.exists()


def test_raw_coverage_campaign_summary_keeps_null_cdpp_as_a_valid_result(
    tmp_path: Path,
) -> None:
    from et_mainsim.raw_coverage_campaign_summary import (
        GalaxyRawCoverageCampaignSummaryRequest,
        publish_galaxy_raw_coverage_campaign_summary_v1,
    )

    manifest_path, qc_path, policy_path, output_dir = _write_complete_campaign_fixture(
        tmp_path,
        minimum_accepted_bins=3,
    )
    result = publish_galaxy_raw_coverage_campaign_summary_v1(
        GalaxyRawCoverageCampaignSummaryRequest(
            production_manifest_path=manifest_path,
            campaign_qc_path=qc_path,
            coverage_policy_path=policy_path,
            output_dir=output_dir,
        )
    )

    manifest = json.loads(result.summary_manifest_path.read_text(encoding="utf-8"))
    assert manifest["ready"] is True
    rows = list(csv.DictReader(result.source_window_metrics_path.open(encoding="utf-8")))
    assert len(rows) == 30
    assert all(row["observed_cdpp_ppm"] == "" for row in rows)
    assert all(row["residual_cdpp_ppm"] == "" for row in rows)


def test_raw_coverage_campaign_summary_rejects_a_mixed_policy_identity(
    tmp_path: Path,
) -> None:
    from et_mainsim.raw_coverage_campaign_summary import (
        GalaxyRawCoverageCampaignSummaryRequest,
        audit_galaxy_raw_coverage_campaign_v1,
    )

    manifest_path, qc_path, policy_path, output_dir = _write_complete_campaign_fixture(
        tmp_path
    )
    coverage_manifest_path = (
        manifest_path.parent
        / "analysis"
        / f"source_{SOURCE_IDS[0]}"
        / "injected"
        / "raw_10s_coverage_v2"
        / "coverage_aware_analysis_manifest.json"
    )
    payload = json.loads(coverage_manifest_path.read_text(encoding="utf-8"))
    payload["frozen_coverage_policy"]["identity"]["sha256"] = "0" * 64
    coverage_manifest_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    audit = audit_galaxy_raw_coverage_campaign_v1(
        GalaxyRawCoverageCampaignSummaryRequest(
            production_manifest_path=manifest_path,
            campaign_qc_path=qc_path,
            coverage_policy_path=policy_path,
            output_dir=output_dir,
        )
    )
    assert audit.ready is False
    assert "policy" in audit.errors_by_source[str(SOURCE_IDS[0])].lower()
