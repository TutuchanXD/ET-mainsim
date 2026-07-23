from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest


def _two_thirty_minute_cadences(
    *,
    invalid_per_bin: int,
    cadence_seconds: float = 60.0,
) -> dict[str, np.ndarray]:
    """Return two physical 30-minute windows at a whole-second cadence."""

    frames_per_bin = int(1_800.0 / cadence_seconds)
    assert frames_per_bin * cadence_seconds == pytest.approx(1_800.0)
    cadence_count = frames_per_bin * 2
    exposure = np.full(cadence_count, cadence_seconds)
    time = np.arange(cadence_count, dtype=float) * cadence_seconds
    valid = np.ones(cadence_count, dtype=bool)
    for start in (0, frames_per_bin):
        valid[start : start + invalid_per_bin] = False
    model = np.concatenate(
        (
            np.full(frames_per_bin, 600.0),
            np.full(frames_per_bin, 660.0),
        )
    )
    # Each three-cadence cycle has zero sum.  Marking the first three samples
    # of every 30-minute bin therefore leaves the model-residual aggregate
    # exactly zero without fabricating replacement samples.
    residual = np.resize(np.array([-6.0, 0.0, 6.0]), cadence_count)
    return {
        "time_seconds": time,
        "exposure_seconds": exposure,
        "flux_e": model + residual,
        "aperture_valid": valid,
        "model_flux_e": model,
        "residual_e": residual,
    }


def test_coverage_aware_cdpp_accepts_ninety_percent_without_imputation() -> None:
    from et_mainsim.coverage_aware_stamp_analysis import (
        CoverageAwareLightCurve,
        compute_coverage_aware_cdpp_v1,
    )

    payload = _two_thirty_minute_cadences(invalid_per_bin=3)
    result = compute_coverage_aware_cdpp_v1(
        CoverageAwareLightCurve(**payload),
        windows_minutes=(30,),
        minimum_coverage_fraction=0.90,
        minimum_accepted_bins=2,
    )

    metric = result.metrics_by_window_minutes[30]
    assert metric.accepted_bin_count == 2
    assert metric.rejected_bin_count == 0
    assert metric.accepted_sample_count == 54
    assert metric.observed_cdpp_ppm == pytest.approx(70_600.0)
    assert metric.residual_cdpp_ppm == pytest.approx(0.0)

    first, second = result.binned_rows
    assert first.coverage_fraction == pytest.approx(0.90)
    assert second.coverage_fraction == pytest.approx(0.90)
    assert first.effective_exposure_seconds == pytest.approx(1_620.0)
    assert first.observed_flux_rate_e_per_s == pytest.approx(10.0)
    # Three invalid input cadences are excluded; their flux is never inserted
    # or estimated, while the rate remains normalized by actual exposure.
    assert first.valid_cadence_count == 27
    assert first.expected_cadence_count == 30


def test_coverage_aware_cdpp_rejects_bins_below_coverage_threshold() -> None:
    from et_mainsim.coverage_aware_stamp_analysis import (
        CoverageAwareLightCurve,
        compute_coverage_aware_cdpp_v1,
    )

    payload = _two_thirty_minute_cadences(invalid_per_bin=4)
    result = compute_coverage_aware_cdpp_v1(
        CoverageAwareLightCurve(**payload),
        windows_minutes=(30,),
        minimum_coverage_fraction=0.90,
        minimum_accepted_bins=2,
    )

    metric = result.metrics_by_window_minutes[30]
    assert metric.accepted_bin_count == 0
    assert metric.rejected_bin_count == 2
    assert np.isnan(metric.observed_cdpp_ppm)
    assert np.isnan(metric.residual_cdpp_ppm)
    assert all(not row.accepted for row in result.binned_rows)


def test_coverage_aware_cdpp_requires_contiguous_physical_cadence() -> None:
    from et_mainsim.coverage_aware_stamp_analysis import (
        CoverageAwareAnalysisError,
        CoverageAwareLightCurve,
        compute_coverage_aware_cdpp_v1,
    )

    payload = _two_thirty_minute_cadences(invalid_per_bin=0)
    payload["time_seconds"][30] += 60.0
    with pytest.raises(CoverageAwareAnalysisError, match="contiguous"):
        compute_coverage_aware_cdpp_v1(
            CoverageAwareLightCurve(**payload),
            windows_minutes=(30,),
            minimum_coverage_fraction=0.90,
            minimum_accepted_bins=2,
        )


def _write_reference_analysis_fixture(tmp_path: Path) -> tuple[Path, Path]:
    """Write a raw-10-s strict-reference output with 90% clean 30-min bins."""

    source_dir = tmp_path / "strict_reference"
    source_dir.mkdir()
    lightcurve_path = source_dir / "reference_lightcurve.csv"
    payload = _two_thirty_minute_cadences(
        invalid_per_bin=18,
        cadence_seconds=10.0,
    )
    fieldnames = [
        "time_start_seconds",
        "exposure_seconds",
        "raw_frame_start_index",
        "raw_frame_stop_index_exclusive",
        "flux_derived_e",
        "aperture_valid",
        "aperture_usable_pixel_count",
        "aperture_invalid_pixel_count",
        "model_flux_e",
        "model_residual_e",
        "injected_model_valid",
    ]
    with lightcurve_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(payload["time_seconds"].size):
            valid = bool(payload["aperture_valid"][index])
            writer.writerow(
                {
                    "time_start_seconds": payload["time_seconds"][index],
                    "exposure_seconds": payload["exposure_seconds"][index],
                    "raw_frame_start_index": index,
                    "raw_frame_stop_index_exclusive": index + 1,
                    "flux_derived_e": payload["flux_e"][index] if valid else "",
                    "aperture_valid": int(valid),
                    "aperture_usable_pixel_count": 169 if valid else 168,
                    "aperture_invalid_pixel_count": 0 if valid else 1,
                    "model_flux_e": payload["model_flux_e"][index] if valid else "",
                    "model_residual_e": payload["residual_e"][index] if valid else "",
                    "injected_model_valid": int(valid),
                }
            )
    manifest_path = source_dir / "analysis_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_id": "et_mainsim.standard_stamp_analysis.v1",
                "schema_version": 1,
                "complete": True,
                "run_id": "fixture",
                "source_id": "42",
                "source_id_int64": 42,
                "case": "injected",
                "observation_product": "final_dn",
                "background_realization_used": False,
                "reference_lightcurve": {
                    "schema_id": "et_mainsim.standard_stamp_reference_lightcurve.v1",
                    "path": "reference_lightcurve.csv",
                    "format": "csv",
                },
                "delivery": {
                    "product_filename": "raw.h5",
                    "cadence_seconds": 10.0,
                    "raw_exposure_seconds": 10.0,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return source_dir, tmp_path / "coverage_aware"


def _write_formal_raw_galaxy_fixture(tmp_path: Path) -> Path:
    """Create one hour of formal raw delivery with 90% clean 30-min bins."""

    from et_mainsim.galaxy_lightcurves import (
        GalaxyLightCurve,
        write_galaxy_factor_snapshot,
    )
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

    source_id = 42
    run_id = "raw-coverage-fixture"
    raw_exposure_seconds = 10.0
    raw_frame_count = 360
    run_root = tmp_path / "formal_raw_run"
    inputs_root = run_root / "inputs"
    factors_root = inputs_root / "galaxy_factor_snapshots"
    factors_root.mkdir(parents=True)
    factors = np.concatenate((np.ones(180), np.full(180, 2.0)))
    curve = GalaxyLightCurve(
        source_id=source_id,
        gaia_g_mag=11.0,
        ra_deg=1.0,
        dec_deg=2.0,
        source_class="fixture",
        native_time_seconds=np.array([0.0, raw_frame_count * raw_exposure_seconds]),
        clean_flux_factor=np.array([1.0, 1.0]),
        input_identity={"fixture": "coverage-aware-raw"},
    )
    snapshot_path = factors_root / f"source_{source_id}.npz"
    snapshot_identity = write_galaxy_factor_snapshot(
        snapshot_path,
        curve=curve,
        factors=factors,
        raw_exposure_seconds=raw_exposure_seconds,
    )
    plan = plan_continuous_time_shards(
        raw_start_index=0,
        raw_stop_index=raw_frame_count,
        coadd_sizes=(3,),
        raw_exposure_seconds=raw_exposure_seconds,
        max_raw_frames_per_shard=180,
    )
    time_plan_path = plan.write_manifest(inputs_root / "time_shards.json")
    manifest_path = run_root / "production_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_id": GALAXY_STAMP_PRODUCTION_SCHEMA_ID,
                "schema_version": GALAXY_STAMP_PRODUCTION_SCHEMA_VERSION,
                "run_id": run_id,
                "run_root": str(run_root),
                "observation_product": "final_dn",
                "background_realization_delivered": False,
                "delivery": {
                    "raw_exposure_seconds": raw_exposure_seconds,
                    "cadence_seconds": [10.0, 30.0],
                    "coadd_sizes": [3],
                    "time_plan_relative_path": "inputs/time_shards.json",
                    "time_plan_identity": file_identity(time_plan_path),
                },
                "targets": [
                    {
                        "source_id": str(source_id),
                        "source_id_int64": source_id,
                        "factor_snapshot": snapshot_identity,
                        "factor_snapshot_relative_path": (
                            f"inputs/galaxy_factor_snapshots/source_{source_id}.npz"
                        ),
                    }
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    for shard in plan.shards:
        starts = np.arange(
            shard.raw_start_index,
            shard.raw_stop_index,
            dtype=np.int64,
        )
        local_factors = factors[shard.raw_start_index : shard.raw_stop_index]
        frame_count = starts.size
        final_dn = np.broadcast_to(
            (local_factors[:, None, None] * 100.0).astype(np.uint16),
            (frame_count, 15, 15),
        ).copy()
        cosmic_count = np.zeros((frame_count, 15, 15), dtype=np.uint16)
        cosmic_count[:18, 7, 7] = 1
        bundle = StampDeliveryBundle.from_arrays(
            product_kind="raw",
            coadd_factor=1,
            final_dn=final_dn,
            background_expectation_e=np.zeros((frame_count, 15, 15)),
            bias_level_sum_dn=np.zeros(frame_count),
            column_noise_sum_dn_by_x=np.zeros((frame_count, 15)),
            valid_mask=np.ones((frame_count, 15, 15), dtype=bool),
            fullwell_count=np.zeros((frame_count, 15, 15), dtype=np.uint16),
            adc_low_count=np.zeros((frame_count, 15, 15), dtype=np.uint16),
            adc_high_count=np.zeros((frame_count, 15, 15), dtype=np.uint16),
            cosmic_count=cosmic_count,
            time_start_seconds=starts.astype(float) * raw_exposure_seconds,
            exposure_seconds=np.full(frame_count, raw_exposure_seconds),
            raw_frame_start_index=starts,
            raw_frame_stop_index_exclusive=starts + 1,
            gain_e_per_dn=np.asarray(1.0),
            manifest={
                "schema_id": "et_mainsim.independent_stamp_production.v1",
                "target_source_id": str(source_id),
                "target_source_id_int64": source_id,
                "stamp_shape": [15, 15],
                "time_shard": {
                    "raw_frame_interval": {
                        "start_index": shard.raw_start_index,
                        "stop_index": shard.raw_stop_index,
                    }
                },
                "caller_manifest": {
                    "run_id": run_id,
                    "case": "injected",
                    "target_input_truth": {
                        "variability": {
                            "enabled": True,
                            "case": "injected",
                            "source_factor_snapshot_identity": snapshot_identity,
                        }
                    },
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
            / "raw.h5"
        )
        write_stamp_delivery_bundle(destination, bundle)
    return manifest_path


def test_coverage_aware_analysis_publishes_an_atomic_receipt(tmp_path) -> None:
    from et_mainsim.coverage_aware_stamp_analysis import (
        CoverageAwareStampAnalysisRequest,
        run_coverage_aware_stamp_analysis_v1,
    )

    source_dir, output_dir = _write_reference_analysis_fixture(tmp_path)
    result = run_coverage_aware_stamp_analysis_v1(
        CoverageAwareStampAnalysisRequest(
            reference_analysis_dir=source_dir,
            output_dir=output_dir,
            windows_minutes=(30,),
            minimum_coverage_fraction=0.90,
            minimum_accepted_bins=2,
        )
    )

    assert result.output_dir == output_dir
    assert result.analysis_manifest_path.is_file()
    assert result.binned_lightcurve_path.is_file()
    manifest = json.loads(result.analysis_manifest_path.read_text(encoding="utf-8"))
    assert manifest["complete"] is True
    assert manifest["input_reference_analysis"]["source_id_int64"] == 42
    assert manifest["coverage_policy"]["minimum_coverage_fraction"] == pytest.approx(
        0.90
    )
    assert manifest["metrics"]["30"]["accepted_bin_count"] == 2
    assert manifest["analysis_implementation"]["module"] == (
        "et_mainsim.coverage_aware_stamp_analysis"
    )
    assert len(manifest["analysis_implementation"]["module_identity"]["sha256"]) == 64
    rows = list(csv.DictReader(result.binned_lightcurve_path.open(encoding="utf-8")))
    assert len(rows) == 2
    assert all(row["accepted"] == "1" for row in rows)


def test_coverage_aware_analysis_rejects_a_coadded_reference_input(
    tmp_path: Path,
) -> None:
    from et_mainsim.coverage_aware_stamp_analysis import (
        CoverageAwareAnalysisError,
        CoverageAwareStampAnalysisRequest,
        run_coverage_aware_stamp_analysis_v1,
    )

    source_dir, output_dir = _write_reference_analysis_fixture(tmp_path)
    manifest_path = source_dir / "analysis_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["delivery"].update(
        {
            "product_filename": "coadd_60s.h5",
            "cadence_seconds": 60.0,
        }
    )
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    with pytest.raises(CoverageAwareAnalysisError, match="raw 10-s"):
        run_coverage_aware_stamp_analysis_v1(
            CoverageAwareStampAnalysisRequest(
                reference_analysis_dir=source_dir,
                output_dir=output_dir,
                windows_minutes=(30,),
                minimum_coverage_fraction=0.90,
                minimum_accepted_bins=2,
            )
        )


def test_coverage_aware_analysis_rejects_non_ten_second_reference_rows(
    tmp_path: Path,
) -> None:
    from et_mainsim.coverage_aware_stamp_analysis import (
        CoverageAwareAnalysisError,
        CoverageAwareStampAnalysisRequest,
        run_coverage_aware_stamp_analysis_v1,
    )

    source_dir, output_dir = _write_reference_analysis_fixture(tmp_path)
    lightcurve_path = source_dir / "reference_lightcurve.csv"
    with lightcurve_path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames
        rows = list(reader)
    assert fieldnames is not None
    for index, row in enumerate(rows):
        row["time_start_seconds"] = str(index * 20.0)
        row["exposure_seconds"] = "20.0"
    with lightcurve_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(CoverageAwareAnalysisError, match="10-s exposure"):
        run_coverage_aware_stamp_analysis_v1(
            CoverageAwareStampAnalysisRequest(
                reference_analysis_dir=source_dir,
                output_dir=output_dir,
                windows_minutes=(30,),
                minimum_coverage_fraction=0.90,
                minimum_accepted_bins=2,
            )
        )


def test_raw_coverage_analysis_uses_only_clean_raw_frames_from_formal_delivery(
    tmp_path: Path,
) -> None:
    from et_mainsim.coverage_aware_stamp_analysis import (
        CoverageAwareStampAnalysisRequest,
        run_coverage_aware_stamp_analysis_v1,
    )
    from et_mainsim.standard_stamp_analysis import (
        StandardStampAnalysisRequest,
        run_standard_stamp_analysis_v1,
    )

    manifest_path = _write_formal_raw_galaxy_fixture(tmp_path)
    strict_dir = tmp_path / "raw_10s_strict"
    standard = run_standard_stamp_analysis_v1(
        StandardStampAnalysisRequest(
            production_manifest_path=manifest_path,
            source_id=42,
            case="injected",
            cadence_seconds=10.0,
            output_dir=strict_dir,
            batch_frames=64,
        )
    )
    assert standard.valid_cadence_count == 324

    output_dir = tmp_path / "raw_10s_coverage_v2"
    coverage = run_coverage_aware_stamp_analysis_v1(
        CoverageAwareStampAnalysisRequest(
            reference_analysis_dir=strict_dir,
            output_dir=output_dir,
            windows_minutes=(30,),
            minimum_coverage_fraction=0.90,
            minimum_accepted_bins=2,
        )
    )

    manifest = json.loads(coverage.analysis_manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_id"] == "et_mainsim.raw_coverage_aware_stamp_analysis.v2"
    assert manifest["input_raw_delivery"] == {
        "product_filename": "raw.h5",
        "cadence_seconds": 10.0,
        "raw_exposure_seconds": 10.0,
        "raw_frame_policy": "one_contiguous_raw_frame_per_reference_cadence",
    }
    metric = manifest["metrics"]["30"]
    assert metric["accepted_bin_count"] == 2
    assert metric["accepted_sample_count"] == 324
    assert metric["residual_cdpp_ppm"] == pytest.approx(0.0, abs=1e-8)
    rows = list(csv.DictReader(coverage.binned_lightcurve_path.open(encoding="utf-8")))
    assert [int(row["valid_cadence_count"]) for row in rows] == [162, 162]
    assert [float(row["coverage_fraction"]) for row in rows] == pytest.approx(
        [0.90, 0.90]
    )


def test_coverage_aware_analysis_cli_requires_an_explicit_coverage_policy(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from et_mainsim.coverage_aware_stamp_analysis import main

    source_dir, output_dir = _write_reference_analysis_fixture(tmp_path)
    assert (
        main(
            (
                "--reference-analysis-dir",
                str(source_dir),
                "--output-dir",
                str(output_dir),
                "--windows-minutes",
                "30",
                "--minimum-coverage-fraction",
                "0.90",
                "--minimum-accepted-bins",
                "2",
            )
        )
        == 0
    )
    completion = json.loads(capsys.readouterr().out)
    assert completion["source_id_int64"] == 42
    assert completion["case"] == "injected"
    assert completion["minimum_coverage_fraction"] == pytest.approx(0.90)
    assert completion["analysis_manifest_path"].endswith(
        "coverage_aware_analysis_manifest.json"
    )
