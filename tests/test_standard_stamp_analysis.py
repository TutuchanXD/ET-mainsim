from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pytest


SOURCE_ID = 42
RUN_ID = "formal-standard-analysis-test"
RAW_EXPOSURE_SECONDS = 10.0
CADENCE_SECONDS = 30.0


def _write_formal_galaxy_run(
    tmp_path: Path,
    *,
    case: str,
    write_delivery: bool = True,
) -> Path:
    """Build a compact, two-shard Galaxy formal-delivery fixture.

    Six hours at 30-second cadence gives twelve complete 30-minute CDPP bins.
    The factor pattern is exactly representable in integer DN, so an injected
    reference curve has a known zero residual after the v1 through-origin fit.
    """

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

    run_root = tmp_path / "formal_run"
    inputs_root = run_root / "inputs"
    factors_root = inputs_root / "galaxy_factor_snapshots"
    factors_root.mkdir(parents=True)
    n_raw_frames = 2_160  # six hours at 10 seconds
    factors = 1.0 + ((np.arange(n_raw_frames) // 3) % 2).astype(float)
    curve = GalaxyLightCurve(
        source_id=SOURCE_ID,
        gaia_g_mag=11.0,
        ra_deg=1.0,
        dec_deg=2.0,
        source_class="test",
        native_time_seconds=np.array([0.0, n_raw_frames * RAW_EXPOSURE_SECONDS]),
        clean_flux_factor=np.array([1.0, 1.0]),
        input_identity={"fixture": "standard-stamp-analysis"},
    )
    snapshot_path = factors_root / f"source_{SOURCE_ID}.npz"
    snapshot_identity = write_galaxy_factor_snapshot(
        snapshot_path,
        curve=curve,
        factors=factors,
        raw_exposure_seconds=RAW_EXPOSURE_SECONDS,
    )
    plan = plan_continuous_time_shards(
        raw_start_index=0,
        raw_stop_index=n_raw_frames,
        coadd_sizes=(3,),
        raw_exposure_seconds=RAW_EXPOSURE_SECONDS,
        max_raw_frames_per_shard=1_080,
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
            "raw_exposure_seconds": RAW_EXPOSURE_SECONDS,
            "cadence_seconds": [CADENCE_SECONDS],
            "coadd_sizes": [3],
            "time_plan_relative_path": "inputs/time_shards.json",
            "time_plan_identity": file_identity(time_plan_path),
        },
        "targets": [
            {
                "source_id": str(SOURCE_ID),
                "source_id_int64": SOURCE_ID,
                "factor_snapshot": snapshot_identity,
                "factor_snapshot_relative_path": (
                    f"inputs/galaxy_factor_snapshots/source_{SOURCE_ID}.npz"
                ),
            }
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    if not write_delivery:
        return manifest_path

    for shard in plan.shards:
        starts = np.arange(
            shard.raw_start_index,
            shard.raw_stop_index,
            3,
            dtype=np.int64,
        )
        stops = starts + 3
        factor_sum = (
            factors[shard.raw_start_index : shard.raw_stop_index]
            .reshape(-1, 3)
            .sum(axis=1)
        )
        n_cadence = starts.size
        final_dn = np.broadcast_to(
            (factor_sum[:, None, None] * 100.0).astype(np.uint64),
            (n_cadence, 15, 15),
        ).copy()
        valid_mask = np.ones((n_cadence, 15, 15), dtype=bool)
        # The standard table must carry a transparent cadence-level quality
        # field rather than silently re-scaling a damaged aperture.
        if shard.shard_id == 0:
            valid_mask[0, 7, 7] = False
        bundle = StampDeliveryBundle.from_arrays(
            product_kind="coadd",
            coadd_factor=3,
            final_dn=final_dn,
            background_expectation_e=np.zeros((n_cadence, 15, 15)),
            bias_level_sum_dn=np.zeros(n_cadence),
            column_noise_sum_dn_by_x=np.zeros((n_cadence, 15)),
            valid_mask=valid_mask,
            fullwell_count=np.zeros((n_cadence, 15, 15), dtype=np.uint16),
            adc_low_count=np.zeros((n_cadence, 15, 15), dtype=np.uint16),
            adc_high_count=np.zeros((n_cadence, 15, 15), dtype=np.uint16),
            cosmic_count=np.zeros((n_cadence, 15, 15), dtype=np.uint16),
            time_start_seconds=starts.astype(float) * RAW_EXPOSURE_SECONDS,
            exposure_seconds=np.full(n_cadence, CADENCE_SECONDS),
            raw_frame_start_index=starts,
            raw_frame_stop_index_exclusive=stops,
            gain_e_per_dn=np.asarray(1.0),
            manifest={
                "schema_id": "et_mainsim.independent_stamp_production.v1",
                "target_source_id": str(SOURCE_ID),
                "target_source_id_int64": SOURCE_ID,
                "stamp_shape": [15, 15],
                "time_shard": {
                    "raw_frame_interval": {
                        "start_index": shard.raw_start_index,
                        "stop_index": shard.raw_stop_index,
                    }
                },
                "caller_manifest": {
                    "run_id": RUN_ID,
                    "case": case,
                    **(
                        {}
                        if case == "static"
                        else {
                            "target_input_truth": {
                                "variability": {
                                    "enabled": True,
                                    "case": "injected",
                                    "source_factor_snapshot_identity": snapshot_identity,
                                }
                            }
                        }
                    ),
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
            / case
            / "stamps"
            / f"target_{SOURCE_ID}"
            / "delivery"
            / f"shard_{shard.shard_id:05d}"
            / "coadd_30s.h5"
        )
        write_stamp_delivery_bundle(destination, bundle)
    return manifest_path


def test_standard_analysis_writes_formal_lightcurve_quality_and_residual(
    tmp_path: Path,
) -> None:
    from et_mainsim.standard_stamp_analysis import (
        StandardStampAnalysisRequest,
        run_standard_stamp_analysis_v1,
    )

    manifest_path = _write_formal_galaxy_run(tmp_path, case="injected")
    result = run_standard_stamp_analysis_v1(
        StandardStampAnalysisRequest(
            production_manifest_path=manifest_path,
            source_id=SOURCE_ID,
            case="injected",
            cadence_seconds=CADENCE_SECONDS,
            output_dir=tmp_path / "analysis",
            batch_frames=64,
        )
    )

    assert result.reference_lightcurve_path.is_file()
    assert result.analysis_manifest_path.is_file()
    with result.reference_lightcurve_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 720
    assert rows[0]["aperture_valid"] == "0"
    assert rows[0]["aperture_invalid_pixel_count"] == "1"
    assert rows[0]["flux_derived_e"] == ""
    assert "injected_raw_factor_sum" in rows[0]
    assert "model_residual_ppm" in rows[0]
    assert float(rows[1]["model_residual_ppm"]) == pytest.approx(0.0, abs=1e-8)

    analysis = json.loads(result.analysis_manifest_path.read_text(encoding="utf-8"))
    assert analysis["schema_id"] == "et_mainsim.standard_stamp_analysis.v1"
    assert analysis["observation_product"] == "final_dn"
    assert analysis["ordinary_cdpp_label"] == (
        "undetrended_astrophysical_plus_instrument_legacy_compatible_diagnostic"
    )
    assert analysis["observed_cdpp"]["30"]["estimator"] == (
        "legacy_mean_absolute_deviation_times_1.4826"
    )
    assert analysis["cdpp"]["detrending"] == "not_applied"
    receipts = analysis["delivery"]["bundle_receipts"]
    assert len(receipts) == 2
    assert receipts[0]["size_bytes"] > 0
    assert len(receipts[0]["sha256"]) == 64
    assert receipts[0]["validation"]["frame_count"] == 360
    assert analysis["quality"]["invalid_cadence_count"] == 1
    assert analysis["legacy_compatibility"]["legacy_pickle_pca_sg_used"] is False
    assert (
        analysis["reference_photometry"]["cdpp_by_window_minutes"]["390"]["cdpp_ppm"]
        is None
    )
    assert "NaN" not in result.analysis_manifest_path.read_text(encoding="utf-8")
    assert analysis["injected_model_residual"]["cdpp_by_window_minutes"]["30"][
        "cdpp_ppm"
    ] == pytest.approx(0.0, abs=1e-8)


def test_standard_analysis_static_case_has_no_injected_model_residual(
    tmp_path: Path,
) -> None:
    from et_mainsim.standard_stamp_analysis import (
        StandardStampAnalysisRequest,
        run_standard_stamp_analysis_v1,
    )

    manifest_path = _write_formal_galaxy_run(tmp_path, case="static")
    result = run_standard_stamp_analysis_v1(
        StandardStampAnalysisRequest(
            production_manifest_path=manifest_path,
            source_id=SOURCE_ID,
            case="static",
            cadence_seconds=CADENCE_SECONDS,
            output_dir=tmp_path / "analysis",
        )
    )

    analysis = json.loads(result.analysis_manifest_path.read_text(encoding="utf-8"))
    assert analysis["ordinary_cdpp_label"] == (
        "undetrended_static_source_legacy_compatible_diagnostic"
    )
    assert analysis["injected_model_residual"] is None
    with result.reference_lightcurve_path.open(newline="", encoding="utf-8") as stream:
        fields = csv.DictReader(stream).fieldnames
    assert fields is not None
    assert "injected_raw_factor_sum" not in fields


def test_standard_analysis_publish_is_transactional_when_manifest_write_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed second file must not expose a lone reference CSV."""

    import et_mainsim.standard_stamp_analysis as analysis_module

    manifest_path = _write_formal_galaxy_run(tmp_path, case="injected")
    output_dir = tmp_path / "analysis"

    def fail_manifest_write(*args, **kwargs):
        raise OSError("injected manifest write failure")

    monkeypatch.setattr(analysis_module, "_atomic_json", fail_manifest_write)
    with pytest.raises(OSError, match="injected manifest write failure"):
        analysis_module.run_standard_stamp_analysis_v1(
            analysis_module.StandardStampAnalysisRequest(
                production_manifest_path=manifest_path,
                source_id=SOURCE_ID,
                case="injected",
                cadence_seconds=CADENCE_SECONDS,
                output_dir=output_dir,
            )
        )

    assert not output_dir.exists()
    assert not list(tmp_path.glob(".analysis.staging-*"))


def test_standard_analysis_never_overwrites_an_existing_complete_output(
    tmp_path: Path,
) -> None:
    from et_mainsim.standard_stamp_analysis import (
        StandardStampAnalysisRequest,
        run_standard_stamp_analysis_v1,
    )

    manifest_path = _write_formal_galaxy_run(tmp_path, case="injected")
    output_dir = tmp_path / "analysis"
    first = run_standard_stamp_analysis_v1(
        StandardStampAnalysisRequest(
            production_manifest_path=manifest_path,
            source_id=SOURCE_ID,
            case="injected",
            cadence_seconds=CADENCE_SECONDS,
            output_dir=output_dir,
        )
    )
    csv_before = first.reference_lightcurve_path.read_bytes()
    manifest_before = first.analysis_manifest_path.read_bytes()

    with pytest.raises(FileExistsError, match="complete standard analysis"):
        run_standard_stamp_analysis_v1(
            StandardStampAnalysisRequest(
                production_manifest_path=manifest_path,
                source_id=SOURCE_ID,
                case="injected",
                cadence_seconds=CADENCE_SECONDS,
                output_dir=output_dir,
                overwrite=True,
            )
        )

    assert first.reference_lightcurve_path.read_bytes() == csv_before
    assert first.analysis_manifest_path.read_bytes() == manifest_before


def test_standard_analysis_overwrite_archives_only_an_incomplete_output(
    tmp_path: Path,
) -> None:
    from et_mainsim.standard_stamp_analysis import (
        StandardStampAnalysisRequest,
        run_standard_stamp_analysis_v1,
    )

    manifest_path = _write_formal_galaxy_run(tmp_path, case="injected")
    output_dir = tmp_path / "analysis"
    output_dir.mkdir()
    stale_csv = output_dir / "reference_lightcurve.csv"
    stale_csv.write_text("stale partial output\n", encoding="utf-8")

    result = run_standard_stamp_analysis_v1(
        StandardStampAnalysisRequest(
            production_manifest_path=manifest_path,
            source_id=SOURCE_ID,
            case="injected",
            cadence_seconds=CADENCE_SECONDS,
            output_dir=output_dir,
            overwrite=True,
        )
    )

    assert result.analysis_manifest_path.is_file()
    assert result.reference_lightcurve_path.read_text(encoding="utf-8") != (
        "stale partial output\n"
    )
    archived = list(tmp_path.glob(".analysis.incomplete-*"))
    assert len(archived) == 1
    assert (archived[0] / "reference_lightcurve.csv").read_text(
        encoding="utf-8"
    ) == "stale partial output\n"


def test_standard_analysis_fails_closed_until_every_manifest_shard_is_published(
    tmp_path: Path,
) -> None:
    from et_mainsim.standard_stamp_analysis import (
        StandardStampAnalysisNotReadyError,
        StandardStampAnalysisRequest,
        run_standard_stamp_analysis_v1,
    )

    manifest_path = _write_formal_galaxy_run(
        tmp_path,
        case="injected",
        write_delivery=False,
    )
    with pytest.raises(
        StandardStampAnalysisNotReadyError, match="missing formal delivery"
    ):
        run_standard_stamp_analysis_v1(
            StandardStampAnalysisRequest(
                production_manifest_path=manifest_path,
                source_id=SOURCE_ID,
                case="injected",
                cadence_seconds=CADENCE_SECONDS,
                output_dir=tmp_path / "analysis",
            )
        )


def test_standard_analysis_verifies_the_injected_delivery_snapshot_identity(
    tmp_path: Path,
) -> None:
    from et_mainsim.standard_stamp_analysis import (
        StandardStampAnalysisError,
        StandardStampAnalysisRequest,
        run_standard_stamp_analysis_v1,
    )

    manifest_path = _write_formal_galaxy_run(tmp_path, case="injected")
    first_bundle = (
        tmp_path
        / "formal_run"
        / "cases"
        / "injected"
        / "stamps"
        / f"target_{SOURCE_ID}"
        / "delivery"
        / "shard_00000"
        / "coadd_30s.h5"
    )
    with h5py.File(first_bundle, "r+") as handle:
        raw = handle["manifest_json"][()]
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        bundle_manifest = json.loads(raw)
        bundle_manifest["caller_manifest"]["target_input_truth"]["variability"][
            "source_factor_snapshot_identity"
        ]["sha256"] = "0" * 64
        handle["manifest_json"][()] = json.dumps(bundle_manifest)

    with pytest.raises(StandardStampAnalysisError, match="factor snapshot identity"):
        run_standard_stamp_analysis_v1(
            StandardStampAnalysisRequest(
                production_manifest_path=manifest_path,
                source_id=SOURCE_ID,
                case="injected",
                cadence_seconds=CADENCE_SECONDS,
                output_dir=tmp_path / "analysis",
            )
        )


def test_standard_analysis_validates_every_selected_delivery_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import et_mainsim.standard_stamp_analysis as analysis_module

    manifest_path = _write_formal_galaxy_run(tmp_path, case="injected")
    calls: list[Path] = []
    original = analysis_module.validate_stamp_delivery_bundle

    def traced_validation(path: Path | str):
        calls.append(Path(path))
        return original(path)

    monkeypatch.setattr(
        analysis_module,
        "validate_stamp_delivery_bundle",
        traced_validation,
    )
    analysis_module.run_standard_stamp_analysis_v1(
        analysis_module.StandardStampAnalysisRequest(
            production_manifest_path=manifest_path,
            source_id=SOURCE_ID,
            case="injected",
            cadence_seconds=CADENCE_SECONDS,
            output_dir=tmp_path / "analysis",
        )
    )

    assert [path.name for path in calls] == ["coadd_30s.h5", "coadd_30s.h5"]
    assert [path.parent.name for path in calls] == ["shard_00000", "shard_00001"]


def test_standard_analysis_rechecks_delivery_context_after_discovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A valid-but-wrong bundle replacement must not cross the TOCTOU gap."""

    import et_mainsim.standard_stamp_analysis as analysis_module

    manifest_path = _write_formal_galaxy_run(tmp_path, case="injected")
    first_bundle = (
        tmp_path
        / "formal_run"
        / "cases"
        / "injected"
        / "stamps"
        / f"target_{SOURCE_ID}"
        / "delivery"
        / "shard_00000"
        / "coadd_30s.h5"
    )
    original_discover = analysis_module.discover_standard_stamp_analysis_input

    def replace_after_initial_discovery(request):
        resolved = original_discover(request)
        with h5py.File(first_bundle, "r+") as handle:
            raw = handle["manifest_json"][()]
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            bundle_manifest = json.loads(raw)
            bundle_manifest["target_source_id_int64"] = SOURCE_ID + 1
            handle["manifest_json"][()] = json.dumps(bundle_manifest)
        return resolved

    monkeypatch.setattr(
        analysis_module,
        "discover_standard_stamp_analysis_input",
        replace_after_initial_discovery,
    )
    output_dir = tmp_path / "analysis"
    with pytest.raises(
        analysis_module.StandardStampAnalysisError,
        match="formal delivery target does not match request",
    ):
        analysis_module.run_standard_stamp_analysis_v1(
            analysis_module.StandardStampAnalysisRequest(
                production_manifest_path=manifest_path,
                source_id=SOURCE_ID,
                case="injected",
                cadence_seconds=CADENCE_SECONDS,
                output_dir=output_dir,
            )
        )
    assert not output_dir.exists()


def test_standard_analysis_rejects_a_delivery_input_changed_during_reduction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import et_mainsim.standard_stamp_analysis as analysis_module
    from et_mainsim.standard_stamp_analysis import StandardStampAnalysisError

    manifest_path = _write_formal_galaxy_run(tmp_path, case="injected")
    first_bundle = (
        tmp_path
        / "formal_run"
        / "cases"
        / "injected"
        / "stamps"
        / f"target_{SOURCE_ID}"
        / "delivery"
        / "shard_00000"
        / "coadd_30s.h5"
    )
    original = analysis_module.reduce_stamp_delivery_series_v1

    def mutating_reduction(*args, **kwargs):
        result = original(*args, **kwargs)
        stat = first_bundle.stat()
        os.utime(first_bundle, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))
        return result

    monkeypatch.setattr(
        analysis_module,
        "reduce_stamp_delivery_series_v1",
        mutating_reduction,
    )
    output_dir = tmp_path / "analysis"
    with pytest.raises(StandardStampAnalysisError, match="changed during validation"):
        analysis_module.run_standard_stamp_analysis_v1(
            analysis_module.StandardStampAnalysisRequest(
                production_manifest_path=manifest_path,
                source_id=SOURCE_ID,
                case="injected",
                cadence_seconds=CADENCE_SECONDS,
                output_dir=output_dir,
            )
        )
    assert not output_dir.exists()


def test_standard_analysis_cli_uses_the_production_manifest_for_path_discovery(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from et_mainsim.standard_stamp_analysis import main

    manifest_path = _write_formal_galaxy_run(tmp_path, case="injected")
    assert (
        main(
            [
                "--production-manifest",
                str(manifest_path),
                "--source-id",
                str(SOURCE_ID),
                "--case",
                "injected",
                "--cadence-seconds",
                str(int(CADENCE_SECONDS)),
                "--output-dir",
                str(tmp_path / "cli-analysis"),
                "--batch-frames",
                "64",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert Path(payload["analysis_manifest_path"]).is_file()
    assert payload["source_id"] == str(SOURCE_ID)
