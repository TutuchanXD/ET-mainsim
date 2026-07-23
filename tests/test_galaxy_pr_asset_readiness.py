"""Tests for formal Galaxy notebook readiness and opt-in PR-asset export."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import pytest


_PNG_BYTES = b"\x89PNG\r\n\x1a\nminimal-test-png"
_SINGLE_SOURCE_MARKER = "ET_STAMP_REPORT_GATE=galaxy_single_source_raw_coverage_v2_ready"
_CAMPAIGN_SUMMARY_MARKER = "ET_STAMP_REPORT_GATE=galaxy_campaign_raw_coverage_v2_ready"


def _write_json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _content_identity(path: Path) -> dict[str, object]:
    from et_mainsim.stamp_inputs import file_identity

    identity = file_identity(path)
    return {
        "sha256": identity["sha256"],
        "size_bytes": identity["size_bytes"],
    }


def _formal_receipts(
    tmp_path: Path,
    *,
    provenance_ready: bool = True,
    chosen_psf_id: object = "psf-12deg",
) -> dict[str, Path]:
    run_root = tmp_path / "galaxy_independent_90d_v3"
    quality_control = run_root / "quality_control"
    quality_control.mkdir(parents=True)
    source_ids = tuple(str(value) for value in range(100, 110))
    time_plan_path = _write_json(
        run_root / "time_plan.json",
        {"schema_id": "test.time_plan.v1", "complete": True},
    )
    manifest_path = _write_json(
        run_root / "production_manifest.json",
        {
            "schema_id": "et_mainsim.galaxy_stamp_production.v1",
            "schema_version": 2,
            "run_id": run_root.name,
            "observation_product": "final_dn",
            "background_realization_delivered": False,
            "delivery": {"time_plan_relative_path": "time_plan.json"},
            "targets": [
                {"source_id_int64": int(source_id)} for source_id in source_ids
            ],
        },
    )
    manifest_identity = _content_identity(manifest_path)
    qc_path = _write_json(
        quality_control / "injected_campaign_delivery_qc.json",
        {
            "schema_id": "et_mainsim.galaxy_campaign_delivery_qc.v1",
            "schema_version": 1,
            "ready": True,
            "run_root": str(run_root),
            "run_id": run_root.name,
            "case": "injected",
            "manifest_identity": manifest_identity,
            "time_plan_identity": _content_identity(time_plan_path),
            "expected_bundle_count": 50,
            "valid_bundle_count": 50,
        },
    )
    audit_path = _write_json(
        quality_control / "injected_campaign_provenance_psf_audit.json",
        {
            "schema_id": "et_mainsim.galaxy_delivery_provenance_audit.v1",
            "schema_version": 1,
            "ready": provenance_ready,
            "run_root": str(run_root),
            "run_id": run_root.name,
            "case": "injected",
            "production_manifest_identity": manifest_identity,
            "expected_bundle_count": 50,
            "valid_bundle_count": 50 if provenance_ready else 49,
            "missing_bundle_count": 0 if provenance_ready else 1,
            "invalid_bundle_count": 0,
            "source_summaries": {
                source_id: {
                    "source_id": source_id,
                    "expected_bundle_count": 5,
                    "valid_bundle_count": 5 if provenance_ready else 4,
                    "missing_bundle_count": 0 if provenance_ready else 1,
                    "invalid_bundle_count": 0,
                    "chosen_psf_id": chosen_psf_id,
                    "node_angle_deg": 12.0,
                    "runtime_registry_attestation_verified": provenance_ready,
                    "runtime_registry_semantic_content_sha256": "a" * 64,
                    "runtime_registry_attestation_record_sha256": "b" * 64,
                }
                for source_id in source_ids
            },
        },
    )
    return {
        "run_root": run_root,
        "manifest": manifest_path,
        "qc": qc_path,
        "audit": audit_path,
    }


def _readiness_request(paths: dict[str, Path]):
    from et_mainsim.notebook_report_assets import GalaxyNotebookReadinessRequest

    return GalaxyNotebookReadinessRequest(
        production_manifest_path=paths["manifest"],
        campaign_qc_path=paths["qc"],
        provenance_audit_path=paths["audit"],
    )


def _campaign_summary_receipts(paths: dict[str, Path]) -> dict[str, Path]:
    run_root = paths["run_root"]
    policy_path = run_root / "analysis" / "raw_10s_coverage_v2_policy.json"
    policy_path.parent.mkdir(parents=True)
    policy = {
        "schema_id": "et_mainsim.raw_coverage_aware_policy.v1",
        "schema_version": 1,
        "complete": True,
        "run_id": run_root.name,
        "case": "injected",
        "observation_product": "final_dn",
        "background_realization_used": False,
        "production_manifest_identity": _content_identity(paths["manifest"]),
        "coverage": {
            "windows_minutes": [30, 90, 390],
            "minimum_coverage_fraction": 0.9,
            "minimum_accepted_bins": 2,
            "bin_origin_seconds": 0.0,
            "invalid_cadence_handling": (
                "omit_whole_invalid_cadences_without_pixel_or_flux_imputation"
            ),
            "accepted_bin_normalization": "actual_effective_exposure_only",
        },
    }
    _write_json(policy_path, policy)
    summary_dir = (
        run_root / "analysis" / "campaign" / "injected" / "raw_10s_coverage_v2_summary"
    )
    summary_dir.mkdir(parents=True)
    source_summary_path = summary_dir / "source_summary.csv"
    source_metrics_path = summary_dir / "source_window_metrics.csv"
    source_summary_path.write_text("source_id\n100\n", encoding="utf-8")
    source_metrics_path.write_text("source_id,observed_cdpp_ppm\n100,12.5\n", encoding="utf-8")
    manifest = {
        "schema_id": "et_mainsim.galaxy_raw_coverage_v2_campaign_summary.v1",
        "schema_version": 1,
        "complete": True,
        "ready": True,
        "run_id": run_root.name,
        "case": "injected",
        "observation_product": "final_dn",
        "background_realization_used": False,
        "source_count": 10,
        "production_manifest": {"identity": _content_identity(paths["manifest"])},
        "campaign_qc": {"identity": _content_identity(paths["qc"])},
        "frozen_coverage_policy": {
            "identity": _content_identity(policy_path),
            "record": policy["coverage"],
        },
        "source_artifacts": [
            {"source_id": str(value)} for value in range(100, 110)
        ],
        "tables": {
            "source_summary": {
                "path": source_summary_path.name,
                "format": "csv",
                "identity": _content_identity(source_summary_path),
            },
            "source_window_metrics": {
                "path": source_metrics_path.name,
                "format": "csv",
                "identity": _content_identity(source_metrics_path),
            },
        },
    }
    summary_path = _write_json(summary_dir / "campaign_summary_manifest.json", manifest)
    return {
        "summary": summary_path,
        "policy": policy_path,
        "source_summary": source_summary_path,
        "source_metrics": source_metrics_path,
    }


def _executed_notebook_path(
    path: Path,
    *,
    marker: str = _SINGLE_SOURCE_MARKER,
) -> Path:
    encoded = base64.b64encode(_PNG_BYTES).decode("ascii")
    return _write_json(
        path,
        {
            "nbformat": 4,
            "nbformat_minor": 5,
            "cells": [
                {
                    "cell_type": "code",
                    "id": "quality-figure",
                    "outputs": [
                        {
                            "output_type": "stream",
                            "name": "stdout",
                            "text": marker,
                        },
                        {
                            "output_type": "display_data",
                            "data": {"image/png": encoded},
                            "metadata": {},
                        },
                    ],
                }
            ],
        },
    )


def test_readiness_requires_a_ready_manifest_bound_provenance_audit(
    tmp_path: Path,
) -> None:
    from et_mainsim.notebook_report_assets import (
        NotebookReportAssetError,
        validate_galaxy_notebook_readiness_v1,
    )

    paths = _formal_receipts(tmp_path, provenance_ready=False)

    with pytest.raises(NotebookReportAssetError, match="provenance audit is not ready"):
        validate_galaxy_notebook_readiness_v1(_readiness_request(paths))

    assert not (paths["run_root"] / "presentation").exists()


def test_nullable_cdpp_values_convert_blanks_and_masks_to_nan() -> None:
    from et_mainsim.notebook_report_assets import nullable_cdpp_values_to_nan_v1

    values = np.ma.array(
        ["12.5", "", "  ", "7.0"],
        mask=[False, False, True, False],
    )

    result = nullable_cdpp_values_to_nan_v1(values)

    np.testing.assert_allclose(result[[0, 3]], [12.5, 7.0])
    assert np.isnan(result[1])
    assert np.isnan(result[2])


def test_readiness_accepts_integer_psf_ids_from_the_provenance_audit(
    tmp_path: Path,
) -> None:
    from et_mainsim.notebook_report_assets import (
        validate_galaxy_notebook_readiness_v1,
    )

    paths = _formal_receipts(tmp_path, chosen_psf_id=1)

    readiness = validate_galaxy_notebook_readiness_v1(_readiness_request(paths))

    assert readiness.run_id == "galaxy_independent_90d_v3"


def test_readiness_binds_provenance_bundle_counts_to_campaign_qc(
    tmp_path: Path,
) -> None:
    from et_mainsim.notebook_report_assets import (
        NotebookReportAssetError,
        validate_galaxy_notebook_readiness_v1,
    )

    paths = _formal_receipts(tmp_path)
    audit = json.loads(paths["audit"].read_text(encoding="utf-8"))
    audit["expected_bundle_count"] = 40
    audit["valid_bundle_count"] = 40
    for source in audit["source_summaries"].values():
        source["expected_bundle_count"] = 4
        source["valid_bundle_count"] = 4
    _write_json(paths["audit"], audit)

    with pytest.raises(
        NotebookReportAssetError,
        match="provenance audit bundle counts conflict with campaign QC",
    ):
        validate_galaxy_notebook_readiness_v1(_readiness_request(paths))


def test_readiness_refuses_a_changed_campaign_summary_metrics_table(
    tmp_path: Path,
) -> None:
    from et_mainsim.notebook_report_assets import (
        GalaxyNotebookReadinessRequest,
        NotebookReportAssetError,
        validate_galaxy_notebook_readiness_v1,
    )

    paths = _formal_receipts(tmp_path)
    summary = _campaign_summary_receipts(paths)
    summary["source_metrics"].write_text(
        "source_id,observed_cdpp_ppm\n100,999.0\n", encoding="utf-8"
    )

    with pytest.raises(NotebookReportAssetError, match="source window metrics table identity"):
        validate_galaxy_notebook_readiness_v1(
            GalaxyNotebookReadinessRequest(
                production_manifest_path=paths["manifest"],
                campaign_qc_path=paths["qc"],
                provenance_audit_path=paths["audit"],
                campaign_summary_manifest_path=summary["summary"],
                coverage_policy_path=summary["policy"],
            )
        )


def test_readiness_refuses_an_unexpected_campaign_summary_table_path(
    tmp_path: Path,
) -> None:
    from et_mainsim.notebook_report_assets import (
        GalaxyNotebookReadinessRequest,
        NotebookReportAssetError,
        validate_galaxy_notebook_readiness_v1,
    )

    paths = _formal_receipts(tmp_path)
    summary = _campaign_summary_receipts(paths)
    payload = json.loads(summary["summary"].read_text(encoding="utf-8"))
    payload["tables"]["source_summary"]["path"] = "other.csv"
    _write_json(summary["summary"], payload)

    with pytest.raises(NotebookReportAssetError, match="source summary table path"):
        validate_galaxy_notebook_readiness_v1(
            GalaxyNotebookReadinessRequest(
                production_manifest_path=paths["manifest"],
                campaign_qc_path=paths["qc"],
                provenance_audit_path=paths["audit"],
                campaign_summary_manifest_path=summary["summary"],
                coverage_policy_path=summary["policy"],
            )
        )


def test_readiness_refuses_a_campaign_qc_copy_outside_the_formal_run_root(
    tmp_path: Path,
) -> None:
    from et_mainsim.notebook_report_assets import (
        GalaxyNotebookReadinessRequest,
        NotebookReportAssetError,
        validate_galaxy_notebook_readiness_v1,
    )

    paths = _formal_receipts(tmp_path)
    copied_qc = _write_json(
        tmp_path / "copied_campaign_qc.json",
        json.loads(paths["qc"].read_text(encoding="utf-8")),
    )

    with pytest.raises(NotebookReportAssetError, match="canonical campaign QC path"):
        validate_galaxy_notebook_readiness_v1(
            GalaxyNotebookReadinessRequest(
                production_manifest_path=paths["manifest"],
                campaign_qc_path=copied_qc,
                provenance_audit_path=paths["audit"],
            )
        )


def test_readiness_refuses_a_provenance_audit_copy_outside_the_formal_run_root(
    tmp_path: Path,
) -> None:
    from et_mainsim.notebook_report_assets import (
        GalaxyNotebookReadinessRequest,
        NotebookReportAssetError,
        validate_galaxy_notebook_readiness_v1,
    )

    paths = _formal_receipts(tmp_path)
    copied_audit = _write_json(
        tmp_path / "copied_provenance_audit.json",
        json.loads(paths["audit"].read_text(encoding="utf-8")),
    )

    with pytest.raises(NotebookReportAssetError, match="canonical provenance audit path"):
        validate_galaxy_notebook_readiness_v1(
            GalaxyNotebookReadinessRequest(
                production_manifest_path=paths["manifest"],
                campaign_qc_path=paths["qc"],
                provenance_audit_path=copied_audit,
            )
        )


def test_readiness_refuses_a_provenance_receipt_with_a_conflicting_run_root(
    tmp_path: Path,
) -> None:
    from et_mainsim.notebook_report_assets import (
        NotebookReportAssetError,
        validate_galaxy_notebook_readiness_v1,
    )

    paths = _formal_receipts(tmp_path)
    audit = json.loads(paths["audit"].read_text(encoding="utf-8"))
    audit["run_root"] = str(tmp_path / "other-run")
    _write_json(paths["audit"], audit)

    with pytest.raises(NotebookReportAssetError, match="provenance audit run_root"):
        validate_galaxy_notebook_readiness_v1(_readiness_request(paths))


def test_readiness_refuses_a_campaign_qc_receipt_with_a_conflicting_run_root(
    tmp_path: Path,
) -> None:
    from et_mainsim.notebook_report_assets import (
        NotebookReportAssetError,
        validate_galaxy_notebook_readiness_v1,
    )

    paths = _formal_receipts(tmp_path)
    campaign_qc = json.loads(paths["qc"].read_text(encoding="utf-8"))
    campaign_qc["run_root"] = str(tmp_path / "other-run")
    _write_json(paths["qc"], campaign_qc)

    with pytest.raises(NotebookReportAssetError, match="campaign QC run_root"):
        validate_galaxy_notebook_readiness_v1(_readiness_request(paths))


@pytest.mark.parametrize("chosen_psf_id", [True, -1, 1.5, "   "])
def test_readiness_refuses_nonphysical_psf_ids(
    tmp_path: Path,
    chosen_psf_id: object,
) -> None:
    from et_mainsim.notebook_report_assets import (
        NotebookReportAssetError,
        validate_galaxy_notebook_readiness_v1,
    )

    paths = _formal_receipts(tmp_path, chosen_psf_id=chosen_psf_id)

    with pytest.raises(NotebookReportAssetError, match="PSF ID"):
        validate_galaxy_notebook_readiness_v1(_readiness_request(paths))


def test_pr_asset_export_is_an_explicit_opt_in_outside_the_run_root(
    tmp_path: Path,
) -> None:
    from et_mainsim.notebook_report_assets import (
        NotebookReportAssetError,
        resolve_galaxy_pr_asset_export_root_v1,
        validate_galaxy_notebook_readiness_v1,
    )

    paths = _formal_receipts(tmp_path)
    readiness = validate_galaxy_notebook_readiness_v1(_readiness_request(paths))

    assert (
        resolve_galaxy_pr_asset_export_root_v1(
            readiness,
            environment={},
        )
        is None
    )

    report_root = tmp_path / "pr-assets"
    assert resolve_galaxy_pr_asset_export_root_v1(
        readiness,
        environment={
            "ET_STAMP_WRITE_PR_ASSETS": "1",
            "ET_STAMP_PRESENTATION_DIR": str(report_root),
        },
    ) == report_root.resolve()

    with pytest.raises(NotebookReportAssetError, match="outside the production root"):
        resolve_galaxy_pr_asset_export_root_v1(
            readiness,
            environment={
                "ET_STAMP_WRITE_PR_ASSETS": "1",
                "ET_STAMP_PRESENTATION_DIR": str(paths["run_root"] / "presentation"),
            },
        )


def test_requested_pr_asset_export_fails_closed_before_readiness(
    tmp_path: Path,
) -> None:
    from et_mainsim.notebook_report_assets import (
        NotebookReportAssetError,
        resolve_galaxy_pr_asset_export_root_v1,
    )

    run_root = tmp_path / "galaxy_independent_90d_v3"
    report_root = tmp_path / "pr-assets"

    with pytest.raises(NotebookReportAssetError, match="before Galaxy readiness"):
        resolve_galaxy_pr_asset_export_root_v1(
            None,
            run_root=run_root,
            environment={
                "ET_STAMP_WRITE_PR_ASSETS": "1",
                "ET_STAMP_PRESENTATION_DIR": str(report_root),
            },
        )

    assert not report_root.exists()


def test_pr_asset_wrapper_refuses_an_incomplete_audit_before_writing(
    tmp_path: Path,
) -> None:
    from et_mainsim.notebook_report_assets import (
        GalaxyPrAssetExportRequest,
        NotebookPngAssetSpec,
        NotebookReportAssetError,
        export_galaxy_pr_assets_v1,
    )

    paths = _formal_receipts(tmp_path, provenance_ready=False)
    report_root = tmp_path / "pr-assets"
    request = GalaxyPrAssetExportRequest(
        readiness=_readiness_request(paths),
        executed_notebook_path=_executed_notebook_path(tmp_path / "executed.ipynb"),
        asset_specs=(NotebookPngAssetSpec("quality-figure", "quality.png"),),
        required_markers=(_SINGLE_SOURCE_MARKER,),
        asset_scope="single_source",
        environment={
            "ET_STAMP_WRITE_PR_ASSETS": "1",
            "ET_STAMP_PRESENTATION_DIR": str(report_root),
        },
    )

    with pytest.raises(NotebookReportAssetError, match="provenance audit is not ready"):
        export_galaxy_pr_assets_v1(request)

    assert not report_root.exists()


def test_pr_asset_wrapper_records_the_final_dn_readiness_chain(tmp_path: Path) -> None:
    from et_mainsim.notebook_report_assets import (
        GalaxyPrAssetExportRequest,
        NotebookPngAssetSpec,
        export_galaxy_pr_assets_v1,
    )

    paths = _formal_receipts(tmp_path)
    report_root = tmp_path / "pr-assets"
    result = export_galaxy_pr_assets_v1(
        GalaxyPrAssetExportRequest(
            readiness=_readiness_request(paths),
            executed_notebook_path=_executed_notebook_path(tmp_path / "executed.ipynb"),
            asset_specs=(NotebookPngAssetSpec("quality-figure", "quality.png"),),
            required_markers=(_SINGLE_SOURCE_MARKER,),
            asset_scope="single_source",
            environment={
                "ET_STAMP_WRITE_PR_ASSETS": "1",
                "ET_STAMP_PRESENTATION_DIR": str(report_root),
            },
        )
    )

    assert result is not None
    receipt = json.loads(result.receipt_path.read_text(encoding="utf-8"))
    readiness = receipt["publication_context"]["galaxy_readiness"]
    assert readiness["observation_product"] == "final_dn"
    assert "legacy-MAD-compatible" in readiness["analysis_boundary"]
    assert receipt["assets"][0]["filename"] == "quality.png"


def test_generic_export_refuses_a_formal_galaxy_manifest_without_readiness(
    tmp_path: Path,
) -> None:
    from et_mainsim.notebook_report_assets import (
        ExecutedNotebookReportAssetRequest,
        NotebookPngAssetSpec,
        NotebookReportAssetError,
        export_executed_notebook_png_assets_v1,
    )

    paths = _formal_receipts(tmp_path)

    with pytest.raises(NotebookReportAssetError, match="require export_galaxy_pr_assets_v1"):
        export_executed_notebook_png_assets_v1(
            ExecutedNotebookReportAssetRequest(
                executed_notebook_path=_executed_notebook_path(tmp_path / "executed.ipynb"),
                report_root=tmp_path / "generic-pr-assets",
                production_manifest_path=paths["manifest"],
                asset_specs=(NotebookPngAssetSpec("quality-figure", "quality.png"),),
                required_markers=("ET_STAMP_REPORT_GATE=ready",),
            )
        )


def test_single_source_scope_refuses_the_campaign_summary_marker(tmp_path: Path) -> None:
    from et_mainsim.notebook_report_assets import (
        GalaxyPrAssetExportRequest,
        NotebookPngAssetSpec,
        NotebookReportAssetError,
        export_galaxy_pr_assets_v1,
    )

    paths = _formal_receipts(tmp_path)
    with pytest.raises(NotebookReportAssetError, match="single_source.*marker"):
        export_galaxy_pr_assets_v1(
            GalaxyPrAssetExportRequest(
                readiness=_readiness_request(paths),
                executed_notebook_path=_executed_notebook_path(
                    tmp_path / "executed.ipynb", marker=_CAMPAIGN_SUMMARY_MARKER
                ),
                asset_specs=(NotebookPngAssetSpec("quality-figure", "quality.png"),),
                required_markers=(_CAMPAIGN_SUMMARY_MARKER,),
                asset_scope="single_source",
                environment={
                    "ET_STAMP_WRITE_PR_ASSETS": "1",
                    "ET_STAMP_PRESENTATION_DIR": str(tmp_path / "pr-assets"),
                },
            )
        )


def test_campaign_summary_scope_requires_summary_and_policy(tmp_path: Path) -> None:
    from et_mainsim.notebook_report_assets import (
        GalaxyPrAssetExportRequest,
        NotebookPngAssetSpec,
        NotebookReportAssetError,
        export_galaxy_pr_assets_v1,
    )

    paths = _formal_receipts(tmp_path)
    with pytest.raises(NotebookReportAssetError, match="campaign_summary.*summary.*policy"):
        export_galaxy_pr_assets_v1(
            GalaxyPrAssetExportRequest(
                readiness=_readiness_request(paths),
                executed_notebook_path=_executed_notebook_path(
                    tmp_path / "executed.ipynb", marker=_CAMPAIGN_SUMMARY_MARKER
                ),
                asset_specs=(NotebookPngAssetSpec("quality-figure", "quality.png"),),
                required_markers=(_CAMPAIGN_SUMMARY_MARKER,),
                asset_scope="campaign_summary",
                environment={
                    "ET_STAMP_WRITE_PR_ASSETS": "1",
                    "ET_STAMP_PRESENTATION_DIR": str(tmp_path / "pr-assets"),
                },
            )
        )


def test_campaign_summary_scope_exports_only_with_summary_receipts(
    tmp_path: Path,
) -> None:
    from et_mainsim.notebook_report_assets import (
        GalaxyNotebookReadinessRequest,
        GalaxyPrAssetExportRequest,
        NotebookPngAssetSpec,
        export_galaxy_pr_assets_v1,
    )

    paths = _formal_receipts(tmp_path)
    summary = _campaign_summary_receipts(paths)
    result = export_galaxy_pr_assets_v1(
        GalaxyPrAssetExportRequest(
            readiness=GalaxyNotebookReadinessRequest(
                production_manifest_path=paths["manifest"],
                campaign_qc_path=paths["qc"],
                provenance_audit_path=paths["audit"],
                campaign_summary_manifest_path=summary["summary"],
                coverage_policy_path=summary["policy"],
            ),
            executed_notebook_path=_executed_notebook_path(
                tmp_path / "executed.ipynb", marker=_CAMPAIGN_SUMMARY_MARKER
            ),
            asset_specs=(NotebookPngAssetSpec("quality-figure", "quality.png"),),
            required_markers=(_CAMPAIGN_SUMMARY_MARKER,),
            asset_scope="campaign_summary",
            environment={
                "ET_STAMP_WRITE_PR_ASSETS": "1",
                "ET_STAMP_PRESENTATION_DIR": str(tmp_path / "pr-assets"),
            },
        )
    )

    assert result is not None
    context = json.loads(result.receipt_path.read_text(encoding="utf-8"))["publication_context"]
    assert context["galaxy_readiness"]["asset_scope"] == "campaign_summary"


def test_generic_cli_refuses_a_formal_galaxy_manifest_without_readiness(
    tmp_path: Path,
) -> None:
    from et_mainsim.notebook_report_assets import main

    paths = _formal_receipts(tmp_path)
    report_root = tmp_path / "generic-cli-assets"

    result = main(
        (
            "--executed-notebook",
            str(_executed_notebook_path(tmp_path / "executed.ipynb")),
            "--report-root",
            str(report_root),
            "--production-manifest",
            str(paths["manifest"]),
            "--asset",
            "quality-figure=quality.png",
            "--required-marker",
            "ET_STAMP_REPORT_GATE=ready",
        )
    )

    assert result == 2
    assert not report_root.exists()


def test_galaxy_safe_cli_exports_only_after_formal_readiness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from et_mainsim.notebook_report_assets import main

    paths = _formal_receipts(tmp_path)
    report_root = tmp_path / "galaxy-cli-assets"
    monkeypatch.setenv("ET_STAMP_WRITE_PR_ASSETS", "1")
    monkeypatch.setenv("ET_STAMP_PRESENTATION_DIR", str(report_root))

    result = main(
        (
            "--executed-notebook",
            str(_executed_notebook_path(tmp_path / "executed.ipynb")),
            "--production-manifest",
            str(paths["manifest"]),
            "--galaxy-campaign-qc",
            str(paths["qc"]),
            "--galaxy-provenance-audit",
            str(paths["audit"]),
            "--galaxy-asset-scope",
            "single_source",
            "--asset",
            "quality-figure=quality.png",
            "--required-marker",
            _SINGLE_SOURCE_MARKER,
        )
    )

    assert result == 0
    assert (report_root / "assets" / "quality.png").is_file()


def test_galaxy_safe_cli_requires_summary_and_policy_together(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from et_mainsim.notebook_report_assets import main

    paths = _formal_receipts(tmp_path)
    monkeypatch.setenv("ET_STAMP_WRITE_PR_ASSETS", "1")
    monkeypatch.setenv("ET_STAMP_PRESENTATION_DIR", str(tmp_path / "pr-assets"))

    result = main(
        (
            "--executed-notebook",
            str(_executed_notebook_path(tmp_path / "executed.ipynb")),
            "--production-manifest",
            str(paths["manifest"]),
            "--galaxy-campaign-qc",
            str(paths["qc"]),
            "--galaxy-provenance-audit",
            str(paths["audit"]),
            "--galaxy-asset-scope",
            "campaign_summary",
            "--galaxy-campaign-summary",
            str(tmp_path / "campaign_summary_manifest.json"),
            "--asset",
            "quality-figure=quality.png",
            "--required-marker",
            _CAMPAIGN_SUMMARY_MARKER,
        )
    )

    assert result == 2
    assert "campaign summary and frozen coverage policy" in capsys.readouterr().err


def test_galaxy_safe_cli_requires_an_explicit_asset_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from et_mainsim.notebook_report_assets import main

    paths = _formal_receipts(tmp_path)
    monkeypatch.setenv("ET_STAMP_WRITE_PR_ASSETS", "1")
    monkeypatch.setenv("ET_STAMP_PRESENTATION_DIR", str(tmp_path / "pr-assets"))

    result = main(
        (
            "--executed-notebook",
            str(_executed_notebook_path(tmp_path / "executed.ipynb")),
            "--production-manifest",
            str(paths["manifest"]),
            "--galaxy-campaign-qc",
            str(paths["qc"]),
            "--galaxy-provenance-audit",
            str(paths["audit"]),
            "--asset",
            "quality-figure=quality.png",
            "--required-marker",
            _SINGLE_SOURCE_MARKER,
        )
    )

    assert result == 2
    assert "--galaxy-asset-scope" in capsys.readouterr().err
