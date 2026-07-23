"""Tests for formal Galaxy notebook readiness and opt-in PR-asset export."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import pytest


_PNG_BYTES = b"\x89PNG\r\n\x1a\nminimal-test-png"


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


def _executed_notebook_path(path: Path) -> Path:
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
                            "text": "ET_STAMP_REPORT_GATE=ready",
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
        required_markers=("ET_STAMP_REPORT_GATE=ready",),
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
            required_markers=("ET_STAMP_REPORT_GATE=ready",),
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
            "--asset",
            "quality-figure=quality.png",
            "--required-marker",
            "ET_STAMP_REPORT_GATE=ready",
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
            "--galaxy-campaign-summary",
            str(tmp_path / "campaign_summary_manifest.json"),
            "--asset",
            "quality-figure=quality.png",
            "--required-marker",
            "ET_STAMP_REPORT_GATE=ready",
        )
    )

    assert result == 2
    assert "campaign summary and frozen coverage policy" in capsys.readouterr().err
