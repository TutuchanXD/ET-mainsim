"""Static contracts for notebooks consuming immutable Galaxy delivery products."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_galaxy_delivery_notebook_is_a_read_only_raw_coverage_v2_consumer() -> None:
    notebook_path = (
        Path(__file__).resolve().parents[2]
        / "Galaxy"
        / "data-2"
        / "analyze_et_stamp_90d_independent_delivery.ipynb"
    )
    if not notebook_path.is_file():
        pytest.skip("the Galaxy science-notebook companion is not in this checkout")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", ())) for cell in notebook["cells"])

    assert "raw_10s_coverage_v2" in source
    assert "coverage_aware_analysis_manifest.json" in source
    assert "injected_campaign_delivery_qc.json" in source
    assert "injected_campaign_provenance_psf_audit.json" in source
    assert "et_mainsim.raw_coverage_aware_stamp_analysis.v3" in source
    assert "et_mainsim.galaxy_delivery_provenance_audit.v1" in source
    assert "provenance_audit['ready'] is True" in source
    assert "provenance_audit['production_manifest_identity']" in source
    assert "ET_STAMP_REPORT_GATE=galaxy_single_source_raw_coverage_v2_ready" in source
    assert "content_identity(" in source
    assert "qc['manifest_identity'] == file_identity(manifest_path)" not in source
    assert "input_reference['analysis_manifest'] == file_identity(" not in source
    assert "ET_STAMP_MINIMUM_COVERAGE_FRACTION" not in source
    assert "run_standard_stamp_analysis_v1" not in source
    assert "StandardStampAnalysisRequest" not in source
    assert "discover_standard_stamp_analysis_input" not in source
    assert "analysis_status.json" not in source
    assert "standard_analysis_pointer.json" not in source
    assert ".write_text(" not in source
    assert ".mkdir(" not in source


def test_galaxy_campaign_summary_notebook_only_consumes_atomic_raw_coverage_summary() -> None:
    notebook_path = (
        Path(__file__).resolve().parents[2]
        / "Galaxy"
        / "data-2"
        / "summarize_et_stamp_90d_independent_delivery.ipynb"
    )
    if not notebook_path.is_file():
        pytest.skip("the Galaxy science-notebook companion is not in this checkout")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", ())) for cell in notebook["cells"])

    assert "raw_10s_coverage_v2_summary" in source
    assert "et_mainsim.galaxy_raw_coverage_v2_campaign_summary.v1" in source
    assert "campaign_summary_manifest.json" in source
    assert "source_window_metrics.csv" in source
    assert "injected_campaign_provenance_psf_audit.json" in source
    assert "et_mainsim.galaxy_delivery_provenance_audit.v1" in source
    assert "provenance_audit['ready'] is True" in source
    assert "provenance_audit['production_manifest_identity']" in source
    assert "ET_STAMP_REPORT_GATE=galaxy_campaign_raw_coverage_v2_ready" in source
    assert "coadd_60s" not in source
    assert "run_standard_stamp_analysis_v1" not in source
    assert ".write_text(" not in source
    assert ".mkdir(" not in source
