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
    assert "run_standard_stamp_analysis_v1" not in source
    assert "StandardStampAnalysisRequest" not in source
    assert "discover_standard_stamp_analysis_input" not in source
    assert "analysis_status.json" not in source
    assert "standard_analysis_pointer.json" not in source
    assert ".write_text(" not in source
    assert ".mkdir(" not in source
