"""Tests for fail-closed publication of executed-notebook PNG report assets."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest


_PNG_BYTES = b"\x89PNG\r\n\x1a\nnot-a-real-image-but-a-valid-signature"


def _write_json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _executed_notebook(*, marker: str = "REPORT_GATE=READY") -> dict[str, object]:
    encoded = base64.b64encode(_PNG_BYTES).decode("ascii")
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "cells": [
            {
                "cell_type": "code",
                "id": "first-figure",
                "source": ["plt.show()\n"],
                "outputs": [
                    {"output_type": "stream", "name": "stdout", "text": marker},
                    {
                        "output_type": "display_data",
                        "data": {"image/png": encoded},
                        "metadata": {},
                    },
                ],
            },
            {
                "cell_type": "code",
                "id": "second-figure",
                "source": ["plt.show()\n"],
                "outputs": [
                    {
                        "output_type": "execute_result",
                        "data": {"image/png": encoded},
                        "metadata": {},
                        "execution_count": 1,
                    }
                ],
            },
        ],
    }


def _production_manifest() -> dict[str, object]:
    return {
        "schema_id": "et_mainsim.galaxy_stamp_production.v1",
        "run_id": "galaxy_independent_90d_v3",
        "observation_product": "final_dn",
    }


def _request(tmp_path: Path, *, marker: str = "REPORT_GATE=READY"):
    from et_mainsim.notebook_report_assets import (
        ExecutedNotebookReportAssetRequest,
        NotebookPngAssetSpec,
    )

    production_root = tmp_path / "production"
    production_root.mkdir()
    manifest_path = _write_json(
        production_root / "production_manifest.json", _production_manifest()
    )
    notebook_path = _write_json(
        tmp_path / "executed.ipynb", _executed_notebook(marker=marker)
    )
    return ExecutedNotebookReportAssetRequest(
        executed_notebook_path=notebook_path,
        report_root=tmp_path / "report",
        production_manifest_path=manifest_path,
        asset_specs=(
            NotebookPngAssetSpec("first-figure", "one.png"),
            NotebookPngAssetSpec("second-figure", "two.png"),
        ),
        required_markers=("REPORT_GATE=READY",),
    )


def test_exports_selected_png_outputs_with_atomic_receipts(tmp_path: Path) -> None:
    from et_mainsim.notebook_report_assets import export_executed_notebook_png_assets_v1

    result = export_executed_notebook_png_assets_v1(_request(tmp_path))

    assert result.published_asset_dir == tmp_path / "report" / "assets"
    assert (result.published_asset_dir / "one.png").read_bytes() == _PNG_BYTES
    assert (result.published_asset_dir / "two.png").read_bytes() == _PNG_BYTES
    receipt = json.loads(result.receipt_path.read_text(encoding="utf-8"))
    assert receipt["schema_id"] == "et_mainsim.executed_notebook_report_assets.v1"
    assert receipt["complete"] is True
    assert receipt["run_id"] == "galaxy_independent_90d_v3"
    assert receipt["observation_product"] == "final_dn"
    assert [asset["filename"] for asset in receipt["assets"]] == ["one.png", "two.png"]
    assert (result.published_asset_dir / "one.png.receipt.json").is_file()
    assert not list((tmp_path / "report").glob("*.partial"))


def test_refuses_pending_notebook_without_publishing_assets(tmp_path: Path) -> None:
    from et_mainsim.notebook_report_assets import (
        NotebookReportAssetError,
        export_executed_notebook_png_assets_v1,
    )

    with pytest.raises(NotebookReportAssetError, match="required readiness marker"):
        export_executed_notebook_png_assets_v1(
            _request(tmp_path, marker="REPORT_GATE=PENDING")
        )

    assert not (tmp_path / "report" / "assets").exists()


def test_refuses_a_report_root_under_the_production_root(tmp_path: Path) -> None:
    from et_mainsim.notebook_report_assets import (
        ExecutedNotebookReportAssetRequest,
        NotebookReportAssetError,
        export_executed_notebook_png_assets_v1,
    )

    request = _request(tmp_path)
    unsafe_request = ExecutedNotebookReportAssetRequest(
        executed_notebook_path=request.executed_notebook_path,
        report_root=request.production_manifest_path.parent / "presentation",
        production_manifest_path=request.production_manifest_path,
        asset_specs=request.asset_specs,
        required_markers=request.required_markers,
    )

    with pytest.raises(NotebookReportAssetError, match="must be outside the production root"):
        export_executed_notebook_png_assets_v1(unsafe_request)

    assert not (unsafe_request.report_root / "assets").exists()


def test_cli_reports_a_pending_notebook_without_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from et_mainsim.notebook_report_assets import main

    request = _request(tmp_path, marker="REPORT_GATE=PENDING")
    exit_code = main(
        [
            "--executed-notebook",
            str(request.executed_notebook_path),
            "--report-root",
            str(request.report_root),
            "--production-manifest",
            str(request.production_manifest_path),
            "--asset",
            "first-figure=one.png",
            "--required-marker",
            "REPORT_GATE=READY",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "report asset export failed" in captured.err
    assert "Traceback" not in captured.err
    assert not (request.report_root / "assets").exists()
