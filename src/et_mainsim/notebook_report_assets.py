"""Fail-closed publication of PNG assets from an executed science notebook.

This module deliberately does not execute notebooks and never writes into a
production run root.  Its only responsibility is to extract explicitly named
``image/png`` outputs from an already executed notebook, verify a readiness
marker emitted by that notebook, and atomically publish the images with their
provenance receipts into a separate report directory.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import argparse
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Literal

import numpy as np

from .stamp_inputs import file_identity


NOTEBOOK_REPORT_ASSETS_SCHEMA_ID = "et_mainsim.executed_notebook_report_assets.v1"
NOTEBOOK_REPORT_ASSETS_SCHEMA_VERSION = 1
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_GALAXY_MANIFEST_SCHEMA_ID = "et_mainsim.galaxy_stamp_production.v1"
_GALAXY_MANIFEST_SCHEMA_VERSION = 2
_GALAXY_CAMPAIGN_QC_SCHEMA_ID = "et_mainsim.galaxy_campaign_delivery_qc.v1"
_GALAXY_CAMPAIGN_QC_SCHEMA_VERSION = 1
_GALAXY_PROVENANCE_AUDIT_SCHEMA_ID = (
    "et_mainsim.galaxy_delivery_provenance_audit.v1"
)
_GALAXY_PROVENANCE_AUDIT_SCHEMA_VERSION = 1
_GALAXY_SUMMARY_SCHEMA_ID = "et_mainsim.galaxy_raw_coverage_v2_campaign_summary.v1"
_GALAXY_SUMMARY_SCHEMA_VERSION = 1
_RAW_COVERAGE_POLICY_SCHEMA_ID = "et_mainsim.raw_coverage_aware_policy.v1"
_RAW_COVERAGE_POLICY_SCHEMA_VERSION = 1
_PR_ASSET_WRITE_ENV = "ET_STAMP_WRITE_PR_ASSETS"
_PR_ASSET_PRESENTATION_DIR_ENV = "ET_STAMP_PRESENTATION_DIR"
_FORMAL_GALAXY_TARGET_COUNT = 10
_GALAXY_CAMPAIGN_QC_RELATIVE_PATH = Path(
    "quality_control/injected_campaign_delivery_qc.json"
)
_GALAXY_PROVENANCE_AUDIT_RELATIVE_PATH = Path(
    "quality_control/injected_campaign_provenance_psf_audit.json"
)
_GALAXY_SINGLE_SOURCE_ASSET_SCOPE = "single_source"
_GALAXY_CAMPAIGN_SUMMARY_ASSET_SCOPE = "campaign_summary"
_GALAXY_ASSET_SCOPE_MARKERS = {
    _GALAXY_SINGLE_SOURCE_ASSET_SCOPE: (
        "ET_STAMP_REPORT_GATE=galaxy_single_source_raw_coverage_v2_ready",
    ),
    _GALAXY_CAMPAIGN_SUMMARY_ASSET_SCOPE: (
        "ET_STAMP_REPORT_GATE=galaxy_campaign_raw_coverage_v2_ready",
    ),
}
_GALAXY_CAMPAIGN_SUMMARY_TABLE_FILENAMES = {
    "source_summary": "source_summary.csv",
    "source_window_metrics": "source_window_metrics.csv",
}


class NotebookReportAssetError(ValueError):
    """Raised when an executed notebook cannot safely publish report assets."""


@dataclass(frozen=True)
class NotebookPngAssetSpec:
    """One selected PNG output identified by its stable notebook cell ID."""

    cell_id: str
    filename: str


@dataclass(frozen=True)
class ExecutedNotebookReportAssetRequest:
    """Inputs for one immutable, externally published notebook asset bundle."""

    executed_notebook_path: Path | str
    report_root: Path | str
    production_manifest_path: Path | str
    asset_specs: Sequence[NotebookPngAssetSpec]
    required_markers: Sequence[str]
    publication_context: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class ExecutedNotebookReportAssetResult:
    """Locations of one atomically published asset bundle and its receipt."""

    published_asset_dir: Path
    receipt_path: Path


@dataclass(frozen=True)
class GalaxyNotebookReadinessRequest:
    """Immutable formal receipts required before a Galaxy PR figure may publish.

    A single-source figure needs the production, campaign-QC, and full
    provenance/PSF receipts.  The campaign-summary figure additionally passes
    both ``campaign_summary_manifest_path`` and ``coverage_policy_path`` so
    the final aggregate product is bound to the same formal run.
    """

    production_manifest_path: Path | str
    campaign_qc_path: Path | str
    provenance_audit_path: Path | str
    campaign_summary_manifest_path: Path | str | None = None
    coverage_policy_path: Path | str | None = None
    case: str = "injected"


@dataclass(frozen=True)
class GalaxyNotebookReadiness:
    """Validated final Galaxy identity available to a presentation notebook."""

    run_root: Path
    run_id: str
    source_ids: tuple[str, ...]
    production_manifest_path: Path
    production_manifest_identity: Mapping[str, Any]
    campaign_qc_path: Path
    campaign_qc_identity: Mapping[str, Any]
    provenance_audit_path: Path
    provenance_audit_identity: Mapping[str, Any]
    campaign_summary_manifest_path: Path | None = None
    campaign_summary_manifest_identity: Mapping[str, Any] | None = None
    coverage_policy_path: Path | None = None
    coverage_policy_identity: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class GalaxyPrAssetExportRequest:
    """One opt-in, receipt-bound PR asset publication request.

    The report root deliberately comes only from ``ET_STAMP_PRESENTATION_DIR``
    after ``ET_STAMP_WRITE_PR_ASSETS=1`` is supplied.  This keeps executed
    notebooks read-only by default and makes accidental writes into a formal
    run root impossible through this interface.
    """

    readiness: GalaxyNotebookReadinessRequest
    asset_scope: Literal["single_source", "campaign_summary"]
    executed_notebook_path: Path | str
    asset_specs: Sequence[NotebookPngAssetSpec]
    required_markers: Sequence[str]
    environment: Mapping[str, str] | None = None


def _path(value: Path | str, *, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise NotebookReportAssetError(f"{label} does not exist or is not a file: {path}")
    return path


def _report_root(value: Path | str) -> Path:
    return Path(value).expanduser().resolve()


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise NotebookReportAssetError(f"{label} must be a JSON object")
    return value


def _text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise NotebookReportAssetError(f"{label} must be a non-empty string")
    return value.strip()


def _read_json(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        return _mapping(json.loads(path.read_text(encoding="utf-8")), label=label)
    except (OSError, json.JSONDecodeError) as error:
        raise NotebookReportAssetError(f"cannot read {label}: {path}") from error


def _same_or_child(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _json_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return "".join(value)
    return ""


def _json_mapping_copy(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    try:
        serialized = json.dumps(dict(value), sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise NotebookReportAssetError(
            "publication context must be a JSON-serializable object"
        ) from error
    copied = json.loads(serialized)
    if not isinstance(copied, dict):  # Defensive: ``dict(value)`` above guarantees this.
        raise NotebookReportAssetError("publication context must be a JSON object")
    return copied


def _notebook_output_text(notebook: Mapping[str, Any]) -> str:
    cells = notebook.get("cells")
    if not isinstance(cells, list):
        raise NotebookReportAssetError("executed notebook cells must be a list")
    fragments: list[str] = []
    for cell in cells:
        if not isinstance(cell, Mapping):
            continue
        outputs = cell.get("outputs")
        if not isinstance(outputs, list):
            continue
        for output in outputs:
            if not isinstance(output, Mapping):
                continue
            fragments.append(_json_text(output.get("text")))
            data = output.get("data")
            if isinstance(data, Mapping):
                fragments.append(_json_text(data.get("text/plain")))
    return "\n".join(fragment for fragment in fragments if fragment)


def _validated_specs(
    specs: Sequence[NotebookPngAssetSpec],
) -> tuple[NotebookPngAssetSpec, ...]:
    if not specs:
        raise NotebookReportAssetError("at least one PNG asset specification is required")
    normalized: list[NotebookPngAssetSpec] = []
    cell_ids: set[str] = set()
    filenames: set[str] = set()
    for spec in specs:
        if not isinstance(spec, NotebookPngAssetSpec):
            raise NotebookReportAssetError("asset specifications must be NotebookPngAssetSpec")
        cell_id = _text(spec.cell_id, label="asset cell ID")
        filename = _text(spec.filename, label="asset filename")
        candidate = Path(filename)
        if (
            candidate.name != filename
            or candidate.suffix.lower() != ".png"
            or filename in {".", ".."}
        ):
            raise NotebookReportAssetError(
                "asset filename must be a simple .png filename without path components"
            )
        if cell_id in cell_ids:
            raise NotebookReportAssetError(f"duplicate notebook cell ID: {cell_id}")
        if filename in filenames:
            raise NotebookReportAssetError(f"duplicate asset filename: {filename}")
        cell_ids.add(cell_id)
        filenames.add(filename)
        normalized.append(NotebookPngAssetSpec(cell_id=cell_id, filename=filename))
    return tuple(normalized)


def _cell_png_bytes(notebook: Mapping[str, Any], *, cell_id: str) -> bytes:
    cells = notebook.get("cells")
    if not isinstance(cells, list):
        raise NotebookReportAssetError("executed notebook cells must be a list")
    matches = [cell for cell in cells if isinstance(cell, Mapping) and cell.get("id") == cell_id]
    if len(matches) != 1:
        raise NotebookReportAssetError(
            f"executed notebook must contain exactly one selected cell ID: {cell_id}"
        )
    outputs = matches[0].get("outputs")
    if not isinstance(outputs, list):
        raise NotebookReportAssetError(f"selected cell {cell_id} has no outputs list")
    encoded_pngs: list[str] = []
    for output in outputs:
        if not isinstance(output, Mapping):
            continue
        data = output.get("data")
        if not isinstance(data, Mapping) or "image/png" not in data:
            continue
        encoded = _json_text(data.get("image/png"))
        if not encoded:
            raise NotebookReportAssetError(f"selected cell {cell_id} has an empty image/png output")
        encoded_pngs.append("".join(encoded.split()))
    if len(encoded_pngs) != 1:
        raise NotebookReportAssetError(
            f"selected cell {cell_id} must contain exactly one image/png output"
        )
    try:
        payload = base64.b64decode(encoded_pngs[0], validate=True)
    except (ValueError, TypeError) as error:
        raise NotebookReportAssetError(
            f"selected cell {cell_id} image/png is not valid base64"
        ) from error
    if not payload.startswith(_PNG_SIGNATURE):
        raise NotebookReportAssetError(
            f"selected cell {cell_id} image/png does not have a PNG signature"
        )
    return payload


def _bytes_identity(payload: bytes, *, filename: str) -> dict[str, Any]:
    return {
        "path_relative_to_asset_dir": filename,
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _atomic_bytes(path: Path, payload: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    _atomic_bytes(path, encoded)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _manifest_contract(manifest_path: Path) -> tuple[Mapping[str, Any], str]:
    manifest = _read_json(manifest_path, label="production manifest")
    if manifest.get("schema_id") != _GALAXY_MANIFEST_SCHEMA_ID:
        raise NotebookReportAssetError(
            "production manifest must be a formal Galaxy stamp-production manifest"
        )
    run_id = _text(manifest.get("run_id"), label="production manifest run_id")
    if manifest.get("observation_product") != "final_dn":
        raise NotebookReportAssetError(
            "production manifest observation_product must be final_dn"
        )
    return manifest, run_id


def _is_formal_galaxy_manifest(manifest: Mapping[str, Any]) -> bool:
    """Return whether the manifest is the v2 formal Galaxy delivery contract."""

    return (
        manifest.get("schema_id") == _GALAXY_MANIFEST_SCHEMA_ID
        and manifest.get("schema_version") == _GALAXY_MANIFEST_SCHEMA_VERSION
        and manifest.get("observation_product") == "final_dn"
    )


def _integer(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise NotebookReportAssetError(f"{label} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise NotebookReportAssetError(f"{label} must be an integer") from error
    if result != value or result < minimum:
        raise NotebookReportAssetError(
            f"{label} must be an integer at least {minimum}"
        )
    return result


def _content_identity(value: Any, *, label: str) -> dict[str, Any]:
    mapping = _mapping(value, label=label)
    sha256 = mapping.get("sha256")
    if not isinstance(sha256, str) or len(sha256) != 64:
        raise NotebookReportAssetError(f"{label}.sha256 must be a SHA-256 hex string")
    try:
        int(sha256, 16)
    except ValueError as error:
        raise NotebookReportAssetError(
            f"{label}.sha256 must be a SHA-256 hex string"
        ) from error
    return {
        "sha256": sha256,
        "size_bytes": _integer(
            mapping.get("size_bytes"), label=f"{label}.size_bytes", minimum=1
        ),
    }


def _same_content_identity(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    return (
        actual.get("sha256") == expected.get("sha256")
        and actual.get("size_bytes") == expected.get("size_bytes")
    )


def _require_content_identity(
    value: Any,
    *,
    expected: Mapping[str, Any],
    label: str,
) -> None:
    actual = _content_identity(value, label=label)
    if not _same_content_identity(actual, expected):
        raise NotebookReportAssetError(f"{label} does not match the frozen input")


def _relative_file_within(
    root: Path,
    relative: Any,
    *,
    label: str,
) -> Path:
    text = _text(relative, label=label)
    candidate = Path(text)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise NotebookReportAssetError(f"{label} must be a safe relative path")
    path = (root / candidate).resolve()
    if not _same_or_child(path, root.resolve()):
        raise NotebookReportAssetError(f"{label} must remain within the formal run root")
    if not path.is_file():
        raise NotebookReportAssetError(f"{label} does not exist or is not a file: {path}")
    return path


def _canonical_formal_receipt_path(
    path: Path,
    *,
    run_root: Path,
    relative_path: Path,
    label: str,
) -> Path:
    """Require one known immutable receipt path beneath the formal run root."""

    expected = (run_root / relative_path).resolve()
    if not _same_or_child(expected, run_root):
        raise NotebookReportAssetError(
            f"canonical {label} path must remain within the formal run root"
        )
    if path != expected:
        raise NotebookReportAssetError(
            f"{label} must use canonical {label} path: {expected}"
        )
    return path


def _require_receipt_run_root(
    receipt: Mapping[str, Any], *, run_root: Path, label: str
) -> None:
    receipt_run_root = Path(
        _text(receipt.get("run_root"), label=f"{label} run_root")
    ).expanduser().resolve()
    if receipt_run_root != run_root:
        raise NotebookReportAssetError(
            f"{label} run_root conflicts with the production manifest"
        )


def _formal_source_ids(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    targets = manifest.get("targets")
    if not isinstance(targets, list) or len(targets) != _FORMAL_GALAXY_TARGET_COUNT:
        raise NotebookReportAssetError(
            f"formal Galaxy manifest must contain exactly {_FORMAL_GALAXY_TARGET_COUNT} targets"
        )
    source_ids: list[str] = []
    for position, target in enumerate(targets):
        target_mapping = _mapping(target, label=f"production target {position}")
        source_id = _integer(
            target_mapping.get("source_id_int64"),
            label=f"production target {position}.source_id_int64",
            minimum=0,
        )
        source_ids.append(str(source_id))
    if len(set(source_ids)) != len(source_ids):
        raise NotebookReportAssetError("formal Galaxy manifest target IDs must be unique")
    return tuple(source_ids)


def _finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise NotebookReportAssetError(f"{label} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise NotebookReportAssetError(f"{label} must be finite") from error
    if not math.isfinite(result):
        raise NotebookReportAssetError(f"{label} must be finite")
    return result


def _psf_id(value: Any, *, label: str) -> str | int:
    """Validate the scalar ID shape emitted by the formal PSF audit.

    The delivery audit reads ``chosen_psf_id`` from the focal-plane registry
    and serializes the registry's integer IDs unchanged.  A future registry
    may instead use a non-empty string key, but booleans, floats, and negative
    IDs are not meaningful PSF selections.
    """

    if isinstance(value, bool):
        raise NotebookReportAssetError(f"{label} must be a non-boolean PSF ID")
    if isinstance(value, (int, np.integer)):
        result = int(value)
        if result < 0 or result > 2**63 - 1:
            raise NotebookReportAssetError(
                f"{label} must be a non-negative signed PSF ID"
            )
        return result
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise NotebookReportAssetError(
        f"{label} must be a non-empty string or non-negative integer PSF ID"
    )


def _validate_campaign_qc(
    qc: Mapping[str, Any],
    *,
    run_root: Path,
    run_id: str,
    manifest_identity: Mapping[str, Any],
    time_plan_identity: Mapping[str, Any],
    case: str,
) -> tuple[int, int]:
    if qc.get("schema_id") != _GALAXY_CAMPAIGN_QC_SCHEMA_ID:
        raise NotebookReportAssetError("campaign QC has an unsupported schema")
    if _integer(qc.get("schema_version"), label="campaign QC schema_version", minimum=1) != (
        _GALAXY_CAMPAIGN_QC_SCHEMA_VERSION
    ):
        raise NotebookReportAssetError("campaign QC has an unsupported schema version")
    if qc.get("ready") is not True:
        raise NotebookReportAssetError("campaign QC is not ready")
    _require_receipt_run_root(qc, run_root=run_root, label="campaign QC")
    if _text(qc.get("run_id"), label="campaign QC run_id") != run_id:
        raise NotebookReportAssetError("campaign QC run_id conflicts with production manifest")
    if qc.get("case") != case:
        raise NotebookReportAssetError("campaign QC case conflicts with requested case")
    _require_content_identity(
        qc.get("manifest_identity"),
        expected=manifest_identity,
        label="campaign QC manifest identity",
    )
    _require_content_identity(
        qc.get("time_plan_identity"),
        expected=time_plan_identity,
        label="campaign QC time-plan identity",
    )
    expected_bundle_count = _integer(
        qc.get("expected_bundle_count"),
        label="campaign QC expected_bundle_count",
        minimum=1,
    )
    valid_bundle_count = _integer(
        qc.get("valid_bundle_count"),
        label="campaign QC valid_bundle_count",
        minimum=0,
    )
    if valid_bundle_count != expected_bundle_count:
        raise NotebookReportAssetError(
            "campaign QC valid bundle count does not meet the expected count"
        )
    return expected_bundle_count, valid_bundle_count


def _validate_provenance_audit(
    audit: Mapping[str, Any],
    *,
    run_root: Path,
    run_id: str,
    source_ids: tuple[str, ...],
    manifest_identity: Mapping[str, Any],
    case: str,
) -> tuple[int, int]:
    if audit.get("schema_id") != _GALAXY_PROVENANCE_AUDIT_SCHEMA_ID:
        raise NotebookReportAssetError("provenance audit has an unsupported schema")
    if _integer(
        audit.get("schema_version"), label="provenance audit schema_version", minimum=1
    ) != _GALAXY_PROVENANCE_AUDIT_SCHEMA_VERSION:
        raise NotebookReportAssetError("provenance audit has an unsupported schema version")
    if audit.get("ready") is not True:
        raise NotebookReportAssetError("provenance audit is not ready")
    _require_receipt_run_root(
        audit, run_root=run_root, label="provenance audit"
    )
    if _text(audit.get("run_id"), label="provenance audit run_id") != run_id:
        raise NotebookReportAssetError(
            "provenance audit run_id conflicts with production manifest"
        )
    if audit.get("case") != case:
        raise NotebookReportAssetError("provenance audit case conflicts with requested case")
    _require_content_identity(
        audit.get("production_manifest_identity"),
        expected=manifest_identity,
        label="provenance audit production-manifest identity",
    )
    expected_bundle_count = _integer(
        audit.get("expected_bundle_count"),
        label="provenance audit expected_bundle_count",
        minimum=1,
    )
    valid_bundle_count = _integer(
        audit.get("valid_bundle_count"),
        label="provenance audit valid_bundle_count",
        minimum=0,
    )
    if valid_bundle_count != expected_bundle_count:
        raise NotebookReportAssetError(
            "provenance audit valid bundle count does not meet the expected count"
        )
    for name in ("missing_bundle_count", "invalid_bundle_count"):
        if _integer(audit.get(name), label=f"provenance audit {name}", minimum=0) != 0:
            raise NotebookReportAssetError(f"provenance audit {name} must be zero")
    source_summaries = _mapping(
        audit.get("source_summaries"), label="provenance audit source_summaries"
    )
    if set(source_summaries) != set(source_ids):
        raise NotebookReportAssetError(
            "provenance audit sources do not exactly match the production manifest"
        )
    source_expected_total = 0
    source_valid_total = 0
    for source_id in source_ids:
        summary = _mapping(
            source_summaries.get(source_id), label=f"provenance audit source {source_id}"
        )
        if _text(summary.get("source_id"), label=f"provenance audit source {source_id} ID") != source_id:
            raise NotebookReportAssetError("provenance audit source ID conflicts with its key")
        source_expected = _integer(
            summary.get("expected_bundle_count"),
            label=f"provenance audit source {source_id} expected_bundle_count",
            minimum=1,
        )
        source_valid = _integer(
            summary.get("valid_bundle_count"),
            label=f"provenance audit source {source_id} valid_bundle_count",
            minimum=0,
        )
        if source_valid != source_expected:
            raise NotebookReportAssetError(
                f"provenance audit source {source_id} is not fully valid"
            )
        for name in ("missing_bundle_count", "invalid_bundle_count"):
            if _integer(
                summary.get(name),
                label=f"provenance audit source {source_id} {name}",
                minimum=0,
            ) != 0:
                raise NotebookReportAssetError(
                    f"provenance audit source {source_id} {name} must be zero"
                )
        _psf_id(
            summary.get("chosen_psf_id"),
            label=f"provenance audit source {source_id} PSF",
        )
        _finite(
            summary.get("node_angle_deg"),
            label=f"provenance audit source {source_id} node angle",
        )
        if summary.get("runtime_registry_attestation_verified") is not True:
            raise NotebookReportAssetError(
                f"provenance audit source {source_id} registry attestation is not verified"
            )
        for name in (
            "runtime_registry_semantic_content_sha256",
            "runtime_registry_attestation_record_sha256",
        ):
            _content_identity(
                {"sha256": summary.get(name), "size_bytes": 1},
                label=f"provenance audit source {source_id} {name}",
            )
        source_expected_total += source_expected
        source_valid_total += source_valid
    if (
        source_expected_total != expected_bundle_count
        or source_valid_total != valid_bundle_count
    ):
        raise NotebookReportAssetError(
            "provenance audit source bundle counts conflict with campaign totals"
        )
    return expected_bundle_count, valid_bundle_count


def _validate_campaign_summary_tables(
    summary: Mapping[str, Any],
    *,
    summary_path: Path,
) -> None:
    """Bind both published campaign CSVs to the immutable summary manifest."""

    tables = _mapping(summary.get("tables"), label="campaign summary tables")
    for table_key, filename in _GALAXY_CAMPAIGN_SUMMARY_TABLE_FILENAMES.items():
        label = table_key.replace("_", " ")
        table = _mapping(
            tables.get(table_key), label=f"campaign summary {label} table"
        )
        path_text = _text(
            table.get("path"), label=f"campaign summary {label} table path"
        )
        if path_text != filename:
            raise NotebookReportAssetError(
                f"campaign summary {label} table path must be {filename}"
            )
        if table.get("format") != "csv":
            raise NotebookReportAssetError(
                f"campaign summary {label} table format must be csv"
            )
        table_path = _relative_file_within(
            summary_path.parent,
            path_text,
            label=f"campaign summary {label} table path",
        )
        _require_content_identity(
            table.get("identity"),
            expected=_content_identity(
                file_identity(table_path),
                label=f"campaign summary {label} table file",
            ),
            label=f"campaign summary {label} table identity",
        )


def _validate_campaign_summary(
    summary: Mapping[str, Any],
    *,
    summary_path: Path,
    policy: Mapping[str, Any],
    policy_path: Path,
    run_root: Path,
    run_id: str,
    source_ids: tuple[str, ...],
    manifest_identity: Mapping[str, Any],
    campaign_qc_identity: Mapping[str, Any],
    case: str,
) -> None:
    if summary.get("schema_id") != _GALAXY_SUMMARY_SCHEMA_ID:
        raise NotebookReportAssetError("campaign summary has an unsupported schema")
    if _integer(
        summary.get("schema_version"), label="campaign summary schema_version", minimum=1
    ) != _GALAXY_SUMMARY_SCHEMA_VERSION:
        raise NotebookReportAssetError("campaign summary has an unsupported schema version")
    if summary.get("complete") is not True or summary.get("ready") is not True:
        raise NotebookReportAssetError("campaign summary is not complete and ready")
    if _text(summary.get("run_id"), label="campaign summary run_id") != run_id:
        raise NotebookReportAssetError("campaign summary run_id conflicts with production manifest")
    if summary.get("case") != case:
        raise NotebookReportAssetError("campaign summary case conflicts with requested case")
    if summary.get("observation_product") != "final_dn":
        raise NotebookReportAssetError("campaign summary observation_product must be final_dn")
    if summary.get("background_realization_used") is not False:
        raise NotebookReportAssetError(
            "campaign summary background_realization_used must be false"
        )
    if _integer(summary.get("source_count"), label="campaign summary source_count", minimum=1) != len(source_ids):
        raise NotebookReportAssetError("campaign summary source_count conflicts with production manifest")
    production = _mapping(
        summary.get("production_manifest"), label="campaign summary production_manifest"
    )
    _require_content_identity(
        production.get("identity"),
        expected=manifest_identity,
        label="campaign summary production-manifest identity",
    )
    campaign_qc = _mapping(summary.get("campaign_qc"), label="campaign summary campaign_qc")
    _require_content_identity(
        campaign_qc.get("identity"),
        expected=campaign_qc_identity,
        label="campaign summary campaign-QC identity",
    )
    if policy.get("schema_id") != _RAW_COVERAGE_POLICY_SCHEMA_ID:
        raise NotebookReportAssetError("frozen coverage policy has an unsupported schema")
    if _integer(
        policy.get("schema_version"), label="frozen coverage policy schema_version", minimum=1
    ) != _RAW_COVERAGE_POLICY_SCHEMA_VERSION:
        raise NotebookReportAssetError("frozen coverage policy has an unsupported schema version")
    if policy.get("complete") is not True or policy.get("case") != case:
        raise NotebookReportAssetError("frozen coverage policy is not complete for the requested case")
    if _text(policy.get("run_id"), label="frozen coverage policy run_id") != run_id:
        raise NotebookReportAssetError("frozen coverage policy run_id conflicts with production manifest")
    if policy.get("observation_product") != "final_dn" or policy.get("background_realization_used") is not False:
        raise NotebookReportAssetError("frozen coverage policy does not preserve final_dn semantics")
    _require_content_identity(
        policy.get("production_manifest_identity"),
        expected=manifest_identity,
        label="frozen coverage policy production-manifest identity",
    )
    policy_identity = _content_identity(file_identity(policy_path), label="frozen coverage policy file")
    frozen_policy = _mapping(
        summary.get("frozen_coverage_policy"), label="campaign summary frozen_coverage_policy"
    )
    _require_content_identity(
        frozen_policy.get("identity"),
        expected=policy_identity,
        label="campaign summary frozen coverage-policy identity",
    )
    if frozen_policy.get("record") != policy.get("coverage"):
        raise NotebookReportAssetError(
            "campaign summary frozen coverage-policy record conflicts with policy"
        )
    artifacts = summary.get("source_artifacts")
    if not isinstance(artifacts, list) or [
        _text(_mapping(item, label="campaign summary source artifact").get("source_id"), label="campaign summary source artifact ID")
        for item in artifacts
    ] != list(source_ids):
        raise NotebookReportAssetError(
            "campaign summary source order does not match the production manifest"
        )
    for label, root, path in (
        ("campaign summary", run_root, summary_path),
        ("frozen coverage policy", run_root, policy_path),
    ):
        if not _same_or_child(path.resolve(), root.resolve()):
            raise NotebookReportAssetError(f"{label} must remain within the formal run root")
    _validate_campaign_summary_tables(summary, summary_path=summary_path)


def validate_galaxy_notebook_readiness_v1(
    request: GalaxyNotebookReadinessRequest,
) -> GalaxyNotebookReadiness:
    """Validate formal Galaxy receipts before an executed notebook may publish.

    The function is read-only.  It binds production manifest, complete campaign
    QC, and all-HDF5 provenance/PSF audit by their content identities.  An
    optional campaign summary is independently bound to its frozen coverage
    policy.  It accepts only the formal ten-source injected ``final_dn``
    delivery, never a partial, raw-electron, or legacy-PCA/SG substitute.
    """

    if not isinstance(request, GalaxyNotebookReadinessRequest):
        raise NotebookReportAssetError("request must be GalaxyNotebookReadinessRequest")
    if request.case != "injected":
        raise NotebookReportAssetError("Galaxy PR assets only support injected formal delivery")
    if (request.campaign_summary_manifest_path is None) != (
        request.coverage_policy_path is None
    ):
        raise NotebookReportAssetError(
            "campaign summary and frozen coverage policy must be supplied together"
        )
    manifest_path = _path(request.production_manifest_path, label="production manifest")
    campaign_qc_path = _path(request.campaign_qc_path, label="campaign QC")
    provenance_audit_path = _path(
        request.provenance_audit_path, label="provenance audit"
    )
    manifest = _read_json(manifest_path, label="production manifest")
    if manifest.get("schema_id") != _GALAXY_MANIFEST_SCHEMA_ID:
        raise NotebookReportAssetError("production manifest has an unsupported schema")
    if _integer(
        manifest.get("schema_version"), label="production manifest schema_version", minimum=1
    ) != _GALAXY_MANIFEST_SCHEMA_VERSION:
        raise NotebookReportAssetError("production manifest has an unsupported schema version")
    if manifest.get("observation_product") != "final_dn":
        raise NotebookReportAssetError("production manifest observation_product must be final_dn")
    if manifest.get("background_realization_delivered") is not False:
        raise NotebookReportAssetError(
            "production manifest background_realization_delivered must be false"
        )
    run_root = manifest_path.parent.resolve()
    run_id = _text(manifest.get("run_id"), label="production manifest run_id")
    if run_root.name != run_id:
        raise NotebookReportAssetError("production manifest run_id must equal its run-root name")
    _canonical_formal_receipt_path(
        campaign_qc_path,
        run_root=run_root,
        relative_path=_GALAXY_CAMPAIGN_QC_RELATIVE_PATH,
        label="campaign QC",
    )
    _canonical_formal_receipt_path(
        provenance_audit_path,
        run_root=run_root,
        relative_path=_GALAXY_PROVENANCE_AUDIT_RELATIVE_PATH,
        label="provenance audit",
    )
    source_ids = _formal_source_ids(manifest)
    manifest_identity = _content_identity(
        file_identity(manifest_path), label="production manifest file"
    )
    delivery = _mapping(manifest.get("delivery"), label="production manifest delivery")
    time_plan_path = _relative_file_within(
        run_root,
        delivery.get("time_plan_relative_path"),
        label="production manifest time-plan path",
    )
    time_plan_identity = _content_identity(
        file_identity(time_plan_path), label="production time-plan file"
    )
    campaign_qc = _read_json(campaign_qc_path, label="campaign QC")
    campaign_qc_bundle_counts = _validate_campaign_qc(
        campaign_qc,
        run_root=run_root,
        run_id=run_id,
        manifest_identity=manifest_identity,
        time_plan_identity=time_plan_identity,
        case=request.case,
    )
    provenance_audit = _read_json(provenance_audit_path, label="provenance audit")
    provenance_audit_bundle_counts = _validate_provenance_audit(
        provenance_audit,
        run_root=run_root,
        run_id=run_id,
        source_ids=source_ids,
        manifest_identity=manifest_identity,
        case=request.case,
    )
    if provenance_audit_bundle_counts != campaign_qc_bundle_counts:
        raise NotebookReportAssetError(
            "provenance audit bundle counts conflict with campaign QC"
        )
    campaign_summary_path: Path | None = None
    campaign_summary_identity: Mapping[str, Any] | None = None
    coverage_policy_path: Path | None = None
    coverage_policy_identity: Mapping[str, Any] | None = None
    if request.campaign_summary_manifest_path is not None:
        campaign_summary_path = _path(
            request.campaign_summary_manifest_path, label="campaign summary manifest"
        )
        coverage_policy_path = _path(
            request.coverage_policy_path, label="frozen coverage policy"
        )
        campaign_summary = _read_json(
            campaign_summary_path, label="campaign summary manifest"
        )
        coverage_policy = _read_json(coverage_policy_path, label="frozen coverage policy")
        _validate_campaign_summary(
            campaign_summary,
            summary_path=campaign_summary_path,
            policy=coverage_policy,
            policy_path=coverage_policy_path,
            run_root=run_root,
            run_id=run_id,
            source_ids=source_ids,
            manifest_identity=manifest_identity,
            campaign_qc_identity=_content_identity(
                file_identity(campaign_qc_path), label="campaign QC file"
            ),
            case=request.case,
        )
        campaign_summary_identity = _content_identity(
            file_identity(campaign_summary_path), label="campaign summary file"
        )
        coverage_policy_identity = _content_identity(
            file_identity(coverage_policy_path), label="frozen coverage policy file"
        )
    return GalaxyNotebookReadiness(
        run_root=run_root,
        run_id=run_id,
        source_ids=source_ids,
        production_manifest_path=manifest_path,
        production_manifest_identity=manifest_identity,
        campaign_qc_path=campaign_qc_path,
        campaign_qc_identity=_content_identity(
            file_identity(campaign_qc_path), label="campaign QC file"
        ),
        provenance_audit_path=provenance_audit_path,
        provenance_audit_identity=_content_identity(
            file_identity(provenance_audit_path), label="provenance audit file"
        ),
        campaign_summary_manifest_path=campaign_summary_path,
        campaign_summary_manifest_identity=campaign_summary_identity,
        coverage_policy_path=coverage_policy_path,
        coverage_policy_identity=coverage_policy_identity,
    )


def nullable_cdpp_values_to_nan_v1(values: Any) -> np.ndarray:
    """Convert nullable CSV/table CDPP values into a one-dimensional float array.

    Formal coverage summary CSVs legitimately represent an insufficient number
    of accepted bins as blank/null CDPP.  The notebook must render those rows
    as missing rather than throw while coercing a whole column to ``float`` or
    fabricate a value through interpolation.
    """

    array = np.ma.asarray(values)
    if array.ndim != 1:
        raise NotebookReportAssetError("nullable CDPP values must be one-dimensional")
    masked = np.ma.getmaskarray(array)
    raw = np.ma.getdata(array)
    result = np.full(array.shape, np.nan, dtype=float)
    for index, value in enumerate(raw):
        if bool(masked[index]) or value is None:
            continue
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if isinstance(value, str):
            value = value.strip()
            if not value:
                continue
        if isinstance(value, bool):
            raise NotebookReportAssetError(f"CDPP value at index {index} must be numeric or blank")
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise NotebookReportAssetError(
                f"CDPP value at index {index} must be numeric or blank"
            ) from error
        if math.isnan(numeric):
            continue
        if not math.isfinite(numeric):
            raise NotebookReportAssetError(
                f"CDPP value at index {index} must be finite or blank"
            )
        result[index] = numeric
    return result


def _pr_asset_export_requested(environment: Mapping[str, str]) -> bool:
    value = environment.get(_PR_ASSET_WRITE_ENV)
    if value is None:
        return False
    if not isinstance(value, str):
        raise NotebookReportAssetError(f"{_PR_ASSET_WRITE_ENV} must be a string")
    normalized = value.strip()
    if normalized in {"", "0"}:
        return False
    if normalized != "1":
        raise NotebookReportAssetError(
            f"{_PR_ASSET_WRITE_ENV} must be exactly '1' to publish PR assets"
        )
    return True


def resolve_galaxy_pr_asset_export_root_v1(
    readiness: GalaxyNotebookReadiness | None,
    *,
    run_root: Path | str | None = None,
    environment: Mapping[str, str] | None = None,
) -> Path | None:
    """Resolve the sole opt-in PR-asset destination without creating it.

    No environment setting means no output.  Once ``ET_STAMP_WRITE_PR_ASSETS``
    is explicitly enabled, missing readiness is an error rather than a silent
    no-op, so ``nbconvert`` cannot appear successful while publishing no valid
    science figure.
    """

    env = os.environ if environment is None else environment
    if not isinstance(env, Mapping):
        raise NotebookReportAssetError("asset-export environment must be a mapping")
    if not _pr_asset_export_requested(env):
        return None
    if readiness is None:
        raise NotebookReportAssetError(
            "PR asset export was requested before Galaxy readiness was validated"
        )
    if not isinstance(readiness, GalaxyNotebookReadiness):
        raise NotebookReportAssetError("readiness must be GalaxyNotebookReadiness")
    if run_root is not None and Path(run_root).expanduser().resolve() != readiness.run_root:
        raise NotebookReportAssetError(
            "requested run_root conflicts with the validated Galaxy readiness"
        )
    destination_value = env.get(_PR_ASSET_PRESENTATION_DIR_ENV)
    if not isinstance(destination_value, str) or not destination_value.strip():
        raise NotebookReportAssetError(
            f"{_PR_ASSET_PRESENTATION_DIR_ENV} must be set when PR asset export is enabled"
        )
    destination = Path(destination_value).expanduser().resolve()
    if _same_or_child(destination, readiness.run_root):
        raise NotebookReportAssetError(
            "PR asset presentation directory must be outside the production root"
        )
    return destination


def _validated_galaxy_asset_scope(
    request: GalaxyPrAssetExportRequest,
) -> Literal["single_source", "campaign_summary"]:
    """Require the figure type, its readiness inputs, and marker to agree."""

    scope = request.asset_scope
    if scope not in _GALAXY_ASSET_SCOPE_MARKERS:
        raise NotebookReportAssetError(
            "Galaxy asset scope must be single_source or campaign_summary"
        )
    has_campaign_summary = request.readiness.campaign_summary_manifest_path is not None
    has_coverage_policy = request.readiness.coverage_policy_path is not None
    if scope == _GALAXY_SINGLE_SOURCE_ASSET_SCOPE:
        if has_campaign_summary or has_coverage_policy:
            raise NotebookReportAssetError(
                "single_source asset scope does not accept campaign summary receipts"
            )
    elif not (has_campaign_summary and has_coverage_policy):
        raise NotebookReportAssetError(
            "campaign_summary asset scope requires both campaign summary and "
            "frozen coverage policy"
        )
    markers = tuple(
        _text(marker, label="required readiness marker")
        for marker in request.required_markers
    )
    expected_markers = _GALAXY_ASSET_SCOPE_MARKERS[scope]
    if markers != expected_markers:
        raise NotebookReportAssetError(
            f"{scope} asset scope requires exactly its readiness marker: "
            f"{expected_markers[0]}"
        )
    return scope


def export_galaxy_pr_assets_v1(
    request: GalaxyPrAssetExportRequest,
) -> ExecutedNotebookReportAssetResult | None:
    """Publish a formal Galaxy notebook's selected PNGs only after readiness.

    This is the presentation-facing wrapper around the generic executed-
    notebook exporter.  The generic helper remains suitable for other report
    types; this wrapper is the only supported path for final Galaxy PR assets.
    """

    if not isinstance(request, GalaxyPrAssetExportRequest):
        raise NotebookReportAssetError("request must be GalaxyPrAssetExportRequest")
    asset_scope = _validated_galaxy_asset_scope(request)
    environment = os.environ if request.environment is None else request.environment
    if not isinstance(environment, Mapping):
        raise NotebookReportAssetError("asset-export environment must be a mapping")
    if not _pr_asset_export_requested(environment):
        return None
    readiness = validate_galaxy_notebook_readiness_v1(request.readiness)
    report_root = resolve_galaxy_pr_asset_export_root_v1(
        readiness,
        environment=environment,
    )
    if report_root is None:  # Defensive: the explicit flag was just checked above.
        raise NotebookReportAssetError("PR asset export unexpectedly resolved to disabled")
    readiness_context: dict[str, Any] = {
        "schema_id": "et_mainsim.galaxy_pr_asset_readiness.v1",
        "run_id": readiness.run_id,
        "case": "injected",
        "asset_scope": asset_scope,
        "observation_product": "final_dn",
        "production_manifest": dict(readiness.production_manifest_identity),
        "campaign_qc": {
            "identity": dict(readiness.campaign_qc_identity),
        },
        "provenance_audit": {
            "identity": dict(readiness.provenance_audit_identity),
        },
        "analysis_boundary": (
            "legacy-MAD-compatible only; no legacy pickle, PCA, or "
            "Savitzky-Golay workflow"
        ),
    }
    if readiness.campaign_summary_manifest_identity is not None:
        readiness_context["campaign_summary"] = {
            "identity": dict(readiness.campaign_summary_manifest_identity),
        }
    if readiness.coverage_policy_identity is not None:
        readiness_context["frozen_coverage_policy"] = {
            "identity": dict(readiness.coverage_policy_identity),
        }
    return _export_executed_notebook_png_assets_v1(
        ExecutedNotebookReportAssetRequest(
            executed_notebook_path=request.executed_notebook_path,
            report_root=report_root,
            production_manifest_path=readiness.production_manifest_path,
            asset_specs=request.asset_specs,
            required_markers=request.required_markers,
            publication_context={"galaxy_readiness": readiness_context},
        ),
        allow_formal_galaxy=True,
    )


def export_executed_notebook_png_assets_v1(
    request: ExecutedNotebookReportAssetRequest,
) -> ExecutedNotebookReportAssetResult:
    """Publish a non-Galaxy executed-notebook report asset bundle.

    The generic exporter intentionally remains available to existing report
    workflows.  A formal Galaxy v2 delivery is different: its final figures
    must be bound to the campaign QC and provenance receipts, so callers must
    use :func:`export_galaxy_pr_assets_v1` instead.
    """

    return _export_executed_notebook_png_assets_v1(
        request,
        allow_formal_galaxy=False,
    )


def _export_executed_notebook_png_assets_v1(
    request: ExecutedNotebookReportAssetRequest,
    *,
    allow_formal_galaxy: bool,
) -> ExecutedNotebookReportAssetResult:
    """Atomically publish selected PNG output cells and their receipts.

    The destination is always ``<report_root>/assets`` and must not already
    exist.  This prevents an older report bundle from being silently mixed
    with the new one.  ``report_root`` is rejected if it lies under the
    production-manifest directory, so this helper cannot alter science data.
    """

    if not isinstance(request, ExecutedNotebookReportAssetRequest):
        raise NotebookReportAssetError("request must be ExecutedNotebookReportAssetRequest")
    notebook_path = _path(request.executed_notebook_path, label="executed notebook")
    manifest_path = _path(request.production_manifest_path, label="production manifest")
    report_root = _report_root(request.report_root)
    production_root = manifest_path.parent.resolve()
    if _same_or_child(report_root, production_root):
        raise NotebookReportAssetError(
            "report root must be outside the production root"
        )
    specs = _validated_specs(request.asset_specs)
    markers = tuple(
        _text(marker, label="required readiness marker")
        for marker in request.required_markers
    )
    if not markers:
        raise NotebookReportAssetError("at least one required readiness marker is required")
    manifest, run_id = _manifest_contract(manifest_path)
    if _is_formal_galaxy_manifest(manifest) and not allow_formal_galaxy:
        raise NotebookReportAssetError(
            "formal Galaxy report assets require export_galaxy_pr_assets_v1"
        )
    publication_context = _json_mapping_copy(request.publication_context)
    notebook = _read_json(notebook_path, label="executed notebook")
    output_text = _notebook_output_text(notebook)
    for marker in markers:
        if marker not in output_text:
            raise NotebookReportAssetError(
                f"executed notebook is missing required readiness marker: {marker}"
            )
    payloads = [
        (spec, _cell_png_bytes(notebook, cell_id=spec.cell_id)) for spec in specs
    ]

    report_root.mkdir(parents=True, exist_ok=True)
    published_asset_dir = report_root / "assets"
    if published_asset_dir.exists():
        raise NotebookReportAssetError(
            f"published asset directory already exists: {published_asset_dir}"
        )
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=".assets-", suffix=".partial", dir=report_root
        )
    )
    try:
        asset_records: list[dict[str, Any]] = []
        for spec, payload in payloads:
            _atomic_bytes(staging_dir / spec.filename, payload)
            identity = _bytes_identity(payload, filename=spec.filename)
            record = {
                "cell_id": spec.cell_id,
                "filename": spec.filename,
                "image": identity,
            }
            asset_records.append(record)
            _atomic_json(
                staging_dir / f"{spec.filename}.receipt.json",
                {
                    "schema_id": NOTEBOOK_REPORT_ASSETS_SCHEMA_ID,
                    "schema_version": NOTEBOOK_REPORT_ASSETS_SCHEMA_VERSION,
                    "complete": True,
                    "run_id": run_id,
                    "observation_product": "final_dn",
                    "production_manifest": file_identity(manifest_path),
                    "executed_notebook": file_identity(notebook_path),
                    "required_readiness_markers": list(markers),
                    "asset": record,
                    **(
                        {}
                        if publication_context is None
                        else {"publication_context": publication_context}
                    ),
                },
            )
        receipt = {
            "schema_id": NOTEBOOK_REPORT_ASSETS_SCHEMA_ID,
            "schema_version": NOTEBOOK_REPORT_ASSETS_SCHEMA_VERSION,
            "complete": True,
            "run_id": run_id,
            "observation_product": "final_dn",
            "production_manifest": file_identity(manifest_path),
            "executed_notebook": file_identity(notebook_path),
            "required_readiness_markers": list(markers),
            "asset_count": len(asset_records),
            "assets": asset_records,
            "producer": {
                "module": "et_mainsim.notebook_report_assets",
                "operation": "export_executed_notebook_png_assets_v1",
            },
            **(
                {}
                if publication_context is None
                else {"publication_context": publication_context}
            ),
        }
        _atomic_json(staging_dir / "report_assets_receipt.json", receipt)
        _fsync_directory(staging_dir)
        os.replace(staging_dir, published_asset_dir)
        _fsync_directory(report_root)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    return ExecutedNotebookReportAssetResult(
        published_asset_dir=published_asset_dir,
        receipt_path=published_asset_dir / "report_assets_receipt.json",
    )


def _asset_spec_from_text(value: str) -> NotebookPngAssetSpec:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--asset must use CELL_ID=FILENAME.png")
    cell_id, filename = value.split("=", 1)
    try:
        return _validated_specs((NotebookPngAssetSpec(cell_id, filename),))[0]
    except NotebookReportAssetError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Atomically export selected PNG outputs from an executed notebook."
    )
    parser.add_argument("--executed-notebook", required=True, type=Path)
    parser.add_argument(
        "--report-root",
        type=Path,
        help="generic-export destination; required unless Galaxy receipt inputs are supplied",
    )
    parser.add_argument("--production-manifest", required=True, type=Path)
    parser.add_argument(
        "--asset",
        required=True,
        action="append",
        type=_asset_spec_from_text,
        metavar="CELL_ID=FILENAME.png",
    )
    parser.add_argument(
        "--required-marker",
        required=True,
        action="append",
        metavar="TEXT",
    )
    parser.add_argument(
        "--galaxy-campaign-qc",
        "--campaign-qc",
        dest="galaxy_campaign_qc",
        type=Path,
        help="formal Galaxy injected campaign-QC receipt",
    )
    parser.add_argument(
        "--galaxy-provenance-audit",
        "--provenance-audit",
        dest="galaxy_provenance_audit",
        type=Path,
        help="formal Galaxy injected provenance/PSF audit receipt",
    )
    parser.add_argument(
        "--galaxy-campaign-summary",
        "--campaign-summary",
        dest="galaxy_campaign_summary",
        type=Path,
        help="optional formal Galaxy campaign-summary manifest",
    )
    parser.add_argument(
        "--galaxy-coverage-policy",
        "--coverage-policy",
        dest="galaxy_coverage_policy",
        type=Path,
        help="frozen coverage policy paired with --galaxy-campaign-summary",
    )
    parser.add_argument(
        "--galaxy-asset-scope",
        choices=(
            _GALAXY_SINGLE_SOURCE_ASSET_SCOPE,
            _GALAXY_CAMPAIGN_SUMMARY_ASSET_SCOPE,
        ),
        help=(
            "formal Galaxy figure type: single_source requires only source receipts; "
            "campaign_summary also requires the summary and frozen policy"
        ),
    )
    return parser


def _galaxy_cli_export_requested(args: argparse.Namespace) -> bool:
    return any(
        getattr(args, field) is not None
        for field in (
            "galaxy_campaign_qc",
            "galaxy_provenance_audit",
            "galaxy_campaign_summary",
            "galaxy_coverage_policy",
            "galaxy_asset_scope",
        )
    )


def _galaxy_pr_asset_export_request_from_args(
    args: argparse.Namespace,
) -> GalaxyPrAssetExportRequest:
    if args.report_root is not None:
        raise NotebookReportAssetError(
            "--report-root is not accepted for formal Galaxy export; set "
            f"{_PR_ASSET_PRESENTATION_DIR_ENV} with {_PR_ASSET_WRITE_ENV}=1"
        )
    if args.galaxy_campaign_qc is None or args.galaxy_provenance_audit is None:
        raise NotebookReportAssetError(
            "formal Galaxy export requires both --galaxy-campaign-qc and "
            "--galaxy-provenance-audit"
        )
    if args.galaxy_asset_scope is None:
        raise NotebookReportAssetError(
            "formal Galaxy export requires --galaxy-asset-scope"
        )
    return GalaxyPrAssetExportRequest(
        readiness=GalaxyNotebookReadinessRequest(
            production_manifest_path=args.production_manifest,
            campaign_qc_path=args.galaxy_campaign_qc,
            provenance_audit_path=args.galaxy_provenance_audit,
            campaign_summary_manifest_path=args.galaxy_campaign_summary,
            coverage_policy_path=args.galaxy_coverage_policy,
        ),
        asset_scope=args.galaxy_asset_scope,
        executed_notebook_path=args.executed_notebook,
        asset_specs=tuple(args.asset),
        required_markers=tuple(args.required_marker),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if _galaxy_cli_export_requested(args):
            result = export_galaxy_pr_assets_v1(
                _galaxy_pr_asset_export_request_from_args(args)
            )
            if result is None:
                raise NotebookReportAssetError(
                    "formal Galaxy asset export is disabled; set "
                    f"{_PR_ASSET_WRITE_ENV}=1 and "
                    f"{_PR_ASSET_PRESENTATION_DIR_ENV}"
                )
        else:
            if args.report_root is None:
                raise NotebookReportAssetError(
                    "--report-root is required for generic report asset export"
                )
            result = export_executed_notebook_png_assets_v1(
                ExecutedNotebookReportAssetRequest(
                    executed_notebook_path=args.executed_notebook,
                    report_root=args.report_root,
                    production_manifest_path=args.production_manifest,
                    asset_specs=tuple(args.asset),
                    required_markers=tuple(args.required_marker),
                )
            )
    except (NotebookReportAssetError, OSError) as error:
        print(f"report asset export failed: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "published_asset_dir": str(result.published_asset_dir),
                "receipt_path": str(result.receipt_path),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
