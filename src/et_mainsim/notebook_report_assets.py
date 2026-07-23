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
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

from .stamp_inputs import file_identity


NOTEBOOK_REPORT_ASSETS_SCHEMA_ID = "et_mainsim.executed_notebook_report_assets.v1"
NOTEBOOK_REPORT_ASSETS_SCHEMA_VERSION = 1
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_GALAXY_MANIFEST_SCHEMA_ID = "et_mainsim.galaxy_stamp_production.v1"


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


@dataclass(frozen=True)
class ExecutedNotebookReportAssetResult:
    """Locations of one atomically published asset bundle and its receipt."""

    published_asset_dir: Path
    receipt_path: Path


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


def export_executed_notebook_png_assets_v1(
    request: ExecutedNotebookReportAssetRequest,
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
    _manifest, run_id = _manifest_contract(manifest_path)
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
    parser.add_argument("--report-root", required=True, type=Path)
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
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
