from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from astropy.table import Table


_COLUMN_TOKEN = re.compile(r"[^a-z0-9]+")
_ALIASES = {
    "source_id": {"source_id", "gaia_source_id", "id"},
    "gaia_g_mag": {
        "gaia_g_mag",
        "gaia_g",
        "gaia_g_vega",
        "g_mean_mag",
        "gmag",
    },
    "psf_id": {"psf_id", "psf_field_id", "field_id"},
    "detector_xpix": {"detector_xpix", "detector_x", "xpix"},
    "detector_ypix": {"detector_ypix", "detector_y", "ypix"},
}


@dataclass(frozen=True)
class StampTarget:
    source_id: int
    gaia_g_mag: float
    psf_id: int
    detector_xpix: float
    detector_ypix: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class LoadedStampTargetTable:
    targets: tuple[StampTarget, ...]
    provenance: dict[str, Any]


def _token(value: str) -> str:
    return _COLUMN_TOKEN.sub("_", str(value).strip().lower()).strip("_")


def _canonical_columns(table: Table) -> dict[str, str]:
    aliases = {
        alias: canonical
        for canonical, values in _ALIASES.items()
        for alias in values
    }
    result: dict[str, str] = {}
    for column in table.colnames:
        canonical = aliases.get(_token(column))
        if canonical is None:
            continue
        if canonical in result:
            raise ValueError(
                f"target table has multiple columns for {canonical!r}"
            )
        result[canonical] = column
    return result


def _integer(value: Any, *, field_name: str, row_index: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"row {row_index} {field_name} must be an integer")
    try:
        converted = int(value)
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"row {row_index} {field_name} must be an integer"
        ) from exc
    if not math.isfinite(numeric) or numeric != converted or converted < 0:
        raise ValueError(
            f"row {row_index} {field_name} must be a non-negative integer"
        )
    return converted


def _finite(value: Any, *, field_name: str, row_index: int) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"row {row_index} {field_name} must be finite") from exc
    if not math.isfinite(converted):
        raise ValueError(f"row {row_index} {field_name} must be finite")
    return converted


def load_stamp_target_table(
    path: Path | str,
    *,
    detector_shape: tuple[int, int],
) -> LoadedStampTargetTable:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"stamp target table does not exist: {source}")
    try:
        rows, cols = (int(value) for value in detector_shape)
    except (TypeError, ValueError) as exc:
        raise ValueError("detector_shape must contain positive rows and columns") from exc
    if rows <= 0 or cols <= 0:
        raise ValueError("detector_shape must contain positive rows and columns")

    table = Table.read(source)
    if len(table) == 0:
        raise ValueError("stamp target table must contain at least one row")
    columns = _canonical_columns(table)
    if "gaia_g_mag" not in columns:
        raise ValueError("stamp target table requires a Gaia G magnitude column")
    if "psf_id" not in columns:
        raise ValueError("stamp target table requires a PSF ID column")
    has_x = "detector_xpix" in columns
    has_y = "detector_ypix" in columns
    if has_x != has_y:
        raise ValueError(
            "Detector Xpix and Detector Ypix columns must be provided together"
        )

    center_x = (cols - 1) / 2.0
    center_y = (rows - 1) / 2.0
    targets: list[StampTarget] = []
    for row_index, row in enumerate(table):
        source_id = (
            row_index
            if "source_id" not in columns
            else _integer(
                row[columns["source_id"]],
                field_name="source_id",
                row_index=row_index,
            )
        )
        gaia_g_mag = _finite(
            row[columns["gaia_g_mag"]],
            field_name="gaia_g_mag",
            row_index=row_index,
        )
        psf_id = _integer(
            row[columns["psf_id"]],
            field_name="psf_id",
            row_index=row_index,
        )
        detector_xpix = (
            center_x
            if not has_x
            else _finite(
                row[columns["detector_xpix"]],
                field_name="detector_xpix",
                row_index=row_index,
            )
        )
        detector_ypix = (
            center_y
            if not has_y
            else _finite(
                row[columns["detector_ypix"]],
                field_name="detector_ypix",
                row_index=row_index,
            )
        )
        if not (0.0 <= detector_xpix <= cols - 1) or not (
            0.0 <= detector_ypix <= rows - 1
        ):
            raise ValueError(
                f"row {row_index} detector coordinates must be inside detector shape"
            )
        targets.append(
            StampTarget(
                source_id=source_id,
                gaia_g_mag=gaia_g_mag,
                psf_id=psf_id,
                detector_xpix=detector_xpix,
                detector_ypix=detector_ypix,
            )
        )

    source_ids = [target.source_id for target in targets]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("stamp target source_id values must be unique")
    stat = source.stat()
    return LoadedStampTargetTable(
        targets=tuple(targets),
        provenance={
            "schema_id": "et_mainsim.stamp_target_table",
            "schema_version": 1,
            "path": str(source),
            "format": source.suffix.lower().lstrip("."),
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "row_count": len(targets),
            "magnitude_column": "gaia_g_mag",
            "magnitude_system": "Gaia_G_Vega",
            "photon_magnitude_system": "ET_AB",
            "magnitude_conversion": "gaia_g_vega_equals_et_ab_g2v_approx",
            "coordinate_default": "physical_detector_center",
            "scene_policy": "one_independent_target_per_row_no_neighbors",
        },
    )


__all__ = [
    "LoadedStampTargetTable",
    "StampTarget",
    "load_stamp_target_table",
]
