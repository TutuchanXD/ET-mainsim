from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import numpy as np
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
    "ra_deg": {"ra_deg", "ra", "icrs_ra_deg"},
    "dec_deg": {"dec_deg", "dec", "icrs_dec_deg"},
    "curve_id": {"curve_id", "lightcurve_id", "light_curve_id"},
    "frame_index": {"frame_index", "raw_frame_index", "frame"},
    "relative_flux": {"relative_flux", "relative_flux_factor", "flux_factor"},
}


@dataclass(frozen=True)
class StampTarget:
    source_id: int
    gaia_g_mag: float
    psf_id: int | None
    detector_xpix: float
    detector_ypix: float
    curve_id: str | None = None
    location_mode: str = "explicit_psf"
    ra_deg: float | None = None
    dec_deg: float | None = None
    field_x_deg: float | None = None
    field_y_deg: float | None = None
    field_angle_deg: float | None = None
    focalplane_residual_arcsec: float | None = None
    psf_node_angle_deg: float | None = None
    psf_angle_delta_deg: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LoadedStampTargetTable:
    targets: tuple[StampTarget, ...]
    provenance: dict[str, Any]


@dataclass(frozen=True)
class LoadedStampVariabilityTable:
    curves: Mapping[str, tuple[float, ...]]
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
                f"input table has multiple columns for {canonical!r}"
            )
        result[canonical] = column
    return result


def _missing(value: Any) -> bool:
    if value is None or np.ma.is_masked(value):
        return True
    if isinstance(value, str):
        return not value.strip()
    return False


def _integer(value: Any, *, field_name: str, row_index: int) -> int:
    if isinstance(value, (bool, np.bool_)):
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


def _nonempty_string(value: Any, *, field_name: str, row_index: int) -> str:
    if _missing(value):
        raise ValueError(f"row {row_index} {field_name} must be non-empty")
    converted = str(value).strip()
    if not converted:
        raise ValueError(f"row {row_index} {field_name} must be non-empty")
    return converted


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError):
        return str(value)
    return value


def file_identity(path: Path | str) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = source.stat()
    return {
        "path": str(source),
        "size_bytes": int(stat.st_size),
        "sha256": digest.hexdigest(),
    }


def directory_identity(path: Path | str) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"directory does not exist: {source}")
    files = sorted(item for item in source.rglob("*") if item.is_file())
    digest = hashlib.sha256()
    entries: list[dict[str, Any]] = []
    size_bytes = 0
    for item in files:
        identity = file_identity(item)
        relative_path = item.relative_to(source).as_posix()
        size_bytes += int(identity["size_bytes"])
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(identity["sha256"]).encode("ascii"))
        digest.update(b"\0")
        entries.append(
            {
                "path": relative_path,
                "size_bytes": int(identity["size_bytes"]),
                "sha256": str(identity["sha256"]),
            }
        )
    return {
        "path": str(source),
        "file_count": len(entries),
        "size_bytes": size_bytes,
        "sha256": digest.hexdigest(),
        "files": entries,
    }


def focalplane_registry_identity(path: Path | str) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    data_dir = source / "data" if (source / "data").is_dir() else source
    return directory_identity(data_dir)


@lru_cache(maxsize=4)
def _load_focalplane_registry(path: str, registry_sha256: str = "") -> Any:
    try:
        from et_coord import load_registry
    except ImportError as exc:
        raise RuntimeError(
            "coordinate targets require et-coord (the et_focalplane package); "
            "install it in the runtime environment and set ET_FOCALPLANE_ROOT"
        ) from exc

    return load_registry(path)


def _sky_to_focal(
    focalplane_registry: Path | str,
    *,
    ra_deg: float,
    dec_deg: float,
    registry_sha256: str = "",
) -> Any:
    try:
        from et_coord import sky_to_focal
    except ImportError as exc:
        raise RuntimeError(
            "coordinate targets require et-coord (the et_focalplane package); "
            "install it in the runtime environment and set ET_FOCALPLANE_ROOT"
        ) from exc

    registry = _load_focalplane_registry(
        str(Path(focalplane_registry).expanduser().resolve()),
        str(registry_sha256),
    )
    return sky_to_focal(registry, ra=ra_deg, dec=dec_deg)


def _coordinate_value(
    row: Any,
    columns: Mapping[str, str],
    name: str,
) -> Any:
    column = columns.get(name)
    return None if column is None else row[column]


def load_stamp_target_table(
    path: Path | str,
    *,
    detector_shape: tuple[int, int],
    detector_id: str | None = None,
    focalplane_registry: Path | str | None = None,
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
    has_x_column = "detector_xpix" in columns
    has_y_column = "detector_ypix" in columns
    if has_x_column != has_y_column:
        raise ValueError(
            "Detector Xpix and Detector Ypix columns must be provided together"
        )
    has_ra_column = "ra_deg" in columns
    has_dec_column = "dec_deg" in columns
    if has_ra_column != has_dec_column:
        raise ValueError("RA and Dec columns must be provided together")
    if "psf_id" not in columns and not has_ra_column:
        raise ValueError("stamp target table requires a PSF ID or RA/Dec columns")

    center_x = (cols - 1) / 2.0
    center_y = (rows - 1) / 2.0
    targets: list[StampTarget] = []
    sky_count = 0
    registry_identity: dict[str, Any] | None = None
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
        curve_value = _coordinate_value(row, columns, "curve_id")
        curve_id = (
            None
            if _missing(curve_value)
            else _nonempty_string(
                curve_value,
                field_name="curve_id",
                row_index=row_index,
            )
        )

        psf_value = _coordinate_value(row, columns, "psf_id")
        ra_value = _coordinate_value(row, columns, "ra_deg")
        dec_value = _coordinate_value(row, columns, "dec_deg")
        x_value = _coordinate_value(row, columns, "detector_xpix")
        y_value = _coordinate_value(row, columns, "detector_ypix")
        has_psf = not _missing(psf_value)
        has_ra = not _missing(ra_value)
        has_dec = not _missing(dec_value)
        has_x = not _missing(x_value)
        has_y = not _missing(y_value)
        if has_ra != has_dec:
            raise ValueError(f"row {row_index} RA and Dec must be provided together")
        if has_x != has_y:
            raise ValueError(
                f"row {row_index} Detector Xpix and Detector Ypix must be provided together"
            )
        if has_ra and (has_psf or has_x):
            raise ValueError(
                f"row {row_index} RA/Dec and explicit PSF/detector coordinates are mutually exclusive"
            )
        if not has_ra and not has_psf:
            raise ValueError(
                f"row {row_index} requires mutually exclusive RA/Dec or explicit PSF ID"
            )

        if has_ra:
            if detector_id is None or not str(detector_id).strip():
                raise ValueError("coordinate targets require the configured detector_id")
            if focalplane_registry is None:
                raise ValueError(
                    "coordinate targets require the fixed transit focalplane registry"
                )
            if registry_identity is None:
                registry_identity = focalplane_registry_identity(
                    focalplane_registry
                )
            ra_deg = _finite(ra_value, field_name="ra_deg", row_index=row_index)
            dec_deg = _finite(dec_value, field_name="dec_deg", row_index=row_index)
            if not 0.0 <= ra_deg < 360.0:
                raise ValueError(f"row {row_index} ra_deg must be in [0, 360)")
            if not -90.0 <= dec_deg <= 90.0:
                raise ValueError(f"row {row_index} dec_deg must be in [-90, 90]")
            result = _sky_to_focal(
                focalplane_registry,
                ra_deg=ra_deg,
                dec_deg=dec_deg,
                registry_sha256=str(registry_identity["sha256"]),
            )
            if str(getattr(result, "status", "")) != "ok":
                raise ValueError(
                    f"row {row_index} ICRS/J2000 coordinate is outside the fixed transit field of view"
                )
            mapped_detector = str(getattr(result, "detector_id", ""))
            if mapped_detector != str(detector_id):
                raise ValueError(
                    f"row {row_index} maps to detector {mapped_detector!r}, not configured detector {detector_id!r}"
                )
            detector_xpix = _finite(
                getattr(result, "xpix", None),
                field_name="mapped detector_xpix",
                row_index=row_index,
            )
            detector_ypix = _finite(
                getattr(result, "ypix", None),
                field_name="mapped detector_ypix",
                row_index=row_index,
            )
            field_x_deg = _finite(
                getattr(result, "field_x_deg", None),
                field_name="mapped field_x_deg",
                row_index=row_index,
            )
            field_y_deg = _finite(
                getattr(result, "field_y_deg", None),
                field_name="mapped field_y_deg",
                row_index=row_index,
            )
            field_angle_deg = float(math.hypot(field_x_deg, field_y_deg))
            focalplane_residual_arcsec = _finite(
                getattr(result, "residual_arcsec", None),
                field_name="focalplane residual_arcsec",
                row_index=row_index,
            )
            if focalplane_residual_arcsec < 0.0:
                raise ValueError(
                    f"row {row_index} focalplane residual_arcsec must be non-negative"
                )
            psf_id = None
            location_mode = "sky_icrs_j2000"
            sky_count += 1
        else:
            psf_id = _integer(
                psf_value,
                field_name="psf_id",
                row_index=row_index,
            )
            detector_xpix = (
                center_x
                if not has_x
                else _finite(
                    x_value,
                    field_name="detector_xpix",
                    row_index=row_index,
                )
            )
            detector_ypix = (
                center_y
                if not has_y
                else _finite(
                    y_value,
                    field_name="detector_ypix",
                    row_index=row_index,
                )
            )
            ra_deg = dec_deg = None
            field_x_deg = field_y_deg = field_angle_deg = None
            focalplane_residual_arcsec = None
            location_mode = "explicit_psf"

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
                curve_id=curve_id,
                location_mode=location_mode,
                ra_deg=ra_deg,
                dec_deg=dec_deg,
                field_x_deg=field_x_deg,
                field_y_deg=field_y_deg,
                field_angle_deg=field_angle_deg,
                focalplane_residual_arcsec=focalplane_residual_arcsec,
            )
        )

    source_ids = [target.source_id for target in targets]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("stamp target source_id values must be unique")
    provenance: dict[str, Any] = {
        "schema_id": "et_mainsim.stamp_target_table",
        "schema_version": 2,
        "path": str(source),
        "format": source.suffix.lower().lstrip("."),
        "row_count": len(targets),
        "file_identity": file_identity(source),
        "table_meta": _json_safe(dict(table.meta)),
        "magnitude_column": "gaia_g_mag",
        "magnitude_system": "Gaia_G_Vega",
        "photon_magnitude_system": "ET_AB",
        "magnitude_conversion": "gaia_g_vega_equals_et_ab_g2v_approx",
        "coordinate_default": "physical_detector_center",
        "coordinate_frame": "ICRS_J2000" if sky_count else None,
        "coordinate_row_count": sky_count,
        "explicit_psf_row_count": len(targets) - sky_count,
        "scene_policy": "one_independent_target_per_row_no_neighbors",
    }
    if sky_count:
        final_registry_identity = focalplane_registry_identity(
            Path(focalplane_registry)  # type: ignore[arg-type]
        )
        if final_registry_identity != registry_identity:
            raise RuntimeError(
                "focalplane registry changed while resolving stamp target coordinates"
            )
        provenance["focalplane_registry_identity"] = registry_identity
    return LoadedStampTargetTable(
        targets=tuple(targets),
        provenance=provenance,
    )


def load_stamp_variability_table(
    path: Path | str,
    *,
    raw_frame_count: int,
) -> LoadedStampVariabilityTable:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(
            f"stamp variability table does not exist: {source}"
        )
    raw_frame_count = int(raw_frame_count)
    if raw_frame_count <= 0:
        raise ValueError("raw_frame_count must be positive")
    table = Table.read(source)
    if len(table) == 0:
        raise ValueError("stamp variability table must contain at least one row")
    columns = _canonical_columns(table)
    for required in ("curve_id", "frame_index", "relative_flux"):
        if required not in columns:
            raise ValueError(
                f"stamp variability table requires a {required} column"
            )

    grouped: dict[str, dict[int, float]] = {}
    for row_index, row in enumerate(table):
        curve_id = _nonempty_string(
            row[columns["curve_id"]],
            field_name="curve_id",
            row_index=row_index,
        )
        frame_index = _integer(
            row[columns["frame_index"]],
            field_name="frame_index",
            row_index=row_index,
        )
        relative_flux = _finite(
            row[columns["relative_flux"]],
            field_name="relative_flux",
            row_index=row_index,
        )
        if relative_flux < 0.0:
            raise ValueError(
                f"row {row_index} relative_flux must be non-negative"
            )
        curve = grouped.setdefault(curve_id, {})
        if frame_index in curve:
            raise ValueError(
                f"curve {curve_id!r} has duplicate frame_index {frame_index}"
            )
        curve[frame_index] = relative_flux

    expected = tuple(range(raw_frame_count))
    curves: dict[str, tuple[float, ...]] = {}
    for curve_id, by_frame in grouped.items():
        actual = tuple(sorted(by_frame))
        if actual != expected:
            raise ValueError(
                f"curve {curve_id!r} frame_index values must be exactly "
                f"0..{raw_frame_count - 1}; got {list(actual)}"
            )
        curves[curve_id] = tuple(by_frame[index] for index in expected)

    canonical_source_columns = set(columns.values())
    ignored_time_columns = sorted(
        str(name)
        for name in table.colnames
        if name not in canonical_source_columns and "time" in _token(name)
    )
    return LoadedStampVariabilityTable(
        curves=curves,
        provenance={
            "schema_id": "et_mainsim.stamp_variability_table",
            "schema_version": 1,
            "path": str(source),
            "format": source.suffix.lower().lstrip("."),
            "row_count": len(table),
            "curve_count": len(curves),
            "curve_ids": sorted(curves),
            "raw_frame_count": raw_frame_count,
            "time_alignment": "simulation_raw_frame_index",
            "input_time_columns_ignored": ignored_time_columns,
            "file_identity": file_identity(source),
            "table_meta": _json_safe(dict(table.meta)),
        },
    )


__all__ = [
    "LoadedStampTargetTable",
    "LoadedStampVariabilityTable",
    "StampTarget",
    "directory_identity",
    "focalplane_registry_identity",
    "file_identity",
    "load_stamp_target_table",
    "load_stamp_variability_table",
]
