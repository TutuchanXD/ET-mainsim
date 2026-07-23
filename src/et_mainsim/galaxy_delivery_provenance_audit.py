"""Full-header provenance and PSF audit for a formal Galaxy delivery.

Campaign delivery QC proves the complete target × time-shard × product matrix
and its compact delivery headers.  This complementary audit opens every final
HDF5 member and validates the embedded source truth, selected PSF, factor
snapshot identity, and source-code commit identity.  It intentionally treats
the Git commit and clean/dirty state as the software identity: installed
distribution-version labels are collected for disclosure but are not required
to match the prepare-time environment.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Literal
import uuid

import numpy as np

from .galaxy_campaign_qc import _expected_bundles, _load_campaign, _target_ids
from .stamp_inputs import file_identity


GALAXY_DELIVERY_PROVENANCE_AUDIT_SCHEMA_ID = (
    "et_mainsim.galaxy_delivery_provenance_audit.v1"
)
GALAXY_DELIVERY_PROVENANCE_AUDIT_SCHEMA_VERSION = 1
_FLOAT_TOLERANCE = 1e-7
_PACKAGES = ("et_mainsim", "photsim7")


class GalaxyDeliveryProvenanceAuditError(ValueError):
    """Raised when a formal provenance-audit contract is malformed."""


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GalaxyDeliveryProvenanceAuditError(f"{label} must be an object")
    return value


def _text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GalaxyDeliveryProvenanceAuditError(f"{label} must be a non-empty string")
    return value.strip()


def _source_id(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise GalaxyDeliveryProvenanceAuditError(f"{label} must be a signed int64")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise GalaxyDeliveryProvenanceAuditError(
            f"{label} must be a signed int64"
        ) from error
    if result < 0 or result > int(np.iinfo(np.int64).max) or result != value:
        raise GalaxyDeliveryProvenanceAuditError(f"{label} must be a signed int64")
    return result


def _finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise GalaxyDeliveryProvenanceAuditError(f"{label} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise GalaxyDeliveryProvenanceAuditError(f"{label} must be finite") from error
    if not math.isfinite(result):
        raise GalaxyDeliveryProvenanceAuditError(f"{label} must be finite")
    return result


def _equal_float(
    actual: Any,
    expected: Any,
    *,
    label: str,
    abs_tolerance: float = _FLOAT_TOLERANCE,
) -> None:
    actual_value = _finite(actual, label=label)
    expected_value = _finite(expected, label=f"expected {label}")
    if not math.isclose(
        actual_value,
        expected_value,
        rel_tol=0.0,
        abs_tol=abs_tolerance,
    ):
        raise GalaxyDeliveryProvenanceAuditError(
            f"{label} conflicts with frozen production manifest"
        )


def _same_identity(actual: Any, expected: Any, *, label: str) -> None:
    actual_mapping = _mapping(actual, label=label)
    expected_mapping = _mapping(expected, label=f"expected {label}")
    if (
        actual_mapping.get("sha256") != expected_mapping.get("sha256")
        or actual_mapping.get("size_bytes") != expected_mapping.get("size_bytes")
    ):
        raise GalaxyDeliveryProvenanceAuditError(
            f"{label} conflicts with frozen production manifest"
        )


def _json_scalar(handle: Any, name: str) -> Mapping[str, Any]:
    if name not in handle:
        raise GalaxyDeliveryProvenanceAuditError(f"delivery is missing {name}")
    value = handle[name][()]
    if isinstance(value, np.ndarray):
        if value.shape != ():
            raise GalaxyDeliveryProvenanceAuditError(f"delivery {name} must be scalar")
        value = value.reshape(()).item()
    if isinstance(value, (bytes, np.bytes_)):
        value = bytes(value).decode("utf-8")
    try:
        return _mapping(json.loads(str(value)), label=name)
    except json.JSONDecodeError as error:
        raise GalaxyDeliveryProvenanceAuditError(
            f"delivery {name} is not valid JSON"
        ) from error


def _expected_software(manifest: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    prepared = _mapping(
        manifest.get("software_provenance_at_prepare"),
        label="production manifest software_provenance_at_prepare",
    )
    result: dict[str, Mapping[str, Any]] = {}
    for package in _PACKAGES:
        record = _mapping(prepared.get(package), label=f"prepare software {package}")
        _text(record.get("commit"), label=f"prepare software {package}.commit")
        if record.get("dirty") is not False:
            raise GalaxyDeliveryProvenanceAuditError(
                f"prepare software {package}.dirty must be false"
            )
        result[package] = record
    return result


def _target_records(manifest: Mapping[str, Any]) -> Mapping[int, Mapping[str, Any]]:
    raw_targets = manifest.get("targets")
    if not isinstance(raw_targets, list) or not raw_targets:
        raise GalaxyDeliveryProvenanceAuditError("production manifest targets must be a list")
    records: dict[int, Mapping[str, Any]] = {}
    for item in raw_targets:
        target = _mapping(item, label="production manifest target")
        source_id = _source_id(
            target.get("source_id_int64"), label="target.source_id_int64"
        )
        if source_id in records:
            raise GalaxyDeliveryProvenanceAuditError(
                "production manifest target source IDs must be unique"
            )
        _text(target.get("magnitude_system"), label="target.magnitude_system")
        _finite(target.get("gaia_g_mag"), label="target.gaia_g_mag")
        _finite(target.get("ra_deg"), label="target.ra_deg")
        _finite(target.get("dec_deg"), label="target.dec_deg")
        focalplane = _mapping(target.get("focalplane_mapping"), label="target.focalplane_mapping")
        _text(focalplane.get("detector_id"), label="target.focalplane_mapping.detector_id")
        for field in ("detector_xpix", "detector_ypix", "field_angle_deg"):
            _finite(focalplane.get(field), label=f"target.focalplane_mapping.{field}")
        _mapping(target.get("factor_snapshot"), label="target.factor_snapshot")
        records[source_id] = target
    return records


def _nearest_psf_id(
    node_angles: Mapping[str, Any], *, field_angle_deg: float
) -> tuple[int, float]:
    candidates: list[tuple[float, int, float]] = []
    for raw_id, raw_angle in node_angles.items():
        if isinstance(raw_id, bool):
            raise GalaxyDeliveryProvenanceAuditError("PSF node ID must be an integer")
        try:
            psf_id = int(raw_id)
        except (TypeError, ValueError, OverflowError) as error:
            raise GalaxyDeliveryProvenanceAuditError(
                "PSF node ID must be an integer"
            ) from error
        if psf_id < 0:
            raise GalaxyDeliveryProvenanceAuditError("PSF node ID must be an integer")
        angle = _finite(raw_angle, label=f"PSF node angle {psf_id}")
        candidates.append((abs(angle - field_angle_deg), psf_id, angle))
    if not candidates:
        raise GalaxyDeliveryProvenanceAuditError("PSF node_angles_deg must not be empty")
    _delta, psf_id, node_angle = min(candidates, key=lambda item: (item[0], item[1]))
    return psf_id, node_angle


def _validate_target_truth(
    truth: Mapping[str, Any], *, source_id: int, target: Mapping[str, Any]
) -> tuple[int, float]:
    if _source_id(truth.get("source_id"), label="target_input_truth.source_id") != source_id:
        raise GalaxyDeliveryProvenanceAuditError(
            "target_input_truth.source_id conflicts with delivery path"
        )
    if truth.get("magnitude_system") != target.get("magnitude_system"):
        raise GalaxyDeliveryProvenanceAuditError(
            "target_input_truth.magnitude_system conflicts with frozen production manifest"
        )
    _equal_float(
        truth.get("gaia_g_mag"), target.get("gaia_g_mag"), label="target_input_truth.gaia_g_mag"
    )
    location = _mapping(truth.get("location"), label="target_input_truth.location")
    if location.get("mode") != "sky_icrs_j2000":
        raise GalaxyDeliveryProvenanceAuditError(
            "target_input_truth.location.mode must be sky_icrs_j2000"
        )
    _equal_float(location.get("ra_deg"), target.get("ra_deg"), label="target_input_truth.location.ra_deg")
    _equal_float(location.get("dec_deg"), target.get("dec_deg"), label="target_input_truth.location.dec_deg")
    focalplane = _mapping(target.get("focalplane_mapping"), label="target.focalplane_mapping")
    for field in ("detector_xpix", "detector_ypix"):
        _equal_float(
            location.get(field),
            focalplane.get(field),
            label=f"target_input_truth.location.{field}",
            abs_tolerance=1e-2,
        )
    _equal_float(
        location.get("field_angle_deg"),
        focalplane.get("field_angle_deg"),
        label="target_input_truth.location.field_angle_deg",
        abs_tolerance=1e-5,
    )

    psf = _mapping(truth.get("psf"), label="target_input_truth.psf")
    if psf.get("selection_policy") != "nearest_radial_field_angle":
        raise GalaxyDeliveryProvenanceAuditError(
            "target_input_truth.psf.selection_policy must be nearest_radial_field_angle"
        )
    bundle = _mapping(psf.get("bundle"), label="target_input_truth.psf.bundle")
    bundle_sha256 = _text(bundle.get("expected_sha256"), label="target_input_truth.psf.bundle.expected_sha256")
    bundle_identity = _mapping(
        bundle.get("file_identity"), label="target_input_truth.psf.bundle.file_identity"
    )
    if bundle_identity.get("sha256") != bundle_sha256:
        raise GalaxyDeliveryProvenanceAuditError(
            "target_input_truth.psf bundle identity conflicts with expected SHA-256"
        )
    expected_id, expected_node_angle = _nearest_psf_id(
        _mapping(bundle.get("node_angles_deg"), label="target_input_truth.psf.bundle.node_angles_deg"),
        field_angle_deg=_finite(focalplane.get("field_angle_deg"), label="target field angle"),
    )
    chosen_id = _source_id(psf.get("chosen_psf_id"), label="target_input_truth.psf.chosen_psf_id")
    if chosen_id != expected_id:
        raise GalaxyDeliveryProvenanceAuditError(
            "target_input_truth.psf.chosen_psf_id conflicts with nearest field-angle node"
        )
    _equal_float(
        psf.get("node_angle_deg"),
        expected_node_angle,
        label="target_input_truth.psf.node_angle_deg",
    )
    return chosen_id, expected_node_angle


def _validate_header(
    path: Path,
    *,
    source_id: int,
    target: Mapping[str, Any],
    run_id: str,
    case: str,
    expected_software: Mapping[str, Mapping[str, Any]],
) -> tuple[int, float, Mapping[str, str | None]]:
    try:
        import h5py
    except ImportError as error:  # pragma: no cover - package guard
        raise RuntimeError("h5py is required for Galaxy provenance audit") from error
    with h5py.File(path, "r") as handle:
        delivery_manifest = _json_scalar(handle, "manifest_json")
        if _source_id(
            delivery_manifest.get("target_source_id_int64"),
            label="delivery target_source_id_int64",
        ) != source_id:
            raise GalaxyDeliveryProvenanceAuditError(
                "delivery target_source_id_int64 conflicts with path"
            )
        caller = _mapping(delivery_manifest.get("caller_manifest"), label="delivery caller_manifest")
        if caller.get("run_id") != run_id or caller.get("case") != case:
            raise GalaxyDeliveryProvenanceAuditError(
                "delivery caller manifest conflicts with frozen campaign"
            )
        chosen_psf_id, node_angle_deg = _validate_target_truth(
            _mapping(caller.get("target_input_truth"), label="caller target_input_truth"),
            source_id=source_id,
            target=target,
        )
        provenance = _json_scalar(handle, "provenance_json")
        caller_provenance = _mapping(
            provenance.get("caller_provenance"), label="delivery caller_provenance"
        )
        _same_identity(
            caller_provenance.get("factor_snapshot_identity"),
            target.get("factor_snapshot"),
            label="delivery factor_snapshot_identity",
        )
        simulation_spec = _mapping(
            caller_provenance.get("simulation_spec"), label="delivery simulation_spec"
        )
        detector = _mapping(
            simulation_spec.get("detector"), label="delivery simulation_spec.detector"
        )
        expected_focalplane = _mapping(
            target.get("focalplane_mapping"), label="target.focalplane_mapping"
        )
        if detector.get("detector_id") != expected_focalplane.get("detector_id"):
            raise GalaxyDeliveryProvenanceAuditError(
                "delivery simulation_spec.detector.detector_id conflicts with target mapping"
            )
        software = _mapping(caller_provenance.get("software"), label="delivery software")
        versions: dict[str, str | None] = {}
        for package in _PACKAGES:
            actual = _mapping(software.get(package), label=f"delivery software {package}")
            expected = expected_software[package]
            if actual.get("commit") != expected.get("commit"):
                raise GalaxyDeliveryProvenanceAuditError(
                    f"delivery software {package}.commit conflicts with prepare-time commit"
                )
            if actual.get("dirty") is not False:
                raise GalaxyDeliveryProvenanceAuditError(
                    f"delivery software {package}.dirty must be false"
                )
            version = actual.get("version")
            versions[package] = version if isinstance(version, str) else None
    return chosen_psf_id, node_angle_deg, versions


@dataclass(frozen=True)
class GalaxyDeliveryProvenanceAuditRequest:
    """One explicit full-header audit for an immutable injected Galaxy delivery."""

    production_manifest_path: Path | str
    case: Literal["injected"] | str = "injected"

    def __post_init__(self) -> None:
        if self.case != "injected":
            raise GalaxyDeliveryProvenanceAuditError(
                "formal Galaxy provenance audit currently supports injected only"
            )
        object.__setattr__(
            self,
            "production_manifest_path",
            Path(self.production_manifest_path).expanduser().resolve(),
        )
        object.__setattr__(self, "case", "injected")


@dataclass(frozen=True)
class GalaxyDeliveryProvenanceAuditResult:
    """Serializable result of a full final-HDF5 provenance/PSF header sweep."""

    production_manifest_path: Path
    run_root: Path
    run_id: str
    case: str
    expected_bundle_count: int
    valid_bundle_count: int
    missing_bundles: tuple[Mapping[str, Any], ...]
    invalid_bundles: tuple[Mapping[str, Any], ...]
    source_summaries: Mapping[str, Mapping[str, Any]]
    expected_software: Mapping[str, Mapping[str, Any]]
    observed_versions: Mapping[str, Mapping[str, int]]

    @property
    def missing_bundle_count(self) -> int:
        return len(self.missing_bundles)

    @property
    def invalid_bundle_count(self) -> int:
        return len(self.invalid_bundles)

    @property
    def ready(self) -> bool:
        return (
            self.valid_bundle_count == self.expected_bundle_count
            and self.missing_bundle_count == 0
            and self.invalid_bundle_count == 0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_id": GALAXY_DELIVERY_PROVENANCE_AUDIT_SCHEMA_ID,
            "schema_version": GALAXY_DELIVERY_PROVENANCE_AUDIT_SCHEMA_VERSION,
            "status": "ready" if self.ready else "incomplete_or_invalid",
            "ready": self.ready,
            "production_manifest_path": str(self.production_manifest_path),
            "production_manifest_identity": file_identity(self.production_manifest_path),
            "run_root": str(self.run_root),
            "run_id": self.run_id,
            "case": self.case,
            "inspection_mode": "all_final_hdf5_headers; no_final_dn_cube_reads",
            "software_identity_policy": {
                "authority": "git_commit_and_dirty_state",
                "distribution_version": "collected_for_disclosure_not_equality_gate",
            },
            "expected_bundle_count": self.expected_bundle_count,
            "valid_bundle_count": self.valid_bundle_count,
            "missing_bundle_count": self.missing_bundle_count,
            "invalid_bundle_count": self.invalid_bundle_count,
            "expected_software": {
                package: {
                    "commit": record.get("commit"),
                    "dirty": record.get("dirty"),
                }
                for package, record in sorted(self.expected_software.items())
            },
            "software": {
                "observed_versions": {
                    package: dict(sorted(counts.items()))
                    for package, counts in sorted(self.observed_versions.items())
                }
            },
            "source_summaries": {
                source_id: dict(summary)
                for source_id, summary in sorted(
                    self.source_summaries.items(), key=lambda item: int(item[0])
                )
            },
            "missing_bundles": [dict(record) for record in self.missing_bundles],
            "invalid_bundles": [dict(record) for record in self.invalid_bundles],
        }


def audit_galaxy_delivery_provenance_v1(
    request: GalaxyDeliveryProvenanceAuditRequest,
) -> GalaxyDeliveryProvenanceAuditResult:
    """Validate provenance and PSF headers for every expected final delivery HDF5."""

    if not isinstance(request, GalaxyDeliveryProvenanceAuditRequest):
        raise TypeError("request must be GalaxyDeliveryProvenanceAuditRequest")
    run_root, manifest, time_plan, _coadd_sizes, _stamp_shape = _load_campaign(
        request.production_manifest_path
    )
    run_id = _text(manifest.get("run_id"), label="production manifest run_id")
    target_records = _target_records(manifest)
    source_ids = _target_ids(manifest)
    if set(target_records) != set(source_ids):
        raise GalaxyDeliveryProvenanceAuditError(
            "production manifest target IDs conflict with frozen time delivery"
        )
    expected_software = _expected_software(manifest)
    bundles = _expected_bundles(
        run_root=run_root,
        case=request.case,
        source_ids=source_ids,
        time_plan=time_plan,
    )
    source_summaries: dict[str, dict[str, Any]] = {
        str(source_id): {
            "source_id": str(source_id),
            "expected_bundle_count": 0,
            "valid_bundle_count": 0,
            "missing_bundle_count": 0,
            "invalid_bundle_count": 0,
            "chosen_psf_id": None,
            "node_angle_deg": None,
        }
        for source_id in source_ids
    }
    observed_versions: dict[str, Counter[str]] = {
        package: Counter() for package in _PACKAGES
    }
    valid_count = 0
    missing: list[Mapping[str, Any]] = []
    invalid: list[Mapping[str, Any]] = []
    for bundle in bundles:
        summary = source_summaries[str(bundle.source_id)]
        summary["expected_bundle_count"] += 1
        if not bundle.path.is_file():
            summary["missing_bundle_count"] += 1
            missing.append(bundle.record())
            continue
        try:
            chosen_psf_id, node_angle_deg, versions = _validate_header(
                bundle.path,
                source_id=bundle.source_id,
                target=target_records[bundle.source_id],
                run_id=run_id,
                case=request.case,
                expected_software=expected_software,
            )
        except (OSError, TypeError, ValueError) as error:
            summary["invalid_bundle_count"] += 1
            record = bundle.record()
            record["error"] = str(error)
            invalid.append(record)
            continue
        for package, version in versions.items():
            observed_versions[package][version if version is not None else "<missing>"] += 1
        if summary["chosen_psf_id"] is None:
            summary["chosen_psf_id"] = chosen_psf_id
            summary["node_angle_deg"] = node_angle_deg
        elif (
            summary["chosen_psf_id"] != chosen_psf_id
            or not math.isclose(
                float(summary["node_angle_deg"]),
                node_angle_deg,
                rel_tol=0.0,
                abs_tol=_FLOAT_TOLERANCE,
            )
        ):
            summary["invalid_bundle_count"] += 1
            record = bundle.record()
            record["error"] = "source PSF selection is inconsistent across delivery members"
            invalid.append(record)
            continue
        summary["valid_bundle_count"] += 1
        valid_count += 1
    return GalaxyDeliveryProvenanceAuditResult(
        production_manifest_path=request.production_manifest_path,
        run_root=run_root,
        run_id=run_id,
        case=request.case,
        expected_bundle_count=len(bundles),
        valid_bundle_count=valid_count,
        missing_bundles=tuple(missing),
        invalid_bundles=tuple(invalid),
        source_summaries=source_summaries,
        expected_software=expected_software,
        observed_versions={
            package: dict(counter) for package, counter in observed_versions.items()
        },
    )


def write_galaxy_delivery_provenance_audit_json(
    result: GalaxyDeliveryProvenanceAuditResult, path: Path | str
) -> Path:
    """Atomically write one machine-readable provenance-audit receipt."""

    if not isinstance(result, GalaxyDeliveryProvenanceAuditResult):
        raise TypeError("result must be GalaxyDeliveryProvenanceAuditResult")
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(result.to_dict(), stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-manifest", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--require-complete", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Write an audit receipt and optionally fail closed on any gap or anomaly."""

    args = _parser().parse_args(None if argv is None else list(argv))
    try:
        result = audit_galaxy_delivery_provenance_v1(
            GalaxyDeliveryProvenanceAuditRequest(
                production_manifest_path=args.production_manifest
            )
        )
        receipt = write_galaxy_delivery_provenance_audit_json(result, args.output_json)
    except (OSError, ValueError) as error:
        print(f"campaign provenance audit failed: {error}")
        return 1
    print(
        json.dumps(
            {
                "receipt": str(receipt),
                "ready": result.ready,
                "expected_bundle_count": result.expected_bundle_count,
                "valid_bundle_count": result.valid_bundle_count,
                "missing_bundle_count": result.missing_bundle_count,
                "invalid_bundle_count": result.invalid_bundle_count,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if result.ready or not args.require_complete else 2


if __name__ == "__main__":  # pragma: no cover - CLI-only path.
    raise SystemExit(main())


__all__ = [
    "GALAXY_DELIVERY_PROVENANCE_AUDIT_SCHEMA_ID",
    "GALAXY_DELIVERY_PROVENANCE_AUDIT_SCHEMA_VERSION",
    "GalaxyDeliveryProvenanceAuditError",
    "GalaxyDeliveryProvenanceAuditRequest",
    "GalaxyDeliveryProvenanceAuditResult",
    "audit_galaxy_delivery_provenance_v1",
    "main",
    "write_galaxy_delivery_provenance_audit_json",
]
