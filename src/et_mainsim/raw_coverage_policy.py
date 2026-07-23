"""Frozen policy contract for formal raw-10-s Galaxy coverage analysis.

The coverage-aware reducer deliberately has no hidden scientific defaults.
This module turns the owner-approved values into one immutable JSON artifact
bound to a particular formal Galaxy production manifest.  Per-source analyses
and the campaign summary consume the file identity, not a collection of
potentially drifting shell variables.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import argparse
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Literal

from .galaxy_stamp_production import (
    GALAXY_STAMP_PRODUCTION_SCHEMA_ID,
    GALAXY_STAMP_PRODUCTION_SCHEMA_VERSION,
)
from .stamp_inputs import file_identity


RAW_COVERAGE_POLICY_SCHEMA_ID = "et_mainsim.raw_coverage_aware_policy.v1"
RAW_COVERAGE_POLICY_SCHEMA_VERSION = 1
RAW_COVERAGE_ANALYSIS_PROFILE = "raw_10s_coverage_v2"
RAW_COVERAGE_STANDARD_WINDOWS_MINUTES = (30, 90, 390)
RAW_COVERAGE_INVALID_CADENCE_HANDLING = (
    "omit_whole_invalid_cadences_without_pixel_or_flux_imputation"
)
RAW_COVERAGE_ACCEPTED_BIN_NORMALIZATION = "actual_effective_exposure_only"
RAW_COVERAGE_STRICT_QUALITY_POLICY = "invalidate_whole_fixed_aperture_cadence"
RAW_COVERAGE_PRODUCT_FILENAME = "raw.h5"
RAW_COVERAGE_SECONDS = 10.0


class FrozenRawCoveragePolicyError(ValueError):
    """Raised when a formal coverage policy is malformed or unbound."""


def _json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FrozenRawCoveragePolicyError(f"{label} does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise FrozenRawCoveragePolicyError(f"{label} is not valid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise FrozenRawCoveragePolicyError(f"{label} must be a JSON object")
    return payload


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FrozenRawCoveragePolicyError(f"{label} must be an object")
    return value


def _require_exact_keys(
    value: Mapping[str, Any],
    *,
    expected: set[str],
    label: str,
) -> None:
    unknown = sorted(set(value) - expected)
    missing = sorted(expected - set(value))
    if unknown:
        raise FrozenRawCoveragePolicyError(
            f"{label} contains unknown field(s): {', '.join(unknown)}"
        )
    if missing:
        raise FrozenRawCoveragePolicyError(
            f"{label} is missing required field(s): {', '.join(missing)}"
        )


def _nonempty_text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FrozenRawCoveragePolicyError(f"{label} must be a non-empty string")
    return value.strip()


def _finite_float(
    value: Any,
    *,
    label: str,
    positive: bool = False,
    fraction: bool = False,
) -> float:
    if isinstance(value, bool):
        raise FrozenRawCoveragePolicyError(f"{label} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise FrozenRawCoveragePolicyError(f"{label} must be finite") from error
    if not math.isfinite(result):
        raise FrozenRawCoveragePolicyError(f"{label} must be finite")
    if positive and result <= 0.0:
        raise FrozenRawCoveragePolicyError(f"{label} must be positive")
    if fraction and not 0.0 < result <= 1.0:
        raise FrozenRawCoveragePolicyError(f"{label} must be in (0, 1]")
    return result


def _positive_int(value: Any, *, label: str, minimum: int = 1) -> int:
    if isinstance(value, bool):
        raise FrozenRawCoveragePolicyError(f"{label} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise FrozenRawCoveragePolicyError(f"{label} must be an integer") from error
    if result != value or result < minimum:
        raise FrozenRawCoveragePolicyError(
            f"{label} must be an integer at least {minimum}"
        )
    return result


def _strict_source_id(value: Any, *, label: str) -> int:
    result = _positive_int(value, label=label, minimum=0)
    if result > 2**63 - 1:
        raise FrozenRawCoveragePolicyError(f"{label} must fit signed int64")
    return result


def _identity_content_fields(value: Any, *, label: str) -> dict[str, Any]:
    mapping = _require_mapping(value, label=label)
    sha256 = mapping.get("sha256")
    size_bytes = mapping.get("size_bytes")
    if not isinstance(sha256, str) or len(sha256) != 64:
        raise FrozenRawCoveragePolicyError(f"{label}.sha256 must be a SHA-256 hex string")
    try:
        int(sha256, 16)
    except ValueError as error:
        raise FrozenRawCoveragePolicyError(
            f"{label}.sha256 must be a SHA-256 hex string"
        ) from error
    return {
        "sha256": sha256,
        "size_bytes": _positive_int(size_bytes, label=f"{label}.size_bytes", minimum=1),
    }


def _same_content_identity(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    return (
        actual.get("sha256") == expected.get("sha256")
        and actual.get("size_bytes") == expected.get("size_bytes")
    )


def _standard_windows(value: Any, *, label: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise FrozenRawCoveragePolicyError(f"{label} must be a list")
    windows = tuple(_positive_int(item, label=label) for item in value)
    if windows != RAW_COVERAGE_STANDARD_WINDOWS_MINUTES:
        raise FrozenRawCoveragePolicyError(
            f"{label} must be exactly [30, 90, 390]"
        )
    return windows


def _require_relative_within(path: Path, *, run_root: Path, label: str) -> str:
    if path.is_symlink():
        raise FrozenRawCoveragePolicyError(f"{label} must not be a symlink")
    try:
        return path.resolve().relative_to(run_root.resolve()).as_posix()
    except ValueError as error:
        raise FrozenRawCoveragePolicyError(
            f"{label} must be located within the formal run root"
        ) from error


@dataclass(frozen=True)
class FrozenRawCoveragePolicyRequest:
    """Owner-approved values required to create one formal policy artifact."""

    production_manifest_path: Path | str
    output_path: Path | str
    minimum_coverage_fraction: float
    minimum_accepted_bins: int
    windows_minutes: tuple[int, ...] = RAW_COVERAGE_STANDARD_WINDOWS_MINUTES
    bin_origin_seconds: float = 0.0
    case: Literal["injected"] | str = "injected"

    def __post_init__(self) -> None:
        manifest_path = Path(self.production_manifest_path).expanduser().resolve()
        output_path = Path(self.output_path).expanduser().resolve()
        windows = _standard_windows(list(self.windows_minutes), label="windows_minutes")
        coverage = _finite_float(
            self.minimum_coverage_fraction,
            label="minimum_coverage_fraction",
            fraction=True,
        )
        minimum_bins = _positive_int(
            self.minimum_accepted_bins,
            label="minimum_accepted_bins",
            minimum=2,
        )
        origin = _finite_float(self.bin_origin_seconds, label="bin_origin_seconds")
        if self.case != "injected":
            raise FrozenRawCoveragePolicyError("case must be exactly 'injected'")
        object.__setattr__(self, "production_manifest_path", manifest_path)
        object.__setattr__(self, "output_path", output_path)
        object.__setattr__(self, "windows_minutes", windows)
        object.__setattr__(self, "minimum_coverage_fraction", coverage)
        object.__setattr__(self, "minimum_accepted_bins", minimum_bins)
        object.__setattr__(self, "bin_origin_seconds", origin)
        object.__setattr__(self, "case", "injected")


@dataclass(frozen=True)
class FrozenRawCoveragePolicy:
    """Validated immutable raw-coverage policy plus its file content identity."""

    path: Path
    identity: Mapping[str, Any]
    schema_id: str
    schema_version: int
    run_id: str
    case: Literal["injected"]
    analysis_profile: str
    production_manifest_identity: Mapping[str, Any]
    windows_minutes: tuple[int, ...]
    minimum_coverage_fraction: float
    minimum_accepted_bins: int
    bin_origin_seconds: float

    @property
    def path_identity(self) -> Mapping[str, Any]:
        """Full local receipt including the mutable mount-specific path."""

        return self.identity

    @property
    def content_identity(self) -> dict[str, Any]:
        """Mount-independent SHA-256 and size identity suitable for manifests."""

        return _identity_content_fields(self.identity, label="policy identity")

    def coverage_policy_record(self) -> dict[str, Any]:
        return {
            "windows_minutes": list(self.windows_minutes),
            "minimum_coverage_fraction": self.minimum_coverage_fraction,
            "minimum_accepted_bins": self.minimum_accepted_bins,
            "bin_origin_seconds": self.bin_origin_seconds,
            "invalid_cadence_handling": RAW_COVERAGE_INVALID_CADENCE_HANDLING,
            "accepted_bin_normalization": RAW_COVERAGE_ACCEPTED_BIN_NORMALIZATION,
        }


def _load_formal_manifest(manifest_path: Path) -> tuple[dict[str, Any], Path, str]:
    manifest = _json_object(manifest_path, label="production manifest")
    if manifest.get("schema_id") != GALAXY_STAMP_PRODUCTION_SCHEMA_ID:
        raise FrozenRawCoveragePolicyError("unsupported Galaxy production manifest")
    if int(manifest.get("schema_version", 0)) != GALAXY_STAMP_PRODUCTION_SCHEMA_VERSION:
        raise FrozenRawCoveragePolicyError("unsupported Galaxy production manifest version")
    if manifest.get("observation_product") != "final_dn":
        raise FrozenRawCoveragePolicyError(
            "production manifest observation_product must be final_dn"
        )
    if manifest.get("background_realization_delivered") is not False:
        raise FrozenRawCoveragePolicyError(
            "production manifest must not deliver a background realization"
        )
    run_id = _nonempty_text(manifest.get("run_id"), label="production manifest run_id")
    delivery = _require_mapping(manifest.get("delivery"), label="production manifest delivery")
    raw_exposure = _finite_float(
        delivery.get("raw_exposure_seconds"),
        label="delivery.raw_exposure_seconds",
        positive=True,
    )
    if not math.isclose(raw_exposure, RAW_COVERAGE_SECONDS, abs_tol=1e-12):
        raise FrozenRawCoveragePolicyError(
            "formal raw coverage policy requires a 10-s raw exposure"
        )
    targets = manifest.get("targets")
    if not isinstance(targets, list) or len(targets) != 10:
        raise FrozenRawCoveragePolicyError(
            "formal raw coverage policy requires exactly 10 Galaxy targets"
        )
    source_ids = tuple(
        _strict_source_id(
            _require_mapping(target, label="production manifest target").get(
                "source_id_int64"
            ),
            label="target.source_id_int64",
        )
        for target in targets
    )
    if len(set(source_ids)) != len(source_ids):
        raise FrozenRawCoveragePolicyError("production manifest target IDs must be unique")
    return manifest, manifest_path.parent.resolve(), run_id


def _policy_payload(
    *,
    request: FrozenRawCoveragePolicyRequest,
    manifest: Mapping[str, Any],
    run_id: str,
) -> dict[str, Any]:
    return {
        "schema_id": RAW_COVERAGE_POLICY_SCHEMA_ID,
        "schema_version": RAW_COVERAGE_POLICY_SCHEMA_VERSION,
        "complete": True,
        "run_id": run_id,
        "case": request.case,
        "analysis_profile": RAW_COVERAGE_ANALYSIS_PROFILE,
        "production_manifest_identity": _identity_content_fields(
            file_identity(Path(request.production_manifest_path)),
            label="production manifest identity",
        ),
        "observation_product": "final_dn",
        "background_realization_used": False,
        "raw_input": {
            "product_filename": RAW_COVERAGE_PRODUCT_FILENAME,
            "cadence_seconds": RAW_COVERAGE_SECONDS,
            "raw_exposure_seconds": RAW_COVERAGE_SECONDS,
            "strict_quality_policy": RAW_COVERAGE_STRICT_QUALITY_POLICY,
        },
        "coverage": {
            "windows_minutes": list(request.windows_minutes),
            "minimum_coverage_fraction": request.minimum_coverage_fraction,
            "minimum_accepted_bins": request.minimum_accepted_bins,
            "bin_origin_seconds": request.bin_origin_seconds,
            "invalid_cadence_handling": RAW_COVERAGE_INVALID_CADENCE_HANDLING,
            "accepted_bin_normalization": RAW_COVERAGE_ACCEPTED_BIN_NORMALIZATION,
        },
    }


def _parse_policy(path: Path, *, identity: Mapping[str, Any] | None = None) -> FrozenRawCoveragePolicy:
    if not path.is_file() or path.is_symlink():
        raise FrozenRawCoveragePolicyError(
            f"frozen coverage policy must be a regular non-symlink file: {path}"
        )
    payload = _json_object(path, label="frozen coverage policy")
    _require_exact_keys(
        payload,
        expected={
            "schema_id",
            "schema_version",
            "complete",
            "run_id",
            "case",
            "analysis_profile",
            "production_manifest_identity",
            "observation_product",
            "background_realization_used",
            "raw_input",
            "coverage",
        },
        label="frozen coverage policy",
    )
    if payload.get("schema_id") != RAW_COVERAGE_POLICY_SCHEMA_ID:
        raise FrozenRawCoveragePolicyError("unsupported frozen coverage policy schema")
    if int(payload.get("schema_version", 0)) != RAW_COVERAGE_POLICY_SCHEMA_VERSION:
        raise FrozenRawCoveragePolicyError("unsupported frozen coverage policy version")
    if payload.get("complete") is not True:
        raise FrozenRawCoveragePolicyError("frozen coverage policy is not complete")
    run_id = _nonempty_text(payload.get("run_id"), label="policy.run_id")
    if payload.get("case") != "injected":
        raise FrozenRawCoveragePolicyError("policy.case must be exactly 'injected'")
    if payload.get("analysis_profile") != RAW_COVERAGE_ANALYSIS_PROFILE:
        raise FrozenRawCoveragePolicyError("policy analysis_profile is unsupported")
    if payload.get("observation_product") != "final_dn":
        raise FrozenRawCoveragePolicyError("policy observation_product must be final_dn")
    if payload.get("background_realization_used") is not False:
        raise FrozenRawCoveragePolicyError(
            "policy background_realization_used must be false"
        )
    production_identity = _identity_content_fields(
        payload.get("production_manifest_identity"),
        label="policy.production_manifest_identity",
    )
    raw_input = _require_mapping(payload.get("raw_input"), label="policy.raw_input")
    _require_exact_keys(
        raw_input,
        expected={
            "product_filename",
            "cadence_seconds",
            "raw_exposure_seconds",
            "strict_quality_policy",
        },
        label="policy.raw_input",
    )
    if raw_input.get("product_filename") != RAW_COVERAGE_PRODUCT_FILENAME:
        raise FrozenRawCoveragePolicyError("policy raw_input.product_filename must be raw.h5")
    for field in ("cadence_seconds", "raw_exposure_seconds"):
        seconds = _finite_float(raw_input.get(field), label=f"policy.raw_input.{field}")
        if not math.isclose(seconds, RAW_COVERAGE_SECONDS, abs_tol=1e-12):
            raise FrozenRawCoveragePolicyError(
                f"policy.raw_input.{field} must be 10 seconds"
            )
    if raw_input.get("strict_quality_policy") != RAW_COVERAGE_STRICT_QUALITY_POLICY:
        raise FrozenRawCoveragePolicyError("policy raw_input.strict_quality_policy is unsupported")
    coverage = _require_mapping(payload.get("coverage"), label="policy.coverage")
    _require_exact_keys(
        coverage,
        expected={
            "windows_minutes",
            "minimum_coverage_fraction",
            "minimum_accepted_bins",
            "bin_origin_seconds",
            "invalid_cadence_handling",
            "accepted_bin_normalization",
        },
        label="policy.coverage",
    )
    windows = _standard_windows(coverage.get("windows_minutes"), label="policy.coverage.windows_minutes")
    minimum_coverage = _finite_float(
        coverage.get("minimum_coverage_fraction"),
        label="policy.coverage.minimum_coverage_fraction",
        fraction=True,
    )
    minimum_bins = _positive_int(
        coverage.get("minimum_accepted_bins"),
        label="policy.coverage.minimum_accepted_bins",
        minimum=2,
    )
    origin = _finite_float(
        coverage.get("bin_origin_seconds"), label="policy.coverage.bin_origin_seconds"
    )
    if coverage.get("invalid_cadence_handling") != RAW_COVERAGE_INVALID_CADENCE_HANDLING:
        raise FrozenRawCoveragePolicyError(
            "policy coverage.invalid_cadence_handling is unsupported"
        )
    if coverage.get("accepted_bin_normalization") != RAW_COVERAGE_ACCEPTED_BIN_NORMALIZATION:
        raise FrozenRawCoveragePolicyError(
            "policy coverage.accepted_bin_normalization is unsupported"
        )
    actual_identity = file_identity(path) if identity is None else dict(identity)
    _identity_content_fields(actual_identity, label="frozen coverage policy identity")
    return FrozenRawCoveragePolicy(
        path=path,
        identity=actual_identity,
        schema_id=RAW_COVERAGE_POLICY_SCHEMA_ID,
        schema_version=RAW_COVERAGE_POLICY_SCHEMA_VERSION,
        run_id=run_id,
        case="injected",
        analysis_profile=RAW_COVERAGE_ANALYSIS_PROFILE,
        production_manifest_identity=production_identity,
        windows_minutes=windows,
        minimum_coverage_fraction=minimum_coverage,
        minimum_accepted_bins=minimum_bins,
        bin_origin_seconds=origin,
    )


def load_frozen_raw_coverage_policy_v1(path: Path | str) -> FrozenRawCoveragePolicy:
    """Load and strictly validate an already-published coverage policy."""

    policy_path = Path(path).expanduser().resolve()
    return _parse_policy(policy_path)


def validate_frozen_raw_coverage_policy_for_run_v1(
    policy: FrozenRawCoveragePolicy,
    *,
    production_manifest_path: Path | str,
    case: str,
) -> tuple[str, str]:
    """Bind a policy to an immutable formal run and return relative paths.

    The return values are the policy and production-manifest paths relative to
    the run root, so downstream manifests remain portable across mounts.
    """

    if not isinstance(policy, FrozenRawCoveragePolicy):
        raise TypeError("policy must be FrozenRawCoveragePolicy")
    if case != "injected":
        raise FrozenRawCoveragePolicyError("formal raw coverage only supports injected")
    manifest_path = Path(production_manifest_path).expanduser().resolve()
    manifest, run_root, run_id = _load_formal_manifest(manifest_path)
    del manifest
    if policy.run_id != run_id:
        raise FrozenRawCoveragePolicyError(
            "frozen coverage policy run_id does not match production manifest"
        )
    if not _same_content_identity(
        file_identity(manifest_path), policy.production_manifest_identity
    ):
        raise FrozenRawCoveragePolicyError(
            "frozen coverage policy binds a different production manifest"
        )
    policy_relative = _require_relative_within(
        policy.path, run_root=run_root, label="frozen coverage policy"
    )
    manifest_relative = _require_relative_within(
        manifest_path, run_root=run_root, label="production manifest"
    )
    return policy_relative, manifest_relative


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_frozen_raw_coverage_policy_v1(
    request: FrozenRawCoveragePolicyRequest,
) -> FrozenRawCoveragePolicy:
    """Atomically create one immutable, manifest-bound formal policy file."""

    if not isinstance(request, FrozenRawCoveragePolicyRequest):
        raise TypeError("request must be FrozenRawCoveragePolicyRequest")
    manifest, run_root, run_id = _load_formal_manifest(
        Path(request.production_manifest_path)
    )
    output_path = Path(request.output_path)
    _require_relative_within(output_path, run_root=run_root, label="policy output")
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"frozen coverage policy already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_path.parent / f".{output_path.name}.lock"
    try:
        lock_path.mkdir()
    except FileExistsError as error:
        raise FileExistsError(
            f"frozen coverage policy publication is already in progress: {output_path}"
        ) from error
    staging_path: Path | None = None
    try:
        if output_path.exists() or output_path.is_symlink():
            raise FileExistsError(f"frozen coverage policy already exists: {output_path}")
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output_path.name}.staging-", dir=output_path.parent
        )
        staging_path = Path(temporary_name)
        payload = _policy_payload(request=request, manifest=manifest, run_id=run_id)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # Validate the exact staged bytes before publication, then publish by a
        # same-filesystem rename while holding the sibling lock.
        _parse_policy(staging_path)
        os.replace(staging_path, output_path)
        _fsync_directory(output_path.parent)
        staging_path = None
    finally:
        if staging_path is not None:
            staging_path.unlink(missing_ok=True)
        lock_path.rmdir()
    policy = load_frozen_raw_coverage_policy_v1(output_path)
    validate_frozen_raw_coverage_policy_for_run_v1(
        policy,
        production_manifest_path=request.production_manifest_path,
        case=request.case,
    )
    return policy


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-manifest", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--minimum-coverage-fraction", required=True, type=float)
    parser.add_argument("--minimum-accepted-bins", required=True, type=int)
    parser.add_argument("--bin-origin-seconds", type=float, default=0.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Publish one owner-approved policy and print a compact receipt."""

    args = _parser().parse_args(None if argv is None else list(argv))
    policy = write_frozen_raw_coverage_policy_v1(
        FrozenRawCoveragePolicyRequest(
            production_manifest_path=args.production_manifest,
            output_path=args.output_path,
            minimum_coverage_fraction=args.minimum_coverage_fraction,
            minimum_accepted_bins=args.minimum_accepted_bins,
            bin_origin_seconds=args.bin_origin_seconds,
        )
    )
    print(
        json.dumps(
            {
                "policy_path": str(policy.path),
                "policy_identity": policy.content_identity,
                "run_id": policy.run_id,
                "minimum_coverage_fraction": policy.minimum_coverage_fraction,
                "minimum_accepted_bins": policy.minimum_accepted_bins,
                "windows_minutes": list(policy.windows_minutes),
                "bin_origin_seconds": policy.bin_origin_seconds,
            },
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI invocation only.
    raise SystemExit(main())


__all__ = [
    "FrozenRawCoveragePolicy",
    "FrozenRawCoveragePolicyError",
    "FrozenRawCoveragePolicyRequest",
    "RAW_COVERAGE_ACCEPTED_BIN_NORMALIZATION",
    "RAW_COVERAGE_ANALYSIS_PROFILE",
    "RAW_COVERAGE_INVALID_CADENCE_HANDLING",
    "RAW_COVERAGE_POLICY_SCHEMA_ID",
    "RAW_COVERAGE_POLICY_SCHEMA_VERSION",
    "RAW_COVERAGE_PRODUCT_FILENAME",
    "RAW_COVERAGE_SECONDS",
    "RAW_COVERAGE_STANDARD_WINDOWS_MINUTES",
    "RAW_COVERAGE_STRICT_QUALITY_POLICY",
    "load_frozen_raw_coverage_policy_v1",
    "main",
    "validate_frozen_raw_coverage_policy_for_run_v1",
    "write_frozen_raw_coverage_policy_v1",
]
