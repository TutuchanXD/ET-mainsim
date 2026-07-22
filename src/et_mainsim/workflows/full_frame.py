from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field, replace
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
from astropy import units as u

from et_mainsim.config import (
    ExecutionConfig,
    ResolvedRunPaths,
    RunConfig,
    SharedExposureStampsConfig,
    parse_frame_indices,
    worker_assignments,
)
from et_mainsim.manifest import RunManifestStore
from et_mainsim.presets import resource_path
from et_mainsim.provenance import collect_provenance
from et_mainsim.scope_artifacts import (
    FullFrameScopeArtifactContract,
    ScopeFrameCompletion,
)
from et_mainsim.selection_schemas import (
    is_supported_cadence_selection_truth_schema,
)


_SELECTION_TRUTH_SCOPE = "geometry_psf_and_jitter_selection_truth_only"
_ET_FULL_FRAME_SPACECRAFT_ID = "et"
_ET_FULL_FRAME_ABSOLUTE_RAW_FRAME_START_INDEX = 0
_SHARED_EXPOSURE_BATCH_KEY_SCHEMA_ID = "et_mainsim.shared_exposure_worker_batch_key.v1"


@dataclass(frozen=True)
class WorkerRequest:
    spec: Any
    execution: ExecutionConfig
    run_dir: Path
    data_root: Path
    catalog_cache: Path
    frame_indices: tuple[int, ...]
    shared_exposure_stamps: SharedExposureStampsConfig = field(
        default_factory=SharedExposureStampsConfig
    )
    shared_exposure_overwrite_prepared: bool = False
    rank: int = 0
    world_size: int = 1

    def __post_init__(self) -> None:
        if int(self.rank) < 0:
            raise ValueError("rank must be non-negative")
        if int(self.world_size) <= 0 or int(self.rank) >= int(self.world_size):
            raise ValueError("rank must be smaller than positive world_size")
        if not isinstance(
            self.shared_exposure_stamps,
            SharedExposureStampsConfig,
        ):
            raise TypeError(
                "shared_exposure_stamps must be a SharedExposureStampsConfig"
            )
        if not isinstance(self.shared_exposure_overwrite_prepared, bool):
            raise TypeError("shared_exposure_overwrite_prepared must be a boolean")
        object.__setattr__(self, "run_dir", Path(self.run_dir))
        object.__setattr__(self, "data_root", Path(self.data_root))
        object.__setattr__(self, "catalog_cache", Path(self.catalog_cache))
        object.__setattr__(
            self,
            "frame_indices",
            tuple(int(value) for value in self.frame_indices),
        )
        object.__setattr__(self, "rank", int(self.rank))
        object.__setattr__(self, "world_size", int(self.world_size))

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "schema_id": "et_mainsim.full_frame_worker_request",
            "schema_version": 2,
            "simulation_spec": self.spec.to_json_dict(),
            "execution": self.execution.to_dict(),
            "run_dir": str(self.run_dir),
            "data_root": str(self.data_root),
            "catalog_cache": str(self.catalog_cache),
            "frame_indices": list(self.frame_indices),
            "shared_exposure_stamps": self.shared_exposure_stamps.to_dict(),
            "shared_exposure_overwrite_prepared": (
                self.shared_exposure_overwrite_prepared
            ),
            "rank": self.rank,
            "world_size": self.world_size,
        }

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, Any]) -> "WorkerRequest":
        if payload.get("schema_id") != "et_mainsim.full_frame_worker_request":
            raise ValueError("Unsupported full-frame worker request")
        if payload.get("schema_version") != 2:
            raise ValueError("Unsupported full-frame worker request version")
        from photsim7.specs import SimulationSpec

        return cls(
            spec=SimulationSpec.from_json_dict(payload["simulation_spec"]),
            execution=ExecutionConfig(**dict(payload["execution"])),
            run_dir=Path(payload["run_dir"]),
            data_root=Path(payload["data_root"]),
            catalog_cache=Path(payload["catalog_cache"]),
            frame_indices=tuple(payload["frame_indices"]),
            shared_exposure_stamps=SharedExposureStampsConfig(
                **dict(payload["shared_exposure_stamps"])
            ),
            shared_exposure_overwrite_prepared=payload.get(
                "shared_exposure_overwrite_prepared",
                False,
            ),
            rank=int(payload["rank"]),
            world_size=int(payload["world_size"]),
        )


@dataclass(frozen=True)
class WorkerResult:
    rank: int
    rendered: tuple[int, ...]
    skipped: tuple[int, ...]
    elapsed_s: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "rendered": list(self.rendered),
            "skipped": list(self.skipped),
            "elapsed_s": self.elapsed_s,
        }


@dataclass(frozen=True)
class FullFrameRunPlan:
    preset_name: str
    run_config: RunConfig
    paths: ResolvedRunPaths
    spec: Any
    run_dir: Path
    catalog_cache: Path
    frame_indices: tuple[int, ...]
    repo_root: Path

    def to_dict(self, *, dry_run: bool) -> dict[str, Any]:
        return {
            "dry_run": bool(dry_run),
            "workflow": "et-full-frame",
            "preset": self.preset_name,
            "run_id": self.run_config.run_id,
            "run_dir": str(self.run_dir),
            "catalog_cache": str(self.catalog_cache),
            "paths": self.paths.to_dict(),
            "execution": self.run_config.execution.to_dict(),
            "workload": _full_frame_workload_identity(self),
            "frame_plan": {
                "requested": list(self.frame_indices),
                "count": len(self.frame_indices),
            },
            "simulation_spec": self.spec.to_json_dict(),
        }


def _science_api() -> SimpleNamespace:
    from photsim7.artifacts import (
        ItemStatus,
        SharedExposureShardReader,
        SharedExposureShardWriter,
        partial_shard_path,
    )
    from photsim7.catalog_sources import PreparedStarCatalog, StarCatalogCache
    from photsim7.data_registry import DataRegistry
    from photsim7.full_frame_artifacts import (
        FullFrameArtifactOptions,
        FullFrameArtifactWriter,
    )
    from photsim7.full_frame_pipeline import (
        iter_single_cadence_full_frame_scopes,
        run_single_cadence_full_frame,
    )
    from photsim7.selection_artifacts import (
        cadence_selection_truth_relative_path,
        read_cadence_selection_truth,
    )
    from photsim7.shared_exposure import (
        SharedExposureTargetIdentity,
        shared_exposure_crop_v1,
    )
    from photsim7.simulation_services import (
        build_catalog_from_spec,
        build_full_frame_services,
        build_multiscope_full_frame_services,
        resolve_full_frame_source_pixel_geometry,
    )
    from photsim7.stamp_products import StampWindow

    return SimpleNamespace(
        PreparedStarCatalog=PreparedStarCatalog,
        StarCatalogCache=StarCatalogCache,
        DataRegistry=DataRegistry,
        FullFrameArtifactOptions=FullFrameArtifactOptions,
        FullFrameArtifactWriter=FullFrameArtifactWriter,
        ItemStatus=ItemStatus,
        SharedExposureShardReader=SharedExposureShardReader,
        SharedExposureShardWriter=SharedExposureShardWriter,
        SharedExposureTargetIdentity=SharedExposureTargetIdentity,
        StampWindow=StampWindow,
        build_catalog_from_spec=build_catalog_from_spec,
        build_full_frame_services=build_full_frame_services,
        build_multiscope_full_frame_services=(
            build_multiscope_full_frame_services
        ),
        iter_single_cadence_full_frame_scopes=(
            iter_single_cadence_full_frame_scopes
        ),
        resolve_full_frame_source_pixel_geometry=(
            resolve_full_frame_source_pixel_geometry
        ),
        run_single_cadence_full_frame=run_single_cadence_full_frame,
        shared_exposure_crop_v1=shared_exposure_crop_v1,
        partial_shard_path=partial_shard_path,
        cadence_selection_truth_relative_path=(cadence_selection_truth_relative_path),
        read_cadence_selection_truth=read_cadence_selection_truth,
    )


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    from et_mainsim.manifest import _atomic_write_json

    _atomic_write_json(path, payload)


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _artifact_paths(run_dir: Path, frame_index: int) -> tuple[Path, Path, Path]:
    stem = f"frame_{int(frame_index):06d}"
    return (
        run_dir / "frames" / f"{stem}.npy",
        run_dir / "frame_summaries" / f"{stem}.json",
        run_dir / "frame_summaries" / f"{stem}_schema.json",
    )


def _scope_contract_for_spec(
    run_dir: Path | str,
    spec: Any,
) -> FullFrameScopeArtifactContract:
    return FullFrameScopeArtifactContract.from_telescope_count(
        run_dir,
        telescope_count=int(spec.instrument.telescope_count),
    )


def _schema_scope_id(schema: Mapping[str, Any]) -> int | None:
    """Return the persisted service scope identity, if this product has one."""

    provenance = schema.get("provenance")
    if not isinstance(provenance, Mapping):
        return None
    services = provenance.get("services")
    if not isinstance(services, Mapping):
        return None
    scope = services.get("scope")
    if not isinstance(scope, Mapping):
        return None
    scope_id = scope.get("scope_id")
    if isinstance(scope_id, bool) or not isinstance(scope_id, int):
        return None
    if int(scope_id) < 0:
        return None
    return int(scope_id)


def _shared_exposure_root(run_dir: Path) -> Path:
    return run_dir / "shared_exposure"


def _shared_exposure_scope_root(
    run_dir: Path,
    *,
    scope_id: int = 0,
    scope_contract: FullFrameScopeArtifactContract | None = None,
) -> Path:
    """Return the scope-local shared-exposure root.

    The legacy single-scope layout remains byte-for-byte path compatible.
    Six scopes receive an explicit storage namespace so a crop shard can never
    be mistaken for an image-level combination.
    """

    normalized_scope_id = int(scope_id)
    if scope_contract is None or scope_contract.is_single_scope:
        if normalized_scope_id != 0:
            raise ValueError("single-scope shared exposure only accepts scope_id=0")
        return _shared_exposure_root(run_dir)
    scope_contract.scope_root(normalized_scope_id)
    return _shared_exposure_root(run_dir) / f"scope_{normalized_scope_id}"


def _clear_shared_exposure_bundle_for_overwrite(run_dir: Path) -> None:
    """Remove exactly the coordinator-owned shared-exposure bundle."""

    shared_root = _shared_exposure_root(run_dir)
    if shared_root.is_symlink() or shared_root.is_file():
        shared_root.unlink()
    elif shared_root.exists():
        shutil.rmtree(shared_root)
    if os.path.lexists(shared_root):
        raise RuntimeError(
            f"shared-exposure overwrite cleanup did not remove {shared_root}"
        )


def _shared_exposure_plan_path(run_dir: Path) -> Path:
    return _shared_exposure_root(run_dir) / "target_plan.json"


def _shared_exposure_completion_path(
    run_dir: Path,
    frame_index: int,
    *,
    scope_id: int = 0,
    scope_contract: FullFrameScopeArtifactContract | None = None,
) -> Path:
    return (
        _shared_exposure_scope_root(
            run_dir,
            scope_id=scope_id,
            scope_contract=scope_contract,
        )
        / "completion"
        / f"frame_{int(frame_index):09d}.json"
    )


def _shared_exposure_shard_root(
    run_dir: Path,
    rank: int,
    *,
    scope_id: int = 0,
    scope_contract: FullFrameScopeArtifactContract | None = None,
) -> Path:
    return (
        _shared_exposure_scope_root(
            run_dir,
            scope_id=scope_id,
            scope_contract=scope_contract,
        )
        / "shards"
        / f"worker_{int(rank):04d}"
    )


@dataclass(frozen=True)
class _SharedExposureFrameBatch:
    batch_index: int
    frame_ids: tuple[int, ...]
    content_sha256: str
    root: Path
    scope_id: int = 0
    scope_explicit: bool = False

    @property
    def case_id(self) -> str:
        worker_rank = self.root.parent.name.removeprefix("worker_")
        if self.scope_explicit:
            return (
                f"full-frame-scope-{self.scope_id}-worker-{worker_rank}-"
                f"batch-{self.batch_index:06d}"
            )
        return f"full-frame-worker-{worker_rank}-batch-{self.batch_index:06d}"


def _canonical_shared_exposure_batch_key(
    *,
    rank: int,
    world_size: int,
    batch_index: int,
    frames_per_shard: int,
    frame_ids: tuple[int, ...],
    scope_id: int | None = None,
) -> tuple[dict[str, Any], str]:
    payload = {
        "schema_id": _SHARED_EXPOSURE_BATCH_KEY_SCHEMA_ID,
        "rank": int(rank),
        "world_size": int(world_size),
        "batch_index": int(batch_index),
        "frames_per_shard": int(frames_per_shard),
        "frame_ids": [int(frame_index) for frame_index in frame_ids],
    }
    if scope_id is not None:
        payload["scope_id"] = int(scope_id)
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return payload, sha256(encoded).hexdigest()


def _shared_exposure_frame_batches(
    request: WorkerRequest,
    assigned: tuple[int, ...],
    *,
    scope_id: int = 0,
    scope_contract: FullFrameScopeArtifactContract | None = None,
) -> tuple[_SharedExposureFrameBatch, ...]:
    frames_per_shard = request.shared_exposure_stamps.frames_per_shard
    scope_explicit = scope_contract is not None and not scope_contract.is_single_scope
    worker_root = _shared_exposure_shard_root(
        request.run_dir,
        request.rank,
        scope_id=scope_id,
        scope_contract=scope_contract,
    )
    batches: list[_SharedExposureFrameBatch] = []
    for start in range(0, len(assigned), frames_per_shard):
        frame_ids = assigned[start : start + frames_per_shard]
        batch_index = len(batches)
        _, content_sha256 = _canonical_shared_exposure_batch_key(
            rank=request.rank,
            world_size=request.world_size,
            batch_index=batch_index,
            frames_per_shard=frames_per_shard,
            frame_ids=frame_ids,
            scope_id=int(scope_id) if scope_explicit else None,
        )
        batches.append(
            _SharedExposureFrameBatch(
                batch_index=batch_index,
                frame_ids=frame_ids,
                content_sha256=content_sha256,
                root=(worker_root / f"batch_{batch_index:06d}_{content_sha256}"),
                scope_id=int(scope_id),
                scope_explicit=scope_explicit,
            )
        )
    return tuple(batches)


def _shared_exposure_batch_by_frame(
    batches: tuple[_SharedExposureFrameBatch, ...],
) -> dict[int, _SharedExposureFrameBatch]:
    return {frame_index: batch for batch in batches for frame_index in batch.frame_ids}


def _shared_exposure_batch_shard_paths(
    batch: _SharedExposureFrameBatch,
    *,
    plan_content_sha256: str,
    product_keys: tuple[str, ...],
) -> dict[str, Path]:
    from et_mainsim.shared_exposure import shared_exposure_product_shard_path

    return {
        product_key: shared_exposure_product_shard_path(
            batch.root,
            plan_content_sha256=plan_content_sha256,
            product_key=product_key,
        )
        for product_key in product_keys
    }


def _validate_shared_exposure_plan_for_request(
    plan: Mapping[str, Any],
    *,
    request: WorkerRequest,
) -> None:
    shared = request.shared_exposure_stamps
    expected_detector = {
        "detector_id": str(request.spec.detector.detector_id),
        "shape": [int(value) for value in request.spec.detector.shape],
    }
    if plan.get("detector") != expected_detector:
        raise RuntimeError(
            "shared-exposure target plan detector identity conflicts with request"
        )
    if plan.get("stamp_shape") != list(shared.stamp_shape):
        raise RuntimeError(
            "shared-exposure target plan stamp shape conflicts with request"
        )
    targets = plan.get("targets")
    if not isinstance(targets, list):
        raise RuntimeError("shared-exposure target plan targets are invalid")
    try:
        target_ids = tuple(int(target["source_id"]) for target in targets)
    except (TypeError, ValueError, KeyError) as exc:
        raise RuntimeError(
            "shared-exposure target plan target identities are invalid"
        ) from exc
    if target_ids != shared.target_source_ids:
        raise RuntimeError(
            "shared-exposure target plan source order conflicts with request"
        )


def _shared_exposure_windows(plan: Mapping[str, Any], *, api: Any) -> dict[int, Any]:
    windows: dict[int, Any] = {}
    for target in plan["targets"]:
        source_id = int(target["source_id"])
        schema = target["window"]
        window = api.StampWindow(
            x_start_detector_pix=schema["x_start_detector_pix"],
            y_start_detector_pix=schema["y_start_detector_pix"],
            shape=tuple(schema["shape"]),
            detector_shape=tuple(schema["detector_shape"]),
            target_x_detector_pix=schema["target_x_detector_pix"],
            target_y_detector_pix=schema["target_y_detector_pix"],
        )
        if window.to_schema() != schema:
            raise RuntimeError(
                f"shared-exposure target plan window {source_id} is non-canonical"
            )
        windows[source_id] = window
    return windows


def _shared_exposure_crop_product(
    crop: Any,
    product_key: str,
) -> tuple[np.ndarray, str, str]:
    direct = {
        "final_stamp": "final_stamp",
        "electron_stamp": "electron_stamp",
        "adu_stamp_pre_adc": "adu_stamp_pre_adc",
        "dn_stamp": "dn_stamp",
    }
    if product_key in direct:
        product = getattr(crop, direct[product_key])
        if product is None:
            raise RuntimeError(f"shared-exposure crop does not provide {product_key}")
        return _as_numpy(product.array), str(product.unit), str(product.domain)
    if product_key.startswith("electron_components."):
        component_name = product_key.split(".", 1)[1]
        product = crop.electron_components.get(component_name)
        if product is None:
            raise RuntimeError(f"shared-exposure crop does not provide {product_key}")
        return _as_numpy(product.array), str(product.unit), str(product.domain)
    if product_key == "cosmic_events.mask":
        cosmic = crop.cosmic_events
        if cosmic is None or cosmic.mask is None:
            raise RuntimeError(
                "shared-exposure crop does not provide cosmic_events.mask"
            )
        return _as_numpy(cosmic.mask), "bool", "detector_footprint"
    raise AssertionError(
        f"validated shared-exposure product is unsupported: {product_key}"
    )


def _selection_artifact_identity(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} selection artifact identity must be a mapping")
    result = dict(value)
    if set(result) != {
        "relative_path",
        "schema_id",
        "schema_version",
        "content_sha256",
    }:
        raise ValueError(f"{label} selection artifact identity keys are invalid")
    relative = Path(str(result["relative_path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} selection artifact path escapes run_dir")
    digest = str(result["content_sha256"])
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{label} selection artifact hash is invalid")
    schema_id = str(result["schema_id"])
    if not schema_id or schema_id != schema_id.strip():
        raise ValueError(f"{label} selection artifact schema_id is invalid")
    schema_version = int(result["schema_version"])
    if schema_version < 1:
        raise ValueError(f"{label} selection artifact schema version is invalid")
    return {
        "relative_path": relative.as_posix(),
        "schema_id": schema_id,
        "schema_version": schema_version,
        "content_sha256": digest,
    }


def _unavailable_selection_is_complete(
    selection: Mapping[str, Any],
    *,
    expected_spec: Any,
) -> bool:
    if set(selection) != {
        "schema_id",
        "schema_version",
        "verification_status",
        "science_conformance_claim",
        "science_conformance_claim_scope",
        "requested_science_profile_id",
        "missing_components",
    }:
        return False
    return bool(
        is_supported_cadence_selection_truth_schema(
            selection.get("schema_id"),
            selection.get("schema_version"),
        )
        and selection.get("verification_status") == "unavailable"
        and selection.get("science_conformance_claim") is False
        and selection.get("science_conformance_claim_scope") == _SELECTION_TRUTH_SCOPE
        and selection.get("requested_science_profile_id")
        == expected_spec.science_profile.profile_id
        and selection.get("missing_components") == ["jitter_model_selection_truth"]
        and not bool(expected_spec.psf.use_jitter_integrated_psf)
    )


def _persisted_selection_is_complete(
    run_dir: Path,
    frame_index: int,
    selection: Mapping[str, Any],
    *,
    expected_spec: Any,
    api: Any,
) -> bool:
    if set(selection) != {
        "schema_id",
        "schema_version",
        "verification_status",
        "science_conformance_claim",
        "science_conformance_claim_scope",
        "requested_science_profile_id",
        "content_sha256",
        "source_geometry_truth",
        "psf_selection_truth",
        "jitter_model_selection_truth",
        "missing_components",
        "artifact",
    }:
        return False
    if (
        not is_supported_cadence_selection_truth_schema(
            selection.get("schema_id"),
            selection.get("schema_version"),
        )
        or selection.get("verification_status") != "persisted_and_verified"
        or selection.get("science_conformance_claim_scope") != _SELECTION_TRUTH_SCOPE
        or selection.get("requested_science_profile_id")
        != expected_spec.science_profile.profile_id
        or selection.get("missing_components") != []
        or not isinstance(selection.get("science_conformance_claim"), bool)
    ):
        return False
    artifacts = selection.get("artifact")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {
        "geometry",
        "psf",
        "cadence",
    }:
        return False
    geometry_artifact = _selection_artifact_identity(
        artifacts["geometry"],
        label="geometry",
    )
    psf_artifact = _selection_artifact_identity(
        artifacts["psf"],
        label="PSF",
    )
    cadence_artifact = _selection_artifact_identity(
        artifacts["cadence"],
        label="cadence",
    )
    absolute_raw_frame_index = _ET_FULL_FRAME_ABSOLUTE_RAW_FRAME_START_INDEX + int(
        frame_index
    )
    cadence_relative = api.cadence_selection_truth_relative_path(
        absolute_raw_frame_index
    ).as_posix()
    if cadence_artifact["relative_path"] != cadence_relative:
        return False
    content_sha256 = str(selection.get("content_sha256", ""))
    if cadence_artifact["content_sha256"] != content_sha256:
        return False
    truth = api.read_cadence_selection_truth(
        run_dir / cadence_relative,
        artifact_root=run_dir,
        expected_sha256=content_sha256,
    )
    try:
        truth.jitter_model_selection_truth.rng_trace_payload(
            expected_spec.rng.to_seed_tree()
        )
    except (AttributeError, TypeError, ValueError):
        return False
    if (
        truth.schema_id != selection["schema_id"]
        or truth.schema_version != int(selection["schema_version"])
        or truth.detector_id != str(expected_spec.detector.detector_id)
        or truth.local_frame_index != int(frame_index)
        or truth.absolute_raw_frame_index != absolute_raw_frame_index
        or truth.spacecraft_id != _ET_FULL_FRAME_SPACECRAFT_ID
        or truth.science_realization_id
        != int(expected_spec.science_profile.science_realization_id)
        or truth.science_conformance_claim is not selection["science_conformance_claim"]
    ):
        return False
    if selection["source_geometry_truth"] != truth.geometry_reference:
        return False
    if selection["psf_selection_truth"] != truth.psf_reference:
        return False
    if (
        selection["jitter_model_selection_truth"]
        != truth.jitter_model_selection_truth.to_json_dict()
    ):
        return False
    for artifact, reference in (
        (geometry_artifact, truth.geometry_reference),
        (psf_artifact, truth.psf_reference),
    ):
        for field_name in (
            "relative_path",
            "schema_id",
            "schema_version",
            "content_sha256",
        ):
            if artifact[field_name] != reference[field_name]:
                return False
    return bool(
        cadence_artifact["schema_id"] == truth.schema_id
        and cadence_artifact["schema_version"] == truth.schema_version
        and cadence_artifact["content_sha256"] == truth.content_sha256
    )


def _selection_is_complete(
    run_dir: Path,
    frame_index: int,
    schema: Mapping[str, Any],
    *,
    expected_spec: Any,
) -> bool:
    selection = schema.get("selection_truth")
    if not isinstance(selection, Mapping):
        return False
    if selection.get("verification_status") == "unavailable":
        return _unavailable_selection_is_complete(
            selection,
            expected_spec=expected_spec,
        )
    return _persisted_selection_is_complete(
        run_dir,
        frame_index,
        selection,
        expected_spec=expected_spec,
        api=_science_api(),
    )


def frame_is_complete(
    run_dir: Path | str,
    frame_index: int,
    *,
    expected_shape: tuple[int, int],
    expected_spec: Any,
    expected_scope_id: int = 0,
    require_scope_identity: bool = False,
) -> bool:
    run_dir = Path(run_dir)
    frame_path, summary_path, schema_path = _artifact_paths(run_dir, frame_index)
    if not all(path.is_file() for path in (frame_path, summary_path, schema_path)):
        return False
    try:
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        if int(summary.get("artifact_schema_version", 0)) != 1:
            return False
        if int(summary.get("frame_index", -1)) != int(frame_index):
            return False

        from photsim7.frame_products import read_frame_product_schema

        schema = read_frame_product_schema(schema_path)
        if int(schema.get("frame_index", -1)) != int(frame_index):
            return False
        if schema.get("detector_id") != str(expected_spec.detector.detector_id):
            return False
        observed_scope_id = _schema_scope_id(schema)
        if require_scope_identity and observed_scope_id != int(expected_scope_id):
            return False
        if (
            not require_scope_identity
            and observed_scope_id is not None
            and observed_scope_id != int(expected_scope_id)
        ):
            return False
        array = np.load(frame_path, mmap_mode="r", allow_pickle=False)
        if tuple(array.shape) != tuple(expected_shape):
            return False
        final_schema = schema.get("arrays", {}).get("final_frame", {})
        if tuple(final_schema.get("shape", ())) != tuple(expected_shape):
            return False
        if str(final_schema.get("dtype")) != str(array.dtype):
            return False
        if not _selection_is_complete(
            run_dir,
            frame_index,
            schema,
            expected_spec=expected_spec,
        ):
            return False
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False
    return True


def frame_completion(
    run_dir: Path | str,
    frame_index: int,
    *,
    expected_shape: tuple[int, int],
    expected_spec: Any,
) -> ScopeFrameCompletion:
    """Validate every scope product required for one logical cadence.

    Single-scope products retain the pre-scope root layout and accept legacy
    schemas without a scope provenance block.  A six-scope cadence only
    completes when every scope-local product carries its matching explicit
    ``provenance.services.scope.scope_id`` value.
    """

    contract = _scope_contract_for_spec(run_dir, expected_spec)
    return contract.frame_completion(
        frame_index=frame_index,
        scope_is_complete=lambda paths: frame_is_complete(
            contract.scope_root(paths.scope_id),
            frame_index,
            expected_shape=expected_shape,
            expected_spec=expected_spec,
            expected_scope_id=paths.scope_id,
            require_scope_identity=not contract.is_single_scope,
        ),
    )


def _has_partial_artifacts(run_dir: Path, frame_index: int) -> bool:
    absolute_raw_frame_index = _ET_FULL_FRAME_ABSOLUTE_RAW_FRAME_START_INDEX + int(
        frame_index
    )
    cadence_path = (
        run_dir
        / "selection_truth"
        / "cadence"
        / f"frame_{absolute_raw_frame_index:09d}.json"
    )
    return any(
        path.exists() for path in (*_artifact_paths(run_dir, frame_index), cadence_path)
    )


def _has_partial_scope_artifacts(
    contract: FullFrameScopeArtifactContract,
    frame_index: int,
) -> bool:
    return any(
        _has_partial_artifacts(contract.scope_root(scope_id), frame_index)
        for scope_id in contract.scope_ids
    )


def _selection_artifact_metadata(artifacts: Any) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "relative_path": identity.relative_path,
            "schema_id": identity.schema_id,
            "schema_version": identity.schema_version,
            "content_sha256": identity.content_sha256,
        }
        for name, identity in (
            ("geometry", artifacts.geometry),
            ("psf", artifacts.psf),
            ("cadence", artifacts.cadence),
        )
    }


def _persisted_selection_metadata(
    spec: Any,
    truth: Any,
    artifacts: Any,
) -> dict[str, Any]:
    """Rebuild Photsim7's public persisted-selection metadata contract."""

    return {
        "schema_id": truth.schema_id,
        "schema_version": truth.schema_version,
        "verification_status": "persisted_and_verified",
        "science_conformance_claim": truth.science_conformance_claim,
        "science_conformance_claim_scope": _SELECTION_TRUTH_SCOPE,
        "requested_science_profile_id": spec.science_profile.profile_id,
        "content_sha256": truth.content_sha256,
        "source_geometry_truth": truth.geometry_reference,
        "psf_selection_truth": truth.psf_reference,
        "jitter_model_selection_truth": (
            truth.jitter_model_selection_truth.to_json_dict()
        ),
        "missing_components": [],
        "artifact": _selection_artifact_metadata(artifacts),
    }


def _persist_scope_frame_result(
    *,
    writer: Any,
    result: Any,
    spec: Any,
) -> None:
    """Write one iterator result into its scope-local artifact root.

    The iterator intentionally has no writer argument, so this application
    layer publishes the same public FullFrameArtifactWriter products after it
    has received one scope result.  This keeps the production path bounded to
    one rendered full frame at a time.
    """

    product = result.frame_products
    selection_truth = getattr(result, "selection_truth", None)
    if selection_truth is not None:
        artifacts = writer.write_selection_truth(selection_truth)
        persisted_selection = _persisted_selection_metadata(
            spec,
            selection_truth,
            artifacts,
        )
        provenance = dict(product.provenance or {})
        provenance["selection_truth"] = persisted_selection
        product = replace(
            product,
            selection_truth=persisted_selection,
            provenance=provenance,
        )
    detector_result = result.detector_result
    writer.write_frame(
        product.frame_index,
        _as_numpy(product.final_frame.array),
        summary=product.frame_summary,
        cosmic_events=getattr(detector_result, "cosmic_metadata", None),
        column_noise_adu=(
            None
            if getattr(detector_result, "bias_metadata", None) is None
            else detector_result.bias_metadata.column_noise_vector_adu
        ),
    )
    writer.write_frame_product_schema(product)


def _scope_frame_is_complete(
    contract: FullFrameScopeArtifactContract,
    *,
    scope_id: int,
    frame_index: int,
    expected_shape: tuple[int, int],
    expected_spec: Any,
) -> bool:
    return frame_is_complete(
        contract.scope_root(scope_id),
        frame_index,
        expected_shape=expected_shape,
        expected_spec=expected_spec,
        expected_scope_id=scope_id,
        require_scope_identity=not contract.is_single_scope,
    )


def _full_frame_renderer_options(request: WorkerRequest) -> dict[str, Any]:
    return {
        "enable_stellar_photon_noise": True,
        "enable_background_light": True,
        "enable_scattered_light": bool(
            request.spec.sky.scattered_light.to_value(u.electron / u.s / u.pix)
        ),
        "enable_dark_current": True,
        "progress": request.execution.progress,
    }


def _render_multiscope_frame(
    *,
    request: WorkerRequest,
    api: Any,
    services: Any,
    contract: FullFrameScopeArtifactContract,
    catalog: Any,
    frame_index: int,
    expected_shape: tuple[int, int],
) -> None:
    """Render and persist one logical cadence scope-by-scope.

    ``iter_single_cadence_full_frame_scopes`` yields one product at a time;
    this function writes or validates that scope before requesting the next.
    It deliberately has no shared-exposure crop handling.
    """

    if contract.is_single_scope:
        raise ValueError("multiscope rendering requires a six-scope contract")
    if request.shared_exposure_stamps.enabled:
        raise NotImplementedError(
            "shared-exposure target crops are not implemented for six scopes"
        )

    if request.execution.overwrite:
        absolute_raw_frame_index = (
            _ET_FULL_FRAME_ABSOLUTE_RAW_FRAME_START_INDEX + int(frame_index)
        )
        for scope_id in contract.scope_ids:
            cadence_path = contract.scope_root(scope_id) / api.cadence_selection_truth_relative_path(
                absolute_raw_frame_index
            )
            cadence_path.unlink(missing_ok=True)

    torch = None
    if request.execution.device == "cuda":
        import torch as torch_module

        torch = torch_module
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA execution requested but torch reports no CUDA device")
        torch.cuda.reset_peak_memory_stats()

    iterator = iter(
        api.iter_single_cadence_full_frame_scopes(
            request.spec,
            services=services,
            frame_index=frame_index,
            renderer_options=_full_frame_renderer_options(request),
            worker_rank=request.rank,
            rng_trace_scope={"run_id": request.run_dir.name},
        )
    )
    for expected_scope_id in contract.scope_ids:
        scope_started = time.perf_counter()
        try:
            scope_id, result = next(iterator)
        except StopIteration as error:
            raise RuntimeError(
                "Photsim7 multiscope iterator ended before every scope was rendered"
            ) from error
        if int(scope_id) != expected_scope_id:
            raise RuntimeError(
                "Photsim7 multiscope iterator returned scope ids out of canonical "
                f"order: expected {expected_scope_id}, received {scope_id}"
            )
        if torch is not None:
            torch.cuda.synchronize()
        pipeline_elapsed_s = time.perf_counter() - scope_started
        scope_root = contract.scope_root(scope_id)
        scope_requires_persist = not _scope_frame_is_complete(
            contract,
            scope_id=scope_id,
            frame_index=frame_index,
            expected_shape=expected_shape,
            expected_spec=request.spec,
        )
        if scope_requires_persist:
            options = api.FullFrameArtifactOptions(
                save_frame_summaries=True,
                save_cosmic_events=True,
                save_bias=bool(request.spec.artifacts.save_bias_artifacts),
                save_preview=frame_index < request.execution.preview_count,
            )
            writer = api.FullFrameArtifactWriter(scope_root, options=options)
            _persist_scope_frame_result(
                writer=writer,
                result=result,
                spec=request.spec,
            )
            if not _scope_frame_is_complete(
                contract,
                scope_id=scope_id,
                frame_index=frame_index,
                expected_shape=expected_shape,
                expected_spec=request.spec,
            ):
                raise RuntimeError(
                    "Photsim7 scope artifacts failed readback validation for "
                    f"frame {frame_index}, scope {scope_id}"
                )
        if scope_requires_persist:
            if request.execution.save_cosmic_mask:
                cosmic = getattr(result.detector_result, "cosmic_metadata", None)
                mask = None if cosmic is None else getattr(cosmic, "mask", None)
                if mask is not None:
                    mask_array = _as_numpy(mask)
                    if mask_array.ndim == 3 and mask_array.shape[0] == 1:
                        mask_array = mask_array[0]
                    mask_path = (
                        scope_root
                        / "cosmic_events"
                        / f"frame_{frame_index:06d}_mask.npy"
                    )
                    mask_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(mask_path, mask_array)
            if request.execution.save_stellar_mean:
                stellar_mean = result.renderer_components.get("stellar_mean")
                if stellar_mean is None:
                    raise KeyError("Photsim7 did not return stellar_mean")
                np.save(
                    scope_root
                    / "frames"
                    / f"frame_{frame_index:06d}_stellar_mean_e.npy",
                    _as_numpy(stellar_mean).astype(np.float32),
                )
            peak_cuda_allocated_mb = (
                None
                if torch is None
                else float(torch.cuda.max_memory_allocated() / 1024**2)
            )
            peak_cuda_reserved_mb = (
                None
                if torch is None
                else float(torch.cuda.max_memory_reserved() / 1024**2)
            )
            _record_frame_metrics(
                scope_root,
                frame_index,
                rank=request.rank,
                device=request.execution.device,
                n_stars=int(catalog.n_sources),
                pipeline_elapsed_s=pipeline_elapsed_s,
                total_elapsed_s=time.perf_counter() - scope_started,
                peak_cuda_allocated_mb=peak_cuda_allocated_mb,
                peak_cuda_reserved_mb=peak_cuda_reserved_mb,
            )
        del result
    try:
        unexpected_scope_id, _ = next(iterator)
    except StopIteration:
        pass
    else:
        raise RuntimeError(
            "Photsim7 multiscope iterator returned an unexpected extra scope "
            f"{unexpected_scope_id}"
        )
    completion = frame_completion(
        request.run_dir,
        frame_index,
        expected_shape=expected_shape,
        expected_spec=request.spec,
    )
    if not completion.is_complete:
        raise RuntimeError(
            "Photsim7 scope artifacts failed all-scope readback validation for "
            f"frame {frame_index}: missing {completion.missing_scope_ids}"
        )


def _record_frame_metrics(
    run_dir: Path,
    frame_index: int,
    *,
    rank: int,
    device: str,
    n_stars: int,
    pipeline_elapsed_s: float,
    total_elapsed_s: float,
    peak_cuda_allocated_mb: float | None,
    peak_cuda_reserved_mb: float | None,
) -> None:
    _, summary_path, _ = _artifact_paths(run_dir, frame_index)
    with summary_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["et_mainsim"] = {
        "schema_id": "et_mainsim.full_frame_metrics",
        "schema_version": 1,
        "rank": int(rank),
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "n_stars": int(n_stars),
        "pipeline_elapsed_s": float(pipeline_elapsed_s),
        "total_elapsed_s": float(total_elapsed_s),
        "peak_cuda_allocated_mb": peak_cuda_allocated_mb,
        "peak_cuda_reserved_mb": peak_cuda_reserved_mb,
    }
    _atomic_json(summary_path, payload)


def _select_brightest_catalog(catalog: Any, max_stars: int | None, api: Any) -> Any:
    if max_stars is None or int(catalog.n_sources) <= int(max_stars):
        return catalog
    magnitude = None
    for name in ("et_mag", "gaia_g_mag", "kp_mag", "g_mean_mag", "gmag"):
        if name in catalog.star_data:
            magnitude = np.asarray(catalog.star_data[name], dtype=float)
            break
    if magnitude is None:
        raise KeyError("max_stars requires a recognized magnitude column")
    order = np.argsort(magnitude)[: int(max_stars)]
    selected: dict[str, Any] = {}
    for name, value in catalog.star_data.items():
        array = np.asarray(value)
        selected[name] = (
            array[order]
            if array.ndim == 1 and len(array) == int(catalog.n_sources)
            else value
        )
    return api.PreparedStarCatalog(
        star_data=selected,
        metadata={
            **dict(catalog.metadata),
            "et_mainsim_selection": {
                "policy": "brightest",
                "input_n_sources": int(catalog.n_sources),
                "output_n_sources": len(order),
            },
        },
        schema_id=catalog.schema_id,
        schema_version=catalog.schema_version,
    )


def _write_effect_timeseries(run_dir: Path, services: Any, rank: int) -> None:
    if rank != 0:
        return
    timeseries = getattr(services, "effect_timeseries", None)
    if timeseries is None:
        scope_services = getattr(services, "services", ())
        if scope_services:
            timeseries = getattr(scope_services[0], "effect_timeseries", None)
    if timeseries is None:
        return
    np.savez_compressed(run_dir / "effects_timeseries.npz", **timeseries.to_arrays())
    _atomic_json(
        run_dir / "effects_timeseries.metadata.json",
        timeseries.to_metadata(),
    )


@dataclass(frozen=True)
class _SharedExposureShardSnapshot:
    path: Path
    is_final: bool
    has_linked_partial: bool
    statuses: Mapping[tuple[int, int], Any]
    dtype: np.dtype[Any]
    unit: str
    domain: str


def _shared_exposure_shard_provenance(
    request: WorkerRequest,
    batch: _SharedExposureFrameBatch,
) -> dict[str, Any]:
    provenance: dict[str, Any] = {
        "workflow": "et-full-frame",
        "orchestrator_schema_id": "et_mainsim.shared_exposure_target_plan.v1",
        "target_plan_content_sha256": None,
        "worker_rank": request.rank,
        "world_size": request.world_size,
        "batch_schema_id": _SHARED_EXPOSURE_BATCH_KEY_SCHEMA_ID,
        "batch_sha256": batch.content_sha256,
        "batch_index": batch.batch_index,
        "frames_per_shard": request.shared_exposure_stamps.frames_per_shard,
    }
    if batch.scope_explicit:
        provenance["scope_id"] = batch.scope_id
    return provenance


def _validate_shared_exposure_shard_reader(
    reader: Any,
    *,
    request: WorkerRequest,
    plan: Mapping[str, Any],
    windows: Mapping[int, Any],
    batch: _SharedExposureFrameBatch,
    product_key: str,
) -> None:
    expected_provenance = _shared_exposure_shard_provenance(request, batch)
    expected_provenance["target_plan_content_sha256"] = plan["content_sha256"]
    actual_windows = {
        int(source_id): window.to_schema()
        for source_id, window in reader.target_windows.items()
    }
    expected_windows = {
        int(source_id): window.to_schema() for source_id, window in windows.items()
    }
    mismatches: list[str] = []
    if reader.run_id != request.run_dir.name:
        mismatches.append("run_id")
    if reader.case_id != batch.case_id:
        mismatches.append("case_id")
    if reader.detector_id != str(request.spec.detector.detector_id):
        mismatches.append("detector_id")
    if reader.target_source_ids != request.shared_exposure_stamps.target_source_ids:
        mismatches.append("target_source_ids")
    if reader.frame_ids != batch.frame_ids:
        mismatches.append("frame_ids")
    if reader.product_key != product_key:
        mismatches.append("product_key")
    if reader.image_shape != request.shared_exposure_stamps.stamp_shape:
        mismatches.append("image_shape")
    if actual_windows != expected_windows:
        mismatches.append("target_windows")
    if reader.provenance != expected_provenance:
        mismatches.append("provenance")
    if mismatches:
        from et_mainsim.shared_exposure import SharedExposureReferenceDriftError

        raise SharedExposureReferenceDriftError(
            "shared-exposure shard contract drift detected for "
            f"{reader.path}: {', '.join(mismatches)}"
        )


def _inspect_shared_exposure_shards(
    *,
    api: Any,
    request: WorkerRequest,
    plan: Mapping[str, Any],
    windows: Mapping[int, Any],
    batch: _SharedExposureFrameBatch,
    shard_paths: Mapping[str, Path],
) -> dict[str, _SharedExposureShardSnapshot]:
    from et_mainsim.shared_exposure import SharedExposureReferenceDriftError

    snapshots: dict[str, _SharedExposureShardSnapshot] = {}
    for product_key, final_path in shard_paths.items():
        partial_path = api.partial_shard_path(final_path)
        final_exists = final_path.exists()
        partial_exists = partial_path.exists()
        if final_exists and partial_exists:
            try:
                linked_publication = final_path.samefile(partial_path)
            except OSError as exc:
                raise SharedExposureReferenceDriftError(
                    "shared-exposure final/partial identity could not be verified: "
                    f"{final_path}"
                ) from exc
            if not linked_publication:
                raise SharedExposureReferenceDriftError(
                    "shared-exposure final and partial paths refer to different "
                    f"files: {final_path}"
                )
        if not final_exists and not partial_exists:
            continue
        path = final_path if final_exists else partial_path
        with api.SharedExposureShardReader(
            path,
            allow_incomplete=not final_exists,
        ) as reader:
            _validate_shared_exposure_shard_reader(
                reader,
                request=request,
                plan=plan,
                windows=windows,
                batch=batch,
                product_key=product_key,
            )
            statuses = {
                (target_source_id, frame_index): reader.item_status(
                    target_source_id,
                    frame_index,
                )
                for target_source_id in request.shared_exposure_stamps.target_source_ids
                for frame_index in batch.frame_ids
            }
            if final_exists and (
                not reader.is_complete
                or any(
                    status is not api.ItemStatus.COMPLETE
                    for status in statuses.values()
                )
            ):
                raise SharedExposureReferenceDriftError(
                    f"published shared-exposure shard is incomplete: {final_path}"
                )
        snapshots[product_key] = _SharedExposureShardSnapshot(
            path=path,
            is_final=final_exists,
            has_linked_partial=final_exists and partial_exists,
            statuses=statuses,
            dtype=reader.dtype,
            unit=reader.unit,
            domain=reader.domain,
        )
    return snapshots


def _recover_linked_shared_exposure_publications(
    *,
    api: Any,
    request: WorkerRequest,
    plan: Mapping[str, Any],
    windows: Mapping[int, Any],
    batch: _SharedExposureFrameBatch,
    shard_paths: Mapping[str, Path],
    snapshots: Mapping[str, _SharedExposureShardSnapshot],
) -> None:
    """Remove only a verified hard-linked partial alias after a link crash."""

    for product_key, snapshot in snapshots.items():
        if not snapshot.has_linked_partial:
            continue
        writer = api.SharedExposureShardWriter(
            shard_paths[product_key],
            run_id=request.run_dir.name,
            case_id=batch.case_id,
            detector_id=str(request.spec.detector.detector_id),
            frame_ids=batch.frame_ids,
            target_windows=windows,
            product_key=product_key,
            dtype=snapshot.dtype,
            unit=snapshot.unit,
            domain=snapshot.domain,
            provenance={
                **_shared_exposure_shard_provenance(request, batch),
                "target_plan_content_sha256": plan["content_sha256"],
            },
            resume=True,
        )
        writer.close()


def _shared_exposure_frame_has_complete_item(
    snapshots: Mapping[str, _SharedExposureShardSnapshot],
    *,
    target_source_ids: tuple[int, ...],
    frame_index: int,
    complete_status: Any,
) -> bool:
    return any(
        snapshot.statuses.get((target_source_id, frame_index)) is complete_status
        for snapshot in snapshots.values()
        for target_source_id in target_source_ids
    )


def _shared_exposure_frame_has_complete_finals(
    snapshots: Mapping[str, _SharedExposureShardSnapshot],
    *,
    product_keys: tuple[str, ...],
    target_source_ids: tuple[int, ...],
    frame_index: int,
    complete_status: Any,
) -> bool:
    return all(
        product_key in snapshots
        and snapshots[product_key].is_final
        and all(
            snapshots[product_key].statuses.get((target_source_id, frame_index))
            is complete_status
            for target_source_id in target_source_ids
        )
        for product_key in product_keys
    )


def _validate_shared_exposure_marker_for_request(
    marker: Mapping[str, Any],
    *,
    request: WorkerRequest,
    plan: Mapping[str, Any],
    frame_index: int,
    shard_paths: Mapping[str, Path],
    parent_root: Path | None = None,
) -> None:
    from et_mainsim.shared_exposure import SharedExposureReferenceDriftError

    expected_frame = {
        "detector_id": str(request.spec.detector.detector_id),
        "frame_index": int(frame_index),
    }
    if parent_root is None:
        parent_root = request.run_dir
    expected_parent = _artifact_paths(parent_root, frame_index)[0]
    expected_plan = _shared_exposure_plan_path(request.run_dir)
    expected_shards = {
        product_key: path.resolve().relative_to(request.run_dir.resolve()).as_posix()
        for product_key, path in shard_paths.items()
    }
    actual_shards = {
        str(record["product_key"]): str(record["path"]) for record in marker["shards"]
    }
    mismatches: list[str] = []
    if marker.get("frame") != expected_frame:
        mismatches.append("frame")
    if marker.get("mode") not in {
        "parent_rendered_this_attempt",
        "deterministic_parent_reconstruction",
        "validated_existing_parent_and_shards",
    }:
        mismatches.append("mode")
    if marker.get("plan", {}).get("content_sha256") != plan["content_sha256"]:
        mismatches.append("plan.content_sha256")
    if (
        marker.get("plan", {}).get("path")
        != expected_plan.resolve().relative_to(request.run_dir.resolve()).as_posix()
    ):
        mismatches.append("plan.path")
    if (
        marker.get("parent_storage_guard", {}).get("path")
        != expected_parent.resolve().relative_to(request.run_dir.resolve()).as_posix()
    ):
        mismatches.append("parent_storage_guard.path")
    if actual_shards != expected_shards:
        mismatches.append("shards")
    if mismatches:
        raise SharedExposureReferenceDriftError(
            "shared-exposure completion marker conflicts with the worker request: "
            + ", ".join(mismatches)
        )


def _build_shared_exposure_completion(
    *,
    request: WorkerRequest,
    frame_index: int,
    mode: str,
    shard_paths: Mapping[str, Path],
    storage_guard_cache: Any | None = None,
    parent_root: Path | None = None,
) -> dict[str, Any]:
    from et_mainsim.shared_exposure import build_shared_exposure_frame_completion

    if parent_root is None:
        parent_root = request.run_dir
    parent_path, _, _ = _artifact_paths(parent_root, frame_index)
    return build_shared_exposure_frame_completion(
        frame_index=frame_index,
        detector_id=str(request.spec.detector.detector_id),
        mode=mode,
        reference_root=request.run_dir,
        parent_path=parent_path,
        plan_path=_shared_exposure_plan_path(request.run_dir),
        product_shards=shard_paths,
        storage_guard_cache=storage_guard_cache,
    )


def _publish_shared_exposure_completion(
    *,
    request: WorkerRequest,
    frame_index: int,
    marker: Mapping[str, Any],
    storage_guard_cache: Any | None = None,
    scope_id: int = 0,
    scope_contract: FullFrameScopeArtifactContract | None = None,
) -> None:
    from et_mainsim.shared_exposure import (
        publish_shared_exposure_frame_completion,
        read_shared_exposure_frame_completion,
    )

    marker_path = _shared_exposure_completion_path(
        request.run_dir,
        frame_index,
        scope_id=scope_id,
        scope_contract=scope_contract,
    )
    publish_shared_exposure_frame_completion(
        marker_path,
        marker,
        reference_root=request.run_dir,
        storage_guard_cache=storage_guard_cache,
    )
    read_shared_exposure_frame_completion(
        marker_path,
        reference_root=request.run_dir,
        storage_guard_cache=storage_guard_cache,
    )


def _close_shared_exposure_writers(
    writers_by_batch: Mapping[int, dict[str, Any]],
) -> list[tuple[int, str, Exception]]:
    """Attempt every close and remove only writers that closed successfully."""

    errors: list[tuple[int, str, Exception]] = []
    for batch_index, writers in writers_by_batch.items():
        for product_key, writer in tuple(writers.items()):
            try:
                writer.close()
            except Exception as exc:
                errors.append((batch_index, product_key, exc))
            else:
                writers.pop(product_key, None)
    return errors


def _raise_shared_exposure_close_errors(
    errors: list[tuple[int, str, Exception]],
) -> None:
    if not errors:
        return
    failures = [error for _batch_index, _product_key, error in errors]
    raise ExceptionGroup(
        "shared-exposure writer cleanup failed",
        failures,
    )


def _shared_exposure_complete_item_fingerprints(
    *,
    api: Any,
    snapshots: Mapping[str, _SharedExposureShardSnapshot],
    frame_indices: tuple[int, ...],
    target_source_ids: tuple[int, ...],
) -> dict[str, dict[tuple[int, int], dict[str, Any]]]:
    from et_mainsim.shared_exposure import array_c_order_fingerprint

    requested_frames = set(frame_indices)
    fingerprints: dict[str, dict[tuple[int, int], dict[str, Any]]] = {}
    for product_key, snapshot in snapshots.items():
        product_fingerprints: dict[tuple[int, int], dict[str, Any]] = {}
        with api.SharedExposureShardReader(
            snapshot.path,
            allow_incomplete=not snapshot.is_final,
        ) as reader:
            for target_source_id in target_source_ids:
                for frame_index in requested_frames:
                    key = (target_source_id, frame_index)
                    if snapshot.statuses.get(key) is api.ItemStatus.COMPLETE:
                        product_fingerprints[key] = array_c_order_fingerprint(
                            reader.read_array(target_source_id, frame_index)
                        )
        fingerprints[product_key] = product_fingerprints
    return fingerprints


def _assert_shared_exposure_fingerprint(
    expected: Mapping[str, Any],
    actual: Any,
) -> None:
    from et_mainsim.shared_exposure import (
        SharedExposureArrayMismatchError,
        array_c_order_fingerprint,
    )

    observed = array_c_order_fingerprint(actual)
    if expected["shape"] != observed["shape"]:
        raise SharedExposureArrayMismatchError(
            f"array shape mismatch: expected {expected['shape']}, "
            f"got {observed['shape']}"
        )
    if expected["dtype"] != observed["dtype"]:
        raise SharedExposureArrayMismatchError(
            f"array dtype mismatch: expected {expected['dtype']!r}, "
            f"got {observed['dtype']!r}"
        )
    if (
        expected["nbytes"] != observed["nbytes"]
        or expected["content_sha256"] != observed["content_sha256"]
    ):
        raise SharedExposureArrayMismatchError("array C-order bytes mismatch")


def _finalize_shared_exposure_batch(
    *,
    api: Any,
    request: WorkerRequest,
    plan: Mapping[str, Any],
    windows: Mapping[int, Any],
    batch: _SharedExposureFrameBatch,
    shard_paths: Mapping[str, Path],
    initial_snapshots: Mapping[str, _SharedExposureShardSnapshot],
    writers: dict[str, Any],
    validated_complete_items: set[tuple[str, int, int]],
    expected_parent_fingerprints: Mapping[int, Mapping[str, Any]],
    expected_final_fingerprints: Mapping[
        str, Mapping[tuple[int, int], Mapping[str, Any]]
    ],
    rendered_frames: tuple[int, ...],
    frame_modes: Mapping[int, str],
    storage_guard_cache: Any | None,
    parent_root: Path | None = None,
    scope_contract: FullFrameScopeArtifactContract | None = None,
) -> None:
    from et_mainsim.shared_exposure import SharedExposureReferenceDriftError

    shared = request.shared_exposure_stamps
    # A crash can leave a partial-path shard with every item marked COMPLETE
    # but without the final publication link.  Open it only after every item
    # has been compared against a deterministic reconstruction in this batch.
    for product_key, snapshot in initial_snapshots.items():
        if snapshot.is_final or product_key in writers:
            continue
        complete_items = {
            (product_key, target_source_id, frame_index)
            for target_source_id in shared.target_source_ids
            for frame_index in batch.frame_ids
            if snapshot.statuses.get((target_source_id, frame_index))
            is api.ItemStatus.COMPLETE
        }
        expected_items = {
            (product_key, target_source_id, frame_index)
            for target_source_id in shared.target_source_ids
            for frame_index in batch.frame_ids
        }
        if complete_items != expected_items:
            raise SharedExposureReferenceDriftError(
                "shared-exposure partial shard still has unwritten items "
                f"after batch reconstruction: {product_key}"
            )
        if not expected_items.issubset(validated_complete_items):
            raise SharedExposureReferenceDriftError(
                "shared-exposure partial shard cannot be published before "
                f"exact batch reconstruction validation: {product_key}"
            )
        writers[product_key] = api.SharedExposureShardWriter(
            shard_paths[product_key],
            run_id=request.run_dir.name,
            case_id=batch.case_id,
            detector_id=str(request.spec.detector.detector_id),
            frame_ids=batch.frame_ids,
            target_windows=windows,
            product_key=product_key,
            dtype=snapshot.dtype,
            unit=snapshot.unit,
            domain=snapshot.domain,
            provenance={
                **_shared_exposure_shard_provenance(request, batch),
                "target_plan_content_sha256": plan["content_sha256"],
            },
            resume=True,
        )

    for writer in writers.values():
        writer.finalize()
    close_errors = _close_shared_exposure_writers({batch.batch_index: writers})
    _raise_shared_exposure_close_errors(close_errors)

    final_snapshots = _inspect_shared_exposure_shards(
        api=api,
        request=request,
        plan=plan,
        windows=windows,
        batch=batch,
        shard_paths=shard_paths,
    )
    completion_markers: dict[int, dict[str, Any]] = {}
    for frame_index in rendered_frames:
        if not _shared_exposure_frame_has_complete_finals(
            final_snapshots,
            product_keys=shared.product_keys,
            target_source_ids=shared.target_source_ids,
            frame_index=frame_index,
            complete_status=api.ItemStatus.COMPLETE,
        ):
            raise RuntimeError(
                "shared-exposure batch shards failed closed readback for "
                f"frame {frame_index}"
            )
        completion_markers[frame_index] = _build_shared_exposure_completion(
            request=request,
            frame_index=frame_index,
            mode=frame_modes[frame_index],
            shard_paths=shard_paths,
            storage_guard_cache=storage_guard_cache,
            parent_root=parent_root,
        )

    # Build every immutable completion candidate first.  Exact array readback
    # must follow those storage-guard hashes so a mutation at guard-build time
    # cannot be blessed by the marker.  Publication validates the same guards
    # once more, closing the remaining readback-to-link race.
    for frame_index, expected_fingerprint in expected_parent_fingerprints.items():
        parent_path, _, _ = _artifact_paths(
            request.run_dir if parent_root is None else parent_root,
            frame_index,
        )
        _assert_shared_exposure_fingerprint(
            expected_fingerprint,
            np.load(parent_path, allow_pickle=False),
        )
    for product_key, product_fingerprints in expected_final_fingerprints.items():
        if not product_fingerprints:
            continue
        snapshot = final_snapshots.get(product_key)
        if snapshot is None or not snapshot.is_final:
            raise RuntimeError(
                "shared-exposure finalized batch shard is missing during "
                f"exact readback: batch {batch.batch_index}, {product_key}"
            )
        with api.SharedExposureShardReader(snapshot.path) as reader:
            for item_key, expected_fingerprint in product_fingerprints.items():
                _assert_shared_exposure_fingerprint(
                    expected_fingerprint,
                    reader.read_array(*item_key),
                )

    for frame_index, marker in completion_markers.items():
        _publish_shared_exposure_completion(
            request=request,
            frame_index=frame_index,
            marker=marker,
            storage_guard_cache=storage_guard_cache,
            scope_id=batch.scope_id,
            scope_contract=scope_contract,
        )
    final_snapshots.clear()


def _write_missing_shared_exposure_batch_items(
    *,
    api: Any,
    request: WorkerRequest,
    plan: Mapping[str, Any],
    windows: Mapping[int, Any],
    batch: _SharedExposureFrameBatch,
    initial_snapshots: Mapping[str, _SharedExposureShardSnapshot],
    writers: dict[str, Any],
    pending_items: list[tuple[int, Any, str, Path, np.ndarray, str, str]],
) -> None:
    from et_mainsim.shared_exposure import SharedExposureReferenceDriftError

    for (
        target_source_id,
        crop,
        product_key,
        shard_path,
        product_array,
        unit,
        domain,
    ) in pending_items:
        writer = writers.get(product_key)
        if writer is None:
            writer = api.SharedExposureShardWriter(
                shard_path,
                run_id=request.run_dir.name,
                case_id=batch.case_id,
                detector_id=str(request.spec.detector.detector_id),
                frame_ids=batch.frame_ids,
                target_windows=windows,
                product_key=product_key,
                dtype=product_array.dtype,
                unit=unit,
                domain=domain,
                provenance={
                    **_shared_exposure_shard_provenance(request, batch),
                    "target_plan_content_sha256": plan["content_sha256"],
                },
                resume=request.execution.resume,
            )
            writers[product_key] = writer
        if (
            writer.item_status(target_source_id, crop.target_identity.frame_index)
            is api.ItemStatus.COMPLETE
        ):
            raise SharedExposureReferenceDriftError(
                "shared-exposure COMPLETE item was not available for exact "
                "pre-resume validation"
            )
        snapshot = initial_snapshots.get(product_key)
        if snapshot is not None and snapshot.is_final:
            raise SharedExposureReferenceDriftError(
                "published shared-exposure shard cannot accept a missing item"
            )
        writer.write_crop(crop)


@dataclass
class _SharedExposureBatchRuntime:
    """Mutable resume state for one physical scope and one worker batch."""

    batch: _SharedExposureFrameBatch
    shard_paths: dict[str, Path]
    snapshots: dict[str, _SharedExposureShardSnapshot]
    existing_fingerprints: dict[str, dict[tuple[int, int], dict[str, Any]]]
    writers: dict[str, Any]
    validated_complete_items: set[tuple[str, int, int]]
    expected_parent_fingerprints: dict[int, dict[str, Any]]
    expected_final_fingerprints: dict[
        str, dict[tuple[int, int], dict[str, Any]]
    ]
    pending_items: list[tuple[int, Any, str, Path, np.ndarray, str, str]]
    storage_guard_cache: Any


def _open_shared_exposure_batch_runtime(
    *,
    api: Any,
    request: WorkerRequest,
    plan: Mapping[str, Any],
    windows: Mapping[int, Any],
    batch: _SharedExposureFrameBatch,
) -> _SharedExposureBatchRuntime:
    """Open one scope-local shard batch without changing its contents."""

    from et_mainsim.shared_exposure import SharedExposureStorageGuardCache

    shard_paths = _shared_exposure_batch_shard_paths(
        batch,
        plan_content_sha256=plan["content_sha256"],
        product_keys=request.shared_exposure_stamps.product_keys,
    )
    snapshots = _inspect_shared_exposure_shards(
        api=api,
        request=request,
        plan=plan,
        windows=windows,
        batch=batch,
        shard_paths=shard_paths,
    )
    existing = _shared_exposure_complete_item_fingerprints(
        api=api,
        snapshots=snapshots,
        frame_indices=batch.frame_ids,
        target_source_ids=request.shared_exposure_stamps.target_source_ids,
    )
    return _SharedExposureBatchRuntime(
        batch=batch,
        shard_paths=shard_paths,
        snapshots=snapshots,
        existing_fingerprints=existing,
        writers={},
        validated_complete_items=set(),
        expected_parent_fingerprints={},
        expected_final_fingerprints={
            product_key: dict(existing.get(product_key, {}))
            for product_key in request.shared_exposure_stamps.product_keys
        },
        pending_items=[],
        storage_guard_cache=SharedExposureStorageGuardCache(),
    )


def _queue_scope_shared_exposure_crops(
    *,
    api: Any,
    request: WorkerRequest,
    plan: Mapping[str, Any],
    windows: Mapping[int, Any],
    runtime: _SharedExposureBatchRuntime,
    parent_root: Path,
    frame_index: int,
    result: Any,
) -> None:
    """Validate and stage exact scope-local crops without new RNG draws."""

    from et_mainsim.shared_exposure import (
        SharedExposureReferenceDriftError,
        array_c_order_fingerprint,
        assert_exact_array_match,
        assert_exact_parent_crop,
    )

    batch = runtime.batch
    parent_path, _, _ = _artifact_paths(parent_root, frame_index)
    parent_array = np.load(parent_path, allow_pickle=False)
    assert_exact_array_match(
        parent_array,
        _as_numpy(result.frame_products.final_frame.array),
    )
    runtime.expected_parent_fingerprints[frame_index] = array_c_order_fingerprint(
        parent_array
    )

    frame_products: list[tuple[int, Any, str, Path, np.ndarray, str, str]] = []
    for target_source_id, window in windows.items():
        crop = api.shared_exposure_crop_v1(
            result,
            window,
            api.SharedExposureTargetIdentity(
                target_source_id=target_source_id,
                detector_id=str(request.spec.detector.detector_id),
                frame_index=frame_index,
                scope_id=batch.scope_id,
            ),
            product_keys=request.shared_exposure_stamps.product_keys,
            materialize_numpy=True,
        )
        if int(crop.target_identity.scope_id) != batch.scope_id:
            raise SharedExposureReferenceDriftError(
                "shared-exposure crop target scope conflicts with its shard batch"
            )
        crop_parent = crop.provenance.get("parent", {})
        if int(crop_parent.get("scope_id", -1)) != batch.scope_id:
            raise SharedExposureReferenceDriftError(
                "shared-exposure crop parent scope conflicts with its shard batch"
            )
        for product_key, shard_path in runtime.shard_paths.items():
            product_array, unit, domain = _shared_exposure_crop_product(
                crop,
                product_key,
            )
            if product_key == "final_stamp":
                assert_exact_parent_crop(parent_array, product_array, window)
            snapshot = runtime.snapshots.get(product_key)
            if snapshot is not None and (
                snapshot.dtype != product_array.dtype
                or snapshot.unit != unit
                or snapshot.domain != domain
            ):
                raise SharedExposureReferenceDriftError(
                    "shared-exposure product contract drift detected for "
                    f"scope {batch.scope_id}, {product_key}"
                )
            item_key = (target_source_id, frame_index)
            previous = runtime.existing_fingerprints.get(product_key, {}).get(
                item_key
            )
            if previous is not None:
                _assert_shared_exposure_fingerprint(previous, product_array)
                runtime.validated_complete_items.add(
                    (product_key, target_source_id, frame_index)
                )
            elif snapshot is not None and snapshot.is_final:
                raise SharedExposureReferenceDriftError(
                    "published shared-exposure shard is missing an expected "
                    f"complete item: scope {batch.scope_id}, {product_key} {item_key}"
                )
            runtime.expected_final_fingerprints[product_key][item_key] = (
                array_c_order_fingerprint(product_array)
            )
            frame_products.append(
                (
                    target_source_id,
                    crop,
                    product_key,
                    shard_path,
                    product_array,
                    unit,
                    domain,
                )
            )

    # Validate all pre-existing COMPLETE siblings before mutating any missing
    # item.  This is the same fail-closed ordering as the single-scope path.
    for item in frame_products:
        target_source_id, _crop, product_key, _path, _array, _unit, _domain = item
        item_key = (target_source_id, frame_index)
        if runtime.existing_fingerprints.get(product_key, {}).get(item_key) is None:
            runtime.pending_items.append(item)


def _finalize_scope_shared_exposure_batch(
    *,
    api: Any,
    request: WorkerRequest,
    plan: Mapping[str, Any],
    windows: Mapping[int, Any],
    runtime: _SharedExposureBatchRuntime,
    rendered_frames: tuple[int, ...],
    frame_modes: Mapping[int, str],
    parent_root: Path,
    scope_contract: FullFrameScopeArtifactContract,
) -> None:
    """Publish one scope-local batch and immutable completion witnesses."""

    _write_missing_shared_exposure_batch_items(
        api=api,
        request=request,
        plan=plan,
        windows=windows,
        batch=runtime.batch,
        initial_snapshots=runtime.snapshots,
        writers=runtime.writers,
        pending_items=runtime.pending_items,
    )
    _finalize_shared_exposure_batch(
        api=api,
        request=request,
        plan=plan,
        windows=windows,
        batch=runtime.batch,
        shard_paths=runtime.shard_paths,
        initial_snapshots=runtime.snapshots,
        writers=runtime.writers,
        validated_complete_items=runtime.validated_complete_items,
        expected_parent_fingerprints=runtime.expected_parent_fingerprints,
        expected_final_fingerprints=runtime.expected_final_fingerprints,
        rendered_frames=rendered_frames,
        frame_modes=frame_modes,
        storage_guard_cache=runtime.storage_guard_cache,
        parent_root=parent_root,
        scope_contract=scope_contract,
    )
    runtime.pending_items.clear()
    runtime.existing_fingerprints.clear()
    runtime.snapshots.clear()
    runtime.shard_paths.clear()
    runtime.storage_guard_cache.clear()


def _render_multiscope_shared_exposure_frame(
    *,
    request: WorkerRequest,
    api: Any,
    services: Any,
    contract: FullFrameScopeArtifactContract,
    catalog: Any,
    frame_index: int,
    expected_shape: tuple[int, int],
    plan: Mapping[str, Any],
    windows: Mapping[int, Any],
    scope_modes: Mapping[int, str],
    runtime_by_scope_batch: dict[tuple[int, int], _SharedExposureBatchRuntime],
    batch_by_scope_frame: Mapping[int, Mapping[int, _SharedExposureFrameBatch]],
    render_frames_by_scope_batch: Mapping[tuple[int, int], tuple[int, ...]],
    all_scope_modes: Mapping[int, Mapping[int, str]],
) -> None:
    """Render six parent images and stage exact same-scope crop products.

    Each yielded result is persisted and cropped before the next scope is
    requested.  There is intentionally no in-memory six-image cube and no
    image-level sum.
    """

    if contract.is_single_scope:
        raise ValueError("six-scope shared exposure requires a multiscope contract")
    if set(scope_modes) != set(contract.scope_ids):
        raise ValueError("scope_modes must contain the complete scope contract")

    if request.execution.overwrite:
        absolute_raw_frame_index = (
            _ET_FULL_FRAME_ABSOLUTE_RAW_FRAME_START_INDEX + int(frame_index)
        )
        for scope_id, mode in scope_modes.items():
            if mode == "skip":
                continue
            cadence_path = contract.scope_root(scope_id) / api.cadence_selection_truth_relative_path(
                absolute_raw_frame_index
            )
            cadence_path.unlink(missing_ok=True)

    torch = None
    if request.execution.device == "cuda":
        import torch as torch_module

        torch = torch_module
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA execution requested but torch reports no CUDA device")
        torch.cuda.reset_peak_memory_stats()

    iterator = iter(
        api.iter_single_cadence_full_frame_scopes(
            request.spec,
            services=services,
            frame_index=frame_index,
            renderer_options=_full_frame_renderer_options(request),
            worker_rank=request.rank,
            rng_trace_scope={"run_id": request.run_dir.name},
        )
    )
    for expected_scope_id in contract.scope_ids:
        scope_started = time.perf_counter()
        try:
            scope_id, result = next(iterator)
        except StopIteration as error:
            raise RuntimeError(
                "Photsim7 multiscope iterator ended before every scope was rendered"
            ) from error
        if int(scope_id) != expected_scope_id:
            raise RuntimeError(
                "Photsim7 multiscope iterator returned scope ids out of canonical "
                f"order: expected {expected_scope_id}, received {scope_id}"
            )
        if torch is not None:
            torch.cuda.synchronize()
        mode = scope_modes[scope_id]
        if mode == "skip":
            del result
            continue
        if mode not in {
            "parent_rendered_this_attempt",
            "deterministic_parent_reconstruction",
        }:
            raise RuntimeError(f"unsupported shared-exposure scope mode {mode!r}")

        scope_root = contract.scope_root(scope_id)
        if mode == "parent_rendered_this_attempt":
            options = api.FullFrameArtifactOptions(
                save_frame_summaries=True,
                save_cosmic_events=True,
                save_bias=bool(request.spec.artifacts.save_bias_artifacts),
                save_preview=frame_index < request.execution.preview_count,
            )
            writer = api.FullFrameArtifactWriter(scope_root, options=options)
            _persist_scope_frame_result(writer=writer, result=result, spec=request.spec)
            if not _scope_frame_is_complete(
                contract,
                scope_id=scope_id,
                frame_index=frame_index,
                expected_shape=expected_shape,
                expected_spec=request.spec,
            ):
                raise RuntimeError(
                    "Photsim7 scope artifacts failed readback validation for "
                    f"frame {frame_index}, scope {scope_id}"
                )
            if request.execution.save_cosmic_mask:
                cosmic = getattr(result.detector_result, "cosmic_metadata", None)
                mask = None if cosmic is None else getattr(cosmic, "mask", None)
                if mask is not None:
                    mask_array = _as_numpy(mask)
                    if mask_array.ndim == 3 and mask_array.shape[0] == 1:
                        mask_array = mask_array[0]
                    mask_path = (
                        scope_root
                        / "cosmic_events"
                        / f"frame_{frame_index:06d}_mask.npy"
                    )
                    mask_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(mask_path, mask_array)
            if request.execution.save_stellar_mean:
                stellar_mean = result.renderer_components.get("stellar_mean")
                if stellar_mean is None:
                    raise KeyError("Photsim7 did not return stellar_mean")
                np.save(
                    scope_root
                    / "frames"
                    / f"frame_{frame_index:06d}_stellar_mean_e.npy",
                    _as_numpy(stellar_mean).astype(np.float32),
                )
            _record_frame_metrics(
                scope_root,
                frame_index,
                rank=request.rank,
                device=request.execution.device,
                n_stars=int(catalog.n_sources),
                pipeline_elapsed_s=time.perf_counter() - scope_started,
                total_elapsed_s=time.perf_counter() - scope_started,
                peak_cuda_allocated_mb=(
                    None
                    if torch is None
                    else float(torch.cuda.max_memory_allocated() / 1024**2)
                ),
                peak_cuda_reserved_mb=(
                    None
                    if torch is None
                    else float(torch.cuda.max_memory_reserved() / 1024**2)
                ),
            )
        elif not _scope_frame_is_complete(
            contract,
            scope_id=scope_id,
            frame_index=frame_index,
            expected_shape=expected_shape,
            expected_spec=request.spec,
        ):
            raise RuntimeError(
                "shared-exposure reconstruction requires a complete scope parent: "
                f"frame {frame_index}, scope {scope_id}"
            )

        batch = batch_by_scope_frame[scope_id][frame_index]
        runtime_key = (scope_id, batch.batch_index)
        runtime = runtime_by_scope_batch.get(runtime_key)
        if runtime is None:
            runtime = _open_shared_exposure_batch_runtime(
                api=api,
                request=request,
                plan=plan,
                windows=windows,
                batch=batch,
            )
            runtime_by_scope_batch[runtime_key] = runtime
        _queue_scope_shared_exposure_crops(
            api=api,
            request=request,
            plan=plan,
            windows=windows,
            runtime=runtime,
            parent_root=scope_root,
            frame_index=frame_index,
            result=result,
        )
        del result

        batch_render_frames = render_frames_by_scope_batch[runtime_key]
        if frame_index == batch_render_frames[-1]:
            _finalize_scope_shared_exposure_batch(
                api=api,
                request=request,
                plan=plan,
                windows=windows,
                runtime=runtime,
                rendered_frames=batch_render_frames,
                frame_modes=all_scope_modes[scope_id],
                parent_root=scope_root,
                scope_contract=contract,
            )
            runtime_by_scope_batch.pop(runtime_key, None)

    try:
        unexpected_scope_id, _ = next(iterator)
    except StopIteration:
        pass
    else:
        raise RuntimeError(
            "Photsim7 multiscope iterator returned an unexpected extra scope "
            f"{unexpected_scope_id}"
        )


def _run_multiscope_shared_exposure_worker(
    *,
    request: WorkerRequest,
    api: Any,
    scope_contract: FullFrameScopeArtifactContract,
    assigned: tuple[int, ...],
    expected_shape: tuple[int, int],
    started: float,
) -> WorkerResult:
    """Run the exposure-first crop contract for the frozen six-scope system."""

    if scope_contract.is_single_scope:
        raise ValueError("multiscope shared exposure requires six scope artifacts")
    shared = request.shared_exposure_stamps
    if not shared.enabled:
        raise ValueError("multiscope shared exposure requires enabled configuration")

    from et_mainsim.shared_exposure import (
        SharedExposureReferenceDriftError,
        SharedExposureStorageGuardCache,
        read_shared_exposure_frame_completion,
        read_shared_exposure_target_plan,
    )

    shared_root = _shared_exposure_root(request.run_dir)
    if (
        not request.execution.resume
        and not request.execution.overwrite
        and os.path.lexists(shared_root)
    ):
        raise FileExistsError(
            "shared-exposure bundle already exists; use resume or overwrite: "
            f"{shared_root}"
        )
    if request.execution.overwrite and not request.shared_exposure_overwrite_prepared:
        try:
            shared_root.mkdir()
        except FileExistsError as exc:
            raise FileExistsError(
                "the coordinator must clear the shared-exposure bundle before "
                "an overwrite worker starts; direct overwrite refuses existing "
                f"bundle state at {shared_root}"
            ) from exc

    plan_path = _shared_exposure_plan_path(request.run_dir)
    shared_plan: dict[str, Any] | None = None
    shared_windows: dict[int, Any] = {}
    plan_exists = plan_path.exists()
    if not plan_exists:
        dependent_artifacts = (
            [
                path
                for path in shared_root.rglob("*")
                if path.is_file()
                and not (
                    path.parent == shared_root
                    and path.name.startswith(f".{plan_path.name}.")
                    and path.name.endswith(".tmp")
                )
            ]
            if shared_root.exists()
            else []
        )
        plan_exists = plan_path.exists()
        if dependent_artifacts and not plan_exists:
            raise SharedExposureReferenceDriftError(
                "shared-exposure target plan is missing while dependent artifacts "
                "remain"
            )
    if plan_exists:
        shared_plan = read_shared_exposure_target_plan(plan_path)
        _validate_shared_exposure_plan_for_request(shared_plan, request=request)
        shared_windows = _shared_exposure_windows(shared_plan, api=api)

    batches_by_scope = {
        scope_id: _shared_exposure_frame_batches(
            request,
            assigned,
            scope_id=scope_id,
            scope_contract=scope_contract,
        )
        for scope_id in scope_contract.scope_ids
    }
    batch_by_scope_frame = {
        scope_id: _shared_exposure_batch_by_frame(batches)
        for scope_id, batches in batches_by_scope.items()
    }
    scope_modes: dict[int, dict[int, str]] = {
        scope_id: {} for scope_id in scope_contract.scope_ids
    }
    skipped: list[int] = []
    resume_snapshots: dict[
        tuple[int, int], dict[str, _SharedExposureShardSnapshot]
    ] = {}
    resume_shard_paths: dict[tuple[int, int], dict[str, Path]] = {}
    resume_guard_caches: dict[tuple[int, int], Any] = {}
    linked_recovery_keys: set[tuple[int, int]] = set()

    def resume_batch_state(
        scope_id: int,
        batch: _SharedExposureFrameBatch,
    ) -> tuple[dict[str, Path], dict[str, _SharedExposureShardSnapshot], Any]:
        key = (scope_id, batch.batch_index)
        if key not in resume_snapshots:
            if shared_plan is None:
                raise RuntimeError("shared-exposure plan is required for resume")
            shard_paths = _shared_exposure_batch_shard_paths(
                batch,
                plan_content_sha256=shared_plan["content_sha256"],
                product_keys=shared.product_keys,
            )
            snapshots = _inspect_shared_exposure_shards(
                api=api,
                request=request,
                plan=shared_plan,
                windows=shared_windows,
                batch=batch,
                shard_paths=shard_paths,
            )
            resume_shard_paths[key] = shard_paths
            resume_snapshots[key] = snapshots
            resume_guard_caches[key] = SharedExposureStorageGuardCache()
            if any(snapshot.has_linked_partial for snapshot in snapshots.values()):
                linked_recovery_keys.add(key)
        return (
            resume_shard_paths[key],
            resume_snapshots[key],
            resume_guard_caches[key],
        )

    for frame_index in assigned:
        every_scope_complete = True
        for scope_id in scope_contract.scope_ids:
            parent_root = scope_contract.scope_root(scope_id)
            parent_complete = _scope_frame_is_complete(
                scope_contract,
                scope_id=scope_id,
                frame_index=frame_index,
                expected_shape=expected_shape,
                expected_spec=request.spec,
            )
            if not request.execution.resume:
                if not request.execution.overwrite and _has_partial_artifacts(
                    parent_root,
                    frame_index,
                ):
                    raise FileExistsError(
                        f"Frame {frame_index}, scope {scope_id} already has "
                        "artifacts; use resume or overwrite"
                    )
                scope_modes[scope_id][frame_index] = "parent_rendered_this_attempt"
                every_scope_complete = False
                continue
            if shared_plan is None:
                scope_modes[scope_id][frame_index] = (
                    "deterministic_parent_reconstruction"
                    if parent_complete
                    else "parent_rendered_this_attempt"
                )
                every_scope_complete = False
                continue

            batch = batch_by_scope_frame[scope_id][frame_index]
            shard_paths, snapshots, guard_cache = resume_batch_state(scope_id, batch)
            marker_path = _shared_exposure_completion_path(
                request.run_dir,
                frame_index,
                scope_id=scope_id,
                scope_contract=scope_contract,
            )
            if marker_path.exists():
                marker = read_shared_exposure_frame_completion(
                    marker_path,
                    reference_root=request.run_dir,
                    storage_guard_cache=guard_cache,
                )
                _validate_shared_exposure_marker_for_request(
                    marker,
                    request=request,
                    plan=shared_plan,
                    frame_index=frame_index,
                    shard_paths=shard_paths,
                    parent_root=parent_root,
                )
                if not parent_complete:
                    raise SharedExposureReferenceDriftError(
                        "shared-exposure completion exists but its scope parent "
                        f"is incomplete for frame {frame_index}, scope {scope_id}"
                    )
                if not _shared_exposure_frame_has_complete_finals(
                    snapshots,
                    product_keys=shared.product_keys,
                    target_source_ids=shared.target_source_ids,
                    frame_index=frame_index,
                    complete_status=api.ItemStatus.COMPLETE,
                ):
                    raise SharedExposureReferenceDriftError(
                        "shared-exposure completion marker does not resolve to "
                        "complete final shards for "
                        f"frame {frame_index}, scope {scope_id}"
                    )
                scope_modes[scope_id][frame_index] = "skip"
                continue

            every_scope_complete = False
            if parent_complete:
                scope_modes[scope_id][frame_index] = (
                    "deterministic_parent_reconstruction"
                )
                continue
            if _shared_exposure_frame_has_complete_item(
                snapshots,
                target_source_ids=shared.target_source_ids,
                frame_index=frame_index,
                complete_status=api.ItemStatus.COMPLETE,
            ):
                raise SharedExposureReferenceDriftError(
                    "shared-exposure crop is complete but its scope parent is "
                    f"incomplete for frame {frame_index}, scope {scope_id}"
                )
            scope_modes[scope_id][frame_index] = "parent_rendered_this_attempt"
        if every_scope_complete:
            skipped.append(frame_index)

    # Validate every marker before cleaning recoverable final/partial hard-link
    # debris.  This mirrors the single-scope ordering and never mutates a bad
    # control plane artifact.
    if shared_plan is not None and request.execution.resume:
        for scope_id, batch_index in sorted(linked_recovery_keys):
            batch = batches_by_scope[scope_id][batch_index]
            shard_paths = resume_shard_paths[(scope_id, batch_index)]
            snapshots = resume_snapshots[(scope_id, batch_index)]
            _recover_linked_shared_exposure_publications(
                api=api,
                request=request,
                plan=shared_plan,
                windows=shared_windows,
                batch=batch,
                shard_paths=shard_paths,
                snapshots=snapshots,
            )
    for snapshots in resume_snapshots.values():
        snapshots.clear()
    for cache in resume_guard_caches.values():
        cache.clear()

    to_render = tuple(
        frame_index
        for frame_index in assigned
        if any(
            scope_modes[scope_id][frame_index] != "skip"
            for scope_id in scope_contract.scope_ids
        )
    )
    if not to_render:
        _atomic_json(
            request.run_dir / f"worker_{request.rank:02d}_start.json",
            {
                "schema_id": "et_mainsim.full_frame_worker",
                "schema_version": 1,
                "rank": request.rank,
                "world_size": request.world_size,
                "pid": os.getpid(),
                "device": request.execution.device,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "assigned_frames": list(assigned),
                "render_frames": [],
                "skipped_frames": skipped,
                "catalog_cache": str(request.catalog_cache),
                "n_sources": None,
            },
        )
        result = WorkerResult(
            rank=request.rank,
            rendered=(),
            skipped=tuple(skipped),
            elapsed_s=time.perf_counter() - started,
        )
        _atomic_json(
            request.run_dir / f"worker_{request.rank:02d}_done.json",
            {
                "schema_id": "et_mainsim.full_frame_worker",
                "schema_version": 1,
                **result.to_dict(),
            },
        )
        return result

    catalog = api.StarCatalogCache.read(request.catalog_cache)
    catalog = _select_brightest_catalog(catalog, request.execution.max_stars, api)
    registry = api.DataRegistry(data_root=request.data_root)
    services = api.build_multiscope_full_frame_services(
        request.spec,
        catalog=catalog,
        data_registry=registry,
    )
    if shared_plan is None:
        from et_mainsim.shared_exposure import (
            build_shared_exposure_target_plan,
            publish_shared_exposure_target_plan,
        )

        geometry = api.resolve_full_frame_source_pixel_geometry(services.services[0])
        shared_plan = build_shared_exposure_target_plan(
            geometry,
            shared.target_source_ids,
            detector_shape=expected_shape,
            stamp_shape=shared.stamp_shape,
        )
        _validate_shared_exposure_plan_for_request(shared_plan, request=request)
        publish_shared_exposure_target_plan(plan_path, shared_plan)
        shared_windows = _shared_exposure_windows(shared_plan, api=api)

    render_frames_by_scope_batch = {
        (scope_id, batch.batch_index): tuple(
            frame_index
            for frame_index in batch.frame_ids
            if scope_modes[scope_id][frame_index] != "skip"
        )
        for scope_id, batches in batches_by_scope.items()
        for batch in batches
        if any(scope_modes[scope_id][frame_index] != "skip" for frame_index in batch.frame_ids)
    }
    if any(
        scope_modes[scope_id][frame_index] == "parent_rendered_this_attempt"
        for scope_id in scope_contract.scope_ids
        for frame_index in to_render
    ):
        _write_effect_timeseries(request.run_dir, services, request.rank)

    _atomic_json(
        request.run_dir / f"worker_{request.rank:02d}_start.json",
        {
            "schema_id": "et_mainsim.full_frame_worker",
            "schema_version": 1,
            "rank": request.rank,
            "world_size": request.world_size,
            "pid": os.getpid(),
            "device": request.execution.device,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "assigned_frames": list(assigned),
            "render_frames": list(to_render),
            "scope_render_modes": {
                str(scope_id): {
                    str(frame_index): scope_modes[scope_id][frame_index]
                    for frame_index in to_render
                }
                for scope_id in scope_contract.scope_ids
            },
            "skipped_frames": skipped,
            "catalog_cache": str(request.catalog_cache),
            "n_sources": int(catalog.n_sources),
        },
    )

    runtime_by_scope_batch: dict[tuple[int, int], _SharedExposureBatchRuntime] = {}
    rendered: list[int] = []
    try:
        for frame_index in to_render:
            _render_multiscope_shared_exposure_frame(
                request=request,
                api=api,
                services=services,
                contract=scope_contract,
                catalog=catalog,
                frame_index=frame_index,
                expected_shape=expected_shape,
                plan=shared_plan,
                windows=shared_windows,
                scope_modes={
                    scope_id: scope_modes[scope_id][frame_index]
                    for scope_id in scope_contract.scope_ids
                },
                runtime_by_scope_batch=runtime_by_scope_batch,
                batch_by_scope_frame=batch_by_scope_frame,
                render_frames_by_scope_batch=render_frames_by_scope_batch,
                all_scope_modes=scope_modes,
            )
            if not frame_completion(
                request.run_dir,
                frame_index,
                expected_shape=expected_shape,
                expected_spec=request.spec,
            ).is_complete:
                raise RuntimeError(
                    "Photsim7 scope artifacts failed all-scope readback "
                    f"validation for frame {frame_index}"
                )
            rendered.append(frame_index)
        if runtime_by_scope_batch:
            raise RuntimeError(
                "shared-exposure scope batches remained open after their final "
                "assigned frame"
            )
    except BaseException as primary_error:
        close_errors: list[tuple[int, str, Exception]] = []
        for runtime in runtime_by_scope_batch.values():
            close_errors.extend(
                _close_shared_exposure_writers(
                    {runtime.batch.batch_index: runtime.writers}
                )
            )
        for batch_index, product_key, close_error in close_errors:
            primary_error.add_note(
                "shared-exposure writer close failed for "
                f"batch {batch_index}, product {product_key!r}: "
                f"{type(close_error).__name__}: {close_error}"
            )
        raise

    result = WorkerResult(
        rank=request.rank,
        rendered=tuple(rendered),
        skipped=tuple(skipped),
        elapsed_s=time.perf_counter() - started,
    )
    _atomic_json(
        request.run_dir / f"worker_{request.rank:02d}_done.json",
        {
            "schema_id": "et_mainsim.full_frame_worker",
            "schema_version": 1,
            **result.to_dict(),
        },
    )
    return result


def run_worker(
    request: WorkerRequest, *, science_api: Any | None = None
) -> WorkerResult:
    api = _science_api() if science_api is None else science_api
    started = time.perf_counter()
    expected_shape = tuple(int(value) for value in request.spec.detector.shape)
    scope_contract = _scope_contract_for_spec(request.run_dir, request.spec)
    assigned = tuple(request.frame_indices[request.rank :: request.world_size])
    request.run_dir.mkdir(parents=True, exist_ok=True)

    shared = request.shared_exposure_stamps
    if shared.enabled and not scope_contract.is_single_scope:
        return _run_multiscope_shared_exposure_worker(
            request=request,
            api=api,
            scope_contract=scope_contract,
            assigned=assigned,
            expected_shape=expected_shape,
            started=started,
        )
    shared_root = _shared_exposure_root(request.run_dir)
    if (
        shared.enabled
        and not request.execution.resume
        and not request.execution.overwrite
        and os.path.lexists(shared_root)
    ):
        raise FileExistsError(
            "shared-exposure bundle already exists; use resume or overwrite: "
            f"{shared_root}"
        )
    if (
        shared.enabled
        and request.execution.overwrite
        and not request.shared_exposure_overwrite_prepared
    ):
        try:
            shared_root.mkdir()
        except FileExistsError as exc:
            raise FileExistsError(
                "the coordinator must clear the shared-exposure bundle before "
                "an overwrite worker starts; direct overwrite refuses existing "
                f"bundle state at {shared_root}"
            ) from exc
    shared_plan: dict[str, Any] | None = None
    shared_windows: dict[int, Any] = {}
    shared_batches = (
        _shared_exposure_frame_batches(request, assigned) if shared.enabled else ()
    )
    shared_batch_by_frame = _shared_exposure_batch_by_frame(shared_batches)
    shared_shard_paths_by_batch: dict[int, dict[str, Path]] = {}
    shared_snapshots_by_batch: dict[int, dict[str, _SharedExposureShardSnapshot]] = {}
    if shared.enabled:
        from et_mainsim.shared_exposure import (
            SharedExposureReferenceDriftError,
            SharedExposureStorageGuardCache,
            read_shared_exposure_target_plan,
        )

        plan_path = _shared_exposure_plan_path(request.run_dir)
        plan_exists = plan_path.exists()
        if not plan_exists:
            shared_root = _shared_exposure_root(request.run_dir)
            dependent_artifacts = (
                [
                    path
                    for path in shared_root.rglob("*")
                    if path.is_file()
                    and not (
                        path.parent == shared_root
                        and path.name.startswith(f".{plan_path.name}.")
                        and path.name.endswith(".tmp")
                    )
                ]
                if shared_root.exists()
                else []
            )
            # A racing worker may have linked the immutable plan after our
            # first existence check.  Recheck before declaring orphan output.
            plan_exists = plan_path.exists()
            if dependent_artifacts and not plan_exists:
                raise SharedExposureReferenceDriftError(
                    "shared-exposure target plan is missing while dependent "
                    "artifacts remain"
                )
        if plan_exists:
            shared_plan = read_shared_exposure_target_plan(plan_path)
            _validate_shared_exposure_plan_for_request(
                shared_plan,
                request=request,
            )
            shared_windows = _shared_exposure_windows(shared_plan, api=api)

    parent_complete_by_frame = {
        frame_index: frame_completion(
            request.run_dir,
            frame_index,
            expected_shape=expected_shape,
            expected_spec=request.spec,
        ).is_complete
        for frame_index in assigned
    }
    frame_modes: dict[int, str] = {}
    skipped: list[int] = []
    resume_batch_index: int | None = None
    resume_shard_paths: dict[str, Path] = {}
    resume_snapshots: dict[str, _SharedExposureShardSnapshot] = {}
    resume_guard_cache: Any | None = None
    linked_publication_batch_indices: set[int] = set()
    for frame_index in assigned:
        parent_complete = parent_complete_by_frame[frame_index]
        if not request.execution.resume:
            if not request.execution.overwrite and _has_partial_scope_artifacts(
                scope_contract,
                frame_index,
            ):
                raise FileExistsError(
                    f"Frame {frame_index} already has artifacts; use resume or "
                    "overwrite"
                )
            frame_modes[frame_index] = "parent_rendered_this_attempt"
            continue
        if not shared.enabled:
            if parent_complete:
                skipped.append(frame_index)
            else:
                frame_modes[frame_index] = "parent_rendered_this_attempt"
            continue
        if shared_plan is None:
            frame_modes[frame_index] = (
                "deterministic_parent_reconstruction"
                if parent_complete
                else "parent_rendered_this_attempt"
            )
            continue

        batch = shared_batch_by_frame[frame_index]
        if batch.batch_index != resume_batch_index:
            if resume_guard_cache is not None:
                resume_guard_cache.clear()
            resume_shard_paths = _shared_exposure_batch_shard_paths(
                batch,
                plan_content_sha256=shared_plan["content_sha256"],
                product_keys=shared.product_keys,
            )
            resume_snapshots = _inspect_shared_exposure_shards(
                api=api,
                request=request,
                plan=shared_plan,
                windows=shared_windows,
                batch=batch,
                shard_paths=resume_shard_paths,
            )
            resume_guard_cache = SharedExposureStorageGuardCache()
            resume_batch_index = batch.batch_index
            if any(
                snapshot.has_linked_partial for snapshot in resume_snapshots.values()
            ):
                linked_publication_batch_indices.add(batch.batch_index)
        shard_paths = resume_shard_paths
        snapshots = resume_snapshots

        marker_path = _shared_exposure_completion_path(
            request.run_dir,
            frame_index,
        )
        if marker_path.exists():
            from et_mainsim.shared_exposure import (
                SharedExposureReferenceDriftError,
                read_shared_exposure_frame_completion,
            )

            marker = read_shared_exposure_frame_completion(
                marker_path,
                reference_root=request.run_dir,
                storage_guard_cache=resume_guard_cache,
            )
            _validate_shared_exposure_marker_for_request(
                marker,
                request=request,
                plan=shared_plan,
                frame_index=frame_index,
                shard_paths=shard_paths,
            )
            if not parent_complete:
                raise SharedExposureReferenceDriftError(
                    "shared-exposure completion exists but its parent frame "
                    f"contract is incomplete for frame {frame_index}"
                )
            if not _shared_exposure_frame_has_complete_finals(
                snapshots,
                product_keys=shared.product_keys,
                target_source_ids=shared.target_source_ids,
                frame_index=frame_index,
                complete_status=api.ItemStatus.COMPLETE,
            ):
                raise SharedExposureReferenceDriftError(
                    "shared-exposure completion marker does not resolve to "
                    f"complete final shards for frame {frame_index}"
                )
            skipped.append(frame_index)
            continue

        if parent_complete:
            # The completion marker is the only durable guard for the entire
            # parent frame.  A complete shard (or another frame's marker) can
            # authenticate the shared HDF5 bytes, but cannot authenticate this
            # frame's parent NPY.  Reconstruct both products deterministically
            # before replacing a missing commit witness.
            frame_modes[frame_index] = "deterministic_parent_reconstruction"
            continue

        if _shared_exposure_frame_has_complete_item(
            snapshots,
            target_source_ids=shared.target_source_ids,
            frame_index=frame_index,
            complete_status=api.ItemStatus.COMPLETE,
        ):
            from et_mainsim.shared_exposure import SharedExposureReferenceDriftError

            raise SharedExposureReferenceDriftError(
                "shared-exposure crop is complete but its parent frame contract "
                f"is incomplete for frame {frame_index}"
            )
        frame_modes[frame_index] = "parent_rendered_this_attempt"

    if resume_guard_cache is not None:
        resume_guard_cache.clear()
    resume_shard_paths = {}
    resume_snapshots = {}
    shard_paths = {}
    snapshots = {}

    # A final/partial hard-link pair is recoverable publication debris, but it
    # is still user-visible storage.  Validate every existing frame marker
    # first so malformed control state never triggers cleanup mutation.
    if shared_plan is not None and request.execution.resume:
        for batch in shared_batches:
            if batch.batch_index not in linked_publication_batch_indices:
                continue
            shard_paths = _shared_exposure_batch_shard_paths(
                batch,
                plan_content_sha256=shared_plan["content_sha256"],
                product_keys=shared.product_keys,
            )
            snapshots = _inspect_shared_exposure_shards(
                api=api,
                request=request,
                plan=shared_plan,
                windows=shared_windows,
                batch=batch,
                shard_paths=shard_paths,
            )
            _recover_linked_shared_exposure_publications(
                api=api,
                request=request,
                plan=shared_plan,
                windows=shared_windows,
                batch=batch,
                shard_paths=shard_paths,
                snapshots=snapshots,
            )
            snapshots.clear()

    to_render = tuple(
        frame_index for frame_index in assigned if frame_index in frame_modes
    )
    if not to_render:
        _atomic_json(
            request.run_dir / f"worker_{request.rank:02d}_start.json",
            {
                "schema_id": "et_mainsim.full_frame_worker",
                "schema_version": 1,
                "rank": request.rank,
                "world_size": request.world_size,
                "pid": os.getpid(),
                "device": request.execution.device,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "assigned_frames": list(assigned),
                "render_frames": [],
                "skipped_frames": skipped,
                "catalog_cache": str(request.catalog_cache),
                "n_sources": None,
            },
        )
        worker_result = WorkerResult(
            rank=request.rank,
            rendered=(),
            skipped=tuple(skipped),
            elapsed_s=time.perf_counter() - started,
        )
        _atomic_json(
            request.run_dir / f"worker_{request.rank:02d}_done.json",
            {
                "schema_id": "et_mainsim.full_frame_worker",
                "schema_version": 1,
                **worker_result.to_dict(),
            },
        )
        return worker_result

    catalog = api.StarCatalogCache.read(request.catalog_cache)
    catalog = _select_brightest_catalog(
        catalog,
        request.execution.max_stars,
        api,
    )
    registry = api.DataRegistry(data_root=request.data_root)
    if scope_contract.is_single_scope:
        services = api.build_full_frame_services(
            request.spec,
            catalog=catalog,
            data_registry=registry,
        )
    else:
        services = api.build_multiscope_full_frame_services(
            request.spec,
            catalog=catalog,
            data_registry=registry,
        )
    shared_writers_by_batch: dict[int, dict[str, Any]] = {}
    if shared.enabled:
        from et_mainsim.shared_exposure import (
            build_shared_exposure_target_plan,
            publish_shared_exposure_target_plan,
        )

        geometry = api.resolve_full_frame_source_pixel_geometry(services)
        shared_plan = build_shared_exposure_target_plan(
            geometry,
            shared.target_source_ids,
            detector_shape=expected_shape,
            stamp_shape=shared.stamp_shape,
        )
        _validate_shared_exposure_plan_for_request(
            shared_plan,
            request=request,
        )
        publish_shared_exposure_target_plan(
            _shared_exposure_plan_path(request.run_dir),
            shared_plan,
        )
        shared_windows = _shared_exposure_windows(shared_plan, api=api)
        shared_shard_paths_by_batch = {}
        shared_snapshots_by_batch = {}
    render_frames_by_batch = {
        batch.batch_index: tuple(
            frame_index for frame_index in batch.frame_ids if frame_index in frame_modes
        )
        for batch in shared_batches
    }
    existing_fingerprints_by_batch: dict[
        int, dict[str, dict[tuple[int, int], dict[str, Any]]]
    ] = {}
    validated_complete_items_by_batch: dict[int, set[tuple[str, int, int]]] = {}
    expected_parent_fingerprints_by_batch: dict[int, dict[int, dict[str, Any]]] = {}
    expected_final_fingerprints_by_batch: dict[
        int, dict[str, dict[tuple[int, int], dict[str, Any]]]
    ] = {}
    pending_items_by_batch: dict[
        int, list[tuple[int, Any, str, Path, np.ndarray, str, str]]
    ] = {}
    shared_guard_caches_by_batch: dict[int, Any] = {}
    if any(
        frame_modes[frame_index] == "parent_rendered_this_attempt"
        for frame_index in to_render
    ):
        _write_effect_timeseries(request.run_dir, services, request.rank)
    _atomic_json(
        request.run_dir / f"worker_{request.rank:02d}_start.json",
        {
            "schema_id": "et_mainsim.full_frame_worker",
            "schema_version": 1,
            "rank": request.rank,
            "world_size": request.world_size,
            "pid": os.getpid(),
            "device": request.execution.device,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "assigned_frames": list(assigned),
            "render_frames": list(to_render),
            "render_modes": {
                str(frame_index): frame_modes[frame_index] for frame_index in to_render
            },
            "skipped_frames": skipped,
            "catalog_cache": str(request.catalog_cache),
            "n_sources": int(catalog.n_sources),
        },
    )

    rendered: list[int] = []
    try:
        for frame_index in to_render:
            if not scope_contract.is_single_scope:
                _render_multiscope_frame(
                    request=request,
                    api=api,
                    services=services,
                    contract=scope_contract,
                    catalog=catalog,
                    frame_index=frame_index,
                    expected_shape=expected_shape,
                )
                rendered.append(frame_index)
                continue
            if request.execution.overwrite:
                absolute_raw_frame_index = (
                    _ET_FULL_FRAME_ABSOLUTE_RAW_FRAME_START_INDEX + int(frame_index)
                )
                cadence_path = (
                    request.run_dir
                    / api.cadence_selection_truth_relative_path(
                        absolute_raw_frame_index
                    )
                )
                cadence_path.unlink(missing_ok=True)
            frame_started = time.perf_counter()
            frame_mode = frame_modes[frame_index]
            reconstructing = frame_mode == "deterministic_parent_reconstruction"
            parent_writer = None
            if not reconstructing:
                options = api.FullFrameArtifactOptions(
                    save_frame_summaries=True,
                    save_cosmic_events=True,
                    save_bias=bool(request.spec.artifacts.save_bias_artifacts),
                    save_preview=frame_index < request.execution.preview_count,
                )
                parent_writer = api.FullFrameArtifactWriter(
                    request.run_dir,
                    options=options,
                )
            torch = None
            if request.execution.device == "cuda":
                import torch as torch_module

                torch = torch_module
                if not torch.cuda.is_available():
                    raise RuntimeError(
                        "CUDA execution requested but torch reports no CUDA device"
                    )
                torch.cuda.reset_peak_memory_stats()
            pipeline_started = time.perf_counter()
            result = api.run_single_cadence_full_frame(
                request.spec,
                services=services,
                frame_index=frame_index,
                renderer_options={
                    "enable_stellar_photon_noise": True,
                    "enable_background_light": True,
                    "enable_scattered_light": bool(
                        request.spec.sky.scattered_light.to_value(
                            u.electron / u.s / u.pix
                        )
                    ),
                    "enable_dark_current": True,
                    "progress": request.execution.progress,
                },
                worker_rank=request.rank,
                rng_trace_scope={"run_id": request.run_dir.name},
                artifact_writer=parent_writer,
            )
            if torch is not None:
                torch.cuda.synchronize()
            pipeline_elapsed_s = time.perf_counter() - pipeline_started
            if not reconstructing and request.execution.save_cosmic_mask:
                cosmic = getattr(result.detector_result, "cosmic_metadata", None)
                mask = None if cosmic is None else getattr(cosmic, "mask", None)
                if mask is not None:
                    mask_array = _as_numpy(mask)
                    if mask_array.ndim == 3 and mask_array.shape[0] == 1:
                        mask_array = mask_array[0]
                    mask_path = (
                        request.run_dir
                        / "cosmic_events"
                        / f"frame_{frame_index:06d}_mask.npy"
                    )
                    mask_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(mask_path, mask_array)
            if not reconstructing and request.execution.save_stellar_mean:
                stellar_mean = result.renderer_components.get("stellar_mean")
                if stellar_mean is None:
                    raise KeyError("Photsim7 did not return stellar_mean")
                np.save(
                    request.run_dir
                    / "frames"
                    / f"frame_{frame_index:06d}_stellar_mean_e.npy",
                    _as_numpy(stellar_mean).astype(np.float32),
                )
            if not reconstructing:
                peak_cuda_allocated_mb = (
                    None
                    if torch is None
                    else float(torch.cuda.max_memory_allocated() / 1024**2)
                )
                peak_cuda_reserved_mb = (
                    None
                    if torch is None
                    else float(torch.cuda.max_memory_reserved() / 1024**2)
                )
                _record_frame_metrics(
                    request.run_dir,
                    frame_index,
                    rank=request.rank,
                    device=request.execution.device,
                    n_stars=int(catalog.n_sources),
                    pipeline_elapsed_s=pipeline_elapsed_s,
                    total_elapsed_s=time.perf_counter() - frame_started,
                    peak_cuda_allocated_mb=peak_cuda_allocated_mb,
                    peak_cuda_reserved_mb=peak_cuda_reserved_mb,
                )
                if not frame_is_complete(
                    request.run_dir,
                    frame_index,
                    expected_shape=expected_shape,
                    expected_spec=request.spec,
                ):
                    raise RuntimeError(
                        f"Photsim7 artifacts for frame {frame_index} failed "
                        "readback validation"
                    )
            if shared_plan is not None:
                from et_mainsim.shared_exposure import (
                    SharedExposureReferenceDriftError,
                    array_c_order_fingerprint,
                    assert_exact_array_match,
                    assert_exact_parent_crop,
                )

                batch = shared_batch_by_frame[frame_index]
                batch_index = batch.batch_index
                if batch_index not in shared_snapshots_by_batch:
                    active_shard_paths = _shared_exposure_batch_shard_paths(
                        batch,
                        plan_content_sha256=shared_plan["content_sha256"],
                        product_keys=shared.product_keys,
                    )
                    active_snapshots = _inspect_shared_exposure_shards(
                        api=api,
                        request=request,
                        plan=shared_plan,
                        windows=shared_windows,
                        batch=batch,
                        shard_paths=active_shard_paths,
                    )
                    active_fingerprints = _shared_exposure_complete_item_fingerprints(
                        api=api,
                        snapshots=active_snapshots,
                        frame_indices=batch.frame_ids,
                        target_source_ids=shared.target_source_ids,
                    )
                    shared_shard_paths_by_batch[batch_index] = active_shard_paths
                    shared_snapshots_by_batch[batch_index] = active_snapshots
                    existing_fingerprints_by_batch[batch_index] = active_fingerprints
                    validated_complete_items_by_batch[batch_index] = set()
                    expected_parent_fingerprints_by_batch[batch_index] = {}
                    expected_final_fingerprints_by_batch[batch_index] = {
                        product_key: dict(active_fingerprints.get(product_key, {}))
                        for product_key in shared.product_keys
                    }
                    pending_items_by_batch[batch_index] = []
                    shared_writers_by_batch[batch_index] = {}
                    shared_guard_caches_by_batch[batch_index] = (
                        SharedExposureStorageGuardCache()
                    )
                shared_shard_paths = shared_shard_paths_by_batch[batch_index]
                shared_snapshots = shared_snapshots_by_batch[batch_index]
                existing_fingerprints = existing_fingerprints_by_batch[batch_index]
                validated_complete_items = validated_complete_items_by_batch[
                    batch_index
                ]
                expected_parent_fingerprints = expected_parent_fingerprints_by_batch[
                    batch_index
                ]
                expected_final_fingerprints = expected_final_fingerprints_by_batch[
                    batch_index
                ]
                parent_path, _, _ = _artifact_paths(request.run_dir, frame_index)
                parent_array = np.load(parent_path, allow_pickle=False)
                assert_exact_array_match(
                    parent_array,
                    _as_numpy(result.frame_products.final_frame.array),
                )
                expected_parent_fingerprints[frame_index] = array_c_order_fingerprint(
                    parent_array
                )
                frame_products: list[
                    tuple[int, Any, str, Path, np.ndarray, str, str]
                ] = []
                for target_source_id, window in shared_windows.items():
                    crop = api.shared_exposure_crop_v1(
                        result,
                        window,
                        api.SharedExposureTargetIdentity(
                            target_source_id=target_source_id,
                            detector_id=str(request.spec.detector.detector_id),
                            frame_index=frame_index,
                        ),
                        product_keys=shared.product_keys,
                        materialize_numpy=True,
                    )
                    for product_key, shard_path in shared_shard_paths.items():
                        product_array, unit, domain = _shared_exposure_crop_product(
                            crop,
                            product_key,
                        )
                        if product_key == "final_stamp":
                            assert_exact_parent_crop(
                                parent_array,
                                product_array,
                                window,
                            )
                        snapshot = shared_snapshots.get(product_key)
                        if snapshot is not None and (
                            snapshot.dtype != product_array.dtype
                            or snapshot.unit != unit
                            or snapshot.domain != domain
                        ):
                            raise SharedExposureReferenceDriftError(
                                "shared-exposure product contract drift detected "
                                f"for {product_key}"
                            )
                        item_key = (target_source_id, frame_index)
                        previous = existing_fingerprints.get(product_key, {}).get(
                            item_key
                        )
                        if previous is not None:
                            _assert_shared_exposure_fingerprint(
                                previous,
                                product_array,
                            )
                            validated_complete_items.add(
                                (product_key, target_source_id, frame_index)
                            )
                        elif snapshot is not None and snapshot.is_final:
                            raise SharedExposureReferenceDriftError(
                                "published shared-exposure shard is missing an "
                                f"expected complete item: {product_key} {item_key}"
                            )
                        expected_final_fingerprints[product_key][item_key] = (
                            array_c_order_fingerprint(product_array)
                        )
                        frame_products.append(
                            (
                                target_source_id,
                                crop,
                                product_key,
                                shard_path,
                                product_array,
                                unit,
                                domain,
                            )
                        )

                # Validate every pre-existing COMPLETE sibling before mutating
                # any missing item.  Missing items remain in memory until the
                # entire batch has passed exact validation.
                for (
                    target_source_id,
                    crop,
                    product_key,
                    shard_path,
                    product_array,
                    unit,
                    domain,
                ) in frame_products:
                    item_key = (target_source_id, frame_index)
                    if (
                        existing_fingerprints.get(product_key, {}).get(item_key)
                        is not None
                    ):
                        continue
                    pending_items_by_batch[batch_index].append(
                        (
                            target_source_id,
                            crop,
                            product_key,
                            shard_path,
                            product_array,
                            unit,
                            domain,
                        )
                    )
            rendered.append(frame_index)
            if shared_plan is not None:
                batch = shared_batch_by_frame[frame_index]
                batch_render_frames = render_frames_by_batch[batch.batch_index]
                if frame_index == batch_render_frames[-1]:
                    _write_missing_shared_exposure_batch_items(
                        api=api,
                        request=request,
                        plan=shared_plan,
                        windows=shared_windows,
                        batch=batch,
                        initial_snapshots=shared_snapshots_by_batch[batch.batch_index],
                        writers=shared_writers_by_batch[batch.batch_index],
                        pending_items=pending_items_by_batch[batch.batch_index],
                    )
                    _finalize_shared_exposure_batch(
                        api=api,
                        request=request,
                        plan=shared_plan,
                        windows=shared_windows,
                        batch=batch,
                        shard_paths=shared_shard_paths_by_batch[batch.batch_index],
                        initial_snapshots=shared_snapshots_by_batch[batch.batch_index],
                        writers=shared_writers_by_batch[batch.batch_index],
                        validated_complete_items=(
                            validated_complete_items_by_batch[batch.batch_index]
                        ),
                        expected_parent_fingerprints=(
                            expected_parent_fingerprints_by_batch[batch.batch_index]
                        ),
                        expected_final_fingerprints=(
                            expected_final_fingerprints_by_batch[batch.batch_index]
                        ),
                        rendered_frames=batch_render_frames,
                        frame_modes=frame_modes,
                        storage_guard_cache=shared_guard_caches_by_batch[
                            batch.batch_index
                        ],
                    )
                    shared_writers_by_batch.pop(batch.batch_index).clear()
                    pending_items_by_batch.pop(batch.batch_index).clear()
                    existing_fingerprints_by_batch.pop(batch.batch_index, None)
                    validated_complete_items_by_batch.pop(batch.batch_index, None)
                    expected_parent_fingerprints_by_batch.pop(
                        batch.batch_index,
                        None,
                    )
                    expected_final_fingerprints_by_batch.pop(
                        batch.batch_index,
                        None,
                    )
                    shared_snapshots_by_batch.pop(batch.batch_index).clear()
                    shared_shard_paths_by_batch.pop(batch.batch_index, None)
                    shared_guard_caches_by_batch.pop(batch.batch_index).clear()
    except BaseException as primary_error:
        close_errors = _close_shared_exposure_writers(shared_writers_by_batch)
        for batch_index, product_key, close_error in close_errors:
            primary_error.add_note(
                "shared-exposure writer close failed for "
                f"batch {batch_index}, product {product_key!r}: "
                f"{type(close_error).__name__}: {close_error}"
            )
        raise
    else:
        _raise_shared_exposure_close_errors(
            _close_shared_exposure_writers(shared_writers_by_batch)
        )

    worker_result = WorkerResult(
        rank=request.rank,
        rendered=tuple(rendered),
        skipped=tuple(skipped),
        elapsed_s=time.perf_counter() - started,
    )
    _atomic_json(
        request.run_dir / f"worker_{request.rank:02d}_done.json",
        {
            "schema_id": "et_mainsim.full_frame_worker",
            "schema_version": 1,
            **worker_result.to_dict(),
        },
    )
    return worker_result


def _resolve_package_catalog(source_path: str) -> str:
    prefix = "package://"
    if not source_path.startswith(prefix):
        return source_path
    name = source_path[len(prefix) :]
    if not name or "/" in name or "\\" in name:
        raise ValueError(f"Invalid packaged catalog reference {source_path!r}")
    return str(resource_path(name))


def resolve_simulation_spec(
    spec: Any,
    *,
    paths: ResolvedRunPaths,
    catalog_cache: Path,
    frames: int | None = None,
    target_epoch_jyear: float | None = None,
    run_seed: int | None = None,
    device: str | None = None,
) -> Any:
    resolved_frames = (
        int(spec.observation.resolved_n_frames) if frames is None else int(frames)
    )
    if resolved_frames <= 0:
        raise ValueError("frames must be positive")
    sampling = spec.observation.sampling_interval.to(u.s)
    catalog_updates: dict[str, Any] = {
        "cache_path": str(catalog_cache),
        "source_path": _resolve_package_catalog(spec.catalog.source_path),
    }
    if target_epoch_jyear is not None:
        catalog_updates["target_epoch_jyear"] = float(target_epoch_jyear)
    if spec.catalog.source_type == "et_focalplane_query":
        if paths.catalog_path is not None:
            catalog_updates["source_path"] = str(paths.catalog_path)
        if paths.focalplane_registry is not None:
            catalog_updates["registry_data_dir"] = str(paths.focalplane_registry)
            options = dict(spec.catalog.query_options)
            options["et_focalplane_src"] = str(paths.focalplane_registry.parent / "src")
            catalog_updates["query_options"] = options

    return replace(
        spec,
        observation=replace(
            spec.observation,
            observing_duration=resolved_frames * sampling,
            n_frames=resolved_frames,
            frame_start_s=None,
        ),
        catalog=replace(spec.catalog, **catalog_updates),
        psf=replace(
            spec.psf,
            compute_device=spec.psf.compute_device if device is None else device,
        ),
        rng=replace(
            spec.rng,
            run_seed=spec.rng.run_seed if run_seed is None else int(run_seed),
        ),
    )


def build_run_plan(
    *,
    preset_name: str,
    run_config: RunConfig,
    spec: Any,
    repo_root: Path | str,
    env: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    frames: int | None = None,
    frame_indices: str | tuple[int, ...] | None = None,
    target_epoch_jyear: float | None = None,
    run_seed: int | None = None,
) -> FullFrameRunPlan:
    paths = run_config.resolve_paths(env=env, cwd=cwd)
    run_dir = paths.output_root / run_config.run_id
    catalog_cache = paths.catalog_cache or run_dir / "cache" / "stars.npz"
    resolved_spec = resolve_simulation_spec(
        spec,
        paths=paths,
        catalog_cache=catalog_cache,
        frames=frames,
        target_epoch_jyear=target_epoch_jyear,
        run_seed=run_seed,
        device=run_config.execution.device,
    )
    requested = (
        run_config.execution.frame_indices if frame_indices is None else frame_indices
    )
    selected = parse_frame_indices(
        requested,
        total_frames=resolved_spec.observation.resolved_n_frames,
    )
    return FullFrameRunPlan(
        preset_name=preset_name,
        run_config=run_config,
        paths=paths,
        spec=resolved_spec,
        run_dir=run_dir,
        catalog_cache=catalog_cache,
        frame_indices=selected,
        repo_root=Path(repo_root).resolve(),
    )


def preflight(plan: FullFrameRunPlan) -> None:
    if plan.paths.data_root is None:
        raise ValueError("ET_DATA_DIR or paths.data_root is required to run")
    if not plan.paths.data_root.is_dir():
        raise FileNotFoundError(
            f"Photsim7 data root does not exist: {plan.paths.data_root}"
        )
    cache_available = (
        plan.catalog_cache.is_file()
        and not plan.run_config.execution.force_catalog_cache
    )
    catalog = plan.spec.catalog
    if catalog.source_type == "et_focalplane_query":
        if (
            not catalog.registry_data_dir
            or not Path(catalog.registry_data_dir).is_dir()
        ):
            raise FileNotFoundError(
                "ET_FOCALPLANE_ROOT or paths.focalplane_registry must reference focal-plane data"
            )
        if cache_available:
            return
        if not catalog.source_path or not Path(catalog.source_path).is_dir():
            raise FileNotFoundError(
                "GAIA_CATALOG_DIR or paths.catalog_path must reference a catalog directory"
            )
        focalplane_src = Path(catalog.query_options["et_focalplane_src"])
        if not focalplane_src.is_dir():
            raise FileNotFoundError(
                f"ET focal-plane source does not exist: {focalplane_src}"
            )
    elif cache_available:
        return
    elif catalog.source_type != "prepared" and not Path(catalog.source_path).is_file():
        raise FileNotFoundError(f"Catalog source does not exist: {catalog.source_path}")


def prepare_catalog(plan: FullFrameRunPlan, *, science_api: Any | None = None) -> Any:
    api = _science_api() if science_api is None else science_api
    if plan.paths.data_root is None:
        raise ValueError("data_root is required")
    if plan.run_config.execution.force_catalog_cache:
        plan.catalog_cache.unlink(missing_ok=True)
    registry = api.DataRegistry(data_root=plan.paths.data_root)
    return api.build_catalog_from_spec(plan.spec, data_registry=registry)


def _write_worker_request(path: Path, request: WorkerRequest) -> None:
    _atomic_json(path, request.to_json_dict())


def run_worker_request_file(path: Path | str) -> WorkerResult:
    with Path(path).open("r", encoding="utf-8") as handle:
        request = WorkerRequest.from_json_dict(json.load(handle))
    return run_worker(request)


def _launch_subprocess_workers(
    plan: FullFrameRunPlan,
    *,
    shared_exposure_overwrite_prepared: bool = False,
) -> list[WorkerResult]:
    if plan.paths.data_root is None:
        raise ValueError("data_root is required")
    assignments = worker_assignments(plan.run_config.execution)
    request_dir = plan.run_dir / "worker_requests"
    log_dir = plan.run_dir / "logs"
    request_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    processes: list[tuple[Any, subprocess.Popen[Any], Any, Path]] = []
    for assignment in assignments:
        request = WorkerRequest(
            spec=plan.spec,
            execution=plan.run_config.execution,
            run_dir=plan.run_dir,
            data_root=plan.paths.data_root,
            catalog_cache=plan.catalog_cache,
            frame_indices=plan.frame_indices,
            shared_exposure_stamps=(plan.run_config.workload.shared_exposure_stamps),
            shared_exposure_overwrite_prepared=(shared_exposure_overwrite_prepared),
            rank=assignment.rank,
            world_size=assignment.world_size,
        )
        request_path = request_dir / f"worker_{assignment.rank:02d}.json"
        _write_worker_request(request_path, request)
        log_path = log_dir / f"worker_{assignment.rank:02d}.log"
        log_handle = log_path.open("w", encoding="utf-8")
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        if assignment.visible_device is not None:
            environment["CUDA_VISIBLE_DEVICES"] = assignment.visible_device
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "et_mainsim",
                "_worker",
                "--request",
                str(request_path),
            ],
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append((assignment, process, log_handle, log_path))

    failures: list[str] = []
    for assignment, process, log_handle, log_path in processes:
        return_code = process.wait()
        log_handle.close()
        if return_code != 0:
            failures.append(
                f"rank {assignment.rank} exited {return_code}; see {log_path}"
            )
    if failures:
        raise RuntimeError("Worker failures: " + "; ".join(failures))

    results = []
    for assignment in assignments:
        with (plan.run_dir / f"worker_{assignment.rank:02d}_done.json").open(
            "r", encoding="utf-8"
        ) as handle:
            payload = json.load(handle)
        results.append(
            WorkerResult(
                rank=assignment.rank,
                rendered=tuple(payload["rendered"]),
                skipped=tuple(payload["skipped"]),
                elapsed_s=float(payload["elapsed_s"]),
            )
        )
    return results


def _manifest_execution(plan: FullFrameRunPlan) -> dict[str, Any]:
    return {
        **plan.run_config.execution.to_dict(),
        "paths": plan.paths.to_dict(),
        "frame_indices": list(plan.frame_indices),
    }


def _full_frame_product_contract() -> dict[str, Any]:
    from photsim7.artifacts import (
        SHARED_EXPOSURE_IMAGE_SHARD_SCHEMA_ID,
        SHARED_EXPOSURE_IMAGE_SHARD_SCHEMA_VERSION,
    )
    from photsim7.frame_products import (
        FRAME_PRODUCT_SCHEMA_ID,
        FRAME_PRODUCT_SCHEMA_VERSION,
    )
    from photsim7.geometry_truth import SOURCE_GEOMETRY_TRUTH_SCHEMA_ID
    from photsim7.psf.selection_truth import PSF_SELECTION_TRUTH_SCHEMA_ID
    from photsim7.selection_artifacts import (
        CADENCE_SELECTION_TRUTH_SCHEMA_ID,
        CADENCE_SELECTION_TRUTH_SCHEMA_VERSION,
    )
    from photsim7.shared_exposure import (
        SHARED_EXPOSURE_CROP_SCHEMA_ID,
        SHARED_EXPOSURE_CROP_SCHEMA_VERSION,
    )
    from photsim7.source_pixel_geometry import (
        FULL_FRAME_SOURCE_PIXEL_GEOMETRY_SCHEMA_ID,
        FULL_FRAME_SOURCE_PIXEL_GEOMETRY_SCHEMA_VERSION,
    )

    return {
        "frame_product_schema_id": FRAME_PRODUCT_SCHEMA_ID,
        "frame_product_schema_version": FRAME_PRODUCT_SCHEMA_VERSION,
        "source_geometry_truth_schema_id": SOURCE_GEOMETRY_TRUTH_SCHEMA_ID,
        "psf_selection_truth_schema_id": PSF_SELECTION_TRUTH_SCHEMA_ID,
        "cadence_selection_truth_schema_id": (CADENCE_SELECTION_TRUTH_SCHEMA_ID),
        "cadence_selection_truth_schema_version": (
            CADENCE_SELECTION_TRUTH_SCHEMA_VERSION
        ),
        "full_frame_source_pixel_geometry_schema_id": (
            FULL_FRAME_SOURCE_PIXEL_GEOMETRY_SCHEMA_ID
        ),
        "full_frame_source_pixel_geometry_schema_version": (
            FULL_FRAME_SOURCE_PIXEL_GEOMETRY_SCHEMA_VERSION
        ),
        "shared_exposure_crop_schema_id": SHARED_EXPOSURE_CROP_SCHEMA_ID,
        "shared_exposure_crop_schema_version": (SHARED_EXPOSURE_CROP_SCHEMA_VERSION),
        "shared_exposure_image_shard_schema_id": (
            SHARED_EXPOSURE_IMAGE_SHARD_SCHEMA_ID
        ),
        "shared_exposure_image_shard_schema_version": (
            SHARED_EXPOSURE_IMAGE_SHARD_SCHEMA_VERSION
        ),
    }


def _full_frame_workload_identity(plan: FullFrameRunPlan) -> dict[str, Any]:
    payload = plan.run_config.workload.to_dict()
    payload["product_contract"] = _full_frame_product_contract()
    return payload


def _shared_exposure_incomplete_frames_for_worker(
    request: WorkerRequest,
    *,
    science_api: Any | None = None,
) -> tuple[int, ...]:
    if not request.shared_exposure_stamps.enabled:
        return ()
    api = _science_api() if science_api is None else science_api
    assigned = tuple(request.frame_indices[request.rank :: request.world_size])
    scope_contract = _scope_contract_for_spec(request.run_dir, request.spec)
    batches_by_scope = {
        scope_id: _shared_exposure_frame_batches(
            request,
            assigned,
            scope_id=scope_id,
            scope_contract=scope_contract,
        )
        for scope_id in scope_contract.scope_ids
    }
    from et_mainsim.shared_exposure import (
        SharedExposureStorageGuardCache,
        read_shared_exposure_frame_completion,
        read_shared_exposure_target_plan,
    )

    plan_path = _shared_exposure_plan_path(request.run_dir)
    if not plan_path.is_file():
        return assigned
    plan = read_shared_exposure_target_plan(plan_path)
    _validate_shared_exposure_plan_for_request(plan, request=request)
    windows = _shared_exposure_windows(plan, api=api)
    incomplete: set[int] = set()
    for scope_id, batches in batches_by_scope.items():
        parent_root = scope_contract.scope_root(scope_id)
        for batch in batches:
            storage_guard_cache = SharedExposureStorageGuardCache()
            shard_paths = _shared_exposure_batch_shard_paths(
                batch,
                plan_content_sha256=plan["content_sha256"],
                product_keys=request.shared_exposure_stamps.product_keys,
            )
            snapshots = _inspect_shared_exposure_shards(
                api=api,
                request=request,
                plan=plan,
                windows=windows,
                batch=batch,
                shard_paths=shard_paths,
            )
            for frame_index in batch.frame_ids:
                marker_path = _shared_exposure_completion_path(
                    request.run_dir,
                    frame_index,
                    scope_id=scope_id,
                    scope_contract=scope_contract,
                )
                if not marker_path.is_file():
                    incomplete.add(frame_index)
                    continue
                marker = read_shared_exposure_frame_completion(
                    marker_path,
                    reference_root=request.run_dir,
                    storage_guard_cache=storage_guard_cache,
                )
                _validate_shared_exposure_marker_for_request(
                    marker,
                    request=request,
                    plan=plan,
                    frame_index=frame_index,
                    shard_paths=shard_paths,
                    parent_root=parent_root,
                )
                if not _shared_exposure_frame_has_complete_finals(
                    snapshots,
                    product_keys=request.shared_exposure_stamps.product_keys,
                    target_source_ids=request.shared_exposure_stamps.target_source_ids,
                    frame_index=frame_index,
                    complete_status=api.ItemStatus.COMPLETE,
                ):
                    incomplete.add(frame_index)
            snapshots.clear()
            storage_guard_cache.clear()
    return tuple(frame_index for frame_index in assigned if frame_index in incomplete)


def run_full_frame(
    plan: FullFrameRunPlan,
    *,
    prepare_catalog_only: bool = False,
    science_api: Any | None = None,
) -> dict[str, Any]:
    preflight(plan)
    scope_contract = _scope_contract_for_spec(plan.run_dir, plan.spec)
    store = RunManifestStore(plan.run_dir / "run_manifest.json")
    if (
        plan.run_dir.exists()
        and not store.path.exists()
        and any(plan.run_dir.iterdir())
    ):
        raise FileExistsError(
            f"Existing nonempty run directory {plan.run_dir} does not contain "
            "run_manifest.json; use a new run id"
        )
    plan.run_dir.mkdir(parents=True, exist_ok=True)
    execution_payload = _manifest_execution(plan)
    spec_payload = plan.spec.to_json_dict()
    workload_payload = _full_frame_workload_identity(plan)
    if store.path.exists():
        store.ensure_identity(
            workflow="et-full-frame",
            run_id=plan.run_config.run_id,
            simulation_spec=spec_payload,
            execution=execution_payload,
            workload=workload_payload,
        )
    else:
        from photsim7.frame_products import (
            FRAME_PRODUCT_SCHEMA_ID,
            FRAME_PRODUCT_SCHEMA_VERSION,
        )

        artifacts: dict[str, Any] = {
            "run_manifest": str(store.path),
            **scope_contract.to_manifest_artifacts(),
            "frame_product_schema_id": FRAME_PRODUCT_SCHEMA_ID,
            "frame_product_schema_version": FRAME_PRODUCT_SCHEMA_VERSION,
            "selection_truth": _full_frame_product_contract(),
        }
        shared = plan.run_config.workload.shared_exposure_stamps
        if shared.enabled:
            from et_mainsim.shared_exposure import (
                FRAME_COMPLETION_SCHEMA_ID,
                FRAME_COMPLETION_SCHEMA_VERSION,
                TARGET_PLAN_SCHEMA_ID,
                TARGET_PLAN_SCHEMA_VERSION,
            )

            shared_root = _shared_exposure_root(plan.run_dir)
            shared_artifacts: dict[str, Any] = {
                "root": str(shared_root),
                "target_plan": str(_shared_exposure_plan_path(plan.run_dir)),
                "target_plan_schema_id": TARGET_PLAN_SCHEMA_ID,
                "target_plan_schema_version": TARGET_PLAN_SCHEMA_VERSION,
                "frame_completion_schema_id": FRAME_COMPLETION_SCHEMA_ID,
                "frame_completion_schema_version": (FRAME_COMPLETION_SCHEMA_VERSION),
                "target_source_ids": list(shared.target_source_ids),
                "stamp_shape": list(shared.stamp_shape),
                "frames_per_shard": shared.frames_per_shard,
                "product_keys": list(shared.product_keys),
                "independent_stamp_simulation": False,
                "zero_new_rng_draws": True,
            }
            if scope_contract.is_single_scope:
                # Preserve the frozen single-scope manifest projection exactly.
                shared_artifacts.update(
                    {
                        "completion_markers": str(shared_root / "completion"),
                        "worker_shards": str(shared_root / "shards"),
                    }
                )
            else:
                shared_artifacts.update(
                    {
                        "layout": "per_scope_directories",
                        "scope_ids": list(scope_contract.scope_ids),
                        "image_level_combination": "forbidden",
                        "scopes": {
                            f"scope_{scope_id}": {
                                "root": str(
                                    _shared_exposure_scope_root(
                                        plan.run_dir,
                                        scope_id=scope_id,
                                        scope_contract=scope_contract,
                                    )
                                ),
                                "completion_markers": str(
                                    _shared_exposure_scope_root(
                                        plan.run_dir,
                                        scope_id=scope_id,
                                        scope_contract=scope_contract,
                                    )
                                    / "completion"
                                ),
                                "worker_shards": str(
                                    _shared_exposure_scope_root(
                                        plan.run_dir,
                                        scope_id=scope_id,
                                        scope_contract=scope_contract,
                                    )
                                    / "shards"
                                ),
                            }
                            for scope_id in scope_contract.scope_ids
                        },
                    }
                )
            artifacts["shared_exposure"] = shared_artifacts
        store.create(
            workflow="et-full-frame",
            preset=plan.preset_name,
            run_id=plan.run_config.run_id,
            simulation_spec=spec_payload,
            execution=execution_payload,
            workload=workload_payload,
            frame_plan={
                "requested": list(plan.frame_indices),
                "count": len(plan.frame_indices),
            },
            provenance=collect_provenance(plan.repo_root),
            artifacts=artifacts,
        )
    try:
        store.start_attempt(
            control={
                "resume": plan.run_config.execution.resume,
                "overwrite": plan.run_config.execution.overwrite,
                "force_catalog_cache": (plan.run_config.execution.force_catalog_cache),
                "progress": plan.run_config.execution.progress,
            }
        )
        catalog = prepare_catalog(plan, science_api=science_api)
        store.update(
            catalog={
                "cache_path": str(plan.catalog_cache),
                "n_sources": int(catalog.n_sources),
                "metadata": dict(catalog.metadata),
            }
        )
        if prepare_catalog_only:
            return store.transition(
                "completed",
                completion={"catalog_only": True, "n_sources": int(catalog.n_sources)},
            )

        shared_overwrite_prepared = bool(
            plan.run_config.workload.shared_exposure_stamps.enabled
            and plan.run_config.execution.overwrite
        )
        if shared_overwrite_prepared:
            _clear_shared_exposure_bundle_for_overwrite(plan.run_dir)

        if plan.run_config.execution.backend == "in-process":
            if plan.paths.data_root is None:
                raise ValueError("data_root is required")
            results = [
                run_worker(
                    WorkerRequest(
                        spec=plan.spec,
                        execution=plan.run_config.execution,
                        run_dir=plan.run_dir,
                        data_root=plan.paths.data_root,
                        catalog_cache=plan.catalog_cache,
                        frame_indices=plan.frame_indices,
                        shared_exposure_stamps=(
                            plan.run_config.workload.shared_exposure_stamps
                        ),
                        shared_exposure_overwrite_prepared=(shared_overwrite_prepared),
                    ),
                    science_api=science_api,
                )
            ]
        else:
            results = _launch_subprocess_workers(
                plan,
                shared_exposure_overwrite_prepared=shared_overwrite_prepared,
            )

        incomplete = [
            frame_index
            for frame_index in plan.frame_indices
            if not frame_completion(
                plan.run_dir,
                frame_index,
                expected_shape=tuple(plan.spec.detector.shape),
                expected_spec=plan.spec,
            ).is_complete
        ]
        if incomplete:
            raise RuntimeError(
                f"Incomplete frame artifacts after worker exit: {incomplete}"
            )
        shared_incomplete: list[int] = []
        if plan.run_config.workload.shared_exposure_stamps.enabled:
            world_size = max((result.rank for result in results), default=0) + 1
            if {result.rank for result in results} != set(range(world_size)):
                raise RuntimeError(
                    "Shared-exposure worker ranks are not contiguous from zero"
                )
            if plan.paths.data_root is None:
                raise ValueError("data_root is required")
            for result in results:
                shared_incomplete.extend(
                    _shared_exposure_incomplete_frames_for_worker(
                        WorkerRequest(
                            spec=plan.spec,
                            execution=plan.run_config.execution,
                            run_dir=plan.run_dir,
                            data_root=plan.paths.data_root,
                            catalog_cache=plan.catalog_cache,
                            frame_indices=plan.frame_indices,
                            shared_exposure_stamps=(
                                plan.run_config.workload.shared_exposure_stamps
                            ),
                            rank=result.rank,
                            world_size=world_size,
                        ),
                        science_api=science_api,
                    )
                )
        if shared_incomplete:
            raise RuntimeError(
                "Incomplete shared-exposure artifacts after worker exit: "
                f"{sorted(shared_incomplete)}"
            )
        rendered = sum(len(result.rendered) for result in results)
        skipped = sum(len(result.skipped) for result in results)
        return store.transition(
            "completed",
            completion={
                "requested": len(plan.frame_indices),
                "completed": len(plan.frame_indices),
                "rendered": rendered,
                "skipped": skipped,
                "workers": [result.to_dict() for result in results],
            },
        )
    except BaseException as error:
        if store.load()["status"] == "running":
            store.fail(error)
        raise


__all__ = [
    "FullFrameRunPlan",
    "WorkerRequest",
    "WorkerResult",
    "build_run_plan",
    "frame_is_complete",
    "preflight",
    "prepare_catalog",
    "resolve_simulation_spec",
    "run_full_frame",
    "run_worker",
    "run_worker_request_file",
]
