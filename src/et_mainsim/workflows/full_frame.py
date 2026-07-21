from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
from astropy import units as u

from et_mainsim.config import (
    ExecutionConfig,
    ResolvedRunPaths,
    RunConfig,
    parse_frame_indices,
    worker_assignments,
)
from et_mainsim.manifest import RunManifestStore
from et_mainsim.presets import resource_path
from et_mainsim.provenance import collect_provenance


_SELECTION_TRUTH_SCOPE = "geometry_psf_and_jitter_selection_truth_only"
_ET_FULL_FRAME_SPACECRAFT_ID = "et"
_ET_FULL_FRAME_ABSOLUTE_RAW_FRAME_START_INDEX = 0


@dataclass(frozen=True)
class WorkerRequest:
    spec: Any
    execution: ExecutionConfig
    run_dir: Path
    data_root: Path
    catalog_cache: Path
    frame_indices: tuple[int, ...]
    rank: int = 0
    world_size: int = 1

    def __post_init__(self) -> None:
        if int(self.rank) < 0:
            raise ValueError("rank must be non-negative")
        if int(self.world_size) <= 0 or int(self.rank) >= int(self.world_size):
            raise ValueError("rank must be smaller than positive world_size")
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
            "schema_version": 1,
            "simulation_spec": self.spec.to_json_dict(),
            "execution": self.execution.to_dict(),
            "run_dir": str(self.run_dir),
            "data_root": str(self.data_root),
            "catalog_cache": str(self.catalog_cache),
            "frame_indices": list(self.frame_indices),
            "rank": self.rank,
            "world_size": self.world_size,
        }

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, Any]) -> "WorkerRequest":
        if payload.get("schema_id") != "et_mainsim.full_frame_worker_request":
            raise ValueError("Unsupported full-frame worker request")
        if int(payload.get("schema_version", 0)) != 1:
            raise ValueError("Unsupported full-frame worker request version")
        from photsim7.specs import SimulationSpec

        return cls(
            spec=SimulationSpec.from_json_dict(payload["simulation_spec"]),
            execution=ExecutionConfig(**dict(payload["execution"])),
            run_dir=Path(payload["run_dir"]),
            data_root=Path(payload["data_root"]),
            catalog_cache=Path(payload["catalog_cache"]),
            frame_indices=tuple(payload["frame_indices"]),
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
    from photsim7.catalog_sources import PreparedStarCatalog, StarCatalogCache
    from photsim7.data_registry import DataRegistry
    from photsim7.full_frame_artifacts import (
        FullFrameArtifactOptions,
        FullFrameArtifactWriter,
    )
    from photsim7.full_frame_pipeline import run_single_cadence_full_frame
    from photsim7.selection_artifacts import (
        cadence_selection_truth_relative_path,
        read_cadence_selection_truth,
    )
    from photsim7.simulation_services import (
        build_catalog_from_spec,
        build_full_frame_services,
    )

    return SimpleNamespace(
        PreparedStarCatalog=PreparedStarCatalog,
        StarCatalogCache=StarCatalogCache,
        DataRegistry=DataRegistry,
        FullFrameArtifactOptions=FullFrameArtifactOptions,
        FullFrameArtifactWriter=FullFrameArtifactWriter,
        build_catalog_from_spec=build_catalog_from_spec,
        build_full_frame_services=build_full_frame_services,
        run_single_cadence_full_frame=run_single_cadence_full_frame,
        cadence_selection_truth_relative_path=(
            cadence_selection_truth_relative_path
        ),
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
        selection.get("schema_id")
        == "photsim7.cadence_selection_truth.v1"
        and int(selection.get("schema_version", 0)) == 1
        and selection.get("verification_status") == "unavailable"
        and selection.get("science_conformance_claim") is False
        and selection.get("science_conformance_claim_scope")
        == _SELECTION_TRUTH_SCOPE
        and selection.get("requested_science_profile_id")
        == expected_spec.science_profile.profile_id
        and selection.get("missing_components")
        == ["jitter_model_selection_truth"]
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
        selection.get("schema_id")
        != "photsim7.cadence_selection_truth.v1"
        or int(selection.get("schema_version", 0)) != 1
        or selection.get("verification_status") != "persisted_and_verified"
        or selection.get("science_conformance_claim_scope")
        != _SELECTION_TRUTH_SCOPE
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
    absolute_raw_frame_index = (
        _ET_FULL_FRAME_ABSOLUTE_RAW_FRAME_START_INDEX + int(frame_index)
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
    if (
        truth.detector_id != str(expected_spec.detector.detector_id)
        or truth.local_frame_index != int(frame_index)
        or truth.absolute_raw_frame_index != absolute_raw_frame_index
        or truth.spacecraft_id != _ET_FULL_FRAME_SPACECRAFT_ID
        or truth.science_realization_id
        != int(expected_spec.science_profile.science_realization_id)
        or truth.science_conformance_claim
        is not selection["science_conformance_claim"]
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


def _has_partial_artifacts(run_dir: Path, frame_index: int) -> bool:
    absolute_raw_frame_index = (
        _ET_FULL_FRAME_ABSOLUTE_RAW_FRAME_START_INDEX + int(frame_index)
    )
    cadence_path = (
        run_dir
        / "selection_truth"
        / "cadence"
        / f"frame_{absolute_raw_frame_index:09d}.json"
    )
    return any(
        path.exists()
        for path in (*_artifact_paths(run_dir, frame_index), cadence_path)
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
    if rank != 0 or getattr(services, "effect_timeseries", None) is None:
        return
    timeseries = services.effect_timeseries
    np.savez_compressed(run_dir / "effects_timeseries.npz", **timeseries.to_arrays())
    _atomic_json(
        run_dir / "effects_timeseries.metadata.json",
        timeseries.to_metadata(),
    )


def run_worker(
    request: WorkerRequest, *, science_api: Any | None = None
) -> WorkerResult:
    api = _science_api() if science_api is None else science_api
    started = time.perf_counter()
    expected_shape = tuple(int(value) for value in request.spec.detector.shape)
    assigned = request.frame_indices[request.rank :: request.world_size]
    to_render: list[int] = []
    skipped: list[int] = []
    for frame_index in assigned:
        complete = frame_is_complete(
            request.run_dir,
            frame_index,
            expected_shape=expected_shape,
            expected_spec=request.spec,
        )
        if request.execution.resume and complete:
            skipped.append(frame_index)
            continue
        if (
            not request.execution.resume
            and not request.execution.overwrite
            and _has_partial_artifacts(request.run_dir, frame_index)
        ):
            raise FileExistsError(
                f"Frame {frame_index} already has artifacts; use resume or overwrite"
            )
        to_render.append(frame_index)

    request.run_dir.mkdir(parents=True, exist_ok=True)
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
    services = api.build_full_frame_services(
        request.spec,
        catalog=catalog,
        data_registry=registry,
    )
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
            "render_frames": to_render,
            "skipped_frames": skipped,
            "catalog_cache": str(request.catalog_cache),
            "n_sources": int(catalog.n_sources),
        },
    )

    rendered: list[int] = []
    for frame_index in to_render:
        frame_started = time.perf_counter()
        options = api.FullFrameArtifactOptions(
            save_frame_summaries=True,
            save_cosmic_events=True,
            save_bias=bool(request.spec.artifacts.save_bias_artifacts),
            save_preview=frame_index < request.execution.preview_count,
        )
        writer = api.FullFrameArtifactWriter(request.run_dir, options=options)
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
                    request.spec.sky.scattered_light.to_value(u.electron / u.s / u.pix)
                ),
                "enable_dark_current": True,
                "progress": request.execution.progress,
            },
            worker_rank=request.rank,
            rng_trace_scope={"run_id": request.run_dir.name},
            artifact_writer=writer,
        )
        if torch is not None:
            torch.cuda.synchronize()
        pipeline_elapsed_s = time.perf_counter() - pipeline_started
        if request.execution.save_cosmic_mask:
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
        if request.execution.save_stellar_mean:
            stellar_mean = result.renderer_components.get("stellar_mean")
            if stellar_mean is None:
                raise KeyError("Photsim7 did not return stellar_mean")
            np.save(
                request.run_dir
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
            None if torch is None else float(torch.cuda.max_memory_reserved() / 1024**2)
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
                f"Photsim7 artifacts for frame {frame_index} failed readback validation"
            )
        rendered.append(frame_index)

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


def _launch_subprocess_workers(plan: FullFrameRunPlan) -> list[WorkerResult]:
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

    return {
        "frame_product_schema_id": FRAME_PRODUCT_SCHEMA_ID,
        "frame_product_schema_version": FRAME_PRODUCT_SCHEMA_VERSION,
        "source_geometry_truth_schema_id": SOURCE_GEOMETRY_TRUTH_SCHEMA_ID,
        "psf_selection_truth_schema_id": PSF_SELECTION_TRUTH_SCHEMA_ID,
        "cadence_selection_truth_schema_id": (
            CADENCE_SELECTION_TRUTH_SCHEMA_ID
        ),
        "cadence_selection_truth_schema_version": (
            CADENCE_SELECTION_TRUTH_SCHEMA_VERSION
        ),
    }


def _full_frame_workload_identity(plan: FullFrameRunPlan) -> dict[str, Any]:
    payload = plan.run_config.workload.to_dict()
    payload["product_contract"] = _full_frame_product_contract()
    return payload


def run_full_frame(
    plan: FullFrameRunPlan,
    *,
    prepare_catalog_only: bool = False,
    science_api: Any | None = None,
) -> dict[str, Any]:
    preflight(plan)
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
            artifacts={
                "run_manifest": str(store.path),
                "frames": str(plan.run_dir / "frames"),
                "frame_summaries": str(plan.run_dir / "frame_summaries"),
                "frame_product_schema_id": FRAME_PRODUCT_SCHEMA_ID,
                "frame_product_schema_version": FRAME_PRODUCT_SCHEMA_VERSION,
                "selection_truth": _full_frame_product_contract(),
            },
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
                    ),
                    science_api=science_api,
                )
            ]
        else:
            results = _launch_subprocess_workers(plan)

        incomplete = [
            frame_index
            for frame_index in plan.frame_indices
            if not frame_is_complete(
                plan.run_dir,
                frame_index,
                expected_shape=tuple(plan.spec.detector.shape),
                expected_spec=plan.spec,
            )
        ]
        if incomplete:
            raise RuntimeError(
                f"Incomplete frame artifacts after worker exit: {incomplete}"
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
