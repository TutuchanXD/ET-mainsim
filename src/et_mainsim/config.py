from __future__ import annotations

import math
import os
import re
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


EXECUTION_SCHEMA_ID = "et_mainsim.execution_config"
EXECUTION_SCHEMA_VERSION = 1
_BACKENDS = frozenset({"in-process", "local-subprocess", "local-ray"})
_DEVICES = frozenset({"cpu", "cuda"})
_WORKFLOW_KINDS = {
    "et-full-frame": "full-frame",
    "et-stamp": "stamp",
    "legacy-sim": "legacy",
}
_ENV_PATTERN = re.compile(
    r"\$(?:\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)\}|(?P<plain>[A-Za-z_][A-Za-z0-9_]*))"
)


@dataclass(frozen=True)
class RunPaths:
    output_root: str = ""
    data_root: str = ""
    catalog_path: str = ""
    focalplane_registry: str = ""
    catalog_cache: str = ""

    def __post_init__(self) -> None:
        for name in (
            "output_root",
            "data_root",
            "catalog_path",
            "focalplane_registry",
            "catalog_cache",
        ):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise ValueError(f"paths.{name} must be a string")


@dataclass(frozen=True)
class ResolvedRunPaths:
    output_root: Path
    data_root: Path | None
    catalog_path: Path | None
    focalplane_registry: Path | None
    catalog_cache: Path | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "output_root": str(self.output_root),
            "data_root": None if self.data_root is None else str(self.data_root),
            "catalog_path": None
            if self.catalog_path is None
            else str(self.catalog_path),
            "focalplane_registry": (
                None
                if self.focalplane_registry is None
                else str(self.focalplane_registry)
            ),
            "catalog_cache": (
                None if self.catalog_cache is None else str(self.catalog_cache)
            ),
        }


@dataclass(frozen=True)
class ExecutionConfig:
    backend: str = "in-process"
    device: str = "cpu"
    gpu_ids: tuple[str, ...] = field(default_factory=tuple)
    workers_per_device: int = 1
    frame_indices: tuple[int, ...] | None = None
    resume: bool = True
    overwrite: bool = False
    force_catalog_cache: bool = False
    preview_count: int = 1
    max_stars: int | None = None
    progress: bool = False
    save_cosmic_mask: bool = False
    save_stellar_mean: bool = False
    ray_actor_count: int = 1
    ray_num_cpus: int = 1
    ray_num_gpus: int = 0

    def __post_init__(self) -> None:
        backend = str(self.backend).strip().lower()
        device = str(self.device).strip().lower()
        gpu_ids = tuple(
            str(value).strip() for value in self.gpu_ids if str(value).strip()
        )
        if backend not in _BACKENDS:
            raise ValueError(f"backend must be one of {sorted(_BACKENDS)}")
        if device not in _DEVICES:
            raise ValueError(f"device must be one of {sorted(_DEVICES)}")
        if int(self.workers_per_device) <= 0:
            raise ValueError("workers_per_device must be positive")
        if bool(self.resume) and bool(self.overwrite):
            raise ValueError("resume and overwrite are mutually exclusive")
        if int(self.preview_count) < 0:
            raise ValueError("preview_count must be non-negative")
        if self.max_stars is not None and int(self.max_stars) <= 0:
            raise ValueError("max_stars must be positive when provided")
        if backend == "local-subprocess" and device == "cuda" and not gpu_ids:
            raise ValueError("local-subprocess CUDA execution requires gpu_ids")
        if backend == "in-process" and device != "cpu":
            raise ValueError(
                "in-process backend supports CPU only; use local-subprocess for CUDA"
            )
        if backend == "in-process" and int(self.workers_per_device) != 1:
            raise ValueError("in-process execution requires workers_per_device=1")
        if int(self.ray_actor_count) <= 0:
            raise ValueError("ray_actor_count must be positive")
        ray_num_cpus = float(self.ray_num_cpus)
        ray_num_gpus = float(self.ray_num_gpus)
        if not math.isfinite(ray_num_cpus) or ray_num_cpus <= 0.0:
            raise ValueError("ray_num_cpus must be finite and positive")
        if not math.isfinite(ray_num_gpus) or ray_num_gpus < 0.0:
            raise ValueError("ray_num_gpus must be finite and non-negative")
        if not ray_num_cpus.is_integer():
            raise ValueError("ray_num_cpus must be an integer")
        if not ray_num_gpus.is_integer():
            raise ValueError("ray_num_gpus must be an integer")
        if backend == "local-ray" and int(self.workers_per_device) != 1:
            raise ValueError("local-ray execution requires workers_per_device=1")

        indices = None
        if self.frame_indices is not None:
            indices = tuple(int(value) for value in self.frame_indices)
            if any(value < 0 for value in indices):
                raise ValueError("frame_indices must be non-negative")
            indices = tuple(dict.fromkeys(indices))

        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "gpu_ids", gpu_ids)
        object.__setattr__(self, "workers_per_device", int(self.workers_per_device))
        object.__setattr__(self, "frame_indices", indices)
        object.__setattr__(self, "resume", bool(self.resume))
        object.__setattr__(self, "overwrite", bool(self.overwrite))
        object.__setattr__(self, "force_catalog_cache", bool(self.force_catalog_cache))
        object.__setattr__(self, "preview_count", int(self.preview_count))
        object.__setattr__(
            self,
            "max_stars",
            None if self.max_stars is None else int(self.max_stars),
        )
        object.__setattr__(self, "progress", bool(self.progress))
        object.__setattr__(self, "save_cosmic_mask", bool(self.save_cosmic_mask))
        object.__setattr__(self, "save_stellar_mean", bool(self.save_stellar_mean))
        object.__setattr__(self, "ray_actor_count", int(self.ray_actor_count))
        object.__setattr__(self, "ray_num_cpus", int(ray_num_cpus))
        object.__setattr__(self, "ray_num_gpus", int(ray_num_gpus))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["gpu_ids"] = list(self.gpu_ids)
        payload["frame_indices"] = (
            None if self.frame_indices is None else list(self.frame_indices)
        )
        return payload


@dataclass(frozen=True)
class WorkerAssignment:
    rank: int
    world_size: int
    visible_device: str | None
    compute_device: str


@dataclass(frozen=True)
class FullFrameWorkload:
    kind: str = "full-frame"

    def __post_init__(self) -> None:
        if str(self.kind).strip().lower() != "full-frame":
            raise ValueError("full-frame workload kind must be 'full-frame'")
        object.__setattr__(self, "kind", "full-frame")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StampWorkload:
    kind: str = "stamp"
    input_mode: str = "catalog"
    input_table: str = ""
    target_source_ids: tuple[int, ...] = field(default_factory=tuple)
    target_limit: int = 0
    stamp_rows: int = 15
    stamp_cols: int = 15
    include_neighbors: bool = True
    save_raw: bool = True
    save_coadd: bool = True
    save_electron_components: bool = False

    def __post_init__(self) -> None:
        kind = str(self.kind).strip().lower()
        input_mode = str(self.input_mode).strip().lower()
        input_table = str(self.input_table).strip()
        if kind != "stamp":
            raise ValueError("stamp workload kind must be 'stamp'")
        if input_mode not in {"catalog", "table"}:
            raise ValueError("stamp input_mode must be 'catalog' or 'table'")
        if input_mode == "table" and not input_table:
            raise ValueError("stamp table input_mode requires input_table")
        if input_mode == "table" and bool(self.include_neighbors):
            raise ValueError(
                "stamp table input_mode requires include_neighbors=false"
            )
        if input_mode == "catalog" and input_table:
            raise ValueError("stamp catalog input_mode cannot set input_table")
        rows = int(self.stamp_rows)
        cols = int(self.stamp_cols)
        target_limit = int(self.target_limit)
        if rows <= 0 or cols <= 0:
            raise ValueError("stamp_rows and stamp_cols must be positive")
        if target_limit < 0:
            raise ValueError("target_limit must be non-negative")
        if not bool(self.save_raw) and not bool(self.save_coadd):
            raise ValueError("stamp workload must save raw, coadd, or both")
        target_ids = tuple(dict.fromkeys(int(value) for value in self.target_source_ids))
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "input_mode", input_mode)
        object.__setattr__(self, "input_table", input_table)
        object.__setattr__(self, "target_source_ids", target_ids)
        object.__setattr__(self, "target_limit", target_limit)
        object.__setattr__(self, "stamp_rows", rows)
        object.__setattr__(self, "stamp_cols", cols)
        object.__setattr__(self, "include_neighbors", bool(self.include_neighbors))
        object.__setattr__(self, "save_raw", bool(self.save_raw))
        object.__setattr__(self, "save_coadd", bool(self.save_coadd))
        object.__setattr__(
            self,
            "save_electron_components",
            bool(self.save_electron_components),
        )

    @property
    def stamp_shape(self) -> tuple[int, int]:
        return self.stamp_rows, self.stamp_cols

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["target_source_ids"] = list(self.target_source_ids)
        return payload


@dataclass(frozen=True)
class LegacyWorkload:
    kind: str = "legacy"
    run_count: int = 1
    stars_per_run: int = 1
    store_images: bool = False
    et_mag_min: float = 7.0
    et_mag_max: float = 17.0

    def __post_init__(self) -> None:
        kind = str(self.kind).strip().lower()
        run_count = int(self.run_count)
        stars_per_run = int(self.stars_per_run)
        et_mag_min = float(self.et_mag_min)
        et_mag_max = float(self.et_mag_max)
        if kind != "legacy":
            raise ValueError("legacy workload kind must be 'legacy'")
        if run_count <= 0 or stars_per_run <= 0:
            raise ValueError("run_count and stars_per_run must be positive")
        if not all(math.isfinite(value) for value in (et_mag_min, et_mag_max)):
            raise ValueError("legacy ET magnitude bounds must be finite")
        if et_mag_min > et_mag_max:
            raise ValueError("et_mag_min must not exceed et_mag_max")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "run_count", run_count)
        object.__setattr__(self, "stars_per_run", stars_per_run)
        object.__setattr__(self, "store_images", bool(self.store_images))
        object.__setattr__(self, "et_mag_min", et_mag_min)
        object.__setattr__(self, "et_mag_max", et_mag_max)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


WorkloadConfig = FullFrameWorkload | StampWorkload | LegacyWorkload


@dataclass(frozen=True)
class RunConfig:
    schema_id: str
    schema_version: int
    workflow: str
    run_id: str
    paths: RunPaths = field(default_factory=RunPaths)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    workload: WorkloadConfig = field(default_factory=FullFrameWorkload)
    source: str = "<memory>"

    def __post_init__(self) -> None:
        if self.schema_id != EXECUTION_SCHEMA_ID:
            raise ValueError(
                f"schema_id must be {EXECUTION_SCHEMA_ID!r}, got {self.schema_id!r}"
            )
        if int(self.schema_version) != EXECUTION_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {EXECUTION_SCHEMA_VERSION}")
        if not str(self.workflow).strip():
            raise ValueError("workflow must be non-empty")
        if not str(self.run_id).strip():
            raise ValueError("run_id must be non-empty")
        workflow = str(self.workflow).strip()
        expected_kind = _WORKFLOW_KINDS.get(workflow)
        if expected_kind is None:
            raise ValueError(
                f"workflow must be one of {sorted(_WORKFLOW_KINDS)}"
            )
        if self.workload.kind != expected_kind:
            raise ValueError(
                f"workload kind {self.workload.kind!r} does not match workflow "
                f"{workflow!r}"
            )
        backend = self.execution.backend
        if workflow == "legacy-sim" and backend != "local-ray":
            raise ValueError("legacy-sim requires the local-ray backend")
        if workflow != "legacy-sim" and backend == "local-ray":
            raise ValueError("local-ray backend is reserved for legacy-sim")
        object.__setattr__(self, "workflow", workflow)

    @classmethod
    def from_toml(cls, text: str | bytes, *, source: str = "<memory>") -> "RunConfig":
        payload = tomllib.loads(text.decode() if isinstance(text, bytes) else text)
        return cls.from_mapping(payload, source=source)

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        source: str = "<mapping>",
    ) -> "RunConfig":
        allowed = {
            "schema_id",
            "schema_version",
            "workflow",
            "run_id",
            "paths",
            "execution",
            "workload",
        }
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError(f"Unknown run config fields: {', '.join(unknown)}")

        path_payload = dict(payload.get("paths", {}))
        execution_payload = dict(payload.get("execution", {}))
        workload_payload = dict(payload.get("workload", {}))
        path_unknown = sorted(set(path_payload) - set(RunPaths.__dataclass_fields__))
        execution_unknown = sorted(
            set(execution_payload) - set(ExecutionConfig.__dataclass_fields__)
        )
        if path_unknown:
            raise ValueError(f"Unknown paths fields: {', '.join(path_unknown)}")
        if execution_unknown:
            raise ValueError(
                f"Unknown execution fields: {', '.join(execution_unknown)}"
            )
        workflow = str(payload.get("workflow", ""))
        expected_kind = _WORKFLOW_KINDS.get(workflow)
        kind = str(workload_payload.get("kind", expected_kind or "")).strip().lower()
        workload_types = {
            "full-frame": FullFrameWorkload,
            "stamp": StampWorkload,
            "legacy": LegacyWorkload,
        }
        workload_type = workload_types.get(kind)
        if workload_type is None:
            raise ValueError(f"Unknown workload kind {kind!r}")
        workload_unknown = sorted(
            set(workload_payload) - set(workload_type.__dataclass_fields__)
        )
        if workload_unknown:
            raise ValueError(
                f"Unknown workload fields: {', '.join(workload_unknown)}"
            )
        workload_payload.setdefault("kind", kind)
        return cls(
            schema_id=str(payload.get("schema_id", "")),
            schema_version=int(payload.get("schema_version", 0)),
            workflow=workflow,
            run_id=str(payload.get("run_id", "")),
            paths=RunPaths(**path_payload),
            execution=ExecutionConfig(**execution_payload),
            workload=workload_type(**workload_payload),
            source=source,
        )

    def resolve_paths(
        self,
        *,
        env: Mapping[str, str] | None = None,
        cwd: Path | str | None = None,
    ) -> ResolvedRunPaths:
        values = dict(os.environ if env is None else env)
        base = Path.cwd() if cwd is None else Path(cwd)
        output_value = self.paths.output_root or values.get("RESULTS_ROOT", "")
        output_root = _resolve_path(
            output_value or str(base / "results" / "et-mainsim"),
            env=values,
            cwd=base,
        )
        data_root = _optional_path(
            self.paths.data_root or values.get("ET_DATA_DIR", ""),
            env=values,
            cwd=base,
        )
        catalog_path = _optional_path(
            self.paths.catalog_path or values.get("GAIA_CATALOG_DIR", ""),
            env=values,
            cwd=base,
        )
        focalplane_value = self.paths.focalplane_registry
        if not focalplane_value and values.get("ET_FOCALPLANE_ROOT"):
            focalplane_value = str(Path(values["ET_FOCALPLANE_ROOT"]) / "data")
        focalplane_registry = _optional_path(
            focalplane_value,
            env=values,
            cwd=base,
        )
        catalog_cache = _optional_path(
            self.paths.catalog_cache or values.get("ET_CATALOG_CACHE", ""),
            env=values,
            cwd=base,
        )
        return ResolvedRunPaths(
            output_root=output_root,
            data_root=data_root,
            catalog_path=catalog_path,
            focalplane_registry=focalplane_registry,
            catalog_cache=catalog_cache,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "workflow": self.workflow,
            "run_id": self.run_id,
            "paths": asdict(self.paths),
            "execution": self.execution.to_dict(),
            "workload": self.workload.to_dict(),
            "source": self.source,
        }


def _expand_environment(value: str, env: Mapping[str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group("braced") or match.group("plain")
        if name not in env:
            raise ValueError(f"Path references unset environment variable {name}")
        return env[name]

    return _ENV_PATTERN.sub(replace, value)


def _resolve_path(value: str, *, env: Mapping[str, str], cwd: Path) -> Path:
    expanded = Path(_expand_environment(value, env)).expanduser()
    if not expanded.is_absolute():
        expanded = cwd / expanded
    return expanded.resolve(strict=False)


def _optional_path(
    value: str,
    *,
    env: Mapping[str, str],
    cwd: Path,
) -> Path | None:
    return None if not str(value).strip() else _resolve_path(value, env=env, cwd=cwd)


def parse_frame_indices(
    value: str | Sequence[int] | None,
    *,
    total_frames: int,
) -> tuple[int, ...]:
    total_frames = int(total_frames)
    if total_frames <= 0:
        raise ValueError("total_frames must be positive")
    if value is None or (isinstance(value, str) and not value.strip()):
        return tuple(range(total_frames))
    if isinstance(value, str):
        try:
            values = [int(token.strip()) for token in value.split(",") if token.strip()]
        except ValueError as exc:
            raise ValueError("frame indices must be comma-separated integers") from exc
    else:
        values = [int(item) for item in value]
    result = tuple(dict.fromkeys(values))
    if not result:
        raise ValueError("frame selection must not be empty")
    for frame_index in result:
        if frame_index < 0 or frame_index >= total_frames:
            raise ValueError(
                f"Frame index {frame_index} is outside 0..{total_frames - 1}"
            )
    return result


def worker_assignments(execution: ExecutionConfig) -> tuple[WorkerAssignment, ...]:
    if execution.backend == "local-ray":
        raise ValueError("local-ray execution is managed by the legacy workflow")
    if execution.backend == "in-process":
        return (
            WorkerAssignment(
                rank=0,
                world_size=1,
                visible_device=None,
                compute_device=execution.device,
            ),
        )
    if execution.device == "cuda":
        devices = tuple(
            device
            for device in execution.gpu_ids
            for _ in range(execution.workers_per_device)
        )
    else:
        devices = (None,) * execution.workers_per_device
    world_size = len(devices)
    return tuple(
        WorkerAssignment(
            rank=rank,
            world_size=world_size,
            visible_device=device,
            compute_device=execution.device,
        )
        for rank, device in enumerate(devices)
    )


__all__ = [
    "EXECUTION_SCHEMA_ID",
    "EXECUTION_SCHEMA_VERSION",
    "ExecutionConfig",
    "FullFrameWorkload",
    "LegacyWorkload",
    "ResolvedRunPaths",
    "RunConfig",
    "RunPaths",
    "StampWorkload",
    "WorkloadConfig",
    "WorkerAssignment",
    "parse_frame_indices",
    "worker_assignments",
]
