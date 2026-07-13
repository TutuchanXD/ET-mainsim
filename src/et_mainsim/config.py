from __future__ import annotations

import os
import re
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


EXECUTION_SCHEMA_ID = "et_mainsim.execution_config"
EXECUTION_SCHEMA_VERSION = 1
_BACKENDS = frozenset({"in-process", "local-subprocess"})
_DEVICES = frozenset({"cpu", "cuda"})
_ENV_PATTERN = re.compile(
    r"\$(?:\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)\}|(?P<plain>[A-Za-z_][A-Za-z0-9_]*))"
)


@dataclass(frozen=True)
class RunPaths:
    output_root: str = ""
    data_root: str = ""
    catalog_path: str = ""
    focalplane_registry: str = ""

    def __post_init__(self) -> None:
        for name in ("output_root", "data_root", "catalog_path", "focalplane_registry"):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise ValueError(f"paths.{name} must be a string")


@dataclass(frozen=True)
class ResolvedRunPaths:
    output_root: Path
    data_root: Path | None
    catalog_path: Path | None
    focalplane_registry: Path | None

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
class RunConfig:
    schema_id: str
    schema_version: int
    workflow: str
    run_id: str
    paths: RunPaths = field(default_factory=RunPaths)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
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
        }
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError(f"Unknown run config fields: {', '.join(unknown)}")

        path_payload = dict(payload.get("paths", {}))
        execution_payload = dict(payload.get("execution", {}))
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
        return cls(
            schema_id=str(payload.get("schema_id", "")),
            schema_version=int(payload.get("schema_version", 0)),
            workflow=str(payload.get("workflow", "")),
            run_id=str(payload.get("run_id", "")),
            paths=RunPaths(**path_payload),
            execution=ExecutionConfig(**execution_payload),
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
        return ResolvedRunPaths(
            output_root=output_root,
            data_root=data_root,
            catalog_path=catalog_path,
            focalplane_registry=focalplane_registry,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "workflow": self.workflow,
            "run_id": self.run_id,
            "paths": asdict(self.paths),
            "execution": self.execution.to_dict(),
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
    "ResolvedRunPaths",
    "RunConfig",
    "RunPaths",
    "WorkerAssignment",
    "parse_frame_indices",
    "worker_assignments",
]
