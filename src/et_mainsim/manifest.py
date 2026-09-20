from __future__ import annotations

import json
import os
import tempfile
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping


MANIFEST_SCHEMA_ID = "et_mainsim.run_manifest"
MANIFEST_SCHEMA_VERSION = 1
_TRANSITIONS = {
    "planned": frozenset({"running", "failed"}),
    "running": frozenset({"completed", "failed"}),
    "failed": frozenset(),
    "completed": frozenset(),
}
_NON_IDENTITY_EXECUTION_FIELDS = frozenset(
    {
        "resume",
        "overwrite",
        "force_catalog_cache",
        "progress",
        "workers_per_device",
        "gpu_ids",
        "ray_actor_count",
        "ray_num_cpus",
        "ray_num_gpus",
        "backend",
    }
)


class ManifestIdentityError(RuntimeError):
    """Raised when resume would cross a run identity boundary."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _execution_identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: deepcopy(value)
        for name, value in payload.items()
        if name not in _NON_IDENTITY_EXECUTION_FIELDS
    }


def _scientific_identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(payload))
    for field in (
        "warp_frame_batch_size",
        "image_gen_batch_size",
        "image_proc_batch_size",
    ):
        result.get("psf", {}).pop(field, None)
    result.get("dynamic_effects", {}).get("psd_motion", {}).pop("chunk_size", None)
    return result


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "to_string"):
        return value.to_string()
    if hasattr(value, "value") and hasattr(value, "unit"):
        numeric = value.value
        if hasattr(numeric, "tolist"):
            numeric = numeric.tolist()
        return {"value": numeric, "unit": str(value.unit)}
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
                default=_json_default,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


class RunManifestStore:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def load(self) -> dict[str, Any]:
        with self.path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("schema_id") != MANIFEST_SCHEMA_ID:
            raise ValueError(f"Unsupported manifest schema at {self.path}")
        if int(payload.get("schema_version", 0)) != MANIFEST_SCHEMA_VERSION:
            raise ValueError(f"Unsupported manifest schema version at {self.path}")
        return payload

    def create(
        self,
        *,
        workflow: str,
        preset: str,
        run_id: str,
        simulation_spec: Mapping[str, Any],
        execution: Mapping[str, Any],
        frame_plan: Mapping[str, Any],
        provenance: Mapping[str, Any],
        workload: Mapping[str, Any] | None = None,
        artifacts: Mapping[str, Any] | None = None,
        input_identity: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.path.exists():
            raise FileExistsError(f"Run manifest already exists: {self.path}")
        now = _utc_now()
        payload: dict[str, Any] = {
            "schema_id": MANIFEST_SCHEMA_ID,
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "workflow": str(workflow),
            "preset": str(preset),
            "run_id": str(run_id),
            "status": "planned",
            "timestamps": {
                "created_at": now,
                "started_at": None,
                "completed_at": None,
                "failed_at": None,
                "updated_at": now,
            },
            "simulation_spec": deepcopy(dict(simulation_spec)),
            "execution": deepcopy(dict(execution)),
            "workload": deepcopy(dict(workload or {})),
            "frame_plan": deepcopy(dict(frame_plan)),
            "provenance": deepcopy(dict(provenance)),
            "input_identity": deepcopy(input_identity),
            "catalog": None,
            "artifacts": deepcopy(dict(artifacts or {})),
            "completion": None,
            "failure": None,
            "attempts": [],
        }
        _atomic_write_json(self.path, payload)
        return payload

    def ensure_identity(
        self,
        *,
        workflow: str,
        run_id: str,
        simulation_spec: Mapping[str, Any],
        execution: Mapping[str, Any],
        workload: Mapping[str, Any] | None = None,
        input_identity: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = self.load()
        previous_inputs = payload.get("input_identity")
        catalog_stage = (payload.get("completion") or {}).get("catalog_only") is True

        def catalog_inputs(value):
            result = deepcopy(dict(value))
            result.pop("assets", None)
            if isinstance(result.get("runtime"), Mapping):
                result["runtime"].pop("rendering", None)
            return result

        promoting_catalog = (
            catalog_stage
            and isinstance(previous_inputs, Mapping)
            and previous_inputs.get("assets") == {}
            and isinstance(input_identity, Mapping)
            and catalog_inputs(previous_inputs) == catalog_inputs(input_identity)
        )
        if previous_inputs != input_identity and not promoting_catalog:
            raise ManifestIdentityError(
                "Existing run input identity conflicts or lacks evidence; use a new run id"
            )
        if payload["workflow"] != workflow or payload["run_id"] != run_id:
            raise ManifestIdentityError("Existing run workflow or run id conflicts")
        previous_spec = _scientific_identity(payload["simulation_spec"])
        current_spec = _scientific_identity(simulation_spec)
        previous_execution = _execution_identity(payload["execution"])
        current_execution = _execution_identity(execution)
        if promoting_catalog:
            for value in (previous_spec, current_spec):
                value.get("psf", {}).pop("compute_device", None)
            for value in (previous_execution, current_execution):
                value.pop("device", None)
                value.pop("preview_count", None)
        if previous_spec != current_spec:
            raise ManifestIdentityError("Existing run scientific spec conflicts")
        if previous_execution != current_execution:
            raise ManifestIdentityError("Existing run execution identity conflicts")
        if payload.get("workload", {}) != dict(workload or {}):
            raise ManifestIdentityError("Existing run workload identity conflicts")
        return payload

    def transition(self, status: str, **updates: Any) -> dict[str, Any]:
        payload = self.load()
        current = str(payload["status"])
        status = str(status)
        if status not in _TRANSITIONS.get(current, frozenset()):
            raise ValueError(f"Invalid manifest transition {current!r} -> {status!r}")
        now = _utc_now()
        payload["status"] = status
        payload["timestamps"]["updated_at"] = now
        if status == "running":
            payload["timestamps"]["started_at"] = now
        elif status == "completed":
            payload["timestamps"]["completed_at"] = now
        elif status == "failed":
            payload["timestamps"]["failed_at"] = now
        attempts = payload.setdefault("attempts", [])
        if status in {"completed", "failed"} and attempts:
            active_attempt = attempts[-1]
            if active_attempt.get("status") == "running":
                active_attempt["status"] = status
                active_attempt["ended_at"] = now
        for name, value in updates.items():
            payload[name] = deepcopy(value)
        _atomic_write_json(self.path, payload)
        return payload

    def start_attempt(
        self,
        *,
        control: Mapping[str, Any] | None = None,
        recover_running: bool = False,
        input_identity: Mapping[str, Any] | None = None,
        simulation_spec: Mapping[str, Any] | None = None,
        execution: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = self.load()
        previous_status = str(payload["status"])
        if previous_status == "running" and not recover_running:
            raise ValueError("Run is already running")
        if previous_status not in {"planned", "completed", "failed", "running"}:
            raise ValueError(f"Cannot start an attempt from {previous_status!r}")
        now = _utc_now()
        attempts = payload.setdefault("attempts", [])
        if previous_status == "running" and attempts:
            attempts[-1].update(status="interrupted", ended_at=now)
        attempts.append(
            {
                "number": len(attempts) + 1,
                "previous_status": previous_status,
                "status": "running",
                "started_at": now,
                "ended_at": None,
                "control": deepcopy(dict(control or {})),
            }
        )
        payload["status"] = "running"
        payload["completion"] = None
        payload["failure"] = None
        payload["timestamps"]["started_at"] = now
        payload["timestamps"]["completed_at"] = None
        payload["timestamps"]["failed_at"] = None
        payload["timestamps"]["updated_at"] = now
        # A staged run must never expose "running" with the old catalog-only
        # baseline: a crash before a second write would prevent recovery.
        for name, value in (
            ("input_identity", input_identity),
            ("simulation_spec", simulation_spec),
            ("execution", execution),
        ):
            if value is not None:
                payload[name] = deepcopy(dict(value))
        _atomic_write_json(self.path, payload)
        return payload

    def update(self, **updates: Any) -> dict[str, Any]:
        payload = self.load()
        if payload["status"] not in {"planned", "running"}:
            raise ValueError("Only active manifests can be updated")
        payload["timestamps"]["updated_at"] = _utc_now()
        for name, value in updates.items():
            payload[name] = deepcopy(value)
        _atomic_write_json(self.path, payload)
        return payload

    def fail(self, error: BaseException) -> dict[str, Any]:
        return self.transition(
            "failed",
            failure={
                "type": type(error).__name__,
                "message": str(error),
            },
        )


__all__ = [
    "MANIFEST_SCHEMA_ID",
    "MANIFEST_SCHEMA_VERSION",
    "ManifestIdentityError",
    "RunManifestStore",
]
