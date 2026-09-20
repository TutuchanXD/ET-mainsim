"""Effective scientific inputs and automatic run-local content evidence."""

from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

from .manifest import ManifestIdentityError, _atomic_write_json


@contextmanager
def worker_lock(run_dir):
    """Keep live workers visible even if their coordinator exits unexpectedly."""
    import fcntl

    run_dir = Path(run_dir)
    path = run_dir.parent / ".run_locks" / f"{run_dir.name}.workers"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_SH)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def launch_worker(command, *, run_dir, **kwargs):
    """Hold a shared lease from before exec until the child exits.

    Closing the parent's duplicate must not explicitly unlock the shared open
    file description inherited by the child.
    """
    import fcntl
    import os
    import subprocess

    run_dir = Path(run_dir)
    path = run_dir.parent / ".run_locks" / f"{run_dir.name}.workers"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_SH)
        inherited = tuple(kwargs.pop("pass_fds", ())) + (descriptor,)
        return subprocess.Popen(command, pass_fds=inherited, **kwargs)
    finally:
        os.close(descriptor)


@contextmanager
def run_lock(run_dir):
    """Single coordinator per run; the OS releases the lock on process exit."""
    import fcntl

    run_dir = Path(run_dir)
    lock = run_dir.parent / ".run_locks" / f"{run_dir.name}.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Run is already active: {run_dir}") from error
        try:
            with lock.with_suffix(".workers").open("a") as workers:
                try:
                    fcntl.flock(workers.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as error:
                    raise RuntimeError(
                        f"Previous workers are still active: {run_dir}"
                    ) from error
                fcntl.flock(workers.fileno(), fcntl.LOCK_UN)
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def locked_run(function):
    @wraps(function)
    def wrapped(plan, *args, **kwargs):
        with run_lock(plan.run_dir):
            return function(plan, *args, **kwargs)

    return wrapped


def locked_worker(function):
    @wraps(function)
    def wrapped(request, *args, **kwargs):
        plan = getattr(request, "plan", request)
        with worker_lock(plan.run_dir):
            return function(request, *args, **kwargs)

    return wrapped


def effective_spec(spec, data_root):
    from photsim7.data_registry import DataRegistry
    from photsim7.pipelines import build_simulation_context

    return build_simulation_context(
        spec, data_registry=DataRegistry(data_root)
    ).effective_spec


def _source_code_identity(package):
    root = Path(package.__file__).parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def runtime_identity(spec, *, catalog_only=False, gpu_ids=()):
    import et_mainsim
    import et_coord
    import photsim7

    def versions(names):
        result = {}
        for name in names:
            try:
                result[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                result[name] = None
        return result

    result = {
        "python": platform.python_version(),
        "machine": platform.machine(),
        "packages": versions(("numpy", "scipy", "astropy", "et-coord")),
        "photsim7_code": _source_code_identity(photsim7),
        "et_mainsim_code": _source_code_identity(et_mainsim),
        "et_coordinate_code": _source_code_identity(et_coord),
    }
    if catalog_only:
        return result
    rendering = {
        "device": spec.psf.compute_device,
        "float_precision": spec.psf.float_precision,
        "packages": versions(("torch", "kornia")),
    }
    if str(spec.psf.compute_device).startswith("cuda"):
        import os
        import torch
        from photsim7._determinism import deterministic_execution

        mask = os.environ.get("CUDA_VISIBLE_DEVICES")
        visible = None if mask is None else [item.strip() for item in mask.split(",")]
        indices = []
        for selector in gpu_ids:
            if visible is not None:
                if selector not in visible:
                    raise ValueError(
                        f"GPU {selector!r} is outside CUDA_VISIBLE_DEVICES={mask!r}"
                    )
                indices.append(visible.index(selector))
            elif str(selector).isdigit():
                indices.append(int(selector))
            else:
                matches = [
                    index
                    for index in range(torch.cuda.device_count())
                    if str(
                        getattr(torch.cuda.get_device_properties(index), "uuid", "")
                    ).startswith(str(selector))
                ]
                if len(matches) != 1:
                    raise ValueError(
                        f"GPU selector {selector!r} does not identify one visible CUDA device"
                    )
                indices.append(matches[0])
        if not indices:
            indices = [torch.device(spec.psf.compute_device).index or 0]
        with deterministic_execution(spec.rng.determinism_mode, device="cuda"):
            devices = [
                {
                    "toolkit": torch.version.cuda,
                    "device_name": torch.cuda.get_device_name(index),
                    "capability": list(torch.cuda.get_device_capability(index)),
                }
                for index in indices
            ]
        if any(device != devices[0] for device in devices[1:]):
            raise ValueError(
                "One resumable run requires homogeneous selected CUDA devices"
            )
        rendering["cuda"] = devices[0]
    result["rendering"] = rendering
    return result


def collect_run_inputs(
    spec, data_root, *, catalog_cache=None, catalog_only=False, gpu_ids=()
):
    from photsim7.data_registry import DataRegistry
    from photsim7.input_identity import (
        content_identity,
        simulation_asset_identity,
        validate_catalog_cache_location,
    )

    registry = DataRegistry(data_root)
    validate_catalog_cache_location(spec, registry)
    result = {
        "schema_version": 1,
        "assets": {} if catalog_only else simulation_asset_identity(spec, registry),
        "runtime": runtime_identity(spec, catalog_only=catalog_only, gpu_ids=gpu_ids),
    }
    source = spec.catalog.source_path
    if source:
        path = registry.resolve_path(source)
        if path.exists():
            result["catalog_source"] = content_identity(path)
        elif catalog_cache is not None and Path(catalog_cache).is_file():
            from photsim7.catalogs.cache import StarCatalogCache

            result["catalog_source"] = (
                StarCatalogCache.read(catalog_cache)
                .metadata.get("request", {})
                .get("source", {})
                .get("content")
            )
        else:
            raise FileNotFoundError(f"Catalog input does not exist: {path}")
    if spec.catalog.registry_data_dir:
        result["catalog_registry"] = content_identity(
            registry.resolve_path(spec.catalog.registry_data_dir)
        )
    return result


def persist_effective_spec(run_dir, spec):
    _atomic_write_json(Path(run_dir) / "effective_spec.json", spec.to_json_dict())


def verify_worker_inputs(run_dir, spec, data_root, *, catalog_cache=None):
    path = Path(run_dir) / "run_manifest.json"
    if not path.is_file():
        return  # Standalone worker API; no parent run to resume.
    manifest = json.loads(path.read_text())
    recorded = manifest.get("input_identity")
    current = collect_run_inputs(spec, data_root, catalog_cache=catalog_cache)
    if current != recorded:
        raise ManifestIdentityError("Worker input identity changed after planning")
    catalog = None
    if catalog_cache is not None and manifest.get("catalog_content") is not None:
        from photsim7.catalogs.cache import StarCatalogCache

        catalog = StarCatalogCache.read(catalog_cache)
        if catalog_identity(catalog) != manifest["catalog_content"]:
            raise ManifestIdentityError(
                "Worker catalog content changed after preparation"
            )
    return catalog


def catalog_identity(catalog):
    """Scientific arrays and declarations, excluding cache/trace bookkeeping."""
    import numpy as np

    digest = hashlib.sha256()
    metadata = {
        name: value
        for name, value in catalog.metadata.items()
        if name
        not in {
            "rng_trace",
            "catalog_generation_rng_trace",
            "request_validation",
            "cache",
        }
    }
    from .manifest import _json_default

    metadata["catalog"] = {
        "schema_id": catalog.schema_id,
        "schema_version": catalog.schema_version,
    }
    digest.update(
        json.dumps(
            metadata,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            default=_json_default,
        ).encode()
    )
    for name, value in sorted(catalog.star_data.items()):
        array = np.asarray(value)
        digest.update(json.dumps([name, array.dtype.str, array.shape]).encode())
        if array.dtype.hasobject:
            digest.update(json.dumps(array.tolist(), sort_keys=True).encode())
        else:
            digest.update(array.tobytes(order="C"))
    return {"schema_version": 1, "sha256": digest.hexdigest()}


def record_catalog_identity(store, catalog):
    identity = catalog_identity(catalog)
    previous = store.load().get("catalog_content")
    if previous is not None and identity != previous:
        raise ManifestIdentityError(
            "Existing run catalog content conflicts; use a new run id"
        )
    store.update(catalog_content=identity)


def prepare_catalog_input(spec, registry, *, api, run_dir, force=False):
    """Refresh incompatible derived caches for new experiments, atomically."""
    from dataclasses import replace
    import os
    import tempfile
    from photsim7.catalogs.cache import CatalogRequestMismatchError, StarCatalogCache

    from photsim7.input_identity import validate_catalog_cache_location

    validate_catalog_cache_location(spec, registry)
    if not force:
        try:
            return api.build_catalog_from_spec(spec, data_registry=registry)
        except CatalogRequestMismatchError:
            manifest = Path(run_dir) / "run_manifest.json"
            if (
                manifest.is_file()
                and json.loads(manifest.read_text()).get("catalog_content") is not None
            ):
                raise
    generated = api.build_catalog_from_spec(
        replace(spec, catalog=replace(spec.catalog, cache_path="")),
        data_registry=registry,
    )
    cache = registry.resolve_path(spec.catalog.cache_path)
    cache.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{cache.name}.", suffix=".npz", dir=cache.parent
    )
    os.close(descriptor)
    try:
        StarCatalogCache.write(temporary, generated)
        os.replace(temporary, cache)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return generated


__all__ = [
    "collect_run_inputs",
    "effective_spec",
    "persist_effective_spec",
    "verify_worker_inputs",
]
