"""Application-owned execution of Photsim7's explicit equivalent products.

Scientific configuration and validity stay in the upstream typed request. This
module owns file inputs, group selection, CPU policy, run manifests and resume.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import importlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import re
import threading
import tomllib
from typing import Mapping

from .inputs import run_lock, _source_code_identity, runtime_identity
from .manifest import RunManifestStore, _atomic_write_json


CONFIG_SCHEMA = "et_mainsim.equivalent_coadd_run.v1"
_NUMERIC_POLICY_LOCK = threading.RLock()


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@dataclass(frozen=True)
class EquivalentRunConfig:
    request_path: Path
    catalog_path: Path
    variability_path: Path
    data_root: Path
    output_root: Path
    run_id: str
    coadd_indices: tuple[int, ...] | None = None
    block_shape: tuple[int, int] = (512, 512)
    cpu_threads: int = 2
    resume: bool = True

    @classmethod
    def from_mapping(cls, payload: Mapping, *, base=None):
        base = Path.cwd() if base is None else Path(base)
        if not isinstance(payload, Mapping):
            raise ValueError("equivalent run config must be a mapping")
        required = {
            "schema_id",
            "request_path",
            "catalog_path",
            "variability_path",
            "data_root",
            "output_root",
            "run_id",
        }
        optional = {"coadd_indices", "block_shape", "cpu_threads", "resume"}
        if not required <= set(payload) or set(payload) - required - optional:
            raise ValueError("equivalent run config has missing or unknown fields")
        if payload["schema_id"] != CONFIG_SCHEMA:
            raise ValueError("unsupported equivalent run config schema")
        run_id = payload["run_id"]
        if (
            not isinstance(run_id, str)
            or not run_id.strip()
            or run_id in (".", "..", ".run_locks")
            or Path(run_id).name != run_id
        ):
            raise ValueError(
                "run_id must be a single nonempty, unreserved path component"
            )
        paths = {}
        for name in required - {"schema_id", "run_id"}:
            value = payload[name]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty path")
            path = Path(value).expanduser()
            paths[name] = (path if path.is_absolute() else Path(base) / path).resolve()
        indices = payload.get("coadd_indices")
        if indices is not None:
            if not isinstance(indices, (list, tuple)) or not indices:
                raise ValueError("coadd_indices must be nonempty when supplied")
            indices = tuple(_integer(v, "coadd index") for v in indices)
            if len(set(indices)) != len(indices):
                raise ValueError(
                    "duplicate coadd indices are not independent exposures"
                )
        block = payload.get("block_shape", (512, 512))
        if not isinstance(block, (list, tuple)) or len(block) != 2:
            raise ValueError("block_shape requires two dimensions")
        block = tuple(_integer(v, "block dimension", 128) for v in block)
        if any(v % 128 for v in block):
            raise ValueError("block dimensions must be multiples of 128")
        threads = _integer(payload.get("cpu_threads", 2), "cpu_threads", 1)
        resume = payload.get("resume", True)
        if type(resume) is not bool:
            raise ValueError("resume must be a boolean")
        return cls(
            **paths,
            run_id=run_id,
            coadd_indices=indices,
            block_shape=block,
            cpu_threads=threads,
            resume=resume,
        )

    @classmethod
    def from_file(cls, path):
        path = Path(path).resolve()
        text = path.read_text()
        payload = (
            json.loads(text) if path.suffix.lower() == ".json" else tomllib.loads(text)
        )
        return cls.from_mapping(payload, base=path.parent)

    def to_dict(self):
        return {
            "schema_id": CONFIG_SCHEMA,
            **{
                name: str(getattr(self, name))
                for name in (
                    "request_path",
                    "catalog_path",
                    "variability_path",
                    "data_root",
                    "output_root",
                )
            },
            "run_id": self.run_id,
            "coadd_indices": None
            if self.coadd_indices is None
            else list(self.coadd_indices),
            "block_shape": list(self.block_shape),
            "cpu_threads": self.cpu_threads,
            "resume": self.resume,
        }


def _api():
    try:
        api = importlib.import_module("photsim7.equivalent_coadd")
    except ImportError as error:
        raise RuntimeError(
            "installed Photsim7 lacks the formal equivalent runtime; install the documented upstream revision"
        ) from error
    for name in (
        "EquivalentCoaddRequest",
        "SourceVariabilityDelivery",
        "run_equivalent_coadd_product",
        "read_equivalent_coadd_product",
    ):
        if not hasattr(api, name):
            raise RuntimeError(
                f"installed Photsim7 lacks {name}; install the documented upstream revision"
            )
    return api


@contextmanager
def _numeric_policy(threads):
    # Torch and native pool settings are process-global, even for distinct
    # run IDs. Keep the complete execution/restoration scope serialized.
    with _NUMERIC_POLICY_LOCK:
        with _locked_numeric_policy(threads):
            yield


@contextmanager
def _locked_numeric_policy(threads):
    import torch
    from numba import get_num_threads, set_num_threads
    from threadpoolctl import threadpool_limits

    # NumPy and SciPy wheels can ship separate BLAS runtimes. Load SciPy's
    # native linear algebra before threadpoolctl snapshots loaded libraries;
    # otherwise _context() can introduce an uncapped pool on the first run.
    importlib.import_module("scipy.linalg")
    old_torch = torch.get_num_threads()
    old_numba = get_num_threads()
    try:
        set_num_threads(threads)
        torch.set_num_threads(threads)
        with threadpool_limits(threads):
            yield
    finally:
        set_num_threads(old_numba)
        torch.set_num_threads(old_torch)


def _inputs(config):
    from photsim7.catalogs import PreparedStarCatalog
    from photsim7.data_registry import DataRegistry

    before = {
        name: _file_sha(getattr(config, name))
        for name in ("request_path", "catalog_path", "variability_path")
    }
    api = _api()
    request = api.EquivalentCoaddRequest.from_json_dict(
        json.loads(config.request_path.read_text())
    )
    payload = request.to_json_dict()
    catalog_data = json.loads(config.catalog_path.read_text())
    if not isinstance(catalog_data, dict) or set(catalog_data) != {
        "data",
        "metadata",
        "raw_source_arrays",
    }:
        raise ValueError("catalog JSON requires data, metadata and raw_source_arrays")
    catalog = PreparedStarCatalog(
        star_data=catalog_data["data"],
        metadata=catalog_data["metadata"],
        raw_source_arrays=catalog_data["raw_source_arrays"],
    )
    delivery = api.SourceVariabilityDelivery.from_json_dict(
        json.loads(config.variability_path.read_text())
    )
    # These are the upstream request's semantic input identities, not a new
    # scientific derivation or permission to use a naked folded spec.
    catalog_sha = hashlib.sha256(
        json.dumps(
            catalog_data, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    if (
        catalog_sha != payload["catalog_sha256"]
        or delivery.content_sha256 != payload["variability_sha256"]
    ):
        raise ValueError("catalog/variability content differs from the typed request")
    indices = config.coadd_indices
    if indices is None:
        indices = tuple(range(payload["coadd_start"], payload["coadd_stop"]))
    if any(not payload["coadd_start"] <= i < payload["coadd_stop"] for i in indices):
        raise ValueError("selected group is outside the declared request family")
    if any(
        _file_sha(getattr(config, name)) != digest for name, digest in before.items()
    ):
        raise ValueError("input files changed while reading")
    return (
        api,
        request,
        catalog,
        delivery,
        DataRegistry(data_root=config.data_root),
        indices,
        before,
    )


def build_equivalent_run_plan(config: EquivalentRunConfig):
    """Validate declared files and group selection without loading PSF assets."""
    config = EquivalentRunConfig.from_mapping(config.to_dict())
    _, request, _, _, _, indices, _ = _inputs(config)
    derivation = request.derivation
    return {
        "workflow": "et-equivalent-coadd",
        "run_dir": str(config.output_root / config.run_id),
        "request_sha256": request.content_sha256,
        "output_kind": request.output_kind,
        "coadd_indices": list(indices),
        "cadence_s": 10 * derivation.n_raw,
        "absolute_raw_frame_start_index": derivation.absolute_raw_frame_start_index,
        "input_accuracy": "unqualified",
        "asset_validation": "required_at_execution",
    }


def _context(config):
    """Bind the configuration, immutable input snapshots and current assets."""
    config = EquivalentRunConfig.from_mapping(config.to_dict())
    api, request, catalog, delivery, registry, indices, input_files = _inputs(config)
    import photsim7
    import et_mainsim
    from photsim7.input_identity import simulation_asset_identity

    derivation = request.derivation
    payload = request.to_json_dict()
    if (
        simulation_asset_identity(derivation.source_spec, registry)
        != payload["asset_identity"]
    ):
        raise ValueError("scientific assets differ from the typed request")
    run_dir = config.output_root / config.run_id
    store = RunManifestStore(run_dir / "run_manifest.json")
    workload = {
        "kind": "equivalent-coadd",
        "request_sha256": request.content_sha256,
        "coadd_indices": list(indices),
        "block_shape": list(config.block_shape),
        "cpu_threads": config.cpu_threads,
    }
    execution = {
        "backend": "in-process",
        "device": "cpu",
        "resume": config.resume,
        "cpu_threads": config.cpu_threads,
    }
    identity = {
        "request_sha256": request.content_sha256,
        "files": input_files,
        "assets": payload["asset_identity"],
        "et_mainsim_source_sha256": _source_code_identity(et_mainsim),
        "photsim7_source_sha256": _source_code_identity(photsim7),
        "runtime": _runtime_identity(derivation.source_spec),
    }
    common = dict(
        workflow="et-equivalent-coadd",
        run_id=config.run_id,
        simulation_spec=derivation.source_spec.to_json_dict(),
        execution=execution,
        workload=workload,
        input_identity=identity,
    )
    return config, api, request, catalog, delivery, registry, indices, store, common


def _runtime_identity(spec):
    """Bind a run before its first product, including interrupted empty runs."""
    import torch
    from threadpoolctl import threadpool_info

    identity = runtime_identity(spec)
    identity["packages"].update(
        {name: version(name) for name in ("numba", "h5py", "threadpoolctl")}
    )
    cpu = Path("/proc/cpuinfo")
    identity.update(
        platform=platform.platform(),
        cpu_model=next(
            (
                line.split(":", 1)[1].strip()
                for line in (cpu.read_text().splitlines() if cpu.exists() else [])
                if line.startswith("model name")
            ),
            platform.processor(),
        ),
        torch_build_sha256=hashlib.sha256(torch.__config__.show().encode()).hexdigest(),
        torch_cpu_capability=torch.backends.cpu.get_cpu_capability(),
        torch_interop_threads=torch.get_num_interop_threads(),
        native_libraries=sorted(
            [
                {
                    key: pool.get(key)
                    for key in (
                        "internal_api",
                        "prefix",
                        "version",
                        "threading_layer",
                        "architecture",
                    )
                }
                for pool in threadpool_info()
            ],
            key=lambda pool: json.dumps(pool, sort_keys=True),
        ),
    )
    return identity


def _sidecars(config, request, run_dir, *, initialize=False):
    expected = {
        "equivalent_request.json": (request.to_json_dict(), "typed request"),
        "equivalent_run_config.json": (config.to_dict(), "application config"),
    }
    missing = []
    # Validate all existing files before repairing any missing initialization.
    for name, (current, label) in expected.items():
        path = run_dir / name
        if path.is_symlink():
            raise ValueError("saved sidecars must not be symlinks")
        if not path.exists() and initialize:
            missing.append((path, current))
            continue
        saved = json.loads(path.read_text())
        comparison = dict(current)
        if name == "equivalent_run_config.json":
            saved.pop("resume", None)
            comparison.pop("resume", None)
        if saved != comparison:
            raise ValueError(f"saved {label} differs from run authority")
    for path, current in missing:
        _atomic_write_json(path, current)


def _verify_inputs(config, request, registry, identity):
    from photsim7.input_identity import simulation_asset_identity

    if any(
        _file_sha(getattr(config, name)) != digest
        for name, digest in identity["files"].items()
    ):
        raise ValueError("run input files changed during execution or verification")
    if (
        simulation_asset_identity(request.derivation.source_spec, registry)
        != identity["assets"]
    ):
        raise ValueError("scientific assets changed during execution or verification")


def _record(path, index, result):
    manifest_bytes = (path / "product_manifest.json").read_bytes()
    if json.loads(manifest_bytes) != result["manifest"]:
        raise ValueError("product manifest changed after upstream verification")
    return {
        "coadd_index": index,
        "path": str(path),
        "product_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "files": result["manifest"]["files"],
        "arrays": result["manifest"]["arrays"],
        "raw_indices": result["metadata"]["raw_indices"],
        "raw_time_windows_s": result["metadata"]["raw_time_windows_s"],
        "clipping_certificate": result["metadata"]["clipping_certificate"],
    }


def _verify_product_files(products):
    """Recheck all bytes that the upstream reader already validated."""
    for record in products:
        path = Path(record["path"])
        files = {
            **record["files"],
            "product_manifest.json": record["product_manifest_sha256"],
        }
        if path.is_symlink() or {p.name for p in path.iterdir()} != set(files):
            raise ValueError("verified product directory changed before completion")
        for name, digest in files.items():
            member = path / name
            if (
                member.is_symlink()
                or not member.is_file()
                or _file_sha(member) != digest
            ):
                raise ValueError("verified product files changed before completion")


def _recover_initial_manifest(config, run_dir):
    if run_dir.is_symlink():
        raise ValueError("run directory must not be a symlink")
    if not run_dir.exists():
        return
    entries = list(run_dir.iterdir())
    # A killed atomic writer can leave its unpublished temporary manifest.
    # Inspect the whole inventory before removing any known orphan; preserve
    # directories, links, sidecars and unknown/user files without modification.
    if entries and (
        not config.resume
        or run_dir.is_symlink()
        or any(
            entry.is_symlink()
            or not entry.is_file()
            or re.fullmatch(r"\.run_manifest\.json\.[a-z0-9_]{8}\.tmp", entry.name)
            is None
            for entry in entries
        )
    ):
        raise ValueError("nonempty run directory lacks its manifest")
    for entry in entries:
        entry.unlink()


def _validate_run_paths(run_dir):
    paths = [
        run_dir,
        *(
            run_dir / name
            for name in (
                "run_manifest.json",
                "equivalent_request.json",
                "equivalent_run_config.json",
                "products",
            )
        ),
    ]
    if any(path.is_symlink() for path in paths):
        raise ValueError("run directories, manifest and sidecars must not be symlinks")


def _prior_products(payload, indices, run_dir):
    records = payload.get("artifacts", {}).get("equivalent_products", [])
    if not isinstance(records, list):
        raise ValueError("invalid application product records")
    prior = {}
    for record in records:
        index = record.get("coadd_index")
        if type(index) is not int or index not in indices or index in prior:
            raise ValueError("application product group inventory differs from request")
        path = run_dir / "products" / f"group_{index:08d}"
        if record.get("path") != str(path) or _file_sha(
            path / "product_manifest.json"
        ) != record.get("product_manifest_sha256"):
            raise ValueError(
                "product manifest differs from the application completion record"
            )
        prior[index] = record
    return prior


def verify_equivalent_run(config: EquivalentRunConfig):
    """Independently consume every completed upstream product without rendering."""
    config = EquivalentRunConfig.from_mapping(config.to_dict())
    run_dir = config.output_root / config.run_id
    _validate_run_paths(run_dir)
    with run_lock(run_dir), _numeric_policy(config.cpu_threads):
        _validate_run_paths(run_dir)
        config, api, request, _, _, registry, indices, store, common = _context(config)
        payload = store.ensure_identity(**common)
        _sidecars(config, request, run_dir)
        expected = {
            "coadd_groups": len(indices),
            "request_sha256": request.content_sha256,
            "input_accuracy": "unqualified",
        }
        if (
            payload.get("status") != "completed"
            or payload.get("completion") != expected
        ):
            raise ValueError("application run is not complete")
        prior = _prior_products(payload, indices, run_dir)
        if set(prior) != set(indices):
            raise ValueError("application completion omits declared groups")
        for index in indices:
            path = run_dir / "products" / f"group_{index:08d}"
            checked = api.read_equivalent_coadd_product(
                path, request=request, coadd_index=index
            )
            if any(
                prior[index].get(k) != v
                for k, v in _record(path, index, checked).items()
            ):
                raise ValueError(
                    "actual upstream product differs from application record"
                )
        _verify_product_files(prior.values())
        _verify_inputs(config, request, registry, common["input_identity"])
        _validate_run_paths(run_dir)
        _sidecars(config, request, run_dir)
        if store.ensure_identity(**common) != payload:
            raise ValueError("application manifest changed during verification")
        return {
            "status": "verified",
            "workflow": "et-equivalent-coadd",
            "run_dir": str(run_dir),
            **expected,
        }


def run_equivalent_coadd(config: EquivalentRunConfig):
    """Produce verified formal upstream products under an application run lock."""
    config = EquivalentRunConfig.from_mapping(config.to_dict())
    run_dir = config.output_root / config.run_id
    _validate_run_paths(run_dir)
    with run_lock(run_dir), _numeric_policy(config.cpu_threads):
        _validate_run_paths(run_dir)
        config, api, request, catalog, delivery, registry, indices, store, common = (
            _context(config)
        )
        derivation = request.derivation
        identity = common["input_identity"]
        if store.path.exists():
            if not config.resume:
                raise FileExistsError(run_dir)
            previous = store.ensure_identity(**common)
            # A manifest is the identity anchor. Repair sidecars only before
            # any attempt or artifact exists, after ensure_identity succeeds.
            _sidecars(
                config,
                request,
                run_dir,
                initialize=(
                    previous.get("status") == "planned"
                    and previous.get("attempts") == []
                    and previous.get("artifacts") == {}
                    and previous.get("completion") is None
                ),
            )
        else:
            _recover_initial_manifest(config, run_dir)
            previous = store.create(
                **common,
                preset="explicit-equivalent-request",
                frame_plan={
                    "coadd_indices": list(indices),
                    "n_raw": derivation.n_raw,
                    "absolute_raw_frame_start_index": derivation.absolute_raw_frame_start_index,
                },
                provenance={
                    "photsim7_version": version("photsim7"),
                    "input_accuracy": "unqualified",
                },
            )
            _sidecars(config, request, run_dir, initialize=True)
        store.start_attempt(control={"resume": config.resume}, recover_running=True)
        try:
            prior = _prior_products(previous, indices, run_dir)
            products = []
            for index in indices:
                path = run_dir / "products" / f"group_{index:08d}"
                result = api.run_equivalent_coadd_product(
                    request,
                    prepared_catalog=catalog,
                    variability_delivery=delivery,
                    data_registry=registry,
                    coadd_index=index,
                    output=path,
                    block_shape=config.block_shape,
                    resume=config.resume,
                )
                # The upstream writer/reuser returns only after complete
                # readback. Consume that verified record without a second full
                # same-process read; verify_equivalent_run provides independent
                # inspection of a completed application run.
                record = _record(path, index, result)
                if index in prior and any(
                    prior[index].get(k) != v for k, v in record.items()
                ):
                    raise ValueError(
                        "actual upstream product differs from application record"
                    )
                products.append({**record, "reused": result["reused"]})
                store.update(artifacts={"equivalent_products": products})
            _verify_product_files(products)
            _verify_inputs(config, request, registry, identity)
            _validate_run_paths(run_dir)
            _sidecars(config, request, run_dir)
            current = store.ensure_identity(**common)
            if current.get("artifacts") != {"equivalent_products": products}:
                raise ValueError(
                    "application product records changed before completion"
                )
            return store.transition(
                "completed",
                completion={
                    "coadd_groups": len(products),
                    "request_sha256": request.content_sha256,
                    "input_accuracy": "unqualified",
                },
            )
        except BaseException as error:
            # Do not follow a replaced publication path while recording an
            # integrity failure; preserve the original exception instead.
            if (
                not run_dir.is_symlink()
                and not store.path.is_symlink()
                and store.path.is_file()
            ):
                store.fail(error)
            raise


__all__ = [
    "EquivalentRunConfig",
    "build_equivalent_run_plan",
    "run_equivalent_coadd",
    "verify_equivalent_run",
]
