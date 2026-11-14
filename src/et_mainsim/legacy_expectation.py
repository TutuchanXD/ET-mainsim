"""Application orchestration for the public Photsim7 electron coadd profile.

One complete time group is the transaction and scheduling unit. Scientific
rendering, photometry and scope combination belong to Photsim7.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
import tempfile
import time
import tomllib

import numpy as np

from .equivalent_coadd import _numeric_policy, _runtime_identity
from .manifest import _atomic_write_json

CONFIG_SCHEMA = "et_mainsim.legacy_expectation_config.v1"
RUN_SCHEMA = "et_mainsim.legacy_expectation_run.v1"
GROUP_SCHEMA = "et_mainsim.legacy_expectation_group.v1"
PROFILE = "legacy_expectation_electrons_v1"
_SINGLE_FIELDS = (
    "frame_start_s", "frame_stop_s", "mid_time_s", "integration_s",
    "flux_electrons", "electron_rate", "normalized_flux",
    "effective_aperture_area_pix", "centroid_xy_pix", "quality", "input_quality",
)


def _json(path):
    return json.loads(Path(path).read_text())


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _int(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True)
class LegacyExpectationConfig:
    """Explicit input files; scheduling selectors are execution arguments only."""

    spec_path: Path
    aperture_path: Path
    data_root: Path
    output_root: Path
    run_id: str
    kind: str
    target_ids: tuple[int, ...]
    catalog_path: Path | None = None
    input_table_path: Path | None = None
    variability_path: Path | None = None
    variability_table_path: Path | None = None
    focalplane_registry: Path | None = None
    stamp_shape: tuple[int, int] = (15, 15)
    output_kind: str = "pixels"
    save_raw_expectations: bool = False
    photon_noise: bool = True
    readout_noise: bool = True
    cpu_threads: int = 2
    absolute_raw_frame_start_index: int = 0
    spacecraft_id: str = "et"
    normalization_electron_rate: float | None = None
    max_group_pixels: int = 200_000_000

    @classmethod
    def from_mapping(cls, payload, *, base=Path.cwd()):
        payload = dict(payload)
        if payload.pop("schema_id", None) != CONFIG_SCHEMA:
            raise ValueError(f"schema_id must be {CONFIG_SCHEMA}")
        unknown = set(payload) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown configuration fields: {sorted(unknown)}")
        base = Path(base).resolve()
        for name in ("spec_path", "aperture_path", "data_root", "output_root",
                     "catalog_path", "input_table_path", "variability_path",
                     "variability_table_path", "focalplane_registry"):
            value = payload.get(name)
            if value is not None:
                if not isinstance(value, (str, Path)) or not str(value).strip():
                    raise ValueError(f"{name} requires a nonempty path")
                path = Path(value).expanduser()
                # Do not resolve output symlinks away before the path guard.
                payload[name] = Path(os.path.abspath(base / path))
        for name in ("target_ids", "stamp_shape"):
            if name in payload:
                if not isinstance(payload[name], (list, tuple)):
                    raise ValueError(f"{name} must be an array")
                payload[name] = tuple(payload[name])
        result = cls(**payload)
        result.validate()
        return result

    def validate(self):
        if (not isinstance(self.run_id, str) or self.run_id in {"", ".", "..", ".run_locks"}
                or Path(self.run_id).name != self.run_id or "\\" in self.run_id):
            raise ValueError("run_id must be a single unreserved path component")
        if self.kind not in {"full-frame", "catalog-stamp", "table-stamp"}:
            raise ValueError("kind must be full-frame, catalog-stamp or table-stamp")
        if self.output_kind not in {"pixels", "lightcurve"}:
            raise ValueError("output_kind must be pixels or lightcurve")
        if not self.target_ids or len(set(self.target_ids)) != len(self.target_ids):
            raise ValueError("target_ids must be nonempty and unique")
        if any(type(v) is not int or not -(2**63) <= v < 2**63 for v in self.target_ids):
            raise ValueError("target_ids require signed 64-bit integers")
        if len(self.stamp_shape) != 2:
            raise ValueError("stamp_shape requires two dimensions")
        for v in self.stamp_shape:
            _int(v, "stamp dimension", 1)
        for name in ("cpu_threads", "max_group_pixels"):
            _int(getattr(self, name), name, 1)
        _int(self.absolute_raw_frame_start_index, "absolute_raw_frame_start_index")
        for name in ("save_raw_expectations", "photon_noise", "readout_noise"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} requires a boolean")
        if self.save_raw_expectations and self.output_kind != "pixels":
            raise ValueError("raw expectations require pixels output")
        if not isinstance(self.spacecraft_id, str) or not self.spacecraft_id.strip():
            raise ValueError("spacecraft_id must be nonempty")
        if self.normalization_electron_rate is not None:
            rate = self.normalization_electron_rate
            if isinstance(rate, bool) or not np.isfinite(rate) or rate <= 0:
                raise ValueError("normalization_electron_rate must be finite and positive")
        if self.kind == "table-stamp":
            if self.input_table_path is None or self.catalog_path or self.variability_path:
                raise ValueError("table-stamp requires input_table_path and no catalog/variability JSON")
        elif self.catalog_path is None or self.input_table_path or self.variability_table_path:
            raise ValueError("catalog/full-frame require catalog_path and no table inputs")

    @classmethod
    def from_file(cls, path):
        path = Path(path).resolve()
        payload = json.loads(path.read_text()) if path.suffix == ".json" else tomllib.loads(path.read_text())
        return cls.from_mapping(payload, base=path.parent)

    def to_dict(self):
        return {"schema_id": CONFIG_SCHEMA, **{
            name: str(value) if isinstance(value, Path) else list(value) if isinstance(value, tuple) else value
            for name, value in vars(self).items()
        }}


def _no_symlinks(path):
    path = Path(path)
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError(f"output paths must not contain symlinks: {path}")


@contextmanager
def _lock(path, *, wait=False):
    _no_symlinks(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
        except BlockingIOError as error:
            raise RuntimeError(f"work item is already active: {path.name}") from error
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _input_files(config):
    from photsim7.input_identity import content_identity
    return {name: content_identity(path) for name, path in vars(config).items()
            if (name.endswith("_path") or name == "focalplane_registry") and path is not None}


def _table_inputs(config, spec):
    """Reuse the application's existing table/geometry/PSF input adapter."""
    from .config import RunConfig, RunPaths, StampWorkload, ResolvedRunPaths
    from .workflows.stamp import (
        StampRunPlan, _table_spec, _prepare_table_inputs, _science_api, _target_spec,
    )
    from astropy.table import Table
    from .stamp_inputs import _canonical_columns

    table = Table.read(config.input_table_path)
    if "source_id" not in _canonical_columns(table):
        raise ValueError("independent table realizations require explicit stable source_id values")
    workload = StampWorkload(input_mode="table", input_table=str(config.input_table_path),
                             include_neighbors=False, target_source_ids=config.target_ids,
                             stamp_rows=config.stamp_shape[0], stamp_cols=config.stamp_shape[1])
    plan = StampRunPlan(
        preset_name="legacy-expectation-input-adapter",
        run_config=RunConfig("et_mainsim.execution_config", 1, "et-stamp", config.run_id,
                             paths=RunPaths(), workload=workload),
        paths=ResolvedRunPaths(config.output_root, config.data_root, None, config.focalplane_registry),
        spec=_table_spec(spec), run_dir=config.output_root / config.run_id,
        catalog_cache=config.output_root / config.run_id / "unused-cache.npz",
        input_table_path=config.input_table_path, variability_table_path=config.variability_table_path,
        repo_root=Path(__file__).resolve().parents[2],
    )
    prepared = _prepare_table_inputs(plan, _science_api())
    scenes = {}
    for target in sorted(config.target_ids):
        target_spec = _target_spec(plan, target=prepared.targets[target], psf_id=prepared.psf_ids[target],
                                   source_input_truth=prepared.source_input_truth[target])
        # Stable physical identity, independent of row order, selected targets,
        # output directories, worker ranks and group scheduling.
        identity = _digest(["et_mainsim.independent_table.v1", spec.science_profile.science_realization_id, target])
        profile = replace(target_spec.science_profile, science_realization_id=int(identity[:15], 16))
        scenes[target] = (replace(target_spec, science_profile=profile), prepared.catalogs[target],
                          prepared.source_variability[target])
    return scenes, dict(prepared.provenance)


def _prepare(config):
    from photsim7.specs import SimulationSpec
    from photsim7.catalogs import PreparedStarCatalog
    from photsim7.data_registry import DataRegistry
    from photsim7.source_variability import SourceVariability
    from photsim7.input_identity import simulation_asset_identity
    from photsim7.analysis import ReferenceAperture
    from photsim7.pipelines import build_simulation_context

    config = LegacyExpectationConfig.from_mapping(config.to_dict())
    files = _input_files(config)
    spec = SimulationSpec.from_json_dict(_json(config.spec_path))
    if spec.science_profile.profile_id != PROFILE:
        raise ValueError(f"explicit science_profile.profile_id={PROFILE!r} is required")
    n_raw = spec.observation.n_raw_frames_per_coadd
    n_frames = spec.observation.resolved_n_frames
    if n_frames % n_raw:
        raise ValueError("the observation must contain complete raw groups")
    if not spec.sky.subtract_nonstellar_mean:
        raise ValueError("this workflow requires subtract_nonstellar_mean=true; no fitted background is implicit")
    if spec.cosmic_rays.enabled:
        raise ValueError("legacy expectation v1 does not support cosmic rays")
    if spec.psf.mode != "stamp":
        raise ValueError("legacy expectation rendering requires psf.mode='stamp'")
    registry = DataRegistry(data_root=config.data_root)
    if config.kind == "table-stamp":
        scenes, source_truth = _table_inputs(config, spec)
    else:
        payload = _json(config.catalog_path)
        if set(payload) != {"data", "metadata", "raw_source_arrays"}:
            raise ValueError("prepared catalog JSON requires data, metadata and raw_source_arrays")
        catalog = PreparedStarCatalog(star_data=payload["data"], metadata=payload["metadata"],
                                      raw_source_arrays=payload["raw_source_arrays"])
        available = set(np.asarray(catalog.star_data["source_id"]).tolist())
        if not set(config.target_ids) <= available:
            raise ValueError("requested targets are missing from the prepared catalog")
        variability = None
        if config.variability_path:
            values = _json(config.variability_path)
            if set(values) != {"source_ids", "relative_flux"}:
                raise ValueError("variability JSON requires source_ids and relative_flux on the complete raw grid")
            variability = SourceVariability(**values)
            if variability.n_raw_frames != n_frames:
                raise ValueError("variability must cover the complete observation raw grid")
            if not set(np.asarray(variability.source_ids).tolist()) <= available:
                raise ValueError("variability names sources absent from the prepared catalog")
        scenes, source_truth = {None: (spec, catalog, variability)}, payload["metadata"]
    shape = tuple(spec.detector.shape) if config.kind == "full-frame" else config.stamp_shape
    with np.load(config.aperture_path, allow_pickle=False) as bundle:
        if set(bundle.files) != {str(v) for v in config.target_ids}:
            raise ValueError("aperture NPZ must name every selected target exactly once")
        apertures = {t: ReferenceAperture(bundle[str(t)]) for t in sorted(config.target_ids)}
    if any(a.weights.shape != shape for a in apertures.values()):
        raise ValueError("aperture shape must match the full image or requested stamp")
    scope_count = spec.instrument.telescope_count
    group_pixels = int(np.prod(shape)) * scope_count * len(config.target_ids)
    if group_pixels > config.max_group_pixels:
        raise ValueError(f"group pixel budget exceeded: {group_pixels} > {config.max_group_pixels}")
    contexts = {key: build_simulation_context(scene[0], data_registry=registry,
                 spacecraft_id=config.spacecraft_id,
                 absolute_raw_frame_start_index=config.absolute_raw_frame_start_index)
                for key, scene in scenes.items()}
    assets = {str(k): simulation_asset_identity(s[0], registry) for k, s in scenes.items()}
    if files != _input_files(config):
        raise ValueError("scientific inputs changed while preparing the run")
    identity_config = config.to_dict()
    for key in ("output_root", "run_id"):
        identity_config.pop(key)
    identity = {
        "config": identity_config, "files": files, "assets": assets,
        "effective_specs": {str(k): c.effective_spec.to_json_dict() for k, c in contexts.items()},
        "runtime": _runtime_identity(spec),
    }
    plan = {
        "schema_id": RUN_SCHEMA, "profile_id": PROFILE, "kind": config.kind,
        "scene_policy": "independent_target_per_row" if config.kind == "table-stamp" else "same_exposure",
        "run_dir": str(config.output_root / config.run_id), "output_kind": config.output_kind,
        "raw_frame_count": n_frames, "raw_frames_per_coadd": n_raw,
        "group_count": n_frames // n_raw, "scope_count": scope_count,
        "target_ids": sorted(config.target_ids), "group_pixels": group_pixels,
        "cadence_policy": "different group lengths are independent realizations, not nested raw noise",
        "remeasurement_supported": config.output_kind == "pixels",
        "identity": identity, "identity_sha256": _digest(identity), "source_truth": source_truth,
    }
    return SimpleNamespace(config=config, spec=spec, registry=registry, scenes=scenes,
                           contexts=contexts, apertures=apertures, plan=plan)


def build_legacy_expectation_plan(config):
    """Resolve inputs and assets without rendering or creating a run directory."""
    with _numeric_policy(config.cpu_threads):
        return _prepare(config).plan


def _assert_inputs(prepared):
    from photsim7.input_identity import simulation_asset_identity
    config, identity = prepared.config, prepared.plan["identity"]
    if _input_files(config) != identity["files"] or any(
        simulation_asset_identity(scene[0], prepared.registry) != identity["assets"][str(key)]
        for key, scene in prepared.scenes.items()
    ):
        raise ValueError("scientific inputs/assets changed during execution")


def _inventory(directory):
    result = {}
    for path in sorted(directory.rglob("*")):
        _no_symlinks(path)
        if path.is_file() and path != directory / "group.json":
            result[path.relative_to(directory).as_posix()] = {"sha256": _sha(path), "size_bytes": path.stat().st_size}
    return result


def read_legacy_expectation_group(directory, *, identity_sha256=None, group_index=None):
    """Verify every companion before returning the compact electron light curves."""
    directory = Path(directory)
    _no_symlinks(directory)
    manifest = _json(directory / "group.json")
    if manifest.get("schema_id") != GROUP_SCHEMA:
        raise ValueError("unsupported legacy expectation group schema")
    if identity_sha256 is not None and manifest.get("identity_sha256") != identity_sha256:
        raise ValueError("group scientific identity differs from the run")
    if group_index is not None and manifest.get("group_index") != group_index:
        raise ValueError("group index differs from its scheduled item")
    if _inventory(directory) != manifest.get("files"):
        raise ValueError("group companions are missing, modified or unexpected")
    with np.load(directory / "light_curves.npz", allow_pickle=False) as bundle:
        arrays = {name: bundle[name] for name in bundle.files}
    layout = {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in arrays.items()}
    if layout != manifest.get("light_curve_arrays"):
        raise ValueError("light curve array layout differs")
    return SimpleNamespace(arrays=arrays, manifest=manifest)


def _coadd_read(path):
    manifest = _json(path / "manifest.json")
    artifact = manifest["artifacts"]["arrays.npz"]
    if _sha(path / "arrays.npz") != artifact["sha256"]:
        raise ValueError("coadd checksum mismatch")
    with np.load(path / "arrays.npz", allow_pickle=False) as bundle:
        arrays = {k: bundle[k] for k in bundle.files}
    layout = {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in arrays.items()}
    if layout != artifact["arrays"]:
        raise ValueError("coadd array layout mismatch")
    return SimpleNamespace(arrays=arrays, manifest=manifest, image=arrays["coadd_electrons"])


def _services(prepared, key):
    from photsim7.pipelines import build_multiscope_full_frame_services, build_multiscope_stamp_services
    factory = build_multiscope_full_frame_services if prepared.config.kind == "full-frame" else build_multiscope_stamp_services
    return factory(prepared.contexts[key], catalog=prepared.scenes[key][1])


def _produce_group(prepared, index, directory, service_cache):
    from photsim7.pipelines import run_legacy_expectation_full_frame, run_legacy_expectation_stamp
    from photsim7.analysis import reduce_coadd_sequence, combine_scope_light_curves

    config = prepared.config
    started = time.perf_counter()
    coadds = {}
    # Full frames are rendered once per scope, even when multiple targets are
    # measured. Temporary pixel bundles bound memory in both delivery modes.
    for target in ([None] if config.kind == "full-frame" else sorted(config.target_ids)):
        key = target if config.kind == "table-stamp" else None
        if key not in service_cache:
            service_cache.clear()  # table targets never accumulate service graphs
            service_cache[key] = _services(prepared, key)
        multi = service_cache[key]
        for service in multi.services:
            scope = str(service.context.scope_identity.scope_id)
            name = f"coadds/target_{target}/scope_{scope}"
            path = directory / name
            path.parent.mkdir(parents=True, exist_ok=True)
            kwargs = dict(coadd_index=index, source_variability=prepared.scenes[key][2],
                          photon_noise=config.photon_noise, readout_noise=config.readout_noise,
                          save_raw_expectations=config.save_raw_expectations)
            if config.kind == "full-frame":
                result = run_legacy_expectation_full_frame(service, **kwargs)
            else:
                result = run_legacy_expectation_stamp(service, target_source_id=target,
                                                      stamp_shape=config.stamp_shape, **kwargs)
            result.save(path)
            coadds[target, scope] = name
            del result
    measurements, arrays = {}, {}
    for target in sorted(config.target_ids):
        curves = []
        for (source, scope), name in coadds.items():
            if source is not None and source != target:
                continue
            result = _coadd_read(directory / name)
            curve = reduce_coadd_sequence([result], aperture=prepared.apertures[target], target_id=target,
                        normalization_electron_rate=config.normalization_electron_rate,
                        provenance={"application_identity_sha256": prepared.plan["identity_sha256"],
                                    "scene_policy": prepared.plan["scene_policy"], "coadd_path": name})
            label = f"target_{target}/scope_{scope}"
            if config.output_kind == "pixels":
                path = directory / "photometry" / label
                path.parent.mkdir(parents=True, exist_ok=True)
                curve.save(path)
            arrays.update({f"{label}__{k}": curve.arrays[k] for k in _SINGLE_FIELDS})
            measurements[label] = curve.manifest
            curves.append(curve)
            del result
        combined = combine_scope_light_curves(curves)
        label = f"target_{target}/combined"
        if config.output_kind == "pixels":
            combined.save(directory / "photometry" / label)
        arrays.update({f"{label}__{k}": v for k, v in combined.arrays.items() if not k.startswith("member_")})
        measurements[label] = combined.manifest
        del curves, combined
    if config.output_kind == "lightcurve":
        shutil.rmtree(directory / "coadds")
    np.savez_compressed(directory / "light_curves.npz", **arrays)
    manifest = {
        "schema_id": GROUP_SCHEMA, "group_index": index,
        "identity_sha256": prepared.plan["identity_sha256"],
        "profile_id": PROFILE, "scene_policy": prepared.plan["scene_policy"],
        "output_kind": config.output_kind, "remeasurement_supported": config.output_kind == "pixels",
        "measurements": measurements, "elapsed_s": time.perf_counter() - started,
        "light_curve_arrays": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in arrays.items()},
        "files": _inventory(directory),
    }
    _atomic_write_json(directory / "group.json", manifest)
    read_legacy_expectation_group(directory, identity_sha256=prepared.plan["identity_sha256"], group_index=index)


def _run_paths(config):
    run = config.output_root / config.run_id
    locks = config.output_root / ".run_locks" / config.run_id
    _no_symlinks(run)
    _no_symlinks(locks)
    return run, locks


def _initialize(prepared):
    run, locks = _run_paths(prepared.config)
    with _lock(locks / "manifest.lock", wait=True):
        if run.exists():
            for name in ("run_manifest.json", "progress.json", "groups", ".staging"):
                _no_symlinks(run / name)
            saved = _json(run / "run_manifest.json")
            if saved != prepared.plan:
                raise ValueError("saved run manifest/configuration/scientific identity differs")
            return
        run.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{run.name}.init-", dir=run.parent))
        try:
            _atomic_write_json(temporary / "run_manifest.json", prepared.plan)
            _atomic_write_json(temporary / "progress.json", {})
            (temporary / "groups").mkdir()
            (temporary / ".staging").mkdir()
            os.rename(temporary, run)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)


def _scan(prepared, *, require_complete, checkpoint=True):
    run, locks = _run_paths(prepared.config)
    with _lock(locks / "manifest.lock", wait=True):
        return _scan_locked(prepared, run, require_complete=require_complete, checkpoint=checkpoint)


def _scan_locked(prepared, run, *, require_complete, checkpoint):
    _no_symlinks(run / "run_manifest.json")
    if _json(run / "run_manifest.json") != prepared.plan:
        raise ValueError("run manifest/scientific identity differs")
    expected = {f"group_{i:08d}": i for i in range(prepared.plan["group_count"])}
    _no_symlinks(run / "groups")
    present = {p.name: p for p in (run / "groups").iterdir()}
    if set(present) - set(expected):
        raise ValueError("unexpected group artifacts")
    _no_symlinks(run / "progress.json")
    prior = _json(run / "progress.json")
    if set(prior) - set(present):
        raise ValueError("previously published groups are missing")
    records = []
    current = {}
    for name in sorted(present):
        checksum = _sha(present[name] / "group.json")
        if name in prior and prior[name] != checksum:
            raise ValueError("published group manifest was modified")
        product = read_legacy_expectation_group(present[name], identity_sha256=prepared.plan["identity_sha256"],
                                               group_index=expected[name])
        current[name] = checksum
        records.append({"group_index": expected[name], "manifest_sha256": checksum,
                        "elapsed_s": product.manifest["elapsed_s"]})
    # A process may die between directory rename and progress checkpoint.
    # Adopt only complete, fully verified matching groups in that window.
    if checkpoint and current != prior:
        _atomic_write_json(run / "progress.json", current)
    missing = sorted(expected[n] for n in set(expected) - set(present))
    if require_complete and missing:
        raise ValueError(f"run is incomplete; missing groups: {missing}")
    return {"schema_id": "et_mainsim.legacy_expectation_status.v1",
            "status": "partial" if missing else "completed", "run_dir": str(run),
            "identity_sha256": prepared.plan["identity_sha256"], "groups": records, "missing_groups": missing}


def verify_legacy_expectation_run(config, *, require_complete=True):
    with _numeric_policy(config.cpu_threads):
        prepared = _prepare(config)
        result = _scan(prepared, require_complete=require_complete, checkpoint=False)
        _assert_inputs(prepared)
        return result


def run_legacy_expectation(config, *, group_indices=None, shard_index=0, shard_count=1):
    """Run/resume complete groups; independent processes may own disjoint shards."""
    _int(shard_count, "shard_count", 1)
    _int(shard_index, "shard_index")
    if shard_index >= shard_count:
        raise ValueError("shard_index must be below shard_count")
    with _numeric_policy(config.cpu_threads):
        prepared = _prepare(config)
        count = prepared.plan["group_count"]
        selected = list(range(count)) if group_indices is None else list(group_indices)
        if len(set(selected)) != len(selected) or any(type(i) is not int or not 0 <= i < count for i in selected):
            raise ValueError("group_indices must be unique indices within the full observation")
        selected = [i for i in selected if i % shard_count == shard_index]
        _initialize(prepared)
        run, locks = _run_paths(config)
        _scan(prepared, require_complete=False)  # reject old corruption before any new publication
        services = {}
        for index in selected:
            with _lock(locks / f"group_{index:08d}.lock"):
                destination = run / "groups" / f"group_{index:08d}"
                if destination.exists():
                    read_legacy_expectation_group(destination, identity_sha256=prepared.plan["identity_sha256"], group_index=index)
                    continue
                _assert_inputs(prepared)
                _no_symlinks(run / ".staging")
                temporary = Path(tempfile.mkdtemp(prefix=f"group_{index:08d}-", dir=run / ".staging"))
                try:
                    _produce_group(prepared, index, temporary, services)
                    _assert_inputs(prepared)
                    if _json(run / "run_manifest.json") != prepared.plan:
                        raise ValueError("run manifest changed before publication")
                    _no_symlinks(destination)
                    with _lock(locks / "manifest.lock", wait=True):
                        os.rename(temporary, destination)
                        progress = _json(run / "progress.json")
                        progress[destination.name] = _sha(destination / "group.json")
                        _atomic_write_json(run / "progress.json", progress)
                finally:
                    if temporary.exists():
                        _no_symlinks(temporary)
                        shutil.rmtree(temporary)
        _assert_inputs(prepared)
        # Never persist a stale global completion flag: truth is the set of
        # atomically published and verified complete groups, even across shards.
        return _scan(prepared, require_complete=False)
