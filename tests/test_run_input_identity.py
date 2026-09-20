import pytest
from dataclasses import replace

from et_mainsim.manifest import ManifestIdentityError, RunManifestStore


def create(store, **changes):
    args = dict(
        workflow="et-full-frame",
        preset="smoke",
        run_id="run",
        simulation_spec={"rng": {"run_seed": 1}},
        execution={"device": "cpu", "workers_per_device": 1, "gpu_ids": []},
        frame_plan={"requested": [0]},
        provenance={},
        input_identity={"assets": {"psf": "old"}, "runtime": "same"},
    )
    args.update(changes)
    return store.create(**args)


def ensure(store, **changes):
    args = dict(
        workflow="et-full-frame",
        run_id="run",
        simulation_spec={"rng": {"run_seed": 1}},
        execution={"device": "cpu", "workers_per_device": 3, "gpu_ids": []},
        input_identity={"assets": {"psf": "old"}, "runtime": "same"},
    )
    args.update(changes)
    return store.ensure_identity(**args)


def test_resume_accepts_worker_change_but_rejects_changed_asset_before_mutation(
    tmp_path,
):
    store = RunManifestStore(tmp_path / "run.json")
    create(store)
    assert ensure(store)["run_id"] == "run"
    original = store.path.read_bytes()
    with pytest.raises(ManifestIdentityError, match="input"):
        ensure(store, input_identity={"assets": {"psf": "edited"}, "runtime": "same"})
    assert store.path.read_bytes() == original


def test_old_run_without_input_evidence_cannot_resume_new_workflow(tmp_path):
    store = RunManifestStore(tmp_path / "run.json")
    create(store, input_identity=None)
    with pytest.raises(ManifestIdentityError, match="input"):
        ensure(store)


def test_backend_change_remains_a_resume_conflict(tmp_path):
    store = RunManifestStore(tmp_path / "run.json")
    create(store)
    with pytest.raises(ManifestIdentityError):
        ensure(
            store,
            execution={"device": "cuda", "workers_per_device": 1, "gpu_ids": ["0"]},
        )


def test_resume_allows_batch_tuning_without_changing_scientific_inputs(tmp_path):
    store = RunManifestStore(tmp_path / "run.json")
    spec = {
        "rng": {"run_seed": 1},
        "psf": {"float_precision": 32, "warp_frame_batch_size": 2},
        "dynamic_effects": {"psd_motion": {"chunk_size": 3, "split_hz": 0.1}},
    }
    create(store, simulation_spec=spec)
    changed = {
        **spec,
        "psf": {**spec["psf"], "warp_frame_batch_size": 7},
        "dynamic_effects": {"psd_motion": {"chunk_size": 11, "split_hz": 0.1}},
    }
    assert ensure(store, simulation_spec=changed)["run_id"] == "run"
    changed["psf"]["float_precision"] = 64
    with pytest.raises(ManifestIdentityError):
        ensure(store, simulation_spec=changed)


@pytest.mark.parametrize("workflow", ["full_frame", "stamp"])
def test_real_workflow_persists_effective_inputs_and_rejects_edited_psf(
    tmp_path, workflow, monkeypatch
):
    from importlib import import_module
    from test_stamp_workflow import _write_test_psf_bundle
    from et_mainsim.presets import load_preset

    module = import_module(f"et_mainsim.workflows.{workflow}")
    loaded = load_preset(
        "et-full-frame-smoke" if workflow == "full_frame" else "et-stamp-smoke"
    )
    root = tmp_path / "data"
    name, _ = _write_test_psf_bundle(root)
    spec = replace(
        loaded.simulation_spec,
        instrument=replace(loaded.simulation_spec.instrument, telescope_count=1),
        detector=replace(loaded.simulation_spec.detector, n_subpixels=3),
        psf=replace(loaded.simulation_spec.psf, bundle_name=name, bundle_sha256=None),
    )
    config = replace(
        loaded.run_config,
        paths=replace(
            loaded.run_config.paths,
            data_root=str(root),
            output_root=str(tmp_path / "out"),
        ),
    )
    plan = module.build_run_plan(
        preset_name=loaded.descriptor.name,
        run_config=config,
        spec=spec,
        repo_root=tmp_path,
    )
    run = module.run_full_frame if workflow == "full_frame" else module.run_stamp
    if workflow == "full_frame":
        import torch

        cuda_plan = replace(
            plan,
            spec=replace(plan.spec, psf=replace(plan.spec.psf, compute_device="cuda")),
            run_config=replace(
                plan.run_config,
                execution=replace(
                    plan.run_config.execution,
                    device="cuda",
                    backend="local-subprocess",
                    gpu_ids=("7",),
                ),
            ),
        )
        with monkeypatch.context() as context:

            def forbidden(*args, **kwargs):
                raise AssertionError("catalog-only preparation queried CUDA")

            context.setattr(torch.cuda, "get_device_name", forbidden)
            prepared = run(cuda_plan, prepare_catalog_only=True)
        assert "rendering" not in prepared["input_identity"]["runtime"]
    manifest = run(plan)
    assert manifest["input_identity"]["assets"]["psf.bundle"]["sha256"]
    control = manifest["attempts"][-1]["control"]
    assert control["execution"]["device"] == plan.spec.psf.compute_device
    assert control["effective_spec"]["rng"]["run_seed"] == plan.spec.rng.run_seed
    assert (plan.run_dir / "effective_spec.json").is_file()
    assert run(plan)["status"] == "completed"
    # Simulate a coordinator killed after publishing its running status.
    state = RunManifestStore(plan.run_dir / "run_manifest.json")
    state.start_attempt()
    recovered = run(plan)
    assert recovered["status"] == "completed"
    assert recovered["attempts"][-2]["status"] == "interrupted"
    from photsim7.psf.paths import resolve_psf_bundle_filename

    path = resolve_psf_bundle_filename(name, root)
    with path.open("ab") as stream:
        stream.write(b"asset edited")
    before = (plan.run_dir / "run_manifest.json").read_bytes()
    with pytest.raises(ManifestIdentityError, match="input"):
        run(plan)
    assert (plan.run_dir / "run_manifest.json").read_bytes() == before


def test_new_experiment_refreshes_stale_external_catalog_atomically(tmp_path):
    from et_mainsim.inputs import prepare_catalog_input
    from photsim7.data_registry import DataRegistry
    from photsim7.specs import SimulationSpec
    from photsim7.pipelines import build_catalog_from_spec
    from photsim7.catalogs.cache import StarCatalogCache
    from types import SimpleNamespace

    source = tmp_path / "mags.csv"
    source.write_text("mwmsc_gmag\n10\n11\n")
    base = SimulationSpec()
    spec = replace(
        base,
        catalog=replace(
            base.catalog,
            source_type="synthetic_mag_distribution",
            source_path=str(source),
            cache_path=str(tmp_path / "cache.npz"),
            magnitude_column="mwmsc_gmag",
            background_stars_max_mag=20,
        ),
    )
    old = build_catalog_from_spec(spec)
    source.write_text("mwmsc_gmag\n12\n13\n")
    api = SimpleNamespace(
        build_catalog_from_spec=build_catalog_from_spec,
        StarCatalogCache=StarCatalogCache,
    )
    fresh = prepare_catalog_input(
        spec, DataRegistry(tmp_path), api=api, run_dir=tmp_path / "new", force=False
    )
    assert (
        fresh.star_data["gaia_g_mag"].tolist() != old.star_data["gaia_g_mag"].tolist()
    )
    assert (
        StarCatalogCache.read(tmp_path / "cache.npz").star_data["gaia_g_mag"].tolist()
        == fresh.star_data["gaia_g_mag"].tolist()
    )


def test_run_lock_refuses_another_coordinator_and_releases_on_error(tmp_path):
    from et_mainsim.inputs import run_lock

    with pytest.raises(RuntimeError, match="intentional"):
        with run_lock(tmp_path / "run"):
            with pytest.raises(RuntimeError, match="already active"):
                with run_lock(tmp_path / "run"):
                    pass
            raise RuntimeError("intentional")
    with run_lock(tmp_path / "run"):
        pass


def test_recovery_refuses_workers_left_running_by_a_dead_coordinator(tmp_path):
    from et_mainsim.inputs import run_lock, worker_lock

    with worker_lock(tmp_path / "run"):
        with pytest.raises(RuntimeError, match="workers are still active"):
            with run_lock(tmp_path / "run"):
                pass
    with run_lock(tmp_path / "run"):
        with worker_lock(tmp_path / "run"):
            pass


def test_catalog_identity_tracks_geometry_and_ignores_only_execution_trace_headers():
    from et_mainsim.inputs import catalog_identity
    from photsim7.catalogs.sources import PreparedStarCatalog

    catalog = PreparedStarCatalog(
        star_data={"x0": [0.0], "y0": [0.0]},
        metadata={
            "geometry": {"reference_field_angle_deg": 10.0},
            "rng_trace": {"run_seed": 1},
            "request_validation": {"mode": "generated"},
        },
    )
    changed_geometry = catalog.with_metadata(
        geometry={"reference_field_angle_deg": 12.0}
    )
    assert catalog_identity(changed_geometry) != catalog_identity(catalog)
    read_back = catalog.with_metadata(
        rng_trace={"run_seed": 2}, request_validation={"mode": "cache_exact"}
    )
    assert catalog_identity(read_back) == catalog_identity(catalog)


def test_worker_consumes_verified_catalog_snapshot_without_reopening_cache(
    tmp_path, monkeypatch
):
    from test_full_frame_workflow import _selection_ready_worker_request
    from et_mainsim.workflows.full_frame import run_worker

    request, api = _selection_ready_worker_request(tmp_path)
    verified_catalog = api.StarCatalogCache.read(request.catalog_cache)
    monkeypatch.setattr(
        "et_mainsim.inputs.verify_worker_inputs",
        lambda *args, **kwargs: verified_catalog,
    )

    class ChangedCache:
        @staticmethod
        def read(path):
            raise AssertionError("mutable cache was reopened after input verification")

    api.StarCatalogCache = ChangedCache
    result = run_worker(request, science_api=api)
    assert result.rendered == request.frame_indices


def test_catalog_identity_is_stable_across_cache_serialization(tmp_path):
    from et_mainsim.inputs import catalog_identity
    from photsim7.catalogs.sources import PreparedStarCatalog
    from photsim7.catalogs.cache import StarCatalogCache

    catalog = PreparedStarCatalog(
        star_data={"x0": [0.0], "y0": [0.0]},
        metadata={"geometry": {"kind": "explicit"}},
    )
    path = tmp_path / "stars.npz"
    StarCatalogCache.write(path, catalog)
    assert catalog_identity(StarCatalogCache.read(path)) == catalog_identity(catalog)


def test_runtime_identity_detects_focalplane_code_edits_at_the_same_version(
    tmp_path, monkeypatch
):
    import et_coord
    from et_mainsim.inputs import runtime_identity
    from et_mainsim.presets import load_preset

    module = tmp_path / "et_coord" / "__init__.py"
    module.parent.mkdir()
    module.write_text("projection_revision = 1\n")
    monkeypatch.setattr(et_coord, "__file__", str(module))
    spec = load_preset("et-full-frame-smoke").simulation_spec
    before = runtime_identity(spec)
    module.write_text("projection_revision = 2\n")
    assert runtime_identity(spec) != before


def test_resume_rejects_preview_change(tmp_path):
    store = RunManifestStore(tmp_path / "run.json")
    create(store, execution={"device": "cpu", "preview_count": 0})
    with pytest.raises(ManifestIdentityError, match="execution"):
        ensure(store, execution={"device": "cpu", "preview_count": 3})


def test_launch_lease_protects_child_before_worker_initialization(tmp_path):
    import sys
    from et_mainsim.inputs import launch_worker, run_lock

    run = tmp_path / "run"
    with run_lock(run):
        process = launch_worker(
            [sys.executable, "-c", "import time; time.sleep(30)"], run_dir=run
        )
    try:
        with pytest.raises(RuntimeError, match="workers are still active"):
            with run_lock(run):
                pass
    finally:
        process.terminate()
        process.wait(timeout=10)
    with run_lock(run):
        pass


def test_catalog_only_runtime_does_not_probe_cuda(monkeypatch):
    import torch
    from et_mainsim.inputs import runtime_identity
    from et_mainsim.presets import load_preset

    spec = load_preset("et-full-frame-production").simulation_spec

    def forbidden(*args, **kwargs):
        raise AssertionError("catalog-only preparation queried CUDA")

    monkeypatch.setattr(torch.cuda, "get_device_name", forbidden)
    result = runtime_identity(spec, catalog_only=True)
    assert "rendering" not in result


@pytest.mark.parametrize("mask,selector", [(None, "1"), ("1,3", "3")])
def test_runtime_uses_selected_gpu_and_matches_worker(
    tmp_path, monkeypatch, mask, selector
):
    import torch
    from et_mainsim.inputs import runtime_identity
    from et_mainsim.presets import load_preset

    spec = load_preset("et-full-frame-production").simulation_spec
    if mask is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_name",
        lambda index=None: ["old GPU", "selected GPU"][index or 0],
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda index=None: [(8, 0), (9, 0)][index or 0],
    )
    parent = runtime_identity(spec, gpu_ids=(selector,))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", selector)
    monkeypatch.setattr(
        torch.cuda, "get_device_name", lambda index=None: "selected GPU"
    )
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index=None: (9, 0))
    assert runtime_identity(spec) == parent


def test_runtime_rejects_gpu_outside_allocation(monkeypatch):
    from et_mainsim.inputs import runtime_identity
    from et_mainsim.presets import load_preset

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES"):
        runtime_identity(
            load_preset("et-full-frame-production").simulation_spec, gpu_ids=("0",)
        )


def test_catalog_stage_promotes_runtime_but_preserves_scientific_identity(tmp_path):
    store = RunManifestStore(tmp_path / "run.json")
    spec = {"rng": {"run_seed": 1}, "psf": {"compute_device": "cpu"}}
    base = {"assets": {}, "runtime": {"python": "same", "photsim7_code": "same"}}
    create(store, simulation_spec=spec, input_identity=base)
    store.start_attempt()
    store.transition("completed", completion={"catalog_only": True})
    rendering = {
        "assets": {"psf": "new"},
        "runtime": {**base["runtime"], "rendering": {"device": "cuda"}},
    }
    new_spec = {**spec, "psf": {"compute_device": "cuda"}}
    ensure(
        store,
        simulation_spec=new_spec,
        input_identity=rendering,
        execution={"device": "cuda"},
    )
    with pytest.raises(ManifestIdentityError, match="input"):
        ensure(
            store,
            input_identity={
                **rendering,
                "runtime": {**rendering["runtime"], "photsim7_code": "changed"},
            },
        )
    with pytest.raises(ManifestIdentityError, match="scientific"):
        ensure(
            store,
            simulation_spec={**new_spec, "rng": {"run_seed": 2}},
            input_identity=rendering,
            execution={"device": "cuda"},
        )


def test_forced_cache_refresh_rejects_source_overlap_before_building(tmp_path):
    from types import SimpleNamespace
    from et_mainsim.inputs import prepare_catalog_input
    from photsim7.data_registry import DataRegistry
    from photsim7.specs import SimulationSpec

    spec = SimulationSpec()
    source = tmp_path / "source"
    source.mkdir()
    spec = replace(
        spec,
        catalog=replace(
            spec.catalog, source_path=str(source), cache_path=str(source / "cache.npz")
        ),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("cache refresh started before path validation")

    with pytest.raises(ValueError, match="outside"):
        prepare_catalog_input(
            spec,
            DataRegistry(tmp_path),
            api=SimpleNamespace(build_catalog_from_spec=forbidden),
            run_dir=tmp_path / "run",
            force=True,
        )


def test_runtime_resolves_gpu_uuid_and_rejects_mixed_architecture(monkeypatch):
    from types import SimpleNamespace
    import torch
    from et_mainsim.inputs import runtime_identity
    from et_mainsim.presets import load_preset

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda index: SimpleNamespace(uuid=f"GPU-{index}abc"),
    )
    monkeypatch.setattr(
        torch.cuda, "get_device_name", lambda index: ["older", "newer"][index]
    )
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda index: [(8, 0), (9, 0)][index]
    )
    spec = load_preset("et-full-frame-production").simulation_spec
    assert (
        runtime_identity(spec, gpu_ids=("GPU-1",))["rendering"]["cuda"]["device_name"]
        == "newer"
    )
    with pytest.raises(ValueError, match="homogeneous"):
        runtime_identity(spec, gpu_ids=("0", "1"))
