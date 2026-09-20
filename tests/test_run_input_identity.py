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
    tmp_path, workflow
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
