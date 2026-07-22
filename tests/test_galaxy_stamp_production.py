from __future__ import annotations

import json
import shutil
from types import SimpleNamespace

import numpy as np
import pytest


def _stub_detector_physical_pixel_shape(
    monkeypatch,
    stamp_inputs,
    *,
    detector_id: str,
    pixel_width: float,
    pixel_height: float,
) -> None:
    detector = SimpleNamespace(
        pixel_width=pixel_width,
        pixel_height=pixel_height,
    )

    class _Registry:
        def get_detector(self, requested_detector_id: str):
            assert requested_detector_id == detector_id
            return detector

    monkeypatch.setattr(
        stamp_inputs,
        "_load_focalplane_registry",
        lambda *_args, **_kwargs: _Registry(),
    )


def test_manifest_resource_can_move_with_run_root_without_losing_identity(tmp_path) -> None:
    """A prepared run copied to H100 must not retain a local absolute path."""

    from et_mainsim.galaxy_stamp_production import (
        _resolve_manifest_resource,
        _same_file_content_identity,
    )
    from et_mainsim.stamp_inputs import file_identity

    local_run_root = tmp_path / "local" / "galaxy-90d"
    local_resource = local_run_root / "inputs" / "time_shards.json"
    local_resource.parent.mkdir(parents=True)
    local_resource.write_text('{"time": "plan"}\n', encoding="utf-8")
    record = {
        "path": str(local_resource),
        "relative_path": "inputs/time_shards.json",
        "file_identity": file_identity(local_resource),
    }

    h100_parent = tmp_path / "h100"
    h100_run_root = h100_parent / local_run_root.name
    shutil.copytree(local_run_root, h100_run_root)

    resolved = _resolve_manifest_resource(
        h100_run_root,
        record,
        label="time shard plan",
    )

    assert resolved == h100_run_root / "inputs" / "time_shards.json"
    assert _same_file_content_identity(file_identity(resolved), record["file_identity"])


def test_manifest_resource_rejects_relative_path_escape(tmp_path) -> None:
    from et_mainsim.galaxy_stamp_production import _resolve_manifest_resource

    run_root = tmp_path / "run"
    run_root.mkdir()

    with pytest.raises(ValueError, match="escapes prepared run root"):
        _resolve_manifest_resource(
            run_root,
            {"relative_path": "../outside.json"},
            label="factor snapshot",
        )


def test_manifest_resource_requires_a_nonabsolute_relative_path(tmp_path) -> None:
    from et_mainsim.galaxy_stamp_production import _resolve_manifest_resource

    run_root = tmp_path / "run"
    resource = run_root / "inputs" / "factor.npz"
    resource.parent.mkdir(parents=True)
    resource.write_bytes(b"frozen factor")

    with pytest.raises(ValueError, match="relative_path must be relative"):
        _resolve_manifest_resource(
            run_root,
            {"path": str(resource), "relative_path": str(resource)},
            label="factor snapshot",
        )

    with pytest.raises(ValueError, match="requires relative_path"):
        _resolve_manifest_resource(
            run_root,
            {"path": str(resource)},
            label="factor snapshot",
        )


def test_semantic_registry_identity_allows_relocation_but_not_content_drift() -> None:
    from et_mainsim.galaxy_stamp_production import _same_semantic_registry_identity

    prepared = {
        "schema_id": "et_coord.semantic_registry_identity.v1",
        "schema_version": 1,
        "registry_data_dir": "/workstation/et_focalplane/data",
        "sha256": "a" * 64,
        "owner_attestation": {"revision": "frozen-v1"},
    }
    relocated = {
        **prepared,
        "registry_data_dir": "/cluster/home/cxgao/ET/et_focalplane/data",
    }

    assert _same_semantic_registry_identity(prepared, relocated)
    assert not _same_semantic_registry_identity(
        prepared,
        {**relocated, "sha256": "b" * 64},
    )


def test_formal_registry_gate_requires_owner_freeze_and_verified_attestation() -> None:
    from et_mainsim.galaxy_stamp_production import _validate_formal_registry_gate

    prepared = {
        "schema_id": "et_coord.semantic_registry_identity.v1",
        "schema_version": 1,
        "registry_data_dir": "/workstation/et_focalplane/data",
        "sha256": "a" * 64,
        "semantic_content_sha256": "b" * 64,
        "freeze_status": "owner_frozen",
        "owner_approval_required": False,
        "owner_attestation": {"record": "frozen"},
        "algorithm": {"revision": "v1"},
    }
    runtime = {
        **prepared,
        "registry_data_dir": "/cluster/home/cxgao/ET/et_focalplane/data",
    }
    verified = {"verified": True, "errors": []}

    _validate_formal_registry_gate(prepared, runtime, verified)

    with pytest.raises(ValueError, match="owner_frozen"):
        _validate_formal_registry_gate(
            {**prepared, "freeze_status": "candidate"},
            runtime,
            verified,
        )
    with pytest.raises(ValueError, match="attestation verification"):
        _validate_formal_registry_gate(
            prepared,
            runtime,
            {"verified": False, "errors": ["content mismatch"]},
        )


def test_map_curve_to_detector_rejects_coordinate_outside_physical_detector_bounds(
    tmp_path,
    monkeypatch,
) -> None:
    import et_mainsim.galaxy_stamp_production as production
    import et_mainsim.stamp_inputs as stamp_inputs

    registry = tmp_path / "focalplane"
    registry.mkdir()
    monkeypatch.setattr(
        stamp_inputs,
        "_sky_to_focal",
        lambda *_args, **_kwargs: SimpleNamespace(
            status="ok",
            detector_id="main_ld",
            xpix=9120.25,
            ypix=100.0,
            field_x_deg=1.0,
            field_y_deg=2.0,
            residual_arcsec=0.01,
        ),
    )
    _stub_detector_physical_pixel_shape(
        monkeypatch,
        stamp_inputs,
        detector_id="main_ld",
        pixel_width=9120.0,
        pixel_height=8900.0,
    )

    with pytest.raises(ValueError, match="physical focal-plane detector bounds"):
        production._map_curve_to_detector(
            focalplane_registry=registry,
            registry_sha256="registry-fixture",
            ra_deg=10.0,
            dec_deg=-20.0,
        )


def test_prepare_rejects_invalid_physical_mapping_before_creating_run_root(
    tmp_path,
    monkeypatch,
) -> None:
    """A bad coordinate must not leave a no-resume production root behind."""

    import et_mainsim.galaxy_stamp_production as production
    import et_mainsim.stamp_inputs as stamp_inputs
    from et_mainsim.galaxy_lightcurves import GalaxyLightCurve

    input_fits = tmp_path / "galaxy.fits"
    input_fits.write_bytes(b"synthetic source fixture")
    data_root = tmp_path / "data-root"
    registry = tmp_path / "registry"
    data_root.mkdir()
    registry.mkdir()
    curve = GalaxyLightCurve(
        source_id=42,
        gaia_g_mag=11.5,
        ra_deg=10.0,
        dec_deg=-20.0,
        source_class="fixture",
        native_time_seconds=np.array([0.0, 30.0]),
        clean_flux_factor=np.array([1.0, 1.3]),
        input_identity={"sha256": "fixture"},
    )

    class _Spec:
        def to_json_dict(self):
            return {"schema": "test-spec"}

    monkeypatch.setattr(
        production,
        "load_galaxy_lightcurves",
        lambda path, source_ids: {42: curve},
    )
    monkeypatch.setattr(
        stamp_inputs,
        "focalplane_registry_identity",
        lambda path: {
            "schema_id": "et_coord.semantic_registry_identity.v1",
            "registry_data_dir": str(path),
            "sha256": "registry-fixture",
        },
    )
    monkeypatch.setattr(
        stamp_inputs,
        "_sky_to_focal",
        lambda *_args, **_kwargs: SimpleNamespace(
            status="ok",
            detector_id="main_ld",
            xpix=9120.25,
            ypix=100.0,
            field_x_deg=1.0,
            field_y_deg=2.0,
            residual_arcsec=0.01,
        ),
    )
    _stub_detector_physical_pixel_shape(
        monkeypatch,
        stamp_inputs,
        detector_id="main_ld",
        pixel_width=9120.0,
        pixel_height=8900.0,
    )
    monkeypatch.setattr(
        production,
        "build_galaxy_independent_production_spec",
        lambda **kwargs: _Spec(),
    )

    config = production.GalaxyStampProductionConfig(
        input_fits=input_fits,
        output_root=tmp_path / "prepared",
        run_id="invalid-physical-mapping",
        data_root=data_root,
        focalplane_registry=registry,
        source_ids=(42,),
        duration_days=30.0 / 86_400.0,
        cadence_seconds=(30.0,),
        max_raw_frames_per_shard=3,
        device="cpu",
    )

    with pytest.raises(ValueError, match="physical focal-plane detector bounds"):
        production.prepare_galaxy_independent_production(config)

    assert not config.run_root.exists()


def test_prepare_v2_manifest_records_relocatable_resources(tmp_path, monkeypatch) -> None:
    """The actual preparation manifest must exercise the v2 relative records."""

    import et_mainsim.galaxy_stamp_production as production
    import et_mainsim.stamp_inputs as stamp_inputs
    from et_mainsim.galaxy_lightcurves import GalaxyLightCurve

    input_fits = tmp_path / "galaxy.fits"
    input_fits.write_bytes(b"synthetic source fixture")
    data_root = tmp_path / "data-root"
    registry = tmp_path / "registry"
    data_root.mkdir()
    registry.mkdir()
    curve = GalaxyLightCurve(
        source_id=42,
        gaia_g_mag=11.5,
        ra_deg=10.0,
        dec_deg=-20.0,
        source_class="fixture",
        native_time_seconds=np.array([0.0, 30.0]),
        clean_flux_factor=np.array([1.0, 1.3]),
        input_identity={"sha256": "fixture"},
    )

    class _Spec:
        def to_json_dict(self):
            return {"schema": "test-spec"}

    monkeypatch.setattr(
        production,
        "load_galaxy_lightcurves",
        lambda path, source_ids: {42: curve},
    )
    monkeypatch.setattr(
        stamp_inputs,
        "focalplane_registry_identity",
        lambda path: {
            "schema_id": "et_coord.semantic_registry_identity.v1",
            "registry_data_dir": str(path),
            "sha256": "registry-fixture",
        },
    )
    monkeypatch.setattr(
        production,
        "_map_curve_to_detector",
        lambda **kwargs: {
            "detector_id": "main_lu",
            "detector_xpix": 10.0,
            "detector_ypix": 20.0,
            "field_x_deg": 1.0,
            "field_y_deg": 2.0,
            "focalplane_residual_arcsec": 0.0,
            "field_angle_deg": float(np.hypot(1.0, 2.0)),
        },
    )
    monkeypatch.setattr(
        production,
        "build_galaxy_independent_production_spec",
        lambda **kwargs: _Spec(),
    )
    monkeypatch.setattr(production, "collect_provenance", lambda path: {"test": True})

    prepared = production.prepare_galaxy_independent_production(
        production.GalaxyStampProductionConfig(
            input_fits=input_fits,
            output_root=tmp_path / "prepared",
            run_id="galaxy-v2",
            data_root=data_root,
            focalplane_registry=registry,
            source_ids=(42,),
            duration_days=30.0 / 86_400.0,
            cadence_seconds=(30.0,),
            max_raw_frames_per_shard=3,
            device="cpu",
        )
    )
    manifest = json.loads(prepared.manifest_path.read_text(encoding="utf-8"))

    assert manifest["schema_version"] == 2
    assert manifest["delivery"]["time_plan_relative_path"] == "inputs/time_shards.json"
    assert manifest["targets"][0]["factor_snapshot_relative_path"] == (
        "inputs/galaxy_factor_snapshots/source_42.npz"
    )
    assert manifest["targets"][0]["target_table"]["relative_path"] == (
        "inputs/target_tables/targets_main_lu.ecsv"
    )

    relocated = tmp_path / "h100" / "galaxy-v2"
    shutil.copytree(prepared.run_root, relocated)
    relocated_manifest_path, relocated_manifest = production._load_manifest(
        relocated / "production_manifest.json"
    )
    recovered_plan = production._load_time_plan(
        relocated_manifest_path.parent,
        relocated_manifest,
    )
    assert recovered_plan.accepted_raw_frame_count == 3


def test_formal_galaxy_production_spec_freezes_delivery_and_sd20_policy() -> None:
    from et_mainsim.galaxy_stamp_production import (
        build_galaxy_independent_production_spec,
    )

    spec = build_galaxy_independent_production_spec(
        n_raw_frames=18,
        raw_exposure_seconds=10.0,
        device="cpu",
        run_seed=12345,
    )

    assert spec.observation.resolved_n_frames == 18
    assert spec.observation.exposure_duration.to_value("s") == 10.0
    assert spec.instrument.telescope_count == 1
    assert spec.psf.mode == "stamp"
    assert spec.psf.compute_device == "cpu"
    assert spec.artifacts.background_output_policy.value == "expectation"
    assert spec.sky.subtract_nonstellar_mean is False
    assert spec.detector.pixel_scale.to_value("arcsec / pix") == 4.83
    assert spec.detector_response.enable_inter_pixel_response is False
    assert spec.detector_response.enable_intra_pixel_response is False
    assert spec.detector_response.enable_pixel_phase_response is False
    assert spec.detector_response.scripted_sensitivity_enabled is False
    assert spec.detector_response.whole_pixel_gain_normal_enabled is False
    assert spec.detector_response.whole_pixel_gain_sinusoidal_enabled is False
    assert spec.detector_response.enable_flat_field_correction is False
    assert spec.rng.run_seed == 12345


def test_galaxy_worker_records_case_invariant_physical_rng_pairing(
    tmp_path,
    monkeypatch,
) -> None:
    """Physical RNG derives from SimulationContext, not execution labels."""

    from dataclasses import replace

    import et_mainsim.galaxy_stamp_production as production
    import et_mainsim.workflows.stamp as stamp_workflow
    from et_mainsim.time_shards import plan_continuous_time_shards
    from photsim7.detector_electronics import _base_seed_scope
    import photsim7.simulation_services as simulation_services
    import photsim7.stamp_pipeline as stamp_pipeline

    detector_by_source = {101: "main_lu", 202: "main_lu", 303: "main_ld"}
    source_ids = tuple(detector_by_source)
    run_root = tmp_path / "galaxy-paired-rng"
    inputs_root = run_root / "inputs"
    inputs_root.mkdir(parents=True)
    target_table = inputs_root / "targets.ecsv"
    target_table.write_text("fixture target table\n", encoding="utf-8")
    time_plan = plan_continuous_time_shards(
        raw_start_index=0,
        raw_stop_index=3,
        coadd_sizes=(3,),
        raw_exposure_seconds=10.0,
        max_raw_frames_per_shard=3,
    )
    time_plan_path = time_plan.write_manifest(inputs_root / "time_shards.json")
    targets = []
    for source_id in source_ids:
        snapshot_path = inputs_root / f"source_{source_id}.npz"
        snapshot_path.write_bytes(f"snapshot-{source_id}".encode("utf-8"))
        targets.append(
            {
                "source_id_int64": source_id,
                "focalplane_mapping": {
                    "detector_id": detector_by_source[source_id]
                },
                "target_table": {
                    "relative_path": "inputs/targets.ecsv",
                    "file_identity": production.file_identity(target_table),
                },
                "factor_snapshot_relative_path": (
                    f"inputs/source_{source_id}.npz"
                ),
                "factor_snapshot": production.file_identity(snapshot_path),
            }
        )
    base_spec = production.build_galaxy_independent_production_spec(
        n_raw_frames=3,
        raw_exposure_seconds=10.0,
        device="cpu",
        run_seed=12345,
    )
    manifest_path = run_root / "production_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_id": production.GALAXY_STAMP_PRODUCTION_SCHEMA_ID,
                "schema_version": production.GALAXY_STAMP_PRODUCTION_SCHEMA_VERSION,
                "run_id": "paired-rng-fixture",
                "runtime_defaults": {
                    "data_root": str(tmp_path / "data"),
                    "focalplane_registry": str(tmp_path / "registry"),
                },
                "delivery": {
                    "stamp_shape": [100, 300],
                    "time_plan_relative_path": "inputs/time_shards.json",
                    "time_plan_identity": production.file_identity(time_plan_path),
                },
                "simulation_spec_base": base_spec.to_json_dict(),
                "targets": targets,
            }
        ),
        encoding="utf-8",
    )
    data_root = tmp_path / "data"
    registry = tmp_path / "registry"
    data_root.mkdir()
    registry.mkdir()

    target_specs = {
        source_id: replace(
            base_spec,
            detector=replace(
                base_spec.detector,
                detector_id=detector_by_source[source_id],
            ),
        )
        for source_id in source_ids
    }
    prepared = SimpleNamespace(
        targets={
            source_id: SimpleNamespace(source_id=source_id)
            for source_id in source_ids
        },
        psf_ids={source_id: 0 for source_id in source_ids},
        source_input_truth={
            source_id: {"source_id": source_id} for source_id in source_ids
        },
    )
    monkeypatch.setattr(
        production,
        "_runtime_paths",
        lambda *_args, **_kwargs: (data_root, registry, {}, {}),
    )
    monkeypatch.setattr(
        production,
        "read_galaxy_factor_snapshot",
        lambda path: SimpleNamespace(
            source_id=int(path.stem.removeprefix("source_")),
            factors=np.ones(3, dtype=np.float64),
            metadata={"fixture": True},
        ),
    )
    monkeypatch.setattr(production, "collect_provenance", lambda *_args: {"test": True})
    monkeypatch.setattr(
        stamp_workflow,
        "build_run_plan",
        lambda **_kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(stamp_workflow, "_science_api", lambda: SimpleNamespace())
    monkeypatch.setattr(
        stamp_workflow,
        "_prepare_table_inputs",
        lambda *_args, **_kwargs: prepared,
    )
    monkeypatch.setattr(
        stamp_workflow,
        "_table_catalog",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        stamp_workflow,
        "_target_spec",
        lambda _plan, *, target, **_kwargs: target_specs[target.source_id],
    )
    original_build_context = simulation_services.build_simulation_context
    contexts = []

    def _capture_context(*args, **kwargs):
        context = original_build_context(*args, **kwargs)
        contexts.append(context)
        return context

    monkeypatch.setattr(
        simulation_services,
        "build_simulation_context",
        _capture_context,
    )
    monkeypatch.setattr(
        simulation_services,
        "build_stamp_services",
        lambda context, **_kwargs: SimpleNamespace(context=context),
    )

    render_calls = []

    def _fake_render(*_args, **kwargs):
        render_calls.append(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(stamp_pipeline, "run_single_cadence_stamp", _fake_render)

    requests = []

    def _fake_delivery(request, *, render_raw, adapt_raw):
        del adapt_raw
        requests.append(request)
        render_raw(request.shard.raw_start_index)
        return SimpleNamespace(
            shard_id=request.shard.shard_id,
            raw_frame_count=request.shard.raw_frame_count,
            raw_path=request.shard_root / "raw.h5",
            coadd_paths={3: request.shard_root / "coadd_30s.h5"},
        )

    monkeypatch.setattr(
        production,
        "run_independent_stamp_time_shard",
        _fake_delivery,
    )

    production.run_galaxy_independent_target(
        manifest_path,
        source_id=101,
        case="static",
        data_root=data_root,
        focalplane_registry=registry,
        device="cpu",
    )
    production.run_galaxy_independent_target(
        manifest_path,
        source_id=101,
        case="injected",
        data_root=data_root,
        focalplane_registry=registry,
        device="cpu",
    )
    production.run_galaxy_independent_target(
        manifest_path,
        source_id=202,
        case="injected",
        data_root=data_root,
        focalplane_registry=registry,
        device="cpu",
    )
    production.run_galaxy_independent_target(
        manifest_path,
        source_id=303,
        case="injected",
        data_root=data_root,
        focalplane_registry=registry,
        device="cpu",
    )

    assert [request.manifest["case"] for request in requests] == [
        "static",
        "injected",
        "injected",
        "injected",
    ]
    assert render_calls[0]["source_variability"] is None
    assert render_calls[1]["source_variability"].relative_flux.shape == (1, 3)
    assert render_calls[0]["rng_trace_scope"] == {
        "workflow": "galaxy-independent-stamp-production",
        "run_id": "paired-rng-fixture",
        "case": "static",
    }
    assert render_calls[1]["rng_trace_scope"] == {
        "workflow": "galaxy-independent-stamp-production",
        "run_id": "paired-rng-fixture",
        "case": "injected",
    }
    assert render_calls[0]["rng_trace_scope"] != render_calls[1]["rng_trace_scope"]

    static_context, injected_context, same_detector_context, other_detector_context = (
        contexts
    )
    shard = time_plan.shards[0]
    for frame_index in range(shard.raw_start_index, shard.raw_stop_index):
        static_scope = static_context.detector_rng_scope(
            local_frame_index=frame_index
        )
        injected_scope = injected_context.detector_rng_scope(
            local_frame_index=frame_index
        )
        assert static_scope == injected_scope
        assert static_context.seed_tree.derive_seed(
            "readout.gaussian",
            scope=static_scope,
        ) == injected_context.seed_tree.derive_seed(
            "readout.gaussian",
            scope=injected_scope,
        )
        # ``case`` reaches the stamp product only as an execution label.  The
        # detector-electronics normalization intentionally discards it (and
        # worker rank) before the physical seed is derived.
        assert _base_seed_scope(
            frame_start=frame_index,
            detector_id="main_lu",
            worker_rank=0,
            rng_trace_scope=render_calls[0]["rng_trace_scope"],
            science_rng_scope=static_scope,
        ) == _base_seed_scope(
            frame_start=frame_index,
            detector_id="main_lu",
            worker_rank=9,
            rng_trace_scope=render_calls[1]["rng_trace_scope"],
            science_rng_scope=injected_scope,
        )

    static_scope = static_context.detector_rng_scope(local_frame_index=0)
    same_detector_scope = same_detector_context.detector_rng_scope(
        local_frame_index=0
    )
    other_detector_scope = other_detector_context.detector_rng_scope(
        local_frame_index=0
    )
    assert static_scope == same_detector_scope
    assert static_scope != other_detector_scope
    assert static_context.seed_tree.derive_seed(
        "readout.gaussian",
        scope=static_scope,
    ) == same_detector_context.seed_tree.derive_seed(
        "readout.gaussian",
        scope=same_detector_scope,
    )
    assert static_context.seed_tree.derive_seed(
        "readout.gaussian",
        scope=static_scope,
    ) != other_detector_context.seed_tree.derive_seed(
        "readout.gaussian",
        scope=other_detector_scope,
    )

    expected_pairing = {
        "schema_id": "et_mainsim.galaxy_physical_rng_pairing.v1",
        "schema_version": 1,
        "seed_tree_run_seed": 12345,
        "canonical_context_scope": {
            "science_realization_id": 0,
            "spacecraft_id": "et",
            "detector_id": "main_lu",
            "scope_id": 0,
        },
        "absolute_raw_frame_index": {
            "formula": "absolute_raw_frame_start_index + local_frame_index",
            "absolute_raw_frame_start_index": 0,
            "selected_shard_absolute_frame_interval": {
                "start_index": 0,
                "stop_index": 3,
            },
        },
        "selected_time_shard": shard.to_manifest_dict(),
        "target_spec_sha256": production._canonical_json_sha256(
            target_specs[101].to_json_dict()
        ),
        "source_id_comparison_label": 101,
        "source_id_in_physical_rng_identity": False,
        "case_not_in_physical_rng_identity": True,
        "rng_trace_scope_role": "execution_label_only",
    }
    assert requests[0].manifest["physical_rng_pairing"] == expected_pairing
    assert requests[0].provenance["physical_rng_pairing"] == expected_pairing
    assert requests[1].manifest["physical_rng_pairing"] == expected_pairing
    assert requests[1].provenance["physical_rng_pairing"] == expected_pairing
    assert requests[2].manifest["physical_rng_pairing"][
        "canonical_context_scope"
    ] == expected_pairing["canonical_context_scope"]
    assert requests[2].manifest["physical_rng_pairing"][
        "source_id_comparison_label"
    ] == 202
    assert requests[3].manifest["physical_rng_pairing"][
        "canonical_context_scope"]["detector_id"] == "main_ld"
