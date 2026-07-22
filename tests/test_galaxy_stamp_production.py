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
