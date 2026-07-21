from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import hashlib
import json
import pickle

import numpy as np
import pytest
from astropy import units as u
from astropy.table import Table


def _table_plan(tmp_path):
    from et_mainsim.config import StampWorkload
    from et_mainsim.presets import load_preset
    from et_mainsim.workflows.stamp import build_run_plan

    loaded = load_preset("et-stamp-smoke")
    table_path = tmp_path / "targets.csv"
    table_path.write_text(
        "gaia_g_mag,psf_id\n12.0,0\n13.5,1\n",
        encoding="utf-8",
    )
    config = replace(
        loaded.run_config,
        paths=replace(
            loaded.run_config.paths,
            output_root=str(tmp_path / "output"),
            data_root=str(tmp_path / "data"),
            catalog_path="",
            focalplane_registry="",
        ),
        workload=StampWorkload(
            input_mode="table",
            input_table=str(table_path),
            target_limit=0,
            include_neighbors=False,
        ),
    )
    return build_run_plan(
        preset_name="et-stamp-smoke",
        run_config=config,
        spec=loaded.simulation_spec,
        repo_root=tmp_path,
    )


def _variable_table_plan(tmp_path, *, target_body=None, curve_body=None):
    from et_mainsim.config import StampWorkload
    from et_mainsim.presets import load_preset
    from et_mainsim.workflows.stamp import build_run_plan

    tmp_path.mkdir(parents=True, exist_ok=True)
    loaded = load_preset("et-stamp-smoke")
    table_path = tmp_path / "targets.csv"
    table_path.write_text(
        target_body
        or (
            "source_id,gaia_g_mag,psf_id,curve_id\n"
            "10,12.0,0,sn\n"
            "11,13.5,1,\n"
        ),
        encoding="utf-8",
    )
    curve_path = tmp_path / "curves.csv"
    curve_path.write_text(
        curve_body
        or (
            "curve_id,frame_index,relative_flux\n"
            "sn,0,0.5\nsn,1,2.0\n"
            "unused,0,1.0\nunused,1,1.0\n"
        ),
        encoding="utf-8",
    )
    config = replace(
        loaded.run_config,
        paths=replace(
            loaded.run_config.paths,
            output_root=str(tmp_path / "output"),
            data_root=str(tmp_path / "data"),
            catalog_path="",
            focalplane_registry="",
        ),
        workload=StampWorkload(
            input_mode="table",
            input_table=str(table_path),
            variability_table=str(curve_path),
            target_limit=0,
            include_neighbors=False,
        ),
    )
    return build_run_plan(
        preset_name="et-stamp-smoke",
        run_config=config,
        spec=loaded.simulation_spec,
        repo_root=tmp_path,
    )


def _fake_table_api():
    from photsim7.catalog_sources import PreparedStarCatalog
    from photsim7.source_variability import SourceVariability

    return SimpleNamespace(
        PreparedStarCatalog=PreparedStarCatalog,
        SourceVariability=SourceVariability,
        load_psf_bundle=lambda *args, **kwargs: {
            "images": {0: object(), 1: object(), 4: object()},
            "angles": np.array([0.0, 2.0, 4.0, 6.0, 8.0]),
        },
    )


def test_table_stamp_plan_does_not_require_full_frame_catalog_assets(tmp_path) -> None:
    from et_mainsim.workflows.stamp import preflight

    plan = _table_plan(tmp_path)
    plan.paths.data_root.mkdir()

    preflight(plan)

    assert plan.input_table_path == tmp_path / "targets.csv"
    assert plan.paths.catalog_path is None
    assert plan.paths.focalplane_registry is None
    assert plan.spec.catalog.source_type == "prepared"
    assert plan.to_dict(dry_run=True)["workload"]["input_mode"] == "table"
    assert not plan.run_dir.exists()


def test_catalog_stamp_preflight_allows_cache_without_query_assets(tmp_path) -> None:
    from et_mainsim.config import RunPaths
    from et_mainsim.presets import load_preset
    from et_mainsim.workflows.stamp import build_run_plan, preflight

    loaded = load_preset("et-stamp-production")
    data_root = tmp_path / "data"
    data_root.mkdir()
    cache = tmp_path / "cache" / "stars.npz"
    cache.parent.mkdir()
    cache.touch()
    registry = tmp_path / "focalplane" / "data"
    registry.mkdir(parents=True)
    config = replace(
        loaded.run_config,
        paths=RunPaths(
            output_root=str(tmp_path / "output"),
            data_root=str(data_root),
            focalplane_registry=str(registry),
            catalog_cache=str(cache),
        ),
    )
    plan = build_run_plan(
        preset_name=loaded.descriptor.name,
        run_config=config,
        spec=loaded.simulation_spec,
        repo_root=tmp_path,
    )

    preflight(plan)

    forced = replace(
        plan,
        run_config=replace(
            plan.run_config,
            execution=replace(plan.run_config.execution, force_catalog_cache=True),
        ),
    )
    with pytest.raises(FileNotFoundError, match="catalog directory"):
        preflight(forced)

    missing_registry = replace(
        plan,
        spec=replace(
            plan.spec,
            catalog=replace(plan.spec.catalog, registry_data_dir=""),
        ),
    )
    with pytest.raises(FileNotFoundError, match="focal-plane data"):
        preflight(missing_registry)


def test_table_stamp_inputs_are_independent_catalogs_without_query(tmp_path) -> None:
    from photsim7.catalog_sources import PreparedStarCatalog
    from et_mainsim.workflows.stamp import prepare_stamp_inputs

    plan = _table_plan(tmp_path)
    plan.paths.data_root.mkdir()
    api = SimpleNamespace(
        PreparedStarCatalog=PreparedStarCatalog,
        load_psf_bundle=lambda *args, **kwargs: {
            "images": {0: object(), 1: object()},
            "angles": np.array([0.0, 2.0]),
        },
        build_catalog_from_spec=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("table mode must not query or build a full-frame catalog")
        ),
    )

    prepared = prepare_stamp_inputs(plan, science_api=api)

    assert prepared.target_ids == (0, 1)
    assert prepared.shared_catalog is None
    assert prepared.provenance["scene_policy"].endswith("no_neighbors")
    first = prepared.catalogs[0]
    assert first.n_sources == 1
    np.testing.assert_allclose(first.star_data["gaia_g_mag"], [12.0])
    np.testing.assert_allclose(first.star_data["detector_xpix"], [31.5])
    assert prepared.psf_ids == {0: 0, 1: 1}


def test_explicit_psf_table_does_not_identity_unused_focalplane_registry(
    tmp_path,
    monkeypatch,
) -> None:
    import et_mainsim.workflows.stamp as stamp_workflow

    plan = _table_plan(tmp_path)
    plan.paths.data_root.mkdir()
    registry = tmp_path / "focalplane" / "data"
    plan = replace(
        plan,
        paths=replace(plan.paths, focalplane_registry=registry),
    )
    monkeypatch.setattr(
        stamp_workflow,
        "_psf_bundle_asset_identity",
        lambda _plan: {"sha256": "test-psf-bundle"},
    )

    prepared = stamp_workflow.prepare_stamp_inputs(
        plan,
        science_api=_fake_table_api(),
    )
    workload_identity = stamp_workflow._workload_identity(plan)

    assert "focalplane_registry" not in prepared.input_identities
    assert "focalplane_registry_identity" not in workload_identity


def test_table_stamp_inputs_bind_variable_and_static_targets_with_hashes(
    tmp_path,
) -> None:
    from et_mainsim.workflows.stamp import prepare_stamp_inputs

    plan = _variable_table_plan(tmp_path)
    plan.paths.data_root.mkdir()

    prepared = prepare_stamp_inputs(plan, science_api=_fake_table_api())

    assert prepared.target_ids == (10, 11)
    variable = prepared.source_variability[10]
    assert variable.source_ids.tolist() == [10]
    np.testing.assert_allclose(variable.relative_flux, [[0.5, 2.0]])
    assert prepared.source_variability[11] is None
    assert prepared.source_input_truth[10]["variability"]["curve_id"] == "sn"
    assert prepared.source_input_truth[11]["variability"]["enabled"] is False
    assert prepared.provenance["variability_selection"] == {
        "referenced_curve_ids": ["sn"],
        "referenced_curve_count": 1,
        "unreferenced_curve_ids": ["unused"],
        "unreferenced_curve_count": 1,
        "static_target_count": 1,
        "variable_target_count": 1,
    }
    assert set(prepared.input_identities) == {
        "target_table",
        "variability_table",
    }
    assert len(prepared.input_identities["target_table"]["sha256"]) == 64


def test_table_stamp_inputs_fail_before_workers_for_missing_curve_or_psf(
    tmp_path,
) -> None:
    from et_mainsim.workflows.stamp import prepare_stamp_inputs

    missing_curve = _variable_table_plan(
        tmp_path / "missing-curve",
        target_body=(
            "source_id,gaia_g_mag,psf_id,curve_id\n10,12.0,0,absent\n"
        ),
    )
    missing_curve.paths.data_root.mkdir(parents=True)
    with pytest.raises(ValueError, match="absent variability curves"):
        prepare_stamp_inputs(missing_curve, science_api=_fake_table_api())

    missing_psf = _variable_table_plan(
        tmp_path / "missing-psf",
        target_body=(
            "source_id,gaia_g_mag,psf_id,curve_id\n10,12.0,9,sn\n"
        ),
    )
    missing_psf.paths.data_root.mkdir(parents=True)
    with pytest.raises(ValueError, match="unavailable PSF ID 9"):
        prepare_stamp_inputs(missing_psf, science_api=_fake_table_api())

    kp_bundle = _variable_table_plan(tmp_path / "kp-bundle")
    kp_bundle.paths.data_root.mkdir(parents=True)
    kp_bundle = replace(
        kp_bundle,
        spec=replace(
            kp_bundle.spec,
            psf=replace(kp_bundle.spec.psf, bundle_name="kp_2"),
        ),
    )
    with pytest.raises(ValueError, match="deterministic ET PSF bundle"):
        prepare_stamp_inputs(kp_bundle, science_api=_fake_table_api())


def test_coordinate_target_selects_nearest_radial_psf_and_records_mapping(
    tmp_path, monkeypatch
) -> None:
    import et_mainsim.stamp_inputs as stamp_inputs
    from et_mainsim.config import RunPaths
    from et_mainsim.workflows.stamp import prepare_stamp_inputs

    plan = _variable_table_plan(
        tmp_path,
        target_body=(
            "source_id,gaia_g_mag,ra_deg,dec_deg,curve_id\n"
            "10,12.0,123.0,-20.0,sn\n"
        ),
    )
    registry = tmp_path / "focalplane" / "data"
    registry.mkdir(parents=True)
    (registry / "fov.csv").write_text("x\n1\n", encoding="utf-8")
    plan = replace(
        plan,
        paths=replace(plan.paths, focalplane_registry=registry),
        run_config=replace(
            plan.run_config,
            paths=RunPaths(
                **{
                    **plan.run_config.paths.__dict__,
                    "focalplane_registry": str(registry),
                }
            ),
        ),
    )
    plan.paths.data_root.mkdir(parents=True)
    monkeypatch.setattr(
        stamp_inputs,
        "_sky_to_focal",
        lambda *_args, **_kwargs: SimpleNamespace(
            status="ok",
            detector_id="main_rd",
            xpix=30.0,
            ypix=31.0,
            field_x_deg=5.0,
            field_y_deg=6.0,
            residual_arcsec=0.02,
        ),
    )

    prepared = prepare_stamp_inputs(plan, science_api=_fake_table_api())

    target = prepared.targets[10]
    assert target.field_angle_deg == pytest.approx(np.hypot(5.0, 6.0))
    assert target.psf_id == 4
    assert target.psf_node_angle_deg == 8.0
    assert target.psf_angle_delta_deg == pytest.approx(
        8.0 - np.hypot(5.0, 6.0)
    )
    assert prepared.catalogs[10].star_data["field_angle_deg"][0] == pytest.approx(
        np.hypot(5.0, 6.0)
    )
    assert "focalplane_registry" in prepared.input_identities


def test_table_targets_attach_row_local_geometry_declarations(
    tmp_path, monkeypatch
) -> None:
    import et_mainsim.stamp_inputs as stamp_inputs
    import photsim7.geometry_truth as geometry_truth
    from et_mainsim.config import RunPaths
    from et_mainsim.workflows.stamp import _target_spec, prepare_stamp_inputs

    if not hasattr(geometry_truth, "reference_field_nonphysical_declaration"):
        def expected_reference_declaration(**parameters):
            return {
                "schema_id": "photsim7.source_geometry_declaration.v1",
                "schema_version": 1,
                "mode": "reference_field_nonphysical",
                "physical_geometry_claim": False,
                **parameters,
            }

        monkeypatch.setattr(
            geometry_truth,
            "reference_field_nonphysical_declaration",
            expected_reference_declaration,
            raising=False,
        )

    plan = _variable_table_plan(
        tmp_path,
        target_body=(
            "source_id,gaia_g_mag,ra_deg,dec_deg,psf_id,curve_id\n"
            "10,12.0,123.0,-20.0,,sn\n"
            "11,13.0,,,6,\n"
        ),
    )
    registry = tmp_path / "focalplane" / "data"
    registry.mkdir(parents=True)
    (registry / "fov.csv").write_text("x\n1\n", encoding="utf-8")
    plan = replace(
        plan,
        paths=replace(plan.paths, focalplane_registry=registry),
        run_config=replace(
            plan.run_config,
            paths=RunPaths(
                **{
                    **plan.run_config.paths.__dict__,
                    "focalplane_registry": str(registry),
                }
            ),
        ),
    )
    plan.paths.data_root.mkdir(parents=True)
    monkeypatch.setattr(
        stamp_inputs,
        "_sky_to_focal",
        lambda *_args, **_kwargs: SimpleNamespace(
            status="ok",
            detector_id="main_rd",
            xpix=30.0,
            ypix=31.0,
            field_x_deg=5.0,
            field_y_deg=6.0,
            residual_arcsec=0.02,
        ),
    )
    api = _fake_table_api()
    api.load_psf_bundle = lambda *args, **kwargs: {
        "images": {0: object(), 4: object(), 6: object()},
        "angles": np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0]),
    }

    prepared = prepare_stamp_inputs(plan, science_api=api)

    physical_catalog = prepared.catalogs[10]
    physical = physical_catalog.metadata["geometry"]
    assert physical["mode"] == "physical_et_focalplane"
    assert physical["physical_geometry_claim"] is True
    assert physical["coordinate_frame"] == "icrs"
    assert physical["coordinate_epoch_jyear"] == pytest.approx(2000.0)
    assert physical["registry_data_dir"] == str(registry)
    assert physical["focalplane_registry_identity"] == (
        prepared.input_identities["focalplane_registry"]
    )
    assert len(physical["focalplane_registry_identity"]["sha256"]) == 64
    np.testing.assert_allclose(
        physical_catalog.star_data["icrs_ra_deg"], [123.0]
    )
    np.testing.assert_allclose(
        physical_catalog.star_data["icrs_dec_deg"], [-20.0]
    )
    np.testing.assert_allclose(
        physical_catalog.star_data["target_epoch"], [2000.0]
    )

    explicit_catalog = prepared.catalogs[11]
    reference = explicit_catalog.metadata["geometry"]
    assert reference["mode"] == "reference_field_nonphysical"
    assert reference["physical_geometry_claim"] is False
    assert reference["reference_field_angle_deg"] == pytest.approx(12.0)
    assert reference["reference_field_polar_angle_rad"] == pytest.approx(
        np.pi / 4.0
    )
    assert reference["reference_pixel_scale_arcsec_per_pix"] == pytest.approx(
        4.83
    )
    assert reference["reference_x_axis_sign"] == pytest.approx(1.0)
    assert reference["reference_y_axis_sign"] == pytest.approx(1.0)
    assert "focalplane_registry_identity" not in reference
    assert physical is not reference
    assert prepared.source_input_truth[10]["schema_id"] == (
        "et_mainsim.stamp_source_input_truth.v2"
    )
    assert prepared.source_input_truth[10]["schema_version"] == 2
    assert (
        prepared.source_input_truth[10]["focalplane_registry_identity"]
        == prepared.input_identities["focalplane_registry"]
    )
    assert prepared.source_input_truth[10]["location"][
        "coordinate_epoch_jyear"
    ] == pytest.approx(2000.0)
    assert prepared.source_input_truth[11]["focalplane_registry_identity"] is None
    assert (
        prepared.source_input_truth[11]["location"]["coordinate_epoch_jyear"]
        is None
    )

    coordinate_spec = _target_spec(
        plan,
        target=prepared.targets[10],
        psf_id=prepared.psf_ids[10],
    )
    assert coordinate_spec.psf.field_id == "nearest"
    assert coordinate_spec.psf.field_id_policy == "nearest"

    target_spec = _target_spec(
        plan,
        target=prepared.targets[11],
        psf_id=prepared.psf_ids[11],
    )
    assert target_spec.catalog.query_options == {
        "reference_field_angle_deg": pytest.approx(12.0),
        "reference_field_polar_angle_rad": pytest.approx(np.pi / 4.0),
        "reference_pixel_scale_arcsec_per_pix": pytest.approx(4.83),
        "reference_x_axis_sign": pytest.approx(1.0),
        "reference_y_axis_sign": pytest.approx(1.0),
    }


def test_table_mode_rejects_non_j2000_epoch_before_preparation(tmp_path) -> None:
    from et_mainsim.presets import load_preset
    from et_mainsim.workflows.stamp import build_run_plan

    loaded = load_preset("et-stamp-smoke")
    table_path = tmp_path / "targets.csv"
    table_path.write_text("gaia_g_mag,psf_id\n12.0,6\n", encoding="utf-8")
    config = replace(
        loaded.run_config,
        paths=replace(
            loaded.run_config.paths,
            output_root=str(tmp_path / "output"),
            data_root=str(tmp_path / "data"),
            catalog_path="",
            focalplane_registry="",
        ),
        workload=replace(
            loaded.run_config.workload,
            input_mode="table",
            input_table=str(table_path),
            include_neighbors=False,
        ),
    )

    with pytest.raises(ValueError, match="J2000.*2000"):
        build_run_plan(
            preset_name="et-stamp-smoke",
            run_config=config,
            spec=loaded.simulation_spec,
            repo_root=tmp_path,
            target_epoch_jyear=2016.0,
        )


def test_table_target_spec_uses_independent_accepted_psf_bundle_sha256(
    tmp_path, monkeypatch
) -> None:
    import et_mainsim.workflows.stamp as stamp_workflow

    expected_sha256 = "a" * 64
    plan = _variable_table_plan(
        tmp_path,
        target_body=(
            "source_id,gaia_g_mag,psf_id,curve_id\n"
            "10,12.0,0,sn\n"
        ),
    )
    plan = replace(
        plan,
        spec=replace(
            plan.spec,
            psf=replace(plan.spec.psf, bundle_sha256=expected_sha256),
        ),
    )
    plan.paths.data_root.mkdir(parents=True)
    monkeypatch.setattr(
        stamp_workflow,
        "_psf_bundle_asset_identity",
        lambda _plan: {
            "path": "/frozen/sim_psf_images.pkl",
            "size_bytes": 123,
            "sha256": expected_sha256,
        },
    )
    prepared = stamp_workflow.prepare_stamp_inputs(
        plan,
        science_api=_fake_table_api(),
    )

    target_spec = stamp_workflow._target_spec(
        plan,
        target=prepared.targets[10],
        psf_id=prepared.psf_ids[10],
        source_input_truth=prepared.source_input_truth[10],
    )

    assert target_spec.psf.bundle_sha256 == expected_sha256


def test_table_target_spec_rejects_observed_psf_hash_mismatch(
    tmp_path, monkeypatch
) -> None:
    import et_mainsim.workflows.stamp as stamp_workflow

    plan = _variable_table_plan(
        tmp_path,
        target_body=(
            "source_id,gaia_g_mag,psf_id,curve_id\n"
            "10,12.0,0,sn\n"
        ),
    )
    plan = replace(
        plan,
        spec=replace(
            plan.spec,
            psf=replace(plan.spec.psf, bundle_sha256="a" * 64),
        ),
    )
    plan.paths.data_root.mkdir(parents=True)
    monkeypatch.setattr(
        stamp_workflow,
        "_psf_bundle_asset_identity",
        lambda _plan: {
            "path": "/changed/sim_psf_images.pkl",
            "size_bytes": 123,
            "sha256": "b" * 64,
        },
    )
    prepared = stamp_workflow.prepare_stamp_inputs(
        plan,
        science_api=_fake_table_api(),
    )

    with pytest.raises(ValueError, match="accepted SimulationSpec identity"):
        stamp_workflow._target_spec(
            plan,
            target=prepared.targets[10],
            psf_id=prepared.psf_ids[10],
            source_input_truth=prepared.source_input_truth[10],
        )


def test_catalog_stamp_inputs_select_requested_targets_from_one_shared_scene(tmp_path) -> None:
    from photsim7.catalog_sources import PreparedStarCatalog
    from et_mainsim.presets import load_preset
    from et_mainsim.workflows.stamp import build_run_plan, prepare_stamp_inputs

    loaded = load_preset("et-stamp-smoke")
    config = replace(
        loaded.run_config,
        paths=replace(
            loaded.run_config.paths,
            output_root=str(tmp_path / "output"),
            data_root=str(tmp_path / "data"),
        ),
    )
    plan = build_run_plan(
        preset_name="et-stamp-smoke",
        run_config=config,
        spec=loaded.simulation_spec,
        repo_root=tmp_path,
    )
    catalog = PreparedStarCatalog(
        star_data={
            "x0": np.array([0.0, 1.0]),
            "y0": np.array([0.0, 0.5]),
            "source_id": np.array([1, 2]),
            "gaia_g_mag": np.array([12.0, 13.0]),
        }
    )
    api = SimpleNamespace(
        PreparedStarCatalog=PreparedStarCatalog,
        DataRegistry=lambda **kwargs: object(),
        build_catalog_from_spec=lambda *args, **kwargs: catalog,
    )

    prepared = prepare_stamp_inputs(plan, science_api=api)

    assert prepared.target_ids == (1,)
    assert prepared.shared_catalog is catalog
    assert prepared.catalogs == {1: catalog}
    assert prepared.psf_ids == {}


def _write_test_psf_bundle(data_root):
    bundle_name = "psf/et/et_mainsim_stamp_test"
    bundle_dir = data_root / bundle_name
    bundle_dir.mkdir(parents=True)
    n_subpixels = 3
    rows = cols = 7
    y, x = np.mgrid[: rows * n_subpixels, : cols * n_subpixels].astype(np.float32)
    yy = (y - y.mean()) / n_subpixels
    xx = (x - x.mean()) / n_subpixels
    image = np.exp(-0.5 * (xx**2 + yy**2) / 0.9**2).astype(np.float32)
    image /= image.sum(dtype=np.float64)
    with (bundle_dir / "sim_psf_images.pkl").open("wb") as handle:
        pickle.dump(
            {
                "images": {0: {n_subpixels: np.stack([xx, yy, image])}},
                "angles": np.array([0.0]),
            },
            handle,
        )
    bundle_sha256 = hashlib.sha256(
        (bundle_dir / "sim_psf_images.pkl").read_bytes()
    ).hexdigest()
    return bundle_name, bundle_sha256


def _selection_sidecar_plan(tmp_path):
    plan = _variable_table_plan(
        tmp_path,
        target_body=(
            "source_id,gaia_g_mag,psf_id,curve_id\n10,12.0,0,sn\n"
        ),
    )
    bundle_name, bundle_sha256 = _write_test_psf_bundle(
        plan.paths.data_root
    )
    return replace(
        plan,
        spec=replace(
            plan.spec,
            detector=replace(plan.spec.detector, n_subpixels=3),
            psf=replace(
                plan.spec.psf,
                bundle_name=bundle_name,
                bundle_sha256=bundle_sha256,
            ),
        ),
    )


def _complete_selection_api(api):
    from photsim7.dynamic_effects import EffectTimeseries, build_frame_timing
    from photsim7.jitter_bank import (
        CANONICAL_JITTER_BANK_EVIDENCE_ID,
        CANONICAL_JITTER_BANK_LOGICAL_ID,
        NATIVE_JITTER_BANK_LOADER_ID,
    )
    from photsim7.jitter_bank_authority import (
        CANONICAL_JITTER_BANK_MANIFEST_SHA256,
        CANONICAL_JITTER_BANK_SHA256,
    )
    from photsim7.jitter_selection_truth import JitterModelSelector

    original_build_services = api.build_stamp_services

    def build_services(spec, *args, **kwargs):
        services = original_build_services(spec, *args, **kwargs)
        selector = JitterModelSelector(
            seed_tree=services.seed_tree,
            bank_identity={
                "logical_bank_id": CANONICAL_JITTER_BANK_LOGICAL_ID,
                "bank_evidence_id": CANONICAL_JITTER_BANK_EVIDENCE_ID,
                "array_sha256": CANONICAL_JITTER_BANK_SHA256,
                "expected_array_sha256": CANONICAL_JITTER_BANK_SHA256,
                "manifest_sha256": CANONICAL_JITTER_BANK_MANIFEST_SHA256,
                "expected_manifest_sha256": (
                    CANONICAL_JITTER_BANK_MANIFEST_SHA256
                ),
                "verification_status": (
                    "array_and_manifest_sha256_verified_before_load"
                ),
                "loader": NATIVE_JITTER_BANK_LOADER_ID,
            },
            n_models=100,
            science_realization_id=(
                spec.science_profile.science_realization_id
            ),
            spacecraft_id=services.context.spacecraft_id,
            absolute_raw_frame_start_index=(
                services.context.absolute_raw_frame_start_index
            ),
        )
        effects = EffectTimeseries(
            timing=build_frame_timing(
                n_frames=spec.observation.resolved_n_frames,
                integration_s=services.frame_exposure.to_value(u.s),
                sampling_interval_s=(
                    spec.observation.sampling_interval.to_value(u.s)
                ),
            ),
            source_geometry=services.source_geometry,
            jitter_integrated_psf_offsets=np.zeros((100, 2, 1)),
            jitter_model_selector=selector,
        )
        return replace(
            services,
            full_frame_services=replace(
                services.full_frame_services,
                effect_timeseries=effects,
            ),
        )

    api.build_stamp_services = build_services
    return api


def test_stamp_run_writes_readable_raw_coadd_truth_and_resumes(tmp_path) -> None:
    from et_mainsim.presets import load_preset
    from et_mainsim.workflows.stamp import build_run_plan, run_stamp
    from photsim7.artifacts import StampShardReader

    loaded = load_preset("et-stamp-smoke")
    data_root = tmp_path / "data"
    bundle_name, bundle_sha256 = _write_test_psf_bundle(data_root)
    spec = replace(
        loaded.simulation_spec,
        detector=replace(loaded.simulation_spec.detector, n_subpixels=3),
        psf=replace(
            loaded.simulation_spec.psf,
            bundle_name=bundle_name,
            bundle_sha256=bundle_sha256,
        ),
        observation=replace(
            loaded.simulation_spec.observation,
            exposure_duration=10 * u.s,
            readout_duration=0 * u.s,
            observing_duration=20 * u.s,
            n_frames=2,
            n_raw_frames_per_coadd=2,
        ),
    )
    config = replace(
        loaded.run_config,
        paths=replace(
            loaded.run_config.paths,
            output_root=str(tmp_path / "output"),
            data_root=str(data_root),
        ),
    )
    plan = build_run_plan(
        preset_name="et-stamp-smoke",
        run_config=config,
        spec=spec,
        repo_root=tmp_path,
    )

    first = run_stamp(plan)

    target_dir = plan.run_dir / "stamps" / "target_1"
    with StampShardReader(target_dir / "raw.h5") as raw_reader:
        assert raw_reader.star_ids == (1,)
        assert raw_reader.frame_ids == (0, 1)
        raw0 = raw_reader.read_stamp(1, 0)
        raw1 = raw_reader.read_stamp(1, 1)
    with StampShardReader(target_dir / "coadd.h5") as coadd_reader:
        assert coadd_reader.frame_ids == (0,)
        coadd = coadd_reader.read_stamp(1, 0)
    np.testing.assert_array_equal(
        coadd,
        np.sum([raw0, raw1], axis=0, dtype=np.uint64),
    )
    raw_schema = json.loads(
        (target_dir / "schemas" / "raw" / "frame_000000.json").read_text(
            encoding="utf-8"
        )
    )
    coadd_schema = json.loads(
        (target_dir / "schemas" / "coadd" / "coadd_000000.json").read_text(
            encoding="utf-8"
        )
    )
    assert raw_schema["target"]["source_id"] == 1
    assert raw_schema["truth"]["fields"]
    assert raw_schema["rng_trace"]["entries"]
    assert coadd_schema["coadd"]["raw_frame_indices"] == [0, 1]
    assert first["status"] == "completed"

    second = run_stamp(plan)

    assert second["status"] == "completed"
    assert second["completion"]["rendered_targets"] == 0
    assert second["completion"]["skipped_targets"] == 1
    assert len(second["attempts"]) == 2

    for run_id, save_raw, save_coadd, absent_name in (
        ("raw-only", True, False, "coadd"),
        ("coadd-only", False, True, "raw"),
    ):
        output_config = replace(
            config,
            run_id=run_id,
            workload=replace(
                config.workload,
                save_raw=save_raw,
                save_coadd=save_coadd,
            ),
        )
        output_plan = build_run_plan(
            preset_name="et-stamp-smoke",
            run_config=output_config,
            spec=spec,
            repo_root=tmp_path,
        )

        run_stamp(output_plan)

        output_target = output_plan.run_dir / "stamps" / "target_1"
        assert not (output_target / f"{absent_name}.h5").exists()
        assert not (output_target / "schemas" / absent_name).exists()


def test_stamp_worker_request_round_trip_preserves_table_contract(tmp_path) -> None:
    from et_mainsim.workflows.stamp import StampWorkerRequest

    plan = _table_plan(tmp_path)
    with pytest.raises(ValueError, match="input identities"):
        StampWorkerRequest.from_plan(plan, target_ids=(0,))
    request = StampWorkerRequest.from_plan(
        plan,
        target_ids=(0, 1),
        rank=1,
        world_size=2,
        input_identities={
            "target_table": {"path": "targets.csv", "sha256": "abc"}
        },
    )

    restored = StampWorkerRequest.from_json_dict(request.to_json_dict())

    assert restored.plan.workload.input_mode == "table"
    assert restored.plan.input_table_path == tmp_path / "targets.csv"
    assert restored.target_ids == (0, 1)
    assert restored.rank == 1
    assert restored.world_size == 2
    assert restored.input_identities == {
        "target_table": {"path": "targets.csv", "sha256": "abc"}
    }


def test_stamp_worker_reloads_and_rejects_changed_variability_identity(
    tmp_path,
) -> None:
    from et_mainsim.workflows.stamp import (
        StampWorkerRequest,
        _worker_inputs,
        prepare_stamp_inputs,
    )

    plan = _variable_table_plan(tmp_path)
    plan.paths.data_root.mkdir()
    api = _fake_table_api()
    prepared = prepare_stamp_inputs(plan, science_api=api)
    request = StampWorkerRequest.from_plan(
        plan,
        target_ids=prepared.target_ids,
        input_identities=prepared.input_identities,
    )
    plan.variability_table_path.write_text(
        "curve_id,frame_index,relative_flux\n"
        "sn,0,0.25\nsn,1,2.0\n"
        "unused,0,1.0\nunused,1,1.0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="input identity mismatch"):
        _worker_inputs(request, api)


def test_variable_table_run_persists_numeric_truth_provenance_and_digest(
    tmp_path,
) -> None:
    from astropy.table import Table
    from et_mainsim.stamp_inputs import file_identity
    from et_mainsim.workflows.stamp import (
        _science_api,
        run_stamp,
        target_is_complete,
    )
    from photsim7.artifacts import StampShardReader

    plan = _variable_table_plan(
        tmp_path,
        target_body=(
            "source_id,gaia_g_mag,psf_id,curve_id\n10,12.0,0,sn\n"
        ),
    )
    data_root = plan.paths.data_root
    bundle_name, bundle_sha256 = _write_test_psf_bundle(data_root)
    plan = replace(
        plan,
        spec=replace(
            plan.spec,
            detector=replace(plan.spec.detector, n_subpixels=3),
            psf=replace(
                plan.spec.psf,
                bundle_name=bundle_name,
                bundle_sha256=bundle_sha256,
            ),
        ),
    )

    manifest = run_stamp(plan)

    target_dir = plan.run_dir / "stamps" / "target_10"
    truth_path = target_dir / "source_variability_truth.ecsv"
    truth = Table.read(truth_path, format="ascii.ecsv")
    np.testing.assert_allclose(truth["relative_flux"], [0.5, 2.0])
    np.testing.assert_allclose(
        truth["effective_photon_count_electron"],
        np.asarray(truth["baseline_photon_count_electron"])
        * np.asarray(truth["relative_flux"]),
    )
    with StampShardReader(target_dir / "raw.h5") as reader:
        provenance = reader.spec.provenance
    assert provenance["source_input_truth"]["variability"]["curve_id"] == "sn"
    raw_schema = json.loads(
        (target_dir / "schemas" / "raw" / "frame_000001.json").read_text(
            encoding="utf-8"
        )
    )
    assert raw_schema["source_input_truth"]["psf"]["chosen_psf_id"] == 0
    artifacts = manifest["completion"]["targets"][0]["artifacts"]
    assert len(artifacts["source_variability_truth_identity"]["sha256"]) == 64
    assert len(manifest["workload"]["psf_bundle_identity"]["sha256"]) == 64
    assert target_is_complete(plan, 10, api=_science_api())

    truth["effective_photon_count_electron"][0] += 1.0
    truth.write(truth_path, format="ascii.ecsv", overwrite=True)
    target_artifact_path = target_dir / "target_artifacts.json"
    target_artifact_payload = json.loads(
        target_artifact_path.read_text(encoding="utf-8")
    )
    target_artifact_payload["source_variability_truth_identity"] = file_identity(
        truth_path
    )
    target_artifact_path.write_text(
        json.dumps(target_artifact_payload), encoding="utf-8"
    )
    assert not target_is_complete(plan, 10, api=_science_api())


def test_stamp_target_persists_closed_selection_sidecar_manifest(
    tmp_path,
) -> None:
    from et_mainsim.workflows.stamp import _science_api, run_stamp
    from photsim7.selection_artifacts import read_cadence_selection_truth

    plan = _selection_sidecar_plan(tmp_path)
    api = _complete_selection_api(_science_api())
    original_run_coadd = api.run_stamp_coadd
    target_dir = plan.run_dir / "stamps" / "target_10"
    observed_roots = []

    def run_coadd(*args, **kwargs):
        observed_roots.append(kwargs.get("run_dir"))
        return original_run_coadd(*args, **kwargs)

    api.run_stamp_coadd = run_coadd
    manifest = run_stamp(plan, science_api=api)

    assert observed_roots == [target_dir]
    artifacts = manifest["completion"]["targets"][0]["artifacts"]
    assert artifacts["schema_version"] == 2
    selection = artifacts["selection_truth"]
    assert selection["schema_id"] == (
        "et_mainsim.stamp_selection_truth_artifacts.v1"
    )
    assert selection["schema_version"] == 1
    assert selection["verification_status"] == "persisted_and_verified"
    assert selection["missing_components"] == []
    assert selection["artifact_root"] == "."
    geometry = selection["source_geometry_truth"]
    psf = selection["psf_selection_truth"]
    cadence = selection["cadence_selection_truth"]
    assert len(geometry["content_sha256"]) == 64
    assert len(psf["content_sha256"]) == 64
    assert (target_dir / geometry["relative_path"]).is_file()
    assert (target_dir / psf["relative_path"]).is_file()
    assert cadence == {
        "schema_id": "photsim7.cadence_selection_truth.v1",
        "schema_version": 1,
        "relative_directory": "selection_truth/cadence",
        "filename_template": "frame_{absolute_raw_frame_index:09d}.json",
        "count": 2,
        "absolute_raw_frame_start_index": 0,
        "spacecraft_id": "et",
        "science_realization_id": 0,
        "index_digest_schema_id": "et_mainsim.selection_truth_index.v1",
        "index_content_sha256": cadence["index_content_sha256"],
    }
    assert len(cadence["index_content_sha256"]) == 64
    for frame_index in (0, 1):
        path = (
            target_dir
            / cadence["relative_directory"]
            / cadence["filename_template"].format(
                absolute_raw_frame_index=frame_index
            )
        )
        restored = read_cadence_selection_truth(
            path,
            artifact_root=target_dir,
        )
        assert restored.local_frame_index == frame_index
        assert restored.source_geometry_truth.content_sha256 == (
            geometry["content_sha256"]
        )
        assert restored.psf_selection_truth.content_sha256 == (
            psf["content_sha256"]
        )


def test_unavailable_selection_truth_still_requires_raw_frame_order(
    tmp_path,
) -> None:
    from et_mainsim.workflows.stamp import _SelectionTruthAccumulator

    accumulator = _SelectionTruthAccumulator(
        target_dir=tmp_path,
        frame_indices=(0,),
        spacecraft_id="et",
        science_realization_id=0,
        absolute_raw_frame_start_index=0,
    )
    raw_result = SimpleNamespace(
        selection_truth=None,
        selection_artifacts=None,
        stamp_products=SimpleNamespace(
            frame_index=1,
            selection_truth={
                "verification_status": "unavailable",
                "science_conformance_claim": False,
            },
        ),
    )

    with pytest.raises(RuntimeError, match="ordered by the raw frame plan"):
        accumulator.add(raw_result)


def test_stamp_completion_and_resume_fail_closed_on_selection_sidecars(
    tmp_path,
) -> None:
    from et_mainsim.workflows.stamp import (
        _science_api,
        run_stamp,
        target_is_complete,
    )
    from photsim7.selection_artifacts import SelectionArtifactConflictError

    plan = _selection_sidecar_plan(tmp_path)
    api = _complete_selection_api(_science_api())
    run_stamp(plan, science_api=api)
    target_dir = plan.run_dir / "stamps" / "target_10"
    manifest_path = target_dir / "target_artifacts.json"
    original_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cadence = original_manifest["selection_truth"][
        "cadence_selection_truth"
    ]
    cadence_path = (
        target_dir
        / cadence["relative_directory"]
        / cadence["filename_template"].format(
            absolute_raw_frame_index=1
        )
    )

    missing_selection = dict(original_manifest)
    missing_selection.pop("selection_truth")
    manifest_path.write_text(json.dumps(missing_selection), encoding="utf-8")
    assert not target_is_complete(plan, 10, api=api)

    bad_index = json.loads(json.dumps(original_manifest))
    bad_index["selection_truth"]["cadence_selection_truth"][
        "index_content_sha256"
    ] = "0" * 64
    manifest_path.write_text(json.dumps(bad_index), encoding="utf-8")
    assert not target_is_complete(plan, 10, api=api)

    manifest_path.write_text(json.dumps(original_manifest), encoding="utf-8")
    cadence_path.unlink()
    assert not target_is_complete(plan, 10, api=api)
    repaired = run_stamp(plan, science_api=api)
    assert repaired["completion"]["rendered_targets"] == 1
    assert cadence_path.is_file()
    assert target_is_complete(plan, 10, api=api)

    cadence_path.write_text("corrupt competing bytes\n", encoding="utf-8")
    assert not target_is_complete(plan, 10, api=api)
    with pytest.raises(SelectionArtifactConflictError, match="conflict"):
        run_stamp(plan, science_api=api)


def test_selection_resume_binds_spacecraft_and_science_realization(tmp_path):
    from et_mainsim.workflows.stamp import (
        _selection_index_record,
        _validate_selection_sidecars,
    )

    plan = _selection_sidecar_plan(tmp_path)
    geometry = {
        "schema_id": "photsim7.source_geometry_truth.v1",
        "schema_version": 1,
        "relative_path": f"selection_truth/geometry/geometry_{'a' * 64}.json",
        "content_sha256": "a" * 64,
    }
    psf = {
        "schema_id": "photsim7.psf_selection_truth.v2",
        "schema_version": 2,
        "relative_path": f"selection_truth/psf/psf_{'b' * 64}.json",
        "content_sha256": "b" * 64,
    }

    def self_consistent_payload(
        spacecraft_id,
        science_realization_id,
        *,
        manifest_claim=True,
        truth_claim=True,
    ):
        truths = {}
        digest = hashlib.sha256()
        for frame_index in (0, 1):
            content_sha256 = hashlib.sha256(
                (
                    f"{spacecraft_id}:{science_realization_id}:"
                    f"{frame_index}"
                ).encode("utf-8")
            ).hexdigest()
            truths[frame_index] = SimpleNamespace(
                local_frame_index=frame_index,
                absolute_raw_frame_index=frame_index,
                detector_id=plan.spec.detector.detector_id,
                spacecraft_id=spacecraft_id,
                science_realization_id=science_realization_id,
                science_conformance_claim=truth_claim,
                geometry_reference=geometry,
                psf_reference=psf,
                content_sha256=content_sha256,
            )
            digest.update(
                _selection_index_record(
                    local_frame_index=frame_index,
                    absolute_raw_frame_index=frame_index,
                    content_sha256=content_sha256,
                )
            )
        selection = {
            "schema_id": "et_mainsim.stamp_selection_truth_artifacts.v1",
            "schema_version": 1,
            "verification_status": "persisted_and_verified",
            "science_conformance_claim": manifest_claim,
            "missing_components": [],
            "artifact_root": ".",
            "source_geometry_truth": geometry,
            "psf_selection_truth": psf,
            "cadence_selection_truth": {
                "schema_id": "photsim7.cadence_selection_truth.v1",
                "schema_version": 1,
                "relative_directory": "selection_truth/cadence",
                "filename_template": (
                    "frame_{absolute_raw_frame_index:09d}.json"
                ),
                "count": 2,
                "absolute_raw_frame_start_index": 0,
                "spacecraft_id": spacecraft_id,
                "science_realization_id": science_realization_id,
                "index_digest_schema_id": (
                    "et_mainsim.selection_truth_index.v1"
                ),
                "index_content_sha256": digest.hexdigest(),
            },
        }
        api = SimpleNamespace(
            cadence_selection_truth_relative_path=lambda frame: (
                f"selection_truth/cadence/frame_{frame:09d}.json"
            ),
            read_cadence_selection_truth=lambda path, **_kwargs: truths[
                int(str(path).rsplit("_", 1)[1].split(".", 1)[0])
            ],
        )
        return {"selection_truth": selection}, api

    expected_realization = plan.spec.science_profile.science_realization_id
    payload, api = self_consistent_payload("et", expected_realization)
    assert _validate_selection_sidecars(
        plan,
        10,
        payload,
        api=api,
    )

    mismatched_claim, mismatched_claim_api = self_consistent_payload(
        "et",
        expected_realization,
        manifest_claim=True,
        truth_claim=False,
    )
    assert not _validate_selection_sidecars(
        plan,
        10,
        mismatched_claim,
        api=mismatched_claim_api,
    )

    for spacecraft_id, realization_id in (
        ("other-spacecraft", expected_realization),
        ("et", expected_realization + 1),
    ):
        transplanted, transplanted_api = self_consistent_payload(
            spacecraft_id,
            realization_id,
        )
        assert not _validate_selection_sidecars(
            plan,
            10,
            transplanted,
            api=transplanted_api,
        )


def test_stamp_run_identity_requires_current_product_contract(tmp_path):
    from et_mainsim.manifest import ManifestIdentityError
    from et_mainsim.workflows.stamp import _science_api, run_stamp

    plan = _selection_sidecar_plan(tmp_path)
    api = _complete_selection_api(_science_api())
    run_stamp(plan, science_api=api)
    manifest_path = plan.run_dir / "run_manifest.json"
    original = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert original["workload"]["product_contract"] == {
        "target_artifact_schema_id": "et_mainsim.stamp_target_artifacts",
        "target_artifact_schema_version": 2,
        "selection_artifact_schema_id": (
            "et_mainsim.stamp_selection_truth_artifacts.v1"
        ),
        "selection_artifact_schema_version": 1,
        "selection_index_schema_id": "et_mainsim.selection_truth_index.v1",
        "source_geometry_truth_schema_id": (
            "photsim7.source_geometry_truth.v1"
        ),
        "psf_selection_truth_schema_id": (
            "photsim7.psf_selection_truth.v2"
        ),
        "cadence_selection_truth_schema_id": (
            "photsim7.cadence_selection_truth.v1"
        ),
    }

    for mutation in ("missing", "old-version"):
        stale = json.loads(json.dumps(original))
        if mutation == "missing":
            stale["workload"].pop("product_contract")
        else:
            stale["workload"]["product_contract"][
                "target_artifact_schema_version"
            ] = 1
        manifest_path.write_text(json.dumps(stale), encoding="utf-8")
        with pytest.raises(ManifestIdentityError, match="workload identity"):
            run_stamp(plan, science_api=api)


def test_variable_table_local_subprocess_worker_preserves_input_contract(
    tmp_path,
) -> None:
    from et_mainsim.workflows.stamp import run_stamp

    plan = _variable_table_plan(
        tmp_path,
        target_body=(
            "source_id,gaia_g_mag,psf_id,curve_id\n10,12.0,0,sn\n"
        ),
    )
    bundle_name, bundle_sha256 = _write_test_psf_bundle(
        plan.paths.data_root
    )
    plan = replace(
        plan,
        run_config=replace(
            plan.run_config,
            execution=replace(
                plan.run_config.execution,
                backend="local-subprocess",
                device="cpu",
            ),
        ),
        spec=replace(
            plan.spec,
            detector=replace(plan.spec.detector, n_subpixels=3),
            psf=replace(
                plan.spec.psf,
                bundle_name=bundle_name,
                bundle_sha256=bundle_sha256,
            ),
        ),
    )

    manifest = run_stamp(plan)

    assert manifest["status"] == "completed"
    request = json.loads(
        (plan.run_dir / "worker_requests" / "stamp_worker_00.json").read_text(
            encoding="utf-8"
        )
    )
    assert request["schema_version"] == 2
    assert set(request["input_identities"]) == {
        "target_table",
        "variability_table",
        "psf_bundle",
    }
    assert (
        plan.run_dir
        / "stamps"
        / "target_10"
        / "source_variability_truth.ecsv"
    ).is_file()


def test_stamp_resume_rejects_changed_direct_target_table(tmp_path) -> None:
    from et_mainsim.config import StampWorkload
    from et_mainsim.manifest import ManifestIdentityError
    from et_mainsim.presets import load_preset
    from et_mainsim.workflows.stamp import build_run_plan, run_stamp

    loaded = load_preset("et-stamp-smoke")
    data_root = tmp_path / "data"
    bundle_name, bundle_sha256 = _write_test_psf_bundle(data_root)
    table_path = tmp_path / "targets.csv"
    table_path.write_text("gaia_g_mag,psf_id\n12.0,0\n", encoding="utf-8")
    spec = replace(
        loaded.simulation_spec,
        detector=replace(loaded.simulation_spec.detector, n_subpixels=3),
        psf=replace(
            loaded.simulation_spec.psf,
            bundle_name=bundle_name,
            bundle_sha256=bundle_sha256,
        ),
    )
    config = replace(
        loaded.run_config,
        paths=replace(
            loaded.run_config.paths,
            output_root=str(tmp_path / "output"),
            data_root=str(data_root),
            catalog_path="",
            focalplane_registry="",
        ),
        workload=StampWorkload(
            input_mode="table",
            input_table=str(table_path),
            include_neighbors=False,
        ),
    )
    plan = build_run_plan(
        preset_name="et-stamp-smoke",
        run_config=config,
        spec=spec,
        repo_root=tmp_path,
    )
    run_stamp(plan)
    static_truth = Table.read(
        plan.run_dir
        / "stamps"
        / "target_0"
        / "source_variability_truth.ecsv",
        format="ascii.ecsv",
    )
    np.testing.assert_allclose(static_truth["relative_flux"], [1.0, 1.0])
    np.testing.assert_allclose(
        static_truth["effective_photon_count_electron"],
        static_truth["baseline_photon_count_electron"],
    )
    table_path.write_text("gaia_g_mag,psf_id\n13.0,0\n", encoding="utf-8")

    with pytest.raises(ManifestIdentityError, match="workload"):
        run_stamp(plan)


def test_stamp_resume_rejects_changed_psf_bundle_content(tmp_path) -> None:
    from et_mainsim.manifest import ManifestIdentityError
    from et_mainsim.workflows.stamp import run_stamp

    plan = _variable_table_plan(
        tmp_path,
        target_body=(
            "source_id,gaia_g_mag,psf_id,curve_id\n10,12.0,0,sn\n"
        ),
    )
    bundle_name, bundle_sha256 = _write_test_psf_bundle(
        plan.paths.data_root
    )
    plan = replace(
        plan,
        spec=replace(
            plan.spec,
            detector=replace(plan.spec.detector, n_subpixels=3),
            psf=replace(
                plan.spec.psf,
                bundle_name=bundle_name,
                bundle_sha256=bundle_sha256,
            ),
        ),
    )
    run_stamp(plan)
    bundle_path = (
        plan.paths.data_root / bundle_name / "sim_psf_images.pkl"
    )
    bundle_path.write_bytes(bundle_path.read_bytes() + b"changed")

    with pytest.raises(ManifestIdentityError, match="workload"):
        run_stamp(plan)
