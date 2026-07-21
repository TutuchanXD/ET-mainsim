from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import json
import pickle

import h5py
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


def test_coadd_shard_provenance_is_constant_size_for_30_day_plan(
    tmp_path,
) -> None:
    from et_mainsim.workflows.stamp import _coadd_shard_provenance

    plan = _table_plan(tmp_path)
    plan = replace(
        plan,
        spec=SimpleNamespace(
            observation=SimpleNamespace(
                resolved_n_frames=259_200,
                n_raw_frames_per_coadd=6,
            )
        ),
    )

    provenance = _coadd_shard_provenance(plan)
    encoded = json.dumps(
        provenance,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")

    assert len(encoded) < 2 * 1024
    assert "selected_global_raw_frame_indices" not in provenance
    assert "selected_global_coadd_indices" not in provenance
    assert provenance["selected_raw_frame_count"] == 259_200
    assert provenance["selected_coadd_count"] == 43_200
    assert provenance["selection_rule"] == {
        "kind": "strided_global_coadds",
        "start": 0,
        "stop_exclusive": 43_200,
        "step": 1,
        "raw_frame_mapping": "contiguous_blocks_by_coadd_index",
    }


def test_artifact_policy_describes_batch_write_with_runtime_fallback(
    tmp_path,
) -> None:
    from et_mainsim.workflows.stamp import _artifact_policy

    policy = _artifact_policy(_table_plan(tmp_path))

    assert policy["write_strategy"] == (
        "batch_preferred_with_single_write_fallback"
    )
    assert "write_api" not in policy


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
    return bundle_name


def test_stamp_run_writes_readable_raw_coadd_truth_and_resumes(tmp_path) -> None:
    from et_mainsim.presets import load_preset
    from et_mainsim.workflows.stamp import build_run_plan, run_stamp
    from photsim7.artifacts import StampShardReader

    loaded = load_preset("et-stamp-smoke")
    data_root = tmp_path / "data"
    bundle_name = _write_test_psf_bundle(data_root)
    spec = replace(
        loaded.simulation_spec,
        detector=replace(loaded.simulation_spec.detector, n_subpixels=3),
        psf=replace(loaded.simulation_spec.psf, bundle_name=bundle_name),
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

    manifest_path = plan.run_dir / "run_manifest.json"
    legacy_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    legacy_manifest["workload"].pop("artifact_profile")
    legacy_manifest["workload"].pop("write_batch_size")
    legacy_manifest["workload"].pop("coadd_shard_index", None)
    legacy_manifest["workload"].pop("coadd_shard_count", None)
    legacy_manifest["artifacts"].pop("artifact_policy")
    legacy_manifest["artifacts"].pop("coadd_shard", None)
    legacy_manifest["provenance"].pop("coadd_shard", None)
    legacy_manifest["frame_plan"].pop("global_raw_frame_count", None)
    legacy_manifest["frame_plan"].pop("global_coadd_count", None)
    legacy_manifest["frame_plan"].pop("coadd_shard_index", None)
    legacy_manifest["frame_plan"].pop("coadd_shard_count", None)
    manifest_path.write_text(json.dumps(legacy_manifest), encoding="utf-8")

    second = run_stamp(plan)

    assert second["status"] == "completed"
    assert second["completion"]["rendered_targets"] == 0
    assert second["completion"]["skipped_targets"] == 1
    assert len(second["attempts"]) == 2
    assert second["workload"]["artifact_profile"] == "detailed"
    assert second["workload"]["write_batch_size"] == 32
    assert second["workload"]["coadd_shard_index"] == 0
    assert second["workload"]["coadd_shard_count"] == 1
    assert second["artifacts"]["artifact_policy"]["profile"] == "detailed"
    assert second["artifacts"]["coadd_shard"]["coadd_shard_index"] == 0
    assert second["frame_plan"]["global_coadd_count"] == 1
    assert second["frame_plan"]["coadd_shard_count"] == 1
    assert second["provenance"]["coadd_shard"]["logical_run_id"] == (
        plan.run_config.run_id
    )

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
    bundle_name = _write_test_psf_bundle(data_root)
    plan = replace(
        plan,
        spec=replace(
            plan.spec,
            detector=replace(plan.spec.detector, n_subpixels=3),
            psf=replace(plan.spec.psf, bundle_name=bundle_name),
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


def test_truth_accumulator_uses_fixed_column_arrays_and_orders_frames() -> None:
    from et_mainsim.workflows.stamp import _SourceVariabilityTruthAccumulator

    def products(frame_index, relative_flux):
        baseline = 100.0
        return SimpleNamespace(
            frame_index=frame_index,
            truth=SimpleNamespace(
                payload={
                    "source_id": np.array([10], dtype=np.int64),
                    "source_relative_flux_factor": np.array([relative_flux]),
                    "source_baseline_photon_count_electron": np.array([baseline]),
                    "source_effective_photon_count_electron": np.array(
                        [baseline * relative_flux]
                    ),
                    "source_psf_field_index": np.array([4], dtype=np.int64),
                }
            ),
        )

    accumulator = _SourceVariabilityTruthAccumulator(
        raw_frame_count=3,
        target_id=10,
        curve_id="sn",
    )
    assert isinstance(accumulator.relative_flux, np.ndarray)
    assert isinstance(accumulator.baseline_photon_count_electron, np.ndarray)
    assert isinstance(accumulator.effective_photon_count_electron, np.ndarray)
    assert isinstance(accumulator.psf_field_id, np.ndarray)
    assert not hasattr(accumulator, "rows")

    accumulator.add(products(2, 0.8))
    accumulator.add(products(0, 0.5))
    accumulator.add(products(1, 1.0))
    table = accumulator.to_table(source_input_truth={"mode": "test"})

    np.testing.assert_array_equal(table["frame_index"], [0, 1, 2])
    np.testing.assert_array_equal(table["source_id"], [10, 10, 10])
    np.testing.assert_array_equal(table["curve_id"], ["sn", "sn", "sn"])
    np.testing.assert_allclose(table["relative_flux"], [0.5, 1.0, 0.8])
    np.testing.assert_allclose(
        table["effective_photon_count_electron"],
        table["baseline_photon_count_electron"] * table["relative_flux"],
    )
    assert table.meta["source_input_truth"] == {"mode": "test"}
    with pytest.raises(RuntimeError, match="duplicate raw frame"):
        accumulator.add(products(1, 1.0))

    incomplete = _SourceVariabilityTruthAccumulator(
        raw_frame_count=2,
        target_id=10,
        curve_id=None,
    )
    incomplete.add(products(0, 1.0))
    with pytest.raises(RuntimeError, match="missing raw frames"):
        incomplete.to_table(source_input_truth={})

    global_frames = _SourceVariabilityTruthAccumulator(
        frame_indices=(2, 3),
        target_id=10,
        curve_id="sn",
        coadd_shard={"selected_global_coadd_indices": [1]},
    )
    global_frames.add(products(3, 4.0))
    global_frames.add(products(2, 3.0))
    global_table = global_frames.to_table(source_input_truth={})
    np.testing.assert_array_equal(global_table["frame_index"], [2, 3])
    np.testing.assert_allclose(global_table["relative_flux"], [3.0, 4.0])
    assert global_table.meta["coadd_shard"] == {
        "selected_global_coadd_indices": [1]
    }


def test_coadd_shard_must_select_at_least_one_global_coadd(tmp_path) -> None:
    from et_mainsim.workflows.stamp import _frame_plan

    plan = _table_plan(tmp_path)
    workload = replace(
        plan.workload,
        coadd_shard_index=1,
        coadd_shard_count=2,
    )

    with pytest.raises(ValueError, match="selects no global coadds"):
        _frame_plan(plan.spec, workload)


def test_stamp_write_buffer_flushes_at_configured_batch_size() -> None:
    from et_mainsim.workflows.stamp import _StampWriteBuffer

    class Writer:
        def __init__(self):
            self.batches = []

        def write_stamps(self, stamps):
            self.batches.append(tuple(stamps))

        def write_stamp(self, *_args, **_kwargs):
            raise AssertionError("batch-capable writer must not use write_stamp")

    writer = Writer()
    buffer = _StampWriteBuffer(writer, batch_size=2)
    for frame_id in range(5):
        buffer.add(
            star_id=10,
            frame_id=frame_id,
            stamp=np.full((2, 2), frame_id, dtype=np.uint16),
            seed=41 + frame_id,
        )

    assert [len(batch) for batch in writer.batches] == [2, 2]
    assert buffer.pending_count == 1
    buffer.flush()
    assert [len(batch) for batch in writer.batches] == [2, 2, 1]
    assert buffer.pending_count == 0


def test_compact_artifact_profile_omits_schema_sidecars_and_resumes(
    tmp_path,
) -> None:
    from et_mainsim.workflows.stamp import (
        _science_api,
        run_stamp,
        target_is_complete,
    )

    plan = _variable_table_plan(
        tmp_path,
        target_body=(
            "source_id,gaia_g_mag,psf_id,curve_id\n10,12.0,0,sn\n"
        ),
    )
    bundle_name = _write_test_psf_bundle(plan.paths.data_root)
    plan = replace(
        plan,
        run_config=replace(
            plan.run_config,
            workload=replace(
                plan.workload,
                artifact_profile="compact",
                write_batch_size=4,
            ),
        ),
        spec=replace(
            plan.spec,
            detector=replace(plan.spec.detector, n_subpixels=3),
            psf=replace(plan.spec.psf, bundle_name=bundle_name),
        ),
    )

    first = run_stamp(plan)

    target_dir = plan.run_dir / "stamps" / "target_10"
    assert (target_dir / "raw.h5").is_file()
    assert (target_dir / "coadd.h5").is_file()
    assert not (target_dir / "schemas").exists()
    truth = Table.read(
        target_dir / "source_variability_truth.ecsv",
        format="ascii.ecsv",
    )
    np.testing.assert_allclose(truth["relative_flux"], [0.5, 2.0])
    expected_policy = {
        "profile": "compact",
        "raw_schema_sidecars": False,
        "coadd_schema_sidecars": False,
        "electron_component_sidecars": False,
        "write_batch_size": 4,
        "write_strategy": "batch_preferred_with_single_write_fallback",
    }
    assert first["workload"]["artifact_profile"] == "compact"
    assert first["workload"]["write_batch_size"] == 4
    assert first["artifacts"]["artifact_policy"] == expected_policy
    target_artifacts = json.loads(
        (target_dir / "target_artifacts.json").read_text(encoding="utf-8")
    )
    assert target_artifacts["artifact_policy"] == expected_policy
    assert target_is_complete(plan, 10, api=_science_api())

    second = run_stamp(plan)

    assert second["completion"]["rendered_targets"] == 0
    assert second["completion"]["skipped_targets"] == 1


def test_coadd_shard_renders_global_indices_with_full_timespan_services(
    tmp_path,
) -> None:
    from et_mainsim.workflows.stamp import (
        _science_api,
        build_run_plan,
        run_stamp,
    )
    from et_mainsim.stamp_inputs import file_identity
    from photsim7.artifacts import StampShardReader

    base_plan = _variable_table_plan(
        tmp_path,
        target_body=(
            "source_id,gaia_g_mag,psf_id,curve_id\n10,12.0,0,sn\n"
        ),
        curve_body=(
            "curve_id,frame_index,relative_flux\n"
            "sn,0,1\nsn,1,2\nsn,2,3\nsn,3,4\nsn,4,5\n"
            "sn,5,6\nsn,6,7\nsn,7,8\nsn,8,9\nsn,9,10\n"
        ),
    )
    bundle_name = _write_test_psf_bundle(base_plan.paths.data_root)
    config = replace(
        base_plan.run_config,
        workload=replace(
            base_plan.workload,
            coadd_shard_index=1,
            coadd_shard_count=2,
        ),
    )
    spec = replace(
        base_plan.spec,
        detector=replace(base_plan.spec.detector, n_subpixels=3),
        psf=replace(base_plan.spec.psf, bundle_name=bundle_name),
        observation=replace(
            base_plan.spec.observation,
            exposure_duration=10 * u.s,
            readout_duration=0 * u.s,
            observing_duration=100 * u.s,
            n_frames=10,
            n_raw_frames_per_coadd=2,
        ),
    )
    plan = build_run_plan(
        preset_name=base_plan.preset_name,
        run_config=config,
        spec=spec,
        repo_root=tmp_path,
    )
    assert plan.run_dir == (
        plan.paths.output_root
        / plan.run_config.run_id
        / "coadd_shard_0001_of_0002"
    )
    api = _science_api()
    original_build_services = api.build_stamp_services
    original_run_coadd = api.run_stamp_coadd
    service_frame_counts = []
    rendered_coadd_indices = []

    def build_services(spec, *args, **kwargs):
        service_frame_counts.append(int(spec.observation.resolved_n_frames))
        return original_build_services(spec, *args, **kwargs)

    def run_coadd(spec, *args, coadd_index, **kwargs):
        rendered_coadd_indices.append(coadd_index)
        assert int(spec.observation.resolved_n_frames) == 10
        return original_run_coadd(
            spec,
            *args,
            coadd_index=coadd_index,
            **kwargs,
        )

    api.build_stamp_services = build_services
    api.run_stamp_coadd = run_coadd

    manifest = run_stamp(plan, science_api=api)

    assert service_frame_counts == [10]
    assert rendered_coadd_indices == [1, 3]
    assert manifest["simulation_spec"]["observation"]["n_frames"] == 10
    assert manifest["frame_plan"] == {
        "global_raw_frame_count": 10,
        "global_coadd_count": 5,
        "n_raw_frames_per_coadd": 2,
        "coadd_shard_index": 1,
        "coadd_shard_count": 2,
        "raw_frame_count": 4,
        "raw_frame_indices": [2, 3, 6, 7],
        "coadd_count": 2,
        "coadd_indices": [1, 3],
    }
    shard_provenance = {
        "logical_run_id": plan.run_config.run_id,
        "output_relative_path": "coadd_shard_0001_of_0002",
        "global_raw_frame_count": 10,
        "global_coadd_count": 5,
        "n_raw_frames_per_coadd": 2,
        "coadd_shard_index": 1,
        "coadd_shard_count": 2,
        "selected_raw_frame_count": 4,
        "selected_coadd_count": 2,
        "selection_rule": {
            "kind": "strided_global_coadds",
            "start": 1,
            "stop_exclusive": 5,
            "step": 2,
            "raw_frame_mapping": "contiguous_blocks_by_coadd_index",
        },
    }
    assert manifest["artifacts"]["coadd_shard"] == shard_provenance

    target_dir = plan.run_dir / "stamps" / "target_10"
    with StampShardReader(target_dir / "raw.h5") as raw_reader:
        assert raw_reader.frame_ids == (2, 3, 6, 7)
        assert raw_reader.spec.provenance["coadd_shard"] == shard_provenance
    with StampShardReader(target_dir / "coadd.h5") as coadd_reader:
        assert coadd_reader.frame_ids == (1, 3)
        assert coadd_reader.spec.provenance["coadd_shard"] == shard_provenance
    truth = Table.read(
        target_dir / "source_variability_truth.ecsv",
        format="ascii.ecsv",
    )
    np.testing.assert_array_equal(truth["frame_index"], [2, 3, 6, 7])
    np.testing.assert_allclose(truth["relative_flux"], [3.0, 4.0, 7.0, 8.0])
    assert truth.meta["coadd_shard"] == shard_provenance
    assert not (target_dir / "schemas" / "raw" / "frame_000000.json").exists()
    raw_schema = json.loads(
        (target_dir / "schemas" / "raw" / "frame_000002.json").read_text(
            encoding="utf-8"
        )
    )
    coadd_schema = json.loads(
        (target_dir / "schemas" / "coadd" / "coadd_000001.json").read_text(
            encoding="utf-8"
        )
    )
    assert raw_schema["coadd_shard"] == shard_provenance
    assert coadd_schema["coadd_shard"] == shard_provenance
    target_artifacts = json.loads(
        (target_dir / "target_artifacts.json").read_text(encoding="utf-8")
    )
    assert target_artifacts["coadd_shard"] == shard_provenance

    legacy_shard_provenance = {
        key: value
        for key, value in shard_provenance.items()
        if key
        not in {
            "selected_raw_frame_count",
            "selected_coadd_count",
            "selection_rule",
        }
    }
    legacy_shard_provenance.update(
        {
            "selected_global_raw_frame_indices": [2, 3, 6, 7],
            "selected_global_coadd_indices": [1, 3],
        }
    )
    for shard_name in ("raw.h5", "coadd.h5"):
        with h5py.File(target_dir / shard_name, "r+") as handle:
            encoded_provenance = handle["provenance"][()]
            if isinstance(encoded_provenance, bytes):
                encoded_provenance = encoded_provenance.decode("utf-8")
            stored_provenance = json.loads(encoded_provenance)
            stored_provenance["coadd_shard"] = legacy_shard_provenance
            handle["provenance"][()] = json.dumps(
                stored_provenance,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
    truth.meta["coadd_shard"] = legacy_shard_provenance
    truth.write(
        target_dir / "source_variability_truth.ecsv",
        format="ascii.ecsv",
        overwrite=True,
    )
    for schema_path in (
        target_dir / "schemas" / "raw" / "frame_000002.json",
        target_dir / "schemas" / "coadd" / "coadd_000001.json",
    ):
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        schema["coadd_shard"] = legacy_shard_provenance
        schema_path.write_text(json.dumps(schema), encoding="utf-8")
    target_artifacts["coadd_shard"] = legacy_shard_provenance
    target_artifacts["source_variability_truth_identity"] = file_identity(
        target_dir / "source_variability_truth.ecsv"
    )
    (target_dir / "target_artifacts.json").write_text(
        json.dumps(target_artifacts),
        encoding="utf-8",
    )
    manifest_path = plan.run_dir / "run_manifest.json"
    legacy_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    legacy_manifest["artifacts"]["coadd_shard"] = legacy_shard_provenance
    legacy_manifest["provenance"]["coadd_shard"] = legacy_shard_provenance
    manifest_path.write_text(json.dumps(legacy_manifest), encoding="utf-8")

    with StampShardReader(target_dir / "raw.h5") as legacy_raw_reader:
        assert legacy_raw_reader.frame_ids == (2, 3, 6, 7)
        assert (
            legacy_raw_reader.spec.provenance["coadd_shard"]
            == legacy_shard_provenance
        )

    second = run_stamp(plan, science_api=api)

    assert second["completion"]["rendered_targets"] == 0
    assert second["completion"]["skipped_targets"] == 1
    assert second["artifacts"]["coadd_shard"] == legacy_shard_provenance
    assert rendered_coadd_indices == [1, 3]


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
    bundle_name = _write_test_psf_bundle(plan.paths.data_root)
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
            psf=replace(plan.spec.psf, bundle_name=bundle_name),
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
    bundle_name = _write_test_psf_bundle(data_root)
    table_path = tmp_path / "targets.csv"
    table_path.write_text("gaia_g_mag,psf_id\n12.0,0\n", encoding="utf-8")
    spec = replace(
        loaded.simulation_spec,
        detector=replace(loaded.simulation_spec.detector, n_subpixels=3),
        psf=replace(loaded.simulation_spec.psf, bundle_name=bundle_name),
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


def test_stamp_rejects_mismatched_spec_before_legacy_manifest_upgrade(
    tmp_path,
) -> None:
    from et_mainsim.manifest import ManifestIdentityError, RunManifestStore
    from et_mainsim.workflows.stamp import (
        _frame_plan,
        _workload_identity,
        run_stamp,
    )

    plan = _table_plan(tmp_path)
    bundle_name = _write_test_psf_bundle(plan.paths.data_root)
    plan = replace(
        plan,
        spec=replace(
            plan.spec,
            detector=replace(plan.spec.detector, n_subpixels=3),
            psf=replace(plan.spec.psf, bundle_name=bundle_name),
        ),
    )
    execution_payload = {
        **plan.run_config.execution.to_dict(),
        "paths": plan.paths.to_dict(),
    }
    store = RunManifestStore(plan.run_dir / "run_manifest.json")
    store.create(
        workflow="et-stamp",
        preset=plan.preset_name,
        run_id=plan.run_config.run_id,
        simulation_spec=plan.spec.to_json_dict(),
        execution=execution_payload,
        workload=_workload_identity(plan),
        frame_plan=_frame_plan(plan.spec, plan.workload),
        provenance={},
        artifacts={},
    )
    manifest_path = store.path
    original_bytes = manifest_path.read_bytes()
    mismatched_plan = replace(
        plan,
        spec=replace(
            plan.spec,
            rng=replace(
                plan.spec.rng,
                run_seed=plan.spec.rng.run_seed + 1,
            ),
        ),
    )

    with pytest.raises(ManifestIdentityError, match="scientific spec"):
        run_stamp(mismatched_plan, science_api=SimpleNamespace())

    assert manifest_path.read_bytes() == original_bytes


def test_stamp_resume_rejects_changed_psf_bundle_content(tmp_path) -> None:
    from et_mainsim.manifest import ManifestIdentityError
    from et_mainsim.workflows.stamp import run_stamp

    plan = _variable_table_plan(
        tmp_path,
        target_body=(
            "source_id,gaia_g_mag,psf_id,curve_id\n10,12.0,0,sn\n"
        ),
    )
    bundle_name = _write_test_psf_bundle(plan.paths.data_root)
    plan = replace(
        plan,
        spec=replace(
            plan.spec,
            detector=replace(plan.spec.detector, n_subpixels=3),
            psf=replace(plan.spec.psf, bundle_name=bundle_name),
        ),
    )
    run_stamp(plan)
    bundle_path = (
        plan.paths.data_root / bundle_name / "sim_psf_images.pkl"
    )
    bundle_path.write_bytes(bundle_path.read_bytes() + b"changed")

    with pytest.raises(ManifestIdentityError, match="workload"):
        run_stamp(plan)
