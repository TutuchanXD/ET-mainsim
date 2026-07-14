from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import json
import pickle

import numpy as np
import pytest
from astropy import units as u


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


def test_table_stamp_inputs_are_independent_catalogs_without_query(tmp_path) -> None:
    from photsim7.catalog_sources import PreparedStarCatalog
    from et_mainsim.workflows.stamp import prepare_stamp_inputs

    plan = _table_plan(tmp_path)
    plan.paths.data_root.mkdir()
    api = SimpleNamespace(
        PreparedStarCatalog=PreparedStarCatalog,
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
    request = StampWorkerRequest.from_plan(
        plan,
        target_ids=(0, 1),
        rank=1,
        world_size=2,
    )

    restored = StampWorkerRequest.from_json_dict(request.to_json_dict())

    assert restored.plan.workload.input_mode == "table"
    assert restored.plan.input_table_path == tmp_path / "targets.csv"
    assert restored.target_ids == (0, 1)
    assert restored.rank == 1
    assert restored.world_size == 2


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
    table_path.write_text("gaia_g_mag,psf_id\n13.0,0\n", encoding="utf-8")

    with pytest.raises(ManifestIdentityError, match="workload"):
        run_stamp(plan)
