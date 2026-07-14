from __future__ import annotations

import json
import pickle
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from astropy import units as u


def _write_complete_frame(run_dir: Path, frame_index: int, shape=(5, 7)) -> None:
    frames = run_dir / "frames"
    summaries = run_dir / "frame_summaries"
    frames.mkdir(parents=True, exist_ok=True)
    summaries.mkdir(parents=True, exist_ok=True)
    np.save(frames / f"frame_{frame_index:06d}.npy", np.zeros(shape, dtype=np.uint16))
    (summaries / f"frame_{frame_index:06d}.json").write_text(
        json.dumps(
            {
                "artifact_schema_version": 1,
                "frame_index": frame_index,
            }
        ),
        encoding="utf-8",
    )
    (summaries / f"frame_{frame_index:06d}_schema.json").write_text(
        json.dumps(
            {
                "schema_id": "photsim7.single_cadence_frame_products.v1",
                "schema_version": 1,
                "frame_index": frame_index,
                "detector_id": "main_rd",
                "coordinate_convention": "frame_yx",
                "arrays": {
                    "final_frame": {
                        "shape": list(shape),
                        "dtype": "uint16",
                        "domain": "dn",
                        "unit": "dn",
                        "coordinate_convention": "frame_yx",
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def test_resume_requires_frame_summary_schema_and_matching_shape(tmp_path) -> None:
    from et_mainsim.workflows.full_frame import frame_is_complete

    run_dir = tmp_path / "run"
    _write_complete_frame(run_dir, 0)

    assert frame_is_complete(run_dir, 0, expected_shape=(5, 7)) is True
    assert frame_is_complete(run_dir, 0, expected_shape=(7, 5)) is False

    (run_dir / "frame_summaries" / "frame_000000_schema.json").unlink()
    assert frame_is_complete(run_dir, 0, expected_shape=(5, 7)) is False


def test_worker_delegates_rendering_to_photsim7_public_pipeline(tmp_path) -> None:
    from et_mainsim.config import ExecutionConfig
    from et_mainsim.workflows.full_frame import WorkerRequest, run_worker
    from photsim7.spec_factories import make_et_main_detector_spec

    calls: list[tuple[str, object]] = []
    spec = make_et_main_detector_spec(shape=(5, 7), run_seed=7)
    catalog = SimpleNamespace(n_sources=1, star_data={"et_mag": np.array([12.0])})
    services = SimpleNamespace(spec=spec)

    class FakeCache:
        @staticmethod
        def read(path):
            calls.append(("read_catalog", Path(path)))
            return catalog

    class FakeRegistry:
        def __init__(self, *, data_root):
            calls.append(("registry", Path(data_root)))

    class FakeWriter:
        def __init__(self, run_dir, *, options):
            self.run_dir = Path(run_dir)
            calls.append(("writer", options))

    def fake_build_services(typed_spec, *, catalog, data_registry):
        calls.append(("build_full_frame_services", typed_spec))
        return services

    def fake_run_frame(typed_spec, **kwargs):
        frame_index = kwargs["frame_index"]
        calls.append(("run_single_cadence_full_frame", frame_index))
        _write_complete_frame(tmp_path / "run", frame_index)
        return SimpleNamespace(
            renderer_components={},
            detector_result=SimpleNamespace(cosmic_metadata=None),
        )

    fake_api = SimpleNamespace(
        DataRegistry=FakeRegistry,
        StarCatalogCache=FakeCache,
        FullFrameArtifactOptions=lambda **kwargs: kwargs,
        FullFrameArtifactWriter=FakeWriter,
        build_full_frame_services=fake_build_services,
        run_single_cadence_full_frame=fake_run_frame,
    )
    request = WorkerRequest(
        spec=spec,
        execution=ExecutionConfig(
            backend="in-process",
            device="cpu",
            resume=True,
            preview_count=0,
        ),
        run_dir=tmp_path / "run",
        data_root=tmp_path / "data",
        catalog_cache=tmp_path / "stars.npz",
        frame_indices=(0,),
        rank=0,
        world_size=1,
    )

    result = run_worker(request, science_api=fake_api)

    assert result.rendered == (0,)
    assert result.skipped == ()
    assert [name for name, _ in calls] == [
        "read_catalog",
        "registry",
        "build_full_frame_services",
        "writer",
        "run_single_cadence_full_frame",
    ]


def test_worker_resume_skips_without_loading_catalog_or_services(tmp_path) -> None:
    from et_mainsim.config import ExecutionConfig
    from et_mainsim.workflows.full_frame import WorkerRequest, run_worker
    from photsim7.spec_factories import make_et_main_detector_spec

    run_dir = tmp_path / "run"
    _write_complete_frame(run_dir, 0)

    class ForbiddenCache:
        @staticmethod
        def read(path):
            raise AssertionError(f"resume should not read catalog cache {path}")

    fake_api = SimpleNamespace(StarCatalogCache=ForbiddenCache)
    request = WorkerRequest(
        spec=make_et_main_detector_spec(shape=(5, 7), run_seed=7),
        execution=ExecutionConfig(
            backend="in-process",
            device="cpu",
            resume=True,
            preview_count=0,
        ),
        run_dir=run_dir,
        data_root=tmp_path / "data",
        catalog_cache=tmp_path / "stars.npz",
        frame_indices=(0,),
    )

    result = run_worker(request, science_api=fake_api)

    assert result.rendered == ()
    assert result.skipped == (0,)


def _write_test_psf_bundle(data_root: Path) -> str:
    bundle_name = "psf/et/et_mainsim_test"
    bundle_dir = data_root / bundle_name
    bundle_dir.mkdir(parents=True)
    n_subpixels = 3
    rows, cols = 5, 7
    y, x = np.mgrid[: rows * n_subpixels, : cols * n_subpixels].astype(np.float32)
    yy = (y - y.mean()) / n_subpixels
    xx = (x - x.mean()) / n_subpixels
    image = np.exp(-0.5 * (xx**2 + yy**2) / 1.2**2).astype(np.float32)
    image /= image.sum(dtype=np.float64)
    payload = {
        "images": {0: {n_subpixels: np.stack([x, y, image])}},
        "angles": np.array([0.0]),
    }
    with (bundle_dir / "sim_psf_images.pkl").open("wb") as handle:
        pickle.dump(payload, handle)
    return bundle_name


def test_tiny_cpu_worker_writes_readable_photsim7_artifacts(tmp_path) -> None:
    from et_mainsim.config import ExecutionConfig
    from et_mainsim.workflows.full_frame import WorkerRequest, run_worker
    from photsim7.catalog_sources import PreparedStarCatalog, StarCatalogCache
    from photsim7.frame_products import read_frame_product_schema
    from photsim7.spec_factories import make_et_main_detector_spec
    from photsim7.specs import (
        CosmicRaySpec,
        DetectorResponseSpec,
        DynamicEffectsSpec,
    )

    data_root = tmp_path / "data"
    bundle_name = _write_test_psf_bundle(data_root)
    base = make_et_main_detector_spec(shape=(5, 7), run_seed=17)
    spec = replace(
        base,
        observation=replace(
            base.observation,
            exposure_duration=1 * u.s,
            readout_duration=0 * u.s,
            observing_duration=1 * u.s,
            n_frames=1,
        ),
        catalog=replace(base.catalog, source_type="prepared"),
        detector=replace(base.detector, n_subpixels=3),
        detector_response=DetectorResponseSpec(
            enable_inter_pixel_response=False,
            enable_intra_pixel_response=False,
            enable_pixel_phase_response=False,
        ),
        cosmic_rays=CosmicRaySpec(enabled=False),
        psf=replace(
            base.psf,
            bundle_name=bundle_name,
            field_id=0,
            field_id_policy=None,
            use_jitter_integrated_psf=False,
            compute_device="cpu",
        ),
        dynamic_effects=DynamicEffectsSpec(),
    )
    catalog = PreparedStarCatalog(
        star_data={
            "x0": np.array([0.0]),
            "y0": np.array([0.0]),
            "ra": np.array([10.0]),
            "dec": np.array([20.0]),
            "source_id": np.array([11], dtype=np.int64),
            "et_mag": np.array([12.0]),
            "frame_xpix": np.array([3.0]),
            "frame_ypix": np.array([2.0]),
            "detector_xpix_shifted": np.array([3.0]),
            "detector_ypix_shifted": np.array([2.0]),
        },
        metadata={"source": {"type": "prepared", "n_sources": 1}},
    )
    cache_path = tmp_path / "stars.npz"
    StarCatalogCache.write(cache_path, catalog)
    run_dir = tmp_path / "run"
    request = WorkerRequest(
        spec=spec,
        execution=ExecutionConfig(
            backend="in-process",
            device="cpu",
            resume=True,
            preview_count=0,
        ),
        run_dir=run_dir,
        data_root=data_root,
        catalog_cache=cache_path,
        frame_indices=(0,),
        rank=0,
        world_size=1,
    )

    result = run_worker(request)

    frame = np.load(run_dir / "frames" / "frame_000000.npy")
    schema = read_frame_product_schema(
        run_dir / "frame_summaries" / "frame_000000_schema.json"
    )
    summary = json.loads(
        (run_dir / "frame_summaries" / "frame_000000.json").read_text(encoding="utf-8")
    )
    assert result.rendered == (0,)
    assert frame.shape == (5, 7)
    assert frame.dtype == np.uint16
    assert schema["arrays"]["final_frame"]["domain"] == "dn"
    assert summary["et_mainsim"]["rank"] == 0
    assert summary["et_mainsim"]["n_stars"] == 1
    assert summary["et_mainsim"]["pipeline_elapsed_s"] >= 0.0


def test_run_refuses_nonempty_legacy_directory_without_manifest(tmp_path) -> None:
    from et_mainsim.config import RunPaths
    from et_mainsim.presets import load_preset
    from et_mainsim.workflows.full_frame import build_run_plan, run_full_frame

    loaded = load_preset("et-full-frame-smoke")
    data_root = tmp_path / "data"
    data_root.mkdir()
    config = replace(
        loaded.run_config,
        paths=RunPaths(
            output_root=str(tmp_path / "results"),
            data_root=str(data_root),
        ),
    )
    plan = build_run_plan(
        preset_name=loaded.descriptor.name,
        run_config=config,
        spec=loaded.simulation_spec,
        repo_root=tmp_path,
    )
    plan.run_dir.mkdir(parents=True)
    (plan.run_dir / "historical-output.npy").write_bytes(b"old")

    with pytest.raises(FileExistsError, match="does not contain run_manifest.json"):
        run_full_frame(plan)

    assert not (plan.run_dir / "run_manifest.json").exists()


def test_run_records_worker_failure_in_manifest(tmp_path) -> None:
    from et_mainsim.config import RunPaths
    from et_mainsim.presets import load_preset
    from et_mainsim.workflows.full_frame import build_run_plan, run_full_frame

    loaded = load_preset("et-full-frame-smoke")
    data_root = tmp_path / "data"
    data_root.mkdir()
    config = replace(
        loaded.run_config,
        paths=RunPaths(
            output_root=str(tmp_path / "results"),
            data_root=str(data_root),
        ),
    )
    plan = build_run_plan(
        preset_name=loaded.descriptor.name,
        run_config=config,
        spec=loaded.simulation_spec,
        repo_root=tmp_path,
    )
    catalog = SimpleNamespace(
        n_sources=1,
        metadata={"request": {"schema_id": "test"}},
        star_data={"et_mag": np.array([12.0])},
    )

    class FakeCache:
        @staticmethod
        def read(path):
            return catalog

    class FakeRegistry:
        def __init__(self, *, data_root):
            self.data_root = data_root

    def fail_services(*args, **kwargs):
        raise RuntimeError("service construction failed")

    fake_api = SimpleNamespace(
        DataRegistry=FakeRegistry,
        StarCatalogCache=FakeCache,
        build_catalog_from_spec=lambda *args, **kwargs: catalog,
        build_full_frame_services=fail_services,
    )

    with pytest.raises(RuntimeError, match="service construction failed"):
        run_full_frame(plan, science_api=fake_api)

    manifest = json.loads(
        (plan.run_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "failed"
    assert manifest["failure"] == {
        "type": "RuntimeError",
        "message": "service construction failed",
    }
    assert manifest["attempts"][-1]["status"] == "failed"


def test_full_frame_plan_accepts_explicit_external_catalog_cache(tmp_path) -> None:
    from et_mainsim.config import RunPaths
    from et_mainsim.presets import load_preset
    from et_mainsim.workflows.full_frame import build_run_plan

    loaded = load_preset("et-full-frame-smoke")
    external_cache = tmp_path / "shared" / "stars.npz"
    config = replace(
        loaded.run_config,
        paths=RunPaths(
            output_root=str(tmp_path / "results"),
            data_root=str(tmp_path / "data"),
            catalog_cache=str(external_cache),
        ),
    )

    plan = build_run_plan(
        preset_name=loaded.descriptor.name,
        run_config=config,
        spec=loaded.simulation_spec,
        repo_root=tmp_path,
    )

    assert plan.catalog_cache == external_cache
    assert plan.spec.catalog.cache_path == str(external_cache)


def test_full_frame_preflight_allows_validatable_cache_without_query_assets(
    tmp_path,
) -> None:
    from et_mainsim.config import RunPaths
    from et_mainsim.presets import load_preset
    from et_mainsim.workflows.full_frame import build_run_plan, preflight

    loaded = load_preset("et-full-frame-production")
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
