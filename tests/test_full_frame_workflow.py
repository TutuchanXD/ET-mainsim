from __future__ import annotations

import hashlib
import json
import pickle
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from astropy import units as u


def _unavailable_selection_marker(
    *, profile_id: str = "unclaimed"
) -> dict[str, object]:
    return {
        "schema_id": "photsim7.cadence_selection_truth.v1",
        "schema_version": 1,
        "verification_status": "unavailable",
        "science_conformance_claim": False,
        "science_conformance_claim_scope": (
            "geometry_psf_and_jitter_selection_truth_only"
        ),
        "requested_science_profile_id": profile_id,
        "missing_components": ["jitter_model_selection_truth"],
    }


def _write_complete_frame(
    run_dir: Path,
    frame_index: int,
    shape=(5, 7),
    *,
    selection_truth: dict[str, object] | None = None,
) -> None:
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
                "selection_truth": (
                    _unavailable_selection_marker()
                    if selection_truth is None
                    else selection_truth
                ),
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


def _artifact_paths_for_test(run_dir: Path, frame_index: int) -> tuple[Path, ...]:
    stem = f"frame_{frame_index:06d}"
    return (
        run_dir / "frames" / f"{stem}.npy",
        run_dir / "frame_summaries" / f"{stem}.json",
        run_dir / "frame_summaries" / f"{stem}_schema.json",
    )


def _legacy_single_scope_spec(spec):
    """Make legacy-root tests explicit without changing science defaults."""

    return replace(
        spec,
        instrument=replace(spec.instrument, telescope_count=1),
    )


def test_scope_persistence_reuses_persisted_selection_metadata_in_provenance() -> None:
    from et_mainsim.workflows.full_frame import _persist_scope_frame_result
    from photsim7.frame_products import FrameArrayProduct, SingleCadenceFrameProducts

    in_memory_selection = {
        "schema_id": "photsim7.cadence_selection_truth.v1",
        "verification_status": "complete_in_memory",
    }
    product = SingleCadenceFrameProducts(
        frame_index=0,
        detector_id="main_rd",
        final_frame=FrameArrayProduct(
            name="final_frame",
            array=np.zeros((3, 5), dtype=np.uint16),
            unit="dn",
            domain="dn",
        ),
        frame_summary={"frame_index": 0},
        selection_truth=in_memory_selection,
        provenance={
            "services": {"scope": {"scope_id": 0, "scope_count": 6}},
            "selection_truth": in_memory_selection,
        },
    )
    truth = SimpleNamespace(
        schema_id="photsim7.cadence_selection_truth.v1",
        schema_version=1,
        science_conformance_claim=True,
        content_sha256="a" * 64,
        geometry_reference={"content_sha256": "b" * 64},
        psf_reference={"content_sha256": "c" * 64},
        jitter_model_selection_truth=SimpleNamespace(
            to_json_dict=lambda: {"model_index": 4}
        ),
    )

    def identity(label: str, digest: str):
        return SimpleNamespace(
            relative_path=f"selection_truth/{label}/{digest}.json",
            schema_id=f"test.{label}.v1",
            schema_version=1,
            content_sha256=digest * 64,
        )

    artifacts = SimpleNamespace(
        geometry=identity("geometry", "b"),
        psf=identity("psf", "c"),
        cadence=identity("cadence", "a"),
    )

    class RecordingWriter:
        written_product = None

        def write_selection_truth(self, value):
            assert value is truth
            return artifacts

        def write_frame(self, *args, **kwargs):
            return None

        def write_frame_product_schema(self, value):
            self.written_product = value

    writer = RecordingWriter()
    result = SimpleNamespace(
        frame_products=product,
        selection_truth=truth,
        detector_result=SimpleNamespace(cosmic_metadata=None, bias_metadata=None),
    )
    spec = SimpleNamespace(
        science_profile=SimpleNamespace(profile_id="legacy_science_v1")
    )

    _persist_scope_frame_result(writer=writer, result=result, spec=spec)

    persisted = writer.written_product.selection_truth
    assert persisted["verification_status"] == "persisted_and_verified"
    assert writer.written_product.provenance["selection_truth"] is persisted
    assert writer.written_product.provenance["services"] == {
        "scope": {"scope_id": 0, "scope_count": 6}
    }


def test_resume_requires_frame_summary_schema_and_matching_shape(tmp_path) -> None:
    from et_mainsim.workflows.full_frame import frame_is_complete
    from photsim7.spec_factories import make_et_main_detector_spec

    run_dir = tmp_path / "run"
    base = _legacy_single_scope_spec(
        make_et_main_detector_spec(shape=(5, 7), run_seed=7)
    )
    spec = replace(
        base,
        psf=replace(base.psf, use_jitter_integrated_psf=False),
    )
    _write_complete_frame(run_dir, 0)

    assert (
        frame_is_complete(
            run_dir,
            0,
            expected_shape=(5, 7),
            expected_spec=spec,
        )
        is True
    )
    requiring_persisted_truth = replace(
        spec,
        psf=replace(spec.psf, use_jitter_integrated_psf=True),
    )
    assert (
        frame_is_complete(
            run_dir,
            0,
            expected_shape=(5, 7),
            expected_spec=requiring_persisted_truth,
        )
        is False
    )
    assert (
        frame_is_complete(
            run_dir,
            0,
            expected_shape=(7, 5),
            expected_spec=spec,
        )
        is False
    )

    (run_dir / "frame_summaries" / "frame_000000_schema.json").unlink()
    assert (
        frame_is_complete(
            run_dir,
            0,
            expected_shape=(5, 7),
            expected_spec=spec,
        )
        is False
    )


def test_six_scope_worker_persists_scope_local_products_and_requires_all_scopes(
    tmp_path,
) -> None:
    from et_mainsim.config import ExecutionConfig
    from et_mainsim.workflows.full_frame import (
        WorkerRequest,
        frame_completion,
        run_worker,
    )
    from photsim7.frame_products import FrameArrayProduct, SingleCadenceFrameProducts
    from photsim7.full_frame_artifacts import (
        FullFrameArtifactOptions,
        FullFrameArtifactWriter,
    )
    from photsim7.spec_factories import make_et_main_detector_spec

    base = make_et_main_detector_spec(shape=(5, 7), run_seed=7)
    spec = replace(
        base,
        instrument=replace(base.instrument, telescope_count=6),
        psf=replace(base.psf, use_jitter_integrated_psf=False),
    )
    catalog = SimpleNamespace(
        n_sources=1,
        star_data={"et_mag": np.array([12.0])},
    )
    yielded_scope_ids: list[int] = []

    class FakeCache:
        @staticmethod
        def read(path):
            return catalog

    class FakeRegistry:
        def __init__(self, *, data_root):
            self.data_root = Path(data_root)

    def fake_iter_scopes(typed_spec, *, services, frame_index, **kwargs):
        assert typed_spec is spec
        assert services == "six-scope-services"
        assert frame_index == 0
        for scope_id in range(6):
            yielded_scope_ids.append(scope_id)
            product = SingleCadenceFrameProducts(
                frame_index=frame_index,
                detector_id=typed_spec.detector.detector_id,
                final_frame=FrameArrayProduct(
                    name="final_frame",
                    array=np.full((5, 7), scope_id, dtype=np.uint16),
                    unit="dn",
                    domain="dn",
                ),
                frame_summary={"frame_index": frame_index},
                selection_truth=_unavailable_selection_marker(
                    profile_id=typed_spec.science_profile.profile_id
                ),
                provenance={
                    "services": {
                        "scope": {"scope_id": scope_id, "scope_count": 6}
                    }
                },
            )
            yield scope_id, SimpleNamespace(
                frame_products=product,
                detector_result=SimpleNamespace(
                    cosmic_metadata=None,
                    bias_metadata=None,
                ),
                renderer_components={},
                selection_truth=None,
            )

    api = SimpleNamespace(
        DataRegistry=FakeRegistry,
        StarCatalogCache=FakeCache,
        FullFrameArtifactOptions=FullFrameArtifactOptions,
        FullFrameArtifactWriter=FullFrameArtifactWriter,
        build_multiscope_full_frame_services=(
            lambda typed_spec, *, catalog, data_registry: "six-scope-services"
        ),
        iter_single_cadence_full_frame_scopes=fake_iter_scopes,
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
    )

    result = run_worker(request, science_api=api)

    assert result.rendered == (0,)
    assert yielded_scope_ids == [0, 1, 2, 3, 4, 5]
    assert not (request.run_dir / "frames" / "frame_000000.npy").exists()
    for scope_id in range(6):
        schema_path = (
            request.run_dir
            / f"scope_{scope_id}"
            / "frame_summaries"
            / "frame_000000_schema.json"
        )
        assert schema_path.is_file()
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        assert schema["provenance"]["services"]["scope"]["scope_id"] == scope_id

    completion = frame_completion(
        request.run_dir,
        0,
        expected_shape=(5, 7),
        expected_spec=spec,
    )
    assert completion.is_complete is True

    scope_zero_summary = (
        request.run_dir
        / "scope_0"
        / "frame_summaries"
        / "frame_000000.json"
    )
    scope_zero_summary_before_repair = scope_zero_summary.read_bytes()
    scope_five_schema = (
        request.run_dir
        / "scope_5"
        / "frame_summaries"
        / "frame_000000_schema.json"
    )
    corrupt = json.loads(scope_five_schema.read_text(encoding="utf-8"))
    corrupt["provenance"]["services"]["scope"]["scope_id"] = 4
    scope_five_schema.write_text(json.dumps(corrupt), encoding="utf-8")

    assert (
        frame_completion(
            request.run_dir,
            0,
            expected_shape=(5, 7),
            expected_spec=spec,
        ).is_complete
        is False
    )

    yielded_scope_ids.clear()
    repaired = run_worker(request, science_api=api)

    assert repaired.rendered == (0,)
    assert yielded_scope_ids == [0, 1, 2, 3, 4, 5]
    assert scope_zero_summary.read_bytes() == scope_zero_summary_before_repair
    assert (
        frame_completion(
            request.run_dir,
            0,
            expected_shape=(5, 7),
            expected_spec=spec,
        ).is_complete
        is True
    )


def test_six_scope_run_manifest_and_completion_use_scope_products(
    tmp_path,
    monkeypatch,
) -> None:
    from et_mainsim.config import (
        EXECUTION_SCHEMA_ID,
        EXECUTION_SCHEMA_VERSION,
        ExecutionConfig,
        FullFrameWorkload,
        RunConfig,
        RunPaths,
    )
    from et_mainsim.workflows import full_frame
    from et_mainsim.workflows.full_frame import (
        FullFrameRunPlan,
        WorkerResult,
        run_full_frame,
    )
    from photsim7.spec_factories import make_et_main_detector_spec

    base = make_et_main_detector_spec(shape=(5, 7), run_seed=7)
    spec = replace(
        base,
        instrument=replace(base.instrument, telescope_count=6),
        psf=replace(base.psf, use_jitter_integrated_psf=False),
    )
    config = RunConfig(
        schema_id=EXECUTION_SCHEMA_ID,
        schema_version=EXECUTION_SCHEMA_VERSION,
        workflow="et-full-frame",
        run_id="six-scope",
        paths=RunPaths(
            output_root=str(tmp_path),
            data_root=str(tmp_path / "data"),
            catalog_cache=str(tmp_path / "stars.npz"),
        ),
        execution=ExecutionConfig(
            backend="in-process",
            device="cpu",
            resume=True,
            preview_count=0,
        ),
        workload=FullFrameWorkload(),
    )
    plan = FullFrameRunPlan(
        preset_name="six-scope-test",
        run_config=config,
        paths=config.resolve_paths(cwd=tmp_path),
        spec=spec,
        run_dir=tmp_path / "six-scope",
        catalog_cache=tmp_path / "stars.npz",
        frame_indices=(0,),
        repo_root=tmp_path,
    )

    monkeypatch.setattr(full_frame, "preflight", lambda _plan: None)
    monkeypatch.setattr(
        full_frame,
        "prepare_catalog",
        lambda _plan, **kwargs: SimpleNamespace(n_sources=1, metadata={}),
    )

    def fake_worker(request, **kwargs):
        for scope_id in range(6):
            _write_complete_frame(request.run_dir / f"scope_{scope_id}", 0)
            schema_path = (
                request.run_dir
                / f"scope_{scope_id}"
                / "frame_summaries"
                / "frame_000000_schema.json"
            )
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
            schema["provenance"] = {
                "services": {
                    "scope": {"scope_id": scope_id, "scope_count": 6}
                }
            }
            schema_path.write_text(json.dumps(schema), encoding="utf-8")
        return WorkerResult(rank=0, rendered=(0,), skipped=(), elapsed_s=0.0)

    monkeypatch.setattr(full_frame, "run_worker", fake_worker)

    result = run_full_frame(plan)

    assert result["status"] == "completed"
    artifacts = result["artifacts"]
    assert artifacts["layout"] == "per_scope_directories"
    assert artifacts["image_level_combination"] == "forbidden"
    assert "frames" not in artifacts
    assert artifacts["scopes"]["scope_0"]["frames"].endswith(
        "six-scope/scope_0/frames"
    )


def test_six_scope_manifest_namespaces_shared_exposure_artifacts(
    tmp_path,
    monkeypatch,
) -> None:
    """The coordinator advertises scope-local crop products, never a sum."""

    from et_mainsim.config import (
        EXECUTION_SCHEMA_ID,
        EXECUTION_SCHEMA_VERSION,
        ExecutionConfig,
        FullFrameWorkload,
        RunConfig,
        RunPaths,
        SharedExposureStampsConfig,
    )
    from et_mainsim.workflows import full_frame
    from et_mainsim.workflows.full_frame import (
        FullFrameRunPlan,
        WorkerResult,
        run_full_frame,
    )
    from photsim7.spec_factories import make_et_main_detector_spec

    base = make_et_main_detector_spec(shape=(5, 7), run_seed=7)
    spec = replace(
        base,
        instrument=replace(base.instrument, telescope_count=6),
        psf=replace(base.psf, use_jitter_integrated_psf=False),
    )
    shared = SharedExposureStampsConfig(
        enabled=True,
        target_source_ids=(11,),
        stamp_rows=5,
        stamp_cols=7,
        frames_per_shard=1,
    )
    config = RunConfig(
        schema_id=EXECUTION_SCHEMA_ID,
        schema_version=EXECUTION_SCHEMA_VERSION,
        workflow="et-full-frame",
        run_id="six-scope-shared",
        paths=RunPaths(
            output_root=str(tmp_path),
            data_root=str(tmp_path / "data"),
            catalog_cache=str(tmp_path / "stars.npz"),
        ),
        execution=ExecutionConfig(
            backend="in-process",
            device="cpu",
            resume=True,
            preview_count=0,
        ),
        workload=FullFrameWorkload(shared_exposure_stamps=shared),
    )
    plan = FullFrameRunPlan(
        preset_name="six-scope-shared-test",
        run_config=config,
        paths=config.resolve_paths(cwd=tmp_path),
        spec=spec,
        run_dir=tmp_path / "six-scope-shared",
        catalog_cache=tmp_path / "stars.npz",
        frame_indices=(0,),
        repo_root=tmp_path,
    )

    monkeypatch.setattr(full_frame, "preflight", lambda _plan: None)
    monkeypatch.setattr(
        full_frame,
        "prepare_catalog",
        lambda _plan, **kwargs: SimpleNamespace(n_sources=1, metadata={}),
    )

    def fake_worker(request, **kwargs):
        for scope_id in range(6):
            _write_complete_frame(request.run_dir / f"scope_{scope_id}", 0)
            schema_path = (
                request.run_dir
                / f"scope_{scope_id}"
                / "frame_summaries"
                / "frame_000000_schema.json"
            )
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
            schema["provenance"] = {
                "services": {
                    "scope": {"scope_id": scope_id, "scope_count": 6}
                }
            }
            schema_path.write_text(json.dumps(schema), encoding="utf-8")
        return WorkerResult(rank=0, rendered=(0,), skipped=(), elapsed_s=0.0)

    audit_requests = []
    monkeypatch.setattr(full_frame, "run_worker", fake_worker)
    monkeypatch.setattr(
        full_frame,
        "_shared_exposure_incomplete_frames_for_worker",
        lambda request, **kwargs: (audit_requests.append(request), ())[1],
    )

    result = run_full_frame(plan)

    assert result["status"] == "completed"
    assert len(audit_requests) == 1
    artifacts = result["artifacts"]["shared_exposure"]
    assert artifacts["layout"] == "per_scope_directories"
    assert artifacts["image_level_combination"] == "forbidden"
    assert artifacts["target_plan"].endswith("shared_exposure/target_plan.json")
    assert "completion_markers" not in artifacts
    assert "worker_shards" not in artifacts
    assert sorted(artifacts["scopes"]) == [f"scope_{scope_id}" for scope_id in range(6)]
    for scope_id in range(6):
        scope_artifacts = artifacts["scopes"][f"scope_{scope_id}"]
        assert scope_artifacts["root"].endswith(
            f"shared_exposure/scope_{scope_id}"
        )
        assert scope_artifacts["completion_markers"].endswith(
            f"shared_exposure/scope_{scope_id}/completion"
        )
        assert scope_artifacts["worker_shards"].endswith(
            f"shared_exposure/scope_{scope_id}/shards"
        )


def test_worker_delegates_rendering_to_photsim7_public_pipeline(tmp_path) -> None:
    from et_mainsim.config import ExecutionConfig
    from et_mainsim.workflows.full_frame import WorkerRequest, run_worker
    from photsim7.spec_factories import make_et_main_detector_spec

    calls: list[tuple[str, object]] = []
    base = _legacy_single_scope_spec(
        make_et_main_detector_spec(shape=(5, 7), run_seed=7)
    )
    spec = replace(
        base,
        psf=replace(base.psf, use_jitter_integrated_psf=False),
    )
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
    base = _legacy_single_scope_spec(
        make_et_main_detector_spec(shape=(5, 7), run_seed=7)
    )
    spec = replace(
        base,
        psf=replace(base.psf, use_jitter_integrated_psf=False),
    )
    request = WorkerRequest(
        spec=spec,
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


def test_worker_request_v2_round_trips_shared_exposure_contract(tmp_path) -> None:
    from et_mainsim.config import (
        ExecutionConfig,
        SharedExposureStampsConfig,
    )
    from et_mainsim.workflows.full_frame import WorkerRequest
    from photsim7.spec_factories import make_et_main_detector_spec

    shared = SharedExposureStampsConfig(
        enabled=True,
        target_source_ids=(17, 23),
        stamp_rows=100,
        stamp_cols=300,
        product_keys=("final_stamp", "electron_stamp"),
    )
    request = WorkerRequest(
        spec=_legacy_single_scope_spec(
            make_et_main_detector_spec(shape=(5, 7), run_seed=7)
        ),
        execution=ExecutionConfig(),
        run_dir=tmp_path / "run",
        data_root=tmp_path / "data",
        catalog_cache=tmp_path / "stars.npz",
        frame_indices=(0, 2),
        shared_exposure_stamps=shared,
        shared_exposure_overwrite_prepared=True,
        rank=1,
        world_size=2,
    )

    payload = request.to_json_dict()

    assert payload["schema_version"] == 2
    assert payload["shared_exposure_stamps"] == shared.to_dict()
    assert payload["shared_exposure_overwrite_prepared"] is True
    assert WorkerRequest.from_json_dict(payload) == request

    stale = dict(payload)
    stale["schema_version"] = 1
    with pytest.raises(ValueError, match="version"):
        WorkerRequest.from_json_dict(stale)


def test_direct_overwrite_worker_refuses_an_existing_shared_bundle(tmp_path) -> None:
    from et_mainsim.workflows.full_frame import run_worker

    request, api = _selection_ready_worker_request(tmp_path, n_frames=1)
    request = replace(
        _enable_shared_exposure(request),
        execution=replace(
            request.execution,
            resume=False,
            overwrite=True,
        ),
    )
    shared_root = request.run_dir / "shared_exposure"
    shared_root.mkdir(parents=True)
    (shared_root / "stale-partial.h5.partial").write_bytes(b"stale")
    unrelated = request.run_dir / "keep-me.txt"
    unrelated.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="coordinator.*shared-exposure"):
        run_worker(request, science_api=api)

    assert (shared_root / "stale-partial.h5.partial").read_bytes() == b"stale"
    assert unrelated.read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize("bundle_kind", ["directory", "dangling-symlink"])
def test_direct_fresh_worker_refuses_existing_shared_bundle_without_parent(
    tmp_path, bundle_kind
) -> None:
    from et_mainsim.workflows.full_frame import run_worker

    request, api = _selection_ready_worker_request(tmp_path, n_frames=1)
    request = _enable_shared_exposure(request)
    run_worker(request, science_api=api)
    for path in _artifact_paths_for_test(request.run_dir, 0):
        path.unlink()
    shutil.rmtree(request.run_dir / "selection_truth")
    shared_root = request.run_dir / "shared_exposure"
    if bundle_kind == "dangling-symlink":
        shutil.rmtree(shared_root)
        shared_root.symlink_to(
            request.run_dir / "missing-shared-exposure",
            target_is_directory=True,
        )

    fresh_request = replace(
        request,
        execution=replace(
            request.execution,
            resume=False,
            overwrite=False,
        ),
    )

    with pytest.raises(FileExistsError, match="resume or overwrite"):
        run_worker(fresh_request, science_api=api)


def test_shared_exposure_batches_preserve_rank_stride_order_and_full_hash(
    tmp_path,
) -> None:
    from et_mainsim.config import ExecutionConfig, SharedExposureStampsConfig
    from et_mainsim.workflows.full_frame import (
        WorkerRequest,
        _canonical_shared_exposure_batch_key,
        _shared_exposure_frame_batches,
    )
    from photsim7.spec_factories import make_et_main_detector_spec

    request = WorkerRequest(
        spec=_legacy_single_scope_spec(
            make_et_main_detector_spec(shape=(5, 7), run_seed=7)
        ),
        execution=ExecutionConfig(),
        run_dir=tmp_path / "run",
        data_root=tmp_path / "data",
        catalog_cache=tmp_path / "stars.npz",
        frame_indices=(9, 3, 8, 1, 7, 2),
        shared_exposure_stamps=SharedExposureStampsConfig(
            enabled=True,
            target_source_ids=(17,),
            frames_per_shard=2,
        ),
        rank=1,
        world_size=2,
    )
    assigned = request.frame_indices[request.rank :: request.world_size]

    batches = _shared_exposure_frame_batches(request, assigned)

    assert assigned == (3, 1, 2)
    assert [batch.frame_ids for batch in batches] == [(3, 1), (2,)]
    for batch in batches:
        payload, content_sha256 = _canonical_shared_exposure_batch_key(
            rank=request.rank,
            world_size=request.world_size,
            batch_index=batch.batch_index,
            frames_per_shard=2,
            frame_ids=batch.frame_ids,
        )
        assert payload == {
            "schema_id": "et_mainsim.shared_exposure_worker_batch_key.v1",
            "rank": 1,
            "world_size": 2,
            "batch_index": batch.batch_index,
            "frames_per_shard": 2,
            "frame_ids": list(batch.frame_ids),
        }
        assert len(content_sha256) == 64
        assert batch.content_sha256 == content_sha256
        assert batch.root.name == (f"batch_{batch.batch_index:06d}_{content_sha256}")


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


def _complete_full_frame_selection_api(api):
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

    original_build_services = api.build_full_frame_services

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
                "expected_manifest_sha256": (CANONICAL_JITTER_BANK_MANIFEST_SHA256),
                "verification_status": (
                    "array_and_manifest_sha256_verified_before_load"
                ),
                "loader": NATIVE_JITTER_BANK_LOADER_ID,
            },
            n_models=100,
            science_realization_id=(spec.science_profile.science_realization_id),
            spacecraft_id=services.context.spacecraft_id,
            absolute_raw_frame_start_index=(
                services.context.absolute_raw_frame_start_index
            ),
        )
        effects = EffectTimeseries(
            timing=build_frame_timing(
                n_frames=spec.observation.resolved_n_frames,
                integration_s=services.frame_exposure.to_value(u.s),
                sampling_interval_s=(spec.observation.sampling_interval.to_value(u.s)),
            ),
            source_geometry=services.source_geometry,
            jitter_integrated_psf_offsets=np.zeros((100, 2, 1)),
            jitter_model_selector=selector,
        )
        return replace(services, effect_timeseries=effects)

    api.build_full_frame_services = build_services
    return api


def _selection_ready_worker_request(tmp_path, *, n_frames: int = 2):
    from et_mainsim.config import ExecutionConfig
    from et_mainsim.workflows.full_frame import WorkerRequest, _science_api
    from photsim7.catalog_sources import PreparedStarCatalog, StarCatalogCache
    from photsim7.geometry_truth import reference_field_nonphysical_declaration
    from photsim7.spec_factories import make_et_main_detector_spec
    from photsim7.specs import (
        CosmicRaySpec,
        DetectorResponseSpec,
        DynamicEffectsSpec,
    )

    data_root = tmp_path / "data"
    bundle_name = _write_test_psf_bundle(data_root)
    bundle_sha256 = hashlib.sha256(
        (data_root / bundle_name / "sim_psf_images.pkl").read_bytes()
    ).hexdigest()
    base = _legacy_single_scope_spec(
        make_et_main_detector_spec(shape=(5, 7), run_seed=17)
    )
    spec = replace(
        base,
        observation=replace(
            base.observation,
            exposure_duration=1 * u.s,
            readout_duration=0 * u.s,
            observing_duration=n_frames * u.s,
            n_frames=n_frames,
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
            bundle_sha256=bundle_sha256,
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
        metadata={
            "source": {"type": "prepared", "n_sources": 1},
            "geometry": reference_field_nonphysical_declaration(
                reference_field_angle_deg=0.0,
                reference_pixel_scale_arcsec_per_pix=4.83,
            ),
        },
    )
    cache_path = tmp_path / "stars.npz"
    StarCatalogCache.write(cache_path, catalog)
    request = WorkerRequest(
        spec=spec,
        execution=ExecutionConfig(
            backend="in-process",
            device="cpu",
            resume=True,
            preview_count=0,
        ),
        run_dir=tmp_path / "run",
        data_root=data_root,
        catalog_cache=cache_path,
        frame_indices=tuple(range(n_frames)),
        rank=0,
        world_size=1,
    )
    api = _complete_full_frame_selection_api(_science_api())
    return request, api


def _enable_shared_exposure(
    request,
    *,
    target_source_ids=(11,),
    product_keys=("final_stamp",),
    frames_per_shard=32,
):
    from et_mainsim.config import SharedExposureStampsConfig

    return replace(
        request,
        shared_exposure_stamps=SharedExposureStampsConfig(
            enabled=True,
            target_source_ids=target_source_ids,
            stamp_rows=100,
            stamp_cols=300,
            frames_per_shard=frames_per_shard,
            product_keys=product_keys,
        ),
    )


def _shared_final_shard_path(request, *, product_key="final_stamp") -> Path:
    from et_mainsim.shared_exposure import (
        read_shared_exposure_target_plan,
        shared_exposure_product_shard_path,
    )

    root = request.run_dir / "shared_exposure"
    plan = read_shared_exposure_target_plan(root / "target_plan.json")
    legacy_path = shared_exposure_product_shard_path(
        root / "shards" / f"worker_{request.rank:04d}",
        plan_content_sha256=plan["content_sha256"],
        product_key=product_key,
    )
    batch_matches = list(
        legacy_path.parent.parent.parent.glob(
            f"batch_??????_*/shared-exposure-products/"
            f"{plan['content_sha256']}/{legacy_path.name}"
        )
    )
    if len(batch_matches) == 1:
        return batch_matches[0]
    if batch_matches:
        raise AssertionError(
            f"expected one {product_key} shard, found {len(batch_matches)}"
        )
    return legacy_path


@pytest.mark.parametrize("telescope_count", (1, 6))
def test_tiny_cpu_worker_writes_readable_photsim7_artifacts(
    tmp_path,
    telescope_count: int,
) -> None:
    from et_mainsim.config import ExecutionConfig
    from et_mainsim.workflows.full_frame import (
        WorkerRequest,
        frame_completion,
        run_worker,
    )
    from photsim7.catalog_sources import PreparedStarCatalog, StarCatalogCache
    from photsim7.frame_products import read_frame_product_schema
    from photsim7.geometry_truth import reference_field_nonphysical_declaration
    from photsim7.spec_factories import make_et_main_detector_spec
    from photsim7.specs import (
        CosmicRaySpec,
        DetectorResponseSpec,
        DynamicEffectsSpec,
    )

    data_root = tmp_path / "data"
    bundle_name = _write_test_psf_bundle(data_root)
    bundle_sha256 = hashlib.sha256(
        (data_root / bundle_name / "sim_psf_images.pkl").read_bytes()
    ).hexdigest()
    base = make_et_main_detector_spec(shape=(5, 7), run_seed=17)
    spec = replace(
        base,
        instrument=replace(base.instrument, telescope_count=telescope_count),
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
            bundle_sha256=bundle_sha256,
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
        metadata={
            "source": {"type": "prepared", "n_sources": 1},
            "geometry": reference_field_nonphysical_declaration(
                reference_field_angle_deg=0.0,
                reference_pixel_scale_arcsec_per_pix=4.83,
            ),
        },
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

    assert result.rendered == (0,)
    scope_roots = (
        (run_dir,)
        if telescope_count == 1
        else tuple(run_dir / f"scope_{scope_id}" for scope_id in range(6))
    )
    if telescope_count == 6:
        assert not (run_dir / "frames" / "frame_000000.npy").exists()
    for scope_id, scope_root in enumerate(scope_roots):
        frame = np.load(scope_root / "frames" / "frame_000000.npy")
        schema = read_frame_product_schema(
            scope_root / "frame_summaries" / "frame_000000_schema.json"
        )
        summary = json.loads(
            (scope_root / "frame_summaries" / "frame_000000.json").read_text(
                encoding="utf-8"
            )
        )
        assert frame.shape == (5, 7)
        assert frame.dtype == np.uint16
        assert schema["arrays"]["final_frame"]["domain"] == "dn"
        if telescope_count == 6:
            assert schema["provenance"]["services"]["scope"]["scope_id"] == scope_id
        assert summary["et_mainsim"]["rank"] == 0
        assert summary["et_mainsim"]["n_stars"] == 1
        assert summary["et_mainsim"]["pipeline_elapsed_s"] >= 0.0
    assert frame_completion(
        run_dir,
        0,
        expected_shape=(5, 7),
        expected_spec=spec,
    ).is_complete


def test_exposure_first_worker_renders_once_and_persists_parent_crops(
    tmp_path,
) -> None:
    from et_mainsim.config import SharedExposureStampsConfig
    from et_mainsim.shared_exposure import (
        read_shared_exposure_frame_completion,
        read_shared_exposure_target_plan,
    )
    from et_mainsim.workflows.full_frame import run_worker
    from photsim7.artifacts import SharedExposureShardReader

    request, api = _selection_ready_worker_request(tmp_path, n_frames=2)
    request = replace(
        request,
        shared_exposure_stamps=SharedExposureStampsConfig(
            enabled=True,
            target_source_ids=(11,),
            stamp_rows=100,
            stamp_cols=300,
            product_keys=("final_stamp",),
        ),
    )
    calls = 0
    run_frame = api.run_single_cadence_full_frame

    def counted_run_frame(*args, **kwargs):
        nonlocal calls
        calls += 1
        return run_frame(*args, **kwargs)

    api.run_single_cadence_full_frame = counted_run_frame

    result = run_worker(request, science_api=api)

    assert result.rendered == (0, 1)
    assert calls == 2
    shared_root = request.run_dir / "shared_exposure"
    plan_path = shared_root / "target_plan.json"
    plan = read_shared_exposure_target_plan(plan_path)
    assert [target["source_id"] for target in plan["targets"]] == [11]
    assert plan["stamp_shape"] == [100, 300]
    shard_path = _shared_final_shard_path(request)
    with SharedExposureShardReader(shard_path) as reader:
        assert reader.target_source_ids == (11,)
        assert reader.frame_ids == (0, 1)
        for frame_index in (0, 1):
            stamp = reader.read_array(11, frame_index)
            parent = np.load(
                request.run_dir / "frames" / f"frame_{frame_index:06d}.npy"
            )
            assert stamp.shape == (100, 300)
            assert np.array_equal(stamp[48:53, 147:154], parent)
            expected = np.zeros_like(stamp)
            expected[48:53, 147:154] = parent
            assert np.array_equal(stamp, expected)
            marker = read_shared_exposure_frame_completion(
                shared_root / "completion" / f"frame_{frame_index:09d}.json",
                reference_root=request.run_dir,
            )
            assert marker["mode"] == "parent_rendered_this_attempt"
            assert marker["upstream_negative_claims"]["lineage_claimed"] is False


def test_six_scope_shared_exposure_crops_each_scope_and_resumes(tmp_path) -> None:
    """Six scopes own independent parents and scope-local shared crops.

    The contract must never create a root-level image or a shared shard whose
    identity could be mistaken for an image-level sum.  A resumed run is only
    complete when every scope parent, crop shard, and completion marker remains
    valid.
    """

    from et_mainsim.config import SharedExposureStampsConfig
    from et_mainsim.shared_exposure import (
        read_shared_exposure_frame_completion,
        read_shared_exposure_target_plan,
    )
    from et_mainsim.workflows.full_frame import (
        _shared_exposure_incomplete_frames_for_worker,
        run_worker,
    )
    from photsim7.artifacts import SharedExposureShardReader

    request, api = _selection_ready_worker_request(tmp_path, n_frames=1)
    request = replace(
        request,
        spec=replace(
            request.spec,
            instrument=replace(request.spec.instrument, telescope_count=6),
        ),
        shared_exposure_stamps=SharedExposureStampsConfig(
            enabled=True,
            target_source_ids=(11,),
            stamp_rows=5,
            stamp_cols=7,
            frames_per_shard=1,
            product_keys=("final_stamp",),
        ),
    )

    first = run_worker(request, science_api=api)

    assert first.rendered == (0,)
    assert not (request.run_dir / "frames" / "frame_000000.npy").exists()
    assert not (
        request.run_dir
        / "shared_exposure"
        / "completion"
        / "frame_000000000.json"
    ).exists()
    plan = read_shared_exposure_target_plan(
        request.run_dir / "shared_exposure" / "target_plan.json"
    )
    marker_bytes: dict[int, bytes] = {}
    for scope_id in range(6):
        scope_root = request.run_dir / f"scope_{scope_id}"
        parent = np.load(scope_root / "frames" / "frame_000000.npy")
        marker_path = (
            request.run_dir
            / "shared_exposure"
            / f"scope_{scope_id}"
            / "completion"
            / "frame_000000000.json"
        )
        marker = read_shared_exposure_frame_completion(
            marker_path,
            reference_root=request.run_dir,
        )
        marker_bytes[scope_id] = marker_path.read_bytes()
        assert marker["parent_storage_guard"]["path"] == (
            f"scope_{scope_id}/frames/frame_000000.npy"
        )
        shard_matches = list(
            (
                request.run_dir
                / "shared_exposure"
                / f"scope_{scope_id}"
                / "shards"
                / "worker_0000"
            ).glob(
                "batch_??????_*/shared-exposure-products/"
                f"{plan['content_sha256']}/*.h5"
            )
        )
        assert len(shard_matches) == 1
        with SharedExposureShardReader(shard_matches[0]) as reader:
            assert reader.provenance["scope_id"] == scope_id
            assert np.array_equal(reader.read_array(11, 0), parent)

    second = run_worker(request, science_api=api)

    assert second.rendered == ()
    assert second.skipped == (0,)
    for scope_id in range(6):
        marker_path = (
            request.run_dir
            / "shared_exposure"
            / f"scope_{scope_id}"
            / "completion"
            / "frame_000000000.json"
        )
        assert marker_path.read_bytes() == marker_bytes[scope_id]

    # The coordinator's post-worker audit must use the same per-scope
    # completion witnesses as the worker.  A root-level marker is forbidden
    # for a six-scope exposure and must not be mistaken for a missing crop.
    assert (
        _shared_exposure_incomplete_frames_for_worker(request, science_api=api)
        == ()
    )


def test_six_scope_shared_exposure_keeps_two_frame_shards_scope_local(tmp_path) -> None:
    """A multi-frame shard carries only one scope's independent parents."""

    from et_mainsim.config import SharedExposureStampsConfig
    from et_mainsim.shared_exposure import read_shared_exposure_target_plan
    from et_mainsim.workflows.full_frame import run_worker
    from photsim7.artifacts import SharedExposureShardReader

    request, api = _selection_ready_worker_request(tmp_path, n_frames=2)
    request = replace(
        request,
        spec=replace(
            request.spec,
            instrument=replace(request.spec.instrument, telescope_count=6),
        ),
        shared_exposure_stamps=SharedExposureStampsConfig(
            enabled=True,
            target_source_ids=(11,),
            stamp_rows=5,
            stamp_cols=7,
            frames_per_shard=2,
            product_keys=("final_stamp",),
        ),
    )

    first = run_worker(request, science_api=api)

    assert first.rendered == (0, 1)
    plan = read_shared_exposure_target_plan(
        request.run_dir / "shared_exposure" / "target_plan.json"
    )
    assert not (request.run_dir / "shared_exposure" / "shards").exists()
    for scope_id in range(6):
        scope_root = request.run_dir / f"scope_{scope_id}"
        shard_matches = list(
            (
                request.run_dir
                / "shared_exposure"
                / f"scope_{scope_id}"
                / "shards"
                / "worker_0000"
            ).glob(
                "batch_??????_*/shared-exposure-products/"
                f"{plan['content_sha256']}/*.h5"
            )
        )
        assert len(shard_matches) == 1
        with SharedExposureShardReader(shard_matches[0]) as reader:
            assert reader.frame_ids == (0, 1)
            assert reader.provenance["scope_id"] == scope_id
            for frame_index in (0, 1):
                parent = np.load(
                    scope_root / "frames" / f"frame_{frame_index:06d}.npy"
                )
                assert np.array_equal(reader.read_array(11, frame_index), parent)
                assert (
                    request.run_dir
                    / "shared_exposure"
                    / f"scope_{scope_id}"
                    / "completion"
                    / f"frame_{frame_index:09d}.json"
                ).is_file()

    second = run_worker(request, science_api=api)

    assert second.rendered == ()
    assert second.skipped == (0, 1)


def test_exposure_first_three_targets_two_products_still_render_once_per_frame(
    tmp_path,
) -> None:
    from et_mainsim.workflows.full_frame import run_worker
    from photsim7.artifacts import SharedExposureShardReader

    request, api = _selection_ready_worker_request(tmp_path, n_frames=2)
    original = api.StarCatalogCache.read(request.catalog_cache)
    expanded = api.PreparedStarCatalog(
        star_data={
            "x0": np.array([-2.0, 0.0, 2.0]),
            "y0": np.array([-1.0, 0.0, 1.0]),
            "ra": np.array([9.9, 10.0, 10.1]),
            "dec": np.array([19.9, 20.0, 20.1]),
            "source_id": np.array([11, 12, 13], dtype=np.int64),
            "et_mag": np.array([12.0, 12.5, 13.0]),
            "frame_xpix": np.array([1.0, 3.0, 5.0]),
            "frame_ypix": np.array([1.0, 2.0, 3.0]),
            "detector_xpix_shifted": np.array([1.0, 3.0, 5.0]),
            "detector_ypix_shifted": np.array([1.0, 2.0, 3.0]),
        },
        metadata=dict(original.metadata),
    )
    request.catalog_cache.unlink()
    api.StarCatalogCache.write(request.catalog_cache, expanded)
    request = _enable_shared_exposure(
        request,
        target_source_ids=(11, 12, 13),
        product_keys=("final_stamp", "electron_stamp"),
    )
    calls = 0
    run_frame = api.run_single_cadence_full_frame

    def counted_run_frame(*args, **kwargs):
        nonlocal calls
        calls += 1
        return run_frame(*args, **kwargs)

    api.run_single_cadence_full_frame = counted_run_frame

    result = run_worker(request, science_api=api)

    assert result.rendered == (0, 1)
    assert calls == 2
    for product_key in ("final_stamp", "electron_stamp"):
        with SharedExposureShardReader(
            _shared_final_shard_path(request, product_key=product_key)
        ) as reader:
            assert reader.target_source_ids == (11, 12, 13)
            assert reader.frame_ids == (0, 1)
            assert reader.product_key == product_key
            for target_source_id in reader.target_source_ids:
                for frame_index in reader.frame_ids:
                    assert reader.read_array(target_source_id, frame_index).shape == (
                        100,
                        300,
                    )


def test_exposure_first_refuses_marker_after_finalized_shard_byte_drift(
    tmp_path,
) -> None:
    import h5py

    from et_mainsim.shared_exposure import SharedExposureArrayMismatchError
    from et_mainsim.workflows.full_frame import run_worker

    request, api = _selection_ready_worker_request(tmp_path, n_frames=2)
    request = _enable_shared_exposure(request)
    writer_type = api.SharedExposureShardWriter

    class CorruptAfterFinalizeWriter:
        def __init__(self, *args, **kwargs):
            self._writer = writer_type(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._writer, name)

        def finalize(self):
            self._writer.finalize()
            with h5py.File(self._writer.final_path, "r+") as handle:
                handle["images"][0, 0, 0, 0] = np.uint16(
                    handle["images"][0, 0, 0, 0] + 1
                )

        def close(self):
            self._writer.close()

    api.SharedExposureShardWriter = CorruptAfterFinalizeWriter

    with pytest.raises(SharedExposureArrayMismatchError, match="C-order bytes"):
        run_worker(request, science_api=api)

    marker = request.run_dir / "shared_exposure/completion/frame_000000000.json"
    assert _shared_final_shard_path(request).is_file()
    assert not marker.exists()


def test_exposure_first_two_workers_publish_disjoint_shards_and_resume(
    tmp_path,
) -> None:
    from et_mainsim.workflows.full_frame import run_worker
    from photsim7.artifacts import SharedExposureShardReader

    base, api = _selection_ready_worker_request(tmp_path, n_frames=4)
    base = _enable_shared_exposure(base)
    rank0 = replace(base, rank=0, world_size=2)
    rank1 = replace(base, rank=1, world_size=2)

    first0 = run_worker(rank0, science_api=api)
    first1 = run_worker(rank1, science_api=api)

    assert first0.rendered == (0, 2)
    assert first1.rendered == (1, 3)
    with SharedExposureShardReader(_shared_final_shard_path(rank0)) as reader:
        assert reader.frame_ids == (0, 2)
    with SharedExposureShardReader(_shared_final_shard_path(rank1)) as reader:
        assert reader.frame_ids == (1, 3)

    class ForbiddenCache:
        @staticmethod
        def read(path):
            raise AssertionError(f"complete worker resume must not load catalog {path}")

    api.StarCatalogCache = ForbiddenCache
    resumed0 = run_worker(rank0, science_api=api)
    resumed1 = run_worker(rank1, science_api=api)

    assert resumed0.rendered == ()
    assert resumed0.skipped == (0, 2)
    assert resumed1.rendered == ()
    assert resumed1.skipped == (1, 3)


def test_exposure_first_batch_boundary_is_a_durable_bounded_replay_checkpoint(
    tmp_path,
) -> None:
    from et_mainsim.workflows.full_frame import run_worker
    from photsim7.artifacts import SharedExposureShardReader

    request, api = _selection_ready_worker_request(tmp_path, n_frames=5)
    request = _enable_shared_exposure(request, frames_per_shard=2)
    run_frame = api.run_single_cadence_full_frame

    def fail_in_last_batch(*args, **kwargs):
        if kwargs["frame_index"] == 4:
            raise RuntimeError("injected final-batch interruption")
        return run_frame(*args, **kwargs)

    api.run_single_cadence_full_frame = fail_in_last_batch

    with pytest.raises(RuntimeError, match="final-batch interruption"):
        run_worker(request, science_api=api)

    worker_root = request.run_dir / "shared_exposure/shards/worker_0000"
    published_batches = sorted(worker_root.glob("batch_??????_*"))
    assert [path.name.split("_", 2)[:2] for path in published_batches] == [
        ["batch", "000000"],
        ["batch", "000001"],
    ]
    for batch_index, frame_ids in ((0, (0, 1)), (1, (2, 3))):
        final_paths = list(published_batches[batch_index].rglob("*.h5"))
        assert len(final_paths) == 1
        with SharedExposureShardReader(final_paths[0]) as reader:
            assert reader.frame_ids == frame_ids
        for frame_index in frame_ids:
            assert (
                request.run_dir
                / "shared_exposure/completion"
                / f"frame_{frame_index:09d}.json"
            ).is_file()
    assert not (
        request.run_dir / "shared_exposure/completion/frame_000000004.json"
    ).exists()

    calls: list[int] = []

    def counted_run_frame(*args, **kwargs):
        calls.append(kwargs["frame_index"])
        return run_frame(*args, **kwargs)

    api.run_single_cadence_full_frame = counted_run_frame
    result = run_worker(request, science_api=api)

    assert calls == [4]
    assert result.rendered == (4,)
    assert result.skipped == (0, 1, 2, 3)
    completed_batches = sorted(worker_root.glob("batch_??????_*"))
    assert len(completed_batches) == 3
    final_paths = list(completed_batches[2].rglob("*.h5"))
    assert len(final_paths) == 1
    with SharedExposureShardReader(final_paths[0]) as reader:
        assert reader.frame_ids == (4,)


def test_exposure_first_releases_completed_batch_crop_payloads(
    tmp_path,
) -> None:
    import gc
    import weakref

    from et_mainsim.workflows.full_frame import run_worker

    request, api = _selection_ready_worker_request(tmp_path, n_frames=3)
    request = _enable_shared_exposure(request, frames_per_shard=1)
    run_frame = api.run_single_cadence_full_frame
    crop_frame = api.shared_exposure_crop_v1
    crop_refs = []
    array_refs = []

    def tracked_crop(*args, **kwargs):
        assert kwargs["product_keys"] == request.shared_exposure_stamps.product_keys
        assert kwargs["materialize_numpy"] is True
        crop = crop_frame(*args, **kwargs)
        crop_refs.append(weakref.ref(crop))
        array_refs.append(weakref.ref(crop.final_stamp.array))
        return crop

    def inspect_before_third_batch(*args, **kwargs):
        if kwargs["frame_index"] == 2:
            gc.collect()
            assert len(crop_refs) == 2
            assert crop_refs[0]() is None
            assert array_refs[0]() is None
        return run_frame(*args, **kwargs)

    api.shared_exposure_crop_v1 = tracked_crop
    api.run_single_cadence_full_frame = inspect_before_third_batch

    result = run_worker(request, science_api=api)

    assert result.rendered == (0, 1, 2)


def test_exposure_first_resume_keeps_only_one_batch_snapshot_resident(
    tmp_path,
    monkeypatch,
) -> None:
    import gc
    import weakref

    from et_mainsim.workflows import full_frame
    from et_mainsim.workflows.full_frame import (
        _shared_exposure_incomplete_frames_for_worker,
        run_worker,
    )

    request, api = _selection_ready_worker_request(tmp_path, n_frames=3)
    request = _enable_shared_exposure(request, frames_per_shard=1)
    run_worker(request, science_api=api)
    for marker in (request.run_dir / "shared_exposure/completion").glob("*.json"):
        marker.unlink()

    inspect_shards = full_frame._inspect_shared_exposure_shards
    snapshot_refs: list[tuple[int, weakref.ReferenceType[dict]]] = []
    resident_batch_counts: list[int] = []

    class TrackedSnapshots(dict):
        pass

    def tracked_inspection(*args, **kwargs):
        gc.collect()
        resident_batch_counts.append(
            len(
                {
                    batch_index
                    for batch_index, snapshot_ref in snapshot_refs
                    if snapshot_ref() is not None
                }
            )
        )
        snapshots = TrackedSnapshots(inspect_shards(*args, **kwargs))
        snapshot_refs.append((kwargs["batch"].batch_index, weakref.ref(snapshots)))
        return snapshots

    monkeypatch.setattr(
        full_frame,
        "_inspect_shared_exposure_shards",
        tracked_inspection,
    )

    result = run_worker(request, science_api=api)

    assert result.rendered == (0, 1, 2)
    assert resident_batch_counts
    assert max(resident_batch_counts) <= 1

    snapshot_refs.clear()
    resident_batch_counts.clear()
    assert _shared_exposure_incomplete_frames_for_worker(request, science_api=api) == ()
    assert resident_batch_counts
    assert max(resident_batch_counts) <= 1


@pytest.mark.parametrize("mutated_artifact", ["parent", "final_shard"])
def test_exposure_first_exact_readback_follows_completion_guard_build(
    tmp_path,
    monkeypatch,
    mutated_artifact,
) -> None:
    import h5py

    from et_mainsim import shared_exposure
    from et_mainsim.shared_exposure import SharedExposureArrayMismatchError
    from et_mainsim.workflows.full_frame import run_worker

    request, api = _selection_ready_worker_request(tmp_path, n_frames=1)
    request = _enable_shared_exposure(request)
    build_completion = shared_exposure.build_shared_exposure_frame_completion

    def mutate_before_guard_build(*args, **kwargs):
        if mutated_artifact == "parent":
            parent_path = Path(kwargs["parent_path"])
            parent = np.load(parent_path, allow_pickle=False)
            parent.flat[0] = np.asarray(parent.flat[0] ^ 1, dtype=parent.dtype)
            np.save(parent_path, parent)
        else:
            shard_path = Path(kwargs["product_shards"]["final_stamp"])
            with h5py.File(shard_path, "r+") as handle:
                handle["images"][0, 0, 0, 0] = np.asarray(
                    handle["images"][0, 0, 0, 0] ^ 1,
                    dtype=handle["images"].dtype,
                )
        return build_completion(*args, **kwargs)

    monkeypatch.setattr(
        shared_exposure,
        "build_shared_exposure_frame_completion",
        mutate_before_guard_build,
    )

    with pytest.raises(SharedExposureArrayMismatchError, match="C-order bytes"):
        run_worker(request, science_api=api)

    assert not (
        request.run_dir / "shared_exposure/completion/frame_000000000.json"
    ).exists()


def test_exposure_first_guard_order_protects_skipped_sibling_items(
    tmp_path,
    monkeypatch,
) -> None:
    import h5py

    from et_mainsim import shared_exposure
    from et_mainsim.shared_exposure import SharedExposureArrayMismatchError
    from et_mainsim.workflows.full_frame import run_worker

    request, api = _selection_ready_worker_request(tmp_path, n_frames=2)
    request = _enable_shared_exposure(request)
    run_worker(request, science_api=api)
    repaired_marker = (
        request.run_dir / "shared_exposure/completion/frame_000000001.json"
    )
    repaired_marker.unlink()
    build_completion = shared_exposure.build_shared_exposure_frame_completion

    def mutate_skipped_sibling_before_guard_build(*args, **kwargs):
        shard_path = Path(kwargs["product_shards"]["final_stamp"])
        with h5py.File(shard_path, "r+") as handle:
            handle["images"][0, 0, 0, 0] = np.asarray(
                handle["images"][0, 0, 0, 0] ^ 1,
                dtype=handle["images"].dtype,
            )
        return build_completion(*args, **kwargs)

    monkeypatch.setattr(
        shared_exposure,
        "build_shared_exposure_frame_completion",
        mutate_skipped_sibling_before_guard_build,
    )

    with pytest.raises(SharedExposureArrayMismatchError, match="C-order bytes"):
        run_worker(request, science_api=api)

    assert not repaired_marker.exists()


def test_exposure_first_closes_every_writer_without_masking_primary_failure(
    tmp_path,
) -> None:
    from et_mainsim.workflows.full_frame import run_worker

    request, api = _selection_ready_worker_request(tmp_path, n_frames=1)
    request = _enable_shared_exposure(
        request,
        product_keys=("electron_stamp", "final_stamp"),
    )
    writer_type = api.SharedExposureShardWriter
    close_attempts = []

    class FailingWriter:
        def __init__(self, *args, **kwargs):
            self.product_key = kwargs["product_key"]
            self._writer = writer_type(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._writer, name)

        def write_crop(self, crop):
            if self.product_key == "final_stamp":
                raise RuntimeError("injected primary write failure")
            return self._writer.write_crop(crop)

        def close(self):
            close_attempts.append(self.product_key)
            self._writer.close()
            if self.product_key == "electron_stamp":
                raise RuntimeError("injected writer close failure")

    api.SharedExposureShardWriter = FailingWriter

    with pytest.raises(RuntimeError, match="injected primary write failure") as caught:
        run_worker(request, science_api=api)

    assert close_attempts == ["electron_stamp", "final_stamp"]
    assert any(
        "injected writer close failure" in note
        for note in getattr(caught.value, "__notes__", ())
    )


def test_exposure_first_resume_cleans_linked_partial_publication(
    tmp_path,
) -> None:
    from et_mainsim.workflows.full_frame import run_worker
    from photsim7.artifacts import partial_shard_path

    request, api = _selection_ready_worker_request(tmp_path, n_frames=1)
    request = _enable_shared_exposure(request)
    run_worker(request, science_api=api)
    shard = _shared_final_shard_path(request)
    partial = partial_shard_path(shard)
    partial.hardlink_to(shard)
    assert partial.samefile(shard)

    result = run_worker(request, science_api=api)

    assert result.rendered == ()
    assert result.skipped == (0,)
    assert shard.is_file()
    assert not partial.exists()


def test_exposure_first_validates_marker_before_cleaning_linked_partial(
    tmp_path,
) -> None:
    from et_mainsim.shared_exposure import SharedExposureContractError
    from et_mainsim.workflows.full_frame import run_worker
    from photsim7.artifacts import partial_shard_path

    request, api = _selection_ready_worker_request(tmp_path, n_frames=1)
    request = _enable_shared_exposure(request)
    run_worker(request, science_api=api)
    shard = _shared_final_shard_path(request)
    partial = partial_shard_path(shard)
    partial.hardlink_to(shard)
    marker = request.run_dir / "shared_exposure/completion/frame_000000000.json"
    marker.write_bytes(b"{")

    with pytest.raises(SharedExposureContractError, match="UTF-8 JSON object"):
        run_worker(request, science_api=api)

    assert partial.exists()
    assert partial.samefile(shard)


def test_exposure_first_resume_rebuilds_missing_marker_by_exact_reconstruction(
    tmp_path,
) -> None:
    from et_mainsim.shared_exposure import read_shared_exposure_frame_completion
    from et_mainsim.workflows.full_frame import run_worker

    request, api = _selection_ready_worker_request(tmp_path, n_frames=2)
    request = _enable_shared_exposure(request)
    run_worker(request, science_api=api)
    marker = request.run_dir / "shared_exposure" / "completion" / "frame_000000000.json"
    marker.unlink()
    calls: list[object] = []
    run_frame = api.run_single_cadence_full_frame

    def counted_run_frame(*args, **kwargs):
        calls.append(kwargs.get("artifact_writer"))
        return run_frame(*args, **kwargs)

    api.run_single_cadence_full_frame = counted_run_frame

    result = run_worker(request, science_api=api)

    assert result.rendered == (0,)
    assert result.skipped == (1,)
    assert calls == [None]
    assert marker.is_file()
    repaired = read_shared_exposure_frame_completion(
        marker,
        reference_root=request.run_dir,
    )
    assert repaired["mode"] == "deterministic_parent_reconstruction"


def test_exposure_first_missing_marker_rejects_tampered_final_shard(
    tmp_path,
) -> None:
    import h5py

    from et_mainsim.shared_exposure import SharedExposureArrayMismatchError
    from et_mainsim.workflows.full_frame import run_worker

    request, api = _selection_ready_worker_request(tmp_path, n_frames=1)
    request = _enable_shared_exposure(request)
    run_worker(request, science_api=api)
    marker = request.run_dir / "shared_exposure/completion/frame_000000000.json"
    marker.unlink()
    shard = _shared_final_shard_path(request)
    with h5py.File(shard, "r+") as handle:
        handle["images"][0, 0, 0, 0] = np.uint16(handle["images"][0, 0, 0, 0] + 1)

    with pytest.raises(SharedExposureArrayMismatchError, match="C-order bytes"):
        run_worker(request, science_api=api)

    assert shard.is_file()
    assert not marker.exists()


def test_exposure_first_missing_marker_rejects_tampered_parent_with_final_shard(
    tmp_path,
) -> None:
    from et_mainsim.shared_exposure import SharedExposureArrayMismatchError
    from et_mainsim.workflows.full_frame import run_worker

    request, api = _selection_ready_worker_request(tmp_path, n_frames=1)
    request = _enable_shared_exposure(request)
    run_worker(request, science_api=api)
    marker = request.run_dir / "shared_exposure/completion/frame_000000000.json"
    marker.unlink()
    parent_path = request.run_dir / "frames/frame_000000.npy"
    parent = np.load(parent_path)
    parent[0, 0] = np.asarray(parent[0, 0] + 1, dtype=parent.dtype)
    np.save(parent_path, parent)

    with pytest.raises(SharedExposureArrayMismatchError, match="C-order bytes"):
        run_worker(request, science_api=api)

    assert _shared_final_shard_path(request).is_file()
    assert not marker.exists()


def test_exposure_first_resume_reconstructs_parent_without_rewriting_it(
    tmp_path,
) -> None:
    from et_mainsim.workflows.full_frame import run_worker

    request, api = _selection_ready_worker_request(tmp_path, n_frames=1)
    request = _enable_shared_exposure(request)
    run_worker(request, science_api=api)
    parent_paths = _artifact_paths_for_test(request.run_dir, 0)
    before = {
        path: (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
        for path in parent_paths
    }
    marker = request.run_dir / "shared_exposure/completion/frame_000000000.json"
    shard = _shared_final_shard_path(request)
    marker.unlink()
    shard.unlink()

    calls: list[object] = []
    run_frame = api.run_single_cadence_full_frame

    def counted_run_frame(*args, **kwargs):
        calls.append(kwargs.get("artifact_writer"))
        return run_frame(*args, **kwargs)

    api.run_single_cadence_full_frame = counted_run_frame

    result = run_worker(request, science_api=api)

    assert result.rendered == (0,)
    assert result.skipped == ()
    assert calls == [None]
    assert marker.is_file()
    assert shard.is_file()
    assert {
        path: (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
        for path in parent_paths
    } == before
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["mode"] == "deterministic_parent_reconstruction"


def test_exposure_first_recovery_fails_before_crop_on_parent_byte_drift(
    tmp_path,
) -> None:
    from et_mainsim.shared_exposure import SharedExposureArrayMismatchError
    from et_mainsim.workflows.full_frame import run_worker

    request, api = _selection_ready_worker_request(tmp_path, n_frames=1)
    request = _enable_shared_exposure(request)
    run_worker(request, science_api=api)
    marker = request.run_dir / "shared_exposure/completion/frame_000000000.json"
    shard = _shared_final_shard_path(request)
    marker.unlink()
    shard.unlink()
    parent_path = request.run_dir / "frames/frame_000000.npy"
    parent = np.load(parent_path)
    parent[0, 0] = np.asarray(parent[0, 0] + 1, dtype=parent.dtype)
    np.save(parent_path, parent)

    with pytest.raises(SharedExposureArrayMismatchError, match="C-order bytes"):
        run_worker(request, science_api=api)

    assert not shard.exists()
    assert not marker.exists()


def test_exposure_first_refuses_complete_crop_when_parent_is_missing(
    tmp_path,
) -> None:
    from et_mainsim.shared_exposure import SharedExposureReferenceDriftError
    from et_mainsim.workflows.full_frame import run_worker

    request, api = _selection_ready_worker_request(tmp_path, n_frames=1)
    request = _enable_shared_exposure(request)
    run_worker(request, science_api=api)
    marker = request.run_dir / "shared_exposure/completion/frame_000000000.json"
    marker.unlink()
    (request.run_dir / "frames/frame_000000.npy").unlink()
    api.run_single_cadence_full_frame = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("missing parent with complete crop must fail before rendering")
    )

    with pytest.raises(SharedExposureReferenceDriftError, match="parent"):
        run_worker(request, science_api=api)


def test_exposure_first_resume_publishes_complete_partial_after_exact_reconstruction(
    tmp_path,
) -> None:
    from et_mainsim.shared_exposure import read_shared_exposure_frame_completion
    from et_mainsim.workflows.full_frame import run_worker
    from photsim7.artifacts import partial_shard_path

    request, api = _selection_ready_worker_request(tmp_path, n_frames=2)
    request = _enable_shared_exposure(request)
    run_worker(request, science_api=api)
    shard = _shared_final_shard_path(request)
    partial = partial_shard_path(shard)
    shard.rename(partial)
    for frame_index in request.frame_indices:
        (
            request.run_dir
            / "shared_exposure"
            / "completion"
            / f"frame_{frame_index:09d}.json"
        ).unlink()

    calls: list[object] = []
    run_frame = api.run_single_cadence_full_frame

    def counted_run_frame(*args, **kwargs):
        calls.append(kwargs.get("artifact_writer"))
        return run_frame(*args, **kwargs)

    api.run_single_cadence_full_frame = counted_run_frame

    result = run_worker(request, science_api=api)

    assert result.rendered == (0, 1)
    assert result.skipped == ()
    assert calls == [None, None]
    assert shard.is_file()
    assert not partial.exists()
    for frame_index in request.frame_indices:
        marker = read_shared_exposure_frame_completion(
            request.run_dir
            / "shared_exposure"
            / "completion"
            / f"frame_{frame_index:09d}.json",
            reference_root=request.run_dir,
        )
        assert marker["mode"] == "deterministic_parent_reconstruction"


def test_exposure_first_ignores_another_workers_plan_publication_tempfile(
    tmp_path,
) -> None:
    from et_mainsim.workflows.full_frame import run_worker

    request, api = _selection_ready_worker_request(tmp_path, n_frames=1)
    request = _enable_shared_exposure(request)
    shared_root = request.run_dir / "shared_exposure"
    shared_root.mkdir(parents=True)
    temporary = shared_root / ".target_plan.json.concurrent.tmp"
    temporary.write_bytes(b"in-flight immutable publication")

    result = run_worker(request, science_api=api)

    assert result.rendered == (0,)
    assert (shared_root / "target_plan.json").is_file()
    assert temporary.is_file()


def test_exposure_first_validates_all_complete_items_before_writing_missing(
    tmp_path,
) -> None:
    import h5py

    from et_mainsim.shared_exposure import SharedExposureArrayMismatchError
    from et_mainsim.workflows.full_frame import run_worker
    from photsim7.artifacts import ItemStatus, SharedExposureShardReader

    request, api = _selection_ready_worker_request(tmp_path, n_frames=2)
    request = _enable_shared_exposure(
        request,
        product_keys=("electron_stamp", "final_stamp"),
    )
    run_worker(request, science_api=api)
    for frame_index in request.frame_indices:
        (
            request.run_dir
            / "shared_exposure"
            / "completion"
            / f"frame_{frame_index:09d}.json"
        ).unlink()

    electron_final = _shared_final_shard_path(
        request,
        product_key="electron_stamp",
    )
    final_final = _shared_final_shard_path(request)
    electron_partial = api.partial_shard_path(electron_final)
    final_partial = api.partial_shard_path(final_final)
    electron_final.rename(electron_partial)
    final_final.rename(final_partial)
    with h5py.File(electron_partial, "r+") as handle:
        handle.attrs["shard_state"] = "partial"
        handle["status"][:] = np.uint8(ItemStatus.UNWRITTEN)
    with h5py.File(final_partial, "r+") as handle:
        handle.attrs["shard_state"] = "partial"
        handle["images"][0, 1, 0, 0] = np.uint16(handle["images"][0, 1, 0, 0] + 1)

    with pytest.raises(SharedExposureArrayMismatchError, match="C-order bytes"):
        run_worker(request, science_api=api)

    with SharedExposureShardReader(
        electron_partial,
        allow_incomplete=True,
    ) as reader:
        assert reader.item_status(11, 0) is ItemStatus.UNWRITTEN
        assert reader.item_status(11, 1) is ItemStatus.UNWRITTEN


def test_full_frame_run_refuses_completion_when_shared_marker_disappears(
    tmp_path,
    monkeypatch,
) -> None:
    from et_mainsim.config import (
        EXECUTION_SCHEMA_ID,
        EXECUTION_SCHEMA_VERSION,
        FullFrameWorkload,
        RunConfig,
        RunPaths,
    )
    from et_mainsim.workflows import full_frame
    from et_mainsim.workflows.full_frame import FullFrameRunPlan, run_full_frame

    request, api = _selection_ready_worker_request(tmp_path, n_frames=1)
    request = _enable_shared_exposure(request)
    config = RunConfig(
        schema_id=EXECUTION_SCHEMA_ID,
        schema_version=EXECUTION_SCHEMA_VERSION,
        workflow="et-full-frame",
        run_id=request.run_dir.name,
        paths=RunPaths(
            output_root=str(request.run_dir.parent),
            data_root=str(request.data_root),
            catalog_cache=str(request.catalog_cache),
        ),
        execution=request.execution,
        workload=FullFrameWorkload(
            shared_exposure_stamps=request.shared_exposure_stamps,
        ),
    )
    plan = FullFrameRunPlan(
        preset_name="test-shared-exposure",
        run_config=config,
        paths=config.resolve_paths(cwd=tmp_path),
        spec=request.spec,
        run_dir=request.run_dir,
        catalog_cache=request.catalog_cache,
        frame_indices=request.frame_indices,
        repo_root=tmp_path,
    )
    real_run_worker = full_frame.run_worker
    api.build_catalog_from_spec = lambda *args, **kwargs: api.StarCatalogCache.read(
        request.catalog_cache
    )

    def sabotaged_worker(*args, **kwargs):
        result = real_run_worker(*args, **kwargs)
        (request.run_dir / "shared_exposure/completion/frame_000000000.json").unlink()
        return result

    monkeypatch.setattr(full_frame, "run_worker", sabotaged_worker)

    with pytest.raises(RuntimeError, match="shared-exposure"):
        run_full_frame(plan, science_api=api)

    manifest = json.loads(
        (request.run_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "failed"
    assert manifest["artifacts"]["shared_exposure"] == {
        "root": str(request.run_dir / "shared_exposure"),
        "target_plan": str(request.run_dir / "shared_exposure" / "target_plan.json"),
        "completion_markers": str(request.run_dir / "shared_exposure" / "completion"),
        "worker_shards": str(request.run_dir / "shared_exposure" / "shards"),
        "target_plan_schema_id": "et_mainsim.shared_exposure_target_plan.v1",
        "target_plan_schema_version": 1,
        "frame_completion_schema_id": (
            "et_mainsim.shared_exposure_frame_completion.v1"
        ),
        "frame_completion_schema_version": 1,
        "target_source_ids": [11],
        "stamp_shape": [100, 300],
        "frames_per_shard": 32,
        "product_keys": ["final_stamp"],
        "independent_stamp_simulation": False,
        "zero_new_rng_draws": True,
    }


@pytest.mark.parametrize("backend", ["in-process", "local-subprocess"])
def test_full_frame_overwrite_replaces_only_the_shared_bundle_before_workers(
    tmp_path,
    monkeypatch,
    backend,
) -> None:
    from et_mainsim.config import (
        EXECUTION_SCHEMA_ID,
        EXECUTION_SCHEMA_VERSION,
        FullFrameWorkload,
        RunConfig,
        RunPaths,
    )
    from et_mainsim.workflows import full_frame
    from et_mainsim.workflows.full_frame import (
        FullFrameRunPlan,
        WorkerResult,
        run_full_frame,
    )

    request, api = _selection_ready_worker_request(tmp_path, n_frames=1)
    request = _enable_shared_exposure(request)
    execution = replace(
        request.execution,
        backend=backend,
        resume=False,
        overwrite=True,
    )
    config = RunConfig(
        schema_id=EXECUTION_SCHEMA_ID,
        schema_version=EXECUTION_SCHEMA_VERSION,
        workflow="et-full-frame",
        run_id=request.run_dir.name,
        paths=RunPaths(
            output_root=str(request.run_dir.parent),
            data_root=str(request.data_root),
            catalog_cache=str(request.catalog_cache),
        ),
        execution=execution,
        workload=FullFrameWorkload(
            shared_exposure_stamps=request.shared_exposure_stamps,
        ),
    )
    plan = FullFrameRunPlan(
        preset_name="test-shared-overwrite",
        run_config=config,
        paths=config.resolve_paths(cwd=tmp_path),
        spec=request.spec,
        run_dir=request.run_dir,
        catalog_cache=request.catalog_cache,
        frame_indices=request.frame_indices,
        repo_root=tmp_path,
    )
    catalog = api.StarCatalogCache.read(request.catalog_cache)
    fake_api = SimpleNamespace(
        DataRegistry=lambda **kwargs: object(),
        build_catalog_from_spec=lambda *args, **kwargs: catalog,
    )
    run_full_frame(plan, prepare_catalog_only=True, science_api=fake_api)

    shared_root = request.run_dir / "shared_exposure"
    (shared_root / "completion").mkdir(parents=True)
    (shared_root / "completion/frame_000000000.json").write_text(
        "stale marker",
        encoding="utf-8",
    )
    (shared_root / "shards/worker_0000").mkdir(parents=True)
    (shared_root / "shards/worker_0000/stale.h5.partial").write_bytes(b"partial")
    (shared_root / "target_plan.json").write_text("stale plan", encoding="utf-8")
    unrelated = request.run_dir / "unrelated" / "keep.txt"
    unrelated.parent.mkdir()
    unrelated.write_text("keep", encoding="utf-8")

    observed_prepared: list[bool] = []

    def assert_clean_bundle_and_rebuild(*args, **kwargs):
        if backend == "in-process":
            worker_request = args[0]
            prepared = worker_request.shared_exposure_overwrite_prepared
        else:
            prepared = kwargs.get("shared_exposure_overwrite_prepared", False)
        observed_prepared.append(prepared)
        assert not shared_root.exists()
        assert unrelated.read_text(encoding="utf-8") == "keep"
        shared_root.mkdir()
        (shared_root / "rebuilt.txt").write_text("new", encoding="utf-8")
        return (
            WorkerResult(rank=0, rendered=(0,), skipped=(), elapsed_s=0.0)
            if backend == "in-process"
            else [WorkerResult(rank=0, rendered=(0,), skipped=(), elapsed_s=0.0)]
        )

    if backend == "in-process":
        monkeypatch.setattr(full_frame, "run_worker", assert_clean_bundle_and_rebuild)
    else:
        monkeypatch.setattr(
            full_frame,
            "_launch_subprocess_workers",
            assert_clean_bundle_and_rebuild,
        )
    monkeypatch.setattr(full_frame, "frame_is_complete", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        full_frame,
        "_shared_exposure_incomplete_frames_for_worker",
        lambda *args, **kwargs: (),
    )

    run_full_frame(plan, science_api=fake_api)

    assert observed_prepared == [True]
    assert (shared_root / "rebuilt.txt").read_text(encoding="utf-8") == "new"
    assert not (shared_root / "target_plan.json").exists()
    assert unrelated.read_text(encoding="utf-8") == "keep"


def test_full_frame_completion_strictly_reads_closed_selection_sidecars(
    tmp_path,
) -> None:
    from et_mainsim.workflows.full_frame import frame_is_complete, run_worker
    from photsim7.selection_artifacts import read_cadence_selection_truth

    request, api = _selection_ready_worker_request(tmp_path, n_frames=2)

    result = run_worker(request, science_api=api)

    assert result.rendered == (0, 1)
    truths = []
    for frame_index in request.frame_indices:
        assert frame_is_complete(
            request.run_dir,
            frame_index,
            expected_shape=tuple(request.spec.detector.shape),
            expected_spec=request.spec,
        )
        schema_path = (
            request.run_dir / "frame_summaries" / f"frame_{frame_index:06d}_schema.json"
        )
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        selection = schema["selection_truth"]
        assert selection["verification_status"] == "persisted_and_verified"
        cadence = selection["artifact"]["cadence"]
        assert cadence["relative_path"] == (
            f"selection_truth/cadence/frame_{frame_index:09d}.json"
        )
        truth = read_cadence_selection_truth(
            request.run_dir / cadence["relative_path"],
            artifact_root=request.run_dir,
            expected_sha256=selection["content_sha256"],
        )
        assert truth.local_frame_index == frame_index
        assert truth.absolute_raw_frame_index == frame_index
        truths.append(truth)

    assert (
        truths[0].source_geometry_truth.content_sha256
        == truths[1].source_geometry_truth.content_sha256
    )
    assert (
        truths[0].psf_selection_truth.content_sha256
        == truths[1].psf_selection_truth.content_sha256
    )
    assert len(list((request.run_dir / "selection_truth/geometry").glob("*.json"))) == 1
    assert len(list((request.run_dir / "selection_truth/psf").glob("*.json"))) == 1

    first_schema = json.loads(
        (request.run_dir / "frame_summaries/frame_000000_schema.json").read_text(
            encoding="utf-8"
        )
    )
    for label in ("geometry", "psf"):
        missing_root = tmp_path / f"missing-{label}"
        shutil.copytree(request.run_dir, missing_root)
        relative = first_schema["selection_truth"]["artifact"][label]["relative_path"]
        (missing_root / relative).unlink()
        assert not frame_is_complete(
            missing_root,
            0,
            expected_shape=tuple(request.spec.detector.shape),
            expected_spec=request.spec,
        )

    schema_path = request.run_dir / "frame_summaries/frame_000000_schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    schema["selection_truth"]["content_sha256"] = "b" * 64
    schema_path.write_text(json.dumps(schema), encoding="utf-8")
    assert not frame_is_complete(
        request.run_dir,
        0,
        expected_shape=tuple(request.spec.detector.shape),
        expected_spec=request.spec,
    )


def test_full_frame_resume_recovers_missing_and_orphan_sidecars_but_conflicts(
    tmp_path,
) -> None:
    from et_mainsim.workflows.full_frame import frame_is_complete, run_worker
    from photsim7.selection_artifacts import SelectionArtifactConflictError

    request, api = _selection_ready_worker_request(tmp_path, n_frames=2)
    run_worker(request, science_api=api)
    cadence = request.run_dir / "selection_truth/cadence/frame_000000000.json"

    cadence.unlink()
    assert not frame_is_complete(
        request.run_dir,
        0,
        expected_shape=tuple(request.spec.detector.shape),
        expected_spec=request.spec,
    )
    repaired = run_worker(request, science_api=api)
    assert repaired.rendered == (0,)
    assert repaired.skipped == (1,)
    assert cadence.is_file()

    for path in (
        request.run_dir / "frames/frame_000000.npy",
        request.run_dir / "frame_summaries/frame_000000.json",
        request.run_dir / "frame_summaries/frame_000000_schema.json",
    ):
        path.unlink()
    assert cadence.is_file()
    non_resuming = replace(
        request,
        execution=replace(request.execution, resume=False),
        frame_indices=(0,),
    )
    with pytest.raises(FileExistsError, match="already has artifacts"):
        run_worker(non_resuming, science_api=api)
    repaired_orphan = run_worker(request, science_api=api)
    assert repaired_orphan.rendered == (0,)
    assert repaired_orphan.skipped == (1,)
    assert cadence.is_file()

    cadence.write_bytes(b"conflicting sidecar owner\n")
    assert not frame_is_complete(
        request.run_dir,
        0,
        expected_shape=tuple(request.spec.detector.shape),
        expected_spec=request.spec,
    )
    with pytest.raises(SelectionArtifactConflictError, match="conflict"):
        run_worker(request, science_api=api)

    overwrite_request = replace(
        request,
        execution=replace(
            request.execution,
            resume=False,
            overwrite=True,
        ),
        frame_indices=(0,),
    )
    overwritten = run_worker(overwrite_request, science_api=api)
    assert overwritten.rendered == (0,)
    assert frame_is_complete(
        request.run_dir,
        0,
        expected_shape=tuple(request.spec.detector.shape),
        expected_spec=request.spec,
    )


def test_full_frame_completion_rejects_self_consistent_identity_transplant(
    tmp_path,
) -> None:
    from et_mainsim.workflows.full_frame import frame_is_complete, run_worker
    from photsim7.full_frame_pipeline import _selection_truth_metadata
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
    from photsim7.selection_artifacts import (
        CadenceSelectionTruth,
        read_cadence_selection_truth,
        write_cadence_selection_truth,
    )

    request, api = _selection_ready_worker_request(tmp_path, n_frames=1)
    run_worker(request, science_api=api)
    original = read_cadence_selection_truth(
        request.run_dir / "selection_truth/cadence/frame_000000000.json",
        artifact_root=request.run_dir,
    )

    bank_identity = {
        "logical_bank_id": CANONICAL_JITTER_BANK_LOGICAL_ID,
        "bank_evidence_id": CANONICAL_JITTER_BANK_EVIDENCE_ID,
        "array_sha256": CANONICAL_JITTER_BANK_SHA256,
        "expected_array_sha256": CANONICAL_JITTER_BANK_SHA256,
        "manifest_sha256": CANONICAL_JITTER_BANK_MANIFEST_SHA256,
        "expected_manifest_sha256": CANONICAL_JITTER_BANK_MANIFEST_SHA256,
        "verification_status": ("array_and_manifest_sha256_verified_before_load"),
        "loader": NATIVE_JITTER_BANK_LOADER_ID,
    }
    expected_realization = request.spec.science_profile.science_realization_id
    for label, spacecraft_id, realization_id, run_seed in (
        (
            "spacecraft",
            "transplanted-spacecraft",
            expected_realization,
            request.spec.rng.run_seed,
        ),
        (
            "realization",
            "et",
            expected_realization + 1,
            request.spec.rng.run_seed,
        ),
        (
            "run-seed",
            "et",
            expected_realization,
            request.spec.rng.run_seed + 1,
        ),
    ):
        transplanted_root = tmp_path / label
        shutil.copytree(request.run_dir, transplanted_root)
        shutil.rmtree(transplanted_root / "selection_truth")
        selector = JitterModelSelector(
            seed_tree=replace(
                request.spec.rng,
                run_seed=run_seed,
            ).to_seed_tree(),
            bank_identity=bank_identity,
            n_models=100,
            science_realization_id=realization_id,
            spacecraft_id=spacecraft_id,
            absolute_raw_frame_start_index=0,
        )
        transplanted = CadenceSelectionTruth(
            detector_id=original.detector_id,
            local_frame_index=0,
            source_geometry_truth=original.source_geometry_truth,
            psf_selection_truth=original.psf_selection_truth,
            jitter_model_selection_truth=selector.select(0),
        )
        artifacts = write_cadence_selection_truth(
            transplanted_root,
            transplanted,
        )
        schema_path = transplanted_root / "frame_summaries/frame_000000_schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        schema["selection_truth"] = _selection_truth_metadata(
            request.spec,
            truth=transplanted,
            missing_components=(),
            artifacts=artifacts,
        )
        schema_path.write_text(json.dumps(schema), encoding="utf-8")

        assert not frame_is_complete(
            transplanted_root,
            0,
            expected_shape=tuple(request.spec.detector.shape),
            expected_spec=request.spec,
        )


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


def test_full_frame_run_identity_requires_current_product_contract(tmp_path) -> None:
    from et_mainsim.config import RunPaths
    from et_mainsim.manifest import ManifestIdentityError
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
    catalog = SimpleNamespace(n_sources=1, metadata={"source": "test"})
    fake_api = SimpleNamespace(
        DataRegistry=lambda **kwargs: object(),
        build_catalog_from_spec=lambda *args, **kwargs: catalog,
    )

    run_full_frame(plan, prepare_catalog_only=True, science_api=fake_api)
    manifest_path = plan.run_dir / "run_manifest.json"
    original = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert original["workload"]["product_contract"] == {
        "frame_product_schema_id": ("photsim7.single_cadence_frame_products.v1"),
        "frame_product_schema_version": 1,
        "source_geometry_truth_schema_id": ("photsim7.source_geometry_truth.v1"),
        "psf_selection_truth_schema_id": ("photsim7.psf_selection_truth.v2"),
        "cadence_selection_truth_schema_id": ("photsim7.cadence_selection_truth.v1"),
        "cadence_selection_truth_schema_version": 1,
        "full_frame_source_pixel_geometry_schema_id": (
            "photsim7.full_frame_source_pixel_geometry.v1"
        ),
        "full_frame_source_pixel_geometry_schema_version": 1,
        "shared_exposure_crop_schema_id": "photsim7.shared_exposure_crop.v1",
        "shared_exposure_crop_schema_version": 1,
        "shared_exposure_image_shard_schema_id": (
            "photsim7.shared_exposure_image_shard.v1"
        ),
        "shared_exposure_image_shard_schema_version": 1,
    }

    for mutation in ("missing", "old-version"):
        stale = json.loads(json.dumps(original))
        if mutation == "missing":
            stale["workload"].pop("product_contract")
        else:
            stale["workload"]["product_contract"][
                "cadence_selection_truth_schema_version"
            ] = 0
        manifest_path.write_text(json.dumps(stale), encoding="utf-8")
        with pytest.raises(ManifestIdentityError, match="workload identity"):
            run_full_frame(
                plan,
                prepare_catalog_only=True,
                science_api=fake_api,
            )


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
        build_multiscope_full_frame_services=fail_services,
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
