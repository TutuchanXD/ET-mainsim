from __future__ import annotations

import json
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _write_minimal_task_manifest(tmp_path: Path) -> Path:
    from et_mainsim.stamp_inputs import file_identity
    from et_mainsim.time_shards import plan_continuous_time_shards

    run_root = tmp_path / "run"
    time_plan = plan_continuous_time_shards(
        raw_start_index=0,
        raw_stop_index=6,
        coadd_sizes=(3,),
        raw_exposure_seconds=10.0,
        max_raw_frames_per_shard=3,
    )
    time_plan_path = time_plan.write_manifest(run_root / "inputs/time_shards.json")
    manifest_path = run_root / "production_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_id": "et_mainsim.science_stamp_production.v1",
                "schema_version": 1,
                "production_track": "varlc",
                "delivery": {
                    "time_plan_relative_path": "inputs/time_shards.json",
                    "time_plan_identity": file_identity(time_plan_path),
                },
                "targets": [
                    {"source_id_int64": 101},
                    {"source_id_int64": 202},
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return manifest_path


def test_write_science_task_list_binds_exact_manifest_case_and_selection(
    tmp_path: Path,
) -> None:
    from et_mainsim.science_stamp_production import (
        write_science_stamp_task_list,
    )

    manifest_path = _write_minimal_task_manifest(tmp_path)
    output_path = tmp_path / "tasks/static_representatives.json"

    result = write_science_stamp_task_list(
        manifest_path,
        case="static",
        tasks=((202, 1), (101, 0)),
        output_path=output_path,
    )

    manifest_raw = manifest_path.read_bytes()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload == {
        "schema_id": "et_mainsim.science_stamp_task_list.v1",
        "schema_version": 1,
        "case": "static",
        "production_manifest_identity": {
            "sha256": hashlib.sha256(manifest_raw).hexdigest(),
            "size_bytes": len(manifest_raw),
        },
        "tasks": [
            {"source_id": 202, "shard_id": 1},
            {"source_id": 101, "shard_id": 0},
        ],
    }
    assert result.path == output_path.resolve()
    assert result.case == "static"
    assert result.task_count == 2
    assert result.identity == {
        "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "size_bytes": output_path.stat().st_size,
    }


@pytest.mark.parametrize(
    ("tasks", "error"),
    [
        (((101, 0), (101, 0)), "duplicate"),
        (((999, 0),), "unknown source_id"),
        (((101, 99),), "unknown shard_id"),
        (((101.5, 0),), "must be an integer"),
        ((), "at least one"),
    ],
)
def test_write_science_task_list_rejects_invalid_selection_without_output(
    tmp_path: Path,
    tasks: tuple[tuple[int, int], ...],
    error: str,
) -> None:
    from et_mainsim.science_stamp_production import (
        write_science_stamp_task_list,
    )

    manifest_path = _write_minimal_task_manifest(tmp_path)
    output_path = tmp_path / "tasks.json"

    with pytest.raises(ValueError, match=error):
        write_science_stamp_task_list(
            manifest_path,
            case="injected",
            tasks=tasks,
            output_path=output_path,
        )

    assert not output_path.exists()


def test_write_science_task_list_never_overwrites_existing_output(
    tmp_path: Path,
) -> None:
    from et_mainsim.science_stamp_production import (
        write_science_stamp_task_list,
    )

    manifest_path = _write_minimal_task_manifest(tmp_path)
    output_path = tmp_path / "tasks.json"
    output_path.write_text("owned-by-user\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        write_science_stamp_task_list(
            manifest_path,
            case="static",
            tasks=((101, 0),),
            output_path=output_path,
        )

    assert output_path.read_text(encoding="utf-8") == "owned-by-user\n"


def test_write_science_task_list_rejects_an_output_symlink(
    tmp_path: Path,
) -> None:
    from et_mainsim.science_stamp_production import (
        write_science_stamp_task_list,
    )

    manifest_path = _write_minimal_task_manifest(tmp_path)
    linked_target = tmp_path / "must-not-be-created.json"
    output_path = tmp_path / "tasks.json"
    output_path.symlink_to(linked_target)

    with pytest.raises(FileExistsError):
        write_science_stamp_task_list(
            manifest_path,
            case="static",
            tasks=((101, 0),),
            output_path=output_path,
        )

    assert output_path.is_symlink()
    assert not linked_target.exists()


def _curve(*, track: str = "varlc"):
    from et_mainsim.stamp_science_inputs import ScienceInputCurve

    return ScienceInputCurve(
        track=track,
        namespace="varlc",
        external_source_id="KIC003331147",
        source_id_int64=3_331_147,
        source_class="pulsating_variable",
        gaia_g_mag=11.5,
        detector_xpix=2_000.0,
        detector_ypix=4_500.0,
        factors=np.asarray([0.9, 1.0, 1.1, 1.2]),
        metadata={
            "q_definition": "normalised_flux",
            "magnitude_origin": "project_default_missing_input",
        },
    )


def test_science_production_config_freezes_90_day_matrix(tmp_path) -> None:
    from et_mainsim.science_stamp_production import ScienceStampProductionConfig

    config = ScienceStampProductionConfig(
        track="aster",
        input_root=tmp_path / "lcdata",
        output_root=tmp_path / "results",
        run_id="aster_90d_v1",
        data_root=tmp_path / "photsim-data",
        focalplane_registry=tmp_path / "focalplane",
    )

    assert config.n_raw_frames == 777_600
    assert config.raw_exposure_seconds == 10.0
    assert config.cadence_seconds == (30.0, 60.0, 120.0, 300.0)
    assert config.coadd_sizes == (3, 6, 12, 30)
    assert config.stamp_shape == (100, 300)
    assert config.delivery_execution_mode == "staged_local_scratch_v1"


def test_science_production_spec_disables_only_dva_for_reference_field() -> None:
    from et_mainsim.galaxy_stamp_production import (
        build_galaxy_independent_production_spec,
    )
    from et_mainsim.science_stamp_production import (
        build_science_independent_production_spec,
    )

    baseline = build_galaxy_independent_production_spec(
        n_raw_frames=12,
        raw_exposure_seconds=10.0,
        device="cpu",
        run_seed=9,
    )
    science = build_science_independent_production_spec(
        n_raw_frames=12,
        raw_exposure_seconds=10.0,
        device="cpu",
        run_seed=9,
    )

    assert baseline.dynamic_effects.dva.enabled is True
    assert science.dynamic_effects.dva.enabled is False
    baseline_dynamic = baseline.to_json_dict()["dynamic_effects"]
    science_dynamic = science.to_json_dict()["dynamic_effects"]
    baseline_dynamic["dva"]["enabled"] = False
    assert science_dynamic == baseline_dynamic


def test_prepare_science_production_publishes_generic_reference_field_manifest(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import et_mainsim.science_stamp_production as production
    from et_mainsim.science_stamp_production import (
        ScienceStampProductionConfig,
        prepare_science_independent_production,
    )

    input_root = tmp_path / "lcdata"
    data_root = tmp_path / "photsim-data"
    focalplane = tmp_path / "focalplane"
    input_root.mkdir()
    data_root.mkdir()
    focalplane.mkdir()
    curve = _curve()
    object.__setattr__(curve, "factors", curve.factors[:3])
    monkeypatch.setattr(
        production,
        "load_science_track_inputs",
        lambda *args, **kwargs: (curve,),
    )
    monkeypatch.setattr(
        production,
        "focalplane_registry_identity",
        lambda _path: {
            "schema_id": "test.registry",
            "sha256": "a" * 64,
            "size_bytes": 1,
        },
    )

    prepared = prepare_science_independent_production(
        ScienceStampProductionConfig(
            track="varlc",
            input_root=input_root,
            output_root=tmp_path / "results",
            run_id="varlc_tiny_v1",
            data_root=data_root,
            focalplane_registry=focalplane,
            duration_days=30.0 / 86_400.0,
            cadence_seconds=(30.0,),
            max_raw_frames_per_shard=3,
            device="cpu",
        )
    )
    manifest = json.loads(prepared.manifest_path.read_text(encoding="utf-8"))

    assert manifest["schema_id"] == "et_mainsim.science_stamp_production.v1"
    assert manifest["production_track"] == "varlc"
    assert manifest["input"]["time_alignment"] == "simulation_raw_frame_index"
    assert manifest["input"]["native_absolute_time_used"] is False
    assert manifest["delivery"]["execution_mode"] == "staged_local_scratch_v1"
    assert manifest["delivery"]["stamp_shape"] == [100, 300]
    assert manifest["simulation_spec_base"]["dynamic_effects"]["dva"][
        "enabled"
    ] is False
    assert len(manifest["targets"]) == 1
    target = manifest["targets"][0]
    assert target["source_id_int64"] == 3_331_147
    assert target["external_source_id"] == "KIC003331147"
    assert target["source_id_namespace"] == "varlc"
    assert target["magnitude_system"] == "Gaia_G_Vega"
    assert target["detector_placement"] == {
        "detector_id": "main_rd",
        "detector_xpix": 2_000.0,
        "detector_ypix": 4_500.0,
        "location_mode": "reference_field_nonphysical",
    }
    assert target["psf"] == {
        "selection": "explicit_field_id",
        "psf_id": 6,
        "node_angle_deg": 12.0,
    }
    assert target["dva_policy"] == "disabled_no_sky_coordinate"
    assert Path(target["factor_snapshot_relative_path"]).is_absolute() is False
    assert prepared.time_plan.accepted_raw_frame_count == 3
    assert len(prepared.time_plan.shards) == 1

    from astropy.table import Table

    table_path = prepared.run_root / target["target_table"]["relative_path"]
    table = Table.read(table_path)
    assert table.colnames == [
        "source_id",
        "gaia_g_mag",
        "psf_id",
        "detector_xpix",
        "detector_ypix",
    ]
    assert "ra_deg" not in table.colnames
    assert int(table["psf_id"][0]) == 6


@pytest.mark.parametrize(
    ("xpix", "ypix"),
    [(149.0, 4_500.0), (8_750.0, 4_500.0), (2_000.0, 49.0), (2_000.0, 9_070.0)],
)
def test_prepare_rejects_a_reference_position_whose_full_stamp_crosses_detector(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    xpix: float,
    ypix: float,
) -> None:
    import et_mainsim.science_stamp_production as production
    from et_mainsim.science_stamp_production import (
        ScienceStampProductionConfig,
        prepare_science_independent_production,
    )

    input_root = tmp_path / "lcdata"
    data_root = tmp_path / "photsim-data"
    focalplane = tmp_path / "focalplane"
    input_root.mkdir()
    data_root.mkdir()
    focalplane.mkdir()
    curve = _curve()
    object.__setattr__(curve, "factors", curve.factors[:3])
    object.__setattr__(curve, "detector_xpix", xpix)
    object.__setattr__(curve, "detector_ypix", ypix)
    monkeypatch.setattr(
        production,
        "load_science_track_inputs",
        lambda *args, **kwargs: (curve,),
    )
    monkeypatch.setattr(
        production,
        "focalplane_registry_identity",
        lambda _path: {"sha256": "a" * 64, "size_bytes": 1},
    )
    config = ScienceStampProductionConfig(
        track="varlc",
        input_root=input_root,
        output_root=tmp_path / "results",
        run_id="invalid",
        data_root=data_root,
        focalplane_registry=focalplane,
        duration_days=30.0 / 86_400.0,
        cadence_seconds=(30.0,),
        max_raw_frames_per_shard=3,
        device="cpu",
    )

    with pytest.raises(ValueError, match="full 100x300 stamp"):
        prepare_science_independent_production(config)
    assert not config.run_root.exists()


def test_science_worker_rejects_staged_mode_without_scratch_before_runtime(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import et_mainsim.science_stamp_production as production

    manifest = tmp_path / "production_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_id": production.SCIENCE_STAMP_PRODUCTION_SCHEMA_ID,
                "schema_version": production.SCIENCE_STAMP_PRODUCTION_SCHEMA_VERSION,
                "delivery": {"execution_mode": "staged_local_scratch_v1"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        production,
        "_runtime_paths",
        lambda *_args, **_kwargs: pytest.fail(
            "writer-mode gate must run before runtime setup"
        ),
    )

    with pytest.raises(ValueError, match="staged_local_scratch_v1.*output_root"):
        production.run_science_independent_target(
            manifest,
            source_id=3_331_147,
        )


def test_science_worker_rejects_factor_snapshot_identity_drift_before_services(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import et_mainsim.science_stamp_production as production
    import et_mainsim.workflows.stamp as stamp_workflow

    input_root = tmp_path / "lcdata"
    data_root = tmp_path / "photsim-data"
    focalplane = tmp_path / "focalplane"
    input_root.mkdir()
    data_root.mkdir()
    focalplane.mkdir()
    curve = _curve()
    object.__setattr__(curve, "factors", curve.factors[:3])
    monkeypatch.setattr(
        production,
        "load_science_track_inputs",
        lambda *args, **kwargs: (curve,),
    )
    monkeypatch.setattr(
        production,
        "focalplane_registry_identity",
        lambda _path: {"sha256": "a" * 64, "size_bytes": 1},
    )
    prepared = production.prepare_science_independent_production(
        production.ScienceStampProductionConfig(
            track="varlc",
            input_root=input_root,
            output_root=tmp_path / "results",
            run_id="varlc_tiny_v1",
            data_root=data_root,
            focalplane_registry=focalplane,
            duration_days=30.0 / 86_400.0,
            cadence_seconds=(30.0,),
            max_raw_frames_per_shard=3,
            device="cpu",
        )
    )
    payload = json.loads(prepared.manifest_path.read_text(encoding="utf-8"))
    snapshot = prepared.run_root / payload["targets"][0][
        "factor_snapshot_relative_path"
    ]
    snapshot.write_bytes(snapshot.read_bytes() + b"changed")
    monkeypatch.setattr(
        production,
        "_runtime_paths",
        lambda *_args, **_kwargs: (data_root, focalplane, {}, {}),
    )
    monkeypatch.setattr(
        stamp_workflow,
        "build_run_plan",
        lambda **_kwargs: pytest.fail(
            "snapshot identity gate must run before service planning"
        ),
    )

    with pytest.raises(ValueError, match="factor snapshot identity changed"):
        production.run_science_independent_target(
            prepared.manifest_path,
            source_id=curve.source_id_int64,
            output_root=tmp_path / "scratch" / "injected",
            data_root=data_root,
            focalplane_registry=focalplane,
            device="cpu",
        )


def test_science_worker_keeps_static_injected_rng_pairing_and_generic_provenance(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import et_mainsim.science_stamp_production as production
    import et_mainsim.workflows.stamp as stamp_workflow
    import photsim7.simulation_services as simulation_services
    import photsim7.stamp_pipeline as stamp_pipeline
    from et_mainsim.stamp_science_inputs import write_science_factor_snapshot
    from et_mainsim.time_shards import plan_continuous_time_shards

    source_id = 3_331_147
    run_root = tmp_path / "varlc"
    inputs = run_root / "inputs"
    inputs.mkdir(parents=True)
    table_path = inputs / "targets.ecsv"
    table_path.write_text("fixture", encoding="utf-8")
    curve = _curve()
    object.__setattr__(curve, "factors", np.ones(3, dtype=np.float64))
    snapshot_path = inputs / "source.npz"
    snapshot_identity = write_science_factor_snapshot(snapshot_path, curve=curve)
    time_plan = plan_continuous_time_shards(
        raw_start_index=0,
        raw_stop_index=3,
        coadd_sizes=(3,),
        raw_exposure_seconds=10.0,
        max_raw_frames_per_shard=3,
    )
    time_path = time_plan.write_manifest(inputs / "time_shards.json")
    base_spec = production.build_science_independent_production_spec(
        n_raw_frames=3,
        raw_exposure_seconds=10.0,
        device="cpu",
        run_seed=12345,
    )
    manifest_path = run_root / "production_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_id": production.SCIENCE_STAMP_PRODUCTION_SCHEMA_ID,
                "schema_version": production.SCIENCE_STAMP_PRODUCTION_SCHEMA_VERSION,
                "run_id": "varlc-fixture",
                "production_track": "varlc",
                "runtime_defaults": {
                    "data_root": str(tmp_path / "data"),
                    "focalplane_registry": str(tmp_path / "registry"),
                },
                "input": {"focalplane_registry": {}},
                "delivery": {
                    "execution_mode": "staged_local_scratch_v1",
                    "stamp_shape": [100, 300],
                    "time_plan_relative_path": "inputs/time_shards.json",
                    "time_plan_identity": production.file_identity(time_path),
                },
                "simulation_spec_base": base_spec.to_json_dict(),
                "targets": [
                    {
                        "source_id_int64": source_id,
                        "external_source_id": "KIC003331147",
                        "source_id_namespace": "varlc",
                        "detector_placement": {"detector_id": "main_rd"},
                        "target_table": {
                            "relative_path": "inputs/targets.ecsv",
                            "file_identity": production.file_identity(table_path),
                        },
                        "factor_snapshot_relative_path": "inputs/source.npz",
                        "factor_snapshot": snapshot_identity,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    data_root = tmp_path / "data"
    registry = tmp_path / "registry"
    data_root.mkdir()
    registry.mkdir()
    prepared = SimpleNamespace(
        targets={source_id: SimpleNamespace(source_id=source_id)},
        psf_ids={source_id: 6},
        source_input_truth={source_id: {"source_id": source_id}},
    )
    monkeypatch.setattr(
        production,
        "_runtime_paths",
        lambda *_args, **_kwargs: (data_root, registry, {}, {}),
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
        lambda *_args, **_kwargs: base_spec,
    )
    original_build_context = simulation_services.build_simulation_context
    contexts = []

    def capture_context(*args, **kwargs):
        context = original_build_context(*args, **kwargs)
        contexts.append(context)
        return context

    monkeypatch.setattr(
        simulation_services,
        "build_simulation_context",
        capture_context,
    )
    monkeypatch.setattr(
        simulation_services,
        "build_stamp_services",
        lambda context, **_kwargs: SimpleNamespace(context=context),
    )
    render_calls = []
    monkeypatch.setattr(
        stamp_pipeline,
        "run_single_cadence_stamp",
        lambda *_args, **kwargs: render_calls.append(kwargs) or SimpleNamespace(),
    )
    requests = []

    def fake_delivery(request, *, render_raw, adapt_raw):
        del adapt_raw
        requests.append(request)
        render_raw(0)
        return SimpleNamespace(
            shard_id=0,
            raw_frame_count=3,
            raw_path=request.shard_root / "raw.h5",
            coadd_paths={3: request.shard_root / "coadd_30s.h5"},
        )

    monkeypatch.setattr(
        production,
        "run_independent_stamp_time_shard",
        fake_delivery,
    )

    production.run_science_independent_target(
        manifest_path,
        source_id=source_id,
        case="static",
        output_root=tmp_path / "scratch-static",
        device="cpu",
    )
    production.run_science_independent_target(
        manifest_path,
        source_id=source_id,
        case="injected",
        output_root=tmp_path / "scratch-injected",
        device="cpu",
    )

    assert render_calls[0]["source_variability"] is None
    np.testing.assert_array_equal(
        render_calls[1]["source_variability"].relative_flux,
        np.ones((1, 3)),
    )
    assert contexts[0].detector_rng_scope(local_frame_index=0) == contexts[
        1
    ].detector_rng_scope(local_frame_index=0)
    assert requests[0].manifest["physical_rng_pairing"] == requests[1].manifest[
        "physical_rng_pairing"
    ]
    assert requests[0].manifest["physical_rng_pairing"]["schema_id"] == (
        "et_mainsim.science_physical_rng_pairing.v1"
    )
    assert requests[1].manifest["production_manifest"] == str(
        manifest_path.resolve()
    )
    assert "galaxy_production_manifest" not in requests[1].manifest
    assert requests[1].manifest["target_input_truth"]["variability"][
        "time_alignment"
    ] == "simulation_raw_frame_index"
    assert requests[1].provenance["simulation_spec"]["dynamic_effects"]["dva"][
        "enabled"
    ] is False
