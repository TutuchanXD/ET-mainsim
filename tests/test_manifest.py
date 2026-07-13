from __future__ import annotations

import json

import pytest


def test_manifest_lifecycle_is_atomic_and_records_completion(tmp_path) -> None:
    from et_mainsim.manifest import RunManifestStore

    path = tmp_path / "run_manifest.json"
    store = RunManifestStore(path)
    created = store.create(
        workflow="et-full-frame",
        preset="et-full-frame-smoke",
        run_id="smoke",
        simulation_spec={"schema_id": "photsim7.simulation_spec"},
        execution={"device": "cpu"},
        frame_plan={"requested": [0]},
        provenance={"et_mainsim": {"commit": "abc"}},
    )

    assert created["status"] == "planned"
    assert path.exists()
    assert list(tmp_path.glob(".run_manifest.json.*.tmp")) == []

    running = store.transition("running", catalog={"cache_path": "stars.npz"})
    completed = store.transition(
        "completed",
        completion={"requested": 1, "completed": 1, "skipped": 0},
    )

    assert running["timestamps"]["started_at"] is not None
    assert completed["timestamps"]["completed_at"] is not None
    assert json.loads(path.read_text(encoding="utf-8"))["completion"]["completed"] == 1

    with pytest.raises(ValueError, match="transition"):
        store.transition("running")


def test_manifest_failure_keeps_original_identity(tmp_path) -> None:
    from et_mainsim.manifest import RunManifestStore

    store = RunManifestStore(tmp_path / "run_manifest.json")
    store.create(
        workflow="et-full-frame",
        preset="et-full-frame-smoke",
        run_id="smoke",
        simulation_spec={"detector": {"shape": [5, 7]}},
        execution={"device": "cpu"},
        frame_plan={"requested": [0]},
        provenance={},
    )
    store.transition("running")

    failed = store.fail(RuntimeError("render failed"))

    assert failed["status"] == "failed"
    assert failed["run_id"] == "smoke"
    assert failed["failure"] == {
        "type": "RuntimeError",
        "message": "render failed",
    }


def test_manifest_existing_run_must_match_identity(tmp_path) -> None:
    from et_mainsim.manifest import ManifestIdentityError, RunManifestStore

    store = RunManifestStore(tmp_path / "run_manifest.json")
    store.create(
        workflow="et-full-frame",
        preset="a",
        run_id="same",
        simulation_spec={"rng": {"run_seed": 1}},
        execution={"device": "cpu"},
        frame_plan={"requested": [0]},
        provenance={},
    )

    with pytest.raises(ManifestIdentityError, match="scientific spec"):
        store.ensure_identity(
            workflow="et-full-frame",
            run_id="same",
            simulation_spec={"rng": {"run_seed": 2}},
            execution={"device": "cpu"},
        )


def test_completed_manifest_can_start_a_new_resume_attempt(tmp_path) -> None:
    from et_mainsim.manifest import RunManifestStore

    store = RunManifestStore(tmp_path / "run_manifest.json")
    store.create(
        workflow="et-full-frame",
        preset="smoke",
        run_id="smoke",
        simulation_spec={"rng": {"run_seed": 1}},
        execution={"device": "cpu"},
        frame_plan={"requested": [0]},
        provenance={},
    )
    first = store.start_attempt()
    store.transition("completed", completion={"completed": 1})

    resumed = store.start_attempt()

    assert first["attempts"][-1]["number"] == 1
    assert resumed["status"] == "running"
    assert resumed["completion"] is None
    assert resumed["failure"] is None
    assert resumed["attempts"][-1]["number"] == 2
    assert resumed["attempts"][-1]["previous_status"] == "completed"

    with pytest.raises(ValueError, match="already running"):
        store.start_attempt()


def test_resume_and_overwrite_controls_do_not_change_run_identity(tmp_path) -> None:
    from et_mainsim.manifest import RunManifestStore

    store = RunManifestStore(tmp_path / "run_manifest.json")
    store.create(
        workflow="et-full-frame",
        preset="smoke",
        run_id="smoke",
        simulation_spec={"rng": {"run_seed": 1}},
        execution={
            "device": "cpu",
            "resume": True,
            "overwrite": False,
            "force_catalog_cache": False,
            "progress": False,
        },
        frame_plan={"requested": [0]},
        provenance={},
    )

    matched = store.ensure_identity(
        workflow="et-full-frame",
        run_id="smoke",
        simulation_spec={"rng": {"run_seed": 1}},
        execution={
            "device": "cpu",
            "resume": False,
            "overwrite": True,
            "force_catalog_cache": True,
            "progress": True,
        },
    )

    assert matched["run_id"] == "smoke"
