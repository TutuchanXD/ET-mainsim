from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
from types import ModuleType, SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = ROOT / "scripts" / "run_science_independent_stamp_production.py"
LAUNCHER_PATH = (
    ROOT / "scripts" / "science_independent_stamp_staged_slurm_array.sh"
)


def _load_cli() -> ModuleType:
    spec = importlib.util.spec_from_file_location("science_production_cli", CLI_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _write_json(path: Path, payload: object) -> dict[str, object]:
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    path.write_bytes(raw)
    return {"sha256": _sha256_bytes(raw), "size_bytes": len(raw)}


def _write_launcher_fixture(
    tmp_path: Path,
    *,
    execution_mode: str = "staged_local_scratch_v1",
    case: str = "injected",
    task_payload: object | None = None,
) -> tuple[dict[str, str], Path]:
    run_root = tmp_path / "run"
    inputs = run_root / "inputs"
    inputs.mkdir(parents=True)
    time_plan_path = inputs / "time_shards.json"
    time_plan_identity = _write_json(
        time_plan_path,
        {
            "shards": [
                {"shard_id": 4},
                {"shard_id": 9},
            ]
        },
    )
    manifest_path = run_root / "production_manifest.json"
    manifest_identity = _write_json(
        manifest_path,
        {
            "schema_id": "et_mainsim.science_stamp_production.v1",
            "schema_version": 1,
            "production_track": "aster",
            "delivery": {
                "execution_mode": execution_mode,
                "time_plan_relative_path": "inputs/time_shards.json",
                "time_plan_identity": time_plan_identity,
            },
            "targets": [
                {"source_id_int64": 101},
                {"source_id_int64": 202},
            ],
        },
    )

    task_list_path = tmp_path / "remaining_tasks.json"
    if task_payload is None:
        task_payload = {
            "schema_id": "et_mainsim.science_stamp_task_list.v1",
            "schema_version": 1,
            "case": case,
            "production_manifest_identity": manifest_identity,
            "tasks": [
                {"source_id": 202, "shard_id": 9},
                {"source_id": 101, "shard_id": 4},
            ],
        }
    task_identity = _write_json(task_list_path, task_payload)

    fake_conda = tmp_path / "conda.sh"
    fake_conda.write_text(
        "conda() {\n"
        "  if [[ \"${1:-}\" != \"activate\" ]]; then return 2; fi\n"
        "  export TEST_CONDA_ENV=\"${2:-}\"\n"
        "}\n",
        encoding="utf-8",
    )
    command_log = tmp_path / "python-commands.log"
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "if [[ \"${1:-}\" == \"-\" ]]; then\n"
        "  /usr/bin/python3 \"$@\"\n"
        "  status=$?\n"
        "  if [[ -n \"${TEST_MUTATE_MANIFEST_AFTER_SELECTION:-}\" && ! -e \"${TEST_MANIFEST_MUTATION_SENTINEL}\" ]]; then\n"
        "    printf '\\n' >> \"${TEST_PRODUCTION_MANIFEST}\"\n"
        "    touch \"${TEST_MANIFEST_MUTATION_SENTINEL}\"\n"
        "  fi\n"
        "  exit \"${status}\"\n"
        "fi\n"
        "printf '%s\\n' \"$*\" >> \"${TEST_PYTHON_COMMAND_LOG}\"\n"
        "if [[ \"$*\" == *\" run-target \"* && -n \"${TEST_MUTATE_MANIFEST_AFTER_WORKER:-}\" ]]; then\n"
        "  printf '\\n' >> \"${TEST_PRODUCTION_MANIFEST}\"\n"
        "fi\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    for name in ("code", "photsim", "coord", "data", "scratch"):
        (tmp_path / name).mkdir()

    env = {
        **os.environ,
        "SLURM_ARRAY_TASK_ID": "0",
        "SLURM_JOB_ID": "1234",
        "ET_STAMP_MANIFEST": str(manifest_path),
        "ET_STAMP_CODE_ROOT": str(tmp_path / "code"),
        "ET_STAMP_PHOTSIM_ROOT": str(tmp_path / "photsim"),
        "ET_STAMP_ET_COORD_ROOT": str(tmp_path / "coord"),
        "ET_STAMP_DATA_ROOT": str(tmp_path / "data"),
        "ET_STAMP_FOCALPLANE_REGISTRY": str(tmp_path / "coord"),
        "ET_STAMP_CONDA_SH": str(fake_conda),
        "ET_STAMP_CONDA_ENV": "test-env",
        "ET_STAMP_PYTHON_BIN": str(fake_python),
        "ET_STAMP_LOCAL_SCRATCH_ROOT": str(tmp_path / "scratch"),
        "ET_STAMP_MIN_SCRATCH_KB": "1",
        "ET_STAMP_CASE": case,
        "ET_STAMP_TASK_LIST": str(task_list_path),
        "ET_STAMP_TASK_LIST_SHA256": str(task_identity["sha256"]),
        "TEST_PYTHON_COMMAND_LOG": str(command_log),
        "TEST_PRODUCTION_MANIFEST": str(manifest_path),
        "TEST_MANIFEST_MUTATION_SENTINEL": str(tmp_path / "manifest-mutated"),
    }
    return env, command_log


def test_science_cli_prepare_forwards_track_inputs_and_emits_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = _load_cli()
    captured = []
    fake_plan = SimpleNamespace(
        shards=(object(), object()),
        accepted_raw_frame_count=18,
    )
    fake_preparation = SimpleNamespace(
        manifest_path=tmp_path / "run" / "production_manifest.json",
        time_plan_path=tmp_path / "run" / "inputs" / "time_shards.json",
        time_plan=fake_plan,
    )
    monkeypatch.setattr(
        cli,
        "prepare_science_independent_production",
        lambda config: captured.append(config) or fake_preparation,
    )

    status = cli.main(
        [
            "prepare",
            "--track",
            "aster",
            "--input-root",
            str(tmp_path / "lcdata"),
            "--output-root",
            str(tmp_path / "results"),
            "--run-id",
            "aster-formal-v1",
            "--data-root",
            str(tmp_path / "photsim-data"),
            "--focalplane-registry",
            str(tmp_path / "focalplane"),
            "--external-source-id",
            "0000000473",
            "--external-source-id",
            "0000000599",
        ]
    )

    assert status == 0
    assert len(captured) == 1
    config = captured[0]
    assert config.track == "aster"
    assert config.input_root == (tmp_path / "lcdata").resolve()
    assert config.external_source_ids == ("0000000473", "0000000599")
    assert config.delivery_execution_mode == "staged_local_scratch_v1"
    assert json.loads(capsys.readouterr().out) == {
        "accepted_raw_frame_count": 18,
        "manifest_path": str(fake_preparation.manifest_path),
        "production_track": "aster",
        "shard_count": 2,
        "time_plan_path": str(fake_preparation.time_plan_path),
    }


def test_science_cli_run_target_forwards_scratch_and_emits_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = _load_cli()
    captured: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_run(*args: object, **kwargs: object):
        captured.append((args, kwargs))
        return ({"source_id": "42", "shard_id": 7, "case": "injected"},)

    monkeypatch.setattr(cli, "run_science_independent_target", fake_run)
    manifest = tmp_path / "production_manifest.json"
    scratch = tmp_path / "scratch" / "injected"

    status = cli.main(
        [
            "run-target",
            "--manifest",
            str(manifest),
            "--source-id",
            "42",
            "--case",
            "injected",
            "--shard-id",
            "7",
            "--shard-id",
            "8",
            "--output-root",
            str(scratch),
            "--device",
            "cuda",
            "--batch-size",
            "64",
        ]
    )

    assert status == 0
    assert captured == [
        (
            (str(manifest),),
            {
                "source_id": 42,
                "case": "injected",
                "shard_ids": [7, 8],
                "data_root": None,
                "focalplane_registry": None,
                "device": "cuda",
                "batch_size": 64,
                "output_root": str(scratch),
            },
        )
    ]
    assert json.loads(capsys.readouterr().out) == [
        {"case": "injected", "shard_id": 7, "source_id": "42"}
    ]


def test_science_cli_write_task_list_forwards_exact_tasks_and_emits_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = _load_cli()
    captured: list[tuple[tuple[object, ...], dict[str, object]]] = []
    output_path = tmp_path / "static_tasks.json"
    fake_result = SimpleNamespace(
        path=output_path.resolve(),
        case="static",
        task_count=2,
        identity={"sha256": "a" * 64, "size_bytes": 321},
    )

    def fake_write(*args: object, **kwargs: object):
        captured.append((args, kwargs))
        return fake_result

    monkeypatch.setattr(cli, "write_science_stamp_task_list", fake_write)

    status = cli.main(
        [
            "write-task-list",
            "--manifest",
            str(tmp_path / "production_manifest.json"),
            "--case",
            "static",
            "--task",
            "622:0",
            "--task",
            "36:4",
            "--output",
            str(output_path),
        ]
    )

    assert status == 0
    assert captured == [
        (
            (str(tmp_path / "production_manifest.json"),),
            {
                "case": "static",
                "tasks": ((622, 0), (36, 4)),
                "output_path": str(output_path),
            },
        )
    ]
    assert json.loads(capsys.readouterr().out) == {
        "case": "static",
        "path": str(output_path.resolve()),
        "sha256": "a" * 64,
        "size_bytes": 321,
        "task_count": 2,
    }


def test_science_staged_launcher_resolves_a_strict_remaining_task_list(
    tmp_path: Path,
) -> None:
    env, command_log = _write_launcher_fixture(tmp_path)

    completed = subprocess.run(
        ["bash", str(LAUNCHER_PATH)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "source_id=202 shard_id=9" in completed.stdout
    commands = command_log.read_text(encoding="utf-8").splitlines()
    assert len(commands) == 2
    assert commands[0].startswith(
        f"{env['ET_STAMP_CODE_ROOT']}/scripts/"
        "run_science_independent_stamp_production.py run-target"
    )
    assert "--source-id 202" in commands[0]
    assert "--shard-id 9" in commands[0]
    assert "--output-root " in commands[0]
    assert commands[1].startswith("-m et_mainsim.staged_stamp_delivery publish")
    assert "--source-id 202" in commands[1]
    assert "--shard-id 9" in commands[1]


def test_science_staged_launcher_defaults_to_manifest_cartesian_tasks(
    tmp_path: Path,
) -> None:
    env, command_log = _write_launcher_fixture(tmp_path)
    env["SLURM_ARRAY_TASK_ID"] = "3"
    env.pop("ET_STAMP_TASK_LIST")
    env.pop("ET_STAMP_TASK_LIST_SHA256")

    completed = subprocess.run(
        ["bash", str(LAUNCHER_PATH)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "source_id=202 shard_id=9" in completed.stdout
    assert "task_count=4 task_mode=manifest-cartesian" in completed.stdout
    commands = command_log.read_text(encoding="utf-8")
    assert "--source-id 202" in commands
    assert "--shard-id 9" in commands


def test_science_staged_launcher_requires_an_explicit_bound_task_list_for_static(
    tmp_path: Path,
) -> None:
    env, command_log = _write_launcher_fixture(tmp_path, case="static")
    env.pop("ET_STAMP_TASK_LIST")
    env.pop("ET_STAMP_TASK_LIST_SHA256")

    completed = subprocess.run(
        ["bash", str(LAUNCHER_PATH)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "static production requires an explicit case-bound task list" in (
        completed.stderr
    )
    assert not command_log.exists()


def test_science_staged_launcher_accepts_static_only_with_a_static_task_list(
    tmp_path: Path,
) -> None:
    env, command_log = _write_launcher_fixture(tmp_path, case="static")

    completed = subprocess.run(
        ["bash", str(LAUNCHER_PATH)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    commands = command_log.read_text(encoding="utf-8")
    assert "--case static" in commands


def test_science_staged_launcher_rejects_a_task_list_bound_to_another_case(
    tmp_path: Path,
) -> None:
    env, command_log = _write_launcher_fixture(tmp_path)
    payload = json.loads(Path(env["ET_STAMP_TASK_LIST"]).read_text())
    payload["case"] = "static"
    identity = _write_json(Path(env["ET_STAMP_TASK_LIST"]), payload)
    env["ET_STAMP_TASK_LIST_SHA256"] = str(identity["sha256"])

    completed = subprocess.run(
        ["bash", str(LAUNCHER_PATH)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "task list case differs from ET_STAMP_CASE" in completed.stderr
    assert not command_log.exists()


@pytest.mark.parametrize(
    "payload_mutation",
    (
        lambda payload: payload.update(schema_version=True),
        lambda payload: payload["production_manifest_identity"].update(
            unexpected="field"
        ),
    ),
)
def test_science_staged_launcher_rejects_nonexact_task_list_schema_types(
    tmp_path: Path,
    payload_mutation,
) -> None:
    env, command_log = _write_launcher_fixture(tmp_path)
    payload = json.loads(Path(env["ET_STAMP_TASK_LIST"]).read_text())
    payload_mutation(payload)
    identity = _write_json(Path(env["ET_STAMP_TASK_LIST"]), payload)
    env["ET_STAMP_TASK_LIST_SHA256"] = str(identity["sha256"])

    completed = subprocess.run(
        ["bash", str(LAUNCHER_PATH)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert __import__("re").search(r"task[- ]list", completed.stderr)
    assert not command_log.exists()


@pytest.mark.parametrize(
    ("task_payload", "expected_error"),
    [
        (
            {
                "schema_id": "et_mainsim.science_stamp_task_list.v1",
                "schema_version": 1,
                "case": "injected",
                "production_manifest_identity": {},
                "tasks": [
                    {"source_id": 101, "shard_id": 4},
                    {"source_id": 101, "shard_id": 4},
                ],
            },
            "production manifest identity differs|duplicate task",
        ),
        (
            {
                "schema_id": "et_mainsim.science_stamp_task_list.v1",
                "schema_version": 1,
                "case": "injected",
                "production_manifest_identity": {},
                "tasks": [{"source_id": 999, "shard_id": 4}],
            },
            "production manifest identity differs|unknown source_id",
        ),
    ],
)
def test_science_staged_launcher_rejects_invalid_remaining_tasks_before_worker(
    tmp_path: Path,
    task_payload: object,
    expected_error: str,
) -> None:
    env, command_log = _write_launcher_fixture(
        tmp_path,
        task_payload=task_payload,
    )
    # Rebind the placeholder identity so each case reaches task validation.
    manifest_raw = Path(env["ET_STAMP_MANIFEST"]).read_bytes()
    payload = json.loads(Path(env["ET_STAMP_TASK_LIST"]).read_text())
    payload["production_manifest_identity"] = {
        "sha256": _sha256_bytes(manifest_raw),
        "size_bytes": len(manifest_raw),
    }
    identity = _write_json(Path(env["ET_STAMP_TASK_LIST"]), payload)
    env["ET_STAMP_TASK_LIST_SHA256"] = str(identity["sha256"])

    completed = subprocess.run(
        ["bash", str(LAUNCHER_PATH)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert __import__("re").search(expected_error, completed.stderr)
    assert not command_log.exists()


def test_science_staged_launcher_rejects_task_list_identity_drift(
    tmp_path: Path,
) -> None:
    env, command_log = _write_launcher_fixture(tmp_path)
    env["ET_STAMP_TASK_LIST_SHA256"] = "0" * 64

    completed = subprocess.run(
        ["bash", str(LAUNCHER_PATH)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "task list SHA-256 differs" in completed.stderr
    assert not command_log.exists()


def test_science_staged_launcher_rejects_manifest_drift_before_worker(
    tmp_path: Path,
) -> None:
    env, command_log = _write_launcher_fixture(tmp_path)
    env["TEST_MUTATE_MANIFEST_AFTER_SELECTION"] = "1"

    completed = subprocess.run(
        ["bash", str(LAUNCHER_PATH)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "production manifest changed after task selection" in completed.stderr
    assert not command_log.exists()


def test_science_staged_launcher_rejects_manifest_drift_before_publisher(
    tmp_path: Path,
) -> None:
    env, command_log = _write_launcher_fixture(tmp_path)
    env["TEST_MUTATE_MANIFEST_AFTER_WORKER"] = "1"

    completed = subprocess.run(
        ["bash", str(LAUNCHER_PATH)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "production manifest changed after task selection" in completed.stderr
    commands = command_log.read_text(encoding="utf-8").splitlines()
    assert len(commands) == 1
    assert " run-target " in f" {commands[0]} "


def test_science_staged_launcher_rejects_direct_manifest_before_worker(
    tmp_path: Path,
) -> None:
    env, command_log = _write_launcher_fixture(
        tmp_path,
        execution_mode="direct_shared_filesystem",
    )

    completed = subprocess.run(
        ["bash", str(LAUNCHER_PATH)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "requires delivery.execution_mode='staged_local_scratch_v1'" in (
        completed.stderr
    )
    assert not command_log.exists()


def test_science_staged_launcher_is_valid_bash_and_has_no_galaxy_contract() -> None:
    completed = subprocess.run(
        ["bash", "-n", str(LAUNCHER_PATH)],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    script = LAUNCHER_PATH.read_text(encoding="utf-8")
    assert "run_science_independent_stamp_production.py" in script
    assert "publish_staged_independent_stamp_shard" in script
    assert "galaxy" not in script.lower()
    assert "ET_STAMP_TASK_LIST" in script
    assert "ET_STAMP_TASK_LIST_SHA256" in script
    assert "staged_local_scratch_v1" in script
    assert "direct_shared_filesystem" in script
