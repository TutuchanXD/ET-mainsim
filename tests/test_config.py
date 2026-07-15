from __future__ import annotations

import pytest


BASE_TOML = """
schema_id = "et_mainsim.execution_config"
schema_version = 1
workflow = "et-full-frame"
run_id = "unit-run"

[paths]
output_root = ""
data_root = ""
catalog_path = ""
focalplane_registry = ""

[execution]
backend = "local-subprocess"
device = "cuda"
gpu_ids = ["2", "4"]
workers_per_device = 2
resume = true
overwrite = false
force_catalog_cache = false
preview_count = 2
progress = false
save_cosmic_mask = false
save_stellar_mean = false
"""


def test_run_config_resolves_environment_paths_without_hostname_fallback(
    tmp_path,
) -> None:
    from et_mainsim.config import RunConfig

    config = RunConfig.from_toml(BASE_TOML, source="unit.toml")
    paths = config.resolve_paths(
        env={
            "RESULTS_ROOT": str(tmp_path / "results"),
            "ET_DATA_DIR": str(tmp_path / "data"),
            "GAIA_CATALOG_DIR": str(tmp_path / "gaia"),
            "ET_FOCALPLANE_ROOT": str(tmp_path / "focalplane"),
        },
        cwd=tmp_path / "cwd",
    )

    assert paths.output_root == tmp_path / "results"
    assert paths.data_root == tmp_path / "data"
    assert paths.catalog_path == tmp_path / "gaia"
    assert paths.focalplane_registry == tmp_path / "focalplane" / "data"


def test_run_config_uses_portable_output_default_and_keeps_required_assets_explicit(
    tmp_path,
) -> None:
    from et_mainsim.config import RunConfig

    paths = RunConfig.from_toml(BASE_TOML).resolve_paths(env={}, cwd=tmp_path)

    assert paths.output_root == tmp_path / "results" / "et-mainsim"
    assert paths.data_root is None
    assert paths.catalog_path is None
    assert paths.focalplane_registry is None


def test_run_config_expands_explicit_environment_references(tmp_path) -> None:
    from et_mainsim.config import RunConfig

    text = BASE_TOML.replace(
        'output_root = ""',
        'output_root = "${PROJECT_ROOT}/products"',
    )
    config = RunConfig.from_toml(text)

    paths = config.resolve_paths(
        env={"PROJECT_ROOT": str(tmp_path / "project")},
        cwd=tmp_path,
    )

    assert paths.output_root == tmp_path / "project" / "products"
    with pytest.raises(ValueError, match="unset environment variable PROJECT_ROOT"):
        config.resolve_paths(env={}, cwd=tmp_path)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        (
            "resume = true\noverwrite = false",
            "resume = true\noverwrite = true",
            "mutually exclusive",
        ),
        ("workers_per_device = 2", "workers_per_device = 0", "positive"),
        ('device = "cuda"', 'device = "tpu"', "device"),
    ],
)
def test_run_config_rejects_ambiguous_or_invalid_execution(old, new, message) -> None:
    from et_mainsim.config import RunConfig

    text = BASE_TOML.replace(old, new)

    with pytest.raises(ValueError, match=message):
        RunConfig.from_toml(text)


def test_in_process_backend_is_cpu_only() -> None:
    from et_mainsim.config import RunConfig

    text = BASE_TOML.replace(
        'backend = "local-subprocess"',
        'backend = "in-process"',
    ).replace("workers_per_device = 2", "workers_per_device = 1")

    with pytest.raises(ValueError, match="in-process.*CPU"):
        RunConfig.from_toml(text)


@pytest.mark.parametrize(
    ("workflow", "message"),
    [(None, "workflow must be non-empty"), ("unknown", "workflow must be one of")],
)
def test_run_config_validates_workflow_before_workload(
    workflow: str | None,
    message: str,
) -> None:
    from et_mainsim.config import RunConfig

    payload = {
        "schema_id": "et_mainsim.execution_config",
        "schema_version": 1,
        "run_id": "invalid-workflow",
    }
    if workflow is not None:
        payload["workflow"] = workflow

    with pytest.raises(ValueError, match=message):
        RunConfig.from_mapping(payload)


def test_frame_selection_and_worker_assignments_are_stable() -> None:
    from et_mainsim.config import (
        RunConfig,
        parse_frame_indices,
        worker_assignments,
    )

    config = RunConfig.from_toml(BASE_TOML)

    assert parse_frame_indices(None, total_frames=4) == (0, 1, 2, 3)
    assert parse_frame_indices("0,3,3", total_frames=4) == (0, 3)
    assert [item.visible_device for item in worker_assignments(config.execution)] == [
        "2",
        "2",
        "4",
        "4",
    ]

    with pytest.raises(ValueError, match="outside"):
        parse_frame_indices("4", total_frames=4)


def test_stamp_workload_is_typed_and_table_mode_is_query_independent() -> None:
    from et_mainsim.config import RunConfig, StampWorkload

    text = """
schema_id = "et_mainsim.execution_config"
schema_version = 1
workflow = "et-stamp"
run_id = "table-stamps"

[workload]
kind = "stamp"
input_mode = "table"
input_table = "targets.csv"
variability_table = "curves.ecsv"
stamp_rows = 15
stamp_cols = 17
include_neighbors = false
save_raw = true
save_coadd = true
artifact_profile = "compact"
write_batch_size = 64

[execution]
backend = "in-process"
device = "cpu"
"""
    config = RunConfig.from_toml(text)

    assert isinstance(config.workload, StampWorkload)
    assert config.workload.input_mode == "table"
    assert config.workload.input_table == "targets.csv"
    assert config.workload.variability_table == "curves.ecsv"
    assert config.workload.stamp_shape == (15, 17)
    assert config.workload.artifact_profile == "compact"
    assert config.workload.write_batch_size == 64
    assert config.to_dict()["workload"]["kind"] == "stamp"


@pytest.mark.parametrize(
    ("workload", "message"),
    [
        (
            'kind = "stamp"\ninput_mode = "table"\ninclude_neighbors = false',
            "input_table",
        ),
        (
            'kind = "stamp"\ninput_mode = "table"\ninput_table = "x.csv"\ninclude_neighbors = true',
            "include_neighbors",
        ),
        (
            'kind = "stamp"\ninput_mode = "catalog"\nvariability_table = "curves.csv"',
            "variability_table",
        ),
        (
            'kind = "stamp"\nartifact_profile = "brief"',
            "artifact_profile",
        ),
        (
            'kind = "stamp"\nwrite_batch_size = 0',
            "write_batch_size",
        ),
        ('kind = "legacy"', "match workflow"),
    ],
)
def test_stamp_workload_rejects_ambiguous_contracts(workload, message) -> None:
    from et_mainsim.config import RunConfig

    text = f"""
schema_id = "et_mainsim.execution_config"
schema_version = 1
workflow = "et-stamp"
run_id = "bad"

[workload]
{workload}

[execution]
backend = "in-process"
device = "cpu"
"""
    with pytest.raises(ValueError, match=message):
        RunConfig.from_toml(text)


def test_legacy_workload_uses_explicit_local_ray_resources() -> None:
    from et_mainsim.config import LegacyWorkload, RunConfig

    text = """
schema_id = "et_mainsim.execution_config"
schema_version = 1
workflow = "legacy-sim"
run_id = "legacy"

[workload]
kind = "legacy"
run_count = 2
stars_per_run = 100
store_images = false
et_mag_min = 7.0
et_mag_max = 17.0

[execution]
backend = "local-ray"
device = "cuda"
gpu_ids = ["0"]
ray_actor_count = 1
ray_num_cpus = 1
ray_num_gpus = 1
"""
    config = RunConfig.from_toml(text)

    assert isinstance(config.workload, LegacyWorkload)
    assert config.workload.run_count == 2
    assert config.workload.stars_per_run == 100
    assert config.execution.ray_actor_count == 1
    assert config.execution.ray_num_gpus == 1


@pytest.mark.parametrize(
    ("field_name", "value"),
    [("ray_num_cpus", 1.5), ("ray_num_gpus", 0.5)],
)
def test_local_ray_resources_reject_fractional_values(field_name, value) -> None:
    from et_mainsim.config import ExecutionConfig

    with pytest.raises(ValueError, match=f"{field_name} must be an integer"):
        ExecutionConfig(backend="local-ray", **{field_name: value})
