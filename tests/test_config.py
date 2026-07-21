from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

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
write_batch_size = 7

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
    assert config.workload.write_batch_size == 7
    assert config.to_dict()["workload"]["kind"] == "stamp"
    assert config.to_dict()["workload"]["write_batch_size"] == 7


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
        ('kind = "stamp"\nwrite_batch_size = 0', "write_batch_size"),
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


def test_full_frame_shared_exposure_defaults_are_frozen_and_identity_bearing() -> None:
    from et_mainsim.config import (
        FullFrameWorkload,
        RunConfig,
        SharedExposureStampsConfig,
    )

    config = RunConfig.from_toml(BASE_TOML)

    assert isinstance(config.workload, FullFrameWorkload)
    shared = config.workload.shared_exposure_stamps
    assert isinstance(shared, SharedExposureStampsConfig)
    assert shared.enabled is False
    assert shared.target_source_ids == ()
    assert shared.stamp_rows == 100
    assert shared.stamp_cols == 300
    assert shared.stamp_shape == (100, 300)
    assert shared.frames_per_shard == 32
    assert shared.product_keys == ("final_stamp",)
    assert config.workload.to_dict() == {
        "kind": "full-frame",
        "shared_exposure_stamps": {
            "enabled": False,
            "target_source_ids": [],
            "stamp_rows": 100,
            "stamp_cols": 300,
            "frames_per_shard": 32,
            "product_keys": ["final_stamp"],
        },
    }
    with pytest.raises(FrozenInstanceError):
        shared.enabled = True


def test_full_frame_shared_exposure_parses_nested_contract_without_reordering() -> None:
    from et_mainsim.config import RunConfig

    text = (
        BASE_TOML
        + """

[workload]
kind = "full-frame"

[workload.shared_exposure_stamps]
enabled = true
target_source_ids = [9003, 17, -4]
stamp_rows = 100
stamp_cols = 300
frames_per_shard = 8
product_keys = [
  "final_stamp",
  "electron_stamp",
  "electron_components.stellar_mean",
]
"""
    )

    config = RunConfig.from_toml(text)
    shared = config.workload.shared_exposure_stamps

    assert shared.enabled is True
    assert shared.target_source_ids == (9003, 17, -4)
    assert shared.frames_per_shard == 8
    assert shared.product_keys == (
        "final_stamp",
        "electron_stamp",
        "electron_components.stellar_mean",
    )
    assert config.to_dict()["workload"]["shared_exposure_stamps"] == {
        "enabled": True,
        "target_source_ids": [9003, 17, -4],
        "stamp_rows": 100,
        "stamp_cols": 300,
        "frames_per_shard": 8,
        "product_keys": [
            "final_stamp",
            "electron_stamp",
            "electron_components.stellar_mean",
        ],
    }


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"enabled": 1}, "enabled must be a boolean"),
        ({"target_source_ids": "12"}, "target_source_ids must be a sequence"),
        ({"target_source_ids": [True]}, "signed 64-bit integers"),
        ({"target_source_ids": [1.5]}, "signed 64-bit integers"),
        ({"target_source_ids": [2**63]}, "signed 64-bit integers"),
        ({"target_source_ids": [11, 11]}, "target_source_ids must be unique"),
        ({"stamp_rows": True}, "stamp_rows must be a positive integer"),
        ({"frames_per_shard": True}, "frames_per_shard must be a positive integer"),
        ({"frames_per_shard": 0}, "frames_per_shard must be a positive integer"),
        ({"stamp_rows": 1.5}, "stamp_rows must be a positive integer"),
        ({"stamp_rows": 0}, "stamp_rows must be a positive integer"),
        ({"stamp_cols": -1}, "stamp_cols must be a positive integer"),
        ({"product_keys": "final_stamp"}, "product_keys must be a sequence"),
        ({"product_keys": [1, "final_stamp"]}, "non-empty strings"),
        ({"product_keys": [" final_stamp"]}, "surrounding whitespace"),
        (
            {"product_keys": ["final_stamp", "final_stamp"]},
            "product_keys must be unique",
        ),
        ({"product_keys": ["electron_stamp"]}, "must include 'final_stamp'"),
        (
            {"product_keys": ["final_stamp", "unsupported"]},
            "unsupported shared-exposure product key",
        ),
        (
            {"product_keys": ["final_stamp", "electron_components."]},
            "unsupported shared-exposure product key",
        ),
        (
            {"product_keys": ["final_stamp", "electron_components.stellar.mean"]},
            "unsupported shared-exposure product key",
        ),
        (
            {"product_keys": ["final_stamp", "electron_components. stellar"]},
            "unsupported shared-exposure product key",
        ),
        ({"enabled": True}, "enabled shared-exposure stamps require"),
    ],
)
def test_shared_exposure_contract_rejects_ambiguous_values(overrides, message) -> None:
    from et_mainsim.config import SharedExposureStampsConfig

    with pytest.raises(ValueError, match=message):
        SharedExposureStampsConfig(**overrides)


def test_shared_exposure_contract_accepts_every_upstream_product_key_form() -> None:
    from et_mainsim.config import SharedExposureStampsConfig

    product_keys = (
        "final_stamp",
        "electron_stamp",
        "adu_stamp_pre_adc",
        "dn_stamp",
        "cosmic_events.mask",
        "electron_components.stellar_mean",
    )

    assert SharedExposureStampsConfig(product_keys=product_keys).product_keys == (
        product_keys
    )


@pytest.mark.parametrize(
    ("shared_payload", "message"),
    [
        ([], "shared_exposure_stamps must be a mapping"),
        ({"future_field": 1}, "Unknown shared_exposure_stamps fields: future_field"),
    ],
)
def test_run_config_rejects_invalid_nested_shared_exposure_mapping(
    shared_payload,
    message,
) -> None:
    from et_mainsim.config import RunConfig

    payload = {
        "schema_id": "et_mainsim.execution_config",
        "schema_version": 1,
        "workflow": "et-full-frame",
        "run_id": "nested-invalid",
        "workload": {
            "kind": "full-frame",
            "shared_exposure_stamps": shared_payload,
        },
    }

    with pytest.raises(ValueError, match=message):
        RunConfig.from_mapping(payload)


def test_shared_exposure_nested_contract_participates_in_resume_identity(
    tmp_path,
) -> None:
    from et_mainsim.config import RunConfig
    from et_mainsim.manifest import ManifestIdentityError, RunManifestStore

    base = RunConfig.from_toml(BASE_TOML)
    shared = replace(
        base.workload.shared_exposure_stamps,
        enabled=True,
        target_source_ids=(101, 202),
        product_keys=("final_stamp", "electron_stamp"),
    )
    workload = replace(base.workload, shared_exposure_stamps=shared).to_dict()
    store = RunManifestStore(tmp_path / "run_manifest.json")
    store.create(
        workflow=base.workflow,
        preset="unit",
        run_id=base.run_id,
        simulation_spec={"schema": "unit"},
        execution=base.execution.to_dict(),
        workload=workload,
        frame_plan={"requested": [0]},
        provenance={},
    )

    store.ensure_identity(
        workflow=base.workflow,
        run_id=base.run_id,
        simulation_spec={"schema": "unit"},
        execution=base.execution.to_dict(),
        workload=workload,
    )

    changed_targets = replace(
        base.workload,
        shared_exposure_stamps=replace(shared, target_source_ids=(101, 303)),
    ).to_dict()
    with pytest.raises(ManifestIdentityError, match="workload identity"):
        store.ensure_identity(
            workflow=base.workflow,
            run_id=base.run_id,
            simulation_spec={"schema": "unit"},
            execution=base.execution.to_dict(),
            workload=changed_targets,
        )

    changed_batching = replace(
        base.workload,
        shared_exposure_stamps=replace(shared, frames_per_shard=64),
    ).to_dict()
    with pytest.raises(ManifestIdentityError, match="workload identity"):
        store.ensure_identity(
            workflow=base.workflow,
            run_id=base.run_id,
            simulation_spec={"schema": "unit"},
            execution=base.execution.to_dict(),
            workload=changed_batching,
        )
