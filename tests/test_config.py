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
