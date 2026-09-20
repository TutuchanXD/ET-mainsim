"""Explicit S3 acceptance: real workbook/assets, small CPU and CUDA scenes.

Run with ET_DATA_DIR, ET_S3_WORKBOOK and ET_S3_DEVICE=cpu|cuda. This suite is
outside ordinary unit-test discovery because it requires external assets.
"""

from dataclasses import replace
import os
from pathlib import Path

import numpy as np
from astropy import units as u
import pytest

from et_mainsim.presets import load_preset
from photsim7.specs import load_simulation_spec


def _plan(tmp_path, workflow, run_id, *, table=False, readout_seed=23):
    from importlib import import_module

    module = import_module(f"et_mainsim.workflows.{workflow}")
    device = os.environ.get("ET_S3_DEVICE", "cpu")
    root = Path(os.environ["ET_DATA_DIR"])
    registry = Path(os.environ["ET_FOCALPLANE_ROOT"]) / "data"
    workbook = Path(os.environ.get("ET_S3_WORKBOOK", root / "default.xlsx"))
    preset = load_preset(
        "et-full-frame-smoke" if workflow == "full_frame" else "et-stamp-smoke"
    )
    spec = load_simulation_spec(workbook, base=preset.simulation_spec)
    for effect in (
        "psd_motion",
        "psf_breathing",
        "thermal_drift",
        "dva",
        "momentum_dump",
    ):
        assert getattr(spec.dynamic_effects, effect).enabled
    spec = replace(
        spec,
        observation=replace(
            spec.observation,
            observing_duration=40 * u.s,
            n_frames=4,
            frame_start_s=None,
            n_raw_frames_per_coadd=1 if workflow == "full_frame" else 2,
        ),
        instrument=replace(spec.instrument, telescope_count=1),
        detector=replace(spec.detector, shape=(31, 31)),
        psf=replace(
            spec.psf,
            compute_device=device,
            n_jitter_integrated_psf_models=2,
            field_id="nearest",
            field_id_policy="nearest",
        ),
        catalog=replace(
            spec.catalog,
            target_ra_deg=304.41406499712303,
            target_dec_deg=51.81987707392268,
            target_detector_xpix=4450.0,
            target_detector_ypix=4560.0,
        ),
        rng=replace(
            spec.rng,
            run_seed=17,
            stream_seeds={**spec.rng.stream_seeds, "readout.gaussian": readout_seed},
        ),
    )
    config = replace(
        preset.run_config,
        run_id=run_id,
        paths=replace(
            preset.run_config.paths,
            data_root=str(root),
            output_root=str(tmp_path),
            focalplane_registry=str(registry),
        ),
        execution=replace(
            preset.run_config.execution,
            device=device,
            backend="local-subprocess" if device == "cuda" else "in-process",
            gpu_ids=("0",) if device == "cuda" else (),
            preview_count=0,
        ),
    )
    target = tmp_path / "targets.csv"
    target.write_text(
        "source_id,gaia_g_mag,ra_deg,dec_deg\n1,12.0,304.41406499712303,51.81987707392268\n"
    )
    from et_mainsim.config import StampWorkload

    table_workload = StampWorkload(
        input_mode="table",
        input_table=str(target),
        include_neighbors=False,
        target_source_ids=(),
        target_limit=0,
    )
    if table:
        config = replace(
            config,
            workload=table_workload,
        )
    else:
        # A one-star physical ET field prepared with the real focal-plane
        # transform. DVA must not be exercised with fabricated reference geometry.
        from et_mainsim.workflows import stamp
        from photsim7.pipelines import build_catalog_from_spec
        from photsim7.data_registry import DataRegistry

        stamp_preset = load_preset("et-stamp-smoke")
        table_config = replace(
            stamp_preset.run_config,
            paths=config.paths,
            execution=config.execution,
            workload=table_workload,
        )
        table_plan = stamp.build_run_plan(
            preset_name=stamp_preset.descriptor.name,
            run_config=table_config,
            spec=replace(spec, psf=replace(spec.psf, mode="stamp")),
            repo_root=Path(__file__).parents[1],
        )
        prepared = stamp.prepare_stamp_inputs(table_plan).catalogs[1]
        cache = tmp_path / "physical_stars.npz"
        spec = replace(
            spec,
            catalog=replace(
                spec.catalog,
                source_type="prepared",
                source_path="",
                registry_data_dir=str(registry),
                cache_path=str(cache),
            ),
        )
        config = replace(config, paths=replace(config.paths, catalog_cache=str(cache)))
        build_catalog_from_spec(
            spec, data_registry=DataRegistry(root), prepared_catalog=prepared
        )
    # Canonical JSON replay must retain the exact workbook-derived specification.
    replay = tmp_path / f"{run_id}.json"
    replay.write_text(spec.to_json())
    spec = load_simulation_spec(replay, base=preset.simulation_spec)
    plan = module.build_run_plan(
        preset_name=preset.descriptor.name,
        run_config=config,
        spec=spec,
        repo_root=Path(__file__).parents[1],
    )
    return module, plan


def _arrays(plan, workflow):
    if workflow == "full_frame":
        return [
            np.load(path) for path in sorted((plan.run_dir / "frames").glob("*.npy"))
        ]
    from photsim7.artifacts import StampShardReader

    result = []
    for path in sorted((plan.run_dir / "stamps").glob("target_*/raw.h5")):
        with StampShardReader(path) as reader:
            result.extend(
                reader.read_stamp(star, frame)
                for star in reader.star_ids
                for frame in reader.frame_ids
            )
    return result


@pytest.mark.parametrize(
    "workflow,table", [("full_frame", False), ("stamp", False), ("stamp", True)]
)
def test_real_workbook_effects_repeat_resume_and_seed_control(
    tmp_path, workflow, table
):
    module, plan = _plan(tmp_path, workflow, "first", table=table)
    run = module.run_full_frame if workflow == "full_frame" else module.run_stamp
    first = run(plan)
    assert first["status"] == "completed"
    a = _arrays(plan, workflow)
    assert len(a) == 4
    resumed = run(plan)
    assert resumed["status"] == "completed"
    _, repeat = _plan(tmp_path, workflow, "repeat", table=table)
    run(repeat)
    b = _arrays(repeat, workflow)
    assert len(b) == len(a)
    for left, right in zip(a, b):
        np.testing.assert_array_equal(left, right)
    _, changed = _plan(tmp_path, workflow, "changed", table=table, readout_seed=29)
    run(changed)
    c = _arrays(changed, workflow)
    assert any(not np.array_equal(left, right) for left, right in zip(a, c))
