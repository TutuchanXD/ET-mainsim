from __future__ import annotations

import json
import pickle
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest


def _plan(tmp_path):
    from et_mainsim.presets import load_preset
    from et_mainsim.workflows.legacy import build_run_plan

    loaded = load_preset("legacy-sim-full-effects-smoke")
    data_root = tmp_path / "data"
    data_root.mkdir()
    config = replace(
        loaded.run_config,
        paths=replace(
            loaded.run_config.paths,
            output_root=str(tmp_path / "output"),
            data_root=str(data_root),
        ),
    )
    return build_run_plan(
        preset_name=loaded.descriptor.name,
        run_config=config,
        contract=loaded.science_contract,
        repo_root=tmp_path,
    )


class _FakeSimulator:
    def __init__(self, calls, contract, manifest_mutator=None):
        self.calls = calls
        self.contract = contract
        self.manifest_mutator = manifest_mutator

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def run(self, **kwargs):
        self.calls.append(("run", kwargs))
        output = kwargs["sim_save_dir"]
        for run_index in range(kwargs["run_count"]):
            run_dir = output / f"run_{run_index}"
            run_dir.mkdir(parents=True)
            payloads = {
                "time.pkl": np.arange(2),
                "psf_field_ids.pkl": np.array([0]),
                "light_curves.pkl": np.ones((1, 1, 1, 2)),
                "centroids.pkl": np.ones((1, 1, 1, 2, 2)),
                "apertures.pkl": np.ones((1, 1, 1, 9, 9)),
                "variants_settings.pkl": {},
                "stars_metadata_df.pkl": {"ET Mag": [12.0], "Star ID": [0]},
                "sim_config.pkl": {},
                "time_manager.pkl": {},
                "telescope_xy_offsets.pkl": np.zeros((1, 2)),
                "dynamic_param_data.pkl": {},
                "dynamic_param_config.pkl": {},
            }
            for name, value in payloads.items():
                with (run_dir / name).open("wb") as handle:
                    pickle.dump(value, handle)
            manifest = deepcopy(self.contract.to_metadata())
            if self.manifest_mutator is not None:
                self.manifest_mutator(manifest)
            (run_dir / "legacy_effect_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )


def _fake_api(calls, *, manifest_mutator=None):
    class Runtime:
        def __init__(self, contract):
            self.contract = contract

        def build_simulator(self, **kwargs):
            calls.append(("build_simulator", kwargs))
            return _FakeSimulator(
                calls,
                self.contract,
                manifest_mutator=manifest_mutator,
            )

    return SimpleNamespace(
        DataRegistry=lambda **kwargs: SimpleNamespace(**kwargs),
        build_runtime=lambda contract, **kwargs: Runtime(contract),
        read_effect_manifest=lambda path: json.loads(path.read_text(encoding="utf-8")),
    )


def _rename_thermal_effect(manifest) -> None:
    thermal = next(
        effect
        for effect in manifest["effects"]
        if effect["effect_id"] == "motion.thermal"
    )
    thermal["effect_id"] = "motion.thermal.renamed"


def _change_thermal_effect_payload(manifest) -> None:
    thermal = next(
        effect
        for effect in manifest["effects"]
        if effect["effect_id"] == "motion.thermal"
    )
    thermal["parameters"]["profile"] = "tampered-profile"


def test_legacy_run_validates_outputs_and_whole_workload_resume(tmp_path) -> None:
    from et_mainsim.workflows.legacy import run_legacy

    plan = _plan(tmp_path)
    calls = []

    first = run_legacy(plan, science_api=_fake_api(calls))

    assert first["status"] == "completed"
    assert first["completion"]["rendered_runs"] == 1
    run_call = next(value for name, value in calls if name == "run")
    assert run_call["run_count"] == 1
    assert run_call["n_stars_per_run"] == 1
    assert run_call["resume"] is False
    assert len(run_call["user_star_field"]) == 1
    assert 12.0 <= run_call["user_star_field"][0]["et_mag"] <= 12.0

    calls.clear()
    second = run_legacy(plan, science_api=_fake_api(calls))

    assert second["completion"]["rendered_runs"] == 0
    assert second["completion"]["skipped_runs"] == 1
    assert calls == []


@pytest.mark.parametrize(
    "manifest_mutator",
    [_rename_thermal_effect, _change_thermal_effect_payload],
    ids=["effect-id", "effect-payload"],
)
def test_legacy_output_validation_rejects_effect_contract_drift(
    tmp_path,
    manifest_mutator,
) -> None:
    from et_mainsim.workflows.legacy import _validate_run

    plan = _plan(tmp_path)
    plan.legacy_root.mkdir(parents=True)
    _FakeSimulator(
        [],
        plan.contract,
        manifest_mutator=manifest_mutator,
    ).run(
        sim_save_dir=plan.legacy_root,
        run_count=1,
        n_stars_per_run=1,
    )

    summary = _validate_run(
        plan.legacy_root / "run_0",
        workload=plan.workload,
        api=_fake_api([]),
        expected_effects=plan.contract.to_metadata()["effects"],
    )

    assert summary is None


def test_legacy_resume_fails_closed_on_partial_completed_run(tmp_path) -> None:
    from et_mainsim.workflows.legacy import run_legacy

    plan = _plan(tmp_path)
    run_legacy(plan, science_api=_fake_api([]))
    (plan.legacy_root / "run_0" / "apertures.pkl").unlink()

    with pytest.raises(RuntimeError, match="partial legacy output"):
        run_legacy(plan, science_api=_fake_api([]))


def test_legacy_resume_rejects_metadata_outside_requested_et_mag_range(
    tmp_path,
) -> None:
    from et_mainsim.workflows.legacy import run_legacy

    plan = _plan(tmp_path)
    run_legacy(plan, science_api=_fake_api([]))
    metadata_path = plan.legacy_root / "run_0" / "stars_metadata_df.pkl"
    with metadata_path.open("wb") as handle:
        pickle.dump({"ET Mag": [13.0], "Star ID": [0]}, handle)

    with pytest.raises(RuntimeError, match="partial legacy output"):
        run_legacy(plan, science_api=_fake_api([]))
