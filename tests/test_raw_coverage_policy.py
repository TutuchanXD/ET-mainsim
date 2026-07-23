from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_formal_manifest(tmp_path: Path) -> Path:
    from et_mainsim.galaxy_stamp_production import (
        GALAXY_STAMP_PRODUCTION_SCHEMA_ID,
        GALAXY_STAMP_PRODUCTION_SCHEMA_VERSION,
    )
    from et_mainsim.stamp_inputs import file_identity

    run_root = tmp_path / "formal_run"
    inputs = run_root / "inputs"
    inputs.mkdir(parents=True)
    time_plan = inputs / "time_shards.json"
    time_plan.write_text("{}\n", encoding="utf-8")
    manifest_path = run_root / "production_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_id": GALAXY_STAMP_PRODUCTION_SCHEMA_ID,
                "schema_version": GALAXY_STAMP_PRODUCTION_SCHEMA_VERSION,
                "run_id": "policy-fixture",
                "observation_product": "final_dn",
                "background_realization_delivered": False,
                "delivery": {
                    "raw_exposure_seconds": 10.0,
                    "time_plan_relative_path": "inputs/time_shards.json",
                    "time_plan_identity": file_identity(time_plan),
                },
                "targets": [
                    {"source_id_int64": source_id}
                    for source_id in range(10, 20)
                ],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest_path


def test_frozen_raw_coverage_policy_is_manifest_bound_and_immutable(
    tmp_path: Path,
) -> None:
    from et_mainsim.raw_coverage_policy import (
        FrozenRawCoveragePolicyRequest,
        RAW_COVERAGE_POLICY_SCHEMA_ID,
        load_frozen_raw_coverage_policy_v1,
        write_frozen_raw_coverage_policy_v1,
    )

    manifest_path = _write_formal_manifest(tmp_path)
    policy_path = manifest_path.parent / "analysis" / "raw_10s_coverage_v2_policy.json"
    policy = write_frozen_raw_coverage_policy_v1(
        FrozenRawCoveragePolicyRequest(
            production_manifest_path=manifest_path,
            output_path=policy_path,
            minimum_coverage_fraction=0.95,
            minimum_accepted_bins=10,
        )
    )

    assert policy.path == policy_path.resolve()
    assert policy.schema_id == RAW_COVERAGE_POLICY_SCHEMA_ID
    assert policy.run_id == "policy-fixture"
    assert policy.case == "injected"
    assert policy.windows_minutes == (30, 90, 390)
    assert policy.minimum_coverage_fraction == pytest.approx(0.95)
    assert policy.minimum_accepted_bins == 10
    assert policy.path_identity["sha256"] == policy.identity["sha256"]
    payload = json.loads(policy_path.read_text(encoding="utf-8"))
    assert payload["complete"] is True
    assert payload["coverage"]["invalid_cadence_handling"] == (
        "omit_whole_invalid_cadences_without_pixel_or_flux_imputation"
    )
    assert (
        load_frozen_raw_coverage_policy_v1(policy_path).identity["sha256"]
        == policy.identity["sha256"]
    )
    with pytest.raises(FileExistsError, match="already exists"):
        write_frozen_raw_coverage_policy_v1(
            FrozenRawCoveragePolicyRequest(
                production_manifest_path=manifest_path,
                output_path=policy_path,
                minimum_coverage_fraction=0.95,
                minimum_accepted_bins=10,
            )
        )


def test_frozen_raw_coverage_policy_rejects_noncanonical_or_unbound_content(
    tmp_path: Path,
) -> None:
    from et_mainsim.raw_coverage_policy import (
        FrozenRawCoveragePolicyError,
        FrozenRawCoveragePolicyRequest,
        write_frozen_raw_coverage_policy_v1,
    )

    manifest_path = _write_formal_manifest(tmp_path)
    with pytest.raises(FrozenRawCoveragePolicyError, match="30, 90, 390"):
        FrozenRawCoveragePolicyRequest(
            production_manifest_path=manifest_path,
            output_path=tmp_path / "policy.json",
            windows_minutes=(30, 60),
            minimum_coverage_fraction=0.95,
            minimum_accepted_bins=10,
        )

    wrong_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    wrong_manifest["observation_product"] = "derived_e"
    manifest_path.write_text(json.dumps(wrong_manifest) + "\n", encoding="utf-8")
    with pytest.raises(FrozenRawCoveragePolicyError, match="final_dn"):
        write_frozen_raw_coverage_policy_v1(
            FrozenRawCoveragePolicyRequest(
                production_manifest_path=manifest_path,
                output_path=tmp_path / "policy.json",
                minimum_coverage_fraction=0.95,
                minimum_accepted_bins=10,
            )
        )


def test_frozen_raw_coverage_policy_rejects_unknown_policy_fields(
    tmp_path: Path,
) -> None:
    from et_mainsim.raw_coverage_policy import (
        FrozenRawCoveragePolicyError,
        FrozenRawCoveragePolicyRequest,
        load_frozen_raw_coverage_policy_v1,
        write_frozen_raw_coverage_policy_v1,
    )

    manifest_path = _write_formal_manifest(tmp_path)
    policy_path = manifest_path.parent / "analysis" / "policy.json"
    write_frozen_raw_coverage_policy_v1(
        FrozenRawCoveragePolicyRequest(
            production_manifest_path=manifest_path,
            output_path=policy_path,
            minimum_coverage_fraction=0.95,
            minimum_accepted_bins=10,
        )
    )
    payload = json.loads(policy_path.read_text(encoding="utf-8"))
    payload["covergae"] = {"silently_ignored": True}
    policy_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(FrozenRawCoveragePolicyError, match="unknown field"):
        load_frozen_raw_coverage_policy_v1(policy_path)


def test_frozen_raw_coverage_policy_cli_requires_explicit_thresholds(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from et_mainsim.raw_coverage_policy import main

    manifest_path = _write_formal_manifest(tmp_path)
    policy_path = manifest_path.parent / "analysis" / "policy.json"
    assert (
        main(
            (
                "--production-manifest",
                str(manifest_path),
                "--output-path",
                str(policy_path),
                "--minimum-coverage-fraction",
                "0.95",
                "--minimum-accepted-bins",
                "10",
            )
        )
        == 0
    )
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["policy_path"] == str(policy_path.resolve())
    assert receipt["minimum_coverage_fraction"] == pytest.approx(0.95)
