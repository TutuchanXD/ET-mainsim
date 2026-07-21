from __future__ import annotations

from pathlib import Path

import pytest


def test_single_scope_contract_preserves_legacy_full_frame_paths(tmp_path: Path) -> None:
    from et_mainsim.scope_artifacts import FullFrameScopeArtifactContract

    run_dir = tmp_path / "run"
    contract = FullFrameScopeArtifactContract.from_telescope_count(
        run_dir,
        telescope_count=1,
    )

    paths = contract.paths_for_scope_frame(scope_id=0, frame_index=17)

    assert contract.scope_ids == (0,)
    assert paths.frame_path == run_dir / "frames" / "frame_000017.npy"
    assert paths.summary_path == run_dir / "frame_summaries" / "frame_000017.json"
    assert paths.schema_path == (
        run_dir / "frame_summaries" / "frame_000017_schema.json"
    )
    completion = contract.frame_completion(
        frame_index=17,
        scope_is_complete=lambda scope_paths: scope_paths.scope_id == 0,
    )
    assert completion.is_complete is True
    assert completion.completed_scope_ids == (0,)
    assert completion.missing_scope_ids == ()
    assert contract.to_manifest_artifacts()["layout"] == "legacy_root_single_scope"


def test_six_scope_contract_uses_only_per_scope_paths_and_requires_all_scopes(
    tmp_path: Path,
) -> None:
    from et_mainsim.scope_artifacts import (
        FullFrameScopeArtifactContract,
        ScopeCadenceIdentity,
    )

    run_dir = tmp_path / "run"
    contract = FullFrameScopeArtifactContract.from_telescope_count(
        run_dir,
        telescope_count=6,
    )

    identity = ScopeCadenceIdentity(scope_id=5, frame_index=17)
    scope_paths = contract.paths_for_frame(17)

    assert contract.scope_ids == (0, 1, 2, 3, 4, 5)
    assert tuple(paths.scope_id for paths in scope_paths) == contract.scope_ids
    assert contract.paths_for_identity(identity).identity == identity
    assert [paths.frame_path for paths in scope_paths] == [
        run_dir / f"scope_{scope_id}" / "frames" / "frame_000017.npy"
        for scope_id in contract.scope_ids
    ]
    assert (run_dir / "frames" / "frame_000017.npy") not in {
        paths.frame_path for paths in scope_paths
    }

    incomplete = contract.frame_completion(
        frame_index=17,
        scope_is_complete=lambda paths: paths.scope_id != 5,
    )
    assert incomplete.is_complete is False
    assert incomplete.completed_scope_ids == (0, 1, 2, 3, 4)
    assert incomplete.missing_scope_ids == (5,)

    complete = contract.frame_completion(
        frame_index=17,
        scope_is_complete=lambda _paths: True,
    )
    assert complete.is_complete is True
    assert complete.completed_scope_ids == contract.scope_ids

    manifest_artifacts = contract.to_manifest_artifacts()
    assert manifest_artifacts["layout"] == "per_scope_directories"
    assert manifest_artifacts["image_level_combination"] == "forbidden"
    assert "frames" not in manifest_artifacts
    assert manifest_artifacts["scopes"]["scope_5"]["frames"] == str(
        run_dir / "scope_5" / "frames"
    )


def test_scope_contract_fails_closed_for_unsupported_count_and_scope_id(
    tmp_path: Path,
) -> None:
    from et_mainsim.scope_artifacts import (
        FullFrameScopeArtifactContract,
        ScopeArtifactContractError,
    )

    with pytest.raises(ScopeArtifactContractError, match="telescope_count"):
        FullFrameScopeArtifactContract.from_telescope_count(
            tmp_path / "run",
            telescope_count=2,
        )

    contract = FullFrameScopeArtifactContract.from_telescope_count(
        tmp_path / "run",
        telescope_count=6,
    )
    with pytest.raises(ScopeArtifactContractError, match="scope_id"):
        contract.paths_for_scope_frame(scope_id=6, frame_index=0)
