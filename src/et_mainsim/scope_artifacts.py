"""Artifact and resume identities for legacy-science scope execution.

The scientific six-scope model is an outer execution dimension: each scope
produces an independently rendered detector image.  This module deliberately
models that fact in the on-disk layout.  It does not provide a path for a
six-scope image sum; any later science-sample reduction must happen after the
per-scope products have been validated.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


FULL_FRAME_SCOPE_ARTIFACT_SCHEMA_ID = "et_mainsim.full_frame_scope_artifacts.v1"
FULL_FRAME_SCOPE_ARTIFACT_SCHEMA_VERSION = 1

_SCOPE_IDS_BY_TELESCOPE_COUNT: dict[int, tuple[int, ...]] = {
    1: (0,),
    6: (0, 1, 2, 3, 4, 5),
}


class ScopeArtifactContractError(ValueError):
    """Raised when a requested artifact layout violates the scope contract."""


def _strict_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ScopeArtifactContractError(f"{field_name} must be an integer")
    return int(value)


@dataclass(frozen=True)
class ScopeCadenceIdentity:
    """The science identity for one scope-local detector cadence.

    A scope directory is only the storage projection of this identity.  Future
    shared-exposure target identities must bind the same ``scope_id`` rather
    than infer it from their parent pathname.
    """

    scope_id: int
    frame_index: int

    def __post_init__(self) -> None:
        scope_id = _strict_int(self.scope_id, field_name="scope_id")
        frame_index = _strict_int(self.frame_index, field_name="frame_index")
        if scope_id < 0:
            raise ScopeArtifactContractError("scope_id must be non-negative")
        if frame_index < 0:
            raise ScopeArtifactContractError("frame_index must be non-negative")
        object.__setattr__(self, "scope_id", scope_id)
        object.__setattr__(self, "frame_index", frame_index)


@dataclass(frozen=True)
class ScopeFrameArtifactPaths:
    """The three durable Photsim7 products for one scope and one cadence."""

    scope_id: int
    frame_index: int
    frame_path: Path
    summary_path: Path
    schema_path: Path

    @property
    def identity(self) -> ScopeCadenceIdentity:
        return ScopeCadenceIdentity(
            scope_id=self.scope_id,
            frame_index=self.frame_index,
        )


@dataclass(frozen=True)
class ScopeFrameCompletion:
    """All-scope completion result for one logical full-frame cadence."""

    frame_index: int
    scope_ids: tuple[int, ...]
    completed_scope_ids: tuple[int, ...]
    missing_scope_ids: tuple[int, ...]

    @property
    def is_complete(self) -> bool:
        return not self.missing_scope_ids

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_id": FULL_FRAME_SCOPE_ARTIFACT_SCHEMA_ID,
            "schema_version": FULL_FRAME_SCOPE_ARTIFACT_SCHEMA_VERSION,
            "frame_index": self.frame_index,
            "scope_ids": list(self.scope_ids),
            "completed_scope_ids": list(self.completed_scope_ids),
            "missing_scope_ids": list(self.missing_scope_ids),
            "completion_rule": "all_scopes_complete",
            "is_complete": self.is_complete,
            "image_level_combination": "forbidden",
        }


@dataclass(frozen=True)
class FullFrameScopeArtifactContract:
    """Maps one logical cadence to scope-local product paths.

    ``telescope_count=1`` preserves the existing root-level paths exactly.
    ``telescope_count=6`` places every scope below ``scope_<id>/``.  The
    contract intentionally supports only these two frozen legacy-science
    configurations and fails closed for all other counts.
    """

    run_dir: Path
    scope_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        normalized_run_dir = Path(self.run_dir)
        normalized_scope_ids = tuple(
            _strict_int(value, field_name="scope_ids") for value in self.scope_ids
        )
        if normalized_scope_ids not in _SCOPE_IDS_BY_TELESCOPE_COUNT.values():
            raise ScopeArtifactContractError(
                "scope_ids must be exactly (0,) or the canonical six-scope "
                "sequence (0, 1, 2, 3, 4, 5)"
            )
        object.__setattr__(self, "run_dir", normalized_run_dir)
        object.__setattr__(self, "scope_ids", normalized_scope_ids)

    @classmethod
    def from_telescope_count(
        cls,
        run_dir: Path | str,
        *,
        telescope_count: int,
    ) -> "FullFrameScopeArtifactContract":
        normalized_count = _strict_int(
            telescope_count,
            field_name="telescope_count",
        )
        try:
            scope_ids = _SCOPE_IDS_BY_TELESCOPE_COUNT[normalized_count]
        except KeyError as error:
            raise ScopeArtifactContractError(
                "legacy-science scope artifacts support telescope_count=1 or "
                "telescope_count=6"
            ) from error
        return cls(run_dir=Path(run_dir), scope_ids=scope_ids)

    @property
    def telescope_count(self) -> int:
        return len(self.scope_ids)

    @property
    def is_single_scope(self) -> bool:
        return self.scope_ids == (0,)

    def scope_root(self, scope_id: int) -> Path:
        normalized_scope_id = _strict_int(scope_id, field_name="scope_id")
        if normalized_scope_id not in self.scope_ids:
            raise ScopeArtifactContractError(
                f"scope_id {normalized_scope_id} is not in {self.scope_ids}"
            )
        if self.is_single_scope:
            return self.run_dir
        return self.run_dir / f"scope_{normalized_scope_id}"

    def paths_for_scope_frame(
        self,
        *,
        scope_id: int,
        frame_index: int,
    ) -> ScopeFrameArtifactPaths:
        return self.paths_for_identity(
            ScopeCadenceIdentity(scope_id=scope_id, frame_index=frame_index)
        )

    def paths_for_identity(
        self,
        identity: ScopeCadenceIdentity,
    ) -> ScopeFrameArtifactPaths:
        if not isinstance(identity, ScopeCadenceIdentity):
            raise ScopeArtifactContractError(
                "identity must be a ScopeCadenceIdentity"
            )
        root = self.scope_root(identity.scope_id)
        stem = f"frame_{identity.frame_index:06d}"
        return ScopeFrameArtifactPaths(
            scope_id=identity.scope_id,
            frame_index=identity.frame_index,
            frame_path=root / "frames" / f"{stem}.npy",
            summary_path=root / "frame_summaries" / f"{stem}.json",
            schema_path=root / "frame_summaries" / f"{stem}_schema.json",
        )

    def paths_for_frame(self, frame_index: int) -> tuple[ScopeFrameArtifactPaths, ...]:
        return tuple(
            self.paths_for_scope_frame(scope_id=scope_id, frame_index=frame_index)
            for scope_id in self.scope_ids
        )

    def frame_completion(
        self,
        *,
        frame_index: int,
        scope_is_complete: Callable[[ScopeFrameArtifactPaths], bool],
    ) -> ScopeFrameCompletion:
        paths_by_scope = self.paths_for_frame(frame_index)
        completed_scope_ids = tuple(
            paths.scope_id for paths in paths_by_scope if bool(scope_is_complete(paths))
        )
        missing_scope_ids = tuple(
            scope_id for scope_id in self.scope_ids if scope_id not in completed_scope_ids
        )
        return ScopeFrameCompletion(
            frame_index=_strict_int(frame_index, field_name="frame_index"),
            scope_ids=self.scope_ids,
            completed_scope_ids=completed_scope_ids,
            missing_scope_ids=missing_scope_ids,
        )

    def to_manifest_artifacts(self) -> dict[str, Any]:
        """Return layout metadata suitable for a run manifest.

        For six scopes the root has no ``frames`` entry by design.  Consumers
        must resolve an explicit scope product and cannot accidentally treat a
        detector-image sum as a persisted full-frame artifact.
        """

        payload: dict[str, Any] = {
            "scope_artifact_schema_id": FULL_FRAME_SCOPE_ARTIFACT_SCHEMA_ID,
            "scope_artifact_schema_version": (
                FULL_FRAME_SCOPE_ARTIFACT_SCHEMA_VERSION
            ),
            "scope_ids": list(self.scope_ids),
            "completion_rule": "all_scopes_complete",
            "image_level_combination": "forbidden",
        }
        if self.is_single_scope:
            payload.update(
                {
                    "layout": "legacy_root_single_scope",
                    "root": str(self.run_dir),
                    "frames": str(self.run_dir / "frames"),
                    "frame_summaries": str(self.run_dir / "frame_summaries"),
                }
            )
            return payload

        payload.update(
            {
                "layout": "per_scope_directories",
                "scopes": {
                    f"scope_{scope_id}": {
                        "root": str(self.scope_root(scope_id)),
                        "frames": str(self.scope_root(scope_id) / "frames"),
                        "frame_summaries": str(
                            self.scope_root(scope_id) / "frame_summaries"
                        ),
                    }
                    for scope_id in self.scope_ids
                },
            }
        )
        return payload


__all__ = [
    "FULL_FRAME_SCOPE_ARTIFACT_SCHEMA_ID",
    "FULL_FRAME_SCOPE_ARTIFACT_SCHEMA_VERSION",
    "FullFrameScopeArtifactContract",
    "ScopeArtifactContractError",
    "ScopeCadenceIdentity",
    "ScopeFrameArtifactPaths",
    "ScopeFrameCompletion",
]
