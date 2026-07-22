"""Pure planning primitives for contiguous global raw-frame time shards.

The planning layer deliberately has no renderer, storage, or Photsim7 dependency.
It defines one globally anchored raw-frame timeline and partitions only complete
coadd windows.  A worker can therefore process any listed shard independently,
while all output cadence products remain composable without a cross-shard coadd.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


CONTINUOUS_TIME_SHARD_SCHEMA_ID = "et_mainsim.continuous_time_shards.v1"
CONTINUOUS_TIME_SHARD_SCHEMA_VERSION = 1


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _positive_seconds(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and positive")
    try:
        seconds = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and positive") from exc
    if not math.isfinite(seconds) or seconds <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return seconds


def _normalise_coadd_sizes(values: Sequence[int]) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError("coadd_sizes must be a non-empty sequence of integers")
    sizes = tuple(_positive_int(value, name="coadd_sizes") for value in values)
    if not sizes:
        raise ValueError("coadd_sizes must not be empty")
    if len(set(sizes)) != len(sizes):
        raise ValueError("coadd_sizes must not contain duplicates")
    return tuple(sorted(sizes))


def coadd_sizes_for_cadences(
    *,
    raw_exposure_seconds: float,
    cadence_seconds: Sequence[float],
) -> tuple[int, ...]:
    """Return exact integral raw-frame coadd sizes for requested cadences.

    A cadence that is not an exact integer multiple of the raw exposure is
    rejected instead of silently rounding a physical timing definition.
    """

    raw_seconds = _positive_seconds(
        raw_exposure_seconds,
        name="raw_exposure_seconds",
    )
    if isinstance(cadence_seconds, (str, bytes)) or not isinstance(
        cadence_seconds,
        Sequence,
    ):
        raise ValueError("cadence_seconds must be a non-empty sequence")
    if not cadence_seconds:
        raise ValueError("cadence_seconds must not be empty")

    sizes: list[int] = []
    for cadence in cadence_seconds:
        seconds = _positive_seconds(cadence, name="cadence_seconds")
        ratio = seconds / raw_seconds
        nearest = round(ratio)
        if nearest <= 0 or not math.isclose(
            ratio,
            float(nearest),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "cadence_seconds must be an exact integer multiple of "
                "raw_exposure_seconds"
            )
        sizes.append(nearest)
    return _normalise_coadd_sizes(sizes)


def _interval_dict(start_index: int, stop_index: int) -> dict[str, int]:
    return {"start_index": int(start_index), "stop_index": int(stop_index)}


def _interval_from_manifest(
    payload: Any,
    *,
    name: str,
) -> tuple[int, int]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{name} must be an object")
    start_index = _nonnegative_int(payload.get("start_index"), name=f"{name}.start_index")
    stop_index = _nonnegative_int(payload.get("stop_index"), name=f"{name}.stop_index")
    if stop_index <= start_index:
        raise ValueError(f"{name} must have stop_index greater than start_index")
    return start_index, stop_index


@dataclass(frozen=True)
class CoaddWindow:
    """One globally indexed, complete coadd window within a raw time shard."""

    coadd_index: int
    raw_start_index: int
    raw_stop_index: int
    raw_exposure_seconds: float

    def __post_init__(self) -> None:
        coadd_index = _nonnegative_int(self.coadd_index, name="coadd_index")
        raw_start_index = _nonnegative_int(
            self.raw_start_index,
            name="raw_start_index",
        )
        raw_stop_index = _nonnegative_int(
            self.raw_stop_index,
            name="raw_stop_index",
        )
        if raw_stop_index <= raw_start_index:
            raise ValueError("raw_stop_index must be greater than raw_start_index")
        raw_exposure_seconds = _positive_seconds(
            self.raw_exposure_seconds,
            name="raw_exposure_seconds",
        )
        object.__setattr__(self, "coadd_index", coadd_index)
        object.__setattr__(self, "raw_start_index", raw_start_index)
        object.__setattr__(self, "raw_stop_index", raw_stop_index)
        object.__setattr__(self, "raw_exposure_seconds", raw_exposure_seconds)

    @property
    def raw_frame_count(self) -> int:
        return self.raw_stop_index - self.raw_start_index

    @property
    def start_time_seconds(self) -> float:
        return self.raw_start_index * self.raw_exposure_seconds

    @property
    def stop_time_seconds(self) -> float:
        return self.raw_stop_index * self.raw_exposure_seconds

    @property
    def midpoint_time_seconds(self) -> float:
        """Midpoint of the complete exposure interval, relative to raw frame 0."""

        return (self.start_time_seconds + self.stop_time_seconds) / 2.0

    @property
    def duration_seconds(self) -> float:
        return self.stop_time_seconds - self.start_time_seconds


@dataclass(frozen=True)
class ContinuousTimeShard:
    """A contiguous interval of global raw frames aligned to every coadd size."""

    shard_id: int
    raw_start_index: int
    raw_stop_index: int
    coadd_sizes: tuple[int, ...]
    raw_exposure_seconds: float

    def __post_init__(self) -> None:
        shard_id = _nonnegative_int(self.shard_id, name="shard_id")
        raw_start_index = _nonnegative_int(
            self.raw_start_index,
            name="raw_start_index",
        )
        raw_stop_index = _nonnegative_int(
            self.raw_stop_index,
            name="raw_stop_index",
        )
        if raw_stop_index <= raw_start_index:
            raise ValueError("raw_stop_index must be greater than raw_start_index")
        coadd_sizes = _normalise_coadd_sizes(self.coadd_sizes)
        for coadd_size in coadd_sizes:
            if raw_start_index % coadd_size or raw_stop_index % coadd_size:
                raise ValueError(
                    "time shard boundaries must align to every requested coadd size"
                )
        raw_exposure_seconds = _positive_seconds(
            self.raw_exposure_seconds,
            name="raw_exposure_seconds",
        )
        object.__setattr__(self, "shard_id", shard_id)
        object.__setattr__(self, "raw_start_index", raw_start_index)
        object.__setattr__(self, "raw_stop_index", raw_stop_index)
        object.__setattr__(self, "coadd_sizes", coadd_sizes)
        object.__setattr__(self, "raw_exposure_seconds", raw_exposure_seconds)

    @property
    def raw_frame_count(self) -> int:
        return self.raw_stop_index - self.raw_start_index

    def coadd_count(self, coadd_size: int) -> int:
        coadd_size = _positive_int(coadd_size, name="coadd_size")
        if coadd_size not in self.coadd_sizes:
            raise ValueError("coadd_size is not part of this time shard plan")
        return self.raw_frame_count // coadd_size

    def iter_coadd_windows(self, coadd_size: int) -> Iterator[CoaddWindow]:
        """Yield complete coadds in global raw-frame order without buffering images."""

        coadd_size = _positive_int(coadd_size, name="coadd_size")
        if coadd_size not in self.coadd_sizes:
            raise ValueError("coadd_size is not part of this time shard plan")
        first_coadd_index = self.raw_start_index // coadd_size
        for offset, raw_start_index in enumerate(
            range(self.raw_start_index, self.raw_stop_index, coadd_size)
        ):
            yield CoaddWindow(
                coadd_index=first_coadd_index + offset,
                raw_start_index=raw_start_index,
                raw_stop_index=raw_start_index + coadd_size,
                raw_exposure_seconds=self.raw_exposure_seconds,
            )

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "shard_id": self.shard_id,
            "raw_frame_interval": _interval_dict(
                self.raw_start_index,
                self.raw_stop_index,
            ),
            "raw_frame_count": self.raw_frame_count,
            "coadd_counts": {
                str(size): self.coadd_count(size) for size in self.coadd_sizes
            },
        }


def validate_time_shard_coverage(
    shards: Iterable[ContinuousTimeShard],
    *,
    raw_start_index: int,
    raw_stop_index: int,
    coadd_sizes: Sequence[int],
) -> tuple[ContinuousTimeShard, ...]:
    """Validate an ordered, exactly-covering set of independently runnable shards.

    The input order is intentional.  A reorder, gap, overlap, foreign coadd
    definition, or non-contiguous shard id is rejected rather than silently
    normalized, so a scheduler manifest cannot hide an ambiguous time mapping.
    """

    expected_start = _nonnegative_int(raw_start_index, name="raw_start_index")
    expected_stop = _nonnegative_int(raw_stop_index, name="raw_stop_index")
    if expected_stop <= expected_start:
        raise ValueError("raw_stop_index must be greater than raw_start_index")
    expected_coadd_sizes = _normalise_coadd_sizes(coadd_sizes)
    normalised_shards = tuple(shards)
    if not normalised_shards:
        raise ValueError("time shard coverage must include at least one shard")

    current_start = expected_start
    for expected_id, shard in enumerate(normalised_shards):
        if not isinstance(shard, ContinuousTimeShard):
            raise TypeError("time shard coverage must contain ContinuousTimeShard")
        if shard.shard_id != expected_id:
            raise ValueError("time shard ids must be contiguous from zero")
        if shard.coadd_sizes != expected_coadd_sizes:
            raise ValueError("time shard coadd_sizes do not match the plan")
        if shard.raw_start_index > current_start:
            raise ValueError("time shard coverage contains a gap")
        if shard.raw_start_index < current_start:
            raise ValueError("time shard coverage contains an overlap")
        current_start = shard.raw_stop_index

    if current_start < expected_stop:
        raise ValueError("time shard coverage contains a gap")
    if current_start > expected_stop:
        raise ValueError("time shard coverage extends beyond the accepted interval")
    return normalised_shards


@dataclass(frozen=True)
class ContinuousTimeShardPlan:
    """A globally aligned, tail-safe partition of one requested raw-frame span."""

    raw_start_index: int
    raw_stop_index: int
    accepted_raw_start_index: int
    accepted_raw_stop_index: int
    coadd_sizes: tuple[int, ...]
    raw_exposure_seconds: float
    max_raw_frames_per_shard: int
    shards: tuple[ContinuousTimeShard, ...]

    def __post_init__(self) -> None:
        raw_start_index = _nonnegative_int(
            self.raw_start_index,
            name="raw_start_index",
        )
        raw_stop_index = _nonnegative_int(self.raw_stop_index, name="raw_stop_index")
        if raw_stop_index <= raw_start_index:
            raise ValueError("raw_stop_index must be greater than raw_start_index")
        accepted_raw_start_index = _nonnegative_int(
            self.accepted_raw_start_index,
            name="accepted_raw_start_index",
        )
        accepted_raw_stop_index = _nonnegative_int(
            self.accepted_raw_stop_index,
            name="accepted_raw_stop_index",
        )
        if accepted_raw_start_index != raw_start_index:
            raise ValueError("accepted raw interval must start at raw_start_index")
        if (
            accepted_raw_stop_index <= accepted_raw_start_index
            or accepted_raw_stop_index > raw_stop_index
        ):
            raise ValueError("accepted raw interval must be non-empty and in range")
        coadd_sizes = _normalise_coadd_sizes(self.coadd_sizes)
        alignment_raw_frames = math.lcm(*coadd_sizes)
        if raw_start_index % alignment_raw_frames:
            raise ValueError(
                "raw_start_index must align to the global coadd boundary "
                f"({alignment_raw_frames} raw frames)"
            )
        if accepted_raw_stop_index % alignment_raw_frames:
            raise ValueError(
                "accepted_raw_stop_index must align to every requested cadence"
            )
        maximal_complete_stop_index = raw_start_index + (
            (raw_stop_index - raw_start_index) // alignment_raw_frames
        ) * alignment_raw_frames
        if accepted_raw_stop_index != maximal_complete_stop_index:
            raise ValueError(
                "accepted raw interval must retain the maximal complete global "
                "coadd interval"
            )
        raw_exposure_seconds = _positive_seconds(
            self.raw_exposure_seconds,
            name="raw_exposure_seconds",
        )
        max_raw_frames_per_shard = _positive_int(
            self.max_raw_frames_per_shard,
            name="max_raw_frames_per_shard",
        )
        if max_raw_frames_per_shard < alignment_raw_frames:
            raise ValueError(
                "max_raw_frames_per_shard must fit at least one global coadd boundary"
            )
        shards = validate_time_shard_coverage(
            self.shards,
            raw_start_index=accepted_raw_start_index,
            raw_stop_index=accepted_raw_stop_index,
            coadd_sizes=coadd_sizes,
        )
        if any(shard.raw_frame_count > max_raw_frames_per_shard for shard in shards):
            raise ValueError("time shard exceeds max_raw_frames_per_shard")
        if any(
            shard.raw_exposure_seconds != raw_exposure_seconds for shard in shards
        ):
            raise ValueError("time shard raw_exposure_seconds do not match the plan")
        object.__setattr__(self, "raw_start_index", raw_start_index)
        object.__setattr__(self, "raw_stop_index", raw_stop_index)
        object.__setattr__(self, "accepted_raw_start_index", accepted_raw_start_index)
        object.__setattr__(self, "accepted_raw_stop_index", accepted_raw_stop_index)
        object.__setattr__(self, "coadd_sizes", coadd_sizes)
        object.__setattr__(self, "raw_exposure_seconds", raw_exposure_seconds)
        object.__setattr__(self, "max_raw_frames_per_shard", max_raw_frames_per_shard)
        object.__setattr__(self, "shards", shards)

    @property
    def alignment_raw_frames(self) -> int:
        return math.lcm(*self.coadd_sizes)

    @property
    def rejected_tail_raw_interval(self) -> tuple[int, int] | None:
        if self.accepted_raw_stop_index == self.raw_stop_index:
            return None
        return (self.accepted_raw_stop_index, self.raw_stop_index)

    @property
    def accepted_raw_frame_count(self) -> int:
        return self.accepted_raw_stop_index - self.accepted_raw_start_index

    def to_manifest_dict(self) -> dict[str, Any]:
        tail_interval = self.rejected_tail_raw_interval
        return {
            "schema_id": CONTINUOUS_TIME_SHARD_SCHEMA_ID,
            "schema_version": CONTINUOUS_TIME_SHARD_SCHEMA_VERSION,
            "time_axis": {
                "kind": "global_raw_frame_index",
                "raw_exposure_seconds": self.raw_exposure_seconds,
                "origin_raw_frame_index": 0,
                "coadd_timestamp": "exposure_interval_midpoint",
            },
            "raw_frame_interval": _interval_dict(
                self.raw_start_index,
                self.raw_stop_index,
            ),
            "accepted_raw_frame_interval": _interval_dict(
                self.accepted_raw_start_index,
                self.accepted_raw_stop_index,
            ),
            "rejected_tail_raw_frame_interval": (
                None
                if tail_interval is None
                else _interval_dict(*tail_interval)
            ),
            "tail_policy": "reject_incomplete_global_coadd_tail",
            "coadd_sizes": list(self.coadd_sizes),
            "cadence_seconds": [
                size * self.raw_exposure_seconds for size in self.coadd_sizes
            ],
            "global_alignment_raw_frames": self.alignment_raw_frames,
            "max_raw_frames_per_shard": self.max_raw_frames_per_shard,
            "shards": [shard.to_manifest_dict() for shard in self.shards],
        }

    @classmethod
    def from_manifest_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "ContinuousTimeShardPlan":
        if payload.get("schema_id") != CONTINUOUS_TIME_SHARD_SCHEMA_ID:
            raise ValueError("Unsupported continuous time shard manifest")
        if int(payload.get("schema_version", 0)) != CONTINUOUS_TIME_SHARD_SCHEMA_VERSION:
            raise ValueError("Unsupported continuous time shard manifest version")
        time_axis = payload.get("time_axis")
        if not isinstance(time_axis, Mapping):
            raise ValueError("time_axis must be an object")
        if time_axis.get("kind") != "global_raw_frame_index":
            raise ValueError("Unsupported time shard time axis")
        if time_axis.get("origin_raw_frame_index") != 0:
            raise ValueError("Unsupported global raw frame origin")
        if time_axis.get("coadd_timestamp") != "exposure_interval_midpoint":
            raise ValueError("Unsupported coadd timestamp policy")

        raw_start_index, raw_stop_index = _interval_from_manifest(
            payload.get("raw_frame_interval"),
            name="raw_frame_interval",
        )
        accepted_raw_start_index, accepted_raw_stop_index = _interval_from_manifest(
            payload.get("accepted_raw_frame_interval"),
            name="accepted_raw_frame_interval",
        )
        coadd_sizes = _normalise_coadd_sizes(payload.get("coadd_sizes"))
        raw_exposure_seconds = _positive_seconds(
            time_axis.get("raw_exposure_seconds"),
            name="time_axis.raw_exposure_seconds",
        )
        max_raw_frames_per_shard = _positive_int(
            payload.get("max_raw_frames_per_shard"),
            name="max_raw_frames_per_shard",
        )
        shard_payloads = payload.get("shards")
        if not isinstance(shard_payloads, list):
            raise ValueError("shards must be a list")
        shards: list[ContinuousTimeShard] = []
        for item in shard_payloads:
            if not isinstance(item, Mapping):
                raise ValueError("shards must contain objects")
            shard_start_index, shard_stop_index = _interval_from_manifest(
                item.get("raw_frame_interval"),
                name="shard.raw_frame_interval",
            )
            shard = ContinuousTimeShard(
                shard_id=_nonnegative_int(item.get("shard_id"), name="shard_id"),
                raw_start_index=shard_start_index,
                raw_stop_index=shard_stop_index,
                coadd_sizes=coadd_sizes,
                raw_exposure_seconds=raw_exposure_seconds,
            )
            expected_count = item.get("raw_frame_count")
            if expected_count is not None and _positive_int(
                expected_count,
                name="shard.raw_frame_count",
            ) != shard.raw_frame_count:
                raise ValueError("shard raw_frame_count conflicts with its interval")
            expected_coadd_counts = item.get("coadd_counts")
            if expected_coadd_counts is not None:
                if not isinstance(expected_coadd_counts, Mapping):
                    raise ValueError("shard coadd_counts must be an object")
                expected = {
                    str(size): shard.coadd_count(size) for size in coadd_sizes
                }
                if dict(expected_coadd_counts) != expected:
                    raise ValueError("shard coadd_counts conflict with its interval")
            shards.append(shard)

        plan = cls(
            raw_start_index=raw_start_index,
            raw_stop_index=raw_stop_index,
            accepted_raw_start_index=accepted_raw_start_index,
            accepted_raw_stop_index=accepted_raw_stop_index,
            coadd_sizes=coadd_sizes,
            raw_exposure_seconds=raw_exposure_seconds,
            max_raw_frames_per_shard=max_raw_frames_per_shard,
            shards=tuple(shards),
        )
        if payload.get("tail_policy") != "reject_incomplete_global_coadd_tail":
            raise ValueError("Unsupported tail policy")
        expected_tail = plan.to_manifest_dict()["rejected_tail_raw_frame_interval"]
        if payload.get("rejected_tail_raw_frame_interval") != expected_tail:
            raise ValueError("rejected tail interval conflicts with raw frame intervals")
        if payload.get("global_alignment_raw_frames") != plan.alignment_raw_frames:
            raise ValueError("global alignment conflicts with coadd_sizes")
        expected_cadence_seconds = plan.to_manifest_dict()["cadence_seconds"]
        if payload.get("cadence_seconds") != expected_cadence_seconds:
            raise ValueError("cadence_seconds conflict with coadd_sizes")
        return plan

    def write_manifest(self, path: Path | str) -> Path:
        """Atomically write the planner manifest without importing a renderer."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(
                    self.to_manifest_dict(),
                    handle,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, destination)
        finally:
            temporary_path.unlink(missing_ok=True)
        return destination


def plan_continuous_time_shards(
    *,
    raw_start_index: int,
    raw_stop_index: int,
    coadd_sizes: Sequence[int],
    raw_exposure_seconds: float,
    max_raw_frames_per_shard: int,
) -> ContinuousTimeShardPlan:
    """Create contiguous globally aligned time shards from a requested raw span.

    ``raw_start_index`` is an inclusive global raw index and ``raw_stop_index``
    is exclusive.  It must already lie on the shared global coadd boundary.  If
    the requested stop leaves a partial multi-cadence window, only that tail is
    excluded and recorded in the manifest; no leading frame is re-anchored.
    """

    raw_start_index = _nonnegative_int(raw_start_index, name="raw_start_index")
    raw_stop_index = _nonnegative_int(raw_stop_index, name="raw_stop_index")
    if raw_stop_index <= raw_start_index:
        raise ValueError("raw_stop_index must be greater than raw_start_index")
    coadd_sizes = _normalise_coadd_sizes(coadd_sizes)
    raw_exposure_seconds = _positive_seconds(
        raw_exposure_seconds,
        name="raw_exposure_seconds",
    )
    max_raw_frames_per_shard = _positive_int(
        max_raw_frames_per_shard,
        name="max_raw_frames_per_shard",
    )
    alignment_raw_frames = math.lcm(*coadd_sizes)
    if raw_start_index % alignment_raw_frames:
        raise ValueError(
            "raw_start_index must align to the global coadd boundary "
            f"({alignment_raw_frames} raw frames)"
        )
    max_complete_shard_frames = (
        max_raw_frames_per_shard // alignment_raw_frames
    ) * alignment_raw_frames
    if max_complete_shard_frames == 0:
        raise ValueError(
            "max_raw_frames_per_shard must fit at least one global coadd boundary"
        )

    requested_frame_count = raw_stop_index - raw_start_index
    accepted_frame_count = (
        requested_frame_count // alignment_raw_frames
    ) * alignment_raw_frames
    if accepted_frame_count == 0:
        raise ValueError(
            "requested raw interval contains no complete global coadd window"
        )
    accepted_raw_stop_index = raw_start_index + accepted_frame_count
    shards: list[ContinuousTimeShard] = []
    for shard_id, shard_start_index in enumerate(
        range(
            raw_start_index,
            accepted_raw_stop_index,
            max_complete_shard_frames,
        )
    ):
        shard_stop_index = min(
            shard_start_index + max_complete_shard_frames,
            accepted_raw_stop_index,
        )
        shards.append(
            ContinuousTimeShard(
                shard_id=shard_id,
                raw_start_index=shard_start_index,
                raw_stop_index=shard_stop_index,
                coadd_sizes=coadd_sizes,
                raw_exposure_seconds=raw_exposure_seconds,
            )
        )
    return ContinuousTimeShardPlan(
        raw_start_index=raw_start_index,
        raw_stop_index=raw_stop_index,
        accepted_raw_start_index=raw_start_index,
        accepted_raw_stop_index=accepted_raw_stop_index,
        coadd_sizes=coadd_sizes,
        raw_exposure_seconds=raw_exposure_seconds,
        max_raw_frames_per_shard=max_raw_frames_per_shard,
        shards=tuple(shards),
    )


__all__ = [
    "CONTINUOUS_TIME_SHARD_SCHEMA_ID",
    "CONTINUOUS_TIME_SHARD_SCHEMA_VERSION",
    "CoaddWindow",
    "ContinuousTimeShard",
    "ContinuousTimeShardPlan",
    "coadd_sizes_for_cadences",
    "plan_continuous_time_shards",
    "validate_time_shard_coverage",
]
