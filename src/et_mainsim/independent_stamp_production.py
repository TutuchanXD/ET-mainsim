"""Streaming production of independent, calibrated ET stamp time shards.

This layer is intentionally narrower than the historical ``et-stamp``
workflow.  It owns one target, one contiguous globally indexed raw-frame
shard, and writes only formal delivery bundles whose sole observation is
``final_dn``.  Background expectations, bias/column calibration, masks and
truth/provenance are companions; sampled background realizations are never
written as a subtractable product.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import ExitStack
from dataclasses import dataclass
import json
import math
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .stamp_delivery import (
    StampDeliveryBundle,
    StampDeliveryBundleAppender,
    StampDeliveryShardSet,
)
from .time_shards import ContinuousTimeShard


INDEPENDENT_STAMP_PRODUCTION_SCHEMA_ID = (
    "et_mainsim.independent_stamp_production.v1"
)


def _json_mapping(value: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    candidate = dict(value)
    try:
        json.dumps(candidate, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be JSON serializable") from error
    return candidate


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _nonnegative_source_id(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a non-negative signed int64 integer")
    result = int(value)
    if result < 0 or result > int(np.iinfo(np.int64).max):
        raise ValueError(f"{name} must be a non-negative signed int64 integer")
    return result


def _positive_float(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite and positive")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be finite and positive") from error
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _to_numpy(value: Any) -> NDArray[np.generic]:
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach()
    if hasattr(value, "cpu") and callable(value.cpu):
        value = value.cpu()
    if hasattr(value, "numpy") and callable(value.numpy):
        value = value.numpy()
    return np.asarray(value)


def _mask(value: ArrayLike, *, name: str, shape: tuple[int, int]) -> NDArray[np.bool_]:
    array = np.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{name} must have stamp shape {shape}, got {array.shape}")
    if array.dtype.kind not in {"b", "i", "u"}:
        raise ValueError(f"{name} must be a boolean/integer mask")
    return np.asarray(array, dtype=bool)


@dataclass(frozen=True)
class RawStampDeliveryFrame:
    """One raw detector stamp plus calibration and quality companions."""

    final_dn: NDArray[np.unsignedinteger]
    background_expectation_e: NDArray[np.float64]
    bias_level_dn: float
    column_noise_dn_by_x: NDArray[np.float64]
    valid_mask: NDArray[np.bool_]
    fullwell_mask: NDArray[np.bool_]
    adc_low_mask: NDArray[np.bool_]
    adc_high_mask: NDArray[np.bool_]
    cosmic_mask: NDArray[np.bool_]

    def __post_init__(self) -> None:
        final = np.asarray(self.final_dn)
        if final.ndim != 2 or any(int(size) <= 0 for size in final.shape):
            raise ValueError("final_dn must be a non-empty two-dimensional stamp")
        if final.dtype.kind != "u":
            raise ValueError("final_dn must use an unsigned detector DN dtype")
        shape = (int(final.shape[0]), int(final.shape[1]))
        background = np.asarray(self.background_expectation_e, dtype=np.float64)
        if background.shape != shape or not np.all(np.isfinite(background)):
            raise ValueError("background_expectation_e must be finite and match final_dn")
        if np.any(background < 0.0):
            raise ValueError("background_expectation_e must be non-negative")
        bias = float(self.bias_level_dn)
        if not math.isfinite(bias):
            raise ValueError("bias_level_dn must be finite")
        column = np.asarray(self.column_noise_dn_by_x, dtype=np.float64)
        if column.shape != (shape[1],) or not np.all(np.isfinite(column)):
            raise ValueError("column_noise_dn_by_x must be finite with shape (nx,)")
        object.__setattr__(self, "final_dn", np.array(final, copy=True, order="C"))
        object.__setattr__(
            self,
            "background_expectation_e",
            np.array(background, copy=True, order="C"),
        )
        object.__setattr__(self, "bias_level_dn", bias)
        object.__setattr__(self, "column_noise_dn_by_x", np.array(column, copy=True))
        for name in (
            "valid_mask",
            "fullwell_mask",
            "adc_low_mask",
            "adc_high_mask",
            "cosmic_mask",
        ):
            object.__setattr__(self, name, _mask(getattr(self, name), name=name, shape=shape))

    @property
    def stamp_shape(self) -> tuple[int, int]:
        return tuple(int(value) for value in self.final_dn.shape)  # type: ignore[return-value]


@dataclass(frozen=True)
class IndependentStampShardRequest:
    """Static identity for one non-resumable target-time production shard."""

    output_root: Path | str
    target_source_id: int
    stamp_shape: tuple[int, int]
    shard: ContinuousTimeShard
    gain_e_per_dn: float
    manifest: Mapping[str, Any]
    provenance: Mapping[str, Any]
    batch_size: int = 32
    overwrite: bool = False

    def __post_init__(self) -> None:
        target_source_id = _nonnegative_source_id(
            self.target_source_id,
            name="target_source_id",
        )
        if not isinstance(self.shard, ContinuousTimeShard):
            raise TypeError("shard must be a ContinuousTimeShard")
        try:
            ny, nx = self.stamp_shape
        except (TypeError, ValueError) as error:
            raise ValueError("stamp_shape must contain two positive integers") from error
        stamp_shape = (_positive_int(ny, name="stamp_shape[0]"), _positive_int(nx, name="stamp_shape[1]"))
        output_root = Path(self.output_root).expanduser()
        object.__setattr__(self, "output_root", output_root)
        object.__setattr__(self, "target_source_id", target_source_id)
        object.__setattr__(self, "stamp_shape", stamp_shape)
        object.__setattr__(self, "gain_e_per_dn", _positive_float(self.gain_e_per_dn, name="gain_e_per_dn"))
        object.__setattr__(self, "manifest", _json_mapping(self.manifest, name="manifest"))
        object.__setattr__(self, "provenance", _json_mapping(self.provenance, name="provenance"))
        object.__setattr__(self, "batch_size", _positive_int(self.batch_size, name="batch_size"))
        object.__setattr__(self, "overwrite", bool(self.overwrite))

    @property
    def target_root(self) -> Path:
        return Path(self.output_root) / "stamps" / f"target_{self.target_source_id}"

    @property
    def shard_root(self) -> Path:
        return self.target_root / "delivery" / f"shard_{self.shard.shard_id:05d}"


@dataclass(frozen=True)
class IndependentStampShardReport:
    """Paths and counts published after every bundle readback succeeds."""

    raw_path: Path
    coadd_paths: Mapping[int, Path]
    raw_frame_count: int
    target_source_id: int
    shard_id: int


@dataclass
class _CoaddAccumulator:
    factor: int
    stamp_shape: tuple[int, int]
    raw_exposure_seconds: float
    count: int = 0
    raw_start_index: int | None = None
    final_dn: NDArray[np.uint64] | None = None
    background_expectation_e: NDArray[np.float64] | None = None
    bias_level_sum_dn: float = 0.0
    column_noise_sum_dn_by_x: NDArray[np.float64] | None = None
    valid_mask: NDArray[np.bool_] | None = None
    fullwell_count: NDArray[np.uint16] | None = None
    adc_low_count: NDArray[np.uint16] | None = None
    adc_high_count: NDArray[np.uint16] | None = None
    cosmic_count: NDArray[np.uint16] | None = None

    def add(self, raw: RawStampDeliveryFrame, *, raw_frame_index: int) -> None:
        if raw.stamp_shape != self.stamp_shape:
            raise ValueError("raw frame stamp shape differs from coadd accumulator")
        if self.count == 0:
            self.raw_start_index = int(raw_frame_index)
            self.final_dn = np.zeros(self.stamp_shape, dtype=np.uint64)
            self.background_expectation_e = np.zeros(self.stamp_shape, dtype=np.float64)
            self.column_noise_sum_dn_by_x = np.zeros(self.stamp_shape[1], dtype=np.float64)
            self.valid_mask = np.ones(self.stamp_shape, dtype=bool)
            self.fullwell_count = np.zeros(self.stamp_shape, dtype=np.uint16)
            self.adc_low_count = np.zeros(self.stamp_shape, dtype=np.uint16)
            self.adc_high_count = np.zeros(self.stamp_shape, dtype=np.uint16)
            self.cosmic_count = np.zeros(self.stamp_shape, dtype=np.uint16)
        assert self.final_dn is not None
        assert self.background_expectation_e is not None
        assert self.column_noise_sum_dn_by_x is not None
        assert self.valid_mask is not None
        assert self.fullwell_count is not None
        assert self.adc_low_count is not None
        assert self.adc_high_count is not None
        assert self.cosmic_count is not None
        self.final_dn += raw.final_dn.astype(np.uint64, copy=False)
        self.background_expectation_e += raw.background_expectation_e
        self.bias_level_sum_dn += raw.bias_level_dn
        self.column_noise_sum_dn_by_x += raw.column_noise_dn_by_x
        self.valid_mask &= raw.valid_mask
        self.fullwell_count += raw.fullwell_mask.astype(np.uint16, copy=False)
        self.adc_low_count += raw.adc_low_mask.astype(np.uint16, copy=False)
        self.adc_high_count += raw.adc_high_mask.astype(np.uint16, copy=False)
        self.cosmic_count += raw.cosmic_mask.astype(np.uint16, copy=False)
        self.count += 1
        if self.count > self.factor:
            raise RuntimeError("coadd accumulator received too many raw frames")

    @property
    def ready(self) -> bool:
        return self.count == self.factor

    def consume(self) -> tuple[RawStampDeliveryFrame, int]:
        if not self.ready or self.raw_start_index is None:
            raise RuntimeError("coadd accumulator is not ready")
        assert self.final_dn is not None
        assert self.background_expectation_e is not None
        assert self.column_noise_sum_dn_by_x is not None
        assert self.valid_mask is not None
        assert self.fullwell_count is not None
        assert self.adc_low_count is not None
        assert self.adc_high_count is not None
        assert self.cosmic_count is not None
        frame = RawStampDeliveryFrame(
            final_dn=self.final_dn,
            background_expectation_e=self.background_expectation_e,
            bias_level_dn=self.bias_level_sum_dn,
            column_noise_dn_by_x=self.column_noise_sum_dn_by_x,
            valid_mask=self.valid_mask,
            fullwell_mask=self.fullwell_count > 0,
            adc_low_mask=self.adc_low_count > 0,
            adc_high_mask=self.adc_high_count > 0,
            cosmic_mask=self.cosmic_count > 0,
        )
        # Counts must survive coaddition, so attach them separately at bundle construction.
        raw_start = self.raw_start_index
        self.count = 0
        self.raw_start_index = None
        self.final_dn = None
        self.background_expectation_e = None
        self.bias_level_sum_dn = 0.0
        self.column_noise_sum_dn_by_x = None
        self.valid_mask = None
        self.fullwell_count = None
        self.adc_low_count = None
        self.adc_high_count = None
        self.cosmic_count = None
        return frame, raw_start


def _bundle_from_frames(
    *,
    frames: list[RawStampDeliveryFrame],
    raw_starts: list[int],
    raw_exposure_seconds: float,
    product_kind: str,
    coadd_factor: int,
    gain_e_per_dn: float,
    manifest: Mapping[str, Any],
    provenance: Mapping[str, Any],
    quality_counts: tuple[
        list[NDArray[np.uint16]],
        list[NDArray[np.uint16]],
        list[NDArray[np.uint16]],
        list[NDArray[np.uint16]],
    ]
    | None = None,
) -> StampDeliveryBundle:
    if not frames or len(frames) != len(raw_starts):
        raise ValueError("frames and raw_starts must be non-empty and aligned")
    shape = frames[0].stamp_shape
    if any(frame.stamp_shape != shape for frame in frames):
        raise ValueError("all bundled frames must share one stamp shape")
    if quality_counts is None:
        fullwell = [frame.fullwell_mask.astype(np.uint16) for frame in frames]
        adc_low = [frame.adc_low_mask.astype(np.uint16) for frame in frames]
        adc_high = [frame.adc_high_mask.astype(np.uint16) for frame in frames]
        cosmic = [frame.cosmic_mask.astype(np.uint16) for frame in frames]
    else:
        fullwell, adc_low, adc_high, cosmic = quality_counts
    starts = np.asarray(raw_starts, dtype=np.int64)
    return StampDeliveryBundle.from_arrays(
        product_kind=product_kind,
        coadd_factor=coadd_factor,
        final_dn=np.stack([frame.final_dn for frame in frames]),
        background_expectation_e=np.stack([frame.background_expectation_e for frame in frames]),
        bias_level_sum_dn=np.asarray([frame.bias_level_dn for frame in frames], dtype=np.float64),
        column_noise_sum_dn_by_x=np.stack([frame.column_noise_dn_by_x for frame in frames]),
        valid_mask=np.stack([frame.valid_mask for frame in frames]),
        fullwell_count=np.stack(fullwell),
        adc_low_count=np.stack(adc_low),
        adc_high_count=np.stack(adc_high),
        cosmic_count=np.stack(cosmic),
        time_start_seconds=starts.astype(np.float64) * raw_exposure_seconds,
        exposure_seconds=np.full(
            len(frames),
            raw_exposure_seconds * coadd_factor,
            dtype=np.float64,
        ),
        raw_frame_start_index=starts,
        raw_frame_stop_index_exclusive=starts + coadd_factor,
        gain_e_per_dn=np.asarray(gain_e_per_dn, dtype=np.float64),
        manifest=manifest,
        provenance=provenance,
    )


def _delivery_manifest(
    request: IndependentStampShardRequest,
    *,
    product_kind: str,
    coadd_factor: int,
) -> dict[str, Any]:
    return {
        "schema_id": INDEPENDENT_STAMP_PRODUCTION_SCHEMA_ID,
        "scene_policy": "independent_target",
        "target_source_id": str(request.target_source_id),
        "target_source_id_int64": request.target_source_id,
        "stamp_shape": list(request.stamp_shape),
        "time_shard": request.shard.to_manifest_dict(),
        "product_kind": product_kind,
        "coadd_factor": coadd_factor,
        "caller_manifest": dict(request.manifest),
    }


def _delivery_provenance(
    request: IndependentStampShardRequest,
    *,
    product_kind: str,
    coadd_factor: int,
) -> dict[str, Any]:
    return {
        "schema_id": INDEPENDENT_STAMP_PRODUCTION_SCHEMA_ID,
        "observation_product": "final_dn",
        "background_realization_used": False,
        "background_companion": "deterministic_background_expectation_e",
        "scene_policy": "independent_target",
        "product_kind": product_kind,
        "coadd_factor": coadd_factor,
        "caller_provenance": dict(request.provenance),
    }


def _bundle_paths(request: IndependentStampShardRequest) -> tuple[Path, dict[int, Path]]:
    root = request.shard_root
    raw = root / "raw.h5"
    coadds = {factor: root / f"coadd_{factor * 10:d}s.h5" for factor in request.shard.coadd_sizes}
    return raw, coadds


def _h5_scalar_text(value: Any, *, name: str) -> str:
    """Decode the scalar string representation used by HDF5 attributes/data."""

    if isinstance(value, np.ndarray):
        if value.shape != ():
            raise RuntimeError(f"{name} must be an HDF5 scalar")
        value = value.item()
    if isinstance(value, np.bytes_):
        value = bytes(value)
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str):
        raise RuntimeError(f"{name} must be an HDF5 string")
    return value


def _read_staged_bundle_coverage(
    path: Path,
) -> tuple[
    str,
    int,
    Mapping[str, Any],
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Read only shard-coverage metadata, not the large image planes.

    ``StampDeliveryBundleAppender.complete`` has already performed the full
    file contract validation.  This narrow readback is deliberately limited to
    vectors and manifest metadata so final shard-set publication does not make
    a second in-memory copy of a multi-day stamp cube.
    """

    try:
        import h5py
    except ImportError as error:  # pragma: no cover - packaging guard
        raise RuntimeError("h5py is required for stamp delivery bundles") from error
    try:
        with h5py.File(path, "r") as handle:
            complete = handle.attrs["complete"]
            if isinstance(complete, np.ndarray):
                complete = complete.item()
            if not bool(complete):
                raise RuntimeError("staged bundle is not marked complete")
            product_kind = _h5_scalar_text(
                handle.attrs["product_kind"],
                name="product_kind",
            )
            coadd_factor = int(handle.attrs["coadd_factor"])
            manifest = json.loads(
                _h5_scalar_text(handle["manifest_json"][()], name="manifest_json")
            )
            if not isinstance(manifest, Mapping):
                raise RuntimeError("staged bundle manifest_json must be an object")
            return (
                product_kind,
                coadd_factor,
                dict(manifest),
                np.asarray(handle["raw_frame_start_index"], dtype=np.int64),
                np.asarray(handle["raw_frame_stop_index_exclusive"], dtype=np.int64),
                np.asarray(handle["time_start_seconds"], dtype=np.float64),
                np.asarray(handle["exposure_seconds"], dtype=np.float64),
            )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read staged bundle coverage from {path}") from error


def _validate_staged_bundle_time_shard_coverage(
    request: IndependentStampShardRequest,
    *,
    path: Path,
    product_kind: str,
    coadd_factor: int,
) -> None:
    """Require a completed staged bundle to declare and exactly cover its shard."""

    (
        actual_product_kind,
        actual_coadd_factor,
        manifest,
        raw_starts,
        raw_stops,
        time_starts,
        exposure_seconds,
    ) = _read_staged_bundle_coverage(path)
    if actual_product_kind != product_kind or actual_coadd_factor != coadd_factor:
        raise RuntimeError(
            f"staged bundle {path.name} product identity does not match its shard member"
        )
    expected_shard = request.shard.to_manifest_dict()
    if manifest.get("time_shard") != expected_shard:
        raise RuntimeError(
            f"staged bundle {path.name} does not declare the expected time_shard coverage"
        )
    expected_starts = np.arange(
        request.shard.raw_start_index,
        request.shard.raw_stop_index,
        coadd_factor,
        dtype=np.int64,
    )
    expected_stops = expected_starts + coadd_factor
    expected_time_starts = (
        expected_starts.astype(np.float64) * request.shard.raw_exposure_seconds
    )
    expected_exposures = np.full(
        expected_starts.shape,
        request.shard.raw_exposure_seconds * coadd_factor,
        dtype=np.float64,
    )
    if not np.array_equal(raw_starts, expected_starts) or not np.array_equal(
        raw_stops,
        expected_stops,
    ):
        raise RuntimeError(
            f"staged bundle {path.name} does not exactly cover its declared raw frame interval"
        )
    if not np.array_equal(time_starts, expected_time_starts) or not np.array_equal(
        exposure_seconds,
        expected_exposures,
    ):
        raise RuntimeError(
            f"staged bundle {path.name} time vectors do not match its declared shard"
        )


def _validate_staged_shard_delivery_coverage(
    request: IndependentStampShardRequest,
    *,
    raw_path: Path,
    coadd_paths: Mapping[int, Path],
) -> None:
    """Validate raw plus every coadd member before the directory is published."""

    _validate_staged_bundle_time_shard_coverage(
        request,
        path=raw_path,
        product_kind="raw",
        coadd_factor=1,
    )
    for factor in request.shard.coadd_sizes:
        _validate_staged_bundle_time_shard_coverage(
            request,
            path=coadd_paths[factor],
            product_kind="coadd",
            coadd_factor=factor,
        )


def run_independent_stamp_time_shard(
    request: IndependentStampShardRequest,
    *,
    render_raw: Callable[[int], Any],
    adapt_raw: Callable[[Any], RawStampDeliveryFrame],
) -> IndependentStampShardReport:
    """Render one global raw-frame shard and atomically publish raw/coadds.

    ``render_raw`` must consume the **global** raw frame index.  The caller is
    responsible for supplying a Photsim7 renderer with the whole-run
    ``SimulationSpec`` and a source-variability provider aligned to that same
    global axis; this function never reindexes a shard locally.
    """

    if not isinstance(request, IndependentStampShardRequest):
        raise TypeError("request must be an IndependentStampShardRequest")
    if not callable(render_raw) or not callable(adapt_raw):
        raise TypeError("render_raw and adapt_raw must be callable")
    final_raw_path, final_coadd_paths = _bundle_paths(request)

    raw_manifest = _delivery_manifest(request, product_kind="raw", coadd_factor=1)
    raw_provenance = _delivery_provenance(request, product_kind="raw", coadd_factor=1)
    coadd_manifests = {
        factor: _delivery_manifest(request, product_kind="coadd", coadd_factor=factor)
        for factor in request.shard.coadd_sizes
    }
    coadd_provenances = {
        factor: _delivery_provenance(request, product_kind="coadd", coadd_factor=factor)
        for factor in request.shard.coadd_sizes
    }
    accumulators = {
        factor: _CoaddAccumulator(
            factor=factor,
            stamp_shape=request.stamp_shape,
            raw_exposure_seconds=request.shard.raw_exposure_seconds,
        )
        for factor in request.shard.coadd_sizes
    }
    raw_buffer: list[RawStampDeliveryFrame] = []
    raw_starts: list[int] = []
    coadd_buffers: dict[int, list[tuple[RawStampDeliveryFrame, int, tuple[NDArray[np.uint16], NDArray[np.uint16], NDArray[np.uint16], NDArray[np.uint16]]]]] = {
        factor: [] for factor in request.shard.coadd_sizes
    }

    expected_members = (
        final_raw_path.name,
        *(path.name for path in final_coadd_paths.values()),
    )
    with StampDeliveryShardSet(
        request.shard_root,
        expected_members=expected_members,
        overwrite=request.overwrite,
    ) as shard_set:
        raw_path = shard_set.member_path(final_raw_path.name)
        coadd_paths = {
            factor: shard_set.member_path(final_coadd_paths[factor].name)
            for factor in request.shard.coadd_sizes
        }
        with ExitStack() as stack:
            raw_appender = stack.enter_context(
                StampDeliveryBundleAppender(
                    raw_path,
                    product_kind="raw",
                    coadd_factor=1,
                    stamp_shape=request.stamp_shape,
                    gain_e_per_dn=request.gain_e_per_dn,
                    manifest=raw_manifest,
                    provenance=raw_provenance,
                    overwrite=request.overwrite,
                )
            )
            coadd_appenders = {
                factor: stack.enter_context(
                    StampDeliveryBundleAppender(
                        coadd_paths[factor],
                        product_kind="coadd",
                        coadd_factor=factor,
                        stamp_shape=request.stamp_shape,
                        gain_e_per_dn=request.gain_e_per_dn,
                        manifest=coadd_manifests[factor],
                        provenance=coadd_provenances[factor],
                        overwrite=request.overwrite,
                    )
                )
                for factor in request.shard.coadd_sizes
            }

            def flush_raw() -> None:
                if raw_buffer:
                    raw_appender.append(
                        _bundle_from_frames(
                            frames=raw_buffer,
                            raw_starts=raw_starts,
                            raw_exposure_seconds=request.shard.raw_exposure_seconds,
                            product_kind="raw",
                            coadd_factor=1,
                            gain_e_per_dn=request.gain_e_per_dn,
                            manifest=raw_manifest,
                            provenance=raw_provenance,
                        )
                    )
                    raw_buffer.clear()
                    raw_starts.clear()

            def flush_coadd(factor: int) -> None:
                buffered = coadd_buffers[factor]
                if not buffered:
                    return
                frames = [item[0] for item in buffered]
                starts = [item[1] for item in buffered]
                counts = tuple(
                    [item[2][count_index] for item in buffered]
                    for count_index in range(4)
                )
                coadd_appenders[factor].append(
                    _bundle_from_frames(
                        frames=frames,
                        raw_starts=starts,
                        raw_exposure_seconds=request.shard.raw_exposure_seconds,
                        product_kind="coadd",
                        coadd_factor=factor,
                        gain_e_per_dn=request.gain_e_per_dn,
                        manifest=coadd_manifests[factor],
                        provenance=coadd_provenances[factor],
                        quality_counts=counts,  # type: ignore[arg-type]
                    )
                )
                buffered.clear()

            for raw_frame_index in range(
                request.shard.raw_start_index,
                request.shard.raw_stop_index,
            ):
                raw = adapt_raw(render_raw(raw_frame_index))
                if not isinstance(raw, RawStampDeliveryFrame):
                    raise TypeError("adapt_raw must return RawStampDeliveryFrame")
                if raw.stamp_shape != request.stamp_shape:
                    raise ValueError(
                        "rendered raw frame stamp shape differs from request.stamp_shape"
                    )
                raw_buffer.append(raw)
                raw_starts.append(raw_frame_index)
                if len(raw_buffer) >= request.batch_size:
                    flush_raw()
                for factor, accumulator in accumulators.items():
                    accumulator.add(raw, raw_frame_index=raw_frame_index)
                    if accumulator.ready:
                        assert accumulator.fullwell_count is not None
                        assert accumulator.adc_low_count is not None
                        assert accumulator.adc_high_count is not None
                        assert accumulator.cosmic_count is not None
                        counts = (
                            np.array(accumulator.fullwell_count, copy=True),
                            np.array(accumulator.adc_low_count, copy=True),
                            np.array(accumulator.adc_high_count, copy=True),
                            np.array(accumulator.cosmic_count, copy=True),
                        )
                        coadd_frame, raw_start = accumulator.consume()
                        coadd_buffers[factor].append((coadd_frame, raw_start, counts))
                        if len(coadd_buffers[factor]) >= request.batch_size:
                            flush_coadd(factor)
            flush_raw()
            for factor, accumulator in accumulators.items():
                if accumulator.count != 0:
                    raise RuntimeError(
                        "time shard ended before a complete configured coadd window"
                    )
                flush_coadd(factor)
            raw_appender.complete()
            for appender in coadd_appenders.values():
                appender.complete()
        _validate_staged_shard_delivery_coverage(
            request,
            raw_path=raw_path,
            coadd_paths=coadd_paths,
        )
        shard_set.publish()

    return IndependentStampShardReport(
        raw_path=final_raw_path,
        coadd_paths=final_coadd_paths,
        raw_frame_count=request.shard.raw_frame_count,
        target_source_id=request.target_source_id,
        shard_id=request.shard.shard_id,
    )


def raw_stamp_delivery_frame_from_photsim7(result: Any) -> RawStampDeliveryFrame:
    """Map one Stage-9/SD-24 Photsim7 stamp result to delivery companions."""

    products = getattr(result, "stamp_products", None)
    detector = getattr(result, "detector_result", None)
    components = getattr(result, "renderer_components", None)
    if products is None or detector is None or not isinstance(components, Mapping):
        raise TypeError("result is not a compatible Photsim7 stamp pipeline result")
    final_product = getattr(products, "final_stamp", None)
    if final_product is None:
        raise ValueError("Photsim7 result is missing final_stamp")
    final_dn = _to_numpy(final_product.array)
    if getattr(final_product, "unit", None) != "dn":
        raise ValueError("formal delivery requires ADC-enabled Photsim7 final_dn")
    if "background_expectation" not in components:
        raise ValueError(
            "Photsim7 result lacks background_expectation; use "
            "artifacts.background_output_policy='expectation'"
        )
    background = _to_numpy(components["background_expectation"])
    bias_metadata = getattr(detector, "bias_metadata", None)
    if bias_metadata is None:
        raise ValueError("Photsim7 result lacks bias/column calibration metadata")
    column = getattr(bias_metadata, "column_noise_vector_adu", None)
    if column is None:
        column_vector = np.zeros(final_dn.shape[1], dtype=np.float64)
    else:
        column_array = _to_numpy(column)
        column_vector = np.asarray(column_array, dtype=np.float64).reshape(-1)
    if column_vector.shape != (final_dn.shape[1],):
        raise ValueError("Photsim7 column noise metadata does not match stamp columns")

    def product_mask(name: str) -> NDArray[np.bool_]:
        product = getattr(products, name, None)
        if product is None:
            return np.zeros(final_dn.shape, dtype=bool)
        return _to_numpy(product.array).astype(bool, copy=False)

    cosmic_product = getattr(products, "cosmic_events", None)
    cosmic_mask = (
        np.zeros(final_dn.shape, dtype=bool)
        if cosmic_product is None
        else _to_numpy(cosmic_product.mask).astype(bool, copy=False)
    )
    return RawStampDeliveryFrame(
        final_dn=final_dn,
        background_expectation_e=background,
        bias_level_dn=float(getattr(bias_metadata, "bias_level_adu")),
        column_noise_dn_by_x=column_vector,
        valid_mask=_to_numpy(getattr(products, "valid_detector_mask")),
        fullwell_mask=product_mask("full_well_clipped_mask"),
        adc_low_mask=product_mask("adc_low_clipped_mask"),
        adc_high_mask=product_mask("adc_high_clipped_mask"),
        cosmic_mask=cosmic_mask,
    )


__all__ = [
    "INDEPENDENT_STAMP_PRODUCTION_SCHEMA_ID",
    "IndependentStampShardReport",
    "IndependentStampShardRequest",
    "RawStampDeliveryFrame",
    "raw_stamp_delivery_frame_from_photsim7",
    "run_independent_stamp_time_shard",
]
