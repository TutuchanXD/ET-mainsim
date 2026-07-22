"""Atomic HDF5 delivery bundles for independent ET stamp science products.

The formal v1 delivery contract has one detector-observation product only:
``final_dn``.  Every other plane in this module is calibration, quality, time,
or provenance metadata.  In particular, ``background_expectation_e`` is an
expectation used to derive a background-subtracted electron image; a sampled
background-realization image is intentionally not part of this contract and
must never be subtracted from ``final_dn``.

One bundle represents one target and one contiguous raw or coadd time shard.
It is written to a sibling partial file, fully re-opened and validated, and
only then atomically renamed to the requested final path.  Readers reject
missing, incomplete, malformed, or semantically inconsistent bundles.

The ``to_reference_photometry_payload`` adapter deliberately matches
``ReferencePhotometryInput.from_arrays`` from ``reference_photometry_v1``.
That keeps the delivery wire format independent while allowing the standard
fixed-aperture/CDPP reduction to use the exact delivered calibration planes.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import uuid
from typing import Any, Iterator, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray


STAMP_DELIVERY_SCHEMA_ID = "et_mainsim.stamp_delivery_bundle.v1"
STAMP_DELIVERY_SCHEMA_VERSION = 1
STAMP_DELIVERY_OBSERVATION_PRODUCT = "final_dn"

DeliveryProductKind = Literal["raw", "coadd"]

_QUALITY_COUNT_NAMES = (
    "fullwell_count",
    "adc_low_count",
    "adc_high_count",
    "cosmic_count",
)
_REQUIRED_DATASETS = (
    "final_dn",
    "background_expectation_e",
    "bias_level_sum_dn",
    "column_noise_sum_dn_by_x",
    "valid_mask",
    *_QUALITY_COUNT_NAMES,
    "saturated_mask",
    "cosmic_mask",
    "time_start_seconds",
    "exposure_seconds",
    "raw_frame_start_index",
    "raw_frame_stop_index_exclusive",
    "manifest_json",
    "provenance_json",
)
_MAX_QUALITY_COUNT = int(np.iinfo(np.uint16).max)


class StampDeliveryBundleContractError(ValueError):
    """Raised when a formal stamp delivery bundle violates the v1 contract."""


def _as_array(value: ArrayLike, *, name: str) -> NDArray[np.generic]:
    array = np.asarray(value)
    if array.size == 0:
        raise StampDeliveryBundleContractError(f"{name} must not be empty")
    return array


def _as_finite_float_array(value: ArrayLike, *, name: str) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise StampDeliveryBundleContractError(
            f"{name} must contain only finite numeric values"
        )
    return array


def _as_unsigned_final_dn(
    value: ArrayLike,
    *,
    product_kind: DeliveryProductKind,
) -> NDArray[np.unsignedinteger]:
    array = _as_array(value, name="final_dn")
    if array.ndim != 3 or any(int(size) <= 0 for size in array.shape):
        raise StampDeliveryBundleContractError(
            "final_dn must have shape (n_frames, ny, nx) with positive dimensions"
        )
    if array.dtype.kind != "u":
        raise StampDeliveryBundleContractError(
            f"final_dn must use an unsigned integer DN dtype, got {array.dtype}"
        )
    if product_kind == "coadd" and array.dtype != np.dtype(np.uint64):
        raise StampDeliveryBundleContractError(
            "coadd final_dn must use uint64 so summed raw DN cannot overflow"
        )
    return array


def _as_binary_mask(
    value: ArrayLike,
    *,
    name: str,
    shape: tuple[int, int, int],
) -> NDArray[np.bool_]:
    array = _as_array(value, name=name)
    if array.shape != shape:
        raise StampDeliveryBundleContractError(
            f"{name} must have shape {shape}, got {array.shape}"
        )
    if array.dtype.kind not in {"b", "i", "u"}:
        raise StampDeliveryBundleContractError(
            f"{name} must be a boolean or integer mask, got {array.dtype}"
        )
    if not np.all((array == 0) | (array == 1)):
        raise StampDeliveryBundleContractError(f"{name} values must be exactly 0 or 1")
    return np.asarray(array, dtype=bool)


def _as_quality_count(
    value: ArrayLike,
    *,
    name: str,
    shape: tuple[int, int, int],
    coadd_factor: int,
) -> NDArray[np.uint16]:
    array = _as_array(value, name=name)
    if array.shape != shape:
        raise StampDeliveryBundleContractError(
            f"{name} must have shape {shape}, got {array.shape}"
        )
    if array.dtype.kind not in {"i", "u"}:
        raise StampDeliveryBundleContractError(
            f"{name} must be an integer count, got {array.dtype}"
        )
    if np.any(array < 0):
        raise StampDeliveryBundleContractError(f"{name} must not contain negative counts")
    if np.any(array > coadd_factor):
        raise StampDeliveryBundleContractError(
            f"{name} cannot exceed coadd_factor={coadd_factor} per output pixel"
        )
    if np.any(array > _MAX_QUALITY_COUNT):
        raise StampDeliveryBundleContractError(
            f"{name} exceeds uint16 delivery capacity {_MAX_QUALITY_COUNT}"
        )
    return np.asarray(array, dtype=np.uint16)


def _as_frame_vector(
    value: ArrayLike,
    *,
    name: str,
    n_frames: int,
    positive: bool = False,
) -> NDArray[np.float64]:
    array = _as_finite_float_array(value, name=name)
    if array.shape != (n_frames,):
        raise StampDeliveryBundleContractError(
            f"{name} must have shape ({n_frames},), got {array.shape}"
        )
    if positive and np.any(array <= 0.0):
        raise StampDeliveryBundleContractError(f"{name} must be positive")
    return array


def _as_frame_index_vector(
    value: ArrayLike,
    *,
    name: str,
    n_frames: int,
) -> NDArray[np.int64]:
    array = _as_array(value, name=name)
    if array.shape != (n_frames,):
        raise StampDeliveryBundleContractError(
            f"{name} must have shape ({n_frames},), got {array.shape}"
        )
    if array.dtype.kind not in {"i", "u"}:
        raise StampDeliveryBundleContractError(
            f"{name} must be an integer vector, got {array.dtype}"
        )
    if np.any(array < 0):
        raise StampDeliveryBundleContractError(f"{name} must not contain negatives")
    return np.asarray(array, dtype=np.int64)


def _as_json_mapping(value: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise StampDeliveryBundleContractError(f"{name} must be a JSON object mapping")
    candidate = dict(value)
    try:
        encoded = json.dumps(candidate, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as error:
        raise StampDeliveryBundleContractError(
            f"{name} must be JSON-serializable"
        ) from error
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):  # defensive: JSON encoder preserves dict here.
        raise StampDeliveryBundleContractError(f"{name} must encode as a JSON object")
    return decoded


def _normalise_product_kind(value: object) -> DeliveryProductKind:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str) or value not in {"raw", "coadd"}:
        raise StampDeliveryBundleContractError(
            "product_kind must be exactly 'raw' or 'coadd'"
        )
    return value  # type: ignore[return-value]


def _normalise_coadd_factor(value: object, *, product_kind: DeliveryProductKind) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise StampDeliveryBundleContractError("coadd_factor must be a positive integer")
    factor = int(value)
    if factor <= 0:
        raise StampDeliveryBundleContractError("coadd_factor must be a positive integer")
    if factor > _MAX_QUALITY_COUNT:
        raise StampDeliveryBundleContractError(
            f"coadd_factor must not exceed uint16 capacity {_MAX_QUALITY_COUNT}"
        )
    if product_kind == "raw" and factor != 1:
        raise StampDeliveryBundleContractError("raw products require coadd_factor=1")
    if product_kind == "coadd" and factor <= 1:
        raise StampDeliveryBundleContractError("coadd products require coadd_factor > 1")
    return factor


def _normalise_gain(
    value: ArrayLike,
    *,
    shape: tuple[int, int, int],
) -> NDArray[np.float64]:
    array = _as_finite_float_array(value, name="gain_e_per_dn")
    if np.any(array <= 0.0):
        raise StampDeliveryBundleContractError("gain_e_per_dn must be positive")
    n_frames, ny, nx = shape
    allowed_shapes = {(), (ny, nx), (n_frames, ny, nx)}
    if array.shape not in allowed_shapes:
        raise StampDeliveryBundleContractError(
            "gain_e_per_dn must be scalar, (ny, nx), or (n_frames, ny, nx); "
            f"got {array.shape}"
        )
    return array


def _h5py():
    try:
        import h5py
    except ImportError as error:  # pragma: no cover - packaging guard
        raise RuntimeError("h5py is required for stamp delivery bundles") from error
    return h5py


def _decode_h5_scalar(value: Any, *, name: str) -> Any:
    if isinstance(value, np.ndarray):
        if value.shape != ():
            raise StampDeliveryBundleContractError(f"{name} must be scalar")
        value = value.item()
    if isinstance(value, np.bytes_):
        value = bytes(value)
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _load_json_dataset(handle: Any, name: str) -> dict[str, Any]:
    if name not in handle:
        raise StampDeliveryBundleContractError(
            f"delivery bundle is missing required dataset {name!r}"
        )
    raw = _decode_h5_scalar(handle[name][()], name=name)
    if not isinstance(raw, str):
        raise StampDeliveryBundleContractError(f"{name} must store a JSON string")
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as error:
        raise StampDeliveryBundleContractError(f"{name} is not valid JSON") from error
    if not isinstance(decoded, dict):
        label = "manifest" if name == "manifest_json" else "provenance"
        raise StampDeliveryBundleContractError(f"{label} must be a JSON object")
    return decoded


def _required_dataset(handle: Any, name: str) -> NDArray[np.generic]:
    if name not in handle:
        raise StampDeliveryBundleContractError(
            f"delivery bundle is missing required dataset {name!r}"
        )
    return np.asarray(handle[name])


def _attr(handle: Any, name: str) -> Any:
    if name not in handle.attrs:
        raise StampDeliveryBundleContractError(
            f"delivery bundle is missing required root attribute {name!r}"
        )
    return _decode_h5_scalar(handle.attrs[name], name=name)


def _as_binary_attribute(value: Any, *, name: str) -> bool:
    """Read a root boolean without treating arbitrary strings as truthy."""

    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in {0, 1}:
        return bool(value)
    raise StampDeliveryBundleContractError(
        f"{name} must be a boolean or integer 0/1 root attribute"
    )


def _as_schema_version(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise StampDeliveryBundleContractError(
            f"unsupported delivery schema_version {value!r}"
        )
    return int(value)


@dataclass(frozen=True)
class StampDeliveryBundle:
    """One validated raw or coadd target-time-shard delivery product.

    Array convention: image planes have shape ``(n_frames, ny, nx)``.  The
    time vector stores absolute starts in seconds.  ``raw_frame_*`` are
    absolute, half-open raw-index intervals; their width is one for raw
    products and exactly ``coadd_factor`` for coadds.
    """

    product_kind: DeliveryProductKind
    coadd_factor: int
    final_dn: NDArray[np.unsignedinteger]
    background_expectation_e: NDArray[np.float64]
    bias_level_sum_dn: NDArray[np.float64]
    column_noise_sum_dn_by_x: NDArray[np.float64]
    valid_mask: NDArray[np.bool_]
    fullwell_count: NDArray[np.uint16]
    adc_low_count: NDArray[np.uint16]
    adc_high_count: NDArray[np.uint16]
    cosmic_count: NDArray[np.uint16]
    time_start_seconds: NDArray[np.float64]
    exposure_seconds: NDArray[np.float64]
    raw_frame_start_index: NDArray[np.int64]
    raw_frame_stop_index_exclusive: NDArray[np.int64]
    gain_e_per_dn: NDArray[np.float64]
    manifest: Mapping[str, Any]
    provenance: Mapping[str, Any]

    @classmethod
    def from_arrays(
        cls,
        *,
        product_kind: DeliveryProductKind | str,
        coadd_factor: int,
        final_dn: ArrayLike,
        background_expectation_e: ArrayLike,
        bias_level_sum_dn: ArrayLike,
        column_noise_sum_dn_by_x: ArrayLike,
        valid_mask: ArrayLike,
        fullwell_count: ArrayLike,
        adc_low_count: ArrayLike,
        adc_high_count: ArrayLike,
        cosmic_count: ArrayLike,
        time_start_seconds: ArrayLike,
        exposure_seconds: ArrayLike,
        raw_frame_start_index: ArrayLike,
        raw_frame_stop_index_exclusive: ArrayLike,
        gain_e_per_dn: ArrayLike,
        manifest: Mapping[str, Any],
        provenance: Mapping[str, Any],
    ) -> "StampDeliveryBundle":
        """Validate and normalize all v1 planes before any file is written."""

        kind = _normalise_product_kind(product_kind)
        factor = _normalise_coadd_factor(coadd_factor, product_kind=kind)
        final = _as_unsigned_final_dn(final_dn, product_kind=kind)
        n_frames, ny, nx = (int(size) for size in final.shape)
        shape = (n_frames, ny, nx)

        background = _as_finite_float_array(
            background_expectation_e,
            name="background_expectation_e",
        )
        if background.shape != shape:
            raise StampDeliveryBundleContractError(
                "background_expectation_e must have the same shape as final_dn; "
                f"got {background.shape} and {shape}"
            )
        if np.any(background < 0.0):
            raise StampDeliveryBundleContractError(
                "background_expectation_e must be non-negative"
            )

        bias = _as_frame_vector(
            bias_level_sum_dn,
            name="bias_level_sum_dn",
            n_frames=n_frames,
        )
        column = _as_finite_float_array(
            column_noise_sum_dn_by_x,
            name="column_noise_sum_dn_by_x",
        )
        if column.shape != (n_frames, nx):
            raise StampDeliveryBundleContractError(
                "column_noise_sum_dn_by_x must have shape "
                f"({n_frames}, {nx}), got {column.shape}"
            )

        starts = _as_frame_vector(
            time_start_seconds,
            name="time_start_seconds",
            n_frames=n_frames,
        )
        if n_frames > 1 and not np.all(np.diff(starts) > 0.0):
            raise StampDeliveryBundleContractError(
                "time_start_seconds must be strictly increasing"
            )
        exposures = _as_frame_vector(
            exposure_seconds,
            name="exposure_seconds",
            n_frames=n_frames,
            positive=True,
        )
        raw_start = _as_frame_index_vector(
            raw_frame_start_index,
            name="raw_frame_start_index",
            n_frames=n_frames,
        )
        raw_stop = _as_frame_index_vector(
            raw_frame_stop_index_exclusive,
            name="raw_frame_stop_index_exclusive",
            n_frames=n_frames,
        )
        if n_frames > 1 and not np.all(np.diff(raw_start) > 0):
            raise StampDeliveryBundleContractError(
                "raw_frame_start_index must be strictly increasing"
            )
        raw_width = raw_stop - raw_start
        expected_width = 1 if kind == "raw" else factor
        if not np.all(raw_width == expected_width):
            if kind == "raw":
                raise StampDeliveryBundleContractError(
                    "raw products must cover exactly one raw frame per output plane"
                )
            raise StampDeliveryBundleContractError(
                "coadd raw-frame intervals must each have width coadd_factor"
            )

        normalised_manifest = _as_json_mapping(manifest, name="manifest")
        normalised_provenance = _as_json_mapping(provenance, name="provenance")
        if normalised_provenance.get("observation_product") != (
            STAMP_DELIVERY_OBSERVATION_PRODUCT
        ):
            raise StampDeliveryBundleContractError(
                "provenance.observation_product must be 'final_dn'"
            )
        if normalised_provenance.get("background_realization_used") is not False:
            raise StampDeliveryBundleContractError(
                "provenance.background_realization_used must be false"
            )

        return cls(
            product_kind=kind,
            coadd_factor=factor,
            final_dn=final,
            background_expectation_e=background,
            bias_level_sum_dn=bias,
            column_noise_sum_dn_by_x=column,
            valid_mask=_as_binary_mask(valid_mask, name="valid_mask", shape=shape),
            fullwell_count=_as_quality_count(
                fullwell_count,
                name="fullwell_count",
                shape=shape,
                coadd_factor=factor,
            ),
            adc_low_count=_as_quality_count(
                adc_low_count,
                name="adc_low_count",
                shape=shape,
                coadd_factor=factor,
            ),
            adc_high_count=_as_quality_count(
                adc_high_count,
                name="adc_high_count",
                shape=shape,
                coadd_factor=factor,
            ),
            cosmic_count=_as_quality_count(
                cosmic_count,
                name="cosmic_count",
                shape=shape,
                coadd_factor=factor,
            ),
            time_start_seconds=starts,
            exposure_seconds=exposures,
            raw_frame_start_index=raw_start,
            raw_frame_stop_index_exclusive=raw_stop,
            gain_e_per_dn=_normalise_gain(gain_e_per_dn, shape=shape),
            manifest=normalised_manifest,
            provenance=normalised_provenance,
        )

    @property
    def shape(self) -> tuple[int, int, int]:
        """Return ``(n_frames, ny, nx)`` for delivered image planes."""

        return tuple(int(size) for size in self.final_dn.shape)  # type: ignore[return-value]

    @property
    def saturated_mask(self) -> NDArray[np.bool_]:
        """Return the derived full-well-or-ADC saturation mask."""

        return (
            (self.fullwell_count > 0)
            | (self.adc_low_count > 0)
            | (self.adc_high_count > 0)
        )

    @property
    def cosmic_mask(self) -> NDArray[np.bool_]:
        """Return the derived any-raw-exposure cosmic-ray mask."""

        return self.cosmic_count > 0

    def to_reference_photometry_payload(self) -> dict[str, Any]:
        """Return an exact ``ReferencePhotometryInput.from_arrays`` payload.

        ``time_index`` is deliberately the delivered physical start time, in
        seconds.  The adapter exposes no sampled background-realization plane,
        so a consumer cannot accidentally remove Poisson background noise by
        using this supported reduction interface.
        """

        return {
            "final_dn": self.final_dn,
            "background_expectation_e": self.background_expectation_e,
            "bias_level_sum_dn": self.bias_level_sum_dn,
            "column_noise_sum_dn_by_x": self.column_noise_sum_dn_by_x,
            "valid_mask": self.valid_mask,
            "saturated_mask": self.saturated_mask,
            "cosmic_mask": self.cosmic_mask,
            "time_index": self.time_start_seconds,
            "gain_e_per_dn": self.gain_e_per_dn,
            "time_index_unit": "seconds",
            "exposure_seconds": self.exposure_seconds,
        }


@dataclass(frozen=True)
class StampDeliveryBundleValidation:
    """Readback evidence returned after a complete bundle passes validation."""

    path: Path
    complete: bool
    product_kind: DeliveryProductKind
    coadd_factor: int
    frame_count: int
    stamp_shape: tuple[int, int]
    final_dn_dtype: str
    observation_product: str


def _write_dataset(handle: Any, name: str, value: ArrayLike) -> None:
    array = np.asarray(value)
    if array.ndim == 3:
        chunks = (min(int(array.shape[0]), 64), int(array.shape[1]), int(array.shape[2]))
        handle.create_dataset(name, data=array, chunks=chunks)
        return
    if array.ndim == 2:
        chunks = (min(int(array.shape[0]), 256), int(array.shape[1]))
        handle.create_dataset(name, data=array, chunks=chunks)
        return
    if array.ndim == 1:
        handle.create_dataset(name, data=array, chunks=(min(int(array.shape[0]), 4096),))
        return
    handle.create_dataset(name, data=array)


def _write_json_dataset(handle: Any, name: str, value: Mapping[str, Any]) -> None:
    h5py = _h5py()
    encoded = json.dumps(dict(value), sort_keys=True, separators=(",", ":"))
    handle.create_dataset(name, data=encoded, dtype=h5py.string_dtype(encoding="utf-8"))


def _write_bundle_file(path: Path, bundle: StampDeliveryBundle) -> None:
    h5py = _h5py()
    with h5py.File(path, "w") as handle:
        handle.attrs["schema_id"] = STAMP_DELIVERY_SCHEMA_ID
        handle.attrs["schema_version"] = STAMP_DELIVERY_SCHEMA_VERSION
        handle.attrs["complete"] = False
        handle.attrs["product_kind"] = bundle.product_kind
        handle.attrs["coadd_factor"] = bundle.coadd_factor
        handle.attrs["observation_product"] = STAMP_DELIVERY_OBSERVATION_PRODUCT
        handle.attrs["background_realization_used"] = False

        _write_dataset(handle, "final_dn", bundle.final_dn)
        _write_dataset(
            handle,
            "background_expectation_e",
            bundle.background_expectation_e,
        )
        _write_dataset(handle, "bias_level_sum_dn", bundle.bias_level_sum_dn)
        _write_dataset(
            handle,
            "column_noise_sum_dn_by_x",
            bundle.column_noise_sum_dn_by_x,
        )
        _write_dataset(handle, "valid_mask", bundle.valid_mask.astype(np.uint8, copy=False))
        _write_dataset(handle, "fullwell_count", bundle.fullwell_count)
        _write_dataset(handle, "adc_low_count", bundle.adc_low_count)
        _write_dataset(handle, "adc_high_count", bundle.adc_high_count)
        _write_dataset(handle, "cosmic_count", bundle.cosmic_count)

        # These two are normalized convenience masks, not independent
        # observations.  They keep the file directly readable by the v1
        # reference-photometry loader while count planes retain coadd detail.
        _write_dataset(
            handle,
            "saturated_mask",
            bundle.saturated_mask.astype(np.uint8, copy=False),
        )
        _write_dataset(
            handle,
            "cosmic_mask",
            bundle.cosmic_mask.astype(np.uint8, copy=False),
        )
        _write_dataset(handle, "time_start_seconds", bundle.time_start_seconds)
        _write_dataset(handle, "exposure_seconds", bundle.exposure_seconds)
        _write_dataset(
            handle,
            "raw_frame_start_index",
            bundle.raw_frame_start_index,
        )
        _write_dataset(
            handle,
            "raw_frame_stop_index_exclusive",
            bundle.raw_frame_stop_index_exclusive,
        )
        if bundle.gain_e_per_dn.shape == ():
            handle.attrs["gain_e_per_dn"] = float(bundle.gain_e_per_dn)
        else:
            _write_dataset(handle, "gain_e_per_dn", bundle.gain_e_per_dn)
        _write_json_dataset(handle, "manifest_json", bundle.manifest)
        _write_json_dataset(handle, "provenance_json", bundle.provenance)

        handle.flush()
        handle.attrs["complete"] = True
        handle.flush()

    with path.open("rb") as stream:
        os.fsync(stream.fileno())


@contextmanager
def _exclusive_output_lock(target: Path) -> Iterator[None]:
    """Acquire a small sibling lock so two writers cannot race a final path."""

    lock_path = target.with_name(f".{target.name}.lock")
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise FileExistsError(
            f"stamp delivery output is already locked: {lock_path}"
        ) from error
    try:
        os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        yield
    finally:
        os.close(descriptor)
        try:
            lock_path.unlink()
        except FileNotFoundError:  # pragma: no cover - defensive cleanup
            pass


def _fsync_directory(directory: Path) -> None:
    """Persist an atomic rename where the platform permits directory fsync."""

    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:  # pragma: no cover - platform fallback
        return
    try:
        os.fsync(descriptor)
    except OSError:  # pragma: no cover - filesystem fallback
        pass
    finally:
        os.close(descriptor)


def write_stamp_delivery_bundle(
    path: Path | str,
    bundle: StampDeliveryBundle,
    *,
    overwrite: bool = False,
) -> StampDeliveryBundleValidation:
    """Atomically write a validated HDF5 delivery bundle.

    A target is never resumable from a partial product: a caller either sees a
    fully validated final filename or no final filename.  ``overwrite`` is
    explicit because formal production uses a new run/shard identity rather
    than appending to an interrupted bundle.
    """

    if not isinstance(bundle, StampDeliveryBundle):
        raise TypeError("bundle must be a StampDeliveryBundle")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.name in {"", ".", ".."}:
        raise StampDeliveryBundleContractError("path must name an HDF5 bundle file")

    partial = target.with_name(f".{target.name}.{uuid.uuid4().hex}.partial")
    with _exclusive_output_lock(target):
        if target.exists() and not overwrite:
            raise FileExistsError(
                f"stamp delivery bundle already exists: {target}; use a new shard "
                "identity instead of resuming it"
            )
        try:
            _write_bundle_file(partial, bundle)
            report = validate_stamp_delivery_bundle(partial)
            os.replace(partial, target)
            _fsync_directory(target.parent)
            return StampDeliveryBundleValidation(
                path=target,
                complete=report.complete,
                product_kind=report.product_kind,
                coadd_factor=report.coadd_factor,
                frame_count=report.frame_count,
                stamp_shape=report.stamp_shape,
                final_dn_dtype=report.final_dn_dtype,
                observation_product=report.observation_product,
            )
        except BaseException:
            try:
                partial.unlink()
            except FileNotFoundError:
                pass
            raise


def read_stamp_delivery_bundle(path: Path | str) -> StampDeliveryBundle:
    """Read a complete v1 bundle and fail closed on every contract mismatch."""

    h5py = _h5py()
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"stamp delivery bundle does not exist: {source}")
    with h5py.File(source, "r") as handle:
        schema_id = _attr(handle, "schema_id")
        if schema_id != STAMP_DELIVERY_SCHEMA_ID:
            raise StampDeliveryBundleContractError(
                f"unsupported delivery schema_id {schema_id!r}"
            )
        schema_version = _as_schema_version(_attr(handle, "schema_version"))
        if schema_version != STAMP_DELIVERY_SCHEMA_VERSION:
            raise StampDeliveryBundleContractError(
                f"unsupported delivery schema_version {schema_version!r}"
            )
        complete = _as_binary_attribute(_attr(handle, "complete"), name="complete")
        if complete is not True:
            raise StampDeliveryBundleContractError("delivery bundle is not complete")
        product_kind = _normalise_product_kind(_attr(handle, "product_kind"))
        coadd_factor = _normalise_coadd_factor(
            _attr(handle, "coadd_factor"),
            product_kind=product_kind,
        )
        observation_product = _attr(handle, "observation_product")
        if observation_product != STAMP_DELIVERY_OBSERVATION_PRODUCT:
            raise StampDeliveryBundleContractError(
                "delivery root observation_product must be 'final_dn'"
            )
        if _as_binary_attribute(
            _attr(handle, "background_realization_used"),
            name="background_realization_used",
        ) is not False:
            raise StampDeliveryBundleContractError(
                "delivery root background_realization_used must be false"
            )

        for name in _REQUIRED_DATASETS:
            if name not in handle:
                raise StampDeliveryBundleContractError(
                    f"delivery bundle is missing required dataset {name!r}"
                )
        if "gain_e_per_dn" in handle and "gain_e_per_dn" in handle.attrs:
            raise StampDeliveryBundleContractError(
                "gain_e_per_dn must be stored as either one dataset or one root attribute"
            )
        if "gain_e_per_dn" in handle:
            gain: ArrayLike = np.asarray(handle["gain_e_per_dn"])
        elif "gain_e_per_dn" in handle.attrs:
            gain = _attr(handle, "gain_e_per_dn")
        else:
            raise StampDeliveryBundleContractError(
                "delivery bundle is missing gain_e_per_dn calibration metadata"
            )

        result = StampDeliveryBundle.from_arrays(
            product_kind=product_kind,
            coadd_factor=coadd_factor,
            final_dn=_required_dataset(handle, "final_dn"),
            background_expectation_e=_required_dataset(
                handle,
                "background_expectation_e",
            ),
            bias_level_sum_dn=_required_dataset(handle, "bias_level_sum_dn"),
            column_noise_sum_dn_by_x=_required_dataset(
                handle,
                "column_noise_sum_dn_by_x",
            ),
            valid_mask=_required_dataset(handle, "valid_mask"),
            fullwell_count=_required_dataset(handle, "fullwell_count"),
            adc_low_count=_required_dataset(handle, "adc_low_count"),
            adc_high_count=_required_dataset(handle, "adc_high_count"),
            cosmic_count=_required_dataset(handle, "cosmic_count"),
            time_start_seconds=_required_dataset(handle, "time_start_seconds"),
            exposure_seconds=_required_dataset(handle, "exposure_seconds"),
            raw_frame_start_index=_required_dataset(
                handle,
                "raw_frame_start_index",
            ),
            raw_frame_stop_index_exclusive=_required_dataset(
                handle,
                "raw_frame_stop_index_exclusive",
            ),
            gain_e_per_dn=gain,
            manifest=_load_json_dataset(handle, "manifest_json"),
            provenance=_load_json_dataset(handle, "provenance_json"),
        )

        persisted_saturated = _as_binary_mask(
            _required_dataset(handle, "saturated_mask"),
            name="saturated_mask",
            shape=result.shape,
        )
        persisted_cosmic = _as_binary_mask(
            _required_dataset(handle, "cosmic_mask"),
            name="cosmic_mask",
            shape=result.shape,
        )
        if not np.array_equal(persisted_saturated, result.saturated_mask):
            raise StampDeliveryBundleContractError(
                "saturated_mask must equal fullwell/ADC quality counts > 0"
            )
        if not np.array_equal(persisted_cosmic, result.cosmic_mask):
            raise StampDeliveryBundleContractError(
                "cosmic_mask must equal cosmic_count > 0"
            )
        return result


def validate_stamp_delivery_bundle(path: Path | str) -> StampDeliveryBundleValidation:
    """Open, validate, and summarize a complete formal delivery bundle."""

    source = Path(path)
    bundle = read_stamp_delivery_bundle(source)
    _, ny, nx = bundle.shape
    return StampDeliveryBundleValidation(
        path=source,
        complete=True,
        product_kind=bundle.product_kind,
        coadd_factor=bundle.coadd_factor,
        frame_count=bundle.shape[0],
        stamp_shape=(ny, nx),
        final_dn_dtype=str(bundle.final_dn.dtype),
        observation_product=STAMP_DELIVERY_OBSERVATION_PRODUCT,
    )


__all__ = [
    "STAMP_DELIVERY_OBSERVATION_PRODUCT",
    "STAMP_DELIVERY_SCHEMA_ID",
    "STAMP_DELIVERY_SCHEMA_VERSION",
    "StampDeliveryBundle",
    "StampDeliveryBundleContractError",
    "StampDeliveryBundleValidation",
    "read_stamp_delivery_bundle",
    "validate_stamp_delivery_bundle",
    "write_stamp_delivery_bundle",
]
