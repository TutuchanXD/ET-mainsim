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

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import uuid
from typing import Any, Iterator, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray


STAMP_DELIVERY_SCHEMA_ID = "et_mainsim.stamp_delivery_bundle.v2"
STAMP_DELIVERY_SCHEMA_VERSION = 2
STAMP_DELIVERY_OBSERVATION_PRODUCT = "final_dn"
STAMP_DELIVERY_CAPTURE_DENOMINATOR = "source_effective_photon_count_electron"
STAMP_DELIVERY_CAPTURE_QA_DEFINITION = (
    "no_detector_edge_or_requested_window_truncation"
)

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
    "captured_flux_fraction",
    "captured_flux_denominator_e",
    "captured_flux_qa_pass",
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
    """Raised when a formal stamp delivery bundle violates the v2 contract."""


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


def _as_captured_flux_fraction(
    value: ArrayLike,
    *,
    n_frames: int,
) -> NDArray[np.float64]:
    array = _as_frame_vector(
        value,
        name="captured_flux_fraction",
        n_frames=n_frames,
    )
    tolerance = 1.0e-6
    if np.any(array < 0.0) or np.any(array > 1.0 + tolerance):
        raise StampDeliveryBundleContractError(
            "captured_flux_fraction must lie in [0, 1] within 1e-6 "
            "numerical tolerance"
        )
    return array


def _as_frame_binary_vector(
    value: ArrayLike,
    *,
    name: str,
    n_frames: int,
) -> NDArray[np.bool_]:
    array = _as_array(value, name=name)
    if array.shape != (n_frames,):
        raise StampDeliveryBundleContractError(
            f"{name} must have shape ({n_frames},), got {array.shape}"
        )
    if array.dtype.kind not in {"b", "i", "u"} or not np.all(
        (array == 0) | (array == 1)
    ):
        raise StampDeliveryBundleContractError(
            f"{name} must contain only boolean/0/1 values"
        )
    return np.asarray(array, dtype=bool)


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


def _validate_capture_semantics(handle: Any) -> None:
    if _attr(handle, "captured_flux_fraction_denominator") != (
        STAMP_DELIVERY_CAPTURE_DENOMINATOR
    ):
        raise StampDeliveryBundleContractError(
            "captured_flux_fraction_denominator has unsupported semantics"
        )
    if _attr(handle, "captured_flux_qa_definition") != (
        STAMP_DELIVERY_CAPTURE_QA_DEFINITION
    ):
        raise StampDeliveryBundleContractError(
            "captured_flux_qa_definition has unsupported semantics"
        )


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
    captured_flux_fraction: NDArray[np.float64]
    captured_flux_denominator_e: NDArray[np.float64]
    captured_flux_qa_pass: NDArray[np.bool_]
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
        captured_flux_fraction: ArrayLike,
        captured_flux_denominator_e: ArrayLike,
        captured_flux_qa_pass: ArrayLike,
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
            captured_flux_fraction=_as_captured_flux_fraction(
                captured_flux_fraction,
                n_frames=n_frames,
            ),
            captured_flux_denominator_e=_as_frame_vector(
                captured_flux_denominator_e,
                name="captured_flux_denominator_e",
                n_frames=n_frames,
                positive=True,
            ),
            captured_flux_qa_pass=_as_frame_binary_vector(
                captured_flux_qa_pass,
                name="captured_flux_qa_pass",
                n_frames=n_frames,
            ),
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
        handle.attrs["captured_flux_fraction_denominator"] = (
            STAMP_DELIVERY_CAPTURE_DENOMINATOR
        )
        handle.attrs["captured_flux_qa_definition"] = (
            STAMP_DELIVERY_CAPTURE_QA_DEFINITION
        )

        _write_dataset(handle, "final_dn", bundle.final_dn)
        _write_dataset(
            handle,
            "background_expectation_e",
            bundle.background_expectation_e,
        )
        _write_dataset(
            handle,
            "captured_flux_fraction",
            bundle.captured_flux_fraction,
        )
        _write_dataset(
            handle,
            "captured_flux_denominator_e",
            bundle.captured_flux_denominator_e,
        )
        _write_dataset(
            handle,
            "captured_flux_qa_pass",
            bundle.captured_flux_qa_pass.astype(np.uint8, copy=False),
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


def _normalise_shard_set_member(value: Path | str) -> Path:
    """Return one safe, relative member name for a staged shard directory."""

    candidate = Path(value)
    if (
        candidate.is_absolute()
        or not candidate.parts
        or any(part in {"", ".", ".."} for part in candidate.parts)
    ):
        raise StampDeliveryBundleContractError(
            "shard-set members must be non-empty relative paths without '..'"
        )
    return candidate


class StampDeliveryShardSet:
    """Atomically publish a complete set of bundle files as one shard directory.

    Each member bundle retains its own file-level atomic write/readback
    contract.  This publisher adds the production-level guarantee that the
    final ``shard_<id>`` directory becomes visible only after *every* expected
    member exists in a private sibling staging directory.  Directory rename is
    atomic on the one filesystem containing the target and staging paths; a
    failure before that rename removes the entire staged set and leaves no
    final member visible.

    Existing formal shard directories are intentionally never replaced.  A
    completed shard is non-resumable, and replacing a non-empty directory
    cannot provide the same all-members-at-once publication guarantee.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        expected_members: Sequence[Path | str],
        overwrite: bool = False,
    ) -> None:
        self._target = Path(path)
        self._target.parent.mkdir(parents=True, exist_ok=True)
        if self._target.name in {"", ".", ".."}:
            raise StampDeliveryBundleContractError(
                "path must name a final shard directory"
            )
        members = tuple(_normalise_shard_set_member(value) for value in expected_members)
        if not members:
            raise StampDeliveryBundleContractError(
                "a shard set must declare at least one expected member"
            )
        if len(set(members)) != len(members):
            raise StampDeliveryBundleContractError(
                "a shard set must not declare duplicate member paths"
            )
        self._expected_members = members
        self._overwrite = bool(overwrite)
        self._partial = self._target.with_name(
            f".{self._target.name}.{uuid.uuid4().hex}.partial"
        )
        self._lock_context = _exclusive_output_lock(self._target)
        self._lock_held = False
        self._state = "new"

        try:
            self._lock_context.__enter__()
            self._lock_held = True
            if self._target.exists():
                overwrite_note = (
                    "; overwrite is not supported for shard sets"
                    if self._overwrite
                    else ""
                )
                raise FileExistsError(
                    "stamp delivery shard set already exists: "
                    f"{self._target}; use a new shard identity instead of resuming it"
                    f"{overwrite_note}"
                )
            self._partial.mkdir()
            self._state = "open"
        except BaseException:
            self.abort()
            raise

    @property
    def path(self) -> Path:
        """Return the final shard directory, invisible until ``publish()``."""

        return self._target

    @property
    def staging_root(self) -> Path:
        """Return the private staging directory while the set is open."""

        if self._state != "open":
            raise RuntimeError("stamp delivery shard set is not open")
        return self._partial

    def _release_lock(self) -> None:
        if self._lock_held:
            self._lock_context.__exit__(None, None, None)
            self._lock_held = False

    def member_path(self, member: Path | str) -> Path:
        """Return the private staged pathname for one declared member."""

        if self._state != "open":
            raise RuntimeError("stamp delivery shard set is not open")
        normalised = _normalise_shard_set_member(member)
        if normalised not in self._expected_members:
            raise StampDeliveryBundleContractError(
                f"undeclared shard-set member: {normalised}"
            )
        return self._partial / normalised

    def publish(self) -> Path:
        """Atomically rename the fully staged member directory into place."""

        if self._state != "open":
            raise RuntimeError("stamp delivery shard set is not open")
        missing = [
            str(member)
            for member in self._expected_members
            if not (self._partial / member).is_file()
        ]
        if missing:
            raise StampDeliveryBundleContractError(
                "cannot publish an incomplete stamp delivery shard set; missing "
                + ", ".join(missing)
            )
        try:
            os.replace(self._partial, self._target)
        except BaseException:
            self.abort()
            raise
        self._state = "completed"
        try:
            _fsync_directory(self._target.parent)
        finally:
            self._release_lock()
        return self._target

    def abort(self) -> None:
        """Discard every unpublished member in the private staging directory."""

        if self._state == "completed":
            return
        try:
            shutil.rmtree(self._partial)
        except FileNotFoundError:
            pass
        self._state = "aborted"
        self._release_lock()

    def __enter__(self) -> "StampDeliveryShardSet":
        if self._state != "open":
            raise RuntimeError("stamp delivery shard set is not open")
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        if self._state == "open":
            self.abort()
        return False


def _validate_appender_stamp_shape(value: tuple[int, int]) -> tuple[int, int]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise StampDeliveryBundleContractError(
            "stamp_shape must be a (ny, nx) tuple"
        )
    ny, nx = value
    if isinstance(ny, bool) or isinstance(nx, bool):
        raise StampDeliveryBundleContractError("stamp_shape dimensions must be integers")
    try:
        normalised = (int(ny), int(nx))
    except (TypeError, ValueError) as error:
        raise StampDeliveryBundleContractError(
            "stamp_shape dimensions must be integers"
        ) from error
    if normalised[0] <= 0 or normalised[1] <= 0:
        raise StampDeliveryBundleContractError("stamp_shape dimensions must be positive")
    return normalised


def _validate_delivery_provenance(value: Mapping[str, Any]) -> dict[str, Any]:
    normalised = _as_json_mapping(value, name="provenance")
    if normalised.get("observation_product") != STAMP_DELIVERY_OBSERVATION_PRODUCT:
        raise StampDeliveryBundleContractError(
            "provenance.observation_product must be 'final_dn'"
        )
    if normalised.get("background_realization_used") is not False:
        raise StampDeliveryBundleContractError(
            "provenance.background_realization_used must be false"
        )
    return normalised


def _create_appendable_dataset(
    handle: Any,
    name: str,
    *,
    tail_shape: tuple[int, ...],
    dtype: np.dtype[Any],
) -> None:
    if len(tail_shape) == 2:
        chunks = (64, *tail_shape)
    elif len(tail_shape) == 1:
        chunks = (256, *tail_shape)
    elif len(tail_shape) == 0:
        chunks = (4096,)
    else:  # pragma: no cover - all v1 planes have 0, 1, or 2 trailing axes.
        raise AssertionError(f"unsupported appendable tail shape {tail_shape}")
    handle.create_dataset(
        name,
        shape=(0, *tail_shape),
        maxshape=(None, *tail_shape),
        chunks=chunks,
        dtype=dtype,
    )


class StampDeliveryBundleAppender:
    """Stream bounded batches into one invisible partial delivery bundle.

    The appender deliberately does not offer resume or post-completion append.
    Batches are appended only to one sibling ``.partial`` file while a small
    output lock is held.  ``complete()`` marks that file complete, closes and
    fully validates it, then atomically renames it to the requested final
    pathname.  If a worker exits its context without ``complete()``, the
    partial product is deleted.

    ``gain_e_per_dn`` is static for one streamed shard: scalar or ``(ny,nx)``.
    A time-varying gain plane is valid in the one-shot bundle API but is not
    accepted here until a producer has a concrete per-frame calibration use
    case.  This avoids silently mixing gain contracts across independently
    produced batches.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        product_kind: DeliveryProductKind | str,
        coadd_factor: int,
        stamp_shape: tuple[int, int],
        gain_e_per_dn: ArrayLike,
        manifest: Mapping[str, Any],
        provenance: Mapping[str, Any],
        overwrite: bool = False,
    ) -> None:
        self._target = Path(path)
        self._target.parent.mkdir(parents=True, exist_ok=True)
        if self._target.name in {"", ".", ".."}:
            raise StampDeliveryBundleContractError(
                "path must name an HDF5 bundle file"
            )
        self._product_kind = _normalise_product_kind(product_kind)
        self._coadd_factor = _normalise_coadd_factor(
            coadd_factor,
            product_kind=self._product_kind,
        )
        self._stamp_shape = _validate_appender_stamp_shape(stamp_shape)
        raw_gain = _as_finite_float_array(gain_e_per_dn, name="gain_e_per_dn")
        if np.any(raw_gain <= 0.0) or raw_gain.shape not in {
            (),
            self._stamp_shape,
        }:
            raise StampDeliveryBundleContractError(
                "streaming gain_e_per_dn must be positive scalar or (ny, nx)"
            )
        self._gain_e_per_dn = raw_gain
        self._manifest = _as_json_mapping(manifest, name="manifest")
        self._provenance = _validate_delivery_provenance(provenance)
        self._overwrite = bool(overwrite)
        self._partial = self._target.with_name(
            f".{self._target.name}.{uuid.uuid4().hex}.partial"
        )
        self._lock_context = _exclusive_output_lock(self._target)
        self._lock_held = False
        self._handle: Any | None = None
        self._frame_count = 0
        self._final_dn_dtype: np.dtype[Any] | None = None
        self._last_time_end: float | None = None
        self._last_raw_stop: int | None = None
        self._state = "new"

        try:
            self._lock_context.__enter__()
            self._lock_held = True
            if self._target.exists() and not self._overwrite:
                raise FileExistsError(
                    f"stamp delivery bundle already exists: {self._target}; use a new "
                    "shard identity instead of resuming it"
                )
            h5py = _h5py()
            self._handle = h5py.File(self._partial, "w")
            self._handle.attrs["schema_id"] = STAMP_DELIVERY_SCHEMA_ID
            self._handle.attrs["schema_version"] = STAMP_DELIVERY_SCHEMA_VERSION
            self._handle.attrs["complete"] = False
            self._handle.attrs["product_kind"] = self._product_kind
            self._handle.attrs["coadd_factor"] = self._coadd_factor
            self._handle.attrs["observation_product"] = STAMP_DELIVERY_OBSERVATION_PRODUCT
            self._handle.attrs["background_realization_used"] = False
            self._handle.attrs["captured_flux_fraction_denominator"] = (
                STAMP_DELIVERY_CAPTURE_DENOMINATOR
            )
            self._handle.attrs["captured_flux_qa_definition"] = (
                STAMP_DELIVERY_CAPTURE_QA_DEFINITION
            )
            if self._gain_e_per_dn.shape == ():
                self._handle.attrs["gain_e_per_dn"] = float(self._gain_e_per_dn)
            else:
                _write_dataset(self._handle, "gain_e_per_dn", self._gain_e_per_dn)
            _write_json_dataset(self._handle, "manifest_json", self._manifest)
            _write_json_dataset(self._handle, "provenance_json", self._provenance)
            self._handle.flush()
            self._state = "open"
        except BaseException:
            self.abort()
            raise

    @property
    def path(self) -> Path:
        """Return the final pathname, which remains invisible until completion."""

        return self._target

    @property
    def frame_count(self) -> int:
        """Return the number of successfully appended output planes."""

        return self._frame_count

    def _require_open(self) -> Any:
        if self._state == "completed":
            raise RuntimeError("stamp delivery appender is already completed")
        if self._state != "open" or self._handle is None:
            raise RuntimeError("stamp delivery appender is not open")
        return self._handle

    def _release_lock(self) -> None:
        if self._lock_held:
            self._lock_context.__exit__(None, None, None)
            self._lock_held = False

    def _create_frame_datasets(self, bundle: StampDeliveryBundle) -> None:
        assert self._handle is not None
        _, ny, nx = bundle.shape
        for name, tail_shape, dtype in (
            ("final_dn", (ny, nx), bundle.final_dn.dtype),
            ("background_expectation_e", (ny, nx), np.dtype(np.float64)),
            ("captured_flux_fraction", (), np.dtype(np.float64)),
            ("captured_flux_denominator_e", (), np.dtype(np.float64)),
            ("captured_flux_qa_pass", (), np.dtype(np.uint8)),
            ("bias_level_sum_dn", (), np.dtype(np.float64)),
            ("column_noise_sum_dn_by_x", (nx,), np.dtype(np.float64)),
            ("valid_mask", (ny, nx), np.dtype(np.uint8)),
            ("fullwell_count", (ny, nx), np.dtype(np.uint16)),
            ("adc_low_count", (ny, nx), np.dtype(np.uint16)),
            ("adc_high_count", (ny, nx), np.dtype(np.uint16)),
            ("cosmic_count", (ny, nx), np.dtype(np.uint16)),
            ("saturated_mask", (ny, nx), np.dtype(np.uint8)),
            ("cosmic_mask", (ny, nx), np.dtype(np.uint8)),
            ("time_start_seconds", (), np.dtype(np.float64)),
            ("exposure_seconds", (), np.dtype(np.float64)),
            ("raw_frame_start_index", (), np.dtype(np.int64)),
            ("raw_frame_stop_index_exclusive", (), np.dtype(np.int64)),
        ):
            _create_appendable_dataset(
                self._handle,
                name,
                tail_shape=tail_shape,
                dtype=np.dtype(dtype),
            )
        self._final_dn_dtype = bundle.final_dn.dtype

    def _validate_batch_identity(self, bundle: StampDeliveryBundle) -> None:
        if not isinstance(bundle, StampDeliveryBundle):
            raise TypeError("bundle must be a StampDeliveryBundle")
        if bundle.product_kind != self._product_kind:
            raise StampDeliveryBundleContractError(
                "streamed batch product_kind differs from appender product_kind"
            )
        if bundle.coadd_factor != self._coadd_factor:
            raise StampDeliveryBundleContractError(
                "streamed batch coadd_factor differs from appender coadd_factor"
            )
        if bundle.shape[1:] != self._stamp_shape:
            raise StampDeliveryBundleContractError(
                "streamed batch stamp shape differs from appender stamp_shape"
            )
        if bundle.manifest != self._manifest:
            raise StampDeliveryBundleContractError(
                "streamed batch manifest differs from appender manifest"
            )
        if bundle.provenance != self._provenance:
            raise StampDeliveryBundleContractError(
                "streamed batch provenance differs from appender provenance"
            )
        if not np.array_equal(bundle.gain_e_per_dn, self._gain_e_per_dn):
            raise StampDeliveryBundleContractError(
                "streamed batch gain_e_per_dn differs from appender calibration"
            )
        if self._final_dn_dtype is not None and bundle.final_dn.dtype != (
            self._final_dn_dtype
        ):
            raise StampDeliveryBundleContractError(
                "streamed batch final_dn dtype differs from the first batch"
            )
        if self._last_time_end is not None and float(bundle.time_start_seconds[0]) < (
            self._last_time_end
        ):
            raise StampDeliveryBundleContractError(
                "streamed batches overlap or reverse time_start_seconds"
            )
        if self._last_raw_stop is not None and int(bundle.raw_frame_start_index[0]) != (
            self._last_raw_stop
        ):
            raise StampDeliveryBundleContractError(
                "streamed batch raw-frame start must equal previous batch stop"
            )

    def append(self, bundle: StampDeliveryBundle) -> int:
        """Append one validated bounded batch and return total output frames."""

        handle = self._require_open()
        self._validate_batch_identity(bundle)
        if self._frame_count == 0:
            self._create_frame_datasets(bundle)
        assert self._final_dn_dtype is not None
        batch_count = bundle.shape[0]
        start = self._frame_count
        stop = start + batch_count
        payloads: tuple[tuple[str, NDArray[np.generic]], ...] = (
            ("final_dn", bundle.final_dn),
            ("background_expectation_e", bundle.background_expectation_e),
            ("captured_flux_fraction", bundle.captured_flux_fraction),
            (
                "captured_flux_denominator_e",
                bundle.captured_flux_denominator_e,
            ),
            (
                "captured_flux_qa_pass",
                bundle.captured_flux_qa_pass.astype(np.uint8, copy=False),
            ),
            ("bias_level_sum_dn", bundle.bias_level_sum_dn),
            ("column_noise_sum_dn_by_x", bundle.column_noise_sum_dn_by_x),
            ("valid_mask", bundle.valid_mask.astype(np.uint8, copy=False)),
            ("fullwell_count", bundle.fullwell_count),
            ("adc_low_count", bundle.adc_low_count),
            ("adc_high_count", bundle.adc_high_count),
            ("cosmic_count", bundle.cosmic_count),
            ("saturated_mask", bundle.saturated_mask.astype(np.uint8, copy=False)),
            ("cosmic_mask", bundle.cosmic_mask.astype(np.uint8, copy=False)),
            ("time_start_seconds", bundle.time_start_seconds),
            ("exposure_seconds", bundle.exposure_seconds),
            ("raw_frame_start_index", bundle.raw_frame_start_index),
            ("raw_frame_stop_index_exclusive", bundle.raw_frame_stop_index_exclusive),
        )
        for name, value in payloads:
            dataset = handle[name]
            dataset.resize((stop, *dataset.shape[1:]))
            dataset[start:stop] = value
        self._frame_count = stop
        self._last_time_end = float(
            bundle.time_start_seconds[-1] + bundle.exposure_seconds[-1]
        )
        self._last_raw_stop = int(bundle.raw_frame_stop_index_exclusive[-1])
        handle.flush()
        return self._frame_count

    def complete(self) -> StampDeliveryBundleValidation:
        """Validate the staged file and atomically publish its final pathname."""

        handle = self._require_open()
        if self._frame_count == 0:
            raise StampDeliveryBundleContractError(
                "cannot complete a delivery bundle with no appended frames"
            )
        try:
            handle.attrs["complete"] = True
            handle.flush()
            handle.close()
            self._handle = None
            with self._partial.open("rb") as stream:
                os.fsync(stream.fileno())
            report = validate_stamp_delivery_bundle(self._partial)
            if self._target.exists() and not self._overwrite:
                raise FileExistsError(
                    f"stamp delivery bundle already exists: {self._target}; refusing "
                    "to replace it"
                )
            os.replace(self._partial, self._target)
            _fsync_directory(self._target.parent)
            self._state = "completed"
            self._release_lock()
            return StampDeliveryBundleValidation(
                path=self._target,
                complete=report.complete,
                product_kind=report.product_kind,
                coadd_factor=report.coadd_factor,
                frame_count=report.frame_count,
                stamp_shape=report.stamp_shape,
                final_dn_dtype=report.final_dn_dtype,
                observation_product=report.observation_product,
            )
        except BaseException:
            self.abort()
            raise

    def abort(self) -> None:
        """Close and delete an unpublished partial bundle, if one exists."""

        if self._state == "completed":
            return
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        try:
            self._partial.unlink()
        except FileNotFoundError:
            pass
        self._state = "aborted"
        self._release_lock()

    def __enter__(self) -> "StampDeliveryBundleAppender":
        self._require_open()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        if self._state == "open":
            self.abort()
        return False


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
        _validate_capture_semantics(handle)

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
            captured_flux_fraction=_required_dataset(
                handle,
                "captured_flux_fraction",
            ),
            captured_flux_denominator_e=_required_dataset(
                handle,
                "captured_flux_denominator_e",
            ),
            captured_flux_qa_pass=_required_dataset(
                handle,
                "captured_flux_qa_pass",
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


def _streaming_gain_for_chunk(
    gain_dataset: Any | None,
    gain_attribute: ArrayLike | None,
    *,
    frame_slice: slice,
    n_frames: int,
    ny: int,
    nx: int,
) -> NDArray[np.generic] | ArrayLike:
    """Read only the gain calibration needed for one validation chunk."""

    if gain_dataset is None:
        assert gain_attribute is not None
        return gain_attribute
    shape = tuple(int(size) for size in gain_dataset.shape)
    allowed_shapes = {(), (ny, nx), (n_frames, ny, nx)}
    if shape not in allowed_shapes:
        raise StampDeliveryBundleContractError(
            "gain_e_per_dn must be scalar, (ny, nx), or (n_frames, ny, nx); "
            f"got {shape}"
        )
    if shape == ():
        return np.asarray(gain_dataset[()])
    if shape == (ny, nx):
        return np.asarray(gain_dataset)
    return np.asarray(gain_dataset[frame_slice])


def _stream_validate_stamp_delivery_bundle(
    path: Path | str,
    *,
    batch_frames: int = 16,
) -> StampDeliveryBundleValidation:
    """Validate a complete HDF5 bundle in bounded frame batches.

    The public reader intentionally returns full image cubes for downstream
    analysis.  Completion-time validation has a different job: it must prove
    the on-disk contract without creating another full in-memory cube.  This
    routine therefore reuses ``StampDeliveryBundle.from_arrays`` on small
    frame slices and checks cross-slice monotonicity plus persisted derived
    masks explicitly.
    """

    if isinstance(batch_frames, (bool, np.bool_)) or not isinstance(
        batch_frames,
        (int, np.integer),
    ) or int(batch_frames) <= 0:
        raise ValueError("batch_frames must be a positive integer")
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"stamp delivery bundle does not exist: {source}")
    h5py = _h5py()
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
        _validate_capture_semantics(handle)

        for name in _REQUIRED_DATASETS:
            if name not in handle:
                raise StampDeliveryBundleContractError(
                    f"delivery bundle is missing required dataset {name!r}"
                )
        if "gain_e_per_dn" in handle and "gain_e_per_dn" in handle.attrs:
            raise StampDeliveryBundleContractError(
                "gain_e_per_dn must be stored as either one dataset or one root attribute"
            )
        gain_dataset: Any | None
        gain_attribute: ArrayLike | None
        if "gain_e_per_dn" in handle:
            gain_dataset = handle["gain_e_per_dn"]
            gain_attribute = None
        elif "gain_e_per_dn" in handle.attrs:
            gain_dataset = None
            gain_attribute = _attr(handle, "gain_e_per_dn")
        else:
            raise StampDeliveryBundleContractError(
                "delivery bundle is missing gain_e_per_dn calibration metadata"
            )

        manifest = _load_json_dataset(handle, "manifest_json")
        provenance = _load_json_dataset(handle, "provenance_json")
        final_dataset = handle["final_dn"]
        final_shape = tuple(int(size) for size in final_dataset.shape)
        if len(final_shape) != 3 or any(size <= 0 for size in final_shape):
            raise StampDeliveryBundleContractError(
                "final_dn must have shape (n_frames, ny, nx) with positive dimensions"
            )
        n_frames, ny, nx = final_shape
        final_dtype = np.dtype(final_dataset.dtype)
        if final_dtype.kind != "u":
            raise StampDeliveryBundleContractError(
                f"final_dn must use an unsigned integer DN dtype, got {final_dtype}"
            )
        if product_kind == "coadd" and final_dtype != np.dtype(np.uint64):
            raise StampDeliveryBundleContractError(
                "coadd final_dn must use uint64 so summed raw DN cannot overflow"
            )

        previous_time_start: float | None = None
        previous_raw_start: int | None = None
        for frame_start in range(0, n_frames, int(batch_frames)):
            frame_stop = min(frame_start + int(batch_frames), n_frames)
            frame_slice = slice(frame_start, frame_stop)
            bundle = StampDeliveryBundle.from_arrays(
                product_kind=product_kind,
                coadd_factor=coadd_factor,
                final_dn=np.asarray(final_dataset[frame_slice]),
                background_expectation_e=np.asarray(
                    handle["background_expectation_e"][frame_slice]
                ),
                captured_flux_fraction=np.asarray(
                    handle["captured_flux_fraction"][frame_slice]
                ),
                captured_flux_denominator_e=np.asarray(
                    handle["captured_flux_denominator_e"][frame_slice]
                ),
                captured_flux_qa_pass=np.asarray(
                    handle["captured_flux_qa_pass"][frame_slice]
                ),
                bias_level_sum_dn=np.asarray(
                    handle["bias_level_sum_dn"][frame_slice]
                ),
                column_noise_sum_dn_by_x=np.asarray(
                    handle["column_noise_sum_dn_by_x"][frame_slice]
                ),
                valid_mask=np.asarray(handle["valid_mask"][frame_slice]),
                fullwell_count=np.asarray(handle["fullwell_count"][frame_slice]),
                adc_low_count=np.asarray(handle["adc_low_count"][frame_slice]),
                adc_high_count=np.asarray(handle["adc_high_count"][frame_slice]),
                cosmic_count=np.asarray(handle["cosmic_count"][frame_slice]),
                time_start_seconds=np.asarray(
                    handle["time_start_seconds"][frame_slice]
                ),
                exposure_seconds=np.asarray(handle["exposure_seconds"][frame_slice]),
                raw_frame_start_index=np.asarray(
                    handle["raw_frame_start_index"][frame_slice]
                ),
                raw_frame_stop_index_exclusive=np.asarray(
                    handle["raw_frame_stop_index_exclusive"][frame_slice]
                ),
                gain_e_per_dn=_streaming_gain_for_chunk(
                    gain_dataset,
                    gain_attribute,
                    frame_slice=frame_slice,
                    n_frames=n_frames,
                    ny=ny,
                    nx=nx,
                ),
                manifest=manifest,
                provenance=provenance,
            )
            persisted_saturated = _as_binary_mask(
                np.asarray(handle["saturated_mask"][frame_slice]),
                name="saturated_mask",
                shape=bundle.shape,
            )
            persisted_cosmic = _as_binary_mask(
                np.asarray(handle["cosmic_mask"][frame_slice]),
                name="cosmic_mask",
                shape=bundle.shape,
            )
            if not np.array_equal(persisted_saturated, bundle.saturated_mask):
                raise StampDeliveryBundleContractError(
                    "saturated_mask must equal fullwell/ADC quality counts > 0"
                )
            if not np.array_equal(persisted_cosmic, bundle.cosmic_mask):
                raise StampDeliveryBundleContractError(
                    "cosmic_mask must equal cosmic_count > 0"
                )
            if (
                previous_time_start is not None
                and float(bundle.time_start_seconds[0]) <= previous_time_start
            ):
                raise StampDeliveryBundleContractError(
                    "time_start_seconds must be strictly increasing"
                )
            if (
                previous_raw_start is not None
                and int(bundle.raw_frame_start_index[0]) <= previous_raw_start
            ):
                raise StampDeliveryBundleContractError(
                    "raw_frame_start_index must be strictly increasing"
                )
            previous_time_start = float(bundle.time_start_seconds[-1])
            previous_raw_start = int(bundle.raw_frame_start_index[-1])

    return StampDeliveryBundleValidation(
        path=source,
        complete=True,
        product_kind=product_kind,
        coadd_factor=coadd_factor,
        frame_count=n_frames,
        stamp_shape=(ny, nx),
        final_dn_dtype=str(final_dtype),
        observation_product=STAMP_DELIVERY_OBSERVATION_PRODUCT,
    )


def validate_stamp_delivery_bundle(path: Path | str) -> StampDeliveryBundleValidation:
    """Stream-validate and summarize a complete formal delivery bundle."""

    return _stream_validate_stamp_delivery_bundle(path)


__all__ = [
    "STAMP_DELIVERY_CAPTURE_DENOMINATOR",
    "STAMP_DELIVERY_CAPTURE_QA_DEFINITION",
    "STAMP_DELIVERY_OBSERVATION_PRODUCT",
    "STAMP_DELIVERY_SCHEMA_ID",
    "STAMP_DELIVERY_SCHEMA_VERSION",
    "StampDeliveryBundle",
    "StampDeliveryBundleAppender",
    "StampDeliveryBundleContractError",
    "StampDeliveryShardSet",
    "StampDeliveryBundleValidation",
    "read_stamp_delivery_bundle",
    "validate_stamp_delivery_bundle",
    "write_stamp_delivery_bundle",
]
