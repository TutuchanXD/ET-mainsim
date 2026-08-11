from __future__ import annotations

import json

import h5py
import numpy as np
import pytest


def _bundle_payload(
    *,
    product_kind: str = "raw",
    coadd_factor: int = 1,
    final_dtype: np.dtype[np.unsignedinteger] = np.dtype(np.uint16),
) -> dict[str, object]:
    """Return a compact valid delivery fixture with nontrivial quality flags."""

    n_frames, ny, nx = 2, 3, 4
    final_dn = np.full((n_frames, ny, nx), 1024, dtype=final_dtype)
    fullwell_count = np.zeros((n_frames, ny, nx), dtype=np.uint16)
    adc_low_count = np.zeros((n_frames, ny, nx), dtype=np.uint16)
    adc_high_count = np.zeros((n_frames, ny, nx), dtype=np.uint16)
    cosmic_count = np.zeros((n_frames, ny, nx), dtype=np.uint16)
    fullwell_count[0, 1, 1] = 1
    adc_high_count[0, 1, 2] = 1
    cosmic_count[1, 0, 0] = 1
    return {
        "product_kind": product_kind,
        "coadd_factor": coadd_factor,
        "final_dn": final_dn,
        "background_expectation_e": np.full((n_frames, ny, nx), 3.5),
        "captured_flux_fraction": np.ones(n_frames),
        "captured_flux_denominator_e": np.full(n_frames, 1_000.0),
        "captured_flux_qa_pass": np.ones(n_frames, dtype=bool),
        "bias_level_sum_dn": np.array([100.0, 100.0]),
        "column_noise_sum_dn_by_x": np.array(
            [[-0.25, 0.0, 0.25, 0.5], [-0.25, 0.0, 0.25, 0.5]]
        ),
        "valid_mask": np.ones((n_frames, ny, nx), dtype=bool),
        "fullwell_count": fullwell_count,
        "adc_low_count": adc_low_count,
        "adc_high_count": adc_high_count,
        "cosmic_count": cosmic_count,
        "time_start_seconds": np.array([0.0, 10.0]),
        "exposure_seconds": np.array([10.0, 10.0]),
        "raw_frame_start_index": np.array([40, 41], dtype=np.int64),
        "raw_frame_stop_index_exclusive": np.array([41, 42], dtype=np.int64),
        "gain_e_per_dn": 2.0,
        "manifest": {"run_id": "science-001", "target_id": 42},
        "provenance": {
            "observation_product": "final_dn",
            "background_realization_used": False,
        },
    }


def test_atomic_write_readback_and_reference_adapter(tmp_path) -> None:
    from et_mainsim.stamp_delivery import (
        StampDeliveryBundle,
        read_stamp_delivery_bundle,
        validate_stamp_delivery_bundle,
        write_stamp_delivery_bundle,
    )

    path = tmp_path / "target-42-raw.h5"
    bundle = StampDeliveryBundle.from_arrays(**_bundle_payload())

    write_stamp_delivery_bundle(path, bundle)

    assert path.is_file()
    assert not list(tmp_path.glob("*.partial"))
    report = validate_stamp_delivery_bundle(path)
    assert report.complete is True
    assert report.product_kind == "raw"
    assert report.frame_count == 2
    assert report.stamp_shape == (3, 4)

    restored = read_stamp_delivery_bundle(path)
    assert restored.final_dn.dtype == np.uint16
    np.testing.assert_array_equal(restored.final_dn, bundle.final_dn)
    assert restored.manifest == {"run_id": "science-001", "target_id": 42}
    assert restored.provenance["observation_product"] == "final_dn"

    # This dictionary is directly consumable by
    # ReferencePhotometryInput.from_arrays once reference_photometry_v1 is
    # present on the integration branch.  In particular, no background
    # realization plane is exposed to the reducer.
    reference = restored.to_reference_photometry_payload()
    assert set(reference) == {
        "final_dn",
        "background_expectation_e",
        "bias_level_sum_dn",
        "column_noise_sum_dn_by_x",
        "valid_mask",
        "saturated_mask",
        "cosmic_mask",
        "time_index",
        "gain_e_per_dn",
        "time_index_unit",
        "exposure_seconds",
    }
    assert reference["time_index_unit"] == "seconds"
    assert reference["saturated_mask"][0, 1, 1]
    assert reference["saturated_mask"][0, 1, 2]
    assert reference["cosmic_mask"][1, 0, 0]

    # A complete final product is immutable by default; the next attempt must
    # use a distinct time-shard identity rather than accidentally append.
    with pytest.raises(FileExistsError, match="already exists"):
        write_stamp_delivery_bundle(path, bundle)
    np.testing.assert_array_equal(
        read_stamp_delivery_bundle(path).final_dn,
        bundle.final_dn,
    )


def test_formal_delivery_roundtrips_captured_flux_truth_and_qa(tmp_path) -> None:
    """Capture truth is a cadence vector; final_dn remains the sole observation."""

    from et_mainsim.stamp_delivery import (
        StampDeliveryBundle,
        read_stamp_delivery_bundle,
        write_stamp_delivery_bundle,
    )

    payload = _bundle_payload()
    payload["captured_flux_fraction"] = np.asarray([1.0, 0.999_8])
    payload["captured_flux_qa_pass"] = np.asarray([True, False])
    payload["captured_flux_denominator_e"] = np.asarray([1_000.0, 1_250.0])
    path = tmp_path / "capture-truth.h5"

    write_stamp_delivery_bundle(
        path,
        StampDeliveryBundle.from_arrays(**payload),
    )
    restored = read_stamp_delivery_bundle(path)

    np.testing.assert_allclose(restored.captured_flux_fraction, [1.0, 0.999_8])
    np.testing.assert_array_equal(restored.captured_flux_qa_pass, [True, False])
    np.testing.assert_allclose(
        restored.captured_flux_denominator_e,
        [1_000.0, 1_250.0],
    )
    with h5py.File(path, "r") as handle:
        assert handle.attrs["observation_product"] == "final_dn"
        assert bool(handle.attrs["background_realization_used"]) is False
        assert handle["captured_flux_fraction"].shape == (2,)
        assert handle["captured_flux_qa_pass"].shape == (2,)
        assert handle["captured_flux_denominator_e"].shape == (2,)


def test_validation_streams_hdf5_without_materializing_the_bundle(
    tmp_path,
    monkeypatch,
) -> None:
    """Appender/readback validation must not depend on the full-array reader."""

    import et_mainsim.stamp_delivery as delivery

    path = tmp_path / "stream-validate.h5"
    delivery.write_stamp_delivery_bundle(
        path,
        delivery.StampDeliveryBundle.from_arrays(**_bundle_payload()),
    )

    def fail_full_reader(*args, **kwargs):
        raise AssertionError("full-array reader must not be used by validation")

    monkeypatch.setattr(delivery, "read_stamp_delivery_bundle", fail_full_reader)
    report = delivery.validate_stamp_delivery_bundle(path)

    assert report.path == path
    assert report.complete is True
    assert report.frame_count == 2


def test_coadd_requires_uint64_final_dn_and_bounded_quality_counts(tmp_path) -> None:
    from et_mainsim.stamp_delivery import (
        StampDeliveryBundle,
        StampDeliveryBundleContractError,
        read_stamp_delivery_bundle,
        write_stamp_delivery_bundle,
    )

    invalid = _bundle_payload(
        product_kind="coadd",
        coadd_factor=3,
        final_dtype=np.dtype(np.uint32),
    )
    with pytest.raises(StampDeliveryBundleContractError, match="uint64"):
        StampDeliveryBundle.from_arrays(**invalid)

    payload = _bundle_payload(
        product_kind="coadd",
        coadd_factor=3,
        final_dtype=np.dtype(np.uint64),
    )
    payload["exposure_seconds"] = np.array([30.0, 30.0])
    payload["raw_frame_start_index"] = np.array([40, 43], dtype=np.int64)
    payload["raw_frame_stop_index_exclusive"] = np.array([43, 46], dtype=np.int64)
    payload["fullwell_count"][0, 1, 1] = 3  # type: ignore[index]
    bundle = StampDeliveryBundle.from_arrays(**payload)
    path = tmp_path / "target-42-coadd.h5"
    write_stamp_delivery_bundle(path, bundle)

    restored = read_stamp_delivery_bundle(path)
    assert restored.final_dn.dtype == np.uint64
    assert restored.coadd_factor == 3
    assert restored.fullwell_count[0, 1, 1] == 3

    payload["cosmic_count"][0, 0, 0] = 4  # type: ignore[index]
    with pytest.raises(StampDeliveryBundleContractError, match="coadd_factor"):
        StampDeliveryBundle.from_arrays(**payload)


def test_reader_rejects_an_incomplete_or_tampered_bundle(tmp_path) -> None:
    from et_mainsim.stamp_delivery import (
        STAMP_DELIVERY_SCHEMA_ID,
        STAMP_DELIVERY_SCHEMA_VERSION,
        StampDeliveryBundle,
        StampDeliveryBundleContractError,
        read_stamp_delivery_bundle,
        write_stamp_delivery_bundle,
    )

    incomplete = tmp_path / "incomplete.h5"
    with h5py.File(incomplete, "w") as handle:
        handle.attrs["schema_id"] = STAMP_DELIVERY_SCHEMA_ID
        handle.attrs["schema_version"] = STAMP_DELIVERY_SCHEMA_VERSION
        handle.attrs["complete"] = False
    with pytest.raises(StampDeliveryBundleContractError, match="not complete"):
        read_stamp_delivery_bundle(incomplete)

    path = tmp_path / "good.h5"
    write_stamp_delivery_bundle(path, StampDeliveryBundle.from_arrays(**_bundle_payload()))
    with h5py.File(path, "r+") as handle:
        handle["provenance_json"][()] = json.dumps(["not", "a", "mapping"])
    with pytest.raises(StampDeliveryBundleContractError, match="provenance"):
        read_stamp_delivery_bundle(path)


def test_reader_rejects_tampered_captured_flux_qa_semantics(tmp_path) -> None:
    from et_mainsim.stamp_delivery import (
        StampDeliveryBundle,
        StampDeliveryBundleContractError,
        read_stamp_delivery_bundle,
        write_stamp_delivery_bundle,
    )

    path = tmp_path / "capture-semantics.h5"
    write_stamp_delivery_bundle(
        path,
        StampDeliveryBundle.from_arrays(**_bundle_payload()),
    )
    with h5py.File(path, "r+") as handle:
        handle.attrs["captured_flux_qa_definition"] = "crop_renormalized"

    with pytest.raises(
        StampDeliveryBundleContractError,
        match="captured_flux_qa_definition",
    ):
        read_stamp_delivery_bundle(path)


def test_raw_products_reject_non_single_raw_frame_intervals() -> None:
    from et_mainsim.stamp_delivery import (
        StampDeliveryBundle,
        StampDeliveryBundleContractError,
    )

    payload = _bundle_payload()
    payload["raw_frame_stop_index_exclusive"] = np.array([42, 43], dtype=np.int64)
    with pytest.raises(StampDeliveryBundleContractError, match="exactly one raw frame"):
        StampDeliveryBundle.from_arrays(**payload)


def test_contract_rejects_a_background_realization_as_a_reduction_input() -> None:
    from et_mainsim.stamp_delivery import (
        StampDeliveryBundle,
        StampDeliveryBundleContractError,
    )

    payload = _bundle_payload()
    payload["provenance"] = {
        "observation_product": "final_dn",
        "background_realization_used": True,
    }
    with pytest.raises(StampDeliveryBundleContractError, match="must be false"):
        StampDeliveryBundle.from_arrays(**payload)


def test_streaming_appender_keeps_partial_invisible_until_complete(tmp_path) -> None:
    from et_mainsim.stamp_delivery import (
        StampDeliveryBundle,
        StampDeliveryBundleAppender,
        read_stamp_delivery_bundle,
    )

    payload = _bundle_payload()
    frame_fields = (
        "final_dn",
        "background_expectation_e",
        "captured_flux_fraction",
        "captured_flux_denominator_e",
        "captured_flux_qa_pass",
        "bias_level_sum_dn",
        "column_noise_sum_dn_by_x",
        "valid_mask",
        "fullwell_count",
        "adc_low_count",
        "adc_high_count",
        "cosmic_count",
        "time_start_seconds",
        "exposure_seconds",
        "raw_frame_start_index",
        "raw_frame_stop_index_exclusive",
    )

    def one_frame(frame_index: int) -> StampDeliveryBundle:
        batch = dict(payload)
        for name in frame_fields:
            batch[name] = np.asarray(batch[name])[frame_index : frame_index + 1]
        return StampDeliveryBundle.from_arrays(**batch)

    path = tmp_path / "streamed-raw.h5"
    appender = StampDeliveryBundleAppender(
        path,
        product_kind="raw",
        coadd_factor=1,
        stamp_shape=(3, 4),
        gain_e_per_dn=2.0,
        manifest=payload["manifest"],
        provenance=payload["provenance"],
    )
    appender.append(one_frame(0))
    assert not path.exists()
    assert len(list(tmp_path.glob("*.partial"))) == 1
    appender.append(one_frame(1))

    report = appender.complete()
    assert report.complete is True
    assert path.is_file()
    assert not list(tmp_path.glob("*.partial"))
    assert read_stamp_delivery_bundle(path).shape == (2, 3, 4)
    with pytest.raises(RuntimeError, match="already completed"):
        appender.append(one_frame(0))


def test_streaming_appender_rejects_a_gap_between_raw_batches(tmp_path) -> None:
    """A streamed bundle may not silently skip raw-frame coverage at a batch edge."""

    from et_mainsim.stamp_delivery import (
        StampDeliveryBundle,
        StampDeliveryBundleAppender,
        StampDeliveryBundleContractError,
    )

    payload = _bundle_payload()
    frame_fields = (
        "final_dn",
        "background_expectation_e",
        "captured_flux_fraction",
        "captured_flux_denominator_e",
        "captured_flux_qa_pass",
        "bias_level_sum_dn",
        "column_noise_sum_dn_by_x",
        "valid_mask",
        "fullwell_count",
        "adc_low_count",
        "adc_high_count",
        "cosmic_count",
        "time_start_seconds",
        "exposure_seconds",
        "raw_frame_start_index",
        "raw_frame_stop_index_exclusive",
    )

    first = dict(payload)
    second = dict(payload)
    for name in frame_fields:
        first[name] = np.asarray(first[name])[:1]
        second[name] = np.asarray(second[name])[1:2]
    # The first bundle ends at raw index 41.  A raw index 42 start leaves one
    # frame uncovered and must be rejected at the appender boundary.
    second["raw_frame_start_index"] = np.array([42], dtype=np.int64)
    second["raw_frame_stop_index_exclusive"] = np.array([43], dtype=np.int64)
    second["time_start_seconds"] = np.array([20.0])

    path = tmp_path / "gap-raw.h5"
    with StampDeliveryBundleAppender(
        path,
        product_kind="raw",
        coadd_factor=1,
        stamp_shape=(3, 4),
        gain_e_per_dn=2.0,
        manifest=payload["manifest"],
        provenance=payload["provenance"],
    ) as appender:
        appender.append(StampDeliveryBundle.from_arrays(**first))
        with pytest.raises(
            StampDeliveryBundleContractError,
            match="must equal previous batch stop",
        ):
            appender.append(StampDeliveryBundle.from_arrays(**second))

    assert not path.exists()
    assert not list(tmp_path.glob("*.partial"))


def test_streaming_appender_aborts_a_partial_product_on_context_exit(tmp_path) -> None:
    from et_mainsim.stamp_delivery import (
        StampDeliveryBundle,
        StampDeliveryBundleAppender,
    )

    payload = _bundle_payload()
    batch = dict(payload)
    for name in (
        "final_dn",
        "background_expectation_e",
        "captured_flux_fraction",
        "captured_flux_denominator_e",
        "captured_flux_qa_pass",
        "bias_level_sum_dn",
        "column_noise_sum_dn_by_x",
        "valid_mask",
        "fullwell_count",
        "adc_low_count",
        "adc_high_count",
        "cosmic_count",
        "time_start_seconds",
        "exposure_seconds",
        "raw_frame_start_index",
        "raw_frame_stop_index_exclusive",
    ):
        batch[name] = np.asarray(batch[name])[:1]
    path = tmp_path / "aborted-raw.h5"
    with StampDeliveryBundleAppender(
        path,
        product_kind="raw",
        coadd_factor=1,
        stamp_shape=(3, 4),
        gain_e_per_dn=2.0,
        manifest=payload["manifest"],
        provenance=payload["provenance"],
    ) as appender:
        appender.append(StampDeliveryBundle.from_arrays(**batch))

    assert not path.exists()
    assert not list(tmp_path.glob("*.partial"))
