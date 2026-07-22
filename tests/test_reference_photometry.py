from __future__ import annotations

import h5py
import numpy as np
import pytest


def _input_arrays(*, n_frames: int = 2, shape: tuple[int, int] = (15, 15)):
    """Build a small, internally consistent delivery-bundle fixture."""

    ny, nx = shape
    gain_e_per_dn = 2.0
    desired_e = np.full((n_frames, ny, nx), 5.0)
    background_expectation_e = np.full((n_frames, ny, nx), 2.0)
    bias_level_sum_dn = np.arange(n_frames, dtype=float) + 10.0
    column_noise_sum_dn_by_x = np.broadcast_to(
        np.arange(nx, dtype=float)[None, :] / 10.0,
        (n_frames, nx),
    ).copy()
    final_dn = (
        (desired_e + background_expectation_e) / gain_e_per_dn
        + bias_level_sum_dn[:, None, None]
        + column_noise_sum_dn_by_x[:, None, :]
    )
    return {
        "final_dn": final_dn,
        "background_expectation_e": background_expectation_e,
        "bias_level_sum_dn": bias_level_sum_dn,
        "column_noise_sum_dn_by_x": column_noise_sum_dn_by_x,
        "valid_mask": np.ones((n_frames, ny, nx), dtype=bool),
        "saturated_mask": np.zeros((n_frames, ny, nx), dtype=bool),
        "cosmic_mask": np.zeros((n_frames, ny, nx), dtype=bool),
        "time_index": np.arange(n_frames, dtype=np.int64),
        "gain_e_per_dn": gain_e_per_dn,
    }


def _write_formal_raw_bundle(tmp_path, name: str, *, n_frames: int, start: int):
    """Persist a compact valid raw formal bundle for corruption tests."""

    from et_mainsim.stamp_delivery import (
        StampDeliveryBundle,
        write_stamp_delivery_bundle,
    )

    ny, nx = 15, 15
    raw = np.arange(start, start + n_frames, dtype=np.int64)
    bundle = StampDeliveryBundle.from_arrays(
        product_kind="raw",
        coadd_factor=1,
        final_dn=np.full((n_frames, ny, nx), 16, dtype=np.uint16),
        background_expectation_e=np.full((n_frames, ny, nx), 2.0),
        bias_level_sum_dn=np.full(n_frames, 10.0),
        column_noise_sum_dn_by_x=np.zeros((n_frames, nx)),
        valid_mask=np.ones((n_frames, ny, nx), dtype=bool),
        fullwell_count=np.zeros((n_frames, ny, nx), dtype=np.uint16),
        adc_low_count=np.zeros((n_frames, ny, nx), dtype=np.uint16),
        adc_high_count=np.zeros((n_frames, ny, nx), dtype=np.uint16),
        cosmic_count=np.zeros((n_frames, ny, nx), dtype=np.uint16),
        time_start_seconds=raw.astype(float) * 10.0,
        exposure_seconds=np.full(n_frames, 10.0),
        raw_frame_start_index=raw,
        raw_frame_stop_index_exclusive=raw + 1,
        gain_e_per_dn=np.asarray(2.0),
        manifest={"target_source_id_int64": 42},
        provenance={
            "observation_product": "final_dn",
            "background_realization_used": False,
        },
    )
    path = tmp_path / name
    write_stamp_delivery_bundle(path, bundle)
    return path


def test_reduce_reference_photometry_derives_electrons_from_final_dn_only() -> None:
    from et_mainsim.reference_photometry import (
        ReferencePhotometryInput,
        reduce_reference_photometry_v1,
    )

    payload = _input_arrays()
    result = reduce_reference_photometry_v1(
        ReferencePhotometryInput.from_arrays(
            **payload,
            time_index_unit="frame_index",
            raw_frame_seconds=60.0,
        )
    )

    assert result.aperture_shape == (13, 13)
    assert result.aperture_pixel_count == 169
    np.testing.assert_allclose(result.time_seconds, [0.0, 60.0])
    np.testing.assert_allclose(result.flux_e, [845.0, 845.0])
    assert result.aperture_valid.tolist() == [True, True]
    assert result.product_semantics["observation_product"] == "final_dn"
    assert result.product_semantics["calibrated_electron_product"] == "derived"
    assert result.product_semantics["background_realization_used"] is False


@pytest.mark.parametrize("mask_name", ["valid_mask", "saturated_mask", "cosmic_mask"])
def test_reduce_reference_photometry_invalidates_a_fixed_aperture_on_any_mask(
    mask_name: str,
) -> None:
    from et_mainsim.reference_photometry import (
        ReferencePhotometryInput,
        reduce_reference_photometry_v1,
    )

    payload = _input_arrays()
    center = (payload["final_dn"].shape[1] // 2, payload["final_dn"].shape[2] // 2)
    if mask_name == "valid_mask":
        payload[mask_name][0, center[0], center[1]] = False
    else:
        payload[mask_name][0, center[0], center[1]] = True

    result = reduce_reference_photometry_v1(
        ReferencePhotometryInput.from_arrays(
            **payload,
            time_index_unit="frame_index",
            raw_frame_seconds=60.0,
        )
    )

    assert result.aperture_valid.tolist() == [False, True]
    assert np.isnan(result.flux_e[0])
    assert result.flux_e[1] == pytest.approx(845.0)


@pytest.mark.parametrize("mask_name", ["valid_mask", "saturated_mask", "cosmic_mask"])
def test_reference_photometry_rejects_nonbinary_input_masks(mask_name: str) -> None:
    """Masks are binary wire values, not arbitrary truthy integer arrays."""

    from et_mainsim.reference_photometry import (
        ReferencePhotometryContractError,
        ReferencePhotometryInput,
    )

    payload = _input_arrays()
    payload[mask_name] = np.asarray(payload[mask_name], dtype=np.uint8)
    payload[mask_name][0, 0, 0] = 2

    with pytest.raises(ReferencePhotometryContractError, match="exactly 0 or 1"):
        ReferencePhotometryInput.from_arrays(
            **payload,
            time_index_unit="frame_index",
            raw_frame_seconds=60.0,
        )


def test_load_reference_photometry_input_reads_composite_hdf5_bundle(tmp_path) -> None:
    from et_mainsim.reference_photometry import (
        load_reference_photometry_input,
        reduce_reference_photometry_bundle_v1,
    )

    payload = _input_arrays(n_frames=3)
    bundle_path = tmp_path / "delivery_bundle.h5"
    with h5py.File(bundle_path, "w") as handle:
        for name, value in payload.items():
            if name != "gain_e_per_dn":
                handle.create_dataset(name, data=value)
        handle.attrs["gain_e_per_dn"] = payload["gain_e_per_dn"]
        handle.attrs["time_index_unit"] = "frame_index"
        handle.attrs["raw_frame_seconds"] = 10.0

    loaded = load_reference_photometry_input(bundle_path)

    np.testing.assert_allclose(loaded.final_dn, payload["final_dn"])
    np.testing.assert_allclose(loaded.time_seconds, [0.0, 10.0, 20.0])
    assert loaded.gain_e_per_dn == pytest.approx(2.0)

    reduced = reduce_reference_photometry_bundle_v1(bundle_path)
    np.testing.assert_allclose(reduced.flux_e, [845.0, 845.0, 845.0])


def test_reduce_stamp_delivery_bundle_reads_the_formal_delivery_schema(tmp_path) -> None:
    from et_mainsim.reference_photometry import reduce_stamp_delivery_bundle_v1
    from et_mainsim.stamp_delivery import (
        StampDeliveryBundle,
        write_stamp_delivery_bundle,
    )

    n_frames, ny, nx = 3, 15, 15
    bundle = StampDeliveryBundle.from_arrays(
        product_kind="raw",
        coadd_factor=1,
        final_dn=np.full((n_frames, ny, nx), 16, dtype=np.uint16),
        background_expectation_e=np.full((n_frames, ny, nx), 2.0),
        bias_level_sum_dn=np.full(n_frames, 10.0),
        column_noise_sum_dn_by_x=np.zeros((n_frames, nx)),
        valid_mask=np.ones((n_frames, ny, nx), dtype=bool),
        fullwell_count=np.zeros((n_frames, ny, nx), dtype=np.uint16),
        adc_low_count=np.zeros((n_frames, ny, nx), dtype=np.uint16),
        adc_high_count=np.zeros((n_frames, ny, nx), dtype=np.uint16),
        cosmic_count=np.zeros((n_frames, ny, nx), dtype=np.uint16),
        time_start_seconds=np.array([0.0, 10.0, 20.0]),
        exposure_seconds=np.full(n_frames, 10.0),
        raw_frame_start_index=np.array([0, 1, 2], dtype=np.int64),
        raw_frame_stop_index_exclusive=np.array([1, 2, 3], dtype=np.int64),
        gain_e_per_dn=np.asarray(2.0),
        manifest={"case": "formal-test"},
        provenance={
            "observation_product": "final_dn",
            "background_realization_used": False,
        },
    )
    path = tmp_path / "formal_delivery.h5"
    write_stamp_delivery_bundle(path, bundle)

    result = reduce_stamp_delivery_bundle_v1(path)

    np.testing.assert_allclose(result.time_seconds, [0.0, 10.0, 20.0])
    np.testing.assert_allclose(result.flux_e, np.full(n_frames, 1690.0))
    assert result.product_semantics["observation_product"] == "final_dn"


def test_reduce_stamp_delivery_series_streams_contiguous_shards(tmp_path) -> None:
    from et_mainsim.reference_photometry import reduce_stamp_delivery_series_v1
    from et_mainsim.stamp_delivery import (
        StampDeliveryBundle,
        write_stamp_delivery_bundle,
    )

    def write_shard(*, name: str, start: int, values: tuple[int, int]):
        n_frames, ny, nx = 2, 15, 15
        raw = np.arange(start, start + n_frames, dtype=np.int64)
        bundle = StampDeliveryBundle.from_arrays(
            product_kind="raw",
            coadd_factor=1,
            final_dn=np.stack(
                [np.full((ny, nx), value, dtype=np.uint16) for value in values]
            ),
            background_expectation_e=np.full((n_frames, ny, nx), 2.0),
            bias_level_sum_dn=np.full(n_frames, 10.0),
            column_noise_sum_dn_by_x=np.zeros((n_frames, nx)),
            valid_mask=np.ones((n_frames, ny, nx), dtype=bool),
            fullwell_count=np.zeros((n_frames, ny, nx), dtype=np.uint16),
            adc_low_count=np.zeros((n_frames, ny, nx), dtype=np.uint16),
            adc_high_count=np.zeros((n_frames, ny, nx), dtype=np.uint16),
            cosmic_count=np.zeros((n_frames, ny, nx), dtype=np.uint16),
            time_start_seconds=raw.astype(float) * 10.0,
            exposure_seconds=np.full(n_frames, 10.0),
            raw_frame_start_index=raw,
            raw_frame_stop_index_exclusive=raw + 1,
            gain_e_per_dn=np.asarray(2.0),
            manifest={
                "schema_id": "test",
                "scene_policy": "independent_target",
                "target_source_id": "42",
                "target_source_id_int64": 42,
                "stamp_shape": [ny, nx],
                "time_shard": {"raw_frame_interval": {"start_index": start}},
                "caller_manifest": {"case": "injected", "input": "test"},
            },
            provenance={
                "observation_product": "final_dn",
                "background_realization_used": False,
                "caller_provenance": {"code": "test"},
            },
        )
        path = tmp_path / name
        write_stamp_delivery_bundle(path, bundle)
        return path

    first = write_shard(name="first.h5", start=0, values=(16, 17))
    second = write_shard(name="second.h5", start=2, values=(18, 19))

    result = reduce_stamp_delivery_series_v1(
        (second, first),
        cdpp_windows_minutes=(30,),
        batch_frames=1,
    )

    np.testing.assert_allclose(result.time_seconds, [0.0, 10.0, 20.0, 30.0])
    np.testing.assert_allclose(result.flux_e, [1690.0, 2028.0, 2366.0, 2704.0])
    assert result.aperture_valid.tolist() == [True, True, True, True]
    assert result.product_semantics["input_mode"] == "streamed_formal_delivery_shards"


def test_reduce_stamp_delivery_series_rejects_a_gap_between_shards(tmp_path) -> None:
    from et_mainsim.reference_photometry import (
        ReferencePhotometryContractError,
        reduce_stamp_delivery_series_v1,
    )
    from et_mainsim.stamp_delivery import (
        StampDeliveryBundle,
        write_stamp_delivery_bundle,
    )

    def write_one(name: str, frame: int):
        shape = (1, 15, 15)
        bundle = StampDeliveryBundle.from_arrays(
            product_kind="raw",
            coadd_factor=1,
            final_dn=np.full(shape, 16, dtype=np.uint16),
            background_expectation_e=np.full(shape, 2.0),
            bias_level_sum_dn=np.array([10.0]),
            column_noise_sum_dn_by_x=np.zeros((1, 15)),
            valid_mask=np.ones(shape, dtype=bool),
            fullwell_count=np.zeros(shape, dtype=np.uint16),
            adc_low_count=np.zeros(shape, dtype=np.uint16),
            adc_high_count=np.zeros(shape, dtype=np.uint16),
            cosmic_count=np.zeros(shape, dtype=np.uint16),
            time_start_seconds=np.array([frame * 10.0]),
            exposure_seconds=np.array([10.0]),
            raw_frame_start_index=np.array([frame], dtype=np.int64),
            raw_frame_stop_index_exclusive=np.array([frame + 1], dtype=np.int64),
            gain_e_per_dn=np.asarray(2.0),
            manifest={"target_source_id_int64": 42},
            provenance={
                "observation_product": "final_dn",
                "background_realization_used": False,
            },
        )
        path = tmp_path / name
        write_stamp_delivery_bundle(path, bundle)
        return path

    first = write_one("first.h5", 0)
    gap = write_one("gap.h5", 2)

    with pytest.raises(ReferencePhotometryContractError, match="globally continuous"):
        reduce_stamp_delivery_series_v1((first, gap), batch_frames=1)


def test_reduce_stamp_delivery_series_rejects_an_intra_shard_time_gap(tmp_path) -> None:
    """Contiguous raw indices cannot conceal a physical-time gap in one shard."""

    from et_mainsim.reference_photometry import (
        ReferencePhotometryContractError,
        reduce_stamp_delivery_series_v1,
    )

    n_frames = 3
    path = _write_formal_raw_bundle(
        tmp_path,
        "intra_shard_gap.h5",
        n_frames=n_frames,
        start=0,
    )
    with h5py.File(path, "r+") as handle:
        handle["time_start_seconds"][...] = np.array([0.0, 20.0, 30.0])

    with pytest.raises(ReferencePhotometryContractError, match="invalid frame intervals"):
        # One batch ensures the reader cannot rely only on its batch-edge check.
        reduce_stamp_delivery_series_v1((path,), batch_frames=n_frames)


@pytest.mark.parametrize("mask_name", ["valid_mask", "saturated_mask", "cosmic_mask"])
def test_reduce_stamp_delivery_series_rejects_nonbinary_formal_masks(
    tmp_path,
    mask_name: str,
) -> None:
    """The streaming path must not bool-coerce corrupted formal mask values."""

    from et_mainsim.reference_photometry import (
        ReferencePhotometryContractError,
        reduce_stamp_delivery_series_v1,
    )

    ny, nx = 15, 15
    path = _write_formal_raw_bundle(
        tmp_path,
        f"nonbinary_{mask_name}.h5",
        n_frames=2,
        start=0,
    )
    with h5py.File(path, "r+") as handle:
        handle[mask_name][0, ny // 2, nx // 2] = 2

    with pytest.raises(ReferencePhotometryContractError, match="exactly 0 or 1"):
        reduce_stamp_delivery_series_v1((path,), batch_frames=1)


def test_reduce_stamp_delivery_series_streams_per_frame_gain_maps(tmp_path) -> None:
    """A valid formal 3-D gain plane is cropped per frame, not rejected or reused."""

    from et_mainsim.reference_photometry import reduce_stamp_delivery_series_v1
    from et_mainsim.stamp_delivery import (
        StampDeliveryBundle,
        write_stamp_delivery_bundle,
    )

    ny, nx = 15, 15

    def write_shard(name: str, *, start: int, gain_values: tuple[int, int]):
        n_frames = len(gain_values)
        gains = np.stack(
            [np.full((ny, nx), value, dtype=float) for value in gain_values]
        )
        # `(final_dn - bias) * gain - background` is exactly 10 e/pixel.
        final = np.stack(
            [
                np.full((ny, nx), 10 + 12 // gain, dtype=np.uint16)
                for gain in gain_values
            ]
        )
        raw = np.arange(start, start + n_frames, dtype=np.int64)
        bundle = StampDeliveryBundle.from_arrays(
            product_kind="raw",
            coadd_factor=1,
            final_dn=final,
            background_expectation_e=np.full((n_frames, ny, nx), 2.0),
            bias_level_sum_dn=np.full(n_frames, 10.0),
            column_noise_sum_dn_by_x=np.zeros((n_frames, nx)),
            valid_mask=np.ones((n_frames, ny, nx), dtype=bool),
            fullwell_count=np.zeros((n_frames, ny, nx), dtype=np.uint16),
            adc_low_count=np.zeros((n_frames, ny, nx), dtype=np.uint16),
            adc_high_count=np.zeros((n_frames, ny, nx), dtype=np.uint16),
            cosmic_count=np.zeros((n_frames, ny, nx), dtype=np.uint16),
            time_start_seconds=raw.astype(float) * 10.0,
            exposure_seconds=np.full(n_frames, 10.0),
            raw_frame_start_index=raw,
            raw_frame_stop_index_exclusive=raw + 1,
            gain_e_per_dn=gains,
            manifest={"target_source_id_int64": 42},
            provenance={
                "observation_product": "final_dn",
                "background_realization_used": False,
            },
        )
        path = tmp_path / name
        write_stamp_delivery_bundle(path, bundle)
        return path

    first = write_shard("gain_first.h5", start=0, gain_values=(1, 2))
    second = write_shard("gain_second.h5", start=2, gain_values=(3, 4))

    result = reduce_stamp_delivery_series_v1(
        (second, first),
        cdpp_windows_minutes=(30,),
        batch_frames=1,
    )

    np.testing.assert_allclose(result.time_seconds, [0.0, 10.0, 20.0, 30.0])
    np.testing.assert_allclose(result.flux_e, np.full(4, 1690.0))


def test_injected_model_residual_cdpp_removes_the_known_variable_source_curve() -> None:
    from et_mainsim.reference_photometry import (
        ReferencePhotometryResult,
        compute_injected_model_residual_v1,
    )

    n_frames = 4 * 30 * 60 // 10
    time = np.arange(n_frames, dtype=float) * 10.0
    factor = 1.0 + 0.05 * np.sin(np.arange(n_frames, dtype=float) / 91.0)
    reference = ReferencePhotometryResult(
        time_seconds=time,
        flux_e=1250.0 * factor,
        aperture_valid=np.ones(n_frames, dtype=bool),
        aperture_usable_pixel_count=np.full(n_frames, 169, dtype=np.int64),
        aperture_mask=np.ones((13, 13), dtype=bool),
        aperture_shape=(13, 13),
        aperture_pixel_count=169,
        exposure_seconds=np.full(n_frames, 10.0),
        cdpp_by_window_minutes={},
        product_semantics={"observation_product": "final_dn"},
        raw_frame_start_index=np.arange(n_frames, dtype=np.int64),
        raw_frame_stop_index_exclusive=np.arange(1, n_frames + 1, dtype=np.int64),
    )

    residual = compute_injected_model_residual_v1(
        reference,
        raw_frame_factors=factor,
        raw_exposure_seconds=10.0,
        windows_minutes=(30,),
        minimum_complete_bins=2,
    )

    assert residual.fit_scale_e_per_raw_factor == pytest.approx(1250.0)
    assert residual.fit_intercept_e == pytest.approx(0.0)
    np.testing.assert_allclose(residual.residual_ppm, 0.0, atol=1e-7)
    assert residual.cdpp_by_window_minutes[30].complete_bin_count == 4
    assert residual.cdpp_by_window_minutes[30].cdpp_ppm == pytest.approx(0.0, abs=1e-6)


def test_cadence_aware_cdpp_uses_complete_time_windows_without_legacy_bin_lcs() -> None:
    from et_mainsim.reference_photometry import compute_cadence_aware_cdpp

    cadence_seconds = 60.0
    # Four complete 390-minute windows are enough to exercise every standard
    # window (30m, 90m, 390m) at the same physical cadence.
    time_seconds = np.arange(4 * 390, dtype=float) * cadence_seconds
    flux_e = np.full(time_seconds.size, 100.0)
    metrics = compute_cadence_aware_cdpp(
        time_seconds=time_seconds,
        flux_e=flux_e,
        aperture_valid=np.ones(time_seconds.size, dtype=bool),
        exposure_seconds=cadence_seconds,
    )

    for minutes in (30, 90, 390):
        metric = metrics[minutes]
        assert metric.window_minutes == minutes
        assert metric.complete_bin_count == 4 * 390 // minutes
        assert metric.cdpp_ppm == pytest.approx(0.0)


def test_cadence_aware_cdpp_rejects_an_incomplete_time_window() -> None:
    from et_mainsim.reference_photometry import compute_cadence_aware_cdpp

    cadence_seconds = 60.0
    time_seconds = np.arange(4 * 30, dtype=float) * cadence_seconds
    aperture_valid = np.ones(time_seconds.size, dtype=bool)
    aperture_valid[5] = False
    metrics = compute_cadence_aware_cdpp(
        time_seconds=time_seconds,
        flux_e=np.full(time_seconds.size, 100.0),
        aperture_valid=aperture_valid,
        exposure_seconds=cadence_seconds,
        windows_minutes=(30,),
    )

    metric = metrics[30]
    assert metric.complete_bin_count == 3
    assert metric.rejected_bin_count == 1
    assert metric.cdpp_ppm == pytest.approx(0.0)


def test_cadence_aware_cdpp_uses_the_legacy_mad_normalization_after_binning() -> None:
    from et_mainsim.reference_photometry import compute_cadence_aware_cdpp

    cadence_seconds = 60.0
    # Four 30-minute bins with electron-count levels 3000, 3030, 2970, 3000.
    # The legacy estimator is 1.4826 * mean(abs(x - mean(x))) / mean(x) ppm.
    time_seconds = np.arange(4 * 30, dtype=float) * cadence_seconds
    flux_e = np.repeat([100.0, 101.0, 99.0, 100.0], 30)
    metric = compute_cadence_aware_cdpp(
        time_seconds=time_seconds,
        flux_e=flux_e,
        aperture_valid=np.ones(time_seconds.size, dtype=bool),
        exposure_seconds=cadence_seconds,
        windows_minutes=(30,),
    )[30]

    assert metric.complete_bin_count == 4
    assert metric.cdpp_ppm == pytest.approx(7_413.0)
