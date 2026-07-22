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
