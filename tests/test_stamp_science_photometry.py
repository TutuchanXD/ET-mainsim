from __future__ import annotations

import numpy as np
import pytest


def _delivery_from_calibrated_bgsub(
    calibrated_bgsub_e: np.ndarray,
    *,
    background_expectation_e: float = 10.0,
    bias_dn: float = 100.0,
    column_dn: float = 3.0,
    gain_e_per_dn: float = 2.0,
    valid_mask: np.ndarray | None = None,
    saturated_mask: np.ndarray | None = None,
    cosmic_mask: np.ndarray | None = None,
):
    from et_mainsim.reference_photometry import ReferencePhotometryInput

    calibrated = np.asarray(calibrated_bgsub_e, dtype=np.float64)
    n_frames, ny, nx = calibrated.shape
    background = np.full(calibrated.shape, background_expectation_e)
    final_dn = (
        (calibrated + background) / gain_e_per_dn
        + bias_dn
        + column_dn
    )
    shape = calibrated.shape
    return ReferencePhotometryInput.from_arrays(
        final_dn=final_dn,
        background_expectation_e=background,
        bias_level_sum_dn=np.full(n_frames, bias_dn),
        column_noise_sum_dn_by_x=np.full((n_frames, nx), column_dn),
        valid_mask=(
            np.ones(shape, dtype=bool)
            if valid_mask is None
            else valid_mask
        ),
        saturated_mask=(
            np.zeros(shape, dtype=bool)
            if saturated_mask is None
            else saturated_mask
        ),
        cosmic_mask=(
            np.zeros(shape, dtype=bool)
            if cosmic_mask is None
            else cosmic_mask
        ),
        time_index=np.arange(n_frames, dtype=float) * 10.0,
        gain_e_per_dn=gain_e_per_dn,
        time_index_unit="seconds",
        raw_frame_seconds=10.0,
        exposure_seconds=np.full(n_frames, 10.0),
    )


def test_optimal_aperture_reuses_legacy_snr_kernel_on_rectangular_maps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import photsim7.aperture as legacy_aperture

    calls: list[tuple[np.ndarray, np.ndarray, bool]] = []
    selected = np.zeros((3, 5), dtype=bool)
    selected[1, 2] = True
    selected[1, 3] = True

    def fake_legacy_kernel(signal, noise, plot=False):
        calls.append(
            (
                signal.detach().cpu().numpy(),
                noise.detach().cpu().numpy(),
                plot,
            )
        )
        import torch

        return torch.as_tensor(selected, device=signal.device), 12.5

    monkeypatch.setattr(
        legacy_aperture,
        "maximize_cumulative_snr",
        fake_legacy_kernel,
    )

    from et_mainsim.stamp_science_photometry import (
        build_science_optimal_aperture_v1,
    )

    signal = np.zeros((3, 5), dtype=float)
    signal[1, 2] = 20.0
    signal[1, 3] = 5.0
    noise = np.full((3, 5), 2.0)
    definition = build_science_optimal_aperture_v1(
        signal_template_e=signal,
        noise_template_e=noise,
    )

    assert len(calls) == 1
    np.testing.assert_allclose(calls[0][0], signal)
    np.testing.assert_allclose(calls[0][1], noise)
    assert calls[0][2] is False
    np.testing.assert_array_equal(definition.aperture_mask, selected)
    assert definition.maximum_cumulative_snr == pytest.approx(12.5)
    assert definition.algorithm == (
        "photsim7.aperture.maximize_cumulative_snr"
    )
    assert definition.signal_template_shape == (3, 5)


def test_reference_fixed13_aperture_freezes_floor_centered_reference_mask() -> None:
    from et_mainsim.stamp_science_photometry import (
        build_reference_fixed13_aperture_v1,
    )

    definition = build_reference_fixed13_aperture_v1((100, 300))

    expected = np.zeros((100, 300), dtype=bool)
    expected[44:57, 144:157] = True
    np.testing.assert_array_equal(definition.aperture_mask, expected)
    assert np.count_nonzero(definition.aperture_mask) == 13 * 13
    assert np.isnan(definition.maximum_cumulative_snr)
    assert definition.algorithm == (
        "et_mainsim.reference_fixed13_aperture_v1"
    )
    assert definition.signal_template_shape == (100, 300)
    assert definition.target_peak_yx == (50, 150)
    assert definition.metadata == {
        "aperture_role": "reference_qa_not_science_optimal",
        "aperture_shape": [13, 13],
        "target_center_yx": [50, 150],
        "target_center_policy": "stamp_floor_center_yx",
        "maximum_cumulative_snr_applicable": False,
    }


def test_reference_fixed13_aperture_accepts_an_explicit_boundary_center() -> None:
    from et_mainsim.stamp_science_photometry import (
        build_reference_fixed13_aperture_v1,
    )

    definition = build_reference_fixed13_aperture_v1(
        (20, 30),
        target_center_yx=(6, 6),
    )

    assert np.all(definition.aperture_mask[:13, :13])
    assert np.count_nonzero(definition.aperture_mask) == 169
    assert definition.target_peak_yx == (6, 6)
    assert definition.metadata["target_center_policy"] == (
        "explicit_integer_target_center_yx"
    )


@pytest.mark.parametrize(
    ("stamp_shape", "target_center_yx", "message"),
    [
        ((12, 30), None, "13x13 aperture does not fit"),
        ((20, 30), (5, 6), "crosses the stamp boundary"),
        ((20, 30), (6, 24), "crosses the stamp boundary"),
        ((20, 30), (6.5, 6), "target_center_yx"),
    ],
)
def test_reference_fixed13_aperture_rejects_invalid_geometry(
    stamp_shape: tuple[int, int],
    target_center_yx: tuple[float, float] | None,
    message: str,
) -> None:
    from et_mainsim.stamp_science_photometry import (
        SciencePhotometryContractError,
        build_reference_fixed13_aperture_v1,
    )

    with pytest.raises(SciencePhotometryContractError, match=message):
        build_reference_fixed13_aperture_v1(
            stamp_shape,
            target_center_yx=target_center_yx,
        )


def test_flux_uncertainty_model_keeps_components_when_quality_is_invalid() -> None:
    from et_mainsim.stamp_science_photometry import (
        compute_science_flux_uncertainty_model_v1,
    )

    aperture = np.zeros((3, 5), dtype=bool)
    aperture[1, 2:4] = True
    background = np.empty((3, 3, 5), dtype=float)
    background[0] = 2.0
    background[1] = 3.0
    background[2] = 4.0
    result = compute_science_flux_uncertainty_model_v1(
        fitted_source_expectation_e=np.asarray([1_000.0, 2_000.0, 3_000.0]),
        background_expectation_e=background,
        aperture_mask=aperture,
        read_noise_e_per_raw_pixel=4.0,
        quantization_noise_e_per_raw_pixel=0.5,
        coadd_factor=np.asarray([1, 3, 6]),
        cadence_valid=np.asarray([True, False, True]),
    )

    np.testing.assert_allclose(
        result.source_variance_e2,
        [1_000.0, 2_000.0, 3_000.0],
    )
    np.testing.assert_allclose(result.background_variance_e2, [4.0, 6.0, 8.0])
    np.testing.assert_allclose(result.read_variance_e2, [32.0, 96.0, 192.0])
    np.testing.assert_allclose(
        result.quantization_variance_e2,
        [0.5, 1.5, 3.0],
    )
    np.testing.assert_allclose(
        result.uncertainty_e[[0, 2]],
        [np.sqrt(1_036.5), np.sqrt(3_203.0)],
    )
    assert np.isnan(result.uncertainty_e[1])
    np.testing.assert_array_equal(result.valid_mask, [True, False, True])
    np.testing.assert_array_equal(result.coadd_factor, [1, 3, 6])
    assert result.metadata["background_components"] == (
        "sky+scattered_light+dark_current"
    )
    assert result.metadata[
        "dark_current_counted_once_via_background_expectation"
    ] is True
    assert result.metadata["invalid_cadence_policy"] == (
        "uncertainty_nan_components_retained"
    )


def test_flux_uncertainty_model_scales_per_raw_read_terms_for_scalar_coadd() -> None:
    from et_mainsim.stamp_science_photometry import (
        compute_science_flux_uncertainty_model_v1,
    )

    aperture = np.ones((2, 2), dtype=bool)
    result = compute_science_flux_uncertainty_model_v1(
        fitted_source_expectation_e=np.asarray([100.0, 200.0]),
        background_expectation_e=np.full((2, 2, 2), 5.0),
        aperture_mask=aperture,
        read_noise_e_per_raw_pixel=2.0,
        quantization_noise_e_per_raw_pixel=1.0,
        coadd_factor=3,
    )

    # background_expectation_e already contains sky + scattered light + dark;
    # no additional dark-current variance term exists or is added.
    np.testing.assert_allclose(result.background_variance_e2, [20.0, 20.0])
    np.testing.assert_allclose(result.read_variance_e2, [48.0, 48.0])
    np.testing.assert_allclose(result.quantization_variance_e2, [12.0, 12.0])
    np.testing.assert_allclose(
        result.uncertainty_e,
        np.sqrt([180.0, 280.0]),
    )
    np.testing.assert_array_equal(result.coadd_factor, [3, 3])


def test_flux_uncertainty_streaming_aperture_sum_is_identical_to_cube() -> None:
    from et_mainsim.stamp_science_photometry import (
        compute_science_flux_uncertainty_model_v1,
    )

    aperture = np.zeros((3, 4), dtype=bool)
    aperture[1, 1:4] = True
    background = np.arange(24, dtype=float).reshape(2, 3, 4) + 1.0
    common = {
        "fitted_source_expectation_e": np.asarray([500.0, 800.0]),
        "aperture_mask": aperture,
        "read_noise_e_per_raw_pixel": 3.0,
        "quantization_noise_e_per_raw_pixel": 0.25,
        "coadd_factor": np.asarray([1, 6]),
        "cadence_valid": np.asarray([True, False]),
    }
    from_cube = compute_science_flux_uncertainty_model_v1(
        **common,
        background_expectation_e=background,
    )
    aperture_sum = np.sum(background[:, aperture], axis=1)
    from_stream = compute_science_flux_uncertainty_model_v1(
        **common,
        background_expectation_aperture_e=aperture_sum,
    )

    for field in (
        "uncertainty_e",
        "source_variance_e2",
        "background_variance_e2",
        "read_variance_e2",
        "quantization_variance_e2",
        "valid_mask",
        "coadd_factor",
    ):
        np.testing.assert_allclose(
            getattr(from_stream, field),
            getattr(from_cube, field),
            equal_nan=True,
        )
    assert from_cube.metadata["background_input_representation"] == (
        "cadence_stamp_cube"
    )
    assert from_stream.metadata["background_input_representation"] == (
        "cadence_aperture_sum_vector"
    )
    assert from_stream.metadata["background_product"] == (
        "background_expectation_aperture_e"
    )


@pytest.mark.parametrize("provided", ["both", "neither"])
def test_flux_uncertainty_background_representations_are_exactly_one_of(
    provided: str,
) -> None:
    from et_mainsim.stamp_science_photometry import (
        SciencePhotometryContractError,
        compute_science_flux_uncertainty_model_v1,
    )

    kwargs: dict[str, object] = {
        "fitted_source_expectation_e": np.asarray([10.0, 20.0]),
        "aperture_mask": np.ones((2, 2), dtype=bool),
        "read_noise_e_per_raw_pixel": 2.0,
        "quantization_noise_e_per_raw_pixel": 0.5,
        "coadd_factor": 1,
    }
    if provided == "both":
        kwargs["background_expectation_e"] = np.ones((2, 2, 2))
        kwargs["background_expectation_aperture_e"] = np.ones(2)

    with pytest.raises(
        SciencePhotometryContractError,
        match="exactly one.*background_expectation",
    ):
        compute_science_flux_uncertainty_model_v1(**kwargs)


def test_flux_uncertainty_rejects_invalid_streaming_background_vector() -> None:
    from et_mainsim.stamp_science_photometry import (
        SciencePhotometryContractError,
        compute_science_flux_uncertainty_model_v1,
    )

    with pytest.raises(
        SciencePhotometryContractError,
        match="background_expectation_aperture_e",
    ):
        compute_science_flux_uncertainty_model_v1(
            fitted_source_expectation_e=np.asarray([10.0, 20.0]),
            background_expectation_aperture_e=np.asarray([1.0, -2.0]),
            aperture_mask=np.ones((2, 2), dtype=bool),
            read_noise_e_per_raw_pixel=2.0,
            quantization_noise_e_per_raw_pixel=0.5,
            coadd_factor=1,
        )


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"coadd_factor": np.asarray([1, 0])}, "coadd_factor"),
        ({"cadence_valid": np.asarray([True])}, "cadence_valid"),
        (
            {"background_expectation_e": np.ones((2, 2, 3))},
            "background_expectation_e",
        ),
        (
            {"fitted_source_expectation_e": np.asarray([10.0, -1.0])},
            "fitted_source_expectation_e",
        ),
    ],
)
def test_flux_uncertainty_model_rejects_ambiguous_or_nonphysical_inputs(
    override: dict[str, object],
    message: str,
) -> None:
    from et_mainsim.stamp_science_photometry import (
        SciencePhotometryContractError,
        compute_science_flux_uncertainty_model_v1,
    )

    kwargs: dict[str, object] = {
        "fitted_source_expectation_e": np.asarray([10.0, 20.0]),
        "background_expectation_e": np.ones((2, 2, 2)),
        "aperture_mask": np.ones((2, 2), dtype=bool),
        "read_noise_e_per_raw_pixel": 2.0,
        "quantization_noise_e_per_raw_pixel": 0.5,
        "coadd_factor": 1,
    }
    kwargs.update(override)
    with pytest.raises(SciencePhotometryContractError, match=message):
        compute_science_flux_uncertainty_model_v1(**kwargs)


def test_optimal_aperture_rejects_a_legacy_mask_that_drops_the_target_peak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import photsim7.aperture as legacy_aperture

    def invalid_mask(signal, noise, plot=False):
        import torch

        mask = torch.zeros_like(signal, dtype=torch.bool)
        mask[0, 0] = True
        return mask, 1.0

    monkeypatch.setattr(
        legacy_aperture,
        "maximize_cumulative_snr",
        invalid_mask,
    )
    from et_mainsim.stamp_science_photometry import (
        SciencePhotometryContractError,
        build_science_optimal_aperture_v1,
    )

    signal = np.zeros((3, 5), dtype=float)
    signal[1, 2] = 10.0
    with pytest.raises(
        SciencePhotometryContractError,
        match="target-signal peak",
    ):
        build_science_optimal_aperture_v1(
            signal_template_e=signal,
            noise_template_e=np.ones_like(signal),
        )


def test_background_mask_excludes_aperture_dilation_edges_and_bad_pixels() -> None:
    from et_mainsim.stamp_science_photometry import (
        build_local_background_mask_v1,
    )

    aperture = np.zeros((7, 9), dtype=bool)
    aperture[3, 4] = True
    permanent_valid = np.ones_like(aperture)
    permanent_valid[1, 1] = False
    mask = build_local_background_mask_v1(
        aperture,
        exclusion_radius_pixels=1,
        border_pixels=1,
        permanent_valid_mask=permanent_valid,
    )

    assert not np.any(mask[2:5, 3:6])
    assert not np.any(mask[[0, -1], :])
    assert not np.any(mask[:, [0, -1]])
    assert not mask[1, 1]
    assert mask[1, 2]


def test_local_background_contract_is_explicit_frozen_and_versioned() -> None:
    from et_mainsim.stamp_science_photometry import (
        SciencePhotometryContractError,
        StampSciencePhotometryPolicy,
        reduce_science_photometry_v1,
    )

    policy = StampSciencePhotometryPolicy()
    assert policy.background_strategy == "delivered_expectation_only"
    assert policy.local_background_enabled is False
    assert policy.local_background_policy_version == 1
    assert policy.local_background_estimator == "per_frame_median"
    assert policy.local_background_sigma_clipping == "none"
    with pytest.raises(
        SciencePhotometryContractError,
        match="local_background_policy_version",
    ):
        StampSciencePhotometryPolicy(local_background_policy_version=2)
    with pytest.raises(
        SciencePhotometryContractError,
        match="local_background_estimator",
    ):
        StampSciencePhotometryPolicy(local_background_estimator="sigma_clipped_mean")
    with pytest.raises(
        SciencePhotometryContractError,
        match="local_background_sigma_clipping",
    ):
        StampSciencePhotometryPolicy(local_background_sigma_clipping="3_sigma")

    aperture = np.zeros((5, 7), dtype=bool)
    aperture[2, 3] = True
    background = np.zeros((5, 7), dtype=bool)
    background[1, 1:5] = True
    result = reduce_science_photometry_v1(
        _delivery_from_calibrated_bgsub(np.zeros((1, 5, 7), dtype=float)),
        aperture_mask=aperture,
        background_mask=background,
        minimum_background_pixels=4,
    )
    assert result.product_semantics["local_background_policy_version"] == 1
    assert result.product_semantics["local_background_estimator"] == (
        "per_frame_median"
    )
    assert result.product_semantics["local_background_sigma_clipping"] == "none"


def test_public_science_photometry_wire_schema_is_v2() -> None:
    """Expectation-background-only products must not reuse the old v1 ID."""

    from et_mainsim.stamp_science_photometry import (
        SCIENCE_PHOTOMETRY_SCHEMA_ID,
        SCIENCE_PHOTOMETRY_SCHEMA_VERSION,
    )

    assert SCIENCE_PHOTOMETRY_SCHEMA_ID == (
        "et_mainsim.stamp_science_photometry.v2"
    )
    assert SCIENCE_PHOTOMETRY_SCHEMA_VERSION == 2


def test_dual_background_photometry_uses_final_dn_and_records_quality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import photsim7.analysis as legacy_analysis

    monkeypatch.setattr(
        legacy_analysis,
        "torch_measure_centroids",
        lambda *_args, **_kwargs: pytest.fail(
            "formal science centroid must not call the unsafe whole-stamp legacy wrapper"
        ),
    )

    from et_mainsim.stamp_science_photometry import (
        ScienceQualityFlag,
        reduce_science_photometry_v1,
    )

    frames = np.zeros((3, 5, 7), dtype=float)
    aperture = np.zeros((5, 7), dtype=bool)
    aperture[2, 3:5] = True
    background = np.zeros((5, 7), dtype=bool)
    background[1, 1:5] = True

    # The expectation-subtracted target sum is 150 e in every frame.
    frames[:, 2, 3] = 100.0
    frames[:, 2, 4] = 50.0
    # Residual local backgrounds are 2, 4 and 6 e/pixel respectively.
    frames[0, 1, 1:5] = 2.0
    frames[1, 1, 1:5] = 4.0
    frames[2, 1, 1:5] = 6.0

    valid = np.ones(frames.shape, dtype=bool)
    saturated = np.zeros(frames.shape, dtype=bool)
    cosmic = np.zeros(frames.shape, dtype=bool)
    saturated[1, 2, 4] = True
    cosmic[2, 1, 1] = True
    delivery = _delivery_from_calibrated_bgsub(
        frames,
        valid_mask=valid,
        saturated_mask=saturated,
        cosmic_mask=cosmic,
    )

    result = reduce_science_photometry_v1(
        delivery,
        aperture_mask=aperture,
        background_mask=background,
        minimum_background_pixels=3,
    )

    np.testing.assert_allclose(
        result.flux_expectation_bgsub_e[[0, 2]],
        [150.0, 150.0],
    )
    assert np.isnan(result.flux_expectation_bgsub_e[1])
    np.testing.assert_allclose(
        result.local_background_e_per_pixel,
        [2.0, 4.0, 6.0],
    )
    np.testing.assert_allclose(
        result.flux_local_bgsub_e[[0, 2]],
        [146.0, 138.0],
    )
    assert np.isnan(result.flux_local_bgsub_e[1])
    np.testing.assert_array_equal(result.aperture_valid, [True, False, True])
    np.testing.assert_array_equal(result.saturated_pixel_count, [0, 1, 0])
    np.testing.assert_array_equal(result.cosmic_pixel_count, [0, 0, 0])
    np.testing.assert_array_equal(result.background_usable_pixel_count, [4, 4, 3])
    assert result.quality_bitmask[0] == int(ScienceQualityFlag.OK)
    assert result.quality_bitmask[1] & int(
        ScienceQualityFlag.APERTURE_SATURATED
    )
    assert not result.quality_bitmask[2] & int(
        ScienceQualityFlag.INSUFFICIENT_BACKGROUND
    )
    np.testing.assert_allclose(
        result.centroid_x,
        [3.0 + 48.0 / 146.0, 3.0, 3.0 + 44.0 / 138.0],
    )
    np.testing.assert_allclose(result.centroid_y, [2.0, 2.0, 2.0])
    assert result.product_semantics["observation_product"] == "final_dn"
    assert result.product_semantics["background_realization_used"] is False
    assert result.product_semantics["centroid_algorithm"] == (
        "legacy_center_of_mass_math_on_aperture_support_v1"
    )


def test_science_photometry_flags_insufficient_local_background() -> None:
    from et_mainsim.stamp_science_photometry import (
        ScienceQualityFlag,
        reduce_science_photometry_v1,
    )

    frames = np.zeros((1, 5, 7), dtype=float)
    frames[0, 2, 3] = 20.0
    aperture = np.zeros((5, 7), dtype=bool)
    aperture[2, 3] = True
    background = np.zeros((5, 7), dtype=bool)
    background[1, 1:3] = True
    cosmic = np.zeros_like(frames, dtype=bool)
    cosmic[0, 1, 1] = True
    result = reduce_science_photometry_v1(
        _delivery_from_calibrated_bgsub(frames, cosmic_mask=cosmic),
        aperture_mask=aperture,
        background_mask=background,
        minimum_background_pixels=2,
    )

    assert result.flux_expectation_bgsub_e[0] == pytest.approx(20.0)
    assert np.isnan(result.flux_local_bgsub_e[0])
    assert np.isnan(result.local_background_e_per_pixel[0])
    assert result.quality_bitmask[0] & int(
        ScienceQualityFlag.INSUFFICIENT_BACKGROUND
    )


def test_expectation_background_is_complete_without_a_local_background_mask() -> None:
    """A compact stamp must reduce from its delivered background component alone."""

    from et_mainsim.stamp_science_photometry import (
        ScienceQualityFlag,
        reduce_science_photometry_v1,
    )

    frames = np.zeros((2, 27, 27), dtype=float)
    frames[:, 13, 13] = [100.0, 125.0]
    aperture = np.zeros((27, 27), dtype=bool)
    aperture[13, 13] = True

    result = reduce_science_photometry_v1(
        _delivery_from_calibrated_bgsub(
            frames,
            background_expectation_e=37.0,
        ),
        aperture_mask=aperture,
    )

    np.testing.assert_allclose(result.flux_expectation_bgsub_e, [100.0, 125.0])
    np.testing.assert_array_equal(result.aperture_valid, [True, True])
    assert not np.any(
        result.quality_bitmask & int(ScienceQualityFlag.INSUFFICIENT_BACKGROUND)
    )
    assert not np.any(result.background_mask)
    assert result.product_semantics["default_background_product"] == (
        "background_expectation_e"
    )
    assert result.product_semantics["local_background_enabled"] is False


def test_default_optimal_aperture_training_supports_a_27_by_27_stamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The formal default must not reserve 1,024 pixels for local background."""

    import photsim7.aperture as legacy_aperture

    def choose_peak(signal, noise, plot=False):
        import torch

        mask = torch.zeros_like(signal, dtype=torch.bool)
        mask[13, 13] = True
        return mask, 20.0

    monkeypatch.setattr(legacy_aperture, "maximize_cumulative_snr", choose_peak)
    from et_mainsim.stamp_science_photometry import (
        StampSciencePhotometryPolicy,
        train_science_optimal_aperture_v1,
    )

    signal = np.zeros((4, 27, 27), dtype=float)
    signal[:, 13, 13] = 100.0
    trained = train_science_optimal_aperture_v1(
        _delivery_from_calibrated_bgsub(signal),
        raw_relative_flux=np.ones(4),
        training_raw_frame_indices=np.arange(4),
        read_noise_e_per_pixel=5.0,
        quantization_noise_e_per_pixel=0.0,
        policy=StampSciencePhotometryPolicy(),
    )

    assert trained.aperture_mask.shape == (27, 27)
    assert trained.background_mask is None
    assert trained.metadata["background_strategy"] == (
        "delivered_expectation_only"
    )


def test_train_aperture_recovers_baseline_template_with_varying_q(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import photsim7.aperture as legacy_aperture

    def choose_positive(signal, noise, plot=False):
        import torch

        assert plot is False
        return signal > 0.0, 42.0

    monkeypatch.setattr(
        legacy_aperture,
        "maximize_cumulative_snr",
        choose_positive,
    )
    from et_mainsim.stamp_science_photometry import (
        StampSciencePhotometryPolicy,
        train_science_optimal_aperture_v1,
    )

    q = np.asarray([0.5, 1.0, 1.5, 2.0], dtype=float)
    baseline = np.zeros((5, 7), dtype=float)
    baseline[2, 3] = 100.0
    baseline[2, 4] = 20.0
    frames = q[:, None, None] * baseline[None, :, :]
    delivery = _delivery_from_calibrated_bgsub(
        frames,
        background_expectation_e=16.0,
    )
    trained = train_science_optimal_aperture_v1(
        delivery,
        raw_relative_flux=q,
        training_raw_frame_indices=np.arange(q.size, dtype=np.int64),
        read_noise_e_per_pixel=3.0,
        quantization_noise_e_per_pixel=0.5,
        policy=StampSciencePhotometryPolicy(
            background_strategy="delivered_expectation_plus_local_diagnostic",
            background_guard_pixels=1,
            background_border_pixels=1,
            minimum_background_pixels=2,
            minimum_training_valid_fraction=1.0,
        ),
    )

    np.testing.assert_allclose(trained.signal_template_e, baseline)
    np.testing.assert_allclose(
        trained.noise_template_e,
        np.sqrt(baseline + 16.0 + 3.0**2 + 0.5**2),
    )
    np.testing.assert_array_equal(trained.aperture_mask, baseline > 0.0)
    assert trained.target_peak_yx == (2, 3)
    np.testing.assert_array_equal(
        trained.training_raw_frame_indices,
        np.arange(q.size),
    )
    assert not np.any(trained.aperture_mask & trained.background_mask)
    assert trained.metadata["template_fit"] == "through_origin_q_weighted_v1"


def test_train_aperture_excludes_bad_samples_per_pixel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import photsim7.aperture as legacy_aperture

    def choose_peak(signal, noise, plot=False):
        import torch

        mask = torch.zeros_like(signal, dtype=torch.bool)
        peak = int(torch.argmax(signal).item())
        mask.reshape(-1)[peak] = True
        return mask, 10.0

    monkeypatch.setattr(legacy_aperture, "maximize_cumulative_snr", choose_peak)
    from et_mainsim.stamp_science_photometry import (
        StampSciencePhotometryPolicy,
        train_science_optimal_aperture_v1,
    )

    baseline = np.zeros((5, 7), dtype=float)
    baseline[2, 3] = 50.0
    frames = np.repeat(baseline[None, :, :], 4, axis=0)
    frames[1, 2, 3] = 50_000.0
    cosmic = np.zeros_like(frames, dtype=bool)
    cosmic[1, 2, 3] = True
    trained = train_science_optimal_aperture_v1(
        _delivery_from_calibrated_bgsub(frames, cosmic_mask=cosmic),
        raw_relative_flux=np.ones(4),
        training_raw_frame_indices=np.arange(4),
        read_noise_e_per_pixel=1.0,
        quantization_noise_e_per_pixel=0.0,
        policy=StampSciencePhotometryPolicy(
            background_strategy="delivered_expectation_plus_local_diagnostic",
            background_guard_pixels=1,
            background_border_pixels=1,
            minimum_background_pixels=2,
            minimum_training_valid_fraction=0.5,
        ),
    )

    assert trained.signal_template_e[2, 3] == pytest.approx(50.0)
    assert trained.metadata["excluded_training_sample_count"] == 1


@pytest.mark.parametrize("factor", [3, 6, 12, 30])
def test_raw_accumulation_matches_direct_coadd_calibration(factor: int) -> None:
    from et_mainsim.stamp_science_photometry import (
        coadd_reference_photometry_input_v1,
        reduce_science_photometry_v1,
    )

    n_frames = factor * 2
    frames = np.arange(n_frames * 5 * 7, dtype=float).reshape(n_frames, 5, 7)
    aperture = np.zeros((5, 7), dtype=bool)
    aperture[2, 3:5] = True
    background = np.zeros((5, 7), dtype=bool)
    background[1, 1:5] = True
    raw = _delivery_from_calibrated_bgsub(frames)
    coadded = coadd_reference_photometry_input_v1(raw, factor=factor)
    result = reduce_science_photometry_v1(
        coadded,
        aperture_mask=aperture,
        background_mask=background,
        minimum_background_pixels=4,
    )

    expected = frames.reshape(2, factor, 5, 7).sum(axis=1)
    np.testing.assert_allclose(
        result.flux_expectation_bgsub_e,
        expected[:, aperture].sum(axis=1),
    )
    np.testing.assert_allclose(coadded.exposure_seconds, factor * 10.0)
    np.testing.assert_allclose(coadded.time_seconds, [0.0, factor * 10.0])


def test_model_fit_and_cdpp_use_median_centered_legacy_mean_mad() -> None:
    from et_mainsim.stamp_science_photometry import (
        compute_science_cdpp_v1,
        fit_science_variability_model_v1,
    )

    # 20 one-minute cadences yield ten accepted 2-minute bins.  The last bin
    # is an outlier chosen to distinguish median-centered from mean-centered
    # mean absolute deviation.
    q = np.ones(120, dtype=float)
    flux = np.full(20, 600.0)
    flux[-2:] = 1_200.0
    model = fit_science_variability_model_v1(
        flux_e=flux,
        aperture_valid=np.ones(20, dtype=bool),
        raw_relative_flux=q,
        raw_frame_start_index=np.arange(20) * 6,
        raw_frame_stop_index_exclusive=(np.arange(20) + 1) * 6,
    )
    metric = compute_science_cdpp_v1(
        time_seconds=np.arange(20, dtype=float) * 60.0,
        exposure_seconds=np.full(20, 60.0),
        flux_e=flux,
        aperture_valid=np.ones(20, dtype=bool),
        model_flux_e=model.fitted_flux_e,
        residual_e=model.residual_e,
        windows_minutes=(2,),
        minimum_coverage_fraction=0.95,
        minimum_accepted_bins=10,
    ).metrics_by_window_minutes[2]

    binned_rates = np.asarray([10.0] * 9 + [20.0])
    expected = (
        1.4826
        * np.mean(np.abs(binned_rates - np.median(binned_rates)))
        / np.median(binned_rates)
        * 1_000_000.0
    )
    assert metric.observed_cdpp_ppm == pytest.approx(expected)
    assert metric.accepted_bin_count == 10
    assert model.fit_intercept_e == 0.0


def test_model_fit_rejects_raw_interval_or_factor_mismatch() -> None:
    from et_mainsim.stamp_science_photometry import (
        SciencePhotometryContractError,
        fit_science_variability_model_v1,
    )

    with pytest.raises(SciencePhotometryContractError, match="raw-relative-flux"):
        fit_science_variability_model_v1(
            flux_e=np.asarray([10.0, 10.0]),
            aperture_valid=np.asarray([True, True]),
            raw_relative_flux=np.ones(3),
            raw_frame_start_index=np.asarray([0, 2]),
            raw_frame_stop_index_exclusive=np.asarray([2, 4]),
        )
