from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest


ASTER_IDS = (
    "0000000473",
    "0000000599",
    "0000000036",
    "0000000086",
    "0000000622",
)
VARLC_IDS = (
    "KIC003331147",
    "KIC011145123",
    "TIC260161111",
)


def _write_aster_file(path: Path, source_id: str, *, bad_cadence: bool = False) -> None:
    times = [13.38, 23.38, 33.38, 43.38]
    if bad_cadence:
        times[2] = 34.38
    flags = [0, 0, 7, 0]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                f"# StarID = {source_id}",
                "# Time [s], Flux variation [ppm], Flag",
                *(
                    f"{time:.2f} {ppm:.8f} {flag}"
                    for time, ppm, flag in zip(
                        times,
                        [10.0, -20.0, 30.0, 0.0],
                        flags,
                        strict=True,
                    )
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )
    path.with_suffix(".txt").write_text(
        "# observations parameters:\n"
        " magnitude = 6, duration = 1461, sampling = 10, white_noise = 0\n",
        encoding="utf-8",
    )


def _write_varlc_file(path: Path, *, negative: bool = False) -> None:
    cadence_day = 10.0 / 86_400.0
    factors = [0.95, 1.0, -0.1 if negative else 1.05, 1.1]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# time_day normalised_flux\n"
        + "".join(
            f"{index * cadence_day:.18e} {factor:.18e}\n"
            for index, factor in enumerate(factors)
        ),
        encoding="utf-8",
    )


def _component_rows(star_type: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for index in range(300):
        rows.append(
            {
                "Index": str(index + 1),
                "Mode index": str(index // 3),
                "Star type": star_type,
                "Mode type": "g",
                "l": "1",
                "m": str(index % 3 - 1),
                "Injected f (uHz)": "1000.0" if index == 0 else "2000.0",
                "Delta_f (uHz)": "0.0",
                "Period (s)": "1000.0",
                "Amplitude (ppt)": "2.0" if index == 0 else "0.0",
                "Phase (rad)": "0.25" if index == 0 else "0.0",
                "f0 (uHz)": "1000.0",
                "Grid f (uHz)": "1000.0",
                "Integer offset (uHz)": "0",
                "Decimal offset (uHz)": "0.0",
                "Delta_f min (uHz)": "0.0",
                "Delta_f max (uHz)": "0.0",
            }
        )
    return rows


def _write_component_csv(path: Path, star_type: str) -> None:
    rows = _component_rows(star_type)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _instantaneous_fractional(
    *, n_frames: int, normalization_frames: int
) -> tuple[np.ndarray, float]:
    frequency_hz = 1000.0e-6
    amplitude = 2.0e-3
    phase = 0.25
    normalization = amplitude * np.sin(
        2.0 * np.pi * frequency_hz * np.arange(normalization_frames) * 10.0
        + phase
    )
    median = float(np.median(normalization))
    values = amplitude * np.sin(
        2.0 * np.pi * frequency_hz * np.arange(n_frames) * 10.0 + phase
    )
    return values - median, median


def _write_wdlc_out(
    path: Path,
    fractional: np.ndarray,
    *,
    electron_rate: bool,
    baseline: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    label = "electron_rate" if electron_rate else "raw"
    flux = baseline * (1.0 + fractional) if electron_rate else fractional
    lines = [
        f"Lightcurve ({label}) : {len(fractional)}",
        "Time (BJD) Flux Status",
        "--------------------------------------",
    ]
    lines.extend(
        f"{index * 10.0 / 86400.0:.5f} {value:.17g} 1"
        for index, value in enumerate(flux)
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_wdlc_tree(root: Path, *, perturb_wd: bool = False) -> None:
    track = root / "wdlc" / "lightcurve_test_for_ET2.0"
    for short_name, external_id, baseline in (
        ("wd", "WD", 1_000_000.0),
        ("sdb", "sdB", 750_000.0),
    ):
        _write_component_csv(
            track / "input_models" / f"{short_name}_components.csv",
            external_id,
        )
        fractional, _median = _instantaneous_fractional(
            n_frames=8,
            normalization_frames=32,
        )
        if perturb_wd and short_name == "wd":
            fractional = fractional.copy()
            fractional[3] += 1.0e-3
        _write_wdlc_out(
            track / "lightcurve" / f"{short_name}_light_curve.out",
            fractional,
            electron_rate=False,
            baseline=baseline,
        )
        # The consistency carrier must represent the unperturbed same signal.
        electron_fractional, _ = _instantaneous_fractional(
            n_frames=8,
            normalization_frames=32,
        )
        _write_wdlc_out(
            track
            / "lightcurve"
            / f"{short_name}_light_curve_electron_rate.out",
            electron_fractional,
            electron_rate=True,
            baseline=baseline,
        )


def test_namespaced_source_ids_use_the_frozen_explicit_int64_mapping() -> None:
    from et_mainsim.stamp_science_inputs import stable_namespaced_source_id

    first = stable_namespaced_source_id("aster", "0000000473")
    assert first == 473
    assert stable_namespaced_source_id("varlc", "KIC003331147") == 3_331_147
    assert stable_namespaced_source_id("wdlc", "WD") == 1
    assert stable_namespaced_source_id("wdlc", "sdB") == 2
    with pytest.raises(ValueError, match="no frozen internal int64 mapping"):
        stable_namespaced_source_id("varlc", "0000000473")


def test_curve_global_identity_requires_namespace_to_equal_track() -> None:
    from et_mainsim.stamp_science_inputs import ScienceInputCurve

    with pytest.raises(ValueError, match="namespace must equal track"):
        ScienceInputCurve(
            track="aster",
            namespace="varlc",
            external_source_id="KIC003331147",
            source_id_int64=3_331_147,
            source_class="invalid_cross_track_identity",
            gaia_g_mag=11.5,
            detector_xpix=1500.0,
            detector_ypix=1500.0,
            factors=np.array([1.0]),
            metadata={},
        )


def test_aster_adapter_freezes_five_sources_and_records_unused_flags(
    tmp_path,
) -> None:
    from et_mainsim.stamp_science_inputs import load_aster_precision_inputs

    for source_id in ASTER_IDS:
        _write_aster_file(
            tmp_path / "Aster" / "lightcurves_test10" / f"{source_id}.dat",
            source_id,
        )

    curves = load_aster_precision_inputs(tmp_path, raw_frame_count=4)

    assert [curve.external_source_id for curve in curves] == list(ASTER_IDS)
    assert [curve.source_id_int64 for curve in curves] == [473, 599, 36, 86, 622]
    assert [curve.source_class for curve in curves] == [
        "F_dwarf",
        "G_dwarf",
        "K_dwarf",
        "subgiant",
        "red_giant",
    ]
    assert [(curve.detector_xpix, curve.detector_ypix) for curve in curves] == [
        (1500.0, 1500.0),
        (3000.0, 1500.0),
        (4500.0, 1500.0),
        (6000.0, 1500.0),
        (7500.0, 1500.0),
    ]
    np.testing.assert_allclose(
        curves[0].factors,
        1.0 + np.array([10.0, -20.0, 30.0, 0.0]) * 1.0e-6,
    )
    assert curves[0].namespace == "aster"
    assert curves[0].gaia_g_mag == 11.5
    assert curves[0].psf_id == 6
    assert curves[0].psf_node_angle_deg == 12.0
    assert curves[0].location_mode == "reference_field_nonphysical"
    assert curves[0].dva_enabled is False
    assert curves[0].metadata["input_flags"] == {
        "policy": "recorded_not_applied",
        "nonzero_count": 1,
        "value_counts": {"0": 3, "7": 1},
    }
    assert curves[0].metadata["input_generator_metadata"] == {
        "magnitude": 6.0,
        "sampling_seconds": 10.0,
        "magnitude_role": "not_adopted_for_precision_absolute_flux",
        "precision_gaia_g_vega": 11.5,
        "g6_role": "separate_saturation_validation",
    }
    assert curves[0].metadata["magnitude_origin"] == (
        "precision_override_of_generator_magnitude_6"
    )
    assert curves[0].metadata["input_log_file"]["path"].endswith(
        "0000000473.txt"
    )
    assert curves[0].metadata["input_provider"] == "PSLS"
    assert curves[0].metadata["input_time"]["absolute_origin_ignored"] is True
    assert curves[0].factors.flags.writeable is False


def test_aster_adapter_fails_closed_on_non_10_second_native_cadence(
    tmp_path,
) -> None:
    from et_mainsim.stamp_science_inputs import load_aster_precision_inputs

    for index, source_id in enumerate(ASTER_IDS):
        _write_aster_file(
            tmp_path / "Aster" / "lightcurves_test10" / f"{source_id}.dat",
            source_id,
            bad_cadence=index == 2,
        )

    with pytest.raises(ValueError, match="native cadence.*10"):
        load_aster_precision_inputs(tmp_path, raw_frame_count=4)


def test_varlc_adapter_uses_all_three_normalized_flux_curves(tmp_path) -> None:
    from et_mainsim.stamp_science_inputs import load_varlc_inputs

    for source_id in VARLC_IDS:
        _write_varlc_file(tmp_path / "varlc" / f"{source_id}_simulated_light_curve.dat")

    curves = load_varlc_inputs(tmp_path, raw_frame_count=4)

    assert [curve.external_source_id for curve in curves] == list(VARLC_IDS)
    assert [curve.source_id_int64 for curve in curves] == [
        3_331_147,
        11_145_123,
        260_161_111,
    ]
    assert [(curve.detector_xpix, curve.detector_ypix) for curve in curves] == [
        (2000.0, 4500.0),
        (4450.0, 4500.0),
        (6900.0, 4500.0),
    ]
    np.testing.assert_array_equal(curves[2].factors, [0.95, 1.0, 1.05, 1.1])
    assert curves[2].metadata["q_definition"] == "normalised_flux"
    assert curves[2].metadata["magnitude_origin"] == (
        "project_default_missing_input"
    )
    assert curves[2].metadata["input_time"]["used_for"] == (
        "cadence_and_order_validation_only"
    )


def test_varlc_adapter_rejects_nonpositive_relative_flux(tmp_path) -> None:
    from et_mainsim.stamp_science_inputs import load_varlc_inputs

    for index, source_id in enumerate(VARLC_IDS):
        _write_varlc_file(
            tmp_path / "varlc" / f"{source_id}_simulated_light_curve.dat",
            negative=index == 1,
        )

    with pytest.raises(ValueError, match="strictly positive"):
        load_varlc_inputs(tmp_path, raw_frame_count=4)


def test_common_track_dispatch_resolves_duration_to_10_second_frames(tmp_path) -> None:
    from et_mainsim.stamp_science_inputs import load_science_track_inputs

    for source_id in VARLC_IDS:
        _write_varlc_file(tmp_path / "varlc" / f"{source_id}_simulated_light_curve.dat")

    curves = load_science_track_inputs(
        "varlc",
        tmp_path,
        duration_days=40.0 / 86_400.0,
        raw_exposure_seconds=10.0,
    )

    assert len(curves) == 3
    assert all(curve.factors.size == 4 for curve in curves)
    with pytest.raises(ValueError, match="10-second"):
        load_science_track_inputs(
            "varlc",
            tmp_path,
            duration_days=40.0 / 86_400.0,
            raw_exposure_seconds=5.0,
        )


def test_wdlc_adapter_reconstructs_modes_gates_out_and_integrates_exposures(
    tmp_path,
    monkeypatch,
) -> None:
    import et_mainsim.stamp_science_inputs as inputs

    _write_wdlc_tree(tmp_path)
    monkeypatch.setattr(inputs, "WDLC_NORMALIZATION_RAW_FRAMES", 32)

    curves = inputs.load_wdlc_inputs(tmp_path, raw_frame_count=8)

    assert [curve.external_source_id for curve in curves] == ["WD", "sdB"]
    assert [curve.source_id_int64 for curve in curves] == [1, 2]
    assert [(curve.detector_xpix, curve.detector_ypix) for curve in curves] == [
        (3000.0, 7500.0),
        (6000.0, 7500.0),
    ]
    _, median = _instantaneous_fractional(n_frames=8, normalization_frames=32)
    starts = np.arange(8, dtype=np.float64) * 10.0
    expected = 1.0 + (
        2.0e-3
        * np.sin(2.0 * np.pi * 1.0e-3 * (starts + 5.0) + 0.25)
        * np.sinc(1.0e-3 * 10.0)
        - median
    )
    np.testing.assert_allclose(curves[0].factors, expected, rtol=0.0, atol=1.0e-12)
    gate = curves[0].metadata["validation"]["mode_reconstruction_vs_fractional_out"]
    assert gate["passed"] is True
    assert gate["max_abs"] <= 1.0e-12
    assert gate["rms"] <= 1.0e-12
    assert gate["max_abs_tolerance"] == pytest.approx(15.0e-6)
    assert gate["rms_tolerance"] == pytest.approx(2.0e-6)
    assert curves[0].metadata["electron_rate_role"] == (
        "consistency_only_not_ET_absolute_flux"
    )
    assert curves[0].metadata["magnitude_origin"] == (
        "project_default_missing_input"
    )
    assert curves[0].metadata["instantaneous_normalization"]["frame_count"] == 32
    assert curves[0].metadata["exposure_sampling"] == "analytic_sinc_interval_mean"


def test_wdlc_180d_mode_rounding_budget_keeps_max_and_rms_gates_independent() -> None:
    import et_mainsim.stamp_science_inputs as inputs

    # The approved 180-day WD audit has an 11.7717-ppm maximum but only a
    # 1.689-ppm RMS.  Keep that finite-precision envelope distinct from the
    # tighter RMS guard so a broad mismatch still fails closed.
    approved = np.full(100, 1.2e-6, dtype=np.float64)
    approved[0] = 11.7717e-6
    accepted = inputs._wdlc_mode_gate_metrics(
        approved,
        np.zeros_like(approved),
    )

    assert accepted["passed"] is True
    assert accepted["max_abs_tolerance"] == pytest.approx(15.0e-6)
    assert accepted["rms_tolerance"] == pytest.approx(2.0e-6)

    max_outlier = np.zeros(100, dtype=np.float64)
    max_outlier[0] = 15.0001e-6
    assert inputs._wdlc_mode_gate_metrics(
        max_outlier,
        np.zeros_like(max_outlier),
    )["passed"] is False

    broad_mismatch = np.full(100, 2.0001e-6, dtype=np.float64)
    assert inputs._wdlc_mode_gate_metrics(
        broad_mismatch,
        np.zeros_like(broad_mismatch),
    )["passed"] is False

    # The measured 270-day WD envelope must remain outside this emergency
    # 180-day budget; approval of the shorter campaign is not an implicit
    # long-baseline tolerance expansion.
    long_baseline = np.full(100, 2.49324e-6, dtype=np.float64)
    long_baseline[0] = 19.1715e-6
    assert inputs._wdlc_mode_gate_metrics(
        long_baseline,
        np.zeros_like(long_baseline),
    )["passed"] is False


def test_wdlc_adapter_fails_closed_when_fractional_out_breaks_mode_gate(
    tmp_path,
    monkeypatch,
) -> None:
    import et_mainsim.stamp_science_inputs as inputs

    _write_wdlc_tree(tmp_path, perturb_wd=True)
    monkeypatch.setattr(inputs, "WDLC_NORMALIZATION_RAW_FRAMES", 32)

    with pytest.raises(ValueError, match="mode reconstruction gate"):
        inputs.load_wdlc_inputs(tmp_path, raw_frame_count=8)


def test_common_factor_snapshot_round_trip_keeps_namespaced_identity(
    tmp_path,
) -> None:
    from et_mainsim.stamp_science_inputs import (
        ScienceInputCurve,
        read_science_factor_snapshot,
        stable_namespaced_source_id,
        write_science_factor_snapshot,
    )

    curve = ScienceInputCurve(
        track="varlc",
        namespace="varlc",
        external_source_id="KIC003331147",
        source_id_int64=stable_namespaced_source_id("varlc", "KIC003331147"),
        source_class="pulsating_variable",
        gaia_g_mag=11.5,
        detector_xpix=2000.0,
        detector_ypix=4500.0,
        factors=np.array([0.9, 1.0, 1.1]),
        metadata={"q_definition": "normalised_flux"},
    )

    identity = write_science_factor_snapshot(tmp_path / "factor.npz", curve=curve)
    loaded = read_science_factor_snapshot(tmp_path / "factor.npz")

    assert len(identity["sha256"]) == 64
    assert loaded.source_id_int64 == curve.source_id_int64
    assert loaded.namespace == "varlc"
    assert loaded.external_source_id == "KIC003331147"
    np.testing.assert_array_equal(loaded.factors, curve.factors)
    assert loaded.metadata["time_alignment"] == "simulation_raw_frame_index"
