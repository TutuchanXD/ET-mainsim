from __future__ import annotations

import json

import numpy as np
import pytest


def test_exposure_averaged_factors_integrate_piecewise_linear_clean_flux() -> None:
    from et_mainsim.galaxy_lightcurves import exposure_averaged_factors

    # q(t) is 1 + t / 20 over 0..20 s.  A 10 s exposure measures its interval
    # mean, not either endpoint value.
    result = exposure_averaged_factors(
        native_time_seconds=np.array([0.0, 20.0]),
        clean_flux_factor=np.array([1.0, 2.0]),
        n_raw_frames=2,
        raw_exposure_seconds=10.0,
    )

    np.testing.assert_allclose(result, [1.25, 1.75], rtol=0.0, atol=1e-14)


def test_exposure_averaged_factors_rejects_insufficient_native_coverage() -> None:
    from et_mainsim.galaxy_lightcurves import exposure_averaged_factors

    with pytest.raises(ValueError, match="does not cover"):
        exposure_averaged_factors(
            native_time_seconds=np.array([0.0, 15.0]),
            clean_flux_factor=np.array([1.0, 1.0]),
            n_raw_frames=2,
            raw_exposure_seconds=10.0,
        )


def test_load_galaxy_fits_filters_padding_and_converts_delta_flux(tmp_path) -> None:
    from astropy.io import fits

    from et_mainsim.galaxy_lightcurves import load_galaxy_lightcurves

    path = tmp_path / "galaxy.fits"
    columns = [
        fits.Column(name="Source", format="K", array=np.array([42], dtype=np.int64)),
        fits.Column(name="Gmag", format="D", array=np.array([11.25])),
        fits.Column(name="RAJ2000", format="D", array=np.array([123.0])),
        fits.Column(name="DEJ2000", format="D", array=np.array([-45.0])),
        fits.Column(name="class", format="12A", array=np.array(["rotation"])),
        fits.Column(
            name="time",
            format="PD()",
            array=np.array([np.array([50.0, 50.0 + 10.0 / 86400.0, np.nan])], dtype=object),
        ),
        fits.Column(
            name="relative_flux",
            format="PE()",
            array=np.array([np.array([0.0, 0.2, np.nan], dtype=np.float32)], dtype=object),
        ),
    ]
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(columns)]).writeto(
        path
    )

    curves = load_galaxy_lightcurves(path, source_ids=(42,))

    assert set(curves) == {42}
    curve = curves[42]
    assert curve.source_id == 42
    assert curve.source_class == "rotation"
    assert curve.gaia_g_mag == pytest.approx(11.25)
    np.testing.assert_allclose(curve.native_time_seconds, [0.0, 10.0])
    np.testing.assert_allclose(curve.clean_flux_factor, [1.0, 1.2])
    assert curve.input_identity["sha256"]


def test_factor_snapshot_round_trip_preserves_identity_and_semantics(tmp_path) -> None:
    from et_mainsim.galaxy_lightcurves import (
        GalaxyLightCurve,
        read_galaxy_factor_snapshot,
        write_galaxy_factor_snapshot,
    )

    curve = GalaxyLightCurve(
        source_id=42,
        gaia_g_mag=11.25,
        ra_deg=123.0,
        dec_deg=-45.0,
        source_class="rotation",
        native_time_seconds=np.array([0.0, 10.0, 20.0]),
        clean_flux_factor=np.array([1.0, 1.1, 1.2]),
        input_identity={"path": "/input.fits", "sha256": "a" * 64},
    )
    path = tmp_path / "factor.npz"

    identity = write_galaxy_factor_snapshot(
        path,
        curve=curve,
        factors=np.array([1.025, 1.075]),
        raw_exposure_seconds=10.0,
    )
    restored = read_galaxy_factor_snapshot(path)

    assert identity["sha256"]
    assert restored.source_id == 42
    np.testing.assert_allclose(restored.factors, [1.025, 1.075])
    assert restored.metadata["q_definition"] == "1_plus_delta_f_over_f_ref"
    assert restored.metadata["interpolation"] == "piecewise_linear_clean_flux"
    assert json.loads(restored.metadata_json)["raw_exposure_seconds"] == 10.0
