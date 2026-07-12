from __future__ import annotations

import os
import pickle
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy import units as u


DEFAULT_LEGACY_CACHE = Path(
    "/home/cxgao/Results/ET-mainsim/main_rd_g18_parallel/"
    "main_rd_500x500_detectorxy_310-50-2420_sky23p2_colnoise0_"
    "straylight10_last90/cache/stars_main_rd_500x500_detectorxy_310-50-2420_"
    "sky23p2_colnoise0_straylight10_last90.npz"
)


def _write_psf_bundle(data_root: Path) -> str:
    bundle_name = "psf/et/contract"
    bundle_dir = data_root / bundle_name
    bundle_dir.mkdir(parents=True)
    n_subpixels = 3
    rows, cols = 5, 7
    y, x = np.mgrid[: rows * n_subpixels, : cols * n_subpixels].astype(np.float32)
    yy = (y - y.mean()) / n_subpixels
    xx = (x - x.mean()) / n_subpixels
    image = np.exp(-0.5 * (xx**2 + yy**2) / 1.2**2).astype(np.float32)
    image /= image.sum(dtype=np.float64)
    payload = {
        "images": {0: {n_subpixels: np.stack([x, y, image])}},
        "angles": np.array([8.5], dtype=np.float64),
    }
    with (bundle_dir / "sim_psf_images.pkl").open("wb") as handle:
        pickle.dump(payload, handle)
    return bundle_name


def _contract_spec(tmp_path: Path):
    from photsim7.spec_factories import make_et_main_detector_spec
    from photsim7.specs import (
        CatalogSpec,
        CosmicRaySpec,
        DetectorResponseSpec,
        DynamicEffectsSpec,
    )

    bundle_name = _write_psf_bundle(tmp_path)
    base = make_et_main_detector_spec(shape=(5, 7), run_seed=20260712)
    return replace(
        base,
        observation=replace(
            base.observation,
            exposure_duration=1 * u.s,
            observing_duration=1 * u.s,
        ),
        catalog=CatalogSpec(
            source_type="detector_xy_csv",
            source_path="detector.csv",
            source_id_column="source_id",
            magnitude_column="gmag",
            x_column="x",
            y_column="y",
            input_magnitude_system="Gaia_G",
            photon_magnitude_system="ET",
            background_stars_max_mag=20.0,
            target_ra_deg=10.0,
            target_dec_deg=20.0,
            query_options={
                "reference_field_angle_deg": 8.5,
                "reference_field_polar_angle_rad": 0.75,
                "metadata": {"default_field_angle_deg": 8.5},
            },
        ),
        detector_response=DetectorResponseSpec(
            enable_inter_pixel_response=False,
            enable_intra_pixel_response=False,
            enable_pixel_phase_response=False,
        ),
        cosmic_rays=CosmicRaySpec(enabled=False),
        psf=replace(
            base.psf,
            bundle_name=bundle_name,
            field_id=0,
            field_id_policy=None,
            use_jitter_integrated_psf=False,
        ),
        dynamic_effects=DynamicEffectsSpec(),
        sky=replace(
            base.sky,
            background_flux=0 * u.electron / u.s / u.pix,
            scattered_light=0 * u.electron / u.s / u.pix,
            dark_current=0 * u.electron / u.s / u.pix,
        ),
        readout=replace(
            base.readout,
            readout_noise=0 * u.electron / u.pix,
            column_noise_sigma_adu=0 * u.adu,
            bias_level_adu=10 * u.adu,
            gain_electrons_per_adu=2 * u.electron / u.adu,
        ),
    )


def test_photsim7_service_pipeline_contract_is_deterministic(tmp_path):
    from photsim7.data_registry import DataRegistry
    from photsim7.frame_products import FRAME_PRODUCT_SCHEMA_ID
    from photsim7.full_frame_pipeline import run_single_cadence_full_frame
    from photsim7.simulation_services import build_full_frame_services

    pd.DataFrame(
        {"source_id": [17], "gmag": [10.0], "x": [0.0], "y": [0.0]}
    ).to_csv(tmp_path / "detector.csv", index=False)
    spec = _contract_spec(tmp_path)
    services = build_full_frame_services(
        spec,
        frame_exposure=1 * u.s,
        data_registry=DataRegistry(data_root=tmp_path),
    )
    renderer_options = {
        "enable_stellar_photon_noise": False,
        "enable_background_light": False,
        "enable_scattered_light": False,
        "enable_dark_current": False,
        "progress": False,
    }

    first = run_single_cadence_full_frame(
        spec,
        services=services,
        frame_index=0,
        renderer_options=renderer_options,
    )
    second = run_single_cadence_full_frame(
        spec,
        services=services,
        frame_index=0,
        renderer_options=renderer_options,
    )

    first_frame = np.asarray(first.frame_products.final_frame.array)
    second_frame = np.asarray(second.frame_products.final_frame.array)
    np.testing.assert_array_equal(first_frame, second_frame)
    assert first_frame.shape == (5, 7)
    assert first_frame.dtype == np.uint16
    assert np.max(first_frame) > 10
    assert first.manifest_payload["schema_id"] == FRAME_PRODUCT_SCHEMA_ID
    assert first.manifest_payload["arrays"]["final_frame"]["unit"] == "dn"
    assert first.provenance["services"]["schema_id"] == (
        "photsim7.full_frame_services.v1"
    )
    np.testing.assert_allclose(services.catalog.star_data["frame_xpix"], [3.0])
    np.testing.assert_allclose(services.catalog.star_data["frame_ypix"], [2.0])
    assert services.catalog.metadata["default_field_angle_deg"] == 8.5


def test_real_legacy_main_rd_cache_reads_through_package_schema():
    from photsim7.catalog_sources import StarCatalogCache

    cache_path = Path(
        os.environ.get("MAIN_RD_LEGACY_CACHE", str(DEFAULT_LEGACY_CACHE))
    ).expanduser()
    if not cache_path.exists():
        pytest.skip(f"legacy main-rd cache is unavailable: {cache_path}")

    catalog = StarCatalogCache.read(cache_path)

    assert catalog.n_sources == 17_779
    assert catalog.metadata["cache"]["schema_id"] == (
        "legacy.et_mainsim.star_cache"
    )
    assert {
        "x0",
        "y0",
        "source_id",
        "detector_xpix_shifted",
        "detector_ypix_shifted",
    }.issubset(catalog.star_data)
    assert {"et_mag", "gaia_g_mag"}.intersection(catalog.star_data)
