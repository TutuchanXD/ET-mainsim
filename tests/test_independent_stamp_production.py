from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np


@dataclass(frozen=True)
class _Raw:
    frame_index: int
    value: int


def _fake_render(frame_index: int) -> _Raw:
    return _Raw(frame_index=frame_index, value=frame_index + 1)


def _fake_adapter(raw: _Raw):
    from et_mainsim.independent_stamp_production import RawStampDeliveryFrame

    shape = (13, 13)
    value = raw.value
    return RawStampDeliveryFrame(
        final_dn=np.full(shape, value, dtype=np.uint16),
        background_expectation_e=np.full(shape, value * 0.5, dtype=np.float64),
        bias_level_dn=float(value),
        column_noise_dn_by_x=np.full(shape[1], value * 0.25, dtype=np.float64),
        valid_mask=np.ones(shape, dtype=bool),
        fullwell_mask=np.full(shape, raw.frame_index == 1, dtype=bool),
        adc_low_mask=np.zeros(shape, dtype=bool),
        adc_high_mask=np.full(shape, raw.frame_index == 4, dtype=bool),
        cosmic_mask=np.full(shape, raw.frame_index == 2, dtype=bool),
    )


def test_streamed_independent_shard_writes_raw_and_all_coadds(tmp_path) -> None:
    from et_mainsim.independent_stamp_production import (
        IndependentStampShardRequest,
        run_independent_stamp_time_shard,
    )
    from et_mainsim.stamp_delivery import read_stamp_delivery_bundle
    from et_mainsim.time_shards import ContinuousTimeShard

    shard = ContinuousTimeShard(
        shard_id=7,
        raw_start_index=0,
        raw_stop_index=6,
        coadd_sizes=(3, 6),
        raw_exposure_seconds=10.0,
    )
    request = IndependentStampShardRequest(
        output_root=tmp_path,
        target_source_id=42,
        stamp_shape=(13, 13),
        shard=shard,
        gain_e_per_dn=2.0,
        manifest={"run_id": "test", "input": "synthetic"},
        provenance={"code": "test"},
        batch_size=2,
    )

    report = run_independent_stamp_time_shard(
        request,
        render_raw=_fake_render,
        adapt_raw=_fake_adapter,
    )

    raw = read_stamp_delivery_bundle(report.raw_path)
    coadd3 = read_stamp_delivery_bundle(report.coadd_paths[3])
    coadd6 = read_stamp_delivery_bundle(report.coadd_paths[6])
    assert raw.product_kind == "raw"
    assert raw.shape == (6, 13, 13)
    assert raw.final_dn.dtype == np.dtype(np.uint16)
    np.testing.assert_array_equal(raw.final_dn[:, 0, 0], [1, 2, 3, 4, 5, 6])
    np.testing.assert_allclose(raw.background_expectation_e[:, 0, 0], [0.5, 1, 1.5, 2, 2.5, 3])
    assert raw.manifest["scene_policy"] == "independent_target"
    assert raw.manifest["time_shard"] == shard.to_manifest_dict()
    assert raw.provenance["observation_product"] == "final_dn"
    assert raw.provenance["background_realization_used"] is False

    assert coadd3.product_kind == "coadd"
    assert coadd3.final_dn.dtype == np.dtype(np.uint64)
    np.testing.assert_array_equal(coadd3.final_dn[:, 0, 0], [6, 15])
    np.testing.assert_allclose(coadd3.background_expectation_e[:, 0, 0], [3.0, 7.5])
    np.testing.assert_allclose(coadd3.bias_level_sum_dn, [6.0, 15.0])
    np.testing.assert_allclose(coadd3.column_noise_sum_dn_by_x[:, 0], [1.5, 3.75])
    np.testing.assert_array_equal(coadd3.fullwell_count[:, 0, 0], [1, 0])
    np.testing.assert_array_equal(coadd3.adc_high_count[:, 0, 0], [0, 1])
    np.testing.assert_array_equal(coadd3.cosmic_count[:, 0, 0], [1, 0])
    np.testing.assert_array_equal(coadd3.raw_frame_start_index, [0, 3])
    np.testing.assert_array_equal(coadd3.raw_frame_stop_index_exclusive, [3, 6])

    np.testing.assert_array_equal(coadd6.final_dn[:, 0, 0], [21])
    np.testing.assert_array_equal(coadd6.fullwell_count[:, 0, 0], [1])
    np.testing.assert_array_equal(coadd6.adc_high_count[:, 0, 0], [1])
    np.testing.assert_array_equal(coadd6.cosmic_count[:, 0, 0], [1])


def test_independent_shard_publishes_no_member_if_one_coadd_fails(
    tmp_path,
    monkeypatch,
) -> None:
    """The final shard directory appears only after every member validates."""

    import et_mainsim.independent_stamp_production as production
    import pytest

    from et_mainsim.independent_stamp_production import (
        IndependentStampShardRequest,
        run_independent_stamp_time_shard,
    )
    from et_mainsim.time_shards import ContinuousTimeShard

    request = IndependentStampShardRequest(
        output_root=tmp_path,
        target_source_id=42,
        stamp_shape=(13, 13),
        shard=ContinuousTimeShard(
            shard_id=7,
            raw_start_index=0,
            raw_stop_index=6,
            coadd_sizes=(3, 6),
            raw_exposure_seconds=10.0,
        ),
        gain_e_per_dn=2.0,
        manifest={"run_id": "test", "input": "synthetic"},
        provenance={"code": "test"},
        batch_size=2,
    )
    original_complete = production.StampDeliveryBundleAppender.complete

    def fail_one_coadd(self):
        if self.path.name == "coadd_30s.h5":
            raise OSError("synthetic coadd validation failure")
        return original_complete(self)

    monkeypatch.setattr(
        production.StampDeliveryBundleAppender,
        "complete",
        fail_one_coadd,
    )

    with pytest.raises(OSError, match="synthetic coadd validation failure"):
        run_independent_stamp_time_shard(
            request,
            render_raw=_fake_render,
            adapt_raw=_fake_adapter,
        )

    assert not request.shard_root.exists()
    assert not (request.shard_root / "raw.h5").exists()
    assert not (request.shard_root / "coadd_30s.h5").exists()
    assert not (request.shard_root / "coadd_60s.h5").exists()
    assert not list(request.shard_root.parent.glob(".shard_00007.*.partial"))


def test_independent_shard_refuses_existing_complete_product(tmp_path) -> None:
    from et_mainsim.independent_stamp_production import (
        IndependentStampShardRequest,
        run_independent_stamp_time_shard,
    )
    from et_mainsim.time_shards import ContinuousTimeShard

    request = IndependentStampShardRequest(
        output_root=tmp_path,
        target_source_id=42,
        stamp_shape=(13, 13),
        shard=ContinuousTimeShard(
            shard_id=0,
            raw_start_index=0,
            raw_stop_index=3,
            coadd_sizes=(3,),
            raw_exposure_seconds=10.0,
        ),
        gain_e_per_dn=2.0,
        manifest={"run_id": "test"},
        provenance={"code": "test"},
    )
    run_independent_stamp_time_shard(
        request,
        render_raw=_fake_render,
        adapt_raw=_fake_adapter,
    )

    import pytest

    with pytest.raises(FileExistsError, match="already exists"):
        run_independent_stamp_time_shard(
            request,
            render_raw=_fake_render,
            adapt_raw=_fake_adapter,
        )


def test_photsim7_adapter_maps_delivery_calibration_and_quality_planes() -> None:
    from et_mainsim.independent_stamp_production import (
        raw_stamp_delivery_frame_from_photsim7,
    )

    shape = (13, 13)
    products = SimpleNamespace(
        final_stamp=SimpleNamespace(
            array=np.full(shape, 1234, dtype=np.uint16),
            unit="dn",
        ),
        valid_detector_mask=np.ones(shape, dtype=bool),
        full_well_clipped_mask=SimpleNamespace(
            array=np.eye(shape[0], dtype=bool),
        ),
        adc_low_clipped_mask=None,
        adc_high_clipped_mask=SimpleNamespace(
            array=np.fliplr(np.eye(shape[0], dtype=bool)),
        ),
        cosmic_events=SimpleNamespace(mask=np.tri(*shape, dtype=bool)),
    )
    detector_result = SimpleNamespace(
        bias_metadata=SimpleNamespace(
            bias_level_adu=3500.0,
            column_noise_vector_adu=np.arange(shape[1], dtype=np.float32)[None, :],
        )
    )
    result = SimpleNamespace(
        stamp_products=products,
        detector_result=detector_result,
        renderer_components={
            "background_expectation": np.full(shape, 41.5, dtype=np.float32)
        },
    )

    delivery = raw_stamp_delivery_frame_from_photsim7(result)

    assert delivery.final_dn.dtype == np.dtype(np.uint16)
    np.testing.assert_array_equal(delivery.final_dn, products.final_stamp.array)
    np.testing.assert_allclose(delivery.background_expectation_e, 41.5)
    assert delivery.bias_level_dn == 3500.0
    np.testing.assert_allclose(delivery.column_noise_dn_by_x, np.arange(shape[1]))
    np.testing.assert_array_equal(
        delivery.fullwell_mask,
        products.full_well_clipped_mask.array,
    )
    assert not np.any(delivery.adc_low_mask)
    np.testing.assert_array_equal(
        delivery.adc_high_mask,
        products.adc_high_clipped_mask.array,
    )
    np.testing.assert_array_equal(delivery.cosmic_mask, products.cosmic_events.mask)
