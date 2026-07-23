from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.table import Table


def _write_aster_inputs(tmp_path, *, n_frames: int = 60):
    source_dat = tmp_path / "0000000622.dat"
    source_dat.write_bytes(b"aster source payload identity\n")
    source_log = tmp_path / "0000000622.txt"
    source_log.write_text(
        "StarID = 0000000622\n"
        "# observations parameters:\n"
        " magnitude = 6, duration = 1461, sampling = 10, white_noise = 0\n",
        encoding="utf-8",
    )
    variability = tmp_path / "aster_psls_0000000622_30d_10s_variability.ecsv"
    table = Table()
    table["curve_id"] = ["aster_native"] * n_frames
    table["frame_index"] = np.arange(n_frames, dtype=np.int64)
    table["relative_flux"] = 1.0 + np.arange(n_frames, dtype=np.float64) * 1e-6
    table["simulation_time_s"] = np.arange(n_frames, dtype=np.float64) * 10.0
    table.meta = {
        "flux_semantics": "q = 1 + ppm * 1e-6",
        "time_alignment": "simulation_raw_frame_index",
    }
    table.write(variability, format="ascii.ecsv", overwrite=True)
    return source_dat, source_log, variability


def test_prepare_aster_g6_freezes_explicit_psf_inputs_and_time_plan(tmp_path) -> None:
    from et_mainsim.aster_saturation_validation import (
        AsterG6SaturationValidationConfig,
        prepare_aster_g6_saturation_validation,
    )

    source_dat, source_log, variability = _write_aster_inputs(tmp_path)
    data_root = tmp_path / "data"
    data_root.mkdir()
    prepared = prepare_aster_g6_saturation_validation(
        AsterG6SaturationValidationConfig(
            source_dat=source_dat,
            source_log=source_log,
            variability_ecsv=variability,
            output_root=tmp_path / "output",
            run_id="aster-g6-fixture",
            data_root=data_root,
            n_raw_frames=60,
            max_raw_frames_per_shard=60,
            device="cpu",
        )
    )

    manifest = json.loads(prepared.manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_id"] == "et_mainsim.aster_g6_saturation_validation.v1"
    assert manifest["observation_product"] == "final_dn"
    assert manifest["scientific_scope"]["purpose"] == "saturation_response_validation"
    assert manifest["scientific_scope"]["not_precision_photometry"] is True
    assert manifest["target"]["gaia_g_mag"] == 6.0
    assert manifest["target"]["psf_id"] == 6
    assert manifest["target"]["psf_node_angle_deg"] == 12.0
    assert manifest["target"]["coordinate_mode"] == "explicit_psf_no_sky_coordinate"
    assert manifest["simulation_spec_base"]["dynamic_effects"]["dva"][
        "enabled"
    ] is False
    assert manifest["delivery"]["coadd_sizes"] == [3, 6, 12, 30]
    assert manifest["delivery"]["time_plan_relative_path"] == "inputs/time_shards.json"
    assert prepared.time_plan.accepted_raw_frame_count == 60
    assert len(prepared.time_plan.shards) == 1

    static_target = Table.read(prepared.run_root / "inputs" / "aster_g6_static_target.ecsv")
    injected_target = Table.read(prepared.run_root / "inputs" / "aster_g6_injected_target.ecsv")
    frozen_curve = Table.read(
        prepared.run_root / "inputs" / "aster_psls_0000000622_g6_1h_10s_variability.ecsv"
    )
    assert static_target.colnames == ["source_id", "gaia_g_mag", "psf_id"]
    assert injected_target["curve_id"].tolist() == ["aster_psls_0000000622_g6_1h"]
    assert frozen_curve["frame_index"].tolist() == list(range(60))
    np.testing.assert_allclose(
        frozen_curve["relative_flux"],
        1.0 + np.arange(60, dtype=np.float64) * 1e-6,
    )
    assert manifest["inputs"]["frozen_variability"]["relative_path"] == (
        "inputs/aster_psls_0000000622_g6_1h_10s_variability.ecsv"
    )


@pytest.mark.parametrize(
    ("log_text", "message"),
    [
        ("magnitude = 11, sampling = 10\n", "magnitude=6"),
        ("magnitude = 6, sampling = 30\n", "sampling=10"),
    ],
)
def test_prepare_aster_g6_rejects_mismatched_native_metadata(
    tmp_path,
    log_text,
    message,
) -> None:
    from et_mainsim.aster_saturation_validation import (
        AsterG6SaturationValidationConfig,
        prepare_aster_g6_saturation_validation,
    )

    source_dat, source_log, variability = _write_aster_inputs(tmp_path)
    source_log.write_text(log_text, encoding="utf-8")
    data_root = tmp_path / "data"
    data_root.mkdir()
    with pytest.raises(ValueError, match=message):
        prepare_aster_g6_saturation_validation(
            AsterG6SaturationValidationConfig(
                source_dat=source_dat,
                source_log=source_log,
                variability_ecsv=variability,
                output_root=tmp_path / "output",
                run_id="aster-g6-fixture",
                data_root=data_root,
                n_raw_frames=60,
                max_raw_frames_per_shard=60,
                device="cpu",
            )
        )


def test_aster_g6_worker_uses_paired_rng_and_formal_delivery_contract(
    tmp_path,
    monkeypatch,
) -> None:
    import et_mainsim.aster_saturation_validation as validation
    import et_mainsim.workflows.stamp as stamp_workflow
    from photsim7.catalog_sources import PreparedStarCatalog
    from photsim7.source_variability import SourceVariability

    source_dat, source_log, variability = _write_aster_inputs(tmp_path)
    data_root = tmp_path / "data"
    data_root.mkdir()
    prepared = validation.prepare_aster_g6_saturation_validation(
        validation.AsterG6SaturationValidationConfig(
            source_dat=source_dat,
            source_log=source_log,
            variability_ecsv=variability,
            output_root=tmp_path / "output",
            run_id="aster-g6-worker-fixture",
            data_root=data_root,
            n_raw_frames=60,
            max_raw_frames_per_shard=60,
            device="cpu",
        )
    )
    api = SimpleNamespace(
        PreparedStarCatalog=PreparedStarCatalog,
        SourceVariability=SourceVariability,
        load_psf_bundle=lambda *args, **kwargs: {
            "images": {6: object()},
            "angles": np.asarray([0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0]),
        },
    )
    monkeypatch.setattr(stamp_workflow, "_science_api", lambda: api)
    monkeypatch.setattr(
        "photsim7.simulation_services.build_simulation_context",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "photsim7.simulation_services.build_stamp_services",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    render_calls = []

    def _fake_render(*args, **kwargs):
        render_calls.append(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr("photsim7.stamp_pipeline.run_single_cadence_stamp", _fake_render)
    requests = []

    def _fake_delivery(request, *, render_raw, adapt_raw):
        requests.append(request)
        render_raw(0)
        return SimpleNamespace(
            shard_id=request.shard.shard_id,
            raw_frame_count=request.shard.raw_stop_index,
            raw_path=request.shard_root / "raw.h5",
            coadd_paths={3: request.shard_root / "coadd_30s.h5"},
        )

    monkeypatch.setattr(validation, "run_independent_stamp_time_shard", _fake_delivery)

    static = validation.run_aster_g6_saturation_validation(
        prepared.manifest_path,
        case="static",
        data_root=data_root,
        device="cpu",
        batch_size=7,
    )
    injected = validation.run_aster_g6_saturation_validation(
        prepared.manifest_path,
        case="injected",
        data_root=data_root,
        device="cpu",
        batch_size=7,
    )

    assert static[0]["case"] == "static"
    assert injected[0]["case"] == "injected"
    assert [request.target_source_id for request in requests] == [
        9000000000000000622,
        9000000000000000622,
    ]
    assert [request.stamp_shape for request in requests] == [(100, 300), (100, 300)]
    assert [request.batch_size for request in requests] == [7, 7]
    assert requests[0].manifest["target_input_truth"]["variability"]["enabled"] is False
    assert requests[1].manifest["target_input_truth"]["variability"]["enabled"] is True
    assert requests[0].provenance["observation_product"] == "final_dn"
    assert requests[1].provenance["background_realization_used"] is False
    assert render_calls[0]["source_variability"] is None
    assert render_calls[1]["source_variability"].relative_flux.shape == (1, 60)
    assert render_calls[0]["rng_trace_scope"] == render_calls[1]["rng_trace_scope"]
    assert render_calls[0]["rng_trace_scope"]["science_realization_id"] == (
        "aster-g6-psf6-paired-v1"
    )
