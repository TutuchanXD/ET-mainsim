from __future__ import annotations

import errno
from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest


def _science_policy(*, require_direct_coadd_parity: bool = True):
    from et_mainsim.stamp_science_analysis import StampScienceAnalysisPolicy
    from et_mainsim.stamp_science_photometry import StampSciencePhotometryPolicy

    return StampScienceAnalysisPolicy(
        coadd_factors=(1, 3),
        stream_batch_frames=2,
        direct_coadd_samples_per_shard=1,
        require_direct_coadd_parity=require_direct_coadd_parity,
        photometry=StampSciencePhotometryPolicy(
            background_strategy="delivered_expectation_plus_local_diagnostic",
            cdpp_windows_minutes=(1,),
            minimum_coverage_fraction=1.0,
            minimum_accepted_bins=2,
            training_blocks_per_shard=1,
            training_block_frames=6,
            minimum_training_valid_fraction=1.0,
            background_guard_pixels=1,
            background_border_pixels=1,
            minimum_background_pixels=2,
        ),
    )


def _raw_planes(
    *,
    start: int,
    n_frames: int,
    gain_mode: str = "scalar",
    stamp_shape: tuple[int, int] = (9, 11),
    target_yx: tuple[int, int] = (4, 5),
):
    ny, nx = stamp_shape
    target_y, target_x = target_yx
    raw_start = np.arange(start, start + n_frames, dtype=np.int64)
    q = 1.0 + 0.1 * (raw_start % 4)
    signal = np.zeros((ny, nx), dtype=np.float64)
    signal[target_y, target_x] = 100.0
    signal[target_y, target_x + 1] = 20.0
    calibrated_bgsub = q[:, None, None] * signal[None, :, :]
    background = np.full((n_frames, ny, nx), 5.0, dtype=np.float64)
    bias = np.full(n_frames, 100.0, dtype=np.float64)
    column = np.zeros((n_frames, nx), dtype=np.float64)
    final = np.rint(
        calibrated_bgsub + background + bias[:, None, None]
    ).astype(np.uint16)
    zeros = np.zeros((n_frames, ny, nx), dtype=np.uint16)
    gain: np.ndarray
    if gain_mode == "per_frame":
        gain = np.ones((n_frames, ny, nx), dtype=np.float64)
    elif gain_mode == "stamp_map":
        gain = np.ones((ny, nx), dtype=np.float64)
    else:
        gain = np.asarray(1.0)
    return {
        "final_dn": final,
        "background_expectation_e": background,
        "captured_flux_fraction": np.ones(n_frames),
        "captured_flux_denominator_e": q * 1_000.0,
        "captured_flux_qa_pass": np.ones(n_frames, dtype=bool),
        "bias_level_sum_dn": bias,
        "column_noise_sum_dn_by_x": column,
        "valid_mask": np.ones((n_frames, ny, nx), dtype=bool),
        "fullwell_count": zeros.copy(),
        "adc_low_count": zeros.copy(),
        "adc_high_count": zeros.copy(),
        "cosmic_count": zeros.copy(),
        "time_start_seconds": raw_start.astype(np.float64) * 10.0,
        "exposure_seconds": np.full(n_frames, 10.0),
        "raw_frame_start_index": raw_start,
        "raw_frame_stop_index_exclusive": raw_start + 1,
        "gain_e_per_dn": gain,
        "q": q,
    }


def _coadd_planes(raw: dict[str, np.ndarray], *, factor: int):
    n_raw, ny, nx = raw["final_dn"].shape
    assert n_raw % factor == 0
    n = n_raw // factor

    def grouped_sum(name: str):
        value = raw[name]
        return value.reshape(n, factor, *value.shape[1:]).sum(axis=1)

    starts = raw["raw_frame_start_index"].reshape(n, factor)[:, 0]
    stops = raw["raw_frame_stop_index_exclusive"].reshape(n, factor)[:, -1]
    gain = raw["gain_e_per_dn"]
    if gain.shape == (n_raw, ny, nx):
        gain = gain.reshape(n, factor, ny, nx)[:, 0]
    return {
        "final_dn": grouped_sum("final_dn").astype(np.uint64),
        "background_expectation_e": grouped_sum("background_expectation_e"),
        "captured_flux_fraction": np.ones(n),
        "captured_flux_denominator_e": grouped_sum(
            "captured_flux_denominator_e"
        ),
        "captured_flux_qa_pass": np.all(
            raw["captured_flux_qa_pass"].reshape(n, factor), axis=1
        ),
        "bias_level_sum_dn": grouped_sum("bias_level_sum_dn"),
        "column_noise_sum_dn_by_x": grouped_sum("column_noise_sum_dn_by_x"),
        "valid_mask": np.all(
            raw["valid_mask"].reshape(n, factor, ny, nx), axis=1
        ),
        "fullwell_count": grouped_sum("fullwell_count").astype(np.uint16),
        "adc_low_count": grouped_sum("adc_low_count").astype(np.uint16),
        "adc_high_count": grouped_sum("adc_high_count").astype(np.uint16),
        "cosmic_count": grouped_sum("cosmic_count").astype(np.uint16),
        "time_start_seconds": raw["time_start_seconds"].reshape(n, factor)[:, 0],
        "exposure_seconds": grouped_sum("exposure_seconds"),
        "raw_frame_start_index": starts,
        "raw_frame_stop_index_exclusive": stops,
        "gain_e_per_dn": gain,
    }


def _write_bundle(
    path: Path,
    *,
    planes: dict[str, np.ndarray],
    product_kind: str,
    factor: int,
    shard_id: int,
    identity_marker: str = "same-series",
    science_change: tuple[str, object] | None = None,
    science_case: str = "injected",
    target_source_id: str = "fixture-1",
    run_id: str = "fixture-run",
    production_manifest_identity: dict[str, object] | None = None,
) -> Path:
    from et_mainsim.stamp_delivery import (
        StampDeliveryBundle,
        write_stamp_delivery_bundle,
    )

    science = {
        "target_source_id": target_source_id,
        "case": science_case,
        "simulation_spec_sha256": "spec-a",
        "seed_tree_run_seed": 123,
    }
    if science_change is not None:
        science[science_change[0]] = science_change[1]
    raw_start = int(planes["raw_frame_start_index"][0])
    raw_stop = int(planes["raw_frame_stop_index_exclusive"][-1])
    bundle = StampDeliveryBundle.from_arrays(
        product_kind=product_kind,
        coadd_factor=factor,
        final_dn=planes["final_dn"],
        background_expectation_e=planes["background_expectation_e"],
        captured_flux_fraction=planes["captured_flux_fraction"],
        captured_flux_denominator_e=planes["captured_flux_denominator_e"],
        captured_flux_qa_pass=planes["captured_flux_qa_pass"],
        bias_level_sum_dn=planes["bias_level_sum_dn"],
        column_noise_sum_dn_by_x=planes["column_noise_sum_dn_by_x"],
        valid_mask=planes["valid_mask"],
        fullwell_count=planes["fullwell_count"],
        adc_low_count=planes["adc_low_count"],
        adc_high_count=planes["adc_high_count"],
        cosmic_count=planes["cosmic_count"],
        time_start_seconds=planes["time_start_seconds"],
        exposure_seconds=planes["exposure_seconds"],
        raw_frame_start_index=planes["raw_frame_start_index"],
        raw_frame_stop_index_exclusive=planes[
            "raw_frame_stop_index_exclusive"
        ],
        gain_e_per_dn=planes["gain_e_per_dn"],
        manifest={
            "schema_id": "test.science.production.v1",
            "target_source_id": science["target_source_id"],
            "time_shard": {"shard_id": shard_id},
            "product_kind": product_kind,
            "coadd_factor": factor,
            "caller_manifest": {
                "case": science["case"],
                "run_id": run_id,
                "production_manifest": "production_manifest.json",
                **(
                    {}
                    if production_manifest_identity is None
                    else {
                        "production_manifest_identity": (
                            production_manifest_identity
                        )
                    }
                ),
                "simulation_spec_sha256": science["simulation_spec_sha256"],
                "identity_marker": identity_marker,
                "physical_rng_pairing": {
                    "schema_id": "test.physical_rng_pairing.v1",
                    "seed_tree_run_seed": science["seed_tree_run_seed"],
                    "target_spec_sha256": science["simulation_spec_sha256"],
                    "canonical_context_scope": {
                        "detector_id": "main_rd",
                        "science_realization_id": 0,
                    },
                    "absolute_raw_frame_index": {
                        "absolute_raw_frame_start_index": 0,
                        "formula": (
                            "absolute_raw_frame_start_index + local_frame_index"
                        ),
                        "selected_shard_absolute_frame_interval": {
                            "start_index": raw_start,
                            "stop_index": raw_stop,
                        },
                    },
                    "selected_time_shard": {
                        "shard_id": shard_id,
                        "raw_frame_count": raw_stop - raw_start,
                        "raw_frame_interval": {
                            "start_index": raw_start,
                            "stop_index": raw_stop,
                        },
                    },
                },
            },
        },
        provenance={
            "schema_id": "test.science.production.v1",
            "observation_product": "final_dn",
            "background_realization_used": False,
            "product_kind": product_kind,
            "coadd_factor": factor,
            "caller_provenance": {
                "identity_marker": identity_marker,
                "case": science["case"],
                "simulation_spec_sha256": science["simulation_spec_sha256"],
                "seed_tree_run_seed": science["seed_tree_run_seed"],
                "target_source_id": science["target_source_id"],
            },
        },
    )
    write_stamp_delivery_bundle(path, bundle)
    return path


def _series_fixture(
    tmp_path: Path,
    *,
    second_start: int = 6,
    second_identity: str = "same-series",
    gain_mode: str = "scalar",
    corrupt_coadd: bool = False,
    second_science_change: tuple[str, object] | None = None,
    science_case: str = "injected",
    stamp_shape: tuple[int, int] = (9, 11),
    target_yx: tuple[int, int] = (4, 5),
    target_source_id: str = "fixture-1",
    run_id: str = "fixture-run",
    production_manifest_identity: dict[str, object] | None = None,
    frames_per_shard: int = 6,
    coadd_factors: tuple[int, ...] = (3,),
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    raw_paths: list[Path] = []
    coadd_paths: dict[int, list[Path]] = {factor: [] for factor in coadd_factors}
    q_parts: list[np.ndarray] = []
    for shard_id, start in enumerate((0, second_start)):
        raw = _raw_planes(
            start=start,
            n_frames=frames_per_shard,
            gain_mode=gain_mode,
            stamp_shape=stamp_shape,
            target_yx=target_yx,
        )
        q_parts.append(raw.pop("q"))
        marker = "same-series" if shard_id == 0 else second_identity
        raw_paths.append(
            _write_bundle(
                tmp_path / f"raw_{shard_id}.h5",
                planes=raw,
                product_kind="raw",
                factor=1,
                shard_id=shard_id,
                identity_marker=marker,
                science_change=(
                    second_science_change if shard_id == 1 else None
                ),
                science_case=science_case,
                target_source_id=target_source_id,
                run_id=run_id,
                production_manifest_identity=production_manifest_identity,
            )
        )
        for factor in coadd_factors:
            coadd = _coadd_planes(raw, factor=factor)
            if corrupt_coadd and shard_id == 1 and factor == coadd_factors[0]:
                coadd["final_dn"][0, 4, 5] += np.uint64(1)
            coadd_paths[factor].append(
                _write_bundle(
                    tmp_path / f"coadd{factor}_{shard_id}.h5",
                    planes=coadd,
                    product_kind="coadd",
                    factor=factor,
                    shard_id=shard_id,
                    identity_marker=marker,
                    science_change=(
                        second_science_change if shard_id == 1 else None
                    ),
                    science_case=science_case,
                    target_source_id=target_source_id,
                    run_id=run_id,
                    production_manifest_identity=production_manifest_identity,
                )
            )
    return (
        tuple(raw_paths),
        {factor: tuple(paths) for factor, paths in coadd_paths.items()},
        np.concatenate(q_parts),
    )


def _request(
    tmp_path: Path,
    *,
    raw_paths: tuple[Path, ...],
    coadd_paths: dict[int, tuple[Path, ...]],
    q: np.ndarray,
    output_name: str = "analysis",
    require_direct_coadd_parity: bool = True,
    case: str = "injected",
    aperture_mode: str = "train",
    frozen_aperture=None,
    aperture_source_identity: dict[str, object] | None = None,
):
    from et_mainsim.stamp_science_analysis import StampScienceAnalysisRequest

    return StampScienceAnalysisRequest(
        raw_bundle_paths=raw_paths,
        direct_coadd_bundle_paths=coadd_paths,
        output_dir=tmp_path / output_name,
        raw_relative_flux=q,
        raw_relative_flux_identity={"source": "unit-test-q"},
        code_identity={"git_commit": "unit-test"},
        analysis_context={
            "production_manifest": "unit-test-production-manifest",
            "source_id": "fixture-1",
            "case": case,
        },
        read_noise_e_per_pixel=1.0,
        quantization_noise_e_per_pixel=0.0,
        policy=_science_policy(
            require_direct_coadd_parity=require_direct_coadd_parity
        ),
        aperture_mode=aperture_mode,
        frozen_aperture=frozen_aperture,
        aperture_source_identity=(aperture_source_identity or {}),
    )


def _select_target_pixels(signal, noise, plot=False):
    import torch

    del noise
    assert plot is False
    return signal > 10.0, 20.0


def _frozen_fixture_aperture():
    from et_mainsim.stamp_science_photometry import ScienceApertureDefinition

    aperture = np.zeros((9, 11), dtype=bool)
    aperture[4, 5:7] = True
    background = np.zeros((9, 11), dtype=bool)
    background[1, 1:5] = True
    signal = np.zeros((9, 11), dtype=np.float64)
    signal[4, 5] = 100.0
    signal[4, 6] = 20.0
    return ScienceApertureDefinition(
        aperture_mask=aperture,
        background_mask=background,
        signal_template_e=signal,
        noise_template_e=np.sqrt(signal + 6.0),
        maximum_cumulative_snr=20.0,
        algorithm="unit-test-frozen-oa",
        signal_template_shape=(9, 11),
        target_peak_yx=(4, 5),
        training_raw_frame_indices=np.asarray([0, 6], dtype=np.int64),
        metadata={"source": "injected-paired-analysis"},
    )


def test_static_analysis_requires_unity_q_and_a_reused_injected_aperture(
    tmp_path: Path,
) -> None:
    from et_mainsim.stamp_science_analysis import (
        StampScienceAnalysisContractError,
    )

    raw_paths, coadd_paths, q = _series_fixture(tmp_path)
    with pytest.raises(
        StampScienceAnalysisContractError,
        match="static analysis requires aperture_mode='reuse_published'",
    ):
        _request(
            tmp_path,
            raw_paths=raw_paths,
            coadd_paths=coadd_paths,
            q=np.ones_like(q),
            case="static",
        )
    with pytest.raises(
        StampScienceAnalysisContractError,
        match="static analysis requires an all-unity raw_relative_flux",
    ):
        _request(
            tmp_path,
            raw_paths=raw_paths,
            coadd_paths=coadd_paths,
            q=q,
            case="static",
            aperture_mode="reuse_published",
            frozen_aperture=_frozen_fixture_aperture(),
            aperture_source_identity={"analysis_manifest_sha256": "abc"},
        )


def test_frozen_aperture_background_mask_requirement_follows_strategy() -> None:
    import et_mainsim.stamp_science_analysis as backend

    local = replace(
        _frozen_fixture_aperture(),
        background_mask=None,
        metadata={
            "background_strategy": (
                "delivered_expectation_plus_local_diagnostic"
            )
        },
    )
    with pytest.raises(
        backend.StampScienceAnalysisContractError,
        match="complete mask/template contract",
    ):
        backend._validate_frozen_aperture_definition(local)

    expectation_only = replace(
        local,
        metadata={"background_strategy": "delivered_expectation_only"},
    )
    assert (
        backend._validate_frozen_aperture_definition(expectation_only)
        is expectation_only
    )


def test_static_analysis_reuses_the_frozen_injected_aperture_without_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_paths, coadd_paths, q = _series_fixture(tmp_path)
    import et_mainsim.stamp_science_analysis as backend

    monkeypatch.setattr(
        backend,
        "_train_aperture",
        lambda *_args, **_kwargs: pytest.fail("static must not retrain its OA"),
    )
    frozen = _frozen_fixture_aperture()
    publication = backend.analyze_stamp_science_series_v1(
        _request(
            tmp_path,
            raw_paths=raw_paths,
            coadd_paths=coadd_paths,
            q=np.ones_like(q),
            case="static",
            aperture_mode="reuse_published",
            frozen_aperture=frozen,
            aperture_source_identity={"analysis_manifest_sha256": "abc"},
        )
    )

    with h5py.File(publication.hdf5_path, "r") as handle:
        np.testing.assert_array_equal(
            handle["aperture/aperture_mask"], frozen.aperture_mask
        )
        contract = json.loads(handle["analysis_contract_json"][()].decode())
    assert contract["aperture_mode"] == "reuse_published"
    assert contract["aperture_source_identity"] == {
        "analysis_manifest_sha256": "abc"
    }


def test_multishard_training_uses_online_sufficient_statistics_and_matches_formula(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import photsim7.aperture as legacy_aperture

    monkeypatch.setattr(
        legacy_aperture,
        "maximize_cumulative_snr",
        _select_target_pixels,
    )
    raw_paths, coadd_paths, q = _series_fixture(tmp_path)
    import et_mainsim.stamp_science_analysis as backend
    from et_mainsim.stamp_science_photometry import (
        train_science_optimal_aperture_v1,
    )

    request = _request(
        tmp_path,
        raw_paths=raw_paths,
        coadd_paths=coadd_paths,
        q=q,
    )
    headers = backend._read_series_headers(
        request.raw_bundle_paths,
        product_kind="raw",
        coadd_factor=1,
    )

    # Build the old in-memory formula only as a tiny test oracle.
    reference_batches = []
    import h5py

    for header in headers:
        with h5py.File(header.formal.path, "r") as handle:
            reference_batches.append(
                backend._read_delivery_batch(
                    handle,
                    header,
                    slice(0, header.formal.frame_count),
                )
            )
    reference_batch = backend._concatenate_batches(
        reference_batches,
        require_contiguous=True,
    )
    expected = train_science_optimal_aperture_v1(
        reference_batch.to_photometry_input(),
        raw_relative_flux=q,
        training_raw_frame_indices=np.arange(12, dtype=np.int64),
        read_noise_e_per_pixel=request.read_noise_e_per_pixel,
        quantization_noise_e_per_pixel=request.quantization_noise_e_per_pixel,
        policy=request.policy.photometry,
    )

    monkeypatch.setattr(
        backend,
        "_concatenate_batches",
        lambda *_args, **_kwargs: pytest.fail(
            "training must not retain/concatenate sampled image cubes"
        ),
    )
    actual = backend._train_aperture(
        headers,
        raw_relative_flux=q,
        first_raw_index=0,
        request=request,
    )

    np.testing.assert_allclose(actual.signal_template_e, expected.signal_template_e)
    np.testing.assert_allclose(actual.noise_template_e, expected.noise_template_e)
    np.testing.assert_array_equal(actual.aperture_mask, expected.aperture_mask)
    np.testing.assert_array_equal(actual.background_mask, expected.background_mask)
    np.testing.assert_array_equal(
        actual.training_raw_frame_indices,
        expected.training_raw_frame_indices,
    )
    assert actual.metadata["training_accumulator"] == (
        "online_per_pixel_sufficient_statistics_v1"
    )


@pytest.mark.parametrize(
    ("production_schema_id", "production_schema_version"),
    [
        ("et_mainsim.science_stamp_production.v1", 1),
        ("et_mainsim.galaxy_stamp_production.v1", 2),
        ("et_mainsim.galaxy_stamp_production.v1", 3),
    ],
)
def test_identity_bound_request_cli_loads_injected_q_for_new_and_galaxy_manifests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    production_schema_id: str,
    production_schema_version: int,
) -> None:
    raw_paths, coadd_paths, q = _series_fixture(tmp_path)
    import et_mainsim.stamp_science_analysis as backend

    production_manifest = tmp_path / "production_manifest.json"
    production_manifest.write_text(
        json.dumps(
            {
                "schema_id": production_schema_id,
                "schema_version": production_schema_version,
                "run_id": "fixture-run",
            }
        ),
        encoding="utf-8",
    )
    snapshot = tmp_path / "factor_snapshot.npz"
    np.savez(
        snapshot,
        source_id=np.asarray("fixture-1"),
        factors=q,
        metadata_json=np.asarray("{}"),
    )
    output = tmp_path / "cli-output"
    payload = {
        "schema_id": backend.STAMP_SCIENCE_ANALYSIS_REQUEST_SCHEMA_ID,
        "schema_version": backend.STAMP_SCIENCE_ANALYSIS_REQUEST_SCHEMA_VERSION,
        "production_manifest": backend._cli_file_binding(production_manifest),
        "source_id": "fixture-1",
        "case": "injected",
        "input_discovery": {"mode": "explicit_identity_bound_paths_v1"},
        "raw_bundles": [backend._cli_bundle_binding(path) for path in raw_paths],
        "coadd_bundles": {
            str(factor): [backend._cli_bundle_binding(path) for path in paths]
            for factor, paths in coadd_paths.items()
        },
        "q": {
            "mode": "factor_snapshot_npz",
            "snapshot": backend._cli_file_binding(snapshot),
        },
        "aperture": {"mode": "train"},
        "output_dir": str(output),
        "read_noise_e_per_pixel": 1.0,
        "quantization_noise_e_per_pixel": 0.0,
        "policy": _science_policy().to_dict(),
        "code_identity": {"git_commit": "unit-test"},
    }
    request_path = tmp_path / "analysis_request.json"
    request_path.write_text(json.dumps(payload), encoding="utf-8")
    captured = []

    def capture(request):
        captured.append(request)
        return SimpleNamespace(output_dir=Path(request.output_dir))

    monkeypatch.setattr(backend, "analyze_stamp_science_product_set_v1", capture)
    with pytest.raises(
        backend.StampScienceAnalysisContractError,
        match="formal_profile_id",
    ):
        backend.main(["run", "--request", str(request_path)])
    assert captured == []
    assert capsys.readouterr().out == ""


def test_request_cli_rejects_bound_production_manifest_identity_drift(
    tmp_path: Path,
) -> None:
    raw_paths, coadd_paths, q = _series_fixture(tmp_path)
    import et_mainsim.stamp_science_analysis as backend

    production_manifest = tmp_path / "production_manifest.json"
    production_manifest.write_text(
        json.dumps(
            {
                "schema_id": "et_mainsim.science_stamp_production.v1",
                "schema_version": 1,
                "run_id": "fixture-run",
            }
        ),
        encoding="utf-8",
    )
    binding = backend._cli_file_binding(production_manifest)
    production_manifest.write_text(
        production_manifest.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    snapshot = tmp_path / "factor_snapshot.npz"
    np.savez(snapshot, source_id=np.asarray("fixture-1"), factors=q)
    payload = {
        "schema_id": backend.STAMP_SCIENCE_ANALYSIS_REQUEST_SCHEMA_ID,
        "schema_version": backend.STAMP_SCIENCE_ANALYSIS_REQUEST_SCHEMA_VERSION,
        "formal_profile_id": backend.STAMP_SCIENCE_FORMAL_PROFILE_ID,
        "production_manifest": binding,
        "source_identity": {
            "production_track": "varlc",
            "namespace": "varlc",
            "external_source_id": "fixture-1",
            "source_id": "fixture-1",
        },
        "source_id": "fixture-1",
        "case": "injected",
        "input_discovery": {"mode": "explicit_identity_bound_paths_v1"},
        "raw_bundles": [backend._cli_bundle_binding(path) for path in raw_paths],
        "coadd_bundles": {
            str(factor): [backend._cli_bundle_binding(path) for path in paths]
            for factor, paths in coadd_paths.items()
        },
        "q": {
            "mode": "factor_snapshot_npz",
            "snapshot": backend._cli_file_binding(snapshot),
        },
        "aperture": {"mode": "train"},
        "output_dir": str(tmp_path / "cli-output"),
        "read_noise_e_per_pixel": 1.0,
        "quantization_noise_e_per_pixel": 0.0,
        "noise_model": {},
        "policy": _science_policy().to_dict(),
        "code_identity": {"git_commit": "unit-test"},
    }
    request_path = tmp_path / "analysis_request.json"
    request_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        backend.StampScienceAnalysisContractError,
        match="production_manifest identity/path drift",
    ):
        backend.load_stamp_science_analysis_request_v1(request_path)


def test_static_cli_loads_unity_q_and_reuses_paired_injected_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import photsim7.aperture as legacy_aperture

    monkeypatch.setattr(
        legacy_aperture,
        "maximize_cumulative_snr",
        _select_target_pixels,
    )
    injected_raw, injected_coadd, injected_q = _series_fixture(
        tmp_path / "injected-input",
        science_case="injected",
    )
    import et_mainsim.stamp_science_analysis as backend

    injected_publication = backend.analyze_stamp_science_series_v1(
        _request(
            tmp_path,
            raw_paths=injected_raw,
            coadd_paths=injected_coadd,
            q=injected_q,
            output_name="paired-injected-analysis",
            case="injected",
        )
    )
    static_raw, static_coadd, _ = _series_fixture(
        tmp_path / "static-input",
        science_case="static",
    )
    production_manifest = tmp_path / "production_manifest.json"
    production_manifest.write_text(
        json.dumps(
            {
                "schema_id": "et_mainsim.science_stamp_production.v1",
                "schema_version": 1,
                "run_id": "fixture-run",
            }
        ),
        encoding="utf-8",
    )
    payload = {
        "schema_id": backend.STAMP_SCIENCE_ANALYSIS_REQUEST_SCHEMA_ID,
        "schema_version": backend.STAMP_SCIENCE_ANALYSIS_REQUEST_SCHEMA_VERSION,
        "production_manifest": backend._cli_file_binding(production_manifest),
        "source_id": "fixture-1",
        "case": "static",
        "raw_bundles": [backend._cli_bundle_binding(path) for path in static_raw],
        "coadd_bundles": {
            str(factor): [backend._cli_bundle_binding(path) for path in paths]
            for factor, paths in static_coadd.items()
        },
        "q": {"mode": "unity"},
        "aperture": {
            "mode": "reuse_published",
            "analysis_manifest": backend._cli_file_binding(
                injected_publication.manifest_path
            ),
        },
        "output_dir": str(tmp_path / "paired-static-analysis"),
        "read_noise_e_per_pixel": 1.0,
        "quantization_noise_e_per_pixel": 0.0,
        "policy": _science_policy().to_dict(),
        "code_identity": {"git_commit": "unit-test"},
    }
    request_path = tmp_path / "static_analysis_request.json"
    request_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        backend.StampScienceAnalysisContractError,
        match="formal_profile_id",
    ):
        backend.load_stamp_science_analysis_request_v1(request_path)


def test_formal_series_analysis_streams_bounded_slices_and_publishes_all_products(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import photsim7.aperture as legacy_aperture

    monkeypatch.setattr(
        legacy_aperture,
        "maximize_cumulative_snr",
        _select_target_pixels,
    )
    raw_paths, coadd_paths, q = _series_fixture(tmp_path)

    import et_mainsim.stamp_science_analysis as backend

    original = backend._read_delivery_batch
    observed_slices: list[tuple[int, int]] = []

    def bounded_reader(handle, header, frame_slice):
        assert frame_slice.step in (None, 1)
        assert frame_slice.start is not None and frame_slice.stop is not None
        observed_slices.append((frame_slice.start, frame_slice.stop))
        return original(handle, header, frame_slice)

    monkeypatch.setattr(backend, "_read_delivery_batch", bounded_reader)
    publication = backend.analyze_stamp_science_series_v1(
        _request(
            tmp_path,
            raw_paths=raw_paths,
            coadd_paths=coadd_paths,
            q=q,
        )
    )

    assert publication.output_dir == (tmp_path / "analysis").resolve()
    assert observed_slices
    assert max(stop - start for start, stop in observed_slices) <= 6
    assert publication.hdf5_path.is_file()
    assert publication.ecsv_path.is_file()
    assert publication.manifest_path.is_file()
    assert publication.aperture_mask_path.is_file()
    assert publication.background_mask_path.is_file()
    assert publication.representative_frames_path.is_file()

    manifest = json.loads(publication.manifest_path.read_text(encoding="utf-8"))
    assert manifest["complete"] is True
    assert manifest["ready"] is True
    assert manifest["contract"]["observation_product"] == "final_dn"
    assert manifest["contract"]["background_realization_used"] is False
    assert manifest["contract"]["code_identity"]["git_commit"] == "unit-test"
    assert manifest["contract"]["raw_relative_flux"]["source_identity"] == {
        "source": "unit-test-q"
    }
    assert len(manifest["contract"]["input_raw_shards"]) == 2
    assert {
        item["byte_identity"]["trust_scope"]
        for item in manifest["contract"]["input_raw_shards"]
    } == {"locally_computed_full_file_sha256_v1"}
    assert {
        item["byte_identity"]["trust_scope"]
        for items in manifest["contract"]["input_direct_coadd_shards"].values()
        for item in items
    } == {"locally_computed_full_file_sha256_v1"}
    assert manifest["contract"]["direct_coadd_parity"]["passed"] is True
    assert set(manifest["artifacts"]) >= {
        "photometry.h5",
        "photometry.ecsv",
        "aperture_definition.json",
        "cdpp.json",
        "aperture_mask.npy",
        "background_mask.npy",
        "representative_calibrated_frames.h5",
    }

    with h5py.File(publication.hdf5_path, "r") as handle:
        assert bool(handle.attrs["complete"]) is True
        assert set(handle["cadences"]) == {"10s", "30s"}
        raw = handle["cadences/10s"]
        coadd = handle["cadences/30s"]
        assert raw["time_start_seconds"].shape == (12,)
        assert coadd["time_start_seconds"].shape == (4,)
        np.testing.assert_array_equal(
            coadd["raw_frame_start_index"], [0, 3, 6, 9]
        )
        np.testing.assert_array_equal(
            coadd["raw_frame_stop_index_exclusive"], [3, 6, 9, 12]
        )
        np.testing.assert_allclose(
            raw["flux_expectation_bgsub_e"], 120.0 * q
        )
        np.testing.assert_allclose(
            raw["background_expectation_aperture_e"],
            np.full(12, 10.0),
        )
        np.testing.assert_allclose(
            raw["source_variance_e2"],
            raw["fitted_flux_expectation_e"],
        )
        np.testing.assert_allclose(raw["background_variance_e2"], 10.0)
        np.testing.assert_allclose(raw["read_variance_e2"], 2.0)
        np.testing.assert_allclose(raw["quantization_variance_e2"], 0.0)
        np.testing.assert_allclose(
            raw["flux_uncertainty_e"],
            np.sqrt(
                np.asarray(raw["source_variance_e2"])
                + np.asarray(raw["background_variance_e2"])
                + np.asarray(raw["read_variance_e2"])
                + np.asarray(raw["quantization_variance_e2"])
            ),
        )
        assert np.nanmax(np.abs(raw["residual_expectation_ppm"])) < 1e-8
        np.testing.assert_allclose(handle["raw_relative_flux"], q)
        contract = json.loads(handle["analysis_contract_json"][()].decode())
        assert contract == manifest["contract"]

    with h5py.File(publication.representative_frames_path, "r") as handle:
        assert bool(handle.attrs["complete"]) is True
        assert [item.decode() for item in handle["selection_role"]] == [
            "first_clean",
            "middle_clean",
            "last_clean",
        ]
        np.testing.assert_array_equal(
            handle["raw_frame_start_index"], [0, 6, 11]
        )
        np.testing.assert_allclose(
            handle["calibrated_bgsub_e"],
            np.asarray(handle["calibrated_e"])
            - np.asarray(handle["background_expectation_e"]),
        )
        assert handle["final_dn"].dtype.kind == "u"

    validation = backend.validate_stamp_science_analysis_v1(
        publication.output_dir
    )
    assert validation.complete is True
    assert validation.cadence_seconds == (10, 30)


@pytest.mark.parametrize(
    ("product", "frame_index"),
    (("raw", 1), ("direct_coadd", -1)),
)
def test_analysis_fails_closed_before_publication_when_any_input_capture_qa_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    product: str,
    frame_index: int,
) -> None:
    """A false raw or unsampled direct-coadd capture gate blocks publication."""

    import photsim7.aperture as legacy_aperture

    monkeypatch.setattr(
        legacy_aperture,
        "maximize_cumulative_snr",
        _select_target_pixels,
    )
    raw_paths, coadd_paths, q = _series_fixture(tmp_path / "inputs")
    tampered = raw_paths[0] if product == "raw" else coadd_paths[3][0]
    with h5py.File(tampered, "r+") as handle:
        handle["captured_flux_qa_pass"][frame_index] = False

    import et_mainsim.stamp_science_analysis as backend

    output_dir = tmp_path / f"blocked-{product}"
    with pytest.raises(
        backend.StampScienceAnalysisContractError,
        match="captured_flux_qa_pass.*false",
    ):
        backend.analyze_stamp_science_series_v1(
            _request(
                tmp_path,
                raw_paths=raw_paths,
                coadd_paths=coadd_paths,
                q=q,
                output_name=output_dir.name,
            )
        )

    assert not output_dir.exists()
    assert not tuple(tmp_path.glob(f".{output_dir.name}.*.partial"))


def test_27_by_27_expectation_only_analysis_publishes_optimal_aperture_and_capture_qa(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compact formal products must not depend on stamp-local background pixels."""

    import photsim7.aperture as legacy_aperture

    monkeypatch.setattr(
        legacy_aperture,
        "maximize_cumulative_snr",
        _select_target_pixels,
    )
    raw_paths, coadd_paths, q = _series_fixture(
        tmp_path,
        stamp_shape=(27, 27),
        target_yx=(13, 13),
    )
    request = _request(
        tmp_path,
        raw_paths=raw_paths,
        coadd_paths=coadd_paths,
        q=q,
        output_name="compact-analysis",
    )
    request = replace(
        request,
        policy=replace(
            request.policy,
            photometry=replace(
                request.policy.photometry,
                background_strategy="delivered_expectation_only",
            ),
        ),
    )

    import et_mainsim.stamp_science_analysis as backend

    publication = backend.analyze_stamp_science_series_v1(request)
    manifest = json.loads(publication.manifest_path.read_text(encoding="utf-8"))

    assert manifest["contract"]["default_background_product"] == (
        "background_expectation_e"
    )
    assert manifest["contract"]["background_products"] == [
        "expectation_background_subtracted"
    ]
    assert manifest["contract"]["captured_flux_qa"]["cadences"]["10s"] == {
        "all_pass": True,
        "minimum_fraction": 1.0,
    }
    assert not np.any(np.load(publication.background_mask_path, allow_pickle=False))
    with h5py.File(publication.hdf5_path, "r") as handle:
        raw = handle["cadences/10s"]
        assert raw["captured_flux_fraction"].shape == (12,)
        np.testing.assert_allclose(raw["captured_flux_fraction"], 1.0)
        np.testing.assert_array_equal(raw["captured_flux_qa_pass"], True)
        assert np.all(np.isnan(raw["flux_local_bgsub_e"]))
        assert np.all(np.isfinite(raw["flux_expectation_bgsub_e"]))


def test_expectation_only_injected_aperture_round_trips_into_static_product_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The formal static path must preserve the injected no-local-BG contract."""

    import photsim7.aperture as legacy_aperture

    monkeypatch.setattr(
        legacy_aperture,
        "maximize_cumulative_snr",
        _select_target_pixels,
    )
    import et_mainsim.stamp_science_analysis as backend

    run_root = tmp_path / "run"
    production_manifest = run_root / "production_manifest.json"
    production_manifest.parent.mkdir(parents=True)
    production_manifest.write_text(
        json.dumps(
            {
                "schema_id": "et_mainsim.galaxy_stamp_production.v1",
                "schema_version": 3,
                "run_id": "fixture-run",
            }
        ),
        encoding="utf-8",
    )
    production_binding = backend._cli_file_binding(production_manifest)
    source_identity = {
        "production_track": "galaxy",
        "namespace": "gaia_dr3",
        "external_source_id": "42",
        "source_id": "42",
    }
    production = SimpleNamespace(
        manifest_path=production_manifest.resolve(),
        manifest_binding=production_binding,
        manifest={},
        run_id="fixture-run",
        source_identity=source_identity,
        target={},
        factor_snapshot_path=run_root / "inputs" / "unused-factor.npz",
        factor_snapshot_binding={},
        read_noise_e_per_raw_pixel=1.0,
        quantization_noise_e_per_raw_pixel=0.0,
    )
    production_identity = production_binding["identity"]
    injected_raw, injected_coadd, injected_q = _series_fixture(
        tmp_path / "injected-input",
        second_start=120,
        stamp_shape=(21, 23),
        target_yx=(10, 11),
        target_source_id="42",
        run_id="fixture-run",
        production_manifest_identity=production_identity,
        frames_per_shard=120,
        coadd_factors=(3, 6, 12, 30),
    )
    static_raw, static_coadd, _ = _series_fixture(
        tmp_path / "static-input",
        second_start=120,
        science_case="static",
        stamp_shape=(21, 23),
        target_yx=(10, 11),
        target_source_id="42",
        run_id="fixture-run",
        production_manifest_identity=production_identity,
        frames_per_shard=120,
        coadd_factors=(3, 6, 12, 30),
    )
    formal_policy = backend.StampScienceAnalysisPolicy()
    code_identity = {
        "schema_id": "et_mainsim.formal_analysis_code_identity.v1",
        "schema_version": 1,
        "provenance": {
            "et_mainsim": {
                "commit": "a" * 40,
                "dirty": False,
                "version": "1",
            },
            "photsim7": {
                "commit": "b" * 40,
                "dirty": False,
                "version": "1",
            },
            "runtime": {"python": "3.13.0"},
        },
        "analysis_dependencies": {},
    }
    injected_request = backend.StampScienceAnalysisRequest(
        raw_bundle_paths=injected_raw,
        direct_coadd_bundle_paths=injected_coadd,
        output_dir=tmp_path / "paired-injected-product-set",
        raw_relative_flux=injected_q,
        raw_relative_flux_identity={"source": "unit-test-q"},
        read_noise_e_per_pixel=1.0,
        quantization_noise_e_per_pixel=0.0,
        policy=formal_policy,
        code_identity=code_identity,
        analysis_context={
            "production_manifest": production_binding,
            "production_track": "galaxy",
            "source_identity": source_identity,
            "source_id": "42",
            "case": "injected",
        },
    )
    monkeypatch.setattr(
        backend,
        "collect_formal_analysis_code_identity_v1",
        lambda: code_identity,
    )
    injected = backend.analyze_stamp_science_product_set_v1(injected_request)

    discovery = backend.StampScienceAnalysisBundleDiscovery(
        raw_bundle_paths=static_raw,
        direct_coadd_bundle_paths=static_coadd,
        shard_ids=(0, 1),
        time_plan_identity={"size_bytes": 1, "sha256": "a" * 64},
        static_task_list_binding={
            "path": str(run_root / "inputs" / "static_representative.json"),
            "identity": {"size_bytes": 1, "sha256": "b" * 64},
        },
    )
    monkeypatch.setattr(
        backend,
        "_resolve_production_source_v1",
        lambda *_args, **_kwargs: production,
    )
    monkeypatch.setattr(
        backend,
        "discover_stamp_science_analysis_bundles_v1",
        lambda *_args, **_kwargs: discovery,
    )
    request_path = tmp_path / "static-request.json"
    backend.write_stamp_science_analysis_request_v1(
        request_path,
        production_manifest=production_manifest,
        source_id="42",
        case="static",
        output_dir=tmp_path / "paired-static-product-set",
        aperture_analysis_manifest=(
            injected.science_optimal_aperture.manifest_path
        ),
    )

    static_request = backend.load_stamp_science_analysis_request_v1(request_path)
    assert static_request.frozen_aperture is not None
    assert static_request.frozen_aperture.background_mask is None
    static = backend.analyze_stamp_science_product_set_v1(static_request)
    validation = backend.validate_stamp_science_analysis_product_set_v1(
        static.output_dir
    )
    assert validation.complete is True
    with h5py.File(static.science_optimal_aperture.hdf5_path, "r") as handle:
        assert not np.any(handle["aperture/background_mask"])


def test_input_hdf_byte_identity_prefers_a_complete_staged_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import photsim7.aperture as legacy_aperture

    monkeypatch.setattr(
        legacy_aperture,
        "maximize_cumulative_snr",
        _select_target_pixels,
    )
    raw_paths, coadd_paths, q = _series_fixture(tmp_path)
    members = {}
    for path in (*raw_paths, *coadd_paths[3]):
        raw = path.read_bytes()
        members[path.name] = {
            "path_relative_to_run_root": path.name,
            "size_bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    (tmp_path / "publication_receipt.json").write_text(
        json.dumps(
            {
                "schema_id": "et_mainsim.stamp_shard_publication_receipt.v1",
                "schema_version": 1,
                "complete": True,
                "run_id": "fixture-run",
                "case": "injected",
                "target_source_id_int64": "fixture-1",
                "shard": {},
                "production_manifest": {
                    "path_relative_to_run_root": "production_manifest.json",
                    "size_bytes": 2,
                    "sha256": "0" * 64,
                },
                "members": members,
            }
        ),
        encoding="utf-8",
    )
    import et_mainsim.stamp_science_analysis as backend

    publication = backend.analyze_stamp_science_series_v1(
        _request(
            tmp_path,
            raw_paths=raw_paths,
            coadd_paths=coadd_paths,
            q=q,
        )
    )

    contract = json.loads(publication.manifest_path.read_text())["contract"]
    byte_identities = [
        item["byte_identity"] for item in contract["input_raw_shards"]
    ] + [
        item["byte_identity"]
        for items in contract["input_direct_coadd_shards"].values()
        for item in items
    ]
    assert {item["trust_scope"] for item in byte_identities} == {
        "publisher_receipt_plus_stat_and_formal_header_v1"
    }
    assert all("publication_receipt" in item for item in byte_identities)


def test_product_set_publishes_reference_fixed13_and_science_oa_from_one_raw_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import photsim7.aperture as legacy_aperture

    monkeypatch.setattr(
        legacy_aperture,
        "maximize_cumulative_snr",
        _select_target_pixels,
    )
    raw_paths, coadd_paths, q = _series_fixture(
        tmp_path / "inputs",
        stamp_shape=(21, 23),
        target_yx=(10, 11),
    )
    import et_mainsim.stamp_science_analysis as backend

    raw_stream_entries = 0
    original_stream = backend._stream_raw_product_analyses

    def count_raw_stream(*args, **kwargs):
        nonlocal raw_stream_entries
        raw_stream_entries += 1
        return original_stream(*args, **kwargs)

    monkeypatch.setattr(
        backend,
        "_stream_raw_product_analyses",
        count_raw_stream,
    )
    publication = backend.analyze_stamp_science_product_set_v1(
        _request(
            tmp_path,
            raw_paths=raw_paths,
            coadd_paths=coadd_paths,
            q=q,
            output_name="analysis-products",
        )
    )

    assert raw_stream_entries == 1
    assert publication.output_dir == (tmp_path / "analysis-products").resolve()
    assert publication.reference_fixed13.output_dir == (
        publication.output_dir / "reference_fixed13_v1"
    )
    assert publication.science_optimal_aperture.output_dir == (
        publication.output_dir / "science_optimal_aperture_v1"
    )
    reference_mask = np.load(
        publication.reference_fixed13.aperture_mask_path,
        allow_pickle=False,
    )
    assert reference_mask.shape == (21, 23)
    assert np.count_nonzero(reference_mask) == 169
    np.testing.assert_array_equal(reference_mask[4:17, 5:18], True)
    science_mask = np.load(
        publication.science_optimal_aperture.aperture_mask_path,
        allow_pickle=False,
    )
    assert np.count_nonzero(science_mask) == 2

    for product_name, product in (
        ("reference_fixed13_v1", publication.reference_fixed13),
        ("science_optimal_aperture_v1", publication.science_optimal_aperture),
    ):
        manifest = json.loads(product.manifest_path.read_text(encoding="utf-8"))
        assert manifest["contract"]["analysis_product"] == product_name
        with h5py.File(product.hdf5_path, "r") as handle:
            assert "flux_uncertainty_e" in handle["cadences/10s"]


@pytest.mark.parametrize(
    ("fixture_kwargs", "message"),
    [
        ({"second_identity": "different-series"}, "incompatible shard identities"),
        ({"second_start": 7}, "not globally continuous"),
        ({"gain_mode": "per_frame"}, "per-frame gain"),
    ],
)
def test_formal_series_analysis_fails_closed_on_identity_gap_or_per_frame_gain(
    tmp_path: Path,
    fixture_kwargs: dict[str, object],
    message: str,
) -> None:
    from et_mainsim.stamp_science_analysis import (
        StampScienceAnalysisContractError,
        analyze_stamp_science_series_v1,
    )

    raw_paths, coadd_paths, q = _series_fixture(tmp_path, **fixture_kwargs)
    with pytest.raises(StampScienceAnalysisContractError, match=message):
        analyze_stamp_science_series_v1(
            _request(
                tmp_path,
                raw_paths=raw_paths,
                coadd_paths=coadd_paths,
                q=q,
            )
        )
    assert not (tmp_path / "analysis").exists()


@pytest.mark.parametrize(
    "science_change",
    [
        ("target_source_id", "fixture-2"),
        ("simulation_spec_sha256", "spec-b"),
        ("seed_tree_run_seed", 124),
        ("case", "static"),
    ],
)
def test_cross_shard_canonical_identity_retains_every_science_defining_field(
    tmp_path: Path,
    science_change: tuple[str, object],
) -> None:
    from et_mainsim.stamp_science_analysis import (
        StampScienceAnalysisContractError,
        analyze_stamp_science_series_v1,
    )

    raw_paths, coadd_paths, q = _series_fixture(
        tmp_path,
        second_science_change=science_change,
    )
    with pytest.raises(
        StampScienceAnalysisContractError,
        match="incompatible shard identities",
    ):
        analyze_stamp_science_series_v1(
            _request(
                tmp_path,
                raw_paths=raw_paths,
                coadd_paths=coadd_paths,
                q=q,
            )
        )


def test_formal_series_analysis_detects_raw_derived_direct_coadd_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import photsim7.aperture as legacy_aperture

    monkeypatch.setattr(
        legacy_aperture,
        "maximize_cumulative_snr",
        _select_target_pixels,
    )
    raw_paths, coadd_paths, q = _series_fixture(tmp_path, corrupt_coadd=True)
    from et_mainsim.stamp_science_analysis import (
        StampScienceAnalysisContractError,
        analyze_stamp_science_series_v1,
    )

    with pytest.raises(
        StampScienceAnalysisContractError,
        match="raw-derived/direct-coadd parity",
    ):
        analyze_stamp_science_series_v1(
            _request(
                tmp_path,
                raw_paths=raw_paths,
                coadd_paths=coadd_paths,
                q=q,
            )
        )
    assert not (tmp_path / "analysis").exists()


def test_formal_series_analysis_accepts_one_static_stamp_gain_map(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import photsim7.aperture as legacy_aperture

    monkeypatch.setattr(
        legacy_aperture,
        "maximize_cumulative_snr",
        _select_target_pixels,
    )
    raw_paths, coadd_paths, q = _series_fixture(tmp_path, gain_mode="stamp_map")
    from et_mainsim.stamp_science_analysis import analyze_stamp_science_series_v1

    publication = analyze_stamp_science_series_v1(
        _request(
            tmp_path,
            raw_paths=raw_paths,
            coadd_paths=coadd_paths,
            q=q,
        )
    )

    assert publication.hdf5_path.is_file()


def test_published_analysis_validator_rejects_a_tampered_portable_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import photsim7.aperture as legacy_aperture

    monkeypatch.setattr(
        legacy_aperture,
        "maximize_cumulative_snr",
        _select_target_pixels,
    )
    raw_paths, coadd_paths, q = _series_fixture(tmp_path)
    import et_mainsim.stamp_science_analysis as backend

    publication = backend.analyze_stamp_science_series_v1(
        _request(
            tmp_path,
            raw_paths=raw_paths,
            coadd_paths=coadd_paths,
            q=q,
        )
    )
    with publication.ecsv_path.open("ab") as stream:
        stream.write(b"# tampered\n")

    with pytest.raises(
        backend.StampScienceAnalysisContractError,
        match="artifact hash/readback mismatch",
    ):
        backend.validate_stamp_science_analysis_v1(publication.output_dir)


def test_published_analysis_readback_rejects_false_captured_flux_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Even a re-hashed HDF5 cannot publish a cadence that failed capture QA."""

    import photsim7.aperture as legacy_aperture

    monkeypatch.setattr(
        legacy_aperture,
        "maximize_cumulative_snr",
        _select_target_pixels,
    )
    raw_paths, coadd_paths, q = _series_fixture(tmp_path)
    import et_mainsim.stamp_science_analysis as backend

    publication = backend.analyze_stamp_science_series_v1(
        _request(
            tmp_path,
            raw_paths=raw_paths,
            coadd_paths=coadd_paths,
            q=q,
        )
    )
    with h5py.File(publication.hdf5_path, "r+") as handle:
        handle["cadences/10s/captured_flux_qa_pass"][0] = False

    manifest = json.loads(publication.manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["photometry.h5"] = backend._file_identity(
        publication.hdf5_path
    )
    publication.manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        backend.StampScienceAnalysisContractError,
        match="authoritative HDF5 cadence 10s capture QA did not pass",
    ):
        backend.validate_stamp_science_analysis_v1(publication.output_dir)


def test_published_analysis_readback_rejects_legacy_photometry_table_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A re-hashed v1 portable table cannot masquerade as the v2 layout."""

    import photsim7.aperture as legacy_aperture
    from astropy.table import Table

    monkeypatch.setattr(
        legacy_aperture,
        "maximize_cumulative_snr",
        _select_target_pixels,
    )
    raw_paths, coadd_paths, q = _series_fixture(tmp_path)
    import et_mainsim.stamp_science_analysis as backend

    publication = backend.analyze_stamp_science_series_v1(
        _request(
            tmp_path,
            raw_paths=raw_paths,
            coadd_paths=coadd_paths,
            q=q,
        )
    )
    table = Table.read(publication.ecsv_path, format="ascii.ecsv")
    table.meta["schema_id"] = "et_mainsim.stamp_science_photometry_table.v1"
    table.meta["schema_version"] = 1
    table.write(publication.ecsv_path, format="ascii.ecsv", overwrite=True)

    manifest = json.loads(publication.manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["photometry.ecsv"] = backend._file_identity(
        publication.ecsv_path
    )
    publication.manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        backend.StampScienceAnalysisContractError,
        match="portable photometry ECSV v2 schema is invalid",
    ):
        backend.validate_stamp_science_analysis_v1(publication.output_dir)


def test_validator_rejects_reference_curve_that_disagrees_with_authoritative_hdf5(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import photsim7.aperture as legacy_aperture
    from astropy.table import Table

    monkeypatch.setattr(
        legacy_aperture,
        "maximize_cumulative_snr",
        _select_target_pixels,
    )
    raw_paths, coadd_paths, q = _series_fixture(tmp_path)
    import et_mainsim.stamp_science_analysis as backend

    publication = backend.analyze_stamp_science_series_v1(
        _request(
            tmp_path,
            raw_paths=raw_paths,
            coadd_paths=coadd_paths,
            q=q,
        )
    )
    reference_path = publication.output_dir / "reference_lightcurve.ecsv"
    reference = Table.read(reference_path, format="ascii.ecsv")
    reference["flux_expectation_bgsub_e"][0] += 1.0
    reference.write(reference_path, format="ascii.ecsv", overwrite=True)

    manifest = json.loads(
        publication.manifest_path.read_text(encoding="utf-8")
    )
    manifest["artifacts"]["reference_lightcurve.ecsv"] = backend._file_identity(
        reference_path
    )
    publication.manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        backend.StampScienceAnalysisContractError,
        match=(
            "reference-lightcurve ECSV column differs from HDF5: "
            "flux_expectation_bgsub_e"
        ),
    ):
        backend.validate_stamp_science_analysis_v1(publication.output_dir)


def test_analysis_publication_readback_failure_is_atomic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import photsim7.aperture as legacy_aperture

    monkeypatch.setattr(
        legacy_aperture,
        "maximize_cumulative_snr",
        _select_target_pixels,
    )
    raw_paths, coadd_paths, q = _series_fixture(tmp_path)
    import et_mainsim.stamp_science_analysis as backend

    def fail_readback(_path):
        raise backend.StampScienceAnalysisContractError("forced readback failure")

    monkeypatch.setattr(backend, "_validate_staged_analysis_v1", fail_readback)
    with pytest.raises(
        backend.StampScienceAnalysisContractError,
        match="forced readback failure",
    ):
        backend.analyze_stamp_science_series_v1(
            _request(
                tmp_path,
                raw_paths=raw_paths,
                coadd_paths=coadd_paths,
                q=q,
            )
        )

    assert not (tmp_path / "analysis").exists()
    assert not list(tmp_path.glob(".analysis.*.partial"))
    assert not (tmp_path / ".analysis.lock").exists()


def test_complete_analysis_is_never_overwritten(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import photsim7.aperture as legacy_aperture

    monkeypatch.setattr(
        legacy_aperture,
        "maximize_cumulative_snr",
        _select_target_pixels,
    )
    raw_paths, coadd_paths, q = _series_fixture(tmp_path)
    import et_mainsim.stamp_science_analysis as backend

    request = _request(
        tmp_path,
        raw_paths=raw_paths,
        coadd_paths=coadd_paths,
        q=q,
    )
    first = backend.analyze_stamp_science_series_v1(request)
    manifest_before = first.manifest_path.read_bytes()

    with pytest.raises(FileExistsError, match="already exists"):
        backend.analyze_stamp_science_series_v1(request)
    assert first.manifest_path.read_bytes() == manifest_before


def test_factor_snapshot_loader_round_trips_real_science_writer_identity(
    tmp_path: Path,
) -> None:
    import et_mainsim.stamp_science_analysis as backend
    from et_mainsim.stamp_science_inputs import (
        ScienceInputCurve,
        stable_namespaced_source_id,
        write_science_factor_snapshot,
    )

    source_id = stable_namespaced_source_id("varlc", "KIC003331147")
    curve = ScienceInputCurve(
        track="varlc",
        namespace="varlc",
        external_source_id="KIC003331147",
        source_id_int64=source_id,
        source_class="pulsating_variable",
        gaia_g_mag=11.5,
        detector_xpix=2000.0,
        detector_ypix=4500.0,
        factors=np.asarray([0.9, 1.0, 1.1, 1.2]),
        metadata={"q_definition": "normalised_flux"},
    )
    snapshot = tmp_path / "science-factor.npz"
    write_science_factor_snapshot(snapshot, curve=curve)

    factors, identity = backend._load_factor_snapshot(
        snapshot,
        expected_source_identity={
            "production_track": "varlc",
            "namespace": "varlc",
            "external_source_id": "KIC003331147",
            "source_id": str(source_id),
        },
        first_raw_index=1,
        last_raw_index=4,
    )

    np.testing.assert_array_equal(factors, [1.0, 1.1, 1.2])
    assert identity == {
        "production_track": "varlc",
        "namespace": "varlc",
        "external_source_id": "KIC003331147",
        "source_id": str(source_id),
        "snapshot_schema_id": "et_mainsim.stamp_science_factor_snapshot.v1",
    }
    with pytest.raises(
        backend.StampScienceAnalysisContractError,
        match="snapshot source identity differs",
    ):
        backend._load_factor_snapshot(
            snapshot,
            expected_source_identity={
                "production_track": "aster",
                "namespace": "varlc",
                "external_source_id": "KIC003331147",
                "source_id": str(source_id),
            },
            first_raw_index=0,
            last_raw_index=4,
        )


def test_factor_snapshot_loader_accepts_real_galaxy_schema_and_rejects_identity_mix(
    tmp_path: Path,
) -> None:
    import et_mainsim.stamp_science_analysis as backend
    from et_mainsim.galaxy_lightcurves import (
        GalaxyLightCurve,
        write_galaxy_factor_snapshot,
    )

    curve = GalaxyLightCurve(
        source_id=42,
        gaia_g_mag=11.0,
        ra_deg=10.0,
        dec_deg=20.0,
        source_class="fixture",
        native_time_seconds=np.asarray([0.0, 10.0]),
        clean_flux_factor=np.asarray([1.0, 1.1]),
        input_identity={"sha256": "0" * 64, "size_bytes": 1, "path": "x"},
    )
    snapshot = tmp_path / "galaxy-factor.npz"
    write_galaxy_factor_snapshot(
        snapshot,
        curve=curve,
        factors=np.asarray([1.0, 1.1, 1.2]),
        raw_exposure_seconds=10.0,
    )

    factors, identity = backend._load_factor_snapshot(
        snapshot,
        expected_source_identity={
            "production_track": "galaxy",
            "namespace": "gaia_dr3",
            "external_source_id": "42",
            "source_id": "42",
        },
        first_raw_index=0,
        last_raw_index=3,
    )
    np.testing.assert_array_equal(factors, [1.0, 1.1, 1.2])
    assert identity["snapshot_schema_id"] == "et_mainsim.galaxy_factor_snapshot.v1"

    with pytest.raises(
        backend.StampScienceAnalysisContractError,
        match="snapshot source identity differs",
    ):
        backend._load_factor_snapshot(
            snapshot,
            expected_source_identity={
                "production_track": "galaxy",
                "namespace": "gaia_dr3",
                "external_source_id": "43",
                "source_id": "43",
            },
            first_raw_index=0,
            last_raw_index=3,
        )


def test_formal_request_writer_derives_noise_and_freezes_ready_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import et_mainsim.stamp_science_analysis as backend
    from et_mainsim.galaxy_lightcurves import (
        GalaxyLightCurve,
        write_galaxy_factor_snapshot,
    )

    run_root = tmp_path / "run"
    snapshot = run_root / "inputs" / "factor.npz"
    curve = GalaxyLightCurve(
        source_id=42,
        gaia_g_mag=11.0,
        ra_deg=10.0,
        dec_deg=20.0,
        source_class="fixture",
        native_time_seconds=np.asarray([0.0, 600.0]),
        clean_flux_factor=np.asarray([1.0, 1.1]),
        input_identity={"sha256": "0" * 64, "size_bytes": 1, "path": "x"},
    )
    snapshot_identity = write_galaxy_factor_snapshot(
        snapshot,
        curve=curve,
        factors=1.0 + 0.01 * np.sin(np.arange(120) / 5.0),
        raw_exposure_seconds=10.0,
    )
    from et_mainsim.time_shards import plan_continuous_time_shards

    time_plan = plan_continuous_time_shards(
        raw_start_index=0,
        raw_stop_index=120,
        coadd_sizes=(3, 6, 12, 30),
        raw_exposure_seconds=10.0,
        max_raw_frames_per_shard=60,
    )
    time_plan_path = time_plan.write_manifest(run_root / "inputs" / "time_shards.json")
    time_plan_identity = backend._file_identity(time_plan_path)
    production_manifest = run_root / "production_manifest.json"
    production_manifest.write_text(
        json.dumps(
            {
                "schema_id": "et_mainsim.galaxy_stamp_production.v1",
                "schema_version": 3,
                "run_id": "galaxy-formal-v1",
                "delivery": {
                    "raw_exposure_seconds": 10.0,
                    "cadence_seconds": [30.0, 60.0, 120.0, 300.0],
                    "coadd_sizes": [3, 6, 12, 30],
                    "time_plan_relative_path": "inputs/time_shards.json",
                    "time_plan_identity": time_plan_identity,
                },
                "simulation_spec_base": {
                    "readout": {
                        "readout_noise": {
                            "unit": "electron / pix",
                            "value": 6.0,
                        },
                        "gain_electrons_per_adu": {
                            "unit": "electron / adu",
                            "value": 1.4,
                        },
                        "enable_adc_digitization": True,
                        "adc_round_values": True,
                    },
                    "observation": {
                        "exposure_duration": {"unit": "s", "value": 10.0}
                    },
                },
                "targets": [
                    {
                        "source_id": "42",
                        "source_id_int64": 42,
                        "factor_snapshot_relative_path": "inputs/factor.npz",
                        "factor_snapshot": snapshot_identity,
                    }
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    production_identity = backend._file_identity(production_manifest)
    raw_paths, coadd_paths, _ = _series_fixture(
        run_root / "temporary-bundles",
        second_start=60,
        target_source_id="42",
        run_id="galaxy-formal-v1",
        production_manifest_identity=production_identity,
        frames_per_shard=60,
        coadd_factors=(3, 6, 12, 30),
    )
    delivery_root = (
        run_root
        / "cases"
        / "injected"
        / "stamps"
        / "target_42"
        / "delivery"
    )
    for shard_index, raw_path in enumerate(raw_paths):
        shard_root = delivery_root / f"shard_{shard_index:05d}"
        shard_root.mkdir(parents=True)
        raw_path.rename(shard_root / "raw.h5")
        for factor, paths in coadd_paths.items():
            paths[shard_index].rename(shard_root / f"coadd_{factor * 10}s.h5")
    request_path = tmp_path / "request.json"
    automatic_code_identity = {
        "schema_id": "et_mainsim.formal_analysis_code_identity.v1",
        "schema_version": 1,
        "provenance": {
            "et_mainsim": {
                "root": "/clean/et-mainsim",
                "commit": "a" * 40,
                "branch": "feat/formal-analysis",
                "dirty": False,
                "version": "0.1.0",
            },
            "photsim7": {
                "root": "/clean/photsim7",
                "commit": "b" * 40,
                "branch": "main",
                "dirty": False,
                "version": "0.1.0",
            },
            "runtime": {
                "python": "3.13.0",
                "executable": "/clean/python",
                "platform": "fixture-platform",
                "hostname": "fixture-host",
            },
        },
        "analysis_dependencies": {
            "astropy": "7.0.0",
            "h5py": "3.13.0",
            "matplotlib": "3.10.0",
            "numpy": "2.2.0",
            "torch": "2.6.0",
        },
    }
    monkeypatch.setattr(
        backend,
        "collect_formal_analysis_code_identity_v1",
        lambda: automatic_code_identity,
    )
    assert {
        "raw_bundle_paths",
        "direct_coadd_bundle_paths",
        "code_identity",
    }.isdisjoint(
        inspect.signature(
            backend.write_stamp_science_analysis_request_v1
        ).parameters
    )

    result = backend.write_stamp_science_analysis_request_v1(
        request_path,
        production_manifest=production_manifest,
        source_id="42",
        case="injected",
        output_dir=tmp_path / "analysis",
    )
    request = backend.load_stamp_science_analysis_request_v1(request_path)
    request_payload = json.loads(request_path.read_text(encoding="utf-8"))

    assert result == request_path.resolve()
    assert request_payload["schema_id"] == (
        "et_mainsim.stamp_science_analysis_request.v2"
    )
    assert request_payload["schema_version"] == 2
    assert request_payload["formal_profile_id"] == "et_stamp_science_formal_10s_v2"
    assert request.read_noise_e_per_pixel == pytest.approx(6.0)
    assert request.quantization_noise_e_per_pixel == pytest.approx(
        1.4 / np.sqrt(12.0)
    )
    assert request.policy.coadd_factors == (1, 3, 6, 12, 30)
    assert request.policy.raw_exposure_seconds == 10.0
    assert request.policy.photometry.cdpp_windows_minutes == (30, 90, 390)
    assert request.policy.photometry.minimum_coverage_fraction == 0.95
    assert request.policy.photometry.minimum_accepted_bins == 10
    assert request.analysis_context["production_track"] == "galaxy"
    assert request.analysis_context["source_identity"] == {
        "production_track": "galaxy",
        "namespace": "gaia_dr3",
        "external_source_id": "42",
        "source_id": "42",
    }
    assert request.analysis_context["production_manifest"]["identity"] == (
        backend._cli_file_binding(production_manifest)["identity"]
    )
    assert request.code_identity == automatic_code_identity
    assert backend.validate_stamp_science_analysis_request_ready_v1(request) is request

    stale_request = tmp_path / "stale-v1-request.json"
    stale_payload = dict(request_payload)
    stale_payload["schema_id"] = "et_mainsim.stamp_science_analysis_request.v1"
    stale_payload["schema_version"] = 1
    stale_request.write_text(json.dumps(stale_payload), encoding="utf-8")
    with pytest.raises(
        backend.StampScienceAnalysisContractError,
        match="unsupported analysis request schema/version",
    ):
        backend.load_stamp_science_analysis_request_v1(stale_request)

    explicit_context = dict(request.analysis_context)
    explicit_context["input_discovery"] = {
        "mode": "explicit_identity_bound_paths_v1"
    }
    with pytest.raises(
        backend.StampScienceAnalysisContractError,
        match="canonical production layout",
    ):
        backend.validate_stamp_science_analysis_request_ready_v1(
            replace(request, analysis_context=explicit_context)
        )

    forged_identity = json.loads(json.dumps(automatic_code_identity))
    forged_identity["provenance"]["photsim7"]["dirty"] = True
    with pytest.raises(
        backend.StampScienceAnalysisContractError,
        match="code identity differs",
    ):
        backend.validate_stamp_science_analysis_request_ready_v1(
            replace(request, code_identity=forged_identity)
        )

    portable_identity = json.loads(json.dumps(automatic_code_identity))
    portable_identity["provenance"]["runtime"]["hostname"] = "compute-node"
    portable_identity["provenance"]["et_mainsim"]["root"] = "/cluster/et-mainsim"
    portable_identity["provenance"]["photsim7"]["root"] = "/cluster/photsim7"
    assert backend.validate_stamp_science_analysis_request_ready_v1(
        replace(request, code_identity=portable_identity)
    ).code_identity == portable_identity


def test_formal_ready_validator_rejects_nonfrozen_request_policy(
    tmp_path: Path,
) -> None:
    import et_mainsim.stamp_science_analysis as backend

    raw_paths, coadd_paths, q = _series_fixture(tmp_path)
    request = _request(
        tmp_path,
        raw_paths=raw_paths,
        coadd_paths=coadd_paths,
        q=q,
    )
    with pytest.raises(
        backend.StampScienceAnalysisContractError,
        match="formal analysis profile",
    ):
        backend.validate_stamp_science_analysis_request_ready_v1(request)


def test_canonical_bundle_discovery_supports_static_subset_and_rejects_injected_gap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import et_mainsim.stamp_science_analysis as backend
    from et_mainsim.time_shards import plan_continuous_time_shards

    run_root = tmp_path / "run"
    production_manifest = run_root / "production_manifest.json"
    production_manifest.parent.mkdir(parents=True)
    production_manifest.write_text("{}\n", encoding="utf-8")
    production = SimpleNamespace(
        manifest_path=production_manifest.resolve(),
        run_id="fixture-run",
        source_identity={"production_track": "galaxy"},
        manifest={"delivery": {}},
        manifest_binding={
            "path": str(production_manifest.resolve()),
            "identity": backend._file_identity(production_manifest),
        },
    )
    plan = plan_continuous_time_shards(
        raw_start_index=0,
        raw_stop_index=120,
        coadd_sizes=(3, 6, 12, 30),
        raw_exposure_seconds=10.0,
        max_raw_frames_per_shard=60,
    )
    monkeypatch.setattr(
        backend,
        "_resolve_production_source_v1",
        lambda *_args, **_kwargs: production,
    )
    monkeypatch.setattr(
        backend,
        "_load_bound_time_plan_v1",
        lambda _production: (plan, {"size_bytes": 1, "sha256": "a" * 64}),
    )

    for case in ("static", "injected"):
        raw_paths, coadd_paths, _ = _series_fixture(
            tmp_path / f"temporary-{case}",
            second_start=60,
            target_source_id="42",
            frames_per_shard=60,
            coadd_factors=(3, 6, 12, 30),
            science_case=case,
        )
        root = (
            run_root
            / "cases"
            / case
            / "stamps"
            / "target_42"
            / "delivery"
        )
        for shard_index, raw_path in enumerate(raw_paths):
            shard_root = root / f"shard_{shard_index:05d}"
            shard_root.mkdir(parents=True)
            raw_path.rename(shard_root / "raw.h5")
            for factor, paths in coadd_paths.items():
                paths[shard_index].rename(
                    shard_root / f"coadd_{factor * 10}s.h5"
                )

    with pytest.raises(
        backend.StampScienceAnalysisContractError,
        match="static task list",
    ):
        backend.discover_stamp_science_analysis_bundles_v1(
            production_manifest,
            source_id="42",
            case="static",
        )
    static_task_list = (
        run_root / "inputs" / "task_lists" / "static_representative.json"
    )
    static_task_list.parent.mkdir(parents=True)
    static_task_list.write_text(
        json.dumps(
            {
                "schema_id": "et_mainsim.science_stamp_task_list.v1",
                "schema_version": 1,
                "case": "static",
                "production_manifest_identity": production.manifest_binding[
                    "identity"
                ],
                "tasks": [{"source_id": 42, "shard_id": 0}],
            }
        ),
        encoding="utf-8",
    )
    static = backend.discover_stamp_science_analysis_bundles_v1(
        production_manifest,
        source_id="42",
        case="static",
        static_task_list=static_task_list,
    )
    assert static.shard_ids == (0,)
    assert len(static.raw_bundle_paths) == 1
    assert set(static.direct_coadd_bundle_paths) == {3, 6, 12, 30}

    old_task_list = run_root / "inputs" / "static_representative_day0.json"
    old_task_list.write_bytes(static_task_list.read_bytes())
    with pytest.raises(
        backend.StampScienceAnalysisContractError,
        match="inputs/task_lists/static_representative.json",
    ):
        backend.discover_stamp_science_analysis_bundles_v1(
            production_manifest,
            source_id="42",
            case="static",
            static_task_list=old_task_list,
        )

    production.source_identity = {"production_track": "varlc"}
    production.manifest = {
        "delivery": {"execution_mode": "staged_local_scratch_v1"}
    }
    with pytest.raises(
        backend.StampScienceAnalysisContractError,
        match="publication receipt",
    ):
        backend.discover_stamp_science_analysis_bundles_v1(
            production_manifest,
            source_id="42",
            case="injected",
        )
    production.source_identity = {"production_track": "galaxy"}
    production.manifest = {"delivery": {}}

    missing = (
        run_root
        / "cases"
        / "injected"
        / "stamps"
        / "target_42"
        / "delivery"
        / "shard_00001"
        / "coadd_300s.h5"
    )
    missing.unlink()
    with pytest.raises(
        backend.StampScienceAnalysisContractError,
        match="exact raw/coadd matrix",
    ):
        backend.discover_stamp_science_analysis_bundles_v1(
            production_manifest,
            source_id="42",
            case="injected",
        )


def test_analysis_cli_help_names_the_authoritative_static_task_list(
    capsys: pytest.CaptureFixture[str],
) -> None:
    import et_mainsim.stamp_science_analysis as backend

    with pytest.raises(SystemExit) as exit_info:
        backend.main(["write-request", "--help"])
    assert exit_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "inputs/task_lists/static_representative.json" in help_text
    assert "static_representative_day0.json" not in help_text
    assert "--gate-task-list" in help_text


def test_injected_gate_discovery_only_admits_bound_contiguous_shards_zero_to_five(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import et_mainsim.stamp_science_analysis as backend
    from et_mainsim.time_shards import plan_continuous_time_shards

    run_root = tmp_path / "run"
    production_manifest = run_root / "production_manifest.json"
    production_manifest.parent.mkdir(parents=True)
    production_manifest.write_text("{}\n", encoding="utf-8")
    production = SimpleNamespace(
        manifest_path=production_manifest.resolve(),
        run_id="fixture-run",
        source_identity={"production_track": "galaxy"},
        manifest={"delivery": {}},
        manifest_binding={
            "path": str(production_manifest.resolve()),
            "identity": backend._file_identity(production_manifest),
        },
    )
    plan = plan_continuous_time_shards(
        raw_start_index=0,
        raw_stop_index=10_800,
        coadd_sizes=(3, 6, 12, 30),
        raw_exposure_seconds=10.0,
        max_raw_frames_per_shard=60,
    )
    monkeypatch.setattr(
        backend,
        "_resolve_production_source_v1",
        lambda *_args, **_kwargs: production,
    )
    monkeypatch.setattr(
        backend,
        "_load_bound_time_plan_v1",
        lambda _production: (plan, {"size_bytes": 1, "sha256": "a" * 64}),
    )
    delivery_root = (
        run_root
        / "cases"
        / "injected"
        / "stamps"
        / "target_42"
        / "delivery"
    )
    for shard_id in range(6):
        start = shard_id * 60
        planes = _raw_planes(start=start, n_frames=60)
        planes.pop("q")
        shard_root = delivery_root / f"shard_{shard_id:05d}"
        shard_root.mkdir(parents=True)
        _write_bundle(
            shard_root / "raw.h5",
            planes=planes,
            product_kind="raw",
            factor=1,
            shard_id=shard_id,
            science_case="injected",
            target_source_id="42",
        )
        for factor in (3, 6, 12, 30):
            _write_bundle(
                shard_root / f"coadd_{factor * 10}s.h5",
                planes=_coadd_planes(planes, factor=factor),
                product_kind="coadd",
                factor=factor,
                shard_id=shard_id,
                science_case="injected",
                target_source_id="42",
            )
    gate_task_list = run_root / "inputs" / "task_lists" / "injected_gate.json"
    gate_task_list.parent.mkdir(parents=True)

    def write_gate(shards: list[int]) -> None:
        gate_task_list.write_text(
            json.dumps(
                {
                    "schema_id": "et_mainsim.science_stamp_task_list.v1",
                    "schema_version": 1,
                    "case": "injected",
                    "production_manifest_identity": production.manifest_binding[
                        "identity"
                    ],
                    "tasks": [
                        {"source_id": 42, "shard_id": shard_id}
                        for shard_id in shards
                    ],
                }
            ),
            encoding="utf-8",
        )

    write_gate([*range(6), 179])
    with pytest.raises(
        backend.StampScienceAnalysisContractError,
        match="explicit shards 0..5",
    ):
        backend.discover_stamp_science_analysis_bundles_v1(
            production_manifest,
            source_id="42",
            case="injected",
            gate_task_list=gate_task_list,
        )
    discovery = backend.discover_stamp_science_analysis_bundles_v1(
        production_manifest,
        source_id="42",
        case="injected",
        shard_ids=tuple(range(6)),
        gate_task_list=gate_task_list,
    )
    assert discovery.shard_ids == tuple(range(6))
    assert all("shard_00179" not in str(path) for path in discovery.raw_bundle_paths)
    assert discovery.gate_task_list_binding == backend._cli_file_binding(
        gate_task_list
    )
    production.read_noise_e_per_raw_pixel = 6.0
    production.quantization_noise_e_per_raw_pixel = 0.0
    production.factor_snapshot_binding = {
        "path": str(run_root / "inputs" / "unused-factor.npz"),
        "identity": {"size_bytes": 1, "sha256": "c" * 64},
    }
    monkeypatch.setattr(
        backend,
        "_validate_production_binding_for_headers",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        backend,
        "collect_formal_analysis_code_identity_v1",
        lambda: {"git_commit": "unit-test"},
    )
    request_path = tmp_path / "injected-gate-request.json"
    backend.write_stamp_science_analysis_request_v1(
        request_path,
        production_manifest=production_manifest,
        source_id="42",
        case="injected",
        shard_ids=tuple(range(6)),
        gate_task_list=gate_task_list,
        output_dir=tmp_path / "gate-analysis",
    )
    request_payload = json.loads(request_path.read_text(encoding="utf-8"))
    assert request_payload["input_discovery"]["gate_task_list"] == (
        backend._cli_file_binding(gate_task_list)
    )

    for invalid in ([*range(6)], [*range(6), 178]):
        write_gate(invalid)
        with pytest.raises(
            backend.StampScienceAnalysisContractError,
            match="0..5 and 179",
        ):
            backend.discover_stamp_science_analysis_bundles_v1(
                production_manifest,
                source_id="42",
                case="injected",
                shard_ids=tuple(range(6)),
                gate_task_list=gate_task_list,
            )


def test_formal_injected_gate_request_writes_loads_runs_and_validates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import photsim7.aperture as legacy_aperture

    monkeypatch.setattr(
        legacy_aperture,
        "maximize_cumulative_snr",
        _select_target_pixels,
    )
    import et_mainsim.stamp_science_analysis as backend
    from et_mainsim.galaxy_lightcurves import (
        GalaxyLightCurve,
        write_galaxy_factor_snapshot,
    )
    from et_mainsim.time_shards import plan_continuous_time_shards

    run_root = tmp_path / "run"
    snapshot = run_root / "inputs" / "factor.npz"
    factor_count = 21_600
    factors = 1.0 + 0.1 * (np.arange(factor_count) % 4)
    snapshot_identity = write_galaxy_factor_snapshot(
        snapshot,
        curve=GalaxyLightCurve(
            source_id=42,
            gaia_g_mag=11.0,
            ra_deg=10.0,
            dec_deg=20.0,
            source_class="fixture",
            native_time_seconds=np.asarray([0.0, 10.0]),
            clean_flux_factor=np.asarray([1.0, 1.1]),
            input_identity={"sha256": "0" * 64, "size_bytes": 1, "path": "x"},
        ),
        factors=factors,
        raw_exposure_seconds=10.0,
    )
    time_plan = plan_continuous_time_shards(
        raw_start_index=0,
        raw_stop_index=factor_count,
        coadd_sizes=(3, 6, 12, 30),
        raw_exposure_seconds=10.0,
        max_raw_frames_per_shard=120,
    )
    time_plan_path = time_plan.write_manifest(
        run_root / "inputs" / "time_shards.json"
    )
    production_manifest = run_root / "production_manifest.json"
    production_manifest.write_text(
        json.dumps(
            {
                "schema_id": "et_mainsim.galaxy_stamp_production.v1",
                "schema_version": 3,
                "run_id": "galaxy-gate-fixture",
                "delivery": {
                    "raw_exposure_seconds": 10.0,
                    "cadence_seconds": [30.0, 60.0, 120.0, 300.0],
                    "coadd_sizes": [3, 6, 12, 30],
                    "time_plan_relative_path": "inputs/time_shards.json",
                    "time_plan_identity": backend._file_identity(time_plan_path),
                },
                "simulation_spec_base": {
                    "readout": {
                        "readout_noise": {
                            "unit": "electron / pix",
                            "value": 6.0,
                        },
                        "gain_electrons_per_adu": {
                            "unit": "electron / adu",
                            "value": 1.4,
                        },
                        "enable_adc_digitization": True,
                        "adc_round_values": True,
                    },
                    "observation": {
                        "exposure_duration": {"unit": "s", "value": 10.0}
                    },
                },
                "targets": [
                    {
                        "source_id": "42",
                        "source_id_int64": 42,
                        "factor_snapshot_relative_path": "inputs/factor.npz",
                        "factor_snapshot": snapshot_identity,
                    }
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    production_identity = backend._file_identity(production_manifest)
    delivery_root = (
        run_root
        / "cases"
        / "injected"
        / "stamps"
        / "target_42"
        / "delivery"
    )
    for shard in time_plan.shards[:6]:
        planes = _raw_planes(
            start=shard.raw_start_index,
            n_frames=shard.raw_frame_count,
            stamp_shape=(21, 23),
            target_yx=(10, 11),
        )
        planes.pop("q")
        shard_root = delivery_root / f"shard_{shard.shard_id:05d}"
        shard_root.mkdir(parents=True)
        _write_bundle(
            shard_root / "raw.h5",
            planes=planes,
            product_kind="raw",
            factor=1,
            shard_id=shard.shard_id,
            science_case="injected",
            target_source_id="42",
            run_id="galaxy-gate-fixture",
            production_manifest_identity=production_identity,
        )
        for factor in (3, 6, 12, 30):
            _write_bundle(
                shard_root / f"coadd_{factor * 10}s.h5",
                planes=_coadd_planes(planes, factor=factor),
                product_kind="coadd",
                factor=factor,
                shard_id=shard.shard_id,
                science_case="injected",
                target_source_id="42",
                run_id="galaxy-gate-fixture",
                production_manifest_identity=production_identity,
            )
    gate_task_list = run_root / "inputs" / "task_lists" / "injected_gate.json"
    gate_task_list.parent.mkdir(parents=True)
    gate_task_list.write_text(
        json.dumps(
            {
                "schema_id": "et_mainsim.science_stamp_task_list.v1",
                "schema_version": 1,
                "case": "injected",
                "production_manifest_identity": production_identity,
                "tasks": [
                    {"source_id": 42, "shard_id": shard_id}
                    for shard_id in (*range(6), 179)
                ],
            }
        ),
        encoding="utf-8",
    )
    code_identity = {
        "schema_id": "et_mainsim.formal_analysis_code_identity.v1",
        "schema_version": 1,
        "provenance": {
            "et_mainsim": {
                "commit": "a" * 40,
                "dirty": False,
                "version": "1",
            },
            "photsim7": {
                "commit": "b" * 40,
                "dirty": False,
                "version": "1",
            },
            "runtime": {"python": "3.13.0"},
        },
        "analysis_dependencies": {},
    }
    monkeypatch.setattr(
        backend,
        "collect_formal_analysis_code_identity_v1",
        lambda: code_identity,
    )
    request_path = tmp_path / "gate-request.json"
    backend.write_stamp_science_analysis_request_v1(
        request_path,
        production_manifest=production_manifest,
        source_id="42",
        case="injected",
        shard_ids=tuple(range(6)),
        gate_task_list=gate_task_list,
        output_dir=tmp_path / "gate-analysis",
    )
    request = backend.load_stamp_science_analysis_request_v1(request_path)
    assert request.analysis_context["input_discovery"]["shard_ids"] == list(
        range(6)
    )
    assert len(request.raw_bundle_paths) == 6
    assert request.raw_relative_flux.shape == (720,)
    assert backend.validate_stamp_science_analysis_request_ready_v1(request) is request

    publication = backend.analyze_stamp_science_product_set_v1(request)
    validation = backend.validate_stamp_science_analysis_product_set_v1(
        publication.output_dir
    )
    assert validation.complete is True


def test_formal_code_identity_is_automatic_and_rejects_dirty_or_unknown_repositories(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import et_mainsim.stamp_science_analysis as backend

    clean = {
        "et_mainsim": {"commit": "a" * 40, "dirty": False, "version": "1"},
        "photsim7": {"commit": "b" * 40, "dirty": False, "version": "2"},
        "runtime": {"python": "3.13.0"},
    }
    monkeypatch.setattr(backend, "collect_provenance", lambda _root: clean)
    identity = backend.collect_formal_analysis_code_identity_v1()
    assert identity["schema_id"] == "et_mainsim.formal_analysis_code_identity.v1"
    assert identity["provenance"] == clean
    assert {"numpy", "h5py", "astropy", "matplotlib", "torch"} <= set(
        identity["analysis_dependencies"]
    )

    for key, value in (("dirty", True), ("commit", None)):
        broken = json.loads(json.dumps(clean))
        broken["photsim7"][key] = value
        monkeypatch.setattr(backend, "collect_provenance", lambda _root, p=broken: p)
        with pytest.raises(
            backend.StampScienceAnalysisContractError,
            match="clean known ET-mainsim and Photsim7 commits",
        ):
            backend.collect_formal_analysis_code_identity_v1()


def test_formal_execution_identity_is_collected_again_at_analysis_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import et_mainsim.stamp_science_analysis as backend

    raw_paths, coadd_paths, q = _series_fixture(tmp_path)
    base_request = _request(
        tmp_path,
        raw_paths=raw_paths,
        coadd_paths=coadd_paths,
        q=q,
    )
    request = replace(
        base_request,
        analysis_context={
            **dict(base_request.analysis_context),
            "formal_profile_id": backend.STAMP_SCIENCE_FORMAL_PROFILE_ID,
        },
    )
    execution_identity = {
        "schema_id": "et_mainsim.formal_analysis_code_identity.v1",
        "schema_version": 1,
        "provenance": {"runtime": {"hostname": "execution-node"}},
        "analysis_dependencies": {"torch": "2.6.0"},
    }
    execution_hardware = {
        "schema_id": "et_mainsim.analysis_execution_hardware.v1",
        "analysis_compute_device": "cpu",
        "cpu_count": 96,
        "machine": "x86_64",
        "cuda_available": True,
        "cuda_device_names": ["NVIDIA H100 80GB HBM3"],
    }
    validated: list[object] = []
    monkeypatch.setattr(
        backend,
        "validate_stamp_science_analysis_request_ready_v1",
        lambda value: validated.append(value) or value,
    )
    monkeypatch.setattr(
        backend,
        "collect_formal_analysis_code_identity_v1",
        lambda: execution_identity,
    )
    monkeypatch.setattr(
        backend,
        "_collect_analysis_execution_hardware_v1",
        lambda: execution_hardware,
    )

    assert backend._collect_analysis_execution_identity_v1(request) == (
        {
            **execution_identity,
            "execution_hardware": execution_hardware,
        }
    )
    assert validated == [request]


def test_analysis_request_publication_does_not_hide_directory_fsync_io_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import et_mainsim.stamp_science_analysis as backend

    target = tmp_path / "request.json"
    original_fsync = backend.os.fsync
    calls = 0

    def fail_parent_fsync(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError(errno.EIO, "synthetic directory fsync failure")
        original_fsync(descriptor)

    monkeypatch.setattr(backend.os, "fsync", fail_parent_fsync)

    with pytest.raises(OSError, match="synthetic directory fsync failure"):
        backend._write_bound_json_noreplace(target, {"complete": True})

    assert json.loads(target.read_text(encoding="utf-8")) == {"complete": True}


def test_product_set_contract_has_rates_tables_figures_and_product_specific_clean_frames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import photsim7.aperture as legacy_aperture

    monkeypatch.setattr(
        legacy_aperture,
        "maximize_cumulative_snr",
        _select_target_pixels,
    )
    raw_paths, _, q = _series_fixture(
        tmp_path / "inputs",
        stamp_shape=(21, 23),
        target_yx=(10, 11),
    )
    # The corner of fixed13 is outside the two-pixel OA.  Make the first frame
    # of each shard saturated there so each product must select its own clean
    # representative frames in the same raw pass.
    for path in raw_paths:
        with h5py.File(path, "r+") as handle:
            handle["fullwell_count"][0, 4, 5] = np.uint16(1)
            handle["saturated_mask"][0, 4, 5] = True
    import et_mainsim.stamp_science_analysis as backend

    publication = backend.analyze_stamp_science_product_set_v1(
        _request(
            tmp_path,
            raw_paths=raw_paths,
            coadd_paths={},
            q=q,
            output_name="complete-product-set",
            require_direct_coadd_parity=False,
        )
    )

    assert publication.manifest_path.is_file()
    product_set_manifest = json.loads(
        publication.manifest_path.read_text(encoding="utf-8")
    )
    assert product_set_manifest["schema_id"] == (
        "et_mainsim.stamp_science_analysis_product_set.v2"
    )
    assert product_set_manifest["schema_version"] == 2
    assert product_set_manifest["complete"] is True
    assert product_set_manifest["ready"] is True
    assert set(product_set_manifest["products"]) == {
        "reference_fixed13_v1",
        "science_optimal_aperture_v1",
    }

    representative_indices = {}
    reference_columns = {
        "cadence_seconds",
        "time_start_seconds",
        "exposure_seconds",
        "raw_frame_start_index",
        "raw_frame_stop_index_exclusive",
        "raw_relative_flux_mean",
        "raw_relative_flux_sum",
        "flux_expectation_bgsub_e",
        "flux_expectation_bgsub_e_per_s",
        "aperture_valid",
            "quality_bitmask",
            "captured_flux_fraction",
            "captured_flux_denominator_e",
            "captured_flux_qa_pass",
            "fitted_flux_expectation_e",
        "fitted_flux_expectation_e_per_s",
        "residual_expectation_e",
        "residual_expectation_ppm",
    }
    for name, product in (
        ("reference_fixed13_v1", publication.reference_fixed13),
        ("science_optimal_aperture_v1", publication.science_optimal_aperture),
    ):
        manifest = json.loads(product.manifest_path.read_text(encoding="utf-8"))
        assert manifest["ready"] is True
        assert manifest["schema_id"] == (
            "et_mainsim.stamp_science_analysis_publication.v2"
        )
        assert manifest["schema_version"] == 2
        assert manifest["contract"]["schema_id"] == (
            "et_mainsim.stamp_science_analysis.v2"
        )
        assert manifest["contract"]["schema_version"] == 2
        assert manifest["contract"]["science_photometry_schema_id"] == (
            "et_mainsim.stamp_science_photometry.v2"
        )
        assert manifest["contract"]["request_code_identity"] == {
            "git_commit": "unit-test"
        }
        assert manifest["contract"]["execution_code_identity"] == {
            "git_commit": "unit-test"
        }
        assert manifest["contract"]["reference_lightcurve"] == {
            "artifact": "reference_lightcurve.ecsv",
            "schema_id": "et_mainsim.stamp_science_reference_lightcurve.v2",
            "schema_version": 2,
            "measured_flux_column": "flux_expectation_bgsub_e",
            "measured_rate_column": "flux_expectation_bgsub_e_per_s",
            "validity_column": "aperture_valid",
            "quality_column": "quality_bitmask",
            "fitted_flux_column": "fitted_flux_expectation_e",
            "residual_columns": [
                "residual_expectation_e",
                "residual_expectation_ppm",
            ],
            "required_columns": sorted(reference_columns),
        }
        assert {
            "reference_lightcurve.ecsv",
            "centroid_quality.ecsv",
            "cdpp.ecsv",
            "quality_summary.json",
            "figures/lightcurve_overview.png",
            "figures/cdpp_summary.png",
            "figures/representative_frames.png",
        } <= set(manifest["artifacts"])
        from astropy.table import Table

        reference = Table.read(
            product.output_dir / "reference_lightcurve.ecsv",
            format="ascii.ecsv",
        )
        assert reference.meta["schema_id"] == (
            "et_mainsim.stamp_science_reference_lightcurve.v2"
        )
        assert reference.meta["schema_version"] == 2
        assert reference_columns <= set(reference.colnames)
        cadence_10s = np.asarray(reference["cadence_seconds"]) == 10
        with h5py.File(product.hdf5_path, "r") as handle:
            cadence = handle["cadences/10s"]
            for dataset in (
                "flux_expectation_bgsub_e_per_s",
                "flux_local_bgsub_e_per_s",
                "fitted_flux_expectation_e_per_s",
                "model_flux_uncertainty_e",
                "model_flux_uncertainty_e_per_s",
            ):
                assert dataset in cadence
            np.testing.assert_allclose(
                cadence["flux_expectation_bgsub_e_per_s"],
                np.asarray(cadence["flux_expectation_bgsub_e"]) / 10.0,
            )
            np.testing.assert_allclose(
                cadence["model_flux_uncertainty_e"],
                cadence["flux_uncertainty_e"],
                equal_nan=True,
            )
            np.testing.assert_allclose(
                np.asarray(reference["flux_expectation_bgsub_e"])[cadence_10s],
                cadence["flux_expectation_bgsub_e"],
                equal_nan=True,
            )
            np.testing.assert_allclose(
                np.asarray(reference["flux_expectation_bgsub_e_per_s"])[
                    cadence_10s
                ],
                cadence["flux_expectation_bgsub_e_per_s"],
                equal_nan=True,
            )
            np.testing.assert_array_equal(
                np.asarray(reference["aperture_valid"])[cadence_10s],
                cadence["aperture_valid"],
            )
            np.testing.assert_array_equal(
                np.asarray(reference["quality_bitmask"])[cadence_10s],
                cadence["quality_bitmask"],
            )
            np.testing.assert_allclose(
                np.asarray(reference["fitted_flux_expectation_e"])[cadence_10s],
                cadence["fitted_flux_expectation_e"],
                equal_nan=True,
            )
            np.testing.assert_allclose(
                np.asarray(reference["residual_expectation_e"])[cadence_10s],
                cadence["residual_expectation_e"],
                equal_nan=True,
            )
            np.testing.assert_allclose(
                np.asarray(reference["residual_expectation_ppm"])[cadence_10s],
                cadence["residual_expectation_ppm"],
                equal_nan=True,
            )
        with h5py.File(product.representative_frames_path, "r") as handle:
            indices = np.asarray(handle["raw_frame_start_index"], dtype=np.int64)
            saturated = np.asarray(handle["saturated_mask"], dtype=bool)
        mask = np.load(product.aperture_mask_path, allow_pickle=False)
        assert not np.any(saturated[:, mask])
        representative_indices[name] = indices.tolist()

    assert representative_indices["reference_fixed13_v1"] == [1, 7, 11]
    assert representative_indices["science_optimal_aperture_v1"] == [0, 6, 11]
    validation = backend.validate_stamp_science_analysis_product_set_v1(
        publication.output_dir
    )
    assert validation.complete is True


def test_product_set_atomic_publish_never_replaces_a_race_created_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import photsim7.aperture as legacy_aperture

    monkeypatch.setattr(
        legacy_aperture,
        "maximize_cumulative_snr",
        _select_target_pixels,
    )
    raw_paths, coadd_paths, q = _series_fixture(
        tmp_path / "inputs",
        stamp_shape=(21, 23),
        target_yx=(10, 11),
    )
    import et_mainsim.stamp_science_analysis as backend

    original_publish = backend._atomic_publish_directory_noreplace
    target = tmp_path / "race-output"

    def race_create(source: Path, destination: Path) -> None:
        destination.mkdir()
        (destination / "race-owner.txt").write_text("owned\n", encoding="utf-8")
        original_publish(source, destination)

    monkeypatch.setattr(
        backend,
        "_atomic_publish_directory_noreplace",
        race_create,
    )
    with pytest.raises(FileExistsError):
        backend.analyze_stamp_science_product_set_v1(
            _request(
                tmp_path,
                raw_paths=raw_paths,
                coadd_paths=coadd_paths,
                q=q,
                output_name="race-output",
            )
        )

    assert (target / "race-owner.txt").read_text(encoding="utf-8") == "owned\n"
    assert not (target / "reference_fixed13_v1").exists()
    assert not list(tmp_path.glob(".race-output.*.partial"))
