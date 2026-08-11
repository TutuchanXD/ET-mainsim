from __future__ import annotations

import errno
from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path
import shutil
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
            "noise_model": {
                "schema_id": "et_mainsim.formal_stamp_noise_parameters.v1",
                "source": "unit-test",
                "read_noise_e_per_raw_pixel": 1.0,
                "quantization_noise_e_per_raw_pixel": 0.0,
                "quantization_formula": "unit-test",
            },
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


def test_frozen_aperture_rejects_training_indices_outside_int64_range() -> None:
    import et_mainsim.stamp_science_analysis as backend

    overflow = replace(
        _frozen_fixture_aperture(),
        training_raw_frame_indices=np.asarray(
            [0, np.iinfo(np.int64).max + 1],
            dtype=np.uint64,
        ),
    )

    with pytest.raises(
        backend.StampScienceAnalysisContractError,
        match="complete mask/template contract",
    ):
        backend._validate_frozen_aperture_definition(overflow)


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
        match="captured_flux_qa_pass must be true",
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
            "noise_model": {
                "schema_id": "et_mainsim.formal_stamp_noise_parameters.v1",
                "source": "unit-test",
                "read_noise_e_per_raw_pixel": 1.0,
                "quantization_noise_e_per_raw_pixel": 0.0,
                "quantization_formula": "unit-test",
            },
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


@pytest.mark.parametrize("tamper_mode", ["swap", "copy_reference"])
def test_product_set_validation_rejects_swapped_or_copied_child_roles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper_mode: str,
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

    publication = backend.analyze_stamp_science_product_set_v1(
        _request(
            tmp_path,
            raw_paths=raw_paths,
            coadd_paths=coadd_paths,
            q=q,
            output_name="role-bound-products",
        )
    )
    root = publication.output_dir
    reference = root / "reference_fixed13_v1"
    science = root / "science_optimal_aperture_v1"
    if tamper_mode == "swap":
        temporary = root / "temporary-child"
        reference.rename(temporary)
        science.rename(reference)
        temporary.rename(science)
    else:
        shutil.rmtree(science)
        shutil.copytree(reference, science)

    product_set = json.loads(
        publication.manifest_path.read_text(encoding="utf-8")
    )
    for name in product_set["products"]:
        product_set["products"][name]["analysis_manifest"] = (
            backend._file_identity(root / name / "analysis_manifest.json")
        )
    publication.manifest_path.write_text(
        json.dumps(product_set),
        encoding="utf-8",
    )

    with pytest.raises(
        backend.StampScienceAnalysisContractError,
        match="analysis product.*directory role",
    ):
        backend.validate_stamp_science_analysis_product_set_v1(root)


@pytest.mark.parametrize("binding_name", ["raw_semantic_identity", "policy"])
def test_product_set_validation_binds_common_raw_semantics_and_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    binding_name: str,
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

    publication = backend.analyze_stamp_science_product_set_v1(
        _request(
            tmp_path,
            raw_paths=raw_paths,
            coadd_paths=coadd_paths,
            q=q,
            output_name="common-binding-products",
        )
    )
    child_manifest_path = publication.science_optimal_aperture.manifest_path
    child_manifest = json.loads(child_manifest_path.read_text(encoding="utf-8"))
    contract = child_manifest["contract"]
    if binding_name == "raw_semantic_identity":
        contract["input_raw_shards"][0]["semantic_sha256"] = "f" * 64
    else:
        contract["policy"]["stream_batch_frames"] += 1

    encoded_contract = json.dumps(
        contract,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    with h5py.File(
        publication.science_optimal_aperture.hdf5_path,
        "r+",
    ) as handle:
        del handle["analysis_contract_json"]
        handle.create_dataset(
            "analysis_contract_json",
            data=np.bytes_(encoded_contract),
        )
    child_manifest["artifacts"]["photometry.h5"] = backend._file_identity(
        publication.science_optimal_aperture.hdf5_path
    )
    child_manifest_path.write_text(
        json.dumps(child_manifest),
        encoding="utf-8",
    )
    product_set = json.loads(
        publication.manifest_path.read_text(encoding="utf-8")
    )
    product_set["products"]["science_optimal_aperture_v1"][
        "analysis_manifest"
    ] = backend._file_identity(child_manifest_path)
    publication.manifest_path.write_text(
        json.dumps(product_set),
        encoding="utf-8",
    )

    with pytest.raises(
        backend.StampScienceAnalysisContractError,
        match="raw semantic identities/policy",
    ):
        backend.validate_stamp_science_analysis_product_set_v1(
            publication.output_dir
        )


@pytest.mark.parametrize(
    "analysis_product",
    [None, "reference_fixed13_v1"],
)
def test_published_aperture_loader_requires_science_optimal_product_role(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    analysis_product: str | None,
) -> None:
    import et_mainsim.stamp_science_analysis as backend

    analysis_dir = tmp_path / "published-aperture"
    analysis_dir.mkdir()
    contract = {}
    if analysis_product is not None:
        contract["analysis_product"] = analysis_product
    (analysis_dir / "analysis_manifest.json").write_text(
        json.dumps({"contract": contract}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        backend,
        "validate_stamp_science_analysis_v1",
        lambda _path: None,
    )

    with pytest.raises(
        backend.StampScienceAnalysisContractError,
        match="science_optimal_aperture_v1",
    ):
        backend._load_published_aperture_v1(analysis_dir)


@pytest.mark.parametrize(
    ("artifact_kind", "required_name"),
    [
        ("hdf5", "signal_template_e"),
        ("hdf5", "noise_template_e"),
        ("hdf5", "training_raw_frame_indices"),
        ("definition", "maximum_cumulative_snr"),
        ("definition", "algorithm"),
        ("definition", "signal_template_shape"),
    ],
)
def test_published_aperture_loader_rejects_missing_required_oa_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact_kind: str,
    required_name: str,
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

    publication = backend.analyze_stamp_science_product_set_v1(
        _request(
            tmp_path,
            raw_paths=raw_paths,
            coadd_paths=coadd_paths,
            q=q,
            output_name="analysis-products",
        )
    )
    product = publication.science_optimal_aperture
    if artifact_kind == "hdf5":
        artifact_path = product.hdf5_path
        with h5py.File(artifact_path, "r+") as handle:
            del handle[f"aperture/{required_name}"]
    else:
        artifact_path = product.aperture_definition_path
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        payload.pop(required_name)
        artifact_path.write_text(json.dumps(payload), encoding="utf-8")

    manifest = json.loads(product.manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"][artifact_path.name] = backend._file_identity(artifact_path)
    product.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        backend.StampScienceAnalysisContractError,
        match="published aperture|authoritative HDF5 aperture|aperture definition",
    ):
        backend._load_published_aperture_v1(product.output_dir)


@pytest.mark.parametrize(
    "invalid_product",
    [
        "signal_nan",
        "signal_negative",
        "noise_nan",
        "noise_zero",
        "indices_float",
        "indices_negative",
        "indices_duplicate",
        "indices_unordered",
        "indices_uint64_overflow",
    ],
)
def test_publication_validator_and_loader_reject_invalid_oa_training_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_product: str,
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

    publication = backend.analyze_stamp_science_product_set_v1(
        _request(
            tmp_path,
            raw_paths=raw_paths,
            coadd_paths=coadd_paths,
            q=q,
            output_name="analysis-products",
        )
    )
    product = publication.science_optimal_aperture
    with h5py.File(product.hdf5_path, "r+") as handle:
        aperture = handle["aperture"]
        if invalid_product == "signal_nan":
            aperture["signal_template_e"][0, 0] = np.nan
        elif invalid_product == "signal_negative":
            aperture["signal_template_e"][0, 0] = -1.0
        elif invalid_product == "noise_nan":
            aperture["noise_template_e"][0, 0] = np.nan
        elif invalid_product == "noise_zero":
            aperture["noise_template_e"][0, 0] = 0.0
        else:
            training = np.asarray(aperture["training_raw_frame_indices"])
            del aperture["training_raw_frame_indices"]
            if invalid_product == "indices_float":
                training = training.astype(np.float64)
            elif invalid_product == "indices_negative":
                training[0] = -1
            elif invalid_product == "indices_duplicate":
                training[1] = training[0]
            elif invalid_product == "indices_unordered":
                training = training[::-1]
            elif invalid_product == "indices_uint64_overflow":
                training = np.asarray(
                    [0, np.iinfo(np.int64).max + 1],
                    dtype=np.uint64,
                )
            aperture.create_dataset("training_raw_frame_indices", data=training)

    if invalid_product == "indices_uint64_overflow":
        definition = json.loads(
            product.aperture_definition_path.read_text(encoding="utf-8")
        )
        definition["training_raw_frame_indices"] = [
            0,
            int(np.iinfo(np.int64).max) + 1,
        ]
        product.aperture_definition_path.write_text(
            json.dumps(definition),
            encoding="utf-8",
        )

    manifest = json.loads(product.manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["photometry.h5"] = backend._file_identity(
        product.hdf5_path
    )
    if invalid_product == "indices_uint64_overflow":
        manifest["artifacts"]["aperture_definition.json"] = (
            backend._file_identity(product.aperture_definition_path)
        )
    product.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    for validator in (
        backend.validate_stamp_science_analysis_v1,
        backend._load_published_aperture_v1,
    ):
        with pytest.raises(
            backend.StampScienceAnalysisContractError,
            match="authoritative HDF5 aperture training products are invalid",
        ):
            validator(product.output_dir)


@pytest.mark.parametrize("maximum_cumulative_snr", [0.0, -1.0])
def test_publication_validator_and_loader_require_positive_maximum_snr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    maximum_cumulative_snr: float,
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

    publication = backend.analyze_stamp_science_product_set_v1(
        _request(
            tmp_path,
            raw_paths=raw_paths,
            coadd_paths=coadd_paths,
            q=q,
            output_name="analysis-products",
        )
    )
    product = publication.science_optimal_aperture
    definition = json.loads(
        product.aperture_definition_path.read_text(encoding="utf-8")
    )
    definition["maximum_cumulative_snr"] = maximum_cumulative_snr
    product.aperture_definition_path.write_text(
        json.dumps(definition),
        encoding="utf-8",
    )
    manifest = json.loads(product.manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["aperture_definition.json"] = backend._file_identity(
        product.aperture_definition_path
    )
    product.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    for validator in (
        backend.validate_stamp_science_analysis_v1,
        backend._load_published_aperture_v1,
    ):
        with pytest.raises(
            backend.StampScienceAnalysisContractError,
            match="aperture definition",
        ):
            validator(product.output_dir)


@pytest.mark.parametrize(
    "schema_version",
    [
        pytest.param(1.0, id="float"),
        pytest.param("1", id="string"),
        pytest.param(True, id="bool"),
    ],
)
def test_production_source_requires_native_integer_schema_version(
    tmp_path: Path,
    schema_version: object,
) -> None:
    import et_mainsim.stamp_science_analysis as backend

    production_manifest = tmp_path / "production_manifest.json"
    production_manifest.write_text(
        json.dumps(
            {
                "schema_id": "et_mainsim.science_stamp_production.v1",
                "schema_version": schema_version,
                "production_track": "aster",
                "run_id": "schema-type-fixture",
                "targets": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        backend.StampScienceAnalysisContractError,
        match="production manifest schema/version is unsupported",
    ):
        backend._resolve_production_source_v1(
            production_manifest,
            source_id="42",
        )


@pytest.mark.parametrize(
    "schema_version",
    [
        pytest.param(1.0, id="float"),
        pytest.param("1", id="string"),
        pytest.param(True, id="bool"),
    ],
)
def test_header_production_binding_requires_native_integer_schema_version(
    tmp_path: Path,
    schema_version: object,
) -> None:
    import et_mainsim.stamp_science_analysis as backend

    production_manifest = tmp_path / "production_manifest.json"
    production_manifest.write_text(
        json.dumps(
            {
                "schema_id": "et_mainsim.science_stamp_production.v1",
                "schema_version": schema_version,
                "run_id": "schema-type-fixture",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        backend.StampScienceAnalysisContractError,
        match="production manifest schema/version/run_id is unsupported",
    ):
        backend._validate_production_binding_for_headers(
            production_manifest,
            source_id="42",
            case="injected",
            headers=(),
        )


@pytest.mark.parametrize(
    "schema_version",
    [
        pytest.param(1.0, id="float"),
        pytest.param("1", id="string"),
        pytest.param(True, id="bool"),
    ],
)
def test_bound_analysis_task_list_requires_native_integer_schema_version(
    tmp_path: Path,
    schema_version: object,
) -> None:
    import et_mainsim.stamp_science_analysis as backend

    production_manifest = tmp_path / "production_manifest.json"
    production_manifest.write_text("{}\n", encoding="utf-8")
    production_binding = backend._cli_file_binding(production_manifest)
    production = SimpleNamespace(
        manifest_path=production_manifest.resolve(),
        manifest_binding=production_binding,
    )
    task_list = tmp_path / "inputs" / "task_lists" / "injected_gate.json"
    task_list.parent.mkdir(parents=True)
    task_list.write_text(
        json.dumps(
            {
                "schema_id": "et_mainsim.science_stamp_task_list.v1",
                "schema_version": schema_version,
                "case": "injected",
                "production_manifest_identity": production_binding["identity"],
                "tasks": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        backend.StampScienceAnalysisContractError,
        match="formal gate task list schema/production identity is invalid",
    ):
        backend._load_bound_analysis_task_list_v1(
            task_list,
            expected_path=task_list,
            expected_case="injected",
            production=production,
            label="gate",
        )


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


@pytest.fixture(scope="module")
def _schema_version_publication_root(
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    import photsim7.aperture as legacy_aperture

    root = tmp_path_factory.mktemp("schema-version-publication")
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            legacy_aperture,
            "maximize_cumulative_snr",
            _select_target_pixels,
        )
        raw_paths, coadd_paths, q = _series_fixture(
            root / "inputs",
            stamp_shape=(21, 23),
            target_yx=(10, 11),
        )
        import et_mainsim.stamp_science_analysis as backend

        publication = backend.analyze_stamp_science_product_set_v1(
            _request(
                root,
                raw_paths=raw_paths,
                coadd_paths=coadd_paths,
                q=q,
                output_name="baseline-products",
            )
        )
    return publication.output_dir


def _rehash_product_artifact(
    root: Path,
    *,
    product_name: str,
    artifact_name: str,
) -> None:
    import et_mainsim.stamp_science_analysis as backend

    product_root = root / product_name
    artifact_path = product_root / artifact_name
    child_manifest_path = product_root / "analysis_manifest.json"
    child_manifest = json.loads(child_manifest_path.read_text(encoding="utf-8"))
    child_manifest["artifacts"][artifact_name] = backend._file_identity(
        artifact_path
    )
    child_manifest_path.write_text(json.dumps(child_manifest), encoding="utf-8")

    product_set_manifest_path = root / "product_set_manifest.json"
    product_set_manifest = json.loads(
        product_set_manifest_path.read_text(encoding="utf-8")
    )
    product_set_manifest["products"][product_name]["analysis_manifest"] = (
        backend._file_identity(child_manifest_path)
    )
    product_set_manifest_path.write_text(
        json.dumps(product_set_manifest),
        encoding="utf-8",
    )


def _replace_product_contract(
    root: Path,
    *,
    product_name: str,
    contract: dict[str, object],
) -> None:
    product_root = root / product_name
    child_manifest_path = product_root / "analysis_manifest.json"
    child_manifest = json.loads(child_manifest_path.read_text(encoding="utf-8"))
    child_manifest["contract"] = contract
    child_manifest_path.write_text(json.dumps(child_manifest), encoding="utf-8")
    with h5py.File(product_root / "photometry.h5", "r+") as handle:
        del handle["analysis_contract_json"]
        handle.create_dataset(
            "analysis_contract_json",
            data=np.bytes_(
                json.dumps(
                    contract,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            ),
        )
    _rehash_product_artifact(
        root,
        product_name=product_name,
        artifact_name="photometry.h5",
    )


@pytest.mark.parametrize(
    ("product_name", "tamper_mode"),
    [
        pytest.param(
            "science_optimal_aperture_v1",
            "extra_schema_field",
            id="exact-schema",
        ),
        pytest.param(
            "science_optimal_aperture_v1",
            "coercible_count_type",
            id="type-sensitive-contract",
        ),
        pytest.param(
            "science_optimal_aperture_v1",
            "aperture_count",
            id="aperture-count",
        ),
        pytest.param(
            "science_optimal_aperture_v1",
            "background_count",
            id="background-count",
        ),
        pytest.param(
            "reference_fixed13_v1",
            "shape",
            id="reference-shape",
        ),
        pytest.param(
            "science_optimal_aperture_v1",
            "files",
            id="product-files",
        ),
        pytest.param(
            "reference_fixed13_v1",
            "training_indices",
            id="reference-training-indices",
        ),
    ],
)
def test_public_readback_binds_exact_aperture_definition_to_contract_and_hdf5(
    tmp_path: Path,
    _schema_version_publication_root: Path,
    product_name: str,
    tamper_mode: str,
) -> None:
    import et_mainsim.stamp_science_analysis as backend

    root = tmp_path / "aperture-definition-products"
    shutil.copytree(_schema_version_publication_root, root)
    product_root = root / product_name
    definition_path = product_root / "aperture_definition.json"
    definition = json.loads(definition_path.read_text(encoding="utf-8"))
    child_manifest = json.loads(
        (product_root / "analysis_manifest.json").read_text(encoding="utf-8")
    )
    contract = child_manifest["contract"]
    contract_aperture = contract["aperture"]
    replace_contract = tamper_mode != "coercible_count_type"
    if tamper_mode == "extra_schema_field":
        definition["unbound_extension"] = "accepted-by-subset-validation"
        contract_aperture["unbound_extension"] = "accepted-by-subset-validation"
    elif tamper_mode == "coercible_count_type":
        definition["aperture_pixel_count"] = float(
            definition["aperture_pixel_count"]
        )
    elif tamper_mode == "aperture_count":
        definition["aperture_pixel_count"] += 1
        contract_aperture["aperture_pixel_count"] += 1
    elif tamper_mode == "background_count":
        definition["background_pixel_count"] += 1
        contract_aperture["background_pixel_count"] += 1
    elif tamper_mode == "shape":
        definition["signal_template_shape"] = [1, 1]
        contract_aperture["signal_template_shape"] = [1, 1]
    elif tamper_mode == "files":
        definition["files"] = {
            "aperture_mask": "aperture_mask.npy",
            "background_mask": "background_mask.npy",
        }
        contract_aperture["files"] = dict(definition["files"])
    else:
        definition["training_raw_frame_indices"] = [0]
        contract_aperture["training_raw_frame_indices"] = [0]
    definition_path.write_text(json.dumps(definition), encoding="utf-8")
    if replace_contract:
        _replace_product_contract(
            root,
            product_name=product_name,
            contract=contract,
        )
    _rehash_product_artifact(
        root,
        product_name=product_name,
        artifact_name="aperture_definition.json",
    )

    for validator, path in (
        (backend.validate_stamp_science_analysis_v1, product_root),
        (backend.validate_stamp_science_analysis_product_set_v1, root),
    ):
        with pytest.raises(
            backend.StampScienceAnalysisContractError,
            match="aperture definition",
        ):
            validator(path)


def test_public_readback_preserves_nullable_aperture_target_peak(
    tmp_path: Path,
    _schema_version_publication_root: Path,
) -> None:
    import et_mainsim.stamp_science_analysis as backend

    root = tmp_path / "nullable-target-peak-products"
    shutil.copytree(_schema_version_publication_root, root)
    product_name = "science_optimal_aperture_v1"
    product_root = root / product_name
    definition_path = product_root / "aperture_definition.json"
    definition = json.loads(definition_path.read_text(encoding="utf-8"))
    definition["target_peak_yx"] = None
    definition_path.write_text(json.dumps(definition), encoding="utf-8")
    child_manifest = json.loads(
        (product_root / "analysis_manifest.json").read_text(encoding="utf-8")
    )
    contract = child_manifest["contract"]
    contract["aperture"]["target_peak_yx"] = None
    _replace_product_contract(
        root,
        product_name=product_name,
        contract=contract,
    )
    _rehash_product_artifact(
        root,
        product_name=product_name,
        artifact_name="aperture_definition.json",
    )

    assert backend.validate_stamp_science_analysis_v1(product_root).complete is True
    assert (
        backend.validate_stamp_science_analysis_product_set_v1(root).complete is True
    )
    aperture, _, _ = backend._load_published_aperture_v1(product_root)
    assert aperture.target_peak_yx is None


def test_public_readback_recomputes_exact_quality_summary_from_hdf5(
    tmp_path: Path,
    _schema_version_publication_root: Path,
) -> None:
    import et_mainsim.stamp_science_analysis as backend

    root = tmp_path / "quality-summary-products"
    shutil.copytree(_schema_version_publication_root, root)
    product_name = "science_optimal_aperture_v1"
    product_root = root / product_name
    quality_path = product_root / "quality_summary.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    quality["cadences"]["10s"]["cadence_count"] += 1
    quality_path.write_text(json.dumps(quality), encoding="utf-8")
    _rehash_product_artifact(
        root,
        product_name=product_name,
        artifact_name="quality_summary.json",
    )

    for validator, path in (
        (backend.validate_stamp_science_analysis_v1, product_root),
        (backend.validate_stamp_science_analysis_product_set_v1, root),
    ):
        with pytest.raises(
            backend.StampScienceAnalysisContractError,
            match="quality summary",
        ):
            validator(path)


@pytest.mark.parametrize(
    "column",
    [
        "time_start_seconds",
        "exposure_seconds",
        "raw_relative_flux_mean",
        "raw_relative_flux_sum",
        "flux_expectation_bgsub_e",
        "flux_expectation_bgsub_e_per_s",
        "fitted_flux_expectation_e",
        "fitted_flux_expectation_e_per_s",
        "residual_expectation_e",
        "residual_expectation_ppm",
        "captured_flux_fraction",
        "captured_flux_denominator_e",
    ],
)
def test_reference_lightcurve_requires_native_float_columns(
    tmp_path: Path,
    _schema_version_publication_root: Path,
    column: str,
) -> None:
    from astropy.table import Table
    import et_mainsim.stamp_science_analysis as backend

    root = tmp_path / "reference-float-dtype-products"
    shutil.copytree(_schema_version_publication_root, root)
    product_name = "science_optimal_aperture_v1"
    product_root = root / product_name
    artifact_path = product_root / "reference_lightcurve.ecsv"
    table = Table.read(artifact_path, format="ascii.ecsv")
    table[column] = np.asarray(table[column]).astype(str)
    table.write(artifact_path, format="ascii.ecsv", overwrite=True)
    _rehash_product_artifact(
        root,
        product_name=product_name,
        artifact_name="reference_lightcurve.ecsv",
    )

    for validator, path in (
        (backend.validate_stamp_science_analysis_v1, product_root),
        (backend.validate_stamp_science_analysis_product_set_v1, root),
    ):
        with pytest.raises(
            backend.StampScienceAnalysisContractError,
            match="reference-lightcurve ECSV column dtype differs",
        ):
            validator(path)


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        pytest.param("complete", 1, id="complete-native-bool"),
        pytest.param(
            "observation_product",
            "calibrated_e",
            id="observation-product",
        ),
        pytest.param(
            "calibrated_electron_products_are_derived",
            1,
            id="derived-native-bool",
        ),
        pytest.param(
            "background_realization_used",
            0,
            id="background-realization-native-bool",
        ),
        pytest.param(
            "background_products",
            ["local_background_diagnostic"],
            id="background-products",
        ),
        pytest.param(
            "default_background_product",
            "local_background_e_per_pixel",
            id="default-background-product",
        ),
    ],
)
def test_public_readback_freezes_analysis_contract_semantics(
    tmp_path: Path,
    _schema_version_publication_root: Path,
    field: str,
    invalid_value: object,
) -> None:
    import et_mainsim.stamp_science_analysis as backend

    root = tmp_path / "analysis-contract-products"
    shutil.copytree(_schema_version_publication_root, root)
    product_name = "science_optimal_aperture_v1"
    product_root = root / product_name
    child_manifest = json.loads(
        (product_root / "analysis_manifest.json").read_text(encoding="utf-8")
    )
    contract = child_manifest["contract"]
    contract[field] = invalid_value
    _replace_product_contract(
        root,
        product_name=product_name,
        contract=contract,
    )

    for validator, path in (
        (backend.validate_stamp_science_analysis_v1, product_root),
        (backend.validate_stamp_science_analysis_product_set_v1, root),
    ):
        with pytest.raises(
            backend.StampScienceAnalysisContractError,
            match="analysis contract",
        ):
            validator(path)


@pytest.mark.parametrize("tamper_mode", ["top-level-mismatch", "unsupported-common"])
def test_product_set_binds_supported_formal_profile_to_children(
    tmp_path: Path,
    _schema_version_publication_root: Path,
    tamper_mode: str,
) -> None:
    import et_mainsim.stamp_science_analysis as backend

    root = tmp_path / "formal-profile-products"
    shutil.copytree(_schema_version_publication_root, root)
    product_set_path = root / "product_set_manifest.json"
    product_set = json.loads(product_set_path.read_text(encoding="utf-8"))
    product_set["formal_profile_id"] = "unsupported-formal-profile"
    if tamper_mode == "unsupported-common":
        product_set["analysis_context"]["formal_profile_id"] = (
            "unsupported-formal-profile"
        )
    product_set_path.write_text(json.dumps(product_set), encoding="utf-8")
    if tamper_mode == "unsupported-common":
        for product_name in (
            "reference_fixed13_v1",
            "science_optimal_aperture_v1",
        ):
            child_manifest = json.loads(
                (root / product_name / "analysis_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            contract = child_manifest["contract"]
            contract["analysis_context"]["formal_profile_id"] = (
                "unsupported-formal-profile"
            )
            _replace_product_contract(
                root,
                product_name=product_name,
                contract=contract,
            )

    with pytest.raises(
        backend.StampScienceAnalysisContractError,
        match="formal profile",
    ):
        backend.validate_stamp_science_analysis_product_set_v1(root)


def test_product_set_compares_top_analysis_context_types_exactly(
    tmp_path: Path,
    _schema_version_publication_root: Path,
) -> None:
    import et_mainsim.stamp_science_analysis as backend

    root = tmp_path / "top-context-type-products"
    shutil.copytree(_schema_version_publication_root, root)
    product_set_path = root / "product_set_manifest.json"
    product_set = json.loads(product_set_path.read_text(encoding="utf-8"))
    noise_model = product_set["analysis_context"]["noise_model"]
    read_noise = noise_model["read_noise_e_per_raw_pixel"]
    assert type(read_noise) is float
    noise_model["read_noise_e_per_raw_pixel"] = int(read_noise)
    product_set_path.write_text(json.dumps(product_set), encoding="utf-8")

    with pytest.raises(
        backend.StampScienceAnalysisContractError,
        match="product-set and child analysis contexts differ",
    ):
        backend.validate_stamp_science_analysis_product_set_v1(root)


def test_public_readback_compares_embedded_contract_type_sensitively(
    tmp_path: Path,
    _schema_version_publication_root: Path,
) -> None:
    import et_mainsim.stamp_science_analysis as backend

    root = tmp_path / "embedded-contract-products"
    shutil.copytree(_schema_version_publication_root, root)
    product_name = "science_optimal_aperture_v1"
    product_root = root / product_name
    child_manifest = json.loads(
        (product_root / "analysis_manifest.json").read_text(encoding="utf-8")
    )
    embedded_contract = child_manifest["contract"]
    embedded_contract["complete"] = 1
    with h5py.File(product_root / "photometry.h5", "r+") as handle:
        del handle["analysis_contract_json"]
        handle.create_dataset(
            "analysis_contract_json",
            data=np.bytes_(json.dumps(embedded_contract).encode("utf-8")),
        )
    _rehash_product_artifact(
        root,
        product_name=product_name,
        artifact_name="photometry.h5",
    )

    for validator, path in (
        (backend.validate_stamp_science_analysis_v1, product_root),
        (backend.validate_stamp_science_analysis_product_set_v1, root),
    ):
        with pytest.raises(
            backend.StampScienceAnalysisContractError,
            match="HDF5 and publication manifest contracts differ",
        ):
            validator(path)


@pytest.mark.parametrize(
    "schema_surface",
    [
        "analysis_manifest",
        "analysis_contract",
        "science_photometry_contract",
        "reference_lightcurve_contract",
        "hdf_analysis_attr",
        "aperture_definition",
        "quality_summary",
        "reference_lightcurve_ecsv",
        "photometry_ecsv",
        "product_set_manifest",
    ],
)
@pytest.mark.parametrize(
    "invalid_version",
    [
        pytest.param(2.0, id="float"),
        pytest.param("2", id="string"),
        pytest.param(True, id="bool"),
    ],
)
def test_public_readback_requires_native_integer_schema_versions(
    tmp_path: Path,
    _schema_version_publication_root: Path,
    schema_surface: str,
    invalid_version: object,
) -> None:
    from astropy.table import Table
    import et_mainsim.stamp_science_analysis as backend

    root = tmp_path / "schema-version-products"
    shutil.copytree(_schema_version_publication_root, root)
    product_name = "science_optimal_aperture_v1"
    product_root = root / product_name
    product_set_manifest_path = root / "product_set_manifest.json"
    if schema_surface == "product_set_manifest":
        product_set_manifest = json.loads(
            product_set_manifest_path.read_text(encoding="utf-8")
        )
        product_set_manifest["schema_version"] = invalid_version
        product_set_manifest_path.write_text(
            json.dumps(product_set_manifest),
            encoding="utf-8",
        )
        with pytest.raises(
            backend.StampScienceAnalysisContractError,
            match="schema/completeness",
        ):
            backend.validate_stamp_science_analysis_product_set_v1(root)
        return

    child_manifest_path = product_root / "analysis_manifest.json"
    child_manifest = json.loads(child_manifest_path.read_text(encoding="utf-8"))
    if schema_surface == "analysis_manifest":
        child_manifest["schema_version"] = invalid_version
    elif schema_surface in {
        "analysis_contract",
        "science_photometry_contract",
        "reference_lightcurve_contract",
    }:
        contract = child_manifest["contract"]
        if schema_surface == "analysis_contract":
            contract["schema_version"] = invalid_version
        elif schema_surface == "science_photometry_contract":
            contract["science_photometry_schema_version"] = invalid_version
        else:
            contract["reference_lightcurve"]["schema_version"] = invalid_version
        with h5py.File(product_root / "photometry.h5", "r+") as handle:
            del handle["analysis_contract_json"]
            handle.create_dataset(
                "analysis_contract_json",
                data=np.bytes_(
                    json.dumps(
                        contract,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ).encode("utf-8")
                ),
            )
        child_manifest["artifacts"]["photometry.h5"] = backend._file_identity(
            product_root / "photometry.h5"
        )
    elif schema_surface == "hdf_analysis_attr":
        with h5py.File(product_root / "photometry.h5", "r+") as handle:
            handle.attrs["schema_version"] = invalid_version
        child_manifest["artifacts"]["photometry.h5"] = backend._file_identity(
            product_root / "photometry.h5"
        )
    elif schema_surface in {"aperture_definition", "quality_summary"}:
        artifact_name = (
            "aperture_definition.json"
            if schema_surface == "aperture_definition"
            else "quality_summary.json"
        )
        artifact_path = product_root / artifact_name
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        payload["schema_version"] = invalid_version
        artifact_path.write_text(json.dumps(payload), encoding="utf-8")
        child_manifest["artifacts"][artifact_name] = backend._file_identity(
            artifact_path
        )
    else:
        artifact_name = (
            "reference_lightcurve.ecsv"
            if schema_surface == "reference_lightcurve_ecsv"
            else "photometry.ecsv"
        )
        artifact_path = product_root / artifact_name
        table = Table.read(artifact_path, format="ascii.ecsv")
        table.meta["schema_version"] = invalid_version
        table.write(artifact_path, format="ascii.ecsv", overwrite=True)
        child_manifest["artifacts"][artifact_name] = backend._file_identity(
            artifact_path
        )

    child_manifest_path.write_text(json.dumps(child_manifest), encoding="utf-8")
    product_set_manifest = json.loads(
        product_set_manifest_path.read_text(encoding="utf-8")
    )
    product_set_manifest["products"][product_name]["analysis_manifest"] = (
        backend._file_identity(child_manifest_path)
    )
    product_set_manifest_path.write_text(
        json.dumps(product_set_manifest),
        encoding="utf-8",
    )
    for validator, path in (
        (backend.validate_stamp_science_analysis_v1, product_root),
        (backend.validate_stamp_science_analysis_product_set_v1, root),
    ):
        with pytest.raises(
            backend.StampScienceAnalysisContractError,
            match="schema|contract",
        ):
            validator(path)


@pytest.mark.parametrize(
    ("attribute_name", "invalid_value"),
    [
        pytest.param("complete", 1, id="complete-integer"),
        pytest.param(
            "background_realization_used",
            0,
            id="background-realization-integer",
        ),
    ],
)
def test_public_readback_requires_native_hdf_boolean_analysis_attributes(
    tmp_path: Path,
    _schema_version_publication_root: Path,
    attribute_name: str,
    invalid_value: int,
) -> None:
    import et_mainsim.stamp_science_analysis as backend

    root = tmp_path / "hdf-boolean-products"
    shutil.copytree(_schema_version_publication_root, root)
    product_name = "science_optimal_aperture_v1"
    product_root = root / product_name
    hdf_path = product_root / "photometry.h5"
    with h5py.File(hdf_path, "r+") as handle:
        handle.attrs[attribute_name] = invalid_value

    child_manifest_path = product_root / "analysis_manifest.json"
    child_manifest = json.loads(child_manifest_path.read_text(encoding="utf-8"))
    child_manifest["artifacts"]["photometry.h5"] = backend._file_identity(hdf_path)
    child_manifest_path.write_text(json.dumps(child_manifest), encoding="utf-8")
    product_set_manifest_path = root / "product_set_manifest.json"
    product_set_manifest = json.loads(
        product_set_manifest_path.read_text(encoding="utf-8")
    )
    product_set_manifest["products"][product_name]["analysis_manifest"] = (
        backend._file_identity(child_manifest_path)
    )
    product_set_manifest_path.write_text(
        json.dumps(product_set_manifest),
        encoding="utf-8",
    )

    for validator, path in (
        (backend.validate_stamp_science_analysis_v1, product_root),
        (backend.validate_stamp_science_analysis_product_set_v1, root),
    ):
        with pytest.raises(
            backend.StampScienceAnalysisContractError,
            match="schema/completeness",
        ):
            validator(path)


@pytest.mark.parametrize(
    ("product_name", "tamper_mode"),
    [
        pytest.param(
            "science_optimal_aperture_v1",
            "missing_base_artifact",
            id="science-missing-base",
        ),
        pytest.param(
            "science_optimal_aperture_v1",
            "extra_artifact",
            id="science-extra",
        ),
        pytest.param(
            "reference_fixed13_v1",
            "forbidden_template",
            id="reference-template",
        ),
        pytest.param(
            "science_optimal_aperture_v1",
            "signal_value",
            id="signal-value",
        ),
        pytest.param(
            "science_optimal_aperture_v1",
            "noise_shape",
            id="noise-shape",
        ),
    ],
)
def test_publication_requires_exact_product_artifacts_and_bound_templates(
    tmp_path: Path,
    _schema_version_publication_root: Path,
    product_name: str,
    tamper_mode: str,
) -> None:
    import et_mainsim.stamp_science_analysis as backend

    root = tmp_path / "artifact-products"
    shutil.copytree(_schema_version_publication_root, root)
    product_root = root / product_name
    manifest_path = product_root / "analysis_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if tamper_mode == "missing_base_artifact":
        manifest["artifacts"].pop("cdpp.ecsv")
    elif tamper_mode == "extra_artifact":
        artifact_path = product_root / "unexpected.txt"
        artifact_path.write_text("unexpected\n", encoding="utf-8")
        manifest["artifacts"][artifact_path.name] = backend._file_identity(
            artifact_path
        )
    elif tamper_mode == "forbidden_template":
        artifact_path = product_root / "signal_template_e.npy"
        np.save(artifact_path, np.ones((21, 23)), allow_pickle=False)
        manifest["artifacts"][artifact_path.name] = backend._file_identity(
            artifact_path
        )
    else:
        artifact_name = (
            "signal_template_e.npy"
            if tamper_mode == "signal_value"
            else "noise_template_e.npy"
        )
        artifact_path = product_root / artifact_name
        template = np.load(artifact_path, allow_pickle=False)
        if tamper_mode == "signal_value":
            template = np.array(template, copy=True)
            template[0, 0] += 1.0
        else:
            template = template[:-1]
        np.save(artifact_path, template, allow_pickle=False)
        manifest["artifacts"][artifact_name] = backend._file_identity(artifact_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        backend.StampScienceAnalysisContractError,
        match="artifact|template",
    ):
        backend.validate_stamp_science_analysis_v1(product_root)


@pytest.mark.parametrize(
    "tamper_mode",
    [
        "missing_bias",
        "extra_dataset",
        "cube_axis",
        "vector_axis",
        "column_axis",
        "gain_axis",
        "schema_float",
        "complete_integer",
        "background_integer",
        "qa_nonbinary",
        "valid_nonbinary",
        "saturated_nonbinary",
        "cosmic_nonbinary",
    ],
)
def test_publication_requires_exact_representative_frame_contract(
    tmp_path: Path,
    _schema_version_publication_root: Path,
    tamper_mode: str,
) -> None:
    import et_mainsim.stamp_science_analysis as backend

    root = tmp_path / "representative-products"
    shutil.copytree(_schema_version_publication_root, root)
    product_root = root / "science_optimal_aperture_v1"
    representative_path = product_root / "representative_calibrated_frames.h5"
    with h5py.File(representative_path, "r+") as handle:
        if tamper_mode == "missing_bias":
            del handle["bias_level_sum_dn"]
        elif tamper_mode == "extra_dataset":
            handle.create_dataset("unexpected", data=np.asarray([1]))
        elif tamper_mode in {
            "cube_axis",
            "vector_axis",
            "column_axis",
            "gain_axis",
        }:
            dataset_name = {
                "cube_axis": "final_dn",
                "vector_axis": "captured_flux_fraction",
                "column_axis": "column_noise_sum_dn_by_x",
                "gain_axis": "gain_e_per_dn",
            }[tamper_mode]
            value = np.asarray(handle[dataset_name])
            del handle[dataset_name]
            if tamper_mode == "cube_axis":
                value = value[..., None]
            elif tamper_mode == "vector_axis":
                value = value[:, None]
            elif tamper_mode == "column_axis":
                value = value[:, :1]
            else:
                ny, nx = handle["final_dn"].shape[1:]
                value = np.ones((3, ny, nx), dtype=np.float64)
            handle.create_dataset(dataset_name, data=value)
        elif tamper_mode == "schema_float":
            handle.attrs["schema_version"] = 2.0
        elif tamper_mode == "complete_integer":
            handle.attrs["complete"] = 1
        elif tamper_mode == "background_integer":
            handle.attrs["background_realization_used"] = 0
        else:
            dataset_name = {
                "qa_nonbinary": "captured_flux_qa_pass",
                "valid_nonbinary": "valid_mask",
                "saturated_nonbinary": "saturated_mask",
                "cosmic_nonbinary": "cosmic_mask",
            }[tamper_mode]
            value = np.asarray(handle[dataset_name], dtype=np.uint8)
            value.flat[0] = 2
            del handle[dataset_name]
            handle.create_dataset(dataset_name, data=value)

    manifest_path = product_root / "analysis_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"][representative_path.name] = backend._file_identity(
        representative_path
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(
        backend.StampScienceAnalysisContractError,
        match="representative calibrated-frame product",
    ):
        backend.validate_stamp_science_analysis_v1(product_root)


@pytest.mark.parametrize(
    "tamper_mode",
    [
        "selection_policy",
        "clean_definition",
        "selection_role",
        "raw_start",
        "raw_stop",
        "input_path",
        "input_sha",
        "string_dtype",
        "contract_frame_length",
        "contract_field_type",
        "contract_raw_start_type",
        "contract_shard_index",
        "contract_selection_role",
        "contract_raw_start",
        "contract_input_path",
        "contract_input_sha",
        "contract_frame_extra_field",
        "index_time_mismatch",
        "uint64_overflow",
        "whitespace_path",
        "noncanonical_path",
        "duplicate_shard_path",
    ],
)
def test_publication_binds_representative_frame_provenance_to_contract(
    tmp_path: Path,
    _schema_version_publication_root: Path,
    tamper_mode: str,
) -> None:
    import et_mainsim.stamp_science_analysis as backend

    root = tmp_path / "representative-provenance-products"
    shutil.copytree(_schema_version_publication_root, root)
    product_name = "science_optimal_aperture_v1"
    product_root = root / product_name
    representative_path = product_root / "representative_calibrated_frames.h5"
    manifest_path = product_root / "analysis_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    contract = manifest["contract"]
    representative_contract = contract["representative_calibrated_frames"]
    contract_changed = tamper_mode in {
        "selection_policy",
        "clean_definition",
        "contract_frame_length",
        "contract_field_type",
        "contract_raw_start_type",
        "contract_shard_index",
        "contract_selection_role",
        "contract_raw_start",
        "contract_input_path",
        "contract_input_sha",
        "contract_frame_extra_field",
        "index_time_mismatch",
        "uint64_overflow",
        "whitespace_path",
        "noncanonical_path",
        "duplicate_shard_path",
    }
    representative_changed = tamper_mode in {
        "selection_role",
        "raw_start",
        "raw_stop",
        "input_path",
        "input_sha",
        "string_dtype",
        "index_time_mismatch",
        "uint64_overflow",
        "whitespace_path",
        "noncanonical_path",
        "duplicate_shard_path",
    }
    if tamper_mode == "selection_policy":
        representative_contract["selection_policy"] = "forged_selection_policy"
    elif tamper_mode == "clean_definition":
        representative_contract["clean_definition"] = "forged_clean_definition"
    elif tamper_mode == "contract_frame_length":
        representative_contract["frames"].pop()
    elif tamper_mode == "contract_field_type":
        representative_contract["frames"][0]["input_shard_index"] = 0.0
    elif tamper_mode == "contract_raw_start_type":
        representative_contract["frames"][0]["raw_frame_start_index"] = 0.0
    elif tamper_mode == "contract_shard_index":
        representative_contract["frames"][0]["input_shard_index"] = 999
    elif tamper_mode == "contract_selection_role":
        representative_contract["frames"][0]["selection_role"] = "forged_role"
    elif tamper_mode == "contract_raw_start":
        representative_contract["frames"][0]["raw_frame_start_index"] = 1_000
    elif tamper_mode == "contract_input_path":
        representative_contract["frames"][0]["input_shard_path"] = (
            "/forged/input.h5"
        )
    elif tamper_mode == "contract_input_sha":
        representative_contract["frames"][0]["input_shard_semantic_sha256"] = (
            "0" * 64
        )
    elif tamper_mode == "contract_frame_extra_field":
        representative_contract["frames"][0]["unexpected"] = "unbound"
    elif tamper_mode == "index_time_mismatch":
        frame = representative_contract["frames"][0]
        forged_start = contract["input_raw_shards"][0][
            "first_raw_frame_start"
        ] + 1
        frame["raw_frame_start_index"] = forged_start
        with h5py.File(representative_path, "r+") as handle:
            handle["raw_frame_start_index"][0] = forged_start
            handle["raw_frame_stop_index_exclusive"][0] = forged_start + 1
    elif tamper_mode == "uint64_overflow":
        overflow_start = int(np.iinfo(np.int64).max) + 1
        raw_shard = contract["input_raw_shards"][0]
        raw_shard["first_raw_frame_start"] = overflow_start
        raw_shard["last_raw_frame_stop"] = overflow_start + 6
        representative_contract["frames"][0][
            "raw_frame_start_index"
        ] = overflow_start
        with h5py.File(representative_path, "r+") as handle:
            starts = np.asarray(handle["raw_frame_start_index"], dtype=np.uint64)
            stops = np.asarray(
                handle["raw_frame_stop_index_exclusive"],
                dtype=np.uint64,
            )
            starts[0] = overflow_start
            stops[0] = overflow_start + 1
            del handle["raw_frame_start_index"]
            del handle["raw_frame_stop_index_exclusive"]
            handle.create_dataset("raw_frame_start_index", data=starts)
            handle.create_dataset("raw_frame_stop_index_exclusive", data=stops)
    elif tamper_mode in {"whitespace_path", "noncanonical_path"}:
        forged_path = (
            " "
            if tamper_mode == "whitespace_path"
            else "/tmp/../tmp/forged-input.h5"
        )
        contract["input_raw_shards"][0]["path"] = forged_path
        selected = [
            index
            for index, frame in enumerate(representative_contract["frames"])
            if frame["input_shard_index"] == 0
        ]
        for index in selected:
            representative_contract["frames"][index][
                "input_shard_path"
            ] = forged_path
        with h5py.File(representative_path, "r+") as handle:
            for index in selected:
                handle["input_shard_path"][index] = forged_path
    elif tamper_mode == "duplicate_shard_path":
        duplicate_path = contract["input_raw_shards"][0]["path"]
        contract["input_raw_shards"][1]["path"] = duplicate_path
        selected = [
            index
            for index, frame in enumerate(representative_contract["frames"])
            if frame["input_shard_index"] == 1
        ]
        for index in selected:
            representative_contract["frames"][index][
                "input_shard_path"
            ] = duplicate_path
        with h5py.File(representative_path, "r+") as handle:
            for index in selected:
                handle["input_shard_path"][index] = duplicate_path
    else:
        with h5py.File(representative_path, "r+") as handle:
            if tamper_mode == "selection_role":
                handle["selection_role"][0] = "forged_role"
            elif tamper_mode == "raw_start":
                handle["raw_frame_start_index"][0] = 1_000
                handle["raw_frame_stop_index_exclusive"][0] = 1_001
            elif tamper_mode == "raw_stop":
                start = int(handle["raw_frame_start_index"][0])
                handle["raw_frame_stop_index_exclusive"][0] = start + 2
            elif tamper_mode == "input_path":
                handle["input_shard_path"][0] = "/forged/input.h5"
            elif tamper_mode == "input_sha":
                handle["input_shard_semantic_sha256"][0] = "0" * 64
            else:
                del handle["selection_role"]
                handle.create_dataset("selection_role", data=np.arange(3))

    if contract_changed:
        hdf_path = product_root / "photometry.h5"
        with h5py.File(hdf_path, "r+") as handle:
            del handle["analysis_contract_json"]
            handle.create_dataset(
                "analysis_contract_json",
                data=np.bytes_(
                    json.dumps(
                        contract,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ).encode("utf-8")
                ),
            )
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        _rehash_product_artifact(
            root,
            product_name=product_name,
            artifact_name="photometry.h5",
        )
    if representative_changed:
        _rehash_product_artifact(
            root,
            product_name=product_name,
            artifact_name="representative_calibrated_frames.h5",
        )

    for validator, path in (
        (backend.validate_stamp_science_analysis_v1, product_root),
        (backend.validate_stamp_science_analysis_product_set_v1, root),
    ):
        with pytest.raises(
            backend.StampScienceAnalysisContractError,
            match="representative calibrated-frame provenance",
        ):
            validator(path)


@pytest.mark.parametrize(
    "tamper_mode",
    [
        "contract_cadences",
        "policy_factors",
        "policy_raw_exposure",
        "policy_background_strategy",
        "captured_cadences",
        "cadence_extra_dataset",
        "cdpp_cadences",
        "quality_cadences",
        "coadd_factor_float",
        "exposure_mismatch",
        "raw_span",
        "uncertainty_factor",
    ],
)
def test_publication_cross_binds_every_cadence_contract_surface(
    tmp_path: Path,
    _schema_version_publication_root: Path,
    tamper_mode: str,
) -> None:
    from astropy.table import Table
    import et_mainsim.stamp_science_analysis as backend

    root = tmp_path / "cadence-products"
    shutil.copytree(_schema_version_publication_root, root)
    product_root = root / "science_optimal_aperture_v1"
    manifest_path = product_root / "analysis_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    contract = manifest["contract"]
    changed_artifacts: set[str] = set()
    hdf_path = product_root / "photometry.h5"
    reference_path = product_root / "reference_lightcurve.ecsv"

    if tamper_mode == "contract_cadences":
        contract["cadence_seconds"] = [10]
    elif tamper_mode == "policy_factors":
        contract["policy"]["coadd_factors"] = [1]
    elif tamper_mode == "policy_raw_exposure":
        contract["policy"]["raw_exposure_seconds"] = 5.0
    elif tamper_mode == "policy_background_strategy":
        contract["policy"]["photometry"]["background_strategy"] = (
            "delivered_expectation_only"
        )
    elif tamper_mode == "captured_cadences":
        captured = contract["captured_flux_qa"]["cadences"]
        captured["31s"] = captured.pop("30s")
    elif tamper_mode == "quality_cadences":
        quality_path = product_root / "quality_summary.json"
        quality = json.loads(quality_path.read_text(encoding="utf-8"))
        quality["cadences"].pop("30s")
        quality_path.write_text(json.dumps(quality), encoding="utf-8")
        changed_artifacts.add(quality_path.name)
    elif tamper_mode == "cdpp_cadences":
        cdpp_path = product_root / "cdpp.json"
        cdpp = json.loads(cdpp_path.read_text(encoding="utf-8"))
        cdpp["cadences"].pop("30s")
        cdpp_path.write_text(json.dumps(cdpp), encoding="utf-8")
        cdpp_table_path = product_root / "cdpp.ecsv"
        table = Table.read(cdpp_table_path, format="ascii.ecsv")
        table = table[np.asarray(table["cadence_seconds"]) != 30]
        table.write(cdpp_table_path, format="ascii.ecsv", overwrite=True)
        with h5py.File(hdf_path, "r+") as handle:
            del handle["cdpp_json"]
            handle.create_dataset(
                "cdpp_json",
                data=np.bytes_(
                    json.dumps(
                        cdpp,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ).encode("utf-8")
                ),
            )
        changed_artifacts.update({"photometry.h5", "cdpp.json", "cdpp.ecsv"})
    else:
        with h5py.File(hdf_path, "r+") as handle:
            group = handle["cadences/30s"]
            if tamper_mode == "cadence_extra_dataset":
                group.create_dataset(
                    "unexpected",
                    data=np.zeros(group["time_start_seconds"].shape),
                )
            elif tamper_mode == "coadd_factor_float":
                group.attrs["coadd_factor"] = 3.0
            elif tamper_mode == "uncertainty_factor":
                group["uncertainty_coadd_factor"][:] = 1
            elif tamper_mode == "raw_span":
                count = group["raw_frame_start_index"].shape[0]
                starts = np.arange(count, dtype=np.int64)
                stops = starts + 1
                group["raw_frame_start_index"][:] = starts
                group["raw_frame_stop_index_exclusive"][:] = stops
                table = Table.read(reference_path, format="ascii.ecsv")
                selected = np.asarray(table["cadence_seconds"]) == 30
                table["raw_frame_start_index"][selected] = starts
                table["raw_frame_stop_index_exclusive"][selected] = stops
                table.write(reference_path, format="ascii.ecsv", overwrite=True)
                changed_artifacts.add(reference_path.name)
            else:
                count = group["exposure_seconds"].shape[0]
                exposure = np.full(count, 20.0)
                time = np.arange(count, dtype=np.float64) * 20.0
                group["time_start_seconds"][:] = time
                group["exposure_seconds"][:] = exposure
                for integrated_name, rate_name in (
                    (
                        "flux_expectation_bgsub_e",
                        "flux_expectation_bgsub_e_per_s",
                    ),
                    ("flux_local_bgsub_e", "flux_local_bgsub_e_per_s"),
                    (
                        "fitted_flux_expectation_e",
                        "fitted_flux_expectation_e_per_s",
                    ),
                    ("fitted_flux_local_e", "fitted_flux_local_e_per_s"),
                    ("model_flux_uncertainty_e", "model_flux_uncertainty_e_per_s"),
                ):
                    group[rate_name][:] = (
                        np.asarray(group[integrated_name]) / exposure
                    )
                table = Table.read(reference_path, format="ascii.ecsv")
                selected = np.asarray(table["cadence_seconds"]) == 30
                table["time_start_seconds"][selected] = time
                table["exposure_seconds"][selected] = exposure
                table["flux_expectation_bgsub_e_per_s"][selected] = np.asarray(
                    group["flux_expectation_bgsub_e_per_s"]
                )
                table["fitted_flux_expectation_e_per_s"][selected] = np.asarray(
                    group["fitted_flux_expectation_e_per_s"]
                )
                table.write(reference_path, format="ascii.ecsv", overwrite=True)
                changed_artifacts.add(reference_path.name)
        changed_artifacts.add("photometry.h5")

    if tamper_mode in {
        "contract_cadences",
        "policy_factors",
        "policy_raw_exposure",
        "policy_background_strategy",
        "captured_cadences",
    }:
        with h5py.File(hdf_path, "r+") as handle:
            del handle["analysis_contract_json"]
            handle.create_dataset(
                "analysis_contract_json",
                data=np.bytes_(
                    json.dumps(
                        contract,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ).encode("utf-8")
                ),
            )
        changed_artifacts.add("photometry.h5")

    for artifact_name in changed_artifacts:
        manifest["artifacts"][artifact_name] = backend._file_identity(
            product_root / artifact_name
        )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(
        backend.StampScienceAnalysisContractError,
        match="cadence|policy|background strategy",
    ):
        backend.validate_stamp_science_analysis_v1(product_root)


@pytest.mark.parametrize(
    "tamper_mode",
    [
        "raw_shard_gap",
        "shifted_cadence_time",
        "fitted_model",
        "observed_flux",
        "positive_variance",
        "missing_cdpp_policy",
        "changed_cdpp_window",
        "cdpp_payload",
    ],
)
def test_publication_recomputes_authoritative_analysis_semantics(
    tmp_path: Path,
    _schema_version_publication_root: Path,
    tamper_mode: str,
) -> None:
    from astropy.table import Table
    import et_mainsim.stamp_science_analysis as backend

    root = tmp_path / "recomputed-semantics-products"
    shutil.copytree(_schema_version_publication_root, root)
    product_name = "science_optimal_aperture_v1"
    product_root = root / product_name
    manifest_path = product_root / "analysis_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    contract = manifest["contract"]
    hdf_path = product_root / "photometry.h5"
    changed_artifacts: set[str] = {"photometry.h5"}
    table_updates: dict[str, dict[str, np.ndarray]] = {}
    contract_changed = tamper_mode in {
        "raw_shard_gap",
        "missing_cdpp_policy",
        "changed_cdpp_window",
    }

    if tamper_mode == "raw_shard_gap":
        first_shard = contract["input_raw_shards"][0]
        first_shard["last_raw_frame_stop"] -= 1
        first_shard["last_time_end_seconds"] -= 10.0
    elif tamper_mode == "missing_cdpp_policy":
        del contract["policy"]["photometry"]["minimum_coverage_fraction"]
    elif tamper_mode == "changed_cdpp_window":
        contract["policy"]["photometry"]["cdpp_windows_minutes"] = [2]

    with h5py.File(hdf_path, "r+") as handle:
        group = handle["cadences/30s"]
        exposure = np.asarray(group["exposure_seconds"], dtype=np.float64)
        if tamper_mode == "shifted_cadence_time":
            shifted = np.asarray(
                group["time_start_seconds"], dtype=np.float64
            ) + 5.0
            group["time_start_seconds"][:] = shifted
            for artifact_name in (
                "photometry.ecsv",
                "reference_lightcurve.ecsv",
                "centroid_quality.ecsv",
            ):
                table_updates[artifact_name] = {
                    "time_start_seconds": shifted,
                }
        elif tamper_mode == "fitted_model":
            flux = np.asarray(
                group["flux_expectation_bgsub_e"], dtype=np.float64
            )
            valid = np.asarray(group["aperture_valid"], dtype=bool)
            fitted = np.asarray(
                group["fitted_flux_expectation_e"], dtype=np.float64
            ) * 1.01
            residual = np.full(fitted.shape, np.nan, dtype=np.float64)
            residual[valid] = flux[valid] - fitted[valid]
            residual_ppm = np.full(fitted.shape, np.nan, dtype=np.float64)
            residual_ppm[valid] = (
                residual[valid] / fitted[valid] * 1_000_000.0
            )
            values = {
                "fitted_flux_expectation_e": fitted,
                "residual_expectation_e": residual,
                "residual_expectation_ppm": residual_ppm,
                "fitted_flux_expectation_e_per_s": fitted / exposure,
            }
            for name, value in values.items():
                group[name][:] = value
            group.attrs["fit_scale_expectation_e_per_raw_factor"] = (
                float(group.attrs["fit_scale_expectation_e_per_raw_factor"])
                * 1.01
            )
            table_updates["photometry.ecsv"] = values
            table_updates["reference_lightcurve.ecsv"] = values
        elif tamper_mode == "observed_flux":
            flux = np.asarray(
                group["flux_expectation_bgsub_e"], dtype=np.float64
            )
            flux[np.isfinite(flux)] += 7.0
            rate = flux / exposure
            group["flux_expectation_bgsub_e"][:] = flux
            group["flux_expectation_bgsub_e_per_s"][:] = rate
            values = {
                "flux_expectation_bgsub_e": flux,
                "flux_expectation_bgsub_e_per_s": rate,
            }
            table_updates["photometry.ecsv"] = values
            table_updates["reference_lightcurve.ecsv"] = values
        elif tamper_mode == "positive_variance":
            background_variance = np.asarray(
                group["background_variance_e2"], dtype=np.float64
            ) + 1.0
            source_variance = np.asarray(
                group["source_variance_e2"], dtype=np.float64
            )
            read_variance = np.asarray(
                group["read_variance_e2"], dtype=np.float64
            )
            quantization_variance = np.asarray(
                group["quantization_variance_e2"], dtype=np.float64
            )
            uncertainty = np.sqrt(
                source_variance
                + background_variance
                + read_variance
                + quantization_variance
            )
            uncertainty_valid = np.asarray(
                group["uncertainty_valid"], dtype=bool
            )
            uncertainty[~uncertainty_valid] = np.nan
            values = {
                "background_variance_e2": background_variance,
                "flux_uncertainty_e": uncertainty,
                "model_flux_uncertainty_e": uncertainty,
                "model_flux_uncertainty_e_per_s": uncertainty / exposure,
            }
            for name, value in values.items():
                group[name][:] = value
            table_updates["photometry.ecsv"] = values
        elif tamper_mode == "cdpp_payload":
            payload = json.loads(handle["cdpp_json"][()].decode("utf-8"))
            row = payload["cadences"]["30s"]["expectation_background"][
                "binned_rows"
            ][0]
            row["model_flux_rate_e_per_s"] *= 1.1
            del handle["cdpp_json"]
            handle.create_dataset(
                "cdpp_json",
                data=np.bytes_(
                    json.dumps(
                        payload,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ).encode("utf-8")
                ),
            )
            (product_root / "cdpp.json").write_text(
                json.dumps(payload),
                encoding="utf-8",
            )
            changed_artifacts.add("cdpp.json")

        if contract_changed:
            del handle["analysis_contract_json"]
            handle.create_dataset(
                "analysis_contract_json",
                data=np.bytes_(
                    json.dumps(
                        contract,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ).encode("utf-8")
                ),
            )

    for artifact_name, updates in table_updates.items():
        table_path = product_root / artifact_name
        table = Table.read(table_path, format="ascii.ecsv")
        selected = np.asarray(table["cadence_seconds"]) == 30
        for name, value in updates.items():
            table[name][selected] = value
        table.write(table_path, format="ascii.ecsv", overwrite=True)
        changed_artifacts.add(artifact_name)

    if contract_changed:
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    for artifact_name in sorted(changed_artifacts):
        _rehash_product_artifact(
            root,
            product_name=product_name,
            artifact_name=artifact_name,
        )

    with pytest.raises(
        backend.StampScienceAnalysisContractError,
        match="authoritative analysis semantics",
    ):
        backend.validate_stamp_science_analysis_v1(product_root)


@pytest.mark.parametrize(
    "tamper_mode",
    [
        "flux_uncertainty_model",
        "noise_declaration",
        "noise_numeric_type",
        "uncertainty_metadata_types",
        "variance_dtype",
        "formal_profile_policy",
    ],
)
def test_publication_freezes_uncertainty_declarations_and_formal_policy(
    tmp_path: Path,
    _schema_version_publication_root: Path,
    tamper_mode: str,
) -> None:
    from astropy.table import Table
    import et_mainsim.stamp_science_analysis as backend

    root = tmp_path / "frozen-uncertainty-products"
    shutil.copytree(_schema_version_publication_root, root)
    product_names = (
        "reference_fixed13_v1",
        "science_optimal_aperture_v1",
    )
    science_name = "science_optimal_aperture_v1"
    science_root = root / science_name

    if tamper_mode in {
        "flux_uncertainty_model",
        "noise_declaration",
        "noise_numeric_type",
        "formal_profile_policy",
    }:
        top_manifest_path = root / "product_set_manifest.json"
        top_manifest = json.loads(top_manifest_path.read_text(encoding="utf-8"))
        target_names = (
            (science_name,)
            if tamper_mode == "flux_uncertainty_model"
            else product_names
        )
        if tamper_mode == "noise_declaration":
            top_noise = top_manifest["analysis_context"]["noise_model"]
            top_noise.update(
                {
                    "schema_id": "forged.noise.v1",
                    "source": "forged-source",
                    "quantization_formula": "forged-formula",
                }
            )
        elif tamper_mode == "noise_numeric_type":
            top_noise = top_manifest["analysis_context"]["noise_model"]
            top_noise["read_noise_e_per_raw_pixel"] = 1
            top_noise["quantization_noise_e_per_raw_pixel"] = 0
        elif tamper_mode == "formal_profile_policy":
            top_manifest["formal_profile_id"] = (
                backend.STAMP_SCIENCE_FORMAL_PROFILE_ID
            )
            top_manifest["analysis_context"]["formal_profile_id"] = (
                backend.STAMP_SCIENCE_FORMAL_PROFILE_ID
            )
        top_manifest_path.write_text(json.dumps(top_manifest), encoding="utf-8")

        for product_name in target_names:
            child_manifest = json.loads(
                (root / product_name / "analysis_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            contract = child_manifest["contract"]
            if tamper_mode == "flux_uncertainty_model":
                contract["flux_uncertainty_model"] = {
                    "schema_id": "forged.uncertainty.v1",
                    "authoritative_dataset": "forged",
                }
            elif tamper_mode == "noise_declaration":
                contract["analysis_context"]["noise_model"].update(
                    {
                        "schema_id": "forged.noise.v1",
                        "source": "forged-source",
                        "quantization_formula": "forged-formula",
                    }
                )
            elif tamper_mode == "noise_numeric_type":
                noise = contract["analysis_context"]["noise_model"]
                noise["read_noise_e_per_raw_pixel"] = 1
                noise["quantization_noise_e_per_raw_pixel"] = 0
            else:
                contract["analysis_context"]["formal_profile_id"] = (
                    backend.STAMP_SCIENCE_FORMAL_PROFILE_ID
                )
            _replace_product_contract(
                root,
                product_name=product_name,
                contract=contract,
            )
    elif tamper_mode == "uncertainty_metadata_types":
        with h5py.File(science_root / "photometry.h5", "r+") as handle:
            group = handle["cadences/10s"]
            metadata = json.loads(group.attrs["uncertainty_model_json"])
            metadata["schema_version"] = 1.0
            metadata["dark_current_counted_once_via_background_expectation"] = 1
            group.attrs.modify(
                "uncertainty_model_json",
                json.dumps(metadata, sort_keys=True, separators=(",", ":")),
            )
        _rehash_product_artifact(
            root,
            product_name=science_name,
            artifact_name="photometry.h5",
        )
    else:
        with h5py.File(science_root / "photometry.h5", "r+") as handle:
            for group in handle["cadences"].values():
                values = np.asarray(group["read_variance_e2"], dtype=np.int64)
                del group["read_variance_e2"]
                group.create_dataset("read_variance_e2", data=values)
        table_path = science_root / "photometry.ecsv"
        table = Table.read(table_path, format="ascii.ecsv")
        table["read_variance_e2"] = np.asarray(
            table["read_variance_e2"], dtype=np.int64
        )
        table.write(table_path, format="ascii.ecsv", overwrite=True)
        for artifact_name in ("photometry.h5", "photometry.ecsv"):
            _rehash_product_artifact(
                root,
                product_name=science_name,
                artifact_name=artifact_name,
            )

    for validator, path in (
        (backend.validate_stamp_science_analysis_v1, science_root),
        (backend.validate_stamp_science_analysis_product_set_v1, root),
    ):
        with pytest.raises(
            backend.StampScienceAnalysisContractError,
            match="uncertainty|noise|formal analysis policy",
        ):
            validator(path)


@pytest.mark.parametrize("field", ["source", "quantization_formula"])
def test_formal_noise_declaration_uses_frozen_text(field: str) -> None:
    import et_mainsim.stamp_science_analysis as backend

    noise_model = {
        "schema_id": "et_mainsim.formal_stamp_noise_parameters.v1",
        "source": "production_manifest.simulation_spec_base.readout",
        "read_noise_e_per_raw_pixel": 1.0,
        "quantization_noise_e_per_raw_pixel": 0.0,
        "quantization_formula": (
            "gain_electrons_per_adu/sqrt(12) when ADC rounding is enabled"
        ),
    }
    noise_model[field] = "forged"
    with pytest.raises(
        backend.StampScienceAnalysisContractError,
        match="noise model",
    ):
        backend._analysis_noise_parameters_v1(
            {
                "analysis_context": {
                    "formal_profile_id": backend.STAMP_SCIENCE_FORMAL_PROFILE_ID,
                    "noise_model": noise_model,
                }
            }
        )


@pytest.mark.parametrize("tamper_mode", ["nan", "positive_drift"])
def test_publication_and_product_set_bind_authoritative_q_content_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper_mode: str,
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

    publication = backend.analyze_stamp_science_product_set_v1(
        _request(
            tmp_path,
            raw_paths=raw_paths,
            coadd_paths=coadd_paths,
            q=q,
            output_name="q-bound-products",
        )
    )
    product = publication.science_optimal_aperture
    with h5py.File(product.hdf5_path, "r+") as handle:
        dataset = handle["raw_relative_flux"]
        if tamper_mode == "nan":
            dataset[0] = np.nan
        else:
            dataset[0] = float(dataset[0]) + 0.125

    child_manifest = json.loads(
        product.manifest_path.read_text(encoding="utf-8")
    )
    child_manifest["artifacts"]["photometry.h5"] = backend._file_identity(
        product.hdf5_path
    )
    product.manifest_path.write_text(
        json.dumps(child_manifest),
        encoding="utf-8",
    )
    product_set_manifest = json.loads(
        publication.manifest_path.read_text(encoding="utf-8")
    )
    product_set_manifest["products"]["science_optimal_aperture_v1"][
        "analysis_manifest"
    ] = backend._file_identity(product.manifest_path)
    publication.manifest_path.write_text(
        json.dumps(product_set_manifest),
        encoding="utf-8",
    )

    for validator, path in (
        (backend.validate_stamp_science_analysis_v1, product.output_dir),
        (
            backend.validate_stamp_science_analysis_product_set_v1,
            publication.output_dir,
        ),
    ):
        with pytest.raises(
            backend.StampScienceAnalysisContractError,
            match="authoritative HDF5 raw_relative_flux",
        ):
            validator(path)


@pytest.mark.parametrize(
    "tamper_mode",
    [
        "json_arbitrary",
        "json_numeric_type",
        "ecsv_schema",
        "ecsv_dtype",
        "ecsv_value",
        "negative_observed",
        "negative_residual",
        "binned_schema",
        "binned_invariant",
        "expectation_status",
        "local_status_wrong_contract",
    ],
)
def test_publication_and_product_set_bind_portable_cdpp_to_authoritative_hdf5(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper_mode: str,
) -> None:
    import photsim7.aperture as legacy_aperture
    from astropy.table import Table

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

    publication = backend.analyze_stamp_science_product_set_v1(
        _request(
            tmp_path,
            raw_paths=raw_paths,
            coadd_paths=coadd_paths,
            q=q,
            output_name="cdpp-bound-products",
        )
    )
    product = publication.science_optimal_aperture
    artifact_paths: list[Path]
    if tamper_mode == "json_arbitrary":
        artifact_path = product.cdpp_path
        artifact_path.write_text(
            json.dumps({"arbitrary": "nonempty"}),
            encoding="utf-8",
        )
        artifact_paths = [artifact_path]
    elif tamper_mode == "json_numeric_type":
        payload = json.loads(product.cdpp_path.read_text(encoding="utf-8"))
        cadence_name = min(
            payload["cadences"],
            key=lambda name: int(name.removesuffix("s")),
        )
        payload["cadences"][cadence_name]["coadd_factor"] = float(
            payload["cadences"][cadence_name]["coadd_factor"]
        )
        product.cdpp_path.write_text(json.dumps(payload), encoding="utf-8")
        artifact_paths = [product.cdpp_path]
    elif tamper_mode in {"ecsv_schema", "ecsv_dtype", "ecsv_value"}:
        artifact_path = product.output_dir / "cdpp.ecsv"
        table = Table.read(artifact_path, format="ascii.ecsv")
        if tamper_mode == "ecsv_schema":
            table.meta["schema_id"] = "example.invalid_cdpp_table.v1"
        elif tamper_mode == "ecsv_dtype":
            table["total_bin_count"] = np.asarray(
                table["total_bin_count"],
                dtype=np.float64,
            )
        else:
            table["observed_cdpp_ppm"][0] += 1.0
        table.write(artifact_path, format="ascii.ecsv", overwrite=True)
        artifact_paths = [artifact_path]
    else:
        with h5py.File(product.hdf5_path, "r+") as handle:
            payload = json.loads(handle["cdpp_json"][()].decode("utf-8"))
            cadence_name = min(
                payload["cadences"],
                key=lambda name: int(name.removesuffix("s")),
            )
            cadence_seconds = int(cadence_name.removesuffix("s"))
            cadence = payload["cadences"][cadence_name]
            expectation = cadence["expectation_background"]
            window_name = min(
                expectation["metrics_by_window_minutes"],
                key=int,
            )
            table = Table.read(
                product.output_dir / "cdpp.ecsv",
                format="ascii.ecsv",
            )
            if tamper_mode in {"negative_observed", "negative_residual"}:
                column = (
                    "observed_cdpp_ppm"
                    if tamper_mode == "negative_observed"
                    else "residual_cdpp_ppm"
                )
                expectation["metrics_by_window_minutes"][window_name][column] = -1.0
                table[column][0] = -1.0
            elif tamper_mode == "binned_schema":
                expectation["binned_rows"][0].pop("coverage_fraction")
            elif tamper_mode == "binned_invariant":
                expectation["binned_rows"][0]["coverage_fraction"] = -1.0
            elif tamper_mode == "expectation_status":
                cadence["expectation_background"] = {
                    "status": "not_computed",
                    "reason": "background_strategy_delivered_expectation_only",
                }
                table = table[
                    ~(
                        (np.asarray(table["cadence_seconds"]) == cadence_seconds)
                        & (
                            np.asarray(table["background_estimator"])
                            == "expectation_background"
                        )
                    )
                ]
            else:
                cadence["local_background"] = {
                    "status": "not_computed",
                    "reason": "background_strategy_delivered_expectation_only",
                }
                table = table[
                    ~(
                        (np.asarray(table["cadence_seconds"]) == cadence_seconds)
                        & (
                            np.asarray(table["background_estimator"])
                            == "local_background"
                        )
                    )
                ]
            del handle["cdpp_json"]
            handle.create_dataset(
                "cdpp_json",
                data=np.bytes_(
                    json.dumps(
                        payload,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ).encode("utf-8")
                ),
            )
        product.cdpp_path.write_text(
            json.dumps(payload),
            encoding="utf-8",
        )
        table.write(
            product.output_dir / "cdpp.ecsv",
            format="ascii.ecsv",
            overwrite=True,
        )
        artifact_paths = [
            product.hdf5_path,
            product.cdpp_path,
            product.output_dir / "cdpp.ecsv",
        ]

    child_manifest = json.loads(
        product.manifest_path.read_text(encoding="utf-8")
    )
    for artifact_path in artifact_paths:
        child_manifest["artifacts"][artifact_path.name] = backend._file_identity(
            artifact_path
        )
    product.manifest_path.write_text(
        json.dumps(child_manifest),
        encoding="utf-8",
    )
    product_set_manifest = json.loads(
        publication.manifest_path.read_text(encoding="utf-8")
    )
    product_set_manifest["products"]["science_optimal_aperture_v1"][
        "analysis_manifest"
    ] = backend._file_identity(product.manifest_path)
    publication.manifest_path.write_text(
        json.dumps(product_set_manifest),
        encoding="utf-8",
    )

    for validator, path in (
        (backend.validate_stamp_science_analysis_v1, product.output_dir),
        (
            backend.validate_stamp_science_analysis_product_set_v1,
            publication.output_dir,
        ),
    ):
        with pytest.raises(
            backend.StampScienceAnalysisContractError,
            match=(
                "CDPP background role"
                if tamper_mode
                in {"expectation_status", "local_status_wrong_contract"}
                else "CDPP"
            ),
        ):
            validator(path)


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


def test_published_analysis_readback_rejects_negative_variance_component_even_when_total_matches(
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
    with h5py.File(publication.hdf5_path, "r+") as handle:
        cadence = handle["cadences/10s"]
        source = cadence["source_variance_e2"]
        background = cadence["background_variance_e2"]
        original_source = float(source[0])
        source[0] = -1.0
        background[0] = float(background[0]) + original_source + 1.0

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
        match="variance components are invalid",
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


@pytest.mark.parametrize(
    "tamper_mode",
    [
        "metadata",
        "column_order",
        "row_order",
        "row_count",
        "cadence_axis",
        "dtype",
        "value",
        "local_valid",
        "qa_nonbinary",
    ],
)
def test_public_readback_binds_portable_photometry_to_authoritative_hdf5(
    tmp_path: Path,
    _schema_version_publication_root: Path,
    tamper_mode: str,
) -> None:
    from astropy.table import Table
    import et_mainsim.stamp_science_analysis as backend

    root = tmp_path / "photometry-parity-products"
    shutil.copytree(_schema_version_publication_root, root)
    product_name = "science_optimal_aperture_v1"
    product_root = root / product_name
    artifact_path = product_root / "photometry.ecsv"
    table = Table.read(artifact_path, format="ascii.ecsv")
    if tamper_mode == "metadata":
        table.meta["unbound_extension"] = "accepted-by-subset-validation"
    elif tamper_mode == "column_order":
        table = table[list(reversed(table.colnames))]
    elif tamper_mode == "row_order":
        order = np.arange(len(table))
        order[:2] = order[1::-1]
        table = table[order]
    elif tamper_mode == "row_count":
        table = table[:-1]
    elif tamper_mode == "cadence_axis":
        table["cadence_seconds"][0] += 1
    elif tamper_mode == "dtype":
        table["quality_bitmask"] = np.asarray(
            table["quality_bitmask"],
            dtype=np.float64,
        )
    elif tamper_mode == "value":
        table["centroid_x"][0] += 0.25
    elif tamper_mode == "local_valid":
        table["local_background_valid"][0] = not bool(
            table["local_background_valid"][0]
        )
    else:
        qa = np.ones(len(table), dtype=np.int16)
        qa[0] = 2
        table["captured_flux_qa_pass"] = qa
    table.write(artifact_path, format="ascii.ecsv", overwrite=True)
    _rehash_product_artifact(
        root,
        product_name=product_name,
        artifact_name="photometry.ecsv",
    )

    for validator, path in (
        (backend.validate_stamp_science_analysis_v1, product_root),
        (backend.validate_stamp_science_analysis_product_set_v1, root),
    ):
        with pytest.raises(
            backend.StampScienceAnalysisContractError,
            match="portable photometry ECSV",
        ):
            validator(path)


@pytest.mark.parametrize(
    "tamper_mode",
    [
        "metadata",
        "column_order",
        "row_order",
        "row_count",
        "cadence_axis",
        "dtype",
        "value",
        "valid_nonbinary",
    ],
)
def test_public_readback_binds_centroid_quality_to_authoritative_hdf5(
    tmp_path: Path,
    _schema_version_publication_root: Path,
    tamper_mode: str,
) -> None:
    from astropy.table import Table
    import et_mainsim.stamp_science_analysis as backend

    root = tmp_path / "centroid-parity-products"
    shutil.copytree(_schema_version_publication_root, root)
    product_name = "science_optimal_aperture_v1"
    product_root = root / product_name
    artifact_path = product_root / "centroid_quality.ecsv"
    table = Table.read(artifact_path, format="ascii.ecsv")
    if tamper_mode == "metadata":
        table.meta["unbound_extension"] = "accepted-without-validation"
    elif tamper_mode == "column_order":
        table = table[list(reversed(table.colnames))]
    elif tamper_mode == "row_order":
        order = np.arange(len(table))
        order[:2] = order[1::-1]
        table = table[order]
    elif tamper_mode == "row_count":
        table = table[:-1]
    elif tamper_mode == "cadence_axis":
        table["cadence_seconds"][0] += 1
    elif tamper_mode == "dtype":
        table["quality_bitmask"] = np.asarray(
            table["quality_bitmask"],
            dtype=np.float64,
        )
    elif tamper_mode == "value":
        table["centroid_x_stamp_pixel_0based"][0] += 0.25
    else:
        validity = np.ones(len(table), dtype=np.int16)
        validity[0] = 2
        table["aperture_valid"] = validity
    table.write(artifact_path, format="ascii.ecsv", overwrite=True)
    _rehash_product_artifact(
        root,
        product_name=product_name,
        artifact_name="centroid_quality.ecsv",
    )

    for validator, path in (
        (backend.validate_stamp_science_analysis_v1, product_root),
        (backend.validate_stamp_science_analysis_product_set_v1, root),
    ):
        with pytest.raises(
            backend.StampScienceAnalysisContractError,
            match="centroid-quality ECSV",
        ):
            validator(path)


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


@pytest.mark.parametrize(
    "column",
    [
        "cadence_seconds",
        "raw_frame_start_index",
        "quality_bitmask",
        "aperture_valid",
        "captured_flux_qa_pass",
    ],
)
def test_reference_lightcurve_rejects_coercible_bool_and_integer_dtypes(
    tmp_path: Path,
    _schema_version_publication_root: Path,
    column: str,
) -> None:
    from astropy.table import Table
    import et_mainsim.stamp_science_analysis as backend

    root = tmp_path / "reference-dtype-products"
    shutil.copytree(_schema_version_publication_root, root)
    product_name = "science_optimal_aperture_v1"
    product_root = root / product_name
    artifact_path = product_root / "reference_lightcurve.ecsv"
    table = Table.read(artifact_path, format="ascii.ecsv")
    if column in {"aperture_valid", "captured_flux_qa_pass"}:
        values = np.asarray(table[column], dtype=np.int16)
        true_indices = np.flatnonzero(values == 1)
        assert true_indices.size > 0
        values[int(true_indices[0])] = 2
        table[column] = values
    else:
        table[column] = np.asarray(table[column], dtype=np.float64)
    table.write(artifact_path, format="ascii.ecsv", overwrite=True)
    _rehash_product_artifact(
        root,
        product_name=product_name,
        artifact_name="reference_lightcurve.ecsv",
    )

    for validator, path in (
        (backend.validate_stamp_science_analysis_v1, product_root),
        (backend.validate_stamp_science_analysis_product_set_v1, root),
    ):
        with pytest.raises(
            backend.StampScienceAnalysisContractError,
            match="reference-lightcurve ECSV column dtype differs",
        ):
            validator(path)


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
        raw_stop_index=5_400,
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

    write_gate([*range(6), 89])
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
    assert all("shard_00089" not in str(path) for path in discovery.raw_bundle_paths)
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

    for invalid in ([*range(6)], [*range(6), 88]):
        write_gate(invalid)
        with pytest.raises(
            backend.StampScienceAnalysisContractError,
            match="0..5 and tail shard 89",
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
            match="clean Git commits or verified installed distributions",
        ):
            backend.collect_formal_analysis_code_identity_v1()


def test_formal_code_identity_accepts_verified_installed_distributions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import et_mainsim.stamp_science_analysis as backend

    installed = {
        "et_mainsim": {
            "commit": None,
            "dirty": None,
            "version": "0.1.0",
            "distribution_identity": {
                "schema_id": "et_mainsim.installed_distribution_identity.v1",
                "name": "et-mainsim",
                "version": "0.1.0",
                "record_entry_count": 12,
                "record_tree_sha256": "a" * 64,
            },
        },
        "photsim7": {
            "commit": None,
            "dirty": None,
            "version": "0.2.5",
            "distribution_identity": {
                "schema_id": "et_mainsim.installed_distribution_identity.v1",
                "name": "photsim7",
                "version": "0.2.5",
                "record_entry_count": 34,
                "record_tree_sha256": "b" * 64,
            },
        },
        "runtime": {"python": "3.12.11", "hostname": "request-host"},
    }
    monkeypatch.setattr(backend, "collect_provenance", lambda _root: installed)

    identity = backend.collect_formal_analysis_code_identity_v1()

    assert identity["provenance"] == installed


def test_formal_code_identity_matcher_uses_distribution_content_not_host_or_path() -> None:
    import et_mainsim.stamp_science_analysis as backend

    def identity(
        *,
        root: str,
        host: str,
        et_digest: str = "a" * 64,
        et_record_entry_count: int = 12,
        schema_version: object = 1,
    ):
        return {
            "schema_id": "et_mainsim.formal_analysis_code_identity.v1",
            "schema_version": schema_version,
            "provenance": {
                "et_mainsim": {
                    "root": root,
                    "commit": None,
                    "dirty": None,
                    "version": "0.1.0",
                    "distribution_identity": {
                        "schema_id": (
                            "et_mainsim.installed_distribution_identity.v1"
                        ),
                        "name": "et-mainsim",
                        "version": "0.1.0",
                        "record_entry_count": et_record_entry_count,
                        "record_tree_sha256": et_digest,
                    },
                },
                "photsim7": {
                    "root": f"{root}/photsim7",
                    "commit": None,
                    "dirty": None,
                    "version": "0.2.5",
                    "distribution_identity": {
                        "schema_id": (
                            "et_mainsim.installed_distribution_identity.v1"
                        ),
                        "name": "photsim7",
                        "version": "0.2.5",
                        "record_entry_count": 34,
                        "record_tree_sha256": "b" * 64,
                    },
                },
                "runtime": {"python": "3.12.11", "hostname": host},
            },
            "analysis_dependencies": {"numpy": "2.2.6"},
        }

    recorded = identity(root="/request/site-packages", host="request")
    current = identity(root="/execution/site-packages", host="execution")

    assert backend._formal_code_identity_matches_execution_v1(recorded, current)
    assert not backend._formal_code_identity_matches_execution_v1(
        recorded,
        identity(root="/execution/site-packages", host="execution", et_digest="c" * 64),
    )
    assert not backend._formal_code_identity_matches_execution_v1(
        recorded,
        identity(
            root="/execution/site-packages",
            host="execution",
            et_record_entry_count=13,
        ),
    )
    for coerced_version in (1.0, True):
        malformed_recorded = identity(
            root="/request/site-packages",
            host="request",
            schema_version=coerced_version,
        )
        malformed_current = identity(
            root="/execution/site-packages",
            host="execution",
            schema_version=coerced_version,
        )
        assert not backend._formal_code_identity_matches_execution_v1(
            malformed_recorded,
            current,
        )
        assert not backend._formal_code_identity_matches_execution_v1(
            recorded,
            malformed_current,
        )


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
