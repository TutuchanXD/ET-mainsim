"""Real upstream equivalent products through the downstream application API."""

import json
import pickle

import h5py
import numpy as np
import pytest
from astropy import units as u

from et_mainsim.equivalent_coadd import (
    CONFIG_SCHEMA,
    EquivalentRunConfig,
    build_equivalent_run_plan,
    run_equivalent_coadd,
    verify_equivalent_run,
)


def make_inputs(root, cadence=120, kind="stamp"):
    # Only contracted public upstream APIs; this fixture also works against an
    # installed wheel without upstream checkout scripts or private modules.
    from photsim7.catalogs import PreparedStarCatalog
    from photsim7.data_registry import DataRegistry
    from photsim7.equivalent_coadd import (
        EquivalentCoaddRequest,
        SourceVariabilityDelivery,
    )
    from photsim7.geometry_truth import reference_field_nonphysical_declaration
    from photsim7.source_variability import SourceVariability
    from photsim7.specs import (
        SimulationSpec,
        ObservationSpec,
        CatalogSpec,
        DetectorSpec,
        DetectorResponseSpec,
        PsfSpec,
        ReadoutSpec,
        SkySpec,
        CosmicRaySpec,
        RngSpec,
    )

    assets = root / "assets"
    bundle = assets / "psf/reference"
    bundle.mkdir(parents=True)
    y, x = np.mgrid[-10:11, -10:11].astype(np.float32) / 3
    psf = np.exp(-(x * x + y * y) / (2 * 0.9**2)).astype(np.float32)
    psf /= psf.sum(dtype=np.float64)
    (bundle / "sim_psf_images.pkl").write_bytes(
        pickle.dumps(
            {"images": {0: {3: np.stack([x, y, psf])}}, "angles": np.array([0.0])},
            protocol=4,
        )
    )
    source = SimulationSpec(
        observation=ObservationSpec(
            simulation_name="equivalent-consumer",
            observing_start_date="2026-09-28T00:00:00",
            exposure_duration=10 * u.s,
            readout_duration=0 * u.s,
            observing_duration=2 * cadence * u.s,
            n_frames=None,
            simulation_cadence_mult=1,
            n_raw_frames_per_coadd=1,
        ),
        catalog=CatalogSpec(source_type="prepared"),
        detector=DetectorSpec(
            detector_id="reference",
            shape=(65, 65),
            pixel_width=6.5 * u.um,
            pixel_scale=4.83 * u.arcsec / u.pix,
            n_subpixels=3,
        ),
        detector_response=DetectorResponseSpec(
            enable_inter_pixel_response=False,
            enable_intra_pixel_response=False,
            enable_pixel_phase_response=False,
            scripted_sensitivity_enabled=False,
            whole_pixel_gain_normal_enabled=False,
            whole_pixel_gain_sinusoidal_enabled=False,
            enable_flat_field_correction=False,
        ),
        psf=PsfSpec(
            mode="stamp",
            bundle_name="psf/reference",
            field_id=0,
            field_id_policy="explicit",
            use_jitter_integrated_psf=False,
            compute_device="cpu",
            float_precision=32,
        ),
        readout=ReadoutSpec(
            full_well_electrons=90680 * u.electron,
            gain_electrons_per_adu=1.4 * u.electron / u.adu,
            readout_noise=5 * u.electron / u.pix,
            bias_level_adu=3500 * u.adu,
            column_noise_sigma_adu=0 * u.adu,
            enable_adc_digitization=True,
            adc_bit_depth=16,
            adc_min_value=0,
            adc_round_values=True,
        ),
        sky=SkySpec(
            background_flux=2 * u.electron / u.s / u.pix,
            scattered_light=0.5 * u.electron / u.s / u.pix,
            dark_current=0.25 * u.electron / u.s / u.pix,
            subtract_nonstellar_mean=False,
        ),
        cosmic_rays=CosmicRaySpec(enabled=False),
        rng=RngSpec(run_seed=20260928),
    )
    data = {
        "source_id": [101, 202],
        "x0": [0.0, 1.2],
        "y0": [0.0, 0.4],
        "ra": [10.0, 10.001],
        "dec": [20.0, 20.001],
        "et_mag": [17.0, 18.0],
        "frame_xpix": [32.0, 33.2],
        "frame_ypix": [32.0, 32.4],
        "detector_xpix": [32.0, 33.2],
        "detector_ypix": [32.0, 32.4],
        "detector_id": "reference",
    }
    catalog = PreparedStarCatalog(
        star_data=data,
        metadata={
            "source": {"type": "prepared"},
            "geometry": reference_field_nonphysical_declaration(
                reference_field_angle_deg=0.0, reference_pixel_scale_arcsec_per_pix=4.83
            ),
        },
    )
    delivery = SourceVariabilityDelivery(
        SourceVariability(
            source_ids=[101], relative_flux=[np.repeat([1.0, 0.5], cadence // 10)]
        ),
        raw_sampling_s=10,
        averaging_window_s=10,
        absolute_raw_frame_start_index=60,
        observing_start_date=source.observation.observing_start_date,
    )
    request = EquivalentCoaddRequest.from_source(
        source,
        n_raw=cadence // 10,
        absolute_raw_frame_start_index=60,
        prepared_catalog=catalog,
        variability_delivery=delivery,
        data_registry=DataRegistry(data_root=assets),
        **(
            {"target_source_id": 101, "stamp_shape": (17, 17)}
            if kind == "stamp"
            else {}
        ),
    )
    for name, value in (
        ("request", request.to_json_dict()),
        (
            "catalog",
            {"data": data, "metadata": catalog.metadata, "raw_source_arrays": None},
        ),
        ("variability", delivery.to_json_dict()),
    ):
        (root / f"{name}.json").write_text(json.dumps(value))
    payload = {
        "schema_id": CONFIG_SCHEMA,
        "request_path": "request.json",
        "catalog_path": "catalog.json",
        "variability_path": "variability.json",
        "data_root": "assets",
        "output_root": "results",
        "run_id": "equivalent-test",
        "cpu_threads": 1,
    }
    config = EquivalentRunConfig.from_mapping(payload, base=root)
    return config, request, payload


@pytest.mark.parametrize("cadence", [30, 60, 120, 300])
@pytest.mark.parametrize("kind", ["stamp", "full_frame"])
def test_real_upstream_products_are_consumed_and_resumed(tmp_path, cadence, kind):
    config, request, _ = make_inputs(tmp_path, cadence, kind)
    plan = build_equivalent_run_plan(config)
    assert plan["cadence_s"] == cadence
    assert not (config.output_root / config.run_id).exists()
    result = run_equivalent_coadd(config)
    assert result["status"] == "completed"
    assert result["completion"]["input_accuracy"] == "unqualified"
    products = result["artifacts"]["equivalent_products"]
    assert len(products) == 2
    for index, product in enumerate(products):
        assert product["raw_indices"] == list(request.derivation.raw_indices_for(index))
        assert product["arrays"]["folded_dn"]["dtype"] == "uint32"
        with h5py.File(
            config.output_root
            / config.run_id
            / "products"
            / f"group_{index:08d}"
            / "arrays.h5"
        ) as handle:
            assert handle["folded_dn"].dtype == np.uint32
            assert handle["folded_dn"].shape == (
                (17, 17) if kind == "stamp" else (65, 65)
            )
    replay = run_equivalent_coadd(config)
    assert all(p["reused"] for p in replay["artifacts"]["equivalent_products"])
    assert [p["arrays"] for p in products] == [
        p["arrays"] for p in replay["artifacts"]["equivalent_products"]
    ]
    assert verify_equivalent_run(config)["status"] == "verified"


def test_cli_routes_explicit_request_without_generic_spec_builder(tmp_path, capsys):
    from et_mainsim.cli import main

    config, _, payload = make_inputs(tmp_path, 30)
    path = tmp_path / "run.json"
    path.write_text(json.dumps(payload))
    assert main(["run", "et-equivalent-coadd", "--config", str(path), "--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)["workflow"] == "et-equivalent-coadd"
    assert main(["run", "et-equivalent-coadd", "--config", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "completed"
    assert (
        main(["run", "et-equivalent-coadd", "--config", str(path), "--verify-only"])
        == 0
    )
    assert json.loads(capsys.readouterr().out)["status"] == "verified"


@pytest.mark.parametrize(
    "field,value",
    [
        ("cpu_threads", True),
        ("cpu_threads", 0),
        ("resume", "yes"),
        ("coadd_indices", []),
        ("coadd_indices", [0, 0]),
        ("coadd_indices", [True]),
        ("block_shape", [129, 128]),
        ("run_id", "../escape"),
        ("run_id", ".run_locks"),
        ("n_raw", 12),
    ],
)
def test_invalid_execution_controls_are_rejected(tmp_path, field, value):
    payload = {
        "schema_id": CONFIG_SCHEMA,
        "request_path": "request.json",
        "catalog_path": "catalog.json",
        "variability_path": "variability.json",
        "data_root": "assets",
        "output_root": "results",
        "run_id": "test",
    }
    payload[field] = value
    with pytest.raises(ValueError):
        EquivalentRunConfig.from_mapping(payload, base=tmp_path)


def test_selected_groups_retain_entire_declared_risk_family(tmp_path):
    config, _, payload = make_inputs(tmp_path)
    payload["coadd_indices"] = [1]
    config = EquivalentRunConfig.from_mapping(payload, base=tmp_path)
    result = run_equivalent_coadd(config)
    products = result["artifacts"]["equivalent_products"]
    assert len(products) == 1 and products[0]["coadd_index"] == 1
    assert products[0]["clipping_certificate"]["family_group_count"] == 2
    payload["coadd_indices"] = [2]
    with pytest.raises(ValueError, match="outside"):
        build_equivalent_run_plan(
            EquivalentRunConfig.from_mapping(payload, base=tmp_path)
        )


def test_failed_later_group_resumes_completed_prefix(tmp_path, monkeypatch):
    import photsim7.equivalent_coadd as upstream

    config, _, _ = make_inputs(tmp_path)
    original = upstream.run_equivalent_coadd_product

    def fail_second(*args, **kwargs):
        if kwargs["coadd_index"] == 1:
            raise RuntimeError("injected second-group failure")
        return original(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(upstream, "run_equivalent_coadd_product", fail_second)
        with pytest.raises(RuntimeError, match="second-group"):
            run_equivalent_coadd(config)
    result = run_equivalent_coadd(config)
    assert [p["reused"] for p in result["artifacts"]["equivalent_products"]] == [
        True,
        False,
    ]
    assert result["attempts"][0]["status"] == "failed"
    assert result["status"] == "completed"


def test_rehashed_product_manifest_is_checked_against_application_record(tmp_path):
    config, _, _ = make_inputs(tmp_path)
    run_equivalent_coadd(config)
    manifest = (
        config.output_root
        / config.run_id
        / "products/group_00000000/product_manifest.json"
    )
    data = json.loads(manifest.read_text())
    data["unused_field"] = "changed"
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="application completion record"):
        verify_equivalent_run(config)
    with pytest.raises(ValueError, match="application completion record"):
        run_equivalent_coadd(config)


def test_request_sidecar_cannot_be_silently_replaced(tmp_path):
    config, _, _ = make_inputs(tmp_path)
    run_equivalent_coadd(config)
    sidecar = config.output_root / config.run_id / "equivalent_request.json"
    data = json.loads(sidecar.read_text())
    data["coadd_stop"] = 1
    sidecar.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="saved typed request"):
        run_equivalent_coadd(config)


def test_changed_catalog_is_rejected_before_run_creation(tmp_path):
    config, _, _ = make_inputs(tmp_path)
    data = json.loads(config.catalog_path.read_text())
    data["data"]["et_mag"][0] += 1
    config.catalog_path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="content differs"):
        run_equivalent_coadd(config)
    assert not (config.output_root / config.run_id).exists()


@pytest.mark.parametrize("operation", ["plan", "run"])
def test_cuda_request_is_rejected_by_typed_authority_before_assets(
    tmp_path, monkeypatch, operation
):
    import photsim7.input_identity as identity

    config, _, _ = make_inputs(tmp_path, 30)
    payload = json.loads(config.request_path.read_text())
    payload["derivation"]["source_spec"]["psf"]["compute_device"] = "cuda"
    config.request_path.write_text(json.dumps(payload))

    def unexpected_assets(*args, **kwargs):
        raise AssertionError("CUDA request reached scientific assets")

    monkeypatch.setattr(identity, "simulation_asset_identity", unexpected_assets)
    with pytest.raises(ValueError, match="CPU only"):
        (build_equivalent_run_plan if operation == "plan" else run_equivalent_coadd)(
            config
        )
    assert not (config.output_root / config.run_id).exists()


def test_changed_actual_product_cannot_resume_as_completed(tmp_path):
    config, _, _ = make_inputs(tmp_path)
    run_equivalent_coadd(config)
    path = config.output_root / config.run_id / "products/group_00000000/arrays.h5"
    with h5py.File(path, "r+") as handle:
        handle["folded_dn"][0, 0] += 1
    with pytest.raises(ValueError, match="integrity"):
        run_equivalent_coadd(config)
    manifest = json.loads(
        (config.output_root / config.run_id / "run_manifest.json").read_text()
    )
    assert manifest["status"] == "failed"


@pytest.mark.parametrize("operation", ["run", "verify"])
def test_assets_changed_during_operation_cannot_pass(tmp_path, monkeypatch, operation):
    import photsim7.equivalent_coadd as upstream

    config, _, _ = make_inputs(tmp_path, 30)
    asset = config.data_root / "psf/reference/sim_psf_images.pkl"
    if operation == "verify":
        run_equivalent_coadd(config)
    name = (
        "run_equivalent_coadd_product"
        if operation == "run"
        else "read_equivalent_coadd_product"
    )
    original = getattr(upstream, name)

    def change_after_last_product(*args, **kwargs):
        result = original(*args, **kwargs)
        if kwargs["coadd_index"] == 1:
            asset.write_bytes(asset.read_bytes() + b"changed-after-render")
        return result

    monkeypatch.setattr(upstream, name, change_after_last_product)
    manifest_path = config.output_root / config.run_id / "run_manifest.json"
    before = manifest_path.read_bytes() if operation == "verify" else None
    with pytest.raises(ValueError, match="scientific assets changed"):
        (run_equivalent_coadd if operation == "run" else verify_equivalent_run)(config)
    if operation == "run":
        assert json.loads(manifest_path.read_text())["status"] == "failed"
    else:
        assert manifest_path.read_bytes() == before


@pytest.mark.parametrize(
    "sidecar", ["equivalent_request.json", "equivalent_run_config.json"]
)
def test_interrupted_initial_sidecars_are_recoverable(tmp_path, monkeypatch, sidecar):
    import et_mainsim.equivalent_coadd as workflow

    config, _, _ = make_inputs(tmp_path, 30)
    original = workflow._atomic_write_json

    def interrupt(path, payload):
        if path.name == sidecar:
            raise OSError("interrupted initial publication")
        return original(path, payload)

    with monkeypatch.context() as patch:
        patch.setattr(workflow, "_atomic_write_json", interrupt)
        with pytest.raises(OSError, match="interrupted initial"):
            run_equivalent_coadd(config)
    run_dir = config.output_root / config.run_id
    planned = json.loads((run_dir / "run_manifest.json").read_text())
    assert planned["status"] == "planned" and planned["attempts"] == []
    result = run_equivalent_coadd(config)
    assert result["status"] == "completed"
    assert verify_equivalent_run(config)["status"] == "verified"


def test_missing_sidecar_after_execution_is_not_repaired(tmp_path):
    config, _, _ = make_inputs(tmp_path, 30)
    run_equivalent_coadd(config)
    path = config.output_root / config.run_id / "equivalent_run_config.json"
    path.unlink()
    with pytest.raises(FileNotFoundError):
        run_equivalent_coadd(config)
    assert not path.exists()


@pytest.mark.parametrize("field", ["python", "machine", "packages", "native_libraries"])
def test_empty_failed_run_cannot_resume_across_numerical_runtime(
    tmp_path, monkeypatch, field
):
    import et_mainsim.equivalent_coadd as workflow
    import photsim7.equivalent_coadd as upstream
    from et_mainsim.manifest import ManifestIdentityError

    config, _, _ = make_inputs(tmp_path, 30)

    def fail_before_first_product(*args, **kwargs):
        raise RuntimeError("failure before first product")

    with monkeypatch.context() as patch:
        patch.setattr(
            upstream, "run_equivalent_coadd_product", fail_before_first_product
        )
        with pytest.raises(RuntimeError, match="before first product"):
            run_equivalent_coadd(config)
    manifest_path = config.output_root / config.run_id / "run_manifest.json"
    before = manifest_path.read_bytes()
    saved_runtime = json.loads(before)["input_identity"]["runtime"]
    assert {"numpy", "scipy", "astropy", "numba", "h5py", "threadpoolctl"} <= set(
        saved_runtime["packages"]
    )
    assert {"torch", "kornia"} <= set(saved_runtime["rendering"]["packages"])
    assert saved_runtime["native_libraries"]
    original = workflow._runtime_identity

    def changed_runtime(spec):
        identity = original(spec)
        identity[field] = "changed-runtime"
        return identity

    monkeypatch.setattr(workflow, "_runtime_identity", changed_runtime)
    with pytest.raises(ManifestIdentityError, match="input identity"):
        run_equivalent_coadd(config)
    assert manifest_path.read_bytes() == before


@pytest.mark.parametrize("extra", [None, "user-file", "directory", "symlink"])
def test_initial_manifest_orphans_recover_without_touching_unknown_files(
    tmp_path, extra
):
    config, _, _ = make_inputs(tmp_path, 30)
    run_dir = config.output_root / config.run_id
    run_dir.mkdir(parents=True)
    orphan = run_dir / ".run_manifest.json.ab12_cd3.tmp"
    orphan.write_text('{"schema_id":')
    other = run_dir / "keep-me"
    if extra == "user-file":
        other.write_text("user data")
    elif extra == "directory":
        other.mkdir()
    elif extra == "symlink":
        other.symlink_to(config.request_path)
    if extra is None:
        result = run_equivalent_coadd(config)
        assert result["status"] == "completed"
        assert not orphan.exists()
    else:
        with pytest.raises(ValueError, match="nonempty run directory"):
            run_equivalent_coadd(config)
        assert orphan.read_text() == '{"schema_id":'
        assert other.exists()
        assert not (run_dir / "run_manifest.json").exists()


@pytest.mark.parametrize(
    "member", ["arrays.h5", "metadata.json", "product_manifest.json"]
)
def test_earlier_product_changed_during_later_group_cannot_complete(
    tmp_path, monkeypatch, member
):
    import photsim7.equivalent_coadd as upstream

    config, _, _ = make_inputs(tmp_path, 30)
    original = upstream.run_equivalent_coadd_product

    def corrupt_previous(*args, **kwargs):
        result = original(*args, **kwargs)
        if kwargs["coadd_index"] == 1:
            path = (
                config.output_root / config.run_id / "products/group_00000000" / member
            )
            path.write_bytes(path.read_bytes() + b"changed-after-verification")
        return result

    monkeypatch.setattr(upstream, "run_equivalent_coadd_product", corrupt_previous)
    with pytest.raises(ValueError, match="product files changed"):
        run_equivalent_coadd(config)
    manifest = json.loads(
        (config.output_root / config.run_id / "run_manifest.json").read_text()
    )
    assert manifest["status"] == "failed"


def test_concurrent_numeric_policies_are_serialized_and_restored():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    import torch
    from numba import config as numba_config
    from et_mainsim.equivalent_coadd import _numeric_policy

    original = torch.get_num_threads()
    first_entered, second_started, second_entered, release = (Event() for _ in range(4))
    other_threads = min(2, numba_config.NUMBA_NUM_THREADS)

    def first():
        with _numeric_policy(1):
            first_entered.set()
            assert release.wait(5)
            assert torch.get_num_threads() == 1

    def second():
        second_started.set()
        with _numeric_policy(other_threads):
            second_entered.set()
            assert torch.get_num_threads() == other_threads

    with ThreadPoolExecutor(max_workers=2) as pool:
        a = pool.submit(first)
        try:
            assert first_entered.wait(5)
            b = pool.submit(second)
            assert second_started.wait(5)
            assert not second_entered.wait(0.1)
        finally:
            release.set()
        a.result(timeout=5)
        b.result(timeout=5)
    assert second_entered.is_set()
    assert torch.get_num_threads() == original


@pytest.mark.parametrize(
    "sidecar", ["equivalent_request.json", "equivalent_run_config.json"]
)
@pytest.mark.parametrize("operation", ["run", "verify"])
def test_sidecars_changed_during_operation_cannot_pass(
    tmp_path, monkeypatch, sidecar, operation
):
    import photsim7.equivalent_coadd as upstream

    config, _, _ = make_inputs(tmp_path, 30)
    if operation == "verify":
        run_equivalent_coadd(config)
    name = (
        "run_equivalent_coadd_product"
        if operation == "run"
        else "read_equivalent_coadd_product"
    )
    original = getattr(upstream, name)
    run_dir = config.output_root / config.run_id

    def change_saved_authority(*args, **kwargs):
        result = original(*args, **kwargs)
        if kwargs["coadd_index"] == 1:
            path = run_dir / sidecar
            data = json.loads(path.read_text())
            data["run_id" if "config" in sidecar else "coadd_stop"] = "changed"
            path.write_text(json.dumps(data))
        return result

    monkeypatch.setattr(upstream, name, change_saved_authority)
    with pytest.raises(ValueError, match="saved .* differs"):
        (run_equivalent_coadd if operation == "run" else verify_equivalent_run)(config)
    if operation == "run":
        assert (
            json.loads((run_dir / "run_manifest.json").read_text())["status"]
            == "failed"
        )


@pytest.mark.parametrize("member", [None, "run_manifest.json", "products"])
@pytest.mark.parametrize("operation", ["run", "verify"])
def test_symlinked_run_publication_paths_are_rejected_without_writing(
    tmp_path, member, operation
):
    config, _, _ = make_inputs(tmp_path, 30)
    run_equivalent_coadd(config)
    run_dir = config.output_root / config.run_id
    path = run_dir if member is None else run_dir / member
    outside = tmp_path / "external-copy"
    path.rename(outside)
    path.symlink_to(outside, target_is_directory=outside.is_dir())
    manifest_path = (
        outside / "run_manifest.json"
        if member is None
        else outside
        if member == "run_manifest.json"
        else run_dir / "run_manifest.json"
    )
    before = manifest_path.read_bytes()
    with pytest.raises(ValueError, match="symlinks"):
        (run_equivalent_coadd if operation == "run" else verify_equivalent_run)(config)
    assert path.is_symlink()
    assert manifest_path.read_bytes() == before


@pytest.mark.parametrize(
    "sidecar", ["equivalent_request.json", "equivalent_run_config.json"]
)
@pytest.mark.parametrize("target_exists", [False, True])
def test_initialization_repair_never_replaces_or_accepts_sidecar_links(
    tmp_path, monkeypatch, sidecar, target_exists
):
    import et_mainsim.equivalent_coadd as workflow

    config, request, _ = make_inputs(tmp_path, 30)
    with monkeypatch.context() as patch:

        def interrupt(*args, **kwargs):
            raise OSError("interrupted before sidecars")

        patch.setattr(workflow, "_atomic_write_json", interrupt)
        with pytest.raises(OSError, match="before sidecars"):
            run_equivalent_coadd(config)
    run_dir = config.output_root / config.run_id
    outside = tmp_path / "external-sidecar.json"
    if target_exists:
        outside.write_text(
            json.dumps(
                config.to_dict() if "config" in sidecar else request.to_json_dict()
            )
        )
    path = run_dir / sidecar
    path.symlink_to(outside)
    before = (run_dir / "run_manifest.json").read_bytes()
    with pytest.raises(ValueError, match="symlinks"):
        run_equivalent_coadd(config)
    assert path.is_symlink() and outside.exists() == target_exists
    assert (run_dir / "run_manifest.json").read_bytes() == before


@pytest.mark.parametrize("kind", ["stamp", "full_frame"])
def test_fresh_process_caps_late_native_pools_and_resumes(tmp_path, kind):
    import os
    from pathlib import Path
    import subprocess
    import sys
    import et_mainsim
    import photsim7

    config, _, _ = make_inputs(tmp_path, 30, kind)
    path = tmp_path / "cold-run.json"
    path.write_text(json.dumps(config.to_dict()))
    env = dict(os.environ)
    # The child starts with a conflicting ambient policy. A warm parent or
    # OMP_NUM_THREADS=1 must not hide a backend loaded after policy entry.
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        env[name] = "3"
    roots = [
        str(Path(package.__file__).resolve().parent.parent)
        for package in (et_mainsim, photsim7)
    ]
    env["PYTHONPATH"] = os.pathsep.join(
        [*dict.fromkeys(roots), env.get("PYTHONPATH", "")]
    )
    probe = """
import json, sys
from pathlib import Path
from et_mainsim.equivalent_coadd import EquivalentRunConfig, run_equivalent_coadd, verify_equivalent_run
config = EquivalentRunConfig.from_file(sys.argv[1])
first = run_equivalent_coadd(config)
assert first["status"] == "completed"
metadata = json.loads((config.output_root / config.run_id / "products/group_00000000/metadata.json").read_text())
assert all(count == config.cpu_threads for _, count in metadata["execution_environment"]["native_thread_limits"])
second = run_equivalent_coadd(config)
assert all(group["reused"] for group in second["artifacts"]["equivalent_products"])
assert verify_equivalent_run(config)["status"] == "verified"
print("cold run, bounded native pools, resume and verification passed")
"""
    result = subprocess.run(
        [sys.executable, "-c", probe, str(path)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
