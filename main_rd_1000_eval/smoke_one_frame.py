from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from astropy import units as u

from main_rd_common import (
    ADC_BIT_DEPTH,
    BIAS_LEVEL_ADU,
    COLUMN_NOISE_SIGMA_ADU,
    COSMIC_RAY_EVENT_RATE,
    COSMIC_RAY_LIBRARY_PATH,
    COSMIC_RAY_PIXEL_SIZE,
    DARK_CURRENT,
    EXPOSURE,
    FRAME_COLS,
    FRAME_ROWS,
    FULL_WELL_ELECTRONS,
    GAIN_ELECTRONS_PER_ADU,
    GAIA_CATALOG_DIR,
    INTER_PIXEL_RESPONSE_NOMINAL,
    INTER_PIXEL_RESPONSE_SIGMA,
    INTRA_PIXEL_RESPONSE_SIGMA,
    N_SUBPIXELS,
    PIXEL_SCALE,
    PSF_BUNDLE_NAME,
    READOUT_NOISE,
    RESULTS_ROOT,
    SCATTERED_LIGHT,
    TARGET_FIELD_ANGLE_DEG,
    build_stars_table,
    ensure_local_imports,
    ensure_results_root,
    experiment_spec_dict,
    query_main_rd_stars,
    select_brightest,
    sim_config_dict,
    star_summary,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render one main_rd 1000x1000 smoke frame and record timing/VRAM."
    )
    parser.add_argument("--mag-limit", type=float, default=16.0)
    parser.add_argument("--max-stars", type=int, default=None)
    parser.add_argument("--catalog-dir", type=Path, default=GAIA_CATALOG_DIR)
    parser.add_argument("--output-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=20260515)
    parser.add_argument("--psf-field-id", type=int, default=None)
    parser.add_argument("--crop-margin-pix", type=float, default=2.0)
    parser.add_argument("--no-detector-response", action="store_true")
    parser.add_argument("--save-npz", action="store_true")
    return parser.parse_args()


def gpu_memory_snapshot() -> str:
    try:
        return subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def build_psf_manager(*, device: str, psf_field_id: int | None):
    ensure_local_imports()
    from photsim7.psf.model import PSFModelManager

    actor_config = {
        "bundle_name": PSF_BUNDLE_NAME,
        "pixel_scale": PIXEL_SCALE,
        "n_rows": FRAME_ROWS,
        "n_cols": FRAME_COLS,
        "n_subpixels": N_SUBPIXELS,
        "integrate_jitter": False,
        "n_jitter_integrated_psf_models": 1,
        "n_jitter_frames": 1,
        "compute_device": device,
        "float_precision": 32,
    }
    probe = PSFModelManager(
        config=actor_config,
        warp_frame_batch_size=10,
        xy_jitter_pix=None,
        intialize=False,
        build_jit_int_models=False,
        pad_to_detector_shape=False,
    )
    probe.load_bundle_data()
    if psf_field_id is None:
        psf_field_id = probe.nearest_field_id_for_angle(TARGET_FIELD_ANGLE_DEG)

    manager = PSFModelManager(
        config=actor_config,
        warp_frame_batch_size=10,
        xy_jitter_pix=None,
        intialize=True,
        build_jit_int_models=True,
        field_ids=[int(psf_field_id)],
        pad_to_detector_shape=False,
    )
    return manager, int(psf_field_id), dict(probe.model_angles_by_field_id)


def build_detector_response_sampler(*, device: str, seed: int):
    ensure_local_imports()
    from photsim7.full_frame_renderer import LazySubpixelResponseSampler

    return LazySubpixelResponseSampler(
        n_rows=FRAME_ROWS,
        n_cols=FRAME_COLS,
        n_subpixels=N_SUBPIXELS,
        inter_pixel_response_sigma=INTER_PIXEL_RESPONSE_SIGMA,
        inter_pixel_nominal_response=INTER_PIXEL_RESPONSE_NOMINAL,
        intra_pixel_response_sigma=INTRA_PIXEL_RESPONSE_SIGMA,
        pixel_response_profile_mod="flux conserved",
        enable_inter_pixel_response=True,
        enable_intra_pixel_response=True,
        enable_pixel_phase_response=True,
        random_seed=seed,
        compute_device=device,
        float_precision=32,
    )


def render_frame(args: argparse.Namespace):
    ensure_local_imports()
    from photsim7.cosmic_rays import (
        CosmicRayEventLibrary,
        CosmicRayInjector,
        apply_adc_digitization,
        clip_full_well_electrons,
        electrons_to_adu,
        mean_events_from_rate,
    )
    from photsim7.full_frame_renderer import SingleCadenceFullFrameRenderer

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    output_root = ensure_results_root(args.output_root)
    star_data_all = query_main_rd_stars(
        mag_limit=args.mag_limit,
        catalog_dir=args.catalog_dir,
        crop_margin_pix=args.crop_margin_pix,
    )
    star_data = select_brightest(star_data_all, args.max_stars)

    psf_model_manager, psf_field_id, psf_angles = build_psf_manager(
        device=args.device,
        psf_field_id=args.psf_field_id,
    )
    stars = build_stars_table(
        star_data,
        psf_field_id=psf_field_id,
    )
    detector_response_sampler = None
    if not args.no_detector_response:
        detector_response_sampler = build_detector_response_sampler(
            device=args.device,
            seed=args.seed,
        )

    renderer = SingleCadenceFullFrameRenderer(
        sim_config=sim_config_dict(),
        stars=stars,
        psf_model_manager=psf_model_manager,
        frame_exposure=EXPOSURE,
        detector_response_sampler=detector_response_sampler,
        compute_device=args.device,
        float_precision=32,
    )

    gpu_before = gpu_memory_snapshot()
    render_start = time.perf_counter()
    components = renderer.render_single_cadence(
        enable_stellar_photon_noise=True,
        enable_background_light=True,
        enable_scattered_light=False,
        enable_dark_current=True,
        enable_readout_noise=True,
        background_flux_per_pixel=sim_config_dict()["Background Flux"],
        scattered_light_per_pixel=SCATTERED_LIGHT,
        dark_current_per_pixel=DARK_CURRENT,
        readout_noise=READOUT_NOISE,
        subtract_nonstellar_mean=False,
        progress=True,
        return_numpy=False,
    )
    image_electrons = components["final_image"]
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    render_elapsed_s = time.perf_counter() - render_start

    electronics_start = time.perf_counter()
    image_clipped_e = clip_full_well_electrons(
        image_electrons,
        full_well_electrons=FULL_WELL_ELECTRONS,
    )
    image_adu = electrons_to_adu(
        image_clipped_e,
        gain_electrons_per_adu=GAIN_ELECTRONS_PER_ADU,
    )
    library = CosmicRayEventLibrary.load(
        COSMIC_RAY_LIBRARY_PATH,
        expected_pixel_size_um=COSMIC_RAY_PIXEL_SIZE.to(u.um).value,
    )
    mean_events = mean_events_from_rate(
        rate_events_per_cm2_s=COSMIC_RAY_EVENT_RATE,
        n_rows=FRAME_ROWS,
        n_cols=FRAME_COLS,
        pixel_size_um=COSMIC_RAY_PIXEL_SIZE,
        exposure_s=EXPOSURE,
    )
    image_adu_stack, cosmic_payload = CosmicRayInjector(library).inject(
        image_adu.unsqueeze(0),
        mean_events_per_frame=mean_events,
        seed=args.seed,
        frame_start=0,
        allow_partial=True,
    )
    image_adu = image_adu_stack[0]
    col_noise = torch.normal(
        mean=0.0,
        std=COLUMN_NOISE_SIGMA_ADU,
        size=(FRAME_COLS,),
        dtype=image_adu.dtype,
        device=image_adu.device,
    )
    image_adu = image_adu + BIAS_LEVEL_ADU + col_noise[None, :]
    image_dn = apply_adc_digitization(
        image_adu,
        enabled=True,
        bit_depth=ADC_BIT_DEPTH,
        min_value=0.0,
        round_values=True,
    )
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    electronics_elapsed_s = time.perf_counter() - electronics_start
    peak_allocated_mb = (
        torch.cuda.max_memory_allocated() / 1024**2 if args.device.startswith("cuda") else None
    )
    peak_reserved_mb = (
        torch.cuda.max_memory_reserved() / 1024**2 if args.device.startswith("cuda") else None
    )
    gpu_after = gpu_memory_snapshot()

    image_np = image_dn.detach().cpu().numpy()
    stellar_mean_np = components["stellar_mean"].detach().cpu().numpy()
    preview_path = output_root / (
        f"smoke_main_rd_g_lt_{args.mag_limit:g}"
        f"_n_{len(stars)}_gpu_{os.environ.get('CUDA_VISIBLE_DEVICES', 'default')}.png"
    )
    plot_preview(
        image_np,
        stellar_mean_np,
        preview_path=preview_path,
        title=(
            f"main_rd 1000x1000 smoke, G<{args.mag_limit:g}, "
            f"stars={len(stars)}, CR={len(cosmic_payload.events)}"
        ),
    )
    npz_path = None
    if args.save_npz:
        npz_path = output_root / (
            f"smoke_main_rd_g_lt_{args.mag_limit:g}_n_{len(stars)}.npz"
        )
        np.savez_compressed(
            npz_path,
            image_dn=image_np,
            stellar_mean_e=stellar_mean_np.astype(np.float32),
            cosmic_mask=cosmic_payload.mask[0],
            cosmic_events=cosmic_payload.events,
            column_noise_vector_adu=col_noise.detach().cpu().numpy(),
        )

    summary = {
        "experiment": experiment_spec_dict(),
        "mag_limit": float(args.mag_limit),
        "max_stars": args.max_stars,
        "all_star_summary": star_summary(star_data_all),
        "rendered_star_summary": star_summary(star_data),
        "psf_field_id": int(psf_field_id),
        "psf_field_angle_deg": float(psf_angles[int(psf_field_id)]),
        "target_field_angle_deg": float(TARGET_FIELD_ANGLE_DEG),
        "device": args.device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "render_elapsed_s": float(render_elapsed_s),
        "electronics_elapsed_s": float(electronics_elapsed_s),
        "total_elapsed_s": float(render_elapsed_s + electronics_elapsed_s),
        "peak_cuda_allocated_mb": peak_allocated_mb,
        "peak_cuda_reserved_mb": peak_reserved_mb,
        "gpu_memory_before": gpu_before,
        "gpu_memory_after": gpu_after,
        "mean_cosmic_events_per_frame": float(mean_events),
        "actual_cosmic_events": int(len(cosmic_payload.events)),
        "cosmic_mask_pixels": int(np.count_nonzero(cosmic_payload.mask)),
        "image_dtype": str(image_np.dtype),
        "image_min": int(np.min(image_np)),
        "image_p50": float(np.percentile(image_np, 50)),
        "image_p99": float(np.percentile(image_np, 99)),
        "image_max": int(np.max(image_np)),
        "saturated_pixels": int(np.count_nonzero(image_np >= (2**ADC_BIT_DEPTH - 1))),
        "preview_path": str(preview_path),
        "npz_path": str(npz_path) if npz_path is not None else None,
    }
    summary_path = output_root / (
        f"smoke_main_rd_g_lt_{args.mag_limit:g}_n_{len(stars)}"
        f"_gpu_{os.environ.get('CUDA_VISIBLE_DEVICES', 'default')}.json"
    )
    write_json(summary_path, summary)
    print(summary_path)
    print(
        f"render={render_elapsed_s:.2f}s electronics={electronics_elapsed_s:.2f}s "
        f"stars={len(stars)} cr={len(cosmic_payload.events)} "
        f"peak_alloc={peak_allocated_mb}MB"
    )


def plot_preview(
    image_dn: np.ndarray,
    stellar_mean_e: np.ndarray,
    *,
    preview_path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8), dpi=140)
    dn_low, dn_high = np.percentile(image_dn, [1, 99.7])
    axes[0].imshow(image_dn, origin="lower", cmap="gray", vmin=dn_low, vmax=dn_high)
    axes[0].set_title("Final DN")
    axes[0].set_xlabel("x pix")
    axes[0].set_ylabel("y pix")
    mean_low, mean_high = np.percentile(stellar_mean_e, [50, 99.8])
    axes[1].imshow(
        stellar_mean_e,
        origin="lower",
        cmap="magma",
        vmin=mean_low,
        vmax=max(mean_high, mean_low + 1.0),
    )
    axes[1].set_title("Stellar mean e-")
    axes[1].set_xlabel("x pix")
    fig.suptitle(title)
    fig.tight_layout()
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(preview_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    render_frame(args)


if __name__ == "__main__":
    main()
