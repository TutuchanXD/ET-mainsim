"""Replay the M6 application with real ET assets; never set science tolerances.

Run in the supported application environment. Assets stay external and are
identified in each run. The output directory must be new.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import platform
import resource
import time

import numpy as np
from astropy import units as u

from et_mainsim.legacy_expectation import (
    LegacyExpectationConfig, CONFIG_SCHEMA, run_legacy_expectation,
    verify_legacy_expectation_run, read_legacy_expectation_group,
)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def prepare(directory, data_root, *, frames=24):
    from photsim7.specs import (
        SimulationSpec, ObservationSpec, InstrumentSpec, DetectorSpec, CatalogSpec,
        PsfSpec, DetectorResponseSpec, ReadoutSpec, SkySpec, CosmicRaySpec, RngSpec, ScienceProfileSpec,
    )
    from photsim7.geometry_truth import reference_field_nonphysical_declaration
    directory.mkdir()
    defaults = SimulationSpec()
    effects = replace(defaults.dynamic_effects,
        psd_motion=replace(defaults.dynamic_effects.psd_motion, enabled=True),
        momentum_dump=replace(defaults.dynamic_effects.momentum_dump, enabled=True,
                              cycle=3*u.day, legacy_radius_arcsec=.15*u.arcsec),
        thermal_drift=replace(defaults.dynamic_effects.thermal_drift, enabled=True,
                              profile="et_temperature_table", time_policy="normalized_observation_phase"),
        psf_breathing=replace(defaults.dynamic_effects.psf_breathing, enabled=True,
                              profile="et_temperature_table", time_policy="normalized_observation_phase"),
    )
    spec = SimulationSpec(
        observation=ObservationSpec(observing_start_date="2026-11-09T00:00:00",
            exposure_duration=10*u.s, readout_duration=0*u.s, n_frames=frames,
            observing_duration=None, n_raw_frames_per_coadd=6, simulation_cadence_mult=1),
        instrument=InstrumentSpec(telescope_count=6),
        detector=DetectorSpec(detector_id="m6-reference", shape=(65,65), n_subpixels=7,
            pixel_width=10*u.um, pixel_scale=4.83*u.arcsec/u.pix),
        catalog=CatalogSpec(source_type="prepared", target_epoch_jyear=2000.),
        psf=PsfSpec(mode="stamp", bundle_name="psf/et/241006/D280mm-focus", field_id=6,
            field_id_policy="explicit", use_jitter_integrated_psf=True,
            n_jitter_integrated_psf_models=2, n_jitter_frames_per_model=64,
            float_precision=64, compute_device="cpu"),
        detector_response=DetectorResponseSpec(enable_inter_pixel_response=True, inter_prv_rms=1*u.percent,
            enable_intra_pixel_response=True, intra_prv_rms=.5*u.percent,
            enable_pixel_phase_response=True, scripted_sensitivity_enabled=False,
            whole_pixel_gain_normal_enabled=False, whole_pixel_gain_sinusoidal_enabled=False,
            enable_flat_field_correction=False),
        readout=ReadoutSpec(readout_noise=5*u.electron/u.pix),
        sky=SkySpec(background_flux=2*u.electron/u.s/u.pix, dark_current=.25*u.electron/u.s/u.pix,
                    scattered_light=.5*u.electron/u.s/u.pix, subtract_nonstellar_mean=True),
        cosmic_rays=CosmicRaySpec(enabled=False), dynamic_effects=effects,
        rng=RngSpec(run_seed=20261109, determinism_mode="bitwise"),
        science_profile=ScienceProfileSpec(profile_id="legacy_expectation_electrons_v1", science_realization_id=29),
    )
    geometry = reference_field_nonphysical_declaration(reference_field_angle_deg=12.,
        reference_field_polar_angle_rad=np.pi/4, reference_pixel_scale_arcsec_per_pix=4.83)
    data = dict(source_id=[101,202], x0=[0.,1.2], y0=[0.,.4], ra=[10.,10.001], dec=[20.,20.001],
                et_mag=[16.,18.], frame_xpix=[32.,33.2], frame_ypix=[32.,32.4],
                detector_xpix=[32.,33.2], detector_ypix=[32.,32.4], detector_id="m6-reference")
    write_json(directory/"catalog.json", dict(data=data, metadata={"source":{"type":"prepared"},"geometry":geometry}, raw_source_arrays=None))
    factors = 1-.01*np.sin(np.linspace(0,np.pi,frames))**2
    write_json(directory/"variability.json", dict(source_ids=[101], relative_flux=[factors.tolist()]))
    (directory/"targets.csv").write_text("source_id,gaia_g_mag,psf_id,curve_id\n101,16,6,variable\n202,18,6,\n")
    (directory/"variability.csv").write_text("curve_id,frame_index,relative_flux\n"+"".join(f"variable,{i},{v:.17g}\n" for i,v in enumerate(factors)))
    for kind, shape in (("full-frame", (65,65)), ("catalog-stamp", (21,21)), ("table-stamp", (21,21))):
        weights=np.zeros(shape)
        y,x=(n//2 for n in shape)
        weights[y-5:y+6,x-5:x+6]=1
        np.savez(directory/f"aperture-{kind}.npz", **({"101":weights,"202":weights} if kind=="table-stamp" else {"101":weights}))
    configs=[]
    for device in ("cpu","cuda"):
        for kind in ("catalog-stamp","full-frame","table-stamp"):
            filename=f"spec-{device}-{kind}.json"
            actual=replace(spec, psf=replace(spec.psf,compute_device=device))
            write_json(directory/filename,actual.to_json_dict())
            payload=dict(schema_id=CONFIG_SCHEMA,spec_path=filename,aperture_path=f"aperture-{kind}.npz",
                data_root=str(data_root),output_root=str(directory.parent/"runs"),run_id=f"{device}-{kind}",
                kind=kind,target_ids=[101,202] if kind=="table-stamp" else [101],stamp_shape=[21,21],cpu_threads=2)
            if kind=="table-stamp":
                payload.update(input_table_path="targets.csv",variability_table_path="variability.csv")
            else:
                payload.update(catalog_path="catalog.json",variability_path="variability.json")
            config=LegacyExpectationConfig.from_mapping(payload,base=directory)
            write_json(directory/f"run-{device}-{kind}.json",config.to_dict())
            configs.append(config)
    return configs


def read_groups(config):
    return [read_legacy_expectation_group(p) for p in sorted((config.output_root/config.run_id/"groups").glob("group_*"))]


def compare(first, second, *, exact=True):
    count=0
    maximum=0.
    by_field={}
    for a,b in zip(read_groups(first),read_groups(second),strict=True):
        if set(a.arrays)!=set(b.arrays):
            raise AssertionError("array identities differ")
        for name,x in a.arrays.items():
            y=b.arrays[name]
            if exact:
                np.testing.assert_array_equal(x,y,err_msg=name)
            finite=np.isfinite(x)&np.isfinite(y)
            if finite.any():
                difference=float(np.max(np.abs(x[finite].astype(float)-y[finite].astype(float))))
                maximum=max(maximum,difference)
                field=name.split("__",1)[1]
                by_field[field]=max(by_field.get(field,0.),difference)
            count+=1
    return {"arrays_compared":count,"bitwise_required":exact,"maximum_absolute_difference":maximum,
            "maximum_absolute_difference_by_field":by_field}


def pixel_parity(configs, device):
    """Same backend, same physical pixels: compare all coadd companions."""
    stamp=next(c for c in configs if c.run_id==f"{device}-catalog-stamp")
    full=next(c for c in configs if c.run_id==f"{device}-full-frame")
    count=0
    for group in sorted((stamp.output_root/stamp.run_id/"groups").glob("group_*")):
        for scope in range(6):
            path=group/f"coadds/target_101/scope_{scope}"
            window=json.loads((path/"manifest.json").read_text())["stamp_window"]
            y,x=window["y_start_detector_pix"],window["x_start_detector_pix"]
            full_path=full.output_root/full.run_id/"groups"/group.name/f"coadds/target_None/scope_{scope}/arrays.npz"
            with np.load(path/"arrays.npz") as a,np.load(full_path) as b:
                for name in a.files:
                    h,w=a[name].shape
                    np.testing.assert_array_equal(a[name],b[name][y:y+h,x:x+w],err_msg=name)
                    count+=1
    return {"arrays_bitwise_equal":count}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--data-root",type=Path,required=True)
    parser.add_argument("--cpu-only",action="store_true")
    args=parser.parse_args(argv)
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    configs=prepare(output/"inputs",args.data_root.resolve())
    import torch
    reports={}
    for config in configs:
        if config.run_id.startswith("cuda") and args.cpu_only:
            continue
        started=time.perf_counter()
        status=run_legacy_expectation(config)
        assert status["status"]=="completed"
        verified=verify_legacy_expectation_run(config)
        assert verified==status
        # Resume an already completed run through the same public entry.
        assert run_legacy_expectation(config)==status
        report={"elapsed_with_verify_resume_s":time.perf_counter()-started,"status":status,
                "host_peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        if config.run_id.startswith("cuda"):
            report["cuda_peak_allocated_bytes"]=torch.cuda.max_memory_allocated()
        # Different process scheduling is separately covered by the regression
        # suite. Here compare all real-asset scalar products under reverse shards.
        other=replace(config,run_id=config.run_id+"-sharded",output_kind="lightcurve")
        assert run_legacy_expectation(other,shard_index=1,shard_count=2)["status"]=="partial"
        assert run_legacy_expectation(other,group_indices=[3,2,1,0])["status"]=="completed"
        verify_legacy_expectation_run(other)
        report["pixel_vs_light_sharded"]=compare(config,other)
        reports[config.run_id]=report
        write_json(output/"progress.json",reports)
        print(config.run_id,report["pixel_vs_light_sharded"],flush=True)
    diagnostics={}
    if not args.cpu_only:
        for kind in ("catalog-stamp","full-frame","table-stamp"):
            cpu=next(c for c in configs if c.run_id==f"cpu-{kind}")
            gpu=next(c for c in configs if c.run_id==f"cuda-{kind}")
            diagnostics[kind]=compare(cpu,gpu,exact=False)
    report={"schema_id":"et_mainsim.m6_validation.v1","platform":platform.platform(),
            "cuda_device":None if args.cpu_only else torch.cuda.get_device_name(),
            "runs":reports,"cpu_cuda_diagnostics":diagnostics,
            "full_stamp_pixel_parity":{device:pixel_parity(configs,device)
                                       for device in (["cpu"] if args.cpu_only else ["cpu","cuda"])},
            "cpu_cuda_policy":"Torch Poisson draws are backend-specific; cross-device noisy flux residuals are descriptive, not a bitwise gate",
            "science_acceptance":"engineering invariants only; no new science residual tolerance"}
    write_json(output/"validation.json",report)
    files={p.relative_to(output).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
           for p in sorted(output.rglob("*")) if p.is_file()}
    write_json(output/"SHA256SUMS.json",files)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
