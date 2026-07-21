from __future__ import annotations

from dataclasses import dataclass, replace
from importlib.resources import files
from typing import Any

from et_mainsim.config import RunConfig


@dataclass(frozen=True)
class PresetDescriptor:
    name: str
    workflow: str
    profile: str
    description: str
    science_resource: str
    execution_resource: str

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "workflow": self.workflow,
            "profile": self.profile,
            "description": self.description,
            "science_resource": self.science_resource,
            "execution_resource": self.execution_resource,
        }


@dataclass(frozen=True)
class LoadedPreset:
    descriptor: PresetDescriptor
    simulation_spec: Any
    run_config: RunConfig
    science_contract: Any | None = None


_PRESETS = {
    "et-full-frame-production": PresetDescriptor(
        name="et-full-frame-production",
        workflow="et-full-frame",
        profile="production",
        description="One physical ET main detector, 180 ten-second cadences, G < 18.",
        science_resource="et_full_frame_production.spec.json",
        execution_resource="et_full_frame_production.run.toml",
    ),
    "et-full-frame-smoke": PresetDescriptor(
        name="et-full-frame-smoke",
        workflow="et-full-frame",
        profile="smoke",
        description="One 64x64 CPU cadence for installation and artifact validation.",
        science_resource="et_full_frame_smoke.spec.json",
        execution_resource="et_full_frame_smoke.run.toml",
    ),
    "et-stamp-production": PresetDescriptor(
        name="et-stamp-production",
        workflow="et-stamp",
        profile="production",
        description="One physical ET target with neighbors, 360 raw cadences and 12 coadds.",
        science_resource="factory:et-stamp-production",
        execution_resource="et_stamp_production.run.toml",
    ),
    "et-stamp-smoke": PresetDescriptor(
        name="et-stamp-smoke",
        workflow="et-stamp",
        profile="smoke",
        description="Two CPU raw cadences and one coadd using a packaged target field.",
        science_resource="factory:et-stamp-smoke",
        execution_resource="et_stamp_smoke.run.toml",
    ),
    "legacy-sim-full-effects-production": PresetDescriptor(
        name="legacy-sim-full-effects-production",
        workflow="legacy-sim",
        profile="full-effects-production",
        description="Legacy 101x101 full-effect parity for 100 targets and 360 cadences.",
        science_resource="factory:legacy-full-effects-production",
        execution_resource="legacy_full_effects_production.run.toml",
    ),
    "legacy-sim-full-effects-smoke": PresetDescriptor(
        name="legacy-sim-full-effects-smoke",
        workflow="legacy-sim",
        profile="full-effects-smoke",
        description="Small local-Ray full-effect parity and artifact validation run.",
        science_resource="factory:legacy-full-effects-smoke",
        execution_resource="legacy_full_effects_smoke.run.toml",
    ),
}


def list_presets(*, workflow: str | None = None) -> tuple[PresetDescriptor, ...]:
    return tuple(
        descriptor
        for name, descriptor in sorted(_PRESETS.items())
        if workflow is None or descriptor.workflow == workflow
    )


def canonical_preset_name(workflow: str, profile_or_name: str) -> str:
    value = str(profile_or_name).strip()
    if value in _PRESETS:
        return value
    candidate = f"{workflow}-{value}"
    if candidate in _PRESETS:
        return candidate
    raise KeyError(
        f"Unknown preset {profile_or_name!r}; available: {', '.join(sorted(_PRESETS))}"
    )


def load_preset(name: str) -> LoadedPreset:
    try:
        descriptor = _PRESETS[str(name)]
    except KeyError as exc:
        raise KeyError(
            f"Unknown preset {name!r}; available: {', '.join(sorted(_PRESETS))}"
        ) from exc

    package_files = files(__package__)
    execution_text = package_files.joinpath(descriptor.execution_resource).read_text(
        encoding="utf-8"
    )
    simulation_spec, science_contract = _load_science(descriptor, package_files)
    return LoadedPreset(
        descriptor=descriptor,
        simulation_spec=simulation_spec,
        run_config=RunConfig.from_toml(
            execution_text,
            source=f"package:{descriptor.execution_resource}",
        ),
        science_contract=science_contract,
    )


def _load_science(descriptor: PresetDescriptor, package_files: Any) -> tuple[Any, Any]:
    from astropy import units as u
    from photsim7.specs import SimulationSpec

    resource = descriptor.science_resource
    if not resource.startswith("factory:"):
        text = package_files.joinpath(resource).read_text(encoding="utf-8")
        return SimulationSpec.from_json(text), None

    if descriptor.workflow == "et-stamp":
        base_name = (
            "et_full_frame_smoke.spec.json"
            if descriptor.profile == "smoke"
            else "et_full_frame_production.spec.json"
        )
        base = SimulationSpec.from_json(
            package_files.joinpath(base_name).read_text(encoding="utf-8")
        )
        if descriptor.profile == "smoke":
            spec = replace(
                base,
                instrument=replace(base.instrument, telescope_count=1),
                observation=replace(
                    base.observation,
                    observing_duration=20 * u.s,
                    n_frames=2,
                    n_raw_frames_per_coadd=2,
                    frame_start_s=None,
                ),
                catalog=replace(
                    base.catalog,
                    source_path="package://et_stamp_smoke_catalog.csv",
                    target_detector_xpix=31.5,
                    target_detector_ypix=31.5,
                ),
                psf=replace(base.psf, mode="stamp"),
            )
        else:
            query_options = dict(base.catalog.query_options)
            query_options.update(
                query_radius_deg=0.07,
                crop_to_simulation_frame=False,
            )
            spec = replace(
                base,
                instrument=replace(base.instrument, telescope_count=1),
                observation=replace(
                    base.observation,
                    observing_duration=3600 * u.s,
                    n_frames=360,
                    n_raw_frames_per_coadd=30,
                    frame_start_s=None,
                ),
                detector=replace(base.detector, n_subpixels=7),
                catalog=replace(base.catalog, query_options=query_options),
                psf=replace(base.psf, mode="stamp"),
            )
        return spec, None

    from photsim7.legacy_workflow import make_et_legacy_full_effect_contract

    if descriptor.profile == "full-effects-smoke":
        contract = make_et_legacy_full_effect_contract(
            shape=(9, 9),
            compute_device="cpu",
            observing_duration=20 * u.s,
            n_jitter_integrated_psf_models=100,
            n_jitter_frames_per_model=300,
            background_stars_max_mag=17.0,
        )
    else:
        contract = make_et_legacy_full_effect_contract(
            shape=(101, 101),
            compute_device="cuda",
            observing_duration=3600 * u.s,
            n_jitter_integrated_psf_models=100,
            n_jitter_frames_per_model=300,
            background_stars_max_mag=17.0,
        )
    return contract.spec, contract


def resource_path(name: str):
    return files(__package__).joinpath(name)


__all__ = [
    "LoadedPreset",
    "PresetDescriptor",
    "canonical_preset_name",
    "list_presets",
    "load_preset",
    "resource_path",
]
