from __future__ import annotations

from dataclasses import dataclass
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

    from photsim7.specs import SimulationSpec

    package_files = files(__package__)
    science_text = package_files.joinpath(descriptor.science_resource).read_text(
        encoding="utf-8"
    )
    execution_text = package_files.joinpath(descriptor.execution_resource).read_text(
        encoding="utf-8"
    )
    return LoadedPreset(
        descriptor=descriptor,
        simulation_spec=SimulationSpec.from_json(science_text),
        run_config=RunConfig.from_toml(
            execution_text,
            source=f"package:{descriptor.execution_resource}",
        ),
    )


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
