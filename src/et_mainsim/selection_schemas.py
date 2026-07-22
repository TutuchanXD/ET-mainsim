from __future__ import annotations

from numbers import Integral
from typing import Any


_LEGACY_CADENCE_SELECTION_TRUTH_SCHEMA = (
    "photsim7.cadence_selection_truth.v1",
    1,
)


def supported_cadence_selection_truth_schemas() -> frozenset[tuple[str, int]]:
    """Return the readable legacy and active Photsim7 cadence schemas."""

    from photsim7.selection_artifacts import (
        CADENCE_SELECTION_TRUTH_SCHEMA_ID,
        CADENCE_SELECTION_TRUTH_SCHEMA_VERSION,
    )

    return frozenset(
        {
            _LEGACY_CADENCE_SELECTION_TRUTH_SCHEMA,
            (
                str(CADENCE_SELECTION_TRUTH_SCHEMA_ID),
                int(CADENCE_SELECTION_TRUTH_SCHEMA_VERSION),
            ),
        }
    )


def is_supported_cadence_selection_truth_schema(
    schema_id: Any,
    schema_version: Any,
) -> bool:
    """Check an exact schema ID/version pair without accepting mixed pairs."""

    if type(schema_id) is not str:
        return False
    if isinstance(schema_version, bool) or not isinstance(
        schema_version,
        Integral,
    ):
        return False
    candidate = (schema_id, int(schema_version))
    return candidate in supported_cadence_selection_truth_schemas()
