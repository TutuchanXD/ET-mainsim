from __future__ import annotations

import pytest


def test_supported_cadence_selection_truth_schemas_accept_exact_pairs() -> None:
    from et_mainsim.selection_schemas import (
        is_supported_cadence_selection_truth_schema,
        supported_cadence_selection_truth_schemas,
    )

    schemas = supported_cadence_selection_truth_schemas()
    assert ("photsim7.cadence_selection_truth.v1", 1) in schemas
    assert all(
        is_supported_cadence_selection_truth_schema(schema_id, schema_version)
        for schema_id, schema_version in schemas
    )


@pytest.mark.parametrize(
    ("schema_id", "schema_version"),
    [
        ("photsim7.cadence_selection_truth.v1", True),
        ("photsim7.cadence_selection_truth.v1", 1.0),
        ("photsim7.cadence_selection_truth.v1", "1"),
        ("photsim7.cadence_selection_truth.v1", None),
        (None, 1),
        (1, 1),
        ("photsim7.cadence_selection_truth.v1 ", 1),
        ("photsim7.cadence_selection_truth.v1", 2),
        ("photsim7.cadence_selection_truth.v2", 1),
    ],
)
def test_supported_cadence_selection_truth_schemas_reject_coercion_and_mixed_pairs(
    schema_id,
    schema_version,
) -> None:
    from et_mainsim.selection_schemas import (
        is_supported_cadence_selection_truth_schema,
    )

    assert not is_supported_cadence_selection_truth_schema(
        schema_id,
        schema_version,
    )
