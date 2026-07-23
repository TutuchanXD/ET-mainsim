#!/usr/bin/env python3
"""Write a metadata-level completion receipt for one Galaxy stamp delivery case.

The command is safe to run while rendering is in progress: missing final HDF5
members and in-progress partial artifacts are reported in the receipt.  Add
``--require-complete`` at the formal hand-off gate to fail closed unless every
target, frozen time shard, and delivered cadence product is present and valid.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
if SOURCE_ROOT.is_dir() and str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from et_mainsim.galaxy_campaign_qc import (  # noqa: E402
    GalaxyCampaignDeliveryQCRequest,
    audit_galaxy_campaign_delivery_v1,
    write_galaxy_campaign_delivery_qc_json,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--production-manifest",
        required=True,
        type=Path,
        help="frozen production_manifest.json at the delivery root",
    )
    parser.add_argument("--case", choices=("static", "injected"), default="injected")
    parser.add_argument(
        "--output-json",
        required=True,
        type=Path,
        help="atomic machine-readable QC receipt destination",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="return status 2 unless the delivery matrix has no gaps or anomalies",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = audit_galaxy_campaign_delivery_v1(
            GalaxyCampaignDeliveryQCRequest(
                production_manifest_path=args.production_manifest,
                case=args.case,
            )
        )
        receipt = write_galaxy_campaign_delivery_qc_json(result, args.output_json)
    except (OSError, ValueError) as error:
        print(f"campaign delivery QC failed: {error}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "receipt": str(receipt),
                "ready": result.ready,
                "expected_bundle_count": result.expected_bundle_count,
                "valid_bundle_count": result.valid_bundle_count,
                "missing_bundle_count": result.missing_bundle_count,
                "invalid_bundle_count": result.invalid_bundle_count,
                "unexpected_final_bundle_count": len(result.unexpected_final_bundles),
                "partial_artifact_count": len(result.partial_artifacts),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if result.ready or not args.require_complete else 2


if __name__ == "__main__":  # pragma: no cover - exercised by CLI subprocess test.
    raise SystemExit(main())
