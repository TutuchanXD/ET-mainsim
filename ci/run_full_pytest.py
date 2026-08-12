from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "ci" / "full_pytest_contract.toml"


class FullPytestContractError(RuntimeError):
    pass


@dataclass(eq=False)
class _SessionReceipt:
    nodeids: list[str] = field(default_factory=list)
    deselected: list[str] = field(default_factory=list)
    passed: set[str] = field(default_factory=set)
    skipped: set[str] = field(default_factory=set)
    xfailed: set[str] = field(default_factory=set)
    failed: set[str] = field(default_factory=set)

    def pytest_collection_finish(self, session: pytest.Session) -> None:
        self.nodeids = [item.nodeid for item in session.items]

    def pytest_deselected(self, items: list[pytest.Item]) -> None:
        self.deselected.extend(item.nodeid for item in items)

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        if report.failed:
            self.failed.add(report.nodeid)
        if hasattr(report, "wasxfail"):
            self.xfailed.add(report.nodeid)
            return
        if report.skipped:
            self.skipped.add(report.nodeid)
        elif report.when == "call" and report.passed:
            self.passed.add(report.nodeid)


def _nodeid_digest(nodeids: list[str]) -> str:
    payload = "\n".join(sorted(nodeids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _tracked_test_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "--", "tests/test_*.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return sorted(line for line in result.stdout.splitlines() if line)


def _validate_environment(contract: dict[str, Any]) -> None:
    if len(sys.argv) != 1:
        raise FullPytestContractError("the full-test runner accepts no selectors or arguments")
    if os.environ.get("PYTEST_ADDOPTS"):
        raise FullPytestContractError("PYTEST_ADDOPTS must be empty")
    if os.environ.get("PYTEST_DISABLE_PLUGIN_AUTOLOAD") != "1":
        raise FullPytestContractError("third-party pytest plugin autoload must be disabled")
    if importlib.metadata.version("pytest") != contract["pytest_version"]:
        raise FullPytestContractError("the installed pytest version is not frozen")
    if f"{sys.version_info.major}.{sys.version_info.minor}" not in contract[
        "python_versions"
    ]:
        raise FullPytestContractError("the Python version is outside the tested matrix")

    data_dir = os.environ.get("ET_DATA_DIR")
    if not data_dir:
        raise FullPytestContractError("ET_DATA_DIR must name an intentionally missing path")
    if Path(data_dir).exists():
        raise FullPytestContractError("ET_DATA_DIR must not exist during the hermetic suite")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise FullPytestContractError("CUDA_VISIBLE_DEVICES must be empty")


def _validate_session(receipt: _SessionReceipt, contract: dict[str, Any]) -> None:
    expected_files = sorted(contract["test_files"])
    tracked_files = _tracked_test_files()
    if tracked_files != expected_files:
        raise FullPytestContractError("tracked test modules differ from the frozen inventory")

    collected_files = sorted({nodeid.split("::", 1)[0] for nodeid in receipt.nodeids})
    if collected_files != expected_files:
        raise FullPytestContractError("pytest did not collect every frozen test module")
    if receipt.deselected:
        raise FullPytestContractError("pytest deselected one or more tests")
    if len(receipt.nodeids) != contract["expected_collected"]:
        raise FullPytestContractError("the collected test count differs from the contract")
    if _nodeid_digest(receipt.nodeids) != contract["nodeid_sha256"]:
        raise FullPytestContractError("the collected node-id inventory differs from the contract")
    if receipt.xfailed:
        raise FullPytestContractError("xfail/xpass outcomes are not permitted in the full suite")
    if sorted(receipt.skipped) != sorted(contract["allowed_skips"]):
        raise FullPytestContractError("the skipped-test inventory differs from the contract")
    terminal = receipt.passed | receipt.skipped | receipt.failed
    if terminal != set(receipt.nodeids):
        raise FullPytestContractError("not every collected test reached a terminal outcome")


def _receipt_payload(
    receipt: _SessionReceipt,
    contract: dict[str, Any],
    exit_code: int,
) -> dict[str, Any]:
    dependencies = {}
    for name in ("et-coord", "photsim7", "pytest"):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = None
    return {
        "schema_id": "et_mainsim.full_pytest_receipt.v1",
        "contract": {
            "schema_version": contract["schema_version"],
            "expected_nodeid_sha256": contract["nodeid_sha256"],
            "expected_collected": contract["expected_collected"],
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "dependencies": dependencies,
        },
        "result": {
            "pytest_exit_code": exit_code,
            "observed_nodeid_sha256": _nodeid_digest(receipt.nodeids),
            "collected": len(receipt.nodeids),
            "passed": len(receipt.passed),
            "skipped": sorted(receipt.skipped),
            "xfailed": sorted(receipt.xfailed),
            "failed": sorted(receipt.failed),
            "deselected": sorted(receipt.deselected),
        },
    }


def _write_receipt(payload: dict[str, Any]) -> None:
    target_text = os.environ.get("FULL_PYTEST_RECEIPT")
    if not target_text:
        raise FullPytestContractError("FULL_PYTEST_RECEIPT must name the CI artifact")
    target = Path(target_text)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def _runner_error_receipt(contract: dict[str, Any], message: str) -> dict[str, Any]:
    return {
        "schema_id": "et_mainsim.full_pytest_receipt.v1",
        "contract": {
            "schema_version": contract.get("schema_version"),
            "expected_nodeid_sha256": contract.get("nodeid_sha256"),
            "expected_collected": contract.get("expected_collected"),
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "result": {
            "runner_error": message,
        },
    }


def main() -> int:
    with CONTRACT_PATH.open("rb") as stream:
        contract = tomllib.load(stream)
    try:
        _validate_environment(contract)
    except FullPytestContractError as exc:
        _write_receipt(_runner_error_receipt(contract, str(exc)))
        raise

    receipt = _SessionReceipt()
    exit_code = int(
        pytest.main(
            ["-q", "--strict-markers", "-p", "no:cacheprovider"],
            plugins=[receipt],
        )
    )
    payload = _receipt_payload(receipt, contract, exit_code)
    try:
        _validate_session(receipt, contract)
        if exit_code != pytest.ExitCode.OK:
            raise FullPytestContractError(f"pytest exited with status {exit_code}")
    except FullPytestContractError as exc:
        payload["result"]["contract_validated"] = False
        payload["result"]["runner_error"] = str(exc)
        _write_receipt(payload)
        raise
    payload["result"]["contract_validated"] = True
    _write_receipt(payload)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, KeyError, TypeError, tomllib.TOMLDecodeError, FullPytestContractError) as exc:
        print(f"full pytest contract rejected: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
