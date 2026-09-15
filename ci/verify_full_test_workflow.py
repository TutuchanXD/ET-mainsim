"""Validate CI security and execution invariants, not incidental YAML layout.

PyYAML is installed by the canonical CI bootstrap (not a runtime dependency).
"""
from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CI_WORKFLOW_PATH = ROOT / ".github/workflows/ci.yml"
FULL_WORKFLOW_PATH = ROOT / ".github/workflows/full-test.yml"
CONTRACT_PATH = ROOT / "ci/full_pytest_contract.toml"
TRUST = "github.event_name != 'pull_request' || github.event.pull_request.head.repo.full_name == github.repository"
ENVIRONMENT = "${{ github.event_name == 'pull_request' && 'full-test-pr-review' || 'full-test-private-dependency' }}"


class WorkflowContractError(RuntimeError):
    pass


def _require(value, message):
    if not value:
        raise WorkflowContractError(message)


class UniqueLoader(yaml.BaseLoader):
    def construct_mapping(self, node, deep=False):
        keys = [self.construct_object(key, deep=deep) for key, _ in node.value]
        _require(len(keys) == len(set(keys)), "duplicate YAML keys")
        return super().construct_mapping(node, deep=deep)


def _load(text):
    try:
        workflow = yaml.load(text, Loader=UniqueLoader)
        _require(set(workflow["on"]) == {"pull_request", "push", "workflow_dispatch"}, "trusted event set")
        _require(workflow["on"]["push"] == {"branches": ["main"]}, "main push boundary")
        _require(workflow["on"]["pull_request"] == "" and workflow["on"]["workflow_dispatch"] == "", "unfiltered PR and dispatch")
        _require(workflow["permissions"] == {"contents": "read"}, "read-only token")
        _require(workflow["concurrency"]["cancel-in-progress"] == "true", "cancel stale runs")
        _require("defaults" not in workflow, "no shell/environment overrides")
        _require("continue-on-error" not in text and "|| true" not in text, "failure masking")
        for job in workflow["jobs"].values():
            _require(job["runs-on"] == "ubuntu-24.04", "public repository must use standard hosted VMs")
            _require("permissions" not in job and "defaults" not in job, "no job permission/shell overrides")
            for step in job["steps"]:
                _require("name" in step and "shell" not in step, "named steps, no shell overrides")
                _require(not ({"working-directory", "continue-on-error"} & step.keys()), "no execution overrides")
                if "uses" in step:
                    action, sha = step["uses"].rsplit("@", 1)
                    _require(re.fullmatch("[0-9a-f]{40}", sha), "immutable action SHA")
                    if action == "actions/checkout":
                        _require(step["with"].get("persist-credentials") == "false", "credential-free checkout")
        return workflow
    except (KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise WorkflowContractError("invalid workflow structure") from exc


def _steps(job, actions, commands):
    steps = job["steps"]
    _require([s.get("uses", "").split("@")[0] for s in steps if "uses" in s] == actions, "action sequence")
    _require([s["run"] for s in steps if "run" in s] == commands, "canonical unfiltered commands")
    _require(len(steps) == len(actions) + len(commands), "unexpected executable step")
    for step in steps:
        if step.get("uses", "").startswith("actions/upload-artifact@"):
            _require(step.get("if") == "always()", "receipt upload even on failure")
        else:
            _require("if" not in step, "critical steps must not be skipped")
    return steps


def verify_workflow_text(ci_workflow, full_workflow, project, contract):
    _require(project.get("optional-dependencies", {}).get("test") == ["pandas>=2.2,<3", "pytest==9.0.3", "PyYAML==6.0.3"], "frozen test dependencies")
    ci, full = _load(ci_workflow), _load(full_workflow)
    _require(set(ci["jobs"]) == {"package-boundary"}, "smoke job inventory")
    _require(set(full["jobs"]) == {"full-test", "full-test-gate"}, "full job inventory")
    package = ci["jobs"]["package-boundary"]
    _require(package["name"] == "package-boundary / py${{ matrix.python-version }}" and "if" not in package, "required smoke check")
    _require(package["strategy"]["matrix"] == {"python-version": contract["python_versions"]}, "dual Python smoke")
    _require(package["strategy"]["fail-fast"] == "false", "complete smoke matrix")
    smoke_steps = _steps(package, ["actions/checkout", "actions/setup-python"], ["./scripts/ci/smoke.sh"])
    _require(smoke_steps[0]["with"] == {"persist-credentials": "false"}, "event checkout")
    _require(smoke_steps[1]["with"]["python-version"] == "${{ matrix.python-version }}", "smoke interpreter")
    job = full["jobs"]["full-test"]
    _require(job["name"] == "full-test / py${{ matrix.python-version }}", "full check names")
    _require(job.get("if") == TRUST, "fork boundary")
    _require(job.get("environment") == {"name": ENVIRONMENT}, "reviewed private-dependency environments")
    _require(job["strategy"]["matrix"] == {"python-version": contract["python_versions"]}, "full Python matrix")
    _require(job["strategy"]["fail-fast"] == "false", "complete full matrix")
    steps = _steps(job, ["actions/checkout"]*3 + ["actions/setup-python", "actions/upload-artifact"], ["./scripts/ci/full.sh"])
    _require(steps[0]["with"] == {"persist-credentials": "false"}, "trusted event ref")
    for step, name, key in [(steps[1], "ET-coordinate", "et_coordinate_commit"), (steps[2], "Photsim7", "photsim7_commit")]:
        settings = {"repository": f"TutuchanXD/{name}", "ref": contract["dependencies"][key],
                    "path": f".ci-dependencies/{name}", "persist-credentials": "false"}
        if name == "Photsim7":
            settings["ssh-key"] = "${{ secrets.PHOTSIM7_READ_ONLY_DEPLOY_KEY }}"
        _require(step["with"] == settings, "frozen dependency / read-only deploy key")
    _require(steps[3]["with"]["python-version"] == "${{ matrix.python-version }}", "full interpreter")
    _require(steps[4].get("env") == {
        "CUDA_VISIBLE_DEVICES": "",
        "ET_DATA_DIR": "${{ runner.temp }}/et-mainsim-ci-missing-data",
        "FULL_PYTEST_RECEIPT": "${{ runner.temp }}/full-pytest-receipt-py${{ matrix.python-version }}.json",
        "MPLBACKEND": "Agg", "PYTHONDONTWRITEBYTECODE": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "PYTEST_ADDOPTS": "",
    }, "hermetic execution and receipt binding")
    _require(steps[5]["with"] == {
        "name": "full-pytest-receipt-py${{ matrix.python-version }}",
        "path": "${{ runner.temp }}/full-pytest-receipt-py${{ matrix.python-version }}.json",
        "if-no-files-found": "error", "retention-days": "14",
    }, "receipt upload semantics")
    gate = full["jobs"]["full-test-gate"]
    _require(gate["name"] == "full-test-gate" and gate.get("if") == "always()" and gate.get("needs") == "full-test", "always instantiated required gate")
    _require("environment" not in gate, "gate cannot receive private secrets")
    gate_steps = _steps(gate, ["actions/checkout"], ["python -m ci.run_full_test_gate"])
    _require(gate_steps[0]["with"] == {"persist-credentials": "false"}, "gate event checkout")
    _require(gate_steps[1].get("env") == {
        "FULL_TEST_EVENT_NAME": "${{ github.event_name }}",
        "FULL_TEST_RESULT": "${{ needs.full-test.result }}",
        "FULL_TEST_SAME_REPOSITORY_PR": "${{ github.event_name == 'pull_request' && github.event.pull_request.head.repo.full_name == github.repository }}",
    }, "gate event result bindings")


def verify_repository(root=ROOT):
    with (root / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)["project"]
    with (root / "ci/full_pytest_contract.toml").open("rb") as stream:
        contract = tomllib.load(stream)
    verify_workflow_text((root / ".github/workflows/ci.yml").read_text(),
                         (root / ".github/workflows/full-test.yml").read_text(), project, contract)


def main():
    try:
        verify_repository()
    except (OSError, KeyError, TypeError, WorkflowContractError) as exc:
        print(f"full-test CI contract rejected: {exc}", file=sys.stderr)
        return 1
    print("full-test CI contract verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
