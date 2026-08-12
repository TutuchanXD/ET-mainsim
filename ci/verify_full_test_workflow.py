from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CI_WORKFLOW_PATH = ROOT / ".github" / "workflows" / "ci.yml"
FULL_WORKFLOW_PATH = ROOT / ".github" / "workflows" / "full-test.yml"
CONTRACT_PATH = ROOT / "ci" / "full_pytest_contract.toml"


class WorkflowContractError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise WorkflowContractError(message)


def _job_block(workflow: str, name: str) -> str:
    lines = workflow.splitlines()
    marker = f"  {name}:"
    try:
        start = lines.index(marker)
    except ValueError as exc:
        raise WorkflowContractError(f"CI workflow is missing the {name!r} job") from exc
    end = len(lines)
    for index in range(start + 1, len(lines)):
        if re.fullmatch(r"  [A-Za-z0-9_-]+:", lines[index]):
            end = index
            break
    return "\n".join(lines[start:end])


def _step_blocks(job: str) -> list[tuple[str, str]]:
    lines = job.splitlines()
    step_lines = [line for line in lines if line.startswith("      - ")]
    _require(
        all(re.fullmatch(r"      - name: .+", line) for line in step_lines),
        "every CI step must be named and part of the frozen sequence",
    )
    starts = [
        (index, match.group(1))
        for index, line in enumerate(lines)
        if (match := re.fullmatch(r"      - name: (.+)", line))
    ]
    steps = []
    for position, (start, name) in enumerate(starts):
        end = starts[position + 1][0] if position + 1 < len(starts) else len(lines)
        steps.append((name, "\n".join(lines[start:end])))
    return steps


def _require_step_contract(
    job: str,
    expected_names: tuple[str, ...],
    allowed_conditions: dict[str, str] | None = None,
) -> dict[str, str]:
    steps = _step_blocks(job)
    names = tuple(name for name, _block in steps)
    _require(names == expected_names, "CI job steps differ from the frozen sequence")
    blocks = dict(steps)
    allowed_conditions = allowed_conditions or {}
    for name, block in steps:
        conditions = [
            line.strip()
            for line in block.splitlines()
            if line.startswith("        if:")
        ]
        expected = allowed_conditions.get(name)
        if expected is None:
            _require(not conditions, f"critical step {name!r} must be unconditional")
        else:
            _require(conditions == [expected], f"step {name!r} has an unsafe condition")
    return blocks


def _require_checkout_credentials_disabled(step: str, name: str) -> None:
    settings = [
        line.strip()
        for line in step.splitlines()
        if line.startswith("          persist-credentials:")
    ]
    _require(
        settings == ["persist-credentials: false"],
        f"checkout step {name!r} must disable credential persistence",
    )


def _verify_shared_workflow_controls(workflow: str) -> None:
    _require("permissions:\n  contents: read" in workflow, "contents must be read-only")
    _require("cancel-in-progress: true" in workflow, "stale CI runs must be cancelled")
    _require("continue-on-error" not in workflow, "CI must not tolerate step failures")
    _require("|| true" not in workflow, "CI commands must not mask failures")

    action_refs = re.findall(
        r"^\s+(?:-\s+)?uses:\s+\S+@(\S+?)(?:\s+#.*)?$",
        workflow,
        re.MULTILINE,
    )
    _require(bool(action_refs), "CI must use explicit actions")
    _require(
        all(re.fullmatch(r"[0-9a-f]{40}", ref) for ref in action_refs),
        "every GitHub Action must be pinned to an immutable commit",
    )


def _require_frozen_triggers(workflow: str) -> None:
    trigger_block = workflow.split("\non:\n", 1)
    _require(len(trigger_block) == 2, "workflow must declare its triggers once")
    actual = trigger_block[1].split("\npermissions:\n", 1)
    _require(len(actual) == 2, "workflow trigger block is malformed")
    _require(
        actual[0]
        == "  pull_request:\n"
        "  push:\n"
        "    branches: [main]\n"
        "  workflow_dispatch:\n",
        "workflow triggers differ from the frozen trusted event set",
    )


def verify_workflow_text(
    ci_workflow: str,
    full_workflow: str,
    project: dict,
    contract: dict,
) -> None:
    test_dependencies = project.get("optional-dependencies", {}).get("test")
    _require(
        test_dependencies == ["pandas>=2.2,<3", "pytest==9.0.3"],
        "project.test must freeze the direct test dependencies",
    )

    _verify_shared_workflow_controls(ci_workflow)
    _verify_shared_workflow_controls(full_workflow)
    _require_frozen_triggers(ci_workflow)
    _require_frozen_triggers(full_workflow)

    package = _job_block(ci_workflow, "package-boundary")
    _require("\n    if:" not in package, "package-boundary must not be skipped")
    package_steps = _require_step_contract(
        package,
        (
            "Check out ET-mainsim",
            "Set up Python",
            "Verify full-test CI contract",
            "Build package without private integration dependency",
            "Verify lightweight boundary",
        ),
    )
    _require_checkout_credentials_disabled(
        package_steps["Check out ET-mainsim"], "package-boundary ET-mainsim"
    )
    _require(
        [
            line.strip()
            for line in package_steps["Verify full-test CI contract"].splitlines()
            if line.startswith("        run:")
        ]
        == ["run: python -m ci.verify_full_test_workflow"],
        "package-boundary must bootstrap the full-test contract verifier",
    )

    full = _job_block(full_workflow, "full-test")
    job_header = full.split("\n    steps:", 1)[0]
    job_conditions = [
        line.strip() for line in job_header.splitlines() if line.startswith("    if:")
    ]
    _require(
        job_conditions
        == [
            "if: github.event_name != 'pull_request' || "
            "github.event.pull_request.head.repo.full_name == github.repository"
        ],
        "full-test must skip untrusted fork pull requests",
    )
    _require(
        "environment:\n      name: ${{ github.event_name == 'pull_request' && "
        "'full-test-pr-review' || 'full-test-private-dependency' }}"
        in job_header,
        "full-test must select the reviewed PR or main-only environment",
    )
    full_steps = _require_step_contract(
        full,
        (
            "Check out ET-mainsim",
            "Check out frozen ET-coordinate",
            "Check out frozen Photsim7 release",
            "Set up Python",
            "Install frozen CPU runtime and test dependencies",
            "Run complete hermetic test suite",
            "Upload full-test receipt",
        ),
        {
            "Upload full-test receipt": "if: always()",
        },
    )
    for checkout_name in (
        "Check out ET-mainsim",
        "Check out frozen ET-coordinate",
        "Check out frozen Photsim7 release",
    ):
        _require_checkout_credentials_disabled(full_steps[checkout_name], checkout_name)
    versions = contract["python_versions"]
    matrix = ", ".join(f'"{version}"' for version in versions)
    _require(
        f"python-version: [{matrix}]" in full,
        "full-test must cover every supported Python version",
    )
    _require(
        [
            line.strip()
            for line in full_steps["Run complete hermetic test suite"].splitlines()
            if line.startswith("        run:")
        ]
        == ["run: python -m ci.run_full_pytest"],
        "full-test must invoke the controlled pytest runner exactly once",
    )
    _require(
        "\n          ref:" not in full_steps["Check out ET-mainsim"],
        "ET-mainsim must be checked out from the trusted event ref",
    )
    _require("python -m pytest" not in full, "workflow must not bypass the runner")
    _require(
        "PYTEST_ADDOPTS: \"\"" in full,
        "ambient pytest options must be disabled",
    )
    _require(
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD: \"1\"" in full,
        "third-party pytest plugin autoload must be disabled",
    )
    for selector in (" -k ", "--ignore", "--deselect", "--collect-only"):
        _require(selector not in full, f"full-test must not filter tests with {selector!r}")

    dependencies = contract["dependencies"]
    for repository, key in (
        ("TutuchanXD/ET-coordinate", "et_coordinate_commit"),
        ("TutuchanXD/Photsim7", "photsim7_commit"),
    ):
        _require(f"repository: {repository}" in full, f"missing checkout for {repository}")
        _require(
            f"ref: {dependencies[key]}" in full,
            f"{repository} must be checked out at its frozen commit",
        )
    _require(
        "ssh-key: ${{ secrets.PHOTSIM7_READ_ONLY_DEPLOY_KEY }}" in full,
        "the private dependency must use the read-only deploy key",
    )
    _require(
        "ET_DATA_DIR: ${{ runner.temp }}/et-mainsim-ci-missing-data" in full,
        "the full suite must prove it does not require scientific data assets",
    )
    _require(
        "CUDA_VISIBLE_DEVICES: \"\"" in full,
        "the full suite must stay on the CPU path",
    )
    _require("python -m pip check" in full, "installed dependencies must be checked")
    _require(
        "actions/upload-artifact@" in full and "if: always()" in full,
        "the test receipt must be uploaded even after a failure",
    )
    _require(
        "if-no-files-found: error" in full_steps["Upload full-test receipt"],
        "a missing full-test receipt must fail the job",
    )

    gate = _job_block(full_workflow, "full-test-gate")
    gate_header = gate.split("\n    steps:", 1)[0]
    _require(
        [line.strip() for line in gate_header.splitlines() if line.startswith("    if:")]
        == ["if: always()"],
        "full-test-gate must run even when the private matrix is skipped",
    )
    _require("\n    needs: full-test" in gate_header, "full-test-gate must depend on the matrix")
    _require("\n    environment:" not in gate_header, "full-test-gate must not receive secrets")
    gate_steps = _require_step_contract(
        gate,
        (
            "Check out ET-mainsim",
            "Evaluate full-test result",
        ),
    )
    _require_checkout_credentials_disabled(
        gate_steps["Check out ET-mainsim"], "full-test-gate ET-mainsim"
    )
    evaluator = gate_steps["Evaluate full-test result"]
    for required in (
        "FULL_TEST_EVENT_NAME: ${{ github.event_name }}",
        "FULL_TEST_RESULT: ${{ needs.full-test.result }}",
        "FULL_TEST_SAME_REPOSITORY_PR: ${{ github.event_name == 'pull_request' && "
        "github.event.pull_request.head.repo.full_name == github.repository }}",
    ):
        _require(required in evaluator, "full-test-gate event binding is incomplete")
    _require(
        [
            line.strip()
            for line in evaluator.splitlines()
            if line.startswith("        run:")
        ]
        == ["run: python -m ci.run_full_test_gate"],
        "full-test-gate must invoke the controlled evaluator exactly once",
    )


def verify_repository(root: Path = ROOT) -> None:
    ci_workflow = (root / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )
    full_workflow = (root / ".github" / "workflows" / "full-test.yml").read_text(
        encoding="utf-8"
    )
    with (root / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)["project"]
    with (root / "ci" / "full_pytest_contract.toml").open("rb") as stream:
        contract = tomllib.load(stream)
    verify_workflow_text(ci_workflow, full_workflow, project, contract)


def main() -> int:
    try:
        verify_repository()
    except (OSError, KeyError, TypeError, tomllib.TOMLDecodeError, WorkflowContractError) as exc:
        print(f"full-test CI contract rejected: {exc}", file=sys.stderr)
        return 1
    print("full-test CI contract verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
