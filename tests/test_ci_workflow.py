from __future__ import annotations

import hashlib
import re
import sys
import tomllib
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from ci.run_full_pytest import (
    FullPytestContractError,
    _runner_error_receipt,
    _SessionReceipt,
    _validate_environment,
    _validate_session,
)
from ci.verify_full_test_workflow import (
    WorkflowContractError,
    verify_repository,
    verify_workflow_text,
)

_ROOT = Path(__file__).resolve().parents[1]
_CI_WORKFLOW_PATH = _ROOT / ".github" / "workflows" / "ci.yml"
_FULL_WORKFLOW_PATH = _ROOT / ".github" / "workflows" / "full-test.yml"


def _workflow_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _job_block(workflow: str, name: str) -> str:
    lines = workflow.splitlines()
    marker = f"  {name}:"
    try:
        start = lines.index(marker)
    except ValueError as exc:
        raise AssertionError(f"CI workflow is missing the {name!r} job") from exc

    end = len(lines)
    for index in range(start + 1, len(lines)):
        if re.fullmatch(r"  [A-Za-z0-9_-]+:", lines[index]):
            end = index
            break
    return "\n".join(lines[start:end])


def test_project_declares_pytest_test_extra() -> None:
    with (_ROOT / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)["project"]

    assert project["optional-dependencies"]["test"] == [
        "pandas>=2.2,<3",
        "pytest==9.0.3",
        "PyYAML==6.0.3",
    ]


def test_public_repository_rejects_persistent_self_hosted_runners() -> None:
    with (_ROOT / 'pyproject.toml').open('rb') as stream:
        project = tomllib.load(stream)['project']
    with (_ROOT / 'ci/full_pytest_contract.toml').open('rb') as stream:
        contract = tomllib.load(stream)
    for target in ('ci', 'full'):
        ci_text = _workflow_text(_CI_WORKFLOW_PATH)
        full_text = _workflow_text(_FULL_WORKFLOW_PATH)
        if target == 'ci':
            ci_text = ci_text.replace('ubuntu-24.04', '[self-hosted, linux, x64]')
        else:
            full_text = full_text.replace('ubuntu-24.04', '[self-hosted, linux, x64]')
        with pytest.raises(WorkflowContractError):
            verify_workflow_text(ci_text, full_text, project, contract)


def test_local_full_entry_uses_frozen_dependencies_and_receipt_runner() -> None:
    from ci.run_local import install_plan, hermetic_environment
    commands = install_plan(Path('/python'), Path('/repo'), Path('/coordinate'), Path('/photsim'))
    assert any('https://download.pytorch.org/whl/cpu' in command for command in commands)
    assert commands[-1][-2:] == ['pip', 'check']
    assert any('/photsim[gpu]' in command for command in commands)
    assert any('/repo[test,release]' in command for command in commands)
    env = hermetic_environment(Path('/isolated'), {'PYTEST_ADDOPTS': '-k subset', 'ET_DATA_DIR': '/science'})
    assert env['PYTEST_ADDOPTS'] == ''
    assert env['ET_DATA_DIR'] == '/isolated/missing-data'
    assert env['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] == '1'
    assert env['CUDA_VISIBLE_DEVICES'] == ''


def test_smoke_wrapper_cannot_be_bypassed_with_ci_python():
    import os
    import subprocess
    result = subprocess.run([str(_ROOT / 'scripts/ci/smoke.sh'), '--help'],
                            env={**os.environ, 'CI_PYTHON': '/bin/true'},
                            capture_output=True, text=True, check=True)
    assert '--python' in result.stdout


def test_test_extra_declares_yaml_parser():
    with (_ROOT / 'pyproject.toml').open('rb') as stream:
        project = tomllib.load(stream)['project']
    assert 'PyYAML==6.0.3' in project['optional-dependencies']['test']


def test_frozen_dependency_rejects_untracked_source_inputs(tmp_path, monkeypatch):
    from ci import run_local
    import subprocess
    root = tmp_path / 'repo'
    target = root / '.ci-dependencies/Photsim7'
    target.mkdir(parents=True)
    subprocess.run(['git', 'init', '-q', target], check=True)
    subprocess.run(['git', '-C', target, '-c', 'user.name=CI Test', '-c',
                    'user.email=test@example.invalid', 'commit', '--allow-empty', '-qm', 'fixture'], check=True)
    sha = subprocess.check_output(['git', '-C', target, 'rev-parse', 'HEAD'], text=True).strip()
    (target / 'unexpected_module.py').write_text('VALUE = 1\n')
    monkeypatch.setattr(run_local, 'ROOT', root)
    with pytest.raises(ValueError, match='clean checkout'):
        run_local.dependency('Photsim7', sha, tmp_path, {})


def test_ci_runs_unfiltered_full_suite_on_supported_python_versions() -> None:
    block = _job_block(_workflow_text(_FULL_WORKFLOW_PATH), "full-test")

    assert re.search(
        r'python-version:\s*\["3\.12",\s*"3\.13"\]',
        block,
    )
    assert "./scripts/ci/full.sh" in block
    assert "python -m pytest" not in block
    assert not re.search(
        r"(?:^|\s)(?:-k|--ignore|--deselect|--collect-only)(?:\s|=)",
        block,
    )
    assert not re.search(r"pytest[^\n]*\s-m(?:\s|=)", block)


def test_full_test_gate_is_always_instantiated_for_branch_protection() -> None:
    block = _job_block(_workflow_text(_FULL_WORKFLOW_PATH), "full-test-gate")

    assert "needs: full-test" in block
    assert "if: always()" in block
    assert "run: python -m ci.run_full_test_gate" in block


def test_full_suite_uses_frozen_runtime_dependencies_and_no_science_data() -> None:
    workflow = _workflow_text(_FULL_WORKFLOW_PATH)
    block = _job_block(workflow, "full-test")

    assert "repository: TutuchanXD/ET-coordinate" in block
    assert "ref: f9cec8038b021c9540a026b94e876dc3240071d1" in block
    assert "repository: TutuchanXD/Photsim7" in block
    assert "ref: 7611ef0857d27847b03306d9d7472b0d5c028a93" in block
    assert "ssh-key: ${{ secrets.PHOTSIM7_READ_ONLY_DEPLOY_KEY }}" in block
    assert "ET_DATA_DIR: ${{ runner.temp }}/et-mainsim-ci-missing-data" in block
    assert "pull_request:" in workflow
    assert (
        "if: github.event_name != 'pull_request' || "
        "github.event.pull_request.head.repo.full_name == github.repository"
    ) in block
    assert (
        "name: ${{ github.event_name == 'pull_request' && "
        "'full-test-pr-review' || 'full-test-private-dependency' }}"
    ) in block


def test_full_suite_installs_release_tools_for_release_engineering_tests() -> None:
    from ci.run_local import install_plan
    commands = install_plan(Path('/python'), Path('/repo'), Path('/coordinate'), Path('/photsim'))
    assert commands == [
        ['/python', '-m', 'pip', 'install', '--upgrade', 'pip'],
        ['/python', '-m', 'pip', 'install', '--index-url', 'https://download.pytorch.org/whl/cpu', 'torch>=2.7,<3'],
        ['/python', '-m', 'pip', 'install', '/coordinate'],
        ['/python', '-m', 'pip', 'install', '/photsim[gpu]'],
        ['/python', '-m', 'pip', 'install', '-e', '/repo[test,release]'],
        ['/python', '-m', 'pip', 'check'],
    ]


def test_ci_actions_and_checkout_credentials_are_locked_down() -> None:
    workflow = _workflow_text(_CI_WORKFLOW_PATH) + _workflow_text(
        _FULL_WORKFLOW_PATH
    )
    action_refs = re.findall(
        r"^\s+(?:-\s+)?uses:\s+\S+@(\S+?)(?:\s+#.*)?$",
        workflow,
        re.MULTILINE,
    )

    assert action_refs
    assert all(re.fullmatch(r"[0-9a-f]{40}", ref) for ref in action_refs)
    assert "permissions:\n  contents: read" in workflow
    assert workflow.count("persist-credentials: false") >= 3


def test_lightweight_job_bootstraps_the_ci_contract_verifier() -> None:
    block = _job_block(_workflow_text(_CI_WORKFLOW_PATH), "package-boundary")

    assert "./scripts/ci/smoke.sh" in block
    source = (_ROOT / 'ci/run_local.py').read_text()
    assert "'ci.verify_full_test_workflow'" in source
    assert "'--version'" in source and "'--help'" in source
    assert (_ROOT / "ci" / "verify_full_test_workflow.py").is_file()
    assert (_ROOT / "ci" / "run_full_pytest.py").is_file()


def test_stdlib_ci_contract_verifier_accepts_repository() -> None:
    verify_repository()
    ci_workflow = _workflow_text(_CI_WORKFLOW_PATH)
    full_workflow = _workflow_text(_FULL_WORKFLOW_PATH)
    with (_ROOT / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)["project"]
    with (_ROOT / "ci/full_pytest_contract.toml").open("rb") as stream:
        contract = tomllib.load(stream)
    # Comments/reformatting are not architecture; executable substitutions are.
    verify_workflow_text(ci_workflow + "\n# harmless comment\n", full_workflow, project, contract)
    for replacement in (
        "run: echo skipped # ./scripts/ci/full.sh",
        "run: |\n          cat <<'EOF'\n          ./scripts/ci/full.sh\n          EOF",
        "shell: 'true {0}'\n        run: ./scripts/ci/full.sh",
    ):
        changed = full_workflow.replace("run: ./scripts/ci/full.sh", replacement)
        assert changed != full_workflow
        with pytest.raises(WorkflowContractError):
            verify_workflow_text(ci_workflow, changed, project, contract)


@pytest.mark.parametrize(
    ("workflow_name", "before", "after"),
    [
        (
            "full",
            "run: ./scripts/ci/full.sh",
            "run: python -m pytest -q",
        ),
        ("full", "timeout-minutes: 30", "continue-on-error: true"),
        (
            "full",
            'python-version: ["3.12", "3.13"]',
            'python-version: ["3.12"]',
        ),
        (
            "full",
            'run: ./scripts/ci/full.sh',
            'run: ./scripts/ci/full.sh --skip-release-tools',
        ),
        (
            "ci",
            "- name: Verify full-test CI contract and package boundary\n        run:",
            "- name: Verify full-test CI contract and package boundary\n        if: ${{ false }}\n        run:",
        ),
        (
            "full",
            "- name: Check out frozen Photsim7 release\n        uses:",
            "- name: Check out frozen Photsim7 release\n        if: ${{ false }}\n        uses:",
        ),
        (
            "full",
            "- name: Run complete hermetic test suite\n        env:",
            "- name: Run complete hermetic test suite\n        if: ${{ false }}\n        env:",
        ),
        (
            "full",
            "on:\n  pull_request:\n  push:",
            "on:\n  pull_request:\n  pull_request_target:\n  push:",
        ),
        (
            "full",
            "github.event.pull_request.head.repo.full_name == github.repository",
            "github.event.pull_request.head.repo.full_name != github.repository",
        ),
        (
            "full",
            "'full-test-pr-review' || 'full-test-private-dependency'",
            "'full-test-private-dependency' || 'full-test-pr-review'",
        ),
        (
            "full",
            "if: github.event_name != 'pull_request' || "
            "github.event.pull_request.head.repo.full_name == github.repository",
            "if: github.event_name != 'pull_request' || "
            "github.event.pull_request.head.repo.full_name == github.repository "
            "&& github.actor == 'nobody'",
        ),
        (
            "full",
            "run: ./scripts/ci/full.sh",
            "run: ./scripts/ci/full.sh || exit 0",
        ),
        (
            "full",
            "run: ./scripts/ci/full.sh",
            "run: ./scripts/ci/full.sh; exit 0",
        ),
        (
            "ci",
            "      - name: Check out ET-mainsim\n"
            "        uses: actions/checkout@9c091bb21b7c1c1d1991bb908d89e4e9dddfe3e0 # v7.0.0\n"
            "        with:\n"
            "          persist-credentials: false",
            "      - name: Check out ET-mainsim\n"
            "        uses: actions/checkout@9c091bb21b7c1c1d1991bb908d89e4e9dddfe3e0 # v7.0.0",
        ),
        (
            "full",
            "      - name: Check out ET-mainsim\n"
            "        uses: actions/checkout@9c091bb21b7c1c1d1991bb908d89e4e9dddfe3e0 # v7.0.0\n"
            "        with:\n"
            "          persist-credentials: false",
            "      - name: Check out ET-mainsim\n"
            "        uses: actions/checkout@9c091bb21b7c1c1d1991bb908d89e4e9dddfe3e0 # v7.0.0",
        ),
        (
            "full",
            "          path: .ci-dependencies/ET-coordinate\n"
            "          persist-credentials: false",
            "          path: .ci-dependencies/ET-coordinate",
        ),
        (
            "full",
            "          ssh-key: ${{ secrets.PHOTSIM7_READ_ONLY_DEPLOY_KEY }}\n"
            "          persist-credentials: false",
            "          ssh-key: ${{ secrets.PHOTSIM7_READ_ONLY_DEPLOY_KEY }}",
        ),
        (
            "full",
            "  full-test-gate:\n"
            "    name: full-test-gate\n"
            "    needs: full-test\n"
            "    if: always()",
            "  full-test-gate:\n"
            "    name: full-test-gate\n"
            "    needs: full-test\n"
            "    if: github.actor == 'nobody'",
        ),
        (
            "full",
            "run: python -m ci.run_full_test_gate",
            "run: python -m ci.run_full_test_gate || exit 0",
        ),
        (
            "full",
            "      - name: Run complete hermetic test suite\n",
            "      - run: python -c \"print(123)\"\n"
            "      - name: Run complete hermetic test suite\n",
        ),
    ],
)
def test_stdlib_ci_contract_verifier_rejects_gate_bypasses(
    workflow_name: str,
    before: str,
    after: str,
) -> None:
    ci_workflow = _workflow_text(_CI_WORKFLOW_PATH)
    full_workflow = _workflow_text(_FULL_WORKFLOW_PATH)
    if workflow_name == "ci":
        assert before in ci_workflow
        ci_workflow = ci_workflow.replace(before, after)
    else:
        assert before in full_workflow
        full_workflow = full_workflow.replace(before, after)
    with (_ROOT / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)["project"]
    with (_ROOT / "ci" / "full_pytest_contract.toml").open("rb") as stream:
        contract = tomllib.load(stream)

    with pytest.raises(WorkflowContractError):
        verify_workflow_text(ci_workflow, full_workflow, project, contract)


def test_stdlib_ci_contract_verifier_rejects_quoted_pr_target_head_checkout() -> None:
    ci_workflow = _workflow_text(_CI_WORKFLOW_PATH)
    full_workflow = _workflow_text(_FULL_WORKFLOW_PATH).replace(
        "on:\n  pull_request:\n  push:",
        'on:\n  pull_request:\n  "pull_request_target":\n  push:',
    ).replace(
        "        with:\n          persist-credentials: false",
        "        with:\n          ref: ${{ github.event.pull_request.head.sha }}\n"
        "          persist-credentials: false",
        1,
    )
    with (_ROOT / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)["project"]
    with (_ROOT / "ci" / "full_pytest_contract.toml").open("rb") as stream:
        contract = tomllib.load(stream)

    with pytest.raises(WorkflowContractError):
        verify_workflow_text(ci_workflow, full_workflow, project, contract)


@pytest.mark.parametrize("tamper", ["deselected", "skip"])
def test_full_pytest_receipt_rejects_incomplete_execution(
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    nodeids = ["tests/test_ci_workflow.py::test_example"]
    contract = {
        "test_files": ["tests/test_ci_workflow.py"],
        "expected_collected": 1,
        "nodeid_sha256": hashlib.sha256(nodeids[0].encode("utf-8")).hexdigest(),
        "allowed_skips": [],
    }
    receipt = _SessionReceipt(nodeids=nodeids, passed=set(nodeids))
    monkeypatch.setattr(
        "ci.run_full_pytest._tracked_test_files",
        lambda: ["tests/test_ci_workflow.py"],
    )
    if tamper == "deselected":
        receipt.deselected.append(nodeids[0])
    else:
        receipt.passed.clear()
        receipt.skipped.add(nodeids[0])

    with pytest.raises(FullPytestContractError):
        _validate_session(receipt, deepcopy(contract))


def test_full_pytest_receipt_treats_xpass_as_forbidden() -> None:
    receipt = _SessionReceipt()
    report = SimpleNamespace(
        nodeid="tests/test_ci_workflow.py::test_example",
        failed=False,
        skipped=False,
        passed=True,
        when="call",
        wasxfail="expected failure",
    )

    receipt.pytest_runtest_logreport(report)

    assert report.nodeid in receipt.xfailed
    assert report.nodeid not in receipt.passed


def test_full_pytest_receipt_makes_teardown_failure_authoritative() -> None:
    receipt = _SessionReceipt()
    nodeid = "tests/test_ci_workflow.py::test_example"
    receipt.pytest_runtest_logreport(
        SimpleNamespace(
            nodeid=nodeid,
            failed=False,
            skipped=False,
            passed=True,
            when="call",
        )
    )
    receipt.pytest_runtest_logreport(
        SimpleNamespace(
            nodeid=nodeid,
            failed=True,
            skipped=False,
            passed=False,
            when="teardown",
        )
    )

    assert nodeid in receipt.failed
    assert nodeid not in receipt.passed


@pytest.mark.parametrize(
    ("event_name", "same_repository_pr", "full_test_result", "expected"),
    [
        ("pull_request", True, "success", True),
        ("pull_request", True, "skipped", False),
        ("pull_request", False, "skipped", True),
        ("pull_request", False, "success", False),
        ("push", False, "success", True),
        ("push", False, "failure", False),
        ("workflow_dispatch", False, "success", True),
        ("workflow_dispatch", False, "cancelled", False),
    ],
)
def test_full_test_gate_truth_table(
    event_name: str,
    same_repository_pr: bool,
    full_test_result: str,
    expected: bool,
) -> None:
    from ci.run_full_test_gate import full_test_gate_allows

    assert (
        full_test_gate_allows(event_name, same_repository_pr, full_test_result)
        is expected
    )


@pytest.mark.parametrize(
    ("argv", "pytest_addopts"),
    [
        (["run_full_pytest.py", "-k", "subset"], ""),
        (["run_full_pytest.py"], "-m subset"),
    ],
)
def test_full_pytest_runner_rejects_external_selectors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    argv: list[str],
    pytest_addopts: str,
) -> None:
    monkeypatch.setattr("ci.run_full_pytest.sys.argv", argv)
    monkeypatch.setenv("PYTEST_ADDOPTS", pytest_addopts)
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    monkeypatch.setenv("ET_DATA_DIR", str(tmp_path / "missing"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    contract = {
        "pytest_version": pytest.__version__,
        "python_versions": [f"{sys.version_info.major}.{sys.version_info.minor}"],
    }

    with pytest.raises(FullPytestContractError):
        _validate_environment(contract)


def test_full_pytest_runner_error_receipt_is_explicit() -> None:
    payload = _runner_error_receipt(
        {
            "schema_version": 1,
            "expected_collected": 912,
            "nodeid_sha256": "a" * 64,
        },
        "synthetic runner rejection",
    )

    assert payload["result"] == {"runner_error": "synthetic runner rejection"}
    assert payload["contract"]["expected_collected"] == 912


def test_full_pytest_receipt_separates_expected_and_observed_inventory() -> None:
    from ci.run_full_pytest import _receipt_payload

    receipt = _SessionReceipt(nodeids=["tests/test_ci_workflow.py::observed"])
    payload = _receipt_payload(
        receipt,
        {
            "schema_version": 1,
            "expected_collected": 1,
            "nodeid_sha256": "a" * 64,
        },
        0,
    )

    assert payload["contract"]["expected_nodeid_sha256"] == "a" * 64
    assert payload["result"]["observed_nodeid_sha256"] != "a" * 64
