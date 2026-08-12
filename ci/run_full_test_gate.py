from __future__ import annotations

import os
import sys


class FullTestGateError(RuntimeError):
    pass


def full_test_gate_allows(
    event_name: str,
    same_repository_pr: bool,
    full_test_result: str,
) -> bool:
    if event_name not in {"pull_request", "push", "workflow_dispatch"}:
        raise FullTestGateError("unsupported workflow event")
    if type(same_repository_pr) is not bool:
        raise FullTestGateError("same-repository PR identity must be boolean")
    if full_test_result not in {"success", "failure", "cancelled", "skipped"}:
        raise FullTestGateError("unsupported full-test result")

    if event_name == "pull_request":
        expected_result = "success" if same_repository_pr else "skipped"
        return full_test_result == expected_result
    return not same_repository_pr and full_test_result == "success"


def main() -> int:
    same_repository_text = os.environ.get("FULL_TEST_SAME_REPOSITORY_PR")
    if same_repository_text not in {"true", "false"}:
        raise FullTestGateError("FULL_TEST_SAME_REPOSITORY_PR must be true or false")
    event_name = os.environ.get("FULL_TEST_EVENT_NAME", "")
    full_test_result = os.environ.get("FULL_TEST_RESULT", "")
    same_repository_pr = same_repository_text == "true"

    if not full_test_gate_allows(
        event_name,
        same_repository_pr,
        full_test_result,
    ):
        raise FullTestGateError("the full-test result does not satisfy the event policy")
    if event_name == "pull_request" and not same_repository_pr:
        print(
            "::notice::Private-dependency tests are skipped for fork pull requests; "
            "a maintainer-owned branch is required before merge."
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FullTestGateError as exc:
        print(f"full-test gate rejected: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
