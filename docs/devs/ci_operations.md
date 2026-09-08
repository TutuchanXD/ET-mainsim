# Public-repository CI and local verification

**ET-mainsim is public; persistent self-hosted runners are intentionally not attached to the repository.**

Public PRs run on standard GitHub-hosted Ubuntu 24.04 disposable VMs. A
same-repository `if` condition is not a justification for attaching a
persistent runner to a public repository: an untrusted PR can modify workflow
code. Repository visibility is unchanged.

Standard GitHub-hosted Actions usage in public repositories has no hosted
minute billing; see [GitHub Actions billing](https://docs.github.com/en/billing/concepts/product-billing/github-actions).
The local full-test entry point is for trusted development/pre-push/agent
iterations, **not** automatic execution of arbitrary public PRs on a workstation.
See also [GitHub's self-hosted security warning](https://docs.github.com/en/actions/reference/security/secure-use).

## Stable check and event contract

- `CI`: `package-boundary / py3.12` and `package-boundary / py3.13` remain
  hosted. Both took only about 25–32 seconds in the audited recent runs.
- `Full test`: both `full-test / py...` jobs and `full-test-gate` remain hosted.
- PR, push main, and manual dispatch remain enabled. Main full-test is retained
  here for exact-commit release/provenance evidence; public minute cost is zero.
- Fork PRs do not enter the private-dependency environment or receive its key.
  The existing always-instantiated gate explicitly reports the fork-policy
  skip, not a fabricated full-suite pass. A maintainer-owned reviewed branch
  is required before integrating external changes. This existing procedural
  requirement is not a new machine-enforced branch rule.
- Same-repository PRs continue through `full-test-pr-review`; main/dispatch
  uses `full-test-private-dependency`. Existing environment policies are not
  weakened or removed.

Required checks are unchanged: `package-boundary / py3.12`,
`package-boundary / py3.13`, `full-test-gate`. Main protection is strict with
admin enforcement and conversation resolution; no rulesets were present at
the audit. No GitHub UI rename is needed.

## One implementation, local and hosted

With `etbase` activated, from the repository root:

```bash
./scripts/ci/smoke.sh --python python3.12
./scripts/ci/smoke.sh --python python3.13
./scripts/ci/full.sh --python python3.12
./scripts/ci/full.sh --python python3.13
python -m ci.verify_full_test_workflow
```

Absolute interpreter paths are supported. Each invocation creates and deletes
a fresh temporary **CI install-proof environment**; the developer's `etbase`
is not modified. No system packages or old editable installation are reused.
The standalone verifier uses CI-only `PyYAML==6.0.3`; smoke/full bootstrap it
automatically. Local full needs authorized ordinary Git SSH access to the
private Photsim7 repository; it never creates, stores, or asks for a deploy key.
Hosted checkouts continue to use the existing read-only deploy key.

`ci.run_local` installs the exact frozen ET-coordinate and Photsim7 snapshots
from `ci/full_pytest_contract.toml`, verifies checkout SHA and cleanliness,
installs CPU Torch and the existing `[test,release]` extras, runs `pip check`,
and calls the unchanged `ci.run_full_pytest` receipt runner. It does not accept
pytest selectors. CUDA, ambient options/plugin autoload, and science-data
paths are disabled; each suite gets a unique intentionally missing data path.
The original allowed-skip list, frozen dependency commits, and full receipt
rules are preserved. The collection inventory changes only for CI contract
tests and their mutation IDs, not scientific tests.

Outputs default to ignored `.ci-results/`; `--output` chooses another directory.
Actions binds `FULL_PYTEST_RECEIPT` to the original per-version artifact path
and still uploads it with `always()` / `if-no-files-found: error`. A setup
failure may produce no pytest receipt; the job and missing-artifact step then
fail, never produce success. Temporary environments are removed on normal
exit or a Python exception; forced machine termination may require local
administrator cleanup of the exact stale CI temp directory.

The verifier now parses YAML and checks structural invariants: event set,
hosted boundary, read-only tokens, immutable actions, dual Python coverage,
frozen dependency and deploy-key binding, canonical execution, environment
and receipt semantics, and gate inputs. Duplicate YAML keys, shell overrides,
conditional critical steps, alternate pytest invocations, and failure masking
are rejected. Harmless comments are not frozen as executable architecture.

`release.yml`, release tool pins, tag rules, approval evidence, wheel inventory,
and release permissions are untouched. The sdist allowlist in the release
contract adds exactly the new CI helper, two shell entry points, and this
operations document; its strict extra-member rejection remains enabled.
