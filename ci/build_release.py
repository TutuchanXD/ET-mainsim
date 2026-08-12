from __future__ import annotations

import argparse
import binascii
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import tarfile
import tempfile
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
import venv
import zipfile
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "ci" / "release_contract.toml"
ARCHIVE_CODEC_CONTRACT = {
    "sdist": {
        "format": "tar+gzip",
        "tar_format": "gnu",
        "gzip_header": "rfc1952-fixed-v1",
        "deflate": "stored-blocks",
    },
    "wheel": {
        "format": "zip",
        "compression": "stored",
        "method": zipfile.ZIP_STORED,
    },
}


class ReleaseContractError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReleaseContractError(message)


def _run(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or str(exc)).strip()
        raise ReleaseContractError(
            f"command failed ({' '.join(command)}): {detail}"
        ) from exc


def load_release_contract(path: Path = CONTRACT_PATH) -> dict[str, Any]:
    with path.open("rb") as stream:
        contract = tomllib.load(stream)
    for key in (
        "schema_version",
        "version",
        "tag",
        "repository",
        "required_checks",
        "release_policy",
        "runtime_dependencies",
        "files",
        "build_tools",
    ):
        _require(key in contract, f"release contract lacks {key!r}")
    _require(contract["schema_version"] == 1, "unsupported release contract schema")
    _require(
        contract["tag"] == f"v{contract['version']}",
        "release tag must be the version prefixed with v",
    )
    required_checks = contract["required_checks"]
    _require(
        isinstance(required_checks, list) and bool(required_checks),
        "release contract required_checks must be a non-empty list",
    )
    check_names: set[str] = set()
    check_targets: set[tuple[str, str]] = set()
    for check in required_checks:
        _require(
            isinstance(check, dict)
            and set(check) == {"name", "workflow_path", "job_name"},
            "release required-check mapping fields are not canonical",
        )
        _require(
            all(isinstance(check[key], str) and bool(check[key]) for key in check),
            "release required-check mapping values must be non-empty strings",
        )
        _require(
            bool(
                re.fullmatch(
                    r"\.github/workflows/[A-Za-z0-9_.-]+\.ya?ml",
                    check["workflow_path"],
                )
            ),
            "release required check has an invalid workflow path",
        )
        _require(
            check["name"] not in check_names,
            "release required check names must be unique",
        )
        target = (check["workflow_path"], check["job_name"])
        _require(
            target not in check_targets,
            "release required workflow-job targets must be unique",
        )
        check_names.add(check["name"])
        check_targets.add(target)
    contract["wheel_files"] = contract["files"]["wheel"]
    contract["sdist_files"] = contract["files"]["sdist"]
    return contract


def validate_release_source(root: Path, contract: dict[str, Any]) -> None:
    with (root / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)["project"]
    _require(project["name"] == "et-mainsim", "unexpected project name")
    _require(
        project["version"] == contract["version"],
        "pyproject version differs from the release contract",
    )

    init_text = (root / "src" / "et_mainsim" / "__init__.py").read_text(
        encoding="utf-8"
    )
    fallback = re.findall(r'^\s*__version__\s*=\s*"([^"]+)"', init_text, re.MULTILINE)
    _require(
        fallback == [contract["version"]],
        "package fallback version differs from the release contract",
    )
    with (root / "ci" / "full_pytest_contract.toml").open("rb") as stream:
        full_test = tomllib.load(stream)
    frozen = full_test["dependencies"]
    runtime_dependencies = contract["runtime_dependencies"]
    _require(
        runtime_dependencies["et_coordinate"]["commit"]
        == frozen["et_coordinate_commit"],
        "release and full-test ET-coordinate commits differ",
    )
    _require(
        runtime_dependencies["photsim7"]["commit"] == frozen["photsim7_commit"],
        "release and full-test Photsim7 commits differ",
    )


def _git_provenance(root: Path, tag: str, require_tag: bool) -> dict[str, Any]:
    inside = _run(["git", "rev-parse", "--is-inside-work-tree"], cwd=root)
    _require(inside.stdout.strip() == "true", "release source must be a Git worktree")
    dirty = _run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"], cwd=root
    ).stdout
    _require(not dirty.strip(), "release source must be completely clean")

    commit = _run(["git", "rev-parse", "HEAD"], cwd=root).stdout.strip()
    tree = _run(["git", "rev-parse", "HEAD^{tree}"], cwd=root).stdout.strip()
    epoch_text = _run(["git", "show", "-s", "--format=%ct", "HEAD"], cwd=root).stdout
    source_date_epoch = int(epoch_text.strip())
    tag_object = None
    main_commit = None
    if require_tag:
        tag_type = _run(["git", "cat-file", "-t", f"refs/tags/{tag}"], cwd=root)
        _require(tag_type.stdout.strip() == "tag", f"{tag} must be an annotated tag")
        tag_object = _run(
            ["git", "rev-parse", f"refs/tags/{tag}"], cwd=root
        ).stdout.strip()
        tag_payload = _run(["git", "cat-file", "-p", tag_object], cwd=root).stdout
        headers = {}
        for line in tag_payload.split("\n\n", 1)[0].splitlines():
            key, separator, value = line.partition(" ")
            if separator:
                headers[key] = value
        _require(headers.get("tag") == tag, f"{tag} tag object has the wrong name")
        _require(
            headers.get("type") == "commit",
            f"{tag} tag object must target a commit",
        )
        _require(
            headers.get("object") == commit,
            f"{tag} tag object does not directly target HEAD",
        )
        peeled_commit = _run(
            ["git", "rev-parse", f"refs/tags/{tag}^{{commit}}"], cwd=root
        ).stdout.strip()
        _require(peeled_commit == commit, f"{tag} does not peel to HEAD")

        main_commit = _run(
            ["git", "rev-parse", "--verify", "refs/remotes/origin/main^{commit}"],
            cwd=root,
        ).stdout.strip()
        reachability = subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, main_commit],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        _require(
            reachability.returncode == 0,
            "release commit is not reachable from origin/main",
        )
    return {
        "git_commit": commit,
        "git_tree": tree,
        "source_date_epoch": source_date_epoch,
        "dirty": False,
        "tag_verified": require_tag,
        "tag_object": tag_object,
        "main_commit": main_commit,
    }


def _github_api_json(url: str, token: str) -> dict[str, Any]:
    _require(bool(token), "GITHUB_TOKEN is required to verify release checks")
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
            payload = json.load(response)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
        raise ReleaseContractError(f"GitHub checks request failed: {exc}") from exc
    _require(isinstance(payload, dict), "GitHub checks response must be an object")
    return payload


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _github_paginated_items(
    url: str,
    *,
    token: str,
    item_key: str,
) -> list[dict[str, Any]]:
    _require(bool(token), "GITHUB_TOKEN is required to read GitHub Actions")
    items: list[dict[str, Any]] = []
    expected_total: int | None = None
    observed_ids: set[int] = set()
    page = 1
    while True:
        separator = "&" if "?" in url else "?"
        payload = _github_api_json(
            f"{url}{separator}per_page=100&page={page}",
            token,
        )
        total = payload.get("total_count")
        page_items = payload.get(item_key)
        _require(
            isinstance(total, int) and not isinstance(total, bool) and total >= 0,
            f"GitHub {item_key} pagination lacks a valid total_count",
        )
        _require(
            isinstance(page_items, list) and len(page_items) <= 100,
            f"GitHub {item_key} pagination returned an invalid page",
        )
        if expected_total is None:
            expected_total = total
        _require(
            total == expected_total,
            f"GitHub {item_key} pagination total_count changed",
        )
        _require(
            len(items) + len(page_items) <= expected_total,
            f"GitHub {item_key} pagination exceeded total_count",
        )
        for item in page_items:
            _require(
                isinstance(item, dict) and _positive_int(item.get("id")),
                f"GitHub {item_key} pagination returned an invalid item",
            )
            item_id = item["id"]
            _require(
                item_id not in observed_ids,
                f"GitHub {item_key} pagination returned a duplicate id",
            )
            observed_ids.add(item_id)
            items.append(item)
        if len(items) == expected_total:
            return items
        _require(
            expected_total < 1000,
            f"GitHub {item_key} pagination reached the filtered search cap",
        )
        _require(
            len(page_items) == 100,
            f"GitHub {item_key} pagination ended before total_count",
        )
        page += 1


def collect_github_approval_evidence(
    *,
    repository: str,
    commit: str,
    main_commit: str,
    required_checks: list[dict[str, str]],
    token: str,
) -> dict[str, Any]:
    _require(
        bool(re.fullmatch(r"[0-9a-f]{40}", commit)),
        "release approval commit must be a full lowercase Git SHA",
    )
    _require(
        bool(re.fullmatch(r"[0-9a-f]{40}", main_commit)),
        "release main commit must be a full lowercase Git SHA",
    )
    _require(
        bool(re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository)),
        "release repository is not canonical",
    )
    api_root = f"https://api.github.com/repos/{repository}"
    workflows = _github_paginated_items(
        f"{api_root}/actions/workflows",
        token=token,
        item_key="workflows",
    )
    workflow_paths = list(dict.fromkeys(check["workflow_path"] for check in required_checks))
    workflow_runs: dict[str, dict[str, Any]] = {}
    workflow_jobs: dict[str, list[dict[str, Any]]] = {}
    workflow_ids: dict[str, int] = {}
    for workflow_path in workflow_paths:
        matching_workflows = [
            workflow for workflow in workflows if workflow.get("path") == workflow_path
        ]
        _require(
            len(matching_workflows) == 1,
            f"required workflow path is missing or ambiguous: {workflow_path}",
        )
        workflow = matching_workflows[0]
        _require(
            _positive_int(workflow.get("id")) and workflow.get("state") == "active",
            f"required workflow is not active: {workflow_path}",
        )
        workflow_id = workflow["id"]
        workflow_ids[workflow_path] = workflow_id
        query = urllib.parse.urlencode(
            {
                "branch": "main",
                "event": "push",
                "head_sha": commit,
                "exclude_pull_requests": "true",
            }
        )
        runs = _github_paginated_items(
            f"{api_root}/actions/workflows/{workflow_id}/runs?{query}",
            token=token,
            item_key="workflow_runs",
        )
        for run in runs:
            _require(
                run.get("workflow_id") == workflow_id
                and run.get("path") == workflow_path
                and run.get("event") == "push"
                and run.get("head_branch") == "main"
                and run.get("head_sha") == commit
                and run.get("status") == "completed"
                and run.get("conclusion") == "success"
                and _positive_int(run.get("run_attempt")),
                f"workflow run provenance is invalid: {workflow_path}",
            )
        _require(
            len(runs) == 1,
            f"required workflow run is missing or ambiguous: {workflow_path}",
        )
        run = runs[0]
        workflow_runs[workflow_path] = run
        workflow_jobs[workflow_path] = _github_paginated_items(
            (
                f"{api_root}/actions/runs/{run['id']}/attempts/"
                f"{run['run_attempt']}/jobs"
            ),
            token=token,
            item_key="jobs",
        )
        confirmed_run = _github_api_json(
            f"{api_root}/actions/runs/{run['id']}",
            token,
        )
        _require(
            all(
                confirmed_run.get(key) == run.get(key)
                for key in (
                    "id",
                    "workflow_id",
                    "path",
                    "run_attempt",
                    "event",
                    "head_branch",
                    "head_sha",
                    "status",
                    "conclusion",
                )
            ),
            f"workflow run changed while collecting evidence: {workflow_path}",
        )

    normalized_jobs = []
    for requirement in required_checks:
        workflow_path = requirement["workflow_path"]
        run = workflow_runs[workflow_path]
        matches = [
            job
            for job in workflow_jobs[workflow_path]
            if job.get("name") == requirement["job_name"]
        ]
        _require(
            len(matches) == 1,
            f"required workflow job is missing or ambiguous: {requirement['job_name']}",
        )
        job = matches[0]
        expected_job_url = (
            f"https://github.com/{repository}/actions/runs/{run['id']}/job/{job['id']}"
        )
        _require(
            job.get("run_id") == run["id"]
            and job.get("head_sha") == commit
            and job.get("status") == "completed"
            and job.get("conclusion") == "success"
            and job.get("html_url") == expected_job_url,
            f"workflow job provenance is invalid: {requirement['job_name']}",
        )
        normalized_jobs.append(
            {
                "name": requirement["name"],
                "workflow_id": workflow_ids[workflow_path],
                "workflow_path": workflow_path,
                "run_id": run["id"],
                "run_attempt": run["run_attempt"],
                "job_name": requirement["job_name"],
                "job_id": job["id"],
                "job_url": job["html_url"],
                "event": "push",
                "head_branch": "main",
                "head_sha": commit,
                "status": "completed",
                "conclusion": "success",
            }
        )
    return {
        "schema_id": "et_mainsim.release_approval.v2",
        "verified": True,
        "repository": repository,
        "commit": commit,
        "main_commit": main_commit,
        "workflow_jobs": normalized_jobs,
    }


def _validate_approval_evidence(
    evidence: dict[str, Any],
    *,
    contract: dict[str, Any],
    source: dict[str, Any],
) -> dict[str, Any]:
    expected_keys = {
        "schema_id",
        "verified",
        "repository",
        "commit",
        "main_commit",
        "workflow_jobs",
    }
    _require(set(evidence) == expected_keys, "approval evidence fields are not canonical")
    _require(
        evidence["schema_id"] == "et_mainsim.release_approval.v2",
        "unsupported release approval evidence schema",
    )
    _require(evidence["verified"] is True, "release approval is not verified")
    _require(
        evidence["repository"] == contract["repository"],
        "release approval repository differs from the contract",
    )
    _require(
        evidence["commit"] == source["git_commit"],
        "release approval commit differs from HEAD",
    )
    _require(
        source["main_commit"] is not None
        and evidence["main_commit"] == source["main_commit"],
        "release approval main baseline differs from origin/main",
    )
    workflow_jobs = evidence["workflow_jobs"]
    _require(
        isinstance(workflow_jobs, list)
        and len(workflow_jobs) == len(contract["required_checks"]),
        "release approval does not contain the exact workflow jobs",
    )
    job_keys = {
        "name",
        "workflow_id",
        "workflow_path",
        "run_id",
        "run_attempt",
        "job_name",
        "job_id",
        "job_url",
        "event",
        "head_branch",
        "head_sha",
        "status",
        "conclusion",
    }
    workflow_identity: dict[str, tuple[int, int, int]] = {}
    for job, required in zip(
        workflow_jobs,
        contract["required_checks"],
        strict=True,
    ):
        _require(
            isinstance(job, dict) and set(job) == job_keys,
            "release approval workflow-job fields are not canonical",
        )
        _require(
            job["name"] == required["name"]
            and job["workflow_path"] == required["workflow_path"]
            and job["job_name"] == required["job_name"],
            "release approval workflow-job mapping differs from the contract",
        )
        _require(
            all(
                _positive_int(job[key])
                for key in ("workflow_id", "run_id", "run_attempt", "job_id")
            ),
            f"workflow job lacks a valid numeric identity: {job['name']}",
        )
        identity = (job["workflow_id"], job["run_id"], job["run_attempt"])
        previous_identity = workflow_identity.setdefault(
            job["workflow_path"],
            identity,
        )
        _require(
            identity == previous_identity,
            f"workflow jobs do not share one run attempt: {job['workflow_path']}",
        )
        _require(
            job["event"] == "push"
            and job["head_branch"] == "main"
            and job["head_sha"] == source["git_commit"]
            and job["status"] == "completed"
            and job["conclusion"] == "success",
            f"workflow job is not an exact successful main push: {job['name']}",
        )
        _require(
            job["job_url"]
            == (
                f"https://github.com/{contract['repository']}/actions/runs/"
                f"{job['run_id']}/job/{job['job_id']}"
            ),
            f"workflow job URL is not canonical: {job['name']}",
        )
    return json.loads(json.dumps(evidence))


def _validate_build_tools(contract: dict[str, Any]) -> dict[str, str]:
    observed = {
        name: importlib.metadata.version(name)
        for name in ("build", "setuptools", "wheel")
    }
    _require(
        observed == contract["build_tools"],
        f"build tool versions differ from the release contract: {observed}",
    )
    return observed


def _export_head(root: Path, destination: Path) -> None:
    try:
        archive = subprocess.run(
            ["git", "archive", "--format=tar", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
    except subprocess.CalledProcessError as exc:
        raise ReleaseContractError("failed to export the release commit") from exc
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as stream:
        stream.extractall(destination, filter="data")


def _canonicalize_sdist(path: Path, source_date_epoch: int) -> None:
    members: list[tuple[tarfile.TarInfo, bytes | None]] = []
    with tarfile.open(path, "r:gz") as source:
        for member in source.getmembers():
            _require(
                member.isfile() or member.isdir(),
                f"unsupported sdist member type: {member.name}",
            )
            payload = None
            if member.isfile():
                extracted = source.extractfile(member)
                _require(
                    extracted is not None, f"cannot read sdist member: {member.name}"
                )
                payload = extracted.read()
            members.append((member, payload))

    tar_buffer = io.BytesIO()
    with tarfile.open(
        fileobj=tar_buffer, mode="w", format=tarfile.GNU_FORMAT
    ) as target:
        for member, payload in sorted(members, key=lambda item: item[0].name):
            canonical = tarfile.TarInfo(member.name)
            canonical.type = member.type
            canonical.linkname = member.linkname
            canonical.mode = 0o755 if member.isdir() or member.mode & 0o111 else 0o644
            canonical.uid = 0
            canonical.gid = 0
            canonical.uname = ""
            canonical.gname = ""
            canonical.mtime = source_date_epoch
            canonical.size = len(payload) if payload is not None else 0
            target.addfile(
                canonical,
                None if payload is None else io.BytesIO(payload),
            )

    payload = tar_buffer.getvalue()
    path.write_bytes(_deterministic_stored_gzip(payload, source_date_epoch))


def _deterministic_stored_gzip(payload: bytes, source_date_epoch: int) -> bytes:
    _require(
        0 <= source_date_epoch <= 0xFFFFFFFF,
        "SOURCE_DATE_EPOCH is outside the gzip timestamp range",
    )
    deflate = bytearray()
    chunks = [payload[offset : offset + 0xFFFF] for offset in range(0, len(payload), 0xFFFF)]
    if not chunks:
        chunks = [b""]
    for index, chunk in enumerate(chunks):
        final = index == len(chunks) - 1
        size = len(chunk)
        deflate.append(1 if final else 0)
        deflate.extend(struct.pack("<HH", size, size ^ 0xFFFF))
        deflate.extend(chunk)
    header = b"\x1f\x8b\x08\x00" + struct.pack("<I", source_date_epoch) + b"\x00\xff"
    trailer = struct.pack(
        "<II",
        binascii.crc32(payload) & 0xFFFFFFFF,
        len(payload) & 0xFFFFFFFF,
    )
    return header + bytes(deflate) + trailer


def _canonicalize_wheel(path: Path, source_date_epoch: int) -> None:
    members: list[tuple[zipfile.ZipInfo, bytes]] = []
    with zipfile.ZipFile(path) as source:
        for info in source.infolist():
            members.append((info, source.read(info.filename)))

    timestamp = time.gmtime(source_date_epoch)[:6]
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as target:
            for original, payload in sorted(members, key=lambda item: item[0].filename):
                canonical = zipfile.ZipInfo(original.filename, timestamp)
                canonical.compress_type = zipfile.ZIP_STORED
                canonical.create_system = 3
                canonical.create_version = 20
                canonical.extract_version = 20
                canonical.extra = b""
                canonical.comment = b""
                executable = bool((original.external_attr >> 16) & 0o111)
                if original.is_dir():
                    mode = 0o40755
                    canonical.external_attr = (mode << 16) | 0x10
                else:
                    mode = 0o100755 if executable else 0o100644
                    canonical.external_attr = mode << 16
                target.writestr(canonical, payload)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _build_once(
    root: Path,
    destination: Path,
    source_date_epoch: int,
) -> tuple[Path, Path]:
    source = destination / "source"
    distributions = destination / "dist"
    source.mkdir(parents=True)
    distributions.mkdir()
    _export_head(root, source)
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.update(
        {
            "LC_ALL": "C.UTF-8",
            "PYTHONHASHSEED": "0",
            "SOURCE_DATE_EPOCH": str(source_date_epoch),
            "TZ": "UTC",
        }
    )
    _run(
        [
            sys.executable,
            "-m",
            "build",
            "--no-isolation",
            "--sdist",
            "--outdir",
            str(distributions),
            ".",
        ],
        cwd=source,
        env=environment,
    )
    sdists = list(distributions.glob("*.tar.gz"))
    _require(len(sdists) == 1, "build produced an unexpected sdist set")
    _canonicalize_sdist(sdists[0], source_date_epoch)

    wheel_source = destination / "wheel-source"
    wheel_source.mkdir()
    with tarfile.open(sdists[0], "r:gz") as archive:
        archive.extractall(wheel_source, filter="data")
    wheel_roots = [path for path in wheel_source.iterdir() if path.is_dir()]
    _require(len(wheel_roots) == 1, "sdist must contain one versioned source root")
    _run(
        [
            sys.executable,
            "-m",
            "build",
            "--no-isolation",
            "--wheel",
            "--outdir",
            str(distributions),
            ".",
        ],
        cwd=wheel_roots[0],
        env=environment,
    )
    wheels = list(distributions.glob("*.whl"))
    _require(len(wheels) == 1, "build produced an unexpected wheel set")
    _canonicalize_wheel(wheels[0], source_date_epoch)
    return wheels[0], sdists[0]


def artifact_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _wheel_metadata(archive: zipfile.ZipFile) -> tuple[str, str]:
    metadata_names = [
        name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
    ]
    _require(len(metadata_names) == 1, "wheel must contain exactly one METADATA file")
    metadata = BytesParser().parsebytes(archive.read(metadata_names[0]))
    return metadata["Name"], metadata["Version"]


def _safe_archive_path(name: str, label: str) -> PurePosixPath:
    stripped = name.removesuffix("/")
    raw_parts = stripped.split("/")
    path = PurePosixPath(stripped)
    _require(
        bool(stripped)
        and "\\" not in name
        and not path.is_absolute()
        and all(part not in {"", ".", ".."} for part in raw_parts)
        and str(path) == stripped,
        f"{label} member does not use a safe canonical path: {name!r}",
    )
    return path


def _allowed_directories(files: list[str], root: str | None = None) -> set[str]:
    directories = set()
    if root is not None:
        directories.add(root)
    for name in files:
        path = PurePosixPath(name)
        parent = path.parent
        while str(parent) != ".":
            directories.add(str(parent) if root is None else f"{root}/{parent}")
            parent = parent.parent
    return directories


def _wheel_cli_version(wheel: Path) -> str:
    environment = os.environ.copy()
    for name in ("PYTHONHOME", "PYTHONPATH", "PYTHONUSERBASE"):
        environment.pop(name, None)
    environment.update(
        {
            "PIP_CONFIG_FILE": os.devnull,
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PYTHONNOUSERSITE": "1",
        }
    )
    with tempfile.TemporaryDirectory(prefix="et-mainsim-release-cli-") as temporary:
        environment_root = Path(temporary) / "venv"
        venv.EnvBuilder(with_pip=True, clear=True).create(environment_root)
        python = environment_root / "bin" / "python"
        console = environment_root / "bin" / "et-mainsim"
        _run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--no-index",
                "--no-deps",
                str(wheel),
            ],
            cwd=Path(temporary),
            env=environment,
        )
        _require(console.is_file(), "wheel did not install the console entry point")
        result = _run(
            [str(console), "--version"],
            cwd=Path(temporary),
            env=environment,
        )
    return result.stdout.strip()


def validate_artifacts(
    wheel: Path,
    sdist: Path,
    contract: dict[str, Any],
) -> dict[str, Any]:
    with zipfile.ZipFile(wheel) as archive:
        wheel_infos = archive.infolist()
        wheel_paths = [
            _safe_archive_path(info.filename, "wheel") for info in wheel_infos
        ]
        normalized_wheel_names = [str(path) for path in wheel_paths]
        _require(
            len(normalized_wheel_names) == len(set(normalized_wheel_names)),
            "wheel contains duplicate member paths",
        )
        wheel_names = [
            str(path)
            for path, info in zip(wheel_paths, wheel_infos, strict=True)
            if not info.is_dir()
        ]
        actual_wheel = sorted(wheel_names)
        name, version = _wheel_metadata(archive)
    expected_wheel = sorted(contract["wheel_files"])
    wheel_directories = {
        str(path)
        for path, info in zip(wheel_paths, wheel_infos, strict=True)
        if info.is_dir()
    }
    _require(
        wheel_directories <= _allowed_directories(expected_wheel),
        "wheel contains an unexpected directory member",
    )
    unexpected_wheel = sorted(set(actual_wheel) - set(expected_wheel))
    missing_wheel = sorted(set(expected_wheel) - set(actual_wheel))
    _require(not unexpected_wheel, f"unexpected wheel member: {unexpected_wheel}")
    _require(not missing_wheel, f"missing wheel member: {missing_wheel}")
    _require(name == "et-mainsim", "wheel metadata has the wrong project name")
    _require(version == contract["version"], "wheel metadata has the wrong version")

    prefix = f"et_mainsim-{contract['version']}/"
    with tarfile.open(sdist, "r:gz") as archive:
        members = archive.getmembers()
        member_paths = [_safe_archive_path(member.name, "sdist") for member in members]
        member_names = [str(path) for path in member_paths]
        _require(
            len(member_names) == len(set(member_names)),
            "sdist contains duplicate member paths",
        )
        _require(
            all(member.isfile() or member.isdir() for member in members),
            "sdist may contain only regular files and directories",
        )
        release_root = f"et_mainsim-{contract['version']}"
        _require(
            all(path.parts[0] == release_root for path in member_paths),
            "sdist member does not use the exact versioned root",
        )
        actual_sdist = sorted(
            str(path).removeprefix(prefix)
            for path, member in zip(member_paths, members, strict=True)
            if member.isfile()
        )
    expected_sdist = sorted(contract["sdist_files"])
    sdist_directories = {
        str(path)
        for path, member in zip(member_paths, members, strict=True)
        if member.isdir()
    }
    _require(
        sdist_directories <= _allowed_directories(expected_sdist, release_root),
        "sdist contains an unexpected directory member",
    )
    unexpected_sdist = sorted(set(actual_sdist) - set(expected_sdist))
    missing_sdist = sorted(set(expected_sdist) - set(actual_sdist))
    _require(not unexpected_sdist, f"unexpected sdist member: {unexpected_sdist}")
    _require(not missing_sdist, f"missing sdist member: {missing_sdist}")

    try:
        cli_version = _wheel_cli_version(wheel)
    except ReleaseContractError as exc:
        raise ReleaseContractError(
            f"console entry point validation failed: {exc}"
        ) from exc
    _require(
        cli_version == f"et-mainsim {contract['version']}",
        "wheel CLI version differs from the release contract",
    )
    return {
        "metadata_name": name,
        "metadata_version": version,
        "cli_version": cli_version,
        "wheel_member_count": len(actual_wheel),
        "sdist_file_count": len(actual_sdist),
        "unexpected_wheel_members": unexpected_wheel,
        "unexpected_sdist_members": unexpected_sdist,
    }


def build_release(
    root: Path = ROOT,
    output: Path | None = None,
    *,
    require_tag: bool = False,
    approval_evidence: dict[str, Any] | Path | None = None,
    github_token: str | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    contract = load_release_contract(root / "ci" / "release_contract.toml")
    validate_release_source(root, contract)
    source = _git_provenance(root, contract["tag"], require_tag)
    if approval_evidence is None:
        _require(
            not require_tag,
            "verified approval evidence is required for a tagged release",
        )
        approval = {
            "schema_id": "et_mainsim.release_approval.v2",
            "verified": False,
            "repository": contract["repository"],
            "commit": source["git_commit"],
            "main_commit": None,
            "workflow_jobs": [],
        }
    else:
        _require(require_tag, "approval evidence is valid only for a tagged release")
        if isinstance(approval_evidence, Path):
            evidence_payload = json.loads(approval_evidence.read_text(encoding="utf-8"))
        else:
            evidence_payload = approval_evidence
        _require(
            isinstance(evidence_payload, dict),
            "release approval evidence must be a JSON object",
        )
        approval = _validate_approval_evidence(
            evidence_payload,
            contract=contract,
            source=source,
        )
        _require(
            bool(github_token),
            "GITHUB_TOKEN is required to authenticate approval evidence",
        )
        observed_approval = collect_github_approval_evidence(
            repository=contract["repository"],
            commit=source["git_commit"],
            main_commit=source["main_commit"],
            required_checks=contract["required_checks"],
            token=github_token,
        )
        _require(
            approval == observed_approval,
            "approval evidence differs from live GitHub check runs",
        )
    build_tools = _validate_build_tools(contract)
    output = (root / "release-output" if output is None else output).resolve()
    _require(output != root, "release output cannot replace the source root")
    if output.exists():
        _require(not any(output.iterdir()), "release output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    artifacts_root = output / "artifacts"
    artifacts_root.mkdir()

    with tempfile.TemporaryDirectory(prefix="et-mainsim-release-build-") as temporary:
        temporary_root = Path(temporary)
        builds = [
            _build_once(
                root, temporary_root / f"build-{index}", source["source_date_epoch"]
            )
            for index in (1, 2)
        ]
        first_wheel, first_sdist = builds[0]
        second_wheel, second_sdist = builds[1]
        _require(
            first_wheel.name == second_wheel.name, "wheel names differ between builds"
        )
        _require(
            first_sdist.name == second_sdist.name, "sdist names differ between builds"
        )
        first_hashes = {
            first_wheel.name: artifact_sha256(first_wheel),
            first_sdist.name: artifact_sha256(first_sdist),
        }
        second_hashes = {
            second_wheel.name: artifact_sha256(second_wheel),
            second_sdist.name: artifact_sha256(second_sdist),
        }
        _require(
            first_hashes == second_hashes, "release builds are not byte reproducible"
        )
        validation = validate_artifacts(first_wheel, first_sdist, contract)
        for source_path in (first_wheel, first_sdist):
            shutil.copyfile(source_path, artifacts_root / source_path.name)

    primary_artifacts = {
        path.name: {"bytes": path.stat().st_size, "sha256": artifact_sha256(path)}
        for path in sorted(artifacts_root.iterdir())
    }
    primary_names = list(primary_artifacts)
    provenance_name = "release-provenance.json"
    checksum_name = "SHA256SUMS"
    bundle_contract = {
        "files": [*primary_names, provenance_name, checksum_name],
        "primary_artifacts": primary_names,
        "provenance_receipt": provenance_name,
        "checksum_manifest": {
            "name": checksum_name,
            "algorithm": "sha256",
            "covers": [*primary_names, provenance_name],
            "excludes": [checksum_name],
        },
    }
    receipt = {
        "schema_id": "et_mainsim.release_receipt.v1",
        "schema_version": 1,
        "release": {"version": contract["version"], "tag": contract["tag"]},
        "source": source,
        "build": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "tools": build_tools,
        },
        "reproducibility": {
            "build_count": 2,
            "byte_identical": True,
            "sha256": first_hashes,
        },
        "archive_codec": ARCHIVE_CODEC_CONTRACT,
        "validation": validation,
        "primary_artifacts": primary_artifacts,
        "bundle_contract": bundle_contract,
        "approval": approval,
        "release_policy": contract["release_policy"],
        "runtime_dependencies": contract["runtime_dependencies"],
    }
    provenance_path = artifacts_root / provenance_name
    provenance_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    checksum_identities = {
        **primary_artifacts,
        provenance_path.name: {
            "bytes": provenance_path.stat().st_size,
            "sha256": artifact_sha256(provenance_path),
        },
    }
    (artifacts_root / checksum_name).write_text(
        "".join(
            f"{identity['sha256']}  {name}\n"
            for name, identity in checksum_identities.items()
        ),
        encoding="utf-8",
    )
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a reproducible ET-mainsim release."
    )
    parser.add_argument("--output", type=Path, default=Path("release-output"))
    parser.add_argument("--require-tag", action="store_true")
    parser.add_argument("--approval-evidence", type=Path)
    parser.add_argument("--collect-approval-evidence", type=Path)
    parser.add_argument("--github-repository")
    parser.add_argument("--github-commit")
    parser.add_argument("--main-commit")
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    try:
        if arguments.collect_approval_evidence is not None:
            contract = load_release_contract(CONTRACT_PATH)
            _require(
                arguments.github_repository == contract["repository"],
                "GitHub repository differs from the release contract",
            )
            _require(arguments.github_commit is not None, "GitHub commit is required")
            _require(arguments.main_commit is not None, "main commit is required")
            evidence = collect_github_approval_evidence(
                repository=arguments.github_repository,
                commit=arguments.github_commit,
                main_commit=arguments.main_commit,
                required_checks=contract["required_checks"],
                token=os.environ.get("GITHUB_TOKEN", ""),
            )
            arguments.collect_approval_evidence.write_text(
                json.dumps(evidence, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(json.dumps(evidence, indent=2, sort_keys=True))
            return 0
        receipt = build_release(
            ROOT,
            arguments.output,
            require_tag=arguments.require_tag,
            approval_evidence=arguments.approval_evidence,
            github_token=os.environ.get("GITHUB_TOKEN"),
        )
    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        tomllib.TOMLDecodeError,
        ReleaseContractError,
    ) as exc:
        print(f"release contract rejected: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
