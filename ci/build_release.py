from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import venv
import zipfile
from email.parser import BytesParser
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "ci" / "release_contract.toml"


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
    if require_tag:
        tag_type = _run(["git", "cat-file", "-t", f"refs/tags/{tag}"], cwd=root)
        _require(tag_type.stdout.strip() == "tag", f"{tag} must be an annotated tag")
        tagged_commit = _run(
            ["git", "rev-list", "-n", "1", f"refs/tags/{tag}"], cwd=root
        ).stdout.strip()
        _require(tagged_commit == commit, f"{tag} does not identify HEAD")
    return {
        "git_commit": commit,
        "git_tree": tree,
        "source_date_epoch": source_date_epoch,
        "dirty": False,
        "tag_verified": require_tag,
    }


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

    with path.open("wb") as raw:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            compresslevel=9,
            fileobj=raw,
            mtime=source_date_epoch,
        ) as compressed:
            compressed.write(tar_buffer.getvalue())


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
) -> dict[str, Any]:
    root = root.resolve()
    contract = load_release_contract(root / "ci" / "release_contract.toml")
    validate_release_source(root, contract)
    source = _git_provenance(root, contract["tag"], require_tag)
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

    artifacts = {
        path.name: {"bytes": path.stat().st_size, "sha256": artifact_sha256(path)}
        for path in sorted(artifacts_root.iterdir())
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
        "validation": validation,
        "artifacts": artifacts,
        "required_checks": contract["required_checks"],
        "release_policy": contract["release_policy"],
        "runtime_dependencies": contract["runtime_dependencies"],
    }
    provenance_path = artifacts_root / "release-provenance.json"
    provenance_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    checksum_identities = {
        **artifacts,
        provenance_path.name: {
            "bytes": provenance_path.stat().st_size,
            "sha256": artifact_sha256(provenance_path),
        },
    }
    (artifacts_root / "SHA256SUMS").write_text(
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
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    try:
        receipt = build_release(
            ROOT, arguments.output, require_tag=arguments.require_tag
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
