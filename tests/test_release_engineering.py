from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import warnings
import zipfile
from pathlib import Path

import pytest

from ci.build_release import (
    ReleaseContractError,
    _git_provenance,
    artifact_sha256,
    build_release,
    load_release_contract,
    validate_artifacts,
    validate_release_source,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "ci" / "release_contract.toml"
WORKFLOW_PATH = ROOT / ".github" / "workflows" / "release.yml"


def _clean_release_source(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    shutil.copytree(
        ROOT,
        source,
        ignore=shutil.ignore_patterns(".git", "build", "dist", "*.egg-info"),
    )
    subprocess.run(["git", "init", "-q"], cwd=source, check=True)
    subprocess.run(["git", "add", "."], cwd=source, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Release Test",
            "-c",
            "user.email=release-test@example.invalid",
            "commit",
            "-qm",
            "test release source",
        ],
        cwd=source,
        check=True,
    )
    return source


def test_release_contract_matches_package_and_cli_version() -> None:
    contract = load_release_contract(CONTRACT_PATH)

    assert contract["version"] == "0.1.0"
    assert contract["tag"] == "v0.1.0"
    validate_release_source(ROOT, contract)

    result = subprocess.run(
        [sys.executable, "-m", "et_mainsim", "--version"],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "et-mainsim 0.1.0"


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        ("pyproject.toml", 'version = "9.9.9"'),
        ("src/et_mainsim/__init__.py", '__version__ = "9.9.9"'),
    ],
)
def test_release_source_rejects_version_drift(
    tmp_path: Path,
    path: str,
    replacement: str,
) -> None:
    contract = load_release_contract(CONTRACT_PATH)
    source = tmp_path / "source"
    source.mkdir()
    (source / "pyproject.toml").write_text(
        (ROOT / "pyproject.toml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    package = source / "src" / "et_mainsim"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        (ROOT / "src" / "et_mainsim" / "__init__.py").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    target = source / path
    text = target.read_text(encoding="utf-8")
    if path == "pyproject.toml":
        text = text.replace('version = "0.1.0"', replacement)
    else:
        text = text.replace('__version__ = "0.1.0"', replacement)
    target.write_text(text, encoding="utf-8")

    with pytest.raises(ReleaseContractError):
        validate_release_source(source, contract)


@pytest.mark.parametrize("tag_kind", ["missing", "lightweight", "wrong-commit"])
def test_release_tag_gate_rejects_invalid_tag(
    tmp_path: Path,
    tag_kind: str,
) -> None:
    source = _clean_release_source(tmp_path)
    if tag_kind == "lightweight":
        subprocess.run(["git", "tag", "v0.1.0"], cwd=source, check=True)
    elif tag_kind == "wrong-commit":
        previous = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=source,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        (source / "tag-test.txt").write_text("second commit\n", encoding="utf-8")
        subprocess.run(["git", "add", "tag-test.txt"], cwd=source, check=True)
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=Release Test",
                "-c",
                "user.email=release-test@example.invalid",
                "commit",
                "-qm",
                "second commit",
            ],
            cwd=source,
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=Release Test",
                "-c",
                "user.email=release-test@example.invalid",
                "tag",
                "-a",
                "v0.1.0",
                "-m",
                "wrong release commit",
                previous,
            ],
            cwd=source,
            check=True,
        )

    with pytest.raises(ReleaseContractError):
        _git_provenance(source, "v0.1.0", require_tag=True)


def test_release_tag_gate_accepts_annotated_tag_at_head(tmp_path: Path) -> None:
    source = _clean_release_source(tmp_path)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Release Test",
            "-c",
            "user.email=release-test@example.invalid",
            "tag",
            "-a",
            "v0.1.0",
            "-m",
            "release v0.1.0",
        ],
        cwd=source,
        check=True,
    )

    provenance = _git_provenance(source, "v0.1.0", require_tag=True)

    assert provenance["tag_verified"] is True


def test_release_builder_produces_reproducible_validated_artifacts(
    tmp_path: Path,
) -> None:
    source = _clean_release_source(tmp_path)
    receipt = build_release(source, tmp_path / "release")

    assert receipt["schema_id"] == "et_mainsim.release_receipt.v1"
    assert receipt["release"] == {"version": "0.1.0", "tag": "v0.1.0"}
    assert (
        receipt["source"]["git_commit"]
        == subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=source,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    assert (
        receipt["source"]["git_tree"]
        == subprocess.run(
            ["git", "rev-parse", "HEAD^{tree}"],
            cwd=source,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    assert receipt["source"]["dirty"] is False
    assert receipt["reproducibility"]["build_count"] == 2
    assert receipt["reproducibility"]["byte_identical"] is True
    assert receipt["validation"]["metadata_version"] == "0.1.0"
    assert receipt["validation"]["cli_version"] == "et-mainsim 0.1.0"
    assert receipt["validation"]["unexpected_wheel_members"] == []
    assert receipt["validation"]["unexpected_sdist_members"] == []

    artifact_root = tmp_path / "release" / "artifacts"
    primary_names = [
        "et_mainsim-0.1.0-py3-none-any.whl",
        "et_mainsim-0.1.0.tar.gz",
    ]
    bundle_files = [
        *primary_names,
        "release-provenance.json",
        "SHA256SUMS",
    ]
    assert sorted(path.name for path in artifact_root.iterdir()) == sorted(bundle_files)
    assert "artifacts" not in receipt
    assert sorted(receipt["primary_artifacts"]) == primary_names
    assert receipt["bundle_contract"] == {
        "files": bundle_files,
        "primary_artifacts": primary_names,
        "provenance_receipt": "release-provenance.json",
        "checksum_manifest": {
            "name": "SHA256SUMS",
            "algorithm": "sha256",
            "covers": [*primary_names, "release-provenance.json"],
            "excludes": ["SHA256SUMS"],
        },
    }
    for name, identity in receipt["primary_artifacts"].items():
        artifact = artifact_root / name
        assert identity == {
            "bytes": artifact.stat().st_size,
            "sha256": artifact_sha256(artifact),
        }
    sums = (artifact_root / "SHA256SUMS").read_text(encoding="utf-8")
    provenance = artifact_root / "release-provenance.json"
    assert set(sums.splitlines()) == {
        *(
            f"{identity['sha256']}  {name}"
            for name, identity in receipt["primary_artifacts"].items()
        ),
        f"{artifact_sha256(provenance)}  release-provenance.json",
    }
    assert (
        json.loads(
            (artifact_root / "release-provenance.json").read_text(encoding="utf-8")
        )
        == receipt
    )


def test_artifact_validator_rejects_unexpected_package_members(tmp_path: Path) -> None:
    contract = load_release_contract(CONTRACT_PATH)
    source = _clean_release_source(tmp_path)
    receipt = build_release(source, tmp_path / "release")
    artifact_root = tmp_path / "release" / "artifacts"
    wheel = artifact_root / "et_mainsim-0.1.0-py3-none-any.whl"
    sdist = artifact_root / "et_mainsim-0.1.0.tar.gz"
    assert receipt["validation"]["unexpected_wheel_members"] == []

    tampered = tmp_path / f"tampered-{wheel.name}"
    tampered.write_bytes(wheel.read_bytes())
    with zipfile.ZipFile(tampered, "a") as archive:
        archive.writestr("science-output.bin", b"not a package member")

    with pytest.raises(ReleaseContractError, match="wheel member"):
        validate_artifacts(tampered, sdist, contract)


def test_artifact_validator_rejects_broken_console_entry_point(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_release_contract(CONTRACT_PATH)
    source = _clean_release_source(tmp_path)
    build_release(source, tmp_path / "release")
    artifact_root = tmp_path / "release" / "artifacts"
    wheel = artifact_root / "et_mainsim-0.1.0-py3-none-any.whl"
    sdist = artifact_root / "et_mainsim-0.1.0.tar.gz"
    tampered = tmp_path / f"tampered-entry-point-{wheel.name}"

    with zipfile.ZipFile(wheel) as original, zipfile.ZipFile(tampered, "w") as target:
        for info in original.infolist():
            payload = original.read(info.filename)
            if info.filename.endswith(".dist-info/entry_points.txt"):
                payload = payload.replace(
                    b"et_mainsim.cli:main",
                    b"et_mainsim.missing:main",
                )
            target.writestr(info, payload)

    fake_site = tmp_path / "fake-site"
    fake_package = fake_site / "et_mainsim"
    fake_package.mkdir(parents=True)
    (fake_package / "missing.py").write_text("def main(): return 0\n", encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", str(fake_site))
    monkeypatch.setenv("PYTHONHOME", str(tmp_path / "fake-python-home"))
    monkeypatch.setenv("PYTHONUSERBASE", str(tmp_path / "fake-user-base"))

    with pytest.raises(ReleaseContractError, match="console entry point"):
        validate_artifacts(tampered, sdist, contract)


def test_wheel_validation_ignores_host_pythonpath(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_release_contract(CONTRACT_PATH)
    source = _clean_release_source(tmp_path)
    build_release(source, tmp_path / "release")
    artifact_root = tmp_path / "release" / "artifacts"
    wheel = artifact_root / "et_mainsim-0.1.0-py3-none-any.whl"
    sdist = artifact_root / "et_mainsim-0.1.0.tar.gz"
    tampered = tmp_path / f"tampered-pythonpath-{wheel.name}"

    with zipfile.ZipFile(wheel) as original, zipfile.ZipFile(tampered, "w") as target:
        for info in original.infolist():
            payload = original.read(info.filename)
            if info.filename == "et_mainsim/cli.py":
                payload = b"raise ImportError('tampered wheel cli')\n"
            target.writestr(info, payload)

    fake_site = tmp_path / "fake-site"
    fake_package = fake_site / "et_mainsim"
    fake_package.mkdir(parents=True)
    (fake_package / "__init__.py").write_text(
        '__version__ = "0.1.0"\n', encoding="utf-8"
    )
    (fake_package / "cli.py").write_text("def main(): return 0\n", encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", str(fake_site))

    with pytest.raises(ReleaseContractError, match="console entry point"):
        validate_artifacts(tampered, sdist, contract)


def test_sdist_contains_only_frozen_release_sources(tmp_path: Path) -> None:
    contract = load_release_contract(CONTRACT_PATH)
    source = _clean_release_source(tmp_path)
    build_release(source, tmp_path / "release")
    sdist = tmp_path / "release" / "artifacts" / "et_mainsim-0.1.0.tar.gz"
    with tarfile.open(sdist, "r:gz") as archive:
        members = sorted(
            member.name for member in archive.getmembers() if member.isfile()
        )

    assert members
    assert all(name.startswith("et_mainsim-0.1.0/") for name in members)
    assert not any(".git" in Path(name).parts for name in members)
    assert not any(name.endswith((".npy", ".h5", ".fits", ".pkl")) for name in members)
    assert contract["sdist_files"]


def test_artifact_validator_rejects_sdist_links(tmp_path: Path) -> None:
    contract = load_release_contract(CONTRACT_PATH)
    source = _clean_release_source(tmp_path)
    build_release(source, tmp_path / "release")
    artifact_root = tmp_path / "release" / "artifacts"
    wheel = artifact_root / "et_mainsim-0.1.0-py3-none-any.whl"
    sdist = artifact_root / "et_mainsim-0.1.0.tar.gz"
    tampered = tmp_path / "tampered-link.tar.gz"

    with (
        tarfile.open(sdist, "r:gz") as original,
        tarfile.open(tampered, "w:gz") as target,
    ):
        for member in original.getmembers():
            extracted = original.extractfile(member) if member.isfile() else None
            target.addfile(member, extracted)
        link = tarfile.TarInfo("et_mainsim-0.1.0/unexpected-link")
        link.type = tarfile.SYMTYPE
        link.linkname = "/etc/passwd"
        target.addfile(link)

    with pytest.raises(ReleaseContractError, match="regular files and directories"):
        validate_artifacts(wheel, tampered, contract)


@pytest.mark.parametrize("archive_kind", ["wheel", "sdist"])
def test_artifact_validator_rejects_duplicate_members(
    tmp_path: Path,
    archive_kind: str,
) -> None:
    contract = load_release_contract(CONTRACT_PATH)
    source = _clean_release_source(tmp_path)
    build_release(source, tmp_path / "release")
    artifact_root = tmp_path / "release" / "artifacts"
    wheel = artifact_root / "et_mainsim-0.1.0-py3-none-any.whl"
    sdist = artifact_root / "et_mainsim-0.1.0.tar.gz"

    if archive_kind == "wheel":
        tampered = tmp_path / "duplicate-member.whl"
        shutil.copyfile(wheel, tampered)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(tampered, "a") as archive:
                archive.writestr("et_mainsim/reference_photometry.py", b"duplicate")
        wheel = tampered
    else:
        tampered = tmp_path / "duplicate-member.tar.gz"
        with (
            tarfile.open(sdist, "r:gz") as original,
            tarfile.open(tampered, "w:gz") as target,
        ):
            duplicate = None
            for member in original.getmembers():
                payload = (
                    original.extractfile(member).read() if member.isfile() else None
                )
                target.addfile(member, None if payload is None else io.BytesIO(payload))
                if member.name.endswith("src/et_mainsim/reference_photometry.py"):
                    duplicate = (member, payload)
            assert duplicate is not None
            member, payload = duplicate
            target.addfile(member, io.BytesIO(payload))
        sdist = tampered

    with pytest.raises(ReleaseContractError, match="duplicate"):
        validate_artifacts(wheel, sdist, contract)


@pytest.mark.parametrize("archive_kind", ["wheel", "sdist"])
def test_artifact_validator_rejects_path_traversal_members(
    tmp_path: Path,
    archive_kind: str,
) -> None:
    contract = load_release_contract(CONTRACT_PATH)
    source = _clean_release_source(tmp_path)
    build_release(source, tmp_path / "release")
    artifact_root = tmp_path / "release" / "artifacts"
    wheel = artifact_root / "et_mainsim-0.1.0-py3-none-any.whl"
    sdist = artifact_root / "et_mainsim-0.1.0.tar.gz"

    if archive_kind == "wheel":
        tampered_dir = tmp_path / "traversal-wheel"
        tampered_dir.mkdir()
        tampered = tampered_dir / wheel.name
        shutil.copyfile(wheel, tampered)
        with zipfile.ZipFile(tampered, "a") as archive:
            archive.writestr("et_mainsim/../outside/", b"")
        wheel = tampered
    else:
        tampered = tmp_path / "traversal-member.tar.gz"
        with (
            tarfile.open(sdist, "r:gz") as original,
            tarfile.open(tampered, "w:gz") as target,
        ):
            for member in original.getmembers():
                payload = (
                    original.extractfile(member).read() if member.isfile() else None
                )
                target.addfile(member, None if payload is None else io.BytesIO(payload))
            traversal = tarfile.TarInfo("et_mainsim-0.1.0/../outside")
            traversal.type = tarfile.DIRTYPE
            target.addfile(traversal)
        sdist = tampered

    with pytest.raises(ReleaseContractError, match="safe canonical path"):
        validate_artifacts(wheel, sdist, contract)


@pytest.mark.parametrize("archive_kind", ["wheel", "sdist"])
def test_artifact_validator_rejects_backslash_path_traversal_members(
    tmp_path: Path,
    archive_kind: str,
) -> None:
    contract = load_release_contract(CONTRACT_PATH)
    source = _clean_release_source(tmp_path)
    build_release(source, tmp_path / "release")
    artifact_root = tmp_path / "release" / "artifacts"
    wheel = artifact_root / "et_mainsim-0.1.0-py3-none-any.whl"
    sdist = artifact_root / "et_mainsim-0.1.0.tar.gz"

    if archive_kind == "wheel":
        tampered_dir = tmp_path / "backslash-traversal-wheel"
        tampered_dir.mkdir()
        tampered = tampered_dir / wheel.name
        shutil.copyfile(wheel, tampered)
        with zipfile.ZipFile(tampered, "a") as archive:
            archive.writestr("et_mainsim\\..\\outside", b"traversal")
        wheel = tampered
    else:
        tampered = tmp_path / "backslash-traversal-member.tar.gz"
        with (
            tarfile.open(sdist, "r:gz") as original,
            tarfile.open(tampered, "w:gz") as target,
        ):
            for member in original.getmembers():
                payload = (
                    original.extractfile(member).read() if member.isfile() else None
                )
                target.addfile(member, None if payload is None else io.BytesIO(payload))
            traversal = tarfile.TarInfo("et_mainsim-0.1.0\\..\\outside")
            traversal.size = len(b"traversal")
            target.addfile(traversal, io.BytesIO(b"traversal"))
        sdist = tampered

    with pytest.raises(ReleaseContractError, match="safe canonical path"):
        validate_artifacts(wheel, sdist, contract)


def test_release_wheel_is_rebuilt_from_the_canonical_sdist(tmp_path: Path) -> None:
    source = _clean_release_source(tmp_path)
    receipt = build_release(source, tmp_path / "release")
    artifact_root = tmp_path / "release" / "artifacts"
    wheel = artifact_root / "et_mainsim-0.1.0-py3-none-any.whl"
    sdist = artifact_root / "et_mainsim-0.1.0.tar.gz"

    with tempfile.TemporaryDirectory(prefix="et-mainsim-sdist-rebuild-") as temporary:
        rebuilt_root = Path(temporary)
        with tarfile.open(sdist, "r:gz") as archive:
            archive.extractall(rebuilt_root, filter="data")
        dist = rebuilt_root / "dist"
        dist.mkdir()
        environment = {
            **os.environ,
            "PYTHONHASHSEED": "0",
            "SOURCE_DATE_EPOCH": str(receipt["source"]["source_date_epoch"]),
        }
        subprocess.run(
            [
                sys.executable,
                "-m",
                "build",
                "--no-isolation",
                "--wheel",
                "--outdir",
                str(dist),
                ".",
            ],
            cwd=rebuilt_root / "et_mainsim-0.1.0",
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        rebuilt = next(dist.glob("*.whl"))
        rebuilt_sha256 = artifact_sha256(rebuilt)

    assert rebuilt_sha256 == artifact_sha256(wheel)


def test_release_workflow_is_manual_build_only_and_immutable() -> None:
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert "workflow_dispatch:" in workflow
    assert "push:" not in workflow
    assert "pull_request_target:" not in workflow
    assert "permissions:\n  contents: read" in workflow
    assert (
        "python -m ci.build_release --output release-output --require-tag" in workflow
    )
    assert "github.ref == 'refs/tags/v0.1.0'" in workflow
    assert "twine upload" not in workflow
    assert "gh release" not in workflow
    assert "contents: write" not in workflow
    assert "persist-credentials: false" in workflow
    for bundle_path in (
        "release-output/artifacts/et_mainsim-0.1.0-py3-none-any.whl",
        "release-output/artifacts/et_mainsim-0.1.0.tar.gz",
        "release-output/artifacts/release-provenance.json",
        "release-output/artifacts/SHA256SUMS",
    ):
        assert bundle_path in workflow
    assert "path: release-output/artifacts\n" not in workflow
    refs = [
        line.split("@", 1)[1].split()[0]
        for line in workflow.splitlines()
        if "uses:" in line
    ]
    assert refs
    assert all(len(ref) == 40 and set(ref) <= set("0123456789abcdef") for ref in refs)


def test_release_contract_freezes_manual_release_policy() -> None:
    contract = load_release_contract(CONTRACT_PATH)

    assert contract["required_checks"] == [
        "full-test-gate",
        "package-boundary / py3.12",
        "package-boundary / py3.13",
    ]
    assert contract["release_policy"] == [
        "annotated_tag",
        "github_release",
        "no_pypi",
        "no_scientific_simulation",
    ]


def test_release_contract_reuses_full_test_runtime_dependency_identity() -> None:
    contract = load_release_contract(CONTRACT_PATH)
    with (ROOT / "ci" / "full_pytest_contract.toml").open("rb") as stream:
        full_test = tomllib.load(stream)

    assert contract["runtime_dependencies"] == {
        "et_coordinate": {
            "distribution": "et-coord",
            "repository": "TutuchanXD/ET-coordinate",
            "commit": full_test["dependencies"]["et_coordinate_commit"],
            "version": "0.1.2",
        },
        "photsim7": {
            "distribution": "photsim7",
            "repository": "TutuchanXD/Photsim7",
            "commit": full_test["dependencies"]["photsim7_commit"],
            "version": "0.2.5",
        },
    }

    with (ROOT / "pyproject.toml").open("rb") as stream:
        dependencies = tomllib.load(stream)["project"]["dependencies"]
    assert "et-coord>=0.1.2,<0.2" in dependencies

    galaxy_runtime = (
        ROOT / "src" / "et_mainsim" / "galaxy_stamp_production.py"
    ).read_text(encoding="utf-8")
    assert "verify_semantic_registry_owner_attestation" in galaxy_runtime
