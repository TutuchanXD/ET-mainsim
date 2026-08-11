from __future__ import annotations

import base64
import csv
import hashlib
from importlib import metadata
import json
from pathlib import Path

import pytest


def _record_hash(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256(path.read_bytes()).digest()
    encoded = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return f"sha256={encoded}", path.stat().st_size


def _fake_wheel_distribution(
    root: Path,
    *,
    editable: bool = False,
    missing_hash_for: str | None = None,
) -> tuple[metadata.Distribution, Path]:
    site = root / "site"
    package = site / "example_pkg"
    dist_info = site / "example_dist-1.2.3.dist-info"
    package.mkdir(parents=True)
    dist_info.mkdir()
    (package / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    (package / "module.py").write_text("VALUE = 2\n", encoding="utf-8")
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: example-dist\nVersion: 1.2.3\n",
        encoding="utf-8",
    )
    (dist_info / "WHEEL").write_text(
        "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\n"
        "Tag: py3-none-any\n",
        encoding="utf-8",
    )
    files = [
        package / "__init__.py",
        package / "module.py",
        dist_info / "METADATA",
        dist_info / "WHEEL",
    ]
    if editable:
        direct_url = dist_info / "direct_url.json"
        direct_url.write_text(
            json.dumps(
                {
                    "dir_info": {"editable": True},
                    "url": "file:///source/example-dist",
                }
            ),
            encoding="utf-8",
        )
        files.append(direct_url)
    record = dist_info / "RECORD"
    with record.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        for path in files:
            relative = path.relative_to(site).as_posix()
            if relative == missing_hash_for:
                writer.writerow((relative, "", ""))
            else:
                hash_value, size = _record_hash(path)
                writer.writerow((relative, hash_value, size))
        writer.writerow((record.relative_to(site).as_posix(), "", ""))
    distribution = next(metadata.distributions(path=[str(site)]))
    return distribution, package


def test_installed_distribution_identity_is_verified_and_path_independent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import et_mainsim.provenance as provenance

    first_dist, first_package = _fake_wheel_distribution(tmp_path / "first")
    second_dist, second_package = _fake_wheel_distribution(tmp_path / "second")

    monkeypatch.setattr(
        provenance.importlib.metadata,
        "distribution",
        lambda _name: first_dist,
    )
    first = provenance._installed_distribution_identity_v1(
        "example-dist",
        first_package,
    )
    monkeypatch.setattr(
        provenance.importlib.metadata,
        "distribution",
        lambda _name: second_dist,
    )
    second = provenance._installed_distribution_identity_v1(
        "example-dist",
        second_package,
    )

    assert first == second
    assert first == {
        "schema_id": "et_mainsim.installed_distribution_identity.v1",
        "name": "example-dist",
        "version": "1.2.3",
        "record_entry_count": 4,
        "record_tree_sha256": first["record_tree_sha256"],
    }
    assert len(first["record_tree_sha256"]) == 64


@pytest.mark.parametrize("failure_mode", ["missing_hash", "content_drift", "editable"])
def test_installed_distribution_identity_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_mode: str,
) -> None:
    import et_mainsim.provenance as provenance

    missing_hash = (
        "example_pkg/module.py" if failure_mode == "missing_hash" else None
    )
    distribution, package = _fake_wheel_distribution(
        tmp_path,
        editable=failure_mode == "editable",
        missing_hash_for=missing_hash,
    )
    if failure_mode == "content_drift":
        (package / "module.py").write_text("VALUE = 999\n", encoding="utf-8")
    monkeypatch.setattr(
        provenance.importlib.metadata,
        "distribution",
        lambda _name: distribution,
    )

    assert (
        provenance._installed_distribution_identity_v1("example-dist", package)
        is None
    )


@pytest.mark.parametrize(
    "git_record",
    [
        {"root": "/checkout", "commit": "a" * 40, "branch": "main", "dirty": False},
        {"root": "/checkout", "commit": "a" * 40, "branch": "main", "dirty": True},
    ],
)
def test_package_provenance_does_not_mask_a_checkout_with_wheel_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    git_record: dict[str, object],
) -> None:
    import et_mainsim.provenance as provenance

    monkeypatch.setattr(provenance, "git_provenance", lambda _root: git_record)
    monkeypatch.setattr(provenance, "_distribution_version", lambda _name: "1.2.3")
    monkeypatch.setattr(
        provenance,
        "_installed_distribution_identity_v1",
        lambda *_args: pytest.fail("checkout provenance must take priority"),
    )

    result = provenance._package_provenance(
        git_root=tmp_path,
        package_root=tmp_path,
        distribution_name="example-dist",
    )

    assert "distribution_identity" not in result
    assert result["commit"] == "a" * 40
    assert result["dirty"] is git_record["dirty"]
