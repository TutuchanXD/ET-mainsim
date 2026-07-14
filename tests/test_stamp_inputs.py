from __future__ import annotations

import pytest
from astropy.table import Table


def test_stamp_target_table_defaults_to_gaia_g_and_detector_center(tmp_path) -> None:
    from et_mainsim.stamp_inputs import load_stamp_target_table

    path = tmp_path / "targets.csv"
    path.write_text("gaia_g_mag,psf_id\n12.0,0\n14.5,6\n", encoding="utf-8")

    loaded = load_stamp_target_table(path, detector_shape=(9120, 8900))

    assert [item.source_id for item in loaded.targets] == [0, 1]
    assert [item.gaia_g_mag for item in loaded.targets] == [12.0, 14.5]
    assert [item.psf_id for item in loaded.targets] == [0, 6]
    assert loaded.targets[0].detector_xpix == pytest.approx(4449.5)
    assert loaded.targets[0].detector_ypix == pytest.approx(4559.5)
    assert loaded.provenance["row_count"] == 2
    assert "sha256" not in loaded.provenance


def test_stamp_target_table_accepts_documented_aliases_and_coordinates(tmp_path) -> None:
    from et_mainsim.stamp_inputs import load_stamp_target_table

    path = tmp_path / "targets.ecsv"
    Table(
        {
            "Source ID": [42],
            "Gaia G Mag": [13.2],
            "PSF ID": [5],
            "Detector Xpix": [100.25],
            "Detector Ypix": [200.75],
        }
    ).write(
        path,
        format="ascii.ecsv",
    )

    loaded = load_stamp_target_table(path, detector_shape=(9120, 8900))

    assert loaded.targets[0].to_dict() == {
        "source_id": 42,
        "gaia_g_mag": 13.2,
        "psf_id": 5,
        "detector_xpix": 100.25,
        "detector_ypix": 200.75,
    }


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("gaia_g_mag\n12.0\n", "PSF"),
        ("psf_id\n0\n", "Gaia G"),
        ("gaia_g_mag,psf_id,detector_xpix\n12,0,2\n", "together"),
        ("gaia_g_mag,psf_id,source_id\n12,0,1\n13,2,1\n", "unique"),
        ("gaia_g_mag,psf_id\nnan,0\n", "finite"),
        ("gaia_g_mag,psf_id,detector_xpix,detector_ypix\n12,0,99,2\n", "inside"),
    ],
)
def test_stamp_target_table_rejects_invalid_rows(tmp_path, body, message) -> None:
    from et_mainsim.stamp_inputs import load_stamp_target_table

    path = tmp_path / "bad.csv"
    path.write_text(body, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_stamp_target_table(path, detector_shape=(9, 11))
