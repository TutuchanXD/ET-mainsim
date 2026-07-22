from __future__ import annotations

from types import SimpleNamespace
import sys

import numpy as np
import pytest
from astropy.table import Table


def _stub_semantic_registry_identity(monkeypatch, stamp_inputs, registry) -> None:
    candidate = {
        "schema_id": "et_coord.semantic_registry_identity.v1",
        "schema_version": 1,
        "freeze_status": "candidate_pending_owner_freeze",
        "owner_approval_required": True,
        "registry_data_dir": str(registry),
        "sha256": "b" * 64,
    }
    monkeypatch.setattr(
        stamp_inputs,
        "focalplane_registry_identity",
        lambda _path: candidate,
    )

    class _Registry:
        def get_detector(self, _detector_id: str):
            # Existing coordinate fixtures use detector_shape=(rows=9,
            # cols=11); their physical geometry is intentionally expressed in
            # x/y order here rather than inheriting raster orientation.
            return SimpleNamespace(pixel_width=11.0, pixel_height=9.0)

    monkeypatch.setattr(
        stamp_inputs,
        "_load_focalplane_registry",
        lambda *_args, **_kwargs: _Registry(),
    )


def _stub_detector_physical_pixel_shape(
    monkeypatch,
    stamp_inputs,
    *,
    detector_id: str,
    pixel_width: float,
    pixel_height: float,
) -> None:
    """Install the focal-plane geometry used by coordinate-target tests."""

    detector = SimpleNamespace(
        pixel_width=pixel_width,
        pixel_height=pixel_height,
    )

    class _Registry:
        def get_detector(self, requested_detector_id: str):
            assert requested_detector_id == detector_id
            return detector

    monkeypatch.setattr(
        stamp_inputs,
        "_load_focalplane_registry",
        lambda *_args, **_kwargs: _Registry(),
    )


def test_focalplane_registry_identity_uses_et_coord_semantic_candidate(
    monkeypatch,
    tmp_path,
) -> None:
    import et_coord
    from et_mainsim.stamp_inputs import focalplane_registry_identity

    candidate = {
        "schema_id": "et_coord.semantic_registry_identity.v1",
        "schema_version": 1,
        "freeze_status": "candidate_pending_owner_freeze",
        "owner_approval_required": True,
        "registry_data_dir": str(tmp_path / "data"),
        "sha256": "a" * 64,
    }
    captured = {}

    def fake_identity(path):
        captured["path"] = path
        return candidate

    monkeypatch.setattr(et_coord, "semantic_registry_identity", fake_identity)

    identity = focalplane_registry_identity(tmp_path)

    assert identity == candidate
    assert captured["path"] == tmp_path.resolve()


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
    assert len(loaded.provenance["file_identity"]["sha256"]) == 64
    assert loaded.provenance["magnitude_system"] == "Gaia_G_Vega"


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
        "curve_id": None,
        "location_mode": "explicit_psf",
        "ra_deg": None,
        "dec_deg": None,
        "field_x_deg": None,
        "field_y_deg": None,
        "field_angle_deg": None,
        "focalplane_residual_arcsec": None,
        "psf_node_angle_deg": None,
        "psf_angle_delta_deg": None,
    }


def test_coordinate_target_uses_physical_focalplane_pixel_bounds_not_raster_shape(
    tmp_path,
    monkeypatch,
) -> None:
    """A rotated main detector may be wider than the renderer's raster columns."""

    import et_mainsim.stamp_inputs as stamp_inputs

    registry = tmp_path / "focalplane"
    registry.mkdir()
    path = tmp_path / "targets.ecsv"
    Table(
        {
            "source_id": [2080632520701306880],
            "gaia_g_mag": [11.0],
            "ra_deg": [10.0],
            "dec_deg": [-20.0],
        }
    ).write(path, format="ascii.ecsv")
    monkeypatch.setattr(
        stamp_inputs,
        "_sky_to_focal",
        lambda *_args, **_kwargs: SimpleNamespace(
            status="ok",
            detector_id="main_ld",
            # main_ld is physically 9120 x 8900 pixels.  The ET simulation
            # raster remains (rows, cols) = (9120, 8900), so this valid x
            # coordinate must not be compared with raster cols=8900.
            xpix=9065.726,
            ypix=6185.964,
            field_x_deg=1.0,
            field_y_deg=2.0,
            residual_arcsec=0.01,
        ),
    )
    _stub_semantic_registry_identity(monkeypatch, stamp_inputs, registry)
    _stub_detector_physical_pixel_shape(
        monkeypatch,
        stamp_inputs,
        detector_id="main_ld",
        pixel_width=9120.0,
        pixel_height=8900.0,
    )

    loaded = stamp_inputs.load_stamp_target_table(
        path,
        detector_shape=(9120, 8900),
        detector_id="main_ld",
        focalplane_registry=registry,
    )

    assert loaded.targets[0].detector_xpix == pytest.approx(9065.726)
    assert loaded.targets[0].detector_ypix == pytest.approx(6185.964)
    assert loaded.provenance["coordinate_physical_detector_bounds"] == {
        "detector_id": "main_ld",
        "pixel_width": 9120.0,
        "pixel_height": 8900.0,
    }


def test_stamp_target_table_preserves_gaia_source_id_decimal_string(
    tmp_path,
) -> None:
    from et_mainsim.stamp_inputs import load_stamp_target_table

    source_id = 2_100_787_084_231_447_424
    path = tmp_path / "target.ecsv"
    Table(
        {
            "source_id": [str(source_id)],
            "gaia_g_mag": [13.2],
            "psf_id": [5],
        }
    ).write(path, format="ascii.ecsv")

    loaded = load_stamp_target_table(path, detector_shape=(9, 11))

    assert loaded.targets[0].source_id == source_id
    assert type(loaded.targets[0].source_id) is int


@pytest.mark.parametrize(
    "value",
    [
        2_100_787_084_231_447_424,
        np.int64(2_100_787_084_231_447_424),
        "2100787084231447424",
        "  +2100787084231447424  ",
        9_000_000_000_000_000_000,
        np.int64(9_000_000_000_000_000_000),
        np.iinfo(np.int64).max,
    ],
)
def test_integer_preserves_exact_source_ids_as_python_int(value) -> None:
    from et_mainsim.stamp_inputs import _integer

    source_id = _integer(value, field_name="source_id", row_index=3)

    assert source_id == int(value)
    assert type(source_id) is int


@pytest.mark.parametrize("value", [float(2**53), np.float64(2**53)])
def test_integer_accepts_exact_float_safety_boundary(value) -> None:
    from et_mainsim.stamp_inputs import _integer

    assert _integer(value, field_name="source_id", row_index=3) == 2**53


@pytest.mark.parametrize(
    "value",
    [
        float(2**53 + 2),
        np.float64(2**53 + 2),
        float(2_100_787_084_231_447_424),
        np.float64(2_100_787_084_231_447_424),
    ],
)
def test_integer_rejects_source_id_float_above_exact_safety_boundary(value) -> None:
    from et_mainsim.stamp_inputs import _integer

    with pytest.raises(ValueError, match="integer"):
        _integer(value, field_name="source_id", row_index=3)


@pytest.mark.parametrize(
    "value",
    [
        True,
        np.bool_(False),
        -1,
        np.int64(-1),
        "-1",
        1.5,
        np.float64(1.5),
        "1.5",
        float("nan"),
        float("inf"),
        float("-inf"),
        np.iinfo(np.int64).max + 1,
        np.uint64(np.iinfo(np.int64).max) + np.uint64(1),
        str(np.iinfo(np.int64).max + 1),
    ],
)
def test_integer_rejects_invalid_or_out_of_int64_source_id(value) -> None:
    from et_mainsim.stamp_inputs import _integer

    with pytest.raises(ValueError, match="integer"):
        _integer(value, field_name="source_id", row_index=3)


def test_stamp_target_table_preserves_curve_and_ecsv_semantics_note(tmp_path) -> None:
    from et_mainsim.stamp_inputs import load_stamp_target_table

    path = tmp_path / "targets.ecsv"
    table = Table(
        {
            "source_id": [7],
            "gaia_g_mag": [18.25],
            "psf_id": [3],
            "curve_id": ["sn_z0p10"],
        }
    )
    table.meta["magnitude_semantics_note"] = (
        "truncated_gaia_g_ab_treated_as_gaia_g_vega_engineering_proxy"
    )
    table.write(path, format="ascii.ecsv")

    loaded = load_stamp_target_table(path, detector_shape=(9, 11))

    assert loaded.targets[0].curve_id == "sn_z0p10"
    assert loaded.provenance["table_meta"] == {
        "magnitude_semantics_note": (
            "truncated_gaia_g_ab_treated_as_gaia_g_vega_engineering_proxy"
        )
    }


def test_stamp_target_table_maps_icrs_to_expected_detector_and_field_angle(
    tmp_path, monkeypatch
) -> None:
    import et_mainsim.stamp_inputs as stamp_inputs

    registry = tmp_path / "focalplane"
    registry.mkdir()
    (registry / "fov.csv").write_text("detector\nmain_rd\n", encoding="utf-8")
    path = tmp_path / "targets.csv"
    path.write_text(
        "source_id,gaia_g_mag,ra_deg,dec_deg,curve_id\n"
        "9,17.5,123.25,-31.5,sn\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        stamp_inputs,
        "_sky_to_focal",
        lambda *_args, **_kwargs: SimpleNamespace(
            status="ok",
            detector_id="main_rd",
            xpix=4.25,
            ypix=6.75,
            field_x_deg=3.0,
            field_y_deg=4.0,
            residual_arcsec=0.02,
        ),
    )
    _stub_semantic_registry_identity(monkeypatch, stamp_inputs, registry)

    loaded = stamp_inputs.load_stamp_target_table(
        path,
        detector_shape=(9, 11),
        detector_id="main_rd",
        focalplane_registry=registry,
    )

    target = loaded.targets[0]
    assert target.location_mode == "sky_icrs_j2000"
    assert target.psf_id is None
    assert target.detector_xpix == pytest.approx(4.25)
    assert target.detector_ypix == pytest.approx(6.75)
    assert target.field_angle_deg == pytest.approx(5.0)
    assert target.focalplane_residual_arcsec == pytest.approx(0.02)
    assert loaded.provenance["coordinate_frame"] == "ICRS_J2000"
    assert len(loaded.provenance["focalplane_registry_identity"]["sha256"]) == 64


def test_stamp_target_table_allows_mixed_sky_and_explicit_rows(
    tmp_path, monkeypatch
) -> None:
    import et_mainsim.stamp_inputs as stamp_inputs

    registry = tmp_path / "focalplane"
    registry.mkdir()
    (registry / "fov.csv").write_text("x\n1\n", encoding="utf-8")
    path = tmp_path / "targets.csv"
    path.write_text(
        "source_id,gaia_g_mag,ra_deg,dec_deg,psf_id\n"
        "1,12.0,123.0,-20.0,\n"
        "2,13.0,,,4\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        stamp_inputs,
        "_sky_to_focal",
        lambda *_args, **_kwargs: SimpleNamespace(
            status="ok",
            detector_id="main_rd",
            xpix=4.0,
            ypix=5.0,
            field_x_deg=3.0,
            field_y_deg=4.0,
            residual_arcsec=0.01,
        ),
    )
    _stub_semantic_registry_identity(monkeypatch, stamp_inputs, registry)

    loaded = stamp_inputs.load_stamp_target_table(
        path,
        detector_shape=(9, 11),
        detector_id="main_rd",
        focalplane_registry=registry,
    )

    assert [target.location_mode for target in loaded.targets] == [
        "sky_icrs_j2000",
        "explicit_psf",
    ]
    assert loaded.targets[1].psf_id == 4
    assert loaded.targets[1].detector_xpix == pytest.approx(5.0)
    assert loaded.targets[1].detector_ypix == pytest.approx(4.0)


@pytest.mark.parametrize(
    ("result", "message"),
    [
        (SimpleNamespace(status="out_of_fov"), "outside"),
        (
            SimpleNamespace(
                status="ok",
                detector_id="main_lu",
                xpix=4.0,
                ypix=4.0,
                field_x_deg=1.0,
                field_y_deg=1.0,
                residual_arcsec=0.1,
            ),
            "main_lu.*main_rd",
        ),
    ],
)
def test_stamp_target_table_rejects_bad_sky_mapping(
    tmp_path, monkeypatch, result, message
) -> None:
    import et_mainsim.stamp_inputs as stamp_inputs

    registry = tmp_path / "focalplane"
    registry.mkdir()
    (registry / "fov.csv").write_text("x\n1\n", encoding="utf-8")
    path = tmp_path / "targets.csv"
    path.write_text("gaia_g_mag,ra_deg,dec_deg\n12,10,20\n", encoding="utf-8")
    monkeypatch.setattr(stamp_inputs, "_sky_to_focal", lambda *_a, **_k: result)
    _stub_semantic_registry_identity(monkeypatch, stamp_inputs, registry)

    with pytest.raises(ValueError, match=message):
        stamp_inputs.load_stamp_target_table(
            path,
            detector_shape=(9, 11),
            detector_id="main_rd",
            focalplane_registry=registry,
        )


def test_coordinate_mode_reports_actionable_missing_et_coord(
    tmp_path, monkeypatch
) -> None:
    import et_mainsim.stamp_inputs as stamp_inputs

    stamp_inputs._load_focalplane_registry.cache_clear()
    monkeypatch.setitem(sys.modules, "et_coord", None)

    with pytest.raises(RuntimeError, match="coordinate targets require et-coord"):
        stamp_inputs._load_focalplane_registry(str(tmp_path))


def test_stamp_variability_table_is_frame_aligned_and_preserves_metadata(
    tmp_path,
) -> None:
    from et_mainsim.stamp_inputs import load_stamp_variability_table

    path = tmp_path / "curves.ecsv"
    table = Table(
        {
            "curve_id": ["static", "static", "sn", "sn"],
            "frame_index": [0, 1, 0, 1],
            "relative_flux": [1.0, 1.0, 0.5, 2.0],
            "observer_time_day": [0.0, 100.0, -20.0, 40.0],
        }
    )
    table.meta["time_semantics"] = "ignored_input_time_frame_index_only"
    table.write(path, format="ascii.ecsv")

    loaded = load_stamp_variability_table(path, raw_frame_count=2)

    assert loaded.curves["static"] == (1.0, 1.0)
    assert loaded.curves["sn"] == (0.5, 2.0)
    assert loaded.provenance["time_alignment"] == "simulation_raw_frame_index"
    assert loaded.provenance["input_time_columns_ignored"] == [
        "observer_time_day"
    ]
    assert loaded.provenance["table_meta"]["time_semantics"].startswith("ignored")
    assert len(loaded.provenance["file_identity"]["sha256"]) == 64


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("curve_id,frame_index\na,0\n", "relative_flux"),
        ("curve_id,relative_flux\na,1\n", "frame_index"),
        ("frame_index,relative_flux\n0,1\n", "curve_id"),
        ("curve_id,frame_index,relative_flux\na,0,1\na,0,2\n", "duplicate"),
        ("curve_id,frame_index,relative_flux\na,0,1\na,2,2\n", "exactly"),
        ("curve_id,frame_index,relative_flux\na,0,nan\na,1,1\n", "finite"),
        ("curve_id,frame_index,relative_flux\na,0,-1\na,1,1\n", "non-negative"),
    ],
)
def test_stamp_variability_table_rejects_invalid_curves(
    tmp_path, body, message
) -> None:
    from et_mainsim.stamp_inputs import load_stamp_variability_table

    path = tmp_path / "bad.csv"
    path.write_text(body, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_stamp_variability_table(path, raw_frame_count=2)


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("gaia_g_mag\n12.0\n", "PSF"),
        ("psf_id\n0\n", "Gaia G"),
        ("gaia_g_mag,psf_id,detector_xpix\n12,0,2\n", "together"),
        ("gaia_g_mag,psf_id,source_id\n12,0,1\n13,2,1\n", "unique"),
        ("gaia_g_mag,psf_id\nnan,0\n", "finite"),
        ("gaia_g_mag,psf_id,detector_xpix,detector_ypix\n12,0,99,2\n", "inside"),
        ("gaia_g_mag,ra_deg,dec_deg,psf_id\n12,10,20,0\n", "mutually"),
        ("gaia_g_mag,ra_deg\n12,10\n", "together"),
        ("gaia_g_mag,psf_id,detector_xpix,detector_ypix,ra_deg,dec_deg\n12,0,1,2,10,20\n", "mutually"),
    ],
)
def test_stamp_target_table_rejects_invalid_rows(tmp_path, body, message) -> None:
    from et_mainsim.stamp_inputs import load_stamp_target_table

    path = tmp_path / "bad.csv"
    path.write_text(body, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_stamp_target_table(path, detector_shape=(9, 11))
