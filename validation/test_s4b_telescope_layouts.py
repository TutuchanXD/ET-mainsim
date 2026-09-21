"""S4b real assets: homogeneous ET scopes and explicit reference translations.

Run explicitly with ET_DATA_DIR, ET_FOCALPLANE_ROOT and ET_S3_DEVICE=cpu|cuda.
The physical cases retain all five ET effects; the reference case has no DVA
physical claim and uses the documented reference-field projection.
"""
from dataclasses import replace
import json

import numpy as np
import pytest
from astropy import units as u

from test_s3_real_assets import _plan
from photsim7.specs import TelescopeSpec


def _scope_arrays(plan):
    return {
        path.relative_to(plan.run_dir).as_posix(): np.load(path)
        for path in sorted(plan.run_dir.glob('scope_*/frames/frame_*.npy'))
    }


def _check_crops(plan, scope_ids):
    from photsim7.artifacts import SharedExposureShardReader
    for scope_id in scope_ids:
        root = plan.run_dir / 'shared_exposure' / f'scope_{scope_id}'
        target_path = root / 'target_plan.json'
        if not target_path.exists():
            target_path = plan.run_dir / 'shared_exposure' / 'target_plan.json'
        target = json.loads(target_path.read_text())['targets'][0]
        for shard in root.rglob('*.h5'):
            with SharedExposureShardReader(shard) as reader:
                assert reader.provenance['scope_id'] == scope_id
                for frame in reader.frame_ids:
                    value = reader.read_array(1, frame)
                    assert value.shape == (9, 9)
                    assert np.all(np.isfinite(value))
                    parent = np.load(plan.run_dir / f'scope_{scope_id}' / 'frames' / f'frame_{frame:06d}.npy')
                    x0 = target['window']['x_start_detector_pix']
                    y0 = target['window']['y_start_detector_pix']
                    expected = np.zeros((9, 9), dtype=parent.dtype)
                    for y in range(9):
                        for x in range(9):
                            if 0 <= y0+y < parent.shape[0] and 0 <= x0+x < parent.shape[1]:
                                expected[y, x] = parent[y0+y, x0+x]
                    np.testing.assert_array_equal(value, expected)
        assert len(list((root / 'completion').glob('*.json'))) == 4


@pytest.mark.parametrize('count', [3, 8])
def test_physical_sparse_scopes_repeat_reorder_resume_and_shared_crops(tmp_path, count):
    from et_mainsim.config import SharedExposureStampsConfig
    scopes = tuple(TelescopeSpec(11 + i*7, 'main_rd') for i in range(count))

    def prepare(run_id):
        module, plan = _plan(tmp_path, 'full_frame', run_id)
        detector_id = plan.spec.detector.detector_id
        configured = tuple(replace(scope, detector_id=detector_id) for scope in scopes)
        spec = replace(plan.spec, instrument=replace(plan.spec.instrument,
            telescope_count=count, telescopes=configured), observation=replace(plan.spec.observation,
                exposure_duration=2*u.s, readout_duration=0.1*u.s, observing_duration=None,
                frame_start_s=(1., 3.2, 6., 9.)))
        shared = SharedExposureStampsConfig(enabled=True, target_source_ids=(1,),
            stamp_rows=9, stamp_cols=9, frames_per_shard=2, product_keys=('final_stamp',))
        config = replace(plan.run_config, workload=replace(plan.run_config.workload, shared_exposure_stamps=shared))
        return module, replace(plan, spec=spec, run_config=config)

    module, plan = prepare('first')
    assert module.run_full_frame(plan)['status'] == 'completed'
    first = _scope_arrays(plan)
    assert len(first) == 4*count
    _check_crops(plan, [scope.scope_id for scope in scopes])
    reordered = replace(plan, spec=replace(plan.spec, instrument=replace(plan.spec.instrument,
        telescopes=plan.spec.instrument.telescopes[::-1])))
    assert module.run_full_frame(reordered)['status'] == 'completed'
    _, repeat = prepare('repeat')
    assert module.run_full_frame(repeat)['status'] == 'completed'
    second = _scope_arrays(repeat)
    assert first.keys() == second.keys()
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    assert not np.array_equal(first['scope_11/frames/frame_000000.npy'], first['scope_18/frames/frame_000000.npy'])
    for path in plan.run_dir.glob('scope_*/frame_summaries/*_schema.json'):
        schema = json.loads(path.read_text())
        assert schema['provenance']['services']['scope_contract']['scope_ids'] == [scope.scope_id for scope in scopes]


def test_reference_translation_real_psf_dynamic_fields_and_crops(tmp_path):
    from et_mainsim.config import SharedExposureStampsConfig
    from photsim7.catalogs.sources import PreparedStarCatalog
    from photsim7.geometry_truth import reference_field_nonphysical_declaration
    module, plan = _plan(tmp_path, 'full_frame', 'reference')
    scopes = tuple(TelescopeSpec(scope_id, plan.spec.detector.detector_id,
        'reference_translation', offset) for scope_id, offset in ((11, (0., 0.)), (23, (2., -1.)), (71, (0.25, -0.5))))
    catalog = PreparedStarCatalog(star_data={
        'source_id': np.array([1]), 'x0': np.array([0.]), 'y0': np.array([0.]),
        'frame_xpix': np.array([15.]), 'frame_ypix': np.array([15.]),
        'et_mag': np.array([12.]), 'ra': np.array([0.]), 'dec': np.array([0.]),
    }, metadata={'source': {'type': 'prepared'}, 'geometry': reference_field_nonphysical_declaration(
        reference_field_angle_deg=plan.spec.catalog.query_options['reference_field_angle_deg'],
        reference_field_polar_angle_rad=plan.spec.catalog.query_options['reference_field_polar_angle_rad'])})
    cache = tmp_path / 'reference_stars.npz'
    spec = replace(plan.spec, instrument=replace(plan.spec.instrument, telescope_count=3, telescopes=scopes),
        catalog=replace(plan.spec.catalog, cache_path=str(cache)),
        dynamic_effects=replace(plan.spec.dynamic_effects, dva=replace(plan.spec.dynamic_effects.dva, enabled=False)),
        psf=replace(plan.spec.psf, jitter_bank_mode='single_reference', field_id=0, field_id_policy=None))
    from photsim7.simulation_services import build_catalog_from_spec
    from photsim7.data_registry import DataRegistry
    build_catalog_from_spec(spec, data_registry=DataRegistry(plan.paths.data_root), prepared_catalog=catalog)
    shared = SharedExposureStampsConfig(enabled=True, target_source_ids=(1,),
        stamp_rows=9, stamp_cols=9, frames_per_shard=2, product_keys=('final_stamp',))
    config = replace(plan.run_config, paths=replace(plan.run_config.paths, catalog_cache=str(cache)),
        workload=replace(plan.run_config.workload, shared_exposure_stamps=shared))
    plan = replace(plan, spec=spec, run_config=config, catalog_cache=cache)
    assert module.run_full_frame(plan)['status'] == 'completed'
    before = _scope_arrays(plan)
    assert len(before) == 12
    assert module.run_full_frame(plan)['status'] == 'completed'
    _check_crops(plan, [scope.scope_id for scope in scopes])
    for scope in scopes:
        target_plan = json.loads((plan.run_dir / 'shared_exposure' / f'scope_{scope.scope_id}' / 'target_plan.json').read_text())
        assert target_plan['targets'][0]['x_frame_pix'] == 15 + scope.placement_offset_xy_pix[0]
        assert target_plan['targets'][0]['y_frame_pix'] == 15 + scope.placement_offset_xy_pix[1]
    for key, value in before.items():
        np.testing.assert_array_equal(value, _scope_arrays(plan)[key])
