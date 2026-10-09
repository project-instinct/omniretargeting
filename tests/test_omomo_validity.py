"""Source gaps, unchanged official rows, and reset provenance integration."""
import copy

import joblib
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from omniretargeting.data_sources.base import MotionData
from omniretargeting.data_sources.omomo import OmomoDataSource
from omniretargeting.data_sources.omomo_validity import prepare_object_frames
from omniretargeting.source_validity import export_object_validity
from scripts.review_penetration_persistence import _reset_report
from scripts.review_training_collision import TrainingTrajectory, audit_reset
from tests.test_omomo_adapter import _write_sequence


def sequence(frames, invalid=()):
    scale = np.linspace(1, 2, frames, dtype=np.float32)
    trans = np.arange(frames * 3, dtype=np.float32).reshape(frames, 3, 1) / 10
    rot = Rotation.from_rotvec(np.column_stack([np.zeros(frames), np.zeros(frames), np.linspace(0, 2 * np.pi / 3, frames)])).as_matrix().astype(np.float32)
    result = dict(seq_name='test_cube_000', obj_scale=scale, obj_trans=trans,
                  obj_rot=rot, obj_com_pos=trans[..., 0].copy(),
                  trans=np.arange(frames * 3, dtype=np.float32).reshape(frames, 3),
                  root_orient=np.zeros((frames, 3), np.float32),
                  pose_body=np.zeros((frames, 63), np.float32),
                  pose_hand=np.zeros((frames, 90), np.float32),
                  betas=np.arange(16, dtype=np.float32)[None],
                  rest_offsets=np.ones((24, 3), np.float32))
    result['obj_scale'][list(invalid)] = 0
    result['obj_trans'][list(invalid)] = 0
    result['obj_rot'][list(invalid)] = 0
    result['obj_com_pos'][list(invalid)] = 0
    return result


def assert_bytes_same(a, b):
    assert a.dtype == b.dtype
    assert a.shape == b.shape
    assert a.tobytes() == b.tobytes()


def test_short_gap_linear_scale_translation_and_slerp_only_missing_rows():
    raw = sequence(9, [3, 4, 5])
    untouched = copy.deepcopy(raw)
    fixed, valid, report = prepare_object_frames(raw, 30)
    assert report['interpolated_frames'] == [3, 4, 5]
    assert report['truncated_frames'] == 0
    assert valid.tolist() == [True] * 3 + [False] * 3 + [True] * 3
    expected = sequence(9)
    for key in ('obj_scale', 'obj_trans', 'obj_com_pos', 'obj_rot'):
        np.testing.assert_allclose(fixed[key], expected[key], atol=1e-7)
        assert_bytes_same(fixed[key][valid], raw[key][valid])
    for key, value in raw.items():
        if isinstance(value, np.ndarray):
            assert_bytes_same(value, untouched[key])
    assert_bytes_same(fixed['trans'], raw['trans'])
    # The half-way rotation is 60 degrees, not a non-orthogonal matrix blend.
    np.testing.assert_allclose(fixed['obj_rot'][4].T @ fixed['obj_rot'][4], np.eye(3), atol=1e-7)


def test_long_gap_keeps_longest_interval_and_slices_all_body_rows():
    raw = sequence(40, range(8, 24))  # 16/30 > .5 s; later interval is longest.
    fixed, valid, report = prepare_object_frames(raw)
    assert (report['retained_start'], report['retained_stop']) == (24, 40)
    assert report['truncated_frames'] == 24
    assert report['interpolated_frames'] == []
    assert valid.all()
    for key in ('trans', 'root_orient', 'pose_body', 'pose_hand', 'obj_scale', 'obj_trans', 'obj_rot', 'obj_com_pos'):
        assert_bytes_same(fixed[key], raw[key][24:])
    # 16-frame retained interval must not turn a 16-dimensional beta into time.
    assert_bytes_same(fixed['betas'], raw['betas'])
    assert_bytes_same(fixed['rest_offsets'], raw['rest_offsets'])
    np.testing.assert_array_equal(fixed['frame_ids'], np.arange(24, 40))


def test_edges_are_truncated_and_short_internal_gap_still_bridged():
    raw = sequence(20, [0, 1, 8, 9, 18, 19])
    fixed, valid, report = prepare_object_frames(raw)
    assert (report['retained_start'], report['retained_stop']) == (2, 18)
    assert report['truncated_frames'] == 4
    assert report['interpolated_frames'] == [8, 9]
    assert np.flatnonzero(~valid).tolist() == [6, 7]
    assert_bytes_same(fixed['trans'], raw['trans'][2:18])


@pytest.mark.parametrize('count,expected_fill', [(15, True), (16, False)])
def test_half_second_boundary(count, expected_fill):
    raw = sequence(50, range(10, 10 + count))
    _, _, report = prepare_object_frames(raw, 30)
    assert bool(report['interpolated_frames']) == expected_fill


def test_no_gap_retains_every_bit_and_origin_translation_is_valid():
    raw = sequence(10)
    raw['obj_trans'][0] = 0
    fixed, valid, report = prepare_object_frames(raw)
    assert fixed is raw
    assert valid.all()
    assert report['truncated_frames'] == report['interpolated_frame_count'] == 0
    for key, value in raw.items():
        if isinstance(value, np.ndarray):
            assert_bytes_same(fixed[key], value)


def test_invalid_scale_includes_negative_and_nonfinite():
    raw = sequence(8)
    raw['obj_scale'][2:6] = [-1, np.nan, np.inf, -np.inf]
    fixed, valid, report = prepare_object_frames(raw)
    assert report['invalid_frames'] == [2, 3, 4, 5]
    assert np.isfinite(fixed['obj_scale']).all()
    assert np.flatnonzero(~valid).tolist() == [2, 3, 4, 5]


def test_no_valid_frames_fail_with_sequence_name_and_ties_choose_earliest():
    with pytest.raises(ValueError, match='test_cube_000: no valid object frames'):
        prepare_object_frames(sequence(3, range(3)))
    _, _, report = prepare_object_frames(sequence(25, range(5, 21)))
    assert (report['retained_start'], report['retained_stop']) == (0, 5)
    _, _, report = prepare_object_frames(sequence(26, range(5, 21)))
    assert (report['retained_start'], report['retained_stop']) == (0, 5)


def test_adapter_truncates_before_smplx_fk_surfaces_and_sole_extraction(monkeypatch, tmp_path):
    path, root, _, _ = _write_sequence(tmp_path)
    raw = sequence(40, range(8, 24))
    raw['seq_name'] = 'seq_cube_000'
    joblib.dump([raw], path)
    captured = {}

    def body_loader(self):
        captured['raw'] = self.sequence
        frames = len(self.sequence['trans'])
        positions = np.tile(self.sequence['trans'][:, None], (1, 22, 1))
        orientations = np.zeros((frames, 22, 4))
        orientations[..., 0] = 1
        return positions, orientations, self.sequence['root_orient'], {
            'body_surface_samples': {'parts': {'hand': {'triangles': np.zeros((frames, 1, 3, 3))}}},
            'foot_sole_centres': {'left': self.sequence['trans'].copy()},
        }

    monkeypatch.setattr(OmomoDataSource, '_load_body_motion', body_loader)
    motion = OmomoDataSource(path, 0, root).load()
    assert len(motion.positions) == len(motion.object_points) == 16
    assert_bytes_same(captured['raw']['trans'], raw['trans'][24:])
    assert len(motion.metadata['body_surface_samples']['parts']['hand']['triangles']) == 16
    assert_bytes_same(motion.metadata['foot_sole_centres']['left'], raw['trans'][24:])
    np.testing.assert_array_equal(motion.metadata['source_frame_indices'], np.arange(24, 40))


def test_npz_export_and_both_reset_audits_keep_source_invalid_even_with_clean_geometry(tmp_path):
    valid = np.array([True, False, True])
    payload = export_object_validity({'object_valid': valid, 'source_frame_indices': np.arange(3)}, 3)
    path = tmp_path / 'output.npz'
    np.savez(path, **payload)
    with np.load(path, allow_pickle=False) as z:
        assert z['object_valid'].dtype == bool
        assert z['reset_invalid'].tolist() == [False, True, False]
        assert z['reset_states'].tolist() == ['unknown', 'invalid', 'unknown']
    zeros = np.zeros(3)
    result = _reset_report(static_depth=zeros, moving_depth=zeros, self_depth=zeros,
                           labels=np.arange(3), scene_unknown=True, self_error_frames=set(),
                           lateral=np.zeros(3, bool), object_valid=valid)
    assert result['states'] == ['unknown', 'invalid', 'unknown']
    assert result['unusable_reasons']['source_object_invalid'] == [[1, 1]]
    trajectory = TrainingTrajectory(np.zeros((3, 7)), np.arange(3), np.arange(3) + 10, {})
    result = audit_reset(payload, {}, trajectory, {}, {}, {})
    assert result['failed_frames'] == [11]
    assert result['source_object_invalid_frames'] == [11]
    assert result['geometry_status'] == 'failed'


def test_resampling_never_hides_invalid_endpoint_and_export_rejects_misalignment():
    motion = MotionData(np.zeros((3, 1, 3)), framerate=30,
                        metadata={'object_valid': np.array([True, False, True]),
                                  'source_frame_indices': np.array([5, 6, 7])})
    result = motion.resample(60)
    assert result.metadata['object_valid'].tolist() == [True, False, False, False, True]
    np.testing.assert_array_equal(result.metadata['source_frame_indices'], [5, 5.5, 6, 6.5, 7])
    with pytest.raises(ValueError, match='object_valid'):
        export_object_validity(motion.metadata, 4)


def test_pair_report_keeps_unknown_counts_without_changing_depth_statistics():
    from scripts.review_penetration_persistence import _pair_json
    pair = dict(key='moving:object:hand', kind='moving', robot_geom={'body': 'hand'},
                other='object', depths=np.array([0.0, 0.004, 0.0]))
    report, _, _ = _pair_json(
        pair, labels=np.arange(3) + 10, frames_per_second=30,
        support_index={'source': 'unavailable_no_prepared_support_evidence'},
        scene_record={}, unknown=np.array([True, False, True]),
    )
    assert report['unknown_frame_count'] == 2
    assert report['unknown_frames'] == [10, 12]
    assert report['max_depth_m'] == 0.004
    assert report['worst_frame'] == 11
    assert report['thresholds_m']['0.002']['frame_fraction'] == 1 / 3
