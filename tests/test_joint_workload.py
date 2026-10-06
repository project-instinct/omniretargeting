"""Signed repair transfer ratios preserve worsening and undefined denominators."""
import json

from scripts.review_joint_workload import ratios, verified_reset


def _stage(depth):
    return dict(summary={k: dict(max_depth_m=depth, max_excess_m=depth) for k in ('static', 'moving', 'self')},
                per_body={k: {'owner': dict(max_depth_m=depth, max_excess_m=depth, reliable=True)}
                          for k in ('static', 'moving', 'self')})


def test_ratio_is_signed_unclipped_and_per_body():
    a = dict(before_repair=_stage(.01), final=_stage(.008))
    e = dict(after_main=_stage(.005))
    result = ratios(a, e)
    assert abs(result['static']['clip']['max_depth_m']['ratio'] - 2.5) < 1e-12
    assert abs(result['static']['per_body']['owner']['max_depth_m']['ratio'] - 2.5) < 1e-12
    e['after_main'] = _stage(.011)
    assert result['moving']['clip']['max_excess_m']['ratio'] > 1
    assert ratios(a, e)['moving']['clip']['max_depth_m']['ratio'] < 0


def test_zero_denominator_and_unknown_never_certify_transfer():
    a = dict(before_repair=_stage(.01), final=_stage(.01))
    e = dict(after_main=_stage(0))
    assert ratios(a, e)['self']['clip']['max_depth_m']['ratio'] is None
    a['final'] = _stage(.005)
    e['after_main']['per_body']['static']['owner']['reliable'] = False
    assert ratios(a, e)['static']['per_body']['owner']['max_excess_m']['ratio'] is None


def test_unknown_frames_are_excluded_from_confirmed_reset_counts(tmp_path):
    report = dict(inputs={'input_frames': 3}, reset={'failed_frames': [0], 'unknown_geometry_frames': [0, 1]})
    (tmp_path / 'training_collision.json').write_text(json.dumps(report))
    result = verified_reset(tmp_path)
    assert result['threshold_only_frames'] == 2
    assert result['confirmed_geometry_frames'] == 1
    assert result['unknown_frames'] == 2
