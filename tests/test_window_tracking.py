"""Collision proxies yield only to the explicitly validated window owner."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from omniretargeting._fast import runtime_options
from omniretargeting._fast.window_tracking import skips_proxy_collision
from omniretargeting._fast.sdf import GridSDF
from omniretargeting.self_collision import SelfCollisionQuery
from tests.test_fast_backend import _objective


class Owner:
    owns_clearance = True

    def __init__(self, poses):
        self.shape = poses.shape
        self.calls = []

    def collision_diagnostics(self, poses, *, frames=None):
        assert poses.shape == self.shape, 'Diagnostics received cropped qpos'
        self.calls.append(frames)
        return dict(scene_depth_m=.003, moving_depth_m=None, self_depth_m=.002,
                    collision_diagnostics_reliable=False,
                    collision_diagnostics_provenance='original_geometry_test_provider')


def _same_rows(actual, expected):
    assert actual[0].cost == expected[0].cost
    np.testing.assert_array_equal(actual[0].frame_cost, expected[0].frame_cost)
    np.testing.assert_array_equal(actual[0].gradient, expected[0].gradient)
    np.testing.assert_array_equal(actual[0].hessian, expected[0].hessian)
    np.testing.assert_array_equal(actual[1], expected[1])
    np.testing.assert_array_equal(actual[2].toarray(), expected[2].toarray())


@pytest.mark.parametrize('windowed', [False, True])
@pytest.mark.parametrize('owned', [False, True, None])
def test_gate_is_fail_closed_and_other_paths_are_bit_identical(monkeypatch, windowed, owned):
    obj, poses, _ = _objective(temporal=True, extra=True)
    monkeypatch.setattr(runtime_options, 'OPTIONS', replace(runtime_options.OPTIONS, windowed_priority=False))
    baseline = obj.evaluate(poses)
    obj.priority_clearance = SimpleNamespace(owns_clearance=owned)
    monkeypatch.setattr(runtime_options, 'OPTIONS', replace(runtime_options.OPTIONS, windowed_priority=windowed))
    assert skips_proxy_collision(obj) == (windowed and owned is True)
    if not skips_proxy_collision(obj):
        actual = obj.evaluate(poses)
        _same_rows(actual, baseline)
        assert actual[3] == baseline[3]


@pytest.mark.parametrize('frames', [None, range(0, 32), range(4, 36)])
def test_gate_omits_exactly_self_and_scene_object_rows_and_merges_full_pose_diagnostics(monkeypatch, frames):
    obj, poses, _ = _objective(temporal=True, extra=True)
    obj.evaluation_backend = 'cpu'
    changed = obj.coords.integrate(poses, np.random.default_rng(5).normal(scale=.01, size=(len(poses), obj.width)), 1.)
    expected = obj.evaluate(changed, frames=frames)
    values = np.full((21, 21, 21), -.03, np.float32)
    field = GridSDF([-1., -1., -1.], .15, values, np.zeros((*values.shape, 3), np.float32))
    obj.scene_sdf = field
    obj.objects = [dict(sdf=field, rotations=np.tile(np.eye(3), (obj.frames, 1, 1)),
                        translations=np.zeros((obj.frames, 3)), scales=None)]
    obj.self_collision = SelfCollisionQuery(obj.model)
    owner = obj.priority_clearance = Owner(poses)
    monkeypatch.setattr(runtime_options, 'OPTIONS', replace(runtime_options.OPTIONS, windowed_priority=True))
    def forbidden(*args, **kwargs):
        raise AssertionError('Skipped proxy was queried')
    obj._deepest_samples = forbidden
    obj.self_collision.query = forbidden
    actual = obj.evaluate(changed, frames=frames)
    _same_rows(actual, expected)
    assert owner.calls == [frames]
    assert actual[3]['proxy_collision_terms_skipped'] is True
    assert actual[3]['clearance_diagnostics_owner'] == 'priority_clearance_original_geometry'
    assert actual[3]['scene_depth_m'] == .003
    assert actual[3]['self_depth_m'] == .002
    assert actual[3]['moving_depth_m'] is None
    assert actual[3]['collision_diagnostics_reliable'] is False
    assert obj.stats == actual[3]


def test_stage_a_never_skips_without_a_coordinator(monkeypatch):
    obj, poses, _ = _objective(temporal=False)
    baseline = obj.evaluate(poses)
    monkeypatch.setattr(runtime_options, 'OPTIONS', replace(runtime_options.OPTIONS, windowed_priority=True))
    assert not skips_proxy_collision(obj)
    actual = obj.evaluate(poses)
    _same_rows(actual, baseline)
    assert actual[3] == baseline[3]


def test_full_gpu_reference_path_skips_shell_prepass(monkeypatch):
    torch = pytest.importorskip('torch')
    pytest.importorskip('triton')
    if not torch.cuda.is_available():
        pytest.skip('CUDA is optional, used here only by the parent full evaluator')
    obj, poses, _ = _objective(temporal=True, extra=True)
    expected = obj.evaluate(poses)
    values = np.full((21, 21, 21), -.03, np.float32)
    obj.scene_sdf = GridSDF([-1., -1., -1.], .15, values, np.zeros((*values.shape, 3), np.float32))
    obj.priority_clearance = Owner(poses)
    obj.evaluation_backend = 'gpu'
    evaluator = obj._gpu_evaluator()
    def forbidden(*args, **kwargs):
        raise AssertionError('GPU shell prepass must also be omitted')
    monkeypatch.setattr(evaluator.shell, 'deepest', forbidden)
    monkeypatch.setattr(runtime_options, 'OPTIONS', replace(runtime_options.OPTIONS, windowed_priority=True))
    actual = obj.evaluate(poses)
    _same_rows(actual, expected)
    assert actual[3]['proxy_collision_terms_skipped'] is True
