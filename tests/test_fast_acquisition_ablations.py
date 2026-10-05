"""Opt-in acquisition rules, historical boundaries, and unchanged default bits."""
from types import SimpleNamespace

import cvxpy as cp
import numpy as np
import pytest

from omniretargeting._fast import acquisition_guard, runtime_options, window_guards
from omniretargeting._fast.reference import ReferenceShape
from omniretargeting._fast.runtime_options import RuntimeOptions
from omniretargeting.pose_relations import RelationEnvelope, LAYOUT_FLIP_MARGIN
from tests.test_fast_contact_guard import _acquired


SWITCHES = ('abl_window_layout_deadband', 'abl_normal_slack', 'abl_normal_slack_1e5')


@pytest.mark.parametrize('name', SWITCHES)
def test_acquisition_ablation_is_independent_and_defaults_off(monkeypatch, name):
    monkeypatch.delenv('OMNIRETARGETING_FAST_DISABLE', raising=False)
    monkeypatch.setenv('OMNIRETARGETING_FAST_ENABLE', name)
    defaults = RuntimeOptions().describe()
    assert not any(defaults[key] for key in SWITCHES)
    enabled = RuntimeOptions.from_environment().describe()
    assert enabled[name]
    assert all(enabled[key] == value for key, value in defaults.items() if key != name)


class _ScalarEnvelope(RelationEnvelope):
    """Analytic one-coordinate FK, with the actual sustained/floor machinery."""
    def __init__(self, count):
        self.layout_sign = np.ones((count, 1, 1))
        for name in ('reference', 'source_components', 'whole_scale', 'part_scale',
                     'soft_scale', 'axes', 'source_axes', 'source_layout_sign'):
            setattr(self, name, None)

    def measure(self, kin, with_jacobian=False):
        value = kin.value
        return None, None, (value, np.ones((*value.shape, 1)) if with_jacobian else None)

    def layout(self, change, pairs=None):
        return change, pairs


def _layout_case(monkeypatch, count=600, enabled=True):
    monkeypatch.setattr(runtime_options, 'OPTIONS', RuntimeOptions(abl_window_layout_deadband=enabled))
    guard = ReferenceShape.__new__(ReferenceShape)
    guard.coords = SimpleNamespace(width=1, root_q=None)
    guard.envelopes = [_ScalarEnvelope(count)]
    guard.seed = np.zeros((count, 1))
    guard.acquired_poses = None
    monkeypatch.setattr(window_guards, '_shape_kin',
                        lambda guard, poses, frames, jacobian:
                        SimpleNamespace(value=poses[frames.start:frames.stop, None, :]))
    return guard, np.full((count, 1), .3)


@pytest.mark.parametrize('duration', [6, 7, 8, 127, 128, 129, 300])
def test_window_verdict_and_floor_retain_full_sustained_history(monkeypatch, duration):
    guard, poses = _layout_case(monkeypatch)
    poses[58:58 + duration] = -.3
    frames = range(64, 65)  # Seventh frame lands exactly on the 63/64 boundary.
    expected = guard.envelopes[0].violations(SimpleNamespace(value=poses[:, None, :]))
    expected[:64], expected[71:] = False, False
    actual = window_guards.guard_violation_frames(guard, poses, frames=frames)
    assert actual.tobytes() == expected.tobytes()
    assert window_guards.guard_dependencies(guard, frames, len(poses)) == range(58, 71)
    full, = guard.envelopes[0].linear_bounds(SimpleNamespace(value=poses[:, None, :]), 1)
    row, = window_guards.guard_qp_bounds(guard, poses, frames=frames)[0]
    assert row[0].toarray().tobytes() == full[0][64:65, 64:65].toarray().tobytes()
    assert row[1].tobytes() == full[1][64:65].tobytes()
    values, = window_guards.guard_qp_values(guard, poses, frames=frames)[0]
    assert values[1].tobytes() == row[1].tobytes()


def test_window_edit_can_change_six_future_verdicts_without_rejecting_unrelated_past(monkeypatch):
    guard, poses = _layout_case(monkeypatch)
    poses[:20] = -.3  # Existing unrelated collapse must not reject this window.
    poses[58:70] = -.3
    frames = range(63, 64)
    bad = window_guards.guard_violation_frames(guard, poses, frames=frames)
    assert not bad[:64].any()
    assert np.flatnonzero(bad).tolist() == list(range(64, 70))
    assert not window_guards.guard_accepts(guard, poses, frames=frames)
    poses[63] = .3  # Break the run; the subsequent six deep frames are transient.
    assert window_guards.guard_accepts(guard, poses, frames=frames)


@pytest.mark.parametrize('frames', [range(0, 1), range(593, 600), range(599, 600)])
def test_layout_history_is_clipped_at_trajectory_edges(monkeypatch, frames):
    guard, poses = _layout_case(monkeypatch)
    poses[:] = -.3
    expected = guard.envelopes[0].violations(SimpleNamespace(value=poses[:, None, :]))
    expected[:frames.start] = False
    expected[min(len(poses), frames.stop + 6):] = False
    assert window_guards.guard_violation_frames(guard, poses, frames=frames).tobytes() == expected.tobytes()


def test_deadband_screening_uses_distance_to_floor(monkeypatch):
    guard, poses = _layout_case(monkeypatch)
    frames = range(63, 64)
    poses[63] = .05
    assert not window_guards.guard_qp_bounds(guard, poses, frames=frames, box=np.array([[.15]]))[0]
    poses[63] = -.1
    row, = window_guards.guard_qp_bounds(guard, poses, frames=frames, box=np.array([[.15]]))[0]
    assert row[1][0] == -LAYOUT_FLIP_MARGIN + .1 - RelationEnvelope.SLACK / 2


def test_enabled_long_window_real_fk_and_two_references_match_full_rows(monkeypatch):
    from tests.test_fast_backend import _objective
    monkeypatch.setattr(runtime_options, 'OPTIONS', RuntimeOptions(abl_window_layout_deadband=True))
    objective, original, _ = _objective(temporal=True)
    guard = ReferenceShape(objective, original)
    acquired = original.copy()
    acquired[:, 2] += .01
    guard.freeze_reference(acquired)
    index = np.arange(600) % len(original)
    guard.seed, guard.acquired_poses = guard.seed[index], guard.acquired_poses[index]
    for envelope in guard.envelopes:
        for name in ('reference', 'source_components', 'whole_scale', 'part_scale', 'soft_scale',
                     'axes', 'source_axes', 'layout_sign', 'source_layout_sign'):
            value = getattr(envelope, name)
            if value is not None:
                setattr(envelope, name, value[index])
    poses = objective.coords.integrate(original[index],
                np.random.default_rng(92).normal(scale=.2, size=(600, objective.width)), 1.)
    frames = range(63, 75)
    linear, cones = window_guards.guard_qp_bounds(guard, poses, frames=frames)
    guard.kin.evaluate(poses)
    for envelope, row in zip(guard.envelopes, linear):
        full, = envelope.linear_bounds(guard.kin, objective.width)
        keep = envelope.layout_sign.ravel() != 0
        row_frames = np.broadcast_to(np.arange(600)[:, None, None], envelope.layout_sign.shape).ravel()[keep]
        selected = (row_frames >= frames.start) & (row_frames < frames.stop)
        matrix = full[0][selected, frames.start * objective.width:frames.stop * objective.width]
        assert matrix.toarray().tobytes() == row[0].toarray().tobytes()
        assert full[1][selected].tobytes() == row[1].tobytes()
    assert len(cones) == 2 and all(row[0].shape[1] == len(frames) * objective.width for row in cones)
    expected = guard.violation_frames(poses)
    expected[:63], expected[81:] = False, False
    assert window_guards.guard_violation_frames(guard, poses, frames=frames).tobytes() == expected.tobytes()


@pytest.mark.parametrize('count,enabled', [(600, False), (599, True), (36, True)])
def test_legacy_window_rows_verdicts_and_proposal_output_are_bit_identical(monkeypatch, count, enabled):
    guard, poses = _layout_case(monkeypatch, count=count, enabled=enabled)
    frames = range(10, 18)
    poses[4:15] = -.3
    poses[15:18, 0] = [-.1, .05, .3]
    row, = window_guards.guard_qp_bounds(guard, poses, frames=frames)[0]
    value = poses[10:18, 0]
    old_lower = -np.maximum(value, 0.) - RelationEnvelope.SLACK / 2
    assert row[1].tobytes() == old_lower.tobytes()
    assert row[0].toarray().tobytes() == np.eye(8).tobytes()
    assert window_guards.guard_dependencies(guard, frames, count) == frames
    old_bad = np.zeros(count, bool)
    old_bad[10:18] = guard.envelopes[0].for_frames(slice(10, 18)).violations(
        SimpleNamespace(value=poses[10:18, None, :]))
    assert window_guards.guard_violation_frames(guard, poses, frames=frames).tobytes() == old_bad.tobytes()
    old_values = -value - RelationEnvelope.SLACK / 2
    assert window_guards.guard_qp_values(guard, poses, frames=frames)[0][0][1].tobytes() == old_values.tobytes()
    # Solve the same small proposal with old vs current guard rows; compare poses.
    proposals = []
    for lower in (old_lower, row[1]):
        step = cp.Variable(8)
        problem = cp.Problem(cp.Minimize(cp.sum_squares(step + .2)), [step >= lower])
        problem.solve(solver='CLARABEL', tol_gap_abs=1e-8, tol_feas=1e-8, tol_gap_rel=1e-8)
        proposals.append(value + step.value)
    assert proposals[0].tobytes() == proposals[1].tobytes()


def _normal_case(monkeypatch, surfaces=False, established=True):
    objective, poses = _acquired(surfaces=surfaces)
    objective.tasks.contact_phase = None
    for eff in objective.effectors:
        eff['normal'][:] = [0., 0., 1.]
        if surfaces:
            eff['active'][:] = False
    for surf in objective.surfaces:
        surf['normals'][:] = [0., 0., 1.]
    monkeypatch.setattr(acquisition_guard, 'on_surface',
                        lambda obj, points: np.full(points.shape[:2], established))
    monkeypatch.setattr(acquisition_guard, 'body_penetration',
                        lambda obj, poses: np.zeros((len(poses), obj.model.nbody)))
    return acquisition_guard.AcquisitionNormalGuard(objective, poses), poses


@pytest.mark.parametrize('flag,slack', [(None, 1e-7), ('abl_normal_slack', 1e-6),
                                      ('abl_normal_slack_1e5', 1e-5)])
@pytest.mark.parametrize('surfaces', [False, True])
@pytest.mark.parametrize('established', [False, True])
def test_normal_slack_whole_window_diagnostics_and_rows_agree(monkeypatch, flag, slack, surfaces, established):
    monkeypatch.setattr(runtime_options, 'OPTIONS', RuntimeOptions(**({flag: True} if flag else {})))
    guard, poses = _normal_case(monkeypatch, surfaces, established)
    frames = range(4, 8)
    for excess, accepted in ((slack * .9, True), (slack * 1.1, False)):
        trial = poses.copy()
        trial[:, 2] += .001 + excess
        assert guard.accepts(trial) is accepted
        assert bool(not guard.violation_frames(trial).any()) is accepted
        assert guard.accepts(trial, frames=frames) is accepted
        assert guard.diagnostics(trial)['accepted'] is accepted
    assert guard.numerical_slack == slack
    whole, cones = guard.repair_bounds(poses)
    local, local_cones = guard.qp_bounds(poses, frames=frames)
    if established:
        assert np.all(whole[0][2] == .001 + slack / 2)
        assert np.all(local[0][2] == .001 + slack / 2)
    else:
        assert np.all(cones[0][2] == .001 + slack / 2)
        assert np.all(local_cones[0][2] == .001 + slack / 2)
    invalid = poses.copy()
    invalid[4, 0] = np.nan
    assert not guard.diagnostics(invalid)['accepted']


def test_normal_modes_do_not_relax_other_families_and_larger_mode_wins(monkeypatch):
    from omniretargeting._fast import contact_guard, continuity
    from omniretargeting._sequence.continuity_guard import ACCEPTANCE_SLACK
    from omniretargeting._fast.priority_window_contacts import frozen_violations
    monkeypatch.setattr(runtime_options, 'OPTIONS', RuntimeOptions(abl_normal_slack=True,
                                                                 abl_normal_slack_1e5=True))
    guard, poses = _normal_case(monkeypatch)
    assert guard.numerical_slack == 1e-5
    assert contact_guard.NUMERICAL_SLACK == 1e-7
    assert continuity.ROUND_OFF == ACCEPTANCE_SLACK == 2e-7
    block = SimpleNamespace(values=np.array([5e-7]), bound=np.zeros(1), own=np.zeros(1), frames=np.array([4]))
    adapter = SimpleNamespace(evaluate=lambda *a, **kw: [block])
    assert frozen_violations(adapter, poses, range(4, 5))[4]
