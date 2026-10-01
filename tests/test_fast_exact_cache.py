"""Same-candidate reuse preserves immutable FK and guard linearizations."""

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from omniretargeting._fast.kinematics import TrajectoryKinematics
from omniretargeting._fast.continuity import CorrectionContinuity
from omniretargeting._sequence.continuity_guard import SequenceContinuityGuard
from tests.test_fast_backend import _objective


def _bytes(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def _sparse(a, b):
    assert a.shape == b.shape
    for name in ('data', 'indices', 'indptr'):
        _bytes(getattr(a, name), getattr(b, name))


def test_shared_fk_reuses_values_and_jacobians_but_retains_trial_states(monkeypatch):
    objective, poses, _ = _objective(temporal=True)
    kin = TrajectoryKinematics(objective.model, objective.coords.cols)
    other = TrajectoryKinematics(objective.model, objective.coords.cols, cache=kin.cache)
    calls = []
    original = mujoco.mj_kinematics
    def counted(*args):
        calls.append(1)
        return original(*args)
    monkeypatch.setattr(mujoco, 'mj_kinematics', counted)
    kin.evaluate(poses)
    first = {k: getattr(kin, k) for k in ('xpos', 'xmat', 'jacp', 'jacr')}
    count = len(calls)
    other.evaluate(poses.copy(), False)
    other.evaluate(poses.copy(), True)
    assert len(calls) == count
    for name, value in first.items():
        assert getattr(other, name) is value
        assert not value.flags.writeable
    trial = poses.copy()
    trial[:, -1] += .01
    other.evaluate(trial)
    kin.evaluate(poses)
    assert len(calls) == count + len(poses)
    fresh = TrajectoryKinematics(objective.model, objective.coords.cols).evaluate(trial)
    for name in first:
        _bytes(getattr(other, name), getattr(fresh, name))
        _bytes(getattr(kin, name), first[name])
    # Mutating a caller's input must not mutate a cached key.
    trial[0, 0] += .001
    other.evaluate(trial)
    assert other.xpos.tobytes() != fresh.xpos.tobytes()


def test_fk_cache_distinguishes_signed_zero_and_coordinate_selection(monkeypatch):
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body><freejoint/>
      <geom size=".1"/></body></worldbody></mujoco>''')
    poses = np.tile(model.qpos0, (2, 1))
    kin = TrajectoryKinematics(model, np.arange(model.nv))
    kin.evaluate(poses)
    poses[:, 0] = -0.
    kin.evaluate(poses)
    assert len(kin.cache) == 2
    selected = TrajectoryKinematics(model, [0, 2, 4], cache=kin.cache).evaluate(poses)
    assert selected.jacp.shape[-1] == 3
    _bytes(selected.jacp, kin.jacp[..., [0, 2, 4]])


def test_continuity_linearizations_reuse_structure_and_invalidate_pose():
    objective, poses, _ = _objective(temporal=True)
    guard = CorrectionContinuity(objective.coords, poses)
    first = guard.linearizations(poses)
    assert guard.linearizations(poses.copy()) is first
    matrices = dict(guard._operators)
    moved = poses.copy()
    moved[:, -1] += np.linspace(0., .01, len(poses))
    cached = guard.linearizations(moved)
    direct = CorrectionContinuity(objective.coords, poses).linearizations(moved)
    for key in direct:
        _bytes(cached[key][0], direct[key][0])
        _sparse(cached[key][1], direct[key][1])
    assert all(guard._operators[k] is v for k, v in matrices.items())


def test_sequence_signal_cache_includes_boundary_span_and_input_bytes():
    objective, poses, _ = _objective(temporal=True)
    rotations = Rotation.from_quat(poses[:, 3:7], scalar_first=True)
    guard = SequenceContinuityGuard(objective.tasks, poses, rotations, windowed=True)
    first = guard.signals(poses, (3, 12))
    assert guard.signals(poses.copy(), (3, 12)) is first
    moved = poses.copy()
    moved[4, -1] += .01
    result = guard.signals(moved, (3, 12))
    direct = SequenceContinuityGuard(objective.tasks, poses, rotations, windowed=True).signals(moved, (3, 12))
    for key in direct:
        for a, b in zip(result[key], direct[key]):
            _bytes(a, b)
    assert guard.signals(poses, (3, 12)) is first
    assert len(guard.signals(poses, (4, 12))['joints'][0]) == 8
