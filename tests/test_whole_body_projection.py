"""Physical-pair policy and fixed-source continuity of simultaneous refinement."""
from types import SimpleNamespace

import numpy as np
import pytest

from omniretargeting._fast.whole_body_projection import Projection, quality, rank
from tests.test_fast_backend import _objective


def test_query_representations_merge_before_pair_persistence_policy():
    depths = {(f, 'moving', 0, 'observed_open_surface', 0, 3): .007 for f in range(12)}
    depths.update({(f, 'moving', 0, 'reverse_surface', -1, 3): .004 for f in range(12)})
    result = quality(dict(depths=depths), 30, 30.)
    assert result['summary']['pair_count'] == 1
    assert result['summary']['classification_pair_counts']['persistent'] == 1
    pair = next(iter(result['pairs'].values()))
    assert pair['stats']['thresholds_m']['0.005']['longest_run_seconds'] == pytest.approx(.4)


def test_quality_can_trade_a_shallow_pair_for_a_deep_pair_without_history_locks():
    first = dict(depths={(0, 'self', 1, 2): .03, (0, 'static', 3): .0002})
    second = dict(depths={(0, 'self', 1, 2): .008, (0, 'static', 3): .0018})
    assert rank(second, 30, 30.) < rank(first, 30, 30.)
    # An unknown surrogate cannot disappear from the optimization quality.
    unknown = dict(depths={(0, 'moving', 0, 'unresolved_material', 0, 3): .025})
    assert rank(unknown, 30, 30.)[0] == pytest.approx(.015)


def test_projection_preserves_source_and_rejects_new_single_frame_chatter():
    obj, poses, _ = _objective(temporal=True)
    obj.source_shape = SimpleNamespace(accepts=lambda q: True)
    source = obj.reference.copy()
    projector = Projection(obj, poses)
    assert projector.rejection(poses) is None
    trial = poses.copy()
    trial[len(trial) // 2, obj.coords.qi[-1]] += .1
    reject = projector.rejection(trial)
    assert reject['kind'] == 'added_chatter'
    np.testing.assert_array_equal(obj.reference, source)


def test_direction_has_root_and_compensating_joints_and_is_temporally_smooth():
    obj, poses, _ = _objective(temporal=True)
    obj.source_shape = SimpleNamespace(accepts=lambda q: True)
    projector = Projection(obj, poses)
    root_col = obj.coords.inverse[obj.coords.root_d + 1]
    joint_col = obj.coords.scalar_cols[-1]
    from scipy import sparse
    jac = np.zeros((len(poses), len(poses) * obj.width))
    for frame in range(len(poses)):
        jac[frame, frame * obj.width + root_col] = 1.
        jac[frame, frame * obj.width + joint_col] = .1
    geometry = SimpleNamespace(linear_rows=lambda m: (sparse.csr_matrix(jac), np.full(len(poses), -.02)))
    # No source contacts here: expose the coupled minimum-correction response.
    obj.surfaces, obj.effectors = [], []
    step = projector.step(poses, geometry, dict(rows=[]))
    assert np.all(step[:, root_col] > 0)
    assert np.all(step[:, joint_col] > 0)
    assert np.max(np.abs(np.diff(step, n=2, axis=0))) < 1e-5


def _legacy_inverse_forces(model, poses, fps):
    # The screen keeps the pre-effort-audit execution stencil (backward
    # differences, raw mj_inverse); opt/screen measured the corrected stencil
    # as a wipe300 regression, so the screen is pinned to this definition.
    import mujoco
    velocity = np.empty((len(poses) - 1, model.nv))
    for frame in range(len(poses) - 1):
        mujoco.mj_differentiatePos(model, velocity[frame], 1. / fps, poses[frame], poses[frame + 1])
    acceleration = np.diff(velocity, axis=0) * fps
    data, force = mujoco.MjData(model), []
    for frame in range(len(poses) - 2):
        data.qpos[:], data.qvel[:], data.qacc[:] = poses[frame + 2], velocity[frame + 1], acceleration[frame]
        mujoco.mj_inverse(model, data)
        force.append(data.qfrc_inverse.copy())
    return np.array(force)


def test_inverse_force_screen_keeps_its_execution_stencil():
    from omniretargeting._fast.projection_mechanics import inverse_forces, MechanicalScreen
    obj, poses, _ = _objective(temporal=True)
    expected = _legacy_inverse_forces(obj.model, poses, obj.framerate)
    np.testing.assert_array_equal(inverse_forces(obj.model, poses, obj.framerate), expected)
    joint = obj.model.dof_jntid[obj.coords.di[-1]]
    obj.model.jnt_actfrclimited[joint] = True
    obj.model.jnt_actfrcrange[joint] = [-.01, .01]
    screen = MechanicalScreen(obj, poses)
    assert screen.rejection(poses) is None
    candidate = poses.copy()
    candidate[10, obj.coords.qi[-1]] += .2
    reject = screen.rejection(candidate)
    assert reject['kind'] == 'effort_screen'
    assert reject['required'] > reject['allowed']


def test_mechanical_stencil_jacobian_matches_perturbed_execution():
    from omniretargeting._fast.projection_mechanics import MechanicalScreen, mechanical_rows, inverse_forces
    obj, poses, _ = _objective(temporal=True)
    joint = obj.model.dof_jntid[obj.coords.di[-1]]
    obj.model.jnt_actfrclimited[joint] = True
    obj.model.jnt_actfrcrange[joint] = [-10., 10.]
    screen = MechanicalScreen(obj, poses)
    jac, lower, upper = mechanical_rows(screen, obj.coords, poses)
    reserve = np.tile(screen.limits * 1e-4, len(poses) - 2)
    assert (lower - reserve <= 1e-9).all() and (upper + reserve >= -1e-9).all()
    assert screen.rejection(poses) is None  # Exact acceptance cap is unchanged.
    rng = np.random.default_rng(14)
    direction = rng.normal(size=(len(poses), obj.width)) * .01
    eps = 1e-5
    actual = []
    for sign in (1., -1.):
        q = obj.coords.integrate(poses, direction, sign * eps)
        actual.append(inverse_forces(obj.model, q, obj.framerate)[:, screen.dofs].ravel())
    np.testing.assert_allclose(jac @ direction.ravel(), (actual[0] - actual[1]) / (2 * eps),
                               atol=1e-3, rtol=2e-3)


def test_temporal_focus_keeps_halo_and_merges_coupled_windows():
    from omniretargeting._fast.projection_windows import focus_frames, intervals
    mask = np.zeros(200, bool)
    mask[[30, 60, 160]] = True
    selected = focus_frames({'pairs': {0: {'events': [{'frames': mask}]}}}, 200)
    assert selected[18:43].all() and not selected[17]
    assert intervals(selected) == [[18, 73], [148, 173]]
