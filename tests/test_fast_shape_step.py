"""Shape steps: only the layout and guards bound a step; no angular budget."""

import mujoco
import numpy as np

from omniretargeting._fast.objective import FastObjective
from omniretargeting._fast.reference import ReferenceShape
from omniretargeting._fast.shape_step import restore_shape_step
from omniretargeting.sequence_retargeting import prepare_sequence, sequence_config


def test_a_large_posture_step_is_kept_without_an_angular_budget():
    model = mujoco.MjModel.from_xml_string('''
    <mujoco><compiler angle="radian"/><worldbody>
      <body name="root"><freejoint/><geom size=".01"/>
        <body name="lever"><joint axis="0 1 0" range="-2 2"/><geom size=".01"/>
          <body name="tip" pos="0 0 1"><geom size=".01"/></body>
        </body>
      </body>
    </worldbody></mujoco>''')
    reference = np.tile(model.qpos0, (5, 1))
    source = np.tile([[[0., 0, 0], [0, 0, 1]]], (len(reference), 1, 1))
    cfg = sequence_config({'self_collision_enabled': False})
    coords, _, tasks, rotations = prepare_sequence(
        model, reference, source, ['root', 'tip'], {'root': 'root', 'tip': 'tip'},
        source[:, 0], {}, {}, cfg,
    )
    objective = FastObjective(tasks, cfg, reference=reference,
                              temporal_reference=reference[:, coords.qi],
                              root_rotations=rotations, framerate=30.)
    shape = objective.source_shape = ReferenceShape(objective, reference)
    shape.freeze_reference(reference)
    origin = reference.copy()
    # One lever relation turned by .12 rad reads as ~.15 rad (random axes).
    origin[:, -1] = .12
    assert shape.accepts(origin)
    step = np.zeros((len(origin), coords.width))
    step[:, coords.scalar_cols] = .04
    step[:, coords.inverse[coords.root_d + 3]] = .08
    proposal = coords.integrate(origin, step, 1.)
    # Beyond the former whole-body and part angles: nothing to restore.
    assert shape.accepts(proposal)
    restored = restore_shape_step(objective, origin, step.ravel())
    np.testing.assert_allclose(restored, objective.project(proposal))
    np.testing.assert_array_equal(shape.seed, reference)


def test_identical_guard_rows_merge_without_changing_the_optimum():
    import cvxpy as cp
    import pytest

    from omniretargeting._fast.continuity import CorrectionContinuity
    from omniretargeting._fast.shape_step import _guard_constraints
    from omniretargeting._sequence.continuity_guard import SequenceContinuityGuard
    from tests.test_fast_backend import _objective

    objective, poses, (task_set, _, _) = _objective(temporal=True)
    coords = objective.coords
    # Both guards bound the same scalar-joint stencils of one fixed reference.
    guards = [SequenceContinuityGuard(task_set, poses, objective.root_rotations),
              CorrectionContinuity(coords, poses)]
    moved = poses.copy()  # linearized at the reference: zero is feasible
    target = np.random.default_rng(5).normal(size=len(poses) * coords.width) * .05

    def solve(merged):
        flat = cp.Variable(len(target))
        constraints = [cp.abs(flat) <= .05]
        if merged:
            constraints += _guard_constraints(guards, moved, flat)
        else:
            for guard in guards:
                constraints += guard.constraints(moved, flat)
        problem = cp.Problem(cp.Minimize(cp.sum_squares(flat - target)), constraints)
        problem.solve(solver=cp.CLARABEL, tol_gap_abs=1e-12, tol_gap_rel=1e-12, tol_feas=1e-12, max_iter=500)
        rows = sum(c.size for c in problem.constraints)
        return flat.value, problem.value, rows

    separate, separate_value, separate_rows = solve(False)
    merged, merged_value, merged_rows = solve(True)
    assert np.isfinite(separate_value) and separate_value > 1e-6
    assert merged_rows < separate_rows
    assert merged_value == pytest.approx(separate_value, rel=1e-8)
    np.testing.assert_allclose(merged, separate, atol=1e-6)
