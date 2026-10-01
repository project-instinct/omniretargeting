"""Collision max-depth proposals must be independent of sampling density."""
from types import SimpleNamespace

import numpy as np
from scipy import sparse

from omniretargeting._fast.repair_limits import LinearBounds
from omniretargeting._fast.repair_step import solve_step


def _solve(jac, phi, *, cap=None, projection=None, keys=None, shift=0.):
    jac = np.asarray(jac, float)
    width = jac.shape[1]
    poses = np.zeros((1, width))
    keys = [(0, "static", 1)] * len(phi) if keys is None else keys
    measurement = {"rows": list(zip(keys, ({"clearance": p} for p in phi), strict=True)),
                   "depths": {key: max(0., -min(p for k, p in zip(keys, phi, strict=True) if k == key))
                              for key in keys}}

    def constraints(_):
        result = LinearBounds(width)
        result.add(sparse.eye(width), -.1, .1)
        if cap is not None:
            result.norm(np.eye(width)[None], poses.copy(), cap)
        return result

    limits = SimpleNamespace(frames=1, width=width, entry=poses.copy() - shift,
                             coords=SimpleNamespace(qi=np.arange(width)),
                             movable_body=None, constraints=constraints)
    geometry = SimpleNamespace(linear_rows=lambda _: (sparse.csr_matrix(jac), np.asarray(phi)))
    return solve_step(limits, poses, geometry, measurement, measurement,
                      sparse.eye(width, format="csc") if projection is None else projection,
                      collision_sigma=1., local_angle=1.)


def test_flat_patch_lifts_without_tilting_other_corner_inward():
    # z + theta and z - theta: both corners must leave a flat plane.
    step, status = _solve([[1, 1], [1, -1]], [-.01, -.01])
    assert status in {"Solved", "AlmostSolved"}
    np.testing.assert_allclose(step, [[.005, 0]], atol=1e-7)
    # Without a lifting DOF, neither pitching direction is a repair.
    step, _ = _solve([[1], [-1]], [-.01, -.01])
    np.testing.assert_allclose(step, 0., atol=1e-8)


def test_duplicating_one_corner_does_not_change_collision_weight():
    original, _ = _solve([[1, 1], [1, -1]], [-.01, -.01])
    repeated, _ = _solve([[1, 1]] * 30 + [[1, -1]], [-.01] * 31)
    np.testing.assert_allclose(repeated, original, atol=1e-7)


def test_epigraph_cannot_spend_contact_norm_budget():
    step, _ = _solve([[1, 1], [1, -1]], [-.01, -.01], cap=.001)
    assert np.linalg.norm(step) <= .001 + 1.1e-7
    assert step[0, 0] > .00099
    assert abs(step[0, 1]) < 1e-8


def test_no_penetration_and_no_movable_joints_do_not_invent_correction():
    assert _solve([[1]], [.005])[1] == 'no_active_collision'
    assert _solve([[1]], [-.01], projection=sparse.csc_matrix((1, 0)))[1] == 'no_local_dofs'


def test_unilateral_depth_does_not_reward_overseparation():
    # Opposite side is already clear. It must not act like a signed spring.
    step, _ = _solve([[1], [-1]], [-.01, .08])
    np.testing.assert_allclose(step, [[.005]], atol=1e-7)


def test_uncontrollable_collision_keeps_residual_without_motion():
    step, status = _solve([[0.]], [-.01])
    assert status in {"Solved", "AlmostSolved"}
    np.testing.assert_allclose(step, 0., atol=1e-9)


def test_second_collision_group_cannot_be_worsened():
    step, _ = _solve([[1.], [-1.]], [-.01, 0.], keys=[(0, "static", 1), (0, "static", 2)])
    np.testing.assert_allclose(step, 0., atol=1e-8)


def test_joint_cost_uses_immutable_entry_not_latest_iterate():
    step, _ = _solve([[1.]], [-.01], shift=.002)
    np.testing.assert_allclose(step, [[.004]], atol=1e-7)
