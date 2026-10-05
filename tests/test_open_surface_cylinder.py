"""Exact solid-cylinder / finite-triangle queries on open scans."""

import mujoco
import numpy as np
import pytest
import trimesh
from scipy.optimize import minimize

from omniretargeting.geometry_query import RigidGeometryQuery, relative_point_jacobian
from omniretargeting.scene_geometry import SceneQuery
from omniretargeting.triangle_cylinder import _support, cylinder_triangle_rows

FLOOR = np.array([[[-2., -2., 0.], [2., -2., 0.], [0., 2., 0.]]])


def test_standing_and_lying_cylinders_over_a_floor_are_exact():
    standing = cylinder_triangle_rows([0, 0, .13], [0, 0, 1], .05, .1, FLOOR)
    assert standing["clearance"][0] == pytest.approx(.03, abs=1e-12)
    np.testing.assert_allclose(standing["direction"][0], [0, 0, 1], atol=1e-12)
    assert standing["gap"][0] < 1e-12
    lying = cylinder_triangle_rows([0, 0, .046], [1, 0, 0], .05, .1, FLOOR)
    assert lying["clearance"][0] == pytest.approx(-.004, abs=1e-12)
    np.testing.assert_allclose(lying["direction"][0], [0, 0, 1], atol=1e-12)
    assert not lying["ambiguous"][0]
    # Its support face is the whole bottom line: the witness lies on it.
    assert lying["point"][0][2] == pytest.approx(-.004, abs=1e-12)
    assert abs(lying["point"][0][0]) <= .1 + 1e-12


def test_symmetric_crossing_keeps_its_depth_without_choosing_an_exit():
    crossing = cylinder_triangle_rows([0, 0, 0], [0, 0, 1], .05, .2, FLOOR)
    assert crossing["clearance"][0] == pytest.approx(-.2)
    assert crossing["ambiguous"][0]


def test_crossing_near_a_finite_edge_exits_sideways():
    triangle = np.array([[[-.5, -.5, 0.], [.5, -.5, 0.], [.5, .5, 0.]]])
    near_edge = cylinder_triangle_rows([.49, 0, 0], [0, 0, 1], .05, .2, triangle)
    assert near_edge["clearance"][0] == pytest.approx(-.06)
    np.testing.assert_allclose(near_edge["direction"][0], [1, 0, 0], atol=1e-9)


def _reference(center, axis, radius, half, triangle):
    """Global minimum of the configuration-obstacle support function."""
    count = 20000
    i = np.arange(count) + .5
    polar, azimuth = np.arccos(1 - 2 * i / count), np.pi * (1 + 5**.5) * i
    sphere = np.c_[np.cos(azimuth) * np.sin(polar), np.sin(azimuth) * np.sin(polar), np.cos(polar)]
    values = _support(center, axis, radius, half, triangle, sphere[None])[0]
    best = values.min()
    for start in sphere[np.argsort(values)[:8]]:
        result = minimize(lambda x: _support(center, axis, radius, half, triangle,
                                             (x / np.linalg.norm(x))[None, None])[0, 0],
                          start, method="Nelder-Mead",
                          options=dict(xatol=1e-12, fatol=1e-14, maxiter=4000))
        best = min(best, result.fun)
    return -best


def test_random_pairs_match_the_global_support_minimum_and_certify_witnesses():
    rng = np.random.default_rng(7)
    for _ in range(24):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        center, radius, half = rng.normal(size=3) * .05, rng.uniform(.008, .05), rng.uniform(.02, .1)
        triangle = (rng.normal(size=(1, 3, 3)) * rng.uniform(.03, .3)
                    + rng.normal(size=3) * rng.uniform(0, .12))
        measured = cylinder_triangle_rows(center, axis, radius, half, triangle)
        clearance = measured["clearance"][0]
        reference = _reference(center, axis, radius, half, triangle)
        # Never above the true clearance; equal within roundoff.
        assert clearance <= reference + 1e-12
        assert clearance == pytest.approx(reference, abs=2e-9)
        if clearance >= 0:
            assert measured["gap"][0] < 1e-7
            assert np.linalg.norm(measured["point"][0] - measured["surface"][0]) == pytest.approx(
                clearance, abs=1e-7)


def _hinged_cylinder():
    return mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body pos="0 0 .3"><freejoint/>
      <inertial pos="0 0 0" mass="1" diaginertia=".1 .1 .1"/>
      <body><joint axis="0 1 0"/><geom type="cylinder" pos=".2 0 0" euler="0 80 0" size=".05 .2"/>
      </body></body></worldbody></mujoco>''')


def test_cylinder_rows_replace_the_unsupported_skip_and_match_finite_differences():
    model = _hinged_cylinder()
    scene = trimesh.Trimesh(FLOOR[0] + [0, 0, .29], [[0, 1, 2]], process=False)
    query = RigidGeometryQuery(model, [0], SceneQuery(scene))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    row = query.query(data, .3)[0]
    assert not row["uncertain"]
    jacobian = relative_point_jacobian(model, data, row["points"], row["body_ids"], row["directions"])
    for dof in range(model.nv):
        values = []
        for sign in (-1, 1):
            data.qpos[:] = model.qpos0
            velocity = np.zeros(model.nv)
            velocity[dof] = sign
            mujoco.mj_integratePos(model, data.qpos, velocity, 1e-6)
            mujoco.mj_forward(model, data)
            values.append(query.query(data, .3)[0]["clearance"])
        assert jacobian[dof] == pytest.approx((values[1] - values[0]) / 2e-6, abs=1e-6)
    assert query.describe()["uncertified_cylinder_witnesses"] == 0
