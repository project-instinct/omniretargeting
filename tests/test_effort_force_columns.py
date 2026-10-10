"""Ray-moment reduction preserves complete force feasibility and effort optimum."""

import mujoco
import numpy as np
import pytest

from omniretargeting.effort_dynamics import allocate, contact_matrix
from omniretargeting.effort_force_columns import reduced_contact_matrix


@pytest.mark.parametrize("outer", [False, True])
def test_dense_patch_reduction_matches_original_cone_and_minimax(outer):
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="root"><freejoint/><geom size=".1" mass="1"/>
        <body name="foot"><joint axis="0 1 0"/><geom size=".1" mass="1"/></body>
      </body></worldbody></mujoco>''')
    data = mujoco.MjData(model); mujoco.mj_forward(model, data)
    grid = np.array([[x,y,0.] for x in np.linspace(-.1,.1,9) for y in np.linspace(-.05,.05,7)])
    contacts = [{"body": 2, "normal": [0.,0.,1.], "point": point} for point in grid]
    full = contact_matrix(model, data, contacts, .6, outer=outer)
    reduced, indices, count = reduced_contact_matrix(model, data, contacts, .6, outer=outer)
    assert len(indices) < count / 3
    np.testing.assert_allclose(reduced, full[:,indices], atol=1e-12)
    base, joints, caps = np.arange(6), np.array([6]), np.array([5.])
    for demand in (np.array([1.,2.,10.,.1,.2,0.,8.]),
                   np.array([0.,0.,-10.,0.,0.,0.,8.]),
                   np.array([0.,0.,10.,0.,4.,0.,8.])):
        before = allocate(demand, full, base, joints, caps)
        after = allocate(demand, reduced, base, joints, caps)
        assert before["status"] == after["status"]
        assert before["utilization"] == pytest.approx(after["utilization"], abs=1e-9)


def test_curved_contact_cloud_keeps_the_same_generalized_force_cone():
    from omniretargeting.effort_force_columns import force_column_indices
    from omniretargeting.effort_dynamics import friction_rays
    rng = np.random.default_rng(42)
    points = rng.normal(size=(200,3)); points /= np.linalg.norm(points, axis=1)[:,None]
    contacts = [{"body": 1, "normal": [0.,0.,1.], "point": p} for p in points]
    indices, count = force_column_indices(contacts, .6)
    rays = friction_rays([0,0,1.], .6)
    full = np.concatenate([np.r_[rays, np.cross(p[None], rays.T).T] for p in points], axis=1)
    assert len(indices) < count
    # Every omitted generator must be reproducible by nonnegative kept weights.
    from scipy.optimize import linprog
    for index in rng.choice(count, 12, replace=False):
        result = linprog(np.zeros(len(indices)), A_eq=full[:,indices], b_eq=full[:,index],
                         bounds=(0,None), method="highs")
        assert result.success
        np.testing.assert_allclose(full[:,indices] @ result.x, full[:,index], atol=1e-8)
