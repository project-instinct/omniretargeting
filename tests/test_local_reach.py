import mujoco
import numpy as np

from omniretargeting._fast.local_reach import geom_reach_bound, unreachable_geoms


def test_local_reach_bounds_finite_joint_motion_and_ignores_fixed_root():
    model = mujoco.MjModel.from_xml_string('''<mujoco><compiler angle="radian"/><worldbody>
      <body><freejoint/><geom size=".01"/>
        <body pos=".1 .02 0"><joint axis="0 0 1" pos=".02 0 0"/>
          <geom size=".01"/>
          <body pos=".2 0 .04"><joint axis="0 1 0"/>
            <geom name="tip" type="box" pos=".07 0 0" size=".03 .02 .01"/>
          </body></body></body></worldbody></mujoco>''')
    poses = model.qpos0[None].copy(); g = model.geom('tip').id
    bound = geom_reach_bound(model, g, poses, .05, .003)
    data = mujoco.MjData(model); data.qpos[:] = poses[0]; mujoco.mj_forward(model, data)
    local = np.array([[a,b,c] for a in [-.03,.03] for b in [-.02,.02] for c in [-.01,.01]])
    before = local @ data.geom_xmat[g].reshape(3,3).T + data.geom_xpos[g]
    for first in [-.05,.05]:
      for second in [-.05,.05]:
        data.qpos[7:] = [first,second]; mujoco.mj_forward(model,data)
        points = local @ data.geom_xmat[g].reshape(3,3).T + data.geom_xpos[g]
        assert np.linalg.norm(points-before,axis=1).max() <= bound
    measured = {'depths': {(0,'moving',0,'reverse_surface',-1,g): bound+.02,
                           (0,'moving',0,'observed_open_surface',0,0): 100.}}
    conflicts = unreachable_geoms(model,poses,measured,angle=.05,slide=.003)
    assert len(conflicts)==1 and conflicts[0]['geom_id']==g
    assert abs(conflicts[0]['residual_lower_bound_m']-.02)<1e-12
