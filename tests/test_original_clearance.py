from types import SimpleNamespace
import numpy as np

from omniretargeting._fast.mesh_repair import OriginalGeometry
from omniretargeting._fast.original_clearance import OriginalClearance,replace_deeper_original
from tests.test_fast_backend import _objective
from tests.test_fast_mesh_repair import _open_object


def test_original_rows_use_global_prescribed_time_when_chunked():
    objective,poses,_=_objective(temporal=True)
    obj=_open_object(objective,poses)
    obj['translations'][:,0]+=np.linspace(0,.01,len(poses))
    geometry=OriginalGeometry(objective,[obj],None)
    g=objective.model.geom('hand_geom').id
    original=OriginalClearance(geometry,[g])
    whole=original.evaluate(poses)
    assert whole and all(key[-1]==g for key in whole)
    for first,last in [(0,11),(11,len(poses))]:
        part=original.evaluate(poses[first:last],first)
        assert len(part)==sum(first<=key[1]<last for key in whole)
        for (field,frame,geom),row in part.items():
            other=whole[field,frame+first,geom]
            for name in ('point','direction','clearance'):
                np.testing.assert_array_equal(row[name],other[name])


def test_static_batches_preserve_moving_object_global_chunk_times():
    class Plane:
        def query(self, points):
            return dict(clearance=points[:, 2] + 10, uncertain=np.zeros(len(points), bool),
                        directions=np.tile([0., 0., 1.], (len(points), 1)))
    objective, poses, _ = _objective(temporal=True)
    obj = _open_object(objective, poses)
    obj['translations'][:, 0] += np.linspace(0, .02, len(poses))
    geom = objective.model.geom('hand_geom').id
    original = OriginalClearance(OriginalGeometry(objective, [obj], Plane()), [geom])
    whole = original.evaluate(poses)
    assert any(key[0] == 1 for key in whole)
    for first, last in [(0, 11), (11, len(poses))]:
        part = original.evaluate(poses[first:last], first)
        assert len(part) == sum(first <= key[1] < last for key in whole)
        for (field, frame, g), row in part.items():
            other = whole[field, frame + first, g]
            for name in ('point', 'direction', 'clearance'):
                np.testing.assert_array_equal(row[name], other[name])


def test_more_restrictive_original_row_replaces_grid_without_duplicate_weight():
    clearance=np.array([[-.001,-.02,np.nan,.01]])
    deficit=np.minimum(clearance,0);deficit[~np.isfinite(deficit)]=0
    rows=np.zeros((1,4,6));kin=SimpleNamespace(xpos=np.zeros((1,4,3)))
    measured={(0,0,g):dict(point=np.array([1.,0,0]),direction=np.array([0.,0,1.]),clearance=p)
              for g,p in enumerate([-.01,-.01,-.03,np.nan])}
    replace_deeper_original(0,np.arange(4),measured,clearance,deficit,rows,kin,np.arange(4),0.)
    np.testing.assert_allclose(deficit,[[-.01,-.02,-.03,0.]])
    np.testing.assert_array_equal(rows[0,0],[0.,0.,1.,0.,-1.,0.])
    assert not rows[0,1].any() and not rows[0,3].any()


def test_original_clearance_term_gradient_and_serial_chunk_agreement():
    from omniretargeting._fast.rows import _Accumulator
    from tests.test_fast_backend import _floor
    class Plane:
        def query(self,points):
            return dict(clearance=points[:,2]-.3,uncertain=np.zeros(len(points),bool),
                        directions=np.tile([0.,0.,1.],(len(points),1)))
    objective,poses,_=_objective(temporal=True)
    objective.scene_sdf=_floor(0.)
    geometry=OriginalGeometry(objective,[],Plane())
    objective.original_clearance=OriginalClearance(geometry,[objective.model.geom('foot_geom').id])
    def evaluate(obj,q,jac=True):
        obj.kin.evaluate(q,jac)
        acc=_Accumulator(len(q),obj.width,jac)
        obj._clearance_terms(acc,{},q)
        return acc
    whole=evaluate(objective,poses)
    assert whole.cost>0
    chunk=objective.for_frames(slice(7,15))
    part=evaluate(chunk,poses[7:15])
    np.testing.assert_allclose(part.frame_cost,whole.frame_cost[7:15],rtol=1e-14)
    # Gauss-Newton gradient stores J.T*r (cost is r.T*r).
    frame=10;column=int(np.flatnonzero(objective.coords.cols==objective.model.joint('ankle').dofadr[0])[0])
    address=int(objective.model.joint('ankle').qposadr[0]);epsilon=1e-6
    plus=poses.copy();minus=poses.copy();plus[frame,address]+=epsilon;minus[frame,address]-=epsilon
    numerical=(evaluate(objective,plus,False).cost-evaluate(objective,minus,False).cost)/(2*epsilon)
    np.testing.assert_allclose(numerical,2*whole.gradient[frame,column],rtol=2e-4,atol=1e-4)


def test_static_original_queries_are_batched_without_changing_first_minima():
    class Plane:
        def __init__(self):self.calls=0
        def query(self,points):
            self.calls+=1
            return dict(clearance=points[:,2]-.3,uncertain=np.zeros(len(points),bool),
                        directions=np.tile([0.,0.,1.],(len(points),1)))
    objective,poses,_=_objective(temporal=True)
    scene=Plane();geometry=OriginalGeometry(objective,[],scene)
    geom=objective.model.geom('foot_geom').id
    raw=OriginalClearance(geometry,[geom])
    measured=raw.evaluate(poses)
    assert len(measured)==len(poses) and scene.calls==1
    for f,pose in enumerate(poses):
        geometry.data.qpos[:]=pose
        import mujoco
        mujoco.mj_forward(geometry.model,geometry.data)
        local=objective.shell_local[objective.shell_sample_geom==geom]
        points=local@geometry.data.geom_xmat[geom].reshape(3,3).T+geometry.data.geom_xpos[geom]
        index=int(np.argmin(points[:,2]))
        row=measured[0,f,geom]
        assert row['point'].tobytes()==points[index].tobytes()
        assert row['clearance']==float(points[index,2]-.3)
