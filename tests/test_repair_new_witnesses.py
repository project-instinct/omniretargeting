from types import SimpleNamespace
import numpy as np

from omniretargeting._fast.mesh_repair import OriginalGeometry
from omniretargeting._fast.repair_witnesses import RepairWitnesses
from tests.test_repair_penetration_yield import _problem


class Plane:
    def query(self, points):
        return dict(clearance=points[:,2].copy(), uncertain=np.zeros(len(points),bool),
                    directions=np.tile([0.,0.,1.],(len(points),1)))


def test_acceptance_sample_is_requeried_at_current_pose_and_preserves_collision_key():
    objective, poses = _problem()
    geometry = OriginalGeometry(objective, [], Plane(), manifold=True)
    gid = objective.model.geom('wet_geom').id
    sample = int(np.flatnonzero(objective.shell_sample_geom == gid)[0])
    key = (0,'static',gid)
    current = dict(rows=[],depths={},failures=[],complete=True)
    candidate = dict(rows=[(key,dict(sample_id=sample,clearance=-1.,point=np.ones(3)*20))])
    working = RepairWitnesses()
    assert working.observe(current,candidate)==1
    assert working.observe(current,candidate)==0
    augmented = working.augment(geometry,poses,current)
    assert len(augmented['rows'])==1
    row_key,row = augmented['rows'][0]
    assert row_key==key and row['sample_id']==sample
    assert row['clearance']==row['point'][2]
    assert row['clearance']!=-1.
    assert np.isfinite(row['jacobian']).all()
    assert augmented['depths']==current['depths']=={}
    assert not current['rows']
    moved=poses.copy();moved[:,objective.model.joint('wet_joint').qposadr[0]]+=.002
    again=working.augment(geometry,moved,current)['rows'][0][1]
    assert abs(again['clearance']-row['clearance']-.002)<1e-12


def test_static_manifold_retains_original_material_ids_across_batch_boundaries():
    objective, poses = _problem()
    geometry=OriginalGeometry(objective,[],Plane(),manifold=True)
    geometry.SCENE_BATCH_POINTS=7
    measured=geometry.measure(poses)
    rows=[(k,r) for k,r in measured['rows'] if k[1]=='static']
    assert rows
    for key,row in rows:
        assert objective.shell_sample_geom[row['sample_id']]==key[-1]


def test_remembered_forward_samples_share_one_exact_current_geom_query(monkeypatch):
    import trimesh
    objective,poses=_problem()
    mesh=trimesh.creation.box(extents=[.02,.02,.02])
    geometry=OriginalGeometry(objective,[dict(mesh=mesh,translations=np.tile([.3,0,0.],(len(poses),1)))],None)
    geom=objective.model.geom('wet_geom').id
    key=(0,'moving',0,'closed_surface',0,geom)
    current=dict(rows=[],depths={},failures=[],complete=True)
    working=RepairWitnesses();working.required={(key,0),(key,1)}
    expected=[]
    for sample in (0,1):
        geometry.data.qpos[:]=poses[0]
        import mujoco
        mujoco.mj_forward(geometry.model,geometry.data)
        expected.append((key,working._query(geometry,poses[0],key,sample)))
    expected=geometry.linearize(poses,{**current,'rows':expected})['rows']
    old=geometry.moving._forward_samples;calls=[]
    def query(*args,**kwargs):
        calls.append(1);return old(*args,**kwargs)
    monkeypatch.setattr(geometry.moving,'_forward_samples',query)
    actual=working.augment(geometry,poses,current)['rows']
    assert len(calls)==1 and len(actual)==2
    for (_,a),(_,b) in zip(actual,expected):
        for field in ('point','direction','clearance','jacobian'):
            av,bv=np.asarray(a[field]),np.asarray(b[field])
            assert av.dtype==bv.dtype and av.shape==bv.shape and av.tobytes()==bv.tobytes()
    moved=poses.copy();moved[:,objective.model.joint('wet_joint').qposadr[0]]+=.001
    changed=working.augment(geometry,moved,current)['rows']
    assert len(calls)==2
    assert any(a[1]['point'].tobytes()!=b[1]['point'].tobytes() for a,b in zip(changed,actual))


def test_remembered_static_points_batch_real_winding_with_exact_rows_and_jacobians(monkeypatch):
    import mujoco
    import trimesh
    from omniretargeting.scene_geometry import SceneQuery
    objective, poses = _problem()
    scene = SceneQuery(trimesh.creation.box(extents=[1., 1., 1.]))
    geometry = OriginalGeometry(objective, [], scene, manifold=True)
    geom = objective.model.geom('wet_geom').id
    samples = np.flatnonzero(objective.shell_sample_geom == geom)[:3]
    working = RepairWitnesses()
    working.required = {((frame, 'static', geom), int(sample)) for frame in (0, 1) for sample in samples}
    current = dict(rows=[], depths={}, failures=[], complete=True)
    expected = []
    for key, sample in sorted(working.required):
        geometry.data.qpos[:] = poses[key[0]]
        mujoco.mj_forward(objective.model, geometry.data)
        expected.append((key, working._query(geometry, poses[key[0]], key, sample)))
    expected = geometry.linearize(poses, {**current, 'rows': expected})['rows']
    original = scene.query
    calls = []
    def query(points, **kwargs):
        calls.append(len(points))
        return original(points, **kwargs)
    monkeypatch.setattr(scene, 'query', query)
    geometry.SCENE_BATCH_POINTS = 4
    actual = working.augment(geometry, poses, current)['rows']
    assert calls == [4, 2]
    for (key, row), (other_key, other) in zip(actual, expected):
        assert key == other_key
        for field in ('point', 'direction', 'clearance', 'jacobian'):
            a, b = np.asarray(row[field]), np.asarray(other[field])
            assert a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()
