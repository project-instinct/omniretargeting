import numpy as np
import trimesh

from omniretargeting.moving_scene import MovingSceneQuery
from omniretargeting.moving_outside import certified_outside
from tests.test_moving_source_evidence import robot, spec


def test_unknown_patch_does_not_replace_lost_penetrating_observation():
    model = robot()
    query = MovingSceneQuery(model, [spec()], 1)
    pose = model.qpos0.copy(); pose[2] = -.2
    required = frozenset(query.query(pose, 0)['open_penetrating_samples'])
    assert required
    pose[0] = .7
    assert not certified_outside(query, pose, 0, required)
    # Returning to a reliably observed free side is an allowed proof.
    pose[0] = 0.; pose[2] = .2
    assert certified_outside(query, pose, 0, required) == required
    assert query.objects[0]['query'].evidence_radius == .4


def test_broadphase_skipped_geom_still_checks_every_required_material_point():
    model = robot()
    query = MovingSceneQuery(model, [dict(mesh=trimesh.creation.box(extents=[.4]*3))], 1)
    pose = model.qpos0.copy()
    gid = int(query.geom_ids[0])
    required = frozenset((0, gid, i) for i in range(len(query.shells[0]['offsets'])))
    assert not certified_outside(query, pose, 0, required)
    pose[0] = 10.
    measured = query.query(pose, 0)
    assert not measured['rows'] and not measured['open_known_samples']
    assert certified_outside(query, pose, 0, required) == required
    assert len(required) == len(query.shells[0]['offsets'])


def test_closed_component_can_certify_cavity_inside_convex_hull():
    model=robot()
    mesh=trimesh.creation.torus(major_radius=.5,minor_radius=.1)
    query=MovingSceneQuery(model,[dict(mesh=mesh)],1)
    gid=int(query.geom_ids[0]);pose=model.qpos0.copy()
    ids=frozenset((0,gid,i) for i in range(len(query.shells[0]['offsets'])))
    assert not query.objects[0]['query'].outside_hull(np.zeros((1,3)))[0]
    assert certified_outside(query,pose,0,ids)==ids


def test_unsigned_component_cannot_prove_hull_interior_outside(monkeypatch):
    from tests.test_moving_source_evidence import panel
    box=trimesh.creation.box(extents=[.4,.4,.4]);box.apply_translation([.5,0,0])
    sheet=panel();sheet.apply_translation([-.5,0,0])
    model=robot();owner=MovingSceneQuery(model,[dict(mesh=trimesh.util.concatenate([box,sheet]))],1)
    query=owner.objects[0]['query'];pose=model.qpos0.copy();gid=int(owner.geom_ids[0])
    required=frozenset((0,gid,i) for i in range(len(owner.shells[0]['offsets'])))
    assert query.has_unsigned_only_components
    points=owner.shells[0]['offsets']
    expected=query.query(points)
    assert expected['uncertain'].all()
    hull_outside=query.outside_hull(points)
    def unnecessary(*args,**kwargs):raise AssertionError('an always-unknown whole-object query is unnecessary')
    monkeypatch.setattr(query,'query',unnecessary)
    result=certified_outside(owner,pose,0,required)
    assert result=={(0,gid,i) for i in np.flatnonzero(hull_outside)}
    pose[0]=5.
    assert certified_outside(owner,pose,0,required)==required
