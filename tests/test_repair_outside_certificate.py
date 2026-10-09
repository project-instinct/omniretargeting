"""Leaving an open object's broad phase is accepted only with geometric proof."""

from types import SimpleNamespace

import numpy as np
import pytest
import trimesh

from omniretargeting._fast.mesh_repair import OriginalGeometry
from omniretargeting.scene_geometry import collision_geom_ids, geom_surface_offsets
from tests.test_moving_source_evidence import robot, spec


def geometry(obj):
    model = robot()
    geoms = collision_geom_ids(model)
    offsets = [geom_surface_offsets(model, int(g), .01) for g in geoms]
    objective = SimpleNamespace(
        model=model, frames=1,
        coords=SimpleNamespace(qi=np.empty(0, int), di=np.empty(0, int)),
        shell_geoms=geoms, shell_local=np.concatenate(offsets),
        shell_sample_geom=np.concatenate([np.full(len(o), g) for g, o in zip(geoms, offsets)]),
    )
    return OriginalGeometry(objective, [obj], None), model.qpos0[None].copy()


def test_departed_open_object_samples_are_certified_after_broadphase_skip():
    mesh = trimesh.creation.box(extents=[.4, .4, .4])
    mesh.update_faces(np.arange(len(mesh.faces) - 1))
    query, poses = geometry(dict(
        mesh=mesh, collision_evidence=dict(points=np.array([[0., 0., .2]]),
                                          normals=np.array([[0., 0., 1.]]), radius=1.)))
    query.freeze_reference_observations(poses)
    required = query._moving_evidence_baseline[0]
    assert required
    initial = query.measure(poses)
    assert not initial['failures']
    poses[:, 0] = 10.
    assert not query.moving.query(poses[0], 0)['open_known_samples']
    checked = query.measure(poses, protected_depths=initial['depths'])
    assert checked['complete'] and not checked['failures']
    assert dict(query._moving_evidence_baseline) == {0: required}
    assert max(checked['depths'].values(), default=0.) == pytest.approx(0.)


def test_unobserved_patch_remains_unknown_and_cannot_erase_penetration():
    query, poses = geometry(spec())
    poses[:, 2] = -.2
    query.freeze_reference_observations(poses)
    initial = query.measure(poses)
    required = query._moving_evidence_baseline[0]
    assert required
    poses[:, 0] = .7
    checked = query.measure(poses, protected_depths=initial['depths'])
    assert any('lost original source-side observation' in f for f in checked['failures'])
    assert query._moving_evidence_baseline[0] == required


def test_unknown_exit_rows_retain_depth_without_claiming_a_known_side():
    query, poses = geometry(spec())
    query.uncertain_exit_rows = True
    poses[:, 2] = -.2
    query.freeze_reference_observations(poses)
    required = query._moving_evidence_baseline[0]
    poses[:, 0] = .7
    checked = query.measure(poses)
    rows = [r for _, r in checked['rows'] if r.get('kind') == 'unresolved_material']
    assert rows and checked['failures']
    assert all(r['uncertain'] and r['clearance'] < 0 for r in rows)
    np.testing.assert_allclose(rows[0]['direction'], [0., 0., 1.], atol=1e-12)
    assert -rows[0]['clearance'] == pytest.approx(.25)
    assert query._moving_evidence_baseline[0] == required
    poses[:, 2] += .01
    closer = query.measure(poses)
    row = next(r for _, r in closer['rows'] if r.get('kind') == 'unresolved_material')
    assert -row['clearance'] == pytest.approx(.24)
    assert closer['failures']


def test_unknown_exit_tangent_jacobian_matches_unsigned_distance():
    from omniretargeting._fast.uncertain_exit import material_rows

    query, poses = geometry(spec())
    poses[:, 2] = -.2
    query.freeze_reference_observations(poses)
    required = query._moving_evidence_baseline[0]
    poses[:, 0] = .7
    row = material_rows(query.moving, poses[0], 0, required, np.arange(6))[0]
    assert row['uncertain'] and row['jacobian'][2] == pytest.approx(1.)
    plus, minus = poses[0].copy(), poses[0].copy()
    plus[2] += 1e-6
    minus[2] -= 1e-6
    difference = (material_rows(query.moving, plus, 0, required, [])[0]['clearance']
                  - material_rows(query.moving, minus, 0, required, [])[0]['clearance']) / 2e-6
    assert row['jacobian'][2] == pytest.approx(difference)
