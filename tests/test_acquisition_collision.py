"""Acquisition sees self-penetration and evidence-signed open objects."""

import numpy as np
import pytest
import trimesh

from omniretargeting._fast.objective import FastObjective
from omniretargeting._fast.sdf import object_sdf
from tests.test_fast_backend import _problem


def _objective(poses, cfg, task_set, rotations, enabled):
    return FastObjective(task_set, dict(cfg, self_collision_enabled=enabled), reference=poses,
                         temporal_reference=poses[:, task_set.coords.qi], root_rotations=rotations,
                         framerate=30.0)


def test_self_penetration_adds_rows_that_push_bodies_apart():
    _, cfg, coords, poses, task_set, rotations = _problem(frames=6)
    colliding = poses.copy()
    # Swing the arm (shoulder about x) through the leg on the other side.
    colliding[:, coords.qi[3]] = 0.7
    enabled = _objective(colliding, cfg, task_set, rotations, True)
    disabled = _objective(colliding, cfg, task_set, rotations, False)
    assert disabled.self_collision is None
    rows = [row for f in range(len(colliding)) for row in enabled.self_collision.query(colliding[f])
            if row["clearance"] < cfg["self_clearance_m"]]
    assert rows, "the test pose must interpenetrate"
    with_rows = enabled.evaluate(colliding)
    without = disabled.evaluate(colliding)
    assert with_rows[3]["self_depth_m"] > 0
    assert with_rows[0].cost > without[0].cost
    # A row's Jacobian points towards more clearance: a small step along it
    # separates the pair (the hinge residual is negative while penetrating).
    query = enabled.self_collision
    row = min(query.query(colliding[0]), key=lambda r: r["clearance"])
    jac = query.jacobian(row, query.data)
    moved = colliding[0].copy()
    moved[coords.qi[3]] += 1e-3 * np.sign(jac[coords.di[3]])
    after = min(r["clearance"] for r in query.query(moved) if r["pair"] == row["pair"])
    assert after > row["clearance"]


def test_open_object_grid_is_signed_only_near_evidence():
    box = trimesh.creation.box(extents=[.4, .4, .4])
    top = np.flatnonzero(box.face_normals[:, 2] > .5)
    open_box = trimesh.Trimesh(box.vertices, np.delete(box.faces, top, axis=0), process=False)
    evidence = {"points": np.array([[.2, 0., 0.]]), "normals": np.array([[1., 0., 0.]]), "radius": .08}
    plain = object_sdf(open_box, .02, .1)
    signed = object_sdf(open_box, .02, .1, evidence)
    near = np.array([[.25, 0., 0.], [.15, 0., 0.]])  # outside / inside the observed wall
    far = np.array([[-.15, 0., 0.]])  # behind the unobserved wall, inside the box's hull
    assert np.isnan(plain.query(near, gradient=False)).all()
    value = signed.query(near, gradient=False)
    assert value[0] > 0 > value[1]
    assert np.isnan(signed.query(far, gradient=False)).all()  # no evidence: still unknown
    # Outside the box's hull no closure of the recorded walls contains a point.
    assert signed.query(np.array([[-.25, 0., 0.]]), gradient=False)[0] == pytest.approx(.05, abs=1e-6)


def test_remote_unobserved_fragment_does_not_hide_an_observed_wall():
    box = trimesh.creation.box(extents=[.4, .4, .4])
    top = np.flatnonzero(box.face_normals[:, 2] > .5)
    walls = trimesh.Trimesh(box.vertices, np.delete(box.faces, top, axis=0), process=False)
    fragment = trimesh.Trimesh([[1., 1., 1.], [1.1, 1., 1.], [1., 1.1, 1.]], [[0, 1, 2]], process=False)
    mesh = trimesh.util.concatenate([walls, fragment])
    evidence = {"points": np.array([[.2, 0., 0.]]), "normals": np.array([[1., 0., 0.]]), "radius": .08}
    from omniretargeting._fast.sdf import scene_query_for
    whole = scene_query_for(mesh, enclosed_free_space=False, evidence=evidence)[0]
    near = np.array([[.25, 0., 0.], [.15, 0., 0.]])
    assert whole.query(near)["uncertain"].all()  # any unobserved open part poisons all points
    grid = object_sdf(mesh, .02, .1, evidence)
    value = grid.query(near, gradient=False)
    assert value[0] > 0 > value[1]
    assert np.isnan(grid.query(np.array([[1.05, 1.02, 1.05]]), gradient=False)).all()
