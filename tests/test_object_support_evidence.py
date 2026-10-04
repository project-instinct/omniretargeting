"""Approximate object net-wrench support: skin contacts a moving object needs."""

import numpy as np
import trimesh

from omniretargeting.object_support_evidence import (
    CombinedSupportEvidence,
    ObjectSupportEvidence,
    build_object_support_evidence,
    object_inertia,
)
from omniretargeting.source_contacts import build_source_contacts
from omniretargeting.surface_contact import surface_contact_config

FRAMES = 40
HALF = 0.15  # box half extent


def floor():
    mesh = trimesh.creation.box(extents=[4, 4, 0.2])
    mesh.apply_translation([0, 0, -0.1])
    return mesh


def box(heights):
    heights = np.broadcast_to(np.asarray(heights, float), (FRAMES,))
    return {"mesh": trimesh.creation.box(extents=[2 * HALF] * 3), "name": "box",
            "translations": np.c_[np.zeros((FRAMES, 2)), heights],
            "rotations": np.tile(np.eye(3), (FRAMES, 1, 1)), "scales": np.ones(FRAMES)}


def patch(center, normal, heights):
    """A 3x3 skin patch 5 mm off a box face, riding with the box."""
    normal = np.asarray(normal, float)
    axes = [a for a in np.eye(3) if abs(a @ normal) < 0.5]
    grid = [center + 0.04 * (i * axes[0] + j * axes[1]) for i in (-1, 0, 1) for j in (-1, 0, 1)]
    points = np.asarray(grid)[None] + np.c_[np.zeros((FRAMES, 2)), heights][:, None]
    return points, np.broadcast_to(normal, points.shape)


def skin(patches, heights):
    heights = np.broadcast_to(np.asarray(heights, float), (FRAMES,))
    points, normals, labels = [], [], []
    for label, (center, normal) in patches.items():
        p, n = patch(np.asarray(center, float), normal, heights)
        points.append(p)
        normals.append(n)
        labels += [label] * p.shape[1]
    points = np.concatenate(points, axis=1)
    return {"points": points, "skin_normals": np.concatenate(normals, axis=1),
            "valid": np.ones(points.shape[:2], bool), "labels": labels}


def evidence(patches, heights, objects=None):
    config = surface_contact_config()
    objects = objects or [box(heights)]
    table = build_source_contacts(skin(patches, heights), floor(), config, framerate=30.0,
                                  scene_objects=objects, scene_enclosed_free_space=False)
    result, diagnostics = build_object_support_evidence(
        table, floor(), objects, config, framerate=30.0, scene_enclosed_free_space=False)
    assert diagnostics["status"] == "active"
    return result, diagnostics


GAP = HALF + 0.005
UNDER = {"palm": ((0, 0, -GAP), (0, 0, 1)), "back": ((0, 0, GAP), (0, 0, -1))}
SIDES = {"left": ((-GAP, 0, 0), (1, 0, 0)), "right": ((GAP, 0, 0), (-1, 0, 0))}


def roles(result, region):
    return np.array([result.role(region, 1, f) for f in range(FRAMES)])


def test_the_hand_under_a_carried_box_holds_it_up_and_a_hand_on_top_does_not():
    result, diagnostics = evidence(UNDER, 1.0)
    assert (roles(result, "palm") == "support").all()
    # A hand on top only pushes the box down: never needed to carry it.
    assert (roles(result, "back") == None).all()  # noqa: E711
    assert diagnostics["objects"][0]["inertia_source"] == "uniform_density_closed_mesh"
    assert diagnostics["environment_carried_frames"] == 0


def test_a_box_resting_on_the_floor_is_carried_by_the_floor_not_by_touching_hands():
    result, diagnostics = evidence(SIDES, HALF)
    assert (roles(result, "left") == None).all()  # noqa: E711
    assert (roles(result, "right") == None).all()  # noqa: E711
    assert diagnostics["environment_carried_frames"] == FRAMES


def test_squeezing_hands_become_support_when_the_box_leaves_the_floor():
    heights = np.r_[np.full(20, HALF), np.linspace(HALF, 0.5, 21)[1:]]
    result, _ = evidence(SIDES, heights)
    for region in SIDES:
        assert (roles(result, region)[:15] == None).all()  # noqa: E711
        assert (roles(result, region)[25:] == "support").all()


def test_a_box_pushed_along_the_floor_stays_carried_by_the_floor():
    x = np.linspace(0.0, 0.8, FRAMES)
    objects = [{**box(HALF), "translations": np.c_[x, np.zeros(FRAMES), np.full(FRAMES, HALF)]}]
    config = surface_contact_config()
    source = skin({"left": SIDES["left"]}, HALF)
    source["points"] = source["points"] + np.c_[x, np.zeros((FRAMES, 2))][:, None]
    table = build_source_contacts(source, floor(), config, framerate=30.0,
                                  scene_objects=objects, scene_enclosed_free_space=False)
    result, diagnostics = build_object_support_evidence(
        table, floor(), objects, config, framerate=30.0, scene_enclosed_free_space=False)
    # The floor's friction cone carries a pushed box: the hand is control, not support.
    assert (roles(result, "left") == None).all()  # noqa: E711
    assert diagnostics["environment_carried_frames"] == FRAMES


def test_a_seat_on_a_narrow_post_is_carried_through_its_base():
    """Objects touching each other are one system; the joint force is internal."""
    post = trimesh.creation.cylinder(radius=0.015, height=0.38)
    post.apply_translation([0, 0, 0.21])
    foot = trimesh.creation.cylinder(radius=0.3, height=0.02)
    foot.apply_translation([0, 0, 0.01])
    base = trimesh.util.concatenate([foot, post])
    # The seat's centre of mass lies 10 cm beside the post: as a lone body on a
    # one-sided contact it would tip, so a touching hand would look necessary.
    # Its eight corner vertices are all far from the post.
    seat = trimesh.creation.box(extents=[0.3, 0.3, 0.04])
    seat.apply_translation([0.1, 0, 0.42])
    still = lambda mesh, name: {"mesh": mesh, "name": name, "translations": np.zeros((FRAMES, 3)),
                                "rotations": np.tile(np.eye(3), (FRAMES, 1, 1)), "scales": np.ones(FRAMES)}
    objects = [still(base, "base"), still(seat, "seat")]
    hand = {"palm": ((0.1, 0, 0.44 + 0.005), (0, 0, -1)), "under": ((0.2, 0, 0.4 - 0.005), (0, 0, 1))}
    config = surface_contact_config()
    table = build_source_contacts(skin(hand, 0.0), floor(), config, framerate=30.0,
                                  scene_objects=objects, scene_enclosed_free_space=False)
    result, diagnostics = build_object_support_evidence(
        table, floor(), objects, config, framerate=30.0, scene_enclosed_free_space=False)
    assert not any(result.decisions)
    assert diagnostics["environment_carried_frames"] == FRAMES


def test_an_open_mesh_uses_its_convex_hull_for_inertia():
    mesh = trimesh.creation.box(extents=[0.2, 0.4, 0.6])
    open_mesh = trimesh.Trimesh(mesh.vertices, mesh.faces[:-2], process=False)
    center, inertia, volume, source = object_inertia(open_mesh)
    assert source == "convex_hull"
    assert np.isclose(volume, 0.2 * 0.4 * 0.6)
    np.testing.assert_allclose(center, 0, atol=1e-9)
    np.testing.assert_allclose(np.diag(inertia), [(0.16 + 0.36) / 12, (0.04 + 0.36) / 12, (0.04 + 0.16) / 12])


def test_object_support_only_promotes_the_human_roles():
    class Human:
        frames = 2
        diagnostics = {"method": "human"}

        def role(self, region, surface, frame):
            return "follow"

    objects = ObjectSupportEvidence([{("hand", 1)}, set()], {"status": "active"})
    combined = CombinedSupportEvidence(Human(), objects)
    assert combined.role("hand", 1, 0) == "support"
    assert combined.role("hand", 1, 1) == "follow"
    assert combined.role("foot", 0, 0) == "follow"
    assert combined.diagnostics["object_support"] == {"status": "active"}
    assert CombinedSupportEvidence(None, objects).role("foot", 0, 0) is None


def test_missing_inputs_disable_object_support():
    config = surface_contact_config()
    assert build_object_support_evidence(None, floor(), [box(1.0)], config, framerate=30.0)[0] is None
    result, diagnostics = build_object_support_evidence(
        object(), floor(), (), config, framerate=30.0)
    assert result is None and diagnostics["reason"] == "no_moving_objects"
