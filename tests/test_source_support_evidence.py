"""Main-pipeline source support evidence: admissible supporting surfaces."""

import numpy as np
import trimesh

from omniretargeting.source_support_evidence import object_grounded


def floor_mesh():
    return trimesh.Trimesh([[-5, -5, 0], [5, -5, 0], [5, 5, 0], [-5, 5, 0]], [[0, 1, 2], [0, 2, 3]])


def spec(mesh, frames, translations):
    translations = np.asarray(translations, float)
    return {"mesh": mesh, "translations": translations,
            "rotations": np.tile(np.eye(3), (frames, 1, 1)), "scales": np.ones(frames)}


def test_a_carried_object_is_not_grounded_while_in_the_air():
    box = trimesh.creation.box(extents=(0.4, 0.4, 0.4))
    heights = np.array([0.2, 0.21, 0.5, 1.0, 0.215])  # resting, lifted, carried, set down
    spec_ = {"mesh": box, "translations": np.c_[np.zeros((5, 2)), heights],
             "rotations": np.tile(np.eye(3), (5, 1, 1)), "scales": np.ones(5)}
    grounded = object_grounded(floor_mesh(), [spec_], 5, 0.03)
    np.testing.assert_array_equal(grounded[0], [True, True, False, False, True])
    assert not object_grounded(None, [spec_], 5, 0.03).any()


def test_only_a_resting_object_is_part_of_the_fixed_environment():
    from omniretargeting.source_support_evidence import object_fixed

    box = trimesh.creation.box(extents=(0.4, 0.4, 0.4))
    x = np.r_[np.zeros(4), 0.1 * np.arange(1, 5)]  # rests, then is pushed along the floor
    spec_ = {"mesh": box, "translations": np.c_[x, np.zeros(8), np.full(8, 0.2)],
             "rotations": np.tile(np.eye(3), (8, 1, 1)), "scales": np.ones(8)}
    fixed = object_fixed(floor_mesh(), [spec_], 8, 0.03, 0.04, 30.0)
    np.testing.assert_array_equal(fixed[0], [True, True, True, False, False, False, False, False])


def test_a_stacked_seat_is_fixed_only_while_its_base_is_fixed():
    from omniretargeting.source_support_evidence import object_fixed

    frames = 7
    # A chair base rests on the floor throughout, then slides away; a seat
    # resting on it stays exactly where it was.
    x = np.r_[np.zeros(4), 0.06 * np.arange(1, 4)]
    base = spec(trimesh.creation.box(extents=(0.4, 0.4, 0.2)), frames,
                np.c_[x, np.zeros(frames), np.full(frames, 0.1)])
    seat = spec(trimesh.creation.box(extents=(0.5, 0.5, 0.04)), frames,
                np.c_[np.zeros(frames), np.zeros(frames), np.full(frames, 0.22)])
    fixed = object_fixed(floor_mesh(), [base, seat], frames, 0.03, 0.04, 30.0)
    # Both fixed while stacked and still; the seat loses fixed-environment
    # status as soon as its support is pushed, even though they still touch.
    np.testing.assert_array_equal(fixed[0], [True, True, True, False, False, False, False])
    np.testing.assert_array_equal(fixed[1], [True, True, True, False, False, False, False])


def test_a_still_object_floating_above_the_scene_is_never_fixed():
    from omniretargeting.source_support_evidence import object_fixed

    frames = 4
    resting = spec(trimesh.creation.box(extents=(0.4, 0.4, 0.4)), frames,
                   np.c_[np.zeros(frames), np.zeros(frames), np.full(frames, 0.2)])
    flying = spec(trimesh.creation.box(extents=(0.4, 0.4, 0.4)), frames,
                  np.c_[np.zeros(frames), np.zeros(frames), np.full(frames, 1.0)])
    fixed = object_fixed(floor_mesh(), [resting, flying], frames, 0.03, 0.04, 30.0)
    assert fixed[0].all()
    assert not fixed[1].any()
