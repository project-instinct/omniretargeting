"""HSOI contracts, discrete motion operations, detection, and graph math."""

from dataclasses import replace
import json

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import trimesh

from omniretargeting import (
    Contact,
    DataSource,
    EntityPose,
    EntityTrajectory,
    MotionData,
    MotionFrame,
    Scene,
    SceneEntity,
    detect_contacts,
    load_contact_trajectory,
    save_hsoi_annotations,
)
from omniretargeting.contacts import resolve_contact_anchors
from omniretargeting.retargeting import GenericInteractionRetargeter
from omniretargeting.utils import (
    calculate_contact_edge_weights,
    calculate_laplacian_coordinates,
    calculate_laplacian_matrix,
    create_flat_terrain,
)


@pytest.fixture
def motion():
    box = trimesh.creation.box()
    scene = Scene(
        {
            "box": SceneEntity(box.copy(), np.array([[0.5, 0, 0]])),
            "cabinet/door": SceneEntity(
                box.copy(), static_pose=EntityPose([2, 0, 1], [1, 0, 0, 0])
            ),
        }
    )
    track = EntityTrajectory(
        np.array([[t, 0, 1] for t in range(4)]),
        Rotation.from_euler(
            "z", np.array([0, 30, 60, 90])[:, None], degrees=True
        ).as_quat(scalar_first=True),
    )
    return MotionData(
        positions=np.zeros((4, 2, 3)),
        target_names=["palm", "knee"],
        framerate=2,
        scene=scene,
        entity_trajectories={"box": track},
        contact_trajectory=[
            [
                Contact("palm", "box", [0.5, 0, 0], confidence=0.8),
                Contact("knee", "cabinet/door", [0, 0, 0.5]),
            ],
            [Contact("palm", "cabinet/door", [0, 0.5, 0])],
            [],
            None,
        ],
    )


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"point_local": [1, 2, np.nan]}, "finite"),
        ({"point_local": [1, 2]}, "shape"),
        ({"confidence": -0.1}, "confidence"),
        ({"confidence": np.inf}, "confidence"),
        ({"normal_local": [2, 0, 0]}, "unit"),
        ({"source_point_id": ""}, "source_point_id"),
    ],
)
def test_contact_validation(changes, match):
    with pytest.raises(ValueError, match=match):
        Contact(
            **{
                "source_point_id": "palm",
                "scene_body_id": "box",
                "point_local": [0, 0, 0],
                **changes,
            }
        )


def test_endpoint_and_timeline_validation(motion):
    with pytest.raises(ValueError, match="exactly T"):
        replace(motion, contact_trajectory=[[]])
    with pytest.raises(ValueError, match="source point 'elbow'"):
        replace(
            motion,
            contact_trajectory=[[Contact("elbow", "box", [0, 0, 0])], [], [], []],
        )
    with pytest.raises(ValueError, match="unknown scene body|Unknown scene body"):
        replace(
            motion,
            contact_trajectory=[[Contact("palm", "wall", [0, 0, 0])], [], [], []],
        )
    with pytest.raises(ValueError, match="same frame count"):
        replace(
            motion,
            entity_trajectories={
                "box": EntityTrajectory(np.zeros((2, 3)), np.tile([1, 0, 0, 0], (2, 1)))
            },
        )
    with pytest.raises(ValueError, match="no trajectory or explicit static pose"):
        replace(motion, entity_trajectories=None)
    with pytest.raises(ValueError, match="unique"):
        replace(motion, target_names=["palm", "palm"])
    with pytest.raises(ValueError, match="unit wxyz"):
        EntityPose([0, 0, 0], [2, 0, 0, 0])


def test_frame_round_trip_and_availability(motion):
    frames = list(motion.iter_frames())
    assert [None if f.contacts is None else len(f.contacts) for f in frames] == [
        2,
        1,
        0,
        None,
    ]
    np.testing.assert_allclose(frames[2].entity_poses["box"].translation, [2, 0, 1])

    class Source(DataSource):
        target_names = motion.target_names
        framerate = motion.framerate

        def iter_frames(self):
            yield from frames

    restored = Source().load()
    assert restored.contact_trajectory[2] == []
    assert restored.contact_trajectory[3] is None
    np.testing.assert_array_equal(
        restored.entity_trajectories["box"].orientations,
        motion.entity_trajectories["box"].orientations,
    )
    restored.contact_trajectory[0][0].point_local[0] = 10
    assert motion.contact_trajectory[0][0].point_local[0] == 0.5
    assert replace(motion, contact_trajectory=None).contact_trajectory is None
    assert all(
        frame.contacts == []
        for frame in replace(
            motion, contact_trajectory=[[] for _ in range(4)]
        ).iter_frames()
    )


def test_copy_slice_and_discrete_resample(motion):
    copied = motion.copy()
    copied.contact_trajectory[0][0].point_local[0] = 4
    assert motion.contact_trajectory[0][0].point_local[0] == 0.5
    sliced = motion.slice_frames(slice(1, 4, 2))
    assert len(sliced.positions) == 2 and sliced.framerate == 1
    assert sliced.contact_trajectory[0][0].scene_body_id == "cabinet/door"
    assert sliced.contact_trajectory[1] is None
    np.testing.assert_array_equal(
        sliced.entity_trajectories["box"].translations[:, 0], [1, 3]
    )
    sampled = motion.resample(4)
    assert len(sampled.positions) == 7
    assert [
        None if contacts is None else len(contacts)
        for contacts in sampled.contact_trajectory
    ] == [2, 2, 1, 1, 0, 0, None]
    assert (
        sampled.contact_trajectory[1][0].scene_body_id == "box"
    )  # midpoint tie -> earlier
    assert sampled.contact_trajectory[2][0].scene_body_id == "cabinet/door"
    np.testing.assert_allclose(
        sampled.entity_trajectories["box"].translations[:, 0], np.arange(7) / 2
    )
    np.testing.assert_allclose(
        np.linalg.norm(sampled.entity_trajectories["box"].orientations, axis=1), 1
    )
    single = motion.slice_frames(slice(0, 1)).resample(120)
    assert len(single.positions) == 1 and len(single.contact_trajectory) == 1


@pytest.mark.parametrize("moving", [False, True])
def test_resample_preserves_geometry_only_entity_samples(moving):
    local_sample = np.array([[0.5, 0, 0]])
    static_pose = EntityPose([0, 0, 1], [1, 0, 0, 0])
    track = EntityTrajectory(
        np.array([[0, 0, 1], [1, 0, 1]]),
        Rotation.from_euler("z", [[0], [90]], degrees=True).as_quat(scalar_first=True),
    )
    scene = Scene({"box": SceneEntity(trimesh.creation.box(), static_pose=static_pose)})
    world_samples = np.stack(
        [
            (track.pose(t) if moving else static_pose).to_world(local_sample)
            for t in range(2)
        ]
    )
    motion = MotionData(
        positions=np.zeros((2, 1, 3)),
        target_names=["palm"],
        object_points=world_samples,
        framerate=2,
        scene=scene,
        entity_trajectories={"box": track} if moving else None,
        metadata={"object_points_body_id": "box"},
    )
    resampled = motion.resample(4)
    assert resampled.object_points.shape == (3, 1, 3)
    for frame in resampled.iter_frames():
        expected = frame.scene.pose("box", frame.entity_poses).to_world(local_sample)
        np.testing.assert_allclose(frame.object_points, expected, atol=1e-7)
    np.testing.assert_array_equal(motion.object_points, world_samples)
    assert len(motion.scene.entities["box"].local_samples) == 0


def test_scale_resolves_identical_world_anchors(motion):
    scaled = motion.scaled(2.5)
    for before, after in zip(motion.iter_frames(), scaled.iter_frames()):
        for old, new in zip(before.contacts or [], after.contacts or []):
            old_world = before.scene.pose(
                old.scene_body_id, before.entity_poses
            ).to_world(old.point_local)
            new_world = after.scene.pose(
                new.scene_body_id, after.entity_poses
            ).to_world(new.point_local)
            np.testing.assert_allclose(new_world, old_world * 2.5)
            assert old.confidence == new.confidence
    np.testing.assert_allclose(
        scaled.scene.entities["box"].local_samples,
        motion.scene.entities["box"].local_samples * 2.5,
    )
    np.testing.assert_allclose(
        scaled.scene.entities["box"].geometry.vertices,
        motion.scene.entities["box"].geometry.vertices * 2.5,
    )
    assert scaled.contact_trajectory[2] == [] and scaled.contact_trajectory[3] is None


def test_scale_reused_records_and_poses_only_once(motion):
    contact = Contact("palm", "box", [0.5, 0, 0])
    shared = replace(
        motion,
        contact_trajectory=[[contact]] * 4,
        root_translations=motion.positions[:, 0],
    )
    scaled = shared.scaled(2)
    for frame in scaled.contact_trajectory:
        np.testing.assert_array_equal(frame[0].point_local, [1, 0, 0])
    np.testing.assert_array_equal(contact.point_local, [0.5, 0, 0])


def test_json_annotations_round_trip(motion, tmp_path):
    path = tmp_path / "contacts.json"
    save_hsoi_annotations(motion, path)
    contacts = load_contact_trajectory(path)
    assert contacts[2] == [] and contacts[3] is None
    assert contacts[0][0].confidence == 0.8
    np.testing.assert_array_equal(contacts[0][0].point_local, [0.5, 0, 0])
    payload = json.loads(path.read_text())
    assert payload["quaternion_convention"] == "wxyz"
    assert len(payload["entity_trajectories"]["box"]["translations"]) == 4
    save_hsoi_annotations(motion.scaled(2), path, source_to_robot_scale=2)
    source_contacts = load_contact_trajectory(path, source_coordinates=True)
    np.testing.assert_array_equal(
        source_contacts[0][0].point_local, motion.contact_trajectory[0][0].point_local
    )


def test_world_frame_change_retains_local_contacts(motion):
    rotation = Rotation.from_rotvec([0.4, 0.2, 0.1])
    translation = np.array([4, -2, 1])
    transformed = motion.transformed(rotation.as_quat(scalar_first=True), translation)
    for old_frame, new_frame in zip(motion.iter_frames(), transformed.iter_frames()):
        for old_contact, new_contact in zip(
            old_frame.contacts or [], new_frame.contacts or []
        ):
            old_world = old_frame.scene.pose(
                old_contact.scene_body_id, old_frame.entity_poses
            ).to_world(old_contact.point_local)
            new_world = new_frame.scene.pose(
                new_contact.scene_body_id, new_frame.entity_poses
            ).to_world(new_contact.point_local)
            np.testing.assert_allclose(
                new_world, rotation.apply(old_world) + translation
            )
            np.testing.assert_array_equal(
                old_contact.point_local, new_contact.point_local
            )


def test_all_unavailable_frames_survive_data_source_load(motion):
    frames = list(replace(motion, contact_trajectory=[None] * 4).iter_frames())

    class Source(DataSource):
        target_names = motion.target_names

        def iter_frames(self):
            yield from frames

    assert Source().load().contact_trajectory == [None] * 4


def detector_motion(local_points, moving=True):
    count = len(local_points)
    angles = np.arange(count) * 45 if moving else np.zeros(count)
    rotations = Rotation.from_euler("z", angles[:, None], degrees=True)
    translations = (
        np.array([[t, 0, 2] for t in range(count)])
        if moving
        else np.tile([0, 0, 2], (count, 1))
    )
    positions = rotations.apply(local_points) + translations
    return MotionData(
        positions=positions[:, None],
        target_names=["palm"],
        framerate=30,
        scene=Scene({"body": SceneEntity(trimesh.creation.box())}),
        entity_trajectories={
            "body": EntityTrajectory(translations, rotations.as_quat(scalar_first=True))
        },
    )


def test_detector_sticking_to_translating_and_rotating_body():
    motion = detector_motion(np.tile([0.51, 0.1, 0.1], (5, 1)))
    contacts = detect_contacts(
        motion, {"distance_threshold": 0.02, "velocity_threshold": 0.01}
    )
    assert all(len(frame) == 1 for frame in contacts)
    for frame in contacts:
        np.testing.assert_allclose(frame[0].point_local, [0.5, 0.1, 0.1], atol=1e-6)
        np.testing.assert_allclose(frame[0].normal_local, [1, 0, 0])
        assert frame[0].source_point_id == "palm"


def test_detector_sliding_speed_and_short_runs():
    motion = detector_motion(
        np.array([[0.51, y, 0.1] for y in [0, 0.1, 0.2, 0.3, 0.4]]), moving=False
    )
    assert all(
        frame == [] for frame in detect_contacts(motion, {"velocity_threshold": 0.1})
    )
    sliding = detect_contacts(motion, {"velocity_threshold": None})
    assert all(len(frame) == 1 for frame in sliding)
    assert sliding[0][0].point_local[1] != sliding[4][0].point_local[1]
    short = detector_motion(
        np.array([[x, 0, 0] for x in [0.51, 0.51, 1, 0.51, 1]]), moving=False
    )
    contacts = detect_contacts(
        short, {"velocity_threshold": None, "min_contact_frames": 2}
    )
    assert [len(frame) for frame in contacts] == [1, 1, 0, 0, 0]


def test_detector_preserves_available_empty_frames_and_annotated_anchors():
    motion = detector_motion(np.tile([0.51, 0.1, 0.1], (3, 1)))
    annotated = Contact("palm", "body", [0.5, 0.2, 0.1])
    motion = replace(motion, contact_trajectory=[[], None, [annotated]])
    contacts = detect_contacts(motion, {})
    assert (
        detect_contacts(replace(motion, contact_trajectory=None), {"enabled": False})
        is None
    )
    disabled = detect_contacts(motion, {"enabled": False})
    assert disabled[0] == [] and disabled[1] is None
    assert contacts[0] == [] and len(contacts[1]) == 1
    np.testing.assert_array_equal(contacts[2][0].point_local, annotated.point_local)
    assert all(
        len(frame) == 1 for frame in detect_contacts(motion, {"overwrite": True})
    )
    with pytest.raises(ValueError, match="unknown scene body"):
        detect_contacts(motion, {"pairs": {"unknown": ["palm"]}})
    with pytest.raises(ValueError, match="unknown source point"):
        detect_contacts(motion, {"pairs": {"body": ["unknown"]}})
    with pytest.raises(ValueError, match="Unknown contact_detection"):
        detect_contacts(motion, {"distance_thresh": 0.1})


@pytest.mark.parametrize("weighting", ["uniform", "exponential"])
def test_contact_graph_normalizes_combined_raw_contributions(weighting):
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    adjacency = [[1], [0, 2], [1]]
    weights = calculate_contact_edge_weights(
        vertices, adjacency, [(0, 2, 0.8)], 10, weighting, kappa=1
    )
    assert 2 in adjacency[0] and 0 in adjacency[2]
    assert weights[0][adjacency[0].index(2)] > 0.8
    assert all(np.isclose(row.sum(), 1) for row in weights)
    laplacian = calculate_laplacian_coordinates(
        vertices, adjacency, edge_weights=weights
    )
    matrix = calculate_laplacian_matrix(vertices, adjacency, edge_weights=weights)
    np.testing.assert_allclose(laplacian, matrix @ vertices)


def test_contact_mapping_body_identity_and_patch_aggregation(motion):
    frame = next(motion.iter_frames())
    frame.contacts = [
        Contact("palm", "box", [0.5, 0, 0], confidence=0.5),
        Contact("palm", "box", [0.5, 0, 0], confidence=0.8),
        Contact("palm", "box", [0.5, 0.1, 0], confidence=0.4),
        Contact("palm", "box", [100, 100, 100], confidence=0),
        Contact("palm", "cabinet/door", [0.5, 0, 0]),
    ]
    anchors, edges = resolve_contact_anchors(frame, ["palm", "knee"])
    assert len(anchors) == 2
    assert edges == [(0, 0, pytest.approx(0.6)), (0, 1, 1)]
    np.testing.assert_allclose(anchors[0], [0.5, 1 / 30, 1])
    np.testing.assert_allclose(anchors[1], [2.5, 0, 1])
    with pytest.raises(ValueError, match="'palm'.*mapping"):
        resolve_contact_anchors(frame, ["knee"])


@pytest.mark.parametrize("contact_weight", [10, 100])
def test_contact_patch_density_preserves_laplacian_objective(
    monkeypatch, contact_weight
):
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <body name="hand" pos="0 0 1"><freejoint/><geom type="sphere" size="0.1"/></body>
    </worldbody></mujoco>""")
    retargeter = GenericInteractionRetargeter(
        model,
        mujoco.MjData(model),
        create_flat_terrain(),
        {"palm": "hand"},
        1,
        terrain_sample_points=4,
        contact_edge_weight=contact_weight,
    )
    retargeter.terrain_points = np.array(
        [[x, y, 0] for x in [-2, 2] for y in [-2, 2]], dtype=float
    )
    frame = MotionFrame(
        positions=np.array([[0, 0, 1.0]]),
        target_names=["palm"],
        object_points=np.array([[1, 0, 1.0]]),
        scene=Scene(
            {"object": SceneEntity(static_pose=EntityPose([0, 0, 0], [1, 0, 0, 0]))}
        ),
    )
    coefficients = []

    def optimize(q_init, target_laplacian, matrix, kron, terrain_points, **kwargs):
        # For a source translation, all reverse environment rows contribute too.
        # The production objective gives every Laplacian residual weight 10.
        coefficients.append(10 * np.sum(matrix[:, 0].toarray() ** 2))
        return q_init

    monkeypatch.setattr(retargeter, "_optimize_configuration", optimize)
    for count in [1, 2, 8]:
        frame.contacts = [
            Contact("palm", "object", [1, 0.1 + i * 1e-5, 1]) for i in range(count)
        ]
        anchors, edges = resolve_contact_anchors(frame, ["palm"])
        retargeter.retarget_frame(
            frame.positions,
            model.qpos0,
            object_points=frame.object_points,
            contact_anchors=anchors,
            contact_edges=edges,
        )
    np.testing.assert_allclose(coefficients, coefficients[0], atol=1e-10, rtol=0)


def test_solver_uses_identical_anchor_order_and_frozen_graph(monkeypatch):
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <body name="hand" pos="0 0 1"><freejoint/><geom type="sphere" size="0.1"/></body>
    </worldbody></mujoco>""")
    data = mujoco.MjData(model)
    retargeter = GenericInteractionRetargeter(
        model,
        data,
        create_flat_terrain(),
        {"palm": "hand"},
        1,
        terrain_sample_points=8,
        contact_edge_weight=10,
    )
    source = np.array([[0, 0, 1.0]])
    objects = np.array([[1, 0, 1.0]])
    anchors = np.array([[1, 0.1, 1.0]])
    captured = {}

    def optimize(q_init, target_laplacian, matrix, kron, terrain_points, **kwargs):
        vertices = np.vstack([source, terrain_points, kwargs["object_points"]])
        np.testing.assert_allclose(target_laplacian, matrix @ vertices)
        np.testing.assert_array_equal(
            kwargs["object_points"], np.vstack([objects, anchors])
        )
        assert matrix.shape[0] == len(vertices)
        assert kron.shape == (3 * len(vertices), 3 * len(vertices))
        captured["vertices"] = vertices
        captured["matrix"] = matrix
        return q_init

    monkeypatch.setattr(retargeter, "_optimize_configuration", optimize)
    retargeter.retarget_frame(
        source,
        model.qpos0,
        object_points=objects,
        contact_anchors=anchors,
        contact_edges=[(0, 0, 1)],
    )
    assert abs(captured["matrix"][0, -1]) > 0.5
    assert len(captured["vertices"]) == 11  # source + terrain + object + anchor


def test_solver_environment_anchor_jacobians_are_zero(monkeypatch):
    import omniretargeting.retargeting as engine

    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <body name="hand" pos="0 0 1"><freejoint/><geom type="sphere" size="0.1"/></body>
    </worldbody></mujoco>""")
    retargeter = GenericInteractionRetargeter(
        model,
        mujoco.MjData(model),
        create_flat_terrain(),
        {"palm": "hand"},
        1,
        terrain_sample_points=8,
        contact_edge_weight=10,
    )
    original_stack = engine.sp.vstack
    environment_blocks = []

    def stack(blocks, *args, **kwargs):
        if len(blocks) == 2 and blocks[0].shape == (3, retargeter.nv_a):
            environment_blocks.append(blocks[1])
        return original_stack(blocks, *args, **kwargs)

    monkeypatch.setattr(engine.sp, "vstack", stack)
    output = retargeter.retarget_frame(
        np.array([[0, 0, 1.0]]),
        model.qpos0,
        max_iter=1,
        object_points=np.array([[1, 0, 1.0]]),
        contact_anchors=np.array([[1, 0.1, 1.0]]),
        contact_edges=[(0, 0, 1)],
    )
    assert np.isfinite(output).all()
    assert environment_blocks
    assert all(
        block.shape == (30, retargeter.nv_a) and block.nnz == 0
        for block in environment_blocks
    )
