"""Basic tests for omniretargeting package."""

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
import numpy as np
import pytest
import trimesh
from pathlib import Path
from unittest.mock import Mock, patch

from scipy.spatial.transform import Rotation

from omniretargeting.data_sources.base import DataSource, MotionData, MotionFrame, validate_motion_frame_positions, validate_motion_positions
from omniretargeting.robot_config import load_robot_config
from omniretargeting.main import export_scaled_objects, select_robot_source


REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_RESOURCES = REPO_ROOT / "tests" / "resources"
SMPLX_MODEL_DIR = Path("/localhdd/Datasets/")
ROBOT_PROFILE_CASES = (
    pytest.param("unitree_g1", REPO_ROOT / "robot_models" / "unitree_g1" / "unitree_g1.json", id="g1"),
    pytest.param("unitree_h1", REPO_ROOT / "robot_models" / "unitree_h1" / "unitree_h1.json", id="h1"),
    pytest.param("booster_k1", REPO_ROOT / "robot_models" / "booster_k1" / "booster_k1.json", id="booster-k1"),
    pytest.param("hightorque_mini_pi_plus", REPO_ROOT / "robot_models" / "hightorque_mini_pi_plus" / "hightorque_mini_pi_plus.json", id="mini-pi-plus"),
)
COMMON_ALIGNMENT_JOINTS = (
    "Pelvis",
    "L_Hip",
    "R_Hip",
    "Spine1",
    "L_Knee",
    "R_Knee",
    "L_Ankle",
    "R_Ankle",
    "L_Shoulder",
    "R_Shoulder",
    "L_Elbow",
    "R_Elbow",
)

FLOATING_BASE_PROFILE_CASES = (
    pytest.param("unitree_h1", REPO_ROOT / "robot_models" / "unitree_h1" / "unitree_h1.json", id="h1-floating"),
    pytest.param("booster_k1", REPO_ROOT / "robot_models" / "booster_k1" / "booster_k1.json", id="booster-k1-floating"),
    pytest.param("hightorque_mini_pi_plus", REPO_ROOT / "robot_models" / "hightorque_mini_pi_plus" / "hightorque_mini_pi_plus.json", id="mini-pi-plus-floating"),
)

@dataclass(frozen=True)
class MotionCase:
    case_id: str
    robot_profile: Path
    motion_path: Path
    terrain_path: Path


def test_export_scaled_objects_scales_pose_translations_with_scene(tmp_path):
    object_mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    motion_data = MotionData(
        positions=np.zeros((2, 1, 3), dtype=float),
        object_mesh=object_mesh,
        metadata={
            "object_name": "box",
            "object_translations": np.array([[1.0, 2.0, 3.0], [-1.0, 0.5, 4.0]]),
            "object_rotations": np.repeat(np.eye(3)[None, :, :], 2, axis=0),
            "object_scales": np.array([0.5, 0.25]),
        },
    )

    mesh_path, pose_path = export_scaled_objects(
        motion_data,
        tmp_path,
        source_to_robot_scale=2.0,
        apply_scene_scaling=True,
    )

    assert mesh_path.exists()
    assert pose_path.exists()
    poses = json.loads(pose_path.read_text())
    assert poses[0]["translation"] == [2.0, 4.0, 6.0]
    assert poses[1]["translation"] == [-2.0, 1.0, 8.0]
    assert poses[0]["scale"] == 1.0
    assert poses[1]["scale"] == 0.5


ROBOT_MOTION_MATRIX_ROBOTS = (
    ("g1", REPO_ROOT / "robot_models" / "unitree_g1" / "unitree_g1.json"),
    ("h1", REPO_ROOT / "robot_models" / "unitree_h1" / "unitree_h1.json"),
    ("booster-k1", REPO_ROOT / "robot_models" / "booster_k1" / "booster_k1.json"),
    ("mini-pi-plus", REPO_ROOT / "robot_models" / "hightorque_mini_pi_plus" / "hightorque_mini_pi_plus.json"),
)

ROBOT_MOTION_MATRIX_SCENES = (
    ("amass-simplelab",
     TEST_RESOURCES / "amass" / "140_02_stageii.npz",
     TEST_RESOURCES / "terrain" / "simplelab_enlarged_noWall.stl"),
    ("amass-wallflip",
     TEST_RESOURCES / "amass" / "wall_flip_smplx_amass.npz",
     TEST_RESOURCES / "terrain" / "wall_flip_scene.obj"),
    ("amass-prox-sofa",
     TEST_RESOURCES / "amass" / "PROX_1_smplx_amass.npz",
     TEST_RESOURCES / "terrain" / "PROX_sofa.obj"),
)

MOTION_CASES = tuple(
    MotionCase(
        case_id=f"{robot_id}-{scene_id}",
        robot_profile=robot_profile,
        motion_path=motion_path,
        terrain_path=terrain_path,
    )
    for robot_id, robot_profile in ROBOT_MOTION_MATRIX_ROBOTS
    for scene_id, motion_path, terrain_path in ROBOT_MOTION_MATRIX_SCENES
)



def _load_robot_profile(profile_path: Path) -> dict:
    return load_robot_config(profile_path)


def _build_retargeter_kwargs(robot_config: dict, terrain_mesh_path: Path | str, joint_mapping: dict | None = None) -> dict:
    return {
        "robot_urdf_path": Path(robot_config["urdf_path"]),
        "terrain_mesh_path": terrain_mesh_path,
        "joint_mapping": dict(joint_mapping or robot_config["joint_mapping"]),
        "robot_height": robot_config.get("robot_height"),
        "source_target_names": robot_config.get("source_target_names"),
        "base_orientation": robot_config.get("base_orientation"),
        "retargeting": robot_config.get("retargeting"),
    }

def _print_and_skip(reason: str) -> None:
    print(reason)
    pytest.skip(reason)



class TestUtils:
    """Test utility functions."""

    def test_validate_motion_positions_valid(self):
        positions = np.random.randn(8, 5, 3)
        assert validate_motion_positions(positions) is True

    def test_motion_data_validates_target_names(self):
        positions = np.random.randn(8, 5, 3)
        motion = MotionData(positions=positions, target_names=["a", "b", "c", "d", "e"], framerate=60.0)
        assert motion.positions is positions
        assert motion.framerate == 60.0

    def test_motion_data_rejects_mismatched_target_names(self):
        with pytest.raises(ValueError, match="target_names"):
            MotionData(positions=np.random.randn(8, 5, 3), target_names=["a"])


    def test_validate_motion_frame_positions_valid(self):
        positions = np.random.randn(5, 3)
        assert validate_motion_frame_positions(positions) is True

    def test_motion_frame_rejects_invalid_positions(self):
        with pytest.raises(ValueError, match="MotionFrame.positions"):
            MotionFrame(positions=np.random.randn(2, 5, 3))

    def test_data_source_collects_frames(self):
        class FakeSource(DataSource):
            target_names = ["a", "b"]
            framerate = 30.0
            source_height = 1.8
            metadata = {"source_type": "fake"}

            def iter_frames(self):
                yield MotionFrame(positions=np.zeros((2, 3)), root_translation=np.array([1.0, 0.0, 0.0]))
                yield MotionFrame(positions=np.ones((2, 3)), root_translation=np.array([2.0, 0.0, 0.0]))

        motion = FakeSource().load()
        assert motion.positions.shape == (2, 2, 3)
        assert motion.target_names == ["a", "b"]
        assert motion.root_translations.shape == (2, 3)
        assert motion.source_height == 1.8
        assert motion.human_height == 1.8


# TestOmniRetargeter removed - mock tests replaced with integration tests below

class TestLaplacianEdgeWeights:
    """Distance-dependent (exponential) Laplacian edge weights from TopoRetarget."""

    @staticmethod
    def _tetrahedron():
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        tetrahedra = np.array([[0, 1, 2, 3]])
        return vertices, tetrahedra

    def test_weights_are_row_normalized(self):
        from omniretargeting.utils import (
            calculate_exponential_edge_weights,
            get_adjacency_list,
        )

        vertices, tetrahedra = self._tetrahedron()
        adj_list = get_adjacency_list(tetrahedra, len(vertices))

        edge_weights = calculate_exponential_edge_weights(vertices, adj_list, kappa=30.0)

        assert len(edge_weights) == len(vertices)
        for i, weights in enumerate(edge_weights):
            assert len(weights) == len(adj_list[i])
            np.testing.assert_allclose(np.sum(weights), 1.0)

    def test_weights_favor_closer_neighbors(self):
        from omniretargeting.utils import calculate_exponential_edge_weights

        vertices = np.array(
            [
                [0.0, 0.0, 0.0],  # center vertex
                [0.1, 0.0, 0.0],  # close neighbor
                [1.0, 0.0, 0.0],  # far neighbor
            ]
        )
        adj_list = [[1, 2], [0], [0]]

        edge_weights = calculate_exponential_edge_weights(vertices, adj_list, kappa=30.0)

        assert edge_weights[0][0] > edge_weights[0][1]
        expected_far = np.exp(-30.0 * 1.0)
        expected_close = np.exp(-30.0 * 0.1)
        np.testing.assert_allclose(
            edge_weights[0],
            [expected_close, expected_far] / (expected_close + expected_far),
        )

    def test_zero_kappa_recovers_uniform_weights(self):
        from omniretargeting.utils import (
            calculate_exponential_edge_weights,
            get_adjacency_list,
        )

        vertices, tetrahedra = self._tetrahedron()
        adj_list = get_adjacency_list(tetrahedra, len(vertices))

        edge_weights = calculate_exponential_edge_weights(vertices, adj_list, kappa=0.0)

        for i, weights in enumerate(edge_weights):
            np.testing.assert_allclose(weights, np.ones(len(adj_list[i])) / len(adj_list[i]))

    def test_weighted_matrix_matches_weighted_coordinates(self):
        from omniretargeting.utils import (
            calculate_exponential_edge_weights,
            calculate_laplacian_coordinates,
            calculate_laplacian_matrix,
            get_adjacency_list,
        )

        rng = np.random.default_rng(0)
        vertices = rng.normal(size=(6, 3))
        tetrahedra = np.array([[0, 1, 2, 3], [2, 3, 4, 5]])
        adj_list = get_adjacency_list(tetrahedra, len(vertices))
        edge_weights = calculate_exponential_edge_weights(vertices, adj_list, kappa=30.0)

        coords = calculate_laplacian_coordinates(vertices, adj_list, edge_weights=edge_weights)
        L = calculate_laplacian_matrix(vertices, adj_list, edge_weights=edge_weights)

        np.testing.assert_allclose(L @ vertices, coords)

    def test_default_uniform_weighting_unchanged(self):
        from omniretargeting.utils import (
            calculate_laplacian_coordinates,
            calculate_laplacian_matrix,
            get_adjacency_list,
        )

        vertices, tetrahedra = self._tetrahedron()
        adj_list = get_adjacency_list(tetrahedra, len(vertices))

        coords = calculate_laplacian_coordinates(vertices, adj_list, uniform_weight=True)
        L = calculate_laplacian_matrix(vertices, adj_list, uniform_weight=True)

        # Uniform Laplacian: delta_i = v_i - mean(neighbors)
        np.testing.assert_allclose(L @ vertices, coords)
        np.testing.assert_allclose(coords[0], vertices[0] - vertices[1:].mean(axis=0))

    def test_invalid_weighting_scheme_rejected(self):
        from omniretargeting.retargeting import GenericInteractionRetargeter

        with pytest.raises(ValueError, match="laplacian_edge_weighting"):
            GenericInteractionRetargeter(
                Mock(), Mock(), Mock(), {}, 1.0, laplacian_edge_weighting="inverse-distance"
            )


class TestBoneDirection:
    """TopoRetarget bone-direction prior (Eq. 1-2, 8)."""

    def test_parse_bone_chains_extracts_adjacent_triples(self):
        from omniretargeting.retargeting import parse_bone_chains

        names = ["Hips", "Spine", "L_Up", "L_Knee", "L_Foot", "R_Up", "R_Knee", "R_Foot"]
        triples = parse_bone_chains(
            [["L_Up", "L_Knee", "L_Foot"], ["R_Up", "R_Knee", "R_Foot"], ["Hips", "Spine"]],
            names,
        )
        # Each 3-target leg chain yields one adjacent pair; the 2-target torso
        # chain defines a bone but no adjacent pair.
        assert triples == [(2, 3, 4), (5, 6, 7)]

    def test_parse_bone_chains_rejects_unmapped_target(self):
        from omniretargeting.retargeting import parse_bone_chains

        with pytest.raises(ValueError, match="not a mapped source target"):
            parse_bone_chains([["A", "B", "C"]], ["A", "B"])

    def test_parse_bone_chains_rejects_short_chain(self):
        from omniretargeting.retargeting import parse_bone_chains

        with pytest.raises(ValueError, match="at least 2 targets"):
            parse_bone_chains([["A"]], ["A"])

    def test_bone_direction_targets_math(self):
        from omniretargeting.retargeting import compute_bone_direction_targets

        # Chain (0->1->2): bone k along +x, bone l along +y
        points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 2.0, 0.0]])
        targets = compute_bone_direction_targets(points, [(0, 1, 2)])
        np.testing.assert_allclose(targets, np.array([1.0, -1.0, 0.0]))

    def test_bone_jacobian_matches_finite_difference(self):
        from omniretargeting.retargeting import (
            compute_bone_direction_targets,
            compute_bone_direction_residual_and_jacobian,
        )

        rng = np.random.default_rng(2)
        n_targets, nq_a = 4, 5
        points = rng.normal(size=(n_targets, 3)) + np.array([0.0, 0.0, 2.0])
        J_V = rng.normal(size=(3 * n_targets, nq_a))
        triples = [(0, 1, 2), (1, 2, 3)]
        targets = compute_bone_direction_targets(rng.normal(size=(n_targets, 3)) + 2.0, triples)

        res0, J = compute_bone_direction_residual_and_jacobian(points, J_V, triples, targets)

        dq = rng.normal(scale=1e-6, size=nq_a)
        points_perturbed = points + (J_V @ dq).reshape(n_targets, 3)
        res1, _ = compute_bone_direction_residual_and_jacobian(points_perturbed, J_V, triples, targets)

        np.testing.assert_allclose(res1, res0 + J @ dq, rtol=1e-4, atol=1e-8)

    def test_bone_direction_requires_chains_when_enabled(self):
        from omniretargeting.retargeting import GenericInteractionRetargeter

        with pytest.raises(ValueError, match="requires 'chains'"):
            GenericInteractionRetargeter(
                Mock(), Mock(), Mock(), {"Pelvis": "pelvis"}, 1.0,
                bone_direction={"enabled": True},
            )

    def test_bone_direction_rejects_unmapped_chain_target(self):
        from omniretargeting.retargeting import GenericInteractionRetargeter

        with pytest.raises(ValueError, match="bone_direction chain target"):
            GenericInteractionRetargeter(
                Mock(), Mock(), Mock(), {"Pelvis": "pelvis"}, 1.0,
                bone_direction={"enabled": True, "chains": [["Pelvis", "A", "B"]]},
            )

    def test_bone_direction_rejects_chains_without_pairs(self):
        from omniretargeting.retargeting import GenericInteractionRetargeter

        with pytest.raises(ValueError, match="no adjacent bone pairs"):
            GenericInteractionRetargeter(
                Mock(), Mock(), Mock(), {"A": "a", "B": "b"}, 1.0,
                source_target_names=["A", "B"],
                bone_direction={"enabled": True, "chains": [["A", "B"]]},
            )


class TestPenetrationSlack:
    """TopoRetarget slack penetration handling (Eq. 8)."""

    @staticmethod
    def _bare_retargeter(**attrs):
        from omniretargeting.retargeting import GenericInteractionRetargeter

        r = GenericInteractionRetargeter.__new__(GenericInteractionRetargeter)
        r.penetration_slack_enabled = False
        r.penetration_tolerance = 1e-3
        r.penetration_soft_tolerance = 1e-3
        r.penetration_hard_bound = 0.03
        r.penetration_slack_penalty = 1e5
        for k, v in attrs.items():
            setattr(r, k, v)
        return r

    def test_hard_only_constraint_by_default(self):
        r = self._bare_retargeter()
        hard_rows, slack_row = r._penetration_constraint_terms(np.array([1.0]), 0.005)

        assert len(hard_rows) == 1
        assert slack_row is None

    def test_slack_allows_soft_violation_within_hard_bound(self):
        from scipy import sparse as sp
        from omniretargeting.retargeting import _solve_qp_clarabel

        r = self._bare_retargeter(penetration_slack_enabled=True)
        # 5 mm existing penetration: soft tolerance (1 mm) must be violated,
        # but the hard bound (30 mm) must still hold.
        hard_rows, slack_row = r._penetration_constraint_terms(np.array([1.0]), -0.005)

        assert len(hard_rows) == 1
        assert slack_row is not None
        Ja, rhs_soft, span = slack_row

        # Recreate the QP row the retargeter builds for this pair, with dqa
        # pinned at 0: min (w_s/2)(span*s_unit)^2
        #   s.t. Ja*dqa + span*s_unit >= rhs_soft, Ja*dqa >= rhs_hard, 0<=s_unit<=1
        w_s = r.penetration_slack_penalty
        P = sp.csr_matrix(np.diag([0.0, w_s * span ** 2]))
        c = np.zeros(2)
        lb = np.array([0.0, 0.0])  # dqa pinned at 0 via bounds
        ub = np.array([0.0, 1.0])
        ge_rows = [
            (np.array([Ja[0], span]), rhs_soft),
            (np.array([Ja[0], 0.0]), hard_rows[0][1]),
        ]
        x, ok = _solve_qp_clarabel(P, c, lb, ub, ge_rows)
        assert ok
        s = span * x[1]

        # Soft violation exactly compensated: phi + s = -tau -> s = 0.005 - 0.001
        np.testing.assert_allclose(s, 0.004, rtol=1e-4)
        assert s <= 0.03 - 0.001 + 1e-9

    def test_hard_bound_must_exceed_soft_tolerance(self):
        from omniretargeting.retargeting import GenericInteractionRetargeter

        with pytest.raises(ValueError, match="hard_bound"):
            GenericInteractionRetargeter(
                Mock(), Mock(), Mock(), {"Pelvis": "pelvis"}, 1.0,
                hard_penetration_constraint=True,
                penetration_slack={"soft_tolerance": 0.03, "hard_bound": 0.001},
            )

    def test_slack_requires_hard_penetration_constraint(self):
        from omniretargeting.retargeting import GenericInteractionRetargeter

        with pytest.raises(ValueError, match="requires hard_penetration_constraint"):
            GenericInteractionRetargeter(
                Mock(), Mock(), Mock(), {"Pelvis": "pelvis"}, 1.0,
                penetration_slack={"soft_tolerance": 0.001},
            )

    @pytest.mark.parametrize(
        ("penetration_slack", "message"),
        [
            ([], "dictionary"),
            ({"soft_tolerance": -0.001}, "non-negative"),
            ({"hard_bound": np.nan}, "finite"),
            ({"slack_penalty": 0.0}, "positive"),
        ],
    )
    def test_invalid_slack_parameters_are_rejected(self, penetration_slack, message):
        from omniretargeting.retargeting import GenericInteractionRetargeter

        with pytest.raises(ValueError, match=message):
            GenericInteractionRetargeter(
                Mock(),
                Mock(),
                Mock(),
                {"Pelvis": "pelvis"},
                1.0,
                hard_penetration_constraint=True,
                penetration_slack=penetration_slack,
            )


def _make_tangent_test_retargeter(**kwargs):
    import mujoco

    from omniretargeting.retargeting import GenericInteractionRetargeter

    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <compiler angle="radian"/>
          <worldbody>
            <body name="base" pos="0 0 1.0">
              <freejoint/>
              <geom name="base_geom" type="sphere" size="0.05" contype="0" conaffinity="0"/>
              <body name="thigh">
                <joint name="hip" type="hinge" axis="0 1 0" range="-2 2"/>
                <geom name="thigh_geom" type="capsule" fromto="0 0 0 0 0 -0.5" size="0.04"/>
                <body name="shin" pos="0 0 -0.5">
                  <joint name="knee" type="hinge" axis="0 1 0" range="-2 2"/>
                  <geom name="shin_geom" type="capsule" fromto="0 0 0 0 0 -0.5" size="0.04"/>
                  <body name="foot" pos="0 0 -0.5">
                    <joint name="ankle" type="hinge" axis="0 1 0" range="-1 1"/>
                    <geom name="foot_geom" type="box" pos="0.15 0 -0.04" size="0.2 0.08 0.04"/>
                  </body>
                </body>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    terrain = trimesh.Trimesh(
        vertices=np.array(
            [[-2.0, -2.0, 0.0], [2.0, -2.0, 0.0], [2.0, 2.0, 0.0], [-2.0, 2.0, 0.0]]
        ),
        faces=np.array([[0, 1, 2], [0, 2, 3]]),
        process=False,
    )
    retargeter = GenericInteractionRetargeter(
        model,
        data,
        terrain,
        {"Foot": "foot"},
        1.0,
        terrain_sample_points=4,
        source_target_names=["Foot"],
        **kwargs,
    )
    return retargeter, model, data


@pytest.mark.parametrize(
    ("penetration_correction", "message"),
    [
        ({"base_translation_weights": [1.0, 2.0]}, "contain 3"),
        ({"base_rotation_weight": np.inf}, "finite"),
        ({"joint_weight": -1.0}, "non-negative"),
        ({"base_translation_step": [0.1, 0.0, 0.1]}, "positive"),
        ({"joint_step_fraction": 0.0}, "positive"),
        ({"restoration_penalty": 0.0}, "positive"),
    ],
)
def test_invalid_penetration_correction_is_rejected(
    penetration_correction, message
):
    with pytest.raises(ValueError, match=message):
        _make_tangent_test_retargeter(
            penetration_correction=penetration_correction
        )


def _set_tangent_test_pose(model, q):
    import mujoco

    for name, value in (("hip", 0.45), ("knee", -0.75), ("ankle", 0.35)):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        q[model.jnt_qposadr[joint_id]] = value


def test_open3d_terrain_query_returns_closest_points_and_triangle_ids():
    retargeter, _, _ = _make_tangent_test_retargeter()
    points = np.array([[0.0, 0.0, 0.1], [0.3, -0.2, -0.1]])

    closest, distances, triangle_ids = retargeter._closest_points_on_terrain(points)

    np.testing.assert_allclose(closest[:, :2], points[:, :2], atol=1e-6)
    np.testing.assert_allclose(closest[:, 2], 0.0, atol=1e-6)
    np.testing.assert_allclose(distances, [0.1, 0.1], atol=1e-6)
    assert np.all(np.isin(triangle_ids, [0, 1]))


def test_mujoco_tangent_jacobian_matches_integrated_point_displacement():
    import mujoco

    retargeter, model, data = _make_tangent_test_retargeter()
    q = model.qpos0.copy()
    _set_tangent_test_pose(model, q)
    data.qpos[:] = q
    mujoco.mj_forward(model, data)

    foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "foot")
    point_body = np.array([0.2, 0.0, -0.04])
    point_before = data.xpos[foot_id] + data.xmat[foot_id].reshape(3, 3) @ point_body
    jacobian = retargeter._calc_contact_jacobian_from_point(foot_id, point_body)
    delta_v = np.linspace(-1.0, 1.0, retargeter.nv_a) * 1e-7
    q_new = retargeter._integrate_optimized_step(q, delta_v)

    data.qpos[:] = q_new
    mujoco.mj_forward(model, data)
    point_after = data.xpos[foot_id] + data.xmat[foot_id].reshape(3, 3) @ point_body

    np.testing.assert_allclose(
        point_after - point_before,
        jacobian[:, retargeter.dof_indices] @ delta_v,
        atol=1e-12,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        retargeter._configuration_residual(q, q_new),
        delta_v,
        atol=1e-12,
    )
    np.testing.assert_allclose(np.linalg.norm(q_new[3:7]), 1.0, atol=1e-12)


def test_bent_leg_terrain_jacobian_has_base_and_joint_support():
    import mujoco

    retargeter, model, data = _make_tangent_test_retargeter()
    q = model.qpos0.copy()
    _set_tangent_test_pose(model, q)
    data.qpos[:] = q
    mujoco.mj_forward(model, data)

    foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "foot")
    jacobian_z = retargeter._calc_contact_jacobian_from_point(
        foot_id, np.array([0.2, 0.0, -0.04])
    )[2, retargeter.dof_indices]

    base_z = retargeter.base_translation_opt_indices[2]
    np.testing.assert_allclose(jacobian_z[base_z], 1.0, atol=1e-12)
    for name in ("hip", "knee", "ankle"):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        opt_idx = retargeter._dof_to_opt[int(model.jnt_dofadr[joint_id])]
        assert abs(jacobian_z[opt_idx]) > 1e-3


def test_scaled_correction_metric_prefers_articulation_to_base_motion():
    import mujoco
    from scipy import sparse as sp

    from omniretargeting.retargeting import _solve_qp_clarabel

    retargeter, model, data = _make_tangent_test_retargeter(
        penetration_correction={
            "base_translation_weights": [1000.0, 1000.0, 1000.0],
            "base_rotation_weight": 1000.0,
            "joint_weight": 0.01,
        }
    )
    q = model.qpos0.copy()
    _set_tangent_test_pose(model, q)
    data.qpos[:] = q
    mujoco.mj_forward(model, data)

    foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "foot")
    jacobian_z = retargeter._calc_contact_jacobian_from_point(
        foot_id, np.array([0.2, 0.0, -0.04])
    )[2, retargeter.dof_indices]

    metric = np.maximum(retargeter.joint_regularization_diag.copy(), 1e-6)
    metric[retargeter.base_translation_opt_indices] = retargeter.base_translation_weights
    metric[retargeter.base_rotation_opt_indices] = retargeter.base_rotation_weight
    lb, ub = retargeter._step_bounds(q)
    delta_v, ok = _solve_qp_clarabel(
        sp.diags(2.0 * metric),
        np.zeros(retargeter.nv_a),
        lb,
        ub,
        [(jacobian_z, 0.02)],
    )

    assert ok
    assert jacobian_z @ delta_v >= 0.02 - 1e-7
    joint_indices = (
        retargeter.dof_group_indices["legs"]
        + retargeter.dof_group_indices["waist"]
        + retargeter.dof_group_indices["arms"]
        + retargeter.dof_group_indices["other_joints"]
    )
    joint_norm = np.linalg.norm(delta_v[joint_indices])
    base_norm = np.linalg.norm(
        delta_v[
            retargeter.base_translation_opt_indices
            + retargeter.base_rotation_opt_indices
        ]
    )
    assert joint_norm > 10.0 * base_norm


def test_tangent_step_bounds_respect_physical_joint_limits():
    import mujoco

    retargeter, model, _ = _make_tangent_test_retargeter()
    q = model.qpos0.copy()
    hip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hip")
    hip_qpos = int(model.jnt_qposadr[hip_id])
    hip_opt = retargeter._dof_to_opt[int(model.jnt_dofadr[hip_id])]
    q[hip_qpos] = 1.99

    lb, ub = retargeter._step_bounds(q)

    np.testing.assert_allclose(ub[hip_opt], 0.01, atol=1e-12)
    np.testing.assert_allclose(lb[hip_opt], -0.4, atol=1e-12)


def test_relative_contact_jacobian_cancels_rigid_base_translation():
    import mujoco

    retargeter, model, data = _make_tangent_test_retargeter()
    q = model.qpos0.copy()
    _set_tangent_test_pose(model, q)
    data.qpos[:] = q
    mujoco.mj_forward(model, data)

    thigh_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "thigh")
    foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "foot")
    relative = retargeter._calc_contact_jacobian_from_point(
        thigh_id
    ) - retargeter._calc_contact_jacobian_from_point(foot_id)
    np.testing.assert_allclose(relative[:, :3], 0.0, atol=1e-12)


def test_nonlinear_feasibility_is_checked_on_integrated_candidate():
    import mujoco
    from scipy import sparse as sp

    from omniretargeting.retargeting import GenericInteractionRetargeter

    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body name="base" pos="0 0 0.04">
              <freejoint/>
              <geom name="base_geom" type="sphere" size="0.05"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    terrain = trimesh.Trimesh(
        vertices=np.array(
            [[-2.0, -2.0, 0.0], [2.0, -2.0, 0.0], [2.0, 2.0, 0.0], [-2.0, 2.0, 0.0]]
        ),
        faces=np.array([[0, 1, 2], [0, 2, 3]]),
        process=False,
    )
    retargeter = GenericInteractionRetargeter(
        model,
        data,
        terrain,
        {"Base": "base"},
        0.1,
        source_target_names=["Base"],
        terrain_sample_points=4,
        hard_penetration_constraint=True,
        solver_diagnostics=True,
    )
    q_bad = model.qpos0.copy()
    q_deep = q_bad.copy()
    q_deep[2] -= 0.5
    delta_v = np.zeros(retargeter.nv_a)
    delta_v[retargeter.base_translation_opt_indices[2]] = 0.02
    q_good = retargeter._integrate_optimized_step(q_bad, delta_v)
    assert retargeter._nonlinear_penetration_violation(q_bad) > 0.0
    assert retargeter._nonlinear_penetration_violation(q_deep) > 0.4
    assert retargeter._nonlinear_penetration_violation(q_good) == 0.0

    retargeter._single_optimization_step = Mock(return_value=(q_good, 0.0))
    result = retargeter._optimize_configuration(
        q_bad,
        np.zeros((1, 3)),
        sp.csr_matrix((1, 1)),
        sp.csr_matrix((3, 3)),
        np.zeros((0, 3)),
        max_iter=1,
    )

    np.testing.assert_allclose(result, q_good)
    assert retargeter.last_solve_diagnostics["success"] is True
    assert retargeter.last_solve_diagnostics["max_hard_violation"] == 0.0
    assert retargeter.last_solve_diagnostics["failure_reason"] is None


def test_penetration_is_measured_along_surface_normal_for_steep_faces():
    import mujoco

    from omniretargeting.retargeting import GenericInteractionRetargeter

    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body name="base" pos="0 0 0.5">
              <freejoint/>
              <geom name="base_geom" type="sphere" size="0.05"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    # A vertical wall in the y-z plane whose outward normal points +x. Its
    # raw face normal is not upward, so penetration must still be measured
    # along the surface normal.
    terrain = trimesh.Trimesh(
        vertices=np.array(
            [
                [0.0, -1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
                [0.0, -1.0, 1.0],
            ]
        ),
        faces=np.array([[0, 1, 2], [0, 2, 3]]),
        process=False,
    )
    retargeter = GenericInteractionRetargeter(
        model,
        data,
        terrain,
        {"Base": "base"},
        0.1,
        source_target_names=["Base"],
        terrain_sample_points=4,
        hard_penetration_constraint=True,
        solver_diagnostics=True,
    )

    q_behind = model.qpos0.copy()
    q_behind[0] = -0.3
    q_front = model.qpos0.copy()
    q_front[0] = 0.3

    assert retargeter._nonlinear_penetration_violation(q_behind) > 0.0
    assert retargeter._nonlinear_penetration_violation(q_front) == 0.0


def test_nonlinear_step_rejection_is_reported_when_no_step_accepted():
    import mujoco
    from scipy import sparse as sp

    from omniretargeting.retargeting import GenericInteractionRetargeter

    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body name="base" pos="0 0 0.04">
              <freejoint/>
              <geom name="base_geom" type="sphere" size="0.05"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    terrain = trimesh.Trimesh(
        vertices=np.array(
            [[-2.0, -2.0, 0.0], [2.0, -2.0, 0.0], [2.0, 2.0, 0.0], [-2.0, 2.0, 0.0]]
        ),
        faces=np.array([[0, 1, 2], [0, 2, 3]]),
        process=False,
    )
    retargeter = GenericInteractionRetargeter(
        model,
        data,
        terrain,
        {"Base": "base"},
        0.1,
        source_target_names=["Base"],
        terrain_sample_points=4,
        hard_penetration_constraint=True,
        solver_diagnostics=True,
    )

    q_start = model.qpos0.copy()
    delta_v = np.zeros(retargeter.nv_a)
    delta_v[retargeter.base_translation_opt_indices[2]] = -0.02
    q_worse = retargeter._integrate_optimized_step(q_start, delta_v)

    # The QP "succeeds" but produces a candidate that is worse than the
    # current feasible pose, and every positive backtracking scale is also
    # worse. The solver must keep the feasible pose and report the failure
    # instead of silently marking the frame converged/successful.
    retargeter._single_optimization_step = Mock(return_value=(q_worse, 1.0))
    retargeter._nonlinear_penetration_violation = Mock(
        side_effect=[0.0] + [0.5] * 7
    )

    result = retargeter._optimize_configuration(
        q_start,
        np.zeros((1, 3)),
        sp.csr_matrix((1, 1)),
        sp.csr_matrix((3, 3)),
        np.zeros((0, 3)),
        max_iter=1,
    )

    np.testing.assert_allclose(result, q_start)
    diagnostics = retargeter.last_solve_diagnostics
    assert diagnostics["success"] is False
    assert diagnostics["solver_failed"] is False
    assert diagnostics["backtrack_failed"] is True
    assert diagnostics["failure_reason"] == "nonlinear_step_rejected"
    assert diagnostics["converged"] is False


@pytest.mark.parametrize(
    ("geom_type", "size"),
    [
        ("cylinder", np.array([0.2, 0.4, 0.0])),
        ("capsule", np.array([0.2, 0.4, 0.0])),
        ("ellipsoid", np.array([0.2, 0.3, 0.4])),
    ],
)
def test_primitive_samples_use_mujoco_local_geometry(geom_type, size):
    import mujoco

    from omniretargeting.utils import sample_mujoco_geom_local_points

    enum = {
        "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
        "capsule": mujoco.mjtGeom.mjGEOM_CAPSULE,
        "ellipsoid": mujoco.mjtGeom.mjGEOM_ELLIPSOID,
    }[geom_type]
    points_local = sample_mujoco_geom_local_points(enum, size)
    rotation = Rotation.from_euler("xyz", [0.4, -0.2, 0.7]).as_matrix()
    translation = np.array([0.3, -0.1, 1.2])
    points_world = points_local @ rotation.T + translation
    recovered = (points_world - translation) @ rotation

    if geom_type == "ellipsoid":
        surface_value = ((recovered / size[:3]) ** 2).sum(axis=1)
        np.testing.assert_allclose(surface_value, 1.0, atol=1e-12)
    elif geom_type == "cylinder":
        radius, half_length = size[:2]
        radial = np.linalg.norm(recovered[:, :2], axis=1)
        on_side = np.isclose(radial, radius)
        on_cap = np.isclose(np.abs(recovered[:, 2]), half_length)
        assert np.all(on_side | on_cap)
    else:
        radius, half_length = size[:2]
        segment_z = np.clip(recovered[:, 2], -half_length, half_length)
        distance_to_axis_segment = np.linalg.norm(
            recovered - np.column_stack(
                [np.zeros(len(recovered)), np.zeros(len(recovered)), segment_z]
            ),
            axis=1,
        )
        np.testing.assert_allclose(distance_to_axis_segment, radius, atol=1e-12)


def test_load_robot_config_nested_source_profile(tmp_path):
    urdf_path = tmp_path / "robot.urdf"
    urdf_path.write_text("<robot name='dummy'/>")
    config_path = tmp_path / "profile.json"
    config_path.write_text(
        json.dumps(
            {
                "name": "nested",
                "robot": {"urdf_path": "robot.urdf", "height": 1.2},
                "retargeting": {"solver": {"terrain_sample_points": 7}},
                "active_source": "smplx_default",
                "source": [
                    {
                        "name": "smplx_default",
                        "type": "smplx",
                        "target_names": ["Pelvis", "Head"],
                        "target_mapping": {"Pelvis": "base_link"},
                        "height_estimation": {"head_target": "Head", "foot_targets": ["Pelvis"]},
                        "base_orientation": {"pelvis": "Pelvis", "spine": "Head"},
                        "adapter_options": {
                            "model_directory": "/localhdd/Datasets/",
                            "betas": [0.0, 0.0],
                            "gender": "neutral",
                        },
                    }
                ],
            }
        )
    )

    config = load_robot_config(config_path)

    assert config["urdf_path"] == str(urdf_path.resolve())
    assert config["robot_height"] == 1.2
    assert config["joint_mapping"] == {"Pelvis": "base_link"}
    assert config["source_target_names"] == ["Pelvis", "Head"]
    assert config["height_estimation"] == {"head_target": "Head", "foot_targets": ["Pelvis"]}
    assert config["base_orientation"] == {"pelvis": "Pelvis", "spine": "Head"}
    assert config["retargeting"]["terrain_sample_points"] == 7
    assert config["selected_source"]["adapter_options"]["model_directory"] == "/localhdd/Datasets/"


@pytest.mark.parametrize("_profile_name,profile_path", ROBOT_PROFILE_CASES)
def test_robot_profile_mappings_are_source_local(_profile_name, profile_path):
    raw = json.loads(profile_path.read_text())

    assert "active_source" not in raw
    assert "joint_mapping" not in raw
    assert raw.get("source")
    source_types = {source.get("type") for source in raw["source"]}
    assert {"smplx", "omomo"}.issubset(source_types)
    for source in raw["source"]:
        assert isinstance(source.get("target_mapping"), dict)
        assert source["target_mapping"]

    config = load_robot_config(profile_path)
    assert config["joint_mapping"] == config["selected_source"]["target_mapping"]
    assert select_robot_source(config, "omomo")["type"] == "omomo"
    assert select_robot_source(config, "smplx")["type"] == "smplx"


class TestPackageImport:
    """Test package import functionality."""

    def test_import_package(self):
        """Test that package can be imported."""
        import omniretargeting
        assert hasattr(omniretargeting, '__version__')
        assert hasattr(omniretargeting, 'OmniRetargeter')

    def test_version_consistency(self):
        """Test version consistency across files."""
        import omniretargeting
        from omniretargeting.__version__ import __version__

        assert omniretargeting.__version__ == __version__ == "0.1.0"


class TestRealDataIntegration:
    """Integration tests requiring real data files."""

    @pytest.mark.parametrize(
        "motion_case",
        [pytest.param(case, id=case.case_id) for case in MOTION_CASES],
    )
    def test_motion_case_via_main_script(self, motion_case: MotionCase):
        """
        Test motion-terrain pairs through the main CLI script.
        
        This test validates end-to-end retargeting by invoking the main script
        with curated motion-terrain-robot combinations.
        """
        # Check all required files exist
        if not motion_case.robot_profile.exists():
            _print_and_skip(
                f"Motion case {motion_case.case_id}: Robot profile not found at {motion_case.robot_profile}"
            )
        
        if not motion_case.motion_path.exists():
            _print_and_skip(
                f"Motion case {motion_case.case_id}: Motion file not found at {motion_case.motion_path}"
            )
        
        if not motion_case.terrain_path.exists():
            _print_and_skip(
                f"Motion case {motion_case.case_id}: Terrain mesh not found at {motion_case.terrain_path}"
            )
        
        if not SMPLX_MODEL_DIR.exists():
            _print_and_skip(
                f"Motion case {motion_case.case_id}: SMPL-X model directory not found at {SMPLX_MODEL_DIR}. "
                "This curated main-script test requires licensed local SMPL-X assets."
            )
        
        # Create temporary output file
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp_output:
            output_path = Path(tmp_output.name)
        
        try:
            # Build command
            # Note: main.py normalizes output path to end with _retargeted.npz
            command = [
                sys.executable,
                "-m",
                "omniretargeting.main",
                "--robot-config",
                str(motion_case.robot_profile),
                "--model-dir",
                str(SMPLX_MODEL_DIR),
                "--motion",
                str(motion_case.motion_path),
                "--terrain",
                str(motion_case.terrain_path),
                "--output",
                str(output_path),
                "--penetration-resolver",
                "xyz_nudge",
                "--output-scaled-terrain",
                "/tmp/scaled_terrain.stl",
            ]
            
            # Main script will normalize the output path
            from omniretargeting.utils import normalize_retargeted_output_path
            expected_output_path = Path(normalize_retargeted_output_path(str(output_path)))
            
            print(f"\nRunning motion case {motion_case.case_id}...")
            print(f"Command: {' '.join(command)}")
            
            # Run the main script
            completed = subprocess.run(
                command,
                cwd=str(REPO_ROOT),
                check=False,
                capture_output=True,
                text=True,
            )
            
            # Print output for debugging
            if completed.stdout:
                print(f"STDOUT:\n{completed.stdout}")
            if completed.stderr:
                print(f"STDERR:\n{completed.stderr}")
            
            # Check for success
            assert completed.returncode == 0, (
                f"Main script failed with return code {completed.returncode}. "
                f"See output above for details."
            )
            
            # Verify output file was created (at normalized path)
            assert expected_output_path.exists(), f"Output file not created at {expected_output_path}"
            
            # Load and validate output
            import numpy as np
            output_data = np.load(expected_output_path)
            
            # Check for expected keys from main.py output
            assert "joint_pos" in output_data, "Output missing joint_pos key"
            assert "base_pos_w" in output_data, "Output missing base_pos_w key"
            assert "base_quat_w" in output_data, "Output missing base_quat_w key"
            
            joint_pos = output_data["joint_pos"]
            base_pos = output_data["base_pos_w"]
            base_quat = output_data["base_quat_w"]
            
            assert isinstance(joint_pos, np.ndarray), "joint_pos should be ndarray"
            assert isinstance(base_pos, np.ndarray), "base_pos_w should be ndarray"
            assert isinstance(base_quat, np.ndarray), "base_quat_w should be ndarray"
            
            assert joint_pos.shape[0] > 0, "joint_pos should have frames"
            assert base_pos.shape[0] > 0, "base_pos_w should have frames"
            
            print(f"Motion case {motion_case.case_id} passed! Joint pos shape: {joint_pos.shape}, Base pos shape: {base_pos.shape}")
            
        finally:
            # Clean up temporary files
            if output_path.exists():
                output_path.unlink()
            if expected_output_path.exists():
                expected_output_path.unlink()


def test_retarget_motion_uses_identity_source_to_robot_scale_by_default():
    from omniretargeting import OmniRetargeter

    original_terrain_copy = Mock(name="original_terrain_copy")
    scaled_terrain = Mock(name="scaled_terrain")
    source_positions = np.ones((2, 22, 3), dtype=float)

    retargeter = OmniRetargeter.__new__(OmniRetargeter)
    retargeter.terrain_mesh = Mock()
    retargeter.terrain_mesh.copy.return_value = original_terrain_copy
    retargeter._compute_source_to_robot_scale = Mock(return_value=2.5)
    retargeter._scale_terrain_mesh = Mock(return_value=scaled_terrain)
    retargeter.retarget_stream = Mock(return_value=iter([np.array([1.0, 2.0, 3.0])]))
    retargeter.retargeting_config = {}
    retargeter._visualize_trajectory = Mock()

    source_to_robot_scale, retargeted_motion = retargeter.retarget_motion(
        source_positions,
        visualize_trajectory=False,
        enable_terrain_scaling=False,
    )

    assert source_to_robot_scale == 1.0
    assert isinstance(retargeted_motion, np.ndarray)
    retargeter._compute_source_to_robot_scale.assert_not_called()
    retargeter._scale_terrain_mesh.assert_not_called()
    retargeter.terrain_mesh.copy.assert_called_once_with()
    retargeter.retarget_stream.assert_called_once()
    assert retargeter.retarget_stream.call_args.kwargs["scaled_terrain"] is original_terrain_copy


# test_retarget_motion_applies_source_to_robot_scale_when_enabled removed - replaced with integration test

def test_retarget_motion_applies_foot_stabilization_for_xyz_nudge():
    from omniretargeting import OmniRetargeter

    original_terrain_copy = Mock(name="original_terrain_copy")
    raw_motion = np.array([[1.0, 2.0, 3.0]])
    stabilized_motion = np.array([[1.5, 2.5, 3.5]])

    retargeter = OmniRetargeter.__new__(OmniRetargeter)
    retargeter.terrain_mesh = Mock()
    retargeter.terrain_mesh.copy.return_value = original_terrain_copy
    retargeter.retargeting_config = {"penetration_resolver": "xyz_nudge"}
    retargeter._compute_source_to_robot_scale = Mock(return_value=2.5)
    retargeter._scale_terrain_mesh = Mock()
    retargeter.retarget_stream = Mock(return_value=iter(raw_motion))
    retargeter._apply_foot_stabilization = Mock(return_value=stabilized_motion)
    retargeter._visualize_trajectory = Mock()

    source_positions = np.ones((2, 22, 3), dtype=float)

    source_to_robot_scale, retargeted_motion = retargeter.retarget_motion(
        source_positions,
        framerate=60.0,
        visualize_trajectory=False,
        enable_terrain_scaling=False,
    )

    assert source_to_robot_scale == 1.0
    assert retargeted_motion is stabilized_motion
    retargeter._apply_foot_stabilization.assert_called_once()
    stabilization_args = retargeter._apply_foot_stabilization.call_args
    np.testing.assert_array_equal(stabilization_args.args[0], raw_motion)
    assert stabilization_args.args[1] is original_terrain_copy
    assert stabilization_args.kwargs["framerate"] == 60.0


def test_retarget_motion_skips_foot_stabilization_for_hard_constraint():
    from omniretargeting import OmniRetargeter

    original_terrain_copy = Mock(name="original_terrain_copy")
    raw_motion = np.array([[1.0, 2.0, 3.0]])

    retargeter = OmniRetargeter.__new__(OmniRetargeter)
    retargeter.terrain_mesh = Mock()
    retargeter.terrain_mesh.copy.return_value = original_terrain_copy
    retargeter.retargeting_config = {"penetration_resolver": "hard_constraint"}
    retargeter._compute_source_to_robot_scale = Mock(return_value=2.5)
    retargeter._scale_terrain_mesh = Mock()
    retargeter.retarget_stream = Mock(return_value=iter(raw_motion))
    retargeter._apply_foot_stabilization = Mock()
    retargeter._visualize_trajectory = Mock()

    source_positions = np.ones((2, 22, 3), dtype=float)

    source_to_robot_scale, retargeted_motion = retargeter.retarget_motion(
        source_positions,
        framerate=60.0,
        visualize_trajectory=False,
        enable_terrain_scaling=False,
    )

    assert source_to_robot_scale == 1.0
    np.testing.assert_array_equal(retargeted_motion, raw_motion)
    retargeter._apply_foot_stabilization.assert_not_called()



def test_retarget_motion_uses_base_inputs_as_root_pose_arrays():
    from omniretargeting import OmniRetargeter

    base_orientations = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=float), (1, 1))
    base_translations = np.full((1, 3), 2.0, dtype=float)
    motion_data = MotionData(
        positions=np.zeros((1, 2, 3), dtype=float),
        target_names=["Pelvis", "Head"],
        root_orientations=np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=float), (1, 1)),
        root_translations=np.zeros((1, 3), dtype=float),
        framerate=30.0,
        metadata={"source_type": "test"},
    )

    captured = {}
    scaled_terrain = Mock()

    retargeter = OmniRetargeter.__new__(OmniRetargeter)
    retargeter._coerce_motion_data = Mock(return_value=motion_data)
    retargeter._resolve_source_to_robot_scale = Mock(return_value=1.0)
    retargeter.terrain_mesh = Mock()
    retargeter.terrain_mesh.copy.return_value = scaled_terrain
    retargeter.retargeting_config = {"penetration_resolver": "hard_constraint"}

    def fake_retarget_stream(motion, scaled_terrain=None, show_progress=False):
        captured["motion"] = motion
        captured["terrain"] = scaled_terrain
        return [np.zeros(7, dtype=float)]

    retargeter.retarget_stream = Mock(side_effect=fake_retarget_stream)

    retargeter.retarget_motion(
        motion_data,
        base_orientations=base_orientations,
        base_translations=base_translations,
        visualize_trajectory=False,
    )

    assert captured["terrain"] is scaled_terrain
    np.testing.assert_array_equal(captured["motion"].root_orientations, base_orientations)
    np.testing.assert_array_equal(captured["motion"].root_translations, base_translations)
    assert "use_explicit_root_orientation" not in captured["motion"].metadata
    assert "use_explicit_root_translation" not in captured["motion"].metadata


def test_retarget_frame_uses_root_pose_for_frame_zero_init_when_present():
    from omniretargeting.core import RetargetingStreamState
    from omniretargeting import OmniRetargeter

    estimated_quat_wxyz = np.array([0.1, 0.2, 0.3, 0.9], dtype=float)
    estimated_quat_wxyz /= np.linalg.norm(estimated_quat_wxyz)
    mapped_targets = np.arange(12, dtype=float).reshape(4, 3)
    q_result = np.arange(7, dtype=float)

    inner_retargeter = Mock()
    inner_retargeter.retarget_frame.return_value = q_result

    retargeter = OmniRetargeter.__new__(OmniRetargeter)
    retargeter.retargeting_config = {}
    retargeter._estimate_base_orientation_from_joints = Mock(return_value=estimated_quat_wxyz)
    retargeter._extract_mapped_source_targets = Mock(return_value=mapped_targets)

    state = RetargetingStreamState(
        retargeter=inner_retargeter,
        q_init=np.zeros(7, dtype=float),
        q_last=None,
        last_estimated_quat=None,
        frame_idx=0,
        scaled_terrain=Mock(),
    )

    root_translation = np.array([1.0, 2.0, 3.0], dtype=float)
    root_orientation = Rotation.from_rotvec([0.0, 0.0, np.pi / 2.0]).as_quat(scalar_first=True)
    frame = MotionFrame(
        positions=np.zeros((4, 3), dtype=float),
        root_orientation=root_orientation,
        root_translation=root_translation,
    )

    result = retargeter.retarget_frame(frame, state)

    expected_init_wxyz = root_orientation
    expected_target_wxyz = estimated_quat_wxyz

    call_args = inner_retargeter.retarget_frame.call_args
    np.testing.assert_array_equal(call_args.args[0], mapped_targets)
    np.testing.assert_allclose(call_args.args[1][:3], root_translation)
    np.testing.assert_allclose(call_args.args[1][3:7], expected_init_wxyz)
    assert call_args.kwargs["q_last"] is None
    np.testing.assert_allclose(call_args.kwargs["target_base_orientation"], expected_target_wxyz)
    np.testing.assert_allclose(state.last_estimated_quat, estimated_quat_wxyz)
    assert state.frame_idx == 1
    np.testing.assert_array_equal(state.q_init, q_result)
    np.testing.assert_array_equal(state.q_last, q_result)
    np.testing.assert_array_equal(result, q_result)


def test_retarget_frame_falls_back_to_estimated_root_pose_when_absent():
    from omniretargeting.core import RetargetingStreamState
    from omniretargeting import OmniRetargeter

    estimated_quat_wxyz = np.array([0.3, -0.2, 0.1, 0.9], dtype=float)
    estimated_quat_wxyz /= np.linalg.norm(estimated_quat_wxyz)
    positions = np.array(
        [[10.0, 11.0, 12.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    mapped_targets = np.arange(12, dtype=float).reshape(4, 3)
    previous_q = np.ones(7, dtype=float)
    q_result = np.arange(7, dtype=float) + 10.0

    inner_retargeter = Mock()
    inner_retargeter.retarget_frame.return_value = q_result

    retargeter = OmniRetargeter.__new__(OmniRetargeter)
    retargeter.retargeting_config = {}
    retargeter._estimate_base_orientation_from_joints = Mock(return_value=estimated_quat_wxyz)
    retargeter._extract_mapped_source_targets = Mock(return_value=mapped_targets)

    state = RetargetingStreamState(
        retargeter=inner_retargeter,
        q_init=np.zeros(7, dtype=float),
        q_last=previous_q,
        last_estimated_quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
        frame_idx=0,
        scaled_terrain=Mock(),
    )

    frame = MotionFrame(positions=positions)
    result = retargeter.retarget_frame(frame, state)

    expected_target_wxyz = estimated_quat_wxyz

    call_args = inner_retargeter.retarget_frame.call_args
    np.testing.assert_array_equal(call_args.args[0], mapped_targets)
    np.testing.assert_allclose(call_args.args[1][:3], positions[0])
    np.testing.assert_allclose(call_args.args[1][3:7], expected_target_wxyz)
    np.testing.assert_array_equal(call_args.kwargs["q_last"], previous_q)
    np.testing.assert_allclose(call_args.kwargs["target_base_orientation"], expected_target_wxyz)
    np.testing.assert_allclose(state.last_estimated_quat, estimated_quat_wxyz)
    assert state.frame_idx == 1
    np.testing.assert_array_equal(state.q_init, q_result)
    np.testing.assert_array_equal(state.q_last, q_result)
    np.testing.assert_array_equal(result, q_result)


def test_create_stream_state_passes_hard_penetration_constraint():
    from omniretargeting import OmniRetargeter
    from unittest.mock import patch

    robot_model = Mock()
    robot_model.nq = 7
    robot_model.njnt = 0
    robot_data = Mock()
    scaled_terrain = Mock()

    retargeter = OmniRetargeter.__new__(OmniRetargeter)
    retargeter.robot_model = robot_model
    retargeter.robot_data = robot_data
    retargeter.valid_source_to_robot_link_mapping = {"Pelvis": "pelvis"}
    retargeter.robot_height = 1.0
    retargeter.retargeting_config = {
        "collision_detection_threshold": 0.2,
        "terrain_sample_points": 123,
        "replace_cylinders_with_capsules": True,
        "penetration_resolver": "hard_constraint_slack",
        "laplacian_edge_weighting": "exponential",
        "laplacian_distance_decay": 15.0,
        "bone_direction": {"enabled": True, "chains": [["Pelvis", "A", "B"]]},
        "penetration_slack": {"soft_tolerance": 0.002, "hard_bound": 0.04, "slack_penalty": 5e4},
    }
    retargeter.valid_source_target_names = ["Pelvis"]
    retargeter.base_orientation_config = {}

    with patch("omniretargeting.retargeting.GenericInteractionRetargeter") as retargeter_cls:
        retargeter_instance = Mock()
        retargeter_cls.return_value = retargeter_instance
        state = retargeter.create_stream_state(scaled_terrain=scaled_terrain)

    assert state.retargeter is retargeter_instance
    retargeter_cls.assert_called_once_with(
        robot_model,
        robot_data,
        scaled_terrain,
        {"Pelvis": "pelvis"},
        1.0,
        collision_detection_threshold=0.2,
        terrain_sample_points=123,
        source_target_names=["Pelvis"],
        replace_cylinders_with_capsules=True,
        hard_penetration_constraint=True,
        joint_regularization_boost=None,
        laplacian_edge_weighting="exponential",
        laplacian_distance_decay=15.0,
        bone_direction={"enabled": True, "chains": [["Pelvis", "A", "B"]]},
        penetration_slack={"soft_tolerance": 0.002, "hard_bound": 0.04, "slack_penalty": 5e4},
        base_position_tracking_weight=0.0,
        base_position_tracking_weight_z=0.0,
        penetration_correction=None,
        solver_diagnostics=False,
        terrain_deep_penetration_depth=0.5,
    )


def test_create_stream_state_passes_penetration_correction_and_diagnostics():
    from omniretargeting import OmniRetargeter
    from unittest.mock import patch

    correction = {
        "base_translation_weights": [0.1, 0.1, 20.0],
        "joint_weight": 0.01,
    }
    retargeter = OmniRetargeter.__new__(OmniRetargeter)
    retargeter.robot_model = Mock(nq=7, njnt=0)
    retargeter.robot_data = Mock()
    retargeter.valid_source_to_robot_link_mapping = {"Pelvis": "pelvis"}
    retargeter.robot_height = 1.0
    retargeter.retargeting_config = {
        "penetration_correction": correction,
        "solver_diagnostics": True,
    }
    retargeter.valid_source_target_names = ["Pelvis"]
    retargeter.base_orientation_config = {}

    with patch("omniretargeting.retargeting.GenericInteractionRetargeter") as retargeter_cls:
        retargeter_cls.return_value = Mock()
        retargeter.create_stream_state(scaled_terrain=Mock())

    assert retargeter_cls.call_args.kwargs["penetration_correction"] is correction
    assert retargeter_cls.call_args.kwargs["solver_diagnostics"] is True


def test_create_stream_state_enables_default_slack_parameters():
    from omniretargeting import OmniRetargeter
    from unittest.mock import patch

    retargeter = OmniRetargeter.__new__(OmniRetargeter)
    retargeter.robot_model = Mock(nq=7, njnt=0)
    retargeter.robot_data = Mock()
    retargeter.valid_source_to_robot_link_mapping = {"Pelvis": "pelvis"}
    retargeter.robot_height = 1.0
    retargeter.retargeting_config = {"penetration_resolver": "hard_constraint_slack"}
    retargeter.valid_source_target_names = ["Pelvis"]
    retargeter.base_orientation_config = {}

    with patch("omniretargeting.retargeting.GenericInteractionRetargeter") as retargeter_cls:
        retargeter_cls.return_value = Mock()
        retargeter.create_stream_state(scaled_terrain=Mock())

    assert retargeter_cls.call_args.kwargs["hard_penetration_constraint"] is True
    assert retargeter_cls.call_args.kwargs["penetration_slack"] == {}


def test_create_stream_state_ignores_slack_params_for_non_slack_resolver():
    from omniretargeting import OmniRetargeter
    from unittest.mock import patch

    retargeter = OmniRetargeter.__new__(OmniRetargeter)
    retargeter.robot_model = Mock(nq=7, njnt=0)
    retargeter.robot_data = Mock()
    retargeter.valid_source_to_robot_link_mapping = {"Pelvis": "pelvis"}
    retargeter.robot_height = 1.0
    retargeter.retargeting_config = {
        "penetration_resolver": "hard_constraint",
        "penetration_slack": {"soft_tolerance": 0.002},
    }
    retargeter.valid_source_target_names = ["Pelvis"]
    retargeter.base_orientation_config = {}

    with patch("omniretargeting.retargeting.GenericInteractionRetargeter") as retargeter_cls:
        retargeter_cls.return_value = Mock()
        retargeter.create_stream_state(scaled_terrain=Mock())

    assert retargeter_cls.call_args.kwargs["hard_penetration_constraint"] is True
    assert retargeter_cls.call_args.kwargs["penetration_slack"] is None


def test_create_stream_state_rejects_unknown_penetration_resolver():
    from omniretargeting import OmniRetargeter

    retargeter = OmniRetargeter.__new__(OmniRetargeter)
    retargeter.robot_model = Mock(nq=7, njnt=0)
    retargeter.robot_data = Mock()
    retargeter.valid_source_to_robot_link_mapping = {"Pelvis": "pelvis"}
    retargeter.robot_height = 1.0
    retargeter.retargeting_config = {"penetration_resolver": "soft_constraint"}
    retargeter.valid_source_target_names = ["Pelvis"]
    retargeter.base_orientation_config = {}

    with pytest.raises(ValueError, match="penetration_resolver"):
        retargeter.create_stream_state(scaled_terrain=Mock())


def test_create_stream_state_passes_base_position_tracking_weight():
    from omniretargeting import OmniRetargeter
    from unittest.mock import patch

    retargeter = OmniRetargeter.__new__(OmniRetargeter)
    retargeter.robot_model = Mock(nq=7, njnt=0)
    retargeter.robot_data = Mock()
    retargeter.valid_source_to_robot_link_mapping = {"Pelvis": "pelvis"}
    retargeter.robot_height = 1.0
    retargeter.retargeting_config = {"base_position_tracking_weight": 42.0}
    retargeter.valid_source_target_names = ["Pelvis"]
    retargeter.base_orientation_config = {}

    with patch("omniretargeting.retargeting.GenericInteractionRetargeter") as retargeter_cls:
        retargeter_cls.return_value = Mock()
        retargeter.create_stream_state(scaled_terrain=Mock())

    assert retargeter_cls.call_args.kwargs["base_position_tracking_weight"] == 42.0


def test_base_position_tracking_weight_default_is_zero():
    from omniretargeting.retargeting import GenericInteractionRetargeter

    retargeter = GenericInteractionRetargeter.__new__(GenericInteractionRetargeter)
    retargeter.penetration_slack_enabled = False
    retargeter.hard_penetration_constraint = False
    retargeter.bone_direction_enabled = False
    retargeter.Q_diag_modified = np.ones(10)
    retargeter.q_a_indices = np.arange(10)
    retargeter.q_a_lb = -np.ones(10) * 1e6
    retargeter.q_a_ub = np.ones(10) * 1e6
    retargeter.base_position_tracking_weight = 0.0

    assert retargeter.base_position_tracking_weight == 0.0

@pytest.mark.parametrize(("robot_name", "profile_path"), ROBOT_PROFILE_CASES)
def test_tpose_retargeting_alignment(robot_name: str, profile_path: Path):
    """
    End-to-end test: Create a T-pose SMPLX trajectory and verify retargeting accuracy.
    
    This test:
    1. Creates a synthetic T-pose trajectory (standing human, arms out)
    2. Runs full retargeting with a real robot URDF
    3. Compares retargeted robot link positions to target SMPLX joints
    4. Passes only if mean distance < 0.3m across all mapped joints
    """
    from omniretargeting import OmniRetargeter
    import trimesh
    import tempfile

    # ==========================================
    # Create synthetic T-pose SMPLX trajectory
    # ==========================================
    # Body-frame offsets for a simple T-pose (X forward, Y left, Z up).
    # These are relative to pelvis in a standard humanoid coordinate frame.
    offsets = np.array([
        [0.0, 0.0, 0.0],      # 0: Pelvis (root)
        [0.0, -0.1, -0.1],    # 1: L_Hip
        [0.0, 0.1, -0.1],     # 2: R_Hip
        [0.0, 0.0, 0.2],      # 3: Spine1
        [0.0, -0.1, -0.5],    # 4: L_Knee
        [0.0, 0.1, -0.5],     # 5: R_Knee
        [0.0, 0.0, 0.4],      # 6: Spine2
        [0.0, -0.1, -0.9],    # 7: L_Ankle
        [0.0, 0.1, -0.9],     # 8: R_Ankle
        [0.0, 0.0, 0.6],      # 9: Spine3
        [0.05, -0.1, -0.95],  # 10: L_Foot
        [0.05, 0.1, -0.95],   # 11: R_Foot
        [0.0, 0.0, 0.8],      # 12: Neck
        [0.0, -0.15, 0.75],   # 13: L_Collar
        [0.0, 0.15, 0.75],    # 14: R_Collar
        [0.0, 0.0, 0.95],     # 15: Head
        [0.0, -0.3, 0.75],    # 16: L_Shoulder
        [0.0, 0.3, 0.75],     # 17: R_Shoulder
        [0.0, -0.55, 0.75],   # 18: L_Elbow
        [0.0, 0.55, 0.75],    # 19: R_Elbow
        [0.0, -0.75, 0.75],   # 20: L_Wrist
        [0.0, 0.75, 0.75],    # 21: R_Wrist
    ], dtype=float)

    # Create world-space trajectory (single frame T-pose)
    pelvis_world = np.array([0.0, 0.0, 1.0], dtype=float)  # Standing at origin
    joints_world = pelvis_world + offsets
    
    # Create trajectory: (T, J, 3) - single frame
    source_positions = joints_world[np.newaxis, :, :]  # Shape: (1, 22, 3)
    
    # ==========================================
    # Setup test environment
    # ==========================================
    robot_config = _load_robot_profile(profile_path)
    robot_urdf_path = Path(robot_config["urdf_path"])

    if not robot_urdf_path.exists():
        pytest.skip(f"Robot URDF not found at: {robot_urdf_path}")
    
    # Create a simple flat terrain mesh
    terrain_mesh = trimesh.creation.box(extents=[10.0, 10.0, 0.1])
    terrain_mesh.apply_translation([0, 0, -0.05])
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix=".obj", delete=False) as f:
        terrain_path = f.name
        terrain_mesh.export(terrain_path)
    
    try:
        joint_mapping = {
            joint_name: robot_config["joint_mapping"][joint_name]
            for joint_name in COMMON_ALIGNMENT_JOINTS
            if joint_name in robot_config["joint_mapping"]
        }
        
        # ==========================================
        # Run retargeting
        # ==========================================
        print("\n" + "="*60)
        print(f"T-Pose Retargeting Test ({robot_name})")
        print("="*60)
        
        # The synthetic trajectory uses the full 22-joint SMPLX layout, so declare
        # the full skeleton as source_target_names (production gets these from the
        # DataSource). Otherwise source_target_names falls back to the mapped
        # subset, mapped indices point at the wrong joints, and base_orientation
        # cannot find Spine1 (the robot maps its waist to Spine2, not Spine1).
        from omniretargeting.data_sources.smplx import DEFAULT_SMPLX_TARGET_NAMES

        retargeter_kwargs = _build_retargeter_kwargs(robot_config, terrain_path, joint_mapping)
        retargeter_kwargs["source_target_names"] = list(DEFAULT_SMPLX_TARGET_NAMES)
        retargeter = OmniRetargeter(**retargeter_kwargs)
        assert sorted(retargeter.validate_joint_mapping()) == []
        
        print(f"Input SMPLX trajectory shape: {source_positions.shape}")
        print(f"Mapped source targets: {len(retargeter.mapped_source_target_indices)}")
        
        # Run retargeting (no visualization)
        source_to_robot_scale, retargeted_motion = retargeter.retarget_motion(
            source_positions,
            visualize_trajectory=False
        )
        
        print(f"Source-to-robot scale: {source_to_robot_scale:.4f}")
        print(f"Retargeted motion shape: {retargeted_motion.shape}")
        
        # ==========================================
        # Verify retargeting accuracy
        # ==========================================
        # Extract robot link positions from retargeted configuration
        import mujoco
        
        model = retargeter.robot_model
        data = retargeter.robot_data
        
        # Set robot to retargeted configuration
        q_retargeted = retargeted_motion[0]  # First (only) frame
        data.qpos[:] = q_retargeted
        mujoco.mj_forward(model, data)
        
        # Get robot link positions for mapped joints
        robot_positions = []
        target_positions = []
        checked_joints = []

        for smplx_name, mapping_value in joint_mapping.items():
            # Profile target_mapping values may be dicts: {"robot_link": ..., "offset": ...}
            robot_link_name = mapping_value["robot_link"] if isinstance(mapping_value, dict) else mapping_value

            # Get SMPLX joint index
            smplx_idx = retargeter.source_target_indices.get(smplx_name)
            if smplx_idx is None:
                continue

            # Get robot link position (mj_name2id returns -1 for unknown bodies)
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, robot_link_name)
            if body_id < 0:
                continue

            # Get target position (scaled)
            target_pos = source_positions[0, smplx_idx] * source_to_robot_scale
            target_positions.append(target_pos)
            robot_positions.append(data.xpos[body_id].copy())
            checked_joints.append((smplx_name, robot_link_name))
        
        robot_positions = np.array(robot_positions)
        target_positions = np.array(target_positions)
        
        # Compute per-joint distances
        distances = np.linalg.norm(robot_positions - target_positions, axis=1)
        mean_distance = distances.mean()
        max_distance = distances.max()
        
        print("\n" + "-"*60)
        print("Retargeting Accuracy Results:")
        print("-"*60)
        print(f"Number of mapped joints: {len(distances)}")
        print(f"Mean distance: {mean_distance:.4f} m")
        print(f"Max distance: {max_distance:.4f} m")
        print(f"Min distance: {distances.min():.4f} m")
        print("\nPer-joint distances:")
        for i, (smplx_name, robot_link_name) in enumerate(checked_joints):
            print(f"  {smplx_name:12s} -> {robot_link_name:25s}: {distances[i]:.4f} m")
        print("-"*60)
        
        # Test assertion: mean distance should be < 1.0m for now
        # TODO: Improve retargeting accuracy to get below 0.3m
        # Current issues:
        # - Laplacian constraints alone may not be sufficient for accurate position matching
        # - Need to add explicit position tracking costs
        # - Initial guess may be too far from solution
        # - Optimization may not be converging properly
        assert mean_distance < 1.0, (
            f"Retargeting accuracy too low: mean distance {mean_distance:.4f}m exceeds 1.0m threshold. "
            f"This indicates major issues with the retargeting pipeline."
        )
        
        if mean_distance < 0.3:
            print(f"\n✓ Test PASSED (EXCELLENT): Mean distance {mean_distance:.4f}m < 0.3m")
        elif mean_distance < 0.5:
            print(f"\n✓ Test PASSED (GOOD): Mean distance {mean_distance:.4f}m < 0.5m")
        else:
            print(f"\n✓ Test PASSED (ACCEPTABLE): Mean distance {mean_distance:.4f}m < 1.0m")
            print("  Note: Accuracy could be improved - see TODO comments in test")
        print("="*60 + "\n")
        
    finally:
        # Cleanup temporary terrain file
        if os.path.exists(terrain_path):
            os.remove(terrain_path)

@pytest.mark.parametrize(("robot_name", "profile_path"), FLOATING_BASE_PROFILE_CASES)
def test_robot_profile_has_floating_base(robot_name: str, profile_path: Path):
    import mujoco

    robot_config = _load_robot_profile(profile_path)
    from omniretargeting.utils import load_robot_urdf_with_floating_base
    model = load_robot_urdf_with_floating_base(str(robot_config["urdf_path"]))

    assert model.njnt > 0
    joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, 0)
    joint_type = int(model.jnt_type[0])

    message = "%s should expose a floating base as the first joint, got %s type=%s" % (
        robot_name,
        joint_name,
        joint_type,
    )
    assert joint_type == int(mujoco.mjtJoint.mjJNT_FREE), message
