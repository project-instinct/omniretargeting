"""Anatomical T-pose frames for LAFAN1's bone-aligned BVH skeleton.

LAFAN1 offsets run along each segment's local X axis. Zero BVH channels
therefore put all segments along that axis, not in an anatomical rest pose.
The reference below uses the source's lengths and local frame conventions;
it is independent of the clip's first frame, scene, and target robot.
"""

import numpy as np
from scipy.spatial.transform import Rotation


def lafan1_reference(names, parents, offsets, local_rotations=None):
    """Return fixed Z-up T-pose metadata for the standard LAFAN1 hierarchy.

    Torso X points up, Y forward, Z left. Limb X points distally; arm Y
    points backward and arm Z is the elbow flexion axis (up on the left,
    down on the right). This preserves the recorded palm twist as well as
    the bone direction. Other BVH skeletons use the existing first-frame
    calibration fallback rather than being assigned LAFAN1's conventions.
    """
    required = {
        "Hips",
        "Spine2",
        "LeftUpLeg",
        "RightUpLeg",
        "LeftLeg",
        "RightLeg",
        "LeftFoot",
        "RightFoot",
        "LeftShoulder",
        "RightShoulder",
        "LeftArm",
        "RightArm",
        "LeftForeArm",
        "RightForeArm",
        "LeftHand",
        "RightHand",
    }
    if not required.issubset(names):
        return {}

    torso = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    leg = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0]])
    # Foot X runs toward the toe, Y is up, Z is lateral toward the right.
    foot = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    left_arm = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    right_arm = np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    frames = {}
    for name in names:
        if name in {"LeftUpLeg", "RightUpLeg", "LeftLeg", "RightLeg"}:
            frames[name] = leg
        elif name in {"LeftFoot", "RightFoot", "LeftToe", "RightToe"}:
            frames[name] = foot
        elif name in {"LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand"}:
            frames[name] = left_arm
        elif name in {"RightShoulder", "RightArm", "RightForeArm", "RightHand"}:
            frames[name] = right_arm
        else:
            frames[name] = torso

    rotations = np.asarray([frames[name] for name in names])
    # LAFAN1's toe channels contain a fixed rig rotation, which sets the
    # ankle-to-ball slope even in neutral stance. Retain that source-authored
    # offset in the reference instead of treating the ankle frame as a flat
    # sole. Articulated toe motion must never become a calibration offset.
    if local_rotations is not None:
        for side in ("Left", "Right"):
            toe_name, ankle_name = side + "Toe", side + "Foot"
            if toe_name not in names:
                continue
            toe = names.index(toe_name)
            ankle = names.index(ankle_name)
            if parents[toe] != ankle:
                continue
            local = Rotation.from_quat(
                np.asarray(local_rotations)[:, toe], scalar_first=True
            )
            if np.all((local[0].inv() * local).magnitude() < 1e-6):
                rotations[ankle] = foot @ local[0].as_matrix().T
    positions = np.zeros((len(names), 3))
    for index, parent in enumerate(parents):
        if parent >= 0:
            positions[index] = (
                positions[parent] + rotations[parent] @ offsets[index] / 100.0
            )
    positions[:, 2] -= positions[:, 2].min()
    return {
        "joint_rest_positions": positions,
        "joint_rest_orientations": Rotation.from_matrix(rotations).as_quat(
            scalar_first=True
        ),
        "joint_rest_pose": "lafan1_anatomical_t_pose",
    }
