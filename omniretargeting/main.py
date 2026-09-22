import argparse
import json
import os
import tempfile
from pathlib import Path

import mujoco
import numpy as np
import trimesh
import yaml
from scipy.spatial.transform import Rotation

from omniretargeting import OmniRetargeter
from omniretargeting.contact import stack_contact_tasks
from omniretargeting.data_sources.registry import create_data_source
from omniretargeting.motion_visualization import (
    save_trajectory_video as save_trajectory_video,
    temporary_visualization_scene as temporary_visualization_scene,
    visualize_trajectory as visualize_trajectory,
)
from omniretargeting.robot_config import load_robot_config
from omniretargeting.utils import (
    load_smplx_trajectory,
    normalize_retargeted_output_path,
)

DEFAULT_ROBOT_CONFIG_PATH = "robot_models/unitree_g1/unitree_g1.json"



        # Cleanup handled by context manager

def create_flat_terrain(size=10.0):
    """Create a simple flat terrain mesh."""
    mesh = trimesh.creation.box(extents=[size, size, 0.1])
    mesh.apply_translation([0, 0, -0.05])
    return mesh

def extract_body_base_pose(urdf_path, trajectory, body_name):
    """Extract a robot body's world pose from a MuJoCo qpos trajectory."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(urdf_path))
    data = mujoco.MjData(model)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"Body '{body_name}' not found in robot model: {urdf_path}")

    base_pos = np.zeros((len(trajectory), 3), dtype=trajectory.dtype)
    base_quat = np.zeros((len(trajectory), 4), dtype=trajectory.dtype)
    for frame_idx, qpos in enumerate(trajectory):
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        base_pos[frame_idx] = data.xpos[body_id]
        base_quat[frame_idx] = data.xquat[body_id]

    return base_pos, base_quat

def extract_omomo_object_pose(motion_path, num_frames):
    """Extract OMOMO object world pose from the source qpos if present."""
    motion_data = np.load(motion_path, allow_pickle=True)
    if "qpos" not in motion_data:
        return None

    qpos = motion_data["qpos"]
    if qpos.ndim != 2 or qpos.shape[0] < num_frames or qpos.shape[1] < 43:
        return None

    # OMOMO stores the object rotation in the opposite convention from MuJoCo/Viser.
    object_quat = np.array(qpos[:num_frames, 36:40], copy=True)
    object_quat[:, 1:] *= -1.0
    object_pos = qpos[:num_frames, 40:43]
    return object_pos, object_quat


def load_source_config(config_path):
    """Read a generic motion-source YAML/JSON config and resolve input paths."""
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Source config at {path} must be a mapping.")
    source_type = config.get("type")
    motion_path = config.get("motion")
    if not source_type or not motion_path:
        raise ValueError(
            f"Source config at {path} must define non-empty 'type' and 'motion'."
        )

    def resolve_input(value):
        value_path = Path(os.path.expandvars(os.path.expanduser(str(value))))
        if not value_path.is_absolute():
            value_path = (path.parent / value_path).resolve()
        return value_path

    resolved = dict(config)
    resolved["motion"] = str(resolve_input(motion_path))
    if config.get("terrain"):
        resolved["terrain"] = str(resolve_input(config["terrain"]))
    return resolved


def direct_source_config(motion_path, source_type=None):
    """Build the minimal source config for a motion passed directly to the CLI.

    Common unambiguous formats are inferred so dataset clips do not need a
    repeated per-file JSON wrapper.  ``--source-type`` remains available for
    formats such as BVH that are shared by several adapters.
    """
    path = Path(
        os.path.expandvars(os.path.expanduser(str(motion_path)))
    ).resolve()
    if source_type is None:
        inferred_types = {
            ".pkl": "parc_ms",
            ".pickle": "parc_ms",
            ".npz": "smplx",
        }
        source_type = inferred_types.get(path.suffix.casefold())
    if not source_type:
        raise ValueError(
            f"Cannot infer the source adapter for '{path.suffix or path.name}'. "
            "Pass --source-type (for example lafan1, nokov, or bones_seed)."
        )
    return {"type": str(source_type), "motion": str(path)}


def _resolve_source_torso_mapping(source_motion_data, source_config):
    """Return an optional source-neutral virtual-torso specification.

    An explicit legacy ``base_orientation_joint`` keeps its historical
    behaviour. Otherwise a source adapter may advertise a semantic default,
    while the source config can always override or disable that default.
    """
    if "torso_mapping" in source_config:
        raw_mapping = source_config.get("torso_mapping")
    elif source_config.get("base_orientation_joint"):
        return None
    else:
        raw_mapping = source_motion_data.metadata.get("torso_mapping")

    if raw_mapping in (None, False):
        return None
    if isinstance(raw_mapping, str):
        mapping = {"preset": raw_mapping}
    elif isinstance(raw_mapping, dict):
        mapping = dict(raw_mapping)
    else:
        raise ValueError("torso_mapping must be a preset name or a mapping.")

    preset = str(mapping.get("preset", "pelvis_shoulders"))
    if preset != "pelvis_shoulders":
        raise ValueError(
            f"Unsupported torso_mapping preset '{preset}'. Supported presets: "
            "pelvis_shoulders."
        )
    mapping["preset"] = preset
    mapping.setdefault("origin", "Pelvis")
    mapping.setdefault("left_shoulder", "L_Shoulder")
    mapping.setdefault("right_shoulder", "R_Shoulder")
    mapping.setdefault("source_body", "Spine1")
    return mapping


def _landmark_torso_rotation(
    positions, name_to_index, *, origin, left_shoulder, right_shoulder
):
    """Build world rotations whose axes follow an anatomical torso frame.

    X points forward, Y toward the actor's left shoulder, and Z from the
    pelvis toward the shoulder midpoint. Using both shoulders retains torso
    twist while the pelvis-to-shoulder direction represents the silhouette of
    a multi-segment human spine on a robot with a mostly rigid trunk.
    """
    required = (origin, left_shoulder, right_shoulder)
    missing = [name for name in required if name not in name_to_index]
    if missing:
        raise ValueError(
            "torso_mapping refers to missing source landmarks "
            f"{missing}; available joints are {list(name_to_index)}."
        )

    values = np.asarray(positions, dtype=float)
    if values.shape[-2:] != (len(name_to_index), 3):
        raise ValueError(
            "torso_mapping positions must end in shape "
            f"({len(name_to_index)}, 3), got {values.shape}."
        )
    pelvis = values[..., name_to_index[origin], :]
    left = values[..., name_to_index[left_shoulder], :]
    right = values[..., name_to_index[right_shoulder], :]
    up = 0.5 * (left + right) - pelvis
    lateral_hint = left - right

    up_norm = np.linalg.norm(up, axis=-1, keepdims=True)
    forward = np.cross(lateral_hint, up)
    forward_norm = np.linalg.norm(forward, axis=-1, keepdims=True)
    if np.any(up_norm < 1e-8) or np.any(forward_norm < 1e-8):
        raise ValueError(
            "torso_mapping cannot build a frame from coincident or collinear "
            "pelvis/shoulder landmarks."
        )
    up = up / up_norm
    forward = forward / forward_norm
    lateral = np.cross(up, forward)
    lateral_norm = np.linalg.norm(lateral, axis=-1, keepdims=True)
    if np.any(lateral_norm < 1e-8):
        raise ValueError("torso_mapping produced a degenerate lateral axis.")
    lateral = lateral / lateral_norm
    return np.stack([forward, lateral, up], axis=-1)


def _select_landmark_torso_orientations(source_motion_data, mapping):
    target_names = list(source_motion_data.target_names or [])
    name_to_index = {str(name): index for index, name in enumerate(target_names)}
    reference_positions = source_motion_data.metadata.get("joint_rest_positions")
    if reference_positions is None:
        raise ValueError(
            "torso_mapping requires source metadata 'joint_rest_positions' so "
            "the anatomical frame can be calibrated instead of treating the "
            "first motion frame as a rest pose."
        )

    frame_kwargs = {
        "origin": str(mapping["origin"]),
        "left_shoulder": str(mapping["left_shoulder"]),
        "right_shoulder": str(mapping["right_shoulder"]),
    }
    current_rotations = _landmark_torso_rotation(
        source_motion_data.positions,
        name_to_index,
        **frame_kwargs,
    )
    reference_rotation = _landmark_torso_rotation(
        np.asarray(reference_positions, dtype=float),
        name_to_index,
        **frame_kwargs,
    )
    # Apply source motion relative to its authored rest frame to a canonical
    # robot base frame. This matches rest-pose calibration used by body
    # orientation retargeters instead of assuming source and robot axes match.
    target_rotations = current_rotations @ reference_rotation.T
    quaternions = Rotation.from_matrix(target_rotations).as_quat(
        scalar_first=True
    )
    for frame_index in range(1, len(quaternions)):
        if np.dot(quaternions[frame_index - 1], quaternions[frame_index]) < 0.0:
            quaternions[frame_index] *= -1.0

    source_body = str(mapping["source_body"])
    if source_body not in name_to_index:
        raise ValueError(
            f"torso_mapping source_body '{source_body}' is not in the source "
            f"joint layout {target_names}."
        )
    return quaternions, source_body


def _source_config_with_profile_torso(source_motion_data, source_config, base_orientation):
    """Apply an anatomical frame declared by a robot profile when supported.

    Explicit source choices win. Positions-only adapters keep their original
    path when they cannot provide the authored rest landmarks for calibration.
    """
    result = dict(source_config)
    profile_mapping = base_orientation.get("torso_mapping")
    if (
        not profile_mapping
        or "torso_mapping" in result
        or result.get("base_orientation_joint")
    ):
        return result
    if source_motion_data.metadata.get("joint_rest_positions") is None:
        print("Profile torso calibration unavailable: source has no rest landmarks; retaining its original orientation path.")
        return result
    result["torso_mapping"] = profile_mapping
    result.setdefault("lock_root_orientation_to_source", False)
    return result


def select_source_base_orientations(source_motion_data, source_config):
    """Resolve the source body frame that should drive a floating robot base.

    ``MotionData.root_orientations`` follows the source file's root joint,
    which is commonly the pelvis.  A torso-base robot instead needs the world
    orientation of the configured torso joint; feeding the pelvis rotation to
    that base while also tracking the torso creates two incompatible targets
    during bends and vaults.
    """
    torso_mapping = _resolve_source_torso_mapping(
        source_motion_data, source_config
    )
    if torso_mapping is not None:
        return _select_landmark_torso_orientations(
            source_motion_data, torso_mapping
        )

    source_joint = source_config.get("base_orientation_joint")
    if not source_joint:
        return source_motion_data.root_orientations, None
    target_names = list(source_motion_data.target_names or [])
    if source_joint not in target_names:
        raise ValueError(
            f"base_orientation_joint '{source_joint}' is not in the source "
            f"joint layout {target_names}."
        )
    joint_orientations = source_motion_data.metadata.get("joint_orientations")
    if joint_orientations is None:
        raise ValueError(
            f"base_orientation_joint '{source_joint}' requires source global "
            "joint orientations."
        )
    joint_orientations = np.asarray(joint_orientations, dtype=float)
    expected_shape = (*source_motion_data.positions.shape[:2], 4)
    if joint_orientations.shape != expected_shape:
        raise ValueError(
            "Source global joint orientations must have shape "
            f"{expected_shape}, got {joint_orientations.shape}."
        )
    return joint_orientations[:, target_names.index(source_joint)], str(source_joint)


def select_mapped_source_base_pose(source_motion_data, model, joint_mapping, *, position_source=None):
    """Use the source body corresponding to the compiled floating robot body.

    The adapter owns coordinate calibration. Raw source joint quaternions must
    not be interpreted as robot base frames. Explicit caller policies are
    resolved before this optional automatic path.
    """
    orientations = source_motion_data.metadata.get("body_aligned_joint_orientations")
    if orientations is None:
        return None
    free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if len(free) != 1:
        return None
    body = model.body(int(model.jnt_bodyid[free[0]])).name
    names = list(source_motion_data.target_names or [])
    matches = [name for name, target in joint_mapping.items() if target == body and name in names]
    if len(matches) != 1:
        raise ValueError(f"Floating body '{body}' needs one unambiguous source mapping; got {matches}.")
    values = np.asarray(orientations, dtype=float)
    if values.shape != (*source_motion_data.positions.shape[:2], 4):
        raise ValueError("body_aligned_joint_orientations must have shape (frames, joints, 4).")
    index = names.index(matches[0])
    selected = values[:, index].copy()
    norms = np.linalg.norm(selected, axis=1)
    if not np.isfinite(selected).all() or np.any(norms < 1e-12):
        raise ValueError("Mapped base orientations must be finite nonzero quaternions.")
    selected /= norms[:, None]
    if position_source is not None and position_source not in names:
        raise ValueError(f"Root position source '{position_source}' is not in the source joint layout.")
    position_index = names.index(position_source) if position_source is not None else index
    return selected, np.asarray(source_motion_data.positions[:, position_index]).copy(), matches[0]


def main():
    parser = argparse.ArgumentParser(description="OmniRetargeting CLI")
    parser.add_argument(
        "--robot-config",
        default=DEFAULT_ROBOT_CONFIG_PATH,
        help=f"Path to robot configuration JSON file (default: {DEFAULT_ROBOT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--source-config",
        default=None,
        help="Generic motion source YAML/JSON (for example type=parc_ms, motion=clip.pkl).",
    )
    parser.add_argument(
        "--source-motion",
        default=None,
        help=(
            "Direct generic motion path. Common formats are inferred "
            "(.pkl=parc_ms, .npz=smplx), so no per-clip source JSON is needed."
        ),
    )
    parser.add_argument(
        "--source-type",
        default=None,
        help="Adapter for --source-motion when its format is ambiguous.",
    )
    parser.add_argument(
        "--smplx_model_dir",
        default=None,
        help="Directory containing SMPLX model files",
    )
    parser.add_argument(
        "--smplx_motion",
        default=None,
        help="Legacy direct path to an SMPLX motion file (.npz)",
    )
    parser.add_argument("--output", required=True, help="Path to save output motion (.npz)")
    parser.add_argument("--solver", choices=["framewise", "sequence"], default=None,
                        help="Retargeting solver; sequence jointly optimizes all input frames")
    parser.add_argument("--terrain", help="Path to terrain mesh file (optional, defaults to flat ground)")
    parser.add_argument(
        "--output-scaled-terrain",
        dest="output_scaled_terrain",
        default=None,
        help="Path to save the scaled terrain mesh. When provided, terrain scaling is enabled.",
    )
    parser.add_argument("--mapping", help="Path to joint mapping JSON file (optional, overrides robot profile mapping)")
    parser.add_argument("--vis", action="store_true", help="Visualize the retargeted motion")
    parser.add_argument("--save-video", dest="save_video", default=None, help="Save retargeted motion video to file (e.g. /tmp/out.mp4). Uses offscreen rendering (set MUJOCO_GL=egl for headless).")
    parser.add_argument("--framerate", type=float, default=None, help="Framerate of the motion (optional, defaults to 30.0 or auto-detected)")
    parser.add_argument(
        "--output-framerate",
        type=float,
        default=None,
        help="Resample a generic source to this framerate before retargeting.",
    )
    parser.add_argument(
        "--scaled-objects",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--output-base-link", default="torso_link",
                        help="Robot body link used for saved base_pos_w/base_quat_w.")
    parser.add_argument("--replace-cylinders-with-capsules", dest="replace_cylinders_with_capsules", action="store_true", default=False,
                        help="Replace cylinder collision geoms with capsules to match IsaacLab/PhysX convention.")
    parser.add_argument("--penetration-resolver", choices=["hard_constraint", "xyz_nudge"], default=None,
                        help="Override the contact handling mode for retargeting.")

    args = parser.parse_args()

    source_inputs = (args.source_config, args.source_motion, args.smplx_motion)
    if sum(value is not None for value in source_inputs) != 1:
        parser.error(
            "provide exactly one of --source-config, --source-motion, or "
            "--smplx_motion"
        )
    if args.source_type is not None and args.source_motion is None:
        parser.error("--source-type is only valid with --source-motion")

    args.output = normalize_retargeted_output_path(args.output)

    if args.source_config:
        source_config = load_source_config(args.source_config)
    elif args.source_motion:
        source_config = direct_source_config(args.source_motion, args.source_type)
    else:
        source_config = None

    # Load robot profile config (default profile path can be overridden by --robot-config).
    robot_config = {}
    if args.robot_config:
        robot_config_path = Path(args.robot_config)
        if robot_config_path.exists():
            robot_config = load_robot_config(robot_config_path)
            profile_name = robot_config.get("name", robot_config_path.stem)
            print(f"Loaded robot config profile: {profile_name}")
        elif args.robot_config == DEFAULT_ROBOT_CONFIG_PATH:
            print(f"Default robot config not found at {DEFAULT_ROBOT_CONFIG_PATH}, continuing without profile.")
        else:
            raise FileNotFoundError(f"Robot config not found: {args.robot_config}")

    # Load joint mapping
    mapping_is_explicit = bool(
        args.mapping
        or (source_config is not None and source_config.get("joint_mapping"))
    )
    if args.mapping:
        with open(args.mapping, 'r') as f:
            joint_mapping = json.load(f)
    elif source_config is not None and source_config.get("joint_mapping"):
        joint_mapping = source_config["joint_mapping"]
    elif "joint_mapping" in robot_config:
        joint_mapping = robot_config["joint_mapping"]
    else:
        raise ValueError(
            "No joint mapping available. Provide --mapping or use a robot profile with 'joint_mapping'."
        )

    if not isinstance(joint_mapping, dict) or not joint_mapping:
        raise ValueError("Joint mapping must be a non-empty JSON object.")

    robot_urdf_path = robot_config.get("urdf_path")
    if not robot_urdf_path:
        raise ValueError(
            "Robot URDF is required. Set 'urdf_path' in the robot profile JSON (--robot-config)."
        )

    robot_height = robot_config.get("robot_height")
    smplx_joint_names = robot_config.get("smplx_joint_names")
    height_estimation = robot_config.get("height_estimation")
    base_orientation = dict(robot_config.get("base_orientation") or {})
    if source_config is not None and source_config.get("base_orientation"):
        base_orientation.update(dict(source_config["base_orientation"]))
    retargeting = dict(robot_config.get("retargeting") or {})

    # Merge CLI flag into retargeting config
    if args.replace_cylinders_with_capsules:
        retargeting["replace_cylinders_with_capsules"] = True
    if args.penetration_resolver is not None:
        retargeting["penetration_resolver"] = args.penetration_resolver

    temp_terrain_path = None
    source_motion_data = None
    source_type = "smplx"
    source_motion_path = Path(args.smplx_motion) if args.smplx_motion else None

    try:
        end_effector_cfg = dict(
            retargeting.get("end_effector_priority", {}) or {}
        )
        end_effector_markers_enabled = bool(
            end_effector_cfg.get("enabled", False)
            or end_effector_cfg.get("world_tracking_only", False)
            or end_effector_cfg.get("world_tracking_enabled", False)
        )
        base_orientations = None
        base_translations = None
        source_reference_joint_orientations = None
        source_reference_joint_positions = None

        if source_config is not None:
            source_type = str(source_config["type"])
            source_motion_path = Path(source_config["motion"])
            print(
                f"Loading {source_type} trajectory from {source_motion_path}..."
            )
            data_source = create_data_source(
                source_type,
                source_motion_path,
                source_config=source_config,
            )
            source_motion_data = data_source.load()
            if (
                not mapping_is_explicit
                and source_motion_data.metadata.get("joint_mapping")
            ):
                joint_mapping = dict(
                    source_motion_data.metadata["joint_mapping"]
                )
                print(
                    f"Using {source_type} adapter's default joint mapping "
                    f"({len(joint_mapping)} joints)."
                )
            adapter_base_orientation = source_motion_data.metadata.get(
                "base_orientation"
            )
            if adapter_base_orientation:
                base_orientation.update(dict(adapter_base_orientation))
                if source_config.get("base_orientation"):
                    base_orientation.update(
                        dict(source_config["base_orientation"])
                    )
            if args.output_framerate is not None:
                source_motion_data = source_motion_data.resample(
                    args.output_framerate
                )
            smplx_trajectory = source_motion_data.positions
            smplx_orientations = source_motion_data.metadata.get(
                "joint_orientations"
            )
            source_reference_joint_orientations = source_motion_data.metadata.get(
                "joint_rest_orientations"
            )
            source_reference_joint_positions = source_motion_data.metadata.get(
                "joint_rest_positions"
            )
            source_config = _source_config_with_profile_torso(
                source_motion_data, source_config, base_orientation
            )
            base_orientations, base_orientation_source_joint = (
                select_source_base_orientations(
                    source_motion_data, source_config
                )
            )
            if base_orientation_source_joint is not None:
                torso_mapping = _resolve_source_torso_mapping(
                    source_motion_data, source_config
                )
                end_effector_cfg = dict(
                    retargeting.get("end_effector_priority", {}) or {}
                )
                end_effector_cfg["_base_orientation_source_joint"] = (
                    base_orientation_source_joint
                )
                lock_root_orientation = bool(
                    source_config.get(
                        "lock_root_orientation_to_source",
                        True,
                    )
                )
                end_effector_cfg["lock_root_orientation_to_source"] = (
                    lock_root_orientation
                )
                balanced_torso = (
                    torso_mapping is not None
                    and not lock_root_orientation
                    and end_effector_cfg.get("free_motion_priority") == "balanced"
                )
                if balanced_torso:
                    # The frame choice must not silently add stricter root
                    # rate limits to balanced free motion. Preserve explicit
                    # temporal settings, with trusted contacts still first.
                    end_effector_cfg["hierarchy"] = True
                if torso_mapping is not None and not lock_root_orientation and not balanced_torso:
                    # Active scene contacts stay lexicographically first. For
                    # free limbs, preserve the actor's calibrated whole-trunk
                    # frame before spending the remaining morphology freedom
                    # on world-space swing targets. A small angular allowance
                    # keeps this a soft cross-morphology task rather than the
                    # brittle quaternion hard lock used by the legacy mode.
                    end_effector_cfg["hierarchy"] = True
                    end_effector_cfg["base_orientation_before_swing"] = True
                    end_effector_cfg[
                        "base_orientation_priority_absolute_tolerance"
                    ] = float(np.deg2rad(5.0))
                    temporal_cfg = dict(
                        end_effector_cfg.get("temporal", {}) or {}
                    )
                    temporal_cfg.setdefault(
                        "max_base_orientation_step_error",
                        float(np.deg2rad(5.0)),
                    )
                    temporal_cfg.setdefault(
                        "base_orientation_catchup_gain", 0.35
                    )
                    temporal_cfg.setdefault(
                        "base_orientation_catchup_deadband",
                        float(np.deg2rad(20.0)),
                    )
                    temporal_cfg.setdefault(
                        "max_base_orientation_catchup_step",
                        float(np.deg2rad(20.0)),
                    )
                    temporal_cfg.setdefault(
                        "max_base_orientation_total_step",
                        float(np.deg2rad(25.0)),
                    )
                    end_effector_cfg["temporal"] = temporal_cfg
                retargeting["end_effector_priority"] = end_effector_cfg
                if torso_mapping is not None:
                    orientation_policy = (
                        "fixed to source"
                        if lock_root_orientation
                        else "soft priority"
                    )
                    print(
                        "Using rest-calibrated anatomical torso preset "
                        f"'{torso_mapping['preset']}' for the robot-base "
                        f"orientation frame ({orientation_policy})."
                    )
                else:
                    print(
                        "Using source joint "
                        f"'{base_orientation_source_joint}' as the robot-base "
                        "orientation frame."
                    )
            base_translations = source_motion_data.root_translations
            end_effector_markers = (
                source_motion_data.metadata.get("end_effector_markers")
                if end_effector_markers_enabled
                else None
            )
            smplx_joint_names = (
                source_motion_data.target_names or smplx_joint_names
            )
            framerate = args.framerate or source_motion_data.framerate
        else:
            if args.output_framerate is not None:
                raise ValueError(
                    "--output-framerate currently requires --source-config."
                )
            print(f"Loading SMPLX trajectory from {source_motion_path}...")
            loaded_motion = load_smplx_trajectory(
                smplx_file=source_motion_path,
                smplx_model_directory=args.smplx_model_dir,
                return_end_effectors=end_effector_markers_enabled,
            )
            smplx_trajectory, smplx_orientations = loaded_motion[:2]
            end_effector_markers = (
                loaded_motion[2] if end_effector_markers_enabled else None
            )

            # Try to detect framerate from a legacy SMPL-X file.
            framerate = args.framerate
            if framerate is None:
                try:
                    smplx_data = np.load(source_motion_path, allow_pickle=True)
                    if isinstance(smplx_data, np.lib.npyio.NpzFile):
                        for key in (
                            "framerate",
                            "mocap_framerate",
                            "mocap_frame_rate",
                            "fps",
                        ):
                            if key in smplx_data:
                                framerate = float(smplx_data[key])
                                print(f"Detected {key} from file: {framerate}")
                                break
                except Exception:
                    pass

        if framerate is None:
            framerate = 30.0
            print(f"Using default framerate: {framerate}")

        terrain_path = args.terrain
        if terrain_path is None and source_config is not None:
            terrain_path = source_config.get("terrain")
        if terrain_path is None and source_motion_data is not None:
            terrain_path = source_motion_data.metadata.get("terrain_mesh")
        if terrain_path is None:
            print("No terrain provided, creating default flat terrain.")
            flat_terrain = create_flat_terrain()
            fd, temp_terrain_path = tempfile.mkstemp(suffix=".obj")
            os.close(fd)
            flat_terrain.export(temp_terrain_path)
            terrain_path = temp_terrain_path
        else:
            terrain_path = str(terrain_path)
            print(f"Using terrain mesh: {terrain_path}")

        print(f"Loaded trajectory with shape: {smplx_trajectory.shape}")
        if smplx_orientations is not None:
            print(f"Loaded orientations with shape: {smplx_orientations.shape}")
        else:
            print("Warning: Orientations not available for this file format.")

        # Initialize Retargeter
        print("Initializing OmniRetargeter...")
        retargeter = OmniRetargeter(
            robot_urdf_path=robot_urdf_path,
            terrain_mesh_path=terrain_path,
            joint_mapping=joint_mapping,
            robot_height=robot_height,
            smplx_joint_names=smplx_joint_names,
            height_estimation=height_estimation,
            base_orientation=base_orientation,
            retargeting=retargeting,
        )
        if args.solver is not None:
            retargeter.retargeting_config.setdefault("sequence_solver", {})["enabled"] = args.solver == "sequence"
        position_anchor = base_orientation.get("root_position")
        if position_anchor:
            retargeter.retargeting_config.setdefault("end_effector_priority", {})[
                "_position_anchor_source_joint"
            ] = position_anchor

        if (
            source_motion_data is not None
            and not source_config.get("base_orientation_joint")
            and _resolve_source_torso_mapping(source_motion_data, source_config) is None
            and source_config.get("auto_mapped_base_pose", True)
        ):
            mapped_pose = select_mapped_source_base_pose(
                source_motion_data, retargeter.robot_model, joint_mapping,
                position_source=base_orientation.get("root_position"),
            )
            if mapped_pose is not None:
                base_orientations, base_translations, mapped_body = mapped_pose
                # This body already has the calibrated base task. A second
                # orientation task calibrated through a different pose can
                # conflict with it during a whole-body rotation.
                retargeter.retargeting_config.setdefault("end_effector_priority", {})[
                    "_base_orientation_source_joint"
                ] = mapped_body
                position_source = base_orientation.get("root_position", mapped_body)
                print(f"Robot base orientation follows calibrated '{mapped_body}'; position anchor is '{position_source}' (no hard orientation lock).")

        # Perform retargeting
        print("Retargeting motion...")
        enable_terrain_scaling = bool(args.output_scaled_terrain)
        retarget_motion_kwargs = dict(
            base_orientations=base_orientations,
            base_translations=base_translations,
            source_joint_orientations=smplx_orientations,
            source_reference_joint_orientations=(
                source_reference_joint_orientations
            ),
            source_reference_joint_positions=source_reference_joint_positions,
            framerate=framerate,
            visualize_trajectory=args.vis,
            enable_terrain_scaling=enable_terrain_scaling,
        )
        if end_effector_markers is not None:
            retarget_motion_kwargs["end_effector_markers"] = end_effector_markers
        if source_motion_data is not None:
            retarget_motion_kwargs["body_surface_samples"] = source_motion_data.metadata.get("body_surface_samples")
            if source_motion_data.object_points is not None:
                retarget_motion_kwargs["object_points"] = source_motion_data.object_points
            if source_motion_data.object_mesh is not None:
                metadata = source_motion_data.metadata
                retarget_motion_kwargs["scene_objects"] = [{
                    "name": metadata.get("object_name", "object"),
                    "mesh": source_motion_data.object_mesh,
                    "translations": metadata.get("object_translations"),
                    "rotations": metadata.get("object_rotations"),
                    "scales": metadata.get("object_scales"),
                }]
        terrain_scale, retargeted_motion = retargeter.retarget_motion(
            smplx_trajectory, **retarget_motion_kwargs
        )

        if args.output_scaled_terrain:
            scaled_terrain = trimesh.load(terrain_path, force="mesh")
            scaled_terrain.apply_scale(terrain_scale)
            output_scaled_terrain_path = Path(args.output_scaled_terrain)
            output_scaled_terrain_path.parent.mkdir(parents=True, exist_ok=True)
            scaled_terrain.export(output_scaled_terrain_path)
            print(f"Saved scaled terrain mesh to {output_scaled_terrain_path}")

        # Save output
        print(f"Saving output to {args.output}...")

        # Extract data for saving
        # retargeted_motion shape: (T, 7 + DOF) -> [root pos(3), root quat(4), joints(DOF)]

        # Get joint names from robot model
        joint_names = retargeter.get_joint_names()

        # Extract components. The MuJoCo qpos root is pelvis; save torso_link as the exported base.
        base_pos, base_quat = extract_body_base_pose(
            robot_urdf_path,
            retargeted_motion,
            args.output_base_link,
        )
        joint_pos = retargeted_motion[:, 7:]
        object_pose = (
            extract_omomo_object_pose(
                source_motion_path, len(retargeted_motion)
            )
            if source_type == "smplx" and source_motion_path is not None
            else None
        )

        # Convert quaternion to xyzw if needed (standard for many tools)
        # MuJoCo uses wxyz, but many other tools use xyzw.
        # The example file has 'base_quat_w' which implies world frame.
        # Let's assume the example file uses xyzw convention as it's common in ROS/scipy
        # But wait, MuJoCo uses wxyz. Let's check the example file values if possible.
        # For now, let's stick to wxyz as it is what MuJoCo uses and what we have.
        # If the user wants xyzw, we can convert.
        # Actually, let's look at the example file keys again:
        # ['framerate', 'joint_names', 'joint_pos', 'base_pos_w', 'base_quat_w']

        # Save as .npz with specific keys
        output_payload = dict(
            framerate=framerate,
            # Preserve the exact MuJoCo generalized configuration.  The
            # exported base_pos_w may refer to torso_link for downstream
            # policy datasets and cannot be used as the floating-root pose
            # without an inverse-FK conversion in viewers.
            qpos=retargeted_motion,
            joint_names=np.array(joint_names),
            joint_pos=joint_pos,
            base_pos_w=base_pos,
            base_quat_w=base_quat, # Saving as wxyz (MuJoCo convention)
            source_type=np.asarray(source_type),
            source_file=np.asarray(str(source_motion_path)),
            source_joint_names=np.asarray(
                smplx_joint_names or retargeter.smplx_joint_names
            ),
            source_joints_w=smplx_trajectory * terrain_scale,
            terrain_scale=np.asarray(terrain_scale),
            # Backward-compatible alias retained for existing consumers.
            smplx_joints_w=smplx_trajectory * terrain_scale,
        )
        if object_pose is not None:
            object_pos, object_quat = object_pose
            output_payload["object_pos_w"] = object_pos
            output_payload["object_quat_w"] = object_quat
        if source_motion_data is not None and source_motion_data.object_mesh is not None:
            metadata = source_motion_data.metadata
            output_payload["object_pos_w"] = np.asarray(metadata["object_translations"])
            output_payload["object_quat_w"] = Rotation.from_matrix(
                metadata["object_rotations"]
            ).as_quat(scalar_first=True)
            output_payload["object_scale"] = np.asarray(metadata["object_scales"])
            if metadata.get("object_mesh_path"):
                output_payload["object_mesh_path"] = np.asarray(metadata["object_mesh_path"])
        output_payload.update(
            stack_contact_tasks(getattr(retargeter, "last_end_effector_tasks", {}))
        )
        surface_contacts = getattr(retargeter, "last_surface_contact_tasks", None)
        if surface_contacts is not None:
            output_payload.update(surface_contacts.export())
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        diagnostics = getattr(retargeter, "last_collision_diagnostics", None)
        relation_tasks = getattr(retargeter, "last_body_relation_tasks", None)
        if relation_tasks is not None and diagnostics is not None:
            diagnostics["body_relations"]["stages"]["final_save"] = relation_tasks.metrics(retargeted_motion)
        if diagnostics is not None:
            from .local_collision_repair import diagnostics_json, trajectory_digest
            if trajectory_digest(retargeted_motion) != diagnostics["accepted_qpos_sha256"]:
                raise RuntimeError("Output qpos changed after final collision/contact validation")
            diagnostic_path = Path(args.output).with_suffix(".diagnostics.json")
            diagnostic_path.write_text(diagnostics_json(diagnostics), encoding="utf-8")
            print(f"Saved collision/contact diagnostics to {diagnostic_path}")
        np.savez(
            args.output,
            **output_payload,
        )

        print(f"Done! Terrain scale used: {terrain_scale}")

        # Load terrain for visualization/video if needed
        vis_terrain = None
        if (args.vis or args.save_video) and terrain_path and os.path.exists(terrain_path):
            try:
                vis_terrain = trimesh.load(terrain_path, force='mesh')
                if args.output_scaled_terrain:
                    vis_terrain.apply_scale(terrain_scale)
            except Exception as e:
                print(f"Could not load terrain for visualization: {e}")

        if args.save_video:
            save_trajectory_video(
                robot_urdf_path, retargeted_motion, args.save_video,
                smplx_trajectory=smplx_trajectory * terrain_scale,
                terrain_mesh=vis_terrain, fps=framerate,
            )

        if args.vis:
            visualize_trajectory(robot_urdf_path, retargeted_motion, smplx_trajectory * terrain_scale, terrain_mesh=vis_terrain)

    finally:
        # Cleanup temp file
        if temp_terrain_path and os.path.exists(temp_terrain_path):
            os.remove(temp_terrain_path)


if __name__ == "__main__":
    main()
