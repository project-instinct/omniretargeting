"""Motion-source configuration and base-orientation frame selection for the CLI.

Reads generic source configs, infers the adapter of a directly passed motion
and resolves which calibrated source frame drives the floating robot base.
"""

import os
from pathlib import Path

import mujoco
import numpy as np
import yaml
from scipy.spatial.transform import Rotation


# Path-valued keys besides ``motion``; relative values are relative to the config file.
PATH_KEYS = ("terrain", "model_directory", "data_root")


def absolute_source_paths(config, base=None):
    """Return ``config`` with relative input paths anchored at ``base`` (default: CWD).

    Use before writing a generated config elsewhere: a saved config resolves
    relative paths against its own directory.
    """
    base = Path.cwd() if base is None else Path(base)
    resolved = dict(config)
    for key in ("motion", *PATH_KEYS):
        if resolved.get(key) and isinstance(resolved[key], (str, os.PathLike)):
            value = Path(os.path.expandvars(os.path.expanduser(str(resolved[key]))))
            resolved[key] = str(value if value.is_absolute() else (base / value).resolve())
    return resolved


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
    for key in PATH_KEYS:
        if config.get(key):
            resolved[key] = str(resolve_input(config[key]))
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
