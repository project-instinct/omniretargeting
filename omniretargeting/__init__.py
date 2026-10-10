"""OmniRetargeting: Generic motion retargeting for any humanoid URDF and terrain mesh."""

from .__version__ import __version__
from .core import OmniRetargeter
from .data_sources.base import (
    DataSource,
    MotionData,
    MotionFrame,
    validate_motion_frame_positions,
    validate_motion_positions,
    validate_object_points,
)
from .robot_config import load_robot_config
from .contacts import (
    Contact,
    EntityPose,
    EntityTrajectory,
    Scene,
    SceneEntity,
    detect_contacts,
    load_contact_trajectory,
    save_hsoi_annotations,
)

__all__ = [
    "DataSource",
    "MotionData",
    "MotionFrame",
    "Contact",
    "EntityPose",
    "EntityTrajectory",
    "Scene",
    "SceneEntity",
    "detect_contacts",
    "load_contact_trajectory",
    "save_hsoi_annotations",
    "OmniRetargeter",
    "load_robot_config",
    "validate_motion_frame_positions",
    "validate_motion_positions",
    "validate_object_points",
    "__version__",
]
