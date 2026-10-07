"""OmniRetargeting: Generic motion retargeting for any humanoid URDF and terrain mesh."""

from .__version__ import __version__
from .core import OmniRetargeter
from .contact_task_output import ContactTaskOutput, build_contact_task_output
from .robot_config import load_robot_config

__all__ = ["OmniRetargeter", "ContactTaskOutput", "build_contact_task_output",
           "load_robot_config", "__version__"]
