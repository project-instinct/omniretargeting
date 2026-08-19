"""OmniRetargeting: Generic motion retargeting for any humanoid URDF and terrain mesh."""

from .__version__ import __version__
from .core import OmniRetargeter
from .robot_config import load_robot_config

__all__ = ["OmniRetargeter", "load_robot_config", "__version__"]
