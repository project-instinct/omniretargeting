"""Robot model configuration helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def load_robot_config(config_path: str | Path) -> Dict[str, Any]:
    """
    Load robot configuration from JSON.

    Supported top-level keys:
    - name: Optional profile name
    - urdf_path: Optional path to URDF (relative paths resolved from config dir)
    - joint_mapping: Required mapping of SMPLX joint names to robot body names
    - robot_height: Optional explicit robot height
    - smplx_joint_names: Optional custom SMPLX joint ordering
    - height_estimation: Optional config forwarded to OmniRetargeter
    - base_orientation: Optional config forwarded to OmniRetargeter
    - retargeting: Optional config forwarded to GenericInteractionRetargeter
    """
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Robot config at {path} must be a JSON object.")

    if "joint_mapping" not in raw or not isinstance(raw["joint_mapping"], dict) or not raw["joint_mapping"]:
        raise ValueError("Robot config must contain non-empty 'joint_mapping'.")

    config = dict(raw)
    urdf_path = config.get("urdf_path")
    if urdf_path:
        urdf = Path(urdf_path)
        if not urdf.is_absolute():
            urdf = (path.parent / urdf).resolve()
        config["urdf_path"] = str(urdf)

    return config

