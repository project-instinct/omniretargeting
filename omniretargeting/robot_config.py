"""Robot model configuration helpers."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List


def load_robot_config(config_path: str | Path) -> Dict[str, Any]:
    """
    Load robot configuration from JSON.

    Supported top-level keys:
    - name: Optional profile name
    - urdf_path: Optional path to URDF, or a list of candidates tried in order
      until one exists. ``~`` and ``${VAR}`` are expanded; relative paths are
      resolved from the config dir. A list lets a profile point at assets that
      live outside the repo without hard-coding one machine's layout.
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
        config["urdf_path"] = _resolve_urdf_path(urdf_path, path)

    return config


_UNEXPANDED_VAR = re.compile(r"\$\{?(\w+)\}?")


def _resolve_urdf_path(urdf_path: Any, config_path: Path) -> str:
    """Resolve urdf_path to the first candidate that exists on this machine.

    A profile may give a single path or a list of candidates. Candidates keep
    ``~`` and ``${VAR}`` unexpanded on disk so the same profile works on hosts
    that store the robot assets under different roots.
    """
    candidates = urdf_path if isinstance(urdf_path, (list, tuple)) else [urdf_path]
    if not candidates:
        raise ValueError(f"Robot config at {config_path} has an empty 'urdf_path'.")

    tried: List[str] = []
    for candidate in candidates:
        expanded = os.path.expandvars(os.path.expanduser(str(candidate)))
        if _UNEXPANDED_VAR.search(expanded):
            # The variable is not set on this host; this candidate cannot apply.
            tried.append(f"{candidate} (unset variable)")
            continue
        urdf = Path(expanded)
        if not urdf.is_absolute():
            urdf = (config_path.parent / urdf).resolve()
        if urdf.exists():
            return str(urdf)
        tried.append(str(urdf))

    # A single candidate keeps the original behaviour: return the resolved path
    # and let the model loader report the missing file.
    if len(candidates) == 1 and not _UNEXPANDED_VAR.search(
        os.path.expandvars(os.path.expanduser(str(candidates[0])))
    ):
        single = Path(os.path.expandvars(os.path.expanduser(str(candidates[0]))))
        if not single.is_absolute():
            single = (config_path.parent / single).resolve()
        return str(single)

    raise FileNotFoundError(
        f"Robot config at {config_path} lists no reachable 'urdf_path' on this "
        f"machine. Tried: " + ", ".join(tried)
    )

