"""Robot model configuration helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def _resolve_path(value: str | None, config_dir: Path) -> str | None:
    if not value:
        return value
    path = Path(value)
    if not path.is_absolute():
        path = (config_dir / path).resolve()
    return str(path)


def _ensure_dict(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Robot config '{name}' must be a JSON object.")
    return dict(value)


def _normalize_source_entry(entry: dict[str, Any]) -> dict[str, Any]:
    source = dict(entry)
    if "target_names" not in source and "joint_names" in source:
        source["target_names"] = source["joint_names"]
    if "target_names" not in source and "smplx_joint_names" in source:
        source["target_names"] = source["smplx_joint_names"]
    if "target_mapping" not in source and "joint_mapping" in source:
        source["target_mapping"] = source["joint_mapping"]
    return source


def _select_source(config: dict[str, Any]) -> dict[str, Any]:
    sources = config.get("source") or []
    if not isinstance(sources, list):
        raise ValueError("Robot config 'source' must be a JSON array.")
    if not sources:
        return {}

    normalized = []
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("Robot config 'source' entries must be JSON objects.")
        normalized.append(_normalize_source_entry(source))
    config["source"] = normalized

    active_source = config.get("active_source")
    if active_source is None:
        return normalized[0]

    matches = [s for s in normalized if s.get("name") == active_source]
    if not matches:
        matches = [s for s in normalized if s.get("type") == active_source]
    if len(matches) != 1:
        raise ValueError(f"Robot config active_source '{active_source}' does not select exactly one source entry.")
    return matches[0]


def load_robot_config(config_path: str | Path) -> Dict[str, Any]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Robot config at {path} must be a JSON object.")

    config = dict(raw)
    robot = _ensure_dict(config.get("robot"), "robot")
    retargeting = _ensure_dict(config.get("retargeting"), "retargeting")
    selected_source = _select_source(config)

    if "solver" in retargeting:
        solver = _ensure_dict(retargeting.get("solver"), "retargeting.solver")
        merged_retargeting = {k: v for k, v in retargeting.items() if k != "solver"}
        merged_retargeting.update(solver)
        retargeting = merged_retargeting

    config["robot"] = robot
    config["retargeting"] = retargeting
    config["selected_source"] = selected_source

    urdf_path = robot.get("urdf_path", config.get("urdf_path"))
    if urdf_path:
        config["urdf_path"] = _resolve_path(urdf_path, path.parent)

    robot_height = robot.get("height", config.get("robot_height"))
    if robot_height is not None:
        config["robot_height"] = robot_height

    if "link_offset_config" in robot or "link_offset_config" in config:
        config["link_offset_config"] = robot.get("link_offset_config", config.get("link_offset_config"))

    target_mapping = selected_source.get("target_mapping", config.get("joint_mapping"))
    if not isinstance(target_mapping, dict) or not target_mapping:
        raise ValueError("Robot config must contain non-empty 'joint_mapping' or selected source 'target_mapping'.")
    config["joint_mapping"] = target_mapping
    config["target_mapping"] = target_mapping

    target_names = selected_source.get("target_names", config.get("smplx_joint_names"))
    if target_names is not None:
        config["smplx_joint_names"] = target_names
        config["source_target_names"] = target_names

    for key in ("height_estimation", "base_orientation"):
        if key in selected_source:
            config[key] = selected_source[key]

    if "betas" in selected_source and "smplx_betas" not in config:
        config["smplx_betas"] = selected_source["betas"]
    if "gender" in selected_source:
        config["source_gender"] = selected_source["gender"]
    if "smpl_model_dir" in selected_source:
        config["smpl_model_dir"] = selected_source["smpl_model_dir"]

    joint_pos_fitting_smplx = config.get("joint_pos_fitting_smplx")
    if joint_pos_fitting_smplx is not None and not isinstance(joint_pos_fitting_smplx, dict):
        raise ValueError("Robot config 'joint_pos_fitting_smplx' must be a JSON object.")

    smplx_betas = config.get("smplx_betas")
    if smplx_betas is not None:
        if not isinstance(smplx_betas, list) or not all(isinstance(v, (int, float)) for v in smplx_betas):
            raise ValueError("Robot config 'smplx_betas' must be a list of numbers.")

    return config
