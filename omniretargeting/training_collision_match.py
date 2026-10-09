"""Match an audited robot collision manifest to training-model provenance."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from .collision_manifest import collision_manifest, compare_collision_manifests
from .robot_config import load_robot_config

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAINING_PROFILE = ROOT / "robot_models/unitree_g1/unitree_g1_mesh.json"


def _scalar(value: Any) -> Any:
    array = np.asarray(value)
    if array.dtype == object and array.size == 1:
        return array.item()
    if array.ndim == 0:
        return array.item()
    return value


def _load_training_manifest(
    training_model_path: str | Path | None,
) -> dict[str, Any]:
    """Load an explicit model, or the user's declared mesh collision contract.

    This establishes asset identity, not evidence that a running training job
    loaded the asset. Environment collision representation is audited separately.
    """
    path: Path | None
    if training_model_path is None:
        try:
            path = Path(load_robot_config(DEFAULT_TRAINING_PROFILE)["urdf_path"])
            reason = "declared mesh collision contract model was not found"
        except (OSError, ValueError, KeyError) as error:
            return {
                "status": "unverified",
                "path": str(DEFAULT_TRAINING_PROFILE),
                "reason": f"could not resolve default training profile: {error}",
            }
    else:
        path = Path(str(training_model_path)).expanduser()
        reason = "training model was not found"
    resolved = path.resolve()
    if not resolved.is_file():
        return {"status": "unverified", "path": str(resolved), "reason": reason}
    try:
        model = mujoco.MjModel.from_xml_path(str(resolved))
        manifest = collision_manifest(model, source_path=resolved)
    except (OSError, ValueError, RuntimeError) as error:
        return {
            "status": "unverified",
            "path": str(resolved),
            "reason": f"could not compile training model: {type(error).__name__}: {error}",
        }
    return {"status": "loaded", "path": str(resolved), "manifest": manifest}


def training_collision_match(
    audited_manifest: dict[str, Any],
    motion_payload: dict[str, Any],
    training_model_path: str | Path | None = None,
    *,
    legacy_asset_sha256: str | None = None,
) -> dict[str, Any]:
    """Compare motion provenance and an optional training MJCF to the audit."""
    expected = _scalar(motion_payload.get("collision_model_sha256"))
    if expected is None:
        motion_check: dict[str, Any] = {
            "status": "unverified",
            "reason": "motion payload has no collision_model_sha256",
        }
    else:
        expected_hash = str(expected)
        motion_check = {
            "status": "matched" if expected_hash == audited_manifest.get("sha256") else "mismatched",
            "expected_sha256": expected_hash,
            "audited_sha256": audited_manifest.get("sha256"),
        }

    loaded = _load_training_manifest(training_model_path)
    training_check: dict[str, Any] = {"status": loaded["status"], "path": loaded["path"]}
    if "manifest" in loaded:
        comparison = compare_collision_manifests(audited_manifest, loaded["manifest"])
        training_check.update(
            {
                "status": comparison["status"],
                "audited_sha256": comparison["first_sha256"],
                "training_sha256": comparison["second_sha256"],
                "difference": comparison["difference"],
            }
        )
    else:
        training_check["reason"] = loaded.get("reason", "training model was not loaded")

    checks = [motion_check["status"], training_check["status"]]
    if "unverified" in checks:
        status = "unverified"
    elif "mismatched" in checks:
        status = "mismatched"
    else:
        status = "matched"
    result = {
        "status": status,
        "motion_payload": motion_check,
        "training_model": training_check,
    }
    if training_check.get("difference") is not None:
        result["geom_difference"] = training_check["difference"]

    # Preserve acceptance of the older XML+all-mesh asset identifier when it is
    # explicitly supplied. It is diagnostic provenance, not a collision manifest.
    legacy = _scalar(motion_payload.get("training_model"))
    if isinstance(legacy, dict) and legacy_asset_sha256 is not None:
        legacy_hash = legacy.get("sha256")
        result["legacy_asset_match"] = {
            "status": (
                "unverified"
                if not legacy_hash
                else "matched"
                if str(legacy_hash) == legacy_asset_sha256
                else "mismatched"
            ),
            "expected_sha256": None if legacy_hash is None else str(legacy_hash),
            "audited_asset_sha256": legacy_asset_sha256,
        }
    return result
