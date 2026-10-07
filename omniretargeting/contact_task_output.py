"""Versioned, source-neutral contact/task output alongside the complete qpos.

Slots are the Cartesian product of body parts and observed scene surfaces.
This preserves simultaneous contacts without averaging targets off a surface.
Importance describes downstream task priority, not solver costs or forces.
"""

from collections.abc import Mapping
from dataclasses import dataclass, fields

import numpy as np

SCHEMA_VERSION = 1
DEFAULT_CANDIDATE_PARTS = (
    "left_hand", "right_hand", "left_foot", "right_foot",
    "left_knee", "right_knee", "left_forearm", "right_forearm", "pelvis", "abdomen",
)
DEFAULT_IMPORTANCE = {"support": 1.0, "hold": 0.75, "control": 0.5, "follow": 0.25}
CONTACT_TYPES = ("none", "support", "control", "touch", "slide")
CONTACT_MODES = ("none", "touch", "sticking", "sliding", "unknown")
CONTACT_ROLES = ("free", *DEFAULT_IMPORTANCE)


def contact_task_output_config(config=None):
    from .config_validation import known_options

    defaults = {
        "candidate_parts": DEFAULT_CANDIDATE_PARTS,
        "body_part_regions": {},
        "importance_weights": DEFAULT_IMPORTANCE,
    }
    known_options(config, defaults, "contact_task_output")
    cfg = {**defaults, **(config or {})}
    cfg["candidate_parts"] = _names(cfg["candidate_parts"], "candidate_parts")
    groups = cfg["body_part_regions"]
    if not isinstance(groups, Mapping):
        raise TypeError("contact_task_output.body_part_regions must be a mapping")
    cfg["body_part_regions"] = {
        _name(part): _names(regions, "body_part_regions") for part, regions in groups.items()
    }
    weights = cfg["importance_weights"]
    if not isinstance(weights, Mapping):
        raise TypeError("contact_task_output.importance_weights must be a mapping")
    known_options(weights, DEFAULT_IMPORTANCE, "contact_task_output.importance_weights")
    cfg["importance_weights"] = {**DEFAULT_IMPORTANCE, **weights}
    values = np.asarray(list(cfg["importance_weights"].values()), float)
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Contact task importance weights must be finite and in [0, 1]")
    return cfg


def _name(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Contact task body parts and region names must be nonempty strings")
    return value


def _names(values, option):
    if isinstance(values, str):
        raise TypeError(f"contact_task_output.{option} must be a sequence of names")
    return tuple(_name(value) for value in values)


@dataclass(frozen=True)
class ContactTaskOutput:
    """Immutable arrays with (T, K) scalars and (T, K, 3) world geometry.

    Inactive slots have zero importance/confidence, no type/mode/role, -1
    scene_object_ids, and NaN targets/normals. ``observed`` distinguishes a
    valid source observation from absent/invalid material samples; it does
    not certify that the robot attained a task or passed collision checks.
    """

    body_parts: np.ndarray                 # (K,), repeated once per surface
    slot_scene_object_ids: np.ndarray      # (K,), indexes scene_object_names
    scene_object_names: np.ndarray         # (S,), includes static "scene"
    timestamps_s: np.ndarray               # (T,), relative to clip start
    observed: np.ndarray
    active: np.ndarray
    contact_types: np.ndarray
    contact_modes: np.ndarray
    roles: np.ndarray
    importance_weights: np.ndarray
    confidence: np.ndarray
    target_positions_w: np.ndarray
    surface_normals_w: np.ndarray
    scene_object_ids: np.ndarray
    source_sample_indices: np.ndarray     # -1 for endpoint-only/unknown slots

    def __post_init__(self):
        # Own the buffers: consumers cannot change source or solver evidence.
        for field in fields(self):
            value = np.array(getattr(self, field.name), copy=True)
            value.setflags(write=False)
            object.__setattr__(self, field.name, value)
        t, k = len(self.timestamps_s), len(self.body_parts)
        shape = (t, k)
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name in {"body_parts", "slot_scene_object_ids"}:
                expected = (k,)
            elif field.name == "scene_object_names":
                expected = (len(value),)
            elif field.name == "timestamps_s":
                expected = (t,)
            else:
                expected = (*shape, 3) if field.name in {
                    "target_positions_w", "surface_normals_w"
                } else shape
            if value.shape != expected or value.dtype.hasobject:
                raise ValueError(f"Contact task {field.name} must have shape {expected} without objects")
        if not np.isfinite(self.timestamps_s).all() or np.any(np.diff(self.timestamps_s) <= 0):
            raise ValueError("Contact task timestamps must be finite and increasing")
        for array in (self.importance_weights, self.confidence):
            if not np.isfinite(array).all() or np.any((array < 0) | (array > 1)):
                raise ValueError("Contact task weights and confidence must be in [0, 1]")
        for array, choices in ((self.contact_types, CONTACT_TYPES),
                               (self.contact_modes, CONTACT_MODES), (self.roles, CONTACT_ROLES)):
            if not np.isin(array, choices).all():
                raise ValueError("Unknown contact task type, mode or role")
        if np.any(self.active & ~self.observed):
            raise ValueError("Active contact tasks require observed material evidence")
        ids = self.slot_scene_object_ids
        if np.any((ids < -1) | (ids >= len(self.scene_object_names))):
            raise ValueError("Contact task scene object IDs must address the surface registry")
        if np.any(self.active & (self.scene_object_ids != ids[None])) or np.any(
            self.active & (self.scene_object_ids < 0)
        ):
            raise ValueError("Active contact tasks must address their slot's scene object")
        if (not np.isfinite(self.target_positions_w[self.active]).all()
                or not np.isfinite(self.surface_normals_w[self.active]).all()
                or not np.allclose(np.linalg.norm(self.surface_normals_w[self.active], axis=-1), 1)):
            raise ValueError("Active contact tasks require finite targets and unit surface normals")
        inactive = ~self.active
        if (np.any(self.importance_weights[inactive]) or np.any(self.confidence[inactive])
                or np.any(self.scene_object_ids[inactive] != -1)
                or np.any(self.contact_types[inactive] != "none")
                or np.any(self.contact_modes[inactive] != "none")
                or np.any(self.roles[inactive] != "free")
                or not np.isnan(self.target_positions_w[inactive]).all()
                or not np.isnan(self.surface_normals_w[inactive]).all()):
            raise ValueError("Inactive contact tasks must use the documented empty values")

    def export(self):
        """A flat NPZ payload readable with allow_pickle=False."""
        return {
            "contact_task_schema_version": np.asarray(SCHEMA_VERSION, dtype=np.int32),
            **{f"contact_task_{field.name}": getattr(self, field.name) for field in fields(self)},
        }

    def frame(self, frame_idx):
        """Plain records for every slot, including inactive and unknown ones."""
        if not 0 <= frame_idx < len(self.timestamps_s):
            raise IndexError("Contact task frame index is outside the clip")
        records = []
        for slot, part in enumerate(self.body_parts):
            active = bool(self.active[frame_idx, slot])
            object_id = int(self.scene_object_ids[frame_idx, slot])
            records.append({
                "body_part": str(part),
                "observed": bool(self.observed[frame_idx, slot]),
                "active": active,
                "contact_type": str(self.contact_types[frame_idx, slot]),
                "contact_mode": str(self.contact_modes[frame_idx, slot]),
                "role": str(self.roles[frame_idx, slot]),
                "importance_weight": float(self.importance_weights[frame_idx, slot]),
                "confidence": float(self.confidence[frame_idx, slot]),
                "target_position_w": self.target_positions_w[frame_idx, slot].tolist() if active else None,
                "surface_normal_w": self.surface_normals_w[frame_idx, slot].tolist() if active else None,
                "scene_object_id": object_id,
                "scene_object": str(self.scene_object_names[object_id]) if active else None,
                "slot_scene_object_id": int(self.slot_scene_object_ids[slot]),
                "source_sample_index": int(self.source_sample_indices[frame_idx, slot]),
            })
        return records


def _role(region, surface, frame, mode, support, control, unresolved):
    role = None if support is None else support.role(region, surface, frame)
    if role == "support":
        return role
    if unresolved[frame]:
        return "hold"
    if control is not None and control.role(region, surface, frame) == "control":
        return "control"
    return role or ("hold" if mode == "sticking" else "follow")


def _activate(arrays, frame, slot, target, normal, confidence, mode, role, surface, sample=-1):
    normal = np.asarray(normal, float)
    length = np.linalg.norm(normal)
    if not np.isfinite(target).all() or not np.isfinite(length) or length < 1e-8:
        return
    arrays["observed"][frame, slot] = True
    arrays["active"][frame, slot] = True
    arrays["confidence"][frame, slot] = confidence
    arrays["contact_modes"][frame, slot] = mode if mode in CONTACT_MODES[1:] else "unknown"
    arrays["roles"][frame, slot] = role
    arrays["contact_types"][frame, slot] = (
        role if role in {"support", "control"} else "slide" if mode == "sliding" else "touch"
    )
    arrays["target_positions_w"][frame, slot] = target
    arrays["surface_normals_w"][frame, slot] = normal / length
    arrays["scene_object_ids"][frame, slot] = surface
    arrays["source_sample_indices"][frame, slot] = sample


def build_contact_task_output(
    source_table=None, *, frames, framerate, primary_tasks=None,
    scene_object_names=(), support_evidence=None, control_evidence=None,
    unresolved_support_frames=None, config=None,
):
    """Read source evidence without altering qpos, source arrays or task inputs.

    One highest-priority, then highest-confidence active sample represents
    each region/surface pair (ties use source sample order). Geometry stays
    on that actual surface.
    Missing anatomy stays unknown; explicit region groups only rename/group
    existing evidence. Primary tasks are a fallback for parts without samples.
    """
    if not isinstance(frames, (int, np.integer)) or frames <= 0:
        raise ValueError("Contact task output requires a positive frame count")
    if not np.isfinite(framerate) or framerate <= 0:
        raise ValueError("Contact task output requires a positive framerate")
    cfg = contact_task_output_config(config)
    primary_tasks = primary_tasks or {}
    regions = np.asarray([] if source_table is None else source_table.regions, str)
    parts = sorted(set(cfg["candidate_parts"]) | set(regions) | set(primary_tasks)
                   | set(cfg["body_part_regions"]))
    if source_table is not None:
        if len(source_table.points) != frames:
            raise ValueError("Source contacts must share the q_ref timeline")
        scene_object_names = source_table.observation["diagnostics"]["surface_names"]
    names = np.asarray(scene_object_names, dtype=str)
    surfaces = range(len(names)) if len(names) else (-1,)
    slots = [(part, surface) for part in parts for surface in surfaces]
    shape = (frames, len(slots))
    arrays = {
        "body_parts": np.asarray([part for part, _ in slots], dtype=str),
        "slot_scene_object_ids": np.asarray([surface for _, surface in slots], dtype=np.int32),
        "scene_object_names": names,
        "timestamps_s": np.arange(frames, dtype=float) / framerate,
        "observed": np.zeros(shape, bool), "active": np.zeros(shape, bool),
        "contact_types": np.full(shape, "none", dtype="U7"),
        "contact_modes": np.full(shape, "none", dtype="U8"),
        "roles": np.full(shape, "free", dtype="U7"),
        "importance_weights": np.zeros(shape), "confidence": np.zeros(shape),
        "target_positions_w": np.full((*shape, 3), np.nan),
        "surface_normals_w": np.full((*shape, 3), np.nan),
        "scene_object_ids": np.full(shape, -1, dtype=np.int32),
        "source_sample_indices": np.full(shape, -1, dtype=np.int64),
    }
    for evidence in (support_evidence, control_evidence):
        if evidence is not None and evidence.frames != frames:
            raise ValueError("Contact task evidence must share the q_ref timeline")
    unresolved = np.zeros(frames, bool)
    if unresolved_support_frames is None and support_evidence is not None:
        unexplained = np.asarray(support_evidence.diagnostics.get("unexplained_frames", []), dtype=int)
        unresolved[unexplained] = True
    elif unresolved_support_frames is not None:
        unresolved = np.asarray(unresolved_support_frames, bool)
        if unresolved.shape != (frames,):
            raise ValueError("Unresolved support must share the q_ref timeline")
    for slot, (part, surface) in enumerate(slots):
        if surface < 0:
            continue
        group = cfg["body_part_regions"].get(part, (part,))
        samples = np.flatnonzero(np.isin(regions, group))
        if len(samples):
            obs = source_table.observation
            for frame in range(frames):
                valid = source_table.valid[frame, samples] & (
                    obs["surface_ids"][frame, samples] == surface
                )
                valid &= np.isfinite(obs["environment"][frame, samples]).all(-1)
                valid &= np.isfinite(obs["normals"][frame, samples]).all(-1)
                valid &= np.linalg.norm(obs["normals"][frame, samples], axis=-1) > 1e-8
                arrays["observed"][frame, slot] = valid.any()
                active = samples[valid & source_table.active[frame, samples]]
                if not len(active):
                    continue
                # The representative itself must carry the exported role.
                # Priority precedes confidence when explicit groups merge regions.
                roles = [_role(str(regions[sample]), surface, frame,
                               str(obs["modes"][frame, sample]), support_evidence,
                               control_evidence, unresolved) for sample in active]
                order = tuple(DEFAULT_IMPORTANCE)
                selected = min(range(len(active)), key=lambda i: (
                    order.index(roles[i]), -obs["confidence"][frame, active[i]], int(active[i])
                ))
                sample, role = active[selected], roles[selected]
                mode = str(obs["modes"][frame, sample])
                _activate(arrays, frame, slot, obs["environment"][frame, sample],
                          obs["normals"][frame, sample], obs["confidence"][frame, sample],
                          mode, role, surface, sample)
        elif part in primary_tasks:
            task = primary_tasks[part]
            if len(task.contact_mask) != frames:
                raise ValueError("Primary contacts must share the q_ref timeline")
            environments = (np.zeros(frames, int) if task.environment_ids is None
                            else np.asarray(task.environment_ids))
            observed = np.isfinite(task.distances) & (environments == surface)
            observed &= np.isfinite(task.surface_points).all(-1)
            observed &= np.isfinite(task.surface_normals).all(-1)
            observed &= np.linalg.norm(task.surface_normals, axis=-1) > 1e-8
            arrays["observed"][:, slot] = observed
            for frame in np.flatnonzero(np.asarray(task.contact_mask, bool) & (environments == surface)):
                mode = "sticking" if task.contact_modes is None else str(task.contact_modes[frame])
                role = _role(part, surface, frame, mode, support_evidence, control_evidence, unresolved)
                if task.contact_phase is not None:
                    phase_role = task.contact_phase.role(part, frame)
                    if phase_role in DEFAULT_IMPORTANCE:
                        role = phase_role
                _activate(arrays, frame, slot, task.contact_positions[frame],
                          task.surface_normals[frame], task.contact_confidence[frame], mode, role, surface)
    for role, weight in cfg["importance_weights"].items():
        arrays["importance_weights"][arrays["roles"] == role] = weight
    return ContactTaskOutput(**arrays)
