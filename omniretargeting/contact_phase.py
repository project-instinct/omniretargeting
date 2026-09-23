"""Contact roles read from the observed source, not a force certificate.

Every body part is observed through the same UMR surface contacts (III-C):
distance, velocity relative to the touched surface, and whether that surface
is the static scene or a moving object. A contact that holds still on the
static scene supports; any other contact follows; an endpoint observed clear of
every surface is free. Solvers order support > follow > free > posture
(Kanajar et al., RO-MAN 2017, Sec. IV-A: support tasks before posture), so a
free limb yields to any contact and a following contact yields to support.
How far anything yields is decided by reachability, not by weights or ramps.
World anchors, contact timing and source arrays are never changed, and no
force or torque is solved.
"""

from dataclasses import replace

import numpy as np

# Numerical slack when a lower level keeps a higher level's optimum (metres).
PRIORITY_TOLERANCE = 0.001
ROLES = ("support", "follow", "free")


def contact_phase_config(config=None):
    cfg = {"enabled": True, **(config or {})}
    removed = sorted(set(cfg) - {"enabled"})
    if removed:
        raise ValueError(
            "contact_phase_priority only accepts 'enabled'; roles come from the "
            f"observed contacts, so {removed} no longer exist"
        )
    if not isinstance(cfg["enabled"], (bool, np.bool_)):
        raise TypeError("contact_phase_priority.enabled must be boolean")
    return cfg


class ContactRoles:
    """One role table shared by optimization, repair and acceptance.

    An endpoint's primary contact decides its role while active: sticking on
    the static scene supports, sliding, transitions and moving objects follow.
    Otherwise a world-tracked endpoint takes the role of the UMR samples on its
    own rigid part, so feet need no separate contact classifier. A free world
    endpoint yields to contacts only with observed clearance from every
    surface; unknown clearance keeps its existing protection. Dense samples on
    a part whose endpoint already holds a world target only inform that role;
    their approximate learned correspondence is not a second target there.
    """

    def __init__(self, surface, clearances, primary_tasks, priority, *, model=None):
        self.surface = surface
        self.model = surface.model if surface is not None else model
        self.tolerance = PRIORITY_TOLERANCE
        frames = (
            len(surface.targets)
            if surface is not None
            else len(next(iter(primary_tasks.values())).contact_mask)
        )
        weld = self.model.body_weldid
        if surface is not None:
            point_weld = np.array(
                [
                    weld[self.model.geom_bodyid[self.model.geom(b.robot_geom).id]]
                    for b in surface.bindings
                ],
                dtype=int,
            )
            static_id = getattr(surface, "static_surface_id", None)
            active = np.asarray(surface.active, bool)
            still = (
                active
                & (surface.modes == "sticking")
                & (surface.surface_ids == static_id if static_id is not None else False)
            )
        else:
            point_weld = np.zeros(0, int)
            active = still = np.zeros((frames, 0), bool)
        self.roles, self.effector_welds, self.clear = {}, {}, {}
        engaged = np.zeros_like(still)
        for name, spec in priority.get("effectors", {}).items():
            body_weld = weld[int(self.model.body(spec["robot_body"]).id)]
            on_part = point_weld == body_weld
            role = np.full(frames, "free", dtype="U7")
            if spec.get("preserve_world_target", False):
                role[active[:, on_part].any(axis=1)] = "follow"
                role[still[:, on_part].any(axis=1)] = "support"
                clearance = np.asarray(
                    clearances.get(name, np.full(frames, np.nan)), float
                )
                if clearance.shape != (frames,):
                    raise ValueError("Source clearance must share the original timeline")
                # Missing or inside samples are not evidence of leaving a surface.
                self.clear[name] = np.isfinite(clearance)
            task = primary_tasks.get(name)
            if task is not None:
                if len(task.contact_mask) != frames:
                    raise ValueError("Primary contacts must share the original timeline")
                contact = np.asarray(task.contact_mask, bool)
                modes = (
                    np.asarray(task.contact_modes)
                    if task.contact_modes is not None
                    else np.where(contact, "sticking", "none")
                )
                environments = (
                    np.asarray(task.environment_ids)
                    if task.environment_ids is not None
                    else np.zeros(frames, int)
                )
                role[contact] = "follow"
                role[contact & (modes == "sticking") & (environments == 0)] = "support"
            role.setflags(write=False)
            self.roles[name] = role
            self.effector_welds[name] = int(body_weld)
            engaged[:, on_part] |= (role != "free")[:, None]
        self.support = still & ~engaged
        self.follow_surface = active & ~self.support & ~engaged
        self.support.setflags(write=False)
        self.follow_surface.setflags(write=False)
        self.diagnostics = {
            "status": "active"
            if self.support.any() or any((r != "free").any() for r in self.roles.values())
            else "inactive",
            "method": "umr_observed_contact_roles",
            "order": ["support", "follow", "free", "posture"],
            "physical_support_verified": False,
            "force_or_torque_optimization": False,
            "role_frames": {
                name: {
                    r: np.flatnonzero(role == r).tolist() for r in ROLES if (role == r).any()
                }
                for name, role in self.roles.items()
            },
            "surface_support_frame_points": int(self.support.sum()),
            "surface_follow_frame_points": int(self.follow_surface.sum()),
            "surface_support_bodies": sorted(
                {
                    self.model.body(int(w)).name
                    for w in point_weld[self.support.any(axis=0)]
                }
            ),
            "phases": self._phases(point_weld),
        }

    def _phases(self, point_weld):
        phases = []
        for frame, mask in enumerate(self.support):
            state = {
                "support": sorted(
                    [n for n, r in self.roles.items() if r[frame] == "support"]
                    + [self.model.body(int(w)).name for w in set(point_weld[mask])]
                ),
                "follow": sorted(n for n, r in self.roles.items() if r[frame] == "follow"),
                "free": sorted(n for n, r in self.roles.items() if r[frame] == "free"),
            }
            if phases and all(phases[-1][k] == v for k, v in state.items()):
                phases[-1]["stop"] = frame + 1
            else:
                phases.append({"start": frame, "stop": frame + 1, **state})
        return phases

    def role(self, name, frame):
        if frame is None or name not in self.roles:
            return None
        return str(self.roles[name][frame])

    def yields(self, name, frame):
        """A world endpoint observed clear of every surface is not held back."""
        return self.role(name, frame) == "free" and bool(
            self.clear.get(name, np.zeros(len(self.support), bool))[frame]
        )

    def rows(self, pose, frame, dofs, *, role="support"):
        """Observed body-surface samples with UMR's fixed per-part weights."""
        if role not in {"support", "follow"}:
            raise ValueError("Surface contact role must be support or follow")
        mask = self.support if role == "support" else self.follow_surface
        scale = self.surface.cfg["weight"] if self.surface is not None else 1.0
        for index in np.flatnonzero(mask[frame]):
            point, _, jac, _ = self.surface.kinematics.at(pose, int(index))
            yield (
                float(self.surface.weights[frame, index]) / scale,
                point - self.surface.targets[frame, index],
                jac[:, dofs],
            )

    def errors(self, poses):
        return np.array(
            [
                np.sqrt(sum(w * float(r @ r) for w, r, _ in self.rows(q, f, [])))
                for f, q in enumerate(poses)
            ]
        )

    def export(self):
        names = list(self.roles)
        return {
            "contact_role_names": np.array(names),
            "contact_roles": np.stack([self.roles[n] for n in names], axis=1)
            if names
            else np.zeros((len(self.support), 0), "U7"),
            "contact_role_surface_support": self.support,
        }


def phase_policy(*tasks):
    """Read the same shared role table without source-specific solver contracts."""
    return next(
        (
            t.contact_phase
            for t in tasks
            if getattr(t, "contact_phase", None) is not None
        ),
        None,
    )


def attach_contact_phase(
    surface,
    pose,
    primary_tasks,
    priority,
    config=None,
    *,
    model=None,
    source_clearances=None,
):
    """Attach one role table; replace frozen primary tasks without changing arrays.

    ``primary_tasks`` is a mutable task mapping. Primary-only callers pass the
    robot model and observed source clearances, never guessed from robot IK.
    """
    cfg = contact_phase_config(config)
    model = getattr(surface, "model", model)
    observed_surface = surface if hasattr(surface, "kinematics") else None
    clearances = {
        **getattr(surface, "source_clearances", {}),
        **(source_clearances or {}),
    }
    policy = None
    if (
        cfg["enabled"]
        and model is not None
        and (observed_surface is not None or primary_tasks)
    ):
        policy = ContactRoles(
            observed_surface, clearances, primary_tasks, priority, model=model
        )
    for task in (surface, pose):
        if task is not None:
            task.contact_phase = policy
            if hasattr(task, "diagnostics"):
                task.diagnostics["contact_phase_priority"] = (
                    policy.diagnostics if policy else None
                )
    for name, task in primary_tasks.items():
        primary_tasks[name] = replace(task, contact_phase=policy)
    return policy
