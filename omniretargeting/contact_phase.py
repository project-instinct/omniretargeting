"""Contact roles read from the observed source, not a force certificate.

Every body part is observed through the same UMR surface contacts (III-C).
Source net-wrench evidence, when supplied, determines support priority for ANY
body region, including sliding contacts and prescribed moving objects. Without
that evidence, stationary scene contacts retain geometric holding priority;
this fallback has the separate, explicitly unverified ``hold`` role.
An endpoint observed clear of every surface is free. Solvers order
support > unresolved hold > control > follow > free > posture
(Kanajar et al., RO-MAN 2017, Sec. IV-A: support tasks before posture), so a
free limb yields to any contact and a following contact yields to support.
How far anything yields is decided by reachability, not by weights or ramps.
World anchors, contact timing and source arrays are never changed, and no
force or torque is solved.
"""

from dataclasses import replace

import numpy as np

from .contact_coverage import attach_surface_coverage, surface_effector_samples

# Numerical slack when a lower level keeps a higher level's optimum (metres).
PRIORITY_TOLERANCE = 0.001
# Unexplained holds stay conservative: a moving object is not evidence that an
# otherwise unexplained standing foot can be lifted. With complete mechanics,
# optional contacts become follow and object control is immediately below support.
CONTACT_ROLES = ("support", "hold", "control", "follow")
ROLES = (*CONTACT_ROLES, "free")


def priority_progress(before, after, tolerance):
    """Accept lower-role tradeoffs only where an earlier role improves.

    Rows are frames, columns are priority levels. A support improvement in one
    frame cannot excuse losing support in another; aggregate errors cannot hide
    that regression. Hard posture/continuity/attained-contact guards are separate.
    """
    before, after = np.asarray(before), np.asarray(after)
    improved = np.zeros(len(before), bool)
    valid = np.ones(len(before), bool)
    for old, new in zip(before.T, after.T):
        valid &= improved | (new <= old + tolerance)
        improved |= new < old - tolerance
    return bool(valid.all()), bool(valid.all() and improved.any())


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

    Source mechanics can assign support to any active region/surface pair.
    Without mechanics, sticking only supplies a geometric hold heuristic;
    it must not be interpreted as evidence that other contacts cannot support.
    Otherwise a world-tracked endpoint takes the role of the UMR samples on its
    own rigid part, so feet need no separate contact classifier. A free world
    endpoint yields to contacts only with observed clearance from every
    surface; unknown clearance keeps its existing protection. Dense samples on
    a part whose endpoint already holds a world target only inform that role;
    their approximate learned correspondence is not a second target there.
    """

    def __init__(
        self,
        surface,
        clearances,
        primary_tasks,
        priority,
        *,
        model=None,
        support_evidence=None,
        control_evidence=None,
    ):
        self.surface = surface
        self.model = surface.model if surface is not None else model
        self.tolerance = PRIORITY_TOLERANCE
        frames = (
            len(surface.targets)
            if surface is not None
            else len(next(iter(primary_tasks.values())).contact_mask)
        )
        if support_evidence is not None and support_evidence.frames != frames:
            raise ValueError("Source support evidence must share the original timeline")
        if control_evidence is not None and control_evidence.frames != frames:
            raise ValueError(
                "Source object-control evidence must share the original timeline"
            )

        def observed_role(region, surface_id, frame, sticking):
            support = (
                None
                if support_evidence is None
                else support_evidence.role(region, surface_id, frame)
            )
            if support == "support":
                return "support"
            if (
                control_evidence is not None
                and control_evidence.role(region, surface_id, frame) == "control"
            ):
                return "control"
            # Optional is not the same as unloaded. It retains its actual
            # contact task, but necessity alone cannot promote it to support.
            return support or ("hold" if sticking else "follow")

        weld = self.model.body_weldid
        if surface is not None:
            point_weld = np.array(
                [
                    weld[self.model.geom_bodyid[self.model.geom(b.robot_geom).id]]
                    for b in surface.bindings
                ],
                dtype=int,
            )
            active = np.asarray(surface.active, bool)
            surface_roles = np.full(active.shape, "free", dtype="U7")
            for index, binding in enumerate(surface.bindings):
                for f in np.flatnonzero(active[:, index]):
                    surface_roles[f, index] = observed_role(
                        binding.part,
                        surface.surface_ids[f, index],
                        f,
                        surface.modes[f, index] == "sticking",
                    )
        else:
            point_weld = np.zeros(0, int)
            active = np.zeros((frames, 0), bool)
            surface_roles = np.full(active.shape, "free", dtype="U7")
        self.roles, self.effector_welds, self.clear = {}, {}, {}
        engaged = np.zeros_like(active)
        for name, spec in priority.get("effectors", {}).items():
            task = primary_tasks.get(name)
            managed = np.zeros(frames, bool) if task is None else task.managed_frames
            dense = task is not None and task.surface_managed
            body_weld = weld[int(self.model.body(spec["robot_body"]).id)]
            on_part = point_weld == body_weld
            region = spec.get("source_contact_part", name)
            # Sharing a robot rigid body does not make two source contact
            # regions interchangeable (e.g. palm and supporting forearm).
            if surface is not None:
                on_part &= surface_effector_samples(surface, name, spec)
            role = np.full(frames, "free", dtype="U7")
            if spec.get("preserve_world_target", False) or dense:
                for label in reversed(CONTACT_ROLES):
                    role[(surface_roles[:, on_part] == label).any(axis=1)] = label
                clearance = np.asarray(
                    clearances.get(name, np.full(frames, np.nan)), float
                )
                if clearance.shape != (frames,):
                    raise ValueError(
                        "Source clearance must share the original timeline"
                    )
                # Missing or inside samples are not evidence of leaving a surface.
                self.clear[name] = np.isfinite(clearance)
            if task is not None:
                if len(task.contact_mask) != frames:
                    raise ValueError(
                        "Primary contacts must share the original timeline"
                    )
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
                for f in np.flatnonzero(contact & ~managed):
                    role[f] = observed_role(
                        region, environments[f], f, modes[f] == "sticking"
                    )
            role.setflags(write=False)
            self.roles[name] = role
            self.effector_welds[name] = int(body_weld)
            duplicate = (role != "free")[:, None] & on_part[None]
            if task is not None and surface is not None:
                duplicate &= ~contact[:, None] | (
                    surface.surface_ids == environments[:, None]
                )
            engaged |= duplicate & ~managed[:, None]
        self.surface_roles = {
            label: (surface_roles == label) & ~engaged for label in CONTACT_ROLES
        }
        for mask in self.surface_roles.values():
            mask.setflags(write=False)
        self.support = self.surface_roles["support"]
        self.follow_surface = self.surface_roles["follow"]
        self.diagnostics = {
            "status": "active"
            if active.any() or any((r != "free").any() for r in self.roles.values())
            else "inactive",
            "method": "umr_observed_contact_roles",
            "order": [*ROLES, "posture"],
            "physical_support_verified": False,
            "force_or_torque_optimization": False,
            "source_support_evidence": None
            if support_evidence is None
            else support_evidence.diagnostics,
            "source_control_evidence": None
            if control_evidence is None
            else control_evidence.diagnostics,
            "fallback_roles_are_geometric_only": True,
            "role_frames": {
                name: {
                    r: np.flatnonzero(role == r).tolist()
                    for r in ROLES
                    if (role == r).any()
                }
                for name, role in self.roles.items()
            },
            "surface_support_frame_points": int(self.support.sum()),
            "surface_follow_frame_points": int(self.follow_surface.sum()),
            "surface_role_frame_points": {
                r: int(m.sum()) for r, m in self.surface_roles.items()
            },
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
        for frame in range(len(self.support)):
            state = {}
            for label in ROLES:
                names = [n for n, r in self.roles.items() if r[frame] == label]
                if label in self.surface_roles:
                    mask = self.surface_roles[label][frame]
                    names += [
                        self.model.body(int(w)).name for w in set(point_weld[mask])
                    ]
                state[label] = sorted(set(names))
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
        if role not in CONTACT_ROLES:
            raise ValueError(f"Surface contact role must be one of {CONTACT_ROLES}")
        mask = self.surface_roles[role]
        scale = self.surface.cfg["weight"] if self.surface is not None else 1.0
        rows = () if self.surface is None else self.surface.position_rows(pose, frame, dofs)
        for index, residual, jac in rows:
            if not mask[frame, index]:
                continue
            yield (
                float(self.surface.weights[frame, index]) / scale,
                residual,
                jac,
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
            **{f"contact_role_surface_{r}": m for r, m in self.surface_roles.items()},
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
    support_evidence=None,
    control_evidence=None,
):
    """Attach one role table; replace frozen primary tasks without changing arrays.

    ``primary_tasks`` is a mutable task mapping. Primary-only callers pass the
    robot model and observed source clearances, never guessed from robot IK.
    """
    cfg = contact_phase_config(config)
    model = getattr(surface, "model", model)
    observed_surface = surface if hasattr(surface, "kinematics") else None
    attach_surface_coverage(observed_surface, primary_tasks, priority)
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
            observed_surface,
            clearances,
            primary_tasks,
            priority,
            model=model,
            support_evidence=support_evidence,
            control_evidence=control_evidence,
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
