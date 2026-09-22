"""Source-neutral kinematic domains for learned surface correspondence.

Geometry alone cannot distinguish two similar limbs. Reuse the motion's joint
map to partition the robot surface before fitting, rather than discovering a
wrong-limb correspondence after it has already driven the trajectory.
"""

from collections import deque

import mujoco
import numpy as np


def surface_domains(model, source, face_geoms, joint_mapping=None):
    """Return source-sample and robot-face domain IDs, or an explicit fallback.

    The robot tree is rerooted at the mapped source root. This also supports
    robots whose floating base is the torso and whose pelvis is a descendant.
    Unmapped source joints inherit the nearest represented source ancestor;
    this fallback is reported, never inferred from a world-space left/right
    test. No source-specific joint or robot-body names live in this module.
    """
    full = source["full_body"]
    names = full.get("joint_names")
    parents = full.get("joint_parents")
    sample_joints = full.get("sample_joint_indices")
    if not joint_mapping or names is None or parents is None or sample_joints is None:
        return (
            None,
            None,
            {"status": "unavailable", "reason": "missing_kinematic_metadata"},
        )
    names = list(names)
    parents = np.asarray(parents, dtype=int)
    sample_joints = np.asarray(sample_joints, dtype=int)
    if (
        parents.shape != (len(names),)
        or sample_joints.shape != (len(full["face_ids"]),)
        or np.any(sample_joints < 0)
        or np.any(sample_joints >= len(names))
        or np.any(parents >= len(names))
    ):
        raise ValueError(
            "Surface kinematic metadata must match source joints and samples"
        )
    anchors = {}
    for index, name in enumerate(names):
        body = joint_mapping.get(name)
        if body is not None:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, str(body))
            if bid > 0:
                anchors[index] = bid
    roots = np.flatnonzero(parents < 0)
    if len(roots) != 1 or int(roots[0]) not in anchors:
        return (
            None,
            None,
            {"status": "unavailable", "reason": "missing_mapped_source_root"},
        )
    resolved, inherited = {}, {}
    for index in range(len(names)):
        current, visited = index, set()
        while current not in anchors and current >= 0:
            if current in visited:
                raise ValueError("Surface source joint parents contain a cycle")
            visited.add(current)
            current = int(parents[current])
        if current < 0:
            raise ValueError(
                "Surface source joints must connect to the mapped source root"
            )
        resolved[index] = anchors[current]
        if current != index:
            inherited[names[index]] = names[current]

    neighbors = [[] for _ in range(model.nbody)]
    for body in range(1, model.nbody):
        parent = int(model.body_parentid[body])
        if parent > 0:
            neighbors[parent].append(body)
            neighbors[body].append(parent)
    anchor_bodies = set(anchors.values())
    root = anchors[int(roots[0])]
    owner = np.full(model.nbody, -1, dtype=int)
    owner[root] = root
    queue = deque([root])
    while queue:
        body = queue.popleft()
        for child in neighbors[body]:
            if owner[child] >= 0:
                continue
            owner[child] = child if child in anchor_bodies else owner[body]
            queue.append(child)
    face_domains = owner[
        np.asarray(model.geom_bodyid)[np.asarray(face_geoms, dtype=int)]
    ]
    source_domains = np.array([resolved[int(j)] for j in sample_joints], dtype=int)
    missing = sorted(set(source_domains) - set(face_domains))
    return (
        source_domains,
        face_domains,
        {
            "status": "active",
            "method": "source_joint_ancestry_and_rerooted_robot_tree",
            "inherited_source_joints": inherited,
            "domains_without_surface": [model.body(i).name for i in missing],
            "source_samples_per_domain": {
                model.body(int(i)).name: int(np.count_nonzero(source_domains == i))
                for i in np.unique(source_domains)
            },
        },
    )
