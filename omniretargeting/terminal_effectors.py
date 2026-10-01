"""Infer mapped-chain terminals and their rigid collision shells from the model."""

from copy import deepcopy

import mujoco
import numpy as np

from .scene_geometry import collision_geom_ids, geom_surface_offsets


def infer_terminal_effectors(model, joint_mapping, overrides=None):
    """No anatomical names are inspected; explicit robot facts override inference.

    A mapped body is terminal when it has no distinct mapped descendant. A
    unique unmapped distal chain is followed to its end (e.g. a compound wrist).
    Only the final body's welded collision geometry defines its contact shell.
    """
    mapping = {
        str(source): int(model.body(body).id) for source, body in joint_mapping.items()
    }
    mapped = set(mapping.values())

    def ancestor(parent, child):
        while child:
            child = int(model.body_parentid[child])
            if child == parent:
                return True
        return False

    terminal = [
        (source, body)
        for source, body in mapping.items()
        if not any(ancestor(body, other) for other in mapped if other != body)
    ]
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    collision = collision_geom_ids(model)
    configured = deepcopy(overrides or {})
    result = {}
    for source, mapped_body in terminal:
        body = mapped_body
        while True:
            weld = int(model.body_weldid[body])
            children = [
                b
                for b in range(1, model.nbody)
                if model.body_weldid[b] != weld
                and model.body_weldid[model.body_parentid[b]] == weld
            ]
            if len(children) != 1:
                break
            body = children[0]
        weld = model.body_weldid[body]
        matching = [
            name
            for name, spec in configured.items()
            if spec.get("source_joint") == source
            or (
                spec.get("robot_body")
                and (
                    model.body_weldid[model.body(spec["robot_body"]).id] == weld
                    or model.body(spec["robot_body"]).id == mapped_body
                )
            )
        ]
        if len(matching) > 1:
            raise ValueError(f"Multiple overrides address mapped terminal {source}")
        name = matching[0] if matching else source
        spec = configured.pop(name, {})
        spec.setdefault("robot_body", model.body(body).name)
        spec.setdefault("source_joint", source)
        chosen_body = int(model.body(spec["robot_body"]).id)
        ids = collision[
            model.body_weldid[model.geom_bodyid[collision]]
            == model.body_weldid[chosen_body]
        ]
        shell = []
        for gid in ids:
            world = (
                geom_surface_offsets(model, int(gid))
                @ data.geom_xmat[gid].reshape(3, 3).T
                + data.geom_xpos[gid]
            )
            shell.extend(
                (world - data.xpos[chosen_body]) @ data.xmat[chosen_body].reshape(3, 3)
            )
        shell = (
            np.unique(np.asarray(shell).reshape(-1, 3), axis=0)
            if shell
            else np.zeros((1, 3))
        )
        center = (shell.min(0) + shell.max(0)) * 0.5
        spec.setdefault("tracking_offset", center.tolist())
        spec.setdefault("contact_offsets", shell.tolist())
        spec.setdefault("track_swing", True)
        # Clearance evidence, not limb identity, will decide the target frame.
        result[name] = spec
    result.update(configured)  # explicitly declared nonterminal tools remain valid
    return result
