"""Explicit near-pass experiment on already reconstructed full recordings.

No source/frame-specific selection is used. The required uncertainty budget is
an experimental assumption, not a measured TRUMANS reconstruction error. This
script does not enable the policy in the production retargeting pipeline.
"""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import trimesh

from omniretargeting.contact_pass import review_contact_passes, without_pass_attraction
from omniretargeting.scene_geometry import SceneQuery


def scene_query(mesh, **kwargs):
    try:
        return SceneQuery(mesh, closed_sign_method="pseudonormal", **kwargs)
    except ValueError as error:
        if str(error) != "Pseudonormals require nondegenerate triangles":
            raise
        return SceneQuery(mesh, **kwargs)


def observe(inputs, source, root):
    """Use all original skin samples, with known signed geometry only."""
    model = mujoco.MjModel.from_xml_path(inputs["model_path"])
    static = scene_query(
        trimesh.load_mesh(root / "static_scene.obj", process=False),
        enclosed_free_space=inputs["scene_enclosed_free_space"],
    )
    objects = np.load(root / "objects.npy", allow_pickle=True).tolist()
    queries = {}
    result = {}
    for name, task in inputs["tasks"].items():
        spec = inputs["effectors"]["effectors"][name]
        samples = source["markers"][name]["contact_samples"]
        frames = len(samples)
        points = source["markers"][name]["tracking_positions"].copy()
        env = (
            np.zeros(frames, int)
            if task.environment_ids is None
            else task.environment_ids
        )
        gap = np.full(frames, np.nan)
        for environment in np.unique(env):
            indices = np.flatnonzero(env == environment)
            local = samples[indices].copy()
            scale = np.ones(len(indices))
            query = static
            if environment:
                obj = objects[environment - 1]
                if environment not in queries:
                    queries[environment] = scene_query(obj["mesh"])
                query = queries[environment]
                rotation = np.asarray(
                    obj.get("rotations", np.tile(np.eye(3), (frames, 1, 1)))
                )[indices]
                translation = np.asarray(
                    obj.get("translations", np.zeros((frames, 3)))
                )[indices]
                scale = np.asarray(obj.get("scales", np.ones(frames)))[indices]
                local = (
                    np.einsum("tni,tij->tnj", local - translation[:, None], rotation)
                    / scale[:, None, None]
                )
                points[indices] = (
                    np.einsum("ti,tij->tj", points[indices] - translation, rotation)
                    / scale[:, None]
                )
                # A varying object scale has no single metric material frame.
                # Keep it unknown instead of silently changing the error budget.
                if not np.allclose(scale, 1.0):
                    continue
            witness = query.query(local.reshape(-1, 3))
            clear = witness["clearance"].copy()
            clear[witness["uncertain"]] = np.nan
            gap[indices] = clear.reshape(len(indices), -1).min(axis=1) * scale
        protected = np.asarray(
            source["markers"][name].get("source_contact_mask", np.zeros(frames)), bool
        ).copy()
        # Existing world-tracking contracts remain protected, independent of
        # anatomy. This is not a claim that a named limb mechanically supports.
        if spec.get("preserve_world_target", False):
            protected |= task.contact_mask
        body = model.body(spec["robot_body"]).id
        weld = model.body_weldid[body]
        surface = inputs["surface"]
        on_part = (
            model.body_weldid[model.geom_bodyid[surface["surface_contact_robot_geoms"]]]
            == weld
        )
        result[name] = {
            "positions": points,
            "clearance": gap,
            "protected": protected,
            "on_part": on_part,
        }
    return result


def prepare(args):
    inputs = np.load(args.prepared, allow_pickle=True).item()
    source = np.load(args.prepared.parent / "source.npy", allow_pickle=True).item()
    args.output.mkdir(parents=True, exist_ok=True)
    cache = args.output / "observations.npy"
    identity = {
        filename: hashlib.sha256(
            (args.prepared.parent / filename).read_bytes()
        ).hexdigest()
        for filename in (
            args.prepared.name,
            "source.npy",
            "objects.npy",
            "static_scene.obj",
        )
    }
    manifest = args.output / "input_hashes.json"
    if cache.exists():
        if not manifest.exists() or json.loads(manifest.read_text()) != identity:
            raise ValueError("Observation cache belongs to different source inputs")
        observations = np.load(cache, allow_pickle=True).item()
    else:
        observations = observe(inputs, source, args.prepared.parent)
        np.save(cache, observations, allow_pickle=True)
        manifest.write_text(json.dumps(identity, indent=2))
    fps = float(np.load(args.prepared.parent / "final.npz")["framerate"])
    budgets = sorted({args.uncertainty, *args.sensitivity})
    decisions, report = (
        {},
        {
            "assumed_uncertainties_m": budgets,
            "uncertainty_calibrated": False,
            "mechanics_available": False,
            "by_budget": {},
        },
    )
    for budget in budgets:
        rows = {}
        for name, task in inputs["tasks"].items():
            obs = observations[name]
            protected = obs["protected"].copy()
            # Dense source surface observations on the same welded region can
            # contradict sparse marker clearance. Keep such contacts.
            surface = inputs["surface"]
            active = surface["surface_contact_mask"][:, obs["on_part"]]
            vectors = surface["surface_contact_source_vectors_w"][:, obs["on_part"]]
            protected |= (active & (np.linalg.norm(vectors, axis=-1) <= budget)).any(
                axis=1
            )
            decision = review_contact_passes(
                obs["positions"],
                obs["clearance"],
                task.contact_mask,
                times=np.arange(len(task.contact_mask)) / fps,
                position_uncertainty_m=budget,
                environment_ids=task.environment_ids,
                protected_mask=protected,
            )
            rows[name] = {
                "pass_frames": int(decision.pass_mask.sum()),
                "events": decision.events,
            }
            if budget == args.uncertainty:
                decisions[name] = decision
        report["by_budget"][str(budget)] = rows
    (args.output / "review.json").write_text(json.dumps(report, indent=2))
    np.save(args.output / "decisions.npy", decisions, allow_pickle=True)
    trial = dict(inputs)
    trial["tasks"] = {
        n: without_pass_attraction(t, decisions[n]) for n, t in inputs["tasks"].items()
    }
    # Secondary surface goals must not restore attraction on rejected patches.
    # Actual scene collision queries remain enabled and original plans are kept
    # separately for audits. Other regions and all retained contacts are intact.
    trial["surface"] = {k: np.array(v, copy=True) for k, v in inputs["surface"].items()}
    trial["proximity"] = (
        None
        if inputs["proximity"] is None
        else {
            k: np.array(v, copy=True) if isinstance(v, np.ndarray) else v
            for k, v in inputs["proximity"].items()
        }
    )
    model = mujoco.MjModel.from_xml_path(inputs["model_path"])
    for name, decision in decisions.items():
        frames = np.flatnonzero(decision.pass_mask)
        if not len(frames):
            continue
        columns = np.flatnonzero(observations[name]["on_part"])
        trial["surface"]["surface_contact_mask"][np.ix_(frames, columns)] = False
        if trial["proximity"] is not None:
            spec = inputs["effectors"]["effectors"][name]
            weld = model.body_weldid[model.body(spec["robot_body"]).id]
            proximity_columns = np.flatnonzero(
                model.body_weldid[model.geom_bodyid[trial["proximity"]["geoms"]]]
                == weld
            )
            trial["proximity"]["valid"][np.ix_(frames, proximity_columns)] = False
    np.save(args.output / "prepared.npy", trial, allow_pickle=True)
    for filename in ("source.npy", "objects.npy", "static_scene.obj"):
        target = args.output / filename
        if not target.exists():
            target.symlink_to((args.prepared.parent / filename).resolve())
    print(
        json.dumps(
            {
                "clip": args.prepared.parent.name,
                "results": {
                    b: {n: v["pass_frames"] for n, v in row.items()}
                    for b, row in report["by_budget"].items()
                },
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--uncertainty", type=float, required=True)
    parser.add_argument("--sensitivity", type=float, nargs="*", default=[])
    prepare(parser.parse_args())
