#!/usr/bin/env python3
"""Measure retargeted foot orientation against the calibrated source foot frame.

Position-based checks (clearance, penetration) can pass while the sole is
grossly misoriented, so this reports orientation directly: the geodesic
distance on SO(3) between the robot sole frame and the source sole frame,
stratified into contact and swing frames, plus the component errors and the
joint saturation that usually accompanies a bad result.

Example:

    python scripts/diagnose_foot_pose.py \
        --retargeted out/run_retargeted.npz \
        --urdf robot_models/unitree_g1/g1_29dof_popsicle.urdf \
        --source ../omniretargeting/test_data/c167_1530_final_v3/smplx_amass.npz \
        --smplx-model-dir /path/to/body_models \
        --terrain out/scene_scaled.obj
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from omniretargeting.foot_frames import calibrate_sole_rotation, world_sole_rotations

# SMPL-X body joint indices.
ANKLE_IDX = {"left": 7, "right": 8}
TOE_IDX = {"left": 10, "right": 11}

DEFAULT_FOOT_BODIES = {"left": "left_ankle_roll_link", "right": "right_ankle_roll_link"}
SATURATION_TOL = 1e-4


# --------------------------------------------------------------------------
# robot side
# --------------------------------------------------------------------------

def load_robot_frames(
    retargeted: Path, urdf: Path, foot_bodies: Dict[str, str]
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], dict]:
    """Forward-kinematics the retargeted trajectory into per-foot world frames."""
    import mujoco

    from omniretargeting.utils import load_robot_urdf_with_floating_base

    data = np.load(retargeted, allow_pickle=True)
    base_pos = np.asarray(data["base_pos_w"], dtype=float)
    base_quat = np.asarray(data["base_quat_w"], dtype=float)  # wxyz
    joint_pos = np.asarray(data["joint_pos"], dtype=float)
    joint_names = [str(n) for n in data["joint_names"]]

    cwd = os.getcwd()
    try:
        os.chdir(urdf.parent)
        model = load_robot_urdf_with_floating_base(Path(urdf.name))
    finally:
        os.chdir(cwd)
    mj_data = mujoco.MjData(model)

    qpos_adr = {}
    jnt_range = {}
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name is None or model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        qpos_adr[name] = int(model.jnt_qposadr[i])
        jnt_range[name] = model.jnt_range[i].copy()

    body_ids = {
        side: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
        for side, body in foot_bodies.items()
    }
    for side, bid in body_ids.items():
        if bid < 0:
            raise SystemExit(f"body {foot_bodies[side]!r} not present in the compiled model")

    n = len(joint_pos)
    rot = {side: np.zeros((n, 3, 3)) for side in body_ids}
    pos = {side: np.zeros((n, 3)) for side in body_ids}
    root_rot = np.zeros((n, 3, 3))

    for t in range(n):
        mj_data.qpos[:3] = base_pos[t]
        mj_data.qpos[3:7] = base_quat[t]
        for name, value in zip(joint_names, joint_pos[t]):
            if name in qpos_adr:
                mj_data.qpos[qpos_adr[name]] = value
        mujoco.mj_kinematics(model, mj_data)
        root_rot[t] = mj_data.xmat[1].reshape(3, 3)
        for side, bid in body_ids.items():
            rot[side][t] = mj_data.xmat[bid].reshape(3, 3)
            pos[side][t] = mj_data.xpos[bid]

    saturation = {}
    for name, adr in qpos_adr.items():
        if name not in joint_names:
            continue
        col = joint_pos[:, joint_names.index(name)]
        lo, hi = jnt_range[name]
        if hi <= lo:
            continue
        at_lo = np.abs(col - lo) <= SATURATION_TOL
        at_hi = np.abs(col - hi) <= SATURATION_TOL
        saturation[name] = {
            "at_lower": float(at_lo.mean()),
            "at_upper": float(at_hi.mean()),
            "range": [float(lo), float(hi)],
        }

    meta = {"frames": n, "saturation": saturation, "root_rot": root_rot, "model": model}
    return rot, pos, root_rot, meta


# --------------------------------------------------------------------------
# source side
# --------------------------------------------------------------------------

def calibrate_source_sole_frames(
    source: Path, model_dir: Path, gender: str, use_clip_betas: bool
) -> Optional[Dict[str, np.ndarray]]:
    """Per-frame world sole frames for the SMPL-X source.

    The raw SMPL-X foot joint frame is not the anatomical sole frame; they
    differ by roughly twenty degrees, mostly toe-out yaw. The sole frame is
    therefore calibrated once on the shaped template (thinnest principal axis
    as sole normal, ankle-to-toe projected into the sole plane as forward) and
    then transported per frame by the joint's world rotation.
    """
    try:
        import torch
        import smplx as smplx_lib
    except ImportError:
        return None

    from scipy.spatial.transform import Rotation as R

    raw = np.load(source, allow_pickle=True)
    if "poses" not in raw.files:
        return None

    poses = np.asarray(raw["poses"], dtype=np.float32)
    trans = np.asarray(raw["trans"], dtype=np.float32)
    n = len(poses)
    betas_full = np.asarray(raw["betas"], dtype=np.float32) if "betas" in raw.files else np.zeros(16, np.float32)
    num_betas = int(len(betas_full))
    betas = betas_full if use_clip_betas else np.zeros_like(betas_full)

    body = smplx_lib.create(
        str(model_dir), "smplx", gender=gender, use_pca=False,
        flat_hand_mean=True, num_betas=num_betas, batch_size=n,
    )
    beta_t = torch.tensor(betas)[None].repeat(n, 1)
    out = body(
        betas=beta_t,
        global_orient=torch.tensor(poses[:, :3]),
        body_pose=torch.tensor(poses[:, 3:66]),
        transl=torch.tensor(trans),
        return_full_pose=True,
    )
    verts = out.vertices.detach().numpy()
    joints = out.joints.detach().numpy()[:, :22]

    # Shaped template in the rest pose, used for the one-time calibration.
    tmpl = body(betas=beta_t[:1])
    tmpl_v = tmpl.vertices.detach().numpy()[0]
    tmpl_j = tmpl.joints.detach().numpy()[0, :22]

    weights = body.lbs_weights.detach().numpy()

    # World joint rotations from the full pose.
    full_pose = out.full_pose.detach().numpy().reshape(n, -1, 3)
    parents = body.parents.detach().numpy()
    local = R.from_rotvec(full_pose[:, :22].reshape(-1, 3)).as_matrix().reshape(n, 22, 3, 3)
    world = np.zeros_like(local)
    world[:, 0] = local[:, 0]
    for j in range(1, 22):
        world[:, j] = world[:, parents[j]] @ local[:, j]
    world_quats = R.from_matrix(world.reshape(-1, 3, 3)).as_quat(scalar_first=True).reshape(n, 22, 4)

    frames = {}
    for side in ("left", "right"):
        a_idx, t_idx = ANKLE_IDX[side], TOE_IDX[side]
        mask = (weights[:, a_idx] + weights[:, t_idx]) > 0.5
        cal = calibrate_sole_rotation(side, tmpl_v, tmpl_j, mask, a_idx, t_idx)
        frames[side] = world_sole_rotations(world_quats, cal)
        frames[f"{side}_calibration"] = cal
        frames[f"{side}_ankle_pos"] = joints[:, a_idx]
        frames[f"{side}_toe_pos"] = joints[:, t_idx]

    frames["_verts"] = verts
    return frames


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def geodesic_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Angle of the relative rotation, in degrees, per frame."""
    rel = np.einsum("tij,tkj->tik", a, b)
    trace = np.trace(rel, axis1=1, axis2=2)
    return np.degrees(np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0)))


def angle_between_deg(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    un = u / np.linalg.norm(u, axis=-1, keepdims=True)
    vn = v / np.linalg.norm(v, axis=-1, keepdims=True)
    return np.degrees(np.arccos(np.clip((un * vn).sum(axis=-1), -1.0, 1.0)))


def summarize(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"n": 0}
    return {
        "n": int(values.size),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "max": float(values.max()),
        "mean": float(values.mean()),
    }


def detect_contact(sole_pos: np.ndarray, rot: np.ndarray, terrain, sole_points: np.ndarray,
                   threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    """Contact when any sole point is within `threshold` of the mesh.

    A single sole-centre probe is not enough: a foot standing on its toe holds
    the sole centre well clear of the ground, so a centre-only test reports no
    contact exactly on the frames that matter most.  Taking the minimum over
    heel, centre and toe keeps the phase label stable as the foot orientation
    changes, which is what makes before/after stratification comparable.
    """
    n = len(sole_pos)
    if terrain is None:
        return np.zeros(n, dtype=bool), np.full(n, np.nan)
    world = sole_pos[:, None, :] + np.einsum("tij,kj->tki", rot, sole_points)
    closest, _, _ = terrain.nearest.on_surface(world.reshape(-1, 3))
    dist = np.linalg.norm(world.reshape(-1, 3) - closest, axis=1).reshape(n, -1)
    nearest = dist.min(axis=1)
    return nearest <= threshold, nearest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--retargeted", type=Path, required=True)
    ap.add_argument("--urdf", type=Path, required=True)
    ap.add_argument("--source", type=Path, default=None, help="raw SMPL-X npz with poses/betas")
    ap.add_argument("--smplx-model-dir", type=Path, default=None)
    ap.add_argument("--gender", default="neutral")
    ap.add_argument("--terrain", type=Path, default=None)
    ap.add_argument("--sole-points", type=float, nargs="+",
                    default=[-0.054, 0.0, -0.035, 0.039, 0.0, -0.035, 0.132, 0.0, -0.035],
                    help="flattened heel/centre/toe sole probes in the foot body frame")
    ap.add_argument("--contact-threshold", type=float, default=0.03)
    ap.add_argument("--zero-betas", action="store_true",
                    help="calibrate against the neutral template instead of the clip's own shape")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    rot, pos, root_rot, meta = load_robot_frames(args.retargeted, args.urdf, DEFAULT_FOOT_BODIES)
    n = meta["frames"]

    terrain = None
    if args.terrain is not None:
        import trimesh
        terrain = trimesh.load(args.terrain, force="mesh")

    src = None
    if args.source is not None and args.smplx_model_dir is not None:
        src = calibrate_source_sole_frames(
            args.source, args.smplx_model_dir, args.gender, use_clip_betas=not args.zero_betas
        )
        if src is None:
            print("[warn] source orientations unavailable; reporting robot-side metrics only")

    report: dict = {"frames": n, "retargeted": str(args.retargeted), "feet": {}}
    sole_points = np.asarray(args.sole_points, dtype=float).reshape(-1, 3)

    print(f"frames: {n}")
    for side in ("left", "right"):
        entry: dict = {}
        contact, nearest = detect_contact(
            pos[side], rot[side], terrain, sole_points, args.contact_threshold
        )
        entry["contact_fraction"] = float(contact.mean()) if n else 0.0
        entry["_contact_flags"] = contact
        if np.isfinite(nearest).any():
            entry["sole_clearance_m"] = summarize(nearest[np.isfinite(nearest)])

        toe_axis = rot[side][:, :, 0]
        sole_normal = rot[side][:, :, 2]
        entry["toe_elevation_deg"] = summarize(90.0 - angle_between_deg(toe_axis, np.array([0.0, 0.0, 1.0])))
        entry["sole_tilt_from_vertical_deg"] = summarize(angle_between_deg(sole_normal, np.array([0.0, 0.0, 1.0])))

        if src is not None:
            target = src[side]
            if len(target) != n:
                m = min(len(target), n)
                target, cur = target[:m], rot[side][:m]
                contact_m = contact[:m]
            else:
                cur = rot[side]
                contact_m = contact
            geo = geodesic_deg(cur, target)
            normal_err = angle_between_deg(cur[:, :, 2], target[:, :, 2])
            fwd_err = angle_between_deg(cur[:, :, 0], target[:, :, 0])
            entry["geodesic_deg"] = {
                "all": summarize(geo),
                "contact": summarize(geo[contact_m]),
                "swing": summarize(geo[~contact_m]),
            }
            entry["sole_normal_err_deg"] = summarize(normal_err)
            entry["forward_axis_err_deg"] = summarize(fwd_err)

        report["feet"][side] = entry

        print(f"\n[{side}]  contact frames: {entry['contact_fraction']*100:.1f}%")
        print(f"  toe elevation      median {entry['toe_elevation_deg']['median']:+7.2f} deg")
        print(f"  sole tilt vs up    median {entry['sole_tilt_from_vertical_deg']['median']:7.2f} deg")
        if "geodesic_deg" in entry:
            g = entry["geodesic_deg"]
            print(f"  SO(3) error   all     median {g['all']['median']:6.2f}  p90 {g['all']['p90']:6.2f}  max {g['all']['max']:6.2f}")
            if g["contact"]["n"]:
                print(f"                contact median {g['contact']['median']:6.2f}  p90 {g['contact']['p90']:6.2f}")
            if g["swing"]["n"]:
                print(f"                swing   median {g['swing']['median']:6.2f}  p90 {g['swing']['p90']:6.2f}")
            print(f"  sole-normal err       median {entry['sole_normal_err_deg']['median']:6.2f}")
            print(f"  forward-axis err      median {entry['forward_axis_err_deg']['median']:6.2f}")

    # Base height continuity.  A base that steps between two levels by contact
    # state reads as convulsing even when every joint-angle statistic looks
    # healthy, which is exactly how one such defect escaped notice: the
    # smoothness numbers covered joint angles and not the floating base.
    base_z = np.asarray(np.load(args.retargeted)["base_pos_w"], dtype=float)[:, 2]
    contact_any = np.zeros(len(base_z), dtype=bool)
    for side in ("left", "right"):
        entry = report["feet"].get(side, {})
        flags = entry.pop("_contact_flags", None)
        if flags is not None and len(flags) == len(contact_any):
            contact_any |= np.asarray(flags, dtype=bool)
    base = {
        "second_difference_rms_mm": float(np.sqrt(np.mean(np.diff(base_z, 2) ** 2)) * 1000.0)
        if len(base_z) > 2 else 0.0,
        "max_step_mm": float(np.abs(np.diff(base_z)).max() * 1000.0) if len(base_z) > 1 else 0.0,
    }
    if contact_any.any() and not contact_any.all():
        base["median_with_contact_m"] = float(np.median(base_z[contact_any]))
        base["median_without_contact_m"] = float(np.median(base_z[~contact_any]))
        base["contact_state_step_mm"] = float(
            (base["median_without_contact_m"] - base["median_with_contact_m"]) * 1000.0
        )
    report["base_height"] = base
    print("\nbase height:")
    print(f"  second-difference RMS  {base['second_difference_rms_mm']:8.2f} mm")
    print(f"  largest single step    {base['max_step_mm']:8.2f} mm")
    if "contact_state_step_mm" in base:
        print(f"  step between contact states {base['contact_state_step_mm']:8.2f} mm"
              f"   ({base['median_with_contact_m']*100:.2f} cm with contact,"
              f" {base['median_without_contact_m']*100:.2f} cm without)")

    sat = {k: v for k, v in meta["saturation"].items() if v["at_lower"] + v["at_upper"] > 0.05}
    report["saturation"] = sat
    if sat:
        print("\nsaturated joints (>5% of frames within 1e-4 rad of a bound):")
        for name, v in sorted(sat.items(), key=lambda kv: -(kv[1]["at_lower"] + kv[1]["at_upper"])):
            print(f"  {name:28s} lower {v['at_lower']*100:5.1f}%  upper {v['at_upper']*100:5.1f}%"
                  f"   range [{v['range'][0]:+.4f}, {v['range'][1]:+.4f}]")

    if args.json:
        args.json.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
