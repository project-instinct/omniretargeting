"""Reproducible whole-clip A/B/C run; preserves every previous output.

Prepare once, then run e.g. .venv/bin/python scripts/body_relation_experiment.py
--output out/body_relation_round --clip 010 --variant A. A imports a frozen copy
of the pre-change package. B/C import the working tree. No source arrays change.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def hashes(paths):
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def prepare(base):
    base.mkdir(parents=True, exist_ok=False)
    shutil.copytree(ROOT / "omniretargeting", base / "baseline_code/omniretargeting",
                    ignore=shutil.ignore_patterns("__pycache__"))
    profile = ROOT / "out/her_c167_010_diagnosis/core_bind_balanced/robot_profile.json"
    shutil.copyfile(profile, base / "robot_profile.json")
    cases = {}
    for clip, old in [("010", "her_c167_010_diagnosis/core_bind_balanced"),
                      ("1530", "her_c167_1530_release_body")]:
        old = ROOT / "out" / old
        item = json.loads((old / "input_manifest.json").read_text())["clips"][0]
        cases[clip] = dict(item, source=json.loads((old / "source.json").read_text()))
    (base / "cases.json").write_text(json.dumps(cases, indent=2))
    (base / "baseline_manifest.json").write_text(json.dumps({
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
        "files": hashes((base / "baseline_code/omniretargeting").rglob("*.py")),
        "profile": hashes([profile]),
    }, indent=2))


def prepare_synthetic(base):
    """A whole 48-frame SMPL-X close-arm gesture, generated before retargeting."""
    import numpy as np
    import trimesh
    from scipy.spatial.transform import Rotation

    folder = base / "synthetic_input"
    folder.mkdir(exist_ok=False)
    frames, fps = 48, 30.
    phase = np.linspace(0, 2 * np.pi, frames)
    body = np.zeros((frames, 21, 3), dtype=np.float32)
    # Source anatomical joints only; this experiment is deliberately SMPL-X.
    # Fold hands toward the chest while twisting and bending the torso.
    body[:, 2, 1] = .25 * np.sin(phase)
    body[:, 5, 0] = .12 * np.sin(phase + .5)
    body[:, 8, 1] = .18 * np.sin(phase)
    for side, shoulder, elbow in [(1, 15, 17), (-1, 16, 18)]:
        body[:, shoulder, 2] = -side * (.8 + .22 * np.cos(phase))
        body[:, shoulder, 1] = -side * (.35 + .25 * np.sin(phase))
        body[:, elbow, 1] = -side * (2.05 + .35 * np.sin(phase + side * .35))
    rotation = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
    root = np.tile(Rotation.from_matrix(rotation).as_rotvec(), (frames, 1))
    translation = np.tile([0., 0., 1.0], (frames, 1))
    np.savez(folder / "motion.npz", pose_body=body.reshape(frames, 63), root_orient=root,
             trans=translation, betas=np.zeros(10), gender="neutral", mocap_frame_rate=fps,
             frame_ids=np.arange(frames))
    scene = trimesh.creation.box(extents=[6, 6, .1])
    scene.apply_translation([0, 0, -.05])
    scene.export(folder / "scene.obj")
    cases = json.loads((base / "cases.json").read_text())
    source = dict(type="smplx", motion=str(folder / "motion.npz"), adapter_options=dict(
        model_directory=str(ROOT / "assets/body_models"), use_smplx_base_pose=True,
        frame_id_semantics="provenance"))
    cases["synthetic"] = dict(clip="synthetic_close_arm", motion=str(folder / "motion.npz"),
        scene=str(folder / "scene.obj"), source=source,
        timeline_evidence=dict(published_rows=frames, published_fps=fps))
    (base / "cases.json").write_text(json.dumps(cases, indent=2))


def run(base, clip, variant):
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "2"
    code = base / "baseline_code" if variant == "A" else ROOT
    sys.path.insert(0, str(code))
    import numpy as np
    import pickle
    import omniretargeting.core as core
    from omniretargeting.main import main
    from omniretargeting.retargeting import GenericInteractionRetargeter

    case = json.loads((base / "cases.json").read_text())[clip]
    out = base / clip / variant
    out.mkdir(parents=True, exist_ok=False)
    profile = json.loads((base / "robot_profile.json").read_text())
    if variant != "A":
        profile["retargeting"]["body_relations"] = dict(enabled=True, weight=0.0 if variant == "B" else 1.0,
                                                       sigma=0.12, reference_source=profile["base_orientation"]["pelvis"])
    (out / "robot_profile.json").write_text(json.dumps(profile, indent=2))
    (out / "source.json").write_text(json.dumps(case["source"], indent=2))
    inputs = [Path(case[k]) for k in ("motion", "scene")]
    before = hashes(inputs)
    (out / "manifest.json").write_text(json.dumps(dict(
        variant=variant, clip=clip, command=sys.argv, input_hashes=before,
        robot_hashes=hashes([Path(profile["urdf_path"])]),
        code_hashes=hashes((code / "omniretargeting").rglob("*.py")),
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    ), indent=2))
    started = time.monotonic()
    real_motion = core.OmniRetargeter.retarget_motion
    real_reference = GenericInteractionRetargeter.configure_source_orientation_reference
    real_refine = core.refine_joint_trajectory
    real_select = core.select_safe_postprocessing
    real_repair = core.repair_local_collisions
    holder = {}

    def capture_motion(self, motion, *args, **kwargs):
        arrays = {k: v for k, v in kwargs.items() if isinstance(v, np.ndarray)}
        np.savez(out / "source_arrays.npz", positions=motion, names=self.smplx_joint_names, **arrays)
        holder["retargeter"] = self
        return real_motion(self, motion, *args, **kwargs)

    def capture_reference(self, *args, **kwargs):
        result = real_reference(self, *args, **kwargs)
        with (out / "calibration.pkl").open("wb") as f:
            pickle.dump(dict(tasks=self.body_orientation_tasks,
                             source=self._source_orientation_references,
                             robot=self._robot_orientation_references,
                             mapping=self.joint_mapping), f)
        return result

    def capture_refine(model, q, *args, **kwargs):
        np.save(out / "framewise_qpos.npy", q)
        result = real_refine(model, q, *args, **kwargs)
        np.save(out / "refinement_qpos.npy", result)
        return result

    def capture_select(model, baseline, candidate, scene, tasks, priority, config, **kwargs):
        np.save(out / "framewise_qpos.npy", baseline)
        np.save(out / "postprocessing_candidate_qpos.npy", candidate)
        with (out / "validation_context.pkl").open("wb") as f:
            pickle.dump(dict(tasks=tasks, priority=priority, config=config, framerate=kwargs["framerate"]), f)
        result = real_select(model, baseline, candidate, scene, tasks, priority, config, **kwargs)
        (out / "postprocessing_report.json").write_text(core.diagnostics_json(result[2]))
        return result

    def capture_repair(model, q, *args, **kwargs):
        np.save(out / "before_local_repair_qpos.npy", q)
        result = real_repair(model, q, *args, **kwargs)
        np.save(out / "after_local_repair_qpos.npy", result[0])
        return result

    core.OmniRetargeter.retarget_motion = capture_motion
    GenericInteractionRetargeter.configure_source_orientation_reference = capture_reference
    core.refine_joint_trajectory = capture_refine
    core.select_safe_postprocessing = capture_select
    core.repair_local_collisions = capture_repair
    sys.argv = ["omniretargeting.main", "--source-config", str(out / "source.json"),
                "--robot-config", str(out / "robot_profile.json"), "--terrain", case["scene"],
                "--output", str(out / "final_retargeted.npz")]
    main()
    seconds = time.monotonic() - started
    retargeter = holder["retargeter"]
    (out / "effective_config.json").write_text(core.diagnostics_json(retargeter.last_effective_config))
    relation = getattr(retargeter, "last_body_relation_tasks", None)
    if relation is not None:
        with (out / "body_relations.pkl").open("wb") as f:
            pickle.dump(relation, f)
    with np.load(out / "final_retargeted.npz", allow_pickle=True) as saved:
        q = saved["qpos"]
        assert len(q) == case["timeline_evidence"]["published_rows"]
        assert float(saved["framerate"]) == case["timeline_evidence"]["published_fps"]
        np.testing.assert_array_equal(q, np.load(out / "after_local_repair_qpos.npy"))
        assert np.isfinite(q).all()
    assert before == hashes(inputs)
    (out / "run_summary.json").write_text(json.dumps(dict(
        seconds=seconds, frames=len(q), input_hashes_unchanged=True,
        saved_equals_accepted=True,
    ), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--add-synthetic", action="store_true")
    parser.add_argument("--clip", choices=["010", "1530", "synthetic"])
    parser.add_argument("--variant", choices=["A", "B", "C"])
    args = parser.parse_args()
    if args.prepare:
        prepare(args.output.resolve())
    elif args.add_synthetic:
        prepare_synthetic(args.output.resolve())
    else:
        run(args.output.resolve(), args.clip, args.variant)
