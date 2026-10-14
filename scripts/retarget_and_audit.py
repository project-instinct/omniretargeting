"""Retarget one source with the production fast solver, then audit it once.

One command for any adapter config (SMPL-X/HER video, OMOMO, TRUMANS, ...):

    .venv/bin/python scripts/retarget_and_audit.py --source source.yaml \
        --out /path/on/data/disk/run --source-denoise auto --record

``--source`` is either an adapter config (YAML/JSON, as for ``omniretargeting.main
--source-config``) or a motion file whose adapter is inferred (.pkl=parc_ms,
.npz=smplx; pass ``--source-type`` otherwise, and ``--smplx-model-dir`` for raw
SMPL-X). The run directory keeps every file the standard audits and
``scripts/review_webviser.py`` read (prepared.npy, fixed_reference.npz,
fast_report.json, timing.json, ...). Audits only read the saved motion; they are
never used to re-solve. Write outputs to the data disk, not the system disk.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_fast import THREADS, capture_run, run_audits, save  # noqa: E402

DEFAULT_PROFILE = ROOT / "robot_models/unitree_g1/unitree_g1_mesh.json"


def solver_argv(args):
    """Extra CLI options forwarded to omniretargeting.main."""
    extra = []
    if args.source_denoise is not None:
        extra += ["--source-denoise", args.source_denoise]
    return extra


def child_env():
    env = dict(os.environ, PYTHONPATH=str(ROOT), **THREADS)
    return env


CONFIG_SUFFIXES = {".yaml", ".yml", ".json"}


def source_config(args):
    """Adapter config for ``--source``: a YAML/JSON config or a motion file."""
    from omniretargeting.source_config import direct_source_config, load_source_config

    if args.source.suffix.casefold() in CONFIG_SUFFIXES:
        if args.source_type is not None:
            raise SystemExit("--source-type is only valid when --source is a motion file")
        config = dict(load_source_config(args.source))
    else:
        config = direct_source_config(args.source, args.source_type)
    if args.smplx_model_dir is not None:
        config["model_directory"] = str(Path(args.smplx_model_dir).resolve())
    return config


def solve(args):
    config = source_config(args)
    capture_run(config, args.out, args.robot_config, source_root=ROOT, extra_argv=solver_argv(args))


def record(out, python, env):
    """Web replay next to the run (port 8183 server serves it if placed under its root)."""
    cmd = [str(python), str(ROOT / "scripts/review_webviser.py"), "--prepared", str(out / "prepared.npy"),
           "--motion", str(out / "final_retargeted.npz"), "--objects", str(out / "objects.npy"),
           "--output", str(out / "web"), "--record-only"]
    with (out / "record.log").open("w") as handle:
        return subprocess.run(cmd, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT).returncode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True,
                        help="adapter config (YAML/JSON) or motion file (.pkl, .npz, ...)")
    parser.add_argument("--source-type", default=None, help="adapter for a motion file whose format is ambiguous")
    parser.add_argument("--smplx-model-dir", default=None, help="SMPL-X model directory for raw SMPL-X sources")
    parser.add_argument("--out", type=Path, required=True, help="new run directory (must not exist)")
    parser.add_argument("--robot-config", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--source-denoise", choices=["off", "auto"], default=None,
                        help="forwarded to the CLI; use auto for video estimates such as HER")
    parser.add_argument("--no-audits", action="store_true", help="solve and capture only")
    parser.add_argument("--serial-audits", action="store_true")
    parser.add_argument("--audit-workers", type=int, default=None)
    parser.add_argument("--record", action="store_true", help="also record a web replay into <out>/web")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    args.out = args.out.resolve()

    if args.child:
        solve(args)
        return 0
    # The solve runs in a child so its threads/patching never leak into audits.
    python = Path(sys.executable)
    env = child_env()
    forwarded = ["--source", str(Path(args.source).resolve()), "--out", str(args.out),
                 "--robot-config", str(Path(args.robot_config).resolve())]
    if args.source_denoise is not None:
        forwarded += ["--source-denoise", args.source_denoise]
    if args.source_type is not None:
        forwarded += ["--source-type", args.source_type]
    if args.smplx_model_dir is not None:
        forwarded += ["--smplx-model-dir", str(Path(args.smplx_model_dir).resolve())]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    log = args.out.with_name(args.out.name + ".solve.log")
    with log.open("w") as handle:
        code = subprocess.run([str(python), str(Path(__file__).resolve()), "--child", *forwarded],
                              cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT).returncode
    if code != 0:
        print(f"solve failed ({code}); see {log}", file=sys.stderr)
        return code
    summary = dict(solve_log=str(log))
    if not args.no_audits:
        summary["audits"] = run_audits(args.out, ROOT, env, python, scene_workers=args.audit_workers,
                                       concurrent=not args.serial_audits)
    if args.record:
        summary["record_returncode"] = record(args.out, python, env)
    save(args.out / "retarget_and_audit.json", summary)
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
