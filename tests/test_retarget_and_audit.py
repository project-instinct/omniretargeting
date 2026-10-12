"""The one-command retarget + audit entry point and its shared capture."""
import argparse
import inspect
import json
import os
from pathlib import Path

import numpy as np
import pytest

from scripts import benchmark_fast, retarget_and_audit

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "test_data/c167_1530_final_v3"


def test_denoise_option_is_forwarded_only_when_given():
    args = argparse.Namespace(source_denoise=None)
    assert retarget_and_audit.solver_argv(args) == []
    args.source_denoise = "auto"
    assert retarget_and_audit.solver_argv(args) == ["--source-denoise", "auto"]


def test_benchmark_child_and_entry_point_share_one_capture():
    assert "capture_run(" in inspect.getsource(benchmark_fast.child)
    assert retarget_and_audit.capture_run is benchmark_fast.capture_run
    signature = inspect.signature(benchmark_fast.capture_run)
    assert {"extra_argv", "source_root"} <= set(signature.parameters)


def test_child_environment_pins_threads_and_source_root():
    env = retarget_and_audit.child_env()
    assert env["PYTHONPATH"] == str(ROOT)
    assert all(env[key] == value for key, value in benchmark_fast.THREADS.items())


@pytest.mark.skipif(os.environ.get("OMNIRETARGETING_SLOW_TESTS") != "1" or not SAMPLE.exists(),
                    reason="end-to-end solve; set OMNIRETARGETING_SLOW_TESTS=1 with test_data present")
def test_end_to_end_solve_keeps_audit_inputs(tmp_path):
    source = tmp_path / "source.json"
    source.write_text(json.dumps(dict(type="smplx", motion=str(SAMPLE / "smplx_amass.npz"),
                                      model_directory=str(ROOT / "assets/body_models"), gender="neutral",
                                      terrain=str(SAMPLE / "scene.obj"))))
    out = tmp_path / "run"
    assert retarget_and_audit.main(["--source", str(source), "--out", str(out),
                                    "--source-denoise", "auto", "--no-audits"]) == 0
    for name in ("prepared.npy", "fixed_reference.npz", "final_retargeted.npz", "fast_report.json",
                 "timing.json", "objects.npy", "static_scene.obj", "invariants.json"):
        assert (out / name).exists(), name
    motion = np.load(out / "final_retargeted.npz")
    # Frame-id gaps are restored by the adapter, so compare with the exported source.
    assert len(motion["qpos"]) == len(motion["source_joints_w"]) >= len(np.load(SAMPLE / "smplx_amass.npz")["trans"])
    report = json.loads((out / "fast_report.json").read_text())
    assert report["deadline"]["expired"] is False
    invocation = json.loads((out / "invocation.json").read_text())
    assert invocation["argv"][-2:] == ["--source-denoise", "auto"]
