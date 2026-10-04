"""Run a Python entry point with the optional QOCO CUDA conic backend.

Usage: .venv/bin/python scripts/run_gpu_solver.py -- -m omniretargeting.main ...
"""

from __future__ import annotations

import os
import sys
import sysconfig
from pathlib import Path


def main() -> None:
    args = sys.argv[1:]
    if args and args[0] == "--":
        args = args[1:]
    if not args:
        raise SystemExit("Pass a Python script or -m module after --")

    site = Path(sysconfig.get_paths()["purelib"])
    cuda_lib = site / "nvidia" / "cu13" / "lib"
    cudss = cuda_lib / "libcudss.so.0"
    if not cudss.is_file():
        raise SystemExit("GPU dependencies missing; install omniretargeting[gpu]")
    link_dir = Path(__file__).resolve().parents[1] / "out" / "gpu_runtime" / "lib"
    link_dir.mkdir(parents=True, exist_ok=True)
    # qoco-cuda loads the library by its unversioned or its SONAME-1 name.
    for name in ("libcudss.so", "libcudss.so.1"):
        link = link_dir / name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(cudss)

    libraries = [str(link_dir), str(cuda_lib)]
    for candidate in (
        Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")) / "targets/x86_64-linux/lib",
        Path("/usr/local/cuda-12.6/targets/x86_64-linux/lib"),
    ):
        if candidate.is_dir():
            libraries.append(str(candidate))
    if os.environ.get("LD_LIBRARY_PATH"):
        libraries.append(os.environ["LD_LIBRARY_PATH"])
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = os.pathsep.join(libraries)
    env.setdefault("OMNIRETARGETING_CONIC_BACKEND", "qoco_cuda")
    os.execvpe(sys.executable, [sys.executable, *args], env)


if __name__ == "__main__":
    main()
