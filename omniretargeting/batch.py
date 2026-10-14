"""
Batch processing script for OmniRetargeting.

Scans a folder of motion files, writes per-motion source configs,
and executes main.py for each file with video export (scene scale stays 1.0).

Usage:
    python -m omniretargeting.batch \
        --source-folder /path/to/motions \
        --source-type smplx \
        --robot-config robot_models/unitree_g1/unitree_g1_mesh.json \
        --output-dir /tmp/batch_output \
        --model-dir /path/to/smplx/models
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml

from omniretargeting.data_sources.registry import get_source_extensions
from omniretargeting.source_config import absolute_source_paths

REPO_ROOT = Path(__file__).resolve().parents[1]


def detect_resources() -> dict:
    """Detect available CPU cores and total system memory."""
    cpu_count = os.cpu_count() or 1
    memory_gb = None
    try:
        import psutil
        mem = psutil.virtual_memory()
        memory_gb = mem.total / (1024**3)
    except ImportError:
        pass
    return {"cpu_count": cpu_count, "memory_gb": memory_gb}


def scan_source_folder(folder: Path, source_type: str, recursive: bool = False, exclude_suffix: str | None = None) -> list[Path]:
    """Return motion files in *folder* matching *source_type*, sorted by name."""
    exts = get_source_extensions(source_type)
    files: list[Path] = []
    for ext in exts:
        pattern = f"**/*{ext}" if recursive else f"*{ext}"
        files.extend(sorted(folder.glob(pattern)))
    if exclude_suffix:
        files = [f for f in files if not f.name.endswith(exclude_suffix)]
    return files


def _resolve_rel_subdir(motion_file: Path, source_folder: Path | None) -> str | None:
    """Return the relative subdirectory of *motion_file* within *source_folder*.

    Returns None for files directly in *source_folder* or when *source_folder*
    is None.
    """
    if source_folder and motion_file.is_relative_to(source_folder):
        s = str(motion_file.parent.relative_to(source_folder))
        return None if s == "." else s
    return None


def _output_exists(
    motion_file: Path,
    output_dir: Path,
    source_folder: Path | None = None,
) -> bool:
    """Return True if the retargeted .npz for *motion_file* already exists."""
    rel_subdir = _resolve_rel_subdir(motion_file, source_folder)
    motion_dir = output_dir / "motions" / (rel_subdir or "") / motion_file.stem
    return (motion_dir / f"{motion_file.stem}_retargeted.npz").is_file()


def write_source_config(
    motion_path: Path,
    source_type: str,
    output_dir: Path,
    terrain_path: str | None = None,
    extra_options: dict | None = None,
    rel_subdir: str | None = None,
) -> Path:
    """Write a per-motion YAML source config and return its path."""
    config: dict = {
        "type": source_type,
        "motion": str(motion_path),
    }
    if terrain_path:
        config["terrain"] = terrain_path
    if extra_options:
        config.update(extra_options)
    config = absolute_source_paths(config)

    configs_dir = output_dir / "configs" / (rel_subdir or "")
    configs_dir.mkdir(parents=True, exist_ok=True)
    config_path = configs_dir / f"{motion_path.stem}_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return config_path


def get_activation_prefix() -> str:
    """Return a shell activation command for the current conda/virtualenv, or ''."""
    conda_env = os.environ.get("CONDA_DEFAULT_ENV")
    if conda_env:
        conda_prefix = os.environ.get("CONDA_PREFIX", "")
        conda_base = conda_prefix.rsplit("/envs", 1)[0] if "/envs" in conda_prefix else ""
        if not conda_base:
            for candidate in [
                os.path.expanduser("~/anaconda3"),
                os.path.expanduser("~/miniconda3"),
            ]:
                if os.path.isdir(candidate):
                    conda_base = candidate
                    break
        if conda_base:
            return f"source {conda_base}/etc/profile.d/conda.sh && conda activate {conda_env}"
        return f"conda activate {conda_env}"

    virtual_env = os.environ.get("VIRTUAL_ENV")
    if virtual_env:
        return f"source {virtual_env}/bin/activate"
    return ""


def _find_git_root(path: Path) -> str | None:
    """Walk upward from *path* to find the nearest .git directory."""
    path = path.resolve()
    while path != path.parent:
        if (path / ".git").exists():
            return str(path)
        path = path.parent
    return None


def _git_status(repo_path: str) -> str:
    """Return branch, commit, and dirty status for a git repository."""
    lines: list[str] = []
    try:
        for args, label in [
            (["rev-parse", "--abbrev-ref", "HEAD"], "Branch"),
            (["rev-parse", "HEAD"], "Commit"),
        ]:
            r = subprocess.run(
                ["git", "-C", repo_path] + args,
                capture_output=True, text=True, timeout=10,
            )
            lines.append(f"{label}: {r.stdout.strip()}")

        r = subprocess.run(
            ["git", "-C", repo_path, "status", "--porcelain"],
            capture_output=True, text=True, timeout=10,
        )
        dirty = r.stdout.strip()
        lines.append(f"Dirty: {'yes' if dirty else 'no'}")
        if dirty:
            lines.append("Changes:")
            for line in dirty.split("\n")[:50]:
                lines.append(f"  {line}")
    except Exception as exc:
        lines.append(f"Error: {exc}")
    return "\n".join(lines)


def _git_diff(repo_path: str) -> str:
    """Return the full git diff (staged + unstaged + untracked) for a repository."""
    parts = []
    try:
        r = subprocess.run(
            ["git", "-C", repo_path, "diff", "HEAD"],
            capture_output=True, text=True, timeout=30,
        )
        if r.stdout.strip():
            parts.append(r.stdout)
    except Exception as exc:
        parts.append(f"Error getting diff: {exc}\n")

    try:
        r = subprocess.run(
            ["git", "-C", repo_path, "ls-files", "--others", "--exclude-standard"],
            capture_output=True, text=True, timeout=30,
        )
        if r.stdout.strip():
            parts.append("# Untracked files:\n")
            for f in r.stdout.strip().splitlines():
                parts.append(f"#   {f}\n")
    except Exception:
        pass

    return "".join(parts)


def _save_repo_snapshot(git_status_dir: Path, repo_name: str, repo_path: str) -> None:
    """Save git status and full diff for a repo into the git_status folder."""
    status = _git_status(repo_path)
    with open(git_status_dir / f"{repo_name}.status", "w") as f:
        f.write(status + "\n")

    diff = _git_diff(repo_path)
    if diff.strip():
        with open(git_status_dir / f"{repo_name}.diff", "w") as f:
            f.write(diff)


def log_repo_status(output_dir: Path, robot_config_path: str | None = None) -> None:
    """Save git status and diffs for relevant repos into ``git_status/``."""
    git_status_dir = output_dir / "git_status"
    git_status_dir.mkdir(parents=True, exist_ok=True)

    _save_repo_snapshot(git_status_dir, "omniretargeting", str(REPO_ROOT))

    if robot_config_path:
        rc_path = Path(robot_config_path)
        if rc_path.exists():
            with open(rc_path) as f:
                robot_config = json.load(f)
            urdf_path = robot_config.get("urdf_path") or robot_config.get("robot", {}).get("urdf_path", "")
            if isinstance(urdf_path, str) and urdf_path.startswith("package://"):
                pkg_name = urdf_path[len("package://"):].split("/")[0]
                try:
                    import importlib
                    mod = importlib.import_module(pkg_name)
                    if mod.__file__:
                        pkg_dir = Path(mod.__file__).parent
                        git_root = _find_git_root(pkg_dir)
                        if git_root:
                            _save_repo_snapshot(git_status_dir, pkg_name, git_root)
                except Exception:
                    pass

    print(f"Repository status saved to {git_status_dir}")


def _build_command(
    source_config_path: Path,
    robot_config_path: str,
    output_dir: Path,
    motion_stem: str,
    framerate: float | None = None,
    output_framerate: float | None = None,
    rel_subdir: str | None = None,
    save_video: bool = True,
) -> list[str]:
    """Build the main.py argument list for one motion file."""
    if rel_subdir:
        motion_dir = output_dir / "motions" / rel_subdir / motion_stem
    else:
        motion_dir = output_dir / "motions" / motion_stem
    motion_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "omniretargeting.main",
        "--robot-config", robot_config_path,
        "--source-config", str(source_config_path),
        "--output", str(motion_dir / f"{motion_stem}_retargeted.npz"),
    ]
    if save_video:
        cmd.extend(["--save-video", str(motion_dir / f"{motion_stem}_retargeted.mp4")])
    if framerate is not None:
        cmd.extend(["--framerate", str(framerate)])
    if output_framerate is not None:
        cmd.extend(["--output-framerate", str(output_framerate)])
    return cmd


def _poll_peak_memory(pid: int, result_holder: list[float], stop_event: threading.Event) -> None:
    """Poll VmHWM (high-water mark) of a process tree until it exits or stop_event is set."""
    peak_mb = 0.0
    while not stop_event.is_set():
        try:
            pids = [pid]
            try:
                import psutil
                parent = psutil.Process(pid)
                pids = [p.pid for p in parent.children(recursive=True)] + [pid]
            except (ImportError, psutil.NoSuchProcess):
                pass

            for p in pids:
                try:
                    with open(f"/proc/{p}/status") as f:
                        for line in f:
                            if line.startswith("VmHWM:"):
                                kb = int(line.split()[1])
                                peak_mb = max(peak_mb, kb / 1024.0)
                                break
                except (FileNotFoundError, ProcessLookupError, ValueError):
                    continue
        except Exception:
            break
        stop_event.wait(1.0)
    result_holder.append(peak_mb)


def run_single_job(
    cmd: list[str],
    activation_prefix: str,
    log_file: Path,
    timeout: float,
    measure_memory: bool = False,
) -> dict:
    """Launch one retargeting subprocess and return timing/status.

    When *measure_memory* is True, poll the child's VmHWM and return
    ``peak_memory_mb`` in the result dict.
    """
    if activation_prefix:
        full_cmd = f"{activation_prefix} && {' '.join(cmd)}"
        shell_cmd: list[str] = ["bash", "-lc", full_cmd]
    else:
        shell_cmd = cmd

    start = time.time()
    log_file.parent.mkdir(parents=True, exist_ok=True)

    stop_event: threading.Event | None = None
    mem_thread: threading.Thread | None = None
    mem_result: list[float] = []

    try:
        with open(log_file, "w") as f:
            proc = subprocess.Popen(shell_cmd, stdout=f, stderr=subprocess.STDOUT, text=True)

            if measure_memory:
                stop_event = threading.Event()
                mem_thread = threading.Thread(
                    target=_poll_peak_memory,
                    args=(proc.pid, mem_result, stop_event),
                    daemon=True,
                )
                mem_thread.start()

            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                elapsed = time.time() - start
                if stop_event:
                    stop_event.set()
                if mem_thread:
                    mem_thread.join(timeout=5)
                with open(log_file, "a") as fa:
                    fa.write(f"\n\n*** Killed after timeout ({timeout:.0f}s, elapsed {elapsed:.0f}s) ***\n")
                result: dict = {"returncode": -1, "elapsed": elapsed, "log_file": str(log_file), "timed_out": True}
                if mem_result:
                    result["peak_memory_mb"] = mem_result[0]
                return result

        elapsed = time.time() - start
        if stop_event:
            stop_event.set()
        if mem_thread:
            mem_thread.join(timeout=5)

        result = {"returncode": proc.returncode, "elapsed": elapsed, "log_file": str(log_file)}
        if mem_result:
            result["peak_memory_mb"] = mem_result[0]
        return result
    except Exception as exc:
        elapsed = time.time() - start
        if stop_event:
            stop_event.set()
        if mem_thread:
            mem_thread.join(timeout=5)
        return {"returncode": -1, "elapsed": elapsed, "log_file": str(log_file), "error": str(exc)}


def _run_test_job(
    motion_files: list[Path],
    source_type: str,
    robot_config_path: str,
    output_dir: Path,
    activation_prefix: str,
    terrain_path: str | None,
    extra_options: dict,
    framerate: float | None,
    timeout: float,
    source_folder: Path | None = None,
    save_video: bool = True,
    output_framerate: float | None = None,
) -> dict:
    """Run the first motion as a probe job.  Returns timing info for batch-size tuning."""
    first = motion_files[0]
    print(f"\n--- Running test job: {first.name} ---")
    rel_subdir = _resolve_rel_subdir(first, source_folder)
    config_path = write_source_config(first, source_type, output_dir, terrain_path, extra_options, rel_subdir)
    cmd = _build_command(config_path, robot_config_path, output_dir, first.stem, framerate, output_framerate, rel_subdir, save_video=save_video)
    log_file = output_dir / "logs" / (f"{rel_subdir}/{first.stem}.log" if rel_subdir else f"{first.stem}.log")

    print(f"  Command: {' '.join(cmd)}")
    result = run_single_job(cmd, activation_prefix, log_file, timeout, measure_memory=True)
    result["motion_file"] = str(first)
    result["motion_stem"] = first.stem

    if result.get("timed_out"):
        print(f"  Test job TIMED OUT ({result['elapsed']:.0f}s > {timeout:.0f}s limit)")
    elif result["returncode"] == 0:
        peak = result.get("peak_memory_mb", 0)
        print(f"  Test job OK ({result['elapsed']:.1f}s, peak memory: {peak:.0f} MB)")
    else:
        print(f"  Test job FAILED (rc={result['returncode']}, {result['elapsed']:.1f}s)")
        print(f"  Log: {result['log_file']}")
    return result


def _determine_workers_from_memory(
    test_peak_mb: float,
    reserved_memory_ratio: float,
    max_workers_cap: int | None,
) -> int:
    """Compute parallel workers from available memory and test job peak usage."""
    try:
        import psutil
        total_mb = psutil.virtual_memory().total / (1024 * 1024)
    except ImportError:
        total_mb = None

    if total_mb is None or test_peak_mb <= 0:
        return 1

    usable_mb = total_mb * (1.0 - reserved_memory_ratio)
    workers = max(1, int(usable_mb / test_peak_mb))

    cpu_count = os.cpu_count() or 1
    workers = min(workers, cpu_count)

    if max_workers_cap is not None:
        workers = min(workers, max_workers_cap)

    return workers


def _run_one_motion(
    motion_file: Path,
    source_type: str,
    robot_config_path: str,
    output_dir: Path,
    activation_prefix: str,
    terrain_path: str | None,
    extra_options: dict,
    framerate: float | None,
    timeout: float,
    source_folder: Path | None = None,
    save_video: bool = True,
    output_framerate: float | None = None,
) -> dict:
    """Process a single motion file (called from worker processes)."""
    motion_stem = motion_file.stem
    rel_subdir = _resolve_rel_subdir(motion_file, source_folder)
    config_path = write_source_config(motion_file, source_type, output_dir, terrain_path, extra_options, rel_subdir)
    cmd = _build_command(config_path, robot_config_path, output_dir, motion_stem, framerate, output_framerate, rel_subdir, save_video=save_video)
    log_file = output_dir / "logs" / (f"{rel_subdir}/{motion_stem}.log" if rel_subdir else f"{motion_stem}.log")
    result = run_single_job(cmd, activation_prefix, log_file, timeout)
    result["motion_file"] = str(motion_file)
    result["motion_stem"] = motion_stem
    return result


def process_batch(
    motion_files: list[Path],
    source_type: str,
    robot_config_path: str,
    output_dir: Path,
    activation_prefix: str,
    terrain_path: str | None = None,
    extra_options: dict | None = None,
    max_workers: int | None = None,
    framerate: float | None = None,
    skip_test_job: bool = False,
    timeout: float = 3600,
    reserved_memory_ratio: float = 0.4,
    source_folder: Path | None = None,
    save_video: bool = True,
    output_framerate: float | None = None,
) -> list[dict]:
    """Process every motion file, returning per-file results.

    Runs a test job first to measure peak memory, then launches remaining
    files in parallel with worker count derived from available memory.
    """
    if extra_options is None:
        extra_options = {}

    results: list[dict] = []
    remaining = list(motion_files)

    test_peak_mb = 0.0
    if remaining and not skip_test_job:
        test_result = _run_test_job(
            remaining, source_type, robot_config_path, output_dir,
            activation_prefix, terrain_path, extra_options, framerate, timeout,
            source_folder=source_folder, save_video=save_video,
            output_framerate=output_framerate,
        )
        results.append(test_result)
        if test_result["returncode"] != 0:
            print("\nTest job failed. Aborting batch. Check the log above.")
            return results
        test_peak_mb = test_result.get("peak_memory_mb", 0)
        remaining = remaining[1:]

    if not remaining:
        return results

    if test_peak_mb > 0:
        num_workers = _determine_workers_from_memory(
            test_peak_mb, reserved_memory_ratio, max_workers,
        )
        print(f"\nMemory-based parallelism: {test_peak_mb:.0f} MB/job, "
              f"reserved ratio={reserved_memory_ratio}, workers={num_workers}")
    else:
        num_workers = max_workers or max(1, (os.cpu_count() or 1) - 1)
        print(f"\nNo memory measurement available, using {num_workers} workers")

    print(f"Processing {len(remaining)} remaining files with {num_workers} parallel workers\n")

    if num_workers == 1:
        for i, motion_file in enumerate(remaining):
            print(f"[{i + len(results) + 1}/{len(motion_files)}] Processing: {motion_file.name}")
            result = _run_one_motion(
                motion_file, source_type, robot_config_path, output_dir,
                activation_prefix, terrain_path, extra_options, framerate, timeout,
                source_folder=source_folder, save_video=save_video,
                output_framerate=output_framerate,
            )
            results.append(result)
            status = "TIMED OUT" if result.get("timed_out") else (
                f"OK ({result['elapsed']:.1f}s)" if result["returncode"] == 0
                else f"FAILED (rc={result['returncode']})"
            )
            print(f"  {status}")
    else:
        future_to_motion = {}
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for motion_file in remaining:
                future = executor.submit(
                    _run_one_motion,
                    motion_file, source_type, robot_config_path, output_dir,
                    activation_prefix, terrain_path, extra_options, framerate, timeout,
                    source_folder=source_folder, save_video=save_video,
                    output_framerate=output_framerate,
                )
                future_to_motion[future] = motion_file

            done_count = len(results)
            for future in as_completed(future_to_motion):
                motion_file = future_to_motion[future]
                done_count += 1
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "returncode": -1, "elapsed": 0, "motion_file": str(motion_file),
                        "motion_stem": motion_file.stem, "log_file": "", "error": str(exc),
                    }
                results.append(result)
                if result.get("timed_out"):
                    status = f"TIMED OUT ({result['elapsed']:.0f}s)"
                elif result["returncode"] == 0:
                    status = f"OK ({result['elapsed']:.1f}s)"
                else:
                    status = f"FAILED (rc={result['returncode']})"
                print(f"[{done_count}/{len(motion_files)}] {motion_file.name}: {status}")

    return results


def _summarize(results: list[dict]) -> int:
    total = len(results)
    ok = sum(1 for r in results if r["returncode"] == 0)
    timed_out = sum(1 for r in results if r.get("timed_out"))
    failed = total - ok
    print(f"\n{'=' * 60}")
    print(f"Batch complete: {ok}/{total} succeeded, {failed} failed")
    if timed_out:
        print(f"  ({timed_out} timed out)")
    if failed:
        print("Failures:")
        for r in results:
            if r.get("timed_out"):
                print(f"  - {r['motion_stem']}: TIMED OUT after {r['elapsed']:.0f}s (log: {r['log_file']})")
            elif r["returncode"] != 0:
                print(f"  - {r['motion_stem']}: rc={r['returncode']} (log: {r['log_file']})")
    return failed


def main() -> None:
    parser = argparse.ArgumentParser(description="OmniRetargeting Batch Processor")
    parser.add_argument("--source-folder", required=True,
                        help="Folder containing motion files to process")
    parser.add_argument("--source-type", required=True,
                        help="Type of motion data source (e.g. smplx, lafan1, nokov, omomo, bones_seed)")
    parser.add_argument("--robot-config", default="robot_models/unitree_g1/unitree_g1_mesh.json",
                        help="Path to robot configuration JSON file (default: G1 mesh-collision profile)")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to save batch outputs")
    parser.add_argument("--model-dir", default=None,
                        help="Model directory (required for SMPL-X source type)")
    parser.add_argument("--terrain", default=None,
                        help="Path to terrain mesh file applied to all motions")
    parser.add_argument("--max-workers", type=int, default=None,
                        help="Maximum parallel workers (default: auto-detect)")
    parser.add_argument("--framerate", type=float, default=None,
                        help="Override framerate for all motions")
    parser.add_argument("--source-options", default=None,
                        help="JSON string of extra source options for all motions")
    parser.add_argument("--skip-test-job", action="store_true",
                        help="Skip the initial probe job and process all files directly")
    parser.add_argument("--timeout", type=float, default=3600,
                        help="Per-file timeout in seconds (default: 3600)")
    parser.add_argument("--reserved-memory-ratio", type=float, default=0.4,
                        help="Fraction of total memory to reserve (default: 0.4). "
                             "Workers are sized from (1 - ratio) * total_memory / per_job_memory.")
    parser.add_argument("--recursive", action="store_true",
                        help="Scan source folder recursively for motion files")
    parser.add_argument("--exclude-suffix", default=None,
                        help="Exclude files ending with this suffix (e.g. '_M.bvh' for mirrored files)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip motions whose retargeted output already exists")
    parser.add_argument("--no-video", action="store_true",
                        help="Skip video rendering (useful for headless servers)")
    parser.add_argument("--output-framerate", type=float, default=None,
                        help="Resample motion to this framerate before retargeting (e.g. 30 to downsample 120fps data)")

    args = parser.parse_args()

    source_folder = Path(args.source_folder)
    if not source_folder.is_dir():
        print(f"Error: source folder not found: {args.source_folder}")
        sys.exit(1)

    robot_config_path = str(Path(args.robot_config).expanduser())
    if not Path(robot_config_path).exists():
        print(f"Error: robot config not found: {args.robot_config}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.recursive and output_dir.resolve().is_relative_to(source_folder.resolve()):
        raise ValueError(
            f"--output-dir ({args.output_dir}) must not be inside "
            f"--source-folder ({args.source_folder}) when --recursive is used"
        )

    # Environment detection
    resources = detect_resources()
    print(f"CPU cores: {resources['cpu_count']}")
    if resources["memory_gb"] is not None:
        print(f"Total memory: {resources['memory_gb']:.1f} GB")

    max_workers = args.max_workers
    if max_workers:
        print(f"Max workers override: {max_workers}")
    else:
        print("Max workers: auto (determined from test job memory)")

    activation_prefix = get_activation_prefix()
    if activation_prefix:
        print(f"Environment activation: detected")
    else:
        print("Warning: no conda/virtualenv detected, using bare Python")

    # Log repository status before processing
    log_repo_status(output_dir, args.robot_config)

    # Scan for motion files
    motion_files = scan_source_folder(source_folder, args.source_type, recursive=args.recursive, exclude_suffix=args.exclude_suffix)
    if not motion_files:
        print(f"Error: no motion files found in {args.source_folder} for type '{args.source_type}'")
        sys.exit(1)
    print(f"Found {len(motion_files)} motion file(s)")

    # Resume: skip motions whose retargeted output already exists
    if args.resume:
        source_root = source_folder if args.recursive else None
        completed = [f for f in motion_files if _output_exists(f, output_dir, source_root)]
        if completed:
            print(f"Resume: skipping {len(completed)} already-processed motion(s)")
        completed_set = set(completed)
        motion_files = [f for f in motion_files if f not in completed_set]
        if not motion_files:
            print("All motions already processed. Nothing to do.")
            sys.exit(0)

    # Build shared extra options
    extra_options: dict = {}
    if args.model_dir:
        extra_options["model_directory"] = args.model_dir
    if args.source_options:
        extra_options.update(json.loads(args.source_options))

    # Process batch
    results = process_batch(
        motion_files=motion_files,
        source_type=args.source_type,
        robot_config_path=robot_config_path,
        output_dir=output_dir,
        activation_prefix=activation_prefix,
        terrain_path=args.terrain,
        extra_options=extra_options,
        max_workers=args.max_workers,
        framerate=args.framerate,
        skip_test_job=args.skip_test_job,
        timeout=args.timeout,
        reserved_memory_ratio=args.reserved_memory_ratio,
        source_folder=source_folder if args.recursive else None,
        save_video=not args.no_video,
        output_framerate=args.output_framerate,
    )

    failed = _summarize(results)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
