"""Shared batch orchestration for OmniRetargeting entry points.

Holds the reusable implementation behind ``omniretargeting.batch`` and
``omniretargeting.edp_batch``: source-config writing, subprocess execution,
memory-based worker sizing, and the first-trial-then-parallel scheduling.

Not re-exported by ``omniretargeting.utils``; import from here explicitly.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

import yaml

from omniretargeting.data_sources.registry import get_source_extensions

REPO_ROOT = Path(__file__).resolve().parents[2]

# Maps a motion file to the source config YAML main.py should consume.
ConfigPathResolver = Callable[[Path], Path]

# Per-motion resolution callables, used when one run spans several source
# types, terrains, or framerates: the callable receives the motion file and
# returns the value for that motion.
SourceTypeResolver = Callable[[Path], str]
TerrainPathResolver = Callable[[Path], str | None]
FramerateResolver = Callable[[Path], float | None]

# Downloads the record identified by an opaque key (URL, EDP file_id, ...) to
# the given destination and returns the local path actually used. A downloader
# may adjust the destination (e.g. fix the suffix) when the real filename is
# only known after the download.
DownloadFn = Callable[[str, Path], Path]

# Called in the main process with each per-motion result dict as soon as the
# job completes (including the test job's).
ResultCallback = Callable[[dict], None]


def _download_url(url: str, dest: Path) -> Path:
    """Download *url* to *dest*, skipping when *dest* already exists."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, open(dest, "wb") as f:
        shutil.copyfileobj(response, f)
    return dest


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
) -> bool:
    """Return True if the retargeted .npz for *motion_file* already exists."""
    return (output_dir / "motions" / f"{motion_file.stem}_retargeted.npz").is_file()


def write_source_config(
    motion_path: Path,
    source_type: str | SourceTypeResolver,
    output_dir: Path,
    terrain_path: str | None | TerrainPathResolver = None,
    extra_options: dict | None = None,
    rel_subdir: str | None = None,
) -> Path:
    """Write a per-motion YAML source config and return its path.

    *source_type* and *terrain_path* may be plain values or per-motion
    resolvers called with the motion file (e.g. when motions span several
    source types or terrains).
    """
    if callable(source_type):
        source_type = source_type(motion_path)
    config: dict = {
        "type": source_type,
        "motion": str(motion_path),
    }
    if callable(terrain_path):
        terrain_path = terrain_path(motion_path)
    if terrain_path:
        config["terrain"] = terrain_path
    if extra_options:
        config.update(extra_options)

    configs_dir = output_dir / "configs" / (rel_subdir or "")
    configs_dir.mkdir(parents=True, exist_ok=True)
    config_path = configs_dir / f"{motion_path.stem}_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return config_path


def same_stem_source_config(motion_file: Path) -> Path:
    """Resolve the source config YAML placed next to *motion_file*.

    Follows the same-stem pairing convention (``<motion_stem>.yaml`` beside
    the data file). Used as a :data:`ConfigPathResolver` when configs are
    pre-written alongside the motions instead of generated per run.
    """
    return motion_file.with_suffix(".yaml")


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


def get_repo_info(repo_path: str | None = None, remote: str = "origin") -> dict:
    """Return ``{repo_url, repo_commit, repo_dirty}`` for a git repository.

    Structured counterpart of :func:`_git_status`, used for provenance
    metadata. Fields are None when they cannot be determined. ``remote`` names
    the git remote whose URL is recorded as ``repo_url``.
    """
    repo_path = repo_path or str(REPO_ROOT)
    info: dict = {"repo_url": None, "repo_commit": None, "repo_dirty": None}
    try:
        r = subprocess.run(
            ["git", "-C", repo_path, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0:
            info["repo_commit"] = r.stdout.strip() or None

        r = subprocess.run(
            ["git", "-C", repo_path, "remote", "get-url", remote],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0:
            info["repo_url"] = r.stdout.strip() or None

        r = subprocess.run(
            ["git", "-C", repo_path, "status", "--porcelain"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0:
            info["repo_dirty"] = bool(r.stdout.strip())
    except Exception:
        pass
    return info


def _build_command(
    source_config_path: Path,
    robot_config_path: str,
    output_dir: Path,
    motion_stem: str,
    framerate: float | None = None,
    output_framerate: float | None = None,
    save_video: bool = False,
    scale_factor: float | None = None,
    progress: bool = False,
) -> list[str]:
    """Build the main.py argument list for one motion file."""
    motion_dir = output_dir / "motions"
    motion_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "omniretargeting.main",
        "--robot-config", robot_config_path,
        "--source-config", str(source_config_path),
        "--output", str(motion_dir / f"{motion_stem}_retargeted.npz"),
    ]
    if scale_factor is not None:
        # Uniform scale: motions scale with one shared factor and the scaled
        # terrain is exported once at the batch level, not per motion.
        cmd.extend(["--scale-factor", str(scale_factor)])
    if save_video:
        cmd.extend(["--save-video", str(motion_dir / f"{motion_stem}_retargeted.mp4")])
    if framerate is not None:
        cmd.extend(["--framerate", str(framerate)])
    if output_framerate is not None:
        cmd.extend(["--output-framerate", str(output_framerate)])
    if progress:
        cmd.append("--progress")
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
    source_type: str | SourceTypeResolver,
    robot_config_path: str,
    output_dir: Path,
    activation_prefix: str,
    terrain_path: str | None | TerrainPathResolver,
    extra_options: dict,
    framerate: float | None | FramerateResolver,
    timeout: float,
    source_folder: Path | None = None,
    save_video: bool = False,
    output_framerate: float | None = None,
    config_path_resolver: ConfigPathResolver | None = None,
    download_key: str | None = None,
    download_fn: DownloadFn | None = None,
    delete_after: bool = False,
    scale_factor: float | None = None,
    progress: bool = False,
) -> dict:
    """Run the first motion as a probe job.  Returns timing info for batch-size tuning."""
    first = motion_files[0]
    print(f"\n--- Running test job: {first.name} ---")
    if download_key is not None:
        try:
            first = (download_fn or _download_url)(download_key, first)
        except Exception as exc:
            print(f"  Test job download FAILED: {exc}")
            return {
                "returncode": -1, "elapsed": 0.0, "log_file": "",
                "motion_file": str(first), "motion_stem": first.stem,
                "error": f"download failed: {exc}",
            }
    rel_subdir = _resolve_rel_subdir(first, source_folder)
    if config_path_resolver is not None:
        config_path = config_path_resolver(first)
    else:
        config_path = write_source_config(first, source_type, output_dir, terrain_path, extra_options, rel_subdir)
    if callable(framerate):
        framerate = framerate(first)
    cmd = _build_command(
        config_path,
        robot_config_path,
        output_dir,
        first.stem,
        framerate,
        output_framerate,
        save_video=save_video,
        scale_factor=scale_factor,
        progress=progress,
    )
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
    if delete_after and result["returncode"] == 0:
        first.unlink(missing_ok=True)
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
    source_type: str | SourceTypeResolver,
    robot_config_path: str,
    output_dir: Path,
    activation_prefix: str,
    terrain_path: str | None | TerrainPathResolver,
    extra_options: dict,
    framerate: float | None | FramerateResolver,
    timeout: float,
    source_folder: Path | None = None,
    save_video: bool = False,
    output_framerate: float | None = None,
    config_path_resolver: ConfigPathResolver | None = None,
    download_key: str | None = None,
    download_fn: DownloadFn | None = None,
    delete_after: bool = False,
    scale_factor: float | None = None,
    progress: bool = False,
) -> dict:
    """Process a single motion file (called from worker processes)."""
    if download_key is not None:
        try:
            motion_file = (download_fn or _download_url)(download_key, motion_file)
        except Exception as exc:
            return {
                "returncode": -1, "elapsed": 0.0, "log_file": "",
                "motion_file": str(motion_file), "motion_stem": motion_file.stem,
                "error": f"download failed: {exc}",
            }
    motion_stem = motion_file.stem
    rel_subdir = _resolve_rel_subdir(motion_file, source_folder)
    if config_path_resolver is not None:
        config_path = config_path_resolver(motion_file)
    else:
        config_path = write_source_config(motion_file, source_type, output_dir, terrain_path, extra_options, rel_subdir)
    if callable(framerate):
        framerate = framerate(motion_file)
    cmd = _build_command(
        config_path,
        robot_config_path,
        output_dir,
        motion_stem,
        framerate,
        output_framerate,
        save_video=save_video,
        scale_factor=scale_factor,
        progress=progress,
    )
    log_file = output_dir / "logs" / (f"{rel_subdir}/{motion_stem}.log" if rel_subdir else f"{motion_stem}.log")
    result = run_single_job(cmd, activation_prefix, log_file, timeout)
    result["motion_file"] = str(motion_file)
    result["motion_stem"] = motion_stem
    if delete_after and result["returncode"] == 0:
        motion_file.unlink(missing_ok=True)
    return result


def process_batch(
    motion_files: list[Path],
    source_type: str | SourceTypeResolver,
    robot_config_path: str,
    output_dir: Path,
    activation_prefix: str,
    terrain_path: str | None | TerrainPathResolver = None,
    extra_options: dict | None = None,
    max_workers: int | None = None,
    framerate: float | None | FramerateResolver = None,
    skip_test_job: bool = False,
    timeout: float = 3600,
    reserved_memory_ratio: float = 0.4,
    source_folder: Path | None = None,
    save_video: bool = False,
    output_framerate: float | None = None,
    config_path_resolver: ConfigPathResolver | None = None,
    download_keys: dict[Path, str] | None = None,
    download_fn: DownloadFn | None = None,
    delete_downloads: bool = False,
    on_result: ResultCallback | None = None,
    worker_initializer: Callable[..., None] | None = None,
    worker_initargs: tuple = (),
    scale_factor: float | None = None,
    progress: bool = False,
) -> list[dict]:
    """Process every motion file, returning per-file results.

    Runs a test job first to measure peak memory, then launches remaining
    files in parallel with worker count derived from available memory.

    When *config_path_resolver* is given, per-motion source configs are
    resolved through it (they must already exist) instead of being written
    from *terrain_path*/*extra_options*.

    When *download_keys* maps a motion file to an opaque download key, the
    file is downloaded just before its job starts (per-record download
    instead of bulk); *download_fn* performs the download (default:
    :func:`_download_url`, treating the key as a URL) and may return an
    adjusted local path, which is then used for the job. With
    *delete_downloads* the file is removed again after a successful job.

    *on_result* is invoked in the main process with each result dict as soon
    as its job completes (including the test job's).

    *worker_initializer*/*worker_initargs* are passed through to the
    ``ProcessPoolExecutor`` (e.g. to set up per-worker download state that a
    *download_fn* relies on).

    *scale_factor* forces one source-to-robot scale for every motion (passed
    to main.py as ``--scale-factor``) instead of per-motion height estimates —
    motions sharing a terrain then get identical scaled terrains.

    *source_type*, *terrain_path*, and *framerate* may be plain values or
    per-motion resolvers (called with the motion file) when a single run
    spans several source types, terrains, or framerates.
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
            config_path_resolver=config_path_resolver,
            download_key=(download_keys or {}).get(remaining[0]),
            download_fn=download_fn,
            delete_after=delete_downloads,
            scale_factor=scale_factor,
            progress=progress,
        )
        results.append(test_result)
        if on_result is not None:
            on_result(test_result)
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
        completed_before_remaining = len(results)
        for i, motion_file in enumerate(remaining):
            print(
                f"[{completed_before_remaining + i + 1}/{len(motion_files)}] "
                f"Processing: {motion_file.name}"
            )
            result = _run_one_motion(
                motion_file, source_type, robot_config_path, output_dir,
                activation_prefix, terrain_path, extra_options, framerate, timeout,
                source_folder=source_folder, save_video=save_video,
                output_framerate=output_framerate,
                config_path_resolver=config_path_resolver,
                download_key=(download_keys or {}).get(motion_file),
                download_fn=download_fn,
                delete_after=delete_downloads,
                scale_factor=scale_factor,
                progress=progress,
            )
            results.append(result)
            if on_result is not None:
                on_result(result)
            status = "TIMED OUT" if result.get("timed_out") else (
                f"OK ({result['elapsed']:.1f}s)" if result["returncode"] == 0
                else f"FAILED (rc={result['returncode']})"
            )
            print(f"  {status}")
    else:
        future_to_motion = {}
        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=worker_initializer,
            initargs=worker_initargs,
        ) as executor:
            for motion_file in remaining:
                future = executor.submit(
                    _run_one_motion,
                    motion_file, source_type, robot_config_path, output_dir,
                    activation_prefix, terrain_path, extra_options, framerate, timeout,
                    source_folder=source_folder, save_video=save_video,
                    output_framerate=output_framerate,
                    config_path_resolver=config_path_resolver,
                    download_key=(download_keys or {}).get(motion_file),
                    download_fn=download_fn,
                    delete_after=delete_downloads,
                    scale_factor=scale_factor,
                    progress=progress,
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
                if on_result is not None:
                    on_result(result)
                if result.get("timed_out"):
                    status = f"TIMED OUT ({result['elapsed']:.0f}s)"
                elif result["returncode"] == 0:
                    status = f"OK ({result['elapsed']:.1f}s)"
                else:
                    status = f"FAILED (rc={result['returncode']})"
                print(f"[{done_count}/{len(motion_files)}] {motion_file.name}: {status}")

    return results


def export_shared_scaled_terrain(
    terrain_path: str,
    scale_factor: float,
    output_dir: Path,
    filename: str = "scaled_terrain.obj",
) -> Path:
    """Export the terrain scaled by the uniform *scale_factor* once.

    Used by batch entry points when a single factor applies to every motion:
    the shared scaled terrain is written once instead of once per motion.
    """
    import trimesh

    scaled_terrain = trimesh.load(terrain_path, force="mesh")
    scaled_terrain.apply_scale(scale_factor)
    terrain_output_dir = output_dir / "terrain"
    terrain_output_dir.mkdir(parents=True, exist_ok=True)
    shared_path = terrain_output_dir / filename
    scaled_terrain.export(shared_path)
    print(f"Saved shared scaled terrain mesh to {shared_path} "
          f"(uniform scale factor {scale_factor})")
    return shared_path


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
