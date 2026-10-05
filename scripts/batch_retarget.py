"""Run independent retargeting clips concurrently with explicit memory admission."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"
TIME_COMMAND = "/usr/bin/time"
DEFAULT_MAX_JOBS = 4
DEFAULT_MEMORY_BUDGET_GB = 46.0
DEFAULT_MEMORY_GB = 10.0
DEFAULT_RESERVE_GB = 10.0
DEFAULT_CPUS_PER_JOB = 6
_EPSILON = 1e-9
_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_RSS_RE = re.compile(r"Maximum resident set size \(kbytes\):\s*(\d+)")


@dataclass(frozen=True)
class JobSpec:
    index: int
    name: str
    args: tuple[str, ...] | None
    command: tuple[str, ...] | None
    memory_gb: float
    env: dict[str, str]


@dataclass
class ActiveJob:
    spec: JobSpec
    process: subprocess.Popen[str]
    log_handle: Any
    log_path: Path
    queued_seconds: float
    started_monotonic: float
    started_at: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return number


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _load_manifest(path: Path, default_memory_gb: float) -> list[JobSpec]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        raw_jobs = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        loaded = json.loads(text)
        raw_jobs = loaded if isinstance(loaded, list) else [loaded]

    if not isinstance(raw_jobs, list) or not all(
        isinstance(item, dict) for item in raw_jobs
    ):
        raise ValueError(
            "manifest must be a JSON array, a JSON object, or JSONL objects"
        )

    jobs: list[JobSpec] = []
    names: set[str] = set()
    for index, item in enumerate(raw_jobs):
        name = item.get("name")
        if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
            raise ValueError(
                f"job {index}: 'name' must be a safe file stem, got {name!r}"
            )
        if name in names:
            raise ValueError(f"duplicate job name: {name}")
        names.add(name)

        has_args = "args" in item
        has_command = "command" in item
        if has_args == has_command:
            raise ValueError(
                f"job {name!r}: provide exactly one of 'args' or 'command'"
            )
        raw_args = item.get("args") if has_args else item.get("command")
        if not isinstance(raw_args, list) or not all(
            isinstance(arg, str) for arg in raw_args
        ):
            raise ValueError(
                f"job {name!r}: 'args'/'command' must be a list of strings"
            )
        if has_command and not raw_args:
            raise ValueError(f"job {name!r}: 'command' cannot be empty")

        memory_gb = item.get("memory_gb", default_memory_gb)
        if isinstance(memory_gb, bool) or not isinstance(memory_gb, (int, float)):
            raise ValueError(f"job {name!r}: 'memory_gb' must be a number")
        memory_gb = float(memory_gb)
        if not math.isfinite(memory_gb) or memory_gb <= 0:
            raise ValueError(f"job {name!r}: 'memory_gb' must be positive and finite")

        raw_env = item.get("env", {})
        if not isinstance(raw_env, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in raw_env.items()
        ):
            raise ValueError(
                f"job {name!r}: 'env' must map string keys to string values"
            )

        jobs.append(
            JobSpec(
                index=index,
                name=name,
                args=tuple(raw_args) if has_args else None,
                command=tuple(raw_args) if has_command else None,
                memory_gb=memory_gb,
                env=dict(raw_env),
            )
        )
    return jobs


def read_mem_available_gb(meminfo_path: Path | str = "/proc/meminfo") -> float:
    """Read MemAvailable from Linux procfs and convert KiB to binary GiB."""
    match = re.search(
        r"^MemAvailable:\s+(\d+) kB$", Path(meminfo_path).read_text(), re.MULTILINE
    )
    if match is None:
        raise RuntimeError(f"MemAvailable not found in {meminfo_path}")
    return int(match.group(1)) * 1024 / (1024**3)


def _job_command(spec: JobSpec) -> list[str]:
    if spec.command is not None:
        return list(spec.command)
    assert spec.args is not None
    return [str(PYTHON), "-m", "omniretargeting.main", *spec.args]


def _job_environment(spec: JobSpec, cpus_per_job: int) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(spec.env)
    environment.update(
        {
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "OMNIRETARGETING_CPUS": str(cpus_per_job),
        }
    )
    return environment


def _launch_command(spec: JobSpec, use_systemd: bool, use_time: bool) -> list[str]:
    payload = _job_command(spec)
    if use_time:
        payload = [TIME_COMMAND, "-v", "--", *payload]
    if not use_systemd:
        return payload
    memory_max = format(spec.memory_gb * 1.5, ".15g")
    return [
        "systemd-run",
        "--user",
        "--scope",
        "-p",
        f"MemoryMax={memory_max}G",
        "-p",
        "MemorySwapMax=0",
        "--",
        *payload,
    ]


def _append_log(handle: Any, message: str) -> None:
    handle.write(message + "\n")
    handle.flush()


def _next_eligible(
    pending: list[JobSpec],
    reserved_gb: float,
    memory_budget_gb: float,
    reserve_gb: float,
    largest_first: bool,
) -> JobSpec | None:
    if largest_first:
        candidates = sorted(pending, key=lambda job: (-job.memory_gb, job.index))
    else:
        # Preserve manifest order: a blocked head-of-line job waits rather than
        # letting later clips overtake it.
        candidates = pending[:1]
    available_gb = read_mem_available_gb()
    for spec in candidates:
        within_budget = reserved_gb + spec.memory_gb <= memory_budget_gb + _EPSILON
        leaves_reserve = available_gb - reserved_gb >= reserve_gb - _EPSILON
        if within_budget and leaves_reserve:
            return spec
    return None


def _parse_peak_rss_gb(log_text: str) -> float | None:
    match = _RSS_RE.search(log_text)
    if match is None:
        return None
    return int(match.group(1)) * 1024 / (1024**3)


def _finalize_job(
    active: ActiveJob,
    exit_code: int | None,
    batch_start: float,
    status: str,
) -> dict[str, Any]:
    ended_monotonic = time.monotonic()
    _append_log(
        active.log_handle,
        f"ended_at={_utc_now()}\nexit_code={exit_code}\nstatus={status}\n",
    )
    active.log_handle.close()
    log_text = active.log_path.read_text(encoding="utf-8", errors="replace")
    return {
        "name": active.spec.name,
        "index": active.spec.index,
        "status": status,
        "exit_code": exit_code,
        "memory_gb": active.spec.memory_gb,
        "queued_seconds": active.queued_seconds,
        "wall_seconds": ended_monotonic - active.started_monotonic,
        "started_at": active.started_at,
        "ended_at": _utc_now(),
        "peak_rss_gb": _parse_peak_rss_gb(log_text),
        "log": str(active.log_path),
    }


def _launch_job(
    spec: JobSpec,
    log_dir: Path,
    batch_start: float,
    cpus_per_job: int,
    use_systemd: bool,
    use_time: bool,
) -> ActiveJob | dict[str, Any]:
    log_path = log_dir / f"{spec.name}.log"
    command = _launch_command(spec, use_systemd=use_systemd, use_time=use_time)
    log_handle = log_path.open("w", encoding="utf-8", buffering=1)
    queued_seconds = time.monotonic() - batch_start
    started_at = _utc_now()
    _append_log(
        log_handle,
        "\n".join(
            [
                f"=== batch job {spec.name} ===",
                f"index={spec.index}",
                f"memory_gb={spec.memory_gb:g}",
                f"command={shlex.join(command)}",
                f"started_at={started_at}",
            ]
        )
        + "\n",
    )
    try:
        process = subprocess.Popen(
            command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=_job_environment(spec, cpus_per_job),
            start_new_session=True,
        )
    except OSError as error:
        _append_log(
            log_handle,
            f"launch_error={error}\nended_at={_utc_now()}\nstatus=launch_error\n",
        )
        log_handle.close()
        return {
            "name": spec.name,
            "index": spec.index,
            "status": "launch_error",
            "exit_code": None,
            "memory_gb": spec.memory_gb,
            "queued_seconds": queued_seconds,
            "wall_seconds": 0.0,
            "started_at": started_at,
            "ended_at": _utc_now(),
            "peak_rss_gb": None,
            "log": str(log_path),
            "error": str(error),
        }
    return ActiveJob(
        spec=spec,
        process=process,
        log_handle=log_handle,
        log_path=log_path,
        queued_seconds=queued_seconds,
        started_monotonic=time.monotonic(),
        started_at=started_at,
    )


def _terminate_active(active: dict[int, ActiveJob]) -> None:
    for record in active.values():
        if record.process.poll() is None:
            try:
                os.killpg(record.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 5.0
    for record in active.values():
        remaining = max(0.0, deadline - time.monotonic())
        try:
            record.process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(record.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            record.process.wait()


def _skipped_result(spec: JobSpec, batch_start: float) -> dict[str, Any]:
    now = _utc_now()
    return {
        "name": spec.name,
        "index": spec.index,
        "status": "skipped",
        "exit_code": None,
        "memory_gb": spec.memory_gb,
        "queued_seconds": time.monotonic() - batch_start,
        "wall_seconds": 0.0,
        "started_at": None,
        "ended_at": now,
        "peak_rss_gb": None,
        "log": None,
    }


def run_batch(
    manifest_path: Path,
    log_dir: Path,
    summary_path: Path,
    *,
    max_jobs: int = DEFAULT_MAX_JOBS,
    memory_budget_gb: float = DEFAULT_MEMORY_BUDGET_GB,
    default_memory_gb: float = DEFAULT_MEMORY_GB,
    reserve_gb: float = DEFAULT_RESERVE_GB,
    cpus_per_job: int = DEFAULT_CPUS_PER_JOB,
    largest_first: bool = False,
    use_systemd: bool = True,
    use_time: bool = True,
    poll_interval: float = 0.2,
) -> dict[str, Any]:
    """Run a manifest and write a JSON summary; failed jobs are not retried."""
    if max_jobs <= 0 or cpus_per_job <= 0:
        raise ValueError("--max-jobs and --cpus-per-job must be positive")
    if (
        memory_budget_gb <= 0
        or default_memory_gb <= 0
        or reserve_gb < 0
        or poll_interval < 0
    ):
        raise ValueError(
            "memory defaults, budget, reserve, and poll interval must be nonnegative"
        )

    jobs = _load_manifest(manifest_path, default_memory_gb)
    impossible = [
        job.name for job in jobs if job.memory_gb > memory_budget_gb + _EPSILON
    ]
    if impossible:
        raise ValueError(f"jobs exceed --memory-budget-gb: {', '.join(impossible)}")

    log_dir.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    started_monotonic = time.monotonic()
    started_at = _utc_now()
    pending = list(jobs)
    active: dict[int, ActiveJob] = {}
    results: dict[int, dict[str, Any]] = {}
    interrupted = False

    try:
        while pending or active:
            for index, record in list(active.items()):
                exit_code = record.process.poll()
                if exit_code is not None:
                    status = "success" if exit_code == 0 else "failed"
                    results[index] = _finalize_job(
                        record, exit_code, started_monotonic, status
                    )
                    del active[index]

            reserved_gb = sum(record.spec.memory_gb for record in active.values())
            while len(active) < max_jobs and pending:
                spec = _next_eligible(
                    pending,
                    reserved_gb,
                    memory_budget_gb,
                    reserve_gb,
                    largest_first,
                )
                if spec is None:
                    if not active:
                        available_gb = read_mem_available_gb()
                        raise RuntimeError(
                            "no eligible job: memory admission would violate the budget or reserve "
                            f"(available={available_gb:.3f} GiB, reserved={reserved_gb:.3f} GiB, "
                            f"budget={memory_budget_gb:.3f} GiB, reserve={reserve_gb:.3f} GiB)"
                        )
                    break
                pending.remove(spec)
                launched = _launch_job(
                    spec,
                    log_dir,
                    started_monotonic,
                    cpus_per_job,
                    use_systemd,
                    use_time,
                )
                if isinstance(launched, dict):
                    results[spec.index] = launched
                else:
                    active[spec.index] = launched
                reserved_gb = sum(record.spec.memory_gb for record in active.values())

            if active:
                time.sleep(poll_interval)
    except KeyboardInterrupt:
        interrupted = True
        _terminate_active(active)

    for index, record in active.items():
        exit_code = record.process.poll()
        results[index] = _finalize_job(
            record,
            exit_code,
            started_monotonic,
            "interrupted",
        )
    active.clear()
    for spec in pending:
        results[spec.index] = _skipped_result(spec, started_monotonic)

    ended_at = _utc_now()
    ordered_results = [results[index] for index in sorted(results)]
    failures = [
        result["name"]
        for result in ordered_results
        if result["status"] in {"failed", "launch_error"}
    ]
    summary = {
        "version": 1,
        "manifest": str(manifest_path),
        "started_at": started_at,
        "ended_at": ended_at,
        "makespan_seconds": time.monotonic() - started_monotonic,
        "max_jobs": max_jobs,
        "memory_budget_gb": memory_budget_gb,
        "reserve_gb": reserve_gb,
        "cpus_per_job": cpus_per_job,
        "largest_first": largest_first,
        "systemd": use_systemd,
        "record_peak_rss": use_time,
        "interrupted": interrupted,
        "jobs": ordered_results,
        "failures": failures,
    }
    temporary_summary = summary_path.with_name(summary_path.name + ".tmp")
    temporary_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_summary.replace(summary_path)
    return summary


def _summary_exit_code(summary: dict[str, Any]) -> int:
    if summary["interrupted"]:
        return 130
    return 1 if summary["failures"] else 0


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Concurrently run independent OmniRetargeting clips",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("manifest", type=Path, help="JSON array or JSONL job manifest")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("batch_logs"),
        help="Directory for job logs",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        help="Summary JSON path (default: <log-dir>/summary.json)",
    )
    parser.add_argument(
        "--max-jobs",
        type=_positive_int,
        default=DEFAULT_MAX_JOBS,
        help="Maximum concurrent jobs",
    )
    parser.add_argument(
        "--memory-budget-gb",
        type=_positive_float,
        default=DEFAULT_MEMORY_BUDGET_GB,
        help="Maximum sum of running job memory reservations, in GiB",
    )
    parser.add_argument(
        "--default-memory-gb",
        type=_positive_float,
        default=DEFAULT_MEMORY_GB,
        help="Default per-job expected peak memory, in GiB",
    )
    parser.add_argument(
        "--reserve-gb",
        type=_positive_float,
        default=DEFAULT_RESERVE_GB,
        help="MemAvailable remaining after running reservations, in GiB",
    )
    parser.add_argument(
        "--cpus-per-job",
        type=_positive_int,
        default=DEFAULT_CPUS_PER_JOB,
        help="Value for OMNIRETARGETING_CPUS; scales geometry/evaluation worker pools",
    )
    parser.add_argument(
        "--largest-first", action="store_true", help="Start larger memory jobs first"
    )
    parser.add_argument(
        "--no-systemd",
        action="store_false",
        dest="systemd",
        help="Do not use systemd-run scopes",
    )
    parser.add_argument(
        "--no-time",
        action="store_false",
        dest="time",
        help="Do not record /usr/bin/time -v output",
    )
    parser.add_argument(
        "--poll-interval",
        type=_positive_float,
        default=0.2,
        help="Seconds between completion and memory-admission checks",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)
    summary_path = args.summary or args.log_dir / "summary.json"
    try:
        summary = run_batch(
            args.manifest,
            args.log_dir,
            summary_path,
            max_jobs=args.max_jobs,
            memory_budget_gb=args.memory_budget_gb,
            default_memory_gb=args.default_memory_gb,
            reserve_gb=args.reserve_gb,
            cpus_per_job=args.cpus_per_job,
            largest_first=args.largest_first,
            use_systemd=args.systemd,
            use_time=args.time,
            poll_interval=args.poll_interval,
        )
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return _summary_exit_code(summary)


if __name__ == "__main__":
    sys.exit(main())
