"""The batch runner schedules resources without changing job commands."""

import json
import sys
from datetime import datetime
from pathlib import Path

import scripts.batch_retarget as batch


def _write_manifest(path: Path, jobs: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(job) for job in jobs) + "\n", encoding="utf-8")
    return path


def _sleep_job(name: str, seconds: float, memory_gb: float) -> dict:
    # A real (but tiny) Python process is intentional: it also exercises logging,
    # environment propagation, and /usr/bin/time peak-RSS extraction.
    command = [
        sys.executable,
        "-c",
        "import sys, time; print('started', flush=True); time.sleep(float(sys.argv[1])); print('finished', flush=True)",
        str(seconds),
    ]
    return {"name": name, "command": command, "memory_gb": memory_gb}


def _run(tmp_path: Path, jobs: list[dict], **overrides):
    manifest = _write_manifest(tmp_path / "jobs.jsonl", jobs)
    log_dir = tmp_path / "logs"
    summary_path = log_dir / "summary.json"
    options = {
        "max_jobs": 4,
        "memory_budget_gb": 100.0,
        "default_memory_gb": 1.0,
        "reserve_gb": 10.0,
        "cpus_per_job": 2,
        "use_systemd": False,
        "poll_interval": 0.01,
    }
    options.update(overrides)
    summary = batch.run_batch(manifest, log_dir, summary_path, **options)
    saved = json.loads(summary_path.read_text(encoding="utf-8"))
    return summary, saved


def _maximum_overlap(results: list[dict]) -> int:
    events: list[tuple[datetime, int]] = []
    for result in results:
        started = datetime.fromisoformat(result["started_at"])
        ended = datetime.fromisoformat(result["ended_at"])
        events.append((started, 1))
        events.append((ended, -1))
    running = maximum = 0
    for _, delta in sorted(events):
        running += delta
        maximum = max(maximum, running)
    return maximum


def test_concurrency_is_limited_by_max_jobs(tmp_path, monkeypatch):
    monkeypatch.setattr(batch, "read_mem_available_gb", lambda: 100.0)
    jobs = [_sleep_job(f"clip_{index}", 0.08, 1.0) for index in range(4)]
    summary, saved = _run(tmp_path, jobs, max_jobs=2)

    assert summary["makespan_seconds"] >= 0.16
    assert _maximum_overlap(summary["jobs"]) <= 2
    assert saved["jobs"] == summary["jobs"]
    assert saved["failures"] == []
    assert all(
        job["exit_code"] == 0 and job["peak_rss_gb"] > 0 for job in summary["jobs"]
    )


def test_running_memory_reservations_are_admitted_against_the_budget(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(batch, "read_mem_available_gb", lambda: 100.0)
    jobs = [_sleep_job("large_a", 0.08, 20.0), _sleep_job("large_b", 0.08, 20.0)]
    summary, _ = _run(tmp_path, jobs, memory_budget_gb=30.0)

    assert _maximum_overlap(summary["jobs"]) == 1
    assert summary["makespan_seconds"] >= 0.16
    assert [job["status"] for job in summary["jobs"]] == ["success", "success"]


def test_mem_available_reserve_blocks_a_second_job(tmp_path, monkeypatch):
    monkeypatch.setattr(batch, "read_mem_available_gb", lambda: 19.0)
    jobs = [_sleep_job("job_a", 0.08, 10.0), _sleep_job("job_b", 0.08, 10.0)]
    summary, _ = _run(tmp_path, jobs, memory_budget_gb=100.0, reserve_gb=10.0)

    assert _maximum_overlap(summary["jobs"]) == 1
    assert all(job["status"] == "success" for job in summary["jobs"])


def test_largest_first_changes_launch_order(tmp_path, monkeypatch):
    monkeypatch.setattr(batch, "read_mem_available_gb", lambda: 100.0)
    jobs = [_sleep_job("small", 0.01, 5.0), _sleep_job("large", 0.01, 20.0)]
    summary, _ = _run(tmp_path, jobs, max_jobs=1, largest_first=True)

    started = {job["name"]: job["started_at"] for job in summary["jobs"]}
    assert started["large"] < started["small"]


def test_manifest_order_is_preserved_when_the_head_job_is_blocked(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(batch, "read_mem_available_gb", lambda: 100.0)
    jobs = [
        _sleep_job("first", 0.08, 8.0),
        _sleep_job("blocked_head", 0.01, 4.0),
        _sleep_job("later", 0.01, 2.0),
    ]
    summary, _ = _run(tmp_path, jobs, memory_budget_gb=10.0)
    started = {job["name"]: job["started_at"] for job in summary["jobs"]}

    assert started["first"] < started["blocked_head"] <= started["later"]


def test_failed_job_is_recorded_in_log_and_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(batch, "read_mem_available_gb", lambda: 100.0)
    command = [
        sys.executable,
        "-c",
        "import os, sys; print('stdout marker'); print('stderr marker', file=sys.stderr); sys.exit(7)",
    ]
    summary, _ = _run(
        tmp_path, [{"name": "fails", "command": command, "memory_gb": 2.0}]
    )
    job = summary["jobs"][0]
    log = Path(job["log"]).read_text(encoding="utf-8")

    assert job["status"] == "failed"
    assert job["exit_code"] == 7
    assert summary["failures"] == ["fails"]
    assert "stdout marker" in log
    assert "stderr marker" in log
    assert "exit_code=7" in log


def test_keyboard_interrupt_terminates_and_skips_jobs(tmp_path, monkeypatch):
    monkeypatch.setattr(batch, "read_mem_available_gb", lambda: 100.0)

    class InterruptingTime:
        monotonic = staticmethod(batch.time.monotonic)

        @staticmethod
        def sleep(_seconds):
            raise KeyboardInterrupt

    monkeypatch.setattr(
        batch,
        "time",
        InterruptingTime,
    )
    jobs = [_sleep_job("active", 30.0, 1.0), _sleep_job("pending", 30.0, 1.0)]
    summary, _ = _run(tmp_path, jobs, max_jobs=1)
    statuses = {job["name"]: job["status"] for job in summary["jobs"]}

    assert summary["interrupted"] is True
    assert statuses == {"active": "interrupted", "pending": "skipped"}
    assert all(
        job["exit_code"] != 0 for job in summary["jobs"] if job["name"] == "active"
    )


def test_job_environment_restricts_thread_pools(tmp_path, monkeypatch):
    monkeypatch.setattr(batch, "read_mem_available_gb", lambda: 100.0)
    command = [
        sys.executable,
        "-c",
        (
            "import json, os; print(json.dumps({name: os.environ[name] for name in "
            "('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'OMNIRETARGETING_CPUS', 'BATCH_TEST_MARKER')}))"
        ),
    ]
    summary, _ = _run(
        tmp_path,
        [
            {
                "name": "env",
                "command": command,
                "memory_gb": 1.0,
                "env": {"BATCH_TEST_MARKER": "yes"},
            }
        ],
        cpus_per_job=3,
    )
    log = Path(summary["jobs"][0]["log"]).read_text(encoding="utf-8")
    reported = json.loads(
        next(line for line in log.splitlines() if line.startswith("{"))
    )

    assert reported == {
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "OMNIRETARGETING_CPUS": "3",
        "BATCH_TEST_MARKER": "yes",
    }
