"""Cores this process may use for its own worker pools.

Parallel time windows share one machine: each window process receives an
explicit budget (``OMNIRETARGETING_CPUS``) so its pools do not oversubscribe
the cores. Scheduling only; worker counts never change computed values.
"""
import os


def available_cpus():
    value = os.environ.get("OMNIRETARGETING_CPUS")
    if value:
        return max(1, int(value))
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover - platforms without affinity
        return os.cpu_count() or 1
