"""Availability of fork workers for the production window schedulers."""
import os
import threading


def fork_status():
    """Unknown live Python/native threading falls back to serial execution."""
    if not hasattr(os, 'fork'):
        return False, 'fork_unavailable'
    if threading.active_count() != 1:
        return False, 'live_python_threads'
    try:
        from threadpoolctl import threadpool_info
        if any(item.get('num_threads', 1) != 1 for item in threadpool_info()):
            return False, 'native_thread_count_not_one'
    except ImportError:
        return False, 'native_thread_state_unknown'
    return True, None
