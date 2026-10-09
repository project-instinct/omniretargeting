"""Small deadline contract for bounded fast-solver phases."""

from __future__ import annotations

import math
import time


class DeadlineExceeded(RuntimeError):
    """The configured production budget, including its export reserve, is gone."""


class Deadline:
    """Track a monotonic production budget and the reserve kept for exports."""

    def __init__(self, budget_s, *, started=None, reserve_s=None,
                 clock=time.monotonic):
        budget_s = float(budget_s)
        if not math.isfinite(budget_s) and budget_s != math.inf:
            raise ValueError("budget_s must be finite or +inf")
        if budget_s < 0:
            raise ValueError("budget_s must be nonnegative")
        if reserve_s is None:
            reserve_s = 0.0 if math.isinf(budget_s) else max(2.0, 0.05 * budget_s)
        reserve_s = float(reserve_s)
        if not math.isfinite(reserve_s) or reserve_s < 0:
            raise ValueError("reserve_s must be finite and nonnegative")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self.clock = clock
        self.budget_s = budget_s
        self.reserve_s = reserve_s
        self.started = self.clock() if started is None else float(started)
        if not math.isfinite(self.started):
            raise ValueError("started must be finite")
        self.checks = 0
        self.first_expired_label = None

    @classmethod
    def for_clip(cls, frames, *, started=None, seconds_per_frame=1.0,
                 framerate=None):
        """Create a per-saved-frame budget.

        ``framerate`` is accepted so callers can pass the clip metadata they
        already hold; the current production contract is explicitly per saved
        frame and therefore does not rescale by framerate.
        """
        frames = float(frames)
        seconds_per_frame = float(seconds_per_frame)
        if not math.isfinite(frames) or frames < 0:
            raise ValueError("frames must be finite and nonnegative")
        if not math.isfinite(seconds_per_frame) or seconds_per_frame <= 0:
            raise ValueError("seconds_per_frame must be finite and positive")
        if framerate is not None:
            framerate = float(framerate)
            if not math.isfinite(framerate) or framerate <= 0:
                raise ValueError("framerate must be finite and positive when supplied")
        return cls(frames * seconds_per_frame, started=started)

    @classmethod
    def unlimited(cls):
        """A non-expiring deadline for tests and non-fast backends."""
        return cls(math.inf, reserve_s=0.0)

    def elapsed(self):
        return float(self.clock() - self.started)

    def remaining(self):
        """Remaining solve time after the immutable export reserve."""
        return float(self.budget_s - self.reserve_s - self.elapsed())

    def expired(self):
        return self.remaining() <= 0

    def reserve_for_export(self):
        """Inspect remaining production time; never withhold the best motion at expiry."""
        remaining = self.budget_s - self.elapsed()
        return dict(remaining_s=float(remaining), reserved_s=self.reserve_s,
                    lightweight=bool(remaining <= self.reserve_s), exhausted=bool(remaining <= 0))

    def check(self, label):
        self.checks += 1
        if self.expired():
            if self.first_expired_label is None:
                self.first_expired_label = label
            raise DeadlineExceeded(str(label))

    def fraction_used(self):
        if math.isinf(self.budget_s):
            return 0.0
        if self.budget_s == 0:
            return math.inf if self.elapsed() > 0 else 0.0
        return float(self.elapsed() / self.budget_s)

    def describe(self):
        return {
            "budget_s": self.budget_s,
            "reserve_s": self.reserve_s,
            "elapsed_s": self.elapsed(),
            "remaining_s": self.remaining(),
            "expired": self.expired(),
            "first_expired_label": self.first_expired_label,
            "checks": self.checks,
        }
