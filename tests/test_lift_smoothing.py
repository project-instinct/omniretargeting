"""The lift that resolves foot penetration, as a constrained smoothing problem.

Three things were tried before this one and each fails a property below. A raw
per-frame lift satisfies the requirement but injects the requirement's own noise
into the base. A low-pass of the requirement is smooth but can dip below it,
which puts the sole back through the surface. Holding each contact interval at
its maximum never dips, but carries a lift through frames that never needed one,
which moves the body off the source for no reason.

The formulation here is the one that satisfies all three at once: minimise
second-difference energy plus a pull toward zero, subject to l >= required.
"""

import numpy as np
import pytest

from omniretargeting.core import OmniRetargeter

lift = OmniRetargeter._minimal_smooth_lift


class TestFeasibility:
    """The property a low-pass filter cannot offer."""

    def test_the_result_is_never_below_the_requirement(self):
        rng = np.random.default_rng(0)
        required = np.maximum(rng.normal(0.01, 0.02, 200), 0.0)
        result = lift(required, window=8.0)
        assert np.all(result >= required - 1e-12)

    def test_an_isolated_spike_is_still_covered(self):
        """The case that punishes any smoother applied without a constraint."""
        required = np.zeros(60)
        required[30] = 0.05
        result = lift(required, window=10.0)
        assert result[30] >= 0.05 - 1e-12

    def test_a_moving_average_of_the_same_spike_would_not_be(self):
        required = np.zeros(60)
        required[30] = 0.05
        averaged = np.convolve(required, np.ones(9) / 9.0, mode="same")
        assert averaged[30] < 0.05 / 2, "the comparison this test exists to make"

    def test_zero_requirement_produces_zero_lift(self):
        np.testing.assert_array_equal(lift(np.zeros(50), window=8.0), np.zeros(50))

    def test_a_requirement_that_is_already_smooth_is_left_alone(self):
        t = np.linspace(0, 1, 120)
        required = 0.02 * (1.0 - np.cos(2 * np.pi * t))
        result = lift(required, window=6.0)
        assert np.max(result - required) < 0.002


class TestMinimality:
    """The property that keeps the body on the source trajectory."""

    def test_frames_far_from_any_requirement_are_not_lifted(self):
        required = np.zeros(120)
        required[55:65] = 0.03
        result = lift(required, window=6.0)
        assert result[0] < 1e-3 and result[-1] < 1e-3

    def test_it_lifts_less_than_holding_the_interval_maximum(self):
        """Directly against the alternative it replaced."""
        required = np.zeros(120)
        required[40:80] = np.linspace(0.0, 0.04, 40)
        held = np.zeros(120)
        held[40:80] = required[40:80].max()
        result = lift(required, window=6.0)
        assert result.sum() < held.sum()

    def test_it_never_lifts_above_the_largest_requirement(self):
        """Feasibility forces the lift up to the requirement; nothing should
        force it past the highest point the motion ever asked for."""
        rng = np.random.default_rng(1)
        for window in (4.0, 8.0, 16.0):
            required = np.maximum(rng.normal(0.005, 0.01, 300), 0.0)
            assert lift(required, window).max() <= required.max() + 1e-6


class TestSmoothness:
    def test_it_removes_most_of_the_requirement_jitter(self):
        rng = np.random.default_rng(2)
        required = 0.02 + 0.01 * rng.normal(size=200)
        required = np.maximum(required, 0.0)
        raw_jitter = np.sqrt(np.mean(np.diff(required, 2) ** 2))
        result = lift(required, window=10.0)
        assert np.sqrt(np.mean(np.diff(result, 2) ** 2)) < raw_jitter / 5

    def test_a_wider_window_is_smoother(self):
        rng = np.random.default_rng(3)
        required = np.maximum(0.02 + 0.01 * rng.normal(size=200), 0.0)
        narrow = np.sqrt(np.mean(np.diff(lift(required, 4.0), 2) ** 2))
        wide = np.sqrt(np.mean(np.diff(lift(required, 16.0), 2) ** 2))
        assert wide < narrow


class TestDegenerateInput:
    def test_short_sequences_pass_through(self):
        for n in (0, 1, 2):
            required = np.full(n, 0.01)
            np.testing.assert_array_equal(lift(required, 8.0), required)

    def test_negative_requirements_are_clamped_to_zero(self):
        """A clearance above target asks for no lift, not a drop."""
        required = np.array([-0.05, -0.01, 0.0, 0.02, -0.03] * 10, dtype=float)
        assert np.all(lift(required, 6.0) >= 0.0)

    def test_non_finite_requirements_do_not_propagate(self):
        required = np.zeros(60)
        required[10] = np.nan
        required[20] = np.inf
        required[30] = 0.02
        assert np.all(np.isfinite(lift(required, 6.0)))


class TestFramerateInvariance:
    """A frame-count window would fail this; the caller derives it from time."""

    def test_the_same_motion_at_two_rates_gets_the_same_lift_shape(self):
        # Several seconds, so the comparison is of the response and not of how
        # two different array lengths handle their own ends.
        duration = 4.0
        t_slow = np.linspace(0, duration, int(duration * 100))
        required_slow = np.maximum(0.03 * np.sin(2 * np.pi * t_slow), 0.0)
        t_fast = np.linspace(0, duration, int(duration * 200))
        required_fast = np.maximum(0.03 * np.sin(2 * np.pi * t_fast), 0.0)

        tau = 0.25
        slow = lift(required_slow, tau * 100.0)
        fast = lift(required_fast, tau * 200.0)

        resampled = np.interp(t_slow, t_fast, fast)
        assert np.max(np.abs(slow - resampled)) < 0.002
