"""Contact detection in units that survive a change of scale.

The gates this replaces were absolute. A 5 cm proximity threshold is 23 percent
of one subject's foot length and 31 percent of the same foot in a scene scaled
to 0.74, and that difference alone once moved detection from 27 and 40 contact
frames to 4 and 15 on the same motion. A speed gate in metres per second has the
same problem across subject sizes.

These tests pin the properties that make the replacement transferable: the gates
are fractions of the source's own geometry, the source's systematic float above
its own scan is estimated rather than absorbed by a loose threshold, and the
state latches rather than flickering on the boundary.
"""

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from omniretargeting.scene_contact import (
    ContactSpec,
    GRAVITY,
    contact_runs,
    estimate_float_bias,
    hysteretic_state,
    infer_contact_targets,
)


def _floor(extent=6.0):
    plane = trimesh.creation.box(extents=(extent, extent, 0.2))
    plane.apply_translation([0.0, 0.0, -0.1])
    return plane


def _step_motion(n=40, plant=(10, 30), height=0.30, float_bias=0.0):
    """A foot that lowers, rests for a while, then lifts again."""
    z = np.full(n, height)
    z[plant[0] : plant[1]] = 0.0
    ramp = 6
    z[plant[0] - ramp : plant[0]] = np.linspace(height, 0.0, ramp)
    z[plant[1] : plant[1] + ramp] = np.linspace(0.0, height, ramp)
    pts = np.zeros((n, 3))
    pts[:, 2] = z + float_bias
    return pts


class TestFloatBiasEstimation:
    def test_a_constant_float_is_recovered(self):
        distance = np.concatenate([np.full(20, 0.03), np.full(20, 0.4)])
        slow = np.concatenate([np.ones(20, bool), np.zeros(20, bool)])
        assert estimate_float_bias(distance, slow, 10.0) == pytest.approx(0.03, abs=1e-9)

    def test_airborne_frames_do_not_inflate_it(self):
        """The bias is the offset when planted, not the average over the clip."""
        distance = np.concatenate([np.full(10, 0.03), np.full(30, 0.5)])
        slow = np.concatenate([np.ones(10, bool), np.zeros(30, bool)])
        biased_by_mean = distance.mean()
        assert estimate_float_bias(distance, slow, 10.0) < biased_by_mean / 4

    def test_it_falls_back_to_all_frames_when_none_are_slow(self):
        distance = np.linspace(0.01, 0.5, 50)
        assert estimate_float_bias(distance, np.zeros(50, bool), 10.0) > 0.0

    def test_it_never_reports_a_negative_offset(self):
        assert estimate_float_bias(np.full(10, -0.05), np.ones(10, bool), 10.0) == 0.0


class TestHysteresis:
    def test_state_latches_until_the_leave_condition_fires(self):
        enter = np.array([0, 1, 0, 0, 0, 0], dtype=bool)
        leave = np.array([0, 0, 0, 0, 1, 0], dtype=bool)
        np.testing.assert_array_equal(
            hysteretic_state(enter, leave), [False, True, True, True, False, False]
        )

    def test_a_frame_on_the_boundary_does_not_flicker(self):
        """Without hysteresis an alternating marginal frame toggles every step."""
        marginal = np.array([1, 0, 1, 0, 1, 0], dtype=bool)
        never_leaves = np.zeros(6, dtype=bool)
        state = hysteretic_state(marginal, never_leaves)
        assert state[1:].all(), "contact should hold through the marginal gaps"

    def test_leaving_takes_precedence_while_active(self):
        enter = np.ones(4, dtype=bool)
        leave = np.array([0, 0, 1, 0], dtype=bool)
        np.testing.assert_array_equal(hysteretic_state(enter, leave), [True, True, False, True])


class TestScaleInvariance:
    """The property the whole reparameterisation exists for."""

    @staticmethod
    def _detect(scale):
        pts = _step_motion(float_bias=0.02) * scale
        floor = _floor()
        floor.apply_scale(scale)
        targets, _, weights, stats = infer_contact_targets(
            {"foot": pts},
            floor,
            framerate=30.0,
            spec=ContactSpec(),
            effector_lengths={"foot": 0.22 * scale},
            leg_length=0.95 * scale,
        )
        return np.isfinite(targets["foot"][:, 0]), stats["foot"], weights["foot"]

    def test_the_same_frames_are_detected_at_any_scale(self):
        base, _, _ = self._detect(1.0)
        for scale in (0.5, 0.7407, 1.5):
            other, _, _ = self._detect(scale)
            np.testing.assert_array_equal(
                other, base, err_msg=f"detection changed at scale {scale}"
            )

    def test_the_estimated_bias_scales_with_the_world(self):
        _, small, _ = self._detect(0.5)
        _, large, _ = self._detect(1.5)
        assert large["float_bias"] == pytest.approx(3.0 * small["float_bias"], rel=1e-6)

    def test_the_relative_part_of_the_gate_scales_with_the_effector(self):
        """The floor deliberately does not scale.

        It stands for reconstruction noise in the scene, which is a property of
        the scanner rather than of the subject, so a small subject in a coarse
        scan should not get a proportionally tighter gate. Only the part
        expressed as a fraction of the effector scales.
        """
        floor = ContactSpec().gap_floor
        _, small, _ = self._detect(0.5)
        _, large, _ = self._detect(1.5)
        assert large["enter_gap"] - floor == pytest.approx(
            3.0 * (small["enter_gap"] - floor), rel=1e-6
        )
        # And the floor is genuinely still there at both scales.
        assert small["enter_gap"] > floor and large["enter_gap"] > floor


class TestSpeedNormalisation:
    def test_resampling_the_same_motion_does_not_change_detection(self):
        """A gate in body-lengths-per-sample would move here; this one must not."""
        floor = _floor()
        slow = _step_motion(n=40)
        # Same continuous motion, sampled twice as densely.
        dense = np.repeat(slow, 2, axis=0)
        dense[:, 2] = np.interp(
            np.linspace(0, len(slow) - 1, len(slow) * 2), np.arange(len(slow)), slow[:, 2]
        )

        kwargs = dict(spec=ContactSpec(), effector_lengths={"foot": 0.22}, leg_length=0.95)
        a, _, _, _ = infer_contact_targets({"foot": slow}, floor, 30.0, **kwargs)
        b, _, _, _ = infer_contact_targets({"foot": dense}, floor, 60.0, **kwargs)

        frac_a = np.isfinite(a["foot"][:, 0]).mean()
        frac_b = np.isfinite(b["foot"][:, 0]).mean()
        assert frac_b == pytest.approx(frac_a, abs=0.05)

    def test_the_speed_gate_is_skipped_when_the_scale_is_unknown(self):
        """Better to decide on proximity alone than to apply a gate in units
        that do not transfer."""
        floor = _floor()
        pts = _step_motion()
        targets, _, _, _ = infer_contact_targets(
            {"foot": pts}, floor, 30.0, ContactSpec(), effector_lengths={"foot": 0.22}
        )
        assert np.isfinite(targets["foot"][:, 0]).any()


class TestConfidence:
    def test_confidence_is_reported_and_bounded(self):
        floor = _floor()
        _, _, weights, _ = infer_contact_targets(
            {"foot": _step_motion()},
            floor,
            30.0,
            ContactSpec(),
            effector_lengths={"foot": 0.22},
            leg_length=0.95,
        )
        w = weights["foot"]
        assert w.min() >= 0.0 and w.max() <= 1.0
        assert (w > 0).any()

    def test_a_planted_foot_reaches_full_confidence(self):
        floor = _floor()
        _, _, weights, _ = infer_contact_targets(
            {"foot": _step_motion()},
            floor,
            30.0,
            ContactSpec(),
            effector_lengths={"foot": 0.22},
            leg_length=0.95,
        )
        assert weights["foot"].max() == pytest.approx(1.0, abs=1e-6)


class TestSpecValidation:
    def test_a_leave_gate_tighter_than_the_enter_gate_is_rejected(self):
        with pytest.raises(ValueError, match="leave_gap_fraction"):
            ContactSpec(enter_gap_fraction=0.2, leave_gap_fraction=0.1)

    def test_a_leave_speed_below_the_enter_speed_is_rejected(self):
        with pytest.raises(ValueError, match="leave_speed_fraction"):
            ContactSpec(enter_speed_fraction=0.3, leave_speed_fraction=0.1)

    def test_a_bias_percentile_outside_the_lower_half_is_rejected(self):
        with pytest.raises(ValueError, match="bias_percentile"):
            ContactSpec(bias_percentile=80.0)


class TestRunExtraction:
    def test_short_runs_are_discarded(self):
        flags = np.array([1, 1, 0, 1, 1, 1, 1, 0], dtype=bool)
        assert contact_runs(flags, 3) == [(3, 7)]

    def test_a_run_reaching_the_end_is_closed(self):
        flags = np.array([0, 1, 1, 1], dtype=bool)
        assert contact_runs(flags, 3) == [(1, 4)]
