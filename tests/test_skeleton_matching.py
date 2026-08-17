"""Bringing the source skeleton to robot proportions about its contacts.

The interaction mesh tracks absolute source joint positions. When the scene is
left at its captured scale and the robot is shorter than the subject, that target
is unreachable: the pelvis cannot be at human height while the feet are on the
real floor. What happened instead was two regimes -- the contact task won on
contact frames and pulled the base down, the interaction mesh won between them
and pulled it back up.

Rescaling the source skeleton to robot proportions removes the conflict instead
of balancing it, provided the rescaling is pinned so the contacting foot lands on
the scene rather than wherever the source had it.
"""

import numpy as np
import pytest

from omniretargeting.core import OmniRetargeter


class TestSkeletonScaleDerivation:
    """The scale is measured from the segment that has to span the gap.

    Which segment that is turned out to matter by several centimetres. The
    contact task pins the sole, so the sole is the point whose reach decides
    whether the robot can stand where the source stood. Measuring to the ankle
    joint instead leaves the human's ankle-to-sole offset, about 8 cm, against
    this robot's 3.5 cm, unaccounted for; the leg then cannot span the gap once
    the sole is planted and the base drops to make up the difference on exactly
    the frames where a contact is active.
    """

    NAMES = ["Pelvis", "L_Ankle", "R_Ankle"]

    def _positions(self, leg_length, n=2):
        p = np.zeros((n, 3, 3))
        p[:, 0] = [0.0, 0.0, leg_length]
        p[:, 1] = [0.0, 0.10, 0.0]
        p[:, 2] = [0.0, -0.10, 0.0]
        return p

    def test_the_sole_is_measured_to_when_it_is_available(self):
        pos = self._positions(0.90)
        # Soles 8 cm below the ankles, as on a human.
        soles = {
            "left": np.tile([0.0, 0.10, -0.08], (2, 1)),
            "right": np.tile([0.0, -0.10, -0.08], (2, 1)),
        }
        to_sole = OmniRetargeter._source_leg_length(pos, self.NAMES, soles)
        to_ankle = OmniRetargeter._source_leg_length(pos, self.NAMES, None)
        assert to_sole == pytest.approx(np.hypot(0.98, 0.10), abs=1e-9)
        assert to_sole - to_ankle == pytest.approx(0.08, abs=1e-3)

    def test_the_ankle_is_the_fallback_when_no_soles_are_known(self):
        pos = self._positions(0.90)
        measured = OmniRetargeter._source_leg_length(pos, self.NAMES, None)
        assert measured == pytest.approx(np.hypot(0.90, 0.10), abs=1e-9)

    def test_the_span_is_taken_at_stance_not_at_the_median(self):
        """Stance is where the leg is most extended, and stance is when it is pinned.

        A median over a clip full of bent-leg swing frames calibrates on the
        wrong part of the motion and leaves the stance demand beyond reach.
        """
        # Ten frames: eight crouched, two extended. The median sees only the
        # crouch; the extended frames are the ones that carry a contact.
        pos = np.zeros((10, 3, 3))
        pos[:, 0] = [0.0, 0.0, 0.70]
        pos[-2:, 0] = [0.0, 0.0, 0.95]
        pos[:, 1] = [0.0, 0.0, 0.0]
        pos[:, 2] = [0.0, 0.0, 0.0]

        measured = OmniRetargeter._source_leg_length(pos, self.NAMES, None)
        assert measured > 0.70, "a median would report the crouch and miss stance"
        assert measured == pytest.approx(
            np.percentile(np.concatenate([pos[:, 0, 2]] * 2), OmniRetargeter._LEG_SPAN_PERCENTILE),
            abs=1e-9,
        )

    def test_missing_pelvis_reports_zero_rather_than_guessing(self):
        pos = self._positions(0.90)
        assert OmniRetargeter._source_leg_length(pos, ["A", "B", "C"], None) == 0.0

    def test_missing_ankles_and_soles_report_zero(self):
        pos = self._positions(0.90)
        assert OmniRetargeter._source_leg_length(pos, ["Pelvis", "B", "C"], None) == 0.0

    def test_height_is_not_used_as_a_proxy(self):
        """Leg length, not height, decides whether the feet reach the ground.

        Two robots of equal height can have very different leg-to-torso
        proportions, and it is the leg that has to span the gap.
        """
        short_legs = OmniRetargeter._source_leg_length(self._positions(0.70), self.NAMES, None)
        long_legs = OmniRetargeter._source_leg_length(self._positions(0.95), self.NAMES, None)
        assert long_legs > short_legs


class TestContactAnchorSeries:
    """What the rescaling is pinned to, frame by frame."""

    NAMES = ["Pelvis", "L_Ankle", "R_Ankle"]

    @staticmethod
    def _positions(n=10):
        p = np.zeros((n, 3, 3))
        p[:, 0] = [0.0, 0.0, 0.9]
        p[:, 1] = [0.0, 0.1, 0.05]
        p[:, 2] = [0.0, -0.1, 0.05]
        return p

    def test_contact_frames_use_their_own_anchor(self):
        pos = self._positions()
        anchors = {"left": np.full((10, 3), np.nan)}
        anchors["left"][3:6] = [1.0, 2.0, 0.0]
        centres = {"left": np.tile([1.0, 2.0, 0.05], (10, 1))}

        surface, source = OmniRetargeter._contact_anchor_series(pos, self.NAMES, anchors, centres)
        np.testing.assert_allclose(surface[4], [1.0, 2.0, 0.0])
        np.testing.assert_allclose(source[4], [1.0, 2.0, 0.05])

    def test_gaps_hold_the_nearest_known_anchor(self):
        """Carrying the anchor across a gap is what keeps the mapping continuous.

        Recentring on the pelvis between contacts would reintroduce the pull
        toward human height that this mechanism exists to remove.
        """
        pos = self._positions()
        anchors = {"left": np.full((10, 3), np.nan)}
        anchors["left"][2:4] = [0.0, 0.0, 0.0]
        anchors["left"][7:9] = [1.0, 0.0, 0.0]
        centres = {"left": np.tile([0.0, 0.0, 0.05], (10, 1))}

        surface, _ = OmniRetargeter._contact_anchor_series(pos, self.NAMES, anchors, centres)
        assert np.isfinite(surface).all(), "no frame may be left without an anchor"
        # Before the first contact and after the last, the nearest one is held.
        np.testing.assert_allclose(surface[0], surface[2])
        np.testing.assert_allclose(surface[9], surface[8])
        # Across the gap it moves monotonically rather than jumping.
        span = surface[2:9, 0]
        assert np.all(np.diff(span) >= -1e-12)

    def test_no_contact_at_all_falls_back_to_the_source_feet(self):
        pos = self._positions()
        anchors = {"left": np.full((10, 3), np.nan)}
        surface, source = OmniRetargeter._contact_anchor_series(pos, self.NAMES, anchors, None)
        expected = pos[:, [1, 2], :].mean(axis=1)
        np.testing.assert_allclose(surface, expected)
        np.testing.assert_allclose(source, expected)


class TestRescalingGeometry:
    """The property the whole mechanism exists for."""

    @staticmethod
    def _rescale(positions, surface, source_point, scale):
        return surface[:, None, :] + scale * (positions - source_point[:, None, :])

    def test_the_contacting_point_lands_on_the_surface_anchor(self):
        pos = np.zeros((1, 2, 3))
        pos[0, 0] = [0.0, 0.0, 0.90]   # pelvis
        pos[0, 1] = [0.0, 0.0, 0.05]   # foot, floating 5 cm above its own scan
        surface = np.array([[0.0, 0.0, 0.0]])
        source_point = np.array([[0.0, 0.0, 0.05]])

        out = self._rescale(pos, surface, source_point, 0.8724)
        np.testing.assert_allclose(out[0, 1], [0.0, 0.0, 0.0], atol=1e-12)

    def test_the_pelvis_descends_by_the_leg_length_difference(self):
        pos = np.zeros((1, 2, 3))
        pos[0, 0] = [0.0, 0.0, 0.90]
        pos[0, 1] = [0.0, 0.0, 0.0]
        surface = np.array([[0.0, 0.0, 0.0]])
        source_point = np.array([[0.0, 0.0, 0.0]])

        scale = 0.8724
        out = self._rescale(pos, surface, source_point, scale)
        assert out[0, 0, 2] == pytest.approx(0.90 * scale)
        assert 0.90 - out[0, 0, 2] == pytest.approx(0.90 * (1.0 - scale))

    def test_a_unit_scale_leaves_the_skeleton_alone_apart_from_the_pin(self):
        rng = np.random.default_rng(1)
        pos = rng.normal(size=(4, 5, 3))
        surface = np.zeros((4, 3))
        source_point = np.zeros((4, 3))
        np.testing.assert_allclose(self._rescale(pos, surface, source_point, 1.0), pos)

    def test_rescaling_preserves_shape_up_to_the_scale(self):
        """Segment lengths must all change by the same factor, nothing else."""
        rng = np.random.default_rng(2)
        pos = rng.normal(size=(3, 6, 3))
        surface = rng.normal(size=(3, 3))
        source_point = rng.normal(size=(3, 3))
        scale = 0.7

        out = self._rescale(pos, surface, source_point, scale)
        before = np.linalg.norm(pos[:, 1:] - pos[:, :-1], axis=-1)
        after = np.linalg.norm(out[:, 1:] - out[:, :-1], axis=-1)
        np.testing.assert_allclose(after, scale * before, rtol=1e-12)
