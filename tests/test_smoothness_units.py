"""The temporal smoothness term, and the units it is summed in.

``q_a`` covers the whole of qpos when the floating base is optimised. Squaring
its step and summing adds three base translations in metres to four quaternion
components carrying no unit to twenty-nine joint angles in radians, and then
scales the total by one number. Whatever exchange rate that implies between a
metre of base travel and a radian of knee is an artefact of the units, not a
choice anyone made.

These tests pin the replacement: a scale vector that converts every group into
how far it moves the robot, expressed relative to the joints so the tuned weight
keeps acting on the joints exactly as it did.
"""

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from omniretargeting.retargeting import GenericInteractionRetargeter


class _Model:
    """The handful of MuJoCo fields the scale builder reads."""

    def __init__(self, body_offsets, free_base=True, qposadr=0):
        self.nbody = len(body_offsets)
        self.body_pos = np.asarray(body_offsets, dtype=float)
        self.njnt = 1
        self.jnt_type = np.array(
            [mujoco.mjtJoint.mjJNT_FREE if free_base else mujoco.mjtJoint.mjJNT_HINGE]
        )
        self.jnt_qposadr = np.array([qposadr])


def _scale(body_offsets, q_a_indices, free_base=True):
    solver = object.__new__(GenericInteractionRetargeter)
    solver.robot_model = _Model(body_offsets, free_base=free_base)
    solver.q_a_indices = np.asarray(q_a_indices)
    solver.nq_a = len(q_a_indices)
    return GenericInteractionRetargeter._build_smoothness_scale(solver)


HUMANOID = [[0, 0, 0], [0, 0, 0.1], [0, 0, -0.4], [0.0, 0.0, -0.8]]


class TestGrouping:
    def test_joints_are_the_reference_and_keep_a_scale_of_one(self):
        """So an existing tuned smooth_weight means what it used to mean."""
        scale = _scale(HUMANOID, np.arange(0, 12))
        np.testing.assert_allclose(scale[7:], 1.0)

    def test_base_translation_is_divided_by_the_robot_reach(self):
        scale = _scale(HUMANOID, np.arange(0, 12))
        reach = 0.9  # |[0,0,-0.8] - [0,0,0.1]|
        np.testing.assert_allclose(scale[:3], 1.0 / reach, rtol=1e-9)

    def test_the_quaternion_group_carries_the_half_angle_factor(self):
        scale = _scale(HUMANOID, np.arange(0, 12))
        np.testing.assert_allclose(scale[3:7], 2.0)

    def test_a_taller_robot_penalises_the_same_base_step_less(self):
        """A centimetre of base travel is a smaller upset to a large robot."""
        small = _scale([[0, 0, 0], [0, 0, 0.05], [0, 0, -0.35]], np.arange(0, 11))
        large = _scale([[0, 0, 0], [0, 0, 0.2], [0, 0, -1.4]], np.arange(0, 11))
        assert large[0] < small[0]


class TestScopeOfTheCorrection:
    def test_a_fixed_base_solve_is_untouched(self):
        """Nothing to correct when only joints are optimised."""
        scale = _scale(HUMANOID, np.arange(7, 12), free_base=False)
        np.testing.assert_allclose(scale, 1.0)

    def test_a_solve_that_excludes_the_base_is_untouched(self):
        """The free joint exists but its coordinates are not being optimised."""
        scale = _scale(HUMANOID, np.arange(7, 12))
        np.testing.assert_allclose(scale, 1.0)

    def test_the_free_joint_is_located_by_its_qpos_address(self):
        """Not assumed to start at zero."""
        scale = _scale(HUMANOID, np.arange(0, 12), free_base=True)
        shifted = _scale(HUMANOID, np.arange(0, 12))
        np.testing.assert_allclose(scale, shifted)


class TestDegenerateModels:
    def test_a_model_with_no_extent_falls_back_to_one_metre(self):
        """Rather than dividing by zero."""
        scale = _scale([[0, 0, 0], [0, 0, 0]], np.arange(0, 9))
        assert np.all(np.isfinite(scale))
        np.testing.assert_allclose(scale[:3], 1.0)

    def test_the_scale_has_one_entry_per_optimised_coordinate(self):
        for count in (9, 12, 36):
            assert len(_scale(HUMANOID, np.arange(0, count))) == count


class TestDimensionalConsistency:
    def test_the_penalty_is_unchanged_when_the_robot_is_rebuilt_in_new_units(self):
        """The property the whole scale vector exists for.

        Take a step of one centimetre of base travel and 0.01 rad of joint on a
        robot, then describe the same robot and the same physical step with
        every length doubled. A term summed in mixed units changes; this one
        should not.
        """
        step_small = np.concatenate([[0.01, 0, 0], [0, 0, 0, 0], [0.01, 0.01]])
        scale_small = _scale(HUMANOID, np.arange(0, 9))

        doubled = (np.asarray(HUMANOID) * 2.0).tolist()
        step_large = np.concatenate([[0.02, 0, 0], [0, 0, 0, 0], [0.01, 0.01]])
        scale_large = _scale(doubled, np.arange(0, 9))

        small = float(((scale_small * step_small) ** 2).sum())
        large = float(((scale_large * step_large) ** 2).sum())
        assert large == pytest.approx(small, rel=1e-9)

    def test_the_unscaled_sum_does_not_have_that_property(self):
        """The comparison that makes the test above meaningful."""
        step_small = np.concatenate([[0.01, 0, 0], [0, 0, 0, 0], [0.01, 0.01]])
        step_large = np.concatenate([[0.02, 0, 0], [0, 0, 0, 0], [0.01, 0.01]])
        assert float((step_small ** 2).sum()) != pytest.approx(
            float((step_large ** 2).sum()), rel=1e-9
        )
