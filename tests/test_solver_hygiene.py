"""Regression tests for SQP loop and sampling hygiene.

These cover three defects that made retargeting output non-reproducible or
quietly worse than the solver had actually computed:

* terrain surface points were drawn without a seed, so two runs of the same
  input could not be compared;
* the SQP loop broke out of the iteration before assigning the configuration
  that satisfied the convergence tolerance, returning the previous iterate;
* the base position tracking term reached the QP matrices but not the scalar
  objective used for the stopping decision.
"""

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from omniretargeting.retargeting import GenericInteractionRetargeter
from omniretargeting.utils import sample_points_on_mesh


def _box():
    return trimesh.creation.box(extents=(2.0, 2.0, 0.1))


class TestTerrainSampleSeeding:
    def test_same_seed_reproduces_the_same_points(self):
        mesh = _box()
        first = sample_points_on_mesh(mesh, 64, seed=7)
        second = sample_points_on_mesh(mesh, 64, seed=7)
        np.testing.assert_array_equal(first, second)

    def test_different_seeds_produce_different_points(self):
        mesh = _box()
        first = sample_points_on_mesh(mesh, 64, seed=7)
        second = sample_points_on_mesh(mesh, 64, seed=8)
        assert not np.array_equal(first, second)

    def test_unseeded_draw_still_works(self):
        mesh = _box()
        points = sample_points_on_mesh(mesh, 32)
        assert points.shape == (32, 3)


class TestConvergedStepIsAccepted:
    """`_optimize_configuration` must return the iterate it converged on."""

    @staticmethod
    def _run(costs):
        """Drive the loop with a scripted cost sequence.

        Each scripted step returns a configuration tagged with its own index so
        the returned value identifies which step was accepted.
        """
        retargeter = GenericInteractionRetargeter.__new__(GenericInteractionRetargeter)
        retargeter.bone_direction_enabled = False
        retargeter.bone_warm_init = False

        calls = {"n": 0}

        def fake_step(q, *args, **kwargs):
            idx = calls["n"]
            calls["n"] += 1
            return np.array([float(idx + 1)]), costs[idx]

        retargeter._single_optimization_step = fake_step

        result = retargeter._optimize_configuration(
            np.array([0.0]),
            None,
            None,
            None,
            None,
            max_iter=len(costs),
        )
        return result, calls["n"]

    def test_step_meeting_the_tolerance_is_not_discarded(self):
        # Third step moves the cost by less than the tolerance, so the loop
        # stops there.  Its configuration is still an improvement and must be
        # the value returned.
        result, n_calls = self._run([10.0, 5.0, 5.0])
        assert n_calls == 3
        assert result[0] == pytest.approx(3.0)

    def test_non_finite_step_is_still_rejected(self):
        # A failed solve must not be accepted; the last good iterate wins.
        result, n_calls = self._run([10.0, 5.0, np.inf])
        assert n_calls == 3
        assert result[0] == pytest.approx(2.0)

    def test_loop_runs_to_the_iteration_budget_when_never_converged(self):
        result, n_calls = self._run([10.0, 5.0, 1.0])
        assert n_calls == 3
        assert result[0] == pytest.approx(3.0)


class TestObjectiveTermParity:
    """Every term in the QP matrices must also reach the scalar objective.

    The scalar objective is what the convergence test compares, so a term that
    is present only in the matrices lets the loop stop while that term is still
    descending.
    """

    def test_base_position_tracking_appears_in_the_scalar_objective(self):
        import inspect

        source = inspect.getsource(GenericInteractionRetargeter._single_optimization_step)

        # The term is assembled into the matrices via P_diag_extra / c and must
        # also be accumulated into `cost` before the function returns.
        assert "base_pos_terms.append" in source, (
            "base position tracking no longer records its terms for scalar evaluation"
        )
        assert "for qa_idx, diff in base_pos_terms:" in source, (
            "base position tracking is assembled into the QP but never added to the "
            "scalar objective used for the convergence test"
        )


class TestBaseOrientationTarget:
    """Which orientation the floating base is asked to track.

    The default infers a frame from four joint positions. That inference is a
    construction, not a measurement, and on SMPL-X it lands the pelvis about
    fourteen degrees away from the source, which the waist then absorbs by
    bending to its limit. Sources carrying a real root orientation can select
    it instead, but the default has to stay put: profiles whose sources have no
    root orientation depend on the inferred frame.
    """

    @staticmethod
    def _run(base_orientation_config):
        from unittest.mock import Mock

        from scipy.spatial.transform import Rotation

        from omniretargeting import OmniRetargeter
        from omniretargeting.core import RetargetingStreamState
        from omniretargeting.data_sources.base import MotionFrame

        estimated = np.array([0.1, 0.2, 0.3, 0.9])
        estimated /= np.linalg.norm(estimated)
        root = Rotation.from_rotvec([0.0, 0.0, np.pi / 2.0]).as_quat(scalar_first=True)

        inner = Mock()
        inner.retarget_frame.return_value = np.arange(7, dtype=float)

        retargeter = OmniRetargeter.__new__(OmniRetargeter)
        retargeter.retargeting_config = {}
        retargeter.base_orientation_config = base_orientation_config
        retargeter._estimate_base_orientation_from_joints = Mock(return_value=estimated)
        retargeter._extract_mapped_source_targets = Mock(return_value=np.zeros((4, 3)))

        state = RetargetingStreamState(
            retargeter=inner,
            q_init=np.zeros(7),
            q_last=None,
            last_estimated_quat=None,
            frame_idx=0,
            scaled_terrain=Mock(),
        )
        frame = MotionFrame(
            positions=np.zeros((4, 3)),
            root_orientation=root,
            root_translation=np.array([1.0, 2.0, 3.0]),
        )
        retargeter.retarget_frame(frame, state)
        return inner.retarget_frame.call_args.kwargs["target_base_orientation"], estimated, root

    def test_default_keeps_the_inferred_frame(self):
        target, estimated, _ = self._run({})
        np.testing.assert_allclose(target, estimated)

    def test_explicit_estimate_keeps_the_inferred_frame(self):
        target, estimated, _ = self._run({"target": "estimate"})
        np.testing.assert_allclose(target, estimated)

    def test_root_option_tracks_the_source_orientation(self):
        target, _, root = self._run({"target": "root"})
        np.testing.assert_allclose(target, root)

    def test_unknown_value_falls_back_to_the_default(self):
        target, estimated, _ = self._run({"target": "nonsense"})
        np.testing.assert_allclose(target, estimated)
