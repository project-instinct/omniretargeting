"""Reusing checks must keep rejection, backtracking and changed-pose audits."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

import omniretargeting.whole_body_correction as correction
from omniretargeting.whole_body_guard import WholeBodyGuard
from tests.test_whole_body_moving import (
    _config,
    _far_scene,
    _model_and_trajectory,
    _moving_box,
)


@pytest.mark.parametrize("rejected_by", ["fixed", "fidelity", "geometry"])
def test_rejected_steps_never_evaluate_objective(rejected_by):
    validator = SimpleNamespace(
        qi=np.array([0]),
        fidelity=Mock(return_value={"reasons": ["posture"] if rejected_by == "fidelity" else []}),
        geometry=Mock(return_value={}),
        compare_geometry=Mock(return_value=["collision"] if rejected_by == "geometry" else []),
    )
    guard = WholeBodyGuard(validator)
    guard.before = {}
    merit = Mock(side_effect=AssertionError("Rejected step evaluated objective"))
    current = np.zeros((2, 1))
    result, report = guard.accept(
        current, np.ones_like(current), merit,
        candidate_acceptance=lambda _: rejected_by != "fixed",
    )
    np.testing.assert_array_equal(result, current)
    assert report["accepted_alpha"] == 0
    assert len(report["rejections"]) == 7
    merit.assert_not_called()


def test_lazy_objective_keeps_original_cost_and_backtracking():
    geometry = {"uncertain_frames": [], "uncertain_intervals": []}
    validator = SimpleNamespace(
        qi=np.array([0]),
        fidelity=Mock(return_value={"reasons": [], "metrics": {}}),
        geometry=Mock(return_value=geometry),
        compare_geometry=Mock(return_value=[]),
    )
    guard = WholeBodyGuard(validator)
    guard.before = geometry
    merit = Mock(side_effect=lambda q: float(np.sum((q - .125) ** 2)))
    current = np.zeros((2, 1))
    result, report = guard.accept(
        current, np.ones_like(current), merit,
        candidate_acceptance=lambda q: np.max(q) <= .25,
    )
    assert report["accepted_alpha"] == .125
    assert report["objective_before"] == .03125
    assert report["objective_after"] == 0
    assert report["rejections"][-1]["reasons"] == ["nonlinear_objective_not_improved"]
    np.testing.assert_array_equal(result, np.full((2, 1), .125))
    assert [float(call.args[0][0, 0]) for call in merit.call_args_list] == [0., .25, .125]


@pytest.mark.parametrize("allow_change", [False, True])
def test_composed_moving_check_reused_only_for_unchanged_trajectory(monkeypatch, allow_change):
    monkeypatch.setattr(correction, "WHOLE_BODY_WINDOW_THRESHOLD", 4)
    monkeypatch.setattr(correction, "WHOLE_BODY_WINDOW_FRAMES", 4)
    model, short = _model_and_trajectory()
    baseline = np.tile(short[0], (8, 1))
    obj = _moving_box()
    obj["translations"] = np.c_[np.linspace(-.15, .15, 8), np.zeros((8, 2))]
    obj["rotations"] = np.tile(np.eye(3), (8, 1, 1))
    obj["scales"] = np.ones(8)
    checked = []
    original = WholeBodyGuard.moving_check

    def record_check(self, candidate):
        checked.append((self.validator.frame_start, candidate.copy()))
        return original(self, candidate)

    monkeypatch.setattr(WholeBodyGuard, "moving_check", record_check)
    result, report = correction.correct_whole_body_trajectory(
        model, baseline, {}, {}, _config(iterations=1),
        scene_query=_far_scene(), scene_objects=[obj],
        candidate_acceptance=lambda q: allow_change or np.array_equal(q, baseline),
    )
    assert report["final_reasons"] == []
    assert report["moving_collision"]["after"]["complete"]
    full_checks = [q for start, q in checked if start == 0 and len(q) == len(baseline)]
    if allow_change:
        assert report["accepted_update_count"] > 0
        assert not np.array_equal(result, baseline)
        assert report["reused_baseline_checks"] == {"static": 0, "moving": 0}
        assert len(full_checks) == 2
        np.testing.assert_array_equal(full_checks[-1], result)
    else:
        np.testing.assert_array_equal(result, baseline)
        assert report["accepted_update_count"] == 0
        assert len(full_checks) == 1
        assert report["reused_baseline_checks"] == {"static": 1, "moving": 1}
        assert report["moving_collision"]["before"] == report["moving_collision"]["after"]


def test_failed_reference_check_remains_a_failure(monkeypatch):
    model, baseline = _model_and_trajectory()
    failed = Mock(return_value=(["moving_geometry_failure"], {"complete": False}))
    monkeypatch.setattr(WholeBodyGuard, "moving_check", failed)
    monkeypatch.setattr(correction, "WHOLE_BODY_WINDOW_THRESHOLD", 1)
    result, report = correction.correct_whole_body_trajectory(
        model, baseline, {}, {}, _config(), scene_query=_far_scene(),
        scene_objects=[_moving_box()],
    )
    np.testing.assert_array_equal(result, baseline)
    assert report["reason"] == "baseline_moving_geometry_failed"
    assert report["reasons"] == ["moving_geometry_failure"]
    assert report["iterations"] == []
    assert failed.call_count == 1
