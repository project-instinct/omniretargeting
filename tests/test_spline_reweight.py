"""Necessary shared-iteration reweighting without reference changes."""

import numpy as np

import omniretargeting._fast.pipeline as fast_pipeline
from omniretargeting._fast.spline_reweight import SharedReweight
from tests.test_fast_backend import _objective


def test_only_new_posture_frames_are_weighted_once(monkeypatch):
    objective, poses, _ = _objective(temporal=True)
    fixed = poses.copy()
    envelope_before = objective.envelope_weight.copy()
    records = []
    updater = SharedReweight(objective, fixed, None, records)
    calls = []

    def verified(model, coords, reference, motion, bodies=None, shape=None):
        calls.append((model is objective.model, reference, motion.copy()))
        budgets = {"layout_flip_frames": 2}
        posture = np.zeros(len(motion), bool)
        posture[[1, 2]] = True
        return budgets, np.zeros(len(motion), bool), posture

    monkeypatch.setattr(fast_pipeline, "verify_budgets", verified)
    assert updater(poses, "spline_local_free") is True
    expected = envelope_before.copy()
    expected[[1, 2]] *= fast_pipeline.REWEIGHT_FACTOR
    np.testing.assert_array_equal(objective.envelope_weight, expected)
    assert len(records) == 1
    assert records[0]["phase"] == "spline_local_free"
    assert records[0]["flagged_frames"] == [1, 2]
    assert records[0]["before"] == {"layout_flip_frames": 2}
    assert records[0]["necessary_rows_only"] is True
    assert records[0]["no_reference_rebase"] is True
    assert calls[0][1] is updater.fixed and calls[0][2] is not poses

    assert updater(poses, "spline_local_free") is False
    np.testing.assert_array_equal(objective.envelope_weight, expected)
    assert len(records) == 1

    def verified_new(model, coords, reference, motion, bodies=None, shape=None):
        posture = np.zeros(len(motion), bool)
        posture[[2, 3]] = True
        return {}, np.zeros(len(motion), bool), posture

    monkeypatch.setattr(fast_pipeline, "verify_budgets", verified_new)
    assert updater(poses, "second_window") is True
    expected[3] *= fast_pipeline.REWEIGHT_FACTOR
    np.testing.assert_array_equal(objective.envelope_weight, expected)
    assert records[-1]["phase"] == "second_window"
    assert records[-1]["flagged_frames"] == [3]
    assert updater.posture_weighted[:4].tolist() == [False, True, True, True]
    np.testing.assert_array_equal(objective.reference, fixed)
    assert not updater.fixed.flags.writeable


def test_chatter_uses_original_global_reweight_at_most_once(monkeypatch):
    objective, poses, _ = _objective(temporal=True)
    records = []
    updater = SharedReweight(objective, poses.copy(), None, records)

    def verified(model, coords, reference, motion, bodies=None, shape=None):
        return {}, np.ones(len(motion), bool), np.zeros(len(motion), bool)

    monkeypatch.setattr(fast_pipeline, "verify_budgets", verified)
    calls = []

    def reweighted(objective_arg, fixed_arg, motion_arg, factor):
        calls.append((objective_arg is objective, fixed_arg, motion_arg, factor))
        objective_arg.joint_temporal_weight[0][1] *= factor
        return {"joint_columns": [0]}

    import omniretargeting._fast.reweight as fast_reweight
    monkeypatch.setattr(fast_reweight, "reweight_temporal_signals", reweighted)
    before = objective.joint_temporal_weight[0].copy()
    assert updater(poses, "shared") is True
    assert len(calls) == 1
    expected = before.copy()
    expected[1] *= fast_pipeline.REWEIGHT_FACTOR
    np.testing.assert_array_equal(objective.joint_temporal_weight[0], expected)

    assert updater(poses, "shared") is False
    assert len(calls) == 1
    np.testing.assert_array_equal(objective.joint_temporal_weight[0], expected)
    assert updater.chatter_rounds == 1
    assert len(records) == 1
    assert records[0]["necessary_rows_only"] is False
    assert records[0]["chatter_global_fallback"] is True
    assert records[0]["chatter_reweighted_signals"] == {"joint_columns": [0]}


def test_real_objective_sees_the_reweighted_cost_and_gradient(monkeypatch):
    objective, poses, _ = _objective(temporal=True, extra=True)
    fixed = poses.copy()
    rng = np.random.default_rng(7)
    motion = objective.coords.integrate(
        poses, rng.normal(scale=.003, size=(len(poses), objective.width)), 1.)
    records = []
    updater = SharedReweight(objective, fixed, None, records)
    body = int(objective.tasks.root_body)

    class WeightedEnvelope:
        def hinge_rows(self, kin, weight, sigma, with_jacobian):
            rows = (kin.xpos[:, body, :1] - objective.root_reference[:, :1]) * weight[:, None]
            jacobian = None
            if with_jacobian:
                jacobian = kin.jacp[:, body, :1, :] * weight[:, None, None]
            return rows, jacobian, 0.

    objective.relation_envelope = WeightedEnvelope()

    def verified(model, coords, reference, motion_arg, bodies=None, shape=None):
        posture = np.zeros(len(motion_arg), bool)
        posture[9:14] = True
        return {"layout_flip_frames": 5}, np.zeros(len(motion_arg), bool), posture

    monkeypatch.setattr(fast_pipeline, "verify_budgets", verified)
    before = objective.evaluate(motion)
    before_bytes = objective.reference.tobytes()
    assert updater(motion, "spline_free") is True
    after = objective.evaluate(motion)

    assert after[0].cost != before[0].cost
    assert np.max(np.abs(after[0].gradient - before[0].gradient)) > 1e-8
    assert np.max(np.abs(after[0].hessian - before[0].hessian)) > 1e-8
    assert records[0]["before"] == {"layout_flip_frames": 5}
    assert objective.reference.tobytes() == before_bytes
    np.testing.assert_array_equal(updater.fixed, fixed)
    assert not updater.fixed.flags.writeable
