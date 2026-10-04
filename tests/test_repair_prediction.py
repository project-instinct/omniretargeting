"""Prediction uses the worst linear witness, and gating precedes candidates."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from omniretargeting._fast import repair_prediction, mesh_repair, repair_proposals
from tests.test_fast_backend import _objective
from tests.test_fast_mesh_repair import _open_object


def _predict(rows, phi, step, depths=None, keys=None, body=None):
    keys = keys or [(0, "static", 0)] * len(phi)
    model = SimpleNamespace(geom_bodyid=np.arange(3), body_weldid=np.arange(3))
    geometry = SimpleNamespace(model=model, linear_rows=lambda _: (sparse.csr_matrix(rows), np.array(phi)))
    measurement = {"rows": [(k, {}) for k in keys],
                   "depths": depths or {k: .01 for k in keys}}
    return repair_prediction.linear_prediction(geometry, measurement, np.array([step]), body, .01)


def test_prediction_uses_worst_corner_not_mean_or_row_count():
    prediction = _predict([[1., 1.], [1., -1.]], [-.01, -.01], [.005, .004])
    assert prediction["predicted_decrease_m"] == pytest.approx(.001)
    repeated = _predict([[1., 1.]] * 20 + [[1., -1.]], [-.01] * 21, [.005, .004])
    assert repeated == prediction


def test_prediction_excludes_other_groups_and_keeps_unrepresented_depth():
    keys = [(0, "static", 0), (0, "self", 1, 2)]
    prediction = _predict([[1.], [-1.]], [-.01, -.02], [.005], keys=keys, body=0)
    assert prediction["predicted_decrease_m"] == pytest.approx(.005)
    prediction = _predict([[1.]], [-.01], [.005],
                          depths={(0, "static", 0): .01, (1, "static", 0): .01})
    assert prediction["predicted_decrease_m"] == 0.
    assert _predict([[1.]], [-.01], [np.nan]) is None


def test_gating_is_default_on_strict_and_requires_finite_prediction(monkeypatch):
    assert repair_prediction.PREDICTIVE_GATING is True
    assert repair_prediction.RECORD_PREDICTIONS is False
    prediction = {"predicted_decrease_m": 0.}
    monkeypatch.setattr(repair_prediction, "PREDICTIVE_GATING", False)
    assert not repair_prediction.should_gate(prediction, 1e-4)
    monkeypatch.setattr(repair_prediction, "PREDICTIVE_GATING", True)
    assert repair_prediction.should_gate(prediction, 1e-4)
    for value in (1e-4, 2e-4, np.nan, np.inf):
        assert not repair_prediction.should_gate({"predicted_decrease_m": value}, 1e-4)
    assert not repair_prediction.should_gate(None, 1e-4)
    assert not repair_prediction.should_gate(prediction, 0.)


def test_gated_attempt_never_starts_candidates_or_trial_geometry(monkeypatch):
    monkeypatch.setenv("OMNIRETARGETING_GPU_GEOMETRY", "0")
    monkeypatch.setattr(repair_prediction, "PREDICTIVE_GATING", True)
    monkeypatch.setattr(repair_prediction, "linear_prediction", lambda *a: {
        "predicted_decrease_m": 0., "entry_depth_m": .01, "predicted_depth_m": .01})
    monkeypatch.setattr(mesh_repair, "_qp_step", lambda limits, *a: (
        np.full((limits.frames, limits.width), .001), "Solved"))

    def forbidden(*args, **kwargs):
        raise AssertionError("gated attempt started candidate checks")

    monkeypatch.setattr(repair_proposals.LocalProposal, "candidates", forbidden)
    original = mesh_repair.OriginalGeometry.measure
    calls = []

    def measure(self, *args, **kwargs):
        calls.append(kwargs)
        assert not kwargs.get("protected_depths")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(mesh_repair.OriginalGeometry, "measure", measure)
    objective, poses, _ = _objective(temporal=True)
    reference = objective.reference.tobytes()
    result, report = mesh_repair.repair_original_meshes(objective, poses, [_open_object(objective, poses)])
    assert result.tobytes() == poses.tobytes()
    assert objective.reference.tobytes() == reference
    assert report["history"]
    assert all(r["stop_reason"] == "predicted_small_progress" for r in report["history"])
    assert len(calls) == 1  # Initial baseline only.


def test_remote_prediction_gates_before_worker_candidates(monkeypatch):
    import os
    if not hasattr(os, "fork"):
        pytest.skip("fork unavailable")
    from tests.test_repair_proposals import _problem
    from omniretargeting._fast.repair_schedule import RepairSchedule
    monkeypatch.setattr(
        repair_proposals, 'OPTIONS',
        replace(repair_proposals.OPTIONS, repair_fast_gate=False))
    monkeypatch.setattr(repair_prediction, "PREDICTIVE_GATING", True)
    monkeypatch.setattr(repair_prediction, "linear_prediction", lambda *a: {"predicted_decrease_m": 0.})

    def forbidden(*args, **kwargs):
        raise AssertionError("gated worker started candidate checks")

    monkeypatch.setattr(repair_proposals.LocalProposal, "candidates", forbidden)
    limits, geometry, poses, measured, protected, group = _problem(monkeypatch)
    schedule = RepairSchedule([group], 12)
    workers = repair_proposals.RepairProposals(limits, geometry, count=1)
    try:
        workers.prefetch(schedule, poses, measured, protected)
        repair_proposals.transition_group(limits, poses, group)
        limits.frame_mask = repair_proposals.group_frames(measured, geometry.model, group, limits.frames,
                                                         repair_proposals.halo_frames(30.))
        depth = max(v for k, v in measured["depths"].items() if any(
            geometry.model.body_weldid[geometry.model.geom_bodyid[g]] == group
            for g in (k[2:] if k[1] == "self" else k[-1:])))
        remote = workers.take(poses, measured, protected, None, depth)
        assert isinstance(remote, repair_proposals.RemoteProposal)
        assert remote.prediction == {"predicted_decrease_m": 0.}
        assert remote.done
        remote.close()
        assert workers.workers[0][0].is_alive()
    finally:
        workers.close(abort=True)
