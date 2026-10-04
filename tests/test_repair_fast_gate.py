"""Reduced prediction is optional; it never changes the full repair contracts."""
from dataclasses import replace

import numpy as np
import pytest

from omniretargeting._fast import mesh_repair, repair_fast_gate, repair_proposals, repair_triage
from omniretargeting._fast.repair_gate_frames import local_limits
from tests.test_repair_proposals import _problem


@pytest.fixture(autouse=True)
def isolate_reduced_gate(monkeypatch):
    # The first-level certificate now defaults ON. Keep it OFF in tests
    # whose assertions specifically exercise the second-level gate.
    monkeypatch.setattr(repair_triage, 'OPTIONS', replace(repair_triage.OPTIONS, repair_triage=False))


def test_reduced_program_has_small_domain_and_leaves_reference_and_anchors_fixed(monkeypatch):
    limits, geometry, poses, measured, protected, body = _problem(monkeypatch)
    limits.movable_body = body
    before = [poses.tobytes(), limits.entry.tobytes(), limits.objective.reference.tobytes(),
              *(a.tobytes() for a in limits.anchors), *(a.tobytes() for a in limits.target_budgets)]
    prediction = repair_fast_gate.predict(limits, poses, geometry, measured, protected,
                                           max(measured['depths'].values()))
    assert prediction is not None and prediction['predicted_decrease_m'] > 1e-4
    assert len(prediction['frames']) <= 3
    assert all(p['coordinates'] <= limits.width for p in prediction['frames'])
    assert before == [poses.tobytes(), limits.entry.tobytes(), limits.objective.reference.tobytes(),
                      *(a.tobytes() for a in limits.anchors), *(a.tobytes() for a in limits.target_budgets)]
    local = local_limits(limits, 27)
    assert local.frames == 1 and not local.differences
    assert np.array_equal(local.entry, limits.entry[27:28])
    assert np.array_equal(local.penetration_yield, limits.penetration_yield[27:28])
    assert np.array_equal(local.objective.reference, limits.objective.reference[27:28])


@pytest.mark.parametrize('failure', ['custom_guard', 'incomplete', 'query_failure',
                                    'solver_failure', 'nonfinite', 'exception'])
def test_uncertain_reduced_models_retain_the_full_attempt(monkeypatch, failure):
    limits, geometry, poses, measured, protected, body = _problem(monkeypatch)
    limits.movable_body = body
    if failure == 'custom_guard':
        limits.objective.candidate_guards = [object()]
    elif failure == 'incomplete':
        measured['complete'] = False
    elif failure == 'query_failure':
        measured['failures'] = ['unknown sign']
    elif failure == 'solver_failure':
        monkeypatch.setattr(repair_fast_gate, 'solve_step', lambda *a, **kw: (None, 'NumericalError'))
    elif failure == 'nonfinite':
        monkeypatch.setattr(repair_fast_gate, 'solve_step',
                            lambda *a, **kw: (np.full((1, limits.width), np.nan), 'Solved'))
    else:
        def failed(*args, **kwargs):
            raise RuntimeError('failed reduced solver')
        monkeypatch.setattr(repair_fast_gate, 'solve_step', failed)
    assert repair_fast_gate.predict(limits, poses, geometry, measured, protected, .01) is None


@pytest.mark.parametrize('enabled,shadow', [(False, False), (False, True), (True, False)])
def test_only_enabled_gate_can_avoid_full_qp(monkeypatch, enabled, shadow):
    limits, geometry, poses, measured, protected, body = _problem(monkeypatch)
    limits.movable_body = body
    monkeypatch.setattr(repair_proposals, 'OPTIONS',
                        replace(repair_proposals.OPTIONS, repair_fast_gate=enabled, repair_triage=False))
    monkeypatch.setattr(repair_fast_gate, 'RECORD_PREDICTIONS', shadow)
    calls = []
    prediction = dict(predicted_decrease_m=0., entry_depth_m=.01,
                      predicted_depth_m=.01, model='reduced_frame_qp')
    def reduced(*args):
        calls.append('reduced')
        return prediction
    def full(*args):
        calls.append('full')
        return np.full((limits.frames, limits.width), .001), 'Solved'
    monkeypatch.setattr(repair_fast_gate, 'predict', reduced)
    monkeypatch.setattr(mesh_repair, '_qp_step', full)
    proposal = repair_proposals.LocalProposal(limits, geometry, poses, measured, protected, None, .01)
    if enabled:
        assert calls == ['reduced'] and proposal.status == 'reduced_predicted_small_progress'
    elif shadow:
        assert calls == ['reduced', 'full'] and proposal.prediction['reduced_gate'] == prediction
    else:
        assert calls == ['full'] and proposal.fast_prediction is None


def test_threshold_is_existing_strict_gate_and_option_is_default_off():
    from omniretargeting._fast.runtime_options import RuntimeOptions
    assert RuntimeOptions().repair_fast_gate is True
    assert not repair_fast_gate.should_skip(None)
    assert repair_fast_gate.should_skip({'predicted_decrease_m': 0.})
    for value in [mesh_repair.MIN_GROUP_PROGRESS_M, np.nan, np.inf]:
        assert not repair_fast_gate.should_skip({'predicted_decrease_m': value})


def test_reduced_prefetch_skip_avoids_transfer_but_actual_attempt_rechecks(monkeypatch):
    from omniretargeting._fast.repair_schedule import RepairSchedule
    limits, geometry, poses, measured, protected, body = _problem(monkeypatch)
    monkeypatch.setattr(repair_proposals, 'OPTIONS',
                        replace(repair_proposals.OPTIONS, repair_fast_gate=True, repair_triage=False))
    calls = []
    def reduced(*args):
        calls.append(args[0].movable_body)
        return dict(predicted_decrease_m=0., model='reduced_frame_qp')
    def forbidden(*args):
        raise AssertionError('rejected dispatch serialized a dependency')
    monkeypatch.setattr(repair_fast_gate, 'predict', reduced)
    monkeypatch.setattr(repair_proposals, 'certificate', forbidden)
    workers = repair_proposals.RepairProposals(limits, geometry, count=1)
    try:
        workers.prefetch(RepairSchedule([body], 12), poses, measured, protected)
        assert not workers.pending and workers.stats['reduced_prefetch_skips'] == 1
        repair_proposals.transition_group(limits, poses, body)
        proposal = workers.take(poses, measured, protected, None, .01)
        assert calls == [body, body] and proposal.status == 'reduced_predicted_small_progress'
    finally:
        workers.close(abort=True)


@pytest.mark.parametrize('sequence,correction', [(False, False), (False, True),
                                               (True, False), (True, True)])
def test_window_temporal_rows_match_original_interior_stencils(monkeypatch, sequence, correction):
    from scipy.spatial.transform import Rotation
    from omniretargeting._fast.continuity import CorrectionContinuity
    from omniretargeting._sequence.continuity_guard import SequenceContinuityGuard
    from omniretargeting._fast.repair_gate_temporal import joint_guard, differences
    limits, _, poses, _, _, _ = _problem(monkeypatch)
    if sequence:
        guard = SequenceContinuityGuard(limits.objective.tasks, poses,
                    Rotation.identity(len(poses)), windowed=True, correction=correction)
    else:
        guard = CorrectionContinuity(limits.objective.coords, poses)
        guard.correction = correction
    candidate = poses.copy()
    candidate[:, limits.coords.qi] += np.sin(np.arange(len(poses)))[:,None]*1e-5
    part = slice(5,25)
    full = guard.repair_bounds(candidate)[0]
    local = joint_guard(guard, part, sequence).repair_bounds(candidate[part])[0]
    width, joint_width = limits.objective.coords.width, limits.width
    for order, (global_row, local_row) in enumerate(zip(full[:3], local),1):
        rows = slice(part.start*joint_width,(part.stop-order)*joint_width)
        cols = slice(part.start*width,part.stop*width)
        assert np.array_equal(global_row[0][rows,cols].toarray(),local_row[0].toarray())
        for first, second in zip(global_row[1:],local_row[1:]):
            assert np.array_equal(first[rows],second)
    for order, (matrix, budget) in enumerate(differences(limits,part),1):
        operator, original_budget = limits.differences[order-1]
        rows=slice(part.start*joint_width,(part.stop-order)*joint_width)
        cols=slice(part.start*joint_width,part.stop*joint_width)
        assert np.array_equal(matrix.toarray(),operator[rows,cols].toarray())
        assert np.array_equal(budget,original_budget[rows])
