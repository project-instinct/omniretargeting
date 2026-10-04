"""An admissible unchanged pose is not a successful priority solve."""

import numpy as np

from omniretargeting._fast.priority_reporting import solver_progress
from tests.test_priority_clearance import _priority


def test_unchanged_pose_is_admissible_but_three_failed_phases_make_no_updates():
    guard, _, _ = _priority()
    history = [dict(phase=phase, alpha=0., priority_status='no_certified_accepted_proposal')
               for phase in ('contact_scene', 'priority_clearance_completion', 'priority_original_mesh_repair')]
    report = guard.diagnostics(guard.current, solver_history=history)
    assert report['accepted'] and report['pose_admissible']
    assert report['accepted_semantics'] == 'priority_pose_admissibility_only_not_solver_success'
    assert report['accepted_update_count'] == 0
    assert report['solver_outcome'] == 'no_certified_accepted_proposal'
    assert not report['convergence_certified'] and not report['hard_infeasibility_certified']
    assert len(report['solver_phases']) == 3
    assert all(item['accepted_update_count'] == 0 for item in report['solver_phases'].values())


def test_committed_progress_does_not_certify_nonlinear_convergence():
    guard, _, _ = _priority()
    candidate = guard.current + np.array([[.01, 0.]])
    guard.certificate = dict(certified=True)
    guard.commit(candidate)
    history = [dict(phase='contact_scene', alpha=1., priority_certificate=dict(certified=True))]
    report = guard.diagnostics(candidate, solver_history=history)
    assert report['accepted_update_count'] == 1
    assert report['solver_outcome'] == 'accepted_updates_without_convergence_certificate'
    assert not report['convergence_certified']
    assert guard.epoch == 1  # Reporting did not commit anything.


def test_missing_history_is_explicit_not_inferred_from_admissibility():
    guard, _, _ = _priority()
    report = guard.diagnostics(guard.current)
    assert report['accepted']
    assert report['solver_outcome'] == 'not_reported'
    assert not report['convergence_certified']


def test_progress_followed_by_stall_keeps_both_facts_visible():
    history = [dict(phase='contact_scene', alpha=1.),
               dict(phase='contact_scene', alpha=0., priority_status='no_certified_accepted_proposal'),
               dict(phase='completion', alpha=0., priority_status='repeated_stalled_problem')]
    report = solver_progress(history)
    assert report['accepted_update_count'] == 1
    assert report['solver_outcome'] == 'repeated_stalled_problem'
    assert report['solver_phases']['contact_scene']['accepted_update_count'] == 1
    assert not report['convergence_certified']
