"""Sound early model bounds, uncertain fallback and no-QP execution."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from omniretargeting._fast import mesh_repair, repair_proposals, repair_triage, repair_prediction


def problem(monkeypatch, jac, phi, *, low=-.05, high=.05, mask=None):
    jac = np.asarray(jac, dtype=float)
    frames, width = jac.shape
    limits = SimpleNamespace(frames=frames, width=width, movable_body=0, frame_mask=mask,
                             coords=SimpleNamespace(qi=np.arange(width)),
                             lower=np.full((frames, width), low), upper=np.full((frames, width), high))
    model = SimpleNamespace(geom_bodyid=np.array([0, 1, 2]), body_weldid=np.arange(3))
    geometry = repair_proposals.GeometryRows(SimpleNamespace(model=model, width=width,
                                                            objective=SimpleNamespace(frames=frames)))
    measurement = dict(rows=[((f, "static", 0), dict(jacobian=row, clearance=float(phi[f])))
                             for f, row in enumerate(jac)],
                       depths={(f, "static", 0): max(0., -float(phi[f])) for f in range(frames)})
    monkeypatch.setattr(mesh_repair, "_joint_projection", lambda *a: sparse.eye(frames * width, format="csc"))
    return limits, np.zeros((frames, width)), geometry, measurement


def test_box_bound_dominates_every_feasible_linear_prediction(monkeypatch):
    rng = np.random.default_rng(9)
    jac = rng.normal(size=(5, 4))
    limits, poses, geometry, measured = problem(monkeypatch, jac, [-.01] * 5, low=-.007, high=.019)
    bound = repair_triage.progress_bound(limits, poses, geometry, measured, .01)
    for _ in range(100):
        step = rng.uniform(limits.lower, limits.upper)
        predicted = repair_prediction.linear_prediction(geometry, measured, step, 0, .01)
        assert predicted['predicted_decrease_m'] <= bound['upper_decrease_m']


def test_frame_mask_and_absent_witness_keep_the_maximum(monkeypatch):
    limits, poses, geometry, measured = problem(monkeypatch, [[1.], [1.]], [-.01, -.01],
                                               mask=np.array([True, False]))
    bound = repair_triage.progress_bound(limits, poses, geometry, measured, .01)
    assert bound['upper_decrease_m'] < 1e-9
    assert bound['binding_key'] == [1, 'static', 0]
    measured['rows'].pop()
    bound = repair_triage.progress_bound(limits, poses, geometry, measured, .01)
    assert bound['upper_decrease_m'] == 0.


def test_zero_projection_is_explicit_and_non_selector_is_uncertain(monkeypatch):
    limits, poses, geometry, measured = problem(monkeypatch, [[1.]], [-.01])
    monkeypatch.setattr(mesh_repair, '_joint_projection', lambda *a: sparse.csc_matrix((1, 0)))
    assert repair_triage.progress_bound(limits, poses, geometry, measured, .01)['reason'] == 'no_local_dofs'
    monkeypatch.setattr(mesh_repair, '_joint_projection', lambda *a: sparse.csc_matrix([[2.]]))
    assert repair_triage.progress_bound(limits, poses, geometry, measured, .01) is None


@pytest.mark.parametrize('bad', ['jacobian', 'clearance', 'clearance_shape', 'depth', 'failure', 'partial'])
def test_uncertainty_cannot_skip(monkeypatch, bad):
    limits, poses, geometry, measured = problem(monkeypatch, [[0.]], [-.01])
    if bad in ('jacobian', 'clearance'):
        measured['rows'][0][1][bad] = np.array([np.nan]) if bad == 'jacobian' else np.nan
    elif bad == 'clearance_shape':
        measured['rows'][0][1]['clearance'] = np.array([-.01, -.005])
    elif bad == 'depth':
        measured['depths'][(0, 'static', 0)] = np.nan
    elif bad == 'failure':
        measured['failures'] = ['unknown']
    else:
        measured['complete'] = False
    assert repair_triage.progress_bound(limits, poses, geometry, measured, .01) is None


def test_opt_in_certificate_precedes_qp_and_preserves_existing_gate(monkeypatch):
    limits, poses, geometry, measured = problem(monkeypatch, [[0.]], [-.01])
    pass
    pass
    monkeypatch.setattr(repair_prediction, 'PREDICTIVE_GATING', True)

    def forbidden(*a, **k):
        raise AssertionError('certified attempt built a QP')
    monkeypatch.setattr(mesh_repair, '_qp_step', forbidden)
    proposal = repair_proposals.LocalProposal(limits, geometry, poses, measured, measured, None, .01)
    assert proposal.status == 'pre_qp_certified_small_progress'
    assert repair_prediction.should_gate(proposal.prediction, mesh_repair.MIN_GROUP_PROGRESS_M)
    monkeypatch.setattr(repair_prediction, 'PREDICTIVE_GATING', False)
    assert repair_triage.certificate(limits, poses, geometry, measured, .01) is None


def test_threshold_is_strict_and_useful_witness_falls_back(monkeypatch):
    limits, poses, geometry, measured = problem(monkeypatch, [[1.]], [-.01])
    pass
    assert repair_triage.certificate(limits, poses, geometry, measured, .01) is None
    monkeypatch.setattr(repair_triage, 'progress_bound', lambda *a: dict(upper_decrease_m=1e-4))
    assert repair_triage.certificate(limits, poses, geometry, measured, .01) is None


def test_opposing_protected_witness_certifies_what_box_cannot(monkeypatch):
    limits, poses, geometry, measured = problem(monkeypatch, [[1.]], [-.01])
    measured['rows'].append(((0, 'static', 1), dict(jacobian=np.array([-1.]), clearance=-.01)))
    measured['depths'][(0, 'static', 1)] = .01
    bound = repair_triage.progress_bound(limits, poses, geometry, measured, .01)
    assert bound['reason'] == 'protected_witness_dual'
    assert 0. <= bound['upper_decrease_m'] < 1e-4


def test_approximate_dual_still_bounds_every_feasible_step(monkeypatch):
    from omniretargeting._fast import repair_triage_dual
    # Deliberately non-optimal, sign-imperfect multipliers. The residual box
    # support must still make the certificate conservative.
    monkeypatch.setattr(repair_triage_dual, 'linprog', lambda *a, **k: SimpleNamespace(
        success=True, ineqlin=SimpleNamespace(marginals=np.array([-.3, .7]))))
    rows = [((0, 'static', i), dict(jacobian=np.array([j]), clearance=-.01))
            for i, j in enumerate([1., -1.])]
    depths = {key: .02 for key, _ in rows}
    upper = repair_triage_dual.maximum(np.array([2.]), np.array([-.05]), np.array([.05]), rows, depths)
    assert upper >= .02  # Feasible delta interval [-.01, .01].


@pytest.mark.parametrize('failure', ['timeout', 'invalid', 'nonfinite'])
def test_unavailable_dual_keeps_the_outer_box(monkeypatch, failure):
    from omniretargeting._fast import repair_triage_dual

    def unavailable(*a, **k):
        if failure == 'invalid':
            raise ValueError('optional LP backend unavailable')
        return SimpleNamespace(success=False)

    monkeypatch.setattr(repair_triage_dual, 'linprog', unavailable)
    rows = [((0, 'static', 0), dict(jacobian=np.array([1.]), clearance=-.01))]
    depths = {rows[0][0]: np.nan if failure == 'nonfinite' else .01}
    assert repair_triage_dual.maximum(np.array([1.]), np.array([-.05]), np.array([.05]), rows, depths) is None


def test_temporal_intervals_retain_linear_program_extrema():
    from omniretargeting._fast.repair_triage_temporal import tighten
    from scipy.optimize import linprog
    from omniretargeting._refinement.math import _difference_operator
    from omniretargeting._fast.repair_limits import NUMERICAL_SLACK
    values = np.zeros((6, 1))
    limits = SimpleNamespace(entry=values, coords=SimpleNamespace(qi=np.array([0])),
                             differences=[(_difference_operator(6, n), np.full(6-n, .001))
                                          for n in (1, 2, 3)])
    low, high = np.full_like(values, -.05), np.full_like(values, .05)
    low[[0, -1]], high[[0, -1]] = 0., 0.
    first, last = tighten(limits, values, low, high)
    matrix = sparse.vstack([op for op, _ in limits.differences])
    cap = np.concatenate([b + NUMERICAL_SLACK / 4 for _, b in limits.differences])
    for index in range(6):
        for sign in [-1., 1.]:
            objective = np.zeros(6)
            objective[index] = sign
            optimum = linprog(objective, A_ub=sparse.vstack([matrix, -matrix]),
                              b_ub=np.r_[cap, cap], bounds=list(zip(low.ravel(), high.ravel())))
            assert optimum.success
            assert first.ravel()[index] <= optimum.x[index] + 1e-12
            assert last.ravel()[index] >= optimum.x[index] - 1e-12
