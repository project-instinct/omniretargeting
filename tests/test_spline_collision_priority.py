"""Small coefficient/geometry fixtures only; no production FK, clips or benches."""

from dataclasses import replace
from types import SimpleNamespace

import clarabel
import mujoco
import numpy as np
import pytest
from scipy import sparse
from scipy.spatial.transform import Rotation

import omniretargeting._fast.spline_collision_priority as priority
from omniretargeting._fast.mesh_repair import OriginalGeometry
from omniretargeting._fast.priority_qp import solve_priority
from omniretargeting._fast.qp_assembly import SoftTerms, assemble_program, quadratic
from omniretargeting._fast.qp_guard_rows import HardBlock, intervals, norms
from omniretargeting._fast.qp_types import IterationModel, settings
from omniretargeting._fast.spline_embedding import CorrectionSpline
from omniretargeting._sequence.coordinates import Coordinates


STATIC = (0, 'static', 21)
SELF = (0, 'self', 11, 24)
MOVING = (0, 'moving', 7, 'closed', 3, 25)


def measured(rows=(), **changes):
    result, depths = [], {}
    for key, clearance, jacobian in rows:
        row = dict(clearance=clearance, jacobian=np.asarray(jacobian, float))
        result.append((key, row))
        depths[key] = max(depths.get(key, 0.), -clearance, 0.)
    return dict(rows=result, depths=depths, failures=[], complete=True, **changes)


def linear_rows(model):
    # Call the actual full-frame production interface, without constructing a
    # geometry provider, kinematics object or any real collision query.
    count, width = model.step_lower.shape
    owner = SimpleNamespace(width=width, objective=SimpleNamespace(frames=count))
    return lambda measurement: OriginalGeometry.linear_rows(owner, measurement)


def model(size=1, *, gradient=None, hessian=None, lower=-1., upper=1.,
          blocks=None, shape=None, soft=None):
    soft = SoftTerms(size) if soft is None else soft
    if blocks is None:
        blocks = [intervals(sparse.eye(size), np.full(size, lower), np.full(size, upper))]
    program = assemble_program(blocks, soft, size)
    gradient = np.ones(size) if gradient is None else np.asarray(gradient, float)
    hessian = sparse.eye(size, format='csc') if hessian is None else sparse.csc_matrix(hessian)
    shape = (1, size) if shape is None else shape
    return IterationModel(np.zeros(shape), None, np.full(shape, lower), np.full(shape, upper),
                          None, hessian, gradient, np.ones(size),
                          max(float(np.max(abs(gradient), initial=0.)), 1.), program, soft)


def proposal(problem, measurement, *, mapping=None, selected=None, remaining=lambda: 60.):
    mapping = sparse.eye(problem.step_lower.size, format='csr') if mapping is None else mapping
    selected = np.arange(mapping.shape[1]) if selected is None else selected
    return priority.collision_first_proposal(problem, mapping, selected, measurement,
                                              linear_rows(problem), .001, remaining)


def assert_counts(diagnostics, problem):
    assert diagnostics['conic_calls'] == len(diagnostics['clarabel_calls'])
    assert len(problem.solve_diagnostics) == (diagnostics['conic_calls']
                                             + len(diagnostics['construction_failures']))
    assert diagnostics['solve_seconds'] == pytest.approx(sum(
        attempt['solve_seconds'] for attempt in diagnostics['clarabel_calls']))


def test_identity_epigraph_rows_and_gradient_equal_original_full_frame_interface():
    problem = model(4, shape=(2, 2))
    data = measured([(STATIC, -.2, [2., -1.]), (STATIC, -.1, [-1., 3.]),
                     ((1, 'self', 4, 8), -.3, [1., 2.])])
    mapping = sparse.eye(4, format='csr')
    layer = priority.build_collision_layer(data, linear_rows(problem), mapping, np.arange(4))
    J, phi = linear_rows(problem)(data)
    ordinary = SoftTerms(4)
    ordinary.add((-J, phi, np.ones(3), [('penetration', key) for key, _ in data['rows']]))
    original = assemble_program(problem.program.hard_blocks, ordinary, 4)
    actual = priority._with_collision_rows(problem.program, layer)
    np.testing.assert_array_equal(actual.A.toarray(), original.A.toarray())
    np.testing.assert_array_equal(actual.b, original.b)
    assert actual.nonneg == original.nonneg and actual.soc == original.soc
    assert len(layer.keys) == 2
    assert layer.ownership[STATIC]['witnesses'] == [0, 1]
    assert layer.ownership[(1, 'self', 4, 8)]['geoms'] == (4, 8)
    # At zero the first static row owns the max; gradient is -J for those
    # unique active witnesses, rather than a sum over repeated static rows.
    direction = np.array([.03, -.02, -.01, .04])
    epsilon = 1e-6
    finite = (sum(layer.values(epsilon * direction).values())
              - sum(layer.values(-epsilon * direction).values())) / (2 * epsilon)
    assert finite == pytest.approx(float(-(J[[0, 2]] @ direction).sum()), abs=2e-10)
    np.testing.assert_array_equal(layer.jacobian.toarray(), J.toarray())


def test_identity_solution_matches_existing_strict_full_frame_hierarchy():
    data = measured([(STATIC, -.3, [1., 0.]), (SELF, -.2, [0., 1.]),
                     (SELF, -.1, [0., .5])])
    problem = model(2, gradient=[2., 1.], upper=.1)
    step, diagnostic = proposal(problem, data)
    assert step is not None and diagnostic['certification']['certified']
    J, phi = linear_rows(problem)(data)
    soft = SoftTerms(2)
    soft.add((-J, phi, np.ones(3), [('penetration', key) for key, _ in data['rows']]))
    ordinary = model(2, gradient=[2., 1.], upper=.1, soft=soft)
    P, c = quadratic(ordinary, .001)
    oracle = solve_priority(ordinary, P, c, strict=True)
    assert oracle is not None and oracle.certificate['certified']
    np.testing.assert_allclose(step, oracle.step, atol=5e-8)
    assert diagnostic['stage_optima']['collision'] == pytest.approx(.3, abs=2e-8)
    assert_counts(diagnostic, problem)


@pytest.mark.parametrize('upper, expected_depth', [(.1, .1), (.3, 0.)])
def test_huge_tracking_cost_cannot_purchase_a_worse_collision(upper, expected_depth):
    problem = model(gradient=[2e7], hessian=[[3e7]], upper=upper)
    data = measured([(STATIC, -.2, [1.])])
    step, diagnostic = proposal(problem, data)
    assert step is not None and step[0] >= .2 - expected_depth - 4e-8
    assert diagnostic['predicted_per_key'][STATIC] == pytest.approx(expected_depth, abs=4e-8)
    assert diagnostic['predicted_tracking_decrease'] < 0
    assert diagnostic['predicted_improvement'] > .09
    assert diagnostic['certification']['certified']
    assert not diagnostic['certification']['collision_free_certificate']
    assert diagnostic['clarabel_calls'][0]['P_nnz'] == 0  # pure first-level LP
    assert diagnostic['clarabel_calls'][0]['stage'] == 'collision'
    assert diagnostic['clarabel_calls'][-1]['stage'] == 'tracking'
    assert_counts(diagnostic, problem)


def test_shared_maximum_witnesses_not_average_depth_or_duplicate_weight():
    samples = [(STATIC, -.2, [1.]), (STATIC, -.1, [-1.]), (SELF, -.3, [0.])]
    first, many = model(gradient=[100.]), model(gradient=[100.])
    a, da = proposal(first, measured(samples))
    b, db = proposal(many, measured(samples + [samples[0]] * 13))
    assert a is not None and b is not None
    np.testing.assert_allclose(a, b, atol=5e-8)
    assert a[0] == pytest.approx(.05, abs=5e-8)
    assert da['epigraphs'] == db['epigraphs'] == 2
    assert da['predicted_per_key'][STATIC] == pytest.approx(.15, abs=3e-8)
    assert da['predicted_per_key'][SELF] == pytest.approx(.3, abs=3e-8)
    assert da['stage_optima']['collision'] == pytest.approx(.45, abs=3e-8)
    assert db['stage_optima']['collision'] == pytest.approx(.45, abs=3e-8)
    assert len(db['key_ownership'][STATIC]['witnesses']) == 15
    assert_counts(db, many)


def test_whole_soc_and_all_hard_rows_survive_collision_and_tracking():
    box = intervals(sparse.eye(2), np.full(2, -.3), np.full(2, .3))
    ball = norms(sparse.eye(2), np.zeros((1, 2)), np.array([.1]))
    constant = HardBlock(sparse.csr_matrix((1, 2)), np.array([.7]))
    problem = model(2, gradient=[3., 2.], blocks=[box, ball, constant])
    data = measured([(STATIC, -.3, np.ones(2) / np.sqrt(2))])
    layer = priority.build_collision_layer(data, linear_rows(problem), sparse.eye(2), np.arange(2))
    combined = priority._with_collision_rows(problem.program, layer)
    assert combined.hard_blocks is problem.program.hard_blocks
    assert combined.soc == problem.program.soc == (3,)
    np.testing.assert_array_equal(combined.A[-3:, :2].toarray(), problem.program.A[-3:].toarray())
    np.testing.assert_array_equal(combined.b[-3:], problem.program.b[-3:])
    step, diagnostic = proposal(problem, data)
    assert step is not None
    assert np.linalg.norm(step) <= .1 + 2e-8
    assert diagnostic['predicted_per_key'][STATIC] == pytest.approx(.2, abs=4e-8)
    for attempt in diagnostic['clarabel_calls']:
        assert attempt['soc'] == [3] and attempt['A_shape'][1] == 3
    # A component box would falsely permit (.1, .1), giving a smaller depth.
    assert diagnostic['stage_optima']['collision'] > .3 - .1 * np.sqrt(2) + .01
    assert_counts(diagnostic, problem)


def test_existing_tracking_epigraph_and_its_cone_are_preserved():
    soft = SoftTerms(2)
    soft.add((sparse.csr_matrix([[0., 1.]]), np.array([0.]), np.array([5.]),
              [('clearance_margin', 0, 1)]))
    soft.add((sparse.csr_matrix([[0., 1.], [0., 0.], [0., 0.]]),
              np.zeros((1, 3)), np.array([.2]), np.array([1.]),
              [('clearance_margin', 0, 2)]), cone=True)
    problem = model(2, gradient=[1., -1.], soft=soft)
    original = problem.program.A.copy(), problem.program.b.copy(), list(soft.heads)
    data = measured([(STATIC, -.2, [1., 0.])])
    step, diagnostic = proposal(problem, data)
    assert step is not None
    assert step[0] == pytest.approx(.2, abs=4e-8)
    assert step[1] == pytest.approx(0., abs=3e-7)
    assert diagnostic['collision_program']['variables'] == 5
    assert diagnostic['tracking_program']['soc'] == [4]
    np.testing.assert_array_equal(problem.program.A.toarray(), original[0].toarray())
    np.testing.assert_array_equal(problem.program.b, original[1])
    for (actual, indices), (before, old_indices) in zip(soft.heads, original[2]):
        np.testing.assert_array_equal(actual, before)
        np.testing.assert_array_equal(indices, old_indices)


def test_strict_fixed_root_and_selected_column_order_cannot_leak_root_dofs():
    mapping, selected = sparse.eye(6, format='csr'), np.array([5, 2])
    lower, upper = np.zeros(6), np.zeros(6)
    lower[[2, 5]], upper[[2, 5]] = -.2, .2
    full_box = intervals(mapping, lower, upper)
    mapped_box = replace(full_box, A=full_box.A[:, selected])
    problem = model(2, shape=(2, 3), gradient=[1., 1.], blocks=[mapped_box])
    data = measured([(STATIC, -.2, [1., 0., 0.]),
                     ((1, 'static', 21), -.1, [0., 0., 1.])])
    layer = priority.build_collision_layer(data, linear_rows(problem), mapping, selected)
    np.testing.assert_array_equal(layer.jacobian.toarray(), [[0., 0.], [1., 0.]])
    step, diagnostic = proposal(problem, data, mapping=mapping, selected=selected)
    assert step is not None
    full = np.zeros(6)
    full[selected] = step
    np.testing.assert_array_equal(full[[0, 1, 3, 4]], np.zeros(4))
    assert full[5] == pytest.approx(.1, abs=4e-8)
    assert diagnostic['predicted_per_key'][STATIC] == pytest.approx(.2)
    assert diagnostic['stage_optima']['collision'] == pytest.approx(.2, abs=3e-8)
    assert diagnostic['collision_program']['rows'] >= full_box.A.shape[0] + 4


@pytest.mark.parametrize('identity', [True, False])
def test_nonzero_rotation_mapped_clearance_and_layer_gradient_finite_difference(identity):
    tiny = mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body><freejoint/>
      <geom type="sphere" size=".1" mass="1"/><body><joint type="hinge"/>
      <geom type="sphere" size=".05" mass="1"/></body></body></worldbody></mujoco>''')
    coords, count = Coordinates(tiny), 5
    reference = np.zeros((count, tiny.nq))
    reference[:, 3] = 1.
    spline = CorrectionSpline(coords, reference, 30, identity=identity)
    rng = np.random.default_rng(19)
    coefficients = rng.normal(scale=.2, size=spline.coefficients.shape)
    coefficients[:, spline.angular] += [.7, -.4, .5]
    poses = spline.embed(coefficients)
    point, normal = np.array([.2, -.3, .4]), np.array([.5, .1, .8])
    skew = np.array([[0., -point[2], point[1]], [point[2], 0., -point[0]],
                     [-point[1], point[0], 0.]])

    def clearance(q):
        rotated = Rotation.from_quat(q[:, 3:7], scalar_first=True).apply(point)
        return (rotated + q[:, :3]) @ normal + q[:, 7] - 10.

    rotation = Rotation.from_quat(poses[:, 3:7], scalar_first=True).as_matrix()
    jac = np.zeros((count, coords.width))
    jac[:, :3] = normal
    jac[:, spline.angular] = -np.einsum('i,nij,jk->nk', normal, rotation, skew)
    jac[:, coords.scalar_cols] = 1.
    phi = clearance(poses)
    data = measured([((frame, 'static', 21), phi[frame], jac[frame]) for frame in range(count)])
    problem = model(coefficients.size, shape=(count, coords.width))
    mapping = spline.jacobian(coefficients)
    # Permute a subset, including rotation, to catch accidental full/prefix
    # coefficient selection or treating the rotation differential as identity.
    selected = rng.permutation(coefficients.size)[:max(5, coefficients.size // 2)]
    layer = priority.build_collision_layer(data, linear_rows(problem), mapping, selected)
    direction = rng.normal(size=len(selected))
    full = np.zeros(coefficients.size)
    full[selected] = direction
    full = full.reshape(coefficients.shape)
    epsilon = 2e-7
    plus = clearance(spline.embed(coefficients + epsilon * full))
    minus = clearance(spline.embed(coefficients - epsilon * full))
    np.testing.assert_allclose(layer.jacobian @ direction, (plus - minus) / (2 * epsilon),
                               rtol=2e-7, atol=2e-8)
    finite_layer = (np.maximum(-plus, 0.).sum() - np.maximum(-minus, 0.).sum()) / (2 * epsilon)
    assert float(-(layer.jacobian @ direction).sum()) == pytest.approx(finite_layer, abs=5e-8)
    np.testing.assert_array_equal(spline.reference, reference)
    assert not spline.reference.flags.writeable


def test_nonlinear_layer_includes_every_original_key_and_new_collision():
    other_self = (0, 'self', 11, 30)
    next_frame = (1, 'static', 21)
    baseline = measured([(STATIC, -.03, [1.]), (STATIC, -.01, [2.]),
                         (SELF, -.02, [1.]), (MOVING, -.04, [1.])])
    before = priority.measurement_collision_layer(baseline)
    assert before['value'] == pytest.approx(.09)
    candidate = measured([(STATIC, -.01, [1.]), (SELF, -.02, [1.]),
                          (MOVING, -.03, [1.]), (other_self, -.025, [1.]),
                          (next_frame, -.015, [1.])])
    after = priority.measurement_collision_layer(candidate, expected_keys=before['per_key'])
    assert after['valid'] and after['value'] == pytest.approx(.1)
    assert len(after['per_key']) == 5
    # Sum over all keys records a regression even when an original static key
    # improves and ordinary tracking cost is lower.
    assert after['value'] > before['value']
    assert after['collision_free_certificate'] is False


def test_complete_omitted_expected_key_is_zero_layer_but_never_safety_certificate():
    complete = measured()
    value = priority.measurement_collision_layer(complete, expected_keys=[STATIC])
    assert value['valid'] and value['per_key'][STATIC] == 0.
    assert value['status'] == 'zero_known_depth' and value['value'] == 0.
    assert value['collision_free_certificate'] is False
    complete['failures'] = ['unknown_query']
    unknown = priority.measurement_collision_layer(complete, expected_keys=[STATIC])
    assert not unknown['valid'] and unknown['value'] is None
    assert STATIC not in unknown['per_key']


@pytest.mark.parametrize('change', [dict(failures=['unknown_query']), dict(complete=False),
                                   dict(reliable=False), dict(frame_complete=[False]),
                                   dict(query_scope={'requested_frames': [1]}),
                                   dict(depths={STATIC: np.nan})])
def test_unknown_measurement_does_not_start_optimization_or_claim_zero(change):
    problem = model()
    data = measured([(STATIC, -.2, [1.])])
    data.update(change)
    step, diagnostic = proposal(problem, data)
    assert step is None and diagnostic['status'] == 'unknown_geometry'
    assert diagnostic['measured_layer']['value'] is None
    assert diagnostic['unknown_reasons']
    assert diagnostic['conic_calls'] == diagnostic['solver_constructions'] == 0
    assert not diagnostic['certification']['certified']
    assert_counts(diagnostic, problem)


@pytest.mark.parametrize('missing', ['clearance', 'jacobian', 'missing_jacobian', 'deepest', 'complete'])
def test_incomplete_collision_evidence_is_unknown(missing):
    problem = model()
    data = measured([(STATIC, -.2, [1.])])
    if missing in {'clearance', 'jacobian'}:
        data['rows'][0][1][missing] = np.nan if missing == 'clearance' else np.array([np.nan])
    elif missing == 'missing_jacobian':
        del data['rows'][0][1]['jacobian']
    elif missing == 'deepest':
        data['rows'] = []
    else:
        del data['complete']
    step, diagnostic = proposal(problem, data)
    assert step is None and diagnostic['status'] == 'unknown_geometry'
    assert diagnostic['conic_calls'] == 0
    assert not diagnostic['certification']['certified']


def test_zero_collision_has_explicit_skip_status_and_no_pass():
    problem = model()
    step, diagnostic = proposal(problem, measured([(STATIC, .01, [1.])]))
    assert step is None and diagnostic['status'] == 'no_known_penetration'
    assert diagnostic['conic_calls'] == 0 and not diagnostic['certification']['certified']


def test_unfeasible_constant_hard_row_is_never_removed():
    problem = model(blocks=[HardBlock(sparse.csr_matrix((1, 1)), np.array([-1.]))])
    step, diagnostic = proposal(problem, measured([(STATIC, -.2, [1.])]))
    assert step is None and diagnostic['status'] == 'solver_failed'
    assert diagnostic['conic_calls'] == 1
    assert diagnostic['clarabel_calls'][0]['status'] == 'PrimalInfeasible'
    assert not diagnostic['certification']['certified']
    assert_counts(diagnostic, problem)


def test_deadline_before_entry_does_not_assemble_or_construct(monkeypatch):
    monkeypatch.setattr(clarabel, 'DefaultSolver', lambda *args: pytest.fail('started expired solver'))
    problem = model()
    step, diagnostic = proposal(problem, measured([(STATIC, -.2, [1.])]), remaining=lambda: 0.)
    assert step is None and diagnostic['status'] == 'budget_exhausted'
    assert diagnostic['conic_calls'] == diagnostic['solver_constructions'] == 0


@pytest.mark.parametrize('expire_after', [1, 2])
def test_started_solver_overrun_discarded_and_no_next_stage(monkeypatch, expire_after):
    original, calls = clarabel.DefaultSolver, []

    def counted(*args):
        solver = original(*args)

        def solve():
            result = solver.solve()
            calls.append(result)
            return result
        return SimpleNamespace(solve=solve)

    monkeypatch.setattr(clarabel, 'DefaultSolver', counted)
    problem = model(upper=.1)
    step, diagnostic = proposal(problem, measured([(STATIC, -.2, [1.])]),
                                remaining=lambda: -.25 if len(calls) >= expire_after else 10.)
    assert step is None and diagnostic['status'] == 'budget_exhausted'
    assert diagnostic['conic_calls'] == len(calls) == expire_after
    assert diagnostic['overrun_seconds'] == .25
    assert diagnostic['clarabel_calls'][-1]['discarded_after_deadline']
    assert not diagnostic['certification']['certified']
    assert_counts(diagnostic, problem)


def test_deadline_between_levels_prevents_tracking_solver(monkeypatch):
    original, state = clarabel.DefaultSolver, dict(solved=False, checks=0)

    def counted(*args):
        solver = original(*args)

        def solve():
            result = solver.solve()
            state['solved'] = True
            return result
        return SimpleNamespace(solve=solve)

    def remaining():
        if state['solved']:
            state['checks'] += 1
        # Permit the after-solve and after-certificate checks, then expire.
        return 0. if state['checks'] >= 3 else 10.

    monkeypatch.setattr(clarabel, 'DefaultSolver', counted)
    problem = model(upper=.1)
    step, diagnostic = proposal(problem, measured([(STATIC, -.2, [1.])]), remaining=remaining)
    assert step is None and diagnostic['status'] == 'budget_exhausted'
    assert diagnostic['conic_calls'] == 1 and diagnostic['solver_constructions'] == 1
    assert 'collision' in diagnostic['stage_optima']
    assert 'tracking' not in diagnostic['stage_optima']
    assert_counts(diagnostic, problem)


def test_deadline_spent_during_construction_prevents_solve(monkeypatch):
    original, state = clarabel.DefaultSolver, dict(expired=False)

    def constructor(*args):
        original(*args)
        state['expired'] = True
        return SimpleNamespace(solve=lambda: pytest.fail('solve started after construction overrun'))

    monkeypatch.setattr(clarabel, 'DefaultSolver', constructor)
    problem = model()
    step, diagnostic = proposal(problem, measured([(STATIC, -.2, [1.])]),
                                remaining=lambda: -.01 if state['expired'] else 10.)
    assert step is None and diagnostic['status'] == 'budget_exhausted'
    assert diagnostic['solver_constructions'] == 1 and diagnostic['conic_calls'] == 0
    assert diagnostic['interrupted_construction']['P_nnz'] == 0
    assert diagnostic['overrun_seconds'] == .01
    assert_counts(diagnostic, problem)


def test_expired_numerical_retry_does_not_start_solver(monkeypatch):
    state = dict(solved=False, checks=0)

    def constructor(*args):
        def solve():
            state['solved'] = True
            return SimpleNamespace(status='InsufficientProgress', iterations=1,
                                   r_prim=.1, r_dual=.1, obj_val=0., obj_val_dual=0.)
        return SimpleNamespace(solve=solve)

    def remaining():
        if state['solved']:
            state['checks'] += 1
        return 0. if state['checks'] >= 2 else 10.

    monkeypatch.setattr(clarabel, 'DefaultSolver', constructor)
    problem = model()
    step, diagnostic = proposal(problem, measured([(STATIC, -.2, [1.])]), remaining=remaining)
    assert step is None and diagnostic['status'] == 'budget_exhausted'
    assert diagnostic['conic_calls'] == diagnostic['solver_constructions'] == 1
    assert diagnostic['clarabel_calls'][0]['status'] == 'InsufficientProgress'
    assert_counts(diagnostic, problem)


def test_precision_retry_reuses_original_settings_and_reports_each_call(monkeypatch):
    original, certificates, options = priority.kkt_residual, [], []
    constructor = clarabel.DefaultSolver

    def first_uncertified(*args):
        certificate = original(*args)
        certificates.append(certificate)
        return dict(certificate, certified=False) if len(certificates) == 1 else certificate

    def counted(*args):
        options.append(args[-1])
        return constructor(*args)

    monkeypatch.setattr(priority, 'kkt_residual', first_uncertified)
    monkeypatch.setattr(clarabel, 'DefaultSolver', counted)
    problem = model(upper=.1)
    step, diagnostic = proposal(problem, measured([(STATIC, -.2, [1.])]), remaining=lambda: 12.5)
    assert step is not None
    assert diagnostic['conic_calls'] == 3
    assert [entry['precision'] for entry in diagnostic['clarabel_calls']] == [False, True, False]
    assert options[0].tol_gap_abs == settings().tol_gap_abs
    assert options[1].tol_gap_abs == settings(precision=True).tol_gap_abs
    assert options[1].direct_solve_method == settings(precision=True).direct_solve_method
    assert all(option.time_limit == 12.5 for option in options)
    assert_counts(diagnostic, problem)


def test_numerical_stall_retry_is_counted_and_uses_existing_retry_settings(monkeypatch):
    constructor, options = clarabel.DefaultSolver, []

    def first_stalls(*args):
        options.append(args[-1])
        if len(options) == 1:
            failure = SimpleNamespace(status='InsufficientProgress', iterations=1,
                                      r_prim=.1, r_dual=.1, obj_val=0., obj_val_dual=0.)
            return SimpleNamespace(solve=lambda: failure)
        return constructor(*args)

    monkeypatch.setattr(clarabel, 'DefaultSolver', first_stalls)
    problem = model(upper=.1)
    step, diagnostic = proposal(problem, measured([(STATIC, -.2, [1.])]))
    assert step is not None and diagnostic['certification']['certified']
    assert diagnostic['clarabel_calls'][0]['status'] == 'InsufficientProgress'
    assert diagnostic['clarabel_calls'][1]['retry']
    assert options[1].tol_feas == settings(retry=True).tol_feas
    assert diagnostic['conic_calls'] == diagnostic['solver_constructions'] == len(options)
    assert_counts(diagnostic, problem)


def test_deadline_after_uncertified_result_prevents_precision_retry(monkeypatch):
    state = dict(expired=False)

    def uncertified(*args):
        state['expired'] = True
        return dict(certified=False, reason='fixture_uncertified')

    monkeypatch.setattr(priority, 'kkt_residual', uncertified)
    problem = model(upper=.1)
    step, diagnostic = proposal(problem, measured([(STATIC, -.2, [1.])]),
                                remaining=lambda: 0. if state['expired'] else 10.)
    assert step is None and diagnostic['status'] == 'budget_exhausted'
    assert diagnostic['conic_calls'] == diagnostic['solver_constructions'] == 1
    assert_counts(diagnostic, problem)


@pytest.mark.parametrize('where', ['construction', 'solve'])
def test_solver_exceptions_have_real_construction_and_solve_counts(monkeypatch, where):
    def constructor(*args):
        if where == 'construction':
            raise RuntimeError('fixture construction failure')

        def solve():
            raise RuntimeError('fixture solve failure')
        return SimpleNamespace(solve=solve)

    monkeypatch.setattr(clarabel, 'DefaultSolver', constructor)
    problem = model()
    step, diagnostic = proposal(problem, measured([(STATIC, -.2, [1.])]))
    assert step is None and diagnostic['status'] == 'solver_failed'
    assert diagnostic['solver_constructions'] == 1
    assert diagnostic['conic_calls'] == (where == 'solve')
    assert len(diagnostic['construction_failures']) == (where == 'construction')
    assert not diagnostic['certification']['certified']
    assert_counts(diagnostic, problem)


def test_repeated_uncertified_results_cannot_return_a_step(monkeypatch):
    monkeypatch.setattr(priority, 'kkt_residual', lambda *args: dict(certified=False))
    problem = model(upper=.1)
    step, diagnostic = proposal(problem, measured([(STATIC, -.2, [1.])]))
    assert step is None and diagnostic['status'] == 'uncertified_subproblem'
    assert diagnostic['conic_calls'] == 2
    assert [entry['precision'] for entry in diagnostic['clarabel_calls']] == [False, True]
    assert 'tracking' not in diagnostic['stage_optima']
    assert_counts(diagnostic, problem)


def test_backtracking_prediction_uses_witness_maxima_again():
    problem = model()
    data = measured([(STATIC, -.2, [1.]), (STATIC, -.1, [-1.])])
    layer = priority.build_collision_layer(data, linear_rows(problem), sparse.eye(1), np.array([0]))
    assert layer.values(np.array([.1]))[STATIC] == pytest.approx(.2)
    assert layer.values(np.array([.05]))[STATIC] == pytest.approx(.15)
    # Interpolating endpoint maxima would report .2 instead of the true .15.
    assert layer.prediction(np.array([.05]))['improvement'] == pytest.approx(.05)


def test_prebuilt_layer_reuses_pullback_and_keeps_caller_prediction(monkeypatch):
    problem = model(upper=.1)
    data = measured([(STATIC, -.2, [1.])])
    mapping, selected = sparse.eye(1, format='csr'), np.array([0])
    layer = priority.build_collision_layer(data, linear_rows(problem), mapping, selected)
    expected, _ = proposal(model(upper=.1), data)
    monkeypatch.setattr(priority, 'build_collision_layer', lambda *args: pytest.fail('rebuilt cached layer'))

    def forbidden_rows(*args):
        pytest.fail('assembled rows for a cached layer')

    step, diagnostic = priority.collision_first_proposal(problem, mapping, selected, data,
                                                         forbidden_rows, .001, lambda: 60., layer=layer)
    assert step is not None and diagnostic['reused_collision_layer']
    np.testing.assert_allclose(step, expected, atol=1e-12)
    assert layer.values(step) == diagnostic['predicted_per_key']
    assert layer.prediction(.5 * step)['after'] == pytest.approx(.15, abs=3e-8)
    assert_counts(diagnostic, problem)


@pytest.mark.parametrize('changed', ['mapping', 'selected', 'clearance', 'jacobian'])
def test_stale_prebuilt_layer_cannot_use_previous_iteration_linearization(changed):
    problem = model(2)
    data = measured([(STATIC, -.2, [1., 0.])])
    mapping, selected = sparse.eye(2, format='csr'), np.array([0, 1])
    layer = priority.build_collision_layer(data, linear_rows(problem), mapping, selected)
    if changed == 'mapping':
        mapping.data[0] = .9
    elif changed == 'selected':
        selected[:] = [1, 0]
    elif changed == 'clearance':
        data['rows'][0][1]['clearance'] = -.3
    else:
        data['rows'][0][1]['jacobian'][0] = .8
    with pytest.raises(ValueError, match='Prebuilt collision layer does not match'):
        priority.collision_first_proposal(problem, mapping, selected, data, linear_rows(problem),
                                          .001, lambda: 60., layer=layer)
    assert not problem.solve_diagnostics
