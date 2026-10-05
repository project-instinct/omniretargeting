"""Default-byte contracts and independently enabled repair ablations."""

from dataclasses import replace
import importlib
from types import SimpleNamespace

import numpy as np
import pytest

from omniretargeting._fast.runtime_options import RuntimeOptions
from omniretargeting._fast.repair_limits import RepairLimits, LOCAL_ANGLE
from omniretargeting._fast.contact_guard import ContactGuard
from omniretargeting._fast import repair_ablation
from tests.test_fast_contact_guard import _acquired
from tests.test_fast_backend import _objective

FLAGS = ('abl_repair_restore_layout', 'abl_repair_local_angle',
         'abl_repair_ratchet', 'abl_contact_regression')


def options(monkeypatch, **changes):
    flags = replace(RuntimeOptions(), **changes)
    for name in ('runtime_options', 'mesh_repair', 'repair_limits', 'repair_step',
                 'repair_restoration', 'repair_fast_gate', 'repair_triage', 'repair_ablation'):
        monkeypatch.setattr(importlib.import_module('omniretargeting._fast.'+name), 'OPTIONS', flags)
    return flags


@pytest.mark.parametrize('name', FLAGS)
def test_each_experiment_is_default_off_and_independently_enabled(monkeypatch, name):
    monkeypatch.delenv('OMNIRETARGETING_FAST_DISABLE', raising=False)
    monkeypatch.setenv('OMNIRETARGETING_FAST_ENABLE', name)
    expected = RuntimeOptions().describe()
    assert all(not expected[key] for key in FLAGS)
    expected[name] = True
    assert RuntimeOptions.from_environment().describe() == expected


def test_disabled_original_repair_matches_pre_ablation_qpos_bytes(monkeypatch):
    from omniretargeting._fast.mesh_repair import repair_original_meshes
    from omniretargeting._fast import repair_contact_ablation
    from tests.test_limb_repair_scope import _floor_problem
    options(monkeypatch)
    monkeypatch.setenv('OMNIRETARGETING_GPU_GEOMETRY', '0')

    def forbidden(*args, **kwargs):
        raise AssertionError('Disabled experiment entered its new path')

    for name in ('collision_budgets', 'regressions', 'group_max'):
        monkeypatch.setattr(repair_ablation, name, forbidden)
    for name in ('prepare', 'add', 'check'):
        monkeypatch.setattr(repair_contact_ablation, name, forbidden)
    obj, poses, scene, _ = _floor_problem(.12)
    actual, report = repair_original_meshes(obj, poses, scene_query=scene)
    # Captured before adding any ablation implementation, in this same
    # five-frame unit fixture. Hex literals preserve every float64 bit.
    row = [0., 0., float.fromhex('0x1.051eb851eb852p-1'), 1., 0., 0., 0., 0.,
           float.fromhex('0x1.666663bbf377fp-2'), -float.fromhex('0x1.80000055e09a7p-1'),
           float.fromhex('0x1.999763fd90c54p-2'), float.fromhex('0x1.66666b6fd0732p-2'),
           -float.fromhex('0x1.7ffffdf0c74b9p-1'), float.fromhex('0x1.99455d8642662p-2'), 0.]
    expected = np.tile(row, (5, 1))
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert actual.tobytes() == expected.tobytes()
    assert [r['alpha'] for r in report['history']] == [1., 0., 1., 0., 1., 0., 1., 0.]
    assert report['local_angle_budget_rad'] == .05


@pytest.mark.parametrize('depth', [0., 1e-6, .0004, .01, np.nan])
def test_disabled_candidate_calls_legacy_prefix_and_predicate(monkeypatch, depth):
    from omniretargeting._fast.mesh_repair import _regressed
    options(monkeypatch)
    key = (0, 'static', 0)
    protected = {'depths': {key: .0003}}
    checked = dict(depths={key: depth}, failures=[], complete=True)
    calls = []

    def measure(candidate, **kwargs):
        calls.append(kwargs)
        return checked

    output = repair_ablation.measure_candidate(None, SimpleNamespace(measure=measure), None,
        None, protected, None, .001, frames=None)
    assert calls == [dict(protected_depths=protected['depths'], frames=None)]
    assert output[0] is checked
    assert output[1] == ([key] if _regressed(depth, .0003) else [])
    assert output[3] == (not (depth**2 >= .001 - 1e-10))


@pytest.mark.parametrize('enabled', [False, True])
def test_angle_flag_changes_joint_box_and_body_rotation_cone_together(monkeypatch, enabled):
    options(monkeypatch, abl_repair_local_angle=enabled)
    obj, poses, _ = _objective(temporal=True)
    limits = RepairLimits(obj, poses)
    # Isolate locality from the independent, unchanged free-tracking guard.
    limits.tracking_budgets = [np.full_like(value, np.inf) for value in limits.tracking_budgets]
    assert LOCAL_ANGLE == .05  # Acquisition's imported constant stays fixed.
    assert limits.local_angle == (.10 if enabled else .05)
    moved = poses.copy()
    moved[:, obj.coords.qi[-1]] += .075
    assert limits.check(moved)[0] == enabled
    bounds = limits.constraints(poses)
    body = [allowed for _, residual, allowed in bounds.norms
            if residual.shape == (len(poses)*len(limits.bodies), 3)]
    assert len(body) == 1
    assert np.all(body[0] == limits.local_angle + 5e-8)
    # Body posture is independently bounded, even if every joint is in box.
    limits.lower.fill(-np.inf)
    limits.upper.fill(np.inf)
    assert limits.check(moved)[0] == enabled


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('run', [6, 7, 8])
def test_restoration_layout_uses_full_acceptance_deadband_and_history(monkeypatch, enabled, run):
    from omniretargeting.pose_relations import RelationEnvelope, LAYOUT_FLIP_MARGIN, _sustained, LAYOUT_FLIP_WINDOW_FRAMES
    from omniretargeting._fast import repair_layout_rows
    options(monkeypatch, abl_repair_restore_layout=enabled)
    obj, poses, _ = _objective(temporal=True)
    width, count = obj.width, len(poses)
    value = np.full((count, 1, 3), .1)
    # Run crosses a 16-frame chunk boundary; the enabled path keeps it whole.
    value[14:14+run, 0, 0] = -LAYOUT_FLIP_MARGIN - .01
    envelope = object.__new__(RelationEnvelope)
    envelope.layout_sign = np.ones_like(value)
    jac = np.ones((*value.shape, width))
    envelope.measure = lambda *args, **kwargs: (None, None, (None, None))
    envelope.layout = lambda *args: (value, jac)
    seen = []

    def rows(poses, *, restore=False):
        seen.append(restore)
        return envelope.linear_bounds(None, width, restore=restore), []

    shape = SimpleNamespace(repair_bounds=rows, accepts=lambda poses: True)
    obj.source_shape = shape
    # Observe that an enabled restoration never invokes the chunk cache.
    cached = []
    def cache(*args, **kwargs):
        cached.append(kwargs['restore'])
        return None
    monkeypatch.setattr(repair_layout_rows, 'layout_bounds', cache)
    limits = RepairLimits(obj, poses)
    from scipy import sparse
    projection = sparse.eye(count*limits.width, format='csc')
    bounds = limits.constraints(poses, restore_layout=True, projection=projection)
    assert seen == [not enabled]
    assert cached == ([] if enabled else [True])
    floor = (np.where(_sustained(value < -LAYOUT_FLIP_MARGIN, LAYOUT_FLIP_WINDOW_FRAMES),
                      0., -LAYOUT_FLIP_MARGIN) if enabled else np.zeros_like(value))
    expected = (floor-value-envelope.SLACK/2).ravel()
    assert bounds.lower[-1].tobytes() == expected.tobytes()


def support_objective():
    model = SimpleNamespace(geom_bodyid=np.array([1, 1, 2]), body_weldid=np.arange(3))
    eff = dict(body=1, name='contact', active=np.ones(1, bool), managed=np.zeros(1, bool),
               normal=np.array([[0., 0., 1.]]))
    return SimpleNamespace(model=model, effectors=[eff], surfaces=[],
        tasks=SimpleNamespace(contact_phase=SimpleNamespace(role=lambda name, frame: 'support')))


def measurement(depths, *, direction=(0., 0., 1.)):
    return dict(depths=depths, rows=[(key, dict(clearance=-depth, direction=np.array(direction)))
                                   for key, depth in depths.items()], failures=[], complete=True)


@pytest.mark.parametrize('increase', [.0005, .00050001])
def test_ratchet_existing_shallow_key_has_half_millimetre_best_so_far_cap(monkeypatch, increase):
    options(monkeypatch, abl_repair_ratchet=True)
    obj = support_objective()
    deep, shallow = (0, 'static', 0), (0, 'static', 1)
    before = measurement({deep: .008, shallow: .0004})
    after = measurement({deep: .006, shallow: .0004+increase})
    assert repair_ablation.regressions(obj, before, after, before) == ([] if increase == .0005 else [shallow])
    cap = repair_ablation.collision_budgets(obj, before, before)
    assert cap[1] == .0004 + .0005


@pytest.mark.parametrize('kind,normal,depth,allowed', [
    ('static', (0., 0., 1.), .0009, True),
    ('static', (0., 0., 1.), .00100001, False),
    ('static', (1., 0., 0.), .0001, False),
    ('self', (0., 0., 1.), .0001, False),
])
def test_ratchet_new_key_requires_support_normal_and_stays_below_activation(monkeypatch, kind, normal, depth, allowed):
    options(monkeypatch, abl_repair_ratchet=True)
    obj = support_objective()
    key = (0, kind, 1) if kind == 'static' else (0, 'self', 0, 2)
    before = measurement({(0, 'static', 0): .008})
    after = measurement({key: depth}, direction=normal)
    assert (not repair_ablation.regressions(obj, before, after, before)) == allowed


def test_ratchet_cannot_hide_a_new_side_witness_behind_improved_geom_minimum(monkeypatch):
    options(monkeypatch, abl_repair_ratchet=True)
    obj, key = support_objective(), (0, 'static', 0)
    before = measurement({key: .008})
    after = measurement({key: .006})
    after['rows'].append((key, dict(clearance=-.0001, direction=np.array([1., 0., 0.]))))
    assert repair_ablation.regressions(obj, before, after, before) == [key]


def test_ratchet_accepts_group_max_descent_without_requiring_squared_sum_descent(monkeypatch):
    options(monkeypatch, abl_repair_ratchet=True)
    obj = support_objective()
    deep = (0, 'static', 0)
    shallow = [(0, 'moving', 0, 'closed_surface', component, 0) for component in range(10)]
    before = measurement({deep: .0011, **{key: .0004 for key in shallow}})
    after = measurement({deep: .001099, **{key: .0009 for key in shallow}})
    assert sum(after['depths'][k]**2 for k in after['depths']) > sum(v**2 for v in before['depths'].values())
    calls = []
    def query(candidate, **kwargs):
        calls.append(kwargs)
        return after
    output = repair_ablation.measure_candidate(obj, SimpleNamespace(measure=query), None,
        before, before, 1, sum(v**2 for v in before['depths'].values()), frames=None)
    assert calls == [dict(protected_depths=None, frames=None)]
    assert output[1] == [] and output[3]
    after['depths'][deep] = .0011
    assert not repair_ablation.measure_candidate(obj, SimpleNamespace(measure=query), None,
        before, before, 1, 0., frames=None)[3]


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('axis,move,allowed', [(0, .0008, True), (2, .0008, True),
                                              (0, .0011, False), (2, -.0002, False)])
def test_contact_full_regression_cap_keeps_one_sided_normal_guard(monkeypatch, enabled, axis, move, allowed):
    options(monkeypatch, abl_contact_regression=enabled)
    obj, poses = _acquired()
    obj.cfg['endpoint_regression_m'] = 0.
    obj.effectors[0]['normal'][:] = [0., 0., 1.]
    guard = ContactGuard(obj, poses, preserve_positions=True)
    candidate = poses.copy()
    candidate[:, axis] += move
    assert guard.accepts(candidate) == (enabled and allowed)
    # Whole and absolute-frame window APIs agree, including inward rejection.
    assert guard.accepts(candidate, frames=range(3, 15)) == (enabled and allowed)
    linear, cones = guard.repair_bounds(poses)
    from tests.test_repair_guard_rows import violation
    delta = obj.coords.difference(poses, candidate).ravel()
    assert (violation((linear, cones), delta) <= 1e-10) == (enabled and allowed)


def test_contact_existing_signed_penetration_may_exit_but_cannot_deepen(monkeypatch):
    options(monkeypatch, abl_contact_regression=True)
    obj, poses = _acquired()
    obj.effectors[0]['normal'][:] = [0., 0., 1.]
    obj.effectors[0]['contact'][:, 2] += .002
    guard = ContactGuard(obj, poses)
    candidate = poses.copy()
    candidate[:, 2] += .001
    assert guard.accepts(candidate)
    candidate[:, 2] = poses[:, 2] - .0001
    assert not guard.accepts(candidate)
    assert guard.diagnostics(candidate)['max_inward_regression_m'] == pytest.approx(.0001)


def test_repair_contact_rows_and_check_reject_inward_slack_spending(monkeypatch):
    from omniretargeting._fast.repair_contact_ablation import check
    options(monkeypatch, abl_contact_regression=True)
    obj, poses = _acquired()
    obj.effectors[0]['normal'][:] = [0., 0., 1.]
    obj.cfg['endpoint_regression_m'] = 0.
    limits = RepairLimits(obj, poses)
    assert limits.contact_tolerances[0] == 0.
    _, bodies, budget = limits.points[0]
    anchor_budget, target_budget = limits.contact_budgets(0, bodies, budget)
    assert np.all(anchor_budget == .002) and np.all(target_budget == .001)
    points = limits.points[0][0].copy()
    points[..., 0] += .0008
    assert check(limits, 0, points) is None
    points[..., 2] -= .0002
    assert check(limits, 0, points)['reason'] == 'contact into supporting surface'
    bounds = limits.constraints(poses)
    assert any(np.isposinf(upper).all() and len(lower) == len(poses)
               and np.all(lower == -5e-8) for lower, upper in zip(bounds.lower, bounds.upper))


@pytest.mark.parametrize('preserve', [False, True])
def test_contact_experiment_adds_one_millimetre_to_existing_cap_without_rebasing(monkeypatch, preserve):
    options(monkeypatch, abl_contact_regression=True)
    obj, poses = _acquired(moving=True)
    obj.cfg['endpoint_regression_m'] = .0002
    obj.effectors[0]['normal'][:] = [0., 0., 1.]
    guard = ContactGuard(obj, poses, preserve_positions=preserve)
    assert guard.groups[0].tolerance == .0002
    for move, accepted in ((.0011, True), (.0013, False), (.0011, True)):
        candidate = poses.copy()
        candidate[:, 0] += move
        assert guard.accepts(candidate) == accepted
        from tests.test_repair_guard_rows import violation
        delta = obj.coords.difference(poses, candidate).ravel()
        assert (violation(guard.repair_bounds(poses), delta) <= 1e-10) == accepted
    # A better contact does not refresh the frozen allowance or plane floor.
    assert guard.groups[0].tolerance == .0002


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('restoration', [False, True])
def test_ratchet_proposal_and_restoration_use_the_same_experimental_caps(monkeypatch, enabled, restoration):
    from scipy import sparse
    from omniretargeting._fast.repair_step import solve_step
    from omniretargeting._fast.repair_restoration import restore_repair
    options(monkeypatch, abl_repair_ratchet=enabled)
    obj = support_objective()
    keys = [(0, 'static', 0), (0, 'static', 1), (0, 'moving', 0, 'closed_surface', 0, 1)]
    baseline = measurement(dict(zip(keys, [.008, .0004, 0.])))
    phi = np.array([-.008, -.0004, .0002])
    measured = dict(baseline, rows=[(key, dict(row, clearance=value))
                                  for (key, row), value in zip(baseline['rows'], phi)])
    collision = sparse.csr_matrix(np.ones((3, 1)))
    captured = []

    class Captured(Exception):
        pass

    class Bounds:
        def add(self, matrix, lower, upper):
            captured.append((matrix, lower, upper))
            raise Captured

    limits = SimpleNamespace(objective=obj, constraints=lambda *args, **kwargs: Bounds(),
                             coords=SimpleNamespace(qi=np.array([0])), check=lambda poses: (False, {}))
    geometry = SimpleNamespace(linear_rows=lambda measurement: (collision, phi))
    origin, candidate = np.zeros((1, 1)), np.array([[.001]])
    projection = sparse.eye(1, format='csc')
    with pytest.raises(Captured):
        if restoration:
            restore_repair(limits, origin, candidate, geometry, measured, baseline, projection)
        else:
            solve_step(limits, origin, geometry, measured, baseline, projection,
                       collision_sigma=.005, local_angle=.05)
    matrix, lower, upper = captured[0]
    assert matrix is collision and np.isposinf(upper)
    caps = np.array([.008 + 1e-6 + 5e-5, .0009, .001]) if enabled else np.array([.008, .0004, 0.])
    expected = -np.maximum(caps, -phi) - phi
    if restoration:
        expected -= collision @ candidate.ravel()
    np.testing.assert_allclose(lower, expected, atol=1e-18, rtol=0)


@pytest.mark.parametrize('evidence', ['missing_policy', 'touch_role', 'zero_normal', 'other_body'])
def test_ratchet_never_relaxes_without_matching_support_evidence(monkeypatch, evidence):
    options(monkeypatch, abl_repair_ratchet=True)
    obj = support_objective()
    if evidence == 'missing_policy':
        obj.tasks.contact_phase = None
    elif evidence == 'touch_role':
        obj.tasks.contact_phase.role = lambda *args: 'touch'
    elif evidence == 'zero_normal':
        obj.effectors[0]['normal'][:] = 0.
    else:
        obj.effectors[0]['body'] = 2
    key = (0, 'static', 0)
    before = measurement({key: 0.})
    checked = measurement({key: .0001})
    assert repair_ablation.regressions(obj, before, checked, before) == [key]
