"""Local hard chatter rows and corrected re-solves retain exact acceptance."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from omniretargeting._fast import repair_sqp as sqp
from omniretargeting._fast.repair_limits import LinearBounds, RepairLimits


class Limits:
    coords = SimpleNamespace(qi=np.array([0]))
    frames = width = 1
    lower, upper = np.array([[-1.]]), np.array([[1.]])
    entry = np.zeros((1, 1))
    movable_body = 0

    def check(self, poses):
        # A nonlinear protected row; the linear tangent alone overshoots it.
        x = poses[0, 0]
        return bool(x + x*x <= .0100001), dict(guard='curved', actual=float(x + x*x))

    def constraints(self, poses, **kwargs):
        x = poses[0, 0]
        bounds = LinearBounds(1)
        bounds.add(sparse.csr_matrix([[1 + 2*x]]), -np.inf, .01 - x - x*x)
        return bounds


def proposal(limits=None):
    p = sqp.SQPProposal.__new__(sqp.SQPProposal)
    p.limits = limits or Limits()
    p.current = np.zeros((1, 1))
    p.step = np.array([[.01]])
    p.projection = sparse.eye(1, format='csc')
    p.fixed_q = None
    p.ledger = sqp.Ledger()
    p.feedback, p.cut_rows, p.cuts = None, 0, []
    p.limit_cuts = []
    p.measured = dict(rows=[((0, 'static', 0), {})], depths={(0, 'static', 0): .02})
    p.protected = dict(depths={(0, 'static', 0): .02})
    p.geometry = SimpleNamespace(model=SimpleNamespace(geom_bodyid=np.arange(1), body_weldid=np.arange(1)),
                                 linear_rows=lambda _: (sparse.csr_matrix([[1.]]), np.array([-.02])))
    return p


def test_minimum_norm_correction_removes_curvature_without_fractions():
    p = proposal()
    record = {}
    iterator = p.candidates(record)
    alpha, candidate = next(iterator)
    iterator.close()
    assert alpha == 1. and p.limits.check(candidate)[0]
    assert .009 < candidate[0, 0] < .01
    assert p.ledger.counts['corrected_resolves'] == p.ledger.counts['fallback_fractions'] == 0
    assert p.ledger.counts['correction'] == 1
    assert p.ledger.counts['limit_rejections'] == 1
    assert not p.limit_cuts


def test_infeasible_correction_retains_origin_without_fraction_ladder():
    class InfeasibleLimits(Limits):
        def constraints(self, poses, **kwargs):
            bounds = super().constraints(poses, **kwargs)
            bounds.add(sparse.csr_matrix([[1.]]), .01, np.inf)
            return bounds
    p = proposal(InfeasibleLimits())
    p.solve = lambda: pytest.fail('an infeasible correction must not trigger another group QP')
    original = p.current.copy()
    assert list(p.candidates({})) == []
    np.testing.assert_array_equal(p.current, original)
    assert p.ledger.counts['fallback_fractions'] == 0
    assert p.ledger.rejection_reasons == {'curved': 1}


def test_repeated_exact_curvature_failure_stops_after_two_corrections(monkeypatch):
    p = proposal()
    p.limits.check = lambda q: (False, dict(guard='curved'))
    p.solve = lambda: pytest.fail('no repeated full group solves')
    visits = []
    def correction(proposal, trial):
        visits.append(trial.copy())
        return np.array([[-1e-5]])
    monkeypatch.setattr(sqp, '_soc', correction)
    assert list(p.candidates({})) == []
    assert len(visits) == 2 and p.soc_attempts == 2
    assert p.ledger.counts['limit_rejections'] == 3
    np.testing.assert_array_equal(p.current, np.zeros((1,1)))


def test_regressed_trial_witness_is_cut_before_fraction_retry(monkeypatch):
    p = proposal()
    p.limits.check = lambda poses: (True, {})
    trial_rows = [((0, 'static', 0), dict(clearance=-.0201, sample_id=27))]
    origins = []

    def measure(trial, *, frames):
        assert list(frames) == [0]
        return dict(rows=trial_rows, depths=p.measured['depths'], failures=[], complete=True)

    def linearize(trial, measurement):
        origins.append(trial.copy())
        assert measurement['rows'] == trial_rows
        return measurement

    p.geometry.measure, p.geometry.linearize = measure, linearize
    p.solve = lambda: (np.array([[.009]]), 'Solved')
    iterator = p.candidates({})
    alpha, trial = next(iterator)
    p.feedback = (trial, dict(rows=trial_rows, failures=[]), [(0, 'static', 0)])
    alpha, cut_trial = next(iterator)
    iterator.close()
    assert alpha == 1. and cut_trial[0, 0] == pytest.approx(.019)
    assert np.array_equal(origins[0], trial)
    assert p.ledger.counts['geometry_resolves'] == 1
    assert p.cut_rows == 1 and p.ledger.counts['cuts'] == 1
    assert p.ledger.counts['fallback_fractions'] == 0
    assert not p.cuts and np.array_equal(p.current, np.zeros((1,1)))


def test_unknown_query_cannot_generate_cut(monkeypatch):
    p = proposal()
    p.limits.check = lambda poses: (True, {})
    p.geometry.linearize = lambda *a: pytest.fail('unknown geometry cannot generate a cut')
    iterator = p.candidates({})
    _, trial = next(iterator)
    p.feedback = (trial, dict(failures=['unknown_query']), [(0, 'static', 0)])
    with pytest.raises(StopIteration):next(iterator)
    assert not p.cuts
    assert p.ledger.counts['fallback_fractions'] == 0


def test_step_qp_preserves_interior_witness_and_trial_cut():
    limits = Limits()
    limits.movable_body = 0
    model = SimpleNamespace(geom_bodyid=np.arange(2), body_weldid=np.arange(2))
    keys = [(0, 'static', 0), (0, 'static', 1)]
    geometry = SimpleNamespace(model=model, linear_rows=lambda _: (
        sparse.csr_matrix([[1.], [-1.]]), np.array([-.02, -.0006])))
    measured = dict(rows=[(key, {}) for key in keys], depths=dict(zip(keys, [.02, .0006])))
    protected = dict(depths=dict(zip(keys, [.02, .001])))
    step, status = sqp.solve_step(limits, limits.entry, geometry, measured, protected,
                                  sparse.eye(1, format='csc'), collision_sigma=.01, local_angle=.1)
    # The inactive key keeps a 0.3 mm margin: only 0.1 mm approach is permitted.
    assert status in ('Solved', 'AlmostSolved')
    assert 0.000099 < step[0, 0] <= .00010001
    geometry.linear_rows = lambda _: (sparse.csr_matrix([[1.]]), np.array([-.02]))
    measured = dict(rows=[(keys[0], {})], depths={keys[0]: .02})
    cut = (sparse.csr_matrix([[-1.]]), np.array([-.002]), np.array([.001]), np.array([.01]))
    step, _ = sqp.solve_step(limits, limits.entry, geometry, measured, protected,
                             sparse.eye(1, format='csc'), collision_sigma=.01, local_angle=.1,
                             cuts=[cut])
    assert .0086 < step[0, 0] <= .00870001


def test_repair_retains_ground_truth_and_complete_limits(monkeypatch):
    from tests.test_fast_backend import _objective
    from tests.test_fast_mesh_repair import _open_object
    monkeypatch.setenv('OMNIRETARGETING_GPU_GEOMETRY', '0')
    objective, poses, _ = _objective(temporal=True)
    obj = _open_object(objective, poses)
    reference, original = objective.reference.copy(), poses.copy()
    vertices = obj['mesh'].vertices.copy()
    result, report = sqp.repair_original_meshes_sqp(objective, poses, [obj])
    np.testing.assert_array_equal(poses, original)
    np.testing.assert_array_equal(objective.reference, reference)
    np.testing.assert_array_equal(obj['mesh'].vertices, vertices)
    np.testing.assert_array_equal(result[:, :7], poses[:, :7])
    assert RepairLimits(objective, poses).check(result)[0]
    assert report['after']['moving']['max_depth_m'] <= report['before']['moving']['max_depth_m'] + 1e-6
    assert all('seconds' in h and 'alpha' in h for h in report['history'])
    assert report['ledger']['counts']['geometry_measures'] >= 1
    assert report['witness_band_m'] == .003


def test_adopt_preserves_accumulated_limits_and_kinematics():
    from tests.test_fast_backend import _objective
    from omniretargeting._fast.local_windows import ClusterLimits
    from omniretargeting._fast.local_slip import entry_points
    objective, poses, _ = _objective(temporal=True, extra=True)
    limits = RepairLimits(objective, poses)
    limits.kin.evaluate(limits.entry, False)
    expected_points = entry_points(limits)
    current = poses.copy()
    current[:, limits.coords.qi[-1]] += .001
    limits.kin.evaluate(current, True)
    state = limits.kin.__dict__.copy()
    data = limits.kin.data.qpos.copy()
    cache_keys = list(limits.kin.cache)
    anchors, targets = limits.anchors, limits.target_budgets
    for anchor in anchors:
        anchor += .00001
    anchor_bytes = [a.tobytes() for a in anchors]
    target_bytes = [a.tobytes() for a in targets]
    assert ClusterLimits.adopt(limits) is limits
    assert type(limits) is ClusterLimits and limits.permission == {}
    assert limits.anchors is anchors and limits.target_budgets is targets
    assert [a.tobytes() for a in anchors] == anchor_bytes
    assert [a.tobytes() for a in targets] == target_bytes
    assert all(limits.kin.__dict__[key] is value for key, value in state.items())
    np.testing.assert_array_equal(limits.kin.data.qpos, data)
    assert list(limits.kin.cache) == cache_keys
    for actual, expected in zip(limits.entry_surface_points, expected_points):
        np.testing.assert_array_equal(actual, expected)
    limits.grant([dict(bodies=[], mask=np.ones(limits.frames, bool))])
    assert not limits.permission
    for _, bodies, budget in limits.point_groups():
        assert not limits.correction_mask(bodies, budget).any()
    limits.release(current)
    assert not limits.permission and limits.movable_body is None


def test_no_chatter_skips_adoption_waves_and_geometry(monkeypatch):
    from tests.test_fast_backend import _objective
    from omniretargeting._fast import local_windows, local_rounds
    objective, poses, _ = _objective(temporal=True)
    limits = RepairLimits(objective, poses)
    def forbidden(*args, **kwargs):
        pytest.fail('zero chatter must skip cleanup work')
    monkeypatch.setattr(local_windows.ClusterLimits, 'adopt', forbidden)
    monkeypatch.setattr(local_rounds, 'run_wave', forbidden)
    geometry = SimpleNamespace(linearize=forbidden)
    measured, protected = dict(rows=[(None, {})]), dict(depths={})
    result, measurement, report = sqp._cleanup_chatter(
        objective, limits, geometry, objective.reference, measured, protected)
    assert result is objective.reference and measurement is measured
    assert type(limits) is RepairLimits
    assert report['status'] == 'skipped_no_chatter'
    assert report['chatter_before'] == report['chatter_after'] == dict(violations=0, excess=0.)
    assert report['history'] == [] and report['counts'] == {}


def test_chatter_cleanup_finishes_group_and_reuses_accumulated_state(monkeypatch):
    from tests.test_fast_backend import _objective
    from omniretargeting._fast import local_rounds
    from omniretargeting._fast.local_windows import ClusterLimits
    objective, poses, _ = _objective(temporal=True)
    current = poses.copy()
    current[10, objective.coords.qi[-1]] += .12
    limits = RepairLimits(objective, current)
    limits.movable_body = int(objective.model.body_weldid[objective.effectors[0]['body']])
    measured, protected = dict(rows=[], depths={}), dict(depths={})
    visits = []
    def wave(ctx, windows, state):
        assert ctx.limits is limits and type(limits) is ClusterLimits
        assert limits.movable_body is None and not limits.permission
        assert all(not w['bodies'] for w in windows)
        assert state.current is current and state.measured is measured
        assert state.protected is protected['depths']
        assert limits.check(current)[0]
        visits.append(windows)
    monkeypatch.setattr(local_rounds, 'run_wave', wave)
    result, measurement, report = sqp._cleanup_chatter(
        objective, limits, SimpleNamespace(), current, measured, protected)
    assert visits and result is current and measurement is measured
    assert report['chatter_before']['violations'] > 0
    assert report['counts']['chatter_windows'] > 0
    assert report['status'] == 'retained'


def test_hard_chatter_rows_cover_only_moved_columns_and_crossing_stencils():
    from omniretargeting._fast.local_chatter import JointChatter
    frames, width = 40, 2
    coords = SimpleNamespace(qi=np.arange(width))
    chatter = JointChatter(coords, np.zeros((frames, width)))
    current = np.zeros((frames, width)); current[20, 0] = .2
    moved = np.zeros((frames, width), bool); moved[18:23, 0] = True
    columns = np.flatnonzero(moved.any(axis=0))
    touched = chatter.stencil_frames(moved.any(axis=1))
    matrix, lower, upper = chatter.rows(current, touched, columns, elastic=False)
    assert np.all(matrix.indices % width == 0)
    assert touched[0] < 18 and touched[-1] > 22
    assert np.all(lower <= 0) and np.all(upper >= 0)
    # A feasible scalar step under these exact rows cannot grow any entry excess.
    step = np.zeros_like(current); step[20, 0] = -.01
    assert np.all(matrix @ step.ravel() >= lower - 1e-12)
    assert np.all(matrix @ step.ravel() <= upper + 1e-12)
    assert np.all(np.maximum(chatter.excess(current + step), 0.)
                  <= np.maximum(chatter.excess(current), 0.) + 1e-12)


def test_unchanged_entry_chatter_skips_cleanup(monkeypatch):
    from tests.test_fast_backend import _objective
    from omniretargeting._fast import local_rounds
    objective, poses, _ = _objective(temporal=True)
    poses = poses.copy(); poses[10, objective.coords.qi[-1]] += .12
    limits = RepairLimits(objective, poses)
    monkeypatch.setattr(local_rounds, 'run_wave', lambda *a: pytest.fail('entry excess must not trigger cleanup'))
    current, measured = poses.copy(), dict(rows=[])
    result, measurement, report = sqp._cleanup_chatter(
        objective, limits, SimpleNamespace(), current, measured, dict(depths={}), introduced_only=True)
    assert result is current and measurement is measured
    assert report['status'] == 'preserved_entry_chatter'
    assert report['chatter_before']['violations'] > 0
    assert report['chatter_after'] == report['chatter_before']
    assert report['counts'] == {}


def test_minimum_norm_correction_retains_vector_cone_and_residual_certificate():
    class NormLimits(Limits):
        def constraints(self, poses, **kwargs):
            x = poses[0, 0]
            bounds = LinearBounds(1)
            bounds.norms = [(sparse.csr_matrix([[1 + 2*x], [0.], [0.]]),
                             np.array([[x + x*x, 0., 0.]]), np.array([.01000005]))]
            return bounds
    p = proposal(NormLimits())
    candidate = p.full_candidate(p.step, {})
    assert candidate is not None and p.limits.check(candidate)[0]
    info = p.ledger.corrections[-1]
    assert info["cones"] == 1 and info["cone_error_absolute"] <= 1e-9
    assert info["trust_error"] <= 1e-9
    assert .009 < candidate[0, 0] < .01


def test_second_limit_failure_after_witness_cut_stops():
    p = proposal()
    p.soc_attempts = 2
    p.solve = lambda: pytest.fail('the fixed correction cap must stop retries')
    assert p.full_candidate(p.step, {}) is None
    assert not p.limit_cuts


def test_exact_chatter_floor_is_checked_before_geometry():
    p = proposal()
    p.limits = Limits()
    p.limits.check = lambda poses: (True, {})
    p.limits._sqp_chatter = SimpleNamespace(excess=lambda q: q - .003)
    record = {}
    assert not p.check(np.array([[.01]]), record, 1.)
    assert record['rejected_limits']['reason'] == 'source-relative chatter floor'
    assert p.ledger.rejection_reasons == {'source-relative chatter floor': 1}


def test_group_qp_targets_existing_chatter_without_elastic_slack():
    from omniretargeting._fast.local_chatter import JointChatter
    frames = 40
    coords = SimpleNamespace(qi=np.array([0]))
    entry = np.zeros((frames, 1)); entry[20, 0] = .03
    chatter = JointChatter(coords, np.zeros_like(entry))
    class SignalLimits:
        width = 1
        movable_body = None
        def constraints(self, poses, **kwargs):
            bounds = LinearBounds(frames)
            bounds.set_box(-.05-poses.ravel(), .05-poses.ravel())
            return bounds
    limits = SignalLimits(); limits.frames = frames; limits.coords = coords
    limits.entry = entry; limits._sqp_chatter = chatter
    key = (20, 'static', 0)
    geometry = SimpleNamespace(linear_rows=lambda _: (sparse.csr_matrix((1, frames)), np.array([-.02])))
    measured = dict(rows=[(key,{})], depths={key:.02})
    step, status = sqp.solve_step(limits, entry, geometry, measured, measured,
                                 sparse.eye(frames,format='csc'), collision_sigma=.01, local_angle=.05)
    assert step is not None and status in ('Solved','AlmostSolved')
    before=np.maximum(chatter.excess(entry),0.); after=np.maximum(chatter.excess(entry+step),0.)
    assert after.sum() < before.sum() - 1e-4
    assert np.all(after <= before + 1e-9)


def test_coupled_collider_tail_improves_secondary_body_without_loosening_protection():
    class CoupledLimits:
        frames, width, movable_body = 1, 2, 0
        _sqp_coupled_geometry = True
        entry = np.zeros((1, 2))
        coords = SimpleNamespace(qi=np.arange(2))
        def constraints(self, poses, **kwargs):
            bounds = LinearBounds(2)
            bounds.set_box(np.full(2,-.05),np.full(2,.05))
            return bounds
    limits = CoupledLimits()
    keys = [(0,'static',0),(0,'static',1)]
    geometry = SimpleNamespace(model=SimpleNamespace(geom_bodyid=np.arange(2),body_weldid=np.arange(2)),
                               linear_rows=lambda _: (sparse.eye(2,format='csr'),np.array([-.02,-.01])))
    measured = dict(rows=[(k,{}) for k in keys],depths=dict(zip(keys,[.02,.01])))
    step,status=sqp.solve_step(limits,limits.entry,geometry,measured,measured,
                              sparse.eye(2,format='csc'),collision_sigma=.01,local_angle=.05)
    assert step is not None and status in ('Solved','AlmostSolved')
    assert step[0,0]>.01 and step[0,1]>.005
    assert np.all(step>=0)  # Neither collider can regress under the original rows.


def test_curvature_retry_does_not_suppress_new_material_witness():
    p = proposal()
    p.limit_cuts = [([], [])]
    p.limits.check = lambda q: (True, {})
    key = (0, 'static', 0)
    p.geometry.measure = lambda *a, **kw: dict(rows=[(key, {})], depths=p.measured['depths'], failures=[])
    p.geometry.linearize = lambda trial, measured: measured
    p.solve = lambda: (np.array([[.007]]), 'Solved')
    iterator = p.candidates({})
    _, trial = next(iterator)
    p.feedback = (trial, dict(rows=[(key,{})], failures=[]), [key])
    _, retry = next(iterator)
    assert retry[0,0] == pytest.approx(.017) and p.ledger.counts['cuts'] == 1
    p.feedback = (retry, dict(rows=[(key,{})], failures=[]), [key])
    _, second_retry = next(iterator)
    assert second_retry[0,0] == pytest.approx(.024)
    assert p.ledger.counts['geometry_resolves'] == sqp.GEOMETRY_REFRESHES
    assert p.cut_rows == p.ledger.counts['cuts'] == 2
    p.feedback = (second_retry, dict(rows=[(key,{})], failures=[]), [key])
    with pytest.raises(StopIteration):next(iterator)


def test_incomplete_trial_geometry_stops_without_using_rows():
    p=proposal();p.limits.check=lambda q:(True,{})
    p.geometry.measure=lambda *a,**kw:dict(complete=False,rows=[],failures=['unknown'])
    record={};iterator=p.candidates(record);_,trial=next(iterator)
    p.feedback=(trial,dict(rows=[((0,'static',0),{})],failures=[]),[(0,'static',0)])
    with pytest.raises(StopIteration):next(iterator)
    assert record['stop_reason']=='unknown_trial_geometry' and not p.cuts


def test_trial_geometry_restoration_forces_return_to_protected_depth():
    limits=Limits();limits.movable_body=None
    key=(0,'static',0)
    geometry=SimpleNamespace(linear_rows=lambda m:(sparse.csr_matrix([[1.]]),np.array([-.003])))
    measured=dict(rows=[(key,{})],depths={key:.003})
    protected=dict(depths={key:.001})
    step,status=sqp.solve_step(limits,np.array([[0.]]),geometry,measured,protected,
                              sparse.eye(1,format='csc'),collision_sigma=1.,local_angle=.05,restore_geometry=True)
    assert step is not None and status in ('Solved','AlmostSolved')
    assert step[0,0]>=.0023-1e-8


def test_new_trial_contact_keeps_clearance_reserve_for_curvature():
    limits=Limits();limits.movable_body=None
    key=(0,'static',0)
    geometry=SimpleNamespace(linear_rows=lambda m:(sparse.csr_matrix([[1.]]),np.array([-.0001])))
    measured=dict(rows=[(key,{})],depths={key:.0001})
    protected=dict(depths={key:0.})
    step,status=sqp.solve_step(limits,np.array([[0.]]),geometry,measured,protected,
                              sparse.eye(1,format='csc'),collision_sigma=1.,local_angle=.05,restore_geometry=True)
    assert step is not None and status in ('Solved','AlmostSolved')
    assert -.0001+step[0,0]>=sqp.CLEARANCE_MARGIN_M-1e-8


def test_geometry_registry_contains_audit_samples_without_mutating_objective():
    from tests.test_fast_backend import _objective
    from omniretargeting.scene_geometry import geom_surface_offsets
    objective, _, _ = _objective(temporal=True)
    original_points=objective.shell_local.copy();original_owners=objective.shell_sample_geom.copy()
    enriched=sqp._geometry_objective(objective)
    assert enriched is not objective and enriched.model is objective.model
    np.testing.assert_array_equal(objective.shell_local,original_points)
    np.testing.assert_array_equal(objective.shell_sample_geom,original_owners)
    for geom in objective.shell_geoms:
        samples=set(map(tuple,enriched.shell_local[enriched.shell_sample_geom==geom]))
        assert (0.,0.,0.) in samples
        assert set(map(tuple,geom_surface_offsets(objective.model,int(geom),.02)))<=samples
        assert set(map(tuple,original_points[original_owners==geom]))<=samples


def test_audit_probes_cannot_enlarge_original_static_budget():
    from omniretargeting._fast.mesh_repair import _regressed
    geometry=sqp.OriginalGeometry.__new__(sqp.OriginalGeometry)
    geometry.objective=SimpleNamespace(_sqp_original_probes=np.array([True,False]))
    geometry.scene=SimpleNamespace(query=lambda points:dict(clearance=points[:,2],
        directions=np.tile([0.,0.,1.],(len(points),1)),uncertain=np.zeros(len(points),bool)))
    geometry.manifold=True;geometry.witness_band=.003;geometry._static_bound_reuse=False
    summary=dict(rows=[],row=None,unknown=False,clearance=np.inf,origin=np.zeros(3))
    geometry._query_static_batch([(summary,np.array([[0.,0.,-.001],[1.,0.,-.003]]),np.array([0,1]))])
    rows,depths=[],{}
    for row in summary['rows']:geometry._record(rows,depths,(0,'static',0),row)
    assert depths=={(0,'static',0):.001,(0,'static',-1,0):.003}
    original=(0,'static',0);extra=(0,'static',-1,0)
    # The new union maximum improves, but the original partition regresses.
    changed={original:.00108,extra:.0029}
    assert max(changed.values())<max(depths.values())
    assert _regressed(changed[original],depths[original])
    assert not _regressed(changed[extra],depths[extra])


def test_original_descent_is_required_with_extra_audit_probes():
    original=(0,'static',0);extra=(0,'static',-1,0)
    before=dict(depths={original:.0001,extra:.003})
    after=dict(depths={original:.00011,extra:.002},failures=[])
    geometry=SimpleNamespace(measure=lambda *a,**kw:after)
    checked,bad,_,descent=sqp.measure_candidate(None,geometry,None,before,before,None,
        sum(v*v for v in before['depths'].values()),frames=range(1))
    assert checked is after and not bad
    assert not descent  # New-probe improvement cannot conceal original deterioration.


def test_probe_partitions_share_one_physical_collision_objective():
    limits=Limits();limits.movable_body=None
    keys=[(0,'static',0),(0,'static',-1,0)]
    geometry=SimpleNamespace(linear_rows=lambda m:(sparse.csr_matrix([[1.],[1.]]),np.array([-.001,-.003])))
    measured=dict(rows=[(k,{}) for k in keys],depths=dict(zip(keys,[.001,.003])))
    diagnostics={}
    step,status=sqp.solve_step(limits,limits.entry,geometry,measured,measured,
        sparse.eye(1,format='csc'),collision_sigma=.001,local_angle=.05,diagnostics=diagnostics)
    assert step is not None and status in ('Solved','AlmostSolved')
    assert diagnostics['variables']==2  # One joint and one geom, despite two budgets.


def test_deep_self_pair_can_recruit_partner_without_mutating_scopes(monkeypatch):
    scope = SimpleNamespace(bodies=frozenset({1}), columns=frozenset({0}))
    limits = SimpleNamespace(movable_body=1, limb_scopes=(scope,))
    model = SimpleNamespace(body_weldid=np.array([0, 1, 2]), geom_bodyid=np.array([1, 2]))
    seen = []
    def projection(limits, model, measured):
        seen.append(limits.limb_scopes)
        return sparse.eye(2, format='csc')
    monkeypatch.setattr(sqp, '_baseline_projection', projection)
    shallow = dict(depths={(0, 'self', 0, 1): .0049})
    sqp._joint_projection(limits, model, shallow)
    assert seen[-1] == (scope,) and not limits._sqp_widened_self
    deep = dict(depths={(0, 'self', 0, 1): .006})
    sqp._joint_projection(limits, model, deep)
    assert seen[-1] == () and limits._sqp_widened_self
    assert limits.limb_scopes == (scope,)
    # Deep static/moving penetration cannot widen the scope.
    sqp._joint_projection(limits, model, dict(depths={(0, 'static', 0): .02}))
    assert seen[-1] == (scope,) and not limits._sqp_widened_self


def test_full_union_budget_and_deepest_extra_probe_keep_old_acceptance():
    geometry = sqp.OriginalGeometry.__new__(sqp.OriginalGeometry)
    geometry.objective = SimpleNamespace(_sqp_original_probes=np.array([True, False, False]))
    queried = []
    def query(points):
        queried.append(len(points))
        return dict(clearance=points[:, 2], directions=np.tile([0., 0., 1.], (len(points), 1)),
                    uncertain=np.zeros(len(points), bool))
    geometry.scene = SimpleNamespace(query=query)
    geometry.manifold = True; geometry.witness_band = .003; geometry._static_bound_reuse = False
    summary = dict(rows=[], row=None, unknown=False, clearance=np.inf, origin=np.zeros(3))
    points = np.array([[0., 0., -.003], [1., 0., -.002], [2., 0., -.001]])
    geometry._query_static_batch([(summary, points, np.arange(3))])
    assert queried == [3]  # Acceptance queries every probe, including omitted model rows.
    assert {r['sample_id'] for r in summary['rows']} == {0, 1, 2}
    rows, depths = [], {}
    for row in summary['rows']:
        geometry._record(rows, depths, (0, 'static', 0), row)
    original, union = (0, 'static', 0), (0, 'static', -1, 0)
    assert depths == {original: .003, union: .003}
    # The extra subset rises, but the full collider improves and old samples
    # improve. A subset-only budget would artificially reject this valid step.
    checked = dict(depths={original: .0029, union: .0029}, failures=[])
    geometry.measure = lambda *args, **kwargs: checked
    before = dict(depths=depths)
    _, bad, _, descent = sqp.measure_candidate(None, geometry, None, before, before, None,
        .003**2, frames=range(1))
    assert not bad and descent
    checked['depths'][union] = .0032
    _, bad, _, descent = sqp.measure_candidate(None, geometry, None, before, before, None,
        .003**2, frames=range(1))
    assert union in bad and not descent


@pytest.mark.parametrize('value', [np.nan, np.inf, 1000.])
def test_soc_never_returns_uncertified_or_outside_screen_solution(monkeypatch, value):
    from omniretargeting._fast import working_set
    monkeypatch.setattr(working_set, 'solve', lambda *a, **k: SimpleNamespace(
        x=np.array([value]), status='Solved', iterations=1))
    p = proposal()
    assert sqp._soc(p, p.step) is None
    assert p.ledger.counts['correction'] == 1
    np.testing.assert_array_equal(p.current, np.zeros((1,1)))


def test_soc_with_fixed_violated_row_stops_without_solver(monkeypatch):
    class FixedViolation(Limits):
        def constraints(self, poses, **kwargs):
            bounds = super().constraints(poses, **kwargs)
            bounds.add(sparse.csr_matrix((1, 1)), 1e-5, np.inf)
            return bounds
    from omniretargeting._fast import working_set
    monkeypatch.setattr(working_set, 'solve', lambda *a, **k: pytest.fail('fixed violation'))
    p = proposal(FixedViolation())
    assert sqp._soc(p, p.step) is None
    assert p.ledger.counts['qp'] == 0
