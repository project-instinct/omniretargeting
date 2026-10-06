"""Joint acquisition's provenance, tangent math and disabled-path contract."""
from dataclasses import replace
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
from scipy import sparse

from omniretargeting._fast import runtime_options, solver
from omniretargeting._fast.mesh_repair import OriginalGeometry
from omniretargeting._fast.support_split import split_bounds, split_error, SupportSplit
from tests.test_original_geometry_batches import _problem, _scene
from tests.test_fast_short_banded import Quadratic


def test_witness_jacobian_finite_difference():
    from omniretargeting._fast.joint_geometry import JointGeometry
    obj, poses = _problem()
    obj.coords.cols = np.arange(obj.model.nv)
    obj._clearance_fields = lambda: []  # Missing proxy activates original geoms.
    geometry = JointGeometry(obj, [], _scene('closed'), poses)
    measured = geometry.measure(poses, frames=range(len(poses)), jacobians=True)
    eps = 1e-6
    for axis, dof in enumerate(geometry.di):
        plus, minus = poses.copy(), poses.copy()
        direction = np.zeros(obj.model.nv)
        direction[dof] = eps
        for f in range(len(poses)):
            mujoco.mj_integratePos(obj.model, plus[f], direction, 1.)
            mujoco.mj_integratePos(obj.model, minus[f], direction, -1.)
        high = geometry.reevaluate_witnesses(plus, measured)
        low = geometry.reevaluate_witnesses(minus, measured)
        fd = (high['clearance'] - low['clearance']) / (2 * eps)
        exact = np.array([row['jacobian'][axis] for _, row in measured['rows']])
        valid = high['reliable'] & low['reliable']
        # Exclude only non-differentiable nearest-face switches via asymmetric
        # directional derivatives, rather than silently accepting bad rows.
        base = np.array([row['clearance'] for _, row in measured['rows']])
        smooth = np.abs((high['clearance'] - base) - (base - low['clearance'])) < 1e-9
        assert np.count_nonzero(valid & smooth) > 10
        np.testing.assert_allclose(exact[valid & smooth], fd[valid & smooth], atol=2e-6)


def test_normal_exit_requires_own_source_penetration_and_tangent_never_grows():
    offset = np.array([[[.02, -.01, -.003], [.02, -.01, .003]]])
    normal = np.broadcast_to([0., 0., 1.], offset.shape)
    no_evidence = split_bounds(offset, normal, np.zeros((1, 2)), np.full((1, 2), .04), .001)
    evidence = split_bounds(offset, normal, np.array([[.01, 0.]]), np.array([[.006, .04]]), .001)
    assert no_evidence[1][0, 0] == .001
    assert evidence[1][0, 0] == .004
    for channel in (0, 2, 3):
        np.testing.assert_array_equal(evidence[channel], no_evidence[channel])
    n, t = split_error(offset, normal)
    np.testing.assert_array_equal(n, [[-.003, .003]])
    np.testing.assert_allclose(np.sum(t * normal, axis=-1), 0.)


def test_disabled_switch_retains_synthetic_pose_bytes(monkeypatch):
    def run(options):
        monkeypatch.setattr(runtime_options, 'OPTIONS', options)
        obj = Quadratic()
        monkeypatch.setattr(solver, '_conic_proposal', lambda *a: (np.array([.02]), np.array([[.02]]), 1.))
        q, history = solver._optimize(obj, np.zeros((1, 1)), 2, None, 'probe', .003)
        return q, [{k: v for k, v in h.items() if k != 'seconds'} for h in history]
    baseline = replace(runtime_options.OPTIONS, joint_witness=False)
    implicit = run(baseline)
    explicit = run(replace(baseline, joint_witness=False))
    assert implicit[0].tobytes() == explicit[0].tobytes()
    assert implicit[1] == explicit[1]
    # Recorded with the unmodified pre-task solver on this same quadratic.
    # Comparing only two False options would miss a shared disabled-path drift.
    assert explicit[0].tobytes().hex() == 'b81e85eb51b89e3f'
    assert explicit[1] == [
        dict(phase='probe', iteration=0, cost=1., after=.9603999999999999,
             alpha=1., damping=.0003333333333333333, step_backend='shape_conic'),
        dict(phase='probe', iteration=1, cost=.9603999999999999, after=.9409,
             alpha=.5, damping=.0001111111111111111, step_backend='shape_conic')]


def test_joint_geometry_selects_only_active_geoms_and_pose_cache():
    from omniretargeting._fast.joint_geometry import JointGeometry
    obj, poses = _problem()
    obj.coords.cols = obj.coords.di
    obj.shell_radius = obj.model.geom_size[:, 0]
    obj.shell_index = np.arange(len(obj.shell_local)).reshape(obj.model.ngeom, -1)
    class Proxy:
        def query(self, p, gradient=False):
            return np.where(p[..., 0] < .1, -.01, 1.)
    obj._clearance_fields = lambda: [('scene', Proxy(), None, None, None)]
    geometry = JointGeometry(obj, [], _scene('closed'), poses)
    geometry._gpu_allowed = False
    first = geometry.measure(poses, frames=range(len(poses)))
    assert all(key[1] != 'static' or key[-1] == 0 for key, _ in first['rows'])
    same = geometry.measure(poses, frames=range(len(poses)))
    assert same['query_scope']['queried_frame_count'] == 0
    changed = poses.copy()
    changed[2, 0] += .001
    result = geometry.measure(changed, frames=range(len(poses)))
    assert result['query_scope']['queried_frames'] == [2]


def test_joint_elastic_cost_proposal_and_exact_trial_agree():
    import trimesh
    from omniretargeting.scene_geometry import SceneQuery
    from omniretargeting._fast.joint_witness import JointWitness
    from omniretargeting._fast.reference import ReferenceShape
    from omniretargeting._fast.acquisition_guard import AcquisitionNormalGuard
    from omniretargeting._fast.shape_step import bounded_shape_step
    from tests.test_fast_backend import _objective
    obj, poses, _ = _objective(temporal=True)
    obj.source_shape = ReferenceShape(obj, poses)
    obj.source_shape.freeze_reference(poses)
    box = trimesh.creation.box(extents=[5, 5, .1])
    box.apply_translation([0., 0., -.05])
    scene = SceneQuery(box)
    class Plane:
        def gradient(self, points):
            return np.broadcast_to([0., 0., 1.], points.shape)
        def query(self, points, gradient=False):
            if gradient:
                return points[..., 2], np.broadcast_to([0., 0., 1.], points.shape)
            return points[..., 2]
    obj.scene_sdf = Plane()
    joint = obj.joint_witness = JointWitness(obj, poses, [], scene)
    try:
        guard = AcquisitionNormalGuard(obj, poses)
        obj.candidate_guards = [guard, joint]
        acc, r, jt, _, _ = obj.evaluate(poses)
        joint.begin(poses, 0)
        grad = acc.gradient.ravel() + jt.T @ r
        system = solver.banded_system(obj.frames, obj.width, acc.hessian, jt)
        diagonal = np.maximum(np.diagonal(acc.hessian, axis1=1, axis2=2).ravel(), 1e-9)
        step = bounded_shape_step(obj, poses, system, grad, diagonal, .01)
        assert step is not None and np.isfinite(step).all()
        assert solver._model_decrease(acc, jt, step, grad) > 0
        candidate = obj.project(obj.coords.integrate(poses, step.reshape(obj.frames, obj.width), 1.))
        trial = obj.evaluate(candidate)
        assert np.isfinite(trial[0].cost)
        diag = joint.diagnostics(candidate)
        assert 'static' in diag['before_free_repair']
    finally:
        joint.close()


def test_banded_rows_share_the_same_pair_slack_as_conic():
    from collections import Counter
    from omniretargeting._fast.joint_witness import JointWitness
    joint = JointWitness.__new__(JointWitness)
    joint.objective = SimpleNamespace(coords=SimpleNamespace(difference=lambda a, b: b - a))
    joint.origin_poses = np.zeros((1, 1))
    joint.raw = np.array([.01, .002, -.001])
    joint.pairs = np.array([0, 0, 1])
    joint.matrix = sparse.csr_matrix([[-1.], [1.], [0.]])
    joint.rejections = Counter()
    # The shallower witness can grow while staying below its collision pair's
    # origin elastic slack. Every row must still satisfy that shared bound.
    assert joint.linear_feasible(np.array([[.005]]))
    assert not joint.linear_feasible(np.array([[.009]]))
    assert joint.rejections['banded_witness_rows'] == 1


def test_moving_diagnostics_accept_empty_optional_jacobian():
    from omniretargeting._fast.joint_witness import JointWitness
    joint = JointWitness.__new__(JointWitness)
    joint.objective = SimpleNamespace(width=2)
    joint.support = SimpleNamespace(allowance=lambda *a: 0.)
    rows = [((0, 'static', 1), dict(clearance=-.01, jacobian=np.array([1., 2.]))),
            ((1, 'moving', 0, 2), dict(clearance=-.02, jacobian=np.array([])))]
    _, raw, jacobian, _ = joint.rows(dict(rows=rows))
    np.testing.assert_array_equal(raw, [.01, .02])
    np.testing.assert_array_equal(jacobian, [[-1., -2.], [0., 0.]])
    rows[1][1]['jacobian'] = np.ones(3)
    with pytest.raises(ValueError, match='Unexpected witness Jacobian shape'):
        joint.rows(dict(rows=rows))


def test_support_curvature_rows_and_allowance_provenance():
    import cvxpy as cp
    from tests.test_fast_contact_guard import _acquired
    obj, poses = _acquired(surfaces=True)
    # Construct a geometric source support on one immutable surface. No
    # endpoint/limb-name heuristics enter its role or residual allowance.
    for eff in obj.effectors:
        eff['active'][:] = False
    surf = obj.surfaces[0]
    surf['normals'][:] = [0., 0., 1.]
    task = surf['task']
    task.source_vectors = np.broadcast_to([0., 0., -.002], task.source_vectors.shape).copy()
    task.surface_ids = np.zeros_like(task.surface_ids)
    task.static_surface_id = 0
    obj.kin.evaluate(poses)
    point = obj.kin.geom_points(surf['geoms'], surf['local'])
    task.environment_points = point + [0., 0., .002]
    class Query:
        def query(self, pp):
            # Every point belongs to the same verified support plane.
            z = task.environment_points[0, 0, 2]
            return dict(clearance=pp[:, 2] - z, uncertain=np.zeros(len(pp), bool),
                        directions=np.tile([0., 0., 1.], (len(pp), 1)), obstacle_ids=np.zeros(len(pp), int))
    geometry = SimpleNamespace(scene=Query(), moving=None)
    obj.tasks.contact_phase = SimpleNamespace(surface=task, surface_roles={'support': surf['active']},
                                             role=lambda *a: 'free')
    split = SupportSplit(obj, poses, geometry)
    split.extract('surface', surf['geoms'], surf['local'], surf['targets'], surf['normals'], surf['active'], .001, 'surface_0')
    split.begin(poses)
    assert split.accepts(poses)
    flat = cp.Variable(obj.frames * obj.width)
    constraints = split.constraints(poses, flat)
    flat.value = np.zeros(flat.size)
    assert max(float(np.max(c.violation())) for c in constraints) < 1e-7
    candidate = poses.copy()
    candidate[:, 0] += .03
    assert not split.accepts(candidate)
    # A lateral exit has no support residual allowance, even on the same body.
    row = dict(geom_id=int(surf['geoms'][0]), point=point[0, 0], clearance=-.002, direction=np.array([1., 0., 0.]))
    assert split.allowance((0, 'static', row['geom_id']), row) == 0.
    assert split.allowance((0, 'self', 0, 1), row) == 0.


def test_skip_free_is_opt_in_and_requires_joint(monkeypatch):
    assert runtime_options.RuntimeOptions().joint_skip_free is False
    monkeypatch.setenv('OMNIRETARGETING_FAST_DISABLE', '')
    monkeypatch.setenv('OMNIRETARGETING_FAST_ENABLE', 'joint_skip_free')
    with pytest.raises(ValueError, match='requires joint_witness'):
        runtime_options.RuntimeOptions.from_environment()
    monkeypatch.setenv('OMNIRETARGETING_FAST_ENABLE', 'joint_witness,joint_skip_free')
    opts = runtime_options.RuntimeOptions.from_environment()
    assert opts.joint_witness and opts.joint_skip_free


def test_workload_reports_attempts_and_overlapping_rejection_reasons():
    from omniretargeting._fast.postprocess_workload import workload
    report = dict(stages={'contact_scene': dict(
        history=[{'phase': 'reweight_1'}, {'phase': 'free_bodies'}], reweight_rounds=[{'rejected': 'reweight_1'}],
        penetration_stages=[dict(stage='free_bodies', before={'depth': .02}, candidate={'depth': .01}, accepted=False)],
        original_mesh_repair=dict(scheduled_groups=['owner'], history=[dict(contact_group='owner', alpha=0.,
            stop_reason='no_admissible_descent', model_progress=dict(entry_depth_m=.02, realized_decrease_m=0.),
            rejected_limits=[{'guard': 'ContactGuard'}])]))},
        repair_trial_workload=[dict(group='owner', complete=True, unknown=0, regressions=2, descent=False, accepted=False)])
    result = workload(report)
    assert result['reweight']['rounds'] == 1
    assert result['reweight']['attempts'][0]['accepted'] is False
    assert result['free_bodies'][0]['after'] == {'depth': .02}
    assert result['repair']['rejected_reasons']['ratchet'] == 1
    assert result['repair']['rejected_reasons']['regression_keys'] == 2
    assert result['repair']['rejected_reasons']['ContactGuard'] == 1
    assert result['repair']['per_group'][0]['after_depth_m'] == .02
