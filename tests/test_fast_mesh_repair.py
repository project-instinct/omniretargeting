"""Original-mesh repair preserves contacts, source shape and temporal budgets."""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import trimesh

from omniretargeting._fast.mesh_repair import (
    OriginalGeometry,
    _group_rank,
    repair_original_meshes,
)
from omniretargeting._fast.reference import ReferenceShape
from omniretargeting._fast.repair_limits import LOCAL_ANGLE, RepairLimits
from tests.test_fast_backend import _objective


def _open_object(objective, poses):
    objective.kin.evaluate(poses, False)
    body = objective.model.body("hand").id
    translation = objective.kin.xpos[:, body] + [0.0, 0.018, 0.0]
    mesh = trimesh.Trimesh(vertices=[[0, -0.025, -0.025], [0, 0.025, -0.025],
                                      [0, 0.025, 0.025], [0, -0.025, 0.025]],
                           faces=[[0, 1, 2], [0, 2, 3]], process=False)
    return {"mesh": mesh, "translations": translation,
            "rotations": np.tile(np.eye(3), (len(poses), 1, 1))}


def test_limits_reject_root_drift_pose_distortion_and_added_joint_chatter():
    objective, poses, _ = _objective(temporal=True)
    limits = RepairLimits(objective, poses)
    assert limits.check(poses)[0]
    candidate = poses.copy()
    candidate[0, 0] += 1e-8
    assert limits.check(candidate)[1]["reason"] == "root changed"
    candidate = poses.copy()
    candidate[:, objective.coords.qi[-1]] += LOCAL_ANGLE + 0.001
    assert limits.check(candidate)[1]["reason"] == "joint envelope"
    candidate = poses.copy()
    # A one-frame spike is screened by final source-relative chatter, not a
    # local derivative cap.
    candidate[10, objective.coords.qi[-1]] += 0.02
    assert limits.check(candidate)[0]


def test_group_order_preserves_unexplained_holds_before_control():
    objective, _, _ = _objective(temporal=True)
    roles = {"foot": "hold", "hand": "control"}
    objective.tasks.contact_phase = SimpleNamespace(role=lambda name, frame: roles[name])
    foot = int(objective.model.body_weldid[objective.model.body("foot").id])
    hand = int(objective.model.body_weldid[objective.model.body("hand").id])
    # The hand is free in this fixture; give it observed contact for this test.
    objective.effectors[1]["active"][:] = True
    assert _group_rank(objective, foot) < _group_rank(objective, hand)
    roles.update(foot="follow", hand="support")
    assert _group_rank(objective, hand) < _group_rank(objective, foot)


def test_repair_limits_linearization_contains_zero_step_and_preserves_reference():
    objective, poses, _ = _objective(temporal=True, extra=True)
    reference = objective.reference.copy()
    limits = RepairLimits(objective, poses)
    matrix, lower, upper = limits.constraints(poses).packed()
    assert matrix.shape[1] == len(poses) * len(objective.coords.qi)
    assert (lower <= 1e-8).all()
    assert (upper >= -1e-8).all()
    assert limits.check(poses)[0]
    np.testing.assert_array_equal(objective.reference, reference)


def test_open_object_has_reverse_collision_rows_without_fabricating_signed_volume():
    objective, poses, _ = _objective(temporal=True)
    obj = _open_object(objective, poses)
    geometry = OriginalGeometry(objective, [obj], None)
    report = geometry.moving.query(poses[8], 8, objective.coords.di)
    assert report["unknown_components"] > 0
    assert report["reverse_depth"].max() > 0
    rows = [r for r in report["rows"] if r["kind"] == "reverse_surface"]
    assert rows
    assert all(np.isfinite(r["jacobian"]).all() for r in rows)
    row = min(rows, key=lambda r: r["clearance"])
    # A fixed material sample and facet give an exact local derivative.
    shell = next(s for s in geometry.moving.shells if s["geom_id"] == row["geom_id"])
    plane = shell["planes"][row["facet_id"]]
    point = row["point"]
    for j in np.flatnonzero(np.abs(row["jacobian"]) > 1e-9):
        distances = []
        for sign in (-1, 1):
            q = poses[8].copy()
            q[objective.coords.qi[j]] += sign * 1e-6
            data = geometry.moving.data
            data.qpos[:] = q
            mujoco.mj_forward(objective.model, data)
            gid = row["geom_id"]
            local = (point - data.geom_xpos[gid]) @ data.geom_xmat[gid].reshape(3, 3)
            distances.append(float(local @ plane[:3] + plane[3]))
        assert (distances[1] - distances[0]) / 2e-6 == pytest.approx(row["jacobian"][j], abs=1e-8)


@pytest.mark.parametrize("lazy", [False, True])
def test_original_mesh_repair_keeps_root_and_all_immutable_limits(monkeypatch, lazy):
    from dataclasses import replace
    from omniretargeting._fast import mesh_repair
    monkeypatch.setattr(mesh_repair, "OPTIONS", replace(mesh_repair.OPTIONS, lazy_repair_restoration=lazy))
    objective, poses, _ = _objective(temporal=True)
    obj = _open_object(objective, poses)
    before = poses.copy()
    reference = objective.reference.copy()
    vertices = obj["mesh"].vertices.copy()
    result, report = repair_original_meshes(objective, poses, [obj])
    assert RepairLimits(objective, poses).check(result)[0]
    np.testing.assert_array_equal(poses, before)
    np.testing.assert_array_equal(objective.reference, reference)
    np.testing.assert_array_equal(obj["mesh"].vertices, vertices)
    np.testing.assert_array_equal(result[:, :7], poses[:, :7])
    assert report["after"]["moving"]["max_depth_m"] <= report["before"]["moving"]["max_depth_m"] + 1e-6
    assert report["after"]["unknown_object_components"] > 0
    assert not report["after"]["collision_free_certificate"]


def test_unconverged_proposal_cannot_bypass_nonlinear_limits(monkeypatch):
    import omniretargeting._fast.mesh_repair as repair
    objective, poses, _ = _objective(temporal=True)
    obj = _open_object(objective, poses)
    monkeypatch.setattr(repair, "_qp_step", lambda *args: (
        np.full((len(poses), len(objective.coords.qi)), 100.), "InsufficientProgress"))
    result, report = repair.repair_original_meshes(objective, poses, [obj])
    np.testing.assert_array_equal(result, poses)
    assert report["status"] == "retained"


def test_constant_motion_can_reduce_open_mesh_intersection_without_added_derivatives():
    from omniretargeting._fast.objective import FastObjective
    from omniretargeting.sequence_retargeting import prepare_sequence, sequence_config
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
        <body name="base" pos="0 0 .5"><freejoint/><geom type="sphere" size=".025"/>
          <body name="hand" pos=".3 0 0"><joint type="slide" axis="1 0 0" range="-.1 .1"/>
            <geom type="sphere" size=".05"/></body></body></worldbody></mujoco>""")
    poses = np.tile(model.qpos0, (12, 1))
    positions = np.tile([[0., 0., .5], [.3, 0., .5]], (len(poses), 1, 1))
    cfg = sequence_config({"self_collision_enabled": False})
    coords, poses, tasks, rotations = prepare_sequence(
        model, poses, positions, ["base", "hand"], {"base": "base", "hand": "hand"},
        positions[:, 0], {}, {"effectors": {}}, cfg)
    objective = FastObjective(tasks, cfg, reference=poses,
                              temporal_reference=poses[:, coords.qi],
                              root_rotations=rotations, framerate=30.)
    mesh = trimesh.Trimesh(vertices=[[.347, -.01, .49], [.347, .01, .49],
                                      [.347, .01, .51], [.347, -.01, .51]],
                           faces=[[0, 1, 2], [0, 2, 3]], process=False)
    result, report = repair_original_meshes(objective, poses, [{"mesh": mesh}])
    assert report["status"] == "corrected"
    assert report["after"]["moving"]["max_depth_m"] < report["before"]["moving"]["max_depth_m"] - 1e-5
    assert np.abs(np.diff(result[:, coords.qi], axis=0)).max() < 2e-7


@pytest.mark.parametrize("world", [False, True])
def test_free_endpoint_can_approach_unchanged_target_without_freezing_old_error(world):
    objective, poses, _ = _objective(temporal=True)
    eff = objective.effectors[1]
    eff["world"][:] = world
    eff["scale"] = 0.8
    desired = poses.copy()
    desired[:, objective.coords.qi[-1]] += 0.04
    objective.kin.evaluate(desired, False)
    target = objective.kin.xpos[:, eff["body"]]
    root = objective.kin.xpos[:, objective.tasks.root_body]
    eff["tracking"] = (target.copy() if world else
                       (target - root) / eff["scale"] + objective.tasks.roots)
    limits = RepairLimits(objective, poses)
    toward = poses.copy()
    toward[:, objective.coords.qi[-1]] += 0.02
    accepted, report = limits.check(toward)
    assert accepted, report
    assert report["max_free_endpoint_displacement_m"] > 0.003
    assert report["free_tracking_error_increase_m"] < 1e-8
    away = poses.copy()
    away[:, objective.coords.qi[-1]] -= 0.01
    valid, rejected = limits.check(away)
    assert not valid
    assert rejected["reason"] == "free endpoint fidelity"
    np.testing.assert_array_equal(toward[:, :7], poses[:, :7])


def test_free_tracking_jacobian_includes_body_relative_root_and_approach_blend():
    objective, poses, _ = _objective(temporal=True)
    objective.tasks.root_body = objective.model.body("arm").id
    hand = objective.effectors[1]
    hand["world"][:] = False
    hand["share"][:] = 0.4
    hand["scale"] = 0.8
    limits = RepairLimits(objective, poses)
    limits.kin.evaluate(poses)
    _, jac, _ = list(limits.tracking_rows(True))[1]
    column = len(objective.coords.qi) - 1
    residuals = []
    for sign in (-1, 1):
        shifted = poses.copy()
        shifted[:, objective.coords.qi[column]] += sign * 1e-6
        limits.kin.evaluate(shifted, False)
        residuals.append(list(limits.tracking_rows(False))[1][0])
    np.testing.assert_allclose((residuals[1] - residuals[0]) / 2e-6,
                               jac[..., column], atol=1e-8)


def test_repair_limit_report_keeps_three_scalar_temporal_excesses():
    import json
    objective, poses, _ = _objective(temporal=True)
    valid, report = RepairLimits(objective, poses).check(poses)
    assert valid
    assert report["temporal_excess"] == []
    json.dumps(report)


def test_local_projection_leaves_unrelated_joint_branches_exactly_fixed():
    from omniretargeting._fast.mesh_repair import _joint_projection
    objective, poses, _ = _objective(temporal=True)
    limits = RepairLimits(objective, poses)
    limits.movable_body = int(objective.model.body_weldid[objective.model.body("hand").id])
    projection = _joint_projection(limits, objective.model)
    assert projection.shape == (limits.frames * limits.width, limits.frames)
    step = (projection @ np.ones(projection.shape[1])).reshape(limits.frames, limits.width)
    np.testing.assert_array_equal(step[:, :-1], 0.)
    np.testing.assert_array_equal(step[:, -1], 1.)
    # A floating-base body has no scalar ancestors; local repair cannot lift it.
    limits.movable_body = objective.model.body("base").id
    assert _joint_projection(limits, objective.model).shape[1] == 0


def test_groups_include_static_and_both_sides_of_self_collision():
    from omniretargeting._fast.mesh_repair import _repair_groups
    objective, _, _ = _objective(temporal=True)
    model = objective.model
    foot = model.body('foot').id
    hand = model.body('hand').id
    a = int(np.flatnonzero(model.geom_bodyid == foot)[0])
    b = int(np.flatnonzero(model.geom_bodyid == hand)[0])
    static = _repair_groups(objective, {'depths': {(0, 'static', a): .02}})
    assert int(model.body_weldid[foot]) in static
    paired = _repair_groups(objective, {'depths': {(0, 'self', a, b): .02}})
    assert {int(model.body_weldid[foot]), int(model.body_weldid[hand])} <= set(paired)


def test_projection_includes_distal_joints_to_preserve_contact_during_proximal_repair():
    from omniretargeting._fast.mesh_repair import _joint_projection
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body name="base">
      <freejoint/><geom type="sphere" size=".1"/>
      <body name="proximal"><joint/><geom type="sphere" size=".04"/>
        <body name="distal" pos=".1 0 0"><joint/><geom type="sphere" size=".03"/></body>
      </body><body name="other"><joint/><geom type="sphere" size=".04"/></body>
      </body></worldbody></mujoco>''')
    limits = SimpleNamespace(movable_body=model.body('proximal').id, width=3, frames=2,
                             coords=SimpleNamespace(di=np.arange(6,9)))
    projection = _joint_projection(limits, model)
    step = (projection @ np.ones(projection.shape[1])).reshape(2, 3)
    np.testing.assert_array_equal(step, [[1, 1, 0], [1, 1, 0]])


def test_next_chain_keeps_zero_feasible_after_accepted_temporal_roundoff():
    objective, poses, _ = _objective(temporal=True)
    limits = RepairLimits(objective, poses)
    candidate = poses.copy()
    candidate[10, objective.coords.qi[-1]] += 4e-8
    assert limits.check(candidate)[0]
    _, lower, upper = limits.constraints(candidate).packed()
    assert np.max(lower) <= 1e-10
    assert np.min(upper) >= -1e-10


def test_relinearizing_verified_witnesses_matches_a_fresh_geometry_query():
    objective, poses, _ = _objective(temporal=True)
    geometry = OriginalGeometry(objective, [_open_object(objective, poses)], None)
    verified = geometry.measure(poses)
    cached = geometry.linearize(poses, verified)
    fresh = geometry.measure(poses, jacobians=True)
    assert cached['depths'] == fresh['depths']
    assert cached['failures'] == fresh['failures']
    assert [k for k, _ in cached['rows']] == [k for k, _ in fresh['rows']]
    for (_, a), (_, b) in zip(cached['rows'], fresh['rows']):
        np.testing.assert_array_equal(a['jacobian'], b['jacobian'])


def test_indeterminate_self_geometry_retains_entry_and_reports_failure(monkeypatch):
    from omniretargeting.geometry_query import GeometryQueryError
    from omniretargeting.self_collision import SelfCollisionQuery
    objective, poses, _ = _objective(temporal=True)

    def unknown(*args, **kwargs):
        raise GeometryQueryError('coincident witnesses without a reliable contact normal')

    monkeypatch.setattr(SelfCollisionQuery, 'query', unknown)
    result, report = repair_original_meshes(objective, poses)
    np.testing.assert_array_equal(result, poses)
    assert report['status'] == 'unknown_query'
    assert report['after']['query_failures'] == len(poses)
    assert not report['after']['collision_free_certificate']


def test_unknown_candidate_geometry_is_rejected_instead_of_treated_as_zero_depth(monkeypatch):
    import omniretargeting._fast.mesh_repair as repair
    from omniretargeting._fast import repair_prediction
    # Exercise trial-query rejection independently of predicted-progress stopping.
    monkeypatch.setattr(repair_prediction, "PREDICTIVE_GATING", False)
    objective, poses, _ = _objective(temporal=True)
    obj = _open_object(objective, poses)
    original = OriginalGeometry.measure
    calls = []

    def measure(self, candidate, **kwargs):
        calls.append(candidate.copy())
        if len(calls) == 1:
            return original(self, candidate, **kwargs)
        return {'rows': [], 'depths': {}, 'failures': ['indeterminate candidate normal']}

    monkeypatch.setattr(OriginalGeometry, 'measure', measure)
    monkeypatch.setattr(RepairLimits, 'check', lambda *args: (True, {}))
    monkeypatch.setattr(repair, '_qp_step', lambda *args: (
        np.full((len(poses), len(objective.coords.qi)), 1e-5), 'Solved'))
    result, report = repair_original_meshes(objective, poses, [obj])
    assert len(calls) > 1
    np.testing.assert_array_equal(result, poses)
    assert report['status'] == 'retained'
    assert any(r.get('query_failures') for r in report['history'])


def test_self_collision_projection_couples_both_sides_without_unrelated_branch():
    from omniretargeting._fast.mesh_repair import _joint_projection
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body name="base">
      <freejoint/><geom type="sphere" size=".1"/>
      <body name="first"><joint/><geom name="a" type="sphere" size=".04"/></body>
      <body name="second"><joint/><geom name="b" type="sphere" size=".04"/></body>
      <body name="other"><joint/><geom type="sphere" size=".04"/></body>
      </body></worldbody></mujoco>''')
    limits = SimpleNamespace(movable_body=model.body('first').id, width=3, frames=2,
                             points=[], coords=SimpleNamespace(di=np.arange(6, 9)))
    measurement = {'depths': {(0, 'self', model.geom('a').id, model.geom('b').id): .02}}
    projection = _joint_projection(limits, model, measurement)
    step = (projection @ np.ones(projection.shape[1])).reshape(2, 3)
    np.testing.assert_array_equal(step, [[1, 1, 0], [1, 1, 0]])


def test_projection_keeps_compensation_on_sibling_contact_chain():
    from omniretargeting._fast.mesh_repair import _joint_projection
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body name="base">
      <freejoint/><geom type="sphere" size=".1"/>
      <body name="shared"><joint/><geom type="sphere" size=".04"/>
        <body name="first"><joint/><geom type="sphere" size=".04"/></body>
        <body name="contact"><joint/><geom type="sphere" size=".04"/></body>
      </body><body name="other"><joint/><geom type="sphere" size=".04"/></body>
      </body></worldbody></mujoco>''')
    limits = SimpleNamespace(movable_body=model.body('first').id, width=4, frames=2,
                             coords=SimpleNamespace(di=np.arange(6, 10)),
                             points=[(None, np.array([model.body('contact').id]),
                                      np.full((2, 1), .001))])
    projection = _joint_projection(limits, model, {'depths': {}})
    step = (projection @ np.ones(projection.shape[1])).reshape(2, 4)
    np.testing.assert_array_equal(step, [[1, 1, 1, 0], [1, 1, 1, 0]])


def test_inherited_layout_flip_is_not_certified_as_original_layout():
    objective, poses, _ = _objective(temporal=True)
    entry = poses.copy()
    entry[:, objective.coords.qi[:2]] -= 1.5
    guard = ReferenceShape(objective, poses)
    report = guard.diagnostics(entry)
    assert report['relation_envelope']['whole_angle_max_rad'] > .3
    assert not report['source_layout_preserved']
    assert not report['layout_preserved']
    np.testing.assert_array_equal(objective.reference, poses)



def test_penetrating_regression_allowance_never_admits_new_penetration(monkeypatch):
    from omniretargeting._fast import mesh_repair

    assert mesh_repair.PENETRATING_REGRESSION_M == 5e-5
    monkeypatch.setattr(mesh_repair, "PENETRATING_REGRESSION_M", 0.0)
    # Disabled: every key keeps the 1 um regression tolerance.
    assert not mesh_repair._regressed(0.005 + 1e-6, 0.005)
    assert mesh_repair._regressed(0.005 + 2e-6, 0.005)
    monkeypatch.setattr(mesh_repair, "PENETRATING_REGRESSION_M", 5e-5)
    # A penetrating key may exceed its best depth by the allowance ...
    assert not mesh_repair._regressed(0.005 + 4e-5, 0.005)
    assert mesh_repair._regressed(0.005 + 6e-5, 0.005)
    # ... a key without penetration still may not penetrate beyond 1 um.
    assert not mesh_repair._regressed(1e-6, 0.0)
    assert mesh_repair._regressed(2e-6, 0.0)
