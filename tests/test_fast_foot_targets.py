"""Morphology and contact handoff contracts, independent of clip labels."""
import mujoco
import numpy as np

from omniretargeting._fast.foot_targets import (
    handover, role_envelope, root_reach_radius, pack_sole_morphology,
    prepare_sole_handover,
)
from omniretargeting._fast.reference import kinematic_tracking_tasks, height_tracking_tasks
from omniretargeting._fast.rows import _Accumulator
from tests.test_fast_backend import _objective


def test_free_sole_uses_morphology_but_height_uses_only_observed_contacts():
    obj, _, _ = _objective()
    task = obj.tasks.effectors['foot']['task']
    spec = dict(track_orientation=True, _scale_calibrated=True, preserve_world_target=True)
    priority = {'effectors': {'foot': spec}}
    original = task.tracking_positions.copy(), task.contact_positions.copy(), task.contact_mask.copy()
    source = kinematic_tracking_tasks({'foot': task}, priority)['foot']
    height = height_tracking_tasks({'foot': task}, priority)['foot']
    assert not source.swing_world_mask.any()
    assert not source.contact_mask.any()
    np.testing.assert_array_equal(height.contact_mask, task.contact_mask)
    assert np.all(height.contact_modes[task.contact_mask] == 'sliding')
    for before, after in zip(original, (task.tracking_positions, task.contact_positions, task.contact_mask)):
        np.testing.assert_array_equal(before, after)


def test_handover_keeps_world_anchors_and_blends_on_both_sides():
    tracking = np.zeros((24, 3))
    anchors = np.tile([.18, 0., 0.], (24, 1))
    mask = np.zeros(24, bool)
    mask[9:14] = True
    weights, targets = handover(mask, tracking, anchors, 5)
    effective = weights[:, None] * targets
    np.testing.assert_array_equal(effective[mask], anchors[mask])
    assert np.all(np.diff(effective[3:10, 0]) >= 0)
    assert np.all(np.diff(effective[13:20, 0]) <= 0)
    assert abs(effective[9, 0] - effective[8, 0]) < .02
    assert abs(effective[14, 0] - effective[13, 0]) < .02
    assert np.max(np.abs(np.diff(effective[:, 0]))) < .05


def test_short_gap_connects_anchors_without_mid_swing_jump():
    tracking = np.zeros((10, 3))
    anchors = tracking.copy(); anchors[:5, 0] = .1; anchors[5:, 0] = -.1
    mask = np.ones(10, bool); mask[3:7] = False
    weight, target = handover(mask, tracking, anchors, 5)
    curve = weight[:, None] * target
    assert np.all(np.diff(curve[2:8, 0]) < 0)
    np.testing.assert_array_equal(curve[mask], anchors[mask])


def test_support_envelope_is_make_before_break_without_creating_contact():
    original = np.array([1., 1., 2., 4., 2., 4., 2., 1.])
    active = np.array([0, 0, 1, 1, 1, 1, 1, 0], bool)
    result = role_envelope(original, active, 3)
    assert result[2] > original[2] and result[6] > original[6]
    assert result[4] > original[4]
    np.testing.assert_array_equal(result[~active], original[~active])
    assert np.all(result >= original)


def test_root_sphere_contains_joint_limit_samples():
    obj, poses, _ = _objective()
    model = obj.model; data = mujoco.MjData(model)
    body = model.body('foot').id; offset = np.array([.08, 0., -.02])
    root, radius = root_reach_radius(model, body, offset)
    rng = np.random.default_rng(43)
    for _ in range(80):
        data.qpos[:] = poses[0]
        data.qpos[obj.coords.qi] = rng.uniform(obj.coords.lower, obj.coords.upper)
        mujoco.mj_kinematics(model, data)
        point = data.xpos[body] + data.xmat[body].reshape(3, 3) @ offset
        assert np.linalg.norm(point - data.xpos[root]) <= radius + 1e-12


def test_hip_relative_target_jacobian_and_chunking():
    obj, poses, _ = _objective()
    spec = obj.tasks.effectors['foot']; spec['_scale_calibrated'] = True
    pack_sole_morphology(obj)
    eff = obj.effectors[0]; eff['active'][:] = False; eff['managed'][:] = False
    eff['share'][:] = .3; eff['world'][:] = False
    frame = 17
    chunk = obj.for_frames(slice(frame, frame + 1))
    q = poses[frame:frame+1]
    def evaluate(value, jacobian):
        chunk.kin.evaluate(value, jacobian)
        acc = _Accumulator(1, chunk.width, jacobian)
        chunk._effector_terms(acc, {})
        return acc
    analytic = evaluate(q, True).gradient[0]
    numeric = np.zeros(chunk.width)
    for col in range(chunk.width):
        step = np.zeros((1, chunk.width)); step[0, col] = 1e-6
        numeric[col] = (evaluate(chunk.coords.integrate(q, step, 1), False).cost
                        - evaluate(chunk.coords.integrate(q, step, -1), False).cost) / 2e-6
    # The accumulator stores J.T @ r, half the derivative of its squared cost.
    np.testing.assert_allclose(2 * analytic, numeric, atol=2e-5, rtol=2e-5)


def test_unreachable_plane_yields_without_modifying_evidence():
    obj, poses, _ = _objective(temporal=True)
    spec = obj.tasks.effectors['foot']; spec['_scale_calibrated'] = True
    pack_sole_morphology(obj)
    eff = obj.effectors[0]; eff['contact'][:, 2] = -2.
    original = spec['task'].contact_mask.copy()
    report = prepare_sole_handover(obj)['foot']
    assert report['unreachable_frames'] == np.flatnonzero(original).tolist()
    assert not eff['active'].any()
    np.testing.assert_array_equal(spec['task'].contact_mask, original)


def test_handoff_does_not_certify_a_foot_that_only_lands_after_seat_release():
    from omniretargeting._fast.support_handoff import handoff_gaps
    outgoing = np.array([1, 1, 1, 0, 0], bool)
    incoming = np.ones((5, 2), bool)
    gaps = np.array([[.1, .1], [.05, .05], [.012, .003], [.002, .002], [0., 0.]])
    report = handoff_gaps(outgoing, incoming, gaps)[0]
    assert report['last_overlap_frame'] == 2
    assert not report['acquired_before_release']
    gaps[2] = .003
    assert handoff_gaps(outgoing, incoming, gaps)[0]['acquired_before_release']


def test_support_hold_role_flapping_is_not_an_outgoing_contact_release():
    from omniretargeting._fast.support_handoff import supported_contact_runs
    active = np.array([1, 1, 1, 1, 0, 1, 1], bool)
    support = np.array([1, 0, 1, 0, 0, 0, 0], bool)
    np.testing.assert_array_equal(supported_contact_runs(active, support),
                                 [1, 1, 1, 1, 0, 0, 0])


def test_height_registration_bound_uses_collision_sole_not_semantic_point():
    from dataclasses import replace
    from omniretargeting._fast.reference import bound_sole_registration
    obj, poses, _ = _objective()
    spec = obj.tasks.effectors['foot']; spec['_scale_calibrated'] = True
    spec['task'] = replace(spec['task'], surface_points=spec['task'].surface_points + [0., 0., .1])
    result, shift = bound_sole_registration(obj, poses)
    assert shift.max() > .05
    unchanged = np.delete(np.arange(poses.shape[1]), obj.coords.root_q + 2)
    np.testing.assert_array_equal(result[:, unchanged], poses[:, unchanged])
    obj.kin.evaluate(result, with_jacobian=False)
    geom = obj.model.geom('foot_geom').id
    from omniretargeting.scene_geometry import geom_surface_offsets
    offsets = geom_surface_offsets(obj.model, geom, obj.cfg['collision_spacing_m'])
    points = obj.kin.geom_points(np.full(len(offsets), geom), offsets)
    # The geometric extent includes the sole's lower corners, not just the tip.
    task = spec['task']
    gaps = (points - task.surface_points[:, None])[..., 2].min(1)
    assert np.all(gaps[task.contact_mask] >= obj.cfg['scene_clearance_m'] - 1e-10)
