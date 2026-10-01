"""Original-mesh contact yield is frozen, per body, and shared by checks/cones."""
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import trimesh

from omniretargeting._fast.mesh_repair import repair_original_meshes
from omniretargeting._fast.objective import FastObjective
from omniretargeting._fast.repair_limits import CONTACT_DRIFT, RepairLimits
from omniretargeting.scene_geometry import SceneQuery
from omniretargeting.sequence_retargeting import prepare_sequence, sequence_config


DEPTH = .0012  # With the 1 mm contact budget, stays inside the unchanged slide box.


def _problem(surfaces=False):
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="root" pos="0 0 .1"><freejoint/><geom size=".01"/>
        <body name="wet" pos=".3 0 -.1">
          <joint name="wet_joint" type="slide" axis="0 0 1" range="-.1 .1"/>
          <geom name="wet_geom" type="sphere" size=".0012" mass=".1"/>
          <body name="wet_marker"><geom name="marker_geom" size=".0006" mass=".1"/></body>
          <body name="articulated" pos=".01 0 0">
            <joint name="distal_joint" type="slide" axis="0 0 1" range="-.1 .1"/>
            <geom name="distal_geom" size=".0003" mass=".1"/>
          </body>
        </body>
        <body name="clean" pos="-.3 0 0">
          <joint name="clean_joint" type="slide" axis="0 0 1" range="-.1 .1"/>
          <geom name="clean_geom" type="sphere" size=".0012" mass=".1"/>
        </body>
      </body></worldbody></mujoco>''')
    entry = np.tile(model.qpos0, (4, 1))
    data = mujoco.MjData(model)
    data.qpos[:] = entry[0]
    mujoco.mj_forward(model, data)
    names = ['root', 'wet', 'clean']
    positions = np.tile([data.xpos[model.body(n).id].copy() for n in names], (4, 1, 1))
    cfg = sequence_config({'self_collision_enabled': False})
    coords, entry, tasks, rotations = prepare_sequence(
        model, entry, positions, names, {n: n for n in names}, positions[:, 0],
        {}, {'effectors': {}}, cfg)
    objective = FastObjective(tasks, cfg, reference=entry,
                              temporal_reference=entry[:, coords.qi],
                              root_rotations=rotations, framerate=30.)
    # Real FK and Jacobians; the task dictionaries use the production layout.
    bodies = np.array([model.body(n).id for n in ('wet_marker', 'clean')])
    points = np.tile(data.xpos[bodies], (4, 1, 1))
    targets = points.copy()
    targets[..., 2] -= .0003  # Exercise acquired error + tolerance, not a zero-error special case.
    if surfaces:
        objective.surfaces = [dict(
            bodies=bodies, geoms=np.array([model.geom(n).id for n in ('marker_geom', 'clean_geom')]),
            local=np.zeros((2, 3)), targets=targets, active=np.ones((4, 2), bool),
            task=SimpleNamespace(cfg={'regression': CONTACT_DRIFT}))]
    else:
        objective.effectors = [dict(
            name=name, body=body, offset=np.zeros((4, 3)), active=np.ones(4, bool),
            managed=np.zeros(4, bool), contact=targets[:, i], tracking=points[:, i],
            world=np.ones(4, bool), scale=1., share=np.zeros(4), approach=points[:, i],
            surface_only=np.zeros(4, bool), normal=np.tile([0., 0., 1.], (4, 1)))
            for i, (name, body) in enumerate(zip(('wet_contact', 'clean_contact'), bodies))]
    return objective, entry


def _baseline(objective, depth=DEPTH):
    gid = objective.model.geom('wet_geom').id
    return dict(depths={(f, 'static', gid): depth for f in range(objective.frames)},
                failures=[], complete=True)


def _move(objective, entry, distance, joint='wet_joint'):
    candidate = entry.copy()
    candidate[:, objective.model.joint(joint).qposadr[0]] += distance
    return candidate


def _linear_accepts(limits, origin, candidate):
    bounds = limits.constraints(origin)
    step = (candidate[:, limits.coords.qi] - origin[:, limits.coords.qi]).ravel()
    matrix, lower, upper = bounds.packed()
    value = matrix @ step
    if np.any(value < lower - 1e-12) or np.any(value > upper + 1e-12):
        return False
    return all(np.all(np.linalg.norm(residual + (jac @ step).reshape(residual.shape), axis=1)
                      <= allowed + 1e-12) for jac, residual, allowed in bounds.cones())


@pytest.mark.parametrize('surfaces', [False, True])
@pytest.mark.parametrize('correcting', [False, True])
def test_own_body_yield_allows_outward_motion_but_caps_extra_drift(surfaces, correcting):
    objective, entry = _problem(surfaces)
    limits = RepairLimits(objective, entry)
    limits.freeze_penetration_yield(_baseline(objective))
    # When another group is selected, this penetrating body's contact is anchored.
    limits.movable_body = objective.model.body('wet' if correcting else 'clean').id
    bound = CONTACT_DRIFT + DEPTH
    for distance, expected in [(DEPTH, True), (bound, True), (bound + 2e-7, False)]:
        candidate = _move(objective, entry, distance)
        assert limits.check(candidate)[0] == expected
        assert _linear_accepts(limits, entry, candidate) == expected
        if expected and not correcting:
            assert limits.check(candidate)[1]['max_contact_drift_m'] == pytest.approx(distance)
    # Neither calls nor rejected proposals change nominal contact classification or references.
    assert all(np.all(budget == CONTACT_DRIFT) for _, _, budget in limits.point_groups())
    np.testing.assert_array_equal(limits.entry, entry)
    np.testing.assert_array_equal(objective.reference, entry)


@pytest.mark.parametrize('surfaces', [False, True])
@pytest.mark.parametrize('correcting', [False, True])
def test_clean_contact_keeps_one_mm_even_while_another_body_penetrates(surfaces, correcting):
    objective, entry = _problem(surfaces)
    limits = RepairLimits(objective, entry)
    limits.freeze_penetration_yield(_baseline(objective))
    limits.movable_body = objective.model.body('clean' if correcting else 'wet').id
    for distance, expected in [(CONTACT_DRIFT, True), (CONTACT_DRIFT + 2e-7, False)]:
        candidate = _move(objective, entry, distance, 'clean_joint')
        assert limits.check(candidate)[0] == expected
        assert _linear_accepts(limits, entry, candidate) == expected


@pytest.mark.parametrize('correcting', [False, True])
def test_existing_half_tolerance_reserve_and_relinearization_are_unchanged(correcting):
    objective, entry = _problem()
    limits = RepairLimits(objective, entry)
    limits.freeze_penetration_yield(_baseline(objective))
    limits.movable_body = objective.model.body('wet' if correcting else 'clean').id
    edge = CONTACT_DRIFT + DEPTH
    in_reserve = _move(objective, entry, edge + 7.5e-8)
    assert limits.check(in_reserve)[0]  # Nonlinear tolerance is still 1e-7.
    assert not _linear_accepts(limits, entry, in_reserve)  # Initial cone reserves half.
    assert _linear_accepts(limits, in_reserve, in_reserve)  # Accepted roundoff keeps zero feasible.
    invalid = _move(objective, entry, edge + 2e-7)
    assert not limits.check(invalid)[0]
    assert not _linear_accepts(limits, invalid, invalid)  # Bad proposals cannot enlarge the cap.


def test_depths_are_frame_local_maxima_on_welds_not_articulated_neighbors_or_self_pairs():
    objective, entry = _problem()
    limits = RepairLimits(objective, entry)
    model = objective.model
    wet, marker, distal, clean = [model.geom(n).id for n in
                                  ('wet_geom', 'marker_geom', 'distal_geom', 'clean_geom')]
    measurement = dict(depths={(0, 'static', wet): .0012,
                               (0, 'moving', 0, 'observed_open_surface', 4, marker): .0008,
                               (0, 'moving', 0, 'reverse_surface', -1, marker): .0015,
                               (1, 'static', wet): .0004,
                               (0, 'self', clean, distal): .1}, failures=[], complete=True)
    limits.freeze_penetration_yield(measurement)
    np.testing.assert_array_equal(limits.penetration_yield[:, model.body('wet').id],
                                  [.0015, .0004, 0., 0.])
    np.testing.assert_array_equal(limits.penetration_yield[:, model.body('clean').id], 0.)
    np.testing.assert_array_equal(limits.penetration_yield[:, model.body('articulated').id], 0.)
    bodies = np.array([model.body('wet_marker').id, model.body('articulated').id])
    anchor, _ = limits.contact_budgets(0, bodies, np.full((4, 2), CONTACT_DRIFT))
    assert anchor[0, 0] == pytest.approx(CONTACT_DRIFT + .0015)
    assert anchor[0, 1] == CONTACT_DRIFT
    measurement['depths'][(0, 'static', wet)] = .5
    assert limits.penetration_yield[0, model.body('wet').id] == .0015
    with pytest.raises(RuntimeError, match='already frozen'):
        limits.freeze_penetration_yield(measurement)
    with pytest.raises(ValueError):
        limits.penetration_yield[0, model.body('wet').id] = 1.


@pytest.mark.parametrize('bad', ['incomplete', 'failed', 'nan', 'infinite', 'negative'])
def test_unreliable_measurement_does_not_release_contacts(bad):
    objective, entry = _problem()
    limits = RepairLimits(objective, entry)
    measured = _baseline(objective)
    if bad == 'incomplete':
        measured['complete'] = False
    elif bad == 'failed':
        measured['failures'] = ['unknown original surface']
    else:
        measured['depths'][next(iter(measured['depths']))] = dict(
            nan=np.nan, infinite=np.inf, negative=-.001)[bad]
    with pytest.raises(ValueError):
        limits.freeze_penetration_yield(measured)
    np.testing.assert_array_equal(limits.penetration_yield, 0.)
    assert not limits.check(_move(objective, entry, .0015))[0]


def test_yield_does_not_relax_root_temporal_or_free_tracking_limits():
    objective, entry = _problem()
    limits = RepairLimits(objective, entry)
    limits.freeze_penetration_yield(_baseline(objective))
    chatter = _move(objective, entry, .0015)
    chatter[1, objective.model.joint('wet_joint').qposadr[0]] += 1e-5
    assert limits.check(chatter)[1]['reason'] == 'per-joint temporal budget'
    drift = entry.copy()
    drift[:, 2] += .001
    assert limits.check(drift)[1]['reason'] == 'root changed'
    objective.effectors[0]['active'][:] = False
    free_limits = RepairLimits(objective, entry)
    free_limits.freeze_penetration_yield(_baseline(objective))
    assert free_limits.check(_move(objective, entry, .0015))[1]['reason'] == 'free endpoint fidelity'


@pytest.mark.parametrize('guard_name', ['contact', 'acquisition'])
def test_same_depth_remains_compatible_with_the_existing_immutable_guards(monkeypatch, guard_name):
    import omniretargeting._fast.contact_guard as contact
    import omniretargeting._fast.acquisition_guard as acquisition
    objective, entry = _problem()
    limits = RepairLimits(objective, entry)
    limits.freeze_penetration_yield(_baseline(objective))
    module = contact if guard_name == 'contact' else acquisition
    # Supply the guards' external geometry provider the same known per-body depths.
    # Their real acceptance code and independent fixed references are unchanged.
    monkeypatch.setattr(module, 'body_penetration', lambda *args: limits.penetration_yield.copy())
    guard = (contact.ContactGuard(objective, entry, preserve_positions=True)
             if guard_name == 'contact' else acquisition.AcquisitionNormalGuard(objective, entry))
    objective.candidate_guards = [guard]
    for body in ('wet', 'clean'):  # Correcting, then anchored, for the penetrating point.
        limits.movable_body = objective.model.body(body).id
        for distance, expected in [(CONTACT_DRIFT + DEPTH, True),
                                   (CONTACT_DRIFT + DEPTH + 2e-7, False)]:
            candidate = _move(objective, entry, distance)
            assert guard.accepts(candidate) == expected
            assert limits.check(candidate)[0] == expected
            assert _linear_accepts(limits, entry, candidate) == expected


def test_entry_original_mesh_measurement_is_wired_before_the_first_qp(monkeypatch):
    import omniretargeting._fast.mesh_repair as repair
    objective, entry = _problem()
    floor = trimesh.creation.box(extents=[2, 2, .2])
    floor.apply_translation([0, 0, -.1])
    observed = []

    def probe(limits, poses, geometry, measured, protected):
        observed.append(limits.penetration_yield.copy())
        assert limits.penetration_yield[:, objective.model.body('wet').id] == pytest.approx(DEPTH)
        np.testing.assert_array_equal(limits.penetration_yield[:, objective.model.body('clean').id], 0.)
        return None, 'diagnostic_no_step'

    monkeypatch.setattr(repair, '_qp_step', probe)
    result, report = repair_original_meshes(objective, entry, scene_query=SceneQuery(floor))
    assert observed
    assert report['before']['static']['max_depth_m'] == pytest.approx(DEPTH)
    np.testing.assert_array_equal(result, entry)
