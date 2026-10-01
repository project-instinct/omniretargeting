"""Contacts cannot spend the same source-shape budget again in each stage."""
import numpy as np

from omniretargeting._fast.reference import ReferenceShape
from omniretargeting._fast.solver import optimize
from tests.test_fast_backend import _objective


def test_reference_tracking_uses_adapter_semantics_without_contact_influence():
    from dataclasses import replace
    from omniretargeting._fast.reference import kinematic_tracking_tasks
    from tests.test_fast_backend import _problem

    _, _, _, _, tasks, _ = _problem(frames=8)
    source = {name: replace(spec['task'], preserve_world_tracking=name == 'foot',
                           swing_world_mask=np.ones(8, bool))
              for name, spec in tasks.effectors.items()}
    reference = kinematic_tracking_tasks(source, {})
    assert reference['foot'].swing_world_mask.all()
    assert not reference['hand'].swing_world_mask.any()
    for name, task in reference.items():
        assert not task.contact_mask.any()
        assert not task.managed_frames.any()
        assert not task.approach[0].any()
        np.testing.assert_array_equal(task.tracking_positions, source[name].tracking_positions)
        np.testing.assert_array_equal(task.contact_positions, source[name].contact_positions)
    overridden = kinematic_tracking_tasks(source, {'effectors': {
        'hand': {'preserve_world_target': True},
        'foot': {'preserve_world_target': False},
    }})
    assert overridden['hand'].swing_world_mask.all()
    assert not overridden['foot'].swing_world_mask.any()


def test_world_pose_and_source_checks_survive_reference_changes():
    objective, seed, _ = _objective()
    guard = ReferenceShape(objective, seed)
    assert guard.accepts(seed)
    lifted = seed.copy()
    lifted[:, objective.coords.root_q + 2] += .4
    assert guard.accepts(lifted)  # morphology height is independent of posture
    changed = seed.copy()
    # A hip turn this large changes the whole body by more than 0.3 rad.
    changed[:, objective.coords.qi[0]] += .4
    assert not guard.accepts(changed)
    # Resetting an objective reference cannot make a repeated correction valid.
    objective.reference = changed.copy()
    assert not guard.accepts(changed)
    np.testing.assert_array_equal(guard.seed, seed)
    offset = seed.copy()
    q = objective.coords.root_q
    offset[:, q:q + 2] += [3, 4]
    projected = guard.project(offset)
    np.testing.assert_allclose(np.linalg.norm(projected[:, q:q + 2] - seed[:, q:q + 2], axis=1), guard.xy)


def test_solver_checks_every_candidate_and_continues_inside_envelope():
    objective, seed, _ = _objective()
    guard = ReferenceShape(objective, seed)
    objective.source_shape = guard
    # Force an incompatible target, while leaving a useful in-bounds descent.
    for eff in objective.effectors:
        eff['contact'] = eff['contact'] + [0.8, 0.0, 0.0]
    accepted = []
    solved, history = optimize(objective, seed.copy(), iterations=5,
                               callback=lambda q, _: accepted.append(q))
    assert history and accepted
    assert all(guard.accepts(q) for q in accepted)
    assert guard.accepts(solved)
    assert np.isfinite(solved).all()
    assert history[-1]['after'] < history[0]['cost']


def test_bad_initial_source_fit_is_reported_against_the_source():
    objective, seed, _ = _objective()
    good = ReferenceShape(objective, seed).diagnostics(seed)
    # Give one source edge a conflicting direction, without touching the robot.
    human = objective.tasks.human.copy()
    human[:, 2] = human[:, 1] + [0.8, 0, 0]
    objective.tasks.human = human
    guard = ReferenceShape(objective, seed)
    report = guard.diagnostics(seed)
    assert guard.accepts(seed)  # repair-only baseline, never a certificate
    assert report['posture_limit'].startswith('none')
    assert report['seed_source_bone_angle_max_rad'] > good['seed_source_bone_angle_max_rad'] + .3
    assert report['seed_source_bone_angle_edge'] == [objective.tasks.names[1], objective.tasks.names[2]]


def test_height_registration_reaches_translation_optimum_without_changing_pose():
    from omniretargeting._fast.reference import register_height

    objective, poses, _ = _objective()
    # An immutable world contact demands another height. The registration
    # solves only translation even though joint motion could lower the cost.
    for eff in objective.effectors:
        eff['contact'] = eff['contact'] + [0., 0., -.15]
    before = objective.evaluate(poses)[0].cost
    fitted, shift = register_height(objective, poses)
    acc, residual, jac, _, _ = objective.evaluate(fitted)
    assert acc.cost < before
    assert np.max(np.abs(shift)) > .01
    q = objective.coords.root_q
    unchanged = np.delete(np.arange(poses.shape[1]), q + 2)
    np.testing.assert_array_equal(fitted[:, unchanged], poses[:, unchanged])
    col = objective.coords.inverse[objective.coords.root_d + 2]
    ids = np.arange(len(poses)) * objective.coords.width + col
    gradient = acc.gradient[:, col] + jac[:, ids].T @ residual
    np.testing.assert_allclose(gradient, 0, atol=1e-5)


def test_contact_height_cannot_redefine_the_source_pose_and_self_repair_is_unconditional(monkeypatch):
    from dataclasses import replace
    from types import SimpleNamespace

    from omniretargeting._fast import pipeline
    from tests.test_fast_backend import _problem

    model, cfg, coords, poses, task_set, _ = _problem(frames=8)
    priority = {'effectors': task_set.effectors}
    originals = {name: spec['task'] for name, spec in task_set.effectors.items()}
    source = task_set.human.copy()
    root = task_set.roots.copy()
    common = {'environment_points': None, 'root_orientation_targets': None,
                  'body_orientation_targets': [], 'framerate': 30., 'scene_query': None}
    repairs = []
    monkeypatch.setattr(pipeline, 'repair_original_meshes',
                        lambda obj, motion, *args: (repairs.append(len(motion)) or motion, {'status': 'tested'}))
    references = []
    for height in (0., .2):
        tasks = {name: replace(task, contact_positions=task.contact_positions + [0., 0., height])
                 for name, task in originals.items()}
        owner = SimpleNamespace(_scene_objects=[])
        _, report = pipeline.solve_fast_sequence(
            owner, model, poses, source, task_set.names,
            {n: model.body(b).name for n, b in zip(task_set.names, task_set.bodyids)},
            root, tasks, priority, dict(cfg, iterations=2, self_collision_enabled=True), common, (),
        )
        references.append(owner.last_fast_source_pose.copy())
        assert report['stages']['contact_scene']['original_mesh_repair']['status'] == 'tested'
    # Object/contact locations cannot redefine even the height of the source
    # reference. They are acquired later inside its immutable pose envelope.
    np.testing.assert_array_equal(references[0], references[1])
    assert repairs == [8, 8]  # no scene and no object-grid failure to trigger it
    np.testing.assert_array_equal(source, task_set.human)
    np.testing.assert_array_equal(root, task_set.roots)
