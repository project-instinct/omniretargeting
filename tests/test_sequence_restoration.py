"""Nonlinear feasibility restoration cannot relax fixed motion budgets."""

import mujoco
import numpy as np

from omniretargeting._sequence.continuity_guard import SequenceContinuityGuard
from omniretargeting._sequence.pose_guard import SequencePoseGuard
from omniretargeting._sequence.surface_guard import SurfaceSequenceGuard
from omniretargeting.sequence_retargeting import prepare_sequence, sequence_config


def test_restoration_preserves_translation_reference_and_each_joint_continuity():
    model = mujoco.MjModel.from_xml_string("""<mujoco><compiler angle="radian"/>
    <worldbody><body name="root"><freejoint/><geom size=".01"/>
      <body name="lever"><joint axis="0 1 0" range="-2 2"/><geom size=".01"/>
        <body name="tip" pos="0 0 1"><geom size=".01"/></body>
      </body></body></worldbody></mujoco>""")
    reference = np.tile(model.qpos0, (5, 1))
    source = np.tile([[[0.0, 0, 0], [0, 0, 1]]], (len(reference), 1, 1))
    coords, _, tasks, rotations = prepare_sequence(
        model,
        reference,
        source,
        ["root", "tip"],
        {"root": "root", "tip": "tip"},
        source[:, 0],
        {},
        {},
        sequence_config({"self_collision_enabled": False}),
    )
    guard = SurfaceSequenceGuard(tasks, reference, 0.001, preserve_contacts=False)
    guard.pose = SequencePoseGuard(tasks, reference)
    guard.continuity = SequenceContinuityGuard(tasks, reference, rotations)
    current = reference.copy()
    current[:, -1] = 0.16
    current[:, coords.root_q] += 0.025
    assert guard.accepts(current)
    step = np.zeros((len(current), coords.width))
    step[:, coords.scalar_cols] = 0.04
    step[:, coords.inverse[coords.root_d + 3]] = 0.08
    step[:, coords.inverse[coords.root_d]] = 0.02  # beyond the root translation ball
    candidate = coords.integrate(current, step, 1)
    assert not guard.accepts(candidate)
    lower, upper = coords.bounds(current, 0.12, 0.03)
    restored = guard.restore(current, candidate, lower, upper)
    assert restored is not None and guard.accepts(restored)
    assert np.linalg.norm(restored - candidate) < np.linalg.norm(current - candidate)
    assert np.linalg.norm(restored - current) > 0.01
    np.testing.assert_array_equal(guard.pose.reference, reference)
    np.testing.assert_allclose(np.diff(restored[:, coords.qi], axis=0), 0, atol=2e-7)
    # Rotations beyond the former 0.2 rad envelope are reported, not restored.
    assert guard.pose.diagnostics(restored)["maximum_body_rotation_change_rad"] > 0.2


def test_scene_clearance_does_not_become_a_per_frame_contact_role():
    from omniretargeting.contact_phase import attach_contact_phase
    from tests.test_contact_phase import case
    from tests.test_full_body_surfaces import _markers

    model, q, surface, endpoints, priority = case(stance=True)
    attach_contact_phase(surface, None, endpoints, priority)
    names = ["root", "abdomen", "foot"]
    source = _markers(model, q, names)
    _, _, tasks, _ = prepare_sequence(
        model,
        q,
        source,
        names,
        {n: n for n in names},
        source[:, 0],
        endpoints,
        priority,
        sequence_config(),
        extra_tasks=(surface,),
    )
    guard = SurfaceSequenceGuard(tasks, q, 0.001, preserve_contacts=False)
    errors = guard.priority_errors(q)

    def conflicting_clearance(*args, **kwargs):
        yield np.array([100.0]), np.zeros(tasks.coords.width)

    tasks.clearance_rows = conflicting_clearance
    np.testing.assert_array_equal(guard.priority_errors(q), errors)
