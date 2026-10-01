"""Skip a centre prepass only when its culling mask is provably all True."""

import mujoco
import numpy as np
import pytest
import trimesh

from omniretargeting.scene_geometry import (
    ScenePenetrationProbe,
    SceneQuery,
    collision_geom_ids,
)
from omniretargeting.trajectory_validation import (
    TrajectoryValidator,
    local_repair_config,
)
from tests.test_scene_query_acceleration import assert_same, wall


def robot():
    return mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="robot"><freejoint/>
        <geom type="capsule" size=".04 .1"/>
        <geom type="sphere" size=".03" pos=".18 .02 .02"/>
      </body></worldbody></mujoco>''')


def scene_args(case):
    box = trimesh.creation.box(extents=[.2, .3, .4])
    box.apply_translation([.3, 0, 0])
    if case == "closed":
        return (box,), False
    if case == "evidence":
        return (wall(0), [[0, 0, 0]], [[1, 0, 0]]), False
    if case == "conflicting_evidence":
        return (wall(0), [[0, 0, 0]] * 2, [[1, 0, 0], [-1, 0, 0]]), False
    if case == "partial_evidence":
        return (trimesh.util.concatenate([wall(0), wall(4)]),
                [[0, 0, 0]], [[1, 0, 0]]), True
    if case == "mixed":
        return (trimesh.util.concatenate([wall(0), box]),), True
    return (wall(0),), True


@pytest.mark.parametrize("case", ["open", "mixed", "partial_evidence", "closed",
                                  "evidence", "conflicting_evidence"])
@pytest.mark.parametrize("full_geometry", [False, True])
def test_prepass_elision_preserves_every_collision_field_and_budget(case, full_geometry):
    model = robot()
    args, skip = scene_args(case)
    fast = SceneQuery(*args, full_geometry=full_geometry)
    legacy = SceneQuery(*args, full_geometry=full_geometry)
    assert fast.has_unsigned_only_components == skip
    # Force only the probe's historical centre prepass. The actual query,
    # triangle data, evidence, sample offsets, and narrow phase stay identical.
    legacy.has_unsigned_only_components = False
    geom_ids = collision_geom_ids(model)
    fast_probe = ScenePenetrationProbe(model, args[0], geom_ids, .05, query=fast)
    legacy_probe = ScenePenetrationProbe(model, args[0], geom_ids, .05, query=legacy)
    trajectory = np.tile(model.qpos0, (4, 1))
    trajectory[:, 0] = [-.01, 0, .04, 2.0]

    actual = fast_probe.probe_trajectory(trajectory)
    expected = legacy_probe.probe_trajectory(trajectory)

    for after, before in zip(actual, expected):
        assert_same(after, before)  # Includes ordering, NaNs, witnesses, IDs, and unknowns.
    assert fast.query_calls == (1 if skip else 2) * len(trajectory)
    assert legacy.query_calls == 2 * len(trajectory)
    assert legacy.query_points - fast.query_points == (len(trajectory) * len(geom_ids) if skip else 0)
    assert fast_probe.query_work_bound == legacy_probe.query_work_bound
    assert fast.query_points <= len(trajectory) * fast_probe.query_work_bound
    if not skip and case != "conflicting_evidence":
        # Confirm the ordinary confirmed-free culling path still removes far samples.
        assert len(actual[-1]["points"]) == 0
    if skip:
        sample_count = sum(map(len, fast_probe.offsets))
        assert np.all(actual[0]["uncertain"][:sample_count])
        assert len(actual[-1]["points"]) >= sample_count


def test_dense_geometry_and_acceptance_are_identical_with_unknown_components():
    model = robot()
    args, _ = scene_args("mixed")
    query = SceneQuery(*args)
    legacy_query = SceneQuery(*args)
    legacy_query.has_unsigned_only_components = False
    cfg = local_repair_config({"self_collision_enabled": False, "max_query_points": None})
    trajectory = np.tile(model.qpos0, (2, 1))
    trajectory[:, 0] = [-.01, .02]
    fast = TrajectoryValidator(model, trajectory, query, {}, {}, cfg, 30)
    legacy = TrajectoryValidator(model, trajectory, legacy_query, {}, {}, cfg, 30)

    actual, expected = fast.geometry(trajectory), legacy.geometry(trajectory)

    # Every reported geometric field and witness is unchanged.
    for key in expected:
        if isinstance(expected[key], np.ndarray):
            np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
        else:
            assert actual[key] == expected[key], key
    assert query.query_points < legacy_query.query_points
    assert fast.compare_geometry(expected, actual, allow_existing_uncertainty=True) == []
    assert fast.compare_geometry(expected, actual) == legacy.compare_geometry(expected, expected)


def test_batched_refinement_halfspaces_match_budgeted_sequential_queries():
    import cvxpy as cp

    from omniretargeting._refinement.guard import RefinementGuard

    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body pos="0 0 .05">
        <joint type="slide" axis="0 0 1"/><geom type="sphere" size=".04"/>
        </body></worldbody></mujoco>''')
    poses = np.array([[0.], [.005], [.01]])
    mesh = trimesh.creation.box(extents=[1, 1, .05])
    values, calls = [], []
    for limit in (None, 10_000_000):
        query = SceneQuery(mesh)
        cfg = local_repair_config({"max_query_points": limit})
        validator = TrajectoryValidator(model, poses, query, {}, {}, cfg, 30)
        guard = RefinementGuard(validator)
        guard._baseline_geometry()
        before = query.query_calls
        step = cp.Parameter(3, value=[.003, -.002, .001])
        constraints = guard._scene_constraints(poses, step)
        assert len(constraints) == 1
        values.append(constraints[0].expr.value)
        calls.append(query.query_calls - before)
    np.testing.assert_allclose(values[0], values[1], atol=1e-14, rtol=1e-14)
    assert calls[0] < calls[1]


@pytest.mark.parametrize("case", ["closed", "open", "mixed", "partial_evidence", "conflicting_evidence"])
@pytest.mark.parametrize("full_geometry", [False, True])
def test_batching_preserves_culling_and_every_original_geometry_witness(case, full_geometry):
    from scipy.spatial.transform import Rotation

    from omniretargeting.scene_probe_batch import batched_probe

    model = robot()
    args, _ = scene_args(case)
    grouped_query = SceneQuery(*args, full_geometry=full_geometry)
    direct_query = SceneQuery(*args, full_geometry=full_geometry)
    ids = collision_geom_ids(model)
    grouped = ScenePenetrationProbe(model, args[0], ids, .05, query=grouped_query)
    direct = ScenePenetrationProbe(model, args[0], ids, .05, query=direct_query)
    poses = np.tile(model.qpos0, (4, 1))
    poses[:, 0] = [-.01, 0, .04, 2.]
    poses[:, 3:7] = Rotation.from_euler("xyz", [[0, 0, 0], [.1, .2, 0], [.2, -.1, .3], [0, .3, 0]]).as_quat(scalar_first=True)
    for first, second in zip(batched_probe(grouped, poses), direct.probe_trajectory(poses)):
        assert_same(first, second)
    assert grouped_query.query_points == direct_query.query_points
    assert grouped_query.query_calls < direct_query.query_calls


def test_interframe_batching_keeps_exact_validation_and_reduces_scene_calls(monkeypatch):
    model = robot()
    args, _ = scene_args("mixed")
    trajectory = np.tile(model.qpos0, (20, 1))
    trajectory[:, 0] = np.linspace(-.1, .3, len(trajectory))
    cfg = local_repair_config({"max_query_points": None, "full_geometry": False})
    grouped_query = SceneQuery(*args, full_geometry=False)
    grouped = TrajectoryValidator(model, trajectory, grouped_query, {}, {}, cfg, 30)
    actual = grouped.geometry(trajectory)

    monkeypatch.setattr(
        "omniretargeting.trajectory_validation.prepare_intervals",
        lambda *args: None,
    )
    per_interval_query = SceneQuery(*args, full_geometry=False)
    per_interval = TrajectoryValidator(
        model, trajectory, per_interval_query, {}, {}, cfg, 30
    )
    expected = per_interval.geometry(trajectory)
    for key in expected:
        if isinstance(expected[key], np.ndarray):
            np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
        else:
            assert actual[key] == expected[key], key
    assert grouped_query.query_points == per_interval_query.query_points
    assert grouped_query.query_calls < per_interval_query.query_calls


def test_confirmed_free_centres_skip_only_impossible_rigid_geometry_pairs():
    model = robot()
    args, _ = scene_args("closed")
    ids = collision_geom_ids(model)
    fast_query = SceneQuery(*args, full_geometry=True)
    full_query = SceneQuery(*args, full_geometry=True)
    fast = ScenePenetrationProbe(model, args[0], ids, .05, query=fast_query)
    full = ScenePenetrationProbe(model, args[0], ids, .05, query=full_query)
    original = full.geometry_backend.query
    full.geometry_backend.query = lambda data, margin, active_geom_ids=None: original(
        data, margin
    )
    poses = np.tile(model.qpos0, (3, 1))
    poses[:, 0] = [0, .02, 2.]
    for actual, expected in zip(fast.probe_trajectory(poses), full.probe_trajectory(poses)):
        assert_same(actual, expected)
