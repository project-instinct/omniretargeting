"""World-tracked effectors follow their source path; the root path drops only rotation chatter."""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

import omniretargeting._fast.objective as fast_objective
from omniretargeting._fast.objective import FastObjective, _without_chatter
from omniretargeting.motion_budgets import CHATTER_SCALE_FRAMES
from tests.test_fast_backend import _problem


def _world_foot_objective():
    model, cfg, coords, poses, task_set, rotations = _problem()
    task_set.effectors["foot"]["preserve_world_target"] = True
    objective = FastObjective(
        task_set, cfg, reference=poses, temporal_reference=None,
        root_rotations=rotations, framerate=30.0,
    )
    return objective, poses


def _path_rows(objective, poses, with_jacobian=True):
    objective.kin.evaluate(poses, with_jacobian)
    points, jacobians = objective._effector_points(with_jacobian)
    rows, blocks = objective._world_path_rows(points, jacobians, with_jacobian)
    return np.concatenate(rows), blocks


def test_path_rows_cover_only_free_world_stencils():
    objective, _ = _world_foot_objective()
    ((index, _, masks),) = objective.world_paths
    effector = objective.effectors[index]
    free = effector["world"] & ~(effector["active"] | effector["managed"])
    assert free.any() and not free.all()
    for order, mask in enumerate(masks, 1):
        for start, valid in enumerate(mask):
            assert valid == free[start:start + order + 1].all()
    # The hand is body-relative: it keeps its per-frame target only.
    assert [objective.effectors[i]["name"] for i, _, _ in objective.world_paths] == ["foot"]


def test_path_rows_follow_the_target_path_without_its_chatter():
    objective, _ = _world_foot_objective()
    index, path, _ = objective.world_paths[0]
    effector = objective.effectors[index]
    target = effector["tracking"] + effector["share"][:, None] * (effector["approach"] - effector["tracking"])
    np.testing.assert_allclose(path, gaussian_filter1d(target, CHATTER_SCALE_FRAMES, axis=0, mode="nearest"))


def test_path_rows_match_finite_differences():
    objective, poses = _world_foot_objective()
    rng = np.random.default_rng(3)
    poses = objective.coords.integrate(poses, 0.05 * rng.normal(size=(len(poses), objective.width)), 1.0)
    residual, blocks = _path_rows(objective, poses)
    jacobian = sum(block.shape[0] for block in blocks)
    assert jacobian == len(residual)
    matrix = np.vstack([block.toarray() for block in blocks])
    direction = rng.normal(size=(len(poses), objective.width))
    step = 1e-6
    ahead, _ = _path_rows(objective, objective.coords.integrate(poses, direction, step), False)
    behind, _ = _path_rows(objective, objective.coords.integrate(poses, direction, -step), False)
    expected = matrix @ direction.ravel()
    np.testing.assert_allclose((ahead - behind) / (2 * step), expected, rtol=1e-5, atol=1e-5 * np.abs(expected).max())


def test_path_rows_ignore_a_constant_offset_but_penalize_chatter():
    objective, poses = _world_foot_objective()
    index, target, _ = objective.world_paths[0]
    frames = len(poses)
    points = np.zeros((frames, len(objective.effectors), 3))
    points[:, index] = target + [0.03, -0.01, 0.02]
    rows, _ = objective._world_path_rows(points, None, False)
    np.testing.assert_allclose(np.concatenate(rows), 0.0, atol=1e-9)
    points[:, index, 0] += 0.005 * (-1.0) ** np.arange(frames)
    rows, _ = objective._world_path_rows(points, None, False)
    assert np.abs(np.concatenate(rows)).max() > 1.0


def test_parallel_evaluation_keeps_the_path_rows(monkeypatch):
    objective, poses = _world_foot_objective()
    rng = np.random.default_rng(1)
    moved = objective.coords.integrate(poses, 0.1 * rng.normal(size=(len(poses), objective.width)), 1.0)
    monkeypatch.setattr(fast_objective, "WORKERS", 1)
    serial = objective.evaluate(moved)
    monkeypatch.setattr(fast_objective, "WORKERS", 3)
    monkeypatch.setattr(fast_objective, "MIN_CHUNK_FRAMES", 5)
    objective.start_workers()
    try:
        assert objective.pool is not None
        parallel = objective.evaluate(moved)
    finally:
        objective.stop_workers()
    assert parallel[0].cost == serial[0].cost
    np.testing.assert_array_equal(parallel[1], serial[1])
    np.testing.assert_array_equal(parallel[2].toarray(), serial[2].toarray())


def test_root_path_drops_rotation_chatter_and_keeps_deliberate_turns():
    frames = 90
    time = np.arange(frames) / 30.0
    # A deliberate turn of 3 rad in three seconds, plus 1 degree of chatter.
    yaw = 1.5 * (1 - np.cos(np.pi * time / 3))
    turn = Rotation.from_rotvec(np.c_[np.zeros((frames, 2)), yaw])
    chatter = Rotation.from_rotvec(np.c_[np.radians(1.0) * (-1.0) ** np.arange(frames), np.zeros((frames, 2))])
    noisy = turn * chatter
    quat = noisy.as_quat()
    quat[::3] *= -1  # quaternion signs carry no rotation information
    path = _without_chatter(Rotation.from_quat(quat))
    # Only the Gaussian's curvature bias of the turn remains (largest at the clip ends).
    assert np.degrees((path.inv() * turn).magnitude()).max() < 0.6
    step = (path[1:] * path[:-1].inv()).as_rotvec()
    noisy_step = (noisy[1:] * noisy[:-1].inv()).as_rotvec()
    assert np.linalg.norm(np.diff(step, axis=0), axis=1).max() < 0.05 * np.linalg.norm(np.diff(noisy_step, axis=0), axis=1).max()
