"""Fast backend: exact rows, banded normal equations, bounds and chunking."""

import mujoco
import numpy as np
import pytest
from scipy import sparse

import omniretargeting._fast.objective as fast_objective
from omniretargeting._fast.objective import FastObjective, _Accumulator, _point_rows
from omniretargeting._fast.pipeline import smooth_correction, verify_budgets
from omniretargeting._fast.sdf import GridSDF
from omniretargeting._fast.solver import banded_system, optimize
from omniretargeting._sequence.proximity import SurfaceProximityTasks
from omniretargeting.contact import SceneContactTask
from omniretargeting.sequence_retargeting import prepare_sequence, sequence_config
from omniretargeting.surface_contact import SurfaceContactTasks
from omniretargeting.surface_tasks import SurfaceBinding

FLOATING_XML = """
<mujoco>
  <compiler angle="radian"/>
  <worldbody>
    <body name="base" pos="0 0 1">
      <freejoint/>
      <geom name="base_geom" type="capsule" fromto="0 0 -.05 0 0 .15" size=".06" mass="4"/>
      <body name="thigh" pos="0 .08 -.08">
        <joint name="hip" type="hinge" axis="0 1 0" range="-1.4 1.4"/>
        <geom type="capsule" fromto="0 0 0 0 0 -.4" size=".045"/>
        <body name="shin" pos="0 0 -.4">
          <joint name="knee" type="hinge" axis="0 1 0" range="-2 .05"/>
          <geom name="shin_geom" type="capsule" fromto="0 0 0 0 0 -.4" size=".04"/>
          <body name="foot" pos="0 0 -.42">
            <joint name="ankle" type="hinge" axis="0 1 0" range="-.7 .7"/>
            <geom name="foot_geom" type="box" size=".09 .04 .02"/>
          </body>
        </body>
      </body>
      <body name="arm" pos="0 -.12 .12">
        <joint name="shoulder" type="hinge" axis="1 0 0" range="-2 2"/>
        <geom type="capsule" fromto="0 0 0 0 0 -.3" size=".03"/>
        <body name="hand" pos="0 0 -.32">
          <geom name="hand_geom" type="sphere" size=".035"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""
NAMES = ["base", "thigh", "shin", "foot", "arm", "hand"]


def _problem(frames=36):
    model = mujoco.MjModel.from_xml_string(FLOATING_XML)
    data = mujoco.MjData(model)
    time = np.arange(frames) / 30.0
    q = np.tile(model.qpos0, (frames, 1))
    q[:, 0] = 0.3 * time
    q[:, 2] = 1.0 + 0.02 * np.sin(4 * time)
    q[:, 3] = np.cos(0.1 * time)
    q[:, 6] = np.sin(0.1 * time)
    q[:, 7] = 0.4 * np.sin(3 * time)
    q[:, 8] = -0.6 + 0.3 * np.sin(3 * time)
    q[:, 9] = 0.1 * np.cos(3 * time)
    q[:, 10] = 0.5 * np.sin(2 * time)
    positions = []
    for pose in q:
        data.qpos[:] = pose
        mujoco.mj_kinematics(model, data)
        positions.append([data.xpos[model.body(n).id].copy() for n in NAMES])
    positions = np.asarray(positions)
    foot = positions[:, NAMES.index("foot")]
    contact = np.zeros(frames, bool)
    contact[4:14] = True
    modes = np.where(contact, "sticking", "swing").astype(object)
    modes[20:26] = "sliding"
    contact[20:26] = True
    up = np.tile([0.0, 0.0, 1.0], (frames, 1))
    forward = np.tile([1.0, 0.0, 0.0], (frames, 1))
    tasks = {
        "foot": SceneContactTask(
            name="foot",
            tracking_positions=foot,
            contact_positions=foot - [0, 0, 0.02],
            surface_points=foot - [0, 0, 0.02],
            surface_normals=up,
            distances=np.zeros(frames),
            relative_speeds=np.zeros(frames),
            contact_confidence=np.ones(frames),
            contact_mask=contact,
            forward_directions=forward,
            up_directions=up,
            contact_modes=modes,
        ),
        "hand": SceneContactTask(
            name="hand",
            tracking_positions=positions[:, NAMES.index("hand")],
            contact_positions=positions[:, NAMES.index("hand")],
            surface_points=positions[:, NAMES.index("hand")],
            surface_normals=up,
            distances=np.ones(frames),
            relative_speeds=np.zeros(frames),
            contact_confidence=np.zeros(frames),
            contact_mask=np.zeros(frames, bool),
        ),
    }
    priority = {
        "effectors": {
            "foot": {
                "robot_body": "foot",
                "tracking_offset": [0, 0, 0],
                "contact_offsets": [[0, 0, -0.02], [0.08, 0, -0.02]],
                "track_orientation": True,
            },
            "hand": {"robot_body": "hand", "tracking_offset": [0, 0, 0]},
        }
    }
    mapping = {n: n for n in NAMES}
    cfg = sequence_config({"self_collision_enabled": False})
    seed = q.copy()
    seed[:, 7:] += 0.05 * np.sin(np.arange(frames))[:, None]
    coords, poses, task_set, rotations = prepare_sequence(
        model, seed, positions, NAMES, mapping, positions[:, 0], tasks, priority, cfg,
    )
    return model, cfg, coords, poses, task_set, rotations


def _surface_tasks(model, frames):
    rng = np.random.default_rng(5)
    bindings = [
        SurfaceBinding("foot", 0, (1, 0, 0), "foot_geom", (0.05, 0, -0.02), (0, 0, -1)),
        SurfaceBinding("foot", 1, (1, 0, 0), "foot_geom", (-0.05, 0, -0.02), (0, 0, -1)),
        SurfaceBinding("shin", 2, (1, 0, 0), "shin_geom", (0.04, 0, -0.2), (1, 0, 0)),
        SurfaceBinding("hand", 3, (1, 0, 0), "hand_geom", (0, 0, -0.035), (0, 0, -1)),
    ]
    count = len(bindings)
    active = rng.random((frames, count)) < 0.7
    normals = rng.normal(size=(frames, count, 3))
    modes = np.where(rng.random((frames, count)) < 0.5, "sticking", "sliding")
    surface = SurfaceContactTasks(
        model, bindings, rng.normal(scale=0.3, size=(frames, count, 3)),
        0.01 * rng.normal(size=(frames, count, 3)), active, normals,
        modes=np.where(active, modes, "none"),
        pose_normal_targets=normals / np.linalg.norm(normals, axis=-1, keepdims=True),
        pose_normal_valid=rng.random((frames, count)) < 0.8,
    )
    geoms = np.array([model.geom(n).id for n in ("foot_geom", "foot_geom", "shin_geom", "hand_geom")])
    proximity = SurfaceProximityTasks(
        model, geoms, ["foot", "foot", "shin", "hand"],
        rng.normal(scale=0.3, size=(frames, len(geoms), 3)),
        0.05 * rng.normal(size=(frames, len(geoms), 3)),
        rng.normal(size=(frames, len(geoms), 3)),
        rng.random((frames, len(geoms))) < 0.8,
        distance_scale=0.1, sigma=0.02, weight=1.0,
    )
    return surface, proximity


def _objective(temporal=False, extra=False):
    model, cfg, coords, poses, task_set, rotations = _problem()
    surface, proximity = _surface_tasks(model, len(poses)) if extra else (None, None)
    objective = FastObjective(
        task_set, cfg, reference=poses, temporal_reference=poses[:, coords.qi] if temporal else None,
        root_rotations=rotations, framerate=30.0,
        surface_tasks=[surface] if extra else (), proximity_tasks=[proximity] if extra else (),
    )
    return objective, poses, (task_set, surface, proximity)


def test_per_frame_rows_equal_the_guarded_sequence_rows():
    objective, poses, (task_set, _, _) = _objective()
    rng = np.random.default_rng(3)
    moved = objective.coords.integrate(poses, 0.05 * rng.normal(size=(len(poses), objective.width)), 1.0)
    for frame in (0, 6, 17, 22, 35):
        residual, jacobian, _ = task_set.frame_rows(moved[frame], frame)
        jacobian = jacobian.toarray()
        chunk = objective.for_frames(slice(frame, frame + 1))
        chunk.kin.evaluate(moved[frame : frame + 1], True)
        acc = _Accumulator(1, objective.width, True)
        stats = {}
        for term in (chunk._mapped_terms, chunk._orientation_terms, chunk._effector_terms):
            term(acc, stats)
        assert acc.cost == pytest.approx(float(residual @ residual), rel=1e-10)
        np.testing.assert_allclose(acc.hessian[0], jacobian.T @ jacobian, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(acc.gradient[0], jacobian.T @ residual, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("term", ["_surface_terms", "_proximity_terms"])
def test_surface_and_proximity_rows_equal_the_guarded_task_rows(term):
    objective, poses, (_, surface, proximity) = _objective(extra=True)
    task = surface if term == "_surface_terms" else proximity
    cols = objective.coords.cols
    for frame in (0, 9, 23):
        chunk = objective.for_frames(slice(frame, frame + 1))
        chunk.kin.evaluate(poses[frame : frame + 1], True)
        acc = _Accumulator(1, objective.width, True)
        getattr(chunk, term)(acc, {})
        cost, hessian, gradient = 0.0, np.zeros((len(cols),) * 2), np.zeros(len(cols))
        for weight, residual, jacobian in task.rows(poses[frame], frame, cols):
            cost += weight * float(residual @ residual)
            hessian += weight * jacobian.T @ jacobian
            gradient += weight * jacobian.T @ residual
        assert acc.cost == pytest.approx(cost, rel=1e-10)
        np.testing.assert_allclose(acc.hessian[0], hessian, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(acc.gradient[0], gradient, rtol=1e-9, atol=1e-9)


def test_parallel_chunks_equal_the_serial_evaluation(monkeypatch):
    objective, poses, _ = _objective(temporal=True, extra=True)
    monkeypatch.setattr(fast_objective, "WORKERS", 1)
    serial = objective.evaluate(poses)
    monkeypatch.setattr(fast_objective, "WORKERS", 3)
    monkeypatch.setattr(fast_objective, "MIN_CHUNK_FRAMES", 8)
    objective.start_workers()
    try:
        assert objective.pool is not None
        parallel = objective.evaluate(poses)
    finally:
        objective.stop_workers()
    assert parallel[0].cost == pytest.approx(serial[0].cost, rel=1e-12)
    np.testing.assert_array_equal(parallel[0].hessian, serial[0].hessian)
    np.testing.assert_array_equal(parallel[0].gradient, serial[0].gradient)
    np.testing.assert_array_equal(parallel[1], serial[1])


def test_body_rows_equal_point_jacobian_accumulation():
    objective, poses, _ = _objective()
    kin = objective.kin.evaluate(poses, True)
    rng = np.random.default_rng(0)
    bodies = np.array([1, 3, 3, 4, 6])
    points = kin.xpos[:, bodies] + 0.1 * rng.normal(size=(len(poses), len(bodies), 3))
    directions = rng.normal(size=points.shape)
    residual = rng.normal(size=points.shape[:2])
    fast = _Accumulator(len(poses), objective.width, True)
    fast.add_body_rows(residual, _point_rows(directions, points - kin.xpos[:, bodies]), bodies, kin)
    point_jac = kin.point_jacobians(bodies, points)
    plain = _Accumulator(len(poses), objective.width, True)
    plain.add(residual, np.einsum("fki,fkiw->fkw", directions, point_jac))
    np.testing.assert_allclose(fast.hessian, plain.hessian, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(fast.gradient, plain.gradient, rtol=1e-10, atol=1e-10)


def test_banded_storage_holds_block_and_temporal_normal_equations():
    rng = np.random.default_rng(1)
    frames, width = 7, 3
    blocks = rng.normal(size=(frames, width, width))
    blocks = np.matmul(blocks, blocks.transpose(0, 2, 1))
    temporal = sparse.random(10, frames * width, density=0.3, random_state=2, format="csr")
    temporal = temporal.multiply(abs(np.subtract.outer(np.arange(10), np.arange(frames * width) // width)) < 3)
    ab = banded_system(frames, width, blocks, sparse.csr_matrix(temporal))
    dense = sparse.block_diag(list(blocks)).toarray() + (temporal.T @ temporal).toarray()
    rebuilt = np.zeros_like(dense)
    for offset in range(ab.shape[0]):
        index = np.arange(frames * width - offset)
        rebuilt[index + offset, index] = ab[offset, : len(index)]
    np.testing.assert_allclose(np.tril(rebuilt), np.tril(dense), atol=1e-12)


def test_optimizer_keeps_joint_limits_exact_without_posture_box():
    objective, poses, _ = _objective(temporal=True)
    # Only mechanical limits bound the joints; posture has no angular box.
    np.testing.assert_array_equal(objective.lower, np.broadcast_to(objective.coords.lower, objective.lower.shape))
    np.testing.assert_array_equal(objective.upper, np.broadcast_to(objective.coords.upper, objective.upper.shape))
    objective.reference = poses.copy()
    target = poses.copy()
    target[:, objective.coords.qi] += 1.0
    objective.reference[:, objective.coords.qi] = target[:, objective.coords.qi]
    solved, history = optimize(objective, poses, iterations=15)
    assert history
    assert np.all(solved[:, objective.coords.qi] >= objective.coords.lower - 1e-12)
    assert np.all(solved[:, objective.coords.qi] <= objective.coords.upper + 1e-12)


def test_root_projection_bounds_translation_only():
    objective, poses, _ = _objective(temporal=True)
    q = objective.coords.root_q
    moved = poses.copy()
    moved[:, q : q + 3] += [0.2, 0.0, 0.0]
    moved[:, q + 3 : q + 7] = [np.cos(0.3), 0.0, 0.0, np.sin(0.3)]
    projected = objective.project(moved)
    shift, _ = objective._root_changes(projected)
    assert np.linalg.norm(shift, axis=1).max() == pytest.approx(objective.root_budget)
    np.testing.assert_array_equal(projected[:, q + 3 : q + 7], moved[:, q + 3 : q + 7])
    assert all(offset == 0 for offset, _ in objective.root_boundaries(projected))
    untouched = objective.project(poses)
    np.testing.assert_array_equal(untouched, poses)


def test_grid_sdf_interpolates_and_marks_unknown(tmp_path, monkeypatch):
    monkeypatch.setattr("omniretargeting._fast.sdf.CACHE_DIR", tmp_path)
    calls = []

    def make_query():
        calls.append(1)

        def query(points):
            distance = np.linalg.norm(points, axis=1)
            return {
                "clearance": distance - 0.5,
                "directions": points / np.maximum(distance, 1e-9)[:, None],
                "uncertain": points[:, 0] > 0.9,
            }

        return query

    grid = GridSDF.build(make_query, [-1, -1, -1], [1, 1, 1], 0.05, key=("sphere",))
    again = GridSDF.build(make_query, [-1, -1, -1], [1, 1, 1], 0.05, key=("sphere",))
    assert len(calls) == 1 and np.array_equal(grid.clearance, again.clearance, equal_nan=True)
    points = np.random.default_rng(4).uniform(-0.7, 0.7, size=(200, 3))
    value, gradient = grid.query(points)
    np.testing.assert_allclose(value, np.linalg.norm(points, axis=1) - 0.5, atol=0.02)
    assert np.all(np.einsum("ni,ni->n", gradient, points / np.linalg.norm(points, axis=1)[:, None]) > 0.95)
    assert np.isnan(grid.query(np.array([[1.5, 0, 0], [0.97, 0, 0]]), gradient=False)).all()


def test_budget_verification_flags_chatter_and_only_reports_joint_change():
    model, _, coords, poses, _, _ = _problem()
    motion = poses.copy()
    motion[18, coords.qi[0]] += 0.05
    motion[30, coords.qi[1]] += 0.65
    result, chatter, posture = verify_budgets(model, coords, poses, motion)
    assert chatter[18] and not chatter[:10].any()
    # Joint change has no angular bound; it is reported, not flagged.
    assert not posture.any() and result["layout_flip_frames"] == 0
    assert result["max_joint_change_rad"] == pytest.approx(0.65)


def test_held_contacts_only_resist_regression_beyond_tolerance():
    objective, poses, _ = _objective(extra=True)
    surf = objective.surfaces[0]
    surf["priority"] = surf["active"].astype(float)
    objective.hold_contacts(poses)
    assert objective.clearance_scale == 1.0
    held = surf["hold"][surf["active"]]
    assert np.all(np.isfinite(held)) and np.all(np.isinf(surf["hold"][~surf["active"]]))

    def cost(q, hold):
        saved = surf["hold"]
        surf["hold"] = hold
        objective.kin.evaluate(q, True)
        acc = _Accumulator(len(q), objective.width, True)
        objective._surface_terms(acc, {})
        surf["hold"] = saved
        return acc.cost

    unbounded = np.full_like(surf["hold"], np.inf)
    assert cost(poses, surf["hold"]) == pytest.approx(cost(poses, unbounded))
    moved = poses.copy()
    moved[:, :3] += [0.0, 0.0, 0.05]
    assert cost(moved, surf["hold"]) > cost(moved, unbounded) + 1.0


def _floor(height):
    """Grid of a solid below ``height``: clearance = z - height, exit +z."""
    lower, spacing = np.array([-1.0, -1.0, height - 0.3]), 0.02
    axes = [lower[i] + spacing * np.arange(n) for i, n in enumerate((101, 101, 31))]
    z = np.meshgrid(*axes, indexing="ij")[2]
    directions = np.zeros((*z.shape, 3))
    directions[..., 2] = 1.0
    return GridSDF(lower, spacing, z - height, directions)


def test_reference_lift_raises_a_floating_base_sunk_into_its_support():
    model, cfg, coords, poses, task_set, rotations = _problem()
    frames = len(poses)
    base_z = poses[:, 2] - 0.05  # bottom of the base capsule
    seat = 0.03  # the base bottom is ~3 cm inside the support it sits on from frame 20
    grid = _floor(float((base_z[20:] - 0.06 + seat).mean()))
    binding = [SurfaceBinding("base", 0, (1, 0, 0), "base_geom", (0, 0, -0.16), (0, 0, -1))]
    active = (np.arange(frames) >= 20)[:, None]
    targets = np.zeros((frames, 1, 3))
    targets[:, 0, :2] = poses[:, :2]
    targets[:, 0, 2] = grid.lower[2] + 0.3
    surface = SurfaceContactTasks(
        model, binding, targets, np.zeros((frames, 1, 3)), active, np.tile([0.0, 0.0, 1.0], (frames, 1, 1)),
        modes=np.where(active, "sticking", "none"),
    )
    objective = FastObjective(
        task_set, cfg, reference=poses, temporal_reference=poses[:, coords.qi],
        root_rotations=rotations, framerate=30.0, scene_sdf=grid, surface_tasks=[surface],
    )
    objective.surfaces[0]["priority"] = active.astype(float)
    lift = objective.reference_lift()
    expected = targets[0, 0, 2] - (poses[:, 2] - 0.11)
    assert lift[:5].max() == pytest.approx(0.0, abs=1e-9)  # the ramp starts at most 0.5 s early
    assert lift[27:].min() >= expected[27:].min() - 0.005
    # The minimal amount reaching the plane through the targets, not a fixed cap.
    assert lift.max() <= expected.max() + 2 * cfg["scene_clearance_m"] + 1e-9
    assert np.abs(np.diff(lift)).max() < 0.5 * lift.max()  # ramps, never a one-frame jump
    reference = objective.reference.copy()
    objective.apply_reference_lift(lift)
    np.testing.assert_allclose(objective.reference[:, 2], reference[:, 2] + lift)
    np.testing.assert_array_equal(objective.reference[:, coords.qi], reference[:, coords.qi])


def test_reference_lift_counts_links_sharing_the_base_contact_region():
    model, cfg, coords, poses, task_set, rotations = _problem()
    frames = len(poses)
    thigh = int(model.geom_bodyid.tolist().index(model.body("thigh").id))
    bindings = [
        SurfaceBinding("seat", 0, (1, 0, 0), "base_geom", (0, 0, -0.11), (0, 0, -1)),
        SurfaceBinding("seat", 1, (1, 0, 0), thigh, (0, 0, -0.05), (0, 0, -1)),
        SurfaceBinding("palm", 2, (1, 0, 0), "hand_geom", (0, 0, -0.035), (0, 0, -1)),
    ]
    active = np.ones((frames, len(bindings)), bool)
    surface = SurfaceContactTasks(
        model, bindings, np.zeros((frames, len(bindings), 3)), np.zeros((frames, len(bindings), 3)), active,
        np.tile([0.0, 0.0, 1.0], (frames, len(bindings), 1)), modes=np.full(active.shape, "sticking"),
    )
    objective = FastObjective(
        task_set, cfg, reference=poses, temporal_reference=poses[:, coords.qi],
        root_rotations=rotations, framerate=30.0, surface_tasks=[surface],
    )
    base = model.geom("base_geom").id
    # The thigh carries the base's region and rests on the same seat; the
    # hand's own region can leave its support by a joint.
    assert objective._base_region_geoms([base]) == sorted([base, thigh])
    assert objective._base_region_geoms([]) == []


def test_smooth_corrections_pass_and_added_jitter_is_rejected():
    model, _, coords, poses, _, _ = _problem()
    entry = poses.copy()
    ramp = entry.copy()
    ramp[:, coords.qi[0]] += 0.02 * (1 - np.cos(np.linspace(0, np.pi, len(entry)))) / 2
    jitter = entry.copy()
    jitter[:, coords.qi[0]] += 0.002 * (-1.0) ** np.arange(len(entry))
    base = verify_budgets(model, coords, poses, entry)[0]
    accepted, _ = smooth_correction(coords, entry, ramp, base, verify_budgets(model, coords, poses, ramp)[0])
    rejected, reasons = smooth_correction(coords, entry, jitter, base, verify_budgets(model, coords, poses, jitter)[0])
    assert accepted and not rejected
    assert any("high-frequency" in reason for reason in reasons)


def test_grid_value_gradient_is_the_derivative_of_the_interpolated_clearance():
    rng = np.random.default_rng(8)
    clearance = rng.normal(size=(6, 5, 7)).astype(np.float32)
    grid = GridSDF([0.0, 0.0, 0.0], 0.1, clearance, np.zeros((6, 5, 7, 3)))
    # Points inside cells, with steps that stay in the cell: trilinear is
    # linear along each axis there, so a central difference is exact.
    cells = rng.integers(0, 4, (50, 3))
    points = (cells + rng.uniform(0.3, 0.7, (50, 3))) * 0.1
    gradient = grid.value_gradient(points)
    step = 0.02
    for axis in range(3):
        shift = np.zeros(3)
        shift[axis] = step
        numeric = (grid.query(points + shift, gradient=False) - grid.query(points - shift, gradient=False)) / (2 * step)
        np.testing.assert_allclose(gradient[:, axis], numeric, rtol=1e-5, atol=1e-5)
