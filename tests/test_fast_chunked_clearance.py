"""Frame-chunked clearance measurements equal the serial ones byte for byte."""
import numpy as np

import omniretargeting._fast.objective as objective_module
from omniretargeting._fast.objective import FastObjective
from omniretargeting.surface_contact import SurfaceContactTasks
from omniretargeting.surface_tasks import SurfaceBinding
from tests.test_fast_backend import _floor, _problem


def _seated():
    model, cfg, coords, poses, task_set, rotations = _problem()
    frames = len(poses)
    grid = _floor(float((poses[20:, 2] - 0.05 - 0.06 + 0.03).mean()))
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
    return objective, poses


def test_chunked_support_depth_and_lift_match_serial(monkeypatch):
    objective, poses = _seated()
    objective.build_support_planes()
    monkeypatch.setattr(objective_module, "TEMPORARY_DEEPEST_FRAMES", 10**9)
    serial = objective.support_depth(poses)
    serial_lift = objective._lift_allowance(serial)
    monkeypatch.setattr(objective_module, "TEMPORARY_DEEPEST_FRAMES", 1)
    monkeypatch.setattr(objective_module, "MIN_CHUNK_FRAMES", 1)
    registry = dict(objective_module._WORKER)
    chunked = objective.support_depth(poses)
    assert dict(objective_module._WORKER) == registry
    assert serial.any()
    assert chunked.tobytes() == serial.tobytes()
    assert objective._lift_allowance(chunked).tobytes() == serial_lift.tobytes()
    objective._clear_planes()
