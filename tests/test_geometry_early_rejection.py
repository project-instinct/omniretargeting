"""A known regression can reject early; successful checks need every frame."""

import numpy as np
import pytest
from omniretargeting.scene_geometry import SceneQuery
from tests.test_source_contacts import floor
from omniretargeting._fast.mesh_repair import OriginalGeometry
from tests.test_fast_mesh_repair import _open_object
from tests.test_fast_backend import _objective


@pytest.mark.parametrize("scene", [False, True])
@pytest.mark.parametrize("reject_after", [0, 6])
def test_early_rejection_matches_full_query_and_acceptance_keeps_full_timeline(
    scene, reject_after
):
    objective, poses, _ = _objective(temporal=True)
    query = OriginalGeometry(
        objective,
        [_open_object(objective, poses)],
        SceneQuery(floor()) if scene else None,
    )
    full = query.measure(poses)
    first_key = next(
        k for k, v in full["depths"].items() if v > 1e-5 and k[0] >= reject_after
    )
    budgets = dict(full["depths"])
    budgets[first_key] = 0.0
    count = query.moving.query_count
    prefix = query.measure(poses, protected_depths=budgets)
    assert query.moving.query_count - count == first_key[0] + 1
    assert not prefix["complete"]
    assert prefix["rejection_frame"] == first_key[0]
    assert prefix["depths"][first_key] == full["depths"][first_key]
    assert max(k[0] for k in prefix["depths"]) == first_key[0]
    accepted = query.measure(poses, protected_depths=full["depths"])
    assert accepted["complete"]
    assert accepted["depths"] == full["depths"]
    assert len(accepted["rows"]) == len(full["rows"])
    for (_, a), (_, b) in zip(accepted["rows"], full["rows"]):
        key = "points" if "points" in a else "point"
        np.testing.assert_array_equal(a[key], b[key])
        np.testing.assert_array_equal(a["direction"], b["direction"])


def test_incomplete_query_cannot_be_accepted_as_improved_clearance(monkeypatch):
    import omniretargeting._fast.mesh_repair as repair
    from omniretargeting._fast import repair_prediction
    from omniretargeting._fast.repair_limits import RepairLimits

    # Reach candidate geometry so this still exercises incomplete-query rejection.
    monkeypatch.setattr(repair_prediction, "PREDICTIVE_GATING", False)
    objective, poses, _ = _objective(temporal=True)
    obj = _open_object(objective, poses)
    original = OriginalGeometry.measure
    calls = []

    def measure(self, candidate, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            return original(self, candidate, **kwargs)
        return {
            "rows": [],
            "depths": {},
            "failures": [],
            "complete": False,
            "rejection_frame": 0,
        }

    monkeypatch.setattr(OriginalGeometry, "measure", measure)
    monkeypatch.setattr(RepairLimits, "check", lambda *args: (True, {}))
    monkeypatch.setattr(
        repair,
        "_qp_step",
        lambda *args: (np.full((len(poses), len(objective.coords.qi)), 1e-5), "Solved"),
    )
    result, report = repair.repair_original_meshes(objective, poses, [obj])
    np.testing.assert_array_equal(result, poses)
    assert report["status"] == "retained"
    assert any(r.get("rejected_geometry_prefix_frame") == 0 for r in report["history"])
