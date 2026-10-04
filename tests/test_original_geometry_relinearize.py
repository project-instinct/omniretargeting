"""Selected origin witnesses retain trial identities, including inactive facets."""

from copy import deepcopy

import mujoco
import numpy as np
import pytest

from omniretargeting._fast.original_geometry_relinearize import pose_identity, witness_identity
from tests.test_original_geometry_api import _reference_jacobian
from tests.test_original_geometry_window import _geometry


@pytest.fixture(autouse=True)
def cpu_queries(monkeypatch):
    monkeypatch.setenv("OMNIRETARGETING_GPU_GEOMETRY", "0")


def test_same_pose_selected_rows_match_trial_values_full_root_jacobian_and_identity():
    geometry, poses = _geometry()
    frames = range(2, 6)
    measured = geometry.measure(poses, frames=frames, jacobians=True)
    expected = deepcopy(measured)
    current = geometry.relinearize_witnesses(poses, poses, measured, frames=frames)
    assert current["request_complete"] and current["reliable"].all()
    assert "complete" not in current and current["coverage"] == "selected_witnesses_only"
    assert current["witness_ids"] == tuple(witness_identity(key, row) for key, row in measured["rows"])
    assert current["pose_id"] == current["origin_pose_id"] == current["trial_pose_id"] == pose_identity(poses, frames)
    assert current["evidence_id"] == geometry.reference_observation_id
    np.testing.assert_allclose(current["phi"], [r["clearance"] for _, r in measured["rows"]], atol=1e-13, rtol=0)
    matrix, _ = geometry.linear_rows(measured, frames=frames)
    np.testing.assert_allclose(current["J"].toarray(), matrix.toarray(), atol=1e-12, rtol=0)
    for (_, row), (_, baseline) in zip(current["rows"], measured["rows"]):
        if "points" in baseline:
            np.testing.assert_allclose(row["points"], baseline["points"], atol=1e-13, rtol=0)
        else:
            np.testing.assert_allclose(row["point"], baseline["point"], atol=1e-13, rtol=0)
    assert measured["rows"][0][1]["point" if measured["rows"][0][0][1] != "self" else "points"].tobytes() == \
        expected["rows"][0][1]["point" if expected["rows"][0][0][1] != "self" else "points"].tobytes()


def test_shifted_origin_uses_fresh_material_points_and_keeps_rows_above_eight_mm():
    geometry, trial = _geometry()
    frames = range(2, 6)
    measured = geometry.measure(trial, frames=frames)
    view = {**measured, "rows": [item for item in measured["rows"] if item[0][1] == "static"]}
    origin = trial.copy()
    # The closed fixture is 1.4 m tall; 12 cm leaves these samples inside it.
    # Move by its full height so this exercises retained far-clear witnesses.
    origin[frames, 2] += 1.4
    current = geometry.relinearize_witnesses(origin, trial, view, frames=frames)
    assert current["request_complete"]
    assert len(current["rows"]) == len(view["rows"])
    assert np.any(current["phi"] > .008)
    assert current["origin_pose_id"] != current["trial_pose_id"]
    data = mujoco.MjData(geometry.model)
    for (key, row), (_, old) in zip(current["rows"], view["rows"]):
        data.qpos[:] = origin[key[0]]
        mujoco.mj_forward(geometry.model, data)
        sample, geom = old["sample_id"], key[-1]
        point = data.geom_xpos[geom] + data.geom_xmat[geom].reshape(3, 3) @ geometry.objective.shell_local[sample]
        exact = geometry.scene.query(point[None])
        assert row["clearance"] == pytest.approx(exact["clearance"][0], abs=1e-13)
        np.testing.assert_allclose(row["point"], point, atol=1e-13, rtol=0)
        np.testing.assert_allclose(row["direction"], exact["directions"][0], atol=1e-13, rtol=0)
        np.testing.assert_allclose(row["jacobian"], _reference_jacobian(geometry.model, data, key, row, geometry.columns),
                                   atol=1e-13, rtol=0)


def test_reverse_frozen_facet_reports_current_maximum_without_replacing_scalar():
    geometry, trial = _geometry()
    frames = range(2, 3)
    measured = geometry.measure(trial, frames=frames)
    reverse = next(item for item in measured["rows"] if item[0][1] == "moving" and item[0][3] == "reverse_surface")
    key, witness = reverse
    origin = trial.copy()
    origin[frames, 0] += .5
    view = {**measured, "rows": [(key, dict(witness))]}
    current = geometry.relinearize_witnesses(origin, trial, view, frames=frames)
    assert current["request_complete"]
    row = current["rows"][0][1]
    assert row["facet_id"] == witness["facet_id"]
    assert row["current_max_plane"] >= row["clearance"]
    assert row["frozen_facet_active"] == (row["facet_id"] == row["current_active_facet"])
    # Request another valid, inactive material facet to expose the distinction.
    shell = next(s for s in geometry.moving.shells if s["geom_id"] == key[-1])
    inactive = (row["current_active_facet"] + 1) % len(shell["planes"])
    view = {**measured, "rows": [(key, {**witness, "facet_id": inactive})]}
    result = geometry.relinearize_witnesses(origin, trial, view, frames=frames)
    assert result["request_complete"]
    row = result["rows"][0][1]
    assert not row["frozen_facet_active"] and row["current_max_plane"] >= row["clearance"]
    assert result["witness_ids"][0][-1] == inactive


def test_selective_rows_view_enforces_budget_without_expanding_a_collision_key(monkeypatch):
    geometry, poses = _geometry()
    frames = range(2, 6)
    measured = geometry.measure(poses, frames=frames)
    chosen = [item for item in measured["rows"] if item[0][1] == "static"][:1]
    key = chosen[0][0]
    assert sum(k == key for k, _ in measured["rows"]) > 1
    visits = []
    original = geometry.scene.query
    monkeypatch.setattr(geometry.scene, "query", lambda points, **kw: visits.append(len(points)) or original(points, **kw))
    result = geometry.relinearize_witnesses(poses, poses, {**measured, "rows": chosen}, frames=frames, keys=[key])
    assert result["request_complete"] and len(result["rows"]) == 1 and visits == [1]
    assert result["query_scope"]["requested_witness_count"] == 1


def test_unknown_keeps_nan_phi_jacobian_alignment_and_explicit_failure(monkeypatch):
    geometry, poses = _geometry()
    frames = range(2, 6)
    measured = geometry.measure(poses, frames=frames)
    view = {**measured, "rows": [item for item in measured["rows"] if item[0][1] == "static"][:2]}
    original = geometry.scene.query
    def unknown(points, **kwargs):
        result = original(points, **kwargs)
        result["uncertain"][:] = True
        return result
    monkeypatch.setattr(geometry.scene, "query", unknown)
    current = geometry.relinearize_witnesses(poses, poses, view, frames=frames)
    assert not current["request_complete"] and not current["reliable"].any()
    assert len(current["rows"]) == 2 and len(current["failures"]) == 2
    assert np.isnan(current["phi"]).all() and np.isnan(current["J"].toarray()).any()
    assert current["witness_ids"] == tuple(witness_identity(key, row) for key, row in view["rows"])


def test_missing_keys_and_wrong_reference_are_explicit():
    geometry, poses = _geometry()
    frames = range(2, 6)
    measured = geometry.measure(poses, frames=frames)
    result = geometry.relinearize_witnesses(poses, poses, measured, frames=frames, keys=[(2, "static", 999)])
    assert not result["request_complete"] and "missing trial witness" in result["failures"][0]
    with pytest.raises(ValueError, match="reference"):
        geometry.relinearize_witnesses(poses, poses, {**measured, "evidence_id": "wrong"}, frames=frames)
