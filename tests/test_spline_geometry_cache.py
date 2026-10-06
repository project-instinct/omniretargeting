"""Selected-witness batching preserves the original per-witness contract."""

from contextlib import ExitStack

import numpy as np
import pytest
from scipy import sparse

from omniretargeting._fast.original_geometry_relinearize import witness_identity
from omniretargeting._fast.spline_geometry import SplineGeometry
from tests.test_original_geometry_batches import _scene
from tests.test_spline_acceptance import _small_geometry


@pytest.fixture(autouse=True)
def cpu_geometry(monkeypatch):
    monkeypatch.setenv("OMNIRETARGETING_GPU_GEOMETRY", "0")


def _identity(key, row):
    return repr(witness_identity(key, row))


def _learn_two_old_origins(geometry, origin, candidate):
    """Install real static and moving witnesses from two accepted origins."""
    measured = geometry.measure(candidate, jacobians=True)
    static = {key for key in measured["depths"] if key[1] == "static"}
    moving = {key for key in measured["depths"] if key[1] == "moving"}
    assert static and moving

    second_origin = origin.copy()
    second_origin[:, 0] = 1.05
    # Two disjoint material families deliberately have two old origins.  They
    # must form two batched calls, not one call per remembered row.
    added_static = geometry.learn(origin, candidate, measured, static)
    added_moving = geometry.learn(second_origin, candidate, measured, moving)
    assert added_static > 1 and added_moving > 1
    assert {value[1][1] for value in geometry.witness_memory.values()} == {
        "static", "moving"}
    return list(geometry.witness_memory.items())


def test_remembered_batches_match_per_witness_oracle_and_reuse_same_origin(monkeypatch):
    objective, origin, objects = _small_geometry(root_x=1.)
    candidate = origin.copy()
    candidate[:, 0] = .35
    target = origin.copy()
    target[:, 0] = 1.10

    with ExitStack() as resources:
        geometry = SplineGeometry(objective, objects, _scene("closed"), resources)
        saved_memory = _learn_two_old_origins(geometry, origin, candidate)
        assert len(saved_memory) > 2

        calls = []
        original = geometry.provider.relinearize_witnesses

        def counted(new_poses, old_origin, measurement, *, frames, keys=None):
            calls.append((len(measurement["rows"]), id(old_origin), len(keys)))
            return original(new_poses, old_origin, measurement,
                            frames=frames, keys=keys)

        monkeypatch.setattr(geometry.provider, "relinearize_witnesses", counted)
        current = geometry.measure(target, jacobians=True)

        # Two old origins, not one selected query per material row.
        assert len(calls) == 2
        assert sorted(call[0] for call in calls) == sorted(
            [sum(value[1][1][1] == "static" for value in saved_memory),
             sum(value[1][1][1] == "moving" for value in saved_memory)])
        assert sum(call[0] for call in calls) == len(saved_memory)
        assert geometry.witness_batches == 4  # two learn calls + two target batches
        assert geometry.witness_queries == 2 * len(saved_memory)

        identities = [_identity(key, row) for key, row in current["rows"]]
        assert len(identities) == len(set(identities))
        assert {key[1] for key, _ in current["rows"]} == {
            "static", "moving", "self"}
        assert not current["failures"] and current["complete"]

        assembled_jacobian, assembled_phi = geometry.provider.linear_rows(current)
        for identity, (old_origin, key, stored_row) in saved_memory:
            oracle = original(
                target, old_origin, {"rows": [(key, stored_row)], "failures": []},
                frames=range(len(target)), keys=[key])
            assert oracle["request_complete"]
            assert len(oracle["rows"]) == len(oracle["witness_ids"]) == 1
            assert not oracle["failures"]
            oracle_identity = repr(oracle["witness_ids"][0])
            matches = [index for index, (row_key, row) in enumerate(current["rows"])
                       if _identity(row_key, row) == oracle_identity]
            assert len(matches) == 1
            index = matches[0]
            _, oracle_row = oracle["rows"][0]
            _, current_row = current["rows"][index]

            for field in ("clearance", "direction", "point", "points",
                          "surface_point", "local_point", "material_local_point"):
                if field in oracle_row or field in current_row:
                    np.testing.assert_allclose(
                        oracle_row[field], current_row[field],
                        rtol=1e-13, atol=1e-13, equal_nan=True)
            np.testing.assert_allclose(
                oracle_row["jacobian"], current_row["jacobian"],
                rtol=1e-13, atol=1e-13)
            np.testing.assert_allclose(
                np.asarray(assembled_jacobian[index].toarray()),
                np.asarray(oracle["J"].toarray()), rtol=1e-13, atol=1e-13)
            np.testing.assert_allclose(assembled_phi[index], oracle["phi"][0],
                                       rtol=1e-13, atol=1e-13)

        first_blocks = geometry.blocks(target)
        assert first_blocks[0].A.shape[0] == len(current["rows"])
        assert geometry.linearized["rows"] is geometry.current["rows"]

        # The new current measurement already contains remembered rows.  A
        # repeated assembly at the same origin must reuse them without another
        # selected-witness query and must not append them a second time.
        call_count = len(calls)
        second_blocks = geometry.blocks(target)
        assert len(calls) == call_count
        np.testing.assert_array_equal(first_blocks[0].A.toarray(),
                                      second_blocks[0].A.toarray())
        repeated_identities = [_identity(key, row)
                               for key, row in geometry.current["rows"]]
        assert len(repeated_identities) == len(set(repeated_identities))
        assert repeated_identities == identities


def test_remembered_rows_participate_in_actual_nonlinear_trial(monkeypatch):
    objective, origin, objects = _small_geometry(root_x=1.)
    candidate = origin.copy()
    candidate[:, 0] = .35
    target = origin.copy()
    target[:, 0] = 1.10

    with ExitStack() as resources:
        geometry = SplineGeometry(objective, objects, _scene("closed"), resources)
        _learn_two_old_origins(geometry, origin, candidate)
        geometry.measure(target, jacobians=True)

        # This is a real provider measurement at the far target pose.  It is
        # used only to isolate the nonlinear trial from ordinary current rows;
        # selected material witnesses are still requeried by the real API.
        far_base = geometry.provider.measure(target, jacobians=False)
        assert {key[1] for key, _ in far_base["rows"]} == {"self"}

        def fixed_base(poses, *, jacobians=False):
            return far_base

        monkeypatch.setattr(geometry.provider, "measure", fixed_base)
        accepted, measured, bad = geometry.accepts(candidate)
        assert not accepted
        assert {key[1] for key in bad} == {"static", "moving"}
        assert not measured["failures"] and measured["complete"]
        assert {key[1] for key, _ in measured["rows"]} == {
            "static", "moving", "self"}
        assert geometry.unknown_trials == 0


def test_incomplete_batch_reports_unknown_and_does_not_invent_zero_depth(monkeypatch):
    objective, origin, objects = _small_geometry(root_x=1.)
    candidate = origin.copy()
    candidate[:, 0] = .35
    target = origin.copy()
    target[:, 0] = 1.10

    with ExitStack() as resources:
        geometry = SplineGeometry(objective, objects, _scene("closed"), resources)
        saved_memory = _learn_two_old_origins(geometry, origin, candidate)
        remember_count = len(saved_memory)

        def incomplete(new_poses, old_origin, measurement, *, frames, keys=None):
            # Field set mirrors original_geometry_relinearize.relinearize_witnesses.
            # An empty, explicitly unreliable result must not be interpreted as
            # zero-depth by callers.
            width = len(new_poses) * objective.coords.width
            return {
                "rows": [],
                "J": sparse.csr_matrix((0, width)),
                "phi": np.empty(0),
                "clearance": np.empty(0),
                "reliable": np.zeros(0, bool),
                "witness_ids": (),
                "pose_id": "not-inspected-by-caller",
                "origin_pose_id": "old-origin",
                "trial_pose_id": "new-origin",
                "evidence_id": geometry.provider.geometry.reference_observation_id,
                "columns": np.array(geometry.provider.geometry.columns),
                "frames": tuple(frames),
                "failures": ["unknown selected remembered witness"],
                "request_complete": False,
                "query_scope": {"requested_witness_count": len(measurement["rows"]),
                                "returned_witness_count": 0},
                "queried_scope": {"requested_witness_count": len(measurement["rows"]),
                                  "returned_witness_count": 0},
                "coverage": "selected_witnesses_only",
            }

        monkeypatch.setattr(geometry.provider, "relinearize_witnesses", incomplete)
        measured = geometry.measure(target, jacobians=True)
        assert measured["failures"]
        assert not measured["complete"]
        for _, (_, key, _) in saved_memory:
            assert key not in measured["depths"]
        assert remember_count == len(saved_memory)
        assert geometry.witness_failures

        accepted, unknown_measured, bad = geometry.accepts(target)
        assert accepted and not bad
        assert unknown_measured["failures"]
        assert geometry.unknown_trials == 1
        report = geometry.report()
        assert report["remembered_witnesses"] == remember_count
        assert report["selected_witness_batches"] == 6  # two learn + two per measure
        assert report["selected_witness_queries"] == remember_count
        assert report["selected_witness_failures"]
