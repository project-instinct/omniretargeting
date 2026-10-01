"""Sub-mm repair eligibility and immutable geometry reuse preserve their contracts."""
from dataclasses import replace
from types import SimpleNamespace
import mujoco
import numpy as np
from omniretargeting._fast import mesh_repair
from omniretargeting._fast.repair_schedule import RepairSchedule
from omniretargeting.moving_scene import MovingSceneQuery
from tests.test_geometry_exact_reuse import _same
from tests.test_open_object_hull import V_SHEET, EVIDENCE
from tests.test_repair_penetration_yield import _problem


def test_submillimetre_group_is_eligible_without_expanding_attempt_budget(monkeypatch):
    objective, poses = _problem()
    geom = objective.model.geom('wet_geom').id
    measurement = {'depths': {(0,'static',geom): .0008}}
    monkeypatch.setattr(mesh_repair, 'OPTIONS', replace(mesh_repair.OPTIONS, submillimeter_repair=False))
    assert not mesh_repair._repair_groups(objective, measurement)
    monkeypatch.setattr(mesh_repair, 'OPTIONS', replace(mesh_repair.OPTIONS, submillimeter_repair=True))
    groups = mesh_repair._repair_groups(objective, measurement)
    assert groups
    assert not mesh_repair._repair_groups(objective, measurement, activation=.001)
    schedule = RepairSchedule(range(4), 12, total_limit=12)
    attempts = []
    while (item := schedule.next()) is not None:
        attempts.append(item);schedule.retry(item[0])
    assert len(attempts) == 12


def test_moving_geometry_clone_has_fresh_pose_caches_and_identical_query_bytes():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body><freejoint/>
       <geom type="sphere" size=".05"/></body></worldbody></mujoco>''')
    spec = dict(mesh=V_SHEET, collision_evidence=EVIDENCE)
    original = MovingSceneQuery(model,[spec],2)
    pose = model.qpos0.copy();pose[:3]=[0,0,.02]
    expected = original.query(pose,0)
    clone = original.fresh_state()
    assert clone.objects is original.objects and clone.shells is original.shells
    assert clone.data is not original.data
    assert clone.query_count == 0 and not clone.material_cache and not clone.sample_keys
    _same(expected, clone.query(pose,0))
    changed = pose.copy();changed[0]+=.01
    clone.query(changed,1)
    assert original.query_count == 1
    _same(expected, original.query(pose,0))
    clone.cfg['activation_distance'] = .004
    assert original.cfg['activation_distance'] == .008


def test_original_geometry_reuses_only_matching_sequence_and_keeps_evidence_registry(monkeypatch):
    objective, poses = _problem()
    specs = [dict(mesh=V_SHEET,collision_evidence=EVIDENCE)]
    provider = SimpleNamespace(geometry=mesh_repair.OriginalGeometry(objective,specs,None))
    objective.frozen_original_depth = provider
    provider.geometry._moving_evidence_baseline[0] = frozenset({(0,0,0)})
    monkeypatch.setattr(mesh_repair,'OPTIONS',replace(mesh_repair.OPTIONS,reuse_original_geometry=True))
    repair = mesh_repair.OriginalGeometry(objective,specs,None,manifold=True)
    assert repair.moving.objects is provider.geometry.moving.objects
    assert repair._moving_evidence_baseline == {}
    different = mesh_repair.OriginalGeometry(objective,[dict(specs[0])],None)
    assert different.moving.objects is not provider.geometry.moving.objects
