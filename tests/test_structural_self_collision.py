"""Always-overlapping model geometry is not a self-collision of the motion."""

import mujoco
import numpy as np
from pathlib import Path

from omniretargeting.self_collision import SelfCollisionQuery, STRUCTURAL_REPLACEMENTS
from omniretargeting.geometry_query import GeometryQueryError

XML = """<mujoco><worldbody>
  <body name="base"><freejoint/><geom size=".05"/>
    <body name="upper" pos="0 0 -.2"><joint axis="0 1 0" range="-1 1"/>
      <geom type="capsule" fromto="0 0 0 0 0 -.2" size=".05"/>
      <body name="lower" pos="0 0 -.2"><joint axis="0 1 0" range="-1 1"/>
        <geom type="capsule" fromto="0 0 -.06 0 0 -.25" size=".04"/>
        <!-- ``shell`` sits on the knee axis inside ``upper``: any angle overlaps. -->
        <body name="shell"><joint axis="0 0 1" range="-.2 .2"/><geom size=".03"/></body>
      </body>
    </body>
    <body name="arm" pos=".3 0 0"><joint axis="0 1 0" range="-3 3"/>
      <geom type="capsule" fromto="0 0 0 0 0 -.3" size=".03"/></body>
  </body>
</worldbody></mujoco>"""


def test_pairs_overlapping_in_every_configuration_are_excluded():
    model = mujoco.MjModel.from_xml_string(XML)
    query = SelfCollisionQuery(model)
    names = [sorted(pair) for pair in query.structural_pairs]
    assert ["shell", "upper"] in names
    kept = {tuple(sorted((model.body(int(model.geom_bodyid[a])).name,
                          model.body(int(model.geom_bodyid[b])).name))) for a, b in query.pairs}
    # A swinging arm collides only in some configurations: still checked.
    assert ("arm", "upper") in kept or ("arm", "lower") in kept


def test_mesh_contract_has_no_remaining_structural_pairs():
    path = Path(__file__).resolve().parents[1] / 'robot_models/unitree_g1/g1_29dof_torsobase_mesh.xml'
    model = mujoco.MjModel.from_xml_path(str(path))
    query = SelfCollisionQuery(model)
    assert model.nexclude == 7
    assert len(query.model_pairs) == len(query.pairs) == 542
    assert not query.structural_pairs and not query.structural_evidence
    assert query.structural_sampling['unknown_samples'] == 0
    assert query.structural_sampling['status'] == 'no_candidates'


def test_failure_is_replaced_by_next_deterministic_configuration(monkeypatch):
    model = mujoco.MjModel.from_xml_string(XML)
    original = SelfCollisionQuery.query
    sampled = []
    def query(self, q, **kwargs):
        sampled.append(q.copy())
        if len(sampled) == 2:
            raise GeometryQueryError('test unknown sample')
        return original(self, q, **kwargs)
    monkeypatch.setattr(SelfCollisionQuery, 'query', query)
    first = SelfCollisionQuery(model)
    first_samples = np.array(sampled)
    sampled.clear()
    second = SelfCollisionQuery(model)
    np.testing.assert_array_equal(first_samples, sampled)
    np.testing.assert_array_equal(first.pairs, second.pairs)
    assert first.structural_evidence
    for evidence in first.structural_evidence.values():
        assert evidence['status'] == 'certified'
        assert evidence['samples'] == evidence['overlapping_samples'] == 65
        assert evidence['unknown_samples'] == evidence['replacement_samples'] == 1
        assert evidence['known_sample_indices'] == [0, *range(2, 66)]
        assert len(evidence['sample_depths_m']) == 65


def test_incomplete_evidence_keeps_pairs_and_records_reason(monkeypatch):
    model = mujoco.MjModel.from_xml_string(XML)
    original = SelfCollisionQuery.query
    calls = []
    def query(self, q, **kwargs):
        calls.append(q.copy())
        if len(calls) > 1:
            raise GeometryQueryError('test unknown sample')
        return original(self, q, **kwargs)
    monkeypatch.setattr(SelfCollisionQuery, 'query', query)
    uncertain = SelfCollisionQuery(model)
    assert len(calls) == 65 + STRUCTURAL_REPLACEMENTS
    np.testing.assert_array_equal(uncertain.pairs, uncertain.model_pairs)
    assert not uncertain.structural_pairs
    assert uncertain.structural_sampling['samples'] == 1
    assert uncertain.structural_sampling['unknown_samples'] == 128
    assert uncertain.structural_sampling['replacement_samples'] == STRUCTURAL_REPLACEMENTS
    assert uncertain.structural_evidence
    for evidence in uncertain.structural_evidence.values():
        assert evidence['status'] == 'uncertified'
        assert evidence['uncertified_reason'] == 'insufficient_determinate_samples'


def test_all_unknown_samples_never_exclude(monkeypatch):
    def query(*args, **kwargs):
        raise GeometryQueryError('all samples unknown')
    monkeypatch.setattr(SelfCollisionQuery, 'query', query)
    query = SelfCollisionQuery(mujoco.MjModel.from_xml_string(XML))
    np.testing.assert_array_equal(query.pairs, query.model_pairs)
    assert not query.structural_evidence
    assert query.structural_sampling['samples'] == 0
    assert query.structural_sampling['unknown_samples'] == 129


def test_no_failure_preserves_legacy_pairs_ids_and_row_indices(monkeypatch):
    # Reproduce the previous classification on the same compiled model.
    from omniretargeting.self_collision import STRUCTURAL_SAMPLES, STRUCTURAL_SEED, _random_quaternion
    model = mujoco.MjModel.from_xml_string(XML)
    current = SelfCollisionQuery(model)
    def legacy(self):
        rng = np.random.default_rng(STRUCTURAL_SEED)
        overlapping = None
        for sample in range(STRUCTURAL_SAMPLES + 1):
            q = model.qpos0.copy()
            for joint in range(model.njnt) if sample else ():
                address, kind = model.jnt_qposadr[joint], model.jnt_type[joint]
                if kind == mujoco.mjtJoint.mjJNT_FREE:
                    q[address:address+3] = rng.uniform(-1., 1., 3)
                    q[address+3:address+7] = _random_quaternion(rng)
                elif kind == mujoco.mjtJoint.mjJNT_BALL:
                    q[address:address+4] = _random_quaternion(rng)
                elif model.jnt_limited[joint]:
                    q[address] = rng.uniform(*model.jnt_range[joint])
                elif kind == mujoco.mjtJoint.mjJNT_HINGE:
                    q[address] = rng.uniform(-np.pi, np.pi)
            indices = {r['pair_index'] for r in self.query(q, distance=0.) if r['clearance'] < 0}
            overlapping = indices if overlapping is None else overlapping & indices
            if not overlapping:
                return
        keep = np.ones(len(self.pairs), bool)
        keep[sorted(overlapping or ())] = False
        self.pairs, self.pair_ids = self.pairs[keep], self.pair_ids[keep]
    monkeypatch.setattr(SelfCollisionQuery, '_exclude_structural_overlaps', legacy)
    previous = SelfCollisionQuery(model)
    assert current.pairs.tobytes() == previous.pairs.tobytes()
    assert current.pair_ids.tobytes() == previous.pair_ids.tobytes()
    for q in (model.qpos0, model.qpos0 + .001):
        assert [(r['pair'], r['pair_id'], r['pair_index']) for r in current.query(q)] == [
            (r['pair'], r['pair_id'], r['pair_index']) for r in previous.query(q)]
