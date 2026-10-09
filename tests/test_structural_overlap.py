"""Audit geom identity and sampled evidence are independent of the solver."""
from types import SimpleNamespace
import mujoco
import numpy as np
from omniretargeting.audit_self_collision import AuditSelfCollisionQuery
from omniretargeting.self_collision import SelfCollisionQuery
from omniretargeting.structural_overlap import StructuralOverlap
from omniretargeting.geometry_query import GeometryQueryError


def model_query():
    model = mujoco.MjModel.from_xml_string('''<mujoco><compiler angle="radian"/><worldbody>
      <geom name="visual" type="sphere" size=".01" contype="0" conaffinity="0"/>
      <body name="a"><joint type="hinge" range="-.01 .01"/><geom type="sphere" size=".1"/></body>
      <body name="b" pos=".15 0 0"><joint type="hinge" range="-.01 .01"/><geom type="sphere" size=".1"/></body>
      <body name="c" pos="1 0 0"><joint type="slide" axis="1 0 0" range="-1 1"/><geom type="sphere" size=".1"/></body>
      </worldbody></mujoco>''')
    query = AuditSelfCollisionQuery(model)
    query.use_model_pairs()
    return model, query


def test_audit_restores_geom_ids_not_body_name_pairs():
    model, query = model_query()
    evidence = query.structural_evidence[1, 2]
    assert evidence['samples'] == evidence['overlapping_samples'] == 65
    assert evidence['unknown_samples'] == 0
    collector = StructuralOverlap(query, 2)
    poses = np.tile(model.qpos0, (2, 1)); poses[1, 2] = -.7
    for f, pose in enumerate(poses):
        rows = collector.collect(query.query(pose), f)
        assert all(tuple(r['pair']) != (1, 2) for r in rows)
        if f == 1:
            assert any(tuple(r['pair']) == (2, 3) for r in rows)
    report = collector.describe()
    assert report['pairs']['self:1:2']['max_depth_m'] > .049
    assert report['pairs']['self:1:2']['bodies'] == ['a', 'b']
    assert report['training_mjcf_recommendation']['applied'] is False
    assert model.nexclude == 0
    legacy = SelfCollisionQuery(model)
    assert legacy.structural_pairs == [['a', 'b']]
    assert not hasattr(legacy, 'structural_evidence')


def test_independent_audits_separate_same_pairs():
    from scripts.review_training_collision import audit_self
    from scripts.review_penetration_persistence import _query_self
    model, _ = model_query(); poses = np.tile(model.qpos0, (2, 1))
    training, _ = audit_self(model, SimpleNamespace(qpos=poses, labels=np.arange(2)), np.arange(3))
    pairs, depths, errors, persistence = _query_self(model, poses, workers=1)
    assert not pairs and not errors and np.max(depths) == 0
    assert training['max_m'] == 0
    assert training['model_structural_overlap']['pairs'] == persistence['model_structural_overlap']['pairs']
    assert training['model_structural_overlap']['pair_count'] == 1


def test_unknown_model_sample_does_not_certify_structural_overlap(monkeypatch):
    model, _ = model_query()
    original = AuditSelfCollisionQuery.query
    calls = []
    def query(self, *args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise GeometryQueryError('test unknown sample')
        return original(self, *args, **kwargs)
    monkeypatch.setattr(AuditSelfCollisionQuery, 'query', query)
    uncertain = AuditSelfCollisionQuery(model)
    assert not uncertain.structural_evidence
    assert [1, 2] in uncertain.pairs.tolist()
