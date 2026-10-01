"""Reused self-collision pair results equal fresh queries exactly."""

import mujoco
import numpy as np

from omniretargeting.self_collision import SelfCollisionQuery

XML = """<mujoco><worldbody>
  <body name="base"><freejoint/><geom size=".05"/>
    <body name="left" pos="0 .08 -.1"><joint axis="1 0 0" range="-1 1"/>
      <geom type="capsule" fromto="0 0 0 0 0 -.3" size=".04"/></body>
    <body name="right" pos="0 -.08 -.1"><joint axis="1 0 0" range="-1 1"/>
      <geom type="capsule" fromto="0 0 0 0 0 -.3" size=".04"/></body>
    <body name="arm" pos=".12 0 0"><joint axis="0 1 0" range="-3 3"/>
      <geom type="box" size=".03 .03 .15" pos="0 0 -.15"/></body>
  </body>
</worldbody></mujoco>"""


def _same(actual, expected):
    assert len(actual) == len(expected)
    for row, reference in zip(actual, expected):
        assert row.keys() == reference.keys()
        for name in row:
            np.testing.assert_array_equal(np.asarray(row[name]), np.asarray(reference[name]))


def test_unchanged_pairs_are_copied_and_moved_pairs_recomputed():
    model = mujoco.MjModel.from_xml_string(XML)
    query = SelfCollisionQuery(model)
    reuse = {}
    q = model.qpos0.copy()
    q[7:10] = [.6, -.6, .4]
    first = query.query(q, reuse=reuse)
    assert first and reuse
    _same(first, query.query(q))
    before = query.pair_queries
    _same(query.query(q, reuse=reuse), first)
    assert query.pair_queries == before
    # Moving only the arm recomputes its pairs; the leg pair is copied.
    q[9] += .3
    moved = query.query(q, reuse=reuse)
    computed = query.pair_queries - before
    _same(moved, query.query(q))
    assert 0 < computed < len(reuse)
    # Returned rows are copies: callers may annotate them.
    moved[0]["points"][:] = np.nan
    moved[0]["jacobian"] = None
    _same(query.query(q, reuse=reuse), query.query(q))


def test_fresh_query_state_reuses_filtering_but_not_mutable_state():
    model = mujoco.MjModel.from_xml_string(XML)
    original = SelfCollisionQuery(model)
    clone = original.fresh_state()
    assert clone.data is not original.data
    assert clone.cfg is not original.cfg
    assert clone.pairs is not original.pairs
    assert clone.pair_queries == 0
    q = model.qpos0.copy();q[7:10] = [.6, -.6, .4]
    for amount in [0., .001, -.03]:
        q[9] += amount
        expected, actual = original.query(q), clone.query(q)
        _same(actual, expected)
        for a, b in zip(actual, expected):
            for key in a:
                x, y = np.asarray(a[key]), np.asarray(b[key])
                assert x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes()
    clone._segment_cache['sentinel'] = None
    clone.cfg['detection_distance'] = .004
    assert not original._segment_cache and original.cfg['detection_distance'] == .03
    if hasattr(clone, '_normal_data'):
        assert clone._normal_data is not original._normal_data
