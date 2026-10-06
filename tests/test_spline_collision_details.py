"""Per-body reports retain both self-pair participants and actual frame IDs."""
import json

import mujoco
import numpy as np

from scripts.spline_collision_details import scene_details, self_details, compare_self


def model():
    return mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
    <body name="left"><geom name="a" size=".1"/>
      <geom name="visual" size=".3" contype="0" conaffinity="0"/></body>
    <body name="right" pos="1 0 0"><geom name="b" size=".1"/></body>
    </worldbody></mujoco>''')


def test_static_body_regression_cannot_hide_in_another_body(tmp_path):
    robot = model()
    a = np.zeros((4, robot.ngeom)); b = a.copy()
    a[:, robot.geom('a').id] = .008
    b[:, robot.geom('b').id] = [0., .003, .004, 0.]
    b[:, robot.geom('visual').id] = .9
    paths = []
    for name, values in [('A', a), ('B', b)]:
        path = tmp_path / f'{name}.json'
        path.write_text(json.dumps(dict(start=12, frames=4, static=dict(uncertain_samples=0))))
        np.savez(path.with_suffix('.npz'), static_depths=values)
        paths.append(path)
    report = scene_details(paths[1], paths[0], robot)
    parts = report['categories']['static']['per_body']
    assert parts['left']['delta']['max_m'] < 0
    assert parts['right']['max_m'] == .004
    assert parts['right']['worst_frame'] == 14
    assert parts['right']['worse_than_A_intervals'] == [dict(start_frame=13,end_frame=14,frames=2)]
    assert not report['collision_free_certificate']


def test_self_pair_reports_both_bodies_and_native_tolerance():
    robot = model()
    pair = (robot.geom('a').id, robot.geom('b').id)
    report = self_details({pair: np.array([.0002,.002,.003])}, robot, np.arange(20,23), .001)
    for name in ('left','right'):
        assert report['per_body'][name]['worst_frame'] == 22
        assert report['per_body'][name]['affected_frames'] == [21,22]
        assert report['per_body'][name]['threshold_m'] == .001


def test_self_comparison_preserves_body_pair_frame_regressions_and_unknown(tmp_path):
    robot = model()
    pair = (robot.geom('a').id, robot.geom('b').id)
    paths = []
    for name, depth in [('A', [.002, .001, 0.]), ('B', [.001, .003, 0.])]:
        path = tmp_path / f'{name}.json'
        path.write_text(json.dumps(dict(inputs=dict(frame_start=20, input_frames=3,
            motion_payload_framerate_hz=30.), model=dict(collision_model_sha256='same'),
            self_collision=dict(unknown_query_count=0,
                **self_details({pair: np.asarray(depth)}, robot, np.arange(20, 23), .001)))))
        paths.append(path)
    result = compare_self(paths[1], paths[0])
    assert result['status'] == 'measured'
    for name in ('left', 'right'):
        row = result['per_body'][name]
        assert row['delta']['max_m'] == .001
        assert row['worst_frame'] == 21
        assert row['worse_than_A_intervals'] == [dict(start_frame=21, end_frame=21, frames=1)]
    assert len(result['per_pair']) == 1 and not result['collision_free_certificate']
    data = json.loads(paths[1].read_text())
    data['self_collision']['unknown_query_count'] = 1
    paths[1].write_text(json.dumps(data))
    assert compare_self(paths[1], paths[0])['status'] == 'unknown'
