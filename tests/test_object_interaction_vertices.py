"""Moving objects must enter the same world-space graph on both sides."""
from types import MethodType

import numpy as np
import pytest

from omniretargeting.retargeting import GenericInteractionRetargeter


def test_per_frame_object_points_enter_source_and_robot_graph_without_mutation():
    solver = GenericInteractionRetargeter.__new__(GenericInteractionRetargeter)
    solver.terrain_points = np.array([[-2., -2., 0.], [2., -2., 0.], [0., 2., 0.]])
    solver.end_effector_config = {}
    solver.last_pose_initialization = {}
    solver._initialize_pose = lambda q, *args: q
    calls = []

    def capture(self, q, target, adjacency, environment, **kwargs):
        calls.append((target.copy(), environment.copy()))
        return q

    solver._optimize_configuration = MethodType(capture, solver)
    human = np.array([[0., 0., 1.], [.2, .1, 1.3], [-.2, .2, 1.1]])
    objects = np.array([[.8, .2, .4], [.9, .3, .6], [.7, -.2, .5]])
    original_human, original_objects = human.copy(), objects.copy()
    for shift in [0., .15]:
        points = objects + [shift, 0., 0.]
        solver.retarget_frame(human, np.zeros(7), object_points=points)
        np.testing.assert_array_equal(calls[-1][1], np.vstack([solver.terrain_points, points]))
        assert len(calls[-1][0]) == len(human) + len(solver.terrain_points) + len(points)
    assert not np.array_equal(calls[0][0], calls[1][0])
    np.testing.assert_array_equal(human, original_human)
    np.testing.assert_array_equal(objects, original_objects)


@pytest.mark.parametrize('points', [np.ones((2, 2)), np.array([[0, 0, np.nan]])])
def test_invalid_object_vertices_fail_before_graph_creation(points):
    solver = GenericInteractionRetargeter.__new__(GenericInteractionRetargeter)
    solver.terrain_points = np.zeros((1, 3))
    with pytest.raises(ValueError, match='object_points'):
        solver.retarget_frame(np.ones((3, 3)), np.zeros(7), object_points=points)
