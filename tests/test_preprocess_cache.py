"""Immutable preparation reuse preserves geometry bits and independent results."""
from dataclasses import replace

import numpy as np
import trimesh

from omniretargeting._core.preprocess_cache import CertificateMap, SourceGraphCache
from omniretargeting._sequence.tasks import graph_operator
from omniretargeting.contact_mechanics import ContactCandidate
from omniretargeting import surface_observation


def candidate(frozen=True):
    points = np.array([[0., 0., 0.], [.1, 0., 0.]])
    normals = np.array([[0., 0., 1.], [0., 0., 1.]])
    if frozen:
        points.setflags(write=False)
        normals.setflags(write=False)
    return ContactCandidate('patch', points, points, normals, .6, .03)


def test_source_graphs_equal_uncached_qhull_and_task_operators_do_not_alias():
    rng = np.random.default_rng(271)
    human, environment = rng.normal(size=(6, 8, 3)), rng.normal(size=(6, 4, 3))
    expected = [graph_operator(np.vstack([a, b]), 8) for a, b in zip(human, environment)]
    calls = []

    def build(*args):
        calls.append(1)
        return graph_operator(*args)

    cache = SourceGraphCache()
    first, second = cache.get(human, environment, build), cache.get(human.copy(), environment.copy(), build)
    assert len(calls) == 6
    assert all(a.tobytes() == b.tobytes() == c.tobytes() for a, b, c in zip(first, second, expected))
    first[0][:] = 23.
    assert second[0].tobytes() == cache.get(human, environment, build)[0].tobytes()
    environment[2, 0, 0] += .1
    cache.get(human, environment, build)
    assert len(calls) == 12
    human[0, 0, 0] = 0.
    cache.get(human, environment, build)
    human[0, 0, 0] = -0.
    cache.get(human, environment, build)
    assert len(calls) == 24  # signed zero also invalidates a bitwise cache


def test_certificates_reuse_across_maps_without_reordering_or_result_aliases():
    contact = candidate()
    calls = []

    def certify(task):
        calls.append(task)
        return dict(status='feasible', wrench=np.array(task[2], copy=True), excluded=[])

    mapper = CertificateMap(lambda function, items: list(map(function, items)), certify)
    a = ([contact], np.zeros(3), np.arange(6.), np.zeros(6))
    b = ([contact], np.zeros(3), np.arange(6.) + 1., np.zeros(6))
    result = mapper(certify, [a, b, a])
    assert len(calls) == 2
    assert result[0]['wrench'].tobytes() == result[2]['wrench'].tobytes()
    result[0]['wrench'][:] = -9.
    result[0]['excluded'].append('mutated')
    later = mapper(certify, [a])[0]
    assert len(calls) == 2 and mapper.hits == 2
    assert later['wrench'].tobytes() == np.arange(6.).tobytes() and not later['excluded']


def test_support_cache_does_not_merge_contact_order_friction_or_uncertainty():
    first = candidate()
    second = replace(first, name='second')
    friction = replace(first, friction=.9)
    inputs = [([first, second], np.zeros(3), np.zeros(6), 0.),
              ([second, first], np.zeros(3), np.zeros(6), 0.),
              ([friction, second], np.zeros(3), np.zeros(6), 0.),
              ([first, second], np.zeros(3), np.zeros(6), .1),
              ([first, second], np.zeros(3), np.zeros(6), -0.)]
    calls = []

    def certify(task):
        calls.append(task)
        return {'status': 'uncertain'}

    mapper = CertificateMap(lambda function, items: list(map(function, items)), certify)
    mapper(certify, inputs + inputs)
    assert len(calls) == len(inputs)


def test_mutable_candidates_and_stateful_functions_keep_original_map():
    contact = candidate(frozen=False)
    task = ([contact], np.zeros(3), np.zeros(6), 0.)
    calls = []

    def certify(task):
        calls.append(task)
        return {'status': 'feasible'}

    mapper = CertificateMap(lambda function, items: list(map(function, items)), certify)
    mapper(certify, [task, task])
    mapper(certify, [task])
    assert len(calls) == 3
    states = []
    def nominal(item):
        states.append(item)
        return len(states)
    assert mapper(nominal, [task, task]) == [1, 2]
    assert mapper(nominal, [task]) == [3]


def test_pose_only_surface_keeps_tree_lazy_and_queries_build_it_once(monkeypatch):
    constructors = []
    original = surface_observation.igl.AABB

    def tree():
        constructors.append(1)
        return original()

    monkeypatch.setattr(surface_observation.igl, 'AABB', tree)
    mesh = trimesh.creation.box()
    vertices = mesh.vertices.tobytes()
    surface = surface_observation.ContactSurface(mesh, 2, name='object')
    assert not constructors and surface._tree is None
    points = np.array([[[0., 0., .6]], [[0., 0., .61]]])
    first, second = surface.observe(points, 30.), surface.observe(points, 30.)
    assert constructors == [1]
    assert all(a.tobytes() == b.tobytes() for a, b in zip(first, second))
    assert vertices == mesh.vertices.tobytes()
