"""Exact resident winding BVH for original static triangle queries."""

import subprocess
import weakref

from ..scene_geometry import SceneQuery
from .original_geometry_cpu import serial_scene

_QUERY = SceneQuery.query


def prepare(geometry):
    """Construct immutable native state once, before measurement workers fork."""
    scene = geometry.scene
    if (type(scene) is not SceneQuery or getattr(scene, '_r4_serial', False)
            or getattr(scene.query, '__func__', None) is not _QUERY):
        return scene
    state = getattr(geometry, '_native_repair_static_query', None)
    if state is None or state[0] is not scene:
        try:
            query = serial_scene(scene)
            for backend in query._r4_winding_cache.values():
                if not hasattr(backend, '_repair_release'):
                    backend._repair_release = weakref.finalize(
                        backend, backend.library.r4_winding_free, backend.state)
        except (OSError, RuntimeError, subprocess.CalledProcessError):
            query = scene
        state = geometry._native_repair_static_query = scene, query
    return state[1]


def query_scene(geometry, points):
    """Return original values; custom query callbacks retain their dispatch."""
    return prepare(geometry).query(points)
