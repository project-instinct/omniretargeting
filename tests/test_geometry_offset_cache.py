"""Content cache must preserve every local-shell byte and isolate consumers."""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from omniretargeting import scene_geometry as geometry


@pytest.fixture(autouse=True)
def empty_cache():
    with geometry._SHELLS_LOCK:
        geometry._SHELLS.clear()
        geometry._SHELLS_BYTES = 0
    yield
    with geometry._SHELLS_LOCK:
        geometry._SHELLS.clear()
        geometry._SHELLS_BYTES = 0


def model(kind="box", size=".1 .2 .3"):
    asset = ''
    mesh = ''
    if kind == "mesh":
        asset = '<asset><mesh name="tetra" vertex="0 0 0  .2 0 0  0 .3 0  0 0 .4"/></asset>'
        mesh = 'mesh="tetra"'
    return mujoco.MjModel.from_xml_string(f'''<mujoco>{asset}<worldbody>
      <body><freejoint/><geom type="{kind}" size="{size}" {mesh}/></body>
    </worldbody></mujoco>''')


def same_bytes(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.tobytes() == expected.tobytes()


def count_sampling(monkeypatch):
    calls = []
    original = geometry._geom_surface_offsets

    def sample(*args):
        calls.append(args)
        return original(*args)

    monkeypatch.setattr(geometry, "_geom_surface_offsets", sample)
    return calls


@pytest.mark.parametrize("kind,size", [
    ("box", ".1 .2 .3"), ("sphere", ".1"), ("ellipsoid", ".1 .2 .3"),
    ("capsule", ".1 .2"), ("cylinder", ".1 .2"), ("mesh", "1 1 1"),
])
@pytest.mark.parametrize("spacing", [None, 0., .03, .02, .008])
def test_cached_and_reloaded_shells_match_uncached_bytes(monkeypatch, kind, size, spacing):
    first, second = model(kind, size), model(kind, size)
    expected = geometry._geom_surface_offsets(first, 0, spacing)
    calls = count_sampling(monkeypatch)
    same_bytes(geometry.geom_surface_offsets(first, 0, spacing), expected)
    same_bytes(geometry.geom_surface_offsets(second, 0, spacing), expected)
    assert len(calls) == 1
    assert not next(iter(geometry._SHELLS.values())).flags.writeable


def test_caller_edits_cannot_pollute_cache():
    robot = model()
    first = geometry.geom_surface_offsets(robot, 0, .02)
    expected = first.copy()
    first[:] = 123.
    same_bytes(geometry.geom_surface_offsets(robot, 0, .02), expected)


@pytest.mark.parametrize("field", ["geom_size", "geom_type", "mesh_vert"])
def test_in_place_geometry_edits_invalidate_cache(monkeypatch, field):
    robot = model("mesh", "1 1 1") if field == "mesh_vert" else model()
    calls = count_sampling(monkeypatch)
    geometry.geom_surface_offsets(robot, 0, .02)
    if field == "geom_size":
        robot.geom_size[0, 0] += .01
    elif field == "geom_type":
        robot.geom_type[0] = mujoco.mjtGeom.mjGEOM_ELLIPSOID
    else:
        robot.mesh_vert[0, 0] += .01
    same_bytes(geometry.geom_surface_offsets(robot, 0, .02),
               geometry._geom_surface_offsets(robot, 0, .02))
    assert len(calls) == 3  # Two cache misses plus the independent oracle.


def test_spacing_and_geom_id_are_in_key(monkeypatch):
    robot = mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body><freejoint/>
      <geom type="box" size=".1 .2 .3"/><geom type="box" size=".1 .2 .3"/>
    </body></worldbody></mujoco>''')
    calls = count_sampling(monkeypatch)
    for gid, spacing in [(0, None), (0, 0.), (0, .02), (0, .008), (1, .02)]:
        geometry.geom_surface_offsets(robot, gid, spacing)
        geometry.geom_surface_offsets(robot, gid, spacing)
    assert len(calls) == 5
    # Distinct longdouble inputs must not collapse to the same Python float key.
    a = np.longdouble('.02')
    b = np.nextafter(a, np.longdouble(1))
    assert float(a) == float(b)
    assert geometry._geom_surface_key(robot, 0, a) != geometry._geom_surface_key(robot, 0, b)


def test_mesh_selection_metadata_is_in_key():
    robot = model("mesh", "1 1 1")
    original = geometry._geom_surface_key(robot, 0, .02)
    for field in ("geom_dataid", "mesh_vertadr", "mesh_vertnum"):
        array = getattr(robot, field)
        saved = array[0].copy()
        array[0] += -1 if field == "geom_dataid" else 1
        try:
            assert geometry._geom_surface_key(robot, 0, .02) != original
        finally:
            array[0] = saved


def test_unread_model_fields_do_not_invalidate_local_shell(monkeypatch):
    robot = model()
    calls = count_sampling(monkeypatch)
    expected = geometry.geom_surface_offsets(robot, 0, .02)
    robot.geom_pos[0] += .2
    robot.geom_quat[0] = [0., 1., 0., 0.]
    same_bytes(geometry.geom_surface_offsets(robot, 0, .02), expected)
    assert len(calls) == 1


def test_acquisition_and_moving_repair_reuse_matching_parameters(monkeypatch):
    from omniretargeting._fast.objective import FastObjective
    from omniretargeting.moving_scene import MovingSceneQuery

    calls = count_sampling(monkeypatch)
    objective = SimpleNamespace(model=model(), cfg={"collision_spacing_m": .02})
    FastObjective._pack_shells(objective)
    query = MovingSceneQuery(model(), [], 1)
    assert [args[2] for args in calls] == [.02, .008]
    expected = np.unique(np.vstack([objective.shell_local, np.zeros((1, 3))]), axis=0)
    same_bytes(query.shells[0]["offsets"], expected)
    MovingSceneQuery(model(), [], 1)
    assert len(calls) == 2


def test_lru_limits_retained_memory_without_changing_bytes(monkeypatch):
    robot = model()
    expected = geometry._geom_surface_offsets(robot, 0, .02)
    monkeypatch.setattr(geometry, "_SHELLS_MAX_BYTES", expected.nbytes)
    monkeypatch.setattr(geometry, "_SHELLS_MAX_ENTRIES", 1)
    calls = count_sampling(monkeypatch)
    geometry.geom_surface_offsets(robot, 0, .02)
    geometry.geom_surface_offsets(robot, 0, None)
    same_bytes(geometry.geom_surface_offsets(robot, 0, .02), expected)
    assert len(calls) == 3
    assert len(geometry._SHELLS) == 1
    assert geometry._SHELLS_BYTES <= geometry._SHELLS_MAX_BYTES
    monkeypatch.setattr(geometry, "_SHELLS_MAX_BYTES", 1)
    geometry.geom_surface_offsets(robot, 0, .008)
    assert len(geometry._SHELLS) == 1  # Oversized values are returned, not retained.
