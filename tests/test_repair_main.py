"""Repair CPU reductions preserve rejection decisions and canonical bytes."""

from dataclasses import replace

import numpy as np
import pytest
from scipy import sparse

from omniretargeting._fast import runtime_options
from omniretargeting._fast.contact_guard import ContactGuard
from omniretargeting._fast.repair_limits import RepairLimits
from omniretargeting._fast.repair_restoration import _restoration_program
from tests.test_fast_contact_guard import _acquired
from tests.test_fast_legacy_window import problem


def _set_main_cache(monkeypatch, flag):
    monkeypatch.setattr(runtime_options, 'OPTIONS',
                        replace(runtime_options.OPTIONS, repair_main_cache=flag == '1'))


def _program_bytes(program):
    result = []
    for value in program:
        if sparse.issparse(value):
            result.append((value.shape, value.indptr.tobytes(), value.indices.tobytes(), value.data.tobytes()))
        elif isinstance(value, np.ndarray):
            result.append((value.shape, value.dtype.str, value.tobytes()))
        else:
            result.append(tuple(map(str, value)))
    return result


@pytest.mark.parametrize('restore', [False, True])
@pytest.mark.parametrize('outside', [0., .03])
def test_optimized_program_raw_bytes_equal_original_for_fixed_invalid_frames(monkeypatch, restore, outside):
    objective, poses = problem(extra=True)
    objective.candidate_guards.append(ContactGuard(objective, poses, preserve_positions=True))
    limits = RepairLimits(objective, poses)
    candidate = poses.copy()
    candidate[12, limits.coords.qi[0]] += .001
    candidate[25, limits.coords.qi[-1]] += outside
    indices = np.arange(limits.frames*limits.width).reshape(limits.frames, limits.width)[9:17, :2].ravel()
    projection = sparse.eye(limits.frames*limits.width, format='csc')[:, indices]
    programs = []
    for flag in ('0', '1', '1'):
        _set_main_cache(monkeypatch, flag)
        bounds = limits.constraints(candidate, restore_layout=restore, projection=projection)
        programs.append(_program_bytes(_restoration_program(bounds, projection, np.zeros(projection.shape[0]), .1)))
    assert programs[0] == programs[1] == programs[2]


@pytest.mark.parametrize('preserve', [False, True])
@pytest.mark.parametrize('ablate', [False, True])
def test_contact_boolean_matches_report_at_margins_and_invalid_inputs(monkeypatch, preserve, ablate):
    from omniretargeting._fast import runtime_options
    monkeypatch.setattr(runtime_options, 'OPTIONS', replace(runtime_options.OPTIONS, abl_contact_regression=ablate))
    objective, poses = _acquired(surfaces=True, moving=True)
    guard = ContactGuard(objective, poses, preserve_positions=preserve)
    _set_main_cache(monkeypatch, '1')
    for lift in (0., .001, np.nextafter(.0010001, 0.), .0010001, np.nextafter(.0010001, 1.), .004):
        candidate = poses.copy()
        candidate[:, 2] += lift
        assert guard.accepts(candidate) == guard.diagnostics(candidate)['accepted']
    candidate[0, 0] = np.nan
    assert guard.accepts(candidate) == guard.diagnostics(candidate)['accepted'] == False
    assert guard.accepts(candidate[:-1]) == guard.diagnostics(candidate[:-1])['accepted'] == False


def test_contact_acceptance_cache_invalidates_group_config_and_pose(monkeypatch):
    from omniretargeting._fast.repair_guard_cache import accepts
    _set_main_cache(monkeypatch, '1')
    objective, poses = _acquired(surfaces=True)
    guard = ContactGuard(objective, poses)
    calls = []
    original = guard.kin.evaluate
    def evaluate(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(guard.kin, 'evaluate', evaluate)
    assert accepts(guard, poses) and accepts(guard, poses.copy())
    assert len(calls) == 1
    guard.preserve_positions = True
    assert accepts(guard, poses)
    assert len(calls) == 2
    guard.groups = tuple(replace(g, tolerance=g.tolerance*.5) for g in guard.groups)
    assert accepts(guard, poses)
    assert len(calls) == 3
    candidate = poses.copy()
    candidate[:, 2] += .01
    assert not accepts(guard, candidate)
    assert len(calls) == 4


def test_unchanged_witness_metadata_does_not_recompute_identical_jacobian(monkeypatch):
    from copy import deepcopy
    from tests.test_original_geometry_window import _geometry
    monkeypatch.setenv('OMNIRETARGETING_GPU_GEOMETRY', '0')
    _set_main_cache(monkeypatch, '1')
    geometry, poses = _geometry()
    measured = geometry.measure(poses, jacobians=True)
    changed = deepcopy(measured)
    for _, row in changed['rows']:
        row['clearance'] += .0001
    actual = geometry.linearize(poses, changed)
    assert geometry.linearization_stats['computed'] == 0
    assert [r['jacobian'].tobytes() for _, r in measured['rows']] == [r['jacobian'].tobytes() for _, r in actual['rows']]


@pytest.mark.parametrize('kind', ['contact', 'acquisition'])
def test_chunk_acceptance_matches_full_native_fk_with_partial_tail(monkeypatch, kind):
    from omniretargeting._fast.acquisition_guard import AcquisitionNormalGuard
    from omniretargeting._fast.repair_guard_cache import accepts
    _set_main_cache(monkeypatch, '1')
    objective, poses = _acquired(surfaces=True, moving=True)
    guard = ContactGuard(objective, poses, preserve_positions=True) if kind == 'contact' else AcquisitionNormalGuard(objective, poses)
    # Extend immutable evidence rather than rebuilding a different reference.
    repeat = 130 // len(poses) + 1
    def tiled(array):
        value = np.tile(array, (repeat, *(1 for _ in array.shape[1:])))
        value.setflags(write=False)
        return value
    poses = tiled(poses).copy()
    groups = []
    for group in guard.groups:
        if kind == 'contact':
            names = ('targets', 'active', 'projected', 'normals', 'acquired', 'positions', 'established', 'yielded')
            values = {name: tiled(getattr(group, name)) for name in names}
            if group.kind == 'endpoint':
                values['local'] = tiled(group.local)
            groups.append(replace(group, **values))
        else:
            gkind, indices, offsets, *arrays = group
            groups.append((gkind, indices, tiled(offsets) if gkind == 'endpoint' else offsets,
                           *(tiled(array) for array in arrays)))
    guard.groups = tuple(groups)
    if kind == 'contact':
        guard.shape = poses.shape
    for frame in (0, 63, 64, len(poses)-1, 64):
        for lift in (0., .0005, .0010001, np.nextafter(.0010001, 1.), .004):
            trial = poses.copy()
            trial[frame, 2] += lift
            assert accepts(guard, trial) == guard.diagnostics(trial)['accepted']


def test_shared_request_snapshots_preserve_bytes_and_private_depths():
    import multiprocessing
    from types import SimpleNamespace
    from omniretargeting._fast.repair_transport import prepare, pose_message, protected_message, decode_protected
    geometry = SimpleNamespace(model=SimpleNamespace(nq=3, ngeom=1))
    prepare(geometry, multiprocessing.get_context('fork'), 2)
    poses = np.array([[0., -0., 1.], [2., 3., 4.]])
    assert pose_message(geometry, poses) is None
    assert geometry._repair_pose_view.tobytes() == poses.tobytes()
    poses[:] = 10.
    assert geometry._repair_pose_view[0, 0] == 0.
    assert pose_message(geometry, poses.astype(np.float32)).dtype == np.float32
    depths = {(np.int64(0), 'scene', 2): np.float64(-0.), (1, 'self', 2, 3): .001}
    message = protected_message(geometry, depths)
    first, second = decode_protected(geometry, message), decode_protected(geometry, message)
    assert first == second == depths
    assert np.signbit(first[(0, 'scene', 2)])
    first[(1, 'self', 2, 3)] = 9.
    assert second[(1, 'self', 2, 3)] == .001
    custom = {(0, object()): 0.}
    assert protected_message(geometry, custom) is custom


def test_custom_guard_diagnostics_keeps_original_dispatch(monkeypatch):
    from omniretargeting._fast.repair_guard_cache import accepts
    _set_main_cache(monkeypatch, '1')
    objective, poses = _acquired()
    guard = ContactGuard(objective, poses)
    guard.diagnostics = lambda candidate: {'accepted': False}
    assert not guard.accepts(poses)
    assert not accepts(guard, poses)


def test_material_point_cache_matches_einsum_bytes_and_owns_results(monkeypatch):
    from omniretargeting._fast.repair_gate_frames import local_limits
    objective, poses = problem(extra=True)
    limits = RepairLimits(objective, poses)
    for frame, delta in ((0, 0.), (3, .001), (30, -1e-6), (3, np.spacing(poses[3, 7]))):
        trial = poses.copy()
        trial[frame, 7] += delta
        limits.kin.evaluate(trial, False)
        _set_main_cache(monkeypatch, '0')
        expected = [tuple(array.tobytes() for array in group) for group in limits.point_groups()]
        _set_main_cache(monkeypatch, '1')
        for _ in range(2):
            actual = limits.point_groups()
            assert [tuple(array.tobytes() for array in group) for group in actual] == expected
            for point, _, _ in actual:
                point[:] = 9.  # Public ownership cannot corrupt the cache.
        view = local_limits(limits, frame)
        view.kin.evaluate(trial[frame:frame+1], False)
        view.point_groups()
        assert [tuple(array.tobytes() for array in group) for group in limits.point_groups()] == expected
