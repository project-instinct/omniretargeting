"""Compact repair retains constraints crossing every variable window edge."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from omniretargeting._fast import repair_sparse_frames as compact, runtime_options
from omniretargeting._fast.repair_frames import group_frames
from omniretargeting._fast.repair_limits import RepairLimits
from omniretargeting._fast.repair_restoration import _restoration_program
from tests.test_fast_legacy_window import problem


def canonical(bounds, selector):
    program = _restoration_program(bounds, selector, np.zeros(selector.shape[0]), .1)
    matrix, rhs = program[2].toarray(), program[3]
    rows, cones, start = [], [], 0
    for cone in program[4]:
        size = int(str(cone).split('(')[1][:-1])
        block = np.c_[matrix[start:start+size], rhs[start:start+size]]
        if str(cone).startswith('Nonnegative'):
            rows.extend(block)
        else:
            cones.append(block)
        start += size
    # Row/cone permutations leave the conic feasible set unchanged.
    rows = sorted(rows, key=lambda row: tuple(np.round(row, 8)))
    cones.sort(key=lambda block: (block.shape, tuple(np.round(block.ravel(), 8))))
    return np.array(rows), cones


@pytest.mark.parametrize('restore', [False, True])
def test_compact_program_retains_crossing_support_and_jerk(restore):
    objective, poses = problem(extra=True)
    objective.tasks.contact_phase = SimpleNamespace(
        roles={e['name']: np.where(np.arange(len(poses)) < 12, 'support', 'follow')
               for e in objective.effectors}, surface=None)
    limits = RepairLimits(objective, poses)
    assert limits.support_continuity.groups
    assert limits.check(poses)[0]
    limits._repair_sparse_origin = poses
    limits._repair_sparse_origin_valid = True
    size = limits.frames * limits.width
    ids = np.arange(size).reshape(limits.frames, limits.width)[8:13, :2].ravel()
    selector = sparse.eye(size, format='csc')[:, ids]
    candidate = poses.copy()
    candidate[8, limits.coords.qi[0]] += .0001
    full = limits.constraints(candidate, restore_layout=restore, projection=selector)
    reduced = compact.constraints(limits, candidate, restore_layout=restore, projection=selector)
    assert reduced is not None
    first, old_cones = canonical(full, selector)
    second, new_cones = canonical(reduced, selector)
    np.testing.assert_allclose(first, second, rtol=1e-12, atol=1e-10)
    assert len(old_cones) == len(new_cones)
    for first, second in zip(old_cones, new_cones):
        np.testing.assert_allclose(first, second, rtol=1e-12, atol=1e-10)


def test_uncertified_or_changed_fixed_frames_and_custom_guards_fall_back():
    objective, poses = problem()
    limits = RepairLimits(objective, poses)
    size = limits.frames * limits.width
    selector = sparse.eye(size, format='csc')[:, [limits.width * 8]]
    assert compact.constraints(limits, poses, projection=selector) is None
    limits._repair_sparse_origin = poses
    limits._repair_sparse_origin_valid = True
    changed = poses.copy()
    changed[20, limits.coords.qi[0]] += .1
    assert compact.constraints(limits, changed, projection=selector) is None
    # Predictor frame views may inherit the full-trajectory origin.
    assert compact.constraints(limits, poses[:1], projection=selector) is None
    objective.candidate_guards.append(SimpleNamespace(accepts=lambda q: True))
    assert compact.constraints(limits, poses, projection=selector) is None


def test_activation_filters_tiny_keys_and_keeps_both_self_partners(monkeypatch):
    monkeypatch.setattr(runtime_options, 'OPTIONS', replace(runtime_options.OPTIONS,
                                                          repair_defect_windows=True))
    model = SimpleNamespace(body_weldid=np.arange(6), geom_bodyid=np.arange(6))
    measured = {'depths': {(3, 'static', 2): .0001, (25, 'self', 2, 4): .0047}}
    for body in (2, 4):
        mask = group_frames(measured, model, body, 40, 2)
        assert np.flatnonzero(mask).tolist() == [23, 24, 25, 26, 27]
    original = group_frames(measured, model, 2, 40, 2, activation=0.)
    assert original[3] and original[25]


def test_defect_windows_are_default_off():
    assert runtime_options.RuntimeOptions().repair_defect_windows is False


def test_disabled_policy_preserves_callers_halo(monkeypatch):
    from omniretargeting._fast.repair_defects import frames
    monkeypatch.setattr(runtime_options, 'OPTIONS', replace(runtime_options.OPTIONS,
                                                          repair_defect_windows=False))
    model = SimpleNamespace(body_weldid=np.arange(6), geom_bodyid=np.arange(6))
    limits = SimpleNamespace(frames=40, objective=SimpleNamespace(framerate=30.))
    measured = {'depths': {(3, 'static', 2): .0001}}
    assert np.flatnonzero(frames(limits, measured, model, 2, padding=0)).tolist() == [3]


def test_entry_window_retains_self_frames_after_partial_repair(monkeypatch):
    from omniretargeting._fast.repair_defects import frames, has_defect
    monkeypatch.setattr(runtime_options, 'OPTIONS', replace(runtime_options.OPTIONS,
                                                          repair_defect_windows=True))
    model = SimpleNamespace(body_weldid=np.arange(6), geom_bodyid=np.arange(6))
    entry = {'depths': {(3, 'static', 2): .008, (25, 'self', 2, 4): .0047}}
    pinned = group_frames(entry, model, 2, 40, 2)
    limits = SimpleNamespace(frames=40, objective=SimpleNamespace(framerate=30.),
                             _repair_defect_masks={2: pinned})
    repaired = {'depths': {(3, 'static', 2): .007, (25, 'self', 2, 4): .0008}}
    assert has_defect(repaired, model, 2)
    assert not group_frames(repaired, model, 2, 40, 2)[25]
    assert frames(limits, repaired, model, 2) is pinned
    assert pinned[25]


def test_work_band_is_generic_and_does_not_activate_at_the_boundary(monkeypatch):
    from omniretargeting._fast.repair_defects import has_defect
    monkeypatch.setattr(runtime_options, 'OPTIONS', replace(runtime_options.OPTIONS,
                                                          repair_defect_windows=True))
    model = SimpleNamespace(body_weldid=np.arange(6), geom_bodyid=np.arange(6))
    for kind in ('static', 'moving'):
        measured = {'depths': {(3, kind, 2): .005}}
        assert not has_defect(measured, model, 2)
        measured['depths'][(3, kind, 2)] = np.nextafter(.005, np.inf)
        assert has_defect(measured, model, 2)
        assert not has_defect(measured, model, 4)
