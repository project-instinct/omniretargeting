"""The correction QP sees every guard's fixed budgets before proposing a step."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from omniretargeting._fast.acquisition_guard import AcquisitionNormalGuard
from omniretargeting._fast.contact_guard import ContactGuard
from omniretargeting._fast.continuity import CorrectionContinuity
from omniretargeting._fast.repair_limits import RepairLimits
from omniretargeting._sequence.continuity_guard import SequenceContinuityGuard
from tests.test_fast_contact_guard import _acquired


def violation(bounds, step):
    linear, norms = bounds
    excess = 0.
    for matrix, lower, upper in linear:
        value = matrix @ step
        excess = max(excess, np.max(value-upper, initial=0), np.max(lower-value, initial=0))
    for matrix, residual, allowed in norms:
        value = (matrix @ step).reshape(residual.shape) + residual
        excess = max(excess, np.max(np.linalg.norm(value, axis=1)-allowed, initial=0))
    return excess


@pytest.mark.parametrize('surfaces',[False,True])
def test_contact_and_acquisition_rows_reject_lift_and_retain_tangent_progress(surfaces):
    objective,poses=_acquired(surfaces=surfaces)
    objective.tasks.contact_phase=None
    for surf in objective.surfaces:
        surf['normals'][:]=[0,0,1.];surf['sticking'][:]=True
    for guard in [ContactGuard(objective,poses),AcquisitionNormalGuard(objective,poses)]:
        bounds=guard.repair_bounds(poses)
        assert violation(bounds,np.zeros(objective.frames*objective.width))<1e-9
        lifted=poses.copy();lifted[:,2]+=.005
        step=objective.coords.difference(poses,lifted).ravel()
        assert violation(bounds,step)>.003
        assert not guard.accepts(lifted)
        # Rows retain their original budget after evaluating a bad proposal.
        assert violation(guard.repair_bounds(poses),step)>.003


def test_all_guards_project_into_fixed_root_repair_coordinates():
    objective,poses=_acquired(surfaces=True)
    objective.tasks.contact_phase=None
    rotations=Rotation.from_quat(poses[:,3:7],scalar_first=True)
    guards=[ContactGuard(objective,poses),AcquisitionNormalGuard(objective,poses),
            CorrectionContinuity(objective.coords,poses),
            SequenceContinuityGuard(objective.tasks,poses,rotations)]
    objective.candidate_guards=guards
    limits=RepairLimits(objective,poses)
    packed=limits.constraints(poses)
    matrix,lower,upper=packed.packed()
    assert matrix.shape[1]==len(poses)*len(objective.coords.qi)
    assert np.max(lower,initial=0)<1e-7
    assert np.min(upper,initial=0)>-1e-7
    for jac,residual,allowed in packed.norms:
        assert jac.shape[1]==matrix.shape[1]
        assert np.max(np.linalg.norm(residual,axis=1)-allowed,initial=0)<1e-7
    # One-frame chatter cannot be proposed by the QP and only rejected later.
    candidate=poses.copy();candidate[10,objective.coords.qi[-1]]+=.001
    step=objective.coords.difference(poses,candidate).ravel()
    for guard in guards[2:]:
        assert violation(guard.repair_bounds(poses),step)>.0009
        assert not guard.accepts(candidate)


@pytest.mark.parametrize("contact_guard", [AcquisitionNormalGuard, ContactGuard])
def test_accepted_roundoff_cannot_make_the_zero_repair_infeasible(contact_guard):
    objective,poses=_acquired()
    objective.tasks.contact_phase=None
    acquisition=contact_guard(objective,poses)
    candidate=poses.copy();candidate[:,2]+=.001+8e-8
    assert acquisition.accepts(candidate)
    assert violation(acquisition.repair_bounds(candidate),np.zeros(objective.frames*objective.width))<1e-12
    candidate[:,2]+=3e-7
    assert not acquisition.accepts(candidate)
    assert violation(acquisition.repair_bounds(candidate),np.zeros(objective.frames*objective.width))>1e-7
    rotations=Rotation.from_quat(poses[:,3:7],scalar_first=True)
    for guard in [CorrectionContinuity(objective.coords,poses),
                  SequenceContinuityGuard(objective.tasks,poses,rotations)]:
        candidate=poses.copy()
        # A ramp adds 1.8e-7 velocity, with zero acceleration and jerk.
        candidate[:,objective.coords.qi[-1]]+=np.arange(len(poses))*1.8e-7
        assert guard.accepts(candidate)
        assert violation(guard.repair_bounds(candidate),np.zeros(objective.frames*objective.width))<1e-12
