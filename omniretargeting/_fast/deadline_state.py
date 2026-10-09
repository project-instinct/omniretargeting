"""Deadline checkpoints at the legacy pipeline's existing acceptance boundaries.

No objective, iteration quota or numerical decision changes before expiry.
Native geometry/linear solves finish before the next cooperative check; their
overshoot is reported rather than treating an unverified trial as accepted.
"""
from contextvars import ContextVar
from contextlib import contextmanager
import os
import time
import mujoco
import numpy as np
from .deadline import Deadline, DeadlineExceeded

_CURRENT = ContextVar('fast_production_deadline', default=None)
STAGES = ('source_pose', 'grid_setup', 'contact_acquisition', 'reweight',
          'free_bodies', 'original_mesh_repair', 'final_audit')


class PipelineDeadline:
    def __init__(self, owner, model, initial, frames, config, framerate):
        self.owner, self.model, self.pid = owner, model, os.getpid()
        self.deadline = Deadline.for_clip(frames, started=getattr(owner, '_production_started', None),
            seconds_per_frame=config.get('time_budget_seconds_per_frame', 1.0))
        owner._production_deadline = self.deadline
        pose = np.asarray(initial, float)
        self.motion = np.tile(pose, (frames, 1)) if pose.ndim == 1 else pose.copy()
        self.stage, self.accepted_stage = 'source_pose', 'initialization'
        self.started = time.monotonic()
        self.report = dict(backend='fast', stages={}, frames=frames, framerate=float(framerate))
        self.framerate = float(framerate)
        self.source = None
        self.reference = None

    def commit(self, poses, label, *, iteration=False):
        # Forked window workers do not own the global accepted trajectory.
        if os.getpid() != self.pid or np.shape(poses) != self.motion.shape:
            return
        if iteration and label not in ('source_pose', 'contact_scene'):
            return  # Reweight/free candidates require the existing stage guard.
        self.motion = np.array(poses, copy=True)
        self.accepted_stage = label

    def finish(self, *, expired=False):
        if expired:
            if self.accepted_stage == 'initialization':
                # Only the fallback needs normalization. The normal solve never
                # sees this projection, preserving its original numeric path.
                for pose in self.motion:
                    mujoco.mj_normalizeQuat(self.model, pose)
                    for j in range(self.model.njnt):
                        if self.model.jnt_limited[j] and self.model.jnt_type[j] in (
                                mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                            q = self.model.jnt_qposadr[j]
                            pose[q] = np.clip(pose[q], *self.model.jnt_range[j])
            self.report.update(status='deadline', frames=len(self.motion), framerate=self.framerate,
                model_nq=self.model.nq, model_nv=self.model.nv, final_audit='not_run_or_incomplete',
                last_accepted_stage=self.accepted_stage, seconds=time.monotonic()-self.started,
                skipped_stages=list(STAGES[STAGES.index(self.stage):]),
                defects=[dict(kind='unfinished_budget', stage=self.stage,
                              accepted_stage=self.accepted_stage)],
                budgets={'audited': False, 'reason': 'deadline'},
                skipped_tasks=['final_diagnostics_due_to_deadline'])
        else:
            self.report.setdefault('skipped_stages', [])
        self.owner.last_fast_source_pose = self.motion.copy() if self.source is None else self.source
        self.owner.last_fast_reference = self.motion.copy() if self.reference is None else self.reference
        self.report['deadline'] = self.deadline.describe()
        self.report['deadline']['export'] = self.deadline.reserve_for_export()
        self.report['deadline']['cooperative_native_call_overshoot_s'] = max(0., -self.deadline.remaining())
        return self.motion, self.report


@contextmanager
def installed(state):
    token = _CURRENT.set(state)
    try:
        yield state
    finally:
        _CURRENT.reset(token)


def check(label, *, stage=None):
    state = _CURRENT.get()
    if state is not None:
        if stage is not None:
            state.stage = stage
        state.deadline.check(label)


def accepted(poses, label, *, iteration=False):
    state = _CURRENT.get()
    if state is not None:
        state.commit(poses, label, iteration=iteration)


def state():
    return _CURRENT.get()
