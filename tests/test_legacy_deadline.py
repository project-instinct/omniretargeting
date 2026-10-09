"""Expiry never exports a rejected reweight/free trial or a partial window."""
from types import SimpleNamespace
import time
import numpy as np
import pytest
from omniretargeting._fast import deadline_state as ds
from omniretargeting._fast.pipeline import solve_fast_sequence
from tests.test_fast_backend import _problem


def make_state():
    model, *rest = _problem()
    initial = model.qpos0
    owner = SimpleNamespace()
    state = ds.PipelineDeadline(owner, model, initial, 4, {}, 30.)
    return owner, state


def test_stage_acceptance_is_not_iteration_acceptance():
    owner, state = make_state()
    accepted = state.motion.copy(); accepted[:, 0] += .001
    rejected = accepted.copy(); rejected[:, 0] += .1
    with ds.installed(state):
        ds.accepted(accepted, 'contact_scene', iteration=True)
        ds.accepted(rejected, 'reweight_1', iteration=True)
        ds.accepted(rejected, 'free_bodies', iteration=True)
        ds.accepted(rejected[:2], 'contact_scene', iteration=True)
    result, report = state.finish(expired=True)
    np.testing.assert_array_equal(result, accepted)
    assert report['last_accepted_stage'] == 'contact_scene'
    assert report['budgets']['audited'] is False
    assert report['deadline']['export']['reserved_s'] == 2.
    assert not ds.state()


def test_stage_guard_can_commit_free_body_result():
    _, state = make_state()
    trial = state.motion.copy(); trial[:, 0] += .01
    with ds.installed(state):
        ds.accepted(trial, 'free_bodies')
    np.testing.assert_array_equal(state.finish(expired=True)[0], trial)


def test_pipeline_expired_entry_exports_joint_valid_fallback(monkeypatch):
    model, *_ = _problem()
    owner = SimpleNamespace(_production_started=time.monotonic()-10.)
    initial = model.qpos0.copy(); initial[-1] = 100.
    def unexpected(*a, **k):
        raise AssertionError('expired pipeline must not start numerical stages')
    monkeypatch.setattr('omniretargeting._fast.pipeline._solve_fast_sequence', unexpected)
    q, report = solve_fast_sequence(owner, model, initial, np.zeros((4, 6, 3)), [], {},
        None, {}, {}, {'time_budget_seconds_per_frame': .05}, {'framerate':30.}, [])
    assert q.shape == (4, model.nq)
    assert q[:, -1].max() == model.jnt_range[-1,1]
    assert report['status'] == 'deadline'
    assert report['skipped_stages'] == list(ds.STAGES)
    assert report['deadline']['first_expired_label'] == 'pipeline_entry'
    assert owner.last_fast_source_pose.shape == q.shape


def test_pipeline_timeout_returns_checkpoint_and_restores_context(monkeypatch):
    owner, state = make_state(); model=state.model
    accepted=state.motion.copy(); accepted[:,0] += .01
    def interrupted(*a, **k):
        ds.accepted(accepted, 'contact_scene', iteration=True)
        ds.state().deadline.started -= 100.
        ds.check('repair_group_attempt', stage='original_mesh_repair')
    monkeypatch.setattr('omniretargeting._fast.pipeline._solve_fast_sequence', interrupted)
    q, report = solve_fast_sequence(owner, model, model.qpos0, np.zeros((4,6,3)), [], {},
        None, {}, {}, {}, {'framerate':30.}, [])
    np.testing.assert_array_equal(q, accepted)
    assert report['skipped_stages'] == ['original_mesh_repair','final_audit']
    assert ds.state() is None


def test_gpu_wait_respects_deadline_and_retires_pending_reply():
    from omniretargeting._fast.gpu_geometry import GeometryService
    _, state = make_state()
    events = []
    class Connection:
        def send(self, message): events.append('send')
        def poll(self, timeout):
            assert 0 < timeout <= state.deadline.remaining()+.01
            state.deadline.started -= 100
            return False
        def recv(self): raise AssertionError('must not consume late reply')
        def close(self): events.append('close')
    class Process:
        def terminate(self): events.append('terminate')
        def wait(self, **kwargs): events.append('wait')
    service=GeometryService.__new__(GeometryService)
    service.connection=Connection();service.process=Process()
    with ds.installed(state), pytest.raises(ds.DeadlineExceeded):
        service.request('build','token',())
    assert events==['send','terminate','wait','close']


def test_gpu_reply_within_budget_is_unmodified():
    from omniretargeting._fast.gpu_geometry import GeometryService
    _, state = make_state()
    answer=object()
    connection=SimpleNamespace(send=lambda msg:None,poll=lambda timeout:True,recv=lambda:(True,answer))
    service=GeometryService.__new__(GeometryService);service.connection=connection
    with ds.installed(state):
        assert service.request('query','token',()) is answer
