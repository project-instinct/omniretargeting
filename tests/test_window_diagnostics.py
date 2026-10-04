"""Numerical captures preserve full domains; tracing does not change a QP."""

import pickle
from dataclasses import replace

import numpy as np

from omniretargeting._fast.qp_assembly import quadratic
from omniretargeting._fast.window_assembly import prepare_window_model, solve_window_model
from omniretargeting._fast.window_diagnostics import capture, source_counts
from tests.test_window_driver import problem


def test_capture_keeps_original_numbers_and_soft_terms_without_provider_closures(tmp_path, monkeypatch):
    objective, poses = problem(8)
    model, mandatory = prepare_window_model(objective, poses, range(0, 4))
    P, c = quadratic(model, .001)
    assert capture(model, P, c, mandatory) is None
    monkeypatch.setenv('OMNIRETARGETING_WINDOW_CAPTURE', str(tmp_path))
    path = capture(model, P, c, mandatory)
    with open(path, 'rb') as stream:
        saved = pickle.load(stream)
    frozen = saved['model']
    assert frozen.coords is None and not frozen.window_soft_sources
    assert not frozen.program.hard_blocks
    for name in ('step_lower', 'step_upper', 'origin', 'gradient', 'diagonal'):
        np.testing.assert_array_equal(getattr(frozen, name), getattr(model, name))
    for name in ('P', 'c', 'mandatory'):
        actual, expected = saved[name], {'P': P, 'c': c, 'mandatory': mandatory}[name]
        if name == 'P':
            actual, expected = actual.toarray(), expected.toarray()
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(frozen.program.A.toarray(), model.program.A.toarray())
    np.testing.assert_array_equal(frozen.program.b, model.program.b)
    assert frozen.program.soc == model.program.soc
    assert frozen.predicted_slacks(np.ones(model.gradient.size) * .01) == model.predicted_slacks(
        np.ones(model.gradient.size) * .01)
    assert sum(saved['sources'].values()) == len(model.program.b)
    assert not list(tmp_path.glob('*.tmp'))


def test_capture_limit_counts_distinct_windows_not_damping_attempts(tmp_path, monkeypatch):
    objective, poses = problem(8)
    model, mandatory = prepare_window_model(objective, poses, range(0, 4))
    P, c = quadratic(model, .001)
    monkeypatch.setenv('OMNIRETARGETING_WINDOW_CAPTURE', str(tmp_path))
    monkeypatch.setenv('OMNIRETARGETING_WINDOW_CAPTURE_MAX', '2')
    assert capture(model, P, c, mandatory)
    assert capture(model, 2 * P, c, mandatory) is None
    assert capture(replace(model, frames=range(2, 6)), P, c, mandatory)
    assert capture(replace(model, frames=range(4, 8)), P, c, mandatory) is None
    assert len(list(tmp_path.glob('*.pkl'))) == 2


def test_source_breakdowns_account_for_all_selected_and_violating_rows():
    objective, poses = problem(8)
    model, mandatory = prepare_window_model(objective, poses, range(0, 4))
    assert model.window_row_sources['codes'].shape == model.program.b.shape
    assert sum(source_counts(model).values()) == len(model.program.b)
    assert any(name.startswith('contact:') for name in source_counts(model))
    assert all('unknown_channel' not in name for name in source_counts(model)
               if name.startswith('contact:'))
    result, trace = solve_window_model(model, .001, mandatory)
    assert result is not None
    for record in trace:
        assert sum(record['sources'].values()) == record['rows']
        assert sum(record['original_sources'].values()) == record['original_rows']
        assert sum(record['violated_sources'].values()) >= record['violated_linear']
        assert record['active_violated_linear'] <= record['violated_linear']
