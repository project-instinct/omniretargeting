"""Diagnostic wrappers preserve actual cones, settings, and solutions."""
import json
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from scripts.trace_structural_spline import bootstrap


def test_trace_sizes_work_inside_production_instrumentation(tmp_path, monkeypatch):
    from omniretargeting._fast import runtime_options
    monkeypatch.setattr(runtime_options, 'OPTIONS', SimpleNamespace(performance_trace=True))
    monkeypatch.setattr(sys, 'argv', sys.argv.copy())
    monkeypatch.setattr(sys, 'path', sys.path.copy())
    runner = tmp_path / 'cone.py'
    runner.write_text('''import clarabel
import numpy as np
from scipy import sparse
from omniretargeting._fast.performance_trace import PerformanceTrace
P=sparse.eye(2,format='csc'); c=np.array([-2.,0.])
A=sparse.csc_matrix([[1.,0.],[0.,0.],[0.,1.]])
b=np.array([1.,1.,0.]); cones=[clarabel.NonnegativeConeT(1),clarabel.SecondOrderConeT(2)]
settings=clarabel.DefaultSettings(); settings.verbose=False
trace=PerformanceTrace(stage='contact_acquisition')
with trace.instrument():
    solver=clarabel.DefaultSolver(P,c,A,b,cones,settings)
    result=solver.solve()
    np.testing.assert_allclose(result.x,[1.,0.],atol=1e-7)
''')
    output = tmp_path / 'results'
    bootstrap(runner, ['--source-root', str(tmp_path), '--output', str(output)])
    report = json.loads((output / 'solver_sizes_trace.json').read_text())
    assert report['cone_solve_calls'] == 1
    row = report['events'][0]
    assert row['variables'] == 2 and row['constraint_rows'] == 3
    assert row['stage'] == 'contact_acquisition'
    assert row['cones'][1] == dict(kind='SecondOrderConeT', dim=2)


def test_diagnostic_wrapper_is_forbidden_in_normal_mode(tmp_path, monkeypatch):
    from omniretargeting._fast import runtime_options
    monkeypatch.setattr(runtime_options, 'OPTIONS', SimpleNamespace(performance_trace=False))
    with pytest.raises(ValueError, match='only in trace mode'):
        bootstrap(tmp_path / 'unused.py', ['--source-root', str(tmp_path), '--output', str(tmp_path)])
