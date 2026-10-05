"""Strict status policy, unchanged canonical programs, and CUDA worker isolation."""
from dataclasses import replace
from types import SimpleNamespace
import sys

import clarabel
import cvxpy as cp
import numpy as np
import pytest
from scipy import sparse

from omniretargeting._fast import gpu_conic as gpu
from omniretargeting import gpu_conic_runtime as runtime
from omniretargeting.conic_solver import solve_conic


@pytest.fixture
def enabled(monkeypatch):
    from omniretargeting._fast import runtime_options
    options = replace(gpu.OPTIONS, gpu_conic=True)
    monkeypatch.setattr(gpu, 'OPTIONS', options)
    monkeypatch.setattr(runtime_options, 'OPTIONS', options)
    monkeypatch.setattr(gpu, 'MIN_VARIABLES', 1)
    monkeypatch.setattr(gpu, 'MIN_ROWS', 1)
    monkeypatch.setattr(gpu, 'MIN_NNZ', 0)
    monkeypatch.setattr(gpu, '_UNAVAILABLE', None)
    monkeypatch.setattr(runtime, '_ERROR', None)
    monkeypatch.setattr(runtime, 'device_error', lambda: None)
    gpu.reset_report()


def _program():
    # Equality, nonnegative, SOC; both equality duals and SOC signs matter.
    x = cp.Variable(3)
    problem = cp.Problem(cp.Minimize(cp.sum_squares(x-[.5, .1, -.2])),
                        [x[1] == .1, x >= -.3, cp.norm(x) <= .4])
    return problem, x


def test_cuda_api_canonical_mapping_matches_clarabel(enabled, monkeypatch):
    qoco = pytest.importorskip('qoco', reason='optional native canonical mapping check')
    original = qoco.QOCO
    received = []

    def builtin(**kwargs):
        received.append(kwargs)
        return original(algebra='builtin')

    monkeypatch.setattr(qoco, 'QOCO', builtin)
    problem, x = _program()
    problem.solve(solver=cp.CLARABEL)
    expected, objective = x.value.copy(), problem.value
    solve_conic(problem, solver=cp.CLARABEL)
    assert received == [dict(algebra='cuda')]
    # A 1e-8 objective gap bounds position error by its square root.
    assert np.linalg.norm(x.value-expected) <= np.sqrt(2e-8)
    assert problem.value == pytest.approx(objective, abs=1e-8)
    assert gpu.report()['counts']['solved'] == 1
    for constraint in problem.constraints:
        assert np.max(constraint.violation()) < 1e-7


@pytest.mark.parametrize('status', ['QOCO_SOLVED_INACCURATE', 'QOCO_NUMERICAL_ERROR', 'QOCO_MAX_ITER'])
def test_non_solved_cuda_status_falls_back_to_original_cpu(enabled, monkeypatch, status):
    class Fake:
        def __init__(self, **kwargs): pass
        def setup(self, *args, **kwargs):
            assert kwargs['abstol'] == kwargs['reltol'] == 1e-8
        def solve(self):
            return SimpleNamespace(status=status, iters=1, pres=0., dres=0., gap=0.)

    monkeypatch.setitem(sys.modules, 'qoco', SimpleNamespace(QOCO=Fake))
    problem, x = _program()
    solve_conic(problem, solver=cp.CLARABEL)
    assert problem.status == cp.OPTIMAL
    assert problem.solver_stats.solver_name == 'CLARABEL'
    assert gpu.report()['counts'][f'fallback_{status}'] == 1


def test_worker_and_small_program_never_initialize_cuda(enabled, monkeypatch):
    class Forbidden:
        def __init__(self, **kwargs):
            raise AssertionError('CUDA must not initialize')

    monkeypatch.setitem(sys.modules, 'qoco', SimpleNamespace(QOCO=Forbidden))
    problem, _ = _program()
    monkeypatch.setattr(runtime, '_OWNER_PID', -1)
    solve_conic(problem, solver=cp.CLARABEL)
    assert problem.status == cp.OPTIMAL
    assert gpu.report()['counts']['worker_cpu'] == 1
    monkeypatch.setattr(runtime, '_OWNER_PID', runtime.os.getpid())
    monkeypatch.setattr(gpu, 'MIN_VARIABLES', 100000)
    solve_conic(problem, solver=cp.CLARABEL)
    assert gpu.report()['counts']['small_cpu'] == 1


def test_missing_cuda_is_reported_and_cpu_remains_available(enabled, monkeypatch):
    class Missing:
        def __init__(self, **kwargs):
            raise RuntimeError('Algebra cuda not available')

    monkeypatch.setitem(sys.modules, 'qoco', SimpleNamespace(QOCO=Missing))
    problem, _ = _program()
    solve_conic(problem, solver=cp.CLARABEL)
    solve_conic(problem, solver=cp.CLARABEL)
    assert problem.status == cp.OPTIMAL
    assert 'not available' in gpu.report()['unavailable']
    assert gpu.report()['counts']['unavailable_cpu'] == 1


def test_csc_conversion_does_not_mutate_original_indices():
    matrix = sparse.eye(3, format='csc')
    matrix.indices = matrix.indices.astype(np.int64)
    matrix.indptr = matrix.indptr.astype(np.int64)
    converted = gpu._csc(matrix)
    assert converted.indices.dtype == converted.indptr.dtype == np.int32
    assert matrix.indices.dtype == matrix.indptr.dtype == np.int64


def test_runtime_preserves_original_wrapper_and_restarts_once(tmp_path, monkeypatch):
    cuda = tmp_path / 'nvidia/cu13/lib'
    cuda.mkdir(parents=True)
    (cuda / 'libcudss.so.0').touch()
    monkeypatch.setenv('OMNIRETARGETING_FAST_ENABLE', 'gpu_conic')
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path/'cache'))
    monkeypatch.delenv(runtime._MARKER, raising=False)
    monkeypatch.delenv('LD_LIBRARY_PATH', raising=False)
    monkeypatch.setattr(runtime.sysconfig, 'get_paths', lambda: {'purelib': str(tmp_path)})
    monkeypatch.setattr(runtime.sys, 'orig_argv', ['python', 'wrapper.py', '--tag', 'original'])
    monkeypatch.setattr(runtime.sys, 'argv', ['cli.py', 'changed'])
    calls = []
    monkeypatch.setattr(runtime.os, 'execve', lambda *args: calls.append(args))
    runtime.prepare()
    _, args, environment = calls[0]
    assert args[1:] == ['wrapper.py', '--tag', 'original']
    assert environment[runtime._MARKER] == '1'
    assert (tmp_path/'cache/omniretargeting/qoco_cuda/lib/libcudss.so.1').resolve() == cuda/'libcudss.so.0'
    monkeypatch.setenv(runtime._MARKER, '1')
    monkeypatch.setattr(runtime, '_ERROR', None)
    runtime.prepare()
    assert len(calls) == 1
    assert runtime.error() == 'loader_restart_exhausted'


def test_large_soc_stays_on_cpu(enabled, monkeypatch):
    class Forbidden:
        def __init__(self, **kwargs):
            raise AssertionError('Large dense SOC must stay on CPU')
    monkeypatch.setitem(sys.modules, 'qoco', SimpleNamespace(QOCO=Forbidden))
    x = cp.Variable(30)
    problem = cp.Problem(cp.Minimize(cp.sum_squares(x-1)), [cp.norm(x) <= 1])
    solve_conic(problem, solver=cp.CLARABEL)
    assert problem.status == cp.OPTIMAL
    assert gpu.report()['counts']['large_soc_cpu'] == 1


def test_nonfinite_solved_result_requests_cpu_fallback(enabled, monkeypatch):
    class Fake:
        def __init__(self, **kwargs): pass
        def setup(self, n, m, p, *args, **kwargs):
            self.n, self.m, self.p = n, m, p
        def solve(self):
            return SimpleNamespace(status='QOCO_SOLVED', iters=1, pres=0., dres=0., gap=0.,
                                   x=np.full(self.n,np.nan),y=np.zeros(self.p),z=np.zeros(self.m))
    monkeypatch.setitem(sys.modules, 'qoco', SimpleNamespace(QOCO=Fake))
    problem, _ = _program()
    solve_conic(problem, solver=cp.CLARABEL)
    assert problem.status == cp.OPTIMAL
    assert gpu.report()['counts']['fallback_invalid_vectors'] == 1


def test_no_visible_cuda_device_uses_cpu_without_importing_extension(enabled, monkeypatch):
    monkeypatch.setattr(runtime, 'device_error', lambda: 'cuda_devices_masked')
    class Forbidden:
        def __init__(self, **kwargs):
            raise AssertionError('No CUDA device is visible')
    monkeypatch.setitem(sys.modules,'qoco',SimpleNamespace(QOCO=Forbidden))
    problem,_=_program()
    solve_conic(problem,solver=cp.CLARABEL)
    assert problem.status==cp.OPTIMAL
    assert gpu.report()['unavailable']=='cuda_devices_masked'


def test_explicit_budget_overrides_legacy_cuda_backend(enabled, monkeypatch):
    from omniretargeting import conic_solver
    monkeypatch.setenv('OMNIRETARGETING_CONIC_BACKEND','qoco_cuda_strict')
    monkeypatch.setattr(conic_solver,'GPU_MIN_VARIABLES',1)
    problem,_=_program()
    solve_conic(problem,solver=cp.CLARABEL,max_iter=80)
    assert problem.status==cp.OPTIMAL
    assert problem.solver_stats.solver_name=='CLARABEL'
