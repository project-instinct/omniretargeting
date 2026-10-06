"""Diagnostic-only solver sizes/counts around the unchanged frozen runner."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import runpy
import sys
import time
from unittest.mock import patch


def bootstrap(runner, argv):
    """Keep numerical inputs/settings untouched; instrument trace runs only."""
    import clarabel
    import scipy.sparse.linalg as linear
    source = Path(argv[argv.index('--source-root') + 1]).resolve()
    sys.path.insert(0, str(source))
    from omniretargeting._fast.runtime_options import OPTIONS
    if not OPTIONS.performance_trace:
        raise ValueError('Solver instrumentation is permitted only in trace mode')
    output = Path(argv[argv.index('--output') + 1])
    events = []
    constructor, sparse_solve = clarabel.DefaultSolver, linear.spsolve

    def stage():
        from omniretargeting._fast.performance_trace import _ACTIVE_TRACE
        trace = _ACTIVE_TRACE.get()
        return None if trace is None else trace.current_stage

    class Solver:
        def __init__(self, P, c, A, b, cones, settings):
            self.row = dict(kind='cone_program', stage=stage(), variables=len(c),
                hessian_nnz=P.nnz, constraint_rows=A.shape[0], constraint_nnz=A.nnz,
                cones=[dict(kind=type(cone).__name__, dim=getattr(cone, 'dim', None)) for cone in cones],
                updates=0, solves=[])
            events.append(self.row)
            self.native = constructor(P, c, A, b, cones, settings)

        def __getattr__(self, name):
            return getattr(self.native, name)

        def update(self, **kwargs):
            self.row['updates'] += 1
            return self.native.update(**kwargs)

        def solve(self):
            tick = time.perf_counter()
            result = self.native.solve()
            self.row['solves'].append(dict(seconds=time.perf_counter() - tick,
                status=str(result.status), iterations=int(result.iterations), stage=stage()))
            return result

    def observed(matrix, rhs, *args, **kwargs):
        tick = time.perf_counter()
        try:
            return sparse_solve(matrix, rhs, *args, **kwargs)
        finally:
            events.append(dict(kind='sparse_proposal', stage=stage(), variables=matrix.shape[0],
                               hessian_nnz=matrix.nnz, seconds=time.perf_counter() - tick))

    try:
        with ExitStack() as stack:
            stack.enter_context(patch.object(clarabel, 'DefaultSolver', Solver))
            stack.enter_context(patch.object(linear, 'spsolve', observed))
            sys.argv = [str(runner), *argv]
            runpy.run_path(str(runner), run_name='__main__')
    finally:
        output.mkdir(parents=True, exist_ok=True)
        (output / 'solver_sizes_trace.json').write_text(json.dumps(dict(
            diagnostic_only=True, solver_settings_unchanged=True, events=events,
            cone_solve_calls=sum(len(e.get('solves', [])) for e in events)), indent=2) + '\n')


def main():
    if '--bootstrap' in sys.argv:
        index = sys.argv.index('--bootstrap')
        bootstrap(Path(sys.argv[index + 1]), sys.argv[index + 2:])
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--source-root', type=Path)
    parser.add_argument('--arm', choices=list('ABCD'), required=True)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--clips', default='')
    args = parser.parse_args()
    runner = args.run_dir.resolve() / 'benchmark_runner.py'
    module = runpy.run_path(str(runner))
    args.trace, args.enable = True, None
    call = module['subprocess'].call

    def instrumented(command, *rest, **kwargs):
        # Instrument production child only, leave independent audit commands.
        if '--child' in command and str(runner) in command:
            index = command.index(str(runner))
            command[index:index + 1] = [str(Path(__file__).resolve()), '--bootstrap', str(runner)]
        return call(command, *rest, **kwargs)

    with patch.object(module['subprocess'], 'call', instrumented):
        module['run'](args)


if __name__ == '__main__':
    main()
