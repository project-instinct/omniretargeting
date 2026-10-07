"""Development-only local repair ablations; all patches are scoped to a call.

Keep elastic chatter cleanup and production acceptance checks. Do not import
this module from the production pipeline. max120 patches the call default,
since changing WINDOW_MAX_FRAMES alone does not change a bound Python default.
"""
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
from scipy import sparse

from . import local_chatter, local_repair, local_rounds, local_slip, local_windows, mesh_repair
from .. import moving_sample_summary


def _repair(objective, poses, scene_objects, scene_query, *, no_chatter=False,
            no_slip=False, cap=None, max_frames=None):
    original_propose = local_rounds.propose
    original_rows = local_chatter.JointChatter.rows
    original_windows = local_windows.penetration_windows

    def empty_hard_rows(self, current, frames, columns, *, elastic=False):
        if elastic:
            return original_rows(self, current, frames, columns, elastic=True)
        return sparse.csr_matrix((0, self.frames * self.width)), np.empty(0), np.empty(0)

    def propose_without_penetration_chatter(ctx, run, state):
        if run.window['kind'] != 'penetration':
            return original_propose(ctx, run, state)
        # This also omits penetration curvature-correction hard chatter rows;
        # chatter-window proposals and their corrections retain all rows.
        with patch.object(local_chatter.JointChatter, 'rows', empty_hard_rows):
            return original_propose(ctx, run, state)

    def shorter_windows(obj, limits, measured, activation, longest=None):
        return original_windows(obj, limits, measured, activation,
                                longest=max_frames if longest is None else longest)

    with ExitStack() as stack:
        if no_chatter:
            stack.enter_context(patch.object(local_rounds, 'propose', propose_without_penetration_chatter))
        if no_slip:
            stack.enter_context(patch.object(local_slip, 'slip_term', return_value=None))
        if cap is not None:
            stack.enter_context(patch.object(mesh_repair, 'WITNESS_BAND_CAP', cap))
            stack.enter_context(patch.object(moving_sample_summary, 'WITNESS_BAND_CAP', cap))
        if max_frames is not None:
            stack.enter_context(patch.object(local_windows, 'WINDOW_MAX_FRAMES', max_frames))
            stack.enter_context(patch.object(local_windows, 'penetration_windows', shorter_windows))
        motion, report = local_repair.repair_local(objective, poses, scene_objects, scene_query)
    report['ablation'] = dict(no_chatter_rows=no_chatter, no_slip=no_slip,
                              witness_band_cap=cap, window_max_frames=max_frames)
    return motion, report


def repair_no_chatter_rows(objective, poses, scene_objects=(), scene_query=None):
    return _repair(objective, poses, scene_objects, scene_query, no_chatter=True)


def repair_no_slip(objective, poses, scene_objects=(), scene_query=None):
    return _repair(objective, poses, scene_objects, scene_query, no_slip=True)


def repair_cap24(objective, poses, scene_objects=(), scene_query=None):
    return _repair(objective, poses, scene_objects, scene_query, cap=24)


def repair_max120(objective, poses, scene_objects=(), scene_query=None):
    return _repair(objective, poses, scene_objects, scene_query, max_frames=120)


def repair_no_chatter_rows_no_slip(objective, poses, scene_objects=(), scene_query=None):
    return _repair(objective, poses, scene_objects, scene_query, no_chatter=True, no_slip=True)


def repair_baseline_profiled(objective, poses, scene_objects=(), scene_query=None):
    """Measure baseline subsolves, including working-set restarts, without edits.

    Per-process JSONL preserves measurements from forked proposal workers.
    This supplementary run must be checked against the direct abA output.
    """
    import json
    import os
    import time
    from pathlib import Path
    import clarabel

    trace_dir = Path(os.environ['ULR_TRACE_DIR'])
    trace_dir.mkdir(parents=True, exist_ok=True)
    original_step = mesh_repair.solve_step
    original_solver = clarabel.DefaultSolver

    def timed_step(limits, current, geometry, measured, baseline, projection, **kwargs):
        started = time.monotonic()
        constraint_seconds = 0.
        subsolves = []
        original_constraints = limits.constraints

        def constraints(*args, **kw):
            nonlocal constraint_seconds
            tick = time.monotonic()
            result = original_constraints(*args, **kw)
            constraint_seconds += time.monotonic() - tick
            return result

        class Solver:
            def __init__(self, quadratic, gradient, matrix, rhs, cones, settings):
                self.tick = time.monotonic()
                self.solver = original_solver(quadratic, gradient, matrix, rhs, cones, settings)
                self.info = dict(variables=quadratic.shape[0], rows=matrix.shape[0], cones=len(cones),
                                 P_nnz=quadratic.nnz, A_nnz=matrix.nnz)

            def solve(self):
                solution = self.solver.solve()
                self.info.update(iterations=solution.iterations, status=str(solution.status),
                                 solve_seconds=time.monotonic() - self.tick)
                subsolves.append(self.info)
                return solution

        with patch.object(limits, 'constraints', constraints), patch.object(clarabel, 'DefaultSolver', Solver):
            result = original_step(limits, current, geometry, measured, baseline, projection, **kwargs)
        total = time.monotonic() - started
        record = dict(pid=os.getpid(), joint_variables=projection.shape[1], status=result[1],
                      total_seconds=total, constraint_seconds=constraint_seconds,
                      assembly_seconds=total - constraint_seconds - sum(s['solve_seconds'] for s in subsolves),
                      subsolves=subsolves)
        with (trace_dir / f'{os.getpid()}.jsonl').open('a') as handle:
            handle.write(json.dumps(record) + '\n')
        return result

    with patch.object(mesh_repair, 'solve_step', timed_step):
        motion, report = mesh_repair.repair_original_meshes(objective, poses, scene_objects, scene_query)
    report['qp_profile'] = [json.loads(line) for path in sorted(trace_dir.glob('*.jsonl'))
                            for line in path.read_text().splitlines()]
    report['ablation'] = dict(baseline_profiled=True)
    return motion, report


def repair_first_qp_probe(objective, poses, scene_objects=(), scene_query=None):
    """Solve one identical penetration linearization with four coupling choices.

    Development-only structural probe: returns the entry motion unchanged;
    the report describes QPs, not a repaired trajectory or quality result.
    ULR_PROBE_MIN_FRAMES chooses the costly window (180 sit / 162 omomo).
    """
    import os
    import clarabel
    from . import local_qp

    original_solve = local_qp.solve
    original_solver = clarabel.DefaultSolver
    minimum = int(os.environ['ULR_PROBE_MIN_FRAMES'])
    report = dict(method='identical_window_qp_probe', probe=[], input_unchanged=True)

    class ProbeDone(Exception):
        pass

    def solve_four(*args, **kwargs):
        for label, chatter_enabled, slip_enabled in [
                ('abN', True, True), ('no_chatter_rows', False, True),
                ('no_slip', True, False), ('no_chatter_rows_no_slip', False, False)]:
            options = dict(kwargs)
            if not chatter_enabled:
                options['linear'] = []
            if not slip_enabled:
                options['quadratic'] = None
            structure = {}

            class Solver:
                def __init__(self, quadratic, gradient, matrix, rhs, cones, settings):
                    structure.update(P_nnz=quadratic.nnz, A_nnz=matrix.nnz)
                    self.solver = original_solver(quadratic, gradient, matrix, rhs, cones, settings)

                def solve(self):
                    result = self.solver.solve()
                    linear_info = self.solver.get_info().linsolver
                    structure.update(kkt_nnzA=linear_info.nnzA, factor_nnzL=linear_info.nnzL,
                                     linear_solver=linear_info.name)
                    return result

            with patch.object(clarabel, 'DefaultSolver', Solver):
                _, status, info = original_solve(*args, **options)
            report['probe'].append(dict(variant=label, status=status, **structure,
                                        **{k:v for k,v in info.items() if k not in ['predicted_depths','slack']}))
        raise ProbeDone

    def first_selected_wave(ctx, windows, state, **kwargs):
        candidates = [w for w in windows if w['kind']=='penetration' and int(w['mask'].sum()) >= minimum]
        if not candidates:
            return
        window = max(candidates, key=lambda w:int(w['mask'].sum()))
        report['window'] = dict(frames=int(window['mask'].sum()), columns=len(window['columns']),
                                bodies=[objective.model.body(b).name for b in window['bodies']])
        report['before'] = report['after'] = ctx.geometry.summary(state.measured)
        ctx.limits.grant(windows)
        run = local_rounds.WindowRun(ctx, window)
        local_rounds.propose(ctx, run, state)

    with patch.object(local_rounds, 'run_wave', first_selected_wave), patch.object(local_qp, 'solve', solve_four):
        try:
            local_repair.repair_local(objective, poses, scene_objects, scene_query)
        except ProbeDone:
            pass
    if len(report['probe']) != 4:
        raise RuntimeError(f'No matching penetration window: minimum={minimum}')
    return np.array(poses, copy=True), report
