"""Sparse acquisition models and public, pickleable priority records."""

from dataclasses import dataclass, field
import time

import numpy as np
from scipy import sparse


LEVELS = ('penetration', 'support', 'control', 'contact')
TRACKING_LEVEL = len(LEVELS)


def priority_level(key):
    if not isinstance(key, tuple) or not key:
        raise ValueError('Soft keys must be nonempty stable tuples')
    if key[0] == 'penetration':
        return 0
    if key[0] == 'clearance_margin':
        return TRACKING_LEVEL
    if key[0] != 'contact' or len(key) < 2:
        raise ValueError(f'Unknown priority key: {key!r}')
    ranks = {'support': 1, 'hold': 1, 'uncertain': 1, 'control': 2,
             'follow': 3, 'touch': 3, 'other': 3}
    if key[1] not in ranks:
        raise ValueError(f'Unknown contact rank: {key[1]!r}')
    return ranks[key[1]]


@dataclass(frozen=True)
class PriorityMerit:
    valid: bool
    levels: tuple
    tolerances: tuple
    slacks: dict
    details: dict = field(default_factory=dict)

    @classmethod
    def coerce(cls, value):
        result = cls(**value) if isinstance(value, dict) else value
        if len(result.levels) != 4 or len(result.tolerances) != 4:
            raise ValueError('Priority merit requires four ordered levels/tolerances')
        if (not np.isfinite(result.levels).all() or not np.isfinite(result.tolerances).all()
                or np.any(np.asarray(result.tolerances) < 0)
                or any(not np.isfinite(v) or v < 0 for v in result.slacks.values())):
            raise ValueError('Priority merit contains invalid values')
        return result


@dataclass
class QPSolution:
    step: np.ndarray
    slacks: dict = field(default_factory=dict)
    certificate: dict = field(default_factory=dict)
    primal: np.ndarray | None = None
    dual: np.ndarray | None = None


@dataclass
class IterationModel:
    origin: np.ndarray
    coords: object
    step_lower: np.ndarray
    step_upper: np.ndarray
    system: np.ndarray
    hessian: sparse.csc_matrix
    gradient: np.ndarray
    diagonal: np.ndarray
    scale: float
    program: object = None
    soft: object = None
    priority_cache: object = None
    solve_diagnostics: list = field(default_factory=list)
    frames: range | None = None
    window_soft_sources: tuple = ()
    window_row_sources: object = None
    window_working_set: object = None

    def predicted_slacks(self, delta):
        return {} if self.soft is None else self.soft.values(np.asarray(delta).ravel())

    def predicted_reductions(self, delta):
        before = self.predicted_slacks(np.zeros(self.gradient.size))
        after = self.predicted_slacks(delta)
        return {key: before[key] - after[key] for key in before}

    def tracking_decrease(self, delta):
        delta = np.asarray(delta).ravel()
        return float(-2 * self.gradient @ delta - delta @ (self.hessian @ delta))


@dataclass
class ConeProgram:
    """Ax+s=b, s in nonnegative orthant followed by complete second-order cones."""
    A: sparse.csc_matrix
    b: np.ndarray
    nonneg: int
    soc: tuple
    step_size: int
    hard_blocks: tuple = ()
    soft: object = None
    last_solve: dict = field(default_factory=dict)

    def kinds(self):
        import clarabel
        return ([clarabel.NonnegativeConeT(self.nonneg)] if self.nonneg else []) + [
            clarabel.SecondOrderConeT(int(size)) for size in self.soc]


def settings(*, retry=False, precision=False):
    import clarabel
    from ..conic_solver import RETRY_TOLERANCE
    result = clarabel.DefaultSettings()
    result.verbose = False
    result.direct_solve_method = 'qdldl'
    if retry:
        result.tol_gap_abs = result.tol_gap_rel = result.tol_feas = RETRY_TOLERANCE
    if precision:
        # A strict priority certificate can need more accuracy than Clarabel's
        # normalized stopping test. Refine the same mathematical program.
        result.tol_gap_abs = result.tol_gap_rel = result.tol_feas = 1e-10
        result.static_regularization_constant = 1e-10
    return result


def solve_program(program, P, c, *, precision=False):
    """Same Clarabel settings/status policy and numerical-stall retry as solve_conic."""
    import clarabel
    attempts = []
    for retry in ((False,) if precision else (False, True)):
        tick = time.monotonic()
        upper, vector, cones = sparse.triu(P).tocsc(), np.asarray(c, float), program.kinds()
        options = settings(retry=retry, precision=precision)
        prepare_seconds = time.monotonic() - tick
        tick = time.monotonic()
        solver = clarabel.DefaultSolver(upper, vector, program.A, program.b, cones, options)
        construct_seconds = time.monotonic() - tick
        tick = time.monotonic()
        result = solver.solve()
        solve_seconds = time.monotonic() - tick
        status = str(result.status)
        attempts.append(dict(retry=retry, status=status, iterations=result.iterations,
                             prepare_seconds=prepare_seconds, construct_seconds=construct_seconds,
                             solve_seconds=solve_seconds,
                             reported_solver_seconds=getattr(result, 'solve_time', None)))
        program.last_solve = dict(status=status, retry=retry, precision=precision, iterations=result.iterations,
                                 primal_residual=float(result.r_prim), dual_residual=float(result.r_dual),
                                 primal_cost=float(result.obj_val), dual_cost=float(result.obj_val_dual),
                                 solver_attempts=list(attempts))
        if status in {'Solved', 'AlmostSolved'}:
            return result if np.isfinite(result.x).all() else None
        if status in {'PrimalInfeasible', 'DualInfeasible'}:
            return None
    return None


def sparse_hessian(system):
    count = system.shape[1]
    bands = [system[k, :count - k] for k in range(len(system))]
    lower = sparse.diags(bands, -np.arange(len(bands)), shape=(count, count), format='csc')
    return (lower + lower.T - sparse.diags(system[0])).tocsc()
