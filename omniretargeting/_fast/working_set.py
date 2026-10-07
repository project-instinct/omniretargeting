"""Exact constraint generation for Clarabel standard-form programs.

Repair QPs and restorations carry tens of thousands of rows and contact cones,
of which well under one percent bind at the optimum. A program is solved on a
working set instead: zero cones always, rows and cones near their bound at the
zero step, plus every omitted one whose slack at a solved point falls below
``MARGIN``. When no omitted constraint is that close, the solved point is
feasible for the complete program with all omitted multipliers zero, so for a
strictly convex objective it is the complete optimum (the same mathematical
answer, found on a much smaller interior-point system).

With an optional bound ``radius`` on the solution norm (a restoration's trust
cone), constraints that cannot come within ``MARGIN`` of their bound anywhere
in that ball are dropped before the first solve (safe screening).

A certified infeasible working set proves the complete program infeasible
(it is a relaxation), so that status is returned without a complete solve.
Any other outcome returns None and the caller solves the complete program.
Set ``OMNIRETARGETING_WORKING_SET_STATS=1`` to print outcome counts at exit.
"""

import atexit
from collections import Counter
import os
import re
from types import SimpleNamespace

import numpy as np
from scipy import sparse

SLACK = 1e-3
MARGIN = 1e-6
ROUNDS = 16
# Beyond this share of all rows the complete program is solved instead.
FRACTION = 0.5
# Cones larger than this (trust cones over every coordinate) are always kept.
LARGE_CONE = 16
ACCEPTED = ("Solved", "AlmostSolved")
_CONE = re.compile(r"(\w+)\((\d+)\)")
STATS = Counter()
if os.environ.get("OMNIRETARGETING_WORKING_SET_STATS"):
    atexit.register(lambda: print("WORKING_SET_STATS", os.getpid(), dict(STATS), flush=True))


class _Layout:
    """Rows of every nonnegative row and second-order cone, vectorized by size."""

    def __init__(self, cones, matrix, radius):
        kinds, sizes = [], []
        for cone in cones:
            kind, size = _CONE.match(repr(cone)).groups()
            kinds.append(kind)
            sizes.append(int(size))
        self.supported = all(k in ("ZeroConeT", "NonnegativeConeT", "SecondOrderConeT") for k in kinds)
        sizes = np.asarray(sizes, dtype=int)
        starts = np.r_[0, np.cumsum(sizes)[:-1]].astype(int)
        kinds = np.asarray(kinds)
        self.fixed = [np.arange(s, s + n) for k, s, n in zip(kinds, starts, sizes)
                      if k == "ZeroConeT" or (k == "SecondOrderConeT" and n > LARGE_CONE)]
        self.fixed_kinds = [(k, n) for k, n in zip(kinds, sizes)
                            if k == "ZeroConeT" or (k == "SecondOrderConeT" and n > LARGE_CONE)]
        self.rows = np.concatenate([np.arange(s, s + n) for k, s, n in zip(kinds, starts, sizes)
                                    if k == "NonnegativeConeT"] or [np.zeros(0, int)]).astype(int)
        small = (kinds == "SecondOrderConeT") & (sizes <= LARGE_CONE)
        self.cone_starts, self.cone_sizes = starts[small], sizes[small]
        self.reach_rows = self.reach_cones = None
        if radius is not None and np.isfinite(radius):
            norm = np.sqrt(np.asarray(sparse.csr_matrix(matrix).multiply(matrix).sum(axis=1)).ravel())
            # A row moves by at most ||a_i|| radius; a cone's margin
            # s0 - ||s_tail|| by at most the sum of its row norms times radius.
            self.reach_rows = norm[self.rows] * radius
            total = np.r_[0., np.cumsum(norm)]
            self.reach_cones = (total[self.cone_starts + self.cone_sizes]
                                - total[self.cone_starts]) * radius

    def margins(self, slack):
        cones = np.empty(len(self.cone_starts))
        for size in np.unique(self.cone_sizes):
            which = np.flatnonzero(self.cone_sizes == size)
            block = slack[self.cone_starts[which, None] + np.arange(size)]
            cones[which] = block[:, 0] - np.linalg.norm(block[:, 1:], axis=1)
        return slack[self.rows], cones


def solve(quadratic, linear, matrix, offset, cones, settings, *, radius=None, diagnostics=None):
    """Solution namespace (x, status, iterations) of the complete program, or None."""
    import clarabel

    matrix = sparse.csr_matrix(matrix)
    offset = np.asarray(offset, dtype=float)
    layout = _Layout(cones, matrix, radius)
    if not layout.supported:
        return None
    rows_entry, cones_entry = layout.margins(offset)
    reach_rows = np.ones(len(rows_entry), bool) if layout.reach_rows is None else (
        rows_entry - layout.reach_rows <= MARGIN)
    reach_cones = np.ones(len(cones_entry), bool) if layout.reach_cones is None else (
        cones_entry - layout.reach_cones <= MARGIN)
    keep_rows = reach_rows & (rows_entry <= SLACK)
    keep_cones = reach_cones & (cones_entry <= SLACK)
    total, iterations = matrix.shape[0], 0
    for rounds in range(1, ROUNDS + 1):
        zero = [rows for rows, (kind, _) in zip(layout.fixed, layout.fixed_kinds) if kind == "ZeroConeT"]
        large = [rows for rows, (kind, _) in zip(layout.fixed, layout.fixed_kinds) if kind != "ZeroConeT"]
        selected = np.flatnonzero(keep_cones)
        blocks = zero + ([layout.rows[keep_rows]] if keep_rows.any() else []) + large + [
            layout.cone_starts[c] + np.arange(layout.cone_sizes[c]) for c in selected]
        reduced = ([clarabel.ZeroConeT(len(r)) for r in zero]
                   + ([clarabel.NonnegativeConeT(int(keep_rows.sum()))] if keep_rows.any() else [])
                   + [clarabel.SecondOrderConeT(len(r)) for r in large]
                   + [clarabel.SecondOrderConeT(int(layout.cone_sizes[c])) for c in selected])
        rows = np.concatenate(blocks) if blocks else np.zeros(0, dtype=int)
        if len(rows) > FRACTION * total:
            STATS["fallback_rows"] += 1
            break
        reduced_matrix = matrix[rows].tocsc()
        solution = None
        if solution is None:
            solution = clarabel.DefaultSolver(quadratic, linear, reduced_matrix, offset[rows],
                                              reduced, settings()).solve()
        iterations += solution.iterations
        status = str(solution.status)
        if status == "PrimalInfeasible":
            STATS["infeasible"] += 1
            return SimpleNamespace(x=None, status=status, iterations=iterations,
                                   rounds=rounds, rows=len(rows))
        if status not in ACCEPTED or not np.isfinite(solution.x).all():
            STATS[f"fallback_{status}"] += 1
            break
        point = np.asarray(solution.x, dtype=float)
        rows_now, cones_now = layout.margins(offset - matrix @ point)
        add_rows = ~keep_rows & reach_rows & (rows_now < MARGIN)
        add_cones = ~keep_cones & reach_cones & (cones_now < MARGIN)
        if not add_rows.any() and not add_cones.any():
            if diagnostics is not None:
                diagnostics.update(working_set_rounds=rounds, working_set_rows=int(len(rows)),
                                   complete_rows=int(total), iterations=iterations)
            STATS["solved"] += 1
            STATS[f"rounds_{rounds}"] += 1
            return SimpleNamespace(x=point, status=status, iterations=iterations,
                                   rounds=rounds, rows=len(rows))
        keep_rows |= add_rows
        keep_cones |= add_cones
    else:
        STATS["fallback_rounds"] += 1
    if diagnostics is not None:
        diagnostics.update(working_set_fallback=True)
    return None
