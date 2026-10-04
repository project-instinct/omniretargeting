"""Exact removal/lift of unused epigraphs in a screened proposal QP."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np


def solve_subset(program, P, c, solve, *, precision=False):
    """Eliminate s only when its entire subproblem is min c*s, s>=0.

    Check all P row/column and A column/row uses. Reconstruct s=0 and the
    omitted nonnegativity dual c, preserving primal/dual KKT algebra. Shared
    slacks with even one retained witness can never be eliminated here.
    """
    matrix = program.A
    columns = np.arange(program.step_size, matrix.shape[1])
    columns = columns[np.diff(matrix.indptr)[columns] == 1]
    offsets = matrix.indptr[columns]
    rows, values = matrix.indices[offsets], matrix.data[offsets]
    eligible = ((rows < program.nonneg) & (values == -1.) & (program.b[rows] == 0.)
                & np.isfinite(c[columns]) & (c[columns] >= 0.)
                & (np.asarray(P.getnnz(axis=0))[columns] == 0)
                & (np.asarray(P.getnnz(axis=1))[columns] == 0)
                & (np.asarray(matrix.getnnz(axis=1))[rows] == 1))
    columns, rows = columns[eligible], rows[eligible]
    if not len(columns):
        result = solve(program, P, c, precision=precision)
        program.last_solve.update(eliminated_free_epigraphs=0, solver_variables=matrix.shape[1],
                                  solver_rows=len(program.b))
        return result
    keep_columns, keep_rows = np.ones(matrix.shape[1], bool), np.ones(len(program.b), bool)
    keep_columns[columns], keep_rows[rows] = False, False
    reduced = replace(program, A=matrix[keep_rows][:, keep_columns].tocsc(), b=program.b[keep_rows],
                      nonneg=program.nonneg - len(rows), hard_blocks=(), soft=None, last_solve={})
    result = solve(reduced, P[keep_columns][:, keep_columns].tocsc(), c[keep_columns], precision=precision)
    program.last_solve = dict(reduced.last_solve, eliminated_free_epigraphs=len(columns),
                              solver_variables=int(keep_columns.sum()), solver_rows=int(keep_rows.sum()))
    if result is None:
        return None
    x, z = np.zeros(matrix.shape[1]), np.zeros(len(program.b))
    x[keep_columns], z[keep_rows], z[rows] = result.x, result.z, c[columns]
    return SimpleNamespace(x=x, z=z)
