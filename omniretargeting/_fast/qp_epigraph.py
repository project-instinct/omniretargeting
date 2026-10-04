"""Exact elimination of unused, independently unbounded soft epigraphs.

This is a solver transformation, never an optimality or infeasibility claim.
Lifted iterates must be certified against the complete original program.
"""

from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np
from scipy import sparse


@dataclass
class EpigraphReduction:
    original: object
    program: object
    columns: np.ndarray
    rows: np.ndarray
    removed: list

    def lift(self, result):
        if result is None:
            return None
        full = self.original
        x = np.zeros(full.A.shape[1])
        x[self.columns] = result.x
        residual = full.b - full.A @ x
        for column, entries in self.removed:
            value = 0.
            for row, size in entries:
                need = (-residual[row] if size == 1 else
                        np.linalg.norm(residual[row + 1:row + size]) - residual[row])
                value = max(value, float(need))
            x[column] = np.nextafter(value, np.inf)
        z = np.zeros(len(full.b))
        z[self.rows] = result.z
        s = full.b - full.A @ x
        s[self.rows] = result.s
        return SimpleNamespace(x=x, z=z, s=s)


def eliminate_unused_epigraphs(program, P, c):
    """Return a reversible reduction, or None when nothing is provably free.

    Check every column use: zero objective; own scalar/cone head exactly -1;
    own nonnegative row only; no hard/freeze incidence or coupled epigraphs.
    A cone is removed whole. No approximately active constraints are removed.
    """
    soft = program.soft
    if soft is None or not soft.keys:
        return None
    A, rows = program.A.tocsc(), program.A.tocsr()
    P = sparse.csc_matrix(P)
    quadratic = np.asarray(abs(P).sum(axis=0)).ravel() + np.asarray(abs(P).sum(axis=1)).ravel()
    head_keys = {int(head): int(key) for heads, keys in soft.heads for head, key in zip(heads, keys)}
    cone_size = {}
    position = program.nonneg
    for size in program.soc:
        cone_size[position] = size
        position += size
    removed, delete = [], np.zeros(A.shape[0], bool)
    for key in range(len(soft.keys)):
        column = program.step_size + key
        if c[column] != 0. or quadratic[column] != 0.:
            continue
        start, stop = A.indptr[column:column + 2]
        indices, values = A.indices[start:stop], A.data[start:stop]
        entries, valid = [], bool(len(indices))
        for row, value in zip(indices, values):
            if value == 0.:
                continue
            row = int(row)
            own_bound = (row < program.nonneg and rows.indptr[row + 1] - rows.indptr[row] == 1
                         and program.b[row] == 0.)
            if value != -1. or (not own_bound and head_keys.get(row) != key):
                valid = False
                break
            size = 1 if row < program.nonneg else cone_size.get(row)
            if size is None:
                valid = False
                break
            extra = rows[row:row + size, program.step_size:]
            if np.any(extra.indices != key):
                valid = False
                break
            entries.append((row, size))
        if valid and entries:
            removed.append((column, entries))
            for row, size in entries:
                delete[row:row + size] = True
    if not removed:
        return None
    keep_rows = np.flatnonzero(~delete)
    keep_columns = np.ones(A.shape[1], bool)
    keep_columns[[column for column, _ in removed]] = False
    keep_columns = np.flatnonzero(keep_columns)
    reduced = replace(program, A=A[keep_rows][:, keep_columns].tocsc(), b=program.b[keep_rows],
                      nonneg=int(np.count_nonzero(~delete[:program.nonneg])),
                      soc=tuple(size for head, size in cone_size.items() if not delete[head]),
                      hard_blocks=(), soft=None, last_solve={})
    return EpigraphReduction(program, reduced, keep_columns, keep_rows, removed)
