"""Borrow CPU units for independent shell frames; join before any later fork."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import copy

import numpy as np

from .window_cpu_budget import native_workers


@contextmanager
def shell_workers(bulk):
    queries = bulk.window_queries
    budget = getattr(bulk.objective, '_window_cpu_budget', None)
    from .window_query_native import WindowQueries, grid_bounds
    if (queries is None or budget is None or native_workers() == 1
            or type(queries) is not WindowQueries or 'shell' in queries.__dict__):
        yield
        return
    with budget.borrow(native_workers(), kind='deepest') as workers:
        if workers == 1:
            yield
            return
        original = queries.shell
        with ThreadPoolExecutor(max_workers=workers) as executor:
            def shell(sdf, geom, frames, arrays, spacing, value, choice):
                # All shared preparation precedes submission. Each frame's
                # seed and output belong to exactly one native slice.
                grid_bounds(sdf)
                queries.seeds.setdefault((sdf, geom), np.full(queries.objective.frames, -1, np.int64))
                count = min(workers, max(1, len(frames)//8))
                if count == 1:
                    return original(sdf, geom, frames, arrays, spacing, value, choice)
                edges = np.linspace(0, len(frames), count+1).astype(int)

                def run(bounds):
                    local = copy(queries)
                    local.diagnostics = {}
                    part = slice(*bounds)
                    local_arrays = (frames[part], *arrays[1:])
                    WindowQueries.shell(local, sdf, geom, frames[part], local_arrays,
                                        spacing, value[part], choice[part])
                    return local.diagnostics

                for counters in executor.map(run, zip(edges[:-1], edges[1:])):
                    # Integer diagnostics also retain deterministic order.
                    for key, number in counters.items():
                        queries.diagnostics[key] = queries.diagnostics.get(key, 0)+number

            queries.shell = shell
            try:
                yield
            finally:
                # Restore the inherited class method, avoiding a bound-method
                # cycle; executor shutdown then joins every native call.
                queries.__dict__.pop('shell', None)
