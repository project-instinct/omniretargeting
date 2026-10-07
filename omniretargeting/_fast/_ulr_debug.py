"""Temporary diagnostics for the local repair prototype (not production)."""
import json
import time

import numpy as np


def debug_contact(objective, poses, scene_objects=(), scene_query=None):
    from . import local_repair as lr
    from .local_qp import solve
    from .mesh_repair import OriginalGeometry, _activation_depth
    from .repair_geometry_setup import prepare_static_bounds, prepare_moving_reduction
    from .contact_guard import ContactGuard
    from .proposals import guards_of

    entry = np.array(poses, copy=True)
    limits = lr.ClusterLimits(objective, entry)
    geometry = OriginalGeometry(objective, scene_objects, scene_query, manifold=True)
    prepare_static_bounds(geometry, entry)
    prepare_moving_reduction(geometry, entry)
    geometry.start_workers(len(entry))
    out = {}
    try:
        measured = geometry.measure(entry, jacobians=True)
        limits.freeze_penetration_yield(measured)
        protected = dict(measured["depths"])
        found = lr.clusters(objective, limits, measured, _activation_depth())
        cluster = found[0]
        limits.release(entry, cluster["bodies"])
        projection = lr._projection(limits, cluster)
        matrix, phi, key_of_row = lr._rows(geometry, measured, cluster["mask"])
        budgets = np.array([protected.get(k, 0.) for k in key_of_row])
        keys, sigmas = lr._objective_keys(objective.model, measured, cluster["mask"], cluster["bodies"], limits)
        step, status, info = solve(limits, entry, projection, (matrix, phi, budgets, key_of_row), keys, sigmas,
                                   local_angle=lr.LOCAL_ANGLE)
        qi = limits.coords.qi
        guards = guards_of(objective)[1]
        out["guards"] = [type(g).__name__ for g in guards]
        out["status"] = status
        out["entry_check"] = limits.check(entry)[0]
        for scale in (1.0, 0.25, 1 / 16, 1 / 256, 1 / 4096):
            trial = entry.copy()
            trial[:, qi] += scale * step
            row = {"scale": scale}
            for guard in guards:
                if isinstance(guard, ContactGuard):
                    d = guard.diagnostics(trial)
                    row[type(guard).__name__] = dict(accepted=d["accepted"], max_excess=d["max_excess_m"],
                        groups=[(g["name"], g["max_excess_m"], g["violating_frames"][:5]) for g in d["groups"]
                                if g["violating_frames"]][:4])
                else:
                    row[type(guard).__name__] = bool(guard.accepts(trial))
            row["limits"] = limits.check(trial)[1].get("reason")
            out.setdefault("scales", []).append(row)
        # entry diagnostics of every contact guard
        for guard in guards:
            if isinstance(guard, ContactGuard):
                d = guard.diagnostics(entry)
                out.setdefault("entry_contact", []).append(dict(accepted=d["accepted"], max_excess=d["max_excess_m"],
                    preserve=guard.preserve_positions))
    finally:
        geometry.stop_workers()
    print("DEBUG", json.dumps(out, default=str, indent=1), flush=True)
    raise SystemExit(0)


def debug_clusters(objective, poses, scene_objects=(), scene_query=None):
    from . import local_repair as lr
    from .mesh_repair import OriginalGeometry, _activation_depth, _joint_projection
    from .repair_geometry_setup import prepare_static_bounds, prepare_moving_reduction
    entry = np.array(poses, copy=True)
    limits = lr.ClusterLimits(objective, entry)
    geometry = OriginalGeometry(objective, scene_objects, scene_query, manifold=True)
    prepare_static_bounds(geometry, entry); prepare_moving_reduction(geometry, entry)
    geometry.start_workers(len(entry))
    model = objective.model
    try:
        measured = geometry.measure(entry, jacobians=True)
        names = lambda cols: [model.joint(int(model.dof_jntid[limits.coords.di[c]])).name for c in sorted(cols)]
        per = {}
        for key, v in measured["depths"].items():
            if v < _activation_depth():
                continue
            for b in lr._key_bodies(model, key):
                per.setdefault(b, []).append((key[0], key[1], v, key))
        for b, items in per.items():
            frames = sorted({f for f, *_ in items})
            runs = []
            for f in frames:
                if runs and f == runs[-1][1] + 1: runs[-1][1] = f
                else: runs.append([f, f])
            kinds = sorted({k for _, k, _, _ in items})
            selfs = sorted({tuple(model.body(int(model.geom_bodyid[g])).name for g in key[2:]) for *_, key in items if key[1] == 'self'})
            print("BODY", model.body(b).name, "max", round(max(v for _, _, v, _ in items) * 1000, 2), "kinds", kinds,
                  "runs", runs[:12], "selfpairs", selfs[:5], flush=True)
            print("   columns", names(lr._body_columns(limits, model, measured, b)), flush=True)
    finally:
        geometry.stop_workers()
    raise SystemExit(0)


def debug_rows(objective, poses, scene_objects=(), scene_query=None):
    import inspect
    from scipy import sparse
    from . import local_repair as lr
    from .mesh_repair import OriginalGeometry, _activation_depth
    from .repair_geometry_setup import prepare_static_bounds, prepare_moving_reduction
    from .repair_projected import ProjectedBounds
    from .repair_limits import LinearBounds
    entry = np.array(poses, copy=True)
    limits = lr.ClusterLimits(objective, entry)
    geometry = OriginalGeometry(objective, scene_objects, scene_query, manifold=True)
    prepare_static_bounds(geometry, entry); prepare_moving_reduction(geometry, entry)
    geometry.start_workers(len(entry))
    try:
        measured = geometry.measure(entry, jacobians=True)
        limits.freeze_penetration_yield(measured)
        cluster = lr.clusters(objective, limits, measured, _activation_depth())[0]
        limits.release(entry, cluster["bodies"])
        projection = lr._projection(limits, cluster)
        tags_lin, tags_norm = [], []
        orig_add, orig_norm = LinearBounds.add, ProjectedBounds.norm

        def where():
            frames = inspect.stack()[2:6]
            return " < ".join(f"{f.function}:{f.lineno}" for f in frames[:3])

        def add(self, matrix, lower, upper):
            tags_lin.append((where(), len(self.matrices)))
            return orig_add(self, matrix, lower, upper)

        def norm(self, jacobian, residual, allowed):
            tags_norm.append((where(), len(self.norms)))
            return orig_norm(self, jacobian, residual, allowed)
        LinearBounds.add, ProjectedBounds.norm = add, norm
        try:
            bounds = limits.constraints(entry, projection=projection)
        finally:
            LinearBounds.add, ProjectedBounds.norm = orig_add, orig_norm
        print("TOTAL linear blocks", len(bounds.matrices), "norm blocks", len(bounds.norms), flush=True)
        for (tag, index) in tags_lin:
            m = sparse.csr_matrix(bounds.matrices[index]) @ projection
            var = int((np.diff(m.tocsr().indptr) > 0).sum())
            print("LIN", tag, "rows", bounds.matrices[index].shape[0], "variable", var, flush=True)
        for i, (jac, residual, allowed) in enumerate(bounds.norms):
            m = sparse.csr_matrix(jac) @ projection
            d = residual.shape[1]
            var = int((np.asarray(m.getnnz(axis=1)).reshape(-1, d).sum(1) > 0).sum())
            slack = allowed - np.linalg.norm(residual, axis=1)
            tag = next((t for t, idx in tags_norm if idx == i), "appended")
            print("NORM", i, tag, "cones", len(allowed), "variable", var, "slack quantiles(mm)",
                  np.round(np.quantile(slack, [0, .1, .5]) * 1000, 3) if len(slack) else None, flush=True)
        rows = [(k, r) for k, r in measured["rows"] if cluster["mask"][k[0]]]
        print("WITNESS rows", len(rows), "keys", len({k for k, _ in rows}), flush=True)
    finally:
        geometry.stop_workers()
    raise SystemExit(0)


def profile_setup(objective, poses, scene_objects=(), scene_query=None):
    import cProfile, pstats, io
    from .mesh_repair import OriginalGeometry
    from .repair_geometry_setup import prepare_static_bounds, prepare_moving_reduction
    from . import local_windows as lw
    entry = np.array(poses, copy=True)
    pr = cProfile.Profile()
    t = {}
    tick = time.monotonic(); pr.enable()
    limits = lw.ClusterLimits(objective, entry); t['limits'] = time.monotonic() - tick
    tick = time.monotonic()
    geometry = OriginalGeometry(objective, scene_objects, scene_query, manifold=True); t['geometry_init'] = time.monotonic() - tick
    tick = time.monotonic(); prepare_static_bounds(geometry, entry); t['static_bounds'] = time.monotonic() - tick
    tick = time.monotonic(); prepare_moving_reduction(geometry, entry); t['moving_reduction'] = time.monotonic() - tick
    tick = time.monotonic(); geometry.start_workers(len(entry)); t['start_workers'] = time.monotonic() - tick
    tick = time.monotonic(); measured = geometry.measure(entry, jacobians=True); t['entry_measure'] = time.monotonic() - tick
    pr.disable()
    geometry.stop_workers()
    s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats('cumulative').print_stats(35)
    print("SETUP", json.dumps(t), flush=True)
    print(s.getvalue()[:9000], flush=True)
    raise SystemExit(0)
