"""Contact-aware coordinate QPs inside the single final joint projection.

The variables remain independent scalar-joint time series with the original
intersected continuity/box constraints. Cross-joint contact Jacobians couple
their objectives through Gauss-Seidel sweeps, without a trajectory conic QP.
Roots and coordinates affecting the anatomical root anchor are held here.
"""
import time

import numpy as np
from scipy import sparse

from .soft_support_rows import SupportRows, SupportWeights


class ContactProjection:
    def __init__(self, objective, target, metric, *, weights=None, physical=None,
                 source_yield=False, world_recovery=False):
        from .soft_projection import root_families
        self.objective, self.target = objective, target
        cols = objective.coords.scalar_cols
        root = root_families(objective, target, True)
        affected = set(np.concatenate([f.matrix.indices % objective.width for f in root])) if root else set()
        self.enabled = np.array([c not in affected for c in cols])
        self.indices = (np.arange(len(target))[:, None] * objective.width + cols).ravel()
        # Contact velocity is refined at the final projection, where each
        # joint's exact linear continuity intersection is present in the QP.
        weights = weights or SupportWeights(support_velocity_sigma=.0006)
        residual, matrix, _ = SupportRows(objective, weights, yield_targets=source_yield).evaluate(target, True)
        if world_recovery:
            from .soft_swing_rows import SwingRows
            rr,rj,_=SwingRows(objective,include_contacts=False).evaluate(target,True)
            residual=np.r_[residual,rr];matrix=sparse.vstack([matrix,rj],format='csr')
        matrix = matrix[:, self.indices]
        self.curvature = (matrix.T @ matrix).tocsr()
        self.gradient = np.asarray(matrix.T @ residual).ravel()
        if physical is not None:
            from .soft_projection_physical import curvature
            acc, residual, temporal = physical[:3]
            self.curvature += curvature(acc, temporal)[self.indices][:, self.indices]
            self.gradient += (acc.gradient.ravel() + temporal.T @ residual)[self.indices]
        proximity = np.maximum(.1 * metric[:, cols].ravel(), 1.)
        self.curvature += sparse.diags(proximity)
        self.cols = len(cols)
        self.diagonals = {}
        from .barrier_rows import BarrierRows
        from .legacy_window_qp import WindowObjective
        layouts=BarrierRows(WindowObjective(objective,target,range(len(target))))
        layouts.guards=[g for g in layouts.guards if g[0]=='shape']
        families=layouts.evaluate(target,True)
        self.layout=(sparse.vstack([f.matrix[:,self.indices] for f in families],format='csr')
                     if families else sparse.csr_matrix((0,len(self.indices))))
        self.layout_slack=np.concatenate([f.slack for f in families]) if families else np.empty(0)

    def program(self, joint, result):
        if not self.enabled[joint]:
            return None, None
        selected = np.arange(len(result)) * self.cols + joint
        diagonal = self.diagonals.get(joint)
        if diagonal is None:
            diagonal = self.curvature[selected][:, selected].tocsc()
            self.diagonals[joint] = diagonal
        correction = (result[:, self.objective.coords.qi] - self.target[:, self.objective.coords.qi]).ravel()
        linear = (self.gradient[selected] + self.curvature[selected] @ correction
                  - diagonal @ correction[selected] - diagonal @ self.target[:, self.objective.coords.qi[joint]])
        return diagonal, np.asarray(linear).ravel()

    def boxes(self, joint, result):
        """Exact scalar interval reduction of the frozen layout halfspaces.

        Other joints retain their current Gauss-Seidel corrections. Every
        layout row touches one frame; neither a tolerance nor a root domain
        is changed. The final nonlinear certificate still owns acceptance.
        """
        selected=np.arange(len(result))*self.cols+joint
        column=self.layout[:,selected].tocoo()
        correction=(result[:,self.objective.coords.qi]-self.target[:,self.objective.coords.qi]).ravel()
        rest=self.layout_slack+self.layout@correction-self.layout[:,selected]@correction[selected]
        limit=self.target[column.col,self.objective.coords.qi[joint]]-rest[column.row]/column.data
        lower=np.full(len(result),-np.inf);upper=np.full(len(result),np.inf)
        positive=column.data>0
        np.maximum.at(lower,column.col[positive],limit[positive])
        np.minimum.at(upper,column.col[~positive],limit[~positive])
        return lower,upper


def refine(objective, projected, bulk, metric, *, sweeps=2, evaluate=None,
           physical=None, source_yield=False, world_recovery=False):
    from .soft_projection import project_joints, certify
    tick = time.monotonic()
    report = dict(mode='coupled_contact_coordinate_projection', sweeps=0, qp_solves=0, qp_seconds=0.)
    model = ContactProjection(objective, projected, metric, physical=physical,
                              source_yield=source_yield,world_recovery=world_recovery)
    baseline = None if evaluate is None else evaluate(projected, False)
    state = projected.copy()
    report['_contact_program'] = model
    from .soft_projection_qp import ProjectionFailure
    try:
        for sweep in range(sweeps):
            state = project_joints(objective, state, report=report)
            report['sweeps'] = sweep + 1
    except ProjectionFailure as error:
        report.update(alpha=0.,seconds=time.monotonic()-tick,stall=str(error),
                      fallback='certified_original_projection')
        return projected,report
    finally:
        report.pop('_contact_program')
    # The root path was fixed during the coordinate programs. Original joint
    # constraints are convex in these scalar coordinates; the nonlinear
    # layout and anatomical-anchor predicates still own certification.
    fraction = 1.
    for _ in range(7):
        trial = projected.copy()
        trial[:, objective.coords.qi] += fraction * (state[:, objective.coords.qi] - projected[:, objective.coords.qi])
        failures = certify(objective, trial)
        if not failures:
            if evaluate is not None:
                value = evaluate(trial, False)
                # Sticking cannot purchase additional overlap. These are
                # numerical projection trials; exact guard tolerances remain
                # owned by certify above. The original physical objective and
                # measured clearance own the accepted projection refinement.
                clearance = all(value[3].get(key, 0.) <= baseline[3].get(key, 0.)
                                for key in ('scene_depth_m', 'self_depth_m', 'plane_depth_m'))
                if not clearance or value[0].cost > baseline[0].cost:
                    report.setdefault('rejected_physical_trials', []).append(dict(
                        alpha=fraction, before_cost=baseline[0].cost, after_cost=value[0].cost,
                        before_stats=baseline[3], after_stats=value[3]))
                    fraction *= .5
                    continue
            report.update(alpha=fraction, seconds=time.monotonic() - tick,
                          exact_guard_failures=[], root_path='held_at_feasible_projection')
            return trial, report
        report.setdefault('rejected_certificates', []).append(dict(alpha=fraction, failures=failures))
        fraction *= .5
    report.update(alpha=0., seconds=time.monotonic() - tick, exact_guard_failures=failures,
                  fallback='certified_original_projection')
    return projected, report
