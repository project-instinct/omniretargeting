"""Immutable exact guard slacks, with window-local tangent derivatives.

Acceptance tolerances are imported from the original guards. Proposal reserves,
current-pose allowances and reference root balls are never used as hard slacks.
Boxes and fixed roots are equalities/active domains owned by the step solver.
"""

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .barrier_assembly import scatter
from .rows import _block_rows


class UnsupportedBarrier(ValueError):
    pass


@dataclass
class Family:
    slack: np.ndarray
    matrix: object  # scalar: ds/dq; ball: du/dq, components consecutively
    label: tuple
    locations: object
    residual: object = None

    def identity(self, index):
        return (*self.label, *self.locations[index])

    @property
    def ball(self):
        return self.residual is not None


def scalar(value, matrix, label, locations):
    return Family(np.asarray(value).ravel(), matrix, label, locations)


def ball(value, radius, matrix, label, locations):
    slack = np.asarray(radius)**2 - np.einsum('ij,ij->i', value, value)
    return Family(slack, matrix, label, locations, value)


def _locations(first, shape):
    indices = np.indices(shape).reshape(len(shape), -1).T
    indices[:, 0] += first
    return indices


class BarrierRows:
    def __init__(self, window):
        from .acquisition_guard import AcquisitionNormalGuard
        from .continuity import CorrectionContinuity
        from .reference import ReferenceShape
        from .priority_contacts import PriorityContacts
        from .._sequence.continuity_guard import SequenceContinuityGuard
        self.window = window
        self.guards = []
        objective = window.global_objective
        if (getattr(objective, 'priority_clearance', None) is not None or
                getattr(objective, 'original_clearance', None) is not None):
            raise UnsupportedBarrier('transactional/original clearance has no fixed barrier domain')
        if not isinstance(objective.source_shape, ReferenceShape):
            raise UnsupportedBarrier('barrier acquisition requires a fixed ReferenceShape')
        for guard in [objective.source_shape, *objective.candidate_guards]:
            if isinstance(guard, AcquisitionNormalGuard):
                if guard.priority is not None:
                    raise UnsupportedBarrier('priority contact domain')
                adapter = PriorityContacts(guard, type(guard).__name__)
                self.guards.append(('contact', guard, adapter))
            elif isinstance(guard, CorrectionContinuity):
                self.guards.append(('correction', guard, None))
            elif isinstance(guard, SequenceContinuityGuard):
                self.guards.append(('sequence', guard, None))
            elif type(guard) is ReferenceShape:
                self.guards.append(('shape', guard, None))
            else:
                raise UnsupportedBarrier(f'unknown barrier adapter: {type(guard).__name__}')
        self.scales = None
        self.evaluations = self.jacobian_evaluations = 0

    def evaluate(self, poses, jacobian=False):
        from .window_guards import _correction, _sequence, _shape_kin
        from .continuity import ROUND_OFF
        from .._sequence.continuity_guard import ACCEPTANCE_SLACK
        window = self.window
        full, frames = window.embed(poses), window.variable_frames
        owner = window.global_objective
        # All geometric guard kernels reuse the objective's exact per-pose FK.
        owner.window_kinematics(full, frames, with_jacobian=jacobian)
        families, joints_seen = [], set()
        for kind, guard, adapter in self.guards:
            guard._window_frame_cache = owner._window_frame_cache
            if kind == 'contact':
                for block in adapter.evaluate(full, frames=frames, jacobian=jacobian):
                    radius = block.bound + block.own + 1e-7
                    label = ('AcquisitionNormalGuard',)
                    if block.scalar:
                        families.append(scalar(radius - block.residual,
                            None if block.matrix is None else -block.matrix, label, block.identities))
                    else:
                        families.append(ball(block.residual, radius, block.matrix, label, block.identities))
            elif kind == 'shape':
                kin = _shape_kin(guard, full, frames, jacobian)
                for number, envelope in enumerate(guard.envelopes):
                    envelope = envelope.for_frames(window.span)
                    _, _, (change, pairs) = envelope.measure(kin, with_jacobian=jacobian)
                    value, derivative = envelope.layout(change, pairs)
                    if value is None:
                        continue
                    keep = (envelope.layout_sign != 0).ravel()
                    matrix = None if derivative is None else _block_rows(
                        derivative.reshape(len(frames), -1, window.width), window.width)[keep]
                    families.append(scalar(value.ravel()[keep] + envelope.SLACK, matrix,
                        ('ReferenceShape', number, 'layout'), _locations(frames.start, value.shape)[keep]))
            else:
                values = (_correction(guard, full, frames, jacobian) if kind == 'correction'
                          else _sequence(guard, full, frames, jacobian))
                for key, (value, matrix, lo, hi, order) in values.items():
                    bound = guard.bounds[key][lo:hi]
                    tolerance = ROUND_OFF if kind == 'correction' else ACCEPTANCE_SLACK
                    name = key.rsplit('_order_', 1)[0] if kind == 'correction' else key[0]
                    label = (type(guard).__name__, name, order)
                    if kind == 'correction' or name == 'joints':
                        # Deduplicate identical immutable joint constraints only.
                        signature = (order, guard.correction, bound.shape, bound.tobytes(), tolerance)
                        if name == 'joints' and signature in joints_seen:
                            continue
                        if name == 'joints':
                            joints_seen.add(signature)
                        limit = bound.ravel() + tolerance
                        locations = _locations(lo, value.shape)
                        families.extend([
                            scalar(limit - value.ravel(), None if matrix is None else -matrix,
                                   (*label, 'upper'), locations),
                            scalar(limit + value.ravel(), matrix, (*label, 'lower'), locations)])
                    else:
                        families.append(ball(value, bound + tolerance, matrix, label,
                                             _locations(lo, (len(value),))))
                    if kind == 'correction' and key in guard.norm_bounds:
                        families.append(ball(value, guard.norm_bounds[key][lo:hi] + ROUND_OFF,
                            matrix, (*label, 'norm'), _locations(lo, (len(value),))))
        self.evaluations += 1
        self.jacobian_evaluations += bool(jacobian)
        return families

    def initialize(self, families):
        self.scales = [family.slack.copy() for family in families]

    def merit(self, families):
        if not strictly_interior(families):
            return np.inf
        return -sum(float(np.log(f.slack / s0).sum()) for f, s0 in zip(families, self.scales))


def strictly_interior(families):
    return all(np.isfinite(f.slack).all() and np.all(f.slack > 0) for f in families)


def minimum(families):
    candidates = [(float(f.slack.min()), f, int(np.argmin(f.slack)))
                  for f in families if len(f.slack)]
    if not candidates:
        return dict(slack=None, identity=None)
    slack, family, index = min(candidates, key=lambda x: x[0])
    return dict(slack=slack, identity=family.identity(index))


def accumulate(families, mu, ab):
    """Add barrier PSD GN curvature directly into the existing physical band."""
    gradient = np.zeros(ab.shape[1])
    for f in families:
        if not len(f.slack):
            continue
        if f.ball:
            dim = f.residual.shape[1]
            count = len(f.slack)
            # One row J^T u per ball, not a normal matrix.
            pull = sparse.csr_matrix((f.residual.ravel(),
                (np.repeat(np.arange(count), dim), np.arange(count * dim))),
                shape=(count, count * dim)) @ f.matrix
            gradient += np.asarray(pull.T @ (2 * mu / f.slack)).ravel()
            scatter(ab, f.matrix, np.repeat(2 * mu / f.slack, dim))
            scatter(ab, pull, 4 * mu / f.slack**2)
        else:
            gradient -= np.asarray(f.matrix.T @ (mu / f.slack)).ravel()
            scatter(ab, f.matrix, mu / f.slack**2)
    return gradient


def boundary_fraction(families, step):
    """0.995 fraction to the affine scalar/vector boundary, capped at one."""
    boundary = np.inf
    for f in families:
        if not len(f.slack):
            continue
        derivative = f.matrix @ step
        if not f.ball:
            decreasing = derivative < 0
            boundary = min(boundary, np.min(f.slack[decreasing] / -derivative[decreasing], initial=np.inf))
        else:
            delta = derivative.reshape(f.residual.shape)
            a = np.einsum('ij,ij->i', delta, delta)
            dot = np.einsum('ij,ij->i', f.residual, delta)
            moving = a > 0
            root = np.sqrt(dot[moving]**2 + a[moving] * f.slack[moving])
            # Stable positive quadratic root, including outward and inward motion.
            distance = np.where(dot[moving] >= 0,
                f.slack[moving] / (root + dot[moving]), (root - dot[moving]) / a[moving])
            boundary = min(boundary, np.min(distance, initial=np.inf))
    return min(1., .995 * boundary)
