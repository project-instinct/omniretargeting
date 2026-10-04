"""Deduplicate identical failed proposal searches, never certify infeasibility.

Keep one SHA-256 signature, not another trajectory-sized QP. Numeric arrays
are compared byte-for-byte (no rounding or feasibility tolerance). Immutable
source/scene/guard references are part of the acquisition lifetime contract;
guard identities additionally prevent reuse across newly constructed guards.
"""

from collections.abc import Mapping
import hashlib

import numpy as np
from scipy import sparse


def _feed(digest, value):
    if sparse.issparse(value):
        _feed(digest, ('sparse', value.format, value.shape))
        if value.format not in ('csr', 'csc'):
            raise TypeError('Unsupported sparse storage')
        for array in (value.indptr, value.indices, value.data):
            _feed(digest, array)
    elif isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError('Object arrays are not an exact numeric fingerprint')
        _feed(digest, ('array', value.dtype.str, value.shape))
        if value.size:
            digest.update(memoryview(np.ascontiguousarray(value)).cast('B'))
    elif isinstance(value, np.generic):
        _feed(digest, np.asarray(value))
    elif isinstance(value, Mapping):
        digest.update(b'mapping[')
        for key in sorted(value, key=repr):
            _feed(digest, key)
            _feed(digest, value[key])
        digest.update(b']')
    elif isinstance(value, (tuple, list)):
        digest.update(b'tuple[' if isinstance(value, tuple) else b'list[')
        for part in value:
            _feed(digest, part)
        digest.update(b']')
    elif value is None or isinstance(value, (str, bytes, bool, int, float)):
        encoded = repr((type(value).__name__, value)).encode()
        digest.update(str(len(encoded)).encode() + b':' + encoded)
    else:
        raise TypeError(f'Unsupported fingerprint value {type(value).__name__}')


def fingerprint(coordinator, model, policy):
    """None means insufficient evidence to skip; even a tiny change is a miss."""
    try:
        program, soft = model.program, model.soft
        objective = coordinator.objective
        measured = coordinator.current_measurement
        geometry = getattr(coordinator.provider, 'geometry', None)
        digest = hashlib.sha256()
        _feed(digest, ('priority-stall-v1', policy, objective.cfg))
        _feed(digest, (model.origin, model.step_lower, model.step_upper, model.system,
                       model.hessian, model.gradient, model.diagonal, model.scale))
        _feed(digest, (program.A, program.b, program.nonneg, program.soc, program.step_size,
                       soft.keys, soft.weights, soft.levels, soft.rows, soft.cones))
        # Rows include witness world points, directions and linear Jacobians.
        _feed(digest, {key: measured.get(key) for key in
                       ('rows', 'depths', 'failures', 'complete', 'reliable', 'frame_complete',
                        'body_reliable', 'evidence_id', 'reference_observation_id')})
        _feed(digest, getattr(geometry, 'reference_observation_id', None))
        _feed(digest, (coordinator.epoch, coordinator.current, coordinator.reference,
                       coordinator.initial_depth, coordinator.current_depth, coordinator.best_depth,
                       coordinator.best_keys, coordinator.spent, coordinator._current_contacts))
        _feed(digest, tuple(id(item) for item in coordinator.contacts))
        _feed(digest, tuple(id(item) for item in getattr(objective, 'candidate_guards', ())))
        _feed(digest, tuple(id(getattr(objective, name, None)) for name in
                           ('coords', 'source_shape', 'reference_shape')))
        return digest.hexdigest()
    except (TypeError, AttributeError, ValueError):
        return None


class StalledSearch:
    def __init__(self):
        self.saved = None
        self.skips = []

    def invalidate(self):
        self.saved = None

    def record(self, coordinator, model, policy, record):
        self.saved = None
        if record.get('priority_status') != 'no_certified_accepted_proposal':
            return
        signature = fingerprint(coordinator, model, policy)
        if signature is not None:
            self.saved = (signature, dict(
                previous_phase=record.get('phase'), previous_iteration=record.get('iteration'),
                original_status=record['priority_status']))

    def repeated(self, coordinator, model, policy):
        if self.saved is None:
            return None
        signature = fingerprint(coordinator, model, policy)
        if signature != self.saved[0]:
            self.invalidate()
            return None
        record = dict(self.saved[1], priority_status='repeated_stalled_problem',
                      problem_signature=signature, hard_infeasibility_certified=False,
                      convergence_certified=False, skipped_qp_search=True,
                      explanation='Identical failed proposal search; residuals remain unresolved.')
        self.skips.append(record)
        return dict(record)
