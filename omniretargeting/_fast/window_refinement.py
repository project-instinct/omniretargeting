"""Add current-pose collision witnesses without rebasing a local LM model."""

from copy import copy
from dataclasses import replace
import hashlib
import time

import numpy as np

from .priority_merit import report_value
from .qp_assembly import assemble_program
from .qp_guard_rows import intervals

MAX_REFINEMENTS = 3
MAX_NEW_WITNESSES = 256


def pose_digest(poses):
    value = np.asarray(poses)
    return hashlib.sha256(repr((value.shape, value.dtype.str)).encode() + value.tobytes()).hexdigest()


def _copy_soft(soft):
    result = copy(soft)
    for name in ('keys', 'weights', 'levels', 'rows', 'cones', 'heads'):
        setattr(result, name, list(getattr(soft, name)))
    result._index, result.limits = dict(soft._index), dict(soft.limits)
    return result


def augment_model(model, mandatory, package):
    """Append proposal-only rows, preserving all original hard/soft domains.

    The coordinator owns bounds, weights and witness selection. New batches
    have no nonlinear SOC evaluator; existing providers retain their original
    batch indices. No coordinator state or previous program is changed here.
    """
    original, old_soft = model.program, model.soft
    if np.shape(mandatory) != original.b.shape:
        raise ValueError('Geometry refinement mandatory mask has wrong shape')
    blocks = list(original.hard_blocks)
    if not blocks:
        raise ValueError('Geometry refinement requires original hard blocks')
    size = model.gradient.size
    added_hard = []
    for matrix, lower, upper in package['hard_linear']:
        lower, upper = np.asarray(lower, float), np.asarray(upper, float)
        if (matrix.shape != (len(upper), size) or lower.shape != upper.shape
                or not np.isfinite(matrix.data).all() or not np.isfinite(upper).all()
                or not np.all(np.isneginf(lower)) or np.any(upper < 0.)):
            # Do not silently widen an incompatible frozen facet or reserve.
            raise ValueError('Invalid or origin-incompatible geometry hard rows')
        added_hard.append(intervals(matrix, lower, upper))
    new_hard_count = sum(len(block.b) for block in added_hard)
    if not new_hard_count:
        raise ValueError('Geometry refinement requires an origin-feasible hard row')
    soft = _copy_soft(old_soft)
    for item in package['soft_rows']:
        if any(key[0] not in ('penetration', 'clearance_margin') for key in item[-1]):
            raise ValueError('Geometry refinement cannot alter contact epigraphs')
        soft.add(item)
    program = assemble_program([*blocks, *added_hard], soft, size)

    # Assembly groups scalar hard, scalar soft, epigraph domain, then cones.
    # Map every previous row, preserving exact mandatory/source labels even
    # when new shared keys insert columns or shift all cone head indices.
    hard_count = sum(len(b.b) for b in blocks if not b.soc)
    soft_count = sum(len(bound) for _, bound, _ in old_soft.rows)
    new_soft_count = sum(len(bound) for _, bound, _ in soft.rows) - soft_count
    domain_count = len(old_soft.keys) + len(old_soft.limits)
    if original.nonneg != hard_count + soft_count + domain_count:
        raise ValueError('Geometry refinement original domain layout changed')
    mapping = np.r_[np.arange(hard_count),
                    hard_count + new_hard_count + np.arange(soft_count),
                    hard_count + new_hard_count + soft_count + new_soft_count + np.arange(domain_count),
                    program.nonneg + np.arange(len(original.b) - original.nonneg)].astype(int)
    keep = np.zeros(len(program.b), bool)
    keep[mapping] = mandatory
    keep[hard_count:hard_count + new_hard_count] = True
    start = hard_count + new_hard_count + soft_count
    keep[start:start + new_soft_count] = True
    keep[program.nonneg - len(soft.keys) - len(soft.limits):program.nonneg] = True

    old_sources = model.window_row_sources
    names = ['unlabelled_original'] if old_sources is None else list(old_sources['names'])
    codes = np.empty(len(program.b), np.int16)
    codes[mapping] = 0 if old_sources is None else old_sources['codes']
    def label(rows, name):
        if name not in names:
            names.append(name)
        codes[rows] = names.index(name)
    label(slice(hard_count, hard_count + new_hard_count), 'geometry_refinement:hard')
    offset = start
    for _, bound, indices in soft.rows[len(old_soft.rows):]:
        for index in indices:
            label(offset, 'geometry_refinement:' + soft.keys[index][0])
            offset += 1
    new_keys = len(soft.keys) - len(old_soft.keys)
    if new_keys:
        label(slice(program.nonneg - new_keys, program.nonneg), 'epigraph_nonnegativity')
    result = replace(model, program=program, soft=soft, priority_cache=None,
                     window_working_set=None, solve_diagnostics=[],
                     window_row_sources=dict(names=names, codes=codes))
    return result, keep, dict(added_hard_rows=new_hard_count, added_soft_rows=new_soft_count,
                              added_slack_keys=new_keys, rows=len(program.b),
                              variables=program.A.shape[1], cones=len(program.soc))


class GeometryRefinement:
    """Bounded requests and re-solves shared by one LM iteration's attempts."""
    def __init__(self, coordinator, *, enabled):
        self.coordinator, self.enabled = coordinator, enabled
        self.known = set()
        self.rounds, self.requests = 0, 0
        self.trace = []

    def refine(self, model, mandatory, candidate):
        tick = time.monotonic()
        record = dict(geometry_generation=self.rounds, request=self.requests,
                      unresolved=True, added=False, optimality_certified=False)
        updated, keep = None, mandatory
        remaining = MAX_NEW_WITNESSES - len(self.known)
        hook = getattr(self.coordinator, 'geometry_refinement', None)
        if not self.enabled:
            record['reason'] = 'disabled'
        elif self.requests >= MAX_REFINEMENTS or remaining <= 0:
            record['reason'] = 'geometry_generation_budget'
        elif hook is None:
            record['reason'] = 'missing_geometry_refinement_hook'
        else:
            self.requests += 1
            try:
                package = hook(model.origin, candidate, frames=model.frames,
                               known_witness_ids=frozenset(self.known), max_new_witnesses=remaining)
                record['report'] = report_value({key: value for key, value in package.items()
                                                if key not in ('hard_linear', 'soft_rows')})
                record['unresolved'] = bool(package['unresolved'])
                if package['hard_linear']:
                    self._validate(package, model, candidate, remaining)
                    updated, keep, details = augment_model(model, mandatory, package)
                    self.known.update(package['witness_ids'])
                    self.rounds += 1
                    record.update(details, added=True, geometry_generation=self.rounds,
                                  reason='geometry_rows_added')
                else:
                    if package['soft_rows']:
                        raise ValueError('Geometry soft rows lack a hard nonregression row')
                    record['reason'] = ('unresolved_geometry' if record['unresolved']
                                        else 'geometry_already_covered')
            except (ValueError, TypeError, KeyError, AttributeError) as error:
                # A bad proposal package cannot authorize a nonlinear trial.
                updated, keep = None, mandatory
                record.update(reason='invalid_geometry_refinement', unresolved=True,
                              error=f'{type(error).__name__}: {error}')
        record.update(seconds=time.monotonic() - tick, requests_used=self.requests,
                      witnesses_used=len(self.known))
        self.trace.append(record)
        return updated, keep, record

    def _validate(self, package, model, candidate, remaining):
        if (package['origin_pose_id'] != pose_digest(model.origin)
                or package['trial_pose_id'] != pose_digest(candidate)
                or package['epoch'] != self.coordinator.epoch):
            raise ValueError('Stale geometry refinement pose or epoch')
        state = getattr(self.coordinator, '_window_state', None)
        expected = getattr(state, 'measured', {}).get('evidence_id')
        if expected is not None and package['evidence_id'] != expected:
            raise ValueError('Geometry refinement evidence changed')
        identities = tuple(package['witness_ids'])
        unique = set(identities)
        if (not identities or len(unique) != len(identities) or len(unique) > remaining
                or self.known.intersection(unique)):
            raise ValueError('Duplicated, empty or excessive geometry witness identities')
