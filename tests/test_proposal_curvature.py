"""Contact-only Newton proposals retain full guards and original step bounds."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from omniretargeting._fast.acquisition_guard import AcquisitionNormalGuard
from omniretargeting._fast.proposal_curvature import restore_curvature


class Coordinates:
    width = 1
    @staticmethod
    def integrate(origin, delta, alpha):
        return origin + delta * alpha

    @staticmethod
    def difference(origin, candidate):
        return candidate - origin

    @staticmethod
    def bounds(origin, angular, translation):
        return np.full_like(origin, -angular), np.full_like(origin, angular)


class QuadraticContact(AcquisitionNormalGuard):
    def __init__(self, cone=False):
        self.cone = cone

    def accepts(self, candidate):
        return bool(np.all(candidate ** 2 <= .81 + 1e-7))

    def qp_bounds(self, candidate):
        x = candidate.ravel()
        if self.cone:
            rows = sparse.csr_matrix((2 * x, (np.arange(len(x)) * 3, np.arange(len(x)))),
                                     shape=(len(x) * 3, len(x)))
            return [], [(rows, np.c_[x ** 2, np.zeros((len(x), 2))], np.full(len(x), .81))]
        return [(sparse.diags(2 * x, format='csr'), np.full(len(x), -np.inf), .81 - x ** 2)], []


class HigherGuard:
    def __init__(self, predicate):
        self.predicate = predicate

    def accepts(self, candidate):
        return bool(self.predicate(candidate))

    def qp_bounds(self, candidate):
        return [], []


def _problem(bound=2.):
    origin = np.zeros((1, 1))
    obj = SimpleNamespace(coords=Coordinates(), frames=1, width=1,
                          cfg={'angular_step_rad': bound, 'translation_step_m': bound},
                          project=lambda poses: np.clip(poses, -2., 2.))
    metric = np.ones((1, 1)), np.ones(1), .001
    return obj, origin, metric


@pytest.mark.parametrize('cone', [False, True])
def test_scalar_and_vector_contact_defects_reach_original_nonlinear_acceptance(cone):
    obj, origin, metric = _problem()
    guard = QuadraticContact(cone)
    result = restore_curvature(obj, origin, np.ones((1, 1)), [guard], metric)
    assert result is not None
    assert guard.accepts(result)
    assert abs(result[0, 0] - .9) < 1e-6


@pytest.mark.parametrize('predicate', [lambda q: np.all(q <= .95), lambda q: np.all(q >= .95)])
def test_any_other_guard_failure_keeps_original_restoration_fallback(predicate):
    obj, origin, metric = _problem()
    assert restore_curvature(obj, origin, np.ones((1, 1)),
                             [HigherGuard(predicate), QuadraticContact()], metric) is None


def test_origin_step_box_is_kept_after_contact_correction():
    obj, origin, metric = _problem(bound=.5)
    assert restore_curvature(obj, origin, np.ones((1, 1)), [QuadraticContact()], metric) is None


def test_no_metric_or_no_contact_keeps_original_restoration_fallback():
    obj, origin, metric = _problem()
    assert restore_curvature(obj, origin, np.ones((1, 1)), [QuadraticContact()], None) is None
    assert restore_curvature(obj, origin, np.ones((1, 1)), [HigherGuard(lambda q: True)], metric) is None
