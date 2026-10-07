from types import SimpleNamespace
import numpy as np
from omniretargeting._fast.acquisition_acceptance import refine_fraction


def objective():
    return SimpleNamespace(frames=1, width=1, project=lambda q: q.copy(),
                           coords=SimpleNamespace(integrate=lambda q, d, a: q+a*d))


def test_refinement_returns_only_exactly_guarded_larger_step():
    guard = SimpleNamespace(accepts=lambda q: bool(q.max()**2 <= .7**2))
    candidates = []
    candidate, alpha = refine_fraction(objective(), np.zeros((1, 1)), np.ones(1),
                                      [guard], np.array([[.5]]), .5, 1.,
                                      observe=lambda a, ok: candidates.append((a, ok)))
    assert .68 <= alpha <= .7
    assert guard.accepts(candidate)
    assert len(candidates) == 4
    np.testing.assert_allclose(candidate, [[alpha]])


def test_every_guard_must_accept_and_original_is_kept_if_all_probes_fail():
    calls = []
    first = SimpleNamespace(accepts=lambda q: True)
    def second(q):
        calls.append(q.item())
        return q.item() <= .5
    original = np.array([[.5]])
    candidate, alpha = refine_fraction(objective(), np.zeros((1, 1)), np.ones(1),
                                      [first, SimpleNamespace(accepts=second)], original, .5, 1.)
    assert candidate is original
    assert alpha == .5
    assert len(calls) == 4


def test_small_fraction_retains_restoration_path_without_guard_calls():
    guard = SimpleNamespace(accepts=lambda q: (_ for _ in ()).throw(AssertionError()))
    candidate = np.array([[.125]])
    result, alpha = refine_fraction(objective(), np.zeros((1, 1)), np.ones(1),
                                    [guard], candidate, .125, .25)
    assert result is candidate
    assert alpha == .125
