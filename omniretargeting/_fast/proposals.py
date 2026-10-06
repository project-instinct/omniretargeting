"""Damping attempts of one Levenberg-Marquardt iteration.

Every rejected attempt grows the damping the same way, so once the first
attempt of an iteration is rejected the damping of all remaining attempts is
known. Their proposals (banded step and backtracking, then, for the attempts
that reach it, the constrained conic fallback) do not depend on each other and
are computed at once in forked processes. The solver still takes them in the
sequential order: the accepted step, the history and the guards' rejection
counters equal those of computing each attempt when it is reached.
"""

import multiprocessing
import os

from scipy.linalg import LinAlgError
from .box_step import BoxStepFailure

ALPHAS = (1., .5, .25, .125, .0625, .03125, .015625, .0078125)
# Scheduling only: attempts are computed in forked processes when possible.
SPECULATE = hasattr(os, "fork")


def guards_of(objective):
    """Source shape (or None) and every candidate guard, shape first."""
    guards = list(getattr(objective, "candidate_guards", ()))
    shape = getattr(objective, "source_shape", None)
    if shape is not None:
        guards.insert(0, shape)
    return shape, guards


def backtrack(objective, poses, direction, guards, alphas=ALPHAS):
    """Largest listed fraction of ``direction`` whose projection every guard accepts."""
    coords = objective.coords
    frames, width = objective.frames, objective.width
    for alpha in ((1.,) if not guards else alphas):
        trial = objective.project(coords.integrate(poses, direction.reshape(frames, width), alpha))
        from .priority_merit import check_trial
        if check_trial(objective, trial, guards):
            return trial, alpha
    return None, 0.


def _counters(guards):
    return [getattr(guard, "rejections", 0) for guard in guards]


def _serve(function, attempt, connection):
    try:
        connection.send(("ok",) + function(attempt))
    except BaseException as error:  # re-raised where the attempt is taken
        connection.send(("error", error, None))
    connection.close()


class Attempts:
    """Banded and conic proposals of the attempts of one iteration."""

    def __init__(self, objective, poses, system, gradient, diagonal, held, radial, rhs,
                 dampings, band_step, conic_step, model=None):
        self.objective, self.poses = objective, poses
        self.system, self.gradient, self.diagonal = system, gradient, diagonal
        self.held, self.radial, self.rhs = held, radial, rhs
        self.dampings = list(dampings)
        self._band_step, self._conic_step = band_step, conic_step
        self.cache = {}
        self.conic_numbers = {}
        self.model = model

    def _banded(self, attempt):
        guards = guards_of(self.objective)[1]
        before = _counters(guards)
        try:
            step = self._band_step(self.system, self.diagonal, self.held, self.radial, self.rhs,
                                   self.dampings[attempt])
        except BoxStepFailure:
            # A bounded numerical failure requests the normal conic fallback.
            # No clipped or unconverged box iterate becomes a candidate.
            result = (None, None, 0.)
        except LinAlgError:
            result = None
        else:
            joint = getattr(self.objective, 'joint_witness', None)
            if joint is None:
                candidate, alpha = backtrack(self.objective, self.poses, step, guards)
            else:
                from .joint_proposal import banded_candidate
                candidate, alpha = banded_candidate(joint, self.poses, step, guards)
            result = (step, candidate, alpha)
        return result, [a - b for a, b in zip(_counters(guards), before)]

    def _conic(self, attempt):
        guards = guards_of(self.objective)[1]
        before = _counters(guards)
        kwargs = {} if self.model is None else {'model': self.model}
        result = self._conic_step(self.objective, self.poses, self.system, self.gradient,
                                  self.diagonal, self.conic_damping(attempt), guards, **kwargs)
        return result, [a - b for a, b in zip(_counters(guards), before)]

    def _take(self, kind, attempt):
        cached = self.cache.pop((kind, attempt), None)
        if cached is None:
            # Computed here: the guards count their own rejections.
            return (self._banded if kind == "banded" else self._conic)(attempt)[0]
        status, result, deltas = cached
        if status == "error":
            raise result
        for guard, delta in zip(guards_of(self.objective)[1], deltas):
            if delta:
                guard.rejections += delta
        return result

    def banded(self, attempt):
        """``(step, candidate, alpha)``, or None for a singular system."""
        return self._take("banded", attempt)

    def conic_damping(self, attempt):
        """Spend the existing three conic solves at distinct trust scales.

        Tiny LM damping can otherwise make all three failed directions nearly
        identical. The final unit damping weights the Gauss-Newton diagonal
        equally; it changes the proposal, never any physical guard budget.
        """
        from .runtime_options import OPTIONS
        value = self.dampings[attempt]
        if OPTIONS.conic_trust_damping:
            number = self.conic_numbers.get(attempt, 1)
            value = max(value, (0., 1e-3, 1.)[min(number - 1, 2)])
        return value

    def conic(self, attempt, number=None):
        """``(step, candidate, alpha)`` of the conic fallback, or None."""
        if number is not None:
            self.conic_numbers[attempt] = number
        return self._take("conic", attempt)

    def speculate(self, first, shape_solves, max_shape_solves, conic_possible):
        """Compute attempts ``first..`` at once; conic ones as the loop would reach them."""
        from .runtime_options import OPTIONS
        if not SPECULATE:
            return
        attempts = range(first, len(self.dampings))
        self._fork("banded", attempts, self._banded)
        needed = []
        for attempt in attempts:
            status, result, _ = self.cache[("banded", attempt)]
            if status != "ok" or result is None:
                continue
            _, candidate, alpha = result
            if OPTIONS.accept_short_banded and candidate is not None and alpha < .25:
                # Its true decrease is checked in the parent before a conic
                # solve is needed. Subsequent conic retry numbers depend on
                # that result, so leave them to the sequential consumer too.
                break
            if (conic_possible and shape_solves < max_shape_solves
                    and (candidate is None or alpha < .25)):
                shape_solves += 1
                self.conic_numbers[attempt] = shape_solves
                needed.append(attempt)
        self._fork("conic", needed, self._conic)

    def _fork(self, kind, attempts, function):
        context = multiprocessing.get_context("fork")
        running = []
        for attempt in attempts:
            parent, child = context.Pipe(duplex=False)
            process = context.Process(target=_serve, args=(function, attempt, child), daemon=True)
            process.start()
            child.close()
            running.append((attempt, process, parent))
        for attempt, process, connection in running:
            self.cache[(kind, attempt)] = connection.recv()
            connection.close()
            process.join()
