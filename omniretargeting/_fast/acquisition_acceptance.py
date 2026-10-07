"""Opt-in guard-feasible step refinement for acquisition experiments.

Dyadic backtracking may discard almost half a useful conic direction. Refine
between its accepted fraction and the preceding rejected one, using the exact
original guards. This uses no additional conic solve or relaxed tolerance.
"""


def refine_fraction(objective, poses, direction, guards, candidate, alpha,
                    rejected_alpha, *, checks=4, observe=None):
    """Keep the largest explicitly accepted trial among four bisection probes.

    No convexity or monotonicity of the nonlinear feasible set is assumed:
    every returned trajectory was checked by every original guard. Fractions
    below .25 retain the established restoration path instead of refinement.
    """
    if alpha < .25 or alpha >= rejected_alpha:
        return candidate, alpha
    low, high = float(alpha), float(rejected_alpha)
    step = direction.reshape(objective.frames, objective.width)
    for _ in range(checks):
        middle = (low + high) / 2
        trial = objective.project(objective.coords.integrate(poses, step, middle))
        accepted = all(guard.accepts(trial) for guard in guards)
        if observe is not None:
            observe(middle, accepted)
        if accepted:
            candidate, low = trial, middle
        else:
            high = middle
    return candidate, low
