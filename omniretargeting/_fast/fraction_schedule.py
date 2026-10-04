"""Start a fraction's pure restoration only when it is requested.

Fractions retain fork isolation and transfer only consumed rejection
counters. Origin, step and physical guard bounds are immutable during the
shortening loop, as required by FractionRestorations itself.
"""

from .shortening import FractionRestorations
from .speculation import GuardedCall


class DeferredFractionRestorations(FractionRestorations):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._launched_fractions = set()

    def _launch(self, fraction):
        self.jobs[fraction] = GuardedCall(self.restore_function, (
            self.objective, self.origin, self.step * (self.alpha * fraction)), self.guards)
        self.stats['launched'] += 1
        self._launched_fractions.add(fraction)

    def __enter__(self):
        return self

    def restore(self, fraction):
        if (self.enabled and fraction not in self._launched_fractions
                and fraction in self.fractions[:self.limit]):
            self._launch(fraction)
        return super().restore(fraction)


def restorations_type(reference=FractionRestorations):
    """An opt-in scheduling experiment; the numerical restoration is unchanged."""
    from .runtime_options import OPTIONS
    return DeferredFractionRestorations if getattr(OPTIONS, 'lazy_fraction_restorations', False) else reference
