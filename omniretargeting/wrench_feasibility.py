"""Six-dimensional dual certificates for large contact-wrench cones.

Farkas' lemma replaces a potentially expensive polyhedral conversion by a
separator search. The only decision variables are the six coordinates of a
net-wrench separating plane. No individual contact force is estimated.
"""

from itertools import product

import numpy as np
from scipy.optimize import linprog


class DualWrenchCone:
    def __init__(self, contacts, *, origin=(0.0, 0.0, 0.0)):
        origin = np.asarray(origin, float)
        if origin.shape != (3,) or not np.isfinite(origin).all():
            raise ValueError("A dual wrench cone needs a finite moment origin")
        rays, self.excluded = [], []
        for contact in contacts:
            block = contact.rays(origin, 1.0)
            if block is None:
                self.excluded.append(contact.name)
            else:
                rays.extend(block)
        self.rays = np.asarray(rays, float).reshape(-1, 6)
        if len(self.rays):
            self.rays /= np.linalg.norm(self.rays, axis=1, keepdims=True)

    def _separator(self, wrench):
        if not len(self.rays):
            return -np.sign(wrench)
        result = linprog(
            wrench,
            A_ub=-self.rays,
            b_ub=np.zeros(len(self.rays)),
            bounds=[(-1.0, 1.0)] * 6,
            method="highs",
            options={
                "primal_feasibility_tolerance": 1e-9,
                "dual_feasibility_tolerance": 1e-9,
            },
        )
        if not result.success:
            raise RuntimeError(f"Net-wrench separator failed: {result.message}")
        return result.x

    def check(self, wrench, uncertainty=0.0, *, tolerance=1e-8):
        wrench = np.asarray(wrench, float)
        radius = np.broadcast_to(np.asarray(uncertainty, float), (6,))
        if (
            wrench.shape != (6,)
            or not np.isfinite(wrench).all()
            or not np.isfinite(radius).all()
            or np.any(radius < 0)
            or not np.isfinite(tolerance)
            or tolerance < 0
        ):
            raise ValueError(
                "Net wrench and uncertainty must be finite and nonnegative"
            )
        separator = self._separator(wrench)
        violation = float(-separator @ wrench)
        nominal = violation <= tolerance
        if not np.any(radius):
            status = "feasible" if nominal else "infeasible"
        elif violation - np.abs(separator) @ radius > tolerance:
            status = "infeasible"
        elif not nominal:
            status = "uncertain"
        else:
            # Convexity: the entire uncertainty box is inside iff its vertices
            # are inside. If a corner fails, do not treat it as proof that all
            # admissible demands fail. Correlated uncertainty is not invented.
            axes = np.flatnonzero(radius)
            status = "feasible"
            for signs in product((-1.0, 1.0), repeat=len(axes)):
                corner = wrench.copy()
                corner[axes] += np.asarray(signs) * radius[axes]
                plane = self._separator(corner)
                if plane @ corner < -tolerance:
                    status = "uncertain"
                    break
        return {
            "nominal_feasible": nominal,
            "status": status,
            "max_scaled_violation": max(0.0, violation),
            "excluded_geometry": list(self.excluded),
            "method": "six_dimensional_dual_separator",
        }
