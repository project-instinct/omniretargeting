"""Reuse the diagnostic that decided a built-in contact guard's acceptance."""

from .acquisition_guard import AcquisitionNormalGuard
from .contact_guard import ContactGuard
from .repair_diagnostics import guard_failure

_METHODS = {kind: (kind.accepts, kind.diagnostics)
            for kind in (AcquisitionNormalGuard, ContactGuard)}


def check_guard(guard, poses):
    methods = _METHODS.get(type(guard))
    if (methods is not None and guard.priority is None
            and getattr(guard.accepts, "__func__", None) is methods[0]
            and getattr(guard.diagnostics, "__func__", None) is methods[1]):
        # These accepts methods return precisely diagnostics()['accepted'];
        # neither has another acceptance predicate or a rejection counter.
        detail = guard.diagnostics(poses)
        if detail["accepted"]:
            return True, None
        if "groups" in detail:
            detail = dict(detail, groups=[g for g in detail["groups"] if g.get("violating_frames")])
        return False, {"reason": "immutable acquired contact/continuity",
                       "guard": type(guard).__name__, "constraint": detail}
    if guard.accepts(poses):
        return True, None
    return False, guard_failure(guard, poses)
