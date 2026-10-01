"""Independent, explicitly recorded switches for runtime regression ablations.

Defaults preserve the current source-layout semantics and exact caches.
Both parallel switches passed six-clip byte-equivalence validation.
Signed layout restoration in mesh repair is unconditional defect correction.
Repair preserves the entry world velocity/acceleration window budgets of
source support and unresolved hold endpoints, including their handoffs.
Disable repair_support_continuity only for same-source validation ablations.
Numerical experiments require explicit opt-in until all six clips pass.
The repair rotation bound is correction locality, not a source posture cap.
"""
from dataclasses import asdict, dataclass
import os


@dataclass(frozen=True)
class RuntimeOptions:
    batch2: bool = False
    outside_proof: bool = False
    deferred_retry: bool = False
    witness_memory: bool = False
    body_rotation_report_only: bool = False
    frozen_depth: bool = False
    original_clearance: bool = False
    repair_aggregate: bool = False
    off_surface_approach_only: bool = False
    submillimeter_repair: bool = False
    reuse_original_geometry: bool = False
    lazy_repair_restoration: bool = False
    conic_trust_damping: bool = False
    correction_budget: bool = False
    parallel_shortening: bool = True
    parallel_repair_restore: bool = True
    repair_support_continuity: bool = True
    accept_short_banded: bool = False
    active_joint_bounds: bool = False

    @classmethod
    def from_environment(cls):
        disabled = set(filter(None, os.environ.get('OMNIRETARGETING_FAST_DISABLE', '').split(',')))
        enabled = set(filter(None, os.environ.get('OMNIRETARGETING_FAST_ENABLE', '').split(',')))
        if disabled & enabled:
            raise ValueError(f'Conflicting fast ablation switches: {sorted(disabled & enabled)}')
        unknown = (disabled | enabled) - cls.__dataclass_fields__.keys()
        if unknown:
            raise ValueError(f'Unknown fast ablation switches: {sorted(unknown)}')
        return cls(**{**{name: False for name in disabled}, **{name: True for name in enabled}})

    def describe(self):
        return asdict(self)


OPTIONS = RuntimeOptions.from_environment()
