"""Independent, explicitly recorded switches for runtime regression ablations.

Defaults use source-relative chatter and sustained source-layout checks.
Both parallel switches passed six-clip byte-equivalence validation.
Signed layout restoration in mesh repair is unconditional defect correction.
Repair preserves the entry world velocity/acceleration window budgets of
source support and unresolved hold endpoints, including their handoffs.
Disable repair_support_continuity only for same-source validation ablations.
Numerical experiments require explicit opt-in until all six clips pass.
reference_derivative_budget restores the retired per-stencil derivative envelope.
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
    direct_repair_restoration: bool = False
    conic_trust_damping: bool = False
    banded_temporal_direction: bool = False
    banded_guard_direction: bool = False
    adaptive_banded_direction: bool = False
    correction_budget: bool = False
    reference_derivative_budget: bool = False
    parallel_shortening: bool = True
    parallel_repair_restore: bool = True
    repair_support_continuity: bool = True
    accept_short_banded: bool = False
    active_joint_bounds: bool = False
    window_qp: bool = False
    local_conic_restoration: bool = False
    direct_acquisition_qp: bool = False
    objective_soc: bool = False
    frame_step_lengths: bool = False
    priority_clearance: bool = False
    priority_zero_yield_descent: bool = False
    windowed_priority: bool = False
    window_geometry_generation: bool = False
    seed_continuity: bool = False
    coarse_acquisition: bool = False
    fast_self_collision: bool = True
    bulk_cpu_geometry: bool = True
    barrier_acquisition: bool = False
    continuity_hinge_acquisition: bool = False
    active_set_acquisition: bool = False
    repair_triage: bool = True
    repair_fast_gate: bool = True
    repair_parallel_groups: bool = True
    repair_defect_windows: bool = False
    repair_working_set: bool = True
    repair_geometry_reduction: bool = True
    repair_main_cache: bool = True
    whole_qoco_proposals: bool = False
    gpu_conic: bool = False
    # Exact assembly validated on nine clips; only used below 600 frames.
    whole_canonical: bool = True
    stationary_window: bool = True
    loose_reweight: bool = False
    soft_acquisition: bool = False
    soft_acquisition_v4: bool = False
    soft_acquisition_v5: bool = False
    performance_trace: bool = False
    # Authorized repair/free ablations; each changes the feasible/accepted path.
    abl_repair_restore_layout: bool = False
    abl_repair_local_angle: bool = False
    abl_repair_ratchet: bool = False
    abl_contact_regression: bool = False
    # Independent, opt-in acquisition ablations; never change defaults.
    abl_window_layout_deadband: bool = False
    abl_normal_slack: bool = False
    abl_normal_slack_1e5: bool = False

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
