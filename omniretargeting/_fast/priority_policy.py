"""Exact strategy identity for repeated-failure deduplication, not infeasibility."""

import clarabel

from .qp_types import settings

TRUST_SHRINK = .5
CERTIFICATE_STEP_TOLERANCE = 1e-8


def _settings_record(value):
    return {name: getattr(value, name) for name in dir(value)
            if not name.startswith('_') and not callable(getattr(value, name))}


def iteration_policy(damping, growth):
    from .runtime_options import OPTIONS
    from .solver import MAX_SHAPE_SOLVES, MIN_DAMPING, MAX_DAMPING, DIAGONAL_FLOOR
    from .proposals import ALPHAS
    from .priority_qp import MAX_ESCALATIONS, ESCALATION
    from .frame_step import MAX_ROUNDS
    from .descent_step import NORMALIZED_PENALTY, PENALTY_GROWTH, STEP_BOX_TOLERANCE
    from .second_order_step import SOC_FRACTIONS, SOC_HALOS, MAX_SOC_SOLVES, SOC_DENSE_FRACTION
    return dict(version=4, initial_damping=float(damping), initial_growth=float(growth),
                max_shape_solves=MAX_SHAPE_SOLVES, alphas=ALPHAS,
                objective_soc=OPTIONS.objective_soc, frame_step_lengths=OPTIONS.frame_step_lengths,
                soc_fractions=SOC_FRACTIONS, soc_halos=SOC_HALOS, soc_max_solves=MAX_SOC_SOLVES,
                soc_dense_fraction=SOC_DENSE_FRACTION, frame_max_rounds=MAX_ROUNDS,
                frame_cap_rule='max(uniform, .5*cap)',
                trust_shrink_rule='trust_fraction *= min(shrink, uniform_alpha*fraction)',
                trust_shrink=TRUST_SHRINK, certificate_step_tolerance=CERTIFICATE_STEP_TOLERANCE,
                diagonal_floor=DIAGONAL_FLOOR, min_damping=MIN_DAMPING, max_damping=MAX_DAMPING,
                max_escalations=MAX_ESCALATIONS, escalation_factor=ESCALATION,
                adaptive_certificate='zero_lower_bound_and_global_kkt_v1',
                strict_certificate='ordered_lp_freezes_and_global_kkt_precision_v2',
                weighted_shortcut='positive_box_lower_bound_outward_reserve_v1',
                zero_yield_descent=OPTIONS.priority_zero_yield_descent,
                descent_certificate='zero_additional_contact_yield_descent_v1',
                descent_penalty=NORMALIZED_PENALTY, descent_penalty_growth=PENALTY_GROWTH,
                descent_box_tolerance=STEP_BOX_TOLERANCE,
                backend='clarabel', backend_version=clarabel.__version__,
                settings=_settings_record(settings()), retry_settings=_settings_record(settings(retry=True)),
                precision_settings=_settings_record(settings(precision=True)))
