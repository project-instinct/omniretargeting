"""Separate admissibility of a pose from solver progress and convergence."""


def solver_progress(history=None, *, accepted_update_count=None):
    history = [] if history is None else history
    updates = sum(row.get('alpha', 0.) > 0 for row in history)
    phases = {}
    for row in history:
        phase = row.get('phase', 'unspecified')
        item = phases.setdefault(phase, dict(accepted_update_count=0, iterations_reported=0))
        item['iterations_reported'] += 1
        item['accepted_update_count'] += int(row.get('alpha', 0.) > 0)
        item['solver_outcome'] = row.get('priority_status', 'accepted_updates_without_convergence_certificate'
                                        if item['accepted_update_count'] else 'no_update_reported')
    if history:
        outcome = phases[history[-1].get('phase', 'unspecified')]['solver_outcome']
    else:
        outcome = 'not_reported'
    return dict(accepted_update_count=updates if accepted_update_count is None else accepted_update_count,
                solver_outcome=outcome, solver_phases=phases,
                # QP optimality, accepted progress, iteration limits and an LM
                # stopping heuristic do not certify nonlinear convergence.
                convergence_certified=False, hard_infeasibility_certified=False)
