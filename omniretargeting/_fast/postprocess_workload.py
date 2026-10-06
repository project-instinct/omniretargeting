"""Read-only workload reduction of the existing downstream reports."""
from collections import Counter, defaultdict


def workload(report):
    stage = report.get('stages', {}).get('contact_scene', {})
    history = stage.get('history', [])
    rounds = stage.get('reweight_rounds', [])
    labels = sorted({r['phase'] for r in history if r.get('phase', '').startswith('reweight_')})
    reweight = []
    for label in labels:
        index = int(label.split('_')[-1]) - 1
        budget = rounds[index] if index < len(rounds) else {}
        reweight.append(dict(label=label, iterations=sum(r.get('phase') == label for r in history),
                             accepted=budget.get('rejected') != label))
    free = []
    for item in stage.get('penetration_stages', []):
        if item.get('stage') != 'free_bodies':
            continue
        free.append(dict(skipped=bool(item.get('skipped')), accepted=bool(item.get('accepted')),
                         iterations=sum(r.get('phase') == 'free_bodies' for r in history),
                         before=item.get('before'), after=(item.get('candidate') if item.get('accepted') else item.get('before')),
                         candidate=item.get('candidate'), rejected=item.get('rejected', [])))
    repair = stage.get('original_mesh_repair') or {}
    groups = defaultdict(list)
    rejected = Counter(ContactGuard=0, ratchet=0, regression=0, no_admissible_descent=0)
    for record in repair.get('history', []):
        groups[record.get('contact_group')].append(record)
        if record.get('stop_reason') == 'no_admissible_descent':
            rejected['no_admissible_descent'] += 1
        for failure in record.get('rejected_limits', []):
            if 'ContactGuard' in str(failure):
                rejected['ContactGuard'] += 1
            else:
                rejected['other_limits'] += 1
    trials = report.get('repair_trial_workload', [])
    for trial in trials:
        if trial.get('regressions'):
            rejected['ratchet'] += bool(trial.get('ratchet_regressions', trial['regressions']))
            rejected['regression'] += bool(trial.get('new_collision_regressions', 0))
            rejected['regression_keys'] += trial['regressions']
        if trial.get('unknown'):
            rejected['unknown_query'] += 1
        if trial.get('complete') and not trial.get('descent'):
            rejected['no_descent'] += 1
    if not trials:
        rejected['ratchet'] = rejected['regression'] = None
        rejected['terminal_regression_observations'] = sum(bool(r.get('collision_regressions')) for rows in groups.values() for r in rows)
    summarized = []
    for group, rows in groups.items():
        predicted = [r.get('model_progress', {}) for r in rows if r.get('model_progress')]
        first = predicted[0] if predicted else {}
        last = predicted[-1] if predicted else {}
        before = first.get('entry_depth_m')
        after = last.get('entry_depth_m')
        if after is not None:
            after -= last.get('realized_decrease_m', 0.)
        group_trials = [t for t in trials if t.get('group') == group]
        summarized.append(dict(body=group, proposal_attempts=len(rows), geometry_trials=len(group_trials) if trials else None,
                               accepted=sum(r.get('alpha', 0) > 0 for r in rows), before_depth_m=before, after_depth_m=after,
                               stop_reasons=dict(Counter(r.get('stop_reason', 'accepted' if r.get('alpha', 0) else 'no_direction') for r in rows))))
    return dict(reweight=dict(triggered=bool(labels), rounds=len(labels), attempts=reweight),
                free_bodies=free, repair=dict(scheduled_groups=repair.get('scheduled_groups', []),
                groups=len(groups), proposal_attempts=sum(len(r) for r in groups.values()),
                geometry_trials=len(trials) if trials else None,
                accepted=sum(r.get('alpha', 0) > 0 for rows in groups.values() for r in rows),
                per_group=summarized, rejected_reasons=dict(rejected), before=repair.get('before'), after=repair.get('after'),
                counter_semantics='ratchet counts failed candidate checks; regression_keys counts failing keys; guard rejects are pre-geometry; overlapping reasons are not summed'))
