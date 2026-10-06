"""Contact material signals use source budgets and the full filter context."""
import copy

import numpy as np

from omniretargeting.motion_budgets import high_frequency, chatter_budget, CHATTER_EXCESS_TOLERANCE
from scripts.review_spline_quality import _contact_chatter


def _signals():
    count = 45
    source = np.column_stack((np.linspace(0, .1, count), np.zeros((count, 2))))
    active = np.zeros(count, bool)
    active[18:28] = True
    binding = dict(part='foot', kind='surface', binding_index=2, active=active,
                   world_positions=source.copy(), source_world_positions=source.copy(),
                   source_provenance='original skin from the immutable prepared plan')
    signals = {role: {'surface:foot:binding_2': copy.deepcopy(binding)}
               for role in ('reference', 'final', 'baseline', 'baseline_reference')}
    return signals


def test_contact_filter_keeps_inactive_neighborhood_and_each_axis():
    signals = _signals()
    key = 'surface:foot:binding_2'
    # An off-contact neighboring pulse must still influence the real Gaussian
    # filter at the first contact frame; replacing inactive points hides it.
    signals['final'][key]['world_positions'][17, 0] += .04
    reference = signals['reference'][key]['world_positions']
    correction = signals['final'][key]['world_positions'] - reference
    content = np.abs(high_frequency(correction))
    excess = content - chatter_budget(reference)
    active = signals['final'][key]['active']
    expected = np.any(excess > CHATTER_EXCESS_TOLERANCE, axis=1) & active
    report = _contact_chatter(signals, 100)
    row = report['roles']['final_minus_reference'][key]
    assert expected[18] and row['active_violating_frames'] == int(expected.sum())
    assert row['active_per_coordinate'][0]['max'] == content[active, 0].max()
    assert row['active_per_coordinate'][1]['max'] == row['active_per_coordinate'][2]['max'] == 0
    delta = report['final_against_A']['metrics']['contact_correction_high_frequency_m']
    assert delta[key + '/axis_0']['summary_increased']
    assert not delta[key + '/axis_1']['summary_increased']
    assert delta[key + '/axis_0']['final']['worst_frame'] == 117


def test_reference_noise_is_audited_against_original_skin_not_rebased():
    signals = _signals()
    key = 'surface:foot:binding_2'
    for role in ('reference', 'final'):
        signals[role][key]['world_positions'][22, 2] += .035
    report = _contact_chatter(signals, 0)
    assert report['roles']['final_minus_reference'][key]['whole_trajectory']['violating_frames'] == 0
    source = report['original_source_audits']['reference'][key]
    assert source['status'] == 'measured' and source['report']['violating_frames'] > 0
    assert report['original_source_audits']['final'][key]['report']['max_excess'] > CHATTER_EXCESS_TOLERANCE


def test_missing_endpoint_source_remains_unverified_and_bindings_do_not_average():
    signals = _signals()
    for bindings in signals.values():
        endpoint = copy.deepcopy(next(iter(bindings.values())))
        endpoint.update(kind='endpoint', binding_index=0, source_world_positions=None,
                        source_provenance='unverified: no same-material source correspondence')
        bindings['endpoint:foot:offset_0'] = endpoint
    signals['final']['endpoint:foot:offset_0']['world_positions'][23, 1] += .02
    report = _contact_chatter(signals, 0)
    assert report['original_source_audits']['reference']['endpoint:foot:offset_0']['status'] == 'unverified'
    assert report['roles']['final_minus_reference']['endpoint:foot:offset_0']['active_violating_frames'] > 0
    assert report['roles']['final_minus_reference']['surface:foot:binding_2']['active_violating_frames'] == 0
    assert report['collision_free_certificate'] is False
