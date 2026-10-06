"""Failed/unknown clips remain in the denominator and cannot become passes."""
import json

import pytest

from scripts.summarize_structural_spline import collect, markdown


def test_failed_clip_cost_and_unknown_audits_are_retained(tmp_path):
    for arm, durations, code in [('A', [100., 110.], 0), ('B', [30., 40.], 1)]:
        for repeat, duration in enumerate(durations, 1):
            out = tmp_path / arm / 'normal' / f'repeat_{repeat}' / 'sit180'
            out.mkdir(parents=True)
            (out / 'receipt.json').write_text(json.dumps(dict(
                end_to_end_seconds=duration, production_process_wall_seconds=duration - 5,
                solver_returncode=code, acceptance=dict(status='unknown_query'))))
            (out / 'supplemental_receipt.json').write_text(json.dumps(dict(seconds=7.)))
    result = collect(tmp_path)
    assert result['completed_normal_runs'] == 4
    assert len(result['missing']) == 76
    comparison = next(r for r in result['comparisons'] if r['clip'] == 'sit180' and r['arm'] == 'B')
    assert comparison['failures'] == 2
    assert comparison['complete']
    assert comparison['candidate_seconds']['mean'] == 42.
    assert comparison['paired'][0]['speed_ratio'] == pytest.approx(107 / 37)
    assert all(not row['quality_pass'] for row in result['rows'])
    assert not result['aggregate_quality_pass'] and not result['default_changed']
    assert '含失败' in markdown(result)


def test_trace_never_enters_normal_speed_statistics(tmp_path):
    out = tmp_path / 'B' / 'trace' / 'repeat_1' / 'sit180'
    out.mkdir(parents=True)
    (out / 'receipt.json').write_text(json.dumps(dict(end_to_end_seconds=.1, solver_returncode=0)))
    (out / 'fast_report.json').write_text(json.dumps(dict(performance=dict(iterations=12))))
    result = collect(tmp_path)
    assert result['completed_normal_runs'] == 0
    assert len(result['diagnostic_trace']) == 1
    assert all(row['speed_statistics'] is None for row in result['comparisons'])


def test_failed_supplement_and_retry_both_count_in_total(tmp_path):
    out = tmp_path / 'A' / 'normal' / 'repeat_1' / 'sit180'
    out.mkdir(parents=True)
    (out / 'receipt.json').write_text(json.dumps(dict(
        end_to_end_seconds=100., production_process_wall_seconds=90.,
        solver_returncode=0, acceptance={})))
    (out / 'supplemental_receipt.json').write_text(json.dumps(dict(seconds=2., status='failed')))
    (out / 'supplemental_receipt_retry_2.json').write_text(json.dumps(dict(seconds=7., status='completed')))
    row = collect(tmp_path)['rows'][0]
    assert row['end_to_end_seconds'] == 109.
    assert row['supplemental_audit_seconds'] == 9.
    assert len(row['supplemental_audit_attempts']) == 2
