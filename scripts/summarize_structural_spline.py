"""Retain every serial clip/repeat/failure in structural-spline comparisons."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np


CLIPS = ['sit180', 'wipe300', 'chair150', 'fridge957', 'omomo_5', 'omomo_78',
         'omomo_80', 'omomo_485', 'omomo_182', 'omomo_254']


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def evidence(path):
    value = read(path)
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                report=value) if value is not None else dict(status='unverified', path=str(path))


def statistics(values):
    return dict(mean=float(np.mean(values)), p95=float(np.quantile(values, .95)),
                min=float(min(values)), max=float(max(values))) if values else None


def collect(run_dir, repeats=2):
    rows, missing, trace = [], [], []
    for arm in 'ABCD':
        for repeat in range(1, repeats + 1):
            for clip in CLIPS:
                out = run_dir / arm / 'normal' / f'repeat_{repeat}' / clip
                receipt = read(out / 'receipt.json')
                if receipt is None:
                    missing.append(dict(arm=arm, repeat=repeat, clip=clip))
                    continue
                timing = read(out / 'timing.json') or {}
                report = read(out / 'fast_report.json') or {}
                stage = report.get('stages', {}).get('contact_scene', {})
                supplements = [read(path) for path in sorted(out.glob('supplemental_receipt*.json'))]
                supplement = supplements[-1] if supplements else {}
                extra_seconds = sum(value.get('seconds', value.get('supplemental_seconds', 0.))
                                    for value in supplements)
                total = receipt['end_to_end_seconds'] + extra_seconds
                row = dict(arm=arm, repeat=repeat, clip=clip,
                    end_to_end_seconds=total, initial_receipt_seconds=receipt['end_to_end_seconds'],
                    supplemental_audit_seconds=extra_seconds if supplement else None,
                    supplemental_audits=supplement, supplemental_audit_attempts=supplements,
                    production_process_wall_seconds=receipt['production_process_wall_seconds'],
                    timing=timing, source_stage=report.get('stages', {}).get('source_pose'),
                    solver_returncode=receipt['solver_returncode'],
                    audits=receipt['acceptance'], status=report.get('status'),
                    budgets=stage.get('budgets'), skipped_tasks=stage.get('skipped_tasks'),
                    spline=stage.get('spline'),
                    geometry=evidence(out / 'scene_metrics.json'),
                    training_collision=evidence(out / 'training_collision.json'),
                    contact=evidence(out / 'contact_metrics.json'),
                    smoothness=evidence(out / 'joint_smoothness.json'),
                    supplemental_smoothness=evidence(out / 'joint_smoothness_supplemental.json'),
                    quality=evidence(out / 'spline_quality.json'),
                    artifact_directory=str(out), quality_pass=False)
                row['quality_pass_reason'] = 'Requires original-source per-part, geometry, chatter and contact non-degradation; no pass inferred from process exit. Training/replay excluded by user; physics/reset remain unverified.'
                rows.append(row)
        for clip in CLIPS:
            out = run_dir / arm / 'trace' / 'repeat_1' / clip
            receipt = read(out / 'receipt.json')
            if receipt is not None:
                report = read(out / 'fast_report.json') or {}
                trace.append(dict(arm=arm, clip=clip, artifact_directory=str(out),
                    receipt=receipt, performance=report.get('performance'),
                    solver_sizes=evidence(out / 'solver_sizes_trace.json'),
                    spline=report.get('stages', {}).get('contact_scene', {}).get('spline')))
    comparisons = []
    for clip in CLIPS:
        for arm in 'BCD':
            a = [r for r in rows if r['clip'] == clip and r['arm'] == 'A']
            b = [r for r in rows if r['clip'] == clip and r['arm'] == arm]
            complete = len(a) == len(b) == repeats
            paired = []
            for base in a:
                candidate = next((r for r in b if r['repeat'] == base['repeat']), None)
                if candidate is not None:
                    paired.append(dict(repeat=base['repeat'],
                        speed_ratio=base['end_to_end_seconds'] / candidate['end_to_end_seconds'],
                        A_seconds=base['end_to_end_seconds'], candidate_seconds=candidate['end_to_end_seconds'],
                        baseline_solver_exit=base['solver_returncode'],
                        candidate_solver_exit=candidate['solver_returncode']))
            comparisons.append(dict(clip=clip, arm=arm, complete=complete,
                A_seconds=statistics([r['end_to_end_seconds'] for r in a]),
                candidate_seconds=statistics([r['end_to_end_seconds'] for r in b]),
                paired=paired, speed_statistics=statistics([p['speed_ratio'] for p in paired]),
                failures=sum(r['solver_returncode'] != 0 for r in b),
                production_non_degradation_certified=False))
    return dict(schema_version=1, generated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                run_directory=str(run_dir), manifest=read(run_dir / 'manifest.json'),
                expected_normal_runs=40 * repeats, completed_normal_runs=len(rows), missing=missing,
                rows=rows, comparisons=comparisons, diagnostic_trace=trace,
                default_changed=False, aggregate_quality_pass=False,
                failures_included_in_timing=True, provisional=bool(missing),
                no_2x_claim_without_complete_quality_evidence=True)


def markdown(result):
    lines = ['# 结构性加速实测', '',
             '状态：实验路径，未设为默认。失败、unknown、未物理验证均不计通过。', '',
             f"正常计时完成 {result['completed_normal_runs']}/{result['expected_normal_runs']} 条；trace 单独记录。", '',
             '| clip | 组 | A 总时间 mean / max (s) | 候选 mean / max (s) | A/候选耗时比（含失败） | 运行失败 | 完整重复 |',
             '|---|---|---:|---:|---:|---:|---|']
    for row in result['comparisons']:
        def duration(key):
            value = row[key]
            return '缺失' if value is None else f"{value['mean']:.3f} / {value['max']:.3f}"
        speed = row['speed_statistics']
        lines.append(f"| {row['clip']} | {row['arm']} | {duration('A_seconds')} | {duration('candidate_seconds')} | "
            f"{speed['mean']:.3f}× | {row['failures']} | {row['complete']} |" if speed else
            f"| {row['clip']} | {row['arm']} | {duration('A_seconds')} | 缺失 | 缺失 | {row['failures']} | False |")
    lines += ['', '逐 clip/部位的 mean/P95/max、最差帧、持续区间、budgets、skipped_tasks 和审计原文在配套 JSON 中。',
              '速度比包含验收和失败成本；仅有速度改善不能作为采用依据。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = collect(args.run_dir.resolve(), args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    args.output.with_suffix('.md').write_text(markdown(result))
    print(json.dumps(dict(completed=result['completed_normal_runs'], missing=len(result['missing']))))


if __name__ == '__main__':
    main()
