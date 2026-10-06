#!/usr/bin/env python3
"""Read-only joint-witness stage/workload comparisons against the measured A arm.

OriginalGeometry queries are full CPU queries (no solver SDF screening), once
per distinct saved pose array. Residuals and unknowns remain sampled lower
bounds, with no physical/reset equilibrium certificate inferred here.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace
from unittest.mock import patch

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import numpy as np
import trimesh

from omniretargeting._fast.mesh_repair import OriginalGeometry
from omniretargeting._fast.postprocess_workload import workload
from omniretargeting._fast.support_split import ROUND_OFF
from omniretargeting.moving_scene import interpolate_object
from omniretargeting.scene_geometry import SceneQuery, collision_geom_ids, geom_surface_offsets
from scripts.measurement_nonregression import row_observations, _clip_comparison


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def payload(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def input_digest(directory):
    """A pose-only cache is valid only with identical source/geometry evidence."""
    source = payload(directory / 'final_retargeted.npz')
    h = hashlib.sha256()
    fields = ('source_joints_w', 'scene_object_pos_w', 'scene_object_quat_w',
              'contact_role_surface_support', 'surface_contact_environment_w',
              'surface_contact_normals_w', 'surface_contact_mask',
              'surface_contact_surface_ids', 'surface_contact_robot_geoms',
              'surface_contact_robot_local_points')
    for name in fields:
        array = source.get(name)
        h.update(name.encode())
        if array is not None:
            h.update(repr((array.shape, array.dtype.str)).encode())
            h.update(array.tobytes())
    h.update(payload(directory / 'fixed_reference.npz')['qpos'].tobytes())
    h.update((directory / 'static_scene.obj').read_bytes())
    return h.hexdigest()


class StageQueries:
    def __init__(self, directory):
        inputs = np.load(directory / 'prepared.npy', allow_pickle=True).item()
        self.model = mujoco.MjModel.from_xml_path(inputs['model_path'])
        self.source = payload(directory / 'final_retargeted.npz')
        self.surface_inputs = inputs['surface']
        reference = payload(directory / 'fixed_reference.npz')['qpos']
        model = self.model
        geoms = collision_geom_ids(model)
        spacing = inputs['captured_solver_config'].get('collision_spacing_m', .02)
        offsets = [geom_surface_offsets(model, int(g), spacing) for g in geoms]
        obj = SimpleNamespace(model=model, frames=len(reference), coords=SimpleNamespace(
            qi=np.array([], int), di=np.arange(model.nv)), shell_geoms=geoms,
            shell_sample_geom=np.concatenate([np.full(len(p), g, int) for g, p in zip(geoms, offsets)]),
            shell_local=np.concatenate(offsets))
        objects = np.load(directory / 'objects.npy', allow_pickle=True).tolist()
        scene = trimesh.load_mesh(directory / 'static_scene.obj', process=False)
        query = SceneQuery(scene, enclosed_free_space=inputs.get('scene_enclosed_free_space', False), full_geometry=False)
        with patch.dict(os.environ, {'OMNIRETARGETING_GPU_GEOMETRY': '0'}):
            self.geometry = OriginalGeometry(obj, objects, query, manifold=True)
        self.geometry._gpu_allowed = False
        self.geometry.freeze_reference_observations(reference)
        self.geometry.start_workers(len(reference))
        self.records = self.support_records()

    def close(self):
        self.geometry.stop_workers()

    def query_for(self, sid, frame):
        if sid == 0:
            return self.geometry.scene, lambda p: p, 1., np.eye(3)
        moving = self.geometry.moving
        oid = sid - 1
        if moving is None or not 0 <= oid < len(moving.objects):
            return None, None, None, None
        obj = moving.objects[oid]
        translation, rotation, scale = interpolate_object(obj['surface'], frame)
        return obj['query'], lambda p: (p - translation) @ rotation / scale, scale, rotation

    def support_records(self):
        source, model = self.source, self.model
        active = source.get('contact_role_surface_support')
        if active is None:
            return {}
        active = active & source['surface_contact_mask']
        targets = source['surface_contact_environment_w']
        normals = source['surface_contact_normals_w']
        sid = source['surface_contact_surface_ids']
        geoms = source['surface_contact_robot_geoms']
        welds = model.body_weldid[model.geom_bodyid[geoms]]
        records = defaultdict(list)
        # Batch immutable source projections on the static scene; moving
        # projections retain each original material time/transform.
        for surface_id in np.unique(sid[active]):
            frames, samples = np.nonzero(active & (sid == surface_id))
            visits = [None] if surface_id == 0 else np.unique(frames)
            for f in visits:
                select = np.ones(len(frames), bool) if f is None else frames == f
                ff, ss = frames[select], samples[select]
                query, transform, scale, rotation = self.query_for(int(surface_id), 0 if f is None else int(f))
                if query is None:
                    continue
                result = query.query(transform(targets[ff, ss]))
                good = ~result['uncertain'] & np.isfinite(result['clearance']) & (np.abs(result['clearance'] * scale) <= ROUND_OFF)
                for i in np.flatnonzero(good):
                    frame, sample = int(ff[i]), int(ss[i])
                    normal = normals[frame, sample]
                    length = np.linalg.norm(normal)
                    if length > 1e-12:
                        records[frame, int(welds[sample]), int(surface_id)].append(
                            (normal / length, targets[frame, sample], int(result['obstacle_ids'][i])))
        return records

    def summarize(self, poses):
        measured = self.geometry.measure(poses, frames=range(len(poses)))
        summary = self.geometry.summary(measured)
        model = self.model
        depths = {kind: np.zeros((len(poses), model.nbody)) for kind in ('static', 'moving', 'self')}
        excess = {kind: np.zeros_like(value) for kind, value in depths.items()}
        projected_queries = {}
        for index, (key, row) in enumerate(measured['rows']):
            if key[1] == 'self':
                continue
            sid = 0 if key[1] == 'static' else key[2] + 1
            group = sid, 0 if sid == 0 else key[0]
            projection = row['point'] - row['clearance'] * np.asarray(row['direction'])
            projected_queries.setdefault(group, []).append((index, projection))
        components = {}
        for (sid, frame), items in projected_queries.items():
            query, transform, _, _ = self.query_for(sid, frame)
            result = query.query(transform(np.array([p for _, p in items])))
            for i, (index, _) in enumerate(items):
                if not result['uncertain'][i] and np.isfinite(result['clearance'][i]):
                    components[index] = int(result['obstacle_ids'][i])
        worst = {}
        for index, (key, row) in enumerate(measured['rows']):
            frame, kind = key[:2]
            geoms = key[2:] if kind == 'self' else key[-1:]
            depth = max(-row['clearance'], 0.)
            allowance = 0.
            if kind != 'self' and index in components:
                sid = 0 if kind == 'static' else key[2] + 1
                body = int(model.body_weldid[model.geom_bodyid[geoms[0]]])
                n = np.asarray(row['direction'])
                projection = row['point'] - row['clearance'] * n
                if any(np.dot(n, normal) >= 1 - ROUND_OFF and abs(np.dot(projection - target, normal)) <= ROUND_OFF
                       and cid == components[index] for normal, target, cid in self.records.get((frame, body, sid), ())):
                    allowance = .005
            for geom in geoms:
                body = int(model.geom_bodyid[geom])
                depths[kind][frame, body] = max(depths[kind][frame, body], depth)
                excess[kind][frame, body] = max(excess[kind][frame, body], max(depth - allowance, 0.))
                token = kind, body
                if depth > worst.get(token, {}).get('depth_m', 0.):
                    worst[token] = dict(frame=int(frame), depth_m=float(depth), geom=int(geom), allowed_m=allowance,
                                        direction=np.asarray(row['direction']).tolist())
        per_body = {}
        body_ids = np.unique(model.geom_bodyid[collision_geom_ids(model)])
        for kind in depths:
            per_body[kind] = {}
            for body in body_ids:
                value = depths[kind][:, body]
                over = excess[kind][:, body]
                per_body[kind][model.body(int(body)).name] = dict(max_depth_m=float(value.max()),
                    p95_depth_m=float(np.quantile(value, .95)), max_excess_m=float(over.max()),
                    p95_excess_m=float(np.quantile(over, .95)), worst=worst.get((kind, int(body))),
                    reliable=not measured['failures'])
            summary[kind]['max_excess_m'] = float(excess[kind].max())
            summary[kind]['reliable'] = not measured['failures']
        return dict(summary=summary, per_body=per_body, support=self.support(poses),
                    unknown=len(measured['failures']), query_failures=measured['failures'],
                    geometry='full CPU original queries; independent of solver SDF; sampled lower bound; no physical validation')

    def support(self, poses):
        source, model = self.source, self.model
        active = source.get('contact_role_surface_support')
        if active is None:
            return {}
        active = active & source['surface_contact_mask']
        geoms = source['surface_contact_robot_geoms']
        local = source['surface_contact_robot_local_points']
        normal = source['surface_contact_normals_w']
        normal = normal / np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-12)
        target = source['surface_contact_environment_w']
        data = mujoco.MjData(model)
        points = []
        for pose in poses:
            data.qpos[:] = pose
            mujoco.mj_forward(model, data)
            points.append(np.einsum('nij,nj->ni', data.geom_xmat[geoms].reshape(-1, 3, 3), local) + data.geom_xpos[geoms])
        offset = np.array(points) - target
        en = np.sum(offset * normal, axis=-1)
        et = np.linalg.norm(offset - en[..., None] * normal, axis=-1)
        world = np.linalg.norm(offset, axis=-1)
        fps = float(source['framerate'])
        mode = source['surface_contact_modes']
        sid = source['surface_contact_surface_ids']
        sticking = (active[1:] & active[:-1] & (mode[1:] == 'sticking') &
                    (mode[:-1] == 'sticking') & (sid[1:] == sid[:-1]))
        # Use the primary contact audit's unchanged target-relative speed,
        # restricted to source support and grouped by collision owner.
        slip = np.linalg.norm(np.diff(np.array(points) - self.surface_inputs['surface_contact_targets_w'], axis=0), axis=-1) * fps
        velocity = np.diff(offset, axis=0) * fps
        tangent_speed = np.linalg.norm(velocity - np.sum(velocity * normal[1:], axis=-1)[..., None] * normal[1:], axis=-1)
        result = {}
        bodies = model.geom_bodyid[geoms]
        for body in np.unique(bodies):
            mask = active & (bodies == body)[None]
            if not mask.any():
                continue
            values = {}
            for name, array in (('eN', abs(en)), ('eT', et), ('world', world)):
                f, s = np.unravel_index(np.argmax(np.where(mask, array, -np.inf)), mask.shape)
                values[name] = dict(max_m=float(array[mask].max()), p95_m=float(np.quantile(array[mask], .95)),
                                    worst_frame=int(f), worst_sample=int(s), signed_eN_m=float(en[f, s]))
            speed_mask = sticking & (bodies == body)[None]
            for name, array in (('support_target_relative_slip', slip), ('source_projection_vT', tangent_speed)):
                if speed_mask.any():
                    f, s = np.unravel_index(np.argmax(np.where(speed_mask, array, -np.inf)), speed_mask.shape)
                    values[name] = dict(max_m_s=float(array[speed_mask].max()),
                        p95_m_s=float(np.quantile(array[speed_mask], .95)), worst_frame=int(f + 1), worst_sample=int(s))
            result[model.body(int(body)).name] = values
        return result


def ratios(a, e):
    result = {}
    for kind in ('static', 'moving', 'self'):
        entries = {}
        a_pre, a_post, e_main = a['before_repair'], a['final'], e['after_main']
        bodies = a_pre['per_body'][kind]
        def calculate(before, after, main):
            result = {}
            for metric in ('max_depth_m', 'max_excess_m'):
                achieved = before[metric] - after[metric]
                already = before[metric] - main[metric]
                result[metric] = dict(A_repair_improvement_m=achieved, joint_main_improvement_m=already,
                    ratio=already / achieved if achieved > 1e-9 and all(v.get('reliable', True) for v in (before, after, main)) else None,
                    meaning='signed, unclipped; denominator<=1nm or unknown -> undefined')
            return result
        for body in bodies:
            entries[body] = calculate(bodies[body], a_post['per_body'][kind][body], e_main['per_body'][kind][body])
        result[kind] = dict(clip=calculate(a_pre['summary'][kind], a_post['summary'][kind], e_main['summary'][kind]), per_body=entries)
    return result


def verified_reset(directory):
    audit = json.loads((directory / 'training_collision.json').read_text())
    reset = audit['reset']
    frames = audit['inputs']['input_frames']
    failed = set(reset.get('failed_frames', []))
    unknown = set(reset.get('unknown_geometry_frames', []))
    return dict(frames=frames, threshold_only_frames=frames - len(failed),
                confirmed_geometry_frames=frames - len(failed | unknown),
                unknown_frames=len(unknown), failed_frames=sorted(failed),
                unknown_geometry_frames=sorted(unknown), physical_validation='unverified')


def process_time(directory, receipt):
    """Separate measured child time from waiting for the shared flock."""
    log = directory.parent / f'{directory.name}.log'
    match = re.search(r'Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ([\d:.]+)', log.read_text())
    if match is None:
        return dict(process_elapsed_s=None, lock_and_launch_wait_s=None, queue_excluded_end_to_end_s=None)
    value = 0.
    for component in match[1].split(':'):
        value = value * 60 + float(component)
    wait = receipt['production_process_wall_seconds'] - value
    return dict(process_elapsed_s=value, lock_and_launch_wait_s=wait,
                queue_excluded_end_to_end_s=receipt['end_to_end_seconds'] - wait,
                meaning='usr/bin/time child elapsed + audit wall; excludes flock wait and wrapper launch overhead')


def render(result, run_dir):
    """Keep the full workload/transfer record reviewable without hiding failures."""
    lines = ['# F1 主求解与后处理职责测量', '',
             'E=joint_witness；E2=joint_witness,joint_skip_free。深度单位 mm，时间单位 s。',
             '原几何为完整 CPU 查询的采样下界；unknown 不计通过；物理与静止 reset 平衡未验证。', '',
             '| clip | arm/repeat | 端到端 | 获取 | reweight | free | repair |',
             '|---|---|---:|---:|---:|---:|---:|']
    details = {}
    for key, run in result['runs'].items():
        arm, repeat, clip = key.split('/')
        directory = run_dir / arm / 'normal' / repeat / clip
        receipt = json.loads((directory / 'receipt.json').read_text())
        timing = json.loads((directory / 'timing.json').read_text())['stage_seconds']
        details[key] = dict(end_to_end_s=receipt['end_to_end_seconds'], stage_seconds=timing)
        values = [receipt['end_to_end_seconds']] + [timing.get(k, 0.) for k in
                  ('contact_acquisition', 'reweight', 'free_bodies', 'original_mesh_repair')]
        lines.append(f'| {clip} | {arm}/{repeat} | ' + ' | '.join(f'{v:.3f}' for v in values) + ' |')
    lines += ['', '| clip | arm/repeat | reweight 轮数/迭代/接受 | free 迭代/接受 | repair 组/提议/几何尝试/接受 | 拒绝计数 |',
              '|---|---|---|---|---|---|']
    for key, run in result['runs'].items():
        arm, repeat, clip = key.split('/')
        w = run['workload']
        rw = ','.join(f"{r['iterations']}/{r['accepted']}" for r in w['reweight']['attempts'])
        free = ','.join(f"{r['iterations']}/{r['accepted']}" + ('(skip)' if r['skipped'] else '') for r in w['free_bodies'])
        p = w['repair']
        lines.append(f"| {clip} | {arm}/{repeat} | {w['reweight']['rounds']}: {rw} | {free} | "
                     f"{p['groups']}/{p['proposal_attempts']}/{p['geometry_trials']}/{p['accepted']} | {p['rejected_reasons']} |")
    lines += ['', '## 锁等待与实际运行耗时', '',
              'receipt 的生产与端到端墙钟从获取全局锁前计时；原汇总值保留，另用 /usr/bin/time 分离排队及启动开销。', '',
              '| run | receipt 端到端 s | child 运行 s | 锁等待及启动 s | 排队除外端到端 s |',
              '|---|---:|---:|---:|---:|']
    for key, value in details.items():
        directory = run_dir / key.split('/')[0] / 'normal' / '/'.join(key.split('/')[1:])
        receipt = json.loads((directory / 'receipt.json').read_text())
        value.update(process_time(directory, receipt))
        cells = [value[k] for k in ('end_to_end_s', 'process_elapsed_s', 'lock_and_launch_wait_s', 'queue_excluded_end_to_end_s')]
        lines.append(f'| {key} | ' + ' | '.join('N/A' if v is None else f'{v:.3f}' for v in cells) + ' |')
    lines += ['', '## reset 几何：剔除 unknown 后的合格帧', '',
              '原汇总脚本的 reset.passed_frames 仅扣深度失败帧，不扣 unknown；这里保留该原数值并另报可确认计数。', '',
              '| run | 总帧 | 仅深度界合格 | 排除 unknown 的可确认合格 | unknown 帧 |',
              '|---|---:|---:|---:|---:|---:|']
    for key, run in result['runs'].items():
        q = run['reset_geometry']
        lines.append(f"| {key} | {q['frames']} | {q['threshold_only_frames']} | {q['confirmed_geometry_frames']} | {q['unknown_frames']} |")
    lines += ['', '| run | 阶段 | static 深度/超额 | moving 深度/超额 | self 深度/超额 | unknown |',
              '|---|---|---:|---:|---:|---:|']
    for key, run in result['runs'].items():
        for stage, measured in run['stages'].items():
            values = [f"{measured['summary'][k]['max_depth_m']*1000:.4f}/{measured['summary'][k]['max_excess_m']*1000:.4f}"
                      for k in ('static', 'moving', 'self')]
            lines.append(f"| {key} | {stage} | " + ' | '.join(values) + f" | {measured['unknown']} |")
    lines += ['', '## A repair 改善由 joint 主求解完成的比例', '',
              '`(A_before_repair − E_after_main)/(A_before_repair − A_final)`，逐 clip、逐 body，分别计算最大深度与超额；',
              '有符号且不裁剪：负数表示主求解结束时更差，>1 表示已超过 A repair 的改善；分母≤1 nm 或 unknown 为 N/A。', '',
              '| repeat/clip | kind/body | A repair 改善 mm（深度/超额） | joint 主求解改善 mm | 比例（深度/超额） |',
              '|---|---|---:|---:|---:|']
    for key, compared in result['ratios'].items():
        for kind, entries in compared.items():
            for body, values in {'clip': entries['clip'], **entries['per_body']}.items():
                d, e = values['max_depth_m'], values['max_excess_m']
                if body != 'clip' and all(abs(v[k]) < 1e-9 for v in (d, e) for k in
                                          ('A_repair_improvement_m', 'joint_main_improvement_m')):
                    continue
                rr = ['N/A' if v['ratio'] is None else f"{v['ratio']:.4g}" for v in (d, e)]
                lines.append(f"| {key} | {kind}/{body} | {d['A_repair_improvement_m']*1000:.5f}/{e['A_repair_improvement_m']*1000:.5f} | "
                             f"{d['joint_main_improvement_m']*1000:.5f}/{e['joint_main_improvement_m']*1000:.5f} | {'/'.join(rr)} |")
    lines += ['', '## repair 各组 before/after', '', '| run | body | before/after mm | 尝试/接受 | 停止原因 |',
              '|---|---|---:|---:|---|']
    for key, run in result['runs'].items():
        for group in run['workload']['repair']['per_group']:
            before, after = group['before_depth_m'], group['after_depth_m']
            value = 'N/A' if before is None or after is None else f'{before*1000:.4f}/{after*1000:.4f}'
            lines.append(f"| {key} | {group['body']} | {value} | {group['proposal_attempts']}/{group['accepted']} | {group['stop_reasons']} |")
    lines += ['', '## 支撑 eN/eT 与世界误差', '', '| run/stage | body | eN P95/max mm | eT P95/max mm | world P95/max mm |',
              '|---|---|---:|---:|---:|']
    for key, run in result['runs'].items():
        for stage in ('after_main', 'final'):
            for body, values in run['stages'][stage]['support'].items():
                cells = [f"{values[k]['p95_m']*1000:.3f}/{values[k]['max_m']*1000:.3f}" for k in ('eN', 'eT', 'world')]
                lines.append(f'| {key}/{stage} | {body} | ' + ' | '.join(cells) + ' |')
    lines += ['', '| run/stage | body | support slip P95/max m/s | projection vT P95/max m/s |',
              '|---|---|---:|---:|']
    for key, run in result['runs'].items():
        for stage in ('after_main', 'final'):
            for body, values in run['stages'][stage]['support'].items():
                cells = []
                for name in ('support_target_relative_slip', 'source_projection_vT'):
                    v = values.get(name)
                    cells.append('N/A' if v is None else f"{v['p95_m_s']:.5f}/{v['max_m_s']:.5f}")
                lines.append(f'| {key}/{stage} | {body} | ' + ' | '.join(cells) + ' |')
    lines += ['', '## skip-free 相对 joint 的同口径 P1–P3 对照', '']
    for clip, compared in result['skip_vs_joint'].items():
        for priority in ('1', '2', '3'):
            block = compared['priorities'][priority]
            lines.append(f"{clip} P{priority}: worse={len(block['worse'])}, better={len(block['better'])}, flat={block['flat_count']}, missing={block['missing_count']}")
            lines += [f"- {r['key']}: {r['A']:.6g} → {r['candidate']:.6g}，Δ={r['delta']:.6g} {r['unit']}" for r in block['worse']]
            lines.append('')
    lines += ['逐帧/逐部位最坏点、原几何未知、free before/after、budgets（含 relation_*、layout_flip_*）、skipped_tasks 的完整记录见 joint_workload.json。',
              'A 对 E/E2 的非退化表见 measurement_summary.md；上述采样几何值不构成严格物理验收。', '']
    result['times'] = details
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--clips', default='sit180,omomo_78,chair150')
    parser.add_argument('--audit-workers', type=int, default=8)
    args = parser.parse_args()
    from omniretargeting._fast import parallel_geometry
    parallel_geometry.WORKERS = max(1, args.audit_workers)
    result = {'runs': {}, 'ratios': {}, 'skip_vs_joint': {}}
    cache = args.output_dir / 'stage_cache'
    for clip in args.clips.split(','):
        queries = None
        context = None
        try:
            for arm in ('A', 'E', 'E2'):
                for repeat_dir in sorted((args.run_dir / arm / 'normal').glob('repeat_*')):
                    directory = repeat_dir / clip
                    if not (directory / 'final_retargeted.npz').exists():
                        continue
                    digest = input_digest(directory)
                    if context is not None and digest != context:
                        raise ValueError(f'Inconsistent immutable inputs: {directory}')
                    context = digest
                    if queries is None:
                        queries = StageQueries(directory)
                    report = json.loads((directory / 'fast_report.json').read_text())
                    stages = {}
                    paths = {'after_main': 'after_main.npz', 'before_free': 'before_free.npz',
                             'before_repair': 'before_mesh_repair.npz', 'final': 'final_retargeted.npz'}
                    for stage, filename in paths.items():
                        path = directory / filename
                        if not path.exists() and stage == 'before_free' and arm == 'E2':
                            path = directory / 'before_mesh_repair.npz'
                        poses = payload(path)['qpos']
                        token = hashlib.sha256(context.encode() + poses.tobytes()).hexdigest()
                        saved = cache / clip / f'{token}.json'
                        if not saved.exists():
                            print(f'STAGE {clip} {arm} {repeat_dir.name} {stage}', flush=True)
                            save(saved, queries.summarize(poses))
                        stages[stage] = json.loads(saved.read_text())
                    key = f'{arm}/{repeat_dir.name}/{clip}'
                    result['runs'][key] = dict(stages=stages, workload=workload(report),
                                               immutable_input_sha256=context,
                                               reset_geometry=verified_reset(directory),
                                               budgets=report['stages']['contact_scene'].get('budgets'),
                                               skipped_tasks=report['stages']['contact_scene'].get('skipped_tasks'))
            for repeat_dir in sorted((args.run_dir / 'E' / 'normal').glob('repeat_*')):
                r = repeat_dir.name
                a, e = f'A/{r}/{clip}', f'E/{r}/{clip}'
                if a in result['runs'] and e in result['runs']:
                    result['ratios'][f'{r}/{clip}'] = ratios(result['runs'][a]['stages'], result['runs'][e]['stages'])
            compare = []
            for arm in ('E', 'E2'):
                for repeat_dir in (args.run_dir / arm / 'normal').glob('repeat_*'):
                    directory = repeat_dir / clip
                    if (directory / 'final_retargeted.npz').exists():
                        compare.append(dict(arm='A' if arm == 'E' else 'candidate', observations=row_observations(directory)))
            result['skip_vs_joint'][clip] = _clip_comparison(compare)
        finally:
            if queries is not None:
                queries.close()
        save(args.output_dir / 'joint_workload.json', result)
    print(args.output_dir / 'joint_workload.json', flush=True)
    (args.output_dir / 'joint_summary.md').write_text(render(result, args.run_dir))
    save(args.output_dir / 'joint_workload.json', result)


if __name__ == '__main__':
    main()
