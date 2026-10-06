"""Compare immutable source, fixed reference, final and approved A per clip.

Run with --prepared/--reference/--final/--baseline, or --manifest containing
{"clips": [{"clip": ..., "prepared": ..., "reference": ..., "final": ...,
"baseline": ..., "baseline_reference": ..., "audits": {...}}]}.
Audit mappings are role -> kind -> existing audit JSON path. No angular cap or
percentage degradation is introduced. Geometry/physics are never inferred from
pose metrics. Existing reports retain their own certification limitations.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from scripts.review_source_pose import evaluate as source_pose_evaluate
from scripts.review_contact_metrics import evaluate as contact_evaluate
from omniretargeting.motion_budgets import (
    CHATTER_EXCESS_TOLERANCE, chatter_budget, high_frequency,
)
from omniretargeting.pose_relations import (
    LAYOUT_FLIP_MARGIN, LAYOUT_FLIP_WINDOW_FRAMES, PoseRelations,
    RelationEnvelope, heading_axes, landmark_parents,
)


def intervals(mask, start=0):
    """All contiguous affected intervals; absolute frame endpoints inclusive."""
    mask = np.asarray(mask, bool).reshape(-1)
    changes = np.diff(np.r_[False, mask, False].astype(int))
    return [{"start_frame": int(a + start), "end_frame": int(b - 1 + start),
             "frames": int(b - a)}
            for a, b in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1))]


def stats(values, start=0, abnormal=None):
    values = np.asarray(values, float).reshape(-1)
    finite = np.isfinite(values)
    if not finite.any():
        return {"count": 0, "mean": None, "p95": None, "max": None,
                "worst_frame": None, "abnormal_intervals": []}
    indices = np.flatnonzero(finite)
    result = {"count": int(finite.sum()), "mean": float(values[finite].mean()),
              "p95": float(np.quantile(values[finite], .95)),
              "max": float(values[finite].max()),
              "worst_frame": int(indices[np.argmax(values[finite])] + start)}
    result["abnormal_intervals"] = intervals(abnormal, start) if abnormal is not None else []
    return result


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _motion(path, model):
    with np.load(path, allow_pickle=False) as saved:
        q = np.asarray(saved["qpos"], float)
        fps = float(saved.get("framerate", 30.))
        start = int(saved.get("frame_start", 0))
    if q.ndim != 2 or q.shape[1] != model.nq or not len(q):
        raise ValueError(f"{path}: expected nonempty (frames, model.nq) qpos")
    if not np.isfinite(q).all() or fps <= 0:
        raise ValueError(f"{path}: nonfinite qpos or invalid framerate")
    return q, fps, start


class _Kinematics:
    """Minimal adapter to the real RelationEnvelope reporting interface."""
    def __init__(self, model, q):
        data = mujoco.MjData(model)
        self.xpos = np.empty((len(q), model.nbody, 3))
        self.xmat = np.empty((len(q), model.nbody, 3, 3))
        for frame, pose in enumerate(q):
            data.qpos[:] = pose
            mujoco.mj_forward(model, data)
            self.xpos[frame] = data.xpos
            self.xmat[frame] = data.xmat.reshape(-1, 3, 3)

    def points(self, bodies, offsets):
        return self.xpos[:, bodies] + np.einsum("fbik,bk->fbi", self.xmat[:, bodies], offsets)


def _continuous_log(rotations):
    """Continuous SO(3) logarithm branch; never filter quaternion components."""
    vectors = rotations.as_rotvec()
    for frame in range(1, len(vectors)):
        angle = np.linalg.norm(vectors[frame])
        if angle > 1e-10:
            axis = vectors[frame] / angle
        else:
            previous = np.linalg.norm(vectors[frame - 1])
            if previous <= 1e-10:
                continue
            axis = vectors[frame - 1] / previous
        previous_projection = vectors[frame - 1] @ axis
        winding = int(np.rint((previous_projection - angle) / (2 * np.pi)))
        vectors[frame] += winding * 2 * np.pi * axis
    return vectors


def _rotation_signal(rotations):
    return _continuous_log(rotations * rotations[0].inv())


def _angular_steps(rotations):
    """World SO(3) increments used by production root chatter checks."""
    return (rotations[1:] * rotations[:-1].inv()).as_rotvec()


def _chatter(reference_signal, correction, start):
    """Actual candidate-reference correction, existing source-local budget."""
    correction = np.asarray(correction, float)
    content = np.abs(high_frequency(correction))
    budget = chatter_budget(reference_signal)
    excess = content - budget
    if content.ndim == 1:
        content, excess = content[:, None], excess[:, None]
    peak = np.max(excess, axis=1)
    result = stats(np.linalg.norm(content, axis=1), start,
                   np.any(excess > CHATTER_EXCESS_TOLERANCE, axis=1))
    result.update(max_excess=float(peak.max(initial=0)),
                  violating_frames=int(np.count_nonzero(peak > CHATTER_EXCESS_TOLERANCE)),
                  excess_tolerance=CHATTER_EXCESS_TOLERANCE,
                  correction_definition="high_frequency(actual candidate-reference correction)")
    result['per_coordinate'] = [dict(stats(content[:, column], start,
        excess[:, column] > CHATTER_EXCESS_TOLERANCE),
        max_excess=float(excess[:, column].max(initial=0)))
        for column in range(content.shape[1])]
    return result


def _direction(first, second):
    denominator = np.linalg.norm(first, axis=-1) * np.linalg.norm(second, axis=-1)
    cosine = np.divide(np.sum(first * second, axis=-1), denominator,
                       out=np.full(denominator.shape, np.nan), where=denominator > 1e-12)
    return np.arccos(np.clip(cosine, -1, 1))


def _measure(graph, kin, desired_points, names, parents, bodies, offsets, targets,
             desired_rotations, *, source=False):
    points = kin.points(bodies, offsets)
    desired = graph.relations(desired_points, source=source)
    actual = graph.relations(points)
    changes = graph.changes(actual, desired)
    whole, parts = graph.equivalent_angles(changes, desired)
    relation_whole, relation_parts = graph.summary(changes)
    series = {"normalized_landmark_relations": {
        name: relation_parts[:, k] for k, name in enumerate(names)},
        "equivalent_landmark_angles_rad": {name: parts[:, k] for k, name in enumerate(names)},
        "whole_relation": {"whole": relation_whole},
        "whole_equivalent_angle_rad": {"whole": whole},
        "bone_directions_rad": {}, "calibrated_body_orientation_rad": {}}
    for child, parent in enumerate(parents):
        if parent >= 0:
            name = f"{names[parent]} -> {names[child]}"
            series["bone_directions_rad"][name] = _direction(
                points[:, child] - points[:, parent],
                desired_points[:, child] - desired_points[:, parent])
    for target, desired_rotation in zip(targets, desired_rotations):
        body = target["body_id"]
        rotation = kin.xmat[:, body]
        if target.get("parent_id") is not None:
            rotation = kin.xmat[:, target["parent_id"]].transpose(0, 2, 1) @ rotation
        series["calibrated_body_orientation_rad"][target["robot_body"]] = Rotation.from_matrix(
            desired_rotation @ rotation.transpose(0, 2, 1)).magnitude()
    return series


def _layout(envelope, kin, start, *, source):
    _, _, (change, _) = envelope.measure(kin)
    if source:
        sign, axes = envelope.source_layout_sign, envelope.source_axes
    else:
        sign, axes = envelope.layout_sign, envelope.axes
    components = np.einsum("fpk,fak->fpa", change + envelope.reference, axes)
    value = sign * components
    excess = envelope._collapse_excess(value)
    result = {}
    axis_names = ("forward", "left", "up")
    for k, (a, b) in enumerate(envelope.graph.pairs):
        for axis in range(3):
            if not np.any(sign[:, k, axis]):
                continue
            label = f"{envelope.names[a]} -> {envelope.names[b]} / {axis_names[axis]}"
            result[label] = stats(excess[:, k, axis], start, excess[:, k, axis] > envelope.SLACK)
            result[label]["clear_opposite_intervals"] = intervals(
                value[:, k, axis] < -LAYOUT_FLIP_MARGIN, start)
    return result


def _compare(final, baseline, start):
    """Exact deltas against A; no invented angle/percentage allowance."""
    rows, regressions = {}, []
    for metric, parts in final.items():
        rows[metric] = {}
        for part, values in parts.items():
            if metric not in baseline or part not in baseline[metric]:
                raise ValueError(f"A baseline lacks {metric}/{part}")
            before = baseline[metric][part]
            finite = np.isfinite(values) & np.isfinite(before)
            worse = finite & (values > before)
            after_stats, before_stats = stats(values, start, worse), stats(before, start)
            deltas = {key: (after_stats[key] - before_stats[key])
                      if after_stats[key] is not None and before_stats[key] is not None else None
                      for key in ("mean", "p95", "max")}
            increased = any(v is not None and v > 0 for v in deltas.values())
            row = {"A": before_stats, "final": after_stats, "delta": deltas,
                   "summary_increased": increased,
                   "worse_than_A_frames": int(worse.sum()),
                   "worse_than_A_intervals": intervals(worse, start)}
            rows[metric][part] = row
            if increased:
                regressions.append({"metric": metric, "part": part, **deltas})
    return {"rule": "per clip/part mean, P95 and max against approved A; exact deltas; no angular cap",
            "numerical_tolerance_applied": None,
            "summary_non_degradation": not regressions,
            "regressions": regressions, "metrics": rows}


def _source_chatter(graph, kin, source, bodies, offsets, names, targets, desired, start):
    robot_relation = graph.relations(kin.points(bodies, offsets))
    source_relation = graph.relations(source, source=True)
    correction = robot_relation - source_relation
    rows = {}
    for part, name in enumerate(names):
        selected = np.any(graph.pairs == part, axis=1)
        if selected.any():
            rows[name] = _chatter(source_relation[:, selected].reshape(len(source), -1),
                                 correction[:, selected].reshape(len(source), -1), start)
    orientation = {}
    for target, wanted in zip(targets, desired):
        actual = kin.xmat[:, target["body_id"]]
        if target.get("parent_id") is not None:
            actual = kin.xmat[:, target["parent_id"]].transpose(0, 2, 1) @ actual
        wanted, actual = Rotation.from_matrix(wanted), Rotation.from_matrix(actual)
        orientation[target["robot_body"]] = _chatter(
            _angular_steps(wanted), _angular_steps(actual) - _angular_steps(wanted), start + 1)
    return {"landmarks": rows, "calibrated_body_orientation": orientation,
            "scalar_robot_joint_source_correspondence": "unverified: original human joints are not robot qpos",
            "any_source_relative_chatter": any(row["violating_frames"] for row in
                [*rows.values(), *orientation.values()])}


def _correction_chatter(model, reference, candidate, source_roots, source_rotations, start):
    from omniretargeting._sequence.coordinates import Coordinates
    from omniretargeting._fast.continuity import temporal_signals, chatter_reports
    coords = Coordinates(model)
    before_signals = temporal_signals(coords, reference)
    after_signals = temporal_signals(coords, candidate)
    production, _ = chatter_reports(coords, reference, candidate)
    joints = {}
    for joint in range(model.njnt):
        kind = model.jnt_type[joint]
        adr = int(model.jnt_qposadr[joint])
        name = model.joint(joint).name
        if kind in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            row = _chatter(reference[:, adr], candidate[:, adr] - reference[:, adr], start)
            row["budget_provenance"] = "fixed reference; conditional on separate original-source reference audit"
            row["units"] = "rad" if kind == mujoco.mjtJoint.mjJNT_HINGE else "m"
            joints[name] = row
        elif kind == mujoco.mjtJoint.mjJNT_BALL:
            before = Rotation.from_quat(reference[:, adr:adr + 4], scalar_first=True)
            after = Rotation.from_quat(candidate[:, adr:adr + 4], scalar_first=True)
            joints[name] = _chatter(_rotation_signal(before),
                                   _continuous_log(after * before.inv()), start)
    root = {}
    free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    for joint in free:
        adr = int(model.jnt_qposadr[joint])
        before = Rotation.from_quat(reference[:, adr + 3:adr + 7], scalar_first=True)
        after = Rotation.from_quat(candidate[:, adr + 3:adr + 7], scalar_first=True)
        root[model.joint(int(joint)).name or str(joint)] = {
            "translation": _chatter(reference[:, adr:adr + 3],
                                     candidate[:, adr:adr + 3] - reference[:, adr:adr + 3], start),
            "rotation": _chatter(before_signals['root_rotation'],
                after_signals['root_rotation'] - before_signals['root_rotation'], start + 1)}
    authority = {name: dict(violations=int(row['violations'].sum()),
        per_coordinate=[dict(stats(row['excess'][:, column],
            start + int(name == 'root_rotation'), row['violations'][:, column]),
            violating_frames=int(row['violations'][:, column].sum()))
            for column in range(row['excess'].shape[1])])
        for name, row in production.items()}
    return {"joints": joints, "root": root,
            "production_motion_budgets_audit": authority}


def _contact_choices(prepared, motion, inputs, model):
    choices_path = Path(motion).parent / "contact_choices.npz"
    if choices_path.exists():
        with np.load(choices_path, allow_pickle=False) as saved:
            choices = {name: saved[name] for name in saved.files}
    else:
        from omniretargeting._refinement.math import _lock_contact_offsets
        specs = {name: dict(inputs["effectors"]["effectors"][name],
                 body_id=model.body(inputs["effectors"]["effectors"][name]["robot_body"]).id,
                 task=task) for name, task in inputs["tasks"].items()}
        choices = _lock_contact_offsets(model, inputs["initial"], specs)
    return choices


def _contacts(prepared, paths, inputs, model, loaded):
    if not all(key in inputs for key in ("tasks", "effectors", "surface", "roots", "root_orientation_targets")):
        return {role: {"status": "unverified", "reason": "prepared contact plan missing"} for role in paths}
    cache, reports = {}, {}
    for role, path in paths.items():
        choices = _contact_choices(prepared, path, inputs, model)
        key = (loaded[role][0].tobytes(),
               tuple((name, np.asarray(value).tobytes()) for name, value in sorted(choices.items())))
        if key not in cache:
            cache[key] = contact_evaluate(prepared, path, choices)
        reports[role] = dict(status='measured', implementation='scripts.review_contact_metrics.evaluate',
                            report=dict(cache[key], motion=str(path)))
    return reports


def _contact_details(prepared, paths, loaded, inputs, model, start):
    if not all(key in inputs for key in ('tasks', 'effectors', 'surface', 'roots')):
        return dict(status='unverified', reason='prepared contact plan missing')
    from scripts.spline_contact_details import measure
    reports, series, binding_signals = {}, {}, {}
    for role, path in paths.items():
        choices = _contact_choices(prepared, path, inputs, model)
        measured = measure(inputs, model, loaded[role][0], choices, fps=loaded[role][1], start=start)
        binding_signals[role] = {row['id']: row for rows in measured['binding_signals'].values()
                                 for row in rows}
        # A signed plane gap has two failure directions. Keep its sign, and
        # compare absolute separation for A deltas, never count deeper negative
        # penetration as an improvement. Original mesh penetration is separate.
        series[role] = {metric: {part: np.abs(value) if 'signed_gap' in metric else value
                        for part, value in parts.items()} for metric, parts in measured['metrics'].items()}
        reports[role] = dict(units=measured['units'], provenance=measured['provenance'],
            limitations=measured['limitations'], metrics={})
        for metric, parts in series[role].items():
            reports[role]['metrics'][metric] = {}
            for part, values in parts.items():
                row = stats(values, start)
                signed = measured['metrics'][metric][part]
                if 'signed_gap' in metric and np.isfinite(signed).any():
                    row.update(absolute_gap_statistics=True, signed_min=float(np.nanmin(signed)),
                        signed_max=float(np.nanmax(signed)),
                        signed_at_worst_frame=float(signed[row['worst_frame'] - start]))
                binding = measured.get('binding_selection', {}).get(metric, {}).get(part)
                if binding is not None and row['worst_frame'] is not None:
                    row['worst_binding'] = int(binding[row['worst_frame'] - start])
                reports[role]['metrics'][metric][part] = row
    compared = _compare(series['final'], series['baseline'], start)
    return dict(status='measured', roles=reports, final_against_A=compared,
                reference_against_A=_compare(series['reference'], series['baseline_reference'], start)
                    if 'baseline_reference' in series else None,
                chatter=_contact_chatter(binding_signals, start),
                abnormal_interval_rule='framewise contact/drift error above corresponding A; signed gap uses absolute value')


def _contact_chatter(signals, start):
    """Filter continuous material trajectories before selecting contact frames."""
    roles, source_audits, comparisons = {}, {}, {}
    for role, reference_role in (('final', 'reference'), ('baseline', 'baseline_reference')):
        if reference_role not in signals:
            continue
        rows = {}
        for binding, current in signals[role].items():
            reference = signals[reference_role][binding]
            before, after = reference['world_positions'], current['world_positions']
            active = current['active']
            correction = after - before
            content = np.abs(high_frequency(correction))
            excess = content - chatter_budget(before)
            bad = (excess > CHATTER_EXCESS_TOLERANCE).any(axis=1) & active
            masked = np.where(active[:, None], content, np.nan)
            row = dict(part=current['part'], kind=current['kind'], unit='m',
                binding_index=current['binding_index'], whole_trajectory=_chatter(before, correction, start),
                active_contact=stats(np.linalg.norm(masked, axis=1), start, bad),
                active_frames=int(active.sum()), active_violating_frames=int(bad.sum()),
                active_per_coordinate=[stats(masked[:, axis], start,
                    (excess[:, axis] > CHATTER_EXCESS_TOLERANCE) & active) for axis in range(3)],
                filter_context='all original frames; inactive material points retain actual FK',
                reference_budget_conditional_on_original_source_audit=True)
            rows[binding] = row
            comparisons.setdefault(role, {})[binding] = content, excess
        roles[role + '_minus_' + reference_role] = rows
    for role, bindings in signals.items():
        source_audits[role] = {}
        for binding, current in bindings.items():
            source = current['source_world_positions']
            source_audits[role][binding] = (dict(status='unverified', part=current['part'],
                reason=current['source_provenance']) if source is None or not np.isfinite(source).all() else dict(status='measured',
                part=current['part'], report=_chatter(source, current['world_positions'] - source, start),
                provenance=current['source_provenance']))
    compared = None
    if 'final' in comparisons and 'baseline' in comparisons:
        native = {}
        for role in ('final', 'baseline'):
            native[role] = {metric: {f'{binding}/axis_{axis}': values[index][:, axis]
                           for binding, values in comparisons[role].items() for axis in range(3)}
                           for index, metric in enumerate(('contact_correction_high_frequency_m',
                                                          'contact_correction_chatter_excess_m'))}
        compared = _compare(native['final'], native['baseline'], start)
    return dict(status='measured', roles=roles, original_source_audits=source_audits,
                final_against_A=compared, budget_implementation='omniretargeting.motion_budgets',
                source_correspondence='surface skin from environment + original source vector; endpoint unknown',
                collision_free_certificate=False)


def _motion_amplitude(graph, source, kin, bodies, offsets, names, start):
    """Original-time excursions of morphology-normalized landmark relations."""
    signals = {'original_source': graph.relations(source, source=True)}
    signals.update({role: graph.relations(value.points(bodies, offsets)) for role, value in kin.items()})
    report = {}
    for pair, (first, second) in enumerate(graph.pairs):
        label = f'{names[first]} -> {names[second]}'
        rows = {}
        for role, values in signals.items():
            value = values[:, pair]
            extent = np.ptp(value, axis=0)
            rows[role] = dict(range_extent_xyz=extent.tolist(),
                range_extent_norm=float(np.linalg.norm(extent)),
                total_motion_length=float(np.linalg.norm(np.diff(value, axis=0), axis=1).sum()),
                minimum_frames=(np.argmin(value, axis=0) + start).tolist(),
                maximum_frames=(np.argmax(value, axis=0) + start).tolist())
        rows['final_minus_A'] = {key: rows['final'][key] - rows['baseline'][key]
                                for key in ('range_extent_norm', 'total_motion_length')}
        rows['peak_frame_delta_against_A_xyz'] = (np.array(rows['final']['maximum_frames'])
                                                - rows['baseline']['maximum_frames']).tolist()
        report[label] = rows
    return dict(per_pair=report, units='morphology-normalized relation units; original frame IDs',
        certification='diagnostic only; no amplitude percentage or timing tolerance introduced',
        frame_timing_changed=False, signals_filtered=False)


def _audit_bundle(paths, supplied):
    kinds = ("scene", "contact", "smoothness", "self_collision", "reset", "physics")
    defaults = {"scene": "scene_metrics.json", "contact": "contact_metrics.json",
                "smoothness": "joint_smoothness.json", "self_collision": "self_collision.json",
                "reset": "reset_audit.json", "physics": "physics_replay.json"}
    result = {}
    for role, motion in paths.items():
        result[role] = {}
        for kind in kinds:
            explicit = supplied.get(role, {}).get(kind)
            path = Path(explicit) if explicit else Path(motion).parent / defaults[kind]
            # Unqualified final audit filenames cannot be borrowed by reference.
            if role.endswith("reference") and not explicit:
                path = Path(motion).parent / ("reference_" + defaults[kind])
            if not path.exists():
                result[role][kind] = {"status": "unverified", "reason": "audit absent"}
                continue
            report = json.loads(path.read_text())
            recorded_motion = report.get("motion", report.get("after"))
            if recorded_motion and Path(recorded_motion).resolve() != Path(motion).resolve():
                raise ValueError(f"Audit {path} belongs to another motion: {recorded_motion}")
            result[role][kind] = {"status": "reported", "path": str(path),
                                  "sha256": _sha(path), "report": report,
                                  "independent_pass_inferred": False}
    return result


def evaluate(prepared, reference, final, baseline, *, clip=None,
             baseline_reference=None, audits=None):
    prepared, reference, final, baseline = map(Path, (prepared, reference, final, baseline))
    inputs = np.load(prepared, allow_pickle=True).item()
    model = mujoco.MjModel.from_xml_path(inputs["model_path"])
    paths = {"reference": reference, "final": final, "baseline": baseline}
    if baseline_reference:
        paths["baseline_reference"] = Path(baseline_reference)
    loaded = {role: _motion(path, model) for role, path in paths.items()}
    qref, fps, start = loaded["reference"]
    for role, (poses, rate, frame_start) in loaded.items():
        if poses.shape != qref.shape or rate != fps or start != frame_start:
            raise ValueError(f"{role}: compared motions need identical frames, frame_start and framerate")
    stop = start + len(qref)
    names = list(inputs["mapping"])
    bodies = np.array([model.body(inputs["mapping"][name]).id for name in names])
    source_ids = [list(inputs["source_names"]).index(name) for name in names]
    source = np.asarray(inputs["positions"], float)[start:stop, source_ids]
    if source.shape != (len(qref), len(names), 3) or not np.isfinite(source).all():
        raise ValueError("original source landmarks missing/nonfinite or wrong frame range")
    offsets = np.zeros((len(bodies), 3))
    parents = landmark_parents(model, bodies)
    kin, kin_cache = {}, {}
    for role, values in loaded.items():
        key = values[0].tobytes()
        if key not in kin_cache:
            kin_cache[key] = _Kinematics(model, values[0])
        kin[role] = kin_cache[key]
    ref_points = kin["reference"].points(bodies, offsets)
    graph = PoseRelations(parents, ref_points, source)
    if not len(graph.pairs):
        raise ValueError("No nondegenerate mapped landmark relation")
    targets = [dict(target, body_id=model.body(target["robot_body"]).id,
                    parent_id=model.body(target["parent_body"]).id if target.get("parent_body") else None)
               for target in inputs.get("body_orientation_targets", ())]
    desired = [np.asarray(target["rotations"])[start:stop] for target in targets]
    root_id = bodies[parents.index(-1)]
    envelope = RelationEnvelope(bodies, offsets, parents, ref_points, names,
                               heading_axes(kin["reference"].xmat[:, root_id]), source_positions=source)
    series = {role: _measure(graph, value, source, names, parents, bodies, offsets,
                             targets, desired, source=True) for role, value in kin.items()}
    source_reports, source_report_cache = {}, {}
    for role, path in paths.items():
        key = id(kin[role])
        if key not in source_report_cache:
            source_report_cache[key] = source_pose_evaluate(prepared, path)
        source_reports[role] = dict(source_report_cache[key], motion=str(path))
    fixed_orientations = []
    for target in targets:
        orientation = kin["reference"].xmat[:, target["body_id"]]
        if target.get("parent_id") is not None:
            orientation = kin["reference"].xmat[:, target["parent_id"]].transpose(0, 2, 1) @ orientation
        fixed_orientations.append(orientation)
    fixed_series = _measure(graph, kin["final"], ref_points, names, parents, bodies,
                            offsets, targets, fixed_orientations)
    comparisons = {}
    for label, role in (("ref_vs_original_source", "reference"),
                        ("final_vs_original_source", "final")):
        baseline_role = 'baseline_reference' if role == 'reference' else 'baseline'
        comparisons[label] = {"metrics": {metric: {part: stats(value, start,
                            value > series[baseline_role][metric][part]
                            if baseline_role in series else None)
                            for part, value in rows.items()} for metric, rows in series[role].items()},
                            "abnormal_interval_rule": 'framewise error above corresponding A trajectory'
                                if baseline_role in series else 'unverified: A fixed reference missing',
                            "existing_source_pose_audit": source_reports[role],
                            "layout": _layout(envelope, kin[role], start, source=True)}
    baseline_fixed_series = None
    if 'baseline_reference' in kin:
        base_points = kin['baseline_reference'].points(bodies, offsets)
        base_orientations = []
        for target in targets:
            orientation = kin['baseline_reference'].xmat[:, target['body_id']]
            if target.get('parent_id') is not None:
                orientation = kin['baseline_reference'].xmat[:, target['parent_id']].transpose(0, 2, 1) @ orientation
            base_orientations.append(orientation)
        baseline_fixed_series = _measure(graph, kin['baseline'], base_points, names, parents,
            bodies, offsets, targets, base_orientations)
    comparisons["final_vs_fixed_ref"] = {"metrics": {metric: {part: stats(value, start,
                value > baseline_fixed_series[metric][part] if baseline_fixed_series else None)
                for part, value in rows.items()} for metric, rows in fixed_series.items()},
                "abnormal_interval_rule": 'framewise correction above A final versus A fixed reference'
                    if baseline_fixed_series else 'unverified: A fixed reference missing',
                "layout": _layout(envelope, kin["final"], start, source=False)}
    roots = np.asarray(inputs.get("roots", source[:, parents.index(-1)]))[start:stop] if "roots" in inputs else source[:, parents.index(-1)]
    if "root_orientation_targets" not in inputs:
        raise ValueError("Original calibrated root orientations required for source-relative root audit")
    source_rotation = Rotation.from_quat(np.asarray(inputs["root_orientation_targets"])[start:stop], scalar_first=True)
    source_anchor_body = inputs.get('effectors', {}).get('source_root_robot_body')
    source_anchor_id = None if source_anchor_body is None else model.body(source_anchor_body).id
    chatter = {role: _source_chatter(graph, value, source, bodies, offsets, names,
                                    targets, desired, start) for role, value in kin.items()}
    for role, value in kin.items():
        pose = loaded[role][0]
        free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
        chatter[role]["root_vs_original_source"] = {}
        for joint in free:
            adr = int(model.jnt_qposadr[joint])
            actual = Rotation.from_quat(pose[:, adr + 3:adr + 7], scalar_first=True)
            # The floating base can be torso_link while the source root is
            # pelvis. Reuse the production anatomical-root correspondence.
            anchor = pose[:, adr:adr + 3] if source_anchor_id is None else value.xpos[:, source_anchor_id]
            chatter[role]["root_vs_original_source"][model.joint(int(joint)).name or str(joint)] = {
                "translation": _chatter(roots, anchor - roots, start),
                "rotation": _chatter(_angular_steps(source_rotation),
                                      _angular_steps(actual) - _angular_steps(source_rotation), start + 1)}
        chatter[role]["any_source_relative_chatter"] |= any(
            row["violating_frames"] for root in chatter[role]["root_vs_original_source"].values()
            for row in root.values())
    correction = _correction_chatter(model, qref, loaded["final"][0], roots, source_rotation, start)
    correction["scalar_joint_budget_verified_against_original_source"] = False
    correction["reference_source_audit_has_chatter"] = chatter["reference"]["any_source_relative_chatter"]
    non_degradation = _compare(series["final"], series["baseline"], start)
    # Report sustained layout changes against A independently of angular averages.
    a_layout = _layout(envelope, kin["baseline"], start, source=True)
    layout_delta = {}
    for part, row in comparisons["final_vs_original_source"]["layout"].items():
        a = a_layout[part]
        layout_delta[part] = {"A": a, "final": row,
                            "affected_frame_delta": sum(x["frames"] for x in row["abnormal_intervals"])
                                - sum(x["frames"] for x in a["abnormal_intervals"])}
    contacts = _contacts(prepared, paths, inputs, model, loaded)
    detailed_contacts = _contact_details(prepared, paths, loaded, inputs, model, start)
    external = _audit_bundle(paths, audits or {})
    from scripts.spline_collision_details import scene_details, compare_self
    scene_path = (audits or {}).get('final', {}).get('scene', final.parent / 'scene_metrics.json')
    a_scene_path = (audits or {}).get('baseline', {}).get('scene', baseline.parent / 'scene_metrics.json')
    per_body_geometry = scene_details(scene_path, a_scene_path, model)
    per_body_self = compare_self(final.parent / 'training_collision.json',
                                 baseline.parent / 'training_collision.json')
    artifacts = {role: {"path": str(path), "sha256": _sha(path)} for role, path in paths.items()}
    return {"schema_version": 1, "clip": clip or final.parent.name,
            "frames": len(qref), "frame_start": start, "fps": fps,
            "immutable_source": {"prepared": str(prepared), "sha256": _sha(prepared),
                                 "source_modified": False, "calibration": "prepared fixed body orientation targets"},
            "artifacts": artifacts, "comparisons": comparisons,
            "non_degradation_against_A": non_degradation,
            "layout_against_A": layout_delta,
            "chatter": {"actual_final_minus_fixed_ref": correction,
                        "original_source_audits": chatter},
            "contact_and_root_drift": contacts,
            "contact_and_root_per_part": detailed_contacts,
            "motion_amplitude_and_original_timing": _motion_amplitude(graph, source, kin, bodies, offsets, names, start),
            "source_root_chatter_correspondence": dict(robot_body=source_anchor_body,
                anatomical_translation_verified=source_anchor_id is not None,
                rotation_definition='calibrated root target world SO(3) increments'),
            "existing_audits": external,
            "original_geometry_per_body": per_body_geometry,
            "original_self_collision_per_body": per_body_self,
            "existing_A_defects": {"original_source_pose": source_reports["baseline"],
                                   "layout": a_layout, "chatter": chatter["baseline"],
                                   "contact": contacts["baseline"], "audits": external["baseline"]},
            "certification": {"physics_verified": False,
                              "all_quality_criteria_pass": None,
                              "reason": "Per-part evidence and exact A deltas only; physical replay/reset/unknown queries retain independent audit status"},
            "layout_rule": {"margin": LAYOUT_FLIP_MARGIN,
                            "sustained_frames": LAYOUT_FLIP_WINDOW_FRAMES},
            "limitations": ["Robot scalar-joint source correspondence unavailable; reference high-frequency allowance is conditional, never a source-quality certificate",
                            "Missing external geometry, self-collision, reset or physics audits are unverified",
                            "No per-part angular cap or percentage degradation allowance; positive A deltas reported for review"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    for name in ("prepared", "reference", "final", "baseline", "baseline-reference", "audits"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--clip")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {args.output}")
    if args.manifest:
        manifest = json.loads(args.manifest.read_text())
        clips = []
        for entry in manifest["clips"]:
            entry = dict(entry)
            if isinstance(entry.get("audits"), str):
                entry["audits"] = json.loads(Path(entry["audits"]).read_text())
            clips.append(evaluate(**entry))
        result = {"schema_version": 1, "clips": clips, "aggregate_pass": None}
    else:
        if any(getattr(args, name) is None for name in ("prepared", "reference", "final", "baseline")):
            parser.error("--prepared, --reference, --final and --baseline required without --manifest")
        audits = json.loads(args.audits.read_text()) if args.audits else None
        result = evaluate(args.prepared, args.reference, args.final, args.baseline,
                          clip=args.clip, baseline_reference=args.baseline_reference, audits=audits)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({"output": str(args.output), "clip_count": len(result.get("clips", [result]))}))


if __name__ == "__main__":
    main()
