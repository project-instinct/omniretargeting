"""Render the Chinese follow-up report and contact-repair handoff from audits."""

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

from scripts.review_effort_baseline import CLIPS

OLD_REASONS = ("unknown_contact_or_cone", "unknown_base_balance", "unknown_object_load")
UNKNOWN_REASONS = ("unknown_contact_geometry", "unknown_object_load", "unknown_effort_limits", "unknown_contact_evidence")


def _load(path):
    return json.loads(path.read_text())


def _counts(rows):
    return Counter(r["status"] for r in rows)


def _cell(counts):
    return "; ".join(f"{key}={value}" for key, value in sorted(counts.items())) or "0"


def _runs(frames):
    runs = []
    for frame in sorted(set(frames)):
        if runs and runs[-1][-1] + 1 == frame:
            runs[-1].append(frame)
        else:
            runs.append([frame])
    return runs


def _range_text(run):
    return str(run[0]) if len(run) == 1 else f"{run[0]}–{run[-1]}"


def render(previous, recomputed, report_path, test_log=None):
    new_summary = _load(recomputed / "summary.json")
    reports = {clip: _load(recomputed / clip / "joint_smoothness.json") for clip in CLIPS}
    old_reports = {clip: _load(previous / clip / "joint_smoothness.json") for clip in CLIPS}
    total = sum(r["frames"] for r in reports.values())
    total_counts = Counter()
    for report in reports.values():
        total_counts.update(report["effort_dynamics"]["frame_status_counts"])
    model = new_summary["model"]
    lines = ["# opt/effort 审计跟进报告", "", "## 结论与范围", "",
             f"在冻结的生产 fast 基线 `1d4e2de` 的全部 10 个 clip 上重算，共 {total} 帧、{total-20} 内点帧、20 个未审计端点。帧号从 0 开始。",
             f"主判定：**{_cell(total_counts)}**。unknown 不算通过；`infeasible_unsupported` 是明确失败，也不计入力矩超限条目。",
             "上一版 2523 个 unknown 的分母是 2570 内点帧；2565 是扣除此前 5 个条件过载帧后的数目。",
             "本次只改独立审计、诊断和文档，不改求解器或动作。所有输入 NPZ 的 SHA-256 在审计前后核对一致。穿透、接触、姿态、生产耗时与原基线逐字相同；没有新动作，因此不需要重跑求解器来比较同一条轨迹。",
             "`out/` 已通过 `git rm --cached` 和 amend 移出提交，磁盘保留；本报告和 JSON/CSV 继续由 `.gitignore` 排除，没有 force-add。", "",
             "## 审计方法分析与修正理由", "",
             "原始 robot-only `mj_inverse` 没有座椅、双脚地面反力，并混入软自碰撞/限位约束力，238 条超限不能直接解释为电机实际负载。生产 MJCF 提供质量、惯量、armature、阻尼和摩擦，原 URDF 只提供 velocity/effort；模型来源和哈希保存在 summary.json，不改变任何阈值。",
             f"动力学模型：`{model['dynamics_model']}`，质量 **{model['mass_kg']:.9f} kg**，gravity={model['gravity_m_s2']}；惯量为生产 MJCF 的显式 body inertial，包含 armature，扣除 passive，运动中补 `sign(qvel)*frictionloss`。URDF 限值来源：`{model['limits_urdf']}`；只读限值，不更换碰撞几何或惯量。",
             f"MJCF SHA-256：`{model['dynamics_model_sha256']}`；编译碰撞语义 SHA-256：`{model['collision_model_sha256']}`；URDF SHA-256：`{model['limits_urdf_sha256']}`。所有 29 个标量电机限值完整，浮动根 6 维无虚构驱动力。",
             "上一版反力分配方向正确，但用碰撞片到**源对应点**的完整距离判断是否接触，把切向滑动与实际法向分离混为一谈；又把无支撑或摩擦锥不能平衡归入 unknown，并让未实际碰到的源物体标签覆盖结论。", "",
             "现在逐帧查询所有 active collision geom 对原静态/移动三角面：20 mm 网格壳含全部凸包顶点，补充源局部片和朝向源/最近面法向的精确凸几何支撑点。有限三角面最近点限制支撑位置，不能把座椅延长成无限平面。实际间隙与源对应误差分别记录。闭合定向面使用原自由侧，开面须有 150 mm 范围内、法向一致的源证据，否则保留缺几何方向的 unknown。网格壳是与现有场景审计一致的采样假设，不是精确无碰撞证明。",
             "物体是否为静止世界支撑仍复用原几何接地、静止和堆叠传递算法，但接地距离从源检测的 30 mm 证据范围改为实际 5 mm；敏感性单独重算 15 mm 接地。不能把悬在地面上方的静止物体当作无限外力源。",
             "主判定保持原 ±5 mm 接触几何门限。>5 mm 间隙不提供力；>5 mm 穿入的 geom 不从壳穿过表面的交圈虚构支撑。±15 mm 同时放行近间隙和轻微穿入，只作敏感性分析，不让主判定过关。",
             "逐原三角面的法向/平面分别判断支撑，侧向家具穿透不删掉真实脚底反力。精确凸几何支撑点到接触平面的 extent gap 只在其法向投影落于真实有限原面时使用；这避免薄地板反面把 10 mm 穿入伪装成 1 mm 最近距离，同时不把有限台面延长。", "",
             "动力学仍为 `d=M*qacc+bias-passive+dry_friction`，解 `d=S.T*tau+J.T*f`，浮动根六维精确平衡，μ=0.6，minimax effort。接触片保留凸包极点，不将脚底简化成单一中心点来误判可行压力中心。同一身体/力射线下，广义力是 `p cross f` 的仿射映射：内部力矩可由极端力矩点的非负组合重现，故仅删去冗余列，保留完整反力锥与 minimax 最优值；不同法向不合并，退化情况保留原列。8 射线内包给可行 witness、外包给不可行/过载下界；多边形夹逼不确定时用圆锥 SOCP 和 primal/dual 数值检查。最接近反力锥的根残差只用于解释，不记作根电机。",
             "原大规模圆锥 primal 求解未给证书时，另解只有 6 个变量的 base 分离问题：逐点检查 `g_n ≥ μ*||g_t||` 且 `y·d < -1e-6*sum(abs(y))`。这样对任意合法反力及允许的根数值误差都不能平衡，才标确定 I；求解器失败本身不是 I。搜索的正 margin 不放宽任何摩擦、平衡或 effort 门限。无法取得数值证书属于 `audit_error_*`，与真正缺数据的 unknown 分开，也不通过。",
             "没有实际反力且根需求非零，或实际支撑锥不能提供所需根合力/力矩，标 `infeasible_unsupported`，分别记录 `no_attained_contact` / `friction_cone_or_support_wrench` 等原因。自由落体根需求为零时，即使无接触也可在此瞬时测试中 feasible，不盲目把离地都判坏。只有机器人实际触到非固定物体时，缺质量/惯量才标 `unknown_object_load`；源标签存在但机器人没碰上则是缺失接触。",
             "这是在几何覆盖与摩擦假设下的瞬时可行性审计。没有物体双向动力学、接触互补/加速度约束或物理回放，`physical_execution_verified=false`。差分仍保留原始噪声/尖峰；1 mrad/1 mm 的逐帧有界误差在 30 Hz 对二阶差分可放大到 3.6 rad/s² 或 m/s²，这是敏感性上界，不能当作实测噪声删掉超限。", "",
             "## 每个 clip 的主判定与 ≤15 mm 敏感性", "",
             "F=feasible；O=overload；I=infeasible_unsupported；U=缺数据 unknown；E=审计数值错误。每行 F+O+I+U+E 等于总帧数−2；E 必须单列，不能伪装成缺数据。", "",
             "| clip | 内点帧 | F | O | I | U：物体载荷 | U：接触几何 | U：其他缺数据 | E | 最差受力 witness 关节 / Nm / 帧 |",
             "|---|---:|---:|---:|---:|---:|---:|---|---:|---|"]
    for clip, report in reports.items():
        counts = report["effort_dynamics"]["frame_status_counts"]
        joints = [(name, r["mechanical_limits"]["effort"]) for name, r in report["joints"].items()
                  if r["mechanical_limits"]["effort"]["peak"] is not None]
        worst = max(joints, key=lambda p: p[1]["peak"] / p[1]["limit"]) if joints else None
        worst_text = f"{worst[0]} / {worst[1]['peak']:.3f} / {worst[1]['worst_frame']}" if worst else "无已确认的力矩 witness，峰值不适用"
        other = {s: counts.get(s, 0) for s in UNKNOWN_REASONS[2:] if counts.get(s, 0)}
        errors = sum(value for status, value in counts.items() if status.startswith("audit_error") or status in {"unknown_numerical", "unknown_cone_boundary"})
        lines.append(f"| {clip} | {report['frames']-2} | {counts.get('feasible',0)} | {counts.get('overload',0)} | {counts.get('infeasible_unsupported',0)} | {counts.get('unknown_object_load',0)} | {counts.get('unknown_contact_geometry',0)} | {_cell(other)} | {errors} | {worst_text} |")
    lines += ["", "无根平衡解的帧没有可信电机力矩，不能把 raw contact-free demand 当作负载峰值。表中峰值属于一个 minimax 分配，不是测量，也不是唯一的关节过载归属。", "",
              "参考轨迹与参考导数反事实的数值证书失败也单独记录，不伪装成源支撑不可行或缺数据：", "",
              *[f"- {clip}: reference={_cell({s:n for s,n in r['effort_dynamics']['reference_frame_status_counts'].items() if s.startswith('audit_error') or s in {'unknown_numerical', 'unknown_cone_boundary'}})}; counterfactual={_cell({s:n for s,n in r['effort_dynamics']['source_derivatives_at_final_pose_frame_status_counts'].items() if s.startswith('audit_error') or s in {'unknown_numerical', 'unknown_cone_boundary'}})}; 未获证书记录={r['effort_dynamics'].get('uncertified_numerical_refinements', [])}。" for clip,r in reports.items()], "",
              "| clip | 实际主判定 | ±15 mm 敏感性 | 原逆动力学超限条目 → 上版条件条目 → 本次实际条目 |",
              "|---|---|---|---|"]
    for clip, report in reports.items():
        dynamics = report["effort_dynamics"]
        lines.append(f"| {clip} | {_cell(dynamics['frame_status_counts'])} | {_cell(dynamics['tolerance_15mm_status_counts'])} | {new_summary['clips'][clip]['old_effort_violations']} → {old_reports[clip]['effort_limit_violations']} → {report['effort_limit_violations']} |")
    sensitivity_errors = {clip: [row['frame'] for row in report['effort_dynamics']['frame_diagnostics']
                                if row['tolerance_15mm']['status'].startswith('audit_error')]
                          for clip, report in reports.items()}
    lines += ["", "15 mm 场景会接纳更多原物体面/扫描边缘；它们可能缺质量或方向信息，因此 F 数不保证单调增加。不能用更宽容差掩盖这类缺数据。",
              f"15 mm 数值证书失败（E，不归入 U）：{ {clip:frames for clip,frames in sensitivity_errors.items() if frames} }。主判定独立保持 5 mm。"]
    lines += ["", "明确不可行帧进一步按几何缺口/反力锥无解拆开。最差帧按平移残差范数排序，同值时比较转动残差范数；力与力矩分量分别列出，不把 N 与 Nm 相加。", "",
              "| clip | I 的具体原因 / 帧数 | 最大残差帧 | 平移残差 N | 转动残差 Nm |",
              "|---|---|---:|---|---|"]
    for clip, report in reports.items():
        rows = [r for r in report["effort_dynamics"]["frame_diagnostics"] if r["status"] == "infeasible_unsupported"]
        worst = max(rows, key=lambda r: (sum(x*x for x in r["closest_outer_cone_residual"][:3]),
                                         sum(x*x for x in r["closest_outer_cone_residual"][3:]))) if rows else None
        residual = worst["closest_outer_cone_residual"] if worst else []
        force = ", ".join(f"{x:.2f}" for x in residual[:3]) or "无"
        moment = ", ".join(f"{x:.2f}" for x in residual[3:]) or "无"
        lines.append(f"| {clip} | {_cell(Counter(r['reason'] for r in rows))} | {worst['frame'] if worst else '无'} | {force} | {moment} |")
    lines += ["", "## 上一版 unknown 的逐原因拆解", "",
              "此表保留上一版三个互斥状态的帧数，并给出它们本次迁移后的状态，便于核对 2523 个 unknown 的去向。", "",
              "| clip | unknown_contact_or_cone | unknown_base_balance | unknown_object_load | 这些旧 unknown 帧的新状态 |",
              "|---|---:|---:|---:|---|"]
    for clip in CLIPS:
        old_rows = old_reports[clip]["effort_dynamics"]["frame_diagnostics"]
        old_counts = _counts(old_rows)
        new_rows = {r["frame"]: r for r in reports[clip]["effort_dynamics"]["frame_diagnostics"]}
        migrated = Counter(new_rows[r["frame"]]["status"] for r in old_rows if r["status"] in OLD_REASONS)
        lines.append(f"| {clip} | {old_counts[OLD_REASONS[0]]} | {old_counts[OLD_REASONS[1]]} | {old_counts[OLD_REASONS[2]]} | {_cell(migrated)} |")
    lines += ["", "按具体原因而非仅按状态拆解这些旧 unknown：", "",
              "| clip | 旧 unknown 类别 | 本次具体原因的逐帧计数 |",
              "|---|---|---|"]
    for clip in CLIPS:
        old_rows = old_reports[clip]["effort_dynamics"]["frame_diagnostics"]
        new_rows = {r["frame"]: r for r in reports[clip]["effort_dynamics"]["frame_diagnostics"]}
        for old_reason in OLD_REASONS:
            reasons = Counter(new_rows[r["frame"]]["reason"] for r in old_rows if r["status"] == old_reason)
            lines.append(f"| {clip} | {old_reason} | {_cell(reasons)} |")
    lines += ["", "典型帧按每类旧 unknown 中最大的已报告根残差选取；没有根残差时选该类中间帧。gap 正值是分离，负值是穿入；几何门限是 ±5 mm。根残差前三维为世界平移力 N，后三维为浮动根转动切空间的广义力矩 Nm，不能直接当作世界坐标下的质心力矩。完整向量、所有 geom/surface 距离和物体表面 ID 在各 clip 的 `joint_smoothness.json`。", "",
              "| clip | 旧原因 | 帧 | 本次状态 / 原因 | 典型支撑/最近片：身体、surface、实际 gap mm | 根广义反力残差：力 N / 力矩 Nm |",
              "|---|---|---:|---|---|---|"]
    for clip in CLIPS:
        old_rows = old_reports[clip]["effort_dynamics"]["frame_diagnostics"]
        new_rows = {r["frame"]: r for r in reports[clip]["effort_dynamics"]["frame_diagnostics"]}
        for reason in OLD_REASONS:
            selected = [new_rows[r["frame"]] for r in old_rows if r["status"] == reason]
            if not selected:
                continue
            candidates = [r for r in selected if "closest_outer_cone_residual" in r]
            row = max(candidates, key=lambda r: sum(abs(x) for x in r["closest_outer_cone_residual"])) if candidates else selected[len(selected)//2]
            pairs = row.get("missing_source_support_pairs", []) or [p for p in row["original_triangle_pairs"] if p["source_candidates"]] or row["original_triangle_pairs"]
            if row["status"] == "unknown_object_load":
                pairs = row["unmodelled_object_contact_pairs"]
            pairs = sorted(pairs, key=lambda p: abs(p["gap_m"]))[:3]
            detail = "; ".join(f"{p['body']} / {p['surface']} / {p['gap_m']*1000:+.2f}"
                               + ("（unsigned，侧未知）" if not p["gap_sign_known"] else "")
                               + f"（source_support={p['source_support']}）" for p in pairs)
            if row["status"] == "unknown_object_load":
                detail += f"; 实际非固定物体 surfaces={row['unmodelled_object_surfaces']}，质量/惯量未提供"
            residual = row.get("closest_outer_cone_residual")
            residual_text = ", ".join(f"{x:.2f}" for x in residual) if residual is not None else "无：未证不可行或缺载荷数据"
            lines.append(f"| {clip} | {reason} | {row['frame']} | {row['status']} / {row['reason']} | {detail} | {residual_text} |")
    lines += ["", "本次真正剩余的缺数据状态单独取典型帧，避免被旧类别中的确定失败帧遮住：", "",
              "| clip | 缺数据状态 / 帧数 | 典型帧 | 原因 | 身体 / surface / gap mm；具体缺失 |",
              "|---|---|---:|---|---|"]
    for clip, report in reports.items():
        groups = defaultdict(list)
        for row in report["effort_dynamics"]["frame_diagnostics"]:
            if row["status"].startswith("unknown"):
                groups[row["status"]].append(row)
        for status, rows in sorted(groups.items()):
            row = rows[len(rows) // 2]
            if status == "unknown_object_load":
                pairs = row["unmodelled_object_contact_pairs"]
                missing = f"实际非固定物体 surfaces={row['unmodelled_object_surfaces']} 的质量/惯量"
            else:
                pairs = [p for p in row.get("original_triangle_pairs", [])
                         if not p["normal_known"] and p["nearest_unsigned_m"] <= .005]
                missing = f"近面采样点自由侧/法向未定={row.get('uncertain_near_surface_points', 0)}；不假设未知面反力为零"
            detail = "; ".join(f"{p['body']} / {p['surface']} / {p['gap_m']*1000:+.2f}"
                               + ("（unsigned）" if not p["gap_sign_known"] else "") for p in pairs[:3])
            lines.append(f"| {clip} | {status} / {len(rows)} | {row['frame']} | {row['reason']} | {detail}；{missing} |")
    object_diagnostics = recomputed.parent / "object_support_diagnostics.json"
    if object_diagnostics.exists():
        lines += ["", "剩余物体载荷 unknown 的具体来源：移动/接地判断复用源准备参数，并查询原物体顶点到原静态三角面的距离。同一帧可同时缺物体载荷与几何方向，主分类优先记录几何；此表保留重叠原因，不能再相加当作独立帧。最近顶点距离只是该接地方法的证据，不是物体的完整动力学或精确接触证明。典型帧速度是所有原顶点的最大中心差分速度；是否静止按已有帧间速度规则判断，两者不必恰好一致。", "",
                  "| clip | 实际接触的 object / surface | 接触帧数（主 U载荷 / U几何） | 移动帧 | 静止但未证接地帧 | 15 mm 可证固定帧 | 典型帧 | 顶点最大速度 / 静止门限 m/s | 最近顶点到静态面 mm |",
                  "|---|---|---|---:|---:|---:|---:|---|---:|"]
        for clip, rows in _load(object_diagnostics).items():
            for row in rows:
                lines.append(f"| {clip} | {row['name']} / {row['surface']} | {len(row['unknown_contact_frames'])}（{row['primary_unknown_object_load_frames']} / {row['additional_geometry_unknown_frames']}） | {row['moving_frames']} | {row['still_but_not_proved_fixed_frames']} | {row['15mm_proved_fixed_frames']} | {row['representative_frame']} | {row['maximum_vertex_speed_m_s']:.5f} / {row['sticking_speed_m_s']:.5f} | {row['nearest_vertex_to_static_triangles_mm']:.2f} |")
    geometry_diagnostics = recomputed.parent / "unknown_contact_geometry_diagnostics.json"
    if geometry_diagnostics.exists():
        lines += ["", "未确定自由侧的典型原三角面查询：距离门限 150 mm、法向一致性 0.8 是原局部方向证据的适用范围，不是接触容差；缺这项证据时不猜测外力方向。最近源法向可能指向扫描边缘，不能直接当该三角面的法向。", "",
                  "| clip / 帧 | 身体 / surface | 未定近面点数 | 最近 unsigned gap mm | 最近方向证据距离 mm | 法向绝对内积 | watertight oriented / 自由侧语义已指定 | 同时缺载荷 surfaces |",
                  "|---|---|---:|---:|---:|---:|---|---|"]
        for clip, row in _load(geometry_diagnostics).items():
            if not row:
                continue
            for pair in row["pairs"]:
                distance = pair["orientation_evidence_distance_m"]
                lines.append(f"| {clip} / {row['frame']} | {pair['body']} / {pair['surface']} | {pair['uncertain_points_5mm']} | {pair['uncertain_closest_unsigned_m']*1000:.3f} | {distance*1000 if distance is not None else '无'} | {pair['orientation_evidence_abs_alignment']:.4f} | {pair['surface_watertight_oriented']} / {pair['surface_free_side_specified']} | {row['actual_nonfixed_object_surfaces']} |")
    lines += ["", "## 可交给穿透/接触修复的缺失支撑", "",
              "主表列原源 support 标签存在、机器人同一 geom/surface 未达到 ±5 mm 的帧。另在 CSV 保留所有源几何接触缺口及 `source_support` 标签：源力学解释失败而未排名为 support 的脚/座椅接触不能丢掉，也不能擅自改成确定支撑。正间隙表示没落到面上，负值表示穿入过深；它们不能统称为源动作不存在支撑。固定参考曾达到而最终失去，标 `lost_after_fixed_reference`；参考也未达到标 `unreached_in_fixed_reference`，不能仅凭源标签把缺口归因到最后一轮修复。其他身体仍可能提供反力，因此缺失片的帧也可能是 F/O/U。",
              "完整逐帧数据为 `support_loss.csv`；`support_loss_segments.json` 同时保留有/无 support 标签的全部连续缺口，标明标签、帧状态计数与 15 mm 可达帧数。下表逐身体/geom/表面/归因给出有 support 标签的全部连续帧段、间隙范围与峰值帧，surface=0 是静态场景，>0 是原对象列表索引+1。", "",
              "| clip | 身体 / geom / surface | 区分 | 帧段 | gap 范围 mm | 最大绝对 gap / 帧 |",
              "|---|---|---|---|---|---|"]
    losses, segments = [], []
    for clip in CLIPS:
        dynamics = reports[clip]["effort_dynamics"]
        references = {r["frame"]: {(p["geom"],p["surface"]): p for p in r["original_triangle_pairs"]}
                      for r in dynamics["reference_frame_diagnostics"]}
        groups = defaultdict(list)
        for row in dynamics["frame_diagnostics"]:
            for pair in row["missing_source_contact_pairs"]:
                reference = references[row["frame"]][pair["geom"], pair["surface"]]
                classification = "lost_after_fixed_reference" if reference["actual_contact"] else "unreached_in_fixed_reference"
                loss = {"clip": clip, "frame": row["frame"], "body": pair["body"], "geom": pair["geom"],
                        "surface": pair["surface"], "surface_name": pair["surface_name"],
                        "parts": ",".join(pair["parts"]), "gap_mm": pair["gap_m"]*1000,
                        "reference_gap_mm": reference["gap_m"]*1000, "classification": classification,
                        "source_support": pair["source_support"],
                        "frame_status": row["status"], "tolerance_contact": pair["tolerance_contact"]}
                losses.append(loss)
                groups[pair["body"], pair["geom"], pair["surface"], classification, pair["source_support"]].append(loss)
        for (body, geom, surface, classification, source_support), group in sorted(groups.items()):
            frame_map = {x["frame"]: x for x in group}
            for run in _runs(frame_map):
                values = [frame_map[f] for f in run]
                worst = max(values, key=lambda x: abs(x["gap_mm"]))
                minimum, maximum = min(x["gap_mm"] for x in values), max(x["gap_mm"] for x in values)
                segment = dict(clip=clip, body=body, geom=geom, surface=surface, classification=classification,
                               source_support=source_support,
                               frames=run, gap_min_mm=minimum, gap_max_mm=maximum,
                               frame_status_counts=dict(Counter(x["frame_status"] for x in values)),
                               tolerance_contact_frames=sum(x["tolerance_contact"] for x in values),
                               worst_frame=worst["frame"], worst_gap_mm=worst["gap_mm"])
                segments.append(segment)
                if source_support:
                    lines.append(f"| {clip} | {body} / {geom} / {surface} | {classification} | {_range_text(run)} | {minimum:+.2f}…{maximum:+.2f} | {worst['gap_mm']:+.2f} / {worst['frame']} |")
    with (recomputed.parent / "support_loss.csv").open("w") as stream:
        keys = ["clip", "frame", "body", "geom", "surface", "surface_name", "parts", "gap_mm", "reference_gap_mm",
                "classification", "source_support", "frame_status", "tolerance_contact"]
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader(); writer.writerows(losses)
    (recomputed.parent / "support_loss_segments.json").write_text(json.dumps(segments, indent=2) + "\n")
    loss_counts = Counter(x["classification"] for x in losses if x["source_support"])
    all_loss_counts = Counter(x["classification"] for x in losses)
    lines += ["", f"有 support 标签的缺失支撑 geom×帧条目：{_cell(loss_counts)}；全部源几何接触缺口：{_cell(all_loss_counts)}。", "",
              "## 源动作与求解器新增超限", "",
              "固定参考是人的动作拟合到机器人后的轨迹，不是人体真实 effort。相同最终位姿/实际接触下代入参考速度加速度，若变为 F，才确认新增惯性需求；参考本身已有过载是继承的机器人拟合负担，不能断言人本身做不到。", "",
              "| clip | 实际过载帧 | 关节超限：条目数、峰值/限值 Nm、帧 | 新增/继承区分 |",
              "|---|---|---|---|"]
    for clip, report in reports.items():
        dynamics = report["effort_dynamics"]
        joint_text = []
        for name, value in report["joints"].items():
            effort = value["mechanical_limits"]["effort"]
            if effort["violations"]:
                joint_text.append(f"{name}: {effort['violations']}, {effort['peak']:.3f}/{effort['limit']}, {effort['violation_frames']}")
        attributions = Counter(r["classification"] for r in dynamics["overload_frame_attribution"])
        lines.append(f"| {clip} | {dynamics['unavoidable_overload_frames']} | {'; '.join(joint_text) or '0，缺失支撑/unknown 不算 0 effort 证书'} | {_cell(attributions)} |")
    lines += ["", "同样对确定不可行帧作参考/导数反事实区分，避免把缺支撑和源拟合根加速度都笼统归到最终修复：", "",
              "| clip | 确定不可行帧归因（帧数） |", "|---|---|"]
    for clip, report in reports.items():
        attribution = Counter(r["classification"] for r in report["effort_dynamics"]["infeasible_frame_attribution"])
        lines.append(f"| {clip} | {_cell(attribution)} |")
    lines += ["", "这里 `added_inertial_demand` 指保持最终几何不动、换回参考导数便能恢复 F；`added_pose_or_contact_geometry` 指参考完整状态 F、最终几何反事实仍失败；其他继承/耦合项不证明人体源运动本身不可执行。逐帧证据在 `infeasible_frame_attribution`。"]
    lines += ["", "原始二阶差分的阶段对照：下表是机器人根平移加速度峰值 m/s² / 帧，与 effort 证书分开。阶段峰值不能证明哪个修复步骤造成具体支撑缺口。", "",
              "| clip | fixed_reference | stage_0 | before_mesh_repair | final | 最大新增关节加速度 rad/s² / 帧 |",
              "|---|---|---|---|---|---|"]
    for clip in CLIPS:
        stages = new_summary["clips"][clip]["stage_root_diagnostics"]
        cells = [f"{stages[s]['linear_acceleration_peak_m_s2']:.3f} / {stages[s]['linear_acceleration_worst_frame']}"
                 for s in ("fixed_reference", "stage_0", "before_mesh_repair", "final")]
        name, effort = max(((name, r["mechanical_limits"]["effort"]) for name, r in reports[clip]["joints"].items()),
                           key=lambda p: p[1]["added_acceleration_peak"])
        lines.append(f"| {clip} | {' | '.join(cells)} | {name}: {effort['added_acceleration_peak']:.3f} / {effort['added_acceleration_worst_frame']} |")
    landmark_path = report_path.parent / "source_landmark_acceleration.json"
    if landmark_path.exists():
        landmarks = _load(landmark_path)
        lines += ["", "此前保留的原始人体 landmark 差分（m/s² / 帧），不是机器人自由根同一个坐标点；不改写或滤波源数据：", "",
                  "| clip | human Pelvis | human Spine3 |", "|---|---|---|"]
        for clip in CLIPS:
            cells = [f"{landmarks[clip][p]['raw_acceleration_peak_m_s2']:.3f} / {landmarks[clip][p]['worst_frame']}"
                     for p in ("Pelvis", "Spine3")]
            lines.append(f"| {clip} | {' | '.join(cells)} |")
    lines += ["", "## 修复、剩余问题与验证", "",
              "已修复审计：实际有限三角面接触、切向对应误差解耦、无支撑/无根平衡解的确定失败状态、仅对实际物体接触保留载荷 unknown、独立 15 mm 敏感性、根残差诊断和逐帧支撑缺口交接。没有放宽 URDF velocity/effort、μ 或主几何门限。",
              "求解器未改。`omni_opt_pen` 正在修改 `_fast/mesh_repair.py` / `_fast/pipeline.py` 的同一原网格修复阶段，跨阶段重复引入支撑投影会冲突。交接建议：在其全身原几何协调步骤中按上表恢复原面支撑；先合后离，分散根/关节修正到邻帧；把无接触与摩擦/合力矩不足分别作为约束冲突，达到接触后再判断并限制真实 effort/新增加速度。对 ±15 mm 能恢复而实际几何不行的帧，优先消除具体法向 gap，不能把审计容差当动作修复。",
              "仍需：实际 support 修复、可信非固定物体质量/惯量及双向受力、源拟合尖峰证据、接触运动学与物理回放。端点未审计，unknown/明确不可行均不通过。原基线已有的穿透/接触/姿态残余继续沿用原报告，不能声称本次把动作修好了。", "",
              "重算在 `flock /tmp/omniretargeting_heavy.lock` 内串行执行，显式 `PYTHONPATH=/home/duanxin/Xyk/omni_opt_effort`；逐 clip 审计耗时保存在 `recomputed/summary.json`。生产轨迹和生产耗时不变。", "",
              "已完成 clip 的补充证书与典型帧诊断也在本任务持有同一排他锁期间串行执行：仅暂停本任务的审计 worker，确认停止后运行补充计算，并在 finally 恢复；没有暂停其他 worktree 的进程。下表是实际运行的墙钟时间，包含上述串行诊断暂停，不宣称为新向量实现的独立性能基准。",
              "完整前后质量、各碰撞对、budgets（含 relation/layout）、skipped_tasks、deadline 与 reset 数据见 [comparison.md](contact_followup/comparison.md) / [comparison.json](contact_followup/comparison.json)。这些数据与冻结基线逐字一致。", "",
              "| clip | 原 timing.production_seconds（未变） | 本次独立审计墙钟 s | 末次证书补充计算 s（计时范围可能重叠，不再相加） |",
              "|---|---:|---:|---:|",
              *[f"| {clip} | {row['production_seconds']:.3f} | {row['audit_seconds']:.2f} | {row.get('numerical_certificate_refinement_seconds',0):.2f} |" for clip,row in new_summary['clips'].items()], "",
              "验证覆盖：真实地面与座椅反力、10 mm 悬空间隙、10 mm 穿入、有限面边缘、沿面对应点漂移、自由落体、实际/源-only物体接触、摩擦锥无解、圆锥内包误拒绝、缺原三角几何、所有现有 URDF/逆动力学/新增惯性测试。"]
    if test_log and test_log.exists():
        tail = test_log.read_text().splitlines()[-1]
        lines += ["", f"完整测试：`{tail}`。日志 `{test_log}`。"]
    report_path.write_text("\n".join(lines) + "\n")
    print(json.dumps({"counts": total_counts, "support_loss_entries": loss_counts, "report": str(report_path)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--recomputed", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--test-log", type=Path)
    args = parser.parse_args()
    render(args.previous, args.recomputed, args.report, args.test_log)
