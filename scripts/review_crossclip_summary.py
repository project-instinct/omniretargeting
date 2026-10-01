"""Summarize completed cross-clip experiments without hiding failed criteria."""

import argparse
import hashlib
import html
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text())


def collect(root):
    records = []
    for entry in read(root / "selection.json")["clips"]:
        directory = root / entry["segment"]
        if not (directory / "completed.json").exists():
            raise RuntimeError(f"Incomplete experiment: {entry['segment']}")
        metrics = {
            s: read(directory / f"{s}_metrics.json") for s in ("acquired", "final")
        }
        scene = {s: read(directory / f"{s}_scene.json") for s in ("acquired", "final")}
        pose = {
            s: read(directory / f"{s}_pose.json")
            for s in ("bootstrap", "acquired", "final")
        }
        config = read(directory / "stage_configs.json")
        result = {
            **entry,
            "scene_name": entry["scene"],
            "metrics": metrics,
            "scene": scene,
            "source_pose": pose,
            "displacement": read(directory / "displacement.json"),
            "checks": read(directory / "acceptance.json"),
            "scene_query": read(directory / "scene_query.json"),
            "iterations_per_phase": {
                key: config[key]["iterations"]
                for key in ("bootstrap", "acquired", "final")
            },
            "config_sha256": hashlib.sha256(
                json.dumps(config, sort_keys=True).encode()
            ).hexdigest(),
        }
        result["scene_within_project_sampled_10cm_goal"] = all(
            value["max_m"] < 0.1
            for key, value in scene["final"].items()
            if key in ("static", "moving")
        )
        result["geometry_updates_accepted"] = sum(
            row["alpha"] > 0 for row in read(directory / "final.json")["history"]
        )
        records.append(result)
    if len({r["config_sha256"] for r in records}) != 1:
        raise ValueError("Experiments used different solver configurations")
    return records


def number(value, factor=1, decimals=2):
    return "—" if value is None else f"{value * factor:.{decimals}f}"


def write_reports(root, records):
    iterations = records[0]["iterations_per_phase"]
    largest_orientation_error = max(
        values["max_rad"]
        for record in records
        for values in record["source_pose"]["final"][
            "calibrated_body_orientation"
        ].values()
    )
    lines = [
        "# TRUMANS 多片段复查",
        "",
        (
            f"{len(records)} 段完整原始录制、{len({r['scene_name'] for r in records})} 个场景，"
            f"共 {sum(r['frames'] for r in records)} 帧 / "
            f"{sum(r['frames'] for r in records) / 30:.1f} 秒。"
        ),
        (
            "所有片段在求解前选定，按动作类型选择较短的完整录制；不包含增强副本。"
            "这不是随机总体评估。各段从机器人模型参考姿态重新初始化，没有借用上一段轨迹。"
        ),
        (
            "所有阶段配置 SHA-256 相同；各段姿态初始化与接触求解的迭代上限分别为 "
            f"{iterations['bootstrap']} / {iterations['acquired']}，受保护几何求解为 "
            f"{iterations['final']}（pose 和 root_continuity 各执行一次该预算）。"
            "固定计算预算可能限制收敛；没有按结果单独加轮次或调权重。"
        ),
        "",
        (
            f"实测结论：{sum(all(r['checks'].values()) for r in records)} / {len(records)} 段通过后续修正的"
            "连续性、累计姿态、接触预算和关节限位检查。但初始接触解仍有局部姿态与全身表面贴合偏差，"
            f"标定目标的最大局部朝向误差约 {np.degrees(largest_orientation_error):.1f}°。"
            "因此目前只能确认保护机制在本批数据上有效，不能宣称所有姿态和支撑已经正确。"
        ),
        "",
        "## 抖动和姿态",
        "",
        (
            "下表的前后比较指本段接触求解结果 → 后续几何修正结果，"
            "不代表与旧版本算法的对比。所有数据保持 30 Hz、原始时序和 1.0 比例。"
        ),
        "",
        "| 片段 | 帧数 | 根部修正最大跳变：前→后 (cm/帧) | 关节二阶差分 RMS：前→后 (rad/帧²) | 姿态/连续性预算 |",
        "|---|---:|---:|---:|---|",
    ]
    for r in records:
        a, b = (r["metrics"][s] for s in ("acquired", "final"))
        lines.append(
            f"| {r['segment']} | {r['frames']} | "
            f"{number(a['root_offset_step_m']['max'], 100)} → {number(b['root_offset_step_m']['max'], 100)} | "
            f"{number(a['joint_acc_rms_rad_per_frame2'], decimals=6)} → {number(b['joint_acc_rms_rad_per_frame2'], decimals=6)} | "
            f"{'通过' if all(r['checks'].values()) else '未通过'} |"
        )
    lines += [
        "",
        (
            "根部修正指机器人根位置减源人体根参考。二阶差分包含真实快速动作，"
            "不能全解释成求解噪声。预算通过表示后续修正没有超过既有逐帧上限，"
            "不表示接触求解阶段本身已无抖动或初始姿态已正确。"
        ),
        "",
        "| 片段 | 根部追加水平/竖直最大位移 (cm) | 身体朝向最大追加变化 | 相对源骨段方向均值/95分位 |",
        "|---|---:|---:|---:|",
    ]
    for r in records:
        d = r["displacement"]
        offset = next(iter(d["roots"].values()))
        pose = r["source_pose"]["final"]
        lines.append(
            f"| {r['segment']} | {number(offset['horizontal_max_m'], 100)} / {number(offset['vertical_max_m'], 100)} | "
            f"{np.degrees(d['maximum_body_orientation_change']['radians']):.2f}° | "
            f"{np.degrees(pose['mean_rad']):.2f}° / {np.degrees(pose['p95_rad']):.2f}° |"
        )
    lines += [
        "",
        "骨段方向不能反映全部轴向扭转。另按相同的机器人标定姿态，测量各部位在世界或父级坐标系中的朝向误差：",
        "",
        "| 片段 | 最大局部朝向误差 | 部位 / 帧 | 根部相对源参考最大水平 / 竖直偏差 (cm) |",
        "|---|---:|---|---:|",
    ]
    for r in records:
        orientations = r["source_pose"]["final"]["calibrated_body_orientation"]
        body, worst = max(orientations.items(), key=lambda row: row[1]["max_rad"])
        m = r["metrics"]["final"]
        lines.append(
            f"| {r['segment']} | {np.degrees(worst['max_rad']):.2f}° | {body} / {worst['worst_frame']} | "
            f"{number(m['root_xy_offset_m']['max'], 100)} / {number(m['root_z_offset_m']['max'], 100)} |"
        )
    lines += [
        "",
        (
            "局部朝向误差可能包含形态、自由度和标定差异；它仍应作为独立检查项，不能被较小的平均骨段误差掩盖。"
            "累计姿态保护限制的是几何阶段相对接触解的新增变化，不会自动纠正接触解已经存在的姿态偏差。"
        ),
    ]
    lines += [
        "",
        "## 接触与世界位置误差",
        "",
        (
            "接触误差统计接触标记点到推断场景点；世界位置误差统计跟踪中心到源目标，"
            "覆盖整段，包括自由手的形态差异。没有接触帧时显示空项。"
        ),
        "",
        "| 片段 / 部位 | 接触帧 | 接触均值 / P95 / 最大 (mm) | 全段世界位置均值 (cm) |",
        "|---|---:|---:|---:|",
    ]
    labels = {
        "left_hand": "左手",
        "right_hand": "右手",
        "left_foot": "左脚",
        "right_foot": "右脚",
    }
    for r in records:
        for name, values in r["metrics"]["final"]["endpoints_m"].items():
            c = values["contact_vector"]
            lines.append(
                f"| {r['segment']} / {labels[name]} | {c['count']} | "
                + " / ".join(number(c[k], 1000) for k in ("mean", "p95", "max"))
                + f" | {number(values['world_position']['mean'], 100)} |"
            )
    lines += [
        "",
        (
            "手脚标记点的毫米级误差不代表全部身体表面都已贴合。下面单独检查全身活跃表面对应；"
            "向量误差包含切向位置偏差，法向误差反映目标表面间距的偏差，均不等于承载力。"
        ),
        "",
        "| 片段 / 表面部位 | 活跃样本数 | 向量均值 (cm) | 法向均值 / P95 / 最大 (cm) |",
        "|---|---:|---:|---:|",
    ]
    for r in records:
        metrics = r["metrics"]["final"]
        for part, values in metrics["surface_normal_error_m"].items():
            vector = metrics["surface_vector_error_m"][part]
            lines.append(
                f"| {r['segment']} / {part} | {values['count']} | {number(vector['mean'], 100)} | "
                + " / ".join(number(values[key], 100) for key in ("mean", "p95", "max"))
                + " |"
            )
    lines += [
        "",
        "## 原始场景穿插",
        "",
        "| 片段 / 几何 | 接触解→最终最大深度 (cm) | 最深身体 / 帧 | 完整退出向量水平/竖直分量 (cm) |",
        "|---|---:|---|---:|",
    ]
    for r in records:
        for key in ("static", "moving"):
            if key not in r["scene"]["final"]:
                continue
            a, b = (r["scene"][s][key] for s in ("acquired", "final"))
            lines.append(
                f"| {r['segment']} / {key} | {number(a['max_m'], 100)} → {number(b['max_m'], 100)} | "
                f"{b['worst_body']} / {b['worst_frame']} | "
                f"{number(b['worst_exit_xy_m'], 100)} / {number(b['worst_exit_z_m'], 100)} |"
            )
    lines += [
        "",
        (
            "10 cm 是项目的采样复查目标，不是零穿插保证。空间采样为 2 cm；"
            "原始静态网格使用 winding 查询，活动物体使用原始网格及反向采样。"
            "没有连续时间无碰撞证明。逐帧受影响列表与深度数组保存在各段 `final_scene.json/.npz`。"
        ),
        "本次只检验共同的分阶段重定向方案，没有给失败片段追加专用碰撞补丁或平滑滤波。",
        "测试输出保留残差；本轮未额外完成各段独立的固定根部局部碰撞修复，不能视为最终已修复轨迹。",
        "",
        "## 数据与结论边界",
        "",
        (
            "本批源帧号均连续，没有命中发布数据的 bad_frames；重建皮肤与发布关节点最大差异低于 1 mm。"
            "原始人体、场景和活动物体保持不变。"
        ),
        "",
        (
            "这里验证几何接触、连续性和姿态保护。未提供可信的人体各段惯性与摩擦参数，"
            "因此没有把这些片段的接触标签当成已通过整体动力学验证的支撑结论。"
            "小物件动作标签也不保证相关物件存在于当前场景几何中。"
        ),
        "",
        (
            "每段完整数据、配置、迭代日志、初始/接触/最终轨迹和独立指标均保留。"
            "保护预算、接触准确度、姿态相似度、穿插深度必须分开判断。"
        ),
        "",
    ]
    for r in records:
        name = r["segment"]
        lines += [
            f"### {name}",
            "",
            f"场景：`{r['scene_name']}`。",
            f"静态查询后端：`{r['scene_query']['method']}`；回退原因：{r['scene_query']['fallback_reason'] or '无'}。",
            "",
            "原始动作标注：",
            "",
            "```text",
            r["actions"].strip(),
            "```",
            "",
            (
                f"[交互预览]({name}/webviser/index.html) · [最终指标]({name}/final_metrics.json) · "
                f"[穿插帧]({name}/final_scene.json) · [预算检查]({name}/acceptance.json)"
            ),
            "",
        ]
    (root / "report.md").write_text("\n".join(lines))
    save = {
        "clips": records,
        "configuration_shared": True,
        "all_guard_checks_pass": all(all(r["checks"].values()) for r in records),
        "clips_within_sampled_penetration_goal": sum(
            r["scene_within_project_sampled_10cm_goal"] for r in records
        ),
    }
    (root / "results.json").write_text(json.dumps(save, indent=2))
    web = root / "web"
    web.mkdir(exist_ok=True)
    rows = []
    for r in records:
        link = web / r["segment"]
        if not link.exists():
            link.symlink_to(
                Path("..") / r["segment"] / "webviser", target_is_directory=True
            )
        b = r["metrics"]["final"]
        depth = max(
            r["scene"]["final"][k]["max_m"]
            for k in ("static", "moving")
            if k in r["scene"]["final"]
        )
        rows.append(
            f"<tr><td><a href='{html.escape(r['segment'])}/'>{html.escape(r.get('review_label', r['segment']))}</a>"
            f"<br><small>{html.escape(r['segment'])}</small></td>"
            f"<td>{r['frames'] / 30:.1f} 秒</td><td>{b['root_offset_step_m']['max'] * 100:.2f} cm/帧</td>"
            f"<td>{b['joint_acc_rms_rad_per_frame2']:.6f}</td><td>{depth * 100:.2f} cm</td></tr>"
        )
    (web / "index.html").write_text(
        "<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width'>"
        "<title>TRUMANS 多片段复查</title><style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 20px;"
        "color:#17304b;background:#f7f9fb}table{border-collapse:collapse;width:100%;background:white}"
        "td,th{padding:14px;border-bottom:1px solid #ddd;text-align:left}a{color:#1465a2}</style>"
        f"<h1>TRUMANS 多片段复查</h1><p>{len(records)} 段完整动作 · 同一套配置 · 原始人体与场景比例 1.0。"
        "点击片段查看机器人、源人体骨架与接触目标。</p>"
        "<p>根部跳变统计的是相对源人体的修正量。关节抖动列为二阶差分 RMS（rad/帧²）。"
        "穿插为原始网格的采样结果，仍有残差。</p>"
        "<p>后续修正的防抖与姿态边界通过，不表示初始接触解已经完全合格；"
        "部分帧仍有较大的局部朝向误差。这里提供的是完整测试输出。</p>"
        "<table><tr><th>片段</th><th>时长</th><th>根部修正最大跳变</th><th>关节抖动</th><th>最大穿插</th></tr>"
        + "".join(rows)
        + "</table><p>蓝色骨架：原始人体；灰色：机器人；橙色：接触目标与误差线。</p></html>"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    write_reports(args.root, collect(args.root))
