"""Default support mechanics must accept source markers without SMPL-X skin."""

from types import SimpleNamespace

import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

from omniretargeting._core.surfaces import _prepare_source_support
from omniretargeting.data_sources.skeleton_inertia import skeleton_segment_model
from omniretargeting.source_contacts import build_source_contacts
from omniretargeting.source_inertia import segment_source_inertia
from omniretargeting.source_support_evidence import build_source_support_evidence
from omniretargeting.surface_contact import surface_contact_config


@pytest.fixture(autouse=True)
def serial_mechanics(monkeypatch):
    # Exercise the same computations without allocating a machine-wide pool
    # for these small fixtures. Production's forked path is unchanged.
    monkeypatch.setenv("OMNIRETARGETING_CPUS", "1")


def source_case(*, airborne=False):
    frames = 40
    names = ["arbitrary_root", "arbitrary_tip"]
    model = skeleton_segment_model(names, [-1, 0], [[0, 0, 0], [0, 0, 1]])
    positions = np.tile([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], (frames, 1, 1))
    if airborne:
        times = np.arange(frames) / 30.0
        positions[:, :, 2] += (20 - 4.905 * times**2)[:, None]
    quats = np.tile([1.0, 0.0, 0.0, 0.0], (frames, 2, 1))
    patch = np.array([[x, y, 0.0] for x in (-0.1, 0.0, 0.1) for y in (-0.1, 0.0, 0.1)])
    points = positions[:, :1] + patch[None]
    normals = np.broadcast_to([0.0, 0.0, -1.0], points.shape)
    skin = {
        "points": points.copy(),
        "skin_normals": normals,
        "labels": ["previously_unseen_region"] * len(patch),
    }
    scene = trimesh.Trimesh(
        [[-5, -5, 0], [5, -5, 0], [5, 5, 0], [-5, 5, 0]],
        [[0, 1, 2], [0, 2, 3]],
        process=False,
    )
    cfg = surface_contact_config()
    table = build_source_contacts(skin, scene, cfg, framerate=30.0)
    return names, model, positions, quats, scene, cfg, table


def test_inertia_uses_segment_com_and_spin_not_the_joint_origin():
    model = skeleton_segment_model(
        ["root", "tip", "zero_length_tip"],
        [-1, 0, 1],
        [[0, 0, 0], [2, 0, 0], [0, 0, 0]],
    )
    np.testing.assert_allclose(model["local_centers"], [[1, 0, 0]])
    np.testing.assert_allclose(model["masses"], [1])
    np.testing.assert_allclose(model["inertias"][0], np.diag([0, 1 / 3, 1 / 3]))
    assert model["assumptions"]["subject_measurement"] is False
    quats = np.tile(
        Rotation.from_euler("z", 90, degrees=True).as_quat(scalar_first=True),
        (20, 3, 1),
    )
    positions = np.tile([[0, 0, 1], [0, 2, 1], [0, 2, 1]], (20, 1, 1))
    before = positions.copy()
    inertia = segment_source_inertia(
        model,
        positions,
        quats,
        ["root", "tip", "zero_length_tip"],
        framerate=30,
    )
    np.testing.assert_allclose(
        inertia["centers"], np.tile([[[0, 1, 1]]], (20, 1, 1)), atol=1e-12
    )
    np.testing.assert_allclose(
        inertia["demand"]["wrench"][:, :3], np.tile([0, 0, 9.81], (20, 1)), atol=1e-10
    )
    np.testing.assert_array_equal(positions, before)


def test_marker_only_contacts_run_the_same_wrench_checker_and_keep_observations():
    names, model, positions, quats, scene, cfg, table = source_case()
    original = (
        positions.copy(),
        quats.copy(),
        table.points.copy(),
        table.active.copy(),
    )
    evidence, same, report = build_source_support_evidence(
        None,
        positions,
        names,
        quats,
        scene,
        cfg,
        framerate=30,
        table=table,
        inertia_model=model,
    )
    assert same is table
    assert report["status"] == "active"
    assert report["enabled"] is True
    assert report["inertia_source"] == "source_skeleton_uniform_linear_density"
    assert report["explained_frames"] == len(positions)
    assert report["physical_support_verified"] is False
    assert all(
        evidence.role("previously_unseen_region", 0, f) == "support"
        for f in range(len(positions))
    )
    for value, expected in zip(
        (positions, quats, table.points, table.active), original
    ):
        np.testing.assert_array_equal(value, expected)


def test_free_fall_never_creates_contact_to_explain_gravity():
    names, model, positions, quats, scene, cfg, table = source_case(airborne=True)
    assert not table.active.any()
    evidence, same, report = build_source_support_evidence(
        None,
        positions,
        names,
        quats,
        scene,
        cfg,
        framerate=30,
        table=table,
        inertia_model=model,
    )
    assert report["status"] == "active"
    assert report["support_region_frames"] == {}
    assert all(not frame for frame in evidence.candidates)
    assert same is table and not table.active.any()


def test_missing_inertia_is_unresolved_and_keeps_existing_material_contacts():
    names, _, positions, quats, scene, cfg, table = source_case()
    active = table.active.copy()
    evidence, same, report = build_source_support_evidence(
        None,
        positions,
        names,
        quats,
        scene,
        cfg,
        framerate=30,
        table=table,
    )
    assert evidence is None and same is table
    assert report["enabled"] is True and report["status"] == "unresolved"
    assert report["reason"] == "missing_source_inertia_model"
    np.testing.assert_array_equal(table.active, active)


def test_main_preparation_runs_by_default_without_a_whole_body_surface():
    names, model, positions, quats, scene, _, table = source_case()
    owner = SimpleNamespace(
        smplx_joint_names=names,
        _source_joint_orientations=quats,
        _source_inertia_model=model,
        _source_framerate=30,
        last_source_contact_table=table,
    )
    _prepare_source_support(owner, None, positions, scene, {})
    assert owner._support_evidence_built
    assert owner.last_source_support_diagnostics["status"] == "active"
    assert (
        owner._source_support_evidence.role("previously_unseen_region", 0, 0)
        == "support"
    )
    # Explicit opt-out is the only configured disabling path.
    other = SimpleNamespace()
    _prepare_source_support(
        other, None, positions, scene, {"contact_phase_priority": {"enabled": False}}
    )
    assert other.last_source_support_diagnostics["reason"] == "contact_phase_disabled"


def test_unexplained_sliding_contact_stays_protected_before_object_control():
    from dataclasses import replace

    from omniretargeting.contact_phase import attach_contact_phase
    from omniretargeting.contact_support import explain_source_support
    from omniretargeting.object_control import SourceObjectControlEvidence
    from tests.test_contact_phase import case

    _, q, surface, tasks, priority = case(stance=True)
    tasks["left_foot"] = replace(
        tasks["left_foot"], contact_modes=np.full(len(q), "sliding")
    )
    active = tasks["left_foot"].contact_mask.copy()
    # Upward demand without any admissible force cone is explicitly unknown.
    evidence = explain_source_support(
        [[]] * len(q), np.tile([0, 0, 9.81, 0, 0, 0], (len(q), 1)), {}
    )
    control = SourceObjectControlEvidence([{("left_foot", 0)}] * len(q), [])
    policy = attach_contact_phase(
        surface,
        None,
        tasks,
        priority,
        support_evidence=evidence,
        control_evidence=control,
    )
    assert policy.role("left_foot", 0) == "hold"
    assert policy.diagnostics["unresolved_support_frames"] == list(range(len(q)))
    np.testing.assert_array_equal(tasks["left_foot"].contact_mask, active)
    assert not policy.support.any()


def test_missing_physical_input_protects_only_existing_contacts():
    from dataclasses import replace

    from omniretargeting._core.surfaces import _unresolved_support_frames
    from omniretargeting.contact_phase import attach_contact_phase
    from tests.test_contact_phase import case

    model, q, _, tasks, priority = case(stance=True)
    active = np.zeros(len(q), bool)
    active[1] = True
    tasks["left_foot"] = replace(
        tasks["left_foot"],
        contact_mask=active,
        contact_modes=np.full(len(q), "sliding"),
    )
    owner = SimpleNamespace(last_source_support_diagnostics={"status": "unresolved"})
    policy = attach_contact_phase(
        None,
        None,
        tasks,
        priority,
        model=model,
        unresolved_support_frames=_unresolved_support_frames(owner, len(q)),
    )
    np.testing.assert_array_equal(
        policy.roles["left_foot"], np.where(active, "hold", "free")
    )
    np.testing.assert_array_equal(tasks["left_foot"].contact_mask, active)


def test_lafan_default_inertia_and_material_markers_survive_resampling(tmp_path):
    from omniretargeting.data_sources.lafan1 import Lafan1DataSource
    from tests.data_sources.test_lafan1 import MINIMAL_BVH

    # Extend the minimal fixture with real foot descendants, as in LAFAN.
    text = MINIMAL_BVH
    for side in ("Left", "Right"):
        start = text.index(f"JOINT {side}Foot")
        end = text.index("\n    }", start)
        text = (
            text[:end]
            + (
                f"\n        JOINT {side}Toe\n        {{\n"
                "            OFFSET 0.0 0.0 15.0\n"
                "            CHANNELS 3 Zrotation Xrotation Yrotation\n        }"
            )
            + text[end:]
        )
    lines = text.splitlines()
    for index in range(-3, 0):
        values = lines[index].split()
        # Both added toe rotations are identity; all old channels remain.
        lines[index] = " ".join(values[:15] + ["0"] * 3 + values[15:] + ["0"] * 3)
    path = tmp_path / "with_toes.bvh"
    path.write_text("\n".join(lines))
    motion = Lafan1DataSource(path).load()
    cropped = Lafan1DataSource(path, start_frame=1).load()
    model = motion.metadata["source_inertia_model"]
    assert model["method"] == "source_skeleton_uniform_linear_density"
    np.testing.assert_array_equal(
        model["masses"], cropped.metadata["source_inertia_model"]["masses"]
    )
    for side in ("left", "right"):
        marker = motion.metadata["end_effector_markers"][side + "_foot"]
        assert marker["world_target"] is True
        assert marker["contact_samples"].shape == (3, 2, 3)
    before = motion.positions.copy()
    resampled = motion.resample(motion.framerate * 2)
    assert resampled.metadata["source_inertia_model"] is model
    for marker in resampled.metadata["end_effector_markers"].values():
        assert bool(marker["world_target"])
        assert marker["contact_samples"].shape == (5, 2, 3)
    np.testing.assert_array_equal(motion.positions, before)
    np.testing.assert_array_equal(resampled.positions[[0, -1]], before[[0, -1]])
