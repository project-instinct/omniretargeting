"""The real CLI appends contact tasks while preserving the solved qpos bytes."""

import sys
from types import SimpleNamespace

import mujoco
import numpy as np

from omniretargeting import OmniRetargeter
from omniretargeting.data_sources.base import MotionData
from omniretargeting.local_collision_repair import trajectory_digest
from tests.test_source_contacts import floor


def test_cli_exports_whole_body_tasks_even_when_solver_contacts_are_disabled(tmp_path, monkeypatch):
    import omniretargeting.main as cli

    model_path = tmp_path / "robot.xml"
    model_path.write_text('''<mujoco><worldbody><body name="root"><freejoint/>
        <geom type="sphere" size="0.1" mass="1"/>
        </body></worldbody></mujoco>''')
    model = mujoco.MjModel.from_xml_path(str(model_path))
    q_ref = np.tile(model.qpos0, (3, 1))
    q_ref[:, 0] = [.123456789012345, -0., .2]
    q_ref[:, 2] = .8
    before = q_ref.tobytes()
    source = MotionData(
        positions=np.tile([[[0., 0., .8]]], (3, 1, 1)), target_names=["Pelvis"], framerate=30,
        metadata={"body_surface_samples": {"full_body": {
            "points": np.tile([[[0., 0., .005]]], (3, 1, 1)),
            "skin_normals": np.tile([[[0., 0., -1.]]], (3, 1, 1)),
            "labels": ["left_knee"],
        }}},
    )
    source_before = source.metadata["body_surface_samples"]["full_body"]["points"].tobytes()
    monkeypatch.setattr(cli, "create_data_source", lambda *args, **kwargs: SimpleNamespace(load=lambda: source))
    monkeypatch.setattr(cli, "direct_source_config", lambda path, kind: {
        "type": "parc_ms", "motion": path, "auto_mapped_base_pose": False,
    })
    profile_path = tmp_path / "robot.json"
    profile_path.write_text("{}")
    profile = {"urdf_path": str(model_path), "joint_mapping": {"Pelvis": "root"},
               "retargeting": {"end_effector_priority": {"enabled": False},
                                "sequence_solver": {"enabled": True}, "self_collision_enabled": False}}
    monkeypatch.setattr(cli, "load_robot_config", lambda path: profile)
    owners = []

    def owner_factory(**kwargs):
        owner = OmniRetargeter.__new__(OmniRetargeter)
        owner.robot_model = model
        owner.terrain_mesh = floor()
        owner.smplx_joint_names = ["Pelvis"]
        owner.retargeting_config = kwargs["retargeting"]
        owner.last_sequence_report = {"backend": "fast"}
        owner._process_smplx_trajectory = lambda positions, scale: positions.copy()
        owner._perform_retargeting = lambda *args, **kwargs: q_ref
        owner._clamp_final_joint_safety_limits = lambda qpos: qpos
        owners.append(owner)
        return owner

    monkeypatch.setattr(cli, "OmniRetargeter", owner_factory)
    path = tmp_path / "result_retargeted.npz"
    monkeypatch.setattr(sys, "argv", ["omniretargeting", "--source-motion", str(tmp_path / "clip.pkl"),
                                    "--robot-config", str(profile_path), "--output", str(path),
                                    "--output-base-link", "root"])
    cli.main()
    with np.load(path, allow_pickle=False) as saved:
        assert saved["qpos"].tobytes() == before
        np.testing.assert_array_equal(saved["joint_pos"], q_ref[:, 7:])
        np.testing.assert_array_equal(saved["base_pos_w"], q_ref[:, :3])
        assert saved["contact_task_schema_version"].item() == 1
        knee = np.flatnonzero(saved["contact_task_body_parts"] == "left_knee")[0]
        assert saved["contact_task_active"][:, knee].all()
        assert (saved["contact_task_contact_types"][:, knee] == "touch").all()
        assert saved["contact_task_scene_object_names"].tolist() == ["scene"]
        np.testing.assert_allclose(saved["contact_task_target_positions_w"][:, knee, 2], 0, atol=1e-12)
    assert q_ref.tobytes() == before
    assert source.metadata["body_surface_samples"]["full_body"]["points"].tobytes() == source_before
    assert owners[0].last_source_contact_table is None  # export inference never changes solver inputs
    assert owners[0].last_end_effector_tasks == {}
    assert owners[0].last_collision_diagnostics["accepted_qpos_sha256"] == trajectory_digest(q_ref)
