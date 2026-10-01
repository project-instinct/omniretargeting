import mujoco
import numpy as np
from omniretargeting.morphology import calibrate_terminal_scales


def test_missing_scale_preserves_world_targets_without_changing_contacts():
    from dataclasses import replace
    from omniretargeting.morphology import calibrated_target_regimes
    from tests._contact_fixtures import _task

    points = np.tile([0.2, 0.0, 0.4], (3, 1))
    task = replace(
        _task(points, [True, False, False]), swing_world_mask=np.zeros(3, bool)
    )
    result = calibrated_target_regimes(
        {"unmeasured_tool": task, "calibrated_tool": task},
        {
            "unmeasured_tool": {"status": "missing_reference"},
            "calibrated_tool": {"status": "measured"},
        },
    )
    assert result["unmeasured_tool"].swing_world_mask.all()
    assert not result["calibrated_tool"].swing_world_mask.any()
    np.testing.assert_array_equal(
        result["unmeasured_tool"].contact_mask, task.contact_mask
    )
    np.testing.assert_array_equal(result["unmeasured_tool"].contact_positions, points)
    assert not task.swing_world_mask.any()


def test_mapped_chain_length_uses_reference_geometry_and_opaque_names():
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody><body name="trunk"><freejoint/><geom size=".1"/>
    <body name="a" pos=".1 0 0"><joint/><geom size=".02"/><body name="b" pos=".2 0 0"><joint/><geom size=".02"/>
    <body name="c" pos=".3 0 0"><joint/><geom size=".02"/></body></body></body>
    <body name="other" pos="0 .2 0"><joint/><geom size=".02"/></body></body></worldbody></mujoco>""")
    source = dict(
        joint_names=["P", "Q", "R", "S", "T"],
        joint_parents=[-1, 0, 1, 2, 0],
        reference_joint_positions=np.array(
            [[0, 0, 0], [0.2, 0, 0], [0.6, 0, 0], [1.2, 0, 0], [0, 0.3, 0]]
        ),
    )
    mapping = {"P": "trunk", "Q": "a", "R": "b", "S": "c", "T": "other"}
    specs, report = calibrate_terminal_scales(
        model,
        model.qpos0,
        mapping,
        source,
        {"opaque": dict(source_joint="S", robot_body="c")},
    )
    assert np.isclose(specs["opaque"]["swing_morphology_scale"], 0.5)
    assert report["opaque"]["source_chain_root"] == "Q"
    assert report["opaque"]["status"] == "measured"
    # A second solve must not mistake its previous inferred result for user policy.
    _, repeated = calibrate_terminal_scales(model, model.qpos0, mapping, source, specs)
    assert repeated["opaque"]["status"] == "measured"


def test_branch_does_not_hide_a_measurable_terminal_chain():
    # Neither source nor robot identifiers carry anatomical information.
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
    <body name="A"><freejoint/><geom size=".1"/>
      <body name="B" pos="0 0 .2"><joint/><geom size=".02"/>
        <body name="C" pos="0 0 .3"><joint/><geom size=".02"/></body>
      </body>
      <body name="D" pos=".1 0 0"><joint/><geom size=".02"/></body>
    </body></worldbody></mujoco>''')
    source = dict(
        joint_names=["s0", "s1", "s2", "s3"],
        joint_parents=[-1, 0, 1, 1],
        reference_joint_positions=np.array([[0, 0, 0], [0, 0, .4], [0, 0, 1], [.1, 0, .4]]),
    )
    mapping = {"s0": "A", "s2": "C", "s3": "D"}
    original = dict(source_joint="s2", robot_body="C")
    for human_scale, expected in ((1, .5), (2, .25)):
        measured_source = dict(source, reference_joint_positions=human_scale * source["reference_joint_positions"])
        specs, report = calibrate_terminal_scales(
            model, model.qpos0, mapping, measured_source, {"tool": original},
        )
        assert np.isclose(specs["tool"]["swing_morphology_scale"], expected)
        assert report["tool"]["source_chain_root"] == "s0"
        assert report["tool"]["chain_selection"] == "nearest_mapped_ancestor"
    assert "swing_morphology_scale" not in original
    # A source ancestor mapped onto the wrong robot branch is not evidence.
    _, incompatible = calibrate_terminal_scales(
        model, model.qpos0, dict(mapping, s0="D"), source, {"tool": original},
    )
    assert incompatible["tool"]["status"] == "degenerate_or_incompatible_chain"


def test_adapter_target_semantics_do_not_assign_support_roles():
    from types import SimpleNamespace
    from omniretargeting._core.contact_inputs import unified_contact_inputs
    from tests.test_source_contacts import skin, floor

    owner = SimpleNamespace(
        smplx_joint_names=["x", "y"],
        retargeting_config={"end_effector_priority": {"effectors": {
            "one": {"source_joint": "x"}, "two": {"source_joint": "y"},
        }}},
    )
    positions = np.tile([[0, 0, .5], [.2, 0, .5]], (4, 1, 1))
    full = skin(positions[:, 0])
    full["triangles"] = np.concatenate(
        [full["triangles"], skin(positions[:, 1])["triangles"]], axis=1,
    )
    full.update(
        barycentric=np.ones((2, 3)) / 3, labels=["r0", "r1"],
        joint_names=["x", "y"], joint_parents=[-1, -1],
        sample_joint_indices=[0, 1], world_target_joints=["x"],
    )
    tasks, _ = unified_contact_inputs(owner, {"full_body": full}, positions, floor(), [], {}, 30)
    assert not tasks["one"].contact_mask.any()
    assert not tasks["two"].contact_mask.any()
    assert tasks["one"].swing_world_mask.all()
    assert not tasks["two"].swing_world_mask.any()
