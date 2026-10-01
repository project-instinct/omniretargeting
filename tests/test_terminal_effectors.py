import mujoco
import numpy as np

from omniretargeting.retargeting_defaults import apply_retargeting_defaults
from omniretargeting.terminal_effectors import infer_terminal_effectors


def test_mapped_terminals_and_welded_shell_do_not_need_anatomical_names():
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
    <body name="base"><freejoint/><geom size=".02"/>
      <body name="branch" pos=".3 0 0"><joint name="j"/><geom size=".02"/>
       <body name="last"><joint name="k"/><geom size=".01"/>
        <body name="tool" pos=".1 0 0"><geom type="box" size=".05 .03 .02"/></body>
       </body></body>
      <body name="other" pos="0 .4 0"><joint name="l"/><geom size=".02"/></body>
    </body></worldbody></mujoco>""")
    effectors = infer_terminal_effectors(
        model, {"A": "base", "B": "branch", "C": "other"}
    )
    assert set(effectors) == {"B", "C"}
    assert effectors["B"]["robot_body"] == "last"
    points = np.array(effectors["B"]["contact_offsets"])
    assert np.isclose(points[:, 0].max(), 0.15)
    assert effectors["B"]["source_joint"] == "B"
    assert "preserve_world_target" not in effectors["B"]
    override = infer_terminal_effectors(
        model,
        {"A": "base", "B": "branch", "C": "other"},
        {
            "custom": {
                "robot_body": "tool",
                "source_joint": "B",
                "tracking_offset": [1, 2, 3],
            }
        },
    )
    assert set(override) == {"custom", "C"}
    assert override["custom"]["tracking_offset"] == [1, 2, 3]


def test_default_algorithm_does_not_require_explicit_effectors():
    config = apply_retargeting_defaults({})
    assert config["surface_contact"]["enabled"]
    assert config["end_effector_priority"]["enabled"]
    assert config["end_effector_priority"]["effectors"] == {}
