"""URDF syntax must not change whether an existing floating base is detected."""

import mujoco
import pytest

from omniretargeting.utils import _has_floating_joint, load_robot_urdf_with_floating_base


@pytest.mark.parametrize("attribute", ['type="floating"', "type='floating'", "type = 'floating'"])
def test_floating_joint_detection_uses_xml_attributes(attribute):
    assert _has_floating_joint(f'<robot name="test"><joint name="root" {attribute}/></robot>')


@pytest.mark.parametrize("nonjoint", [
    '<!-- <joint name="root" type="floating"/> -->',
    '<transmission name="root" type="floating"/>',
    '<gazebo><joint name="root" type="floating"/></gazebo>',
])
def test_floating_joint_detection_ignores_comments_and_non_urdf_joints(nonjoint):
    assert not _has_floating_joint(f'<robot name="test">{nonjoint}</robot>')


@pytest.mark.parametrize("existing", [False, True])
def test_robot_loading_keeps_exactly_one_floating_base(tmp_path, existing):
    links = """<link name='base'>
      <inertial><mass value='1'/><inertia ixx='0.1' iyy='0.1' izz='0.1'
        ixy='0' ixz='0' iyz='0'/></inertial>
    </link>"""
    joint = """<link name='world'/><joint name='source_root' type = 'floating'>
      <parent link='world'/><child link='base'/>
    </joint>""" if existing else ""
    path = tmp_path / 'robot.urdf'
    path.write_text(f"<robot name='test'>{links}{joint}</robot>")

    model = load_robot_urdf_with_floating_base(path)

    assert model.njnt == 1
    assert model.nq == 7
    assert model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
    assert model.joint(0).name == ('source_root' if existing else 'floating_base')
