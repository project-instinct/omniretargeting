"""Always-overlapping model geometry is not a self-collision of the motion."""

import mujoco

from omniretargeting.self_collision import SelfCollisionQuery

XML = """<mujoco><worldbody>
  <body name="base"><freejoint/><geom size=".05"/>
    <body name="upper" pos="0 0 -.2"><joint axis="0 1 0" range="-1 1"/>
      <geom type="capsule" fromto="0 0 0 0 0 -.2" size=".05"/>
      <body name="lower" pos="0 0 -.2"><joint axis="0 1 0" range="-1 1"/>
        <geom type="capsule" fromto="0 0 -.06 0 0 -.25" size=".04"/>
        <!-- ``shell`` sits on the knee axis inside ``upper``: any angle overlaps. -->
        <body name="shell"><joint axis="0 0 1" range="-.2 .2"/><geom size=".03"/></body>
      </body>
    </body>
    <body name="arm" pos=".3 0 0"><joint axis="0 1 0" range="-3 3"/>
      <geom type="capsule" fromto="0 0 0 0 0 -.3" size=".03"/></body>
  </body>
</worldbody></mujoco>"""


def test_pairs_overlapping_in_every_configuration_are_excluded():
    model = mujoco.MjModel.from_xml_string(XML)
    query = SelfCollisionQuery(model)
    names = [sorted(pair) for pair in query.structural_pairs]
    assert ["shell", "upper"] in names
    kept = {tuple(sorted((model.body(int(model.geom_bodyid[a])).name,
                          model.body(int(model.geom_bodyid[b])).name))) for a, b in query.pairs}
    # A swinging arm collides only in some configurations: still checked.
    assert ("arm", "upper") in kept or ("arm", "lower") in kept
