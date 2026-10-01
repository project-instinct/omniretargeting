"""Mesh restoration must recover the immutable legal layout, in its local scope."""
import mujoco
import numpy as np
from scipy import sparse
from omniretargeting._fast.objective import FastObjective
from omniretargeting._fast.reference import ReferenceShape
from omniretargeting._fast.repair_limits import RepairLimits
from omniretargeting._fast.repair_restoration import restore_repair
from omniretargeting.sequence_retargeting import prepare_sequence, sequence_config


def test_mesh_restore_returns_flipped_candidate_without_changing_reference(monkeypatch):
    model = mujoco.MjModel.from_xml_string('''
    <mujoco><compiler angle="radian"/><worldbody>
      <body name="root"><freejoint/><geom size=".01"/>
        <body name="chest" pos="0 0 .5"><geom size=".01"/>
          <body name="lever"><joint axis="0 1 0" range="-1 1"/><geom size=".01"/>
            <body name="hand" pos="0 0 -.3"><geom size=".01"/></body></body>
          <body name="other" pos="0 -.3 -.3"><geom size=".01"/></body>
        </body></body></worldbody></mujoco>''')
    reference = np.tile(model.qpos0, (5, 1)); reference[:, -1] = -.3
    names = ['root', 'chest', 'hand', 'other']
    data = mujoco.MjData(model); data.qpos[:] = reference[0]; mujoco.mj_forward(model, data)
    human = np.tile(data.xpos[[model.body(n).id for n in names]], (5, 1, 1))
    cfg = sequence_config({'self_collision_enabled': False})
    coords, _, tasks, rotations = prepare_sequence(model, reference, human, names,
        {n:n for n in names}, human[:,0], {}, {}, cfg)
    objective = FastObjective(tasks, cfg, reference=reference,
        temporal_reference=reference[:,coords.qi], root_rotations=rotations, framerate=30.)
    shape = objective.source_shape = ReferenceShape(objective, reference)
    shape.freeze_reference(reference)
    objective.candidate_guards=[]
    origin=reference.copy();origin[:,-1]=-.02
    limits=RepairLimits(objective,origin)
    assert limits.check(origin)[0]
    candidate=origin.copy();candidate[:,-1]=.01
    assert not limits.check(candidate)[0]
    class Geometry:
        def linear_rows(self, measurement):
            return sparse.csr_matrix((0,len(origin))), np.zeros(0)
    measurement=dict(rows=[],depths={})
    args=(limits,origin,candidate,Geometry(),measurement,measurement,sparse.eye(len(origin),format='csc'))
    original_bounds=shape.repair_bounds
    with monkeypatch.context() as context:
        context.setattr(shape,'repair_bounds',lambda q, **kwargs:original_bounds(q,restore=False))
        old, report=restore_repair(*args)
    assert old is None, report
    restored,report=restore_repair(*args)
    assert restored is not None,report
    assert limits.check(restored)[0]
    assert shape.accepts(restored)
    assert np.max(np.abs(restored[:,-1]))<1e-5
    np.testing.assert_array_equal(restored[:,:7],origin[:,:7])
    np.testing.assert_array_equal(shape.seed,reference)
