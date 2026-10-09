import numpy as np

from scripts.review_penetration_persistence import compress_frame_indices


def test_compress_frame_indices_uses_inclusive_intervals():
    compressed = compress_frame_indices(np.array([1, 2, 3, 7, 8, 10, 20], dtype=int))
    assert compressed == [[1, 3], [7, 8], [10, 10], [20, 20]]


def test_missing_support_metadata_is_not_a_support_certificate():
    from scripts.review_penetration_persistence import _support_fields
    result = _support_fields({}, np.zeros(2), np.arange(2),
        {'source':'unavailable_no_prepared_support_evidence'}, {})
    assert not result['support_normal']
    assert not result['support_frames'].any()


def test_secondary_lateral_pair_invalidates_reset_despite_deeper_support():
    import mujoco
    from scripts.review_penetration_persistence import _lateral_reset_mask, _reset_report
    model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="foot"><geom type="sphere" size=".1"/></body>
      <body name="arm"><geom type="sphere" size=".1"/></body></worldbody></mujoco>''')
    depth=np.array([[.004,.002],[.004,0.],[.004,.0010005]])
    support={'by_body_surface':{(model.body('foot').id,0):np.ones(3,bool)}}
    lateral=_lateral_reset_mask(model,depth,[],np.zeros_like(depth),support)
    assert lateral.tolist()==[True,False,False]
    r=_reset_report(static_depth=depth.max(axis=1),moving_depth=np.zeros(3),self_depth=np.zeros(3),
        labels=np.arange(3),scene_unknown=False,self_error_frames=set(),lateral=lateral,
        support_unverified=np.ones(3,bool))
    assert r['states']==['invalid','unknown','unknown']
