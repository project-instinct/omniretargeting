import numpy as np
from omniretargeting.contact_side import ContactSides


def test_contact_side_preserves_sign_groups_and_sample_frame_denominator():
    sides=ContactSides()
    sides.add('left_wrist',1,[0,0,1,2],[.02,-.02,-.01,-.03])
    sides.add('left_wrist',0,[0],[.03])
    sides.add('right_wrist',1,[1],[.01])
    result=sides.report()['pairs']
    item=result['left_wrist|surface:1']
    assert item['median_signed_m']==-.015
    assert item['backside_gt1cm_fraction']==.5
    assert item['count']==4 and item['active_frames']==3
    assert item['backside_frames']==[0,2] and item['worst_frame']==2
    assert item['p95_abs_m']==np.quantile([.02,.02,.01,.03],.95)
    assert len(result)==3
