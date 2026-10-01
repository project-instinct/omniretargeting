"""Sole/contact extraction follows shaped skin, including heel-only contact."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from omniretargeting.data_sources.smplx_feet import smplx_sole_metadata
from omniretargeting.data_sources.smplx_motion import _extract_smplx_end_effector_markers


def subject(scale=1.):
    # Whole feet with heel assigned to ankle and forefoot assigned to toe.
    one = np.array([[x, y, z] for x in [-.13, -.09, -.05, 0, .04, .08, .13]
                    for y in [-.045, 0, .045] for z in [0, .035]])
    vertices = np.vstack([one + [0, .15, 0], one - [0, .15, 0]]) * scale
    weights = np.zeros((len(vertices), 55))
    joints = np.zeros((55, 3))
    for side, (ankle, toe) in enumerate([(7, 10), (8, 11)]):
        ids = np.arange(len(one)) + len(one)*side
        weights[ids, np.where(one[:, 0] < 0, ankle, toe)] = 1
        joints[ankle] = [-.07*scale, (.15 if side == 0 else -.15)*scale, .09*scale]
        joints[toe] = [.07*scale, joints[ankle, 1], .04*scale]
    # Make the synthetic regressor reproduce the supplied anatomical joints.
    regressor = joints @ np.linalg.pinv(vertices)
    model = SimpleNamespace(v_template=torch.tensor(vertices), lbs_weights=torch.tensor(weights),
                            J_regressor=torch.tensor(regressor))
    return vertices, joints, weights, model


@pytest.mark.parametrize('scale', [.7, 1.8])
def test_skin_frame_shared_with_marker_while_toe_joint_articulates(scale):
    vertices, joints, weights, model = subject(scale)
    rotation = Rotation.from_euler('y', [[0], [80]], degrees=True).as_matrix()
    skin = np.einsum('tij,nj->tni', rotation, vertices) + [1, 2, 3]
    orientations = np.tile([1., 0, 0, 0], (2, 22, 1))
    # Joint-only transport disagrees with the observable skin.
    orientations[:, [10, 11]] = Rotation.from_euler('y', [[20], [120]], degrees=True).as_quat(scalar_first=True)[:, None]
    metadata = smplx_sole_metadata(vertices, joints, weights, orientations, skin)
    output = SimpleNamespace(vertices=torch.tensor(skin), joints=torch.tensor(np.tile(joints, (2, 1, 1))))
    markers = _extract_smplx_end_effector_markers(model, output, world_joint_orientations=orientations,
                                                sole_metadata=metadata, max_samples=len(vertices))
    for side in ['left', 'right']:
        sole = metadata['link_orientations'][side]
        np.testing.assert_allclose(sole, rotation, atol=1e-12)
        np.testing.assert_array_equal(markers[side+'_foot']['up_directions'], sole[:, :, 2])
        np.testing.assert_allclose(metadata['foot_sole_fit_rms'][side], 0, atol=1e-12)
        ids = markers[side+'_foot']['vertex_ids']
        ankle, toe = (7, 10) if side == 'left' else (8, 11)
        assert np.any(weights[ids, ankle] == 1), 'heel-only contacts must remain observable'
        np.testing.assert_allclose(markers[side+'_foot']['tracking_positions'], skin[:, weights[:, toe] > .25].mean(1))
        np.testing.assert_array_equal(markers[side+'_foot']['contact_samples'], skin[:, ids])
