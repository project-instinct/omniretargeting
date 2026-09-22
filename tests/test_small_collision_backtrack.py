"""A valid small self-collision backtrack must not freeze every joint."""
from pathlib import Path

import mujoco
import numpy as np
import pytest

from omniretargeting.retargeting import GenericInteractionRetargeter
from omniretargeting.self_collision import SelfCollisionQuery


@pytest.mark.parametrize("coordinates", ["qpos", "tangent"])
def test_g1_cylinder_feature_change_retains_small_valid_joint_step(coordinates):
    model = mujoco.MjModel.from_xml_path(str(
        Path(__file__).resolve().parents[1] / "robot_models/unitree_g1/g1_29dof_popsicle.urdf"
    ))
    before = np.array([
        0.000653256311654401,
        0.05652767419921015,
        0.7881028062811212,
        -0.5246026118633837,
        -0.08002062121615125,
        -0.1683083656825532,
        0.8306991596514138,
        -0.6828273791901646,
        0.4505656130049214,
        0.43514701821015106,
        0.6615014497608909,
        -0.16251984554347007,
        -0.02241357380169661,
        -0.8069252770609918,
        -0.01068463837464875,
        -0.13275263346997526,
        0.7491463331440918,
        -0.054606959397847134,
        0.11533980545599393,
        0.5855471473266821,
        0.3213725790940052,
        0.38031900778831124,
        0.25945780810117286,
        0.022619067897191213,
        -0.3268730030536619,
        0.6086246943261363,
        -0.8363439754352162,
        -0.6093741711018223,
        0.02472672255361655,
        -0.04935421672012251,
        0.0845855264472311,
        -0.9827914495984453,
        0.9472187587361741,
        0.09721247557896107,
        0.8942781091613734,
        0.41368582822674
    ])
    candidate = np.array([
        0.0006532563128280793,
        0.05652767419793872,
        0.7881255589174163,
        -0.5246047368064025,
        -0.08004174186159714,
        -0.1683869556956255,
        0.830679855798614,
        -0.6614083073786517,
        0.45161131840215313,
        0.43308069072015665,
        0.6651244675535118,
        -0.16332976492809845,
        -0.021171507246996287,
        -0.7923163009295381,
        -0.007018432338259979,
        -0.13154913269085483,
        0.7471069405765501,
        -0.05551311548550512,
        0.11508441034539162,
        0.568885200911031,
        0.31287160045323803,
        0.35174117369148394,
        0.15918947734558625,
        -0.01767962605559353,
        -0.3280553430662231,
        0.538117812846669,
        -0.8354563462695638,
        -0.608493925380207,
        0.024374409224688864,
        -0.1658776874781641,
        0.048166427110949275,
        -0.9483631792932804,
        0.8739713510391866,
        0.0994193095555332,
        0.8916349936670763,
        0.41515147136176167
    ])
    query = SelfCollisionQuery(model)
    solver = GenericInteractionRetargeter.__new__(GenericInteractionRetargeter)
    solver.robot_model = model
    solver.self_collision_enabled = True
    solver.end_effector_config = {"step_coordinates": coordinates}
    solver._defer_self_collision_path_repair = True
    solver._self_collision_query = query
    original = query.depths(before)
    assert not query.acceptable(original, query.depths(candidate))
    result = solver._guard_self_collision_step(before, candidate, before)
    # The rejected full proposal crosses a cylinder contact feature. A much
    # smaller internal-joint step is valid at the existing 1 mm tolerance.
    assert np.linalg.norm(result[7:] - before[7:]) > 1e-5
    assert np.linalg.norm(result[7:] - before[7:]) < np.linalg.norm(candidate[7:] - before[7:])
    if coordinates == "tangent":
        proposed, accepted = np.zeros(model.nv), np.zeros(model.nv)
        mujoco.mj_differentiatePos(model, proposed, 1., before, candidate)
        mujoco.mj_differentiatePos(model, accepted, 1., before, result)
        fraction = np.linalg.norm(accepted[6:]) / np.linalg.norm(proposed[6:])
        np.testing.assert_allclose(accepted, fraction * proposed, atol=1e-12)
    else:
        np.testing.assert_array_equal(result[:7], candidate[:7])
    assert query.acceptable(original, query.depths(result))
    path = query.segment(before, result)
    assert not path["subdivision_limit"]
    assert query.acceptable(original, path["depths"])
