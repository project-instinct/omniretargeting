"""Missing provenance is unknown; known asset/geometry differences are visible."""
import mujoco
import numpy as np
from omniretargeting.collision_manifest import collision_manifest
from omniretargeting.training_collision_match import training_collision_match


def test_collision_identity_and_missing_provenance(tmp_path):
    path = tmp_path/'robot.xml'
    path.write_text('<mujoco><worldbody><body name="robot"><freejoint/>'
                    '<geom type="sphere" size=".1"/></body></worldbody></mujoco>')
    model = mujoco.MjModel.from_xml_path(str(path)); manifest=collision_manifest(model)
    result=training_collision_match(manifest, {'collision_model_sha256':np.asarray(manifest['sha256'])},path)
    assert result['status']=='matched'
    missing=training_collision_match(manifest,{},path)
    assert missing['status']=='unverified'
    assert missing['training_model']['status']=='matched'
    mismatch=training_collision_match(manifest,{'collision_model_sha256':'wrong'},path)
    assert mismatch['status']=='mismatched'
    model.geom_size[0,0]+=.01
    other=training_collision_match(collision_manifest(model),
        {'collision_model_sha256':manifest['sha256']},path)
    assert other['status']=='mismatched'
    assert other['training_model']['status']=='mismatched'
    absent=training_collision_match(manifest,{},tmp_path/'missing.xml')
    assert absent['status']=='unverified'
