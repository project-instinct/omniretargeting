import numpy as np

from omniretargeting._fast.mesh_repair import OriginalGeometry
from tests.test_repair_penetration_yield import _problem


class Plane:
    def query(self, points):
        return dict(clearance=points[:,2].copy(), uncertain=np.zeros(len(points),bool),
                    directions=np.tile([0.,0.,1.],(len(points),1)))




def test_static_manifold_retains_original_material_ids_across_batch_boundaries():
    objective, poses = _problem()
    geometry=OriginalGeometry(objective,[],Plane(),manifold=True)
    geometry.SCENE_BATCH_POINTS=7
    measured=geometry.measure(poses)
    rows=[(k,r) for k,r in measured['rows'] if k[1]=='static']
    assert rows
    for key,row in rows:
        assert objective.shell_sample_geom[row['sample_id']]==key[-1]
