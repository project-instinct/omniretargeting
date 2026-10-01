"""Compare source skin and robot meshes against the same original scene.

Closed-object depths and open-scan inside candidates are distinct outputs.
Generalized winding close to one on an open scan is intersection evidence,
not a certificate of a solid or its shortest full exit distance. No input geometry is sealed, convexified or repositioned.
"""
import argparse
import json
from pathlib import Path

import igl
import mujoco
import numpy as np
import trimesh

from omniretargeting.scene_geometry import SceneQuery


def summary(values, names):
    frame, region = np.unravel_index(np.argmax(values), values.shape)
    return {
        'max_m': float(values[frame, region]), 'worst_frame': int(frame),
        'worst_region': str(names[region]),
        'affected_frames': np.flatnonzero(values.max(1) > 1e-6).tolist(),
        'by_region_max_m': dict(zip(map(str, names), map(float, values.max(0)))),
    }


def measure(args):
    source = np.load(args.skin)
    vertices, faces = source['vertices'], source['faces']
    inputs = np.load(args.prepared, allow_pickle=True).item()
    poses = np.load(args.motion)['qpos']
    if len(poses) != len(vertices):
        raise ValueError('Source skin and robot must share the complete timeline')
    model = mujoco.MjModel.from_xml_path(inputs['model_path'])
    data = mujoco.MjData(model)
    groups = []
    for geom in range(model.ngeom):
        if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = model.geom_dataid[geom]
            start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            groups.append((geom, np.array(model.mesh_vert[start:start+count])))
    if len(groups) != model.ngeom:
        raise ValueError('This original-mesh audit requires every robot geom to be a mesh')
    source_names, source_regions = np.unique(source['vertex_regions'], return_inverse=True)
    robot_names, geom_regions = np.unique([model.body(model.geom_bodyid[g]).name for g, _ in groups], return_inverse=True)
    robot_regions = np.concatenate([np.full(len(p), geom_regions[i], int) for i, (_, p) in enumerate(groups)])
    objects = np.load(args.objects, allow_pickle=True).tolist()
    queries = []
    topology = []
    for obj in objects:
        mesh = obj['mesh']
        tree = igl.AABB()
        tree.init(np.asarray(mesh.vertices), np.asarray(mesh.faces))
        evidence = obj.get('collision_evidence', {})
        query = SceneQuery(mesh, evidence.get('points'), evidence.get('normals'),
                           evidence_radius=evidence.get('radius', .15), full_geometry=False)
        closed = np.zeros(len(mesh.faces), bool)
        components = np.empty(len(mesh.faces), int)
        for cid, (_, ids, solid, _) in enumerate(query.components):
            closed[ids] = solid
            components[ids] = cid
        queries.append((tree, closed, query, components))
        topology.append({'name': obj.get('name', 'object'), 'faces': len(mesh.faces),
                         'closed_faces': int(closed.sum()), 'open_faces': int((~closed).sum()),
                         'winding_consistent': bool(mesh.is_winding_consistent)})
    names = {'source': source_names, 'robot': robot_names}
    regions = {'source': source_regions, 'robot': robot_regions}
    arrays = {f'{kind}_{metric}': np.zeros((len(poses), len(labels)))
              for kind, labels in names.items()
              for metric in ['ground_depth', 'open_scan_inside_candidate_depth', 'closed_object_depth', 'observed_open_depth']}
    worst = {'source': None, 'robot': None}
    worst_depth = {'source': 0., 'robot': 0.}
    floor_mesh = trimesh.load_mesh(args.prepared.parent/'static_scene.obj', process=False)
    top = floor_mesh.triangles[np.asarray(floor_mesh.face_normals)[:, 2] > .999999]
    if not len(top) or not np.allclose(top[:, :, 2], top[:, :, 2].mean(), atol=1e-8):
        raise ValueError('Ground skin audit currently requires a planar original floor')
    floor = float(top[:, :, 2].mean())
    for frame, pose in enumerate(poses):
        data.qpos[:] = pose
        mujoco.mj_kinematics(model, data)
        points = {'source': vertices[frame], 'robot': np.concatenate([
            p @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g] for g, p in groups])}
        for kind, world in points.items():
            owners = regions[kind]
            np.maximum.at(arrays[f'{kind}_ground_depth'][frame], owners, np.maximum(floor-world[:, 2], 0))
            for oid, (obj, (tree, closed, query, components)) in enumerate(zip(objects, queries)):
                scale, rot, trans = obj['scales'][frame], obj['rotations'][frame], obj['translations'][frame]
                local = (world-trans) @ rot / scale
                mesh = obj['mesh']
                d2, triangle, closest = tree.squared_distance(np.asarray(mesh.vertices), np.asarray(mesh.faces), local)
                offset = local-closest
                normal = np.asarray(mesh.face_normals)[triangle]
                # A nearest-face halfspace alone is invalid on a scan: an
                # isolated/reversed fragment can put a distant torso "inside".
                # Keep only well separated winding candidates within the
                # original bounds, and report them independently of closed solids.
                winding = np.zeros(len(local))
                in_bounds = np.all((local >= mesh.bounds[0]) & (local <= mesh.bounds[1]), axis=1)
                if np.any(in_bounds):
                    winding[in_bounds] = np.abs(igl.fast_winding_number(
                        np.asarray(mesh.vertices), np.asarray(mesh.faces), local[in_bounds]))
                candidate = (np.abs(winding - 1.) < .01) & ~closed[triangle]
                depth = np.where(candidate, np.sqrt(np.maximum(d2, 0))*scale, 0.)
                np.maximum.at(arrays[f'{kind}_open_scan_inside_candidate_depth'][frame], owners, depth)
                peak = int(np.argmax(depth))
                if depth[peak] > worst_depth[kind]:
                    worst_depth[kind] = float(depth[peak])
                    worst[kind] = dict(frame=frame, region=str(names[kind][owners[peak]]),
                                      object=oid, depth_m=float(depth[peak]),
                                      world_point=world[peak].tolist(),
                                      world_witness=(closest[peak]*scale @ rot.T+trans).tolist(),
                                      world_oriented_normal=(normal[peak] @ rot.T).tolist(),
                                      winding_diagnostic=float(winding[peak]),
                                      nearest_component_closed=bool(closed[triangle[peak]]))
                evidence_distance = (None if query.evidence_tree is None
                                     else query.evidence_tree.query(local)[0])
                for cid, (part, _, solid, _) in enumerate(query.components):
                    if solid:
                        ids = np.flatnonzero(np.all((local >= part.bounds[0]) & (local <= part.bounds[1]), axis=1))
                    elif query._component_query_data[cid][3]:
                        ids = np.flatnonzero((components[triangle] == cid)
                                             & (evidence_distance <= query.evidence_radius))
                    else:
                        continue
                    if not len(ids):
                        continue
                    signed = query.query(local[ids], component_ids=[cid])
                    valid = ~signed['uncertain'] & np.isfinite(signed['clearance'])
                    label = 'closed_object_depth' if solid else 'observed_open_depth'
                    np.maximum.at(arrays[f'{kind}_{label}'][frame], owners[ids[valid]],
                                  np.maximum(-signed['clearance'][valid]*scale, 0))
        if frame % 30 == 0:
            print('MEASURED', frame, '/', len(poses), flush=True)
    report = dict(frames=len(poses), frame_indexing='zero_based', source_unchanged=True,
                  scene_unchanged=True, metric_scale=1., ground_z_m=floor, objects=topology,
                  open_candidate_caveat='Unsigned nearest original-triangle distance where whole-scan winding differs from 1 by less than 0.01. This is tentative intersection evidence for open scans, not certified solid depth or a shortest full exit vector.',
                  sampling='All original source skin vertices and every original robot mesh vertex; vertex sampling can miss triangle intersections.',
                  collision_free_certificate=False, worst_open_candidate=worst,
                  metrics={key: summary(value, names[key.split('_')[0]]) for key, value in arrays.items()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    np.savez_compressed(args.output.with_suffix('.npz'), **arrays, source_regions=source_names, robot_regions=robot_names)
    print(json.dumps({k:{'max_cm':v['max_m']*100, 'frame':v['worst_frame'], 'region':v['worst_region']} for k,v in report['metrics'].items()}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['skin', 'prepared', 'motion', 'objects', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    measure(parser.parse_args())
