"""Compare opt-in source filtering with identical original-source stance masks.

The 0.2 s high pass is x minus centred cubic regression (odd window, interp
edges). RMS is the norm of vector residuals, not a pooled component RMS.
Acceleration uses second finite differences at the published fps. All maxima
retain frame/body attribution; no metric in this script changes a solver/audit.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import mujoco
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

from omniretargeting.data_sources.source_denoise import runs
from omniretargeting.motion_budgets import chatter_excess, CHATTER_EXCESS_TOLERANCE

PARENTS = [-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19]


def stats(values, frame_offset=0):
    x = np.asarray(values, float)
    if not x.size:
        return dict(rms=None, p99=None, max=None, worst_frame=None)
    return dict(rms=float(np.sqrt(np.mean(x*x))), p99=float(np.quantile(x,.99)),
                max=float(x.max()), worst_frame=int(np.argmax(x)+frame_offset))


def lowpass(values, fps):
    size = min(int(round(.2*fps)) | 1, len(values) if len(values)%2 else len(values)-1)
    return savgol_filter(values, max(5,size), 3, axis=0, mode='interp') if len(values)>=5 else values.copy()


def highpass(values, fps):
    return np.linalg.norm(values-lowpass(values,fps),axis=-1)


def contacts(saved, fps):
    out = {}
    for column,name in enumerate(saved.get('contact_names', [])):
        if 'foot' not in str(name):
            continue
        mask = np.asarray(saved['contact_mask'][:,column],bool)
        intervals = runs(mask)
        out[str(name)] = dict(segments=len(intervals), short_segments=int(sum((b-a)/fps<.1 for a,b in intervals)),
                             frames=int(mask.sum()), intervals=[[int(a),int(b)] for a,b in intervals])
    return out


def robot_points(prepared, saved):
    model = mujoco.MjModel.from_xml_path(prepared['model_path'])
    data = mujoco.MjData(model)
    points = {}
    for name,spec in prepared['effectors']['effectors'].items():
        if 'foot' in name:
            points[name] = np.empty((len(saved['qpos']),3))
    for frame,qpos in enumerate(saved['qpos']):
        data.qpos[:]=qpos
        mujoco.mj_kinematics(model,data)
        for name,series in points.items():
            spec=prepared['effectors']['effectors'][name]
            body=model.body(spec['robot_body']).id
            series[frame]=data.xpos[body]+data.xmat[body].reshape(3,3)@spec['tracking_offset']
    return points



def material_stance(prepared, saved):
    """Fixed collision material points: ankle/sole-centre motion includes rolls."""
    surface=prepared['surface']
    model=mujoco.MjModel.from_xml_path(prepared['model_path'])
    data=mujoco.MjData(model)
    ids=np.flatnonzero(np.isin(surface.get('surface_contact_parts',[]),['left_foot','right_foot']))
    points=np.empty((len(saved['qpos']),len(ids),3))
    for frame,q in enumerate(saved['qpos']):
        data.qpos[:]=q
        mujoco.mj_kinematics(model,data)
        for j,k in enumerate(ids):
            gid=int(surface['surface_contact_robot_geoms'][k])
            points[frame,j]=data.geom_xpos[gid]+data.geom_xmat[gid].reshape(3,3)@surface['surface_contact_robot_local_points'][k]
    out={}
    fps=float(saved['framerate'])
    for side in ['left_foot','right_foot']:
        rows=[];jitter=[];jitter_frames=[];jitter_samples=[]
        for j,k in enumerate(ids):
            if surface['surface_contact_parts'][k]!=side:continue
            mask=surface['surface_contact_mask'][:,k] & (surface['surface_contact_modes'][:,k]=='sticking')
            mask &= surface['surface_contact_surface_ids'][:,k]==0
            p=points[:,j];hp=highpass(p,fps)
            for a,b in runs(mask):
                if (b-a)/fps<.15:continue
                n=surface['surface_contact_normals_w'][a:b,k]
                n=n/np.maximum(np.linalg.norm(n,axis=1)[:,None],1e-12)
                drift=p[a:b]-p[a]
                tangent=drift-n*np.sum(drift*n,axis=1)[:,None]
                value=np.linalg.norm(tangent,axis=1)
                rows.append(dict(sample=int(k),start=int(a),stop=int(b),
                    tangent_drift_max_m=float(value.max()),worst_frame=int(a+value.argmax())))
                jitter.extend(hp[a:b].tolist())
                jitter_frames.extend(range(int(a),int(b)))
                jitter_samples.extend([int(k)]*int(b-a))
        jitter_stats=stats(jitter)
        if jitter:
            index=int(np.argmax(jitter))
            jitter_stats.update(worst_frame=jitter_frames[index],sample=jitter_samples[index])
        out[side]=dict(material_frame_samples=len(jitter),jitter=jitter_stats,
            tangent_drift_max_m=max((r['tangent_drift_max_m'] for r in rows),default=None),intervals=rows)
    return out


def evaluate(off_dir, auto_dir):
    dirs = [Path(off_dir),Path(auto_dir)]
    saved = [np.load(d/'final_retargeted.npz') for d in dirs]
    prepared = [np.load(d/'prepared.npy',allow_pickle=True).item() for d in dirs]
    fps=float(saved[0]['framerate'])
    if fps!=float(saved[1]['framerate']) or saved[0]['qpos'].shape!=saved[1]['qpos'].shape:
        raise ValueError('Comparisons require matching source fps, frames and robot model')
    positions=[s['source_joints_w'] for s in saved]
    source_names=list(saved[0]['source_joint_names'])
    robot = [robot_points(p,s) for p,s in zip(prepared,saved)]
    # One audit mask derived from the original source, used for both motions.
    stance={}
    slow_foot={}
    for name in robot[0]:
        spec=prepared[0]['effectors']['effectors'][name]
        joint=source_names.index(spec['source_joint'])
        smooth=lowpass(positions[0][:,joint],fps)
        speed=np.linalg.norm(np.gradient(smooth,1/fps,axis=0),axis=1)
        mask=(speed<.20)&(smooth[:,2]<np.quantile(smooth[:,2],.05)+.04)
        slow_foot[name]=[(a,b) for a,b in runs(mask) if (b-a)/fps>=.15]
        column=list(saved[0]['contact_names']).index(name)
        mask &= saved[0]['contact_mask'][:,column]
        mask &= saved[0]['contact_modes'][:,column]=='sticking'
        mask &= saved[0]['contact_environment_ids'][:,column]==0
        stance[name]=[(a,b) for a,b in runs(mask) if (b-a)/fps>=.15]
    result=dict(fps=fps, frames=len(positions[0]), highpass_method='centred_cubic_0.2s_vector_RMS',
        acceleration_method='second_finite_difference_vector_RMS', stance_mask='original_source_slow_low_foot_AND_original_geometric_sticking_static_contact',
        off={},auto={})
    for mode,s,source,points,directory in zip(['off','auto'],saved,positions,robot,dirs):
        out=result[mode]
        out['source_highpass_m']={name:stats(highpass(source[:,source_names.index(name)],fps))
                                 for name in ['Pelvis','L_Ankle','R_Ankle']}
        out['source_root_acceleration_m_s2']=stats(np.linalg.norm(np.diff(source[:,0],n=2,axis=0),axis=1)*fps**2,1)
        out['robot_root_acceleration_m_s2']=stats(np.linalg.norm(np.diff(s['qpos'][:,:3],n=2,axis=0),axis=1)*fps**2,1)
        out['robot_highpass_m']={name:stats(highpass(value,fps)) for name,value in points.items()}
        out['stance']={}
        for name,intervals in stance.items():
            p=points[name];hp=highpass(p,fps)
            rows=[]
            for a,b in intervals:
                rows.append(dict(start=int(a),stop=int(b),hp_rms_m=float(np.sqrt(np.mean(hp[a:b]**2))),
                    drift_max_m=float(np.linalg.norm(p[a:b]-p[a],axis=1).max()),
                    speed_p99_m_s=float(np.quantile(np.linalg.norm(np.diff(p[a:b],axis=0),axis=1)*fps,.99))))
            frames=np.concatenate([np.arange(a,b) for a,b in intervals]) if intervals else np.array([],int)
            jitter=stats(hp[frames])
            if len(frames):jitter['worst_frame']=int(frames[np.argmax(hp[frames])])
            out['stance'][name]=dict(frames=len(frames),jitter=jitter,
                drift_max_m=max((r['drift_max_m'] for r in rows),default=None),intervals=rows)
        reference=np.load(dirs[0]/'fixed_reference.npz')['qpos']
        excess=chatter_excess(reference[:,7:],s['qpos'][:,7:])
        frame,joint=np.unravel_index(excess.argmax(),excess.shape)
        out['chatter_original_reference']=dict(violations=int(np.count_nonzero(excess>CHATTER_EXCESS_TOLERANCE)),
            worst_frame=int(frame),joint_index=int(joint),max_excess_rad=float(excess.max()),
            root_translation_violations=int(np.count_nonzero(chatter_excess(reference[:,:3],s['qpos'][:,:3])>CHATTER_EXCESS_TOLERANCE)))
        out['material_stance']=material_stance(prepared[0],s)
        out['contacts']=contacts(s,fps)
        out['production_seconds']=json.loads((directory/'timing.json').read_text()).get('production_seconds')
        out['denoise']=json.loads(str(s['source_denoise_json'])) if 'source_denoise_json' in s else dict(mode='off')
    delta=saved[1]['qpos']-saved[0]['qpos']
    frame,column=np.unravel_index(np.abs(delta).argmax(),delta.shape)
    result['qpos_difference']=dict(bitwise_equal=bool(np.array_equal(saved[0]['qpos'],saved[1]['qpos'])),
        absolute_max=float(np.abs(delta).max()),worst_frame=int(frame),qpos_column=int(column),
        joint_max_rad=float(np.abs(delta[:,7:]).max()),root_max_m=float(np.linalg.norm(delta[:,:3],axis=1).max()))
    angles=[];rows={}
    for joint,parent in enumerate(PARENTS[1:],1):
        bones=[p[:,joint]-p[:,parent] for p in positions]
        unit=[v/np.maximum(np.linalg.norm(v,axis=1)[:,None],1e-12) for v in bones]
        angle=np.arccos(np.clip(np.sum(unit[0]*unit[1],axis=1),-1,1))
        rows[source_names[joint]]=stats(angle)
        angles.append(angle)
    result['source_pose_change_rad']=dict(mean=float(np.mean(angles)),p99=float(np.quantile(angles,.99)),
                                        joints=rows)
    rotations=[]
    for d in dirs:
        with (d/'source.pkl').open('rb') as handle:rotations.append(pickle.load(handle).metadata['joint_orientations'])
    a,b=[Rotation.from_quat(r.reshape(-1,4),scalar_first=True) for r in rotations]
    angle=(a.inv()*b).magnitude().reshape(rotations[0].shape[:2])
    frame,joint=np.unravel_index(angle.argmax(),angle.shape)
    result['source_rotation_change_rad']=dict(mean=float(angle.mean()),p99=float(np.quantile(angle,.99)),
        max=float(angle.max()),worst_frame=int(frame),worst_joint=str(source_names[joint]))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--off-dir',type=Path,required=True)
    parser.add_argument('--auto-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=evaluate(args.off_dir,args.auto_dir)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in {'off','auto'}},indent=2))


if __name__=='__main__':main()
