"""Exact source-only solve overlapped with independent contact certificates."""
from contextlib import redirect_stdout
from dataclasses import fields, is_dataclass, replace
import hashlib
import io
import struct
import time
from types import SimpleNamespace

import numpy as np

from .preprocessing_parallel import ForkJob, can_parallelize


def source_key(initial, positions, names, mapping, roots, tasks, priority, config, common):
    """All source-phase inputs, including bits and order; unknown types decline."""
    from .._fast.reference import kinematic_tracking_tasks
    digest = hashlib.sha256()
    def visit(value):
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                raise TypeError('object array')
            visit((value.dtype.str, value.shape));digest.update(value.tobytes())
        elif is_dataclass(value):
            visit(type(value).__module__+'.'+type(value).__qualname__)
            visit([(f.name,getattr(value,f.name)) for f in fields(value)])
        elif isinstance(value, dict):
            digest.update(b'dict');visit(list(value.items()))
        elif isinstance(value, (tuple,list)):
            digest.update(type(value).__name__.encode());digest.update(struct.pack('Q',len(value)))
            for item in value:visit(item)
        elif value is None or isinstance(value,(str,bool,int,float,np.generic)):
            digest.update(type(value).__name__.encode())
            encoded = struct.pack('d',value) if isinstance(value,float) else repr(value).encode()
            digest.update(struct.pack('Q',len(encoded)));digest.update(encoded)
        else:
            raise TypeError(type(value).__name__)
    source_tasks = {name:replace(task,surface_managed_mask=None)
                    for name,task in kinematic_tracking_tasks(tasks,priority).items()}
    # All source tasks have surface_managed=False and no contact phase;
    # subsequently attached surface coverage cannot affect their rows.
    try:
        visit((initial,positions,names,mapping,roots,source_tasks,
               {name:task.contact_mask for name,task in tasks.items()},
               priority,config,{k:common[k] for k in ('framerate','environment_points',
                   'root_orientation_targets','body_orientation_targets')}))
    except TypeError:
        return None
    return digest.digest()


def compact_source_transport(shape,registration):
    """Evict completed registration Jacobians before IPC; guards rebuild them."""
    for kin in (shape.kin,registration.kin):
        for values in kin.cache.values():
            values.pop('jacp',None);values.pop('jacr',None)
        for name in ('jacp','jacr'):
            if hasattr(kin,name):delattr(kin,name)
        kin.with_jacobian = False


def source_registration_view(registration,shape):
    """The exact fields ReferenceShape consumes, without finished native handles."""
    return SimpleNamespace(model=registration.model,coords=registration.coords,
                           kin=shape.kin,envelope_bodies=registration.envelope_bodies,
                           tasks=registration.tasks,cfg=registration.cfg)


def _prepare(owner, retargeter, positions, tasks, orientations, objects, joint_orientations):
    from .sequence import solve_core_sequence
    stream = io.StringIO()
    with redirect_stdout(stream):
        result = solve_core_sequence(owner,retargeter,positions,tasks,orientations,objects,
                                     source_joint_orientations=joint_orientations,source_only=True)
    result['output'] = stream.getvalue()
    return result


class SourcePosePrefetch:
    def __init__(self, owner, retargeter, positions, tasks, orientations, objects,
                 reference, joint_orientations, active):
        from .._fast.runtime_options import OPTIONS
        self.owner = owner;self.job = None
        if not active or reference is not None or not can_parallelize() or len(positions)<128 or OPTIONS.performance_trace:
            return
        self.job = ForkJob(_prepare,(owner,retargeter,positions,tasks,orientations,objects,joint_orientations),
                           references=(owner.robot_model,))
        owner._source_pose_prefetch = self

    def consume(self, key, callback):
        if self.job is None:
            return None
        tick = time.monotonic()
        try:
            result = self.job.receive()
        except Exception:
            # A speculative solve/transport failure is reproduced by the
            # ordinary source path, where the caller consumes its exception.
            return None
        finally:
            self.job = None
        if key is None or key != result['key']:
            return None
        print(result.pop('output'),end='')
        for pose, record in result.pop('callbacks'):
            if callback is not None:callback(pose,record)
        result['stage']['precompute_reused'] = True
        result['stage']['precomputed_seconds'] = result['stage']['seconds']
        result['stage']['seconds'] = time.monotonic()-tick
        return result

    def close(self):
        if self.job is not None:
            self.job.close();self.job = None
        if getattr(self.owner,'_source_pose_prefetch',None) is self:
            del self.owner._source_pose_prefetch

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
