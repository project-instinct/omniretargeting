"""Ordered preprocessing futures; independent evidence retains its exact math."""
from contextlib import redirect_stdout
import io
import multiprocessing
import os
import signal
from types import SimpleNamespace

import numpy as np

from ..cpu_budget import available_cpus

# Frozen ten-clip comparison retains every accepted qpos bit. Serial remains
# available for small inputs, daemon workers, or platforms without fork.
ENABLED = True


def can_parallelize():
    return (ENABLED and hasattr(os, 'fork') and available_cpus() > 1
            and not multiprocessing.current_process().daemon)


def _readonly_paths(value, path=(), seen=None):
    seen = set() if seen is None else seen
    if id(value) in seen:
        return []
    seen.add(id(value))
    if isinstance(value, np.ndarray):
        return [path] if not value.flags.writeable else []
    if isinstance(value, dict):
        return [p for k, v in value.items() for p in _readonly_paths(v, path+(k,), seen)]
    if isinstance(value, (tuple, list)):
        return [p for k, v in enumerate(value) for p in _readonly_paths(v, path+(k,), seen)]
    if (type(value).__module__.startswith('omniretargeting')
            or isinstance(value,SimpleNamespace)) and hasattr(value,'__dict__'):
        return _readonly_paths(vars(value), path+('__dict__',), seen)
    return []


def _serve(function, args, connection, transport=None):
    # A cancelled future owns its pool descendants as well as its coordinator.
    if hasattr(os,'setsid'):os.setsid()
    from multiprocessing.util import register_after_fork
    register_after_fork(connection,lambda pipe:pipe.close())
    try:
        result = function(*args)
        message = ('ok', result, _readonly_paths(result))
    except BaseException as error:
        message = ('error', error, [])
    try:
        try:
            if transport is None:connection.send(message)
            else:connection.send_bytes(transport.pack(message))
        except Exception as error:
            failure=('error',error,[])
            if transport is None:connection.send(failure)
            else:connection.send_bytes(transport.pack(failure))
    finally:
        connection.close()


class ForkJob:
    """A bounded caller owns every process and receives in original order."""
    def __init__(self, function, args, *, references=None):
        context = multiprocessing.get_context('fork')
        self.transport = None
        if references is not None and os.path.isdir('/dev/shm'):
            from .preprocess_transport import ArrayTransport
            self.transport = ArrayTransport(references)
        self.connection, child = context.Pipe(duplex=False)
        self.process = context.Process(target=_serve,args=(function,args,child,self.transport))
        self.closed = False
        try:
            self.process.start()
        except BaseException:
            self.connection.close();child.close()
            if self.transport is not None:self.transport.close()
            raise
        child.close()

    def receive(self):
        try:
            message = (self.connection.recv() if self.transport is None
                       else self.transport.unpack(self.connection.recv_bytes()))
            status, result, readonly = message
            self.process.join()
            if status == 'error':
                raise result
            for path in readonly:
                value = result
                for key in path:
                    value = vars(value) if key == '__dict__' else value[key]
                value.setflags(write=False)
            return result
        finally:
            self.close()

    def close(self):
        if self.closed:
            return
        if self.process.is_alive():
            try:
                if hasattr(os,'killpg') and os.getpgid(self.process.pid)==self.process.pid:
                    os.killpg(self.process.pid,signal.SIGTERM)
                else:
                    self.process.terminate()
            except ProcessLookupError:
                pass
        self.process.join();self.connection.close();self.closed = True
        if self.transport is not None:self.transport.close()


def _evidence(kind, owner, source, positions, scene, config, workers):
    from . import surfaces
    from ..support_parallel import SupportPool
    output = io.StringIO()
    with redirect_stdout(output), SupportPool(workers=workers) as pool:
        if kind == 'source':
            surfaces._prepare_source_support(owner,source,positions,scene,config,map_function=pool.map)
            names = ('_support_evidence_built','_source_support_evidence','last_source_support_diagnostics')
        else:
            surfaces._prepare_object_support(owner,scene,config,map_function=pool.map)
            names = ('_object_support_built','_object_support_evidence','last_object_support_diagnostics')
    return {name:getattr(owner,name) for name in names if hasattr(owner,name)}, output.getvalue()


class EvidencePrefetch:
    """Compute source/object certificates while the owner binds robot surfaces."""
    def __init__(self, owner, source, positions, scene, config, active):
        self.owner = owner;self.jobs = {}
        if (not active or not can_parallelize() or len(positions) < 128
                or getattr(owner,'last_source_contact_table',None) is None):
            return
        cores = available_cpus()
        try:
            for kind, built, workers in (('source','_support_evidence_built',min(16,max(1,cores-5))),
                                         ('object','_object_support_built',min(4,cores))):
                if getattr(owner,built,False):
                    continue
                self.jobs[kind] = ForkJob(_evidence,(kind,owner,source,positions,scene,config,workers))
        except BaseException:
            self.close();raise

    def __enter__(self):
        if self.jobs:self.owner._preprocess_evidence_prefetch = self
        return self

    def __exit__(self,*args):
        self.close()
        if getattr(self.owner,'_preprocess_evidence_prefetch',None) is self:
            del self.owner._preprocess_evidence_prefetch

    def consume(self, kind):
        job = self.jobs.pop(kind,None)
        if job is None:
            return False
        attributes, output = job.receive()
        for name, value in attributes.items():
            setattr(self.owner,name,value)
        print(output,end='')
        return True

    def close(self):
        for job in self.jobs.values():
            job.close()
        self.jobs.clear()
