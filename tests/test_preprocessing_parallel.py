"""Scheduling-only preprocessing must retain exact query order and ownership."""
from types import SimpleNamespace

import numpy as np
import pytest

from omniretargeting._core import preprocessing_parallel as parallel
from omniretargeting.support_parallel import SupportPool


def _identity(value):
    return value


def _fail():
    raise ValueError('preprocessing failure')


def test_fork_roundtrip_retains_readonly_arrays_and_aliases():
    a=np.array([1.,-0.]);a.setflags(write=False)
    job=parallel.ForkJob(_identity,({'a':a,'b':a,'nested':SimpleNamespace(v=a)},))
    try:
        result=job.receive()
        assert result['a'].tobytes()==a.tobytes()
        assert result['a'] is result['b'] is result['nested'].v
        assert not result['a'].flags.writeable
    finally:
        job.close()
    assert not job.process.is_alive()


def test_fork_exception_is_consumed_and_child_is_joined():
    job=parallel.ForkJob(_fail,())
    with pytest.raises(ValueError,match='preprocessing failure'):
        job.receive()
    assert job.closed and not job.process.is_alive()


def test_support_pool_worker_limit_is_explicit_and_bounded(monkeypatch):
    import omniretargeting.support_parallel as module
    monkeypatch.setattr(module,'available_cpus',lambda:12)
    assert SupportPool().workers==8
    assert SupportPool(workers=16).workers==12
    assert SupportPool(workers=1).workers==1


def test_evidence_is_installed_only_when_consumed_in_original_order(monkeypatch,capsys):
    from omniretargeting._core import surfaces
    def source(owner,*a,**k):
        owner._support_evidence_built=True
        owner._source_support_evidence=np.array([7.])
        owner._source_support_evidence.setflags(write=False)
        owner.last_source_support_diagnostics={'status':'source'}
        print('source done')
    def objects(owner,*a,**k):
        owner._object_support_built=True;owner._object_support_evidence={'object':1}
        owner.last_object_support_diagnostics={'status':'object'}
        print('object done')
    monkeypatch.setattr(surfaces,'_prepare_source_support',source)
    monkeypatch.setattr(surfaces,'_prepare_object_support',objects)
    monkeypatch.setattr(parallel,'ENABLED',True)
    monkeypatch.setattr(parallel,'available_cpus',lambda:12)
    table=object();owner=SimpleNamespace(last_source_contact_table=table)
    prep=parallel.EvidencePrefetch(owner,None,np.zeros((128,3)),None,{},True)
    try:
        assert not hasattr(owner,'_source_support_evidence')
        assert prep.consume('source')
        assert owner.last_source_contact_table is table
        assert not owner._source_support_evidence.flags.writeable
        assert prep.consume('object')
        assert not prep.consume('source')
        assert capsys.readouterr().out=='source done\nobject done\n'
    finally:prep.close()


def test_omomo_incremental_sampler_retains_bytes_and_tie_order(monkeypatch):
    from omniretargeting.data_sources.omomo import OmomoDataSource
    import trimesh
    for vertices in (np.random.default_rng(7).normal(size=(513,3)),
                     np.tile(np.array([[1.,0.,0.],[0.,1.,0.],[-1.,0.,0.]]),(60,1))):
        source=OmomoDataSource.__new__(OmomoDataSource)
        source.object_mesh=trimesh.Trimesh(vertices=vertices,process=False)
        source.n_object_samples=100
        monkeypatch.setattr(parallel,'ENABLED',False)
        expected=source._sample_object_points()
        monkeypatch.setattr(parallel,'ENABLED',True)
        monkeypatch.setattr(parallel,'available_cpus',lambda:4)
        actual=source._sample_object_points()
        assert expected.shape==actual.shape and expected.dtype==actual.dtype
        assert expected.tobytes()==actual.tobytes()


def _block_with_child(path):
    import multiprocessing
    import os
    from pathlib import Path
    ready=multiprocessing.get_context('fork').Process(target=lambda:multiprocessing.Event().wait())
    ready.start()
    Path(path).write_text(str(ready.pid))
    multiprocessing.Event().wait()


def test_cancellation_releases_owned_descendants(tmp_path):
    import os
    from pathlib import Path
    import time
    path=tmp_path/'child.pid'
    job=parallel.ForkJob(_block_with_child,(str(path),))
    try:
        deadline=time.monotonic()+3
        while not path.exists() and time.monotonic()<deadline:time.sleep(.01)
        assert path.exists()
        pid=int(path.read_text())
        assert os.getpgid(pid)==job.process.pid
        job.close()
        # A terminated orphan may briefly await init's reap; it no longer runs.
        deadline=time.monotonic()+3
        while time.monotonic()<deadline:
            status=Path(f'/proc/{pid}/stat')
            try:state=status.read_text().split()[2]
            except FileNotFoundError:break
            if state=='Z':break
            time.sleep(.01)
        else:raise AssertionError('owned child still running')
    finally:job.close()


def test_shared_array_transport_keeps_bits_aliases_and_exact_reference():
    import mujoco
    from pathlib import Path
    model=mujoco.MjModel.from_xml_string('<mujoco/>')
    a=np.arange(20000,dtype=np.float64).reshape(100,200).T
    a.flat[0]=-0.;a.setflags(write=False)
    job=parallel.ForkJob(_identity,({'model':model,'a':a,'again':a},),references=(model,))
    directory=Path(job.transport.directory.name)
    result=job.receive()
    assert result['model'] is model
    assert result['a'] is result['again'] and not result['a'].flags.writeable
    assert result['a'].shape==a.shape and result['a'].dtype==a.dtype
    assert result['a'].tobytes()==a.tobytes()
    assert not directory.exists()
