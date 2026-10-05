"""Solve-local immutable references and RAM-backed NumPy transport."""
import io
import pickle
from pathlib import Path
import tempfile

import numpy as np


class ArrayTransport:
    def __init__(self,references):
        self.references=tuple(references)
        self.directory=tempfile.TemporaryDirectory(prefix='orx-prep-',dir='/dev/shm')

    def pack(self,value):
        arrays={};stream=io.BytesIO();directory=Path(self.directory.name)
        references={id(v):i for i,v in enumerate(self.references)}
        def persistent(item):
            if id(item) in references:return ('reference',references[id(item)])
            if not isinstance(item,np.ndarray) or item.dtype.hasobject or item.nbytes<65536:
                return None
            if id(item) not in arrays:
                index=len(arrays)
                mapped=np.memmap(directory/str(index),dtype=item.dtype,mode='w+',shape=item.shape)
                mapped[:]=item;mapped.flush();mapped._mmap.close()
                arrays[id(item)]=(index,item.shape,item.dtype,not item.flags.writeable)
            return ('array',*arrays[id(item)])
        pickler=pickle.Pickler(stream,protocol=5);pickler.persistent_id=persistent
        pickler.dump(value)
        return stream.getvalue()

    def unpack(self,payload):
        arrays={};directory=Path(self.directory.name)
        def persistent(descriptor):
            kind,*fields=descriptor
            if kind=='reference':return self.references[fields[0]]
            if kind!='array':raise pickle.UnpicklingError('unknown preprocessing descriptor')
            index,shape,dtype,readonly=fields
            if index not in arrays:
                mapped=np.memmap(directory/str(index),dtype=dtype,mode='r+',shape=shape)
                # The ndarray owns the mapping through its base. Unlinking
                # the solve-local files does not invalidate accepted arrays.
                value=np.asarray(mapped)
                value.setflags(write=not readonly);arrays[index]=value
            return arrays[index]
        unpickler=pickle.Unpickler(io.BytesIO(payload));unpickler.persistent_load=persistent
        return unpickler.load()

    def close(self):self.directory.cleanup()
