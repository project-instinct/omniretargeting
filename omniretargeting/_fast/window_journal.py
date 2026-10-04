"""Parent-only durable window history and atomic committed-pose checkpoints."""

from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np

from .priority_merit import report_value


def ledger_snapshot(coordinator):
    """Pure data read: no coordinator evaluation or geometry query."""
    pairs = lambda values: [dict(key=report_value(k), value=report_value(v)) for k, v in values.items()]
    return dict(epoch=int(coordinator.epoch), spent=pairs(coordinator.spent),
                yields=report_value(coordinator.yields), best_keys=pairs(coordinator.best_keys),
                reference_sha256=sha256(np.asarray(coordinator.reference).tobytes()).hexdigest())


def _json_safe(value):
    value = report_value(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


class WindowJournal:
    """Enabled by OMNIRETARGETING_WINDOW_TRACE; never writes from a child."""
    def __init__(self, path=None):
        path = os.environ.get('OMNIRETARGETING_WINDOW_TRACE') if path is None else path
        self.path = Path(path).expanduser().absolute() if path else None
        self.checkpoint_path = None if self.path is None else self.path.with_suffix('.committed.npz')
        self.owner = os.getpid()
        self.run_id = f'{self.owner}-{time.time_ns()}'
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event, **data):
        if self.path is None:
            return
        if os.getpid() != self.owner:
            raise RuntimeError('Window trace writes belong to the parent')
        value = dict(event=event, run_id=self.run_id, pid=self.owner, **data)
        with self.path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(_json_safe(value), allow_nan=False, separators=(',', ':')) + '\n')
            stream.flush()
            os.fsync(stream.fileno())

    def checkpoint(self, poses, coordinator, **context):
        if self.path is None:
            return
        if os.getpid() != self.owner:
            raise RuntimeError('Window checkpoints belong to the parent')
        metadata = dict(run_id=self.run_id, **context, ledger=ledger_snapshot(coordinator))
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.path.parent, prefix=self.path.name + '.',
                                             suffix='.tmp', delete=False) as stream:
                temporary = stream.name
                np.savez(stream, poses=poses, current_depth=coordinator.current_depth,
                         best_depth=coordinator.best_depth,
                         metadata=np.array(json.dumps(_json_safe(metadata), allow_nan=False)))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.checkpoint_path)
            temporary = None
            directory = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if temporary is not None:
                os.unlink(temporary)

    def window(self, record, poses, coordinator, epoch_before):
        if self.path is None:
            return
        self.checkpoint(poses, coordinator, phase=record['phase'], iteration=record['iteration'])
        self.emit('window', record=record, epoch_before=epoch_before,
                  committed=coordinator.epoch != epoch_before, ledger=ledger_snapshot(coordinator),
                  checkpoint=str(self.checkpoint_path))
