"""Source labels and bounded, opt-in numerical captures for local QPs."""

from dataclasses import replace
import os
from pathlib import Path
import pickle
import time

import numpy as np


def contact_batch_labels(provider):
    """Read frozen contact identities for diagnostics; never evaluate again."""
    state = getattr(provider, '_window_state', None)
    result = {}
    domain = getattr(state, 'contact_domain', None)
    blocks = getattr(domain, 'soft_blocks', None) if domain is not None else None
    for block in (getattr(state, 'blocks', ()) if blocks is None else blocks):
        channels = sorted({str(identity[-1]) for identity in block.identities})
        result[id(block.matrix)] = '+'.join(channels)
    return result


def row_sources(blocks, labels, soft, soft_labels=None):
    soft_labels = {} if soft_labels is None else soft_labels
    names, codes = [], []
    def add(name, count):
        if name not in names:
            names.append(name)
        codes.extend([names.index(name)] * count)
    def soft_label(key, kind, batch):
        if key[0] == 'contact':
            channel = soft_labels.get((kind, batch), 'unknown_channel')
            return f'contact:{key[1]}:{kind}:{channel}:batch{batch}'
        return f'{key[0]}:{kind}'
    for block in blocks:
        if not block.soc:
            add(labels[id(block)], len(block.b))
    for batch, (_, _, indices) in enumerate(soft.rows):
        for index in indices:
            add(soft_label(soft.keys[index], 'scalar', batch), 1)
    add('epigraph_upper_limits', len(getattr(soft, 'limits', {})))
    add('epigraph_nonnegativity', len(soft.keys))
    for block in blocks:
        if block.soc:
            add(labels[id(block)], len(block.b))
    for batch, (_, _, _, indices) in enumerate(soft.cones):
        for index in indices:
            add(soft_label(soft.keys[index], 'norm', batch), 4)
    return dict(names=names, codes=np.asarray(codes, np.int16))


def source_counts(model, rows=None):
    sources = model.window_row_sources
    if sources is None:
        return {}
    codes = sources['codes'] if rows is None else sources['codes'][rows]
    counts = np.bincount(codes, minlength=len(sources['names']))
    return {name: int(count) for name, count in zip(sources['names'], counts) if count}


def capture(model, P, c, mandatory):
    """The first QP of at most N distinct frame windows per run directory.

    Reservation uses exclusive file creation. No closures/providers/robot
    runtime are serialized; full numerical domains and soft terms remain.
    """
    directory = os.environ.get('OMNIRETARGETING_WINDOW_CAPTURE')
    if not directory:
        return None
    limit = int(os.environ.get('OMNIRETARGETING_WINDOW_CAPTURE_MAX', '2'))
    root = Path(directory).expanduser().absolute()
    root.mkdir(parents=True, exist_ok=True)
    frames = model.frames
    identity = 'full' if frames is None else f'{frames.start}_{frames.stop}_{frames.step}'
    # A damping retry or a later local iteration must not consume the next
    # capture slot. O_EXCL also prevents duplicate captures across children.
    try:
        descriptor = os.open(root / f'frames_{identity}.seen',
                             os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return None
    os.close(descriptor)
    for number in range(max(0, limit)):
        lock = root / f'window_{number:03d}.reserved'
        try:
            descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            continue
        os.close(descriptor)
        program = replace(model.program, hard_blocks=(), last_solve={})
        frozen = replace(model, coords=None, program=program, window_soft_sources=(),
                         window_working_set=None, priority_cache=None, solve_diagnostics=[])
        payload = dict(model=frozen, program=program, P=P, c=c, mandatory=mandatory,
                       sources=source_counts(model), frames=model.frames, captured_pid=os.getpid(),
                       timestamp=time.time(), numerical_only=True)
        target = root / f'window_{number:03d}.pkl'
        temp = root / f'window_{number:03d}.{os.getpid()}.tmp'
        with temp.open('wb') as stream:
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, target)
        return str(target)
    return None
