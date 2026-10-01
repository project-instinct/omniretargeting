"""A forked pure proposal with only consumed guard counters copied back."""
import multiprocessing
import os
import time


def can_fork():
    return hasattr(os, 'fork') and not multiprocessing.current_process().daemon


def _serve(function, args, guards, connection):
    before = [getattr(g, 'rejections', 0) for g in guards]
    try:
        result, status = function(*args), 'ok'
    except BaseException as error:
        result, status = error, 'error'
    deltas = [getattr(g, 'rejections', 0) - n for g, n in zip(guards, before)]
    try:
        connection.send((status, result, deltas))
    finally:
        connection.close()


class GuardedCall:
    def __init__(self, function, args, guards):
        context = multiprocessing.get_context('fork')
        parent, child = context.Pipe(duplex=False)
        self.guards, self.connection = guards, parent
        self.process = context.Process(target=_serve, daemon=True,
                                       args=(function, args, guards, child))
        self.closed, self.wait_seconds = False, 0.
        try:
            self.process.start()
        except BaseException:
            parent.close()
            child.close()
            raise
        child.close()

    def receive(self):
        started = time.monotonic()
        try:
            status, result, deltas = self.connection.recv()
        except BaseException:
            self.close()
            raise
        self.connection.close()
        self.process.join()
        self.closed = True
        self.wait_seconds = time.monotonic() - started
        for guard, delta in zip(self.guards, deltas):
            if delta:
                guard.rejections += delta
        if status == 'error':
            raise result
        return result

    def close(self):
        if self.closed:
            return
        if self.process.is_alive():
            self.process.terminate()
        self.process.join()
        self.connection.close()
        self.closed = True
