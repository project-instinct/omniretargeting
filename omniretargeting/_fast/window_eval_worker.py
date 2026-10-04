"""Single CUDA-owning executable serving coalesced window evaluations."""
from multiprocessing.connection import Connection, Listener, wait
import os
import pickle
import queue
import sys
import time
import traceback


def serve(control, address, token, payload):
    listener, evaluator = None, None
    connections = []
    finished = queue.SimpleQueue()
    wake_reader, wake_writer = os.pipe()

    def completed(connection, future):
        finished.put((connection, future))
        os.write(wake_writer, b'1')
    try:
        import torch
        from threadpoolctl import threadpool_limits
        from .window_eval_batch import WindowEvaluationBatch
        torch.set_num_threads(1)
        threadpool_limits(limits=1)
        with open(payload, 'rb') as handle:
            objective = pickle.load(handle)
        evaluator = WindowEvaluationBatch(objective)
        listener = Listener(address, family='AF_UNIX', authkey=token, backlog=32)
        control.send(('ready', None))
        while True:
            ready = wait([control, listener._listener._socket, wake_reader, *connections])
            if control in ready:
                if control.recv() is None:
                    control.send(dict(requests=evaluator.requests, batches=evaluator.batches,
                                      evaluation_seconds=evaluator.seconds))
                    return
            if wake_reader in ready:
                os.read(wake_reader, 4096)
                while not finished.empty():
                    connection, future = finished.get()
                    try:
                        result = future.result()
                    except BaseException:
                        connection.send(('error', traceback.format_exc()))
                    else:
                        connection.send(('ok', result))
            if listener._listener._socket in ready:
                connections.append(listener.accept())
            pending = []
            for connection in ready:
                if connection not in connections:
                    continue
                try:
                    pending.append((connection, connection.recv()))
                except EOFError:
                    connection.close()
                    connections.remove(connection)
            if not pending:
                continue
            # Timing selects batch membership only. It never selects poses,
            # row definitions, floating reduction dimensions or commit order.
            deadline = time.monotonic() + .001
            while len(pending) < 8 and time.monotonic() < deadline:
                others = [c for c in connections if all(c is not p[0] for p in pending)]
                for connection in wait(others, max(0., deadline - time.monotonic())) if others else ():
                    try:
                        pending.append((connection, connection.recv()))
                    except EOFError:
                        connection.close()
                        connections.remove(connection)
            try:
                futures = evaluator.submit([(id(c), r) for c, r in pending])
                for (connection, _), future in zip(pending, futures):
                    future.add_done_callback(lambda f, c=connection: completed(c, f))
            except BaseException:
                error = traceback.format_exc()
                for connection, _ in pending:
                    connection.send(('error', error))
    except BaseException:
        control.send(('error', traceback.format_exc()))
    finally:
        if evaluator is not None:
            evaluator.close()
        for connection in connections:
            connection.close()
        if listener is not None:
            listener.close()
        os.close(wake_reader)
        os.close(wake_writer)
        control.close()


if __name__ == '__main__':
    serve(Connection(int(sys.argv[1])), sys.argv[2], bytes.fromhex(sys.argv[3]), sys.argv[4])
