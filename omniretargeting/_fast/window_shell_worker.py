"""Service executable: CUDA geometry only, with the reference CPU reconstructions."""

from multiprocessing.connection import Connection, Listener, wait
import sys
import traceback
from types import SimpleNamespace


def serve(control, address, token):
    connections = []
    listener = None
    try:
        from .gpu_objective import GPUFrameEvaluator
        objective, fields = control.recv()
        evaluator = GPUFrameEvaluator(objective)
        grids = [evaluator.grid(field) for field in fields]
        listener = Listener(address, family='AF_UNIX', authkey=token, backlog=32)
        control.send(('ready', None))
        while True:
            ready = wait([control, listener._listener._socket, *connections])
            if control in ready:
                if control.recv() is None:
                    return
            if listener._listener._socket in ready:
                connections.append(listener.accept())
            for connection in ready:
                if connection not in connections:
                    continue
                try:
                    field, position, rotation, field_r, field_t, scales = connection.recv()
                except EOFError:
                    connection.close()
                    connections.remove(connection)
                    continue
                try:
                    objective.frames = len(position)
                    objective.kin = SimpleNamespace(geom_xpos=position, geom_xmat=rotation)
                    evaluator.shell.prepare(objective.kin)
                    result = evaluator.shell.deepest(grids[field], field_r, field_t, scales)
                    connection.send(('ok', result))
                except BaseException:
                    connection.send(('error', traceback.format_exc()))
    except BaseException:
        control.send(('error', traceback.format_exc()))
    finally:
        for connection in connections:
            connection.close()
        if listener is not None:
            listener.close()
        control.close()


if __name__ == '__main__':
    serve(Connection(int(sys.argv[1])), sys.argv[2], bytes.fromhex(sys.argv[3]))
