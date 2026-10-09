"""In-memory Viser scene owner for record-only exports, with no listener.

The pinned Viser SceneApi/GuiApi and serializer are reused. Its transport is
constructed but never started; a local event loop only drains buffer signals.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading

import viser
from viser import infra
from viser import _messages
from viser.infra._async_message_buffer import AsyncMessageBuffer


class RecordingScene(viser.ViserServer):
    def __init__(self, label=None):
        self._event_loop = asyncio.new_event_loop()
        self._websock_server = infra.WebsockServer(host='127.0.0.1', port=8183, message_class=_messages.Message,
                                                   http_server_root=None, verbose=False)
        self._websock_server._broadcast_buffer = AsyncMessageBuffer(self._event_loop, persistent_messages=True)
        self._connection = self._websock_server
        self._initial_camera = viser.InitialCameraConfig(broadcast=self._connection.queue_message)
        self._connected_clients = {}
        self._client_lock = threading.Lock()
        self._client_connect_cb, self._client_disconnect_cb = [], []
        self._thread_executor = ThreadPoolExecutor(max_workers=1)
        self._share_tunnel = None
        self.scene = viser.SceneApi(self, thread_executor=self._thread_executor, event_loop=self._event_loop)
        self.gui = viser.GuiApi(self, thread_executor=self._thread_executor, event_loop=self._event_loop)
        self.scene.reset()
        self.scene.set_up_direction('+z')
        self.gui.reset()
        self.gui.set_panel_label(label)

    def flush(self):
        # Drain scheduled buffer signals without starting HTTP/WebSocket tasks.
        self._event_loop.run_until_complete(asyncio.sleep(0))

    def stop(self):
        self.flush()
        self._thread_executor.shutdown(wait=True)
        self._event_loop.close()
