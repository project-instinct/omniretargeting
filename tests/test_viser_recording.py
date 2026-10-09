"""Record-only construction must never start a network listener."""

import zstandard
import msgspec
import numpy as np


def test_offline_scene_serializes_animation_without_transport_start(monkeypatch):
    from viser import infra
    from omniretargeting.viser_recording import RecordingScene
    def forbidden(*args, **kwargs):
        raise AssertionError('record-only must not start a WebSocket/HTTP server')
    monkeypatch.setattr(infra.WebsockServer, 'start', forbidden)
    scene = RecordingScene(label='test')
    try:
        assert scene._websock_server._server_thread is None
        scene.initial_camera.position=(2.,-2.,1.)
        frame=scene.scene.add_frame('/test',show_axes=True)
        scene.gui.add_checkbox('test',initial_value=True)
        serializer=scene.get_scene_serializer()
        serializer.insert_sleep(1/30)
        frame.position=np.array([1.,0.,0.])
        scene.flush()
        blob=serializer.serialize()
        assert len(blob)>100
        raw=zstandard.ZstdDecompressor().decompress(blob[8:], max_output_size=int.from_bytes(blob[:8],"little"))
        size=int.from_bytes(raw[:8],"little")
        payload=msgspec.msgpack.decode(raw[8:8+size])
        assert payload['durationSeconds'] >= 1/30
        assert any('/test' in str(message) for message in payload['messages'])
    finally:scene.stop()
