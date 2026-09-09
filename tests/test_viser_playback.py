"""Playback regressions: source time, overload, cache fidelity, and HTTP isolation."""

import json
from pathlib import Path
import subprocess
import sys
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest
import msgspec
import zstandard

from omniretargeting.viser_playback import (
    FramePacer, PlaybackClock, recording_times, serve_recording, write_recording,
    use_rigid_instance_updates,
)


def test_speed_change_settles_old_speed_and_pause_does_not_jump():
    clock = PlaybackClock(100, 25, now=0)
    clock.set_speed(2, now=1)
    assert clock.frame == 25
    assert clock.sample(1.5) == 50
    clock.toggle(1.6)
    assert clock.sample(20) == pytest.approx(55)
    clock.toggle(20)
    assert clock.sample(20.1) == pytest.approx(60)


def test_seek_is_exact_and_no_loop_stops_at_last_sample():
    clock = PlaybackClock(99, 25, now=0, loop=False)
    assert clock.sample(20) == 98
    assert not clock.playing
    clock.toggle(20)
    assert clock.sample(20.04) == pytest.approx(1)
    clock.seek(45, 21)
    assert clock.sample(30) == 45
    assert not clock.playing


def test_native_last_frame_hold_and_idle_reconnect():
    clock = PlaybackClock(99, 25, now=0)
    assert clock.sample(3.94) == 98
    assert clock.sample(3.97) == pytest.approx(0.25)
    clock.rebase(100)
    assert clock.sample(100.04) == pytest.approx(1.25)


def test_pacer_subtracts_work_and_skips_overdue_frames():
    pacer = FramePacer(60, now=0)
    assert pacer.delay(0.006) == pytest.approx(1 / 60 - 0.006)
    assert pacer.delay(1 / 60 + 0.004) == pytest.approx(1 / 60 - 0.004)
    assert 0 <= pacer.delay(0.2) <= 1 / 60
    assert 0 <= pacer.delay(0.205) <= 1 / 60


@pytest.mark.parametrize("frames,fps", [(99, 25), (143, 29.97002983), (1, 30)])
def test_recording_preserves_all_source_timestamps(frames, fps):
    times = recording_times(frames, fps, 60)
    assert times[0] == 0
    assert times[-1] == (frames - 1) / fps
    assert set(np.arange(frames) / fps).issubset(set(times))
    assert np.all(np.diff(times) > 0)
    assert np.all(np.diff(times) <= 1 / 60 + 1e-12)


def test_single_frame_is_static():
    clock = PlaybackClock(1, 25, now=0)
    assert clock.sample(100) == 0
    np.testing.assert_array_equal(recording_times(1, 25, 60), [0])


@pytest.mark.parametrize("fps", [0, -1, float("nan"), float("inf")])
def test_reject_invalid_rate(fps):
    with pytest.raises(ValueError):
        PlaybackClock(99, fps, now=0)
    with pytest.raises(ValueError):
        FramePacer(fps, now=0)


def test_bundle_serves_exact_bytes_cache_etag_and_only_assets(tmp_path):
    payload = b"recording fixture, not source motion"
    manifest = write_recording(tmp_path, payload, camera_position=[1, 2, 3],
                               camera_look_at=[0, 0, 0], metadata={"source_frames": 99})
    # Existing browser tabs keep their immutable URL across a new export.
    latest = write_recording(tmp_path, b"new recording", camera_position=[1, 2, 3],
                             camera_look_at=[0, 0, 0], metadata={"source_frames": 99})
    (tmp_path / "private.txt").write_text("not a playback asset")
    server = serve_recording(tmp_path, 0)
    root = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(root + "/" + manifest["recording"]) as response:
            assert response.read() == payload
            assert "immutable" in response.headers["Cache-Control"]
            etag = response.headers["ETag"]
        with pytest.raises(HTTPError) as error:
            urlopen(Request(root + "/" + manifest["recording"], headers={"If-None-Match": etag}))
        assert error.value.code == 304
        for path in ["/private.txt", "/../private.txt", "/%2e%2e/private.txt", "/subdir/"]:
            with pytest.raises(HTTPError) as error:
                urlopen(root + path)
            assert error.value.code == 404
        with urlopen(root + "/manifest.json") as response:
            assert json.load(response)["source_frames"] == 99
        with urlopen(root + "/" + latest["recording"]) as response:
            assert response.read() == b"new recording"
            assert response.headers["ETag"] != etag
        with urlopen(root + "/viewer.html") as response:
            assert b"<script" in response.read()
    finally:
        server.shutdown()
        server.server_close()


def test_real_viser_export_has_moving_poses_and_exact_source_samples(tmp_path):
    model = tmp_path / "robot.xml"
    model.write_text('<mujoco><worldbody><body name="root"><freejoint/>'
                     '<geom type="box" size=".1 .1 .1"/>'
                     '</body></worldbody></mujoco>')
    qpos = np.zeros((4, 7))
    qpos[:, 0] = [0, 0.1, 0.25, 0.2]
    qpos[:, 2] = 1
    qpos[:, 3] = 1
    motion = tmp_path / "motion.npy"
    np.save(motion, qpos)
    original = motion.read_bytes()
    bundle = tmp_path / "playback"
    result = subprocess.run(
        [sys.executable, "-m", "omniretargeting.mjviser_viewer", str(motion),
         "--model", str(model), "--port", "0", "--playback-dir", str(bundle),
         "--export-only", "--no-object"],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert motion.read_bytes() == original
    manifest = json.loads((bundle / "manifest.json").read_text())
    raw = (bundle / manifest["recording"]).read_bytes()
    inner = zstandard.ZstdDecompressor().decompress(raw[8:])
    n = int.from_bytes(inner[:8], "little")
    recording = msgspec.msgpack.decode(inner[8:8 + n])
    offsets = []
    offset = 8 + n
    for length in recording["binaryBufferLengths"]:
        offset = (offset + 7) // 8 * 8
        offsets.append(offset)
        offset += length

    def array(placeholder):
        index = placeholder["__binary_index"]
        dtype = np.dtype(placeholder["dtype"])
        return np.frombuffer(inner, dtype, recording["binaryBufferLengths"][index] // dtype.itemsize,
                             offset=offsets[index]).reshape(-1, 3)

    positions = {}
    for timestamp, message in recording["messages"]:
        update = message.get("updates", {})
        if "batched_positions" in update:
            values = array(update["batched_positions"])
            if values.shape == (1, 3):
                positions[timestamp] = values[0]
        if message["type"] == "SetPositionMessage" and message["name"].startswith("/bodies/"):
            positions[timestamp] = np.array(message["position"])
        assert not (timestamp > 0 and message["type"].endswith("MeshMessage"))
    # Compare actual serialized robot transforms, not just the timestamp list.
    for frame in range(1, 4):
        timestamp = min(positions, key=lambda t: abs(t - frame / 30))
        assert timestamp == pytest.approx(frame / 30)
        np.testing.assert_allclose(positions[timestamp], qpos[frame, :3], atol=1e-7)
    assert recording["durationSeconds"] == pytest.approx(4 / 30)


def test_rigid_instance_browser_transform_matches_fk_and_keeps_real_batches():
    import mujoco
    import viser
    from mjviser import ViserMujocoScene

    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="root"><freejoint/><geom type="box" size=".1 .2 .3"/>
        <body name="a" pos=".2 0 0"><joint axis="0 1 0"/>
          <geom type="sphere" size=".1"/></body>
        <body name="b" pos="-.2 0 0"><joint axis="0 1 0"/>
          <geom type="sphere" size=".1"/></body>
      </body></worldbody></mujoco>''')
    server = viser.ViserServer(port=0)
    try:
        data = mujoco.MjData(model)
        scene = ViserMujocoScene(server, model, num_envs=1)
        scene.camera_tracking_enabled = False
        mujoco.mj_forward(model, data)
        scene.update_from_mjdata(data)
        assert use_rigid_instance_updates(scene) == 1
        assert use_rigid_instance_updates(scene) == 0
        original_batch = next(g.handle for g in scene._mesh_groups if len(g.body_ids) > 1)
        recording = server.get_scene_serializer()
        recording.insert_sleep(1)
        data.qpos[:3] = [1, 2, 3]
        data.qpos[3:7] = [np.cos(.3), 0, 0, np.sin(.3)]
        data.qpos[7:] = [.2, -.4]
        mujoco.mj_forward(model, data)
        scene.update_from_mjdata(data)
        for group in scene._mesh_groups:
            # Reconstruct the actual frontend transform: outer node pose times
            # local instance pose. Compare with MuJoCo, including rotations.
            outer = getattr(group.handle, "_handle", group.handle)
            matrix = np.zeros(9)
            mujoco.mju_quat2Mat(matrix, np.asarray(outer.wxyz))
            positions = np.asarray(outer.position) + np.asarray(outer.batched_positions) @ matrix.reshape(3, 3).T
            np.testing.assert_allclose(positions, data.xpos[group.body_ids], atol=1e-6)
            for local_q, body in zip(outer.batched_wxyzs, group.body_ids):
                combined = np.zeros(4)
                mujoco.mju_mulQuat(combined, np.asarray(outer.wxyz), np.asarray(local_q, dtype=float))
                assert abs(np.dot(combined, data.xquat[body])) == pytest.approx(1, abs=1e-6)
        assert next(g.handle for g in scene._mesh_groups if len(g.body_ids) > 1) is original_batch
        raw = zstandard.ZstdDecompressor().decompress(recording.serialize()[8:])
        n = int.from_bytes(raw[:8], "little")
        messages = [m for t, m in msgspec.msgpack.decode(raw[8:8 + n])["messages"] if t > 0]
        singleton_name = next(g.handle.name for g in scene._mesh_groups if len(g.body_ids) == 1)
        types = {m["type"] for m in messages if m.get("name") == singleton_name}
        assert types == {"SetPositionMessage", "SetOrientationMessage"}
    finally:
        server.stop()
