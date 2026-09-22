"""Playback timing and portable, browser-local Viser recordings.

The cached player uses Viser's version-matched frontend and recording format.
It downloads a recording before playing; it is not a live video stream.
"""

from __future__ import annotations

import hashlib
import html
import json
import math
import shutil
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlencode, urlsplit

import numpy as np
import viser


class _RigidInstanceHandle:
    """Adapt mjviser's singleton batch to Viser's direct node-pose messages.

    Keep the same GLB, material, LOD, instance scale, and visibility. Its one
    instance stays at identity; only the outer scene node moves. Viser handles
    node poses through mutable refs before drawing, without React prop updates.
    """

    def __init__(self, handle):
        self._handle = handle
        position = np.asarray(handle.batched_positions)[0].copy()
        quaternion = np.asarray(handle.batched_wxyzs)[0].copy()
        # mjviser creates these handles at an identity outer transform.
        # Refuse a composed transform rather than silently losing it.
        if not (np.allclose(handle.position, 0) and np.allclose(handle.wxyz, [1, 0, 0, 0])):
            raise ValueError("Rigid instance adaptation requires an identity outer transform.")
        handle.batched_positions = np.zeros((1, 3), dtype=np.float32)
        handle.batched_wxyzs = np.array([[1, 0, 0, 0]], dtype=np.float32)
        handle.position = position
        handle.wxyz = quaternion

    def __getattr__(self, name):
        return getattr(self._handle, name)

    @property
    def visible(self):
        return self._handle.visible

    @visible.setter
    def visible(self, value):
        self._handle.visible = value

    @property
    def batched_positions(self):
        return np.asarray(self._handle.position)[None]

    @batched_positions.setter
    def batched_positions(self, value):
        if np.shape(value) != (1, 3):
            raise ValueError("Expected exactly one rigid instance position.")
        self._handle.position = np.asarray(value)[0]

    @property
    def batched_wxyzs(self):
        return np.asarray(self._handle.wxyz)[None]

    @batched_wxyzs.setter
    def batched_wxyzs(self, value):
        if np.shape(value) != (1, 4):
            raise ValueError("Expected exactly one rigid instance orientation.")
        self._handle.wxyz = np.asarray(value)[0]


def use_rigid_instance_updates(scene) -> int:
    """Optimize singleton groups in the project's pinned mjviser adapter.

    Actual batches keep mjviser's original path. No Viser dependency is patched.
    ``_mesh_groups`` is the sole private integration point; the project pins the
    mjviser commit and integration tests check geometry and wire-message parity.
    """
    if scene.num_envs != 1:
        return 0
    changed = 0
    with scene.server.atomic():
        for group in scene._mesh_groups:
            if len(group.body_ids) != 1 or isinstance(group.handle, _RigidInstanceHandle):
                continue
            group.handle = _RigidInstanceHandle(group.handle)
            changed += 1
    return changed


class PlaybackClock:
    """Monotonic source-time clock; controls first settle elapsed old state."""

    def __init__(self, frames, fps, *, now, frame=0, playing=True, speed=1.0, loop=True):
        if frames < 1 or not math.isfinite(fps) or fps <= 0:
            raise ValueError("Playback needs at least one frame and a positive finite fps.")
        if not math.isfinite(speed) or speed <= 0:
            raise ValueError("Playback speed must be positive and finite.")
        self.frames, self.fps = frames, fps
        self.frame = float(np.clip(frame, 0, frames - 1))
        self.playing, self.speed, self.loop = playing, speed, loop
        self._time = now

    def sample(self, now):
        dt = max(0.0, now - self._time)
        self._time = now
        if self.playing and self.frames > 1:
            self.frame += dt * self.speed * self.fps
            if self.loop:
                # Hold the last sample for its native frame period. Do not
                # interpolate the end into an unrelated first pose.
                self.frame %= self.frames
            elif self.frame >= self.frames - 1:
                self.frame = float(self.frames - 1)
                self.playing = False
        return min(self.frame, self.frames - 1)

    def rebase(self, now):
        """Freeze media time while there are no viewers, without catch-up."""
        self._time = now

    def seek(self, frame, now):
        self.frame = float(np.clip(round(frame), 0, self.frames - 1))
        self.playing = False
        self._time = now

    def toggle(self, now):
        self.sample(now)
        if not self.playing and self.frame >= self.frames - 1:
            self.frame = 0.0
        self.playing = not self.playing

    def set_speed(self, speed, now):
        if not math.isfinite(speed) or speed <= 0:
            raise ValueError("Playback speed must be positive and finite.")
        self.sample(now)
        self.speed = speed


class FramePacer:
    """Deadline scheduling that skips missed refreshes instead of queuing them."""

    def __init__(self, hz, now):
        if not math.isfinite(hz) or hz <= 0:
            raise ValueError("Refresh rate must be positive and finite.")
        self.period = 1.0 / hz
        self.deadline = now

    def delay(self, now):
        if self.deadline <= now:
            self.deadline += (math.floor((now - self.deadline) / self.period) + 1) * self.period
        return max(0.0, self.deadline - now)


def recording_times(frames: int, fps: float, refresh_hz: float) -> np.ndarray:
    """Include every exact source timestamp plus display-rate interpolants."""
    PlaybackClock(frames, fps, now=0.0)
    FramePacer(refresh_hz, now=0.0)
    duration = (frames - 1) / fps
    source = np.arange(frames, dtype=float) / fps
    display = np.arange(math.ceil(duration * refresh_hz), dtype=float) / refresh_hz
    # Round only for deduplication, retaining source times preferentially.
    times = {round(float(t), 12): float(t) for t in display if t <= duration}
    times.update({round(float(t), 12): float(t) for t in source})
    return np.array(sorted(times.values()))


def write_recording(directory: Path, payload: bytes, *, camera_position, camera_look_at,
                    metadata: dict) -> dict:
    """Write a portable bundle without touching source motion or dependencies."""
    directory.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(payload).hexdigest()
    filename = f"motion-{digest[:16]}.viser"
    target = directory / filename
    temporary = target.with_suffix(".tmp")
    temporary.write_bytes(payload)
    temporary.replace(target)
    frontend = Path(viser.__file__).parent / "client" / "build" / "index.html"
    shutil.copyfile(frontend, directory / "viewer.html")
    params = {
        "playbackPath": filename,
        "initialCameraPosition": ",".join(map(str, camera_position)),
        "initialCameraLookAt": ",".join(map(str, camera_look_at)),
        "initialCameraUp": "0,0,1",
    }
    url = "viewer.html?" + urlencode(params)
    # Relative redirect also makes the exported directory work with any
    # ordinary static HTTP server, without a Python/Viser runtime.
    (directory / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>OmniRetargeting playback</title>'
        f'<meta http-equiv="refresh" content="0;url={html.escape(url, quote=True)}">'
        f'<a href="{html.escape(url, quote=True)}">Open cached 3D playback</a>',
        encoding="utf-8",
    )
    manifest = {**metadata, "viser_version": viser.__version__, "sha256": digest,
                "bytes": len(payload), "recording": filename, "url": url}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def serve_recording(directory: Path, port: int, *, host: str = "0.0.0.0"):
    """Serve only bundle assets, with immutable recording cache validation."""
    directory = directory.resolve()
    recordings = {"/" + path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in directory.glob("motion-*.viser") if path.is_file()}
    allowed = {"/", "/index.html", "/viewer.html", "/manifest.json",
               *recordings}

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(directory), **kwargs)

        def send_head(self):
            path = urlsplit(self.path).path
            if path not in allowed:
                self.send_error(404)
                return None
            if path in recordings:
                etag = '"' + recordings[path] + '"'
                if self.headers.get("If-None-Match") == etag:
                    self.send_response(304)
                    self.end_headers()
                    return None
            return super().send_head()

        def end_headers(self):
            path = urlsplit(self.path).path
            if path in recordings:
                self.send_header("Cache-Control", "public, max-age=31536000, immutable")
                self.send_header("ETag", '"' + recordings[path] + '"')
            else:
                self.send_header("Cache-Control", "no-cache")
            super().end_headers()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer((host, port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server
