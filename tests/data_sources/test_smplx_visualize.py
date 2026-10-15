"""Tests for the standalone SMPL-X visualization helpers."""

import subprocess
import sys

from omniretargeting.data_sources import smplx_visualize


def test_default_coordinates_are_native_smplx():
    args = smplx_visualize.build_parser().parse_args(
        ["--smplx-model-dir", "/tmp/models", "--terrain", "/tmp/terrain.obj"]
    )

    assert args.coords == "smplx"


def test_macos_motion_picker_uses_osascript(monkeypatch):
    completed = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout="/tmp/motion.npz\n",
        stderr="",
    )
    calls = []

    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(smplx_visualize.os.path, "expanduser", lambda path: "/Users/test/Datasets")
    monkeypatch.setattr(smplx_visualize.os.path, "isdir", lambda path: True)
    monkeypatch.setattr(
        smplx_visualize.shutil,
        "which",
        lambda name: "/usr/bin/osascript" if name == "osascript" else None,
    )

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return completed

    monkeypatch.setattr(smplx_visualize.subprocess, "run", run)

    assert smplx_visualize._ask_open_motion() == "/tmp/motion.npz"
    assert calls == [
        (
            [
                "/usr/bin/osascript",
                "-l",
                "JavaScript",
                "-e",
                "const app = Application.currentApplication();\n"
                "app.includeStandardAdditions = true;\n"
                "const selectedFile = app.chooseFile({\n"
                '  withPrompt: "Select SMPL-X motion file",\n'
                '  defaultLocation: Path("/Users/test/Datasets"),\n'
                "});\n"
                "selectedFile.toString();",
            ],
            {"capture_output": True, "text": True},
        )
    ]


def test_linux_motion_picker_still_uses_zenity(monkeypatch):
    completed = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout="/tmp/motion.npz\n",
        stderr="",
    )
    calls = []

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(smplx_visualize.os.path, "expanduser", lambda path: "/home/test/Datasets")
    monkeypatch.setattr(smplx_visualize.os.path, "isdir", lambda path: True)
    monkeypatch.setattr(
        smplx_visualize.shutil,
        "which",
        lambda name: "/usr/bin/zenity" if name == "zenity" else None,
    )

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return completed

    monkeypatch.setattr(smplx_visualize.subprocess, "run", run)

    assert smplx_visualize._ask_open_motion() == "/tmp/motion.npz"
    assert calls == [
        (
            [
                "zenity",
                "--file-selection",
                "--title=Select SMPL-X motion file",
                "--file-filter=SMPL-X motion (*.npz) | *.npz",
                "--filename=/home/test/Datasets",
            ],
            {"capture_output": True, "text": True},
        )
    ]
