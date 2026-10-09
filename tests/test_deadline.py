import json
import sys
import time

import numpy as np
import pytest

from omniretargeting._fast.deadline import Deadline, DeadlineExceeded
from omniretargeting import main as cli
from omniretargeting.sequence_retargeting import sequence_config


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def test_deadline_tracks_elapsed_reserve_and_checks():
    clock = FakeClock()
    deadline = Deadline(10.0, started=20.0, clock=clock)

    assert deadline.elapsed() == pytest.approx(80.0)
    assert deadline.reserve_s == pytest.approx(2.0)
    assert deadline.remaining() == pytest.approx(-72.0)
    assert deadline.expired() is True
    assert deadline.fraction_used() == pytest.approx(8.0)

    with pytest.raises(DeadlineExceeded):
        deadline.check("stage")
    with pytest.raises(DeadlineExceeded):
        deadline.check("later")
    assert deadline.first_expired_label == "stage"
    assert deadline.checks == 2


def test_deadline_default_reserve_and_describe():
    clock = FakeClock()
    deadline = Deadline(100.0, started=90.0, clock=clock)
    assert deadline.reserve_s == pytest.approx(5.0)
    assert deadline.remaining() == pytest.approx(85.0)
    assert deadline.expired() is False
    deadline.check("safe")
    assert deadline.describe() == {
        "budget_s": 100.0,
        "reserve_s": 5.0,
        "elapsed_s": 10.0,
        "remaining_s": 85.0,
        "expired": False,
        "first_expired_label": None,
        "checks": 1,
    }


def test_deadline_for_clip_and_unlimited():
    deadline = Deadline.for_clip(30, seconds_per_frame=2.0, framerate=30.0)
    assert deadline.budget_s == pytest.approx(60.0)
    assert deadline.reserve_s == pytest.approx(3.0)

    unlimited = Deadline.unlimited()
    assert unlimited.budget_s == float("inf")
    assert unlimited.reserve_s == 0.0
    assert unlimited.expired() is False
    assert unlimited.fraction_used() == 0.0
    unlimited.check("never")
    assert unlimited.describe()["checks"] == 1


def test_sequence_config_has_positive_default_deadline():
    cfg = sequence_config()
    assert cfg["time_budget_seconds_per_frame"] == 1.0
    assert sequence_config({
        "time_budget_seconds_per_frame": 0.5
    })["time_budget_seconds_per_frame"] == 0.5


def test_production_start_is_attached_before_solving(tmp_path, monkeypatch):
    class StopAfterProductionStart(RuntimeError):
        pass

    class FakeOwner:
        def __init__(self, *_args, **_kwargs):
            pass

        def __setattr__(self, name, value):
            if name == "_production_started":
                assert source.loaded
                assert call_started < value <= time.monotonic()
                raise StopAfterProductionStart
            super().__setattr__(name, value)

    class FakeSource:
        def __init__(self):
            self.loaded = False
            self.positions = np.zeros((2, 3, 3))
            self.root_translations = np.zeros((2, 3))
            self.root_orientations = None
            self.target_names = ["A", "B", "C"]
            self.framerate = 30.0
            self.object_points = None
            self.object_mesh = None
            self.metadata = {"joint_mapping": {"A": "link_a"}}

        def load(self):
            return self

    source = FakeSource()
    robot = tmp_path / "robot.json"
    robot.write_text(json.dumps({
        "urdf_path": "fake.urdf",
        "joint_mapping": {"A": "link_a"},
    }))
    source_config = tmp_path / "source.json"
    source_config.write_text(json.dumps({
        "type": "fake",
        "motion": str(tmp_path / "motion.pkl"),
        "terrain": str(tmp_path / "terrain.obj"),
    }))
    output = tmp_path / "output.npz"

    def fake_create_data_source(*_args, **_kwargs):
        source.loaded = True
        return source

    monkeypatch.setattr(cli, "OmniRetargeter", FakeOwner)
    monkeypatch.setattr(cli, "create_data_source", fake_create_data_source)
    monkeypatch.setattr(sys, "argv", [
        "omniretargeting", "--source-config", str(source_config),
        "--robot-config", str(robot), "--solver", "fast",
        "--output", str(output),
    ])

    call_started = time.monotonic()
    with pytest.raises(StopAfterProductionStart):
        cli.main()
