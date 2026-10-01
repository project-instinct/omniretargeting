import numpy as np
from omniretargeting._fast.repair_schedule import RepairSchedule


def test_deferred_deep_group_wakes_only_after_related_change_and_keeps_total_cap():
    q = np.zeros((3, 4))
    schedule = RepairSchedule([2, 3], 2)
    assert schedule.next() == (2, 0)
    schedule.note_progress(2, .02, .019)
    schedule.defer(2, q, [0, 1], 'small_progress')
    assert schedule.next() == (3, 0)
    q[:, 3] += .01
    schedule.changed(q)
    assert schedule.next() is None
    q[:, 1] += .002
    schedule.changed(q)
    assert schedule.next() == (2, 1)
    schedule.note_progress(2, .02, .019)
    schedule.defer(2, q, [0, 1], 'small_progress')
    q[:, 1] += .002
    schedule.changed(q)
    schedule.retry(2)
    assert schedule.next() is None


def test_ordinary_continuation_does_not_need_a_pose_wakeup():
    schedule = RepairSchedule([2], 3)
    assert schedule.next() == (2, 0)
    schedule.defer(2, np.zeros((2, 1)), [0], 'geometry')
    schedule.retry(2)
    assert schedule.next() == (2, 1)
    assert not schedule.deferred
