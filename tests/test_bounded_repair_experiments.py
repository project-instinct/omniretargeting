"""Slow experimental mechanisms cannot schedule unbounded or unproductive work."""
import numpy as np
from omniretargeting._fast.repair_schedule import RepairSchedule
from omniretargeting._fast.repair_witnesses import RepairWitnesses


def test_only_deep_accepted_descent_enables_wakeup():
    q = np.zeros((2, 2))
    for before, after in [(.02,.02), (.02,.0201), (.005,.004), (.02,.02-1e-8)]:
        schedule = RepairSchedule([1,2], 12)
        schedule.next()
        schedule.note_progress(1, before, after)
        schedule.defer(1,q,[0],'small_progress')
        schedule.changed(q+1)
        assert not schedule.deferred and schedule.ready == [2]
    schedule.note_progress(1,.02,.019)
    schedule.defer(1,q,[0],'small_progress')
    schedule.changed(q+1)
    assert schedule.ready == [1,2]


def test_witness_memory_is_deduplicated_bounded_and_order_independent():
    rows = [((frame,'static',3),dict(sample_id=i,clearance=-.001*(i+1)))
            for frame in range(5) for i in range(20)]
    keep = []
    for order in [rows, list(reversed(rows)), rows+rows]:
        memory = RepairWitnesses(max_samples=7, per_key=2)
        measured = {'rows': []}
        checked = {'rows': order}
        assert memory.observe(measured,checked) == 7
        assert memory.observe(measured,checked) == 0
        assert len(memory.required) == len(memory.depths) == 7
        assert all(sum(k==key for k,_ in memory.required)<=2 for key,_ in memory.required)
        assert all(sample>=18 for _,sample in memory.required)
        assert checked['rows'] is order and measured['rows'] == []
        keep.append(memory.required)
    assert keep[0] == keep[1] == keep[2]


def test_witness_retry_requires_accepted_deep_progress_and_is_marked():
    schedule = RepairSchedule([1],12)
    schedule.next()
    assert not schedule.witness_retry(1)
    schedule.note_progress(1,.02,.019)
    assert schedule.witness_retry(1)
    assert schedule.next() == (1,1) and schedule.was_wakeup
    schedule.note_progress(1,.019,.008)
    assert not schedule.witness_retry(1)
