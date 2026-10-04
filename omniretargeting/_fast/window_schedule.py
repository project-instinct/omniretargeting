"""Overlapping SQP windows colored by complete, conservative read footprints."""

from dataclasses import dataclass


WINDOW_SIZE = 32
WINDOW_STRIDE = 16
WINDOW_SWEEPS = 2
MAX_LOCAL_ITERATIONS = 8


@dataclass(frozen=True)
class Window:
    frames: range
    footprint: range


def frame_range(frames, count):
    if (not isinstance(frames, range) or frames.step != 1 or not len(frames)
            or frames.start < 0 or frames.stop > count):
        raise ValueError('Window frames must be a nonempty, in-range, step-one range')
    return frames


def sweep_windows(count, sweep, *, size=WINDOW_SIZE, stride=WINDOW_STRIDE):
    """Shift the second sweep by half a stride; truncate only at true endpoints."""
    if count <= 0 or size <= 0 or not 0 < stride <= size:
        raise ValueError('Invalid trajectory/window dimensions')
    shift = (sweep % 2) * (stride // 2)
    return [range(max(0, start), min(count, start + size))
            for start in range(-shift, count, stride) if start + size > 0]


def dependency_window(frames, count, components):
    """Unknown/global dependency serializes instead of guessing independence."""
    frame_range(frames, count)
    first, stop = frames.start, frames.stop
    for component in components:
        function = getattr(component, 'window_dependencies', None)
        footprint = None if function is None else function(frames)
        if footprint is None:
            return Window(frames, range(count))
        frame_range(footprint, count)
        first, stop = min(first, footprint.start), max(stop, footprint.stop)
    return Window(frames, range(first, stop))


def overlaps(left, right):
    return left.start < right.stop and right.start < left.stop


def independent_batches(windows):
    """Stable greedy coloring: half-overlap index parity is not sufficient."""
    batches = []
    for window in windows:
        for batch in batches:
            if all(not overlaps(window.footprint, other.footprint) for other in batch):
                batch.append(window)
                break
        else:
            batches.append([window])
    return batches
