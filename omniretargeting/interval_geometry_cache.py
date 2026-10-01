"""Replay unchanged interframe collision and contact scans during local repair."""

from dataclasses import dataclass

import numpy as np


def interval_key(frame, left, right):
    return (
        frame,
        np.ascontiguousarray(left).tobytes(),
        np.ascontiguousarray(right).tobytes(),
    )


@dataclass
class CachedInterval:
    scene_depth: np.ndarray
    self_depth: np.ndarray
    reliable_depth: np.ndarray
    unknown_geoms: np.ndarray
    unknown: bool
    unresolved: bool
    failures: list[dict]
    contact_change: float
    samples: int
    records: list[dict]

    @classmethod
    def capture(
        cls, scene_depth, self_depth, reliable_depth, unknown_geoms,
        unknown, unresolved, failures, contact_change, samples, records,
    ):
        return cls(
            scene_depth.copy(), self_depth.copy(), reliable_depth.copy(),
            unknown_geoms.copy(), bool(unknown), bool(unresolved),
            [dict(failure) for failure in failures], contact_change, samples,
            [dict(record) for record in records],
        )

    def replay(
        self, frame, scene_depth, self_depth, reliable_depth,
        unknown_geoms, unknown, failures, records,
    ):
        scene_depth[frame] = self.scene_depth
        self_depth[frame] = self.self_depth
        reliable_depth[frame] = self.reliable_depth
        unknown_geoms[frame] = self.unknown_geoms
        unknown[frame] = self.unknown
        failures.extend(dict(failure, frame=frame) for failure in self.failures)
        records.extend(dict(record, frame=frame) for record in self.records)
        return self.unresolved, self.contact_change, self.samples
