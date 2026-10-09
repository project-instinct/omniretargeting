"""Material-point side of immutable observed contact planes (positive = free)."""

import numpy as np


class ContactSides:
    def __init__(self):
        self.groups = {}

    def add(self, body, surface_id, frames, signed):
        frames, signed = np.asarray(frames, int).reshape(-1), np.asarray(signed, float).reshape(-1)
        valid = np.isfinite(signed)
        if valid.any():
            self.groups.setdefault((str(body), int(surface_id)), []).append((frames[valid], signed[valid]))

    def report(self):
        result = {}
        for (body, surface), chunks in self.groups.items():
            frames, values = (np.concatenate([c[i] for c in chunks]) for i in (0, 1))
            back = values < -.01
            worst = int(np.argmin(values))
            result[f'{body}|surface:{surface}'] = dict(body=body, surface_id=surface, count=len(values),
                active_frames=len(np.unique(frames)), median_signed_m=float(np.median(values)),
                backside_gt1cm_fraction=float(np.mean(back)), p95_abs_m=float(np.quantile(abs(values),.95)),
                min_signed_m=float(values[worst]), worst_frame=int(frames[worst]),
                backside_frames=np.unique(frames[back]).tolist())
        return dict(definition='unit source free-side n dot (robot material binding p - observed environment point e); '
                    'active source contact sample-frames, grouped by robot body and source surface id; '
                    'backside is strictly below -0.01m; no yield/reference subtraction', pairs=result)
