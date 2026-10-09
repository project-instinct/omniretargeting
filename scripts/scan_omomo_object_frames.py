"""Scan official train/test OMOMO object gaps without loading body models."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from omniretargeting.data_sources.omomo_validity import object_gap_plan


def scan(data_root: Path, framerate: float = 30.0) -> dict:
    affected, datasets = [], {}
    for split in ("train", "test"):
        path = data_root / "data" / f"{split}_diffusion_manip_seq_joints24.p"
        data = joblib.load(path)
        count = 0
        for index, sequence in (data.items() if isinstance(data, dict) else enumerate(data)):
            try:
                valid, report = object_gap_plan(sequence, framerate)
            except ValueError as error:
                raise ValueError(f"{split}[{index}]: {error}") from error
            if valid.all():
                continue
            report.update(split=split, sequence_index=int(index))
            trans = np.asarray(sequence["obj_trans"]).reshape(len(valid), 3)
            report["zero_scale_and_translation_frames"] = np.flatnonzero(
                (np.asarray(sequence["obj_scale"]) == 0) & np.all(trans == 0, axis=1)
            ).tolist()
            affected.append(report)
            count += report["invalid_frame_count"]
            print(json.dumps(report, ensure_ascii=False), flush=True)
        with path.open("rb") as handle:
            sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
        datasets[split] = dict(path=str(path), sha256=sha256,
                               sequences=len(data), invalid_frame_count=count)
    return dict(datasets=datasets, framerate=framerate, affected_sequences=len(affected),
                invalid_frame_count=sum(r["invalid_frame_count"] for r in affected),
                interpolated_frame_count=sum(r["interpolated_frame_count"] for r in affected),
                truncated_frame_count=sum(r["truncated_frames"] for r in affected), sequences=affected)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=Path.home() / "Datasets/OMOMO")
    p.add_argument("--framerate", type=float, default=30.0)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = scan(args.data_root, args.framerate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"{result['affected_sequences']} sequences, {result['invalid_frame_count']} invalid frames -> {args.output}")


if __name__ == "__main__":
    main()
