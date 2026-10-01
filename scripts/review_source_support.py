"""Evaluate source wrench hypotheses under explicitly supplied physical assumptions."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from omniretargeting.contact_mechanics import explain_contacts
from omniretargeting.contact_support import SourceSupportEvidence
from omniretargeting.surface_contact import surface_contact_config
from omniretargeting.wrench_feasibility import DualWrenchCone


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contacts", type=Path, required=True)
    parser.add_argument("--inertia", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--friction", type=float, nargs="+", required=True)
    parser.add_argument("--gap", type=float, nargs="+", required=True)
    args = parser.parse_args()
    table = np.load(args.contacts, allow_pickle=True).item()
    inertia = np.load(args.inertia, allow_pickle=True).item()
    demand = inertia["demand"]
    cfg = surface_contact_config()
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {
        "inertia": inertia["properties"]["assumptions"],
        "source_only": True,
        "force_distribution_inferred": False,
        "physical_execution_verified": False,
        "scenarios": [],
        "uncertainty_interpretation": "window disagreement is sensitivity, not a confidence interval",
    }
    for gap in args.gap:
        for friction in args.friction:
            label = f"gap_{gap:g}_friction_{friction:g}"
            output = args.output / (label + ".npy")
            if output.exists():
                result = np.load(output, allow_pickle=True).item()
            else:
                frames, regions = table.candidates(
                    friction=friction,
                    reconstruction_bound_m=gap,
                    normal_speed_m_s=cfg["normal_speed"],
                    touch_distance_m=cfg["touch_distance"],
                    normal_opposition=cfg["normal_opposition"],
                    sticking_speed_m_s=cfg["sticking_speed"],
                )
                explanations, sensitivity = [], []
                for f, contacts in enumerate(frames):
                    explanations.append(
                        explain_contacts(
                            contacts,
                            demand["wrench"][f],
                            origin=demand["origin"],
                            method="dual",
                            max_candidates=24,
                        )
                    )
                    sensitivity.append(
                        DualWrenchCone(contacts, origin=demand["origin"]).check(
                            demand["wrench"][f], demand["derivative_disagreement"][f]
                        )
                    )
                    if f % 100 == 0:
                        print(label, f, explanations[-1]["full"]["status"], flush=True)
                evidence = SourceSupportEvidence(
                    explanations, regions, candidates=frames
                )
                result = {
                    "evidence": evidence,
                    "derivative_sensitivity": sensitivity,
                    "friction": friction,
                    "reconstruction_bound_m": gap,
                }
                np.save(output, result, allow_pickle=True)
            evidence = result["evidence"]
            nominal = Counter(x["full"]["status"] for x in evidence.explanations)
            sensitivity = Counter(x["status"] for x in result["derivative_sensitivity"])
            necessary = Counter(
                region
                for frame in evidence.explanations
                for name, role in frame["roles"].items()
                if role == "necessary_support_candidate"
                for region in [evidence.regions[name][0]]
            )
            row = {
                "friction": friction,
                "reconstruction_bound_m": gap,
                "nominal": dict(nominal),
                "derivative_sensitivity": dict(sensitivity),
                "necessary_patch_frames": dict(necessary),
                "unexplained_frames": evidence.diagnostics["unexplained_frames"],
                "output": str(output),
            }
            summary["scenarios"].append(row)
            (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
            print(label, nominal, sensitivity, flush=True)


if __name__ == "__main__":
    main()
