"""Audit removal of entire contacted surfaces across physical sensitivity cases."""
import argparse
import json
from pathlib import Path
from collections import Counter
import numpy as np
from omniretargeting.support_removal import removal_evidence


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenarios',nargs='+',type=Path,required=True)
    parser.add_argument('--inertia',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    demand=np.load(args.inertia,allow_pickle=True).item()['demand']
    cases=[np.load(p,allow_pickle=True).item()['evidence'] for p in args.scenarios]
    surface_ids=sorted({surface for e in cases for _,surface in e.regions.values()})
    records=[]
    for f in range(len(demand['wrench'])):
        group_results=[]
        for e in cases:
            groups={str(surface):[name for name,(_,sid) in e.regions.items() if sid==surface] for surface in surface_ids}
            group_results.append(removal_evidence(e.candidates[f],demand['wrench'][f],groups,origin=demand['origin'],uncertainty=demand['derivative_disagreement'][f]))
        common={}
        for surface in map(str,surface_ids):
            roles=[r['groups'][surface]['role'] for r in group_results]
            common[surface]=roles[0] if len(set(roles))==1 else 'scenario_disagreement'
        records.append(dict(frame=f,consensus=common,scenarios=group_results))
        if f%100==0: print(f,common,flush=True)
    result=dict(records=records,scenarios=list(map(str,args.scenarios)),source_only=True,
                summary={str(s):dict(Counter(row['consensus'][str(s)] for row in records)) for s in surface_ids},
                uncertainty='derivative-window sensitivity box; not a confidence interval',force_distribution_inferred=False)
    args.output.write_text(json.dumps(result,indent=2))
    print(result['summary'],flush=True)


if __name__=='__main__':main()
