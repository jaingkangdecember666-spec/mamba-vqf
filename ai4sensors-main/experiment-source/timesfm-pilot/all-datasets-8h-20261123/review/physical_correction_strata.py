"""Compare the neural and physical directions at pre-existing external training instants, without scoring holdouts."""
import argparse
import json
from pathlib import Path

import numpy as np

from anchor_blend_features import included_degrees
from check_history_router import prediction
from lock_from_frontier import ROOT,digest
from source_router_trust import TrustRisk,weights


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('batch',choices=['C023','C035'])
    args=ap.parse_args()
    src=ROOT/'review/raw-anchor-training-141.npz'
    ablation=ROOT/'review/external-physical-anchor-141.npz'
    with np.load(src)as a,np.load(ablation)as p:
        for k in ('domain','record','native_release'):np.testing.assert_array_equal(a[k],p[k])
        dom,n=a['domain'],a['record']
        model,phys,ref=a['C023'],p['physical'],a['truth']
    proof={'raw_bank':digest(src),'physical_bank':digest(ablation)}
    if args.batch!='C023':
        risk=TrustRisk()
        source,prior=weights(args.batch)
        model,*_=prediction(risk.z,prior,source,risk.center,risk.f,risk.s,risk.enabled)
        np.testing.assert_allclose(ref,risk.truth,rtol=0,atol=1.e-10)
        proof['weights']=digest(ROOT/args.batch/'router.pt')
    delta=included_degrees(model,phys)
    angles=np.percentile(delta,[25,50,75,90])
    delta_sq=included_degrees(model,ref)**2-included_degrees(phys,ref)**2
    out={}
    for d in sorted(set(dom)):
        m=dom==d
        def summarize(lo,hi):
            ix=m&(delta>=lo)&(delta<hi); ids=np.unique(n[ix])
            value=float(np.mean([delta_sq[ix&(n==s)].mean()for s in ids])) if len(ids) else None
            return {'slots':int(ix.sum()),'records':len(ids),'mean_sq_error_model_minus_physical_deg2_across_present_records':value}
        out[d]={'median_model_to_physical_deg':float(np.median(delta[m])),'p90_deg':float(np.percentile(delta[m],90)),
                'by_joint_quartile':[summarize(lo,hi)for lo,hi in zip([0.,*angles[:3]],[*angles[:3],np.inf])]}
    counts={'base':args.batch,'external_slot_count':len(n),'joint_quartiles_and_p90_deg':angles.tolist(),'groups':out,'source_hashes':proof,
            'scope':'External source-only comparison at unchanged training slots. Each quartile averages recording means among recordings present in that bin. No coefficients or validation data are used.'}
    destination=ROOT/'review'/f'{args.batch}_physical_correction_strata.json'
    destination.write_text(json.dumps(counts,indent=2)+'\n',encoding='utf8')
    print(args.batch,angles,{d:round(x['by_joint_quartile'][-1]['mean_sq_error_model_minus_physical_deg2_across_present_records'],4)for d,x in out.items()})


if __name__=='__main__':main()
