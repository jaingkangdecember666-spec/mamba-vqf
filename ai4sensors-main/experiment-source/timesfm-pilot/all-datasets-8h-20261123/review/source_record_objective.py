"""Compare the current source mean squared chord with recording-equal angular RMSE on the actual training draws."""
import json
from statistics import fmean

import numpy as np
import torch

from check_history_router import prediction
from check_sources import ROOT
from lock_from_frontier import digest


def main():
    with np.load(ROOT/'C024/training-inputs.npz')as f:
        a={k:f[k].reshape((-1,)+f[k].shape[2:])for k in ('center','truth','fast','slow','enabled','source_record_ordinal','domain','z')}
    with np.load(ROOT/'C025/training-context.npz')as f:
        four=f['features'].reshape(-1,28)
    base,truth,fast,slow,use=(a[k]for k in ('center','truth','fast','slow','enabled'))
    scores={}
    for batch in ('C022','C023','C024','C025'):
        if batch=='C022':
            ck=torch.load(ROOT/'C023/router.pt',map_location='cpu',weights_only=True)
            p=np.zeros(50);features=a['z']
        else:
            ck=torch.load(ROOT/batch/'router.pt',map_location='cpu',weights_only=True)
            p=np.r_[ck['state_dict']['weight'].numpy().ravel(),ck['state_dict']['bias'].numpy()]
            features=four if len(p)==58 else a['z']
        u,*_=prediction(features,ck['p0'].numpy(),p,base,fast,slow,use)
        radial=np.linalg.norm(np.cross(u,truth),axis=1)
        angles=np.degrees(np.arctan2(radial,(u*truth).sum(1)))
        scores[batch]={'squared_chord':((u-truth)**2).sum(1),'deg_squared':angles**2}
    summary={};rows=[]
    for d in sorted(set(a['domain'])):
        part=[]
        for i in sorted(set(a['source_record_ordinal'][a['domain']==d])):
            sel=a['source_record_ordinal']==i
            r={'source':d,'ordinal':int(i),'training_slots':int(sel.sum())}
            for b,metrics in scores.items():
                r[b]={'mean_squared_chord':float(metrics['squared_chord'][sel].mean()),
                      'angular_rmse_deg':float(np.sqrt(metrics['deg_squared'][sel].mean()))}
            part.append(r)
        n=len(part);worst_n=max(1,int(np.ceil(.1*n)))
        groups={}
        for b in scores:
            items=[(r[b]['mean_squared_chord'],r[b]['angular_rmse_deg'])for r in part]
            xx=np.array(items)
            ang=xx[:,1];ch=xx[:,0]
            groups[b]={'mean_record_angular_rmse_deg':float(ang.mean()),'mean_record_squared_chord':float(ch.mean()),
                'largest_tenth_record_share_of_mean_chord':float(np.sort(ch)[-worst_n:].sum()/ch.sum()),
                'largest_tenth_record_share_of_macro_rmse':float(np.sort(ang)[-worst_n:].sum()/ang.sum()),
                'records_within_source_better_than_C022':sum(r[b]['angular_rmse_deg']<r['C022']['angular_rmse_deg'] for r in part)}
        baseline=groups['C022']
        for g in groups.values():
            g['macro_angle_ratio']=g['mean_record_angular_rmse_deg']/baseline['mean_record_angular_rmse_deg']
            g['source_chord_ratio']=g['mean_record_squared_chord']/baseline['mean_record_squared_chord']
        summary[d]={'record_count':n,'largest_tenth_record_count':worst_n,'methods':groups}
        rows.extend(part)
    out={'scope':'Same 145 admitted external records and 16,640 training draws at their fixed native release, with one vote per record. Angular RMSE is first computed inside each record. Existing optimizer minimizes mean squared chord then normalizes at source level. No additional fitting, target record, or inference adaptation. The largest-tenth summaries use ceil(n/10) records.','training_sha256':digest(ROOT/'C024/training-inputs.npz'),'source_summary':summary,'records':rows}
    (ROOT/'review/source_record_objective.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
    for d,g in summary.items():
        m=g['methods']['C022']
        print(d,g['record_count'],round(m['largest_tenth_record_share_of_mean_chord'],3),round(m['largest_tenth_record_share_of_macro_rmse'],3))


if __name__=='__main__':main()
