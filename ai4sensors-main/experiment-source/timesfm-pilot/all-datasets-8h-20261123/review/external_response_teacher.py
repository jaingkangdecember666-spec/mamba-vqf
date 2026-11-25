"""Compare source-only relative evidence for the three physical responses with the current router choices."""
import json

import numpy as np

from check_sources import ROOT
from lock_from_frontier import digest


def teacher(center,fast,slow,truth,enabled):
    # Class ordering matches the deployed softmax. All three directions include the same frozen C020 center.
    v=np.stack((center+fast,center+slow,center),axis=1)
    v/=np.linalg.norm(v,axis=2,keepdims=True)
    angle=np.degrees(np.arctan2(np.linalg.norm(np.cross(v,truth[:,None,:]),axis=2),np.einsum('nkd,nd->nk',v,truth)))
    sq=angle**2
    sorted_error=np.sort(sq,axis=1)
    best=np.argmin(sq,axis=1)
    clear=(sorted_error[:,1]-sorted_error[:,0])*enabled
    return best,clear,sq


def main():
    path=ROOT/'C026/training-inputs.npz'
    with np.load(path)as f:a={k:f[k].reshape((-1,)+f[k].shape[2:]) for k in ('center','fast','slow','truth','enabled','domain','condition','source_record_ordinal')}
    win,gap,loss=teacher(*[a[k] for k in ('center','fast','slow','truth','enabled')])
    with np.load(ROOT/'C026/training-io.npz')as f:prob=f['C026_prob'].reshape(-1,3)
    names=('fast','slow','original')
    rows={};bank=[]
    groups=np.where(a['domain']=='EuRoC',a['condition'],a['domain'])
    for group in sorted(set(groups)):
        ids=groups==group
        records=np.unique(a['source_record_ordinal'][ids])
        weighted=np.zeros(len(win))
        counts=[]
        for i in records:
            idx=ids&(a['source_record_ordinal']==i)
            total=gap[idx].sum()
            if total>1e-12:
                weighted[idx]=gap[idx]/total
                counts.append(i)
            bank.append({'group':group,'ordinal':int(i),'slots':int(idx.sum()),'positive_gap_slots':int((gap[idx]>0.).sum()),'sum_angular_squared_gap':float(total)})
        weighted/=max(1,len(counts))
        separate={}
        for i,k in enumerate(names):
            matched=ids&(win==i)
            m=float(weighted[matched].sum())
            separate[k]={'record_balanced_confidence_mass':m,'active_slots_won':int((matched&(gap>0.)).sum()),
                         'mean_C026_probability_on_winning_slots':float((weighted[matched]@prob[matched,i])/m) if m>0 else None}
        rows[group]={'record_conditions':len(records),'records_with_confident_slots':len(counts),'raw_slots':int(ids.sum()),
                     'record_balanced_argmax_accuracy':float(weighted@(np.argmax(prob,axis=1)==win)),
                     'mean_C026_probability_for_externally_best_choice':float(weighted@prob[np.arange(len(win)),win]),
                     'classes':separate}
    output={'scope':'Training-only analysis of the three deployable, capped responses at each admitted native release. The external reference chooses one discrete response after normalizing each candidate direction. Confidence is the squared-angular error gap to second-best, set to zero outside the existing intrinsic/nonrest gate. Each original recording has equal total weight inside its source/condition. No fit or target scores. Ground-truth-dependent class names may supervise source training but are never inference variables.','inputs_sha256':digest(path),'router_sha256':digest(ROOT/'C026/router.pt'),'groups':rows,'per_original_record_condition':bank}
    (ROOT/'review/external_response_teacher.json').write_text(json.dumps(output,indent=2)+'\n',encoding='utf8')
    for d,r in rows.items():print(d,[round(r['classes'][k]['record_balanced_confidence_mass'],2)for k in names],round(r['record_balanced_argmax_accuracy'],3))


if __name__=='__main__':main()
