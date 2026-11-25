"""After sealing, perturb or erase the future of native IMU streams and compare all earlier tilt outputs."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from anchor_blend_features import included_degrees
from davis_handheld import Davis
from lock_from_frontier import ROOT,read,digest
from locked_replay import SelectedPredictor
from native_catalog import Catalog
from penn_handheld import Penn
from supplemental_data import Supplemental


def compare(model,signal,hz):
    n=min(len(signal),round(16*hz)); signal=np.array(signal[:n],copy=True)
    cut=min(n-1,max(round(3.1*hz),int(.73*n)))
    if not 0<cut<n:raise ValueError('A stream is too short to test a nonempty future')
    valid=np.isfinite(signal[:cut]).all(1)
    samples=np.where(valid)[0]
    assert len(samples)>5
    # Erase two past packets in the second trial while retaining all the same scored rows.
    holes=samples[[len(samples)//3,len(samples)//2]]
    checks=[]
    for interrupted in (False,True):
        x=signal.copy();ix=samples
        if interrupted:
            x[holes]=np.nan;ix=np.setdiff1d(samples,holes)
        base=model.predict(x,hz,ix)
        erased=model.predict(x[:cut],hz,ix)
        x[cut:,3:]*=-1.;x[cut:,0:2]*=-1.;x[cut::7]=np.nan
        altered=model.predict(x,hz,ix)
        gaps={key:{tag:float(included_degrees(base[key],other[key]).max())for tag,other in [('truncate',erased),('alter_future',altered)]}
              for key in ('candidate_deg','full_vqf_deg','physical_deg')}
        assert max(v for g in gaps.values()for v in g.values())<1.e-4,gaps
        checks.append({'past_two_packet_gaps_injected':interrupted,'paired_past_rows':len(ix),'max_direction_change_deg':gaps})
    return {'native_rate_hz':hz,'original_excerpt_rows':n,'cut_before_native_row':cut,
            'future_rows_changed_or_erased':n-cut,'original_nonfinite_past_rows':int((~valid).sum()),'comparisons':checks}


def main():
    p=argparse.ArgumentParser();p.add_argument('selection_lock',type=Path);args=p.parse_args()
    torch.set_num_threads(1)
    model=SelectedPredictor(args.selection_lock); core=Catalog(args.selection_lock)
    supplemental=Supplemental(args.selection_lock); davis=Davis(args.selection_lock); penn=Penn(args.selection_lock)
    anchors=read(ROOT/'review/final-anchor-spec.json')['anchors']
    cases=[(d,sorted(seq)[0],lambda d=d,n=sorted(seq)[0]:core.native(d,n)[0])for d,seq in anchors.items()]
    for d in ('RIDI','EuRoC'):
        n=sorted(name for dom,name,*_ in supplemental.cases() if dom==d and (d,name) not in supplemental.missing)[0]
        cases.append((d,n,lambda d=d,n=n:supplemental.native(d,n)[0]))
    for d,db,where in [('DAVIS',davis,'davis-handheld'),('PennCOSYVIO',penn,'penn-handheld')]:
        name=sorted(read(ROOT/'external-sources'/where/'preprocessing-lock.json')['sealed_records'])[0]
        cases.append((d,name,lambda db=db,name=name:db.native(name)[0]))
    audit=[]
    for d,name,raw in cases:
        try:
            signal,hz,_,_=raw();data=compare(model,signal,hz)
        except (OSError,ValueError,RuntimeError,AssertionError)as exc:
            audit.append({'dataset':d,'record':name,'status':'failed','detail':f'{type(exc).__name__}: {exc}'})
        else:audit.append({'dataset':d,'record':name,'status':'checked',**data})
    assert len(cases)==len({r['dataset']for r in audit})==12
    report={'selection_lock_sha256':digest(args.selection_lock),'cases':audit,'scope':'Preselected one named recording per domain. The two altered packets are artificial stress inputs, never included in a scored performance table. Only the original raw IMU enters the unmodified branch.'}
    (ROOT/'review/final-causality.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print({r['dataset']:r['status']for r in audit})


if __name__=='__main__':main()
