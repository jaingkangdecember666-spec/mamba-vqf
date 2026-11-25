"""Re-score the unchanged vehicle and fixture predictions under the preregistered alternative references."""
import argparse
import csv
import json
from pathlib import Path
from statistics import fmean

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from check_sources import ROOT
from check_mount_pilots import source
from fixture_level import transport_prior
from locked_replay import SelectedPredictor,sha256
from native_catalog import Catalog
from optical_score import metric


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('lock',type=Path)
    ap.add_argument('main_csv',type=Path)
    args=ap.parse_args()
    torch.set_num_threads(2)
    catalog=Catalog(args.lock)
    model=SelectedPredictor(args.lock)
    anch=json.loads((ROOT/'review/final-anchor-spec.json').read_bytes())
    cfg=anch['supplemental_reference']
    for name,expected in cfg['code'].items():
        assert sha256(ROOT/'review'/name)==expected
    prior=np.array(cfg['prior_world_at_identity_mount'])
    with args.main_csv.open(encoding='utf-8-sig',newline='') as f:
        rows=[r for r in csv.DictReader(f) if r['dataset'] in ('RepoIMU','IMU-Array')]
    assert len(rows)==42
    output=[]
    for r in rows:
        d,n=r['dataset'],r['record']
        if not r.get('candidate_deg'):
            output.append({'dataset':d,'record':n,'status':r['status'],'error':r.get('error')})
            continue
        key=d,n
        (sig,hz,truth,mask),quality=catalog.native(*key)
        ids=np.flatnonzero(mask & np.isfinite(sig).all(1))
        assert len(ids)==int(r['scored_frames'])
        pred=model.predict(sig,hz,ids)
        primary={k:metric(v,truth[ids]) for k,v in pred.items()}
        for field in primary:
            if field in r:
                assert abs(primary[field]-float(r[field]))<2e-4,(key,field)
        changed=False
        if d=='IMU-Array':
            nominal=quality.pop('nominal_reference')
            alternative=nominal[ids]
            changed=True
            flags=quality['reference_flags']
        else:
            flags=quality['quality_flags']
            alternative=truth[ids]
            if quality['world_up_source']=='fixture_X_fallback_unverified':
                t,_,_,q=source(catalog.rows[key])
                mount=Rotation.from_quat(quality['mount_xyzw'])
                level=transport_prior(q[t<quality['fit_through_s']],mount,prior)
                assert r['group'].startswith('Pendulum/') and quality['applied_imu_minus_optical_rows']==0
                alternative=mount.inv().apply(Rotation.from_quat(q[ids]).inv().apply(level))
                changed=True
        extra={k:metric(v,alternative) for k,v in pred.items()}
        output.append({'dataset':d,'record':n,'group':r['group'],'status':'scored','frames':len(ids),
                       'scoring_indices_sha256':r['scoring_indices_sha256'],'alternative_applies':changed,
                       'primary':primary,'alternative':extra,'quality_flags':flags,'reference_checks':quality})
        print(d,n,changed,flush=True)
    summary={}
    for d in ('IMU-Array','RepoIMU'):
        own=[r for r in output if r['dataset']==d]
        scored=[r for r in own if r['status']=='scored']
        summary[d]={'requested':len(own),'scored':len(scored),'changed':sum(r['alternative_applies'] for r in scored),
                    'records_with_flags':sum(bool(r['quality_flags']) for r in scored)}
        for scope,group in [('all',scored),('alternative_changes_reference',[r for r in scored if r['alternative_applies']])]:
            if group:
                summary[d][scope]={s:{k:fmean(r[s][k] for r in group) for k in group[0][s]} for s in ('primary','alternative')}
    target=ROOT/'review/final-geometry.json'
    target.write_text(json.dumps({'lock_sha256':sha256(args.lock),'main_sha256':sha256(args.main_csv),
         'scope':'Vehicle alternative uses the nominal installation; RepoIMU alternative uses the previously specified fixture-level prior '
                 'only when the original static calibration prefix had inadequate information. Original native rows and predictions stay fixed.',
         'datasets':summary,'rows':output},indent=2)+'\n',encoding='utf8')
    print(summary)


if __name__=='__main__':
    main()
