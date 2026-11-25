"""Check the four fixed controls and two deployment-only mixtures without reusing saved predictions."""
import hashlib
import json

import numpy as np
import torch

from check_sources import ROOT,TRANSFER,weights
from check_training_batch import csv_rows,read,digest
from group_uncertainty import paired_by_group
from mixture_replay import replay
from native_catalog import Catalog
from optical_score import metric


def main():
    torch.set_num_threads(2)
    own=ROOT/'C014'
    before={(x['dataset'],x['record']):x for x in csv_rows(ROOT/'C013/sequences.csv')}
    rows=csv_rows(own/'sequences.csv')
    assert set(before)=={(r['dataset'],r['record']) for r in rows} and len(rows)==213
    source={'c003_deg':(TRANSFER/'batch6/mamba-intrinsic-selected.pt','always'),'C006_deg':(ROOT/'C004/B-readout.pt','rest_off'),
            'seed_C003_deg':(TRANSFER/'batch7/mamba-intrinsic-seed314159-selected.pt','always'),
            'seed_C006_deg':(ROOT/'C013/seed-C006-readout.pt','rest_off')}
    models={k:(weights(path,'mamba_v1'),.25,gate) for k,(path,gate) in source.items()}
    mixtures={k:{k:1.} for k in models}
    mixtures.update(A_deg={'c003_deg':.5,'C006_deg':.5},B_deg={'c003_deg':.5,'seed_C006_deg':.5})
    wanted={('SHICE','R1-1'),('OxIOD','running/data1/syn/imu1.csv'),('TUM-VI','room2'),('IMU-Array','calibration_trial/ImuArray_1_1_IMU')}
    catalog=Catalog();samples=[]
    for r in rows:
        key=r['dataset'],r['record'];old=before[key]
        for f in ('scoring_indices_sha256','scored_frames','group'):
            assert r[f]==old[f]
        assert r['status']=='scored' and not r.get('error')
        for f in ['full_vqf_deg',*models]:
            assert abs(float(r[f])-float(old[f]))<1e-6
        if key in wanted:
            (raw,hz,ref,mask),_=catalog.native(*key)
            idx=np.flatnonzero(mask & np.isfinite(raw).all(1))
            assert hashlib.sha256(idx.astype('<i8').tobytes()).hexdigest()==r['scoring_indices_sha256']
            pred=replay(raw,hz,models,mixtures,idx)
            scores={c:metric(d,ref[idx]) for c,d in pred.items()}
            error=max(abs(v-float(r[c])) for c,v in scores.items())
            assert error<1e-5,(key,error)
            samples.append({'dataset':key[0],'record':key[1],'scores_deg':scores,'max_abs_error_deg':error})
    assert len(samples)==len(wanted)
    out={'common_development_records':len(rows),'component_weights_sha256':{k:digest(p) for k,(p,_) in source.items()},
         'raw_replays':samples,'results_sha256':digest(own/'results.json'),
         'domains':{k:{d:paired_by_group([r for r in rows if r['dataset']==d],'full_vqf_deg',k)
                       for d in sorted({r['dataset'] for r in rows})} for k in ('A_deg','B_deg')}}
    (ROOT/'review/C014_audit.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
    print('independent native rows',len(rows),'probes',len(samples),'worst delta deg',max(r['max_abs_error_deg'] for r in samples))


if __name__=='__main__':
    main()
