"""Replay all 37 SHICE development recordings and compare C014-B within fixed motion strata."""
import hashlib
import json
from collections import defaultdict
from statistics import fmean

import numpy as np
import torch

from check_sources import ROOT,TRANSFER,weights
from check_training_batch import csv_rows
from mixture_replay import replay
from native_catalog import Catalog


def angles(pred,y):
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(pred,y),axis=1),(pred*y).sum(1)))


def main():
    torch.set_num_threads(2)
    catalog=Catalog()
    rows=[r for r in csv_rows(ROOT/'C014/sequences.csv') if r['dataset']=='SHICE']
    assert len(rows)==37 and all(r['group'] in ('R1','R2','R3','R4') for r in rows)
    heads={'C003':(weights(TRANSFER/'batch6/mamba-intrinsic-selected.pt','mamba_v1'),.25,'always'),
           'C013_readout':(weights(ROOT/'C013/seed-C006-readout.pt','mamba_v1'),.25,'rest_off')}
    all_stats=[];bins=defaultdict(list)
    for r in rows:
        (signal,hz,target,mask),_=catalog.native('SHICE',r['record'])
        ids=np.flatnonzero(mask & np.isfinite(signal).all(1))
        assert hashlib.sha256(ids.astype('<i8').tobytes()).hexdigest()==r['scoring_indices_sha256']
        pred=replay(signal,hz,heads,{'B_deg':{'C003':.5,'C013_readout':.5}},ids)
        truth=target[ids]
        base=angles(pred['full_vqf_deg'],truth)**2
        ours=angles(pred['B_deg'],truth)**2
        speed=np.linalg.norm(signal[ids,3:],axis=1)
        load=np.abs(np.linalg.norm(signal[ids,:3],axis=1)/9.80665-1.)
        groups={'all':np.ones(len(ids),bool),'gyro_lt_0.05':speed<.05,'gyro_0.05_to_0.5':(speed>=.05)&(speed<.5),
                'gyro_0.5_to_2':(speed>=.5)&(speed<2.),'gyro_ge_2':speed>=2.,
                'force_deviation_below_0.05g':load<.05,'force_deviation_0.05_to_0.2g':(load>=.05)&(load<.2),
                'force_deviation_ge_0.2g':load>=.2}
        entry={'record':r['record'],'device':r['group'],'scoring_indices_sha256':r['scoring_indices_sha256'],'strata':{}}
        for name,keep in groups.items():
            if not keep.any():continue
            v={'n':int(keep.sum()),'baseline_deg':float(np.sqrt(base[keep].mean())),
               'C014_B_deg':float(np.sqrt(ours[keep].mean()))}
            entry['strata'][name]=v
            bins[name].append(v)
        assert abs(entry['strata']['all']['C014_B_deg']-float(r['B_deg']))<1e-5
        assert abs(entry['strata']['all']['baseline_deg']-float(r['full_vqf_deg']))<1e-5
        all_stats.append(entry)
        print(r['record'],len(ids),flush=True)
    combined={}
    for k,v in bins.items():
        n=sum(r['n'] for r in v)
        diff=[r['C014_B_deg']-r['baseline_deg'] for r in v]
        combined[k]={'supporting_records':len(v),'frames':n,'record_equal_delta_deg':fmean(diff),
                     'record_equal_baseline_deg':fmean(r['baseline_deg'] for r in v),
                     'record_equal_C014_B_deg':fmean(r['C014_B_deg'] for r in v),
                     'records_improved':sum(d<0. for d in diff),'record_median_delta_deg':float(np.median(diff))}
    out={'scope':'SHICE R1-R4 only. C014-B is fixed for this descriptive analysis. Gyro thresholds are rad/s; '
                 'force deviation is the absolute departure of raw acceleration magnitude from 9.80665 divided by 9.80665. '
                 'A record with any matching frame has one vote within that stratum. Speed and force tables overlap.',
         'common_37_native_masks':True,'groups':combined,'records':all_stats}
    (ROOT/'review/SHICE_development_motion.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
    print(combined,flush=True)


if __name__=='__main__':
    main()
