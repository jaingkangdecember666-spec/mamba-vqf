"""Recalculate a frozen external angular-cap objective and control rows on the common development panel."""
import argparse
import json

import numpy as np
import torch

from anchor_blend_features import included_degrees
from check_training_batch import csv_rows
from lock_from_frontier import ROOT,read,digest,inference
from locked_replay import Predictor
from native_catalog import Catalog
from physical_cap_certificate import ExternalCap
from physical_angle_cap import cap_direction


def main():
    ap=argparse.ArgumentParser();ap.add_argument('batch',choices=('C036','C037'));args=ap.parse_args()
    torch.set_num_threads(2)
    b=ROOT/args.batch; lock=read(b/'coefficient-lock.json'); bank=ExternalCap()
    c=float(lock['cap_deg']);actual=bank.risks(c)
    assert abs(actual.max()-lock['objective'])<1e-8
    for d,v in zip(bank.groups,actual):
        assert abs(lock['groups'][d]['ratio_to_default']-v)<1e-8
    caps=read(ROOT/'review/candidate_recipes.json')
    registry={(r['batch'],r['column']):r for r in caps}
    spec=inference(('C023','C023_deg'),registry)
    spec['physical_angle_cap']={'degrees':c}
    model=Predictor(spec);db=Catalog()
    reference={(r['dataset'],r['record']):r for r in csv_rows(ROOT/'C022/sequences.csv')}
    table=csv_rows(b/'sequences.csv');assert len(reference)==len(table)==213
    native=[];overall_gap=0.
    wanted={('BROAD','04_undisturbed_slow_rotation_with_breaks_A'),('SHICE','R1-1'),('TUM-VI','room2'),('OxIOD','running/data1/syn/imu1.csv')}
    for r in table:
        key=r['dataset'],r['record'];old=reference[key]
        for k in ('group','scored_frames','scoring_indices_sha256','status'):assert r[k]==old[k]
        for k in ('physical_only_deg','full_vqf_deg'):
            assert abs(float(r[k])-float(old[k]))<1e-5
        if c==0.:
            overall_gap=max(overall_gap,abs(float(r[args.batch+'_deg'])-float(old['physical_only_deg'])))
        if key in wanted:
            raw,_=db.native(*key);out=model.score(raw)
            assert out['scoring_indices_sha256']==r['scoring_indices_sha256']
            gap=abs(out['candidate_deg']-float(r[args.batch+'_deg']));assert gap<1.e-5
            native.append({'dataset':key[0],'record':key[1],'error_deg':gap})
    assert len(native)==len(wanted) and overall_gap<1e-5
    with np.load(b/'training-io.npz')as io:
        rebuilt=cap_direction(bank.physical,bank.model,c)
        gap=float(included_degrees(rebuilt,io[args.batch]).max());assert gap<1e-5
    receipt={'batch':args.batch,'primary_development_records':len(table),'trained_cap_deg':c,'slot_max_disagreement_deg':gap,
             'source_objective':float(actual.max()),'external_ratios':dict(zip(bank.groups,actual.tolist())),
             'native_development_rows':native,'coefficient_lock_sha256':digest(b/'coefficient-lock.json')}
    if c==0.:receipt['max_development_gap_against_existing_C022_physical_deg']=overall_gap
    (ROOT/'review'/f'{args.batch}_audit.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8')
    print(args.batch,c,list(actual),gap,overall_gap)


if __name__=='__main__':main()
