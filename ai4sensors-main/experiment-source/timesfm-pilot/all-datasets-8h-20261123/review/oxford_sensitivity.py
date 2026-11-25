"""Paired sensitivity of two frozen candidates to small common Oxford reference perturbations."""
import csv
import hashlib
import json
import argparse
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from check_sources import ROOT, TRANSFER, weights
from native_catalog import Catalog
from native_blocks import block_inputs
from optical_score import estimate,infer_canonical,metric


def predictions(raw,models):
    signal,hz,_,_=raw
    observed=np.isfinite(signal).all(1)
    previous=np.maximum.accumulate(np.where(observed,np.arange(len(observed)),-1))
    filled=signal[previous.clip(min=0)].copy();filled[previous<0]=0.
    q,w,at,release=block_inputs(filled,hz)
    vertical=np.array([0.,0.,1.])
    predicted={'default':q.inv().apply(vertical)}
    bias,active,_,_=estimate(filled,observed,1./hz)
    if active.any():
        filled[:,:3]-=bias
        q,w,at,release=block_inputs(filled,hz)
    lookup=np.searchsorted(release,np.arange(len(signal)),side='right')
    for name,model in models.items():
        offset=q[at].apply(infer_canonical(model,w))
        if name=='C006':offset[w[:,-1,12].astype(bool)]=0
        up=vertical+.25*offset
        up/=np.linalg.norm(up,axis=1,keepdims=True)
        predicted[name]=q.inv().apply(np.vstack((vertical,up))[lookup])
    return predicted


def perturbed_scores(ref,mask,predict):
    indices=np.flatnonzero(mask[1:-1]&mask[2:]&mask[:-2])+1
    if not len(indices):
        raise ValueError('Oxford reference perturbations have no common rows')
    current=ref[indices]
    refs={'nominal':current,'earlier_10ms':ref[indices-1],'later_10ms':ref[indices+1]}
    for axis in (0,1):
        for sign in (-1,1):
            label=f'body_{"xy"[axis]}_{sign:+d}deg'
            refs[label]=Rotation.from_rotvec(np.deg2rad(sign)*np.eye(3)[axis]).apply(current)
    predicted=predict(indices)
    return indices,{label:{k:metric(v,target) for k,v in predicted.items()} for label,target in refs.items()}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--selection-lock',type=Path)
    parser.add_argument('--main-csv',type=Path)
    parser.add_argument('--development-batch')
    parser.add_argument('--column')
    args=parser.parse_args()
    if bool(args.selection_lock)!=bool(args.main_csv):
        parser.error('Use --selection-lock and --main-csv together')
    if bool(args.development_batch)!=bool(args.column) or (args.development_batch and args.selection_lock):
        parser.error('Use either the formal lock and table or a development batch and column')
    torch.set_num_threads(2)
    catalog=Catalog(args.selection_lock)
    if args.selection_lock:
        from locked_replay import SelectedPredictor
        model=SelectedPredictor(args.selection_lock)
        source=args.main_csv
        names=['candidate_deg','physical_deg']
        baseline='full_vqf_deg'
        candidate_field='candidate_deg'
    elif args.development_batch:
        from lock_from_frontier import inference,read
        from locked_replay import Predictor
        index={(r['batch'],r['column']):r for r in read(ROOT/'review/candidate_recipes.json')}
        model=Predictor(inference((args.development_batch,args.column),index))
        source=ROOT/args.development_batch/'sequences.csv'
        names=['candidate_deg','physical_deg']
        baseline='full_vqf_deg'
        candidate_field=args.column
    else:
        models={'C003':weights(TRANSFER/'batch6/mamba-intrinsic-selected.pt','mamba_v1'),
                'C006':weights(ROOT/'C004/B-readout.pt','mamba_v1')}
        names=list(models)
        baseline='default'
        source=ROOT/'C006/sequences.csv'
    with source.open(encoding='utf-8-sig',newline='') as f:
        all_rows=[r for r in csv.DictReader(f) if r['dataset']=='OxIOD']
    expected=68 if args.selection_lock else 66
    assert len(all_rows)==len({r['record'] for r in all_rows})==expected
    rows=[]
    missing=[]
    for row in all_rows:
        if not row.get('full_vqf_deg'):
            missing.append(row['record'])
            continue
        raw,info=catalog.native('OxIOD',row['record'])
        signal,hz,ref,mask=raw
        assert abs(hz-100.)<1.e-6
        if args.selection_lock or args.development_batch:
            if args.selection_lock:
                assert catalog.rows['OxIOD',row['record']]['side'].startswith('sealed')
            use=np.flatnonzero(mask)
            assert hashlib.sha256(use.astype('<i8').tobytes()).hexdigest()==row['scoring_indices_sha256']
            pred=model.predict(signal,hz,use)
            official={k:metric(v,ref[use]) for k,v in pred.items()}
            fields=[(baseline,baseline),('candidate_deg',candidate_field)]
            if 'physical_deg' in row:
                fields.append(('physical_deg','physical_deg'))
            indices,results=perturbed_scores(ref,mask,lambda i:{k:v[np.searchsorted(use,i)] for k,v in pred.items()})
        else:
            pred=predictions(raw,models)
            official={k:metric(v[mask],ref[mask]) for k,v in pred.items()}
            fields=[('default','full_vqf_deg'),('C003','c003_deg'),('C006','B_rest_gated_deg')]
            indices,results=perturbed_scores(ref,mask,lambda i:{k:v[i] for k,v in pred.items()})
        for k,col in fields:
            assert abs(official[k]-float(row[col]))<1e-5
        rows.append({'record':row['record'],'group':row['group'],'reference_flags':info['quality_flags'],
                     'primary_frames':int(mask.sum()),'common_frames':len(indices),
                     'common_indices_sha256':hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest(),
                     'primary_exact_replay_deg':official,'common_frame_scores_deg':results})
    overview={}
    for subset,rr in [('all',rows),('independently_checked',[r for r in rows if not r['reference_flags']])]:
        out={}
        if rr:
            for scenario in rr[0]['common_frame_scores_deg']:
                records=[r['common_frame_scores_deg'][scenario] for r in rr]
                out[scenario]={k:float(np.mean([v[k]-v[baseline] for v in records])) for k in names}
        overview[subset]={'records':len(rr),'candidate_minus_default_deg':out}
    result={'scope':('68 formal Oxford sequences after selection lock. ' if args.selection_lock else '66 development-only Oxford sequences. ')+
                    'Same frozen predictions, paired on the same intersection '
                    'of nominal and neighboring valid optical rows. Perturbations are prescribed illustrations, not estimated '
                    'error bars. No predictor training or official mask change.',
            'base_results_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'requested_records':expected,
            'missing_base_score':missing,'summary':overview,'sequences':rows}
    if args.selection_lock:
        result['selection_lock_sha256']=hashlib.sha256(args.selection_lock.read_bytes()).hexdigest()
    target='oxford_final_sensitivity.json' if args.selection_lock else 'oxford_development_sensitivity.json'
    if args.development_batch:
        result['development_recipe']=[args.development_batch,args.column]
        target=f'oxford_{args.development_batch}_{args.column}_sensitivity.json'
    (ROOT/'review'/target).write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print('records',len(rows),'official frames',sum(r['primary_frames'] for r in rows),
          'common',sum(r['common_frames'] for r in rows),overview,flush=True)


if __name__=='__main__':
    main()
