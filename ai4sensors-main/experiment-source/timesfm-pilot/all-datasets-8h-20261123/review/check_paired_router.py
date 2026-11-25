"""Verify the three physical recordings and two externally defined gyro input conditions behind C026."""
import hashlib
import json

import numpy as np
import torch

from check_history_router import prediction
from check_motion_router import verify_optimizer
from check_sources import ROOT
from check_training_batch import csv_rows
from euroc_paired_sources import paired,NAMES
from lock_from_frontier import read,digest,inference
from locked_replay import Predictor
from native_blocks import block_inputs
from native_catalog import Catalog
from optical_score import canonical_frame
from response_router import release_alternatives
from router_motion_context import series
from source_motion_router_diagnostic import native


def compare_sources(a,core,b):
    catalog=read(b/'sampling-lock.json')
    entries=[]
    for j,name in enumerate(NAMES):
        original,variant=paired(name)
        slots=a['native_release'][:,104+2*j]
        np.testing.assert_array_equal(slots,a['native_release'][:,105+2*j])
        assert len(set(slots))==160
        np.testing.assert_array_equal(original[0][:,:3],variant[0][:,:3])
        np.testing.assert_array_equal(original[2],variant[2]);np.testing.assert_array_equal(original[3],variant[3])
        both=[];errors={}
        for t,raw in enumerate((original,variant)):
            x,hz=native(raw);q,win,cuts,releases=block_inputs(x,hz)
            allowed=raw[3][releases]&canonical_frame(win)[2]
            both.append(releases[allowed])
            pos=np.searchsorted(releases,slots);np.testing.assert_array_equal(releases[pos],slots)
            at=cuts[pos];column=104+2*j+t
            np.testing.assert_array_equal(a['native_input_last'][:,column],at)
            # The four context and all three world-space targets are recomputed from the source-specific stream.
            center=q[slots].apply(core.predict(raw[0],hz,slots)['candidate_deg'])
            center/=center[:,2:3]
            fast,slow=release_alternatives(x,hz,q,at)
            y=q[slots].apply(raw[2][slots])
            dx={}
            for k,v in [('center',center),('fast',fast),('slow',slow),('truth',y)]:
                dx[k]=float(np.abs(v-a[k][:,column]).max())
            v=series(x,hz)[at]
            stats=read(b/'normalizers.json')
            zn=(v-np.asarray(stats['context_mean']['values']))/np.asarray(stats['context_std']['values'])
            dx['normalized_context']=float(np.abs(zn-a['features'][:,column,24:]).max())
            assert max(dx.values())<2e-7,(name,column,dx)
            errors[catalog['slot_order'][2*j+t]['condition']]=dx
        common=np.intersect1d(*both)
        expected=np.array(sorted(common,key=lambda ix:(hashlib.sha256(f'C026|{name}|{ix}'.encode()).digest(),int(ix)))[:160])
        np.testing.assert_array_equal(slots,expected)
        entries.append({'record':name,'eligible_common_releases':len(common),'paired_native_releases':len(slots),'checked_against_original':errors})
    return entries


def main():
    torch.set_num_threads(2);b=ROOT/'C026'
    with np.load(b/'training-inputs.npz')as f:a={k:f[k]for k in f.files}
    with np.load(ROOT/'C024/training-inputs.npz')as f:
        for k in ('center','fast','slow','truth','enabled','native_input_last','native_release','source_record_ordinal','domain'):
            np.testing.assert_array_equal(a[k][:,:104],f[k])
    with np.load(ROOT/'C025/training-context.npz')as f:
        np.testing.assert_array_equal(a['features'][:,:104],f['features'])
    with np.load(ROOT/'C024/training-io.npz')as f:
        np.testing.assert_array_equal(a['effective_loss_weights'][:,:104].ravel(),f['effective_loss_weights'])
    z=a['features'].reshape(-1,28)
    cp=torch.load(b/'router.pt',map_location='cpu',weights_only=True)
    parent=torch.load(ROOT/'C025/router.pt',map_location='cpu',weights_only=True)
    for k in ('old_feature_mean','old_feature_std','context_mean','context_std','p0','log_p0'):
        torch.testing.assert_close(cp[k],parent[k],rtol=0.,atol=0.)
    direct,_,_,_=prediction(z,cp['p0'].numpy(),np.zeros(58),*[a[k].reshape(-1,3)for k in ('center','fast','slow')],a['enabled'].ravel())
    zero=((direct-a['truth'].reshape(-1,3))**2).sum(1)
    ordinary=a['domain'].ravel()!='EuRoC'
    sets={'frozen':ordinary}
    for case in ('raw','fixed_prefix_bias_removed'):
        sel=a['condition'].ravel()==case
        assert sel.sum()==480 and np.unique(a['source_record_ordinal'].ravel()[sel],return_counts=True)[1].tolist()==[160]*3
        value=1./(2*480*zero[sel].mean())
        np.testing.assert_allclose(a['effective_loss_weights'].ravel()[sel],value,rtol=1e-11)
        sets[case]=sel
    assert np.all(sum(v.astype(int)for v in sets.values())==1)
    report=verify_optimizer(z,a,a['effective_loss_weights'].ravel(),cp,b,160./6,alignment=sets)
    desc={(r['batch'],r['column']):r for r in read(ROOT/'review/candidate_recipes.json')}
    report['external_source_raw_checks']=compare_sources(a,Predictor(inference(('C020','C020_deg'),desc)),b)
    db=Catalog();model=Predictor(inference(('C026','C026_deg'),desc))
    rows=csv_rows(b/'sequences.csv');prior={(r['dataset'],r['record']):r for r in csv_rows(ROOT/'C025/sequences.csv')}
    assert len(rows)==len(prior)==213
    selected={('SHICE','R1-1'),('TUM-VI','room2'),('OxIOD','running/data1/syn/imu1.csv'),('BROAD','04_undisturbed_slow_rotation_with_breaks_A')}
    proof=[]
    for r in rows:
        key=r['dataset'],r['record'];old=prior[key]
        assert all(r[k]==old[k]for k in ('scoring_indices_sha256','scored_frames','group','status'))
        for k in ('full_vqf_deg','C023_deg','C025_deg','fixed_tau1p5_deg','fixed_tau3_deg','fixed_tau6_deg'):
            assert abs(float(r[k])-float(old[k]))<1e-5
        if key in selected:
            raw,_=db.native(*key);score=model.score(raw)
            gap=max(abs(score[k]-float(r[c]))for k,c in [('candidate_deg','C026_deg'),('full_vqf_deg','full_vqf_deg')])
            assert gap<1e-5 and r['scoring_indices_sha256']==score['scoring_indices_sha256']
            proof.append({'dataset':key[0],'record':key[1],'max_abs_gap_deg':gap})
    assert len(proof)==len(selected)
    report.update(raw_development_checks=proof,development_records=213,actual_external_acquisitions=148,
                  old_slots_preserved=16640,additional_condition_slots=960,original_temporal_pairs=480,checkpoint_sha256=digest(b/'router.pt'))
    (ROOT/'review/C026_audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print(report['max_gradient_component_gap'],max(r['max_abs_gap_deg']for r in proof))


if __name__=='__main__':main()
