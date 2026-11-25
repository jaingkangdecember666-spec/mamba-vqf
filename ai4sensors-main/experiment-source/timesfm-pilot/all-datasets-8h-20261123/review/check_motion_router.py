"""Replay five-source motion normalization and the 58-parameter optimizer from its analytical gradient."""
import json

import numpy as np
import torch

from check_history_router import gradient,prediction
from check_sources import ROOT
from check_training_batch import csv_rows
from lock_from_frontier import read,digest,inference
from locked_replay import Predictor
from native_catalog import Catalog
from router_motion_context import NAMES


def verify_inputs(a,cp):
    b=ROOT/'C025'
    archive=read(ROOT/'review/source_motion_router_diagnostic.json')
    with np.load(b/'training-context.npz')as io:
        x=io['features'];raw=io['context_raw']
        for k in ('native_release','native_input_last','source_record_ordinal','domain','record_equal_weights'):
            np.testing.assert_array_equal(a[k],io[k])
    with np.load(ROOT/'review/source_motion_router_features.npz')as f:
        np.testing.assert_array_equal(raw[:,:96].reshape(-1,4),f['features'])
    for i,n in enumerate(read(ROOT/'C024/training-preparation.json')['rows']):
        idx=a['source_record_ordinal']==139+i
        src=ROOT/archive['davis'][n['record']]['file']
        assert digest(src)==archive['davis'][n['record']]['sha256']
        indexed=np.load(src,mmap_mode='r')[a['native_input_last'][idx]]
        np.testing.assert_array_equal(raw[idx],indexed)
    raw=raw.reshape(-1,4)
    ww=a['record_equal_weights'].ravel();ww=ww/ww.sum()
    avg=ww@raw;std=np.sqrt(ww@(raw-avg)**2)
    np.testing.assert_allclose(avg,cp['context_mean'].numpy(),rtol=1e-13)
    np.testing.assert_allclose(std,cp['context_std'].numpy(),rtol=1e-13)
    assert tuple(cp['feature_names'])==NAMES
    np.testing.assert_array_equal(x[:,:,:24],a['z'])
    np.testing.assert_allclose(x[:,:,24:].reshape(-1,4),(raw-avg)/std,rtol=1e-9,atol=1e-12)
    return x.reshape(-1,28)


def verify_optimizer(z,a,w,cp,b,scale=32.,selection=None,alignment=None):
    flat=lambda k:a[k].reshape((-1,)+a[k].shape[2:])
    p0=cp['p0'].numpy()
    center,fast,slow,truth,use=(flat(k)for k in ('center','fast','slow','truth','enabled'))
    steps=[json.loads(line)for line in (b/'optimizer-steps.jsonl').read_text().splitlines()]
    assert len(steps)==160
    weights=np.zeros(58);m=np.zeros(58);v=np.zeros(58)
    max_grad=max_update=max_loss=max_part=0.
    width=a['native_release'].shape[1]
    for i,row in enumerate(steps):
        assert row['step']==i+1
        order=i if selection is None else int(selection[i])
        s=np.arange(order*width,(order+1)*width)
        def grad_at(ix):
            return gradient(z[ix],p0,weights,center[ix],fast[ix],slow[ix],truth[ix],use[ix],scale*w[ix])
        base_error,grad=grad_at(s)
        if alignment:
            for key,select in alignment.items():
                _,g=grad_at(s[select[s]])
                expected=np.asarray(row['optimizer_gradient_components'][key])
                max_part=max(max_part,float(np.abs(g-expected).max()))
        max_loss=max(max_loss,abs(base_error-row['objective']))
        max_grad=max(max_grad,np.abs(grad-row['gradient_before_clip']).max())
        grad*=min(1.,2./(np.linalg.norm(grad)+1e-6))
        m=.9*m+.1*grad;v=.999*v+.001*grad**2
        weights=.9999*weights-.01*(m/(1.-.9**(i+1)))/(np.sqrt(v/(1.-.999**(i+1)))+1e-8)
        max_update=max(max_update,np.abs(weights-row['weights_after']).max())
    assert max(max_grad,max_update,max_loss,max_part)<1e-10
    np.testing.assert_allclose(weights,np.r_[cp['state_dict']['weight'].numpy().ravel(),cp['state_dict']['bias'].numpy()],atol=1e-12)
    y,_,_,p=prediction(z,p0,weights,center,fast,slow,use)
    with np.load(b/'training-io.npz') as f:
        np.testing.assert_allclose(p,f[b.name+'_prob'].reshape(-1,3),atol=1e-12)
        np.testing.assert_allclose(((y-truth)**2).sum(1),f[b.name+'_chord'].ravel(),atol=1e-12)
    return {'optimizer_steps':len(steps),'scalar_weight_updates':len(steps)*58,'max_loss_gap':max_loss,'max_gradient_gap':max_grad,'max_optimizer_gap':max_update,'max_gradient_component_gap':max_part}


def main():
    torch.set_num_threads(2)
    b=ROOT/'C025'
    with np.load(ROOT/'C024/training-inputs.npz') as f:a={k:f[k]for k in f.files}
    with np.load(ROOT/'C024/training-io.npz') as f:w=f['effective_loss_weights']
    cp=torch.load(b/'router.pt',map_location='cpu',weights_only=True)
    z=verify_inputs(a,cp)
    evidence=verify_optimizer(z,a,w,cp,b)
    defs={(r['batch'],r['column']):r for r in read(ROOT/'review/candidate_recipes.json')}
    model=Predictor(inference(('C025','C025_deg'),defs));catalog=Catalog()
    other={(r['dataset'],r['record']):r for r in csv_rows(ROOT/'C024/sequences.csv')}
    rows=csv_rows(b/'sequences.csv');assert len(rows)==len(other)==213
    samples={('SHICE','R1-1'),('TUM-VI','room2'),('OxIOD','running/data1/syn/imu1.csv'),('BROAD','04_undisturbed_slow_rotation_with_breaks_A')}
    out=[]
    for r in rows:
        key=r['dataset'],r['record'];previous=other[key]
        for k in ('scoring_indices_sha256','scored_frames','group','status'):assert previous[k]==r[k]
        for k in ('full_vqf_deg','C023_deg','C024_deg','fixed_tau1p5_deg','fixed_tau3_deg','fixed_tau6_deg'):assert abs(float(r[k])-float(previous[k]))<1e-5
        if key not in samples:continue
        data,_=catalog.native(*key);result=model.score(data)
        err=max(abs(result[k]-float(r[c]))for k,c in [('candidate_deg','C025_deg'),('full_vqf_deg','full_vqf_deg')])
        assert r['scoring_indices_sha256']==result['scoring_indices_sha256'] and err<1e-5
        out.append({'dataset':key[0],'record':key[1],'max_abs_gap_deg':err})
    assert len(out)==len(samples)
    evidence.update(records=213,source_records=145,source_slots=len(z),context_normalization='five source and within-source recording equal',
                    raw_replays=out,checkpoint_sha256=digest(b/'router.pt'))
    (ROOT/'review/C025_audit.json').write_text(json.dumps(evidence,indent=2)+'\n',encoding='utf8')
    print(evidence['max_gradient_gap'],max(r['max_abs_gap_deg'] for r in out))


if __name__=='__main__':main()
