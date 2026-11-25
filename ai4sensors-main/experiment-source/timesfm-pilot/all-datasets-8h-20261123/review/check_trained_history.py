"""Reconstruct full Mamba optimizer updates and native deployment for the short and long temporal branches."""
import argparse
import hashlib
import json

import numpy as np
import torch
from torch.nn import functional as F

from check_sources import ROOT,TRANSFER
from check_training_batch import csv_rows,read,digest
from locked_replay import Predictor
from lock_from_frontier import inference
from native_catalog import Catalog
from models import ResidualHead
from temporal_readout import trained_layout


def loss_at(net,cp,canonical,source,select):
    def array(k):
        return torch.from_numpy(source[k][select])
    x=torch.from_numpy(np.array(canonical[select],copy=True))
    x=((x-cp['B6_input_mean'])/cp['B6_input_std']).clamp(-12.,12.)
    h=net.input(x)
    h=h+net.sequence(net.norm(h))
    feature=h[:,:56].mean(1)-h[:,-8:].mean(1)
    *_,mean_key,std_key=trained_layout(cp)
    local=net.output((feature-cp[mean_key])/cp[std_key])*.1
    up=array('canonical_up')
    local=local-(local*up).sum(1,keepdim=True)*up
    local=local*(np.tan(np.deg2rad(6.))/local.norm(dim=1,keepdim=True).clamp_min(1e-8)).clamp_max(1.)
    basis=array('world_basis')
    world=torch.bmm(basis,local.to(basis.dtype).unsqueeze(-1)).squeeze(-1)
    world=torch.cat((world[:,:2],torch.zeros_like(world[:,:1])),1)
    on=array('valid') & ~array('rest')
    direction=F.normalize(array('center')+.125*world*on[:,None],dim=1)
    return (direction-array('truth')).square().sum(1).mean()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('batch',nargs='?',choices=('C020','C021'),default='C020')
    args=parser.parse_args()
    coarse=args.batch=='C021'
    torch.set_num_threads(2)
    b=ROOT/args.batch
    checkpoint=b/('mamba-coarse.pt' if coarse else 'mamba-end-to-end.pt')
    cp=torch.load(checkpoint,map_location='cpu',weights_only=True)
    b6=torch.load(TRANSFER/'batch6/mamba-intrinsic-selected.pt',map_location='cpu',weights_only=True)
    prior=torch.load(ROOT/'C019/readout.pt',map_location='cpu',weights_only=True)
    np.testing.assert_array_equal(cp['B6_input_mean'].numpy(),b6['mean'].numpy())
    np.testing.assert_array_equal(cp['B6_input_std'].numpy(),b6['std'].numpy())
    with np.load(ROOT/'C019/training-inputs.npz') as fine:
        source={k:fine[k] for k in ('canonical_up','world_basis','center','truth','valid','rest','native_release','native_input_last','source_record_ordinal','global_window')}
    if coarse:
        with np.load(b/'training-coarse-inputs.npz') as extra:
            for k in ('native_release','native_input_last','source_record_ordinal','global_window','center','truth','rest'):
                np.testing.assert_array_equal(extra[k],source[k])
            source.update(canonical_up=extra['coarse_up'],world_basis=extra['coarse_world_basis'],valid=extra['coarse_valid'])
            assert not np.any(source['valid']&~extra['coarse_ready'])
            hidden=extra['initial_coarse_B_features'][source['valid']].astype('f8')
            np.testing.assert_array_equal(cp['feature_mean'].numpy(),hidden.mean(0).astype('f4'))
            np.testing.assert_array_equal(cp['feature_std'].numpy(),np.maximum(hidden.std(0),1e-4).astype('f4'))
    else:
        for k in ('mean','std'):
            np.testing.assert_array_equal(cp['C019_feature_'+k].numpy(),prior['feature_'+k].numpy())
    net=ResidualHead(cp['model']).eval()
    named=list(net.named_parameters())
    hist=[json.loads(s) for s in (b/'optimizer-steps.jsonl').read_text(encoding='utf8').splitlines() if s]
    assert [s['step'] for s in hist]==list(range(1,161))
    canonical=np.load(b/'training-canonical-history.npy',mmap_mode='r')
    assert canonical.shape==(15360,64,13)
    probes=[]
    tensor_steps={}
    with np.load(b/'optimizer-tensors.npz') as f:
        first=f['initial_weights'].astype('f8')
        raw=f['grad_raw'];decay=f['decay_delta'];adaptive=f['adam_gradient_delta']
        assert raw.shape==decay.shape==adaptive.shape==(160,5595)
        current=first.copy()
        for j,item in enumerate(hist):
            cut=0
            check=j in (0,79,159)
            for name,value in named:
                end=cut+value.numel()
                if j==0:
                    expected=np.zeros(value.numel()) if name.startswith('output.') else b6['state_dict'][name].numpy().ravel()
                    np.testing.assert_array_equal(current[cut:end],expected)
                    present=np.any(raw[:,cut:end]!=0.,axis=1)
                    tensor_steps[name]=int(present.sum())
                    assert present[1:].any() and bool(present[0])==name.startswith('output.')
                if check:
                    with torch.no_grad():
                        value.copy_(torch.from_numpy(current[cut:end].reshape(tuple(value.shape))))
                cut=end
            assert cut==len(current)
            selection=slice(j*96,(j+1)*96)
            ids=np.asarray(source['native_release'][selection],dtype='<i8')
            assert hashlib.sha256(ids.tobytes()).hexdigest()==item['native_release_sha256']
            if check:
                net.zero_grad(set_to_none=True)
                loss=loss_at(net,cp,canonical,source,selection)
                loss.backward()
                observed=np.concatenate([p.grad.detach().numpy().ravel() for _,p in named]).astype(float)
                target=raw[j].astype(float)
                cosine=float(np.dot(observed,target)/(np.linalg.norm(observed)*np.linalg.norm(target)))
                gap=float(np.max(abs(observed-target)))
                assert cosine>.99999 and gap<2e-6,(j,cosine,gap)
                loss_gap=abs(loss.item()-item['loss' if coarse else 'batch_mean_chord_loss'])
                assert loss_gap<1e-7,(j,loss_gap)
                probes.append({'step':j+1,'loss_gap':loss_gap,'gradient_cosine':cosine,'max_abs_gradient_gap':gap})
            current+=decay[j]+adaptive[j]
            assert hashlib.sha256(current.astype('<f8').tobytes()).hexdigest()==item['weight_after_sha256' if coarse else 'weights_after_sha256']
        expected=np.concatenate([cp['state_dict'][n].numpy().ravel() for n,p in named])
        np.testing.assert_array_equal(current,expected)
        np.testing.assert_array_equal(current,f['final_weights'])
    rows=csv_rows(b/'sequences.csv')
    last={(r['dataset'],r['record']):r for r in csv_rows(ROOT/('C020' if coarse else 'C019')/'sequences.csv')}
    assert len(rows)==len(last)==213
    spec={(r['batch'],r['column']):r for r in read(ROOT/'review/candidate_recipes.json')}
    column=args.batch+'_deg'
    predictor=Predictor(inference((args.batch,column),spec));catalog=Catalog()
    choices={('SHICE','R1-1'),('TUM-VI','room2'),('IMU-Array','calibration_trial/ImuArray_1_1_IMU'),
             ('OxIOD','running/data1/syn/imu1.csv')}
    native=[]
    for r in rows:
        key=r['dataset'],r['record']
        assert r['status']=='scored' and r['scoring_indices_sha256']==last[key]['scoring_indices_sha256']
        assert r['scored_frames']==last[key]['scored_frames']
        checks=['full_vqf_deg','C017_deg','C018_B_deg','C019_deg']+(['C020_deg'] if coarse else [])
        for col in checks:
            assert abs(float(r[col])-float(last[key][col]))<1e-5
        if key in choices:
            payload,_=catalog.native(*key)
            independent=predictor.score(payload)
            assert independent['scoring_indices_sha256']==r['scoring_indices_sha256']
            error=max(abs(independent['candidate_deg']-float(r[column])),abs(independent['full_vqf_deg']-float(r['full_vqf_deg'])))
            assert error<1e-5,(key,error)
            native.append({'dataset':key[0],'record':key[1],'max_abs_gap_deg':error})
    assert len(native)==len(choices)
    (ROOT/'review'/(args.batch+'_audit.json')).write_text(json.dumps({'actual_steps':160,'parameters':5595,'parameter_steps_with_real_gradient':tensor_steps,
         'native_ordinal_schedule_verified_steps':160,'step_after_weights_reconstructed':160,'cpu_probe_gradients':probes,
         'matched_development_sequences':213,'independent_native_replays':native,'checkpoint_sha256':digest(checkpoint),
         'sequences_sha256':digest(b/'sequences.csv')},indent=2)+'\n',encoding='utf8')
    print('full weight reconstruction, three gradient cosines',[k['gradient_cosine'] for k in probes],
          'largest native gap',max(r['max_abs_gap_deg'] for r in native))


if __name__=='__main__':
    main()
