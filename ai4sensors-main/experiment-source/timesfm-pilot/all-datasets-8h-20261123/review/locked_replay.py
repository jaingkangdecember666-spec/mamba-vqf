"""Independent predictions and paired tilt scores for one fully specified selection lock."""
import hashlib
import json
from pathlib import Path

import numpy as np

from check_sources import ROOT, weights
from mixture_replay import replay
from optical_score import metric


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


class Predictor:
    def __init__(self,spec,device='cpu'):
        self.spec=spec
        if self.spec['front_end']!='causal_sphere':
            raise ValueError(f"Unrecognized front end {self.spec['front_end']}")
        self.models={}
        self.mix={}
        for component in self.spec['heads']:
            path=ROOT/component['weights']
            if sha256(path)!=component['sha256']:
                raise ValueError(f'Locked checkpoint changed: {path.name}')
            name=component['name']
            if name in self.models or component['gate'] not in ('always','rest_off'):
                raise ValueError(f'Invalid component name or gate: {name}')
            head=weights(path,component['kind'],device)
            self.models[name]=(head,component['gain'],component['gate'])
            self.mix[name]=component['fraction']
        if not self.mix or min(self.mix.values())<0. or not np.isclose(sum(self.mix.values()),1.):
            raise ValueError('Expected nonnegative Mamba component fractions summing to one')
        self.extra=[]
        for item in self.spec.get('additions',[]):
            from temporal_readout import load_readout,load_trained_readout
            path=ROOT/item['weights']
            if sha256(path)!=item['sha256']:
                raise ValueError('Changed additive temporal weights')
            if item['name'] in self.models or item['gate']!='rest_off':
                raise ValueError('Invalid temporal readout identity or gating')
            if item['kind']=='frozen_mamba_temporal_readout':
                head,gain=load_readout(self.models[item['backbone']][0],path)
            elif item['kind']=='trained_mamba_temporal_readout':
                head,gain=load_trained_readout(path,device)
            else:
                raise ValueError(f"Unrecognized additive network {item['kind']}")
            if gain!=item['gain']:
                raise ValueError('Readout deployment gain differs from the lock')
            if item['kind']=='trained_mamba_temporal_readout' and head[0].history_scale!=item.get('history_scale',1):
                raise ValueError('Checkpoint block duration differs from the registered duration')
            head[0].history_scale=item.get('history_scale',1)
            if head[0].history_scale not in (1,4):
                raise ValueError('Unrecognized temporal head sampling interval')
            self.extra.append(item['name'])
            self.models[item['name']]=(head,gain,item['gate'])
        self.force={}
        self.bank={}
        self.router=None
        physical=self.spec.get('stable_force')
        if physical:
            if physical['tau_s']!=.25 or physical['spread_deg']!=3. or physical['cap_deg']!=6.:
                raise ValueError('Unrecognized stable-force definition')
            self.force={'candidate_deg':physical['gain'],'physical_deg':physical['gain']}
        response=self.spec.get('response_bank')
        if response:
            if (response['time_constants_s']!=[1.5,6.] or response['fast_to_slow_ratio']!=2. or response['cap_deg']!=6.
                    or response['gate']!='valid_nonrest' or not 0.<=response['gain']<=1.):
                raise ValueError('Unrecognized auxiliary gravity-filter recipe')
            self.bank={'candidate_deg':response['gain'],'physical_deg':response['gain']}
        if self.spec.get('response_router'):
            from response_router import ResponseRouter
            rule=self.spec['response_router']
            if rule['kind']!='history_softmax' or rule['gate']!='valid_nonrest' or self.bank:
                raise ValueError('Invalid or conflicting gravity response selection')
            p=ROOT/rule['weights']
            if sha256(p)!=rule['sha256']:
                raise ValueError('Router checkpoint differs from the lock')
            self.router=ResponseRouter(self.models[rule['backbone']][0],p,rule['prior_gain'],rule.get('context'))
            self.bank={'physical_deg':rule['prior_gain']}

    def predict(self,signal,hz,indices):
        groups={'candidate_deg':self.mix,'physical_deg':{}}
        out=replay(signal,hz,self.models,groups,indices,self.force,{'candidate_deg':self.extra},self.bank,self.router)
        if 'raw_anchor_blend' in self.spec:
            from raw_vqf_blend import apply_offline
            out=apply_offline(signal,hz,indices,out,self.spec['raw_anchor_blend']['beta'])
        if 'physical_angle_cap' in self.spec:
            from physical_angle_cap import cap_direction
            out['candidate_deg']=cap_direction(out['physical_deg'],out['candidate_deg'],self.spec['physical_angle_cap']['degrees'])
        return out

    def stream_heads(self):
        return [(h,g*(self.mix[k] if k in self.mix else 1.),gate) for k,(h,g,gate) in self.models.items()]

    def score(self,raw):
        signal,hz,reference,mask=raw
        indices=np.flatnonzero(mask & np.isfinite(signal).all(axis=1))
        if not len(indices):
            raise ValueError('No paired native scoring rows')
        truth=reference[indices]
        if not np.isfinite(truth).all() or not np.allclose(np.linalg.norm(truth,axis=1),1.,atol=1e-6):
            raise ValueError('Reference is not a finite unit direction')
        predictions=self.predict(signal,hz,indices)
        scores={k:metric(v,truth) for k,v in predictions.items()}
        return {'scored_frames':len(indices),'native_frames':len(signal),'rate_hz':hz,
                'scoring_indices_sha256':hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest(),**scores}


class SelectedPredictor(Predictor):
    def __init__(self,lock_path,device='cpu'):
        self.lock_path=Path(lock_path)
        self.lock=json.loads(self.lock_path.read_text(encoding='utf-8-sig'))
        if sha256(ROOT/'C001/inventory.json')!=self.lock['inventory_sha256']:
            raise ValueError('The selection lock refers to a different inventory')
        super().__init__(self.lock['review_inference'],device)
