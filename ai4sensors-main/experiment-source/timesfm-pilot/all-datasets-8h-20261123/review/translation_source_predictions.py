"""Check whether measured horizontal translation supplies a binding external constraint at the C030 solution."""
import json

import numpy as np
import torch

from check_sources import ROOT
from euroc_paired_sources import NAMES
from external_translation_variation import motion
from lock_from_frontier import read,digest,inference
from locked_replay import Predictor


def errors(up,truth):
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(up,truth),axis=1),np.einsum('ij,ij->i',up,truth)))


def main():
    torch.set_num_threads(2)
    slots=ROOT/'review/translation_training_slots.json'
    chosen=read(slots)['records'];defs={(r['batch'],r['column']):r for r in read(ROOT/'review/candidate_recipes.json')}
    models={b:Predictor(inference((b,b+'_deg'),defs))for b in ('C023','C030')}
    with np.load(ROOT/'C030/training-io.npz')as f:
        saved={b:f[b+'_tilt_deg'].reshape(160,110)for b in models}
    tallies={};rows=[]
    for j,name in enumerate(NAMES):
        entry=chosen[j];assert entry['record']==name
        indices=np.asarray(entry['native_indices']);steps=entry['original_step_indices']
        pair,delta,mask,info=motion(name);assert mask[indices].all()
        for i,((signal,hz,truth,_),condition) in enumerate(zip(pair,('raw_gyro','fixed_bias'))):
            for label,factor in [('unaltered',0.),('doubled',1.)]:
                x=signal.copy();x[:,:3]+=factor*delta;metrics={};checks={}
                for b,model in models.items():
                    predictions=model.predict(x,hz,indices)['candidate_deg']
                    angles=errors(predictions,truth[indices])
                    if factor==0.:
                        gap=float(np.max(np.abs(angles-saved[b][steps,104+2*j+i])));assert gap<2e-5,(b,name,gap)
                        checks[b]=gap
                    metrics[b]=float(np.sqrt(np.mean(angles**2)))
                key=condition+'/'+label
                tallies.setdefault(key,[]).append(metrics)
                rows.append({'record':name,'condition':key,'usable_slots':len(indices),'degree_RMSE':metrics,'original_archive_max_gap':checks})
    base=read(ROOT/'C030/solver-receipt.json')['terminal_epigraph_max_group_ratio']
    means={}
    for key,group in tallies.items():
        avg={b:float(np.mean([g[b]for g in group]))for b in models}
        ratio=avg['C030']/avg['C023']
        means[key]={'degree_RMSE':avg,'C030_relative_to_C023':ratio,'exceeds_locked_seven_group_worst':ratio>base}
    report={'scope':'External-only counterfactual. Recompute both frozen development predictors from original or altered six-axis inputs at precisely the original C026 timestamps retained by the independent checks. A potential constraint can change the old minimax solve only if it exceeds the previous epigraph at the old solution. No new weights are fitted.',
            'known_source_slots_sha256':digest(slots),'C030_checkpoint_sha256':digest(ROOT/'C030/router.pt'),
            'seven_group_terminal_worst':base,'by_condition':means,'per_record':rows}
    (ROOT/'review/translation_source_predictions.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print(means)


if __name__=='__main__':main()
