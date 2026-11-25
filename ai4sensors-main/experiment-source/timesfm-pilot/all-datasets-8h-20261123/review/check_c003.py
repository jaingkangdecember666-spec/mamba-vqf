"""Check every C003 record and array calibration update against independent raw-source calculations."""
import csv
import hashlib
import json
import numpy as np
import torch
from check_sources import ROOT, DATA, TRANSFER, weights, score
from check_array import base_inputs


def read_csv(batch, name):
    with (ROOT/batch/name).open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def independent_trace(signal, hz):
    # Re-form one observation per completed quiet second; solve a sphere with a free squared radius.
    width=round(hz);centers=[];anchors=[];ns=[];times=[];values=[];bias=np.zeros(3)
    for end in range(width,len(signal)+1,width):
        s=signal[end-width:end];a=s[:,:3];m=a.mean(0);l=np.linalg.norm(m)
        if not (8<=l<=12) or np.linalg.norm(s[:,3:],axis=1).max()>.05 or np.linalg.norm(a-m,axis=1).max()>.10:continue
        alpha=m/l
        sims=np.dot(anchors,alpha) if anchors else []
        if len(sims) and max(sims)>=np.cos(np.pi/6):
            j=int(np.argmax(sims));ns[j]+=1;centers[j]+=(m-centers[j])/ns[j]
        else:
            ns.append(1);centers.append(m.copy());anchors.append(alpha)
        x=np.asarray(centers)[np.array(ns)>=3]
        if len(x)<4:continue
        centered=x-x.mean(0);sv=np.linalg.svd(centered,compute_uv=False)
        if sv[-1]/np.sqrt(len(x))<.75 or sv[-1]/sv[0]<.15:continue
        fit=np.linalg.lstsq(np.c_[2*x,np.ones(len(x))],(x*x).sum(1),rcond=None)[0]
        target=fit[:3];length=np.linalg.norm(x-target,axis=1);radius=np.sqrt(np.mean(length**2))
        if np.linalg.norm(target)>1.5 or not 8.8<=radius<=10.8 or np.sqrt(np.mean((length-radius)**2))>.20:continue
        diff=.2*(target-bias);bias=bias+diff*min(1.,.025/max(1e-15,np.linalg.norm(diff)))
        times.append(end);values.append(bias.copy())
    return np.asarray(times),np.asarray(values).reshape(-1,3)


def main():
    torch.set_num_threads(2)
    head=weights(TRANSFER/'batch6/mamba-intrinsic-selected.pt','mamba_v1')
    table=np.memmap(DATA/'IMU-Array-main/Calibration/Group1_ref_data.bin',dtype='<f8',mode='r').reshape(-1,7)
    rows=read_csv('C003','sequences.csv')
    prior={(r['dataset'],r['record']):r for r in read_csv('C002','sequences.csv')}
    inventory=json.loads((ROOT/'C001/inventory.json').read_text('utf8'))
    arrays=[r for r in inventory['records'] if r['side']=='development' and r['dataset']=='IMU-Array']
    specs={r['record']:r for r in arrays}
    updates={(r['dataset'],r['record']):r['updates'] for r in json.loads((ROOT/'C003/bias-updates.json').read_text('utf8'))}
    assert len(set(prior))==len(updates)==len(rows)==85
    anchors=[]; physical=[]
    for r in rows:
        k=r['dataset'],r['record'];old=prior[k]
        assert all(r[f]==old[f] for f in ('scoring_indices_sha256','scored_frames','full_vqf_deg','b6_gain_025_deg'))
        if k[0]!='IMU-Array':
            assert not updates[k] and int(r['active_scored_frames'])==0 and r['full_vqf_deg']==r['debiased_full_vqf_deg'] and r['b6_gain_025_deg']==r['debiased_b6_gain_025_deg']
            continue
        (signal,hz,up,mask),stats=base_inputs(specs[k[1]],table)
        ix,values=independent_trace(signal,hz)
        published=updates[k]
        np.testing.assert_array_equal(ix,[v['effective_from_native_index'] for v in published])
        update_difference=np.max(abs(values-np.array([v['bias_xyz_m_s2'] for v in published])))
        assert update_difference<1e-9 and int(mask[ix[0]:].sum())==int(r['active_scored_frames'])
        # Opposite known poses cancel the true force without supplying a target to the online estimator.
        acceleration=signal[:,:3]
        pose_means=[]
        for axis in (0,1,2):
            for sign in (-1,1):
                chosen=mask & (up[:,axis]*sign>.99)
                pose_means.append(np.median(acceleration[chosen],axis=0))
        bench_bias=np.mean(pose_means,axis=0)
        physical.append({'record':k[1],'updates':len(ix),'first_time_s':float(ix[0]/hz),'max_trace_error_m_s2':float(update_difference),
                         'inferred_bias':values[-1].tolist(),'six_pose_opposition_center':bench_bias.tolist(),
                         'distance_to_opposition_center_m_s2':float(np.linalg.norm(values[-1]-bench_bias))})
        if k[1] in (arrays[0]['record'],arrays[-1]['record']):
            available=np.searchsorted(ix,np.arange(len(signal)),side='right')
            corrected=signal.copy();corrected[:,:3]-=np.vstack((np.zeros(3),values))[available]
            out=score(corrected,hz,up,mask,{'mamba':head},{'mamba':{'gain025':.25}},1.,{'mamba'})
            discrepancy=max(abs(out['rmse_deg'][a]-float(r[k])) for a,k in [('full_vqf','debiased_full_vqf_deg'),('gain025','debiased_b6_gain_025_deg')])
            assert discrepancy<1e-6
            anchors.append({'record':r['record'],'rmse_deg':out['rmse_deg'],'independent_max_delta_deg':discrepancy})
    report={'reviewed_rows':len(rows),'unchanged_inactive_non_array':len(rows)-len(physical),
            'array_active_frames':sum(int(r['active_scored_frames']) for r in rows),'array_raw_update_replay':physical,
            'native_score_anchors':anchors,'source_results_sha256':hashlib.sha256((ROOT/'C003/results.json').read_bytes()).hexdigest()}
    (ROOT/'review/C003_audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print(len(rows),len(physical),sum(v['updates'] for v in physical),
          max(r['max_trace_error_m_s2'] for r in physical),max(r['independent_max_delta_deg'] for r in anchors),
          np.median([r['distance_to_opposition_center_m_s2'] for r in physical]))


if __name__=='__main__':main()

