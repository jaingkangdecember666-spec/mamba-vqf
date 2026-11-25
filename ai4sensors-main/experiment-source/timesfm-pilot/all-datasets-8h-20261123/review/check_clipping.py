"""Use reference-only development diagnostics to audit fast gyro clipping and a causal acceleration cue."""
import json
import numpy as np
from scipy.spatial.transform import Rotation
from check_mount_pilots import ROOT,source
from mount_check import normalized


def clues(acc,gyro,hz):
    n=len(gyro);hop=max(2,round(.05*hz))
    ids=np.arange(hop,n)
    x=normalized(acc)
    old=x[ids-hop];new=x[ids];cross=np.cross(new,old)
    sine=np.linalg.norm(cross,axis=1)
    theta=np.arctan2(sine,np.einsum('ij,ij->i',new,old))
    implied=cross*(theta/(np.maximum(sine,1e-9)*(hop/hz)))[:,None]
    # Local average points along the physical rotation axis, independent of the sensor's mounting.
    past=np.lib.stride_tricks.sliding_window_view(gyro,hop+1,axis=0)
    avg=past.mean(2);axis=normalized(avg);rate=np.linalg.norm(avg,axis=1)
    perp=1.-np.einsum('ij,ij->i',axis,new)**2
    inferred=np.einsum('ij,ij->i',implied,axis)/np.maximum(perp,.05)
    spread=np.ptp(np.linalg.norm(past,axis=1),axis=1)
    const=(rate>4.)&(spread<.025*rate)
    force=np.linalg.norm(acc,axis=1)
    consistent=(perp>.45)&(abs(force[ids]-9.80665)<1.20)&(abs(force[ids-hop]-9.80665)<1.20)
    use=const&consistent&(inferred>rate+1.)&(inferred<2.5*rate)
    return ids,hop,rate,axis,inferred,use,const,perp


def analyze(row,C):
    t,a,w,q=source(row)
    dt=float(np.median(np.diff(t)))
    ids,hop,rate,axis,inferred,use,flat,perp=clues(a,w,1./dt)
    good=np.isfinite(q).all(1)&(np.linalg.norm(q,axis=1)>.5)
    q[~good]=[0,0,0,1]
    omc=Rotation.from_quat(q)
    delta=(omc[ids-hop].inv()*omc[ids]).as_rotvec()/(t[ids]-t[ids-hop])[:,None]
    true=np.einsum('ij,ij->i',C.inv().apply(delta),axis)
    valid=good[ids]&good[ids-hop]&np.isfinite(true)
    chosen=use&valid
    def metrics(mask):
        if not np.any(mask):return None
        r,actual,cue=rate[mask],true[mask],inferred[mask]
        return {'frames':int(mask.sum()),'observed_rate_median':float(np.median(r)),'reference_rate_median':float(np.median(actual)),
                'acceleration_rate_median':float(np.median(cue)),'baseline_rate_mae':float(np.mean(abs(r-actual))),
                'causal_cue_rate_mae':float(np.mean(abs(cue-actual))),
                'reference_exceeds_report_by_1':float(np.mean(actual>r+1.))}
    return {'record':row['record'],'group':row['group'],'frames':len(t),'native_hz':1/dt,
            'high_constant':metrics(valid&flat),'physically_qualified':metrics(chosen),
            'max_gyro_norm':float(np.max(np.linalg.norm(w,axis=1))),
            'reference_speed_p99':float(np.quantile(np.linalg.norm(delta[valid],axis=1),.99))}


def main():
    inventory=json.loads((ROOT/'C001/inventory.json').read_text('utf8'))
    mounts={r['record']:Rotation.from_quat(r['mount_xyzw']) for r in json.loads((ROOT/'review/optical_development_calibration.json').read_text('utf8'))}
    rows=[r for r in inventory['records'] if r['side']=='development' and r['dataset']=='RepoIMU' and r['group'].startswith('TStick/')]
    out=[analyze(r,mounts[r['record']]) for r in rows]
    (ROOT/'review/repo_clipping_diagnosis.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
    print([(r['record'],r['physically_qualified']) for r in out if r['physically_qualified']])


if __name__=='__main__':main()

