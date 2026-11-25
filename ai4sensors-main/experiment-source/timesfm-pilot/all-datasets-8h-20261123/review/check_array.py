"""Read the calibration trial of IMU-Array, independently of the sealed vehicle drive."""
import hashlib
import json
from pathlib import Path
import numpy as np
from check_sources import DATA, ROOT, TRANSFER, measured, weights


def base_inputs(record, ref_all):
    raw=np.memmap(record['path'],dtype='<f8',mode='r').reshape(-1,7)
    t=raw[:,0]
    n=len(t)
    # Map onto the predeclared common time axis. The table's six static positions have known specific force.
    ref_t=ref_all[:,0]
    j=np.searchsorted(ref_t,t,side='left').clip(0,len(ref_t)-1)
    left=np.maximum(0,j-1)
    j=np.where(np.abs(t-ref_t[left])<=np.abs(t-ref_t[j]),left,j)
    delta=np.abs(t-ref_t[j]); dt=np.median(np.diff(t))
    exact=delta<1e-7
    ref=np.asarray(ref_all[j,4:],dtype=np.float64)
    length=np.linalg.norm(ref,axis=1)
    finite=np.isfinite(length)&(length>1)
    u=np.zeros_like(ref)
    u[finite]=ref[finite]/length[finite,None]
    a=raw[:,4:]
    w=np.deg2rad(raw[:,1:4])
    valid=finite & exact & (t-t[0]>=10.) & np.isfinite(np.c_[a,w]).all(1)
    axis=np.sign(u)*np.round(np.abs(u))
    names,ns=np.unique(axis[valid].astype(int),axis=0,return_counts=True)
    stats={'name':record['record'],'frames':n,'dt_s':float(dt),'identical_native_clock_rows':int(exact.sum()),
           'eligible_frames':int(valid.sum()),'sha256':hashlib.sha256(np.flatnonzero(valid).astype('<i8').tobytes()).hexdigest(),
           'known_static_pose_frame_counts':{str(k.tolist()):int(v) for k,v in zip(names,ns)}}
    return (np.c_[a,w],1/dt,u,valid),stats


def main():
    import torch
    torch.set_num_threads(2)
    inv=json.loads((ROOT/'C001/inventory.json').read_text('utf8'))
    dev=[r for r in inv['records'] if r['side']=='development' and r['dataset']=='IMU-Array']
    assert len(dev)==16 and all('Calibration' in r['path'] for r in dev)
    table=np.memmap(DATA/'IMU-Array-main/Calibration/Group1_ref_data.bin',dtype='<f8',mode='r').reshape(-1,7)
    head=weights(TRANSFER/'batch6/mamba-intrinsic-selected.pt','mamba_v1')
    rows=[]
    for i,r in enumerate(dev):
        source,row=base_inputs(r,table)
        if i in (0,len(dev)-1) and row['eligible_frames']:
            row['scores']=measured(source,head)
        rows.append(row)
    (ROOT/'review/array_calibration_audit.json').write_text(json.dumps(rows,indent=2)+'\n',encoding='utf8')
    for r in rows:
        print(r['name'],r['identical_native_clock_rows'],r['eligible_frames'],r.get('scores',{}).get('rmse_deg'))


if __name__=='__main__':
    main()
