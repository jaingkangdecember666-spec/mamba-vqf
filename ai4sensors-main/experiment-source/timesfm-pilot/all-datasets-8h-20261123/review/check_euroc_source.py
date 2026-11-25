"""Check external EuRoC training-source units and paired reference geometry without model outputs."""
import json
import numpy as np
from scipy.spatial.transform import Rotation
from prepare_euroc_source import OUT,ROOM1,h5py
from mount_check import angle


def load_source(name):
    with h5py.File(OUT/(name+'.hdf5'),'r') as f:
        a=np.column_stack([f['acc_'+k][...] for k in 'xyz']).astype(float)
        w=np.column_stack([f['gyr_'+k][...] for k in 'xyz']).astype(float)
        q=np.column_stack([f['opt_'+k][...] for k in 'abcd']).astype(float)
        dt=f['dt'][...].reshape(-1)
        mask=f['movement_mask'][...].astype(bool).reshape(-1)
    return a,w,q,dt,mask


def main():
    records=[]
    for name in ROOM1:
        a,w,q,dt,mask=load_source(name)
        quality=np.isfinite(np.c_[q,a,w,dt]).all(1)&(np.linalg.norm(q,axis=1)>.5)
        quat=np.tile([1.,0.,0.,0.],(len(a),1));quat[quality]=q[quality]
        orientation=Rotation.from_quat(quat,scalar_first=True)
        up=orientation.inv().apply([0.,0.,1.])
        still=quality&(np.linalg.norm(w,axis=1)<.10)&(abs(np.linalg.norm(a,axis=1)-9.80665)<.30)
        level=angle(a[still],up[still])
        hop=10
        x=(orientation[:-hop].inv()*orientation[hop:]).as_rotvec()/(dt[hop:,None]*hop)
        summed=np.vstack([np.zeros(3),np.cumsum(w,axis=0)])
        avg=(summed[1+hop:]-summed[1:-hop])/hop
        use=quality[hop:]&quality[:-hop]&(np.linalg.norm(x,axis=1)>.20)&(np.linalg.norm(avg,axis=1)>.20)
        ang=angle(x[use],avg[use]); record={'record':name,'rows':len(a),'finite_with_pose':int(quality.sum()),
            'declared_dt_min_max':list(map(float,(np.min(dt),np.max(dt)))),'movement_rows':int(mask.sum()),
            'static_rows':int(still.sum()),'static_gravity_median_deg':float(np.median(level)) if len(level) else None,
            'static_gravity_p90_deg':float(np.percentile(level,90)) if len(level) else None,
            'dynamic_gyro_pairs':int(use.sum()),'dynamic_gyro_vs_orientation_median_deg':float(np.median(ang)) if len(ang) else None,
            'median_acc_magnitude_m_s2':float(np.median(np.linalg.norm(a,axis=1)))}
        records.append(record)
    (OUT/'reference-quality.json').write_text(json.dumps(records,indent=2)+'\n',encoding='utf8')
    print(records)


if __name__=='__main__':main()

