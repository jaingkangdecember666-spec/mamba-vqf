"""Check IMU-to-marker mounting on a designated calibration interval, independently of any predictor."""
import numpy as np
from scipy.spatial.transform import Rotation


def angle(a,b):
    return np.rad2deg(np.arctan2(np.linalg.norm(np.cross(a,b),axis=-1),np.sum(a*b,axis=-1)))


def normalized(a):
    return a / np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1.e-12)


def paired_vectors(time,accel,gyro,quat_xyzw,quality,world_up=(0.,0.,1.)):
    """Independent vector pairs for Wahba's problem. Output time indices never cross a bad quaternion."""
    t=np.asarray(time,dtype=float)
    quality=quality & np.isfinite(quat_xyzw).all(axis=1)&(np.linalg.norm(quat_xyzw,axis=1)>.5)
    quality &= np.isfinite(gyro).all(axis=1)&np.isfinite(accel).all(axis=1)
    good=np.flatnonzero(quality)
    q=Rotation.identity(len(t)).as_quat()
    q[good]=Rotation.from_quat(quat_xyzw[good]).as_quat()
    q=Rotation.from_quat(q)
    dt=np.median(np.diff(t))
    hop=max(1,round(.06/dt))
    start=np.arange(0,len(t)-hop,hop)
    stop=start+hop
    bad=np.r_[0,np.cumsum(~quality)]
    observed=(bad[stop+1]==bad[start]) & ((t[stop]-t[start])>.75*hop*dt) & ((t[stop]-t[start])<1.25*hop*dt)
    start,stop=start[observed],stop[observed]
    gw=(q[start].inv()*q[stop]).as_rotvec()/(t[stop]-t[start])[:,None]
    cumulative=np.vstack([np.zeros(3),np.cumsum(np.nan_to_num(gyro),axis=0)])
    iw=(cumulative[stop+1]-cumulative[start+1])/hop
    motion=(np.linalg.norm(iw,axis=1)>.22)&(np.linalg.norm(gw,axis=1)>.22)
    quiet=(np.linalg.norm(gyro[good],axis=1)<.12) & (np.abs(np.linalg.norm(accel[good],axis=1)-9.80665)<.4)
    static_ids=good[quiet][::max(1,round(.20/dt))]
    return {'gyro':(normalized(iw[motion]),normalized(gw[motion]),start[motion]),
            'gravity':(normalized(accel[static_ids]),q[static_ids].inv().apply(world_up),static_ids)}


def fit_from_pairs(pairs):
    parts=[(a,b) for a,b,_ in pairs.values() if len(a)]
    if not parts:
        raise ValueError('Calibration interval yielded no vector pairs')
    x,y=np.concatenate([p[0] for p in parts]),np.concatenate([p[1] for p in parts])
    family=np.concatenate([np.full(len(a),1/len(a)) for a,b in parts])
    w=family.copy()
    for _ in range(6):
        u,singular,vt=np.linalg.svd((y*w[:,None]).T@x)
        rotation=Rotation.from_matrix(u@np.diag([1,1,np.linalg.det(u@vt)])@vt)
        residual=angle(rotation.apply(x),y)
        w=family*np.minimum(1.,5./np.maximum(.001,residual))
    metrics={}
    for source,(a,b,_) in pairs.items():
        err=angle(rotation.apply(a),b) if len(a) else []
        metrics[source]={'pairs':len(err),'median_deg':float(np.median(err)) if len(err) else None,
                         'p90_deg':float(np.percentile(err,90)) if len(err) else None}
    return rotation,{'singular_values':singular.tolist(),'second_over_first':float(singular[1]/max(1e-12,singular[0])),
                     'rotation_xyzw':rotation.as_quat().tolist(),'calibration_residual':metrics}


def verify_synthetic():
    rng=np.random.default_rng(73631)
    true=Rotation.from_rotvec([.24,-.83,1.24])
    x=normalized(rng.normal(size=(50,3)))
    noise=Rotation.from_rotvec(rng.normal(size=(len(x),3))*1e-5)
    y=noise.apply(true.apply(x))
    result,audit=fit_from_pairs({'gyro':(x[:30],y[:30],np.arange(30)), 'gravity':(x[30:],y[30:],np.arange(20))})
    delta=np.rad2deg((result.inv()*true).magnitude())
    assert delta<.001 and audit['second_over_first']>.4
    return float(delta)


if __name__=='__main__':
    print(verify_synthetic())
