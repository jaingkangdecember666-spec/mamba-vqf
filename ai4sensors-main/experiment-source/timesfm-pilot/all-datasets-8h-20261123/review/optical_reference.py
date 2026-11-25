"""Candidate-independent optical geometry for Oxford phones and RepoIMU rigs."""
import json
import numpy as np
from scipy.spatial.transform import Rotation
from check_mount_pilots import ROOT, source
from check_oxiod_timing import speeds
from mount_check import normalized, paired_vectors, angle
from optical_translation import integration_windows, fit_mount, level_check

Z = np.array([0.,0.,1.])
X = np.array([1.,0.,0.])


def time_fit(w,q,dt):
    """Positive lag maps IMU j to Vicon j-lag; use this partition only."""
    if len(w)<120:
        return {'lag':0,'peak':None,'at_zero':None}
    a,b=speeds(w,q,dt); half=min(round(1./dt),len(w)//8)
    idx=np.arange(half+7,len(w)-half-7,2)
    idx=idx[np.isfinite(b[idx]) & (b[idx]<15)]
    scores={}
    for k in range(-half,half+1):
        x=a[idx+k];y=b[idx]
        valid=np.isfinite(x) & (x<15)
        if valid.sum()<45 or min(np.std(x[valid]),np.std(y[valid]))<.05:
            continue
        scores[k]=float(np.corrcoef(x[valid],y[valid])[0,1])
    if not scores:
        return {'lag':0,'peak':None,'at_zero':None}
    k=max(scores,key=scores.get)
    return {'lag':k,'peak':scores[k],'at_zero':scores.get(0),'border':abs(k)==half}


def apply_lag(q,delta):
    i=np.arange(len(q))-delta
    valid=(i>=0)&(i<len(q))
    shifted=np.full(q.shape,np.nan)
    shifted[valid]=q[i[valid]]
    return shifted


def fit_rotation(pairs,include_gravity):
    """Robust Wahba with a small identity prior for unobserved rotation axes."""
    a,b,weight=[],[],[]
    for name,(src,dst,_) in pairs.items():
        if len(src)<3 or (name=='gravity' and not include_gravity):continue
        a.append(src); b.append(dst); weight.append(np.full(len(src),1./len(src)))
    if not a:
        return Rotation.identity(),0.,False
    a=np.concatenate(a);b=np.concatenate(b);nominal=np.concatenate(weight)
    w=nominal
    for _ in range(8):
        h=(b*w[:,None]).T@a
        u,sv,vt=np.linalg.svd(h+np.eye(3)*.0005)
        fitted=Rotation.from_matrix(u@np.diag([1,1,np.linalg.det(u@vt)])@vt)
        err=angle(fitted.apply(a),b)
        w=nominal*np.minimum(1.,5./np.maximum(.001,err))
    sv=np.linalg.svd(h,compute_uv=False)
    return fitted,float(sv[1]/max(sv[0],1e-12)),True


def still_world(t,a,w,q,mount):
    """One direction per strictly low-motion 0.25 s block."""
    dt=np.median(np.diff(t)); width=max(3,round(.25/dt))
    world=[]
    for k in range(0,len(a)-width+1,width):
        ix=slice(k,k+width)
        obs=np.c_[a[ix],w[ix],q[ix]]
        if not np.isfinite(obs).all() or (np.linalg.norm(q[ix],axis=1)<.5).any():
            continue
        if (np.linalg.norm(w[ix],axis=1)>.18).any() or (abs(np.linalg.norm(a[ix],axis=1)-9.80665)>.45).any():
            continue
        v=Rotation.from_quat(q[ix]).apply(mount.apply(normalized(a[ix])))
        world.append(normalized(np.mean(v,axis=0)[None,:])[0])
    return np.asarray(world).reshape(-1,3)


def extract(row):
    """Return full native input, declared rate, calibrated reference and suffix mask."""
    t,a,w,q=source(row); group=row['group']
    duration=float(t[-1]-t[0])
    fit_s=min(60.,duration*.25); check_s=min(fit_s*1.6,duration*.40)
    fit=(t>=0)&(t<fit_s); check=(t>=fit_s)&(t<check_s)
    if min(fit.sum(),check.sum())<12: raise ValueError('Insufficient calibration or verification span')
    dt=float(np.median(np.diff(t)))
    ox=row['dataset']=='OxIOD'; pendulum=group.startswith('Pendulum/')
    tf=tc=None;lag=0
    if ox:
        tf=time_fit(w[fit],q[fit],dt);tc=time_fit(w[check],q[check],dt)
        if tf['peak'] is not None and tf['peak']>=.55 and not tf['border']:
            lag=tf['lag']
    q=apply_lag(q,lag)
    native_index=np.arange(len(t))
    optical_index=native_index-lag
    fit &= (optical_index>=0) & (optical_index<np.searchsorted(t,fit_s))
    check &= (optical_index>=np.searchsorted(t,fit_s)) & (optical_index<np.searchsorted(t,check_s))
    def observations(idx):
        return paired_vectors(t[idx],a[idx],w[idx],q[idx],np.ones(idx.sum(),bool))
    pair=observations(fit)
    mount,rank,has_fit=fit_rotation(pair,not pendulum)
    dynamic_check=None
    if ox:
        position=np.loadtxt(row['reference_path'],delimiter=',',usecols=(2,3,4),ndmin=2)
        position=apply_lag(position,lag)
        training_windows=integration_windows(fit,q,position,a,dt)
        mount=fit_mount(mount,pair,training_windows)
        dynamic_check=level_check(mount,integration_windows(check,q,position,a,dt))
    axis=Z;axis_kind='optical_world_Z'
    if pendulum:
        levels=still_world(t[fit],a[fit],w[fit],q[fit],mount)
        if len(levels):
            axis=normalized(np.median(levels,axis=0)[None,:])[0]
            axis_kind='calibrated_from_prefix_specific_force'
        else:
            axis=X;axis_kind='fixture_X_fallback_unverified'
    held=observations(check);v1,v2,_=held['gyro']
    gyro_err=angle(mount.apply(v1),v2) if len(v1) else np.array([])
    levels=still_world(t[check],a[check],w[check],q[check],mount)
    level_err=angle(levels,axis) if len(levels) else np.array([])
    gmed=float(np.median(gyro_err)) if len(gyro_err) else None
    amed=float(np.median(level_err)) if len(level_err) else None
    issues=[]
    if rank<.02 or not has_fit:issues.append('weak_mount_excitation')
    if len(gyro_err)<6 or gmed>12:issues.append('gyro_check_unverified')
    quiet_ok=len(level_err)>=2 and amed<=6
    moving_ok=bool(dynamic_check and dynamic_check['count']>=3 and dynamic_check['median_deg']<=6 and dynamic_check['median_force_residual_m_s2']<=1.2)
    if not (quiet_ok or moving_ok):issues.append('level_check_unverified')
    if ox and (tc['peak'] is None or tc.get('border') or tc['peak']<.55 or abs(tc['lag']-lag)>4):
        issues.append('timing_check_unverified')
    if pendulum:issues.append('relative_level_and_repeated_samples')
    good=np.isfinite(q).all(1)&(np.linalg.norm(q,axis=1)>.5)&(np.linalg.norm(q,axis=1)<1.5)
    ref=np.zeros_like(a)
    ref[good]=mount.inv().apply(Rotation.from_quat(q[good]).inv().apply(axis))
    # Guard the reference as well as the IMU. Corrected optical times cannot re-enter the fit/check prefix.
    used=good & (t>=max(10.,check_s)) & ((np.arange(len(t))-lag)>=np.searchsorted(t,check_s)) & np.isfinite(np.c_[a,w]).all(1)
    obs={'dataset':row['dataset'],'record':row['record'],'group':group,'hz_declared_from_clock':1/dt,
         'duration_s':duration,'fit_through_s':fit_s,'check_through_s':check_s,'fit_lag':tf,'check_lag':tc,
         'applied_imu_minus_optical_rows':lag,'mount_xyzw':mount.as_quat().tolist(),'mount_second_over_first':rank,
         'world_up':axis.tolist(),'world_up_source':axis_kind,'gyro_check_count':len(gyro_err),
         'optical_translation_check':dynamic_check,
         'gyro_check_median_deg':gmed,'level_check_count':len(level_err),'level_check_median_deg':amed,
         'native_frames':len(t),'scoring_frames':int(used.sum()),'quality_flags':issues,
         'measured_distinct_acc_hz':float((np.diff(a,axis=0)!=0).any(1).sum()/duration),
         'measured_distinct_gyr_hz':float((np.diff(w,axis=0)!=0).any(1).sum()/duration)}
    return np.c_[a,w],1/dt,ref,used,obs


def main():
    inv=json.loads((ROOT/'C001/inventory.json').read_text('utf8'))
    chosen=[r for r in inv['records'] if r['side']=='development' and r['dataset'] in ('OxIOD','RepoIMU')]
    out=[]
    for r in chosen:
        try:
            *_,info=extract(r);out.append(info)
        except (ValueError,OSError,IndexError) as exc:
            out.append({'dataset':r['dataset'],'group':r['group'],'record':r['record'],'error':str(exc)})
    (ROOT/'review/optical_development_calibration.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
    for d in ('OxIOD','RepoIMU'):
        a=[s for s in out if s['dataset']==d]
        print(d,len(a),sum(bool(s.get('scoring_frames')) for s in a),sum(s.get('quality_flags')==[] for s in a),
              [(s['record'],s.get('error')) for s in a if 'error' in s])


if __name__=='__main__':main()

