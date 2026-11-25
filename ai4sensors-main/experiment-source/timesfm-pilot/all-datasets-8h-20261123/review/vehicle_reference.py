"""Vehicle optical/navigation reference preparation. The development check uses synthetic motion only."""
import json
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from mount_check import angle, normalized, paired_vectors
from optical_reference import fit_rotation
from optical_translation import fit_mount, level_check

Z=np.array([0.,0.,1.])
RAW_FRD_TO_FLU=np.diag([1.,-1.,-1.])
NED_TO_ENU=Rotation.from_matrix([[0.,1.,0.],[1.,0.,0.],[0.,0.,-1.]])


def reference_clock(t,nav):
    clock=nav[:,1]
    good=np.isfinite(nav[:,[1,5,6,7,8,9,10]]).all(1)
    tc=clock[good]
    if len(tc)<2 or not np.all(np.diff(tc)>0):
        raise ValueError('Nonmonotone navigation clock or missing navigation solution')
    rr=nav[good,8:11]
    attitude=NED_TO_ENU * Rotation.from_euler('ZYX',rr[:,[2,1,0]],degrees=True)
    j=np.searchsorted(tc,t,side='right')-1
    inside=(j>=0)&(j+1<len(tc));j=np.clip(j,0,len(tc)-2)
    ok=inside & ((tc[j+1]-tc[j])<=.025)
    q=np.full((len(t),4),np.nan);velocity=np.full((len(t),3),np.nan)
    q[ok]=Slerp(tc-tc[0],attitude)(t[ok]-tc[0]).as_quat()
    vel=NED_TO_ENU.apply(nav[good,5:8]);a=(t[ok]-tc[j[ok]])/(tc[j[ok]+1]-tc[j[ok]])
    velocity[ok]=vel[j[ok]]*(1.-a[:,None])+vel[j[ok]+1]*a[:,None]
    return q,velocity,ok


def velocity_windows(t,acc,q,vel,window,dt):
    """Compute force/rotation integrals and navigation velocity increments on nonoverlapping 2 s windows."""
    span=max(2,round(2./dt))
    idx=np.flatnonzero(window);pieces=np.split(idx,np.where(np.diff(idx)!=1)[0]+1)
    matrices=[]; acceleration=[]
    for part in pieces:
        if len(part)<=span:continue
        for j in range(part[0],part[-1]-span+1,span):
            end=j+span
            elapsed=t[end]-t[j]
            if not 1.90<=elapsed<=2.10:continue
            qt=q[j:end+1];af=acc[j:end+1];vv=vel[[j,end]]
            if not np.isfinite(qt).all() or not np.isfinite(af).all() or not np.isfinite(vv).all():continue
            weight=np.diff(t[j:end+1])/elapsed;weights=np.r_[weight[0],weight[:-1]+weight[1:],weight[-1]]*.5
            m=np.einsum('n,nij,nk->ijk',weights,Rotation.from_quat(qt).as_matrix(),af)
            matrices.append(m);acceleration.append((vv[1]-vv[0])/elapsed)
    return np.asarray(matrices).reshape(-1,3,3,3),np.asarray(acceleration).reshape(-1,3)


def prepare_vehicle(raw,nav,common_start,fit_seconds=180.,check_seconds=60.,nominal_hz=200.):
    """No consumer prediction enters this function. Supplied common start is set once across the 16 channels."""
    t=raw[:,0]
    if len(t)<2 or not np.isfinite(t).all() or not np.all(np.diff(t)>0):
        raise ValueError('Nonmonotone native IMU clock')
    dt=1./nominal_hz
    packet_length=np.r_[dt,np.diff(t)]
    observed=np.isfinite(raw[:,1:]).all(1)&(packet_length>0)&(packet_length<=.03)
    acc=raw[:,4:]/packet_length[:,None]
    gyro=raw[:,1:4]/packet_length[:,None]
    q,vel,ref_valid=reference_clock(t,nav)
    t1=common_start+fit_seconds;t2=t1+check_seconds
    fit=(t>=common_start)&(t<t1);check=(t>=t1)&(t<t2)
    def pairs(m):
        if m.sum()<10:
            return {key:(np.empty((0,3)),np.empty((0,3)),np.empty(0,dtype=int)) for key in ('gyro','gravity')}
        return paired_vectors(t[m],acc[m],gyro[m],q[m],(ref_valid&observed)[m])
    rel=pairs(fit);mount,rank,measured=fit_rotation(rel,True)
    fit_windows=velocity_windows(t,acc,q,vel,fit&observed&ref_valid,dt)
    mount=fit_mount(mount,rel,fit_windows)
    checked=level_check(mount,velocity_windows(t,acc,q,vel,check&observed&ref_valid,dt))
    raw_axes=pairs(check)['gyro']
    gyro_err=angle(mount.apply(raw_axes[0]),raw_axes[1]) if len(raw_axes[0]) else np.array([])
    # Input conversion and reference mounting must act on the same side of the orientation.
    body_ref=np.zeros_like(acc);nominal_ref=np.zeros_like(acc)
    nav_up=Rotation.from_quat(q[ref_valid]).inv().apply(Z)
    nominal_ref[ref_valid]=nav_up@RAW_FRD_TO_FLU.T
    body_ref[ref_valid]=mount.inv().apply(nav_up)@RAW_FRD_TO_FLU.T
    signal=np.c_[acc@RAW_FRD_TO_FLU.T,gyro@RAW_FRD_TO_FLU.T];signal[~observed]=np.nan
    use=(t>=t2)&(t-t[0]>=10.)&observed&ref_valid
    flags=[]
    if not measured or rank<.02:flags.append('weak_mount_excitation')
    if checked.get('count',0)<3 or checked.get('median_deg',180)>3:flags.append('unverified_level')
    if len(gyro_err)<6 or np.median(gyro_err)>10:flags.append('unverified_angular_check')
    diagnostics={'common_fit_start_s':float(common_start),'fit_end_s':float(t1),'check_end_s':float(t2),
                 'raw_to_reference_xyzw':mount.as_quat().tolist(),'mount_s2_over_s1':rank,
                 'calibration_velocity_windows':len(fit_windows[0]),'velocity_check':checked,
                 'gyro_check_pairs':len(gyro_err),'gyro_check_median_deg':float(np.median(gyro_err)) if len(gyro_err) else None,
                 'native_frames':len(t),'scored_frames':int(use.sum()),'reference_flags':flags}
    return signal,nominal_hz,body_ref,use,nominal_ref,diagnostics


def synthetic():
    d=.005;t=np.arange(60000)*d;physics=np.array([.8,1.1,1.4])
    rpy=np.c_[.08*np.sin(t*.75),.10*np.sin(t*.95),.5*np.sin(t*.36)]
    R=Rotation.from_euler('ZYX',rpy[:,[2,1,0]])
    mount=Rotation.from_rotvec([.05,-.06,.12])
    imu=R*mount
    vel_ned=.3*np.cos(t[:,None]*physics)
    accel_ned=-.3*physics*np.sin(t[:,None]*physics)
    force=imu.inv().apply(accel_ned+np.array([0.,0.,-9.80665]))
    w=np.zeros_like(force);w[1:]=(imu[:-1].inv()*imu[1:]).as_rotvec()/d;w[0]=w[1]
    raw=np.c_[t,w*d,force*d];ii=np.arange(0,len(t),2)
    nav=np.zeros((len(ii),11));nav[:,1]=t[ii];nav[:,5:8]=vel_ned[ii];nav[:,8:11]=np.rad2deg(rpy[ii])
    _,_,ref,ok,nominal,log=prepare_vehicle(raw,nav,0.)
    ideal=imu.inv().apply([0.,0.,-1.])@RAW_FRD_TO_FLU.T
    delta=angle(ideal[ok],ref[ok]);nomerr=angle(ideal[ok],nominal[ok])
    assert np.max(delta)<.2 and np.median(nomerr)>3 and not ok[t<240].any()
    # Modifying later navigation references cannot change the fitted mounting transform.
    changed=nav.copy();changed[nav[:,1]>245,8]+=12.
    *_, log2=prepare_vehicle(raw,changed,0.)
    np.testing.assert_allclose(log['raw_to_reference_xyzw'],log2['raw_to_reference_xyzw'],rtol=0.,atol=0.)
    out={'synthetic_only':True,'tilt_p95_deg':float(np.quantile(delta,.95)),'tilt_max_deg':float(np.max(delta)),
         'nominal_uncalibrated_median_deg':float(np.median(nomerr)),'fit_invariant_to_changed_evaluation_truth':True,**log}
    return out


if __name__=='__main__':
    from pathlib import Path
    r=synthetic()
    Path(__file__).with_name('vehicle_synthetic_audit.json').write_text(json.dumps(r,indent=2)+'\n',encoding='utf8')
    print(r)

