"""Check optical vertical on moving Oxford phones using changes in Vicon velocity."""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

G=9.80665
UP=np.array([0.,0.,1.])


def integration_windows(imu_ids, q, positions, acc, dt):
    """Mean world specific force equals gravity plus the endpoint change in velocity over time."""
    span=round(2./dt); margin=max(3,round(.15/dt))
    good=np.flatnonzero(imu_ids)
    # Never bridge a missing sample or an optical tracking gap.
    pieces=np.split(good,np.where(np.diff(good)!=1)[0]+1)
    matrix=[];accel=[]
    centered=np.arange(-margin,margin+1)*dt
    slope=centered/np.sum(centered**2)
    for part in pieces:
        if len(part)<span+2*margin+1:continue
        for left in range(part[0]+margin,part[-1]-span-margin+1,span):
            right=left+span
            nearby=positions[left-margin:right+margin+1]
            if not np.isfinite(nearby).all() or np.max(np.linalg.norm(np.diff(nearby,axis=0),axis=1))>max(.12,12.*dt):
                continue
            qq=q[left:right+1]; aa=acc[left:right+1]
            if not np.isfinite(np.c_[qq,aa]).all() or np.min(np.linalg.norm(qq,axis=1))<.5:
                continue
            p0=positions[left-margin:left+margin+1]
            p1=positions[right-margin:right+margin+1]
            # Difference of local linear velocity fits, normalized by elapsed time.
            acceleration=(slope@(p1-p0))/(span*dt)
            rr=Rotation.from_quat(qq).as_matrix()
            wt=np.full(len(rr),1./span);wt[[0,-1]]*=.5
            projected=np.einsum('n,nij,nk->ijk',wt,rr,aa)
            matrix.append(projected);accel.append(acceleration)
    return np.asarray(matrix).reshape(-1,3,3,3),np.asarray(accel).reshape(-1,3)


def fit_mount(initial, pairs, windows):
    matrices,linear=windows
    if len(matrices)<3:return initial
    x,y,_=pairs['gyro']
    stride=max(1,len(x)//120);x=x[::stride];y=y[::stride]
    target=G*UP + linear
    def residual(vector):
        trial=Rotation.from_rotvec(vector)*initial
        predicted=np.einsum('nijk,jk->ni',matrices,trial.as_matrix())
        # Robust angular-direction constraints and independent integrated-force constraints.
        f=(predicted-target)/(.35*np.sqrt(len(target)))
        g=(trial.apply(x)-y)/(.10*np.sqrt(max(1,len(x))))
        return np.r_[f.ravel(),g.ravel(),vector/(.50*np.sqrt(max(1,len(target))))]
    sol=least_squares(residual,np.zeros(3),loss='soft_l1',f_scale=.25,max_nfev=30)
    return Rotation.from_rotvec(sol.x)*initial


def level_check(mount,windows):
    matrices,linear=windows
    if not len(matrices):return {'count':0}
    projected=np.einsum('nijk,jk->ni',matrices,mount.as_matrix())
    vertical=projected-linear
    angle=np.rad2deg(np.arctan2(np.linalg.norm(vertical[:,:2],axis=1),vertical[:,2]))
    residual=vertical-G*UP
    err=np.linalg.norm(residual,axis=1)
    return {'count':len(vertical),'median_deg':float(np.median(angle)),'p90_deg':float(np.quantile(angle,.9)),
            'median_force_residual_m_s2':float(np.median(err)),
            'median_without_velocity_m_s2':float(np.median(np.linalg.norm(projected-G*UP,axis=1))),
            'median_with_sign_flipped_m_s2':float(np.median(np.linalg.norm(projected+linear-G*UP,axis=1))),
            'median_inferred_g_m_s2':float(np.median(np.linalg.norm(vertical,axis=1)))}


def synthetic():
    dt=.01;t=np.arange(6000)*dt
    attitude=Rotation.from_rotvec(np.c_[.47*np.sin(t),.38*np.cos(.65*t),.52*np.sin(1.1*t)])
    C=Rotation.from_rotvec([.22,-.55,.33])
    frequencies=np.array([1.7,1.3,2.1]);amp=np.array([.8,.9,.3])
    pos=amp*np.sin(t[:,None]*frequencies)
    a_world=-(frequencies**2)*pos
    a=C.inv().apply(attitude.inv().apply(a_world+G*UP))
    measures=integration_windows(np.ones(len(t),bool),attitude.as_quat(),pos,a,dt)
    x=np.random.default_rng(8103).normal(size=(120,3));x/=np.linalg.norm(x,axis=1,keepdims=True)
    trial=fit_mount(Rotation.from_rotvec([.03,-.04,.02])*C,{'gyro':(x,C.apply(x),None)},measures)
    error=float(np.rad2deg((trial.inv()*C).magnitude()))
    quality=level_check(trial,measures)
    assert error<.05 and quality['median_deg']<.1 and quality['median_force_residual_m_s2']<.03
    return error,quality


if __name__=='__main__':
    print(synthetic())

