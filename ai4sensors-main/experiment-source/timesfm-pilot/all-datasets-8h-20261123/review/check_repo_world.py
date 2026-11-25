"""Check what the motion-capture file calls vertical using development-only low-motion gravity, without scoring a predictor."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from check_mount_pilots import source, ROOT
from mount_check import angle, normalized


def low_motion_world(row):
    t,a,w,q=source(row)
    eligible=np.isfinite(np.column_stack((a,w,q))).all(1)&(np.linalg.norm(q,axis=1)>.5)
    dt=np.median(np.diff(t)); width=max(4,round(.5/dt)); vectors=[]; times=[]
    for j in range(0,len(t)-width+1,width):
        sl=slice(j,j+width)
        if not eligible[sl].all(): continue
        if np.max(np.linalg.norm(w[sl],axis=1))>.15 or np.max(abs(np.linalg.norm(a[sl],axis=1)-9.80665))>.40: continue
        n=Rotation.from_quat(q[sl]).apply(normalized(a[sl]))
        vectors.append(normalized(n.mean(0)[None,:])[0]); times.append(t[j])
    m=np.array(vectors).reshape(-1,3)
    return m,np.array(times),float(t[-1]-t[0])


def main():
    data=json.loads((ROOT/'C001/inventory.json').read_text('utf8'))
    available=[r for r in data['records'] if r['dataset']=='RepoIMU' and r['side']=='development']
    out=[]; train=[]; verify=[]
    for r in available:
        m,tm,duration=low_motion_world(r)
        if len(m):
            n=normalized(np.median(m,axis=0)[None,:])[0]
            angles={name:float(np.median(angle(m,vec))) for name,vec in [('x',np.array([1,0,0])),('z',np.array([0,0,1]))]}
            spread=float(np.median(angle(m,n)))
        else: n=np.zeros(3); angles={}; spread=None
        out.append({'name':r['record'],'group':r['group'],'low_motion_blocks':len(m),'duration_s':duration,
                    'median_world_direction_identity_mount':n.tolist(),'angle_to_axes_deg':angles,'within_record_median_spread_deg':spread})
        if len(m) and r['group'].startswith('Pendulum/'):
            if r['group']=='Pendulum/Test02': train.append(n)
            else: verify.append((r['record'],n))
    center=normalized(np.median(train,axis=0)[None,:])[0]
    differences=[float(angle(center,n)) for _,n in verify]
    summary={'fit_group':'Pendulum/Test02','calibration_records':len(train),'other_dev_records_with_quiet':len(differences),
             'fitted_world_direction_identity_mount':center.tolist(), 'other_dev_median_difference_deg':float(np.median(differences)),
             'other_dev_max_difference_deg':max(differences),'other_dev_q90_difference_deg':float(np.quantile(differences,.90)),
             'other_dev_direction_checks':[(key,x) for (key,_),x in zip(verify,differences)]}
    (ROOT/'review/repo_world_axis_audit.json').write_text(json.dumps({'summary':summary,'records':out},indent=2)+'\n',encoding='utf8')
    print({k:v for k,v in summary.items() if k!='other_dev_direction_checks'})
    for d in ['Pendulum','TStick']:
        r=[x for x in out if x['name'].startswith(d) and x['low_motion_blocks']];print(d,len(r),'toX/toZ',*[float(np.median([x['angle_to_axes_deg'][a] for x in r])) for a in ['x','z']])


if __name__=='__main__':
    main()
