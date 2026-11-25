"""Cross-check the hinge's unobservable rotation using long, disjoint development calibration trials."""
import json
import numpy as np
from scipy.spatial.transform import Rotation
from check_mount_pilots import ROOT,source
from check_repo_world import low_motion_world
from mount_check import angle,normalized,paired_vectors
from optical_reference import still_world


def fit_hinge(xs,ys):
    """For essentially one-axis motion use the minimum-angle rotation mapping the two measured hinge axes."""
    A=np.concatenate(xs,axis=0);B=np.concatenate(ys,axis=0)
    u,s,vt=np.linalg.svd(A.T@A)
    ua=u[:,0]
    v,ss,vs=np.linalg.svd(B.T@B)
    ub=v[:,0]
    if np.dot(A@ua,B@ub)<0:ub=-ub
    result,_=Rotation.align_vectors(ub[None,:],ua[None,:])
    return result,{'gyro_axis_imu':ua.tolist(),'gyro_axis_marker':ub.tolist(),'imu_secondary_excitation_ratio':float(s[1]/s[0]),
                   'marker_secondary_excitation_ratio':float(ss[1]/ss[0]),'mount_xyzw':result.as_quat().tolist(),
                   'about_unobserved_axis':'minimum proper rotation; the angle about the hinge is unresolved'}


def build():
    inventory=json.loads((ROOT/'C001/inventory.json').read_text('utf8'))
    rows=[r for r in inventory['records'] if r['side']=='development' and r['dataset']=='RepoIMU' and r['group'].startswith('Pendulum/')]
    v1={r['record']:r for r in json.loads((ROOT/'review/optical_development_calibration.json').read_text('utf8'))}
    solution={};reports={}
    for seg in (1,2,3):
        donors=[r for r in rows if r['group']=='Pendulum/Test02' and r['record'].endswith('Segment_'+str(seg))]
        assert len(donors)==2
        vs,ws=[],[]
        for d in donors:
            t,a,w,q=source(d);pairs=paired_vectors(t,a,w,q,np.ones(len(t),bool));x,y,_=pairs['gyro']
            # Equal representation of each donor recording.
            take=np.linspace(0,len(x)-1,min(400,len(x)),dtype=int);vs.append(x[take]);ws.append(y[take])
        C,meta=fit_hinge(vs,ws)
        levels=[]
        for d in donors:
            t,a,w,q=source(d); lv=still_world(t,a,w,q,C)
            if len(lv):levels.append(normalized(np.median(lv,axis=0)[None,:])[0])
        h=normalized(np.median(levels,axis=0)[None,:])[0]
        solution[seg]=(C,h);meta.update({'donors':[r['record'] for r in donors],'donor_quiet_up':h.tolist()})
        reports[str(seg)]=meta
    checks=[]
    for r in rows:
        seg=int(r['record'].split('_')[-1]);C,h=solution[seg]
        s=v1[r['record']];t,a,w,q=source(r);fit=(t<s['fit_through_s']);check=(t>=s['fit_through_s'])
        # Validation starts after the per-record fit boundary and never supplies its level.
        lv=still_world(t[fit],a[fit],w[fit],q[fit],C);held=still_world(t[check],a[check],w[check],q[check],C)
        level=normalized(np.median(lv,axis=0)[None,:])[0] if len(lv) else h
        prior_C=Rotation.from_quat(s['mount_xyzw']);prior_held=still_world(t[check],a[check],w[check],q[check],prior_C)
        old_angle=angle(prior_held,np.array(s['world_up']));new_angle=angle(held,level)
        pp=paired_vectors(t[check],a[check],w[check],q[check],np.ones(check.sum(),bool))
        x,y,_=pp['gyro']
        checks.append({'record':r['record'],'group':r['group'],'is_calibration_group':r['group']=='Pendulum/Test02','valid_quiet_blocks':len(held),
                       'prior_quiet_median_deg':float(np.median(old_angle)) if len(old_angle) else None,
                       'new_quiet_median_deg':float(np.median(new_angle)) if len(new_angle) else None,
                       'prior_gyro_median_deg':float(np.median(angle(prior_C.apply(x),y))) if len(x) else None,
                       'new_gyro_median_deg':float(np.median(angle(C.apply(x),y))) if len(x) else None,
                       'prefix_level_measured':bool(len(lv)),'separate_reference_prefix_world_up':level.tolist()})
    return {'source':'Development only; two complete long donor trials per segment, candidate-free geometric assessment. Existing C005 remains untouched.',
            'segments':reports,'dev_checks':checks}


if __name__=='__main__':
    result=build()
    (ROOT/'review/repo_fixture_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    a=[r for r in result['dev_checks'] if not r['is_calibration_group'] and r['valid_quiet_blocks']]
    print('independent',len(a))
    for k in ['prior_quiet_median_deg','new_quiet_median_deg','prior_gyro_median_deg','new_gyro_median_deg']:
        values=[r[k] for r in a if r[k] is not None]
        print(k,np.percentile(values,[50,90,100]))

