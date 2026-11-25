"""Resolve the per-record hinge-axis rotation ambiguity with a common fixture-level convention."""
import json
import numpy as np
from scipy.spatial.transform import Rotation
from check_mount_pilots import ROOT,source
from mount_check import angle,normalized,paired_vectors
from optical_reference import fit_rotation,still_world


def reference_fit(t,a,w,q,cut,prior_level):
    fit=t<cut
    # Long-axis approximate mounting is the second constraint for a one-axis hinge.
    # It never forces a fixed sign for the spin-axis measurement itself.
    pairs=paired_vectors(t[fit],a[fit],w[fit],q[fit],np.ones(fit.sum(),bool),world_up=prior_level)
    xx,yy,_=pairs['gravity']
    nom=np.array([1.,0.,0.])
    if len(xx)>=4:
        # A bounded independent axis prior plus observed low-motion samples in the reference prefix.
        select=np.linspace(0,len(xx)-1,min(len(xx),8),dtype=int)
        xx=np.vstack((xx[select],np.tile(nom,(3,1))))
        yy=np.vstack((yy[select],np.tile(prior_level,(3,1))))
    else:
        xx=np.tile(nom,(3,1));yy=np.tile(prior_level,(3,1))
    pairs['gravity']=(xx,yy,None)
    C,rank,has_fit=fit_rotation(pairs,True)
    levels=still_world(t[fit],a[fit],w[fit],q[fit],C)
    h=normalized(np.median(levels,axis=0)[None,:])[0] if len(levels) else prior_level
    return C,h,{'right_mount_xyzw':C.as_quat().tolist(),'world_up':h.tolist(),'prefix_quiet':len(levels),'rank_with_prior':rank}


def run():
    inventory=json.loads((ROOT/'C001/inventory.json').read_text('utf8'))
    checks=json.loads((ROOT/'review/repo_fixture_audit.json').read_text('utf8'))
    trial={j['record']:j for j in checks['dev_checks']}
    a=json.loads((ROOT/'review/repo_world_axis_audit.json').read_text('utf8'))
    prior=np.array(a['summary']['fitted_world_direction_identity_mount'])
    validation=[]
    for row in inventory['records']:
        if row['side']!='development' or not row['group'].startswith('Pendulum/'):continue
        t,ac,w,q=source(row);cut=min(60.,t[-1]*.25);use=t>=cut
        C,h,log=reference_fit(t,ac,w,q,cut,prior)
        lv=still_world(t[use],ac[use],w[use],q[use],C)
        pairs=paired_vectors(t[use],ac[use],w[use],q[use],np.ones(use.sum(),bool))
        x,y,_=pairs['gyro'];old=trial[row['record']]
        validation.append({'record':row['record'],'group':row['group'],'donor':old['is_calibration_group'],
             'quiet_blocks':len(lv),'baseline_quiet':old['prior_quiet_median_deg'],'baseline_gyro':old['prior_gyro_median_deg'],
             'new_quiet':float(np.median(angle(lv,h))) if len(lv) else None,
             'new_gyro':float(np.median(angle(C.apply(x),y))) if len(x) else None,**log})
    out={'donor_group':'Pendulum/Test02','donor_level':prior.tolist(),'checks':validation,
         'qualification':'Calibrated relative reference with a fixture prior. Not external absolute attitude truth.'}
    (ROOT/'review/repo_rank_one_audit.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
    held=[r for r in validation if not r['donor'] and r['quiet_blocks']]
    for key in ('baseline_quiet','new_quiet','baseline_gyro','new_gyro'):
        print(key,np.percentile([q[key] for q in held],[50,90,100]))


if __name__=='__main__':run()

