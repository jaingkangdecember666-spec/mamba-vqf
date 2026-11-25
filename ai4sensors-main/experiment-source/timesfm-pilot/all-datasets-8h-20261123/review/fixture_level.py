"""Carry the fixture-level prior into the same mounting convention as the per-record fit."""
import json
import numpy as np
from scipy.spatial.transform import Rotation
from check_mount_pilots import ROOT,source
from optical_reference import still_world
from mount_check import angle


def transport_prior(q_xyzw, C, independent_world_level):
    # A datum expressed in body axes at the reference origin needs both rotations when the right mount changes.
    good=np.flatnonzero(np.isfinite(q_xyzw).all(1)&(np.linalg.norm(q_xyzw,axis=1)>.5))
    if not len(good):raise ValueError('No optical orientation in the prescribed calibration prefix')
    origin=Rotation.from_quat(q_xyzw[good[0]])
    return (origin*C*origin.inv()).apply(independent_world_level)


def main():
    i=json.loads((ROOT/'C001/inventory.json').read_text('utf8'))
    v={r['record']:r for r in json.loads((ROOT/'review/optical_development_calibration.json').read_text('utf8'))}
    prior=np.array(json.loads((ROOT/'review/repo_world_axis_audit.json').read_text('utf8'))['summary']['fitted_world_direction_identity_mount'])
    summary=[]
    for r in i['records']:
        if r['side']!='development' or not r['group'].startswith('Pendulum/'):continue
        old=v[r['record']];t,a,w,q=source(r);fit=t<old['fit_through_s']
        C=Rotation.from_quat(old['mount_xyzw']);original=np.array(old['world_up'])
        fallback=old['world_up_source']=='fixture_X_fallback_unverified'
        revised=transport_prior(q[fit],C,prior) if fallback else original
        later=still_world(t[~fit],a[~fit],w[~fit],q[~fit],C)
        summary.append({'record':r['record'],'group':r['group'],'affected':fallback,
                        'independent_quiet_blocks':len(later),'prior_world_level':original.tolist(),'revised_world_level':revised.tolist(),
                        'old_quiet_mae_deg':float(np.median(angle(later,original))) if len(later) else None,
                        'new_quiet_mae_deg':float(np.median(angle(later,revised))) if len(later) else None})
    held=[r for r in summary if r['group']!='Pendulum/Test02' and r['independent_quiet_blocks']]
    output={'reference_donors':'Pendulum/Test02, no model predictions','world_prior_at_identity_mount':prior.tolist(),'records':summary}
    (ROOT/'review/fixture_level_transport.json').write_text(json.dumps(output,indent=2)+'\n',encoding='utf8')
    for k in ('old_quiet_mae_deg','new_quiet_mae_deg'):
        print(k,np.percentile([r[k] for r in held],[50,90,100]))
    print('affected',[(r['record'],round(r['old_quiet_mae_deg'],1),round(r['new_quiet_mae_deg'],1)) for r in held if r['affected']])


if __name__=='__main__':main()

