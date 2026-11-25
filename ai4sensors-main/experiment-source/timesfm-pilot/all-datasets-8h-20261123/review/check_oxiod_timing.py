"""Check row synchronization on disjoint development prefixes using angular-speed magnitudes only."""
import json
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation
from check_mount_pilots import ROOT, source


def speeds(gyro,q,dt):
    valid=np.isfinite(q).all(1)&(np.linalg.norm(q,axis=1)>.5)
    qq=np.zeros_like(q);qq[:,3]=1;qq[valid]=q[valid]
    rot=Rotation.from_quat(qq)
    hop=6
    mid=np.arange(hop//2,len(q)-hop//2)
    w=gaussian_filter1d(np.linalg.norm(np.nan_to_num(gyro),axis=1),4.)
    v=np.full(len(q),np.nan)
    diff=(rot[mid-hop//2].inv()*rot[mid+hop//2]).magnitude()/(hop*dt)
    ok=valid[mid-hop//2]&valid[mid+hop//2]
    v[mid[ok]]=diff[ok]
    # Smooth the optical magnitude only on intervals with signal; truncated ends are dropped by the alignment mask.
    u=np.nan_to_num(v);weight=gaussian_filter1d(np.isfinite(v).astype(float),4.)
    v=gaussian_filter1d(u,4.)/np.maximum(weight,1e-8);v[weight<.999]=np.nan
    return w,v


def offsets(w,v,t,mask):
    i=np.flatnonzero(mask&np.isfinite(v))
    i=i[(i>=26)&(i+26<len(w))]
    scores={}
    for delta in range(-25,26):
        o=w[i+delta];z=v[i]; good=np.isfinite(o+z)&(np.maximum(o,z)<15)
        if good.sum()<100 or min(np.std(o[good]),np.std(z[good]))<.05:
            continue
        scores[delta]=float(np.corrcoef(o[good],z[good])[0,1])
    if not scores:return {'rows':len(i),'best_lag':None}
    best=max(scores,key=scores.get)
    return {'rows':len(i),'best_lag':best,'best_correlation':scores[best],'zero_correlation':scores[0],
            'correlations':scores}


def main():
    inv=json.loads((ROOT/'C001/inventory.json').read_text('utf8'))
    picks=json.loads((ROOT/'review/mount_development_audit.json').read_text('utf8'))
    rows=[]
    for item in picks:
        if item.get('domain')!='OxIOD':continue
        r=next(r for r in inv['records'] if r['dataset']=='OxIOD' and r['record']==item['name'] and r['side']=='development')
        t,a,g,q=source(r); w,v=speeds(g,q,.01); first=item['fit_through_s']; last=item['quality_check_through_s']
        s0=offsets(w,v,t,(t>=0)&(t<first));s1=offsets(w,v,t,(t>=first)&(t<last));
        rows.append({'name':r['record'],'nominal_hz':100,'fit':s0,'independent_check':s1})
        print(r['record'],[(s.get('best_lag'),round(s.get('best_correlation',0),3),round(s.get('zero_correlation',0),3)) for s in [s0,s1]])
    (ROOT/'review/oxiod_timing_audit.json').write_text(json.dumps(rows,indent=2)+'\n',encoding='utf8')


if __name__=='__main__':main()
