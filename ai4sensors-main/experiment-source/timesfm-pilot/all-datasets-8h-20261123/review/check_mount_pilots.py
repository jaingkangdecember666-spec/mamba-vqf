"""Inspect mounting observability on development-only Vicon calibration prefixes and a disjoint check interval."""
import io
import json
from pathlib import Path
import numpy as np
from mount_check import paired_vectors, fit_from_pairs, angle

ROOT=Path(__file__).resolve().parent.parent


def source(row):
    if row['dataset']=='OxIOD':
        a=np.loadtxt(row['path'],delimiter=',',ndmin=2)
        b=np.loadtxt(row['reference_path'],delimiter=',',ndmin=2)
        assert len(a)==len(b) and b.shape[1]>=9
        t=np.arange(len(a))*.01
        acc=-(a[:,7:10]+a[:,10:13])*9.80665 if a.shape[1]==16 else a[:,1:4]
        gyro=a[:,4:7]
        return t,acc,gyro,b[:,5:9]
    with Path(row['path']).open(encoding='utf-8-sig') as stream:
        next(stream); next(stream)
        a=np.genfromtxt(io.StringIO(stream.read().replace(',','.')),delimiter=';',usecols=tuple(range(11)),ndmin=2)
    return a[:,0]-a[0,0], a[:,5:8], a[:,8:11], np.roll(a[:,1:5],-1,axis=1)


def assess(row):
    t,a,w,q=source(row)
    duration=t[-1]-t[0];cut=min(60., duration*.25); end=min(cut*1.6, duration*.40)
    intervals=[(t>=0)&(t<cut),(t>=cut)&(t<end)]
    parts=[]
    for mask in intervals:
        n=mask.sum()
        parts.append(paired_vectors(t[mask],a[mask],w[mask],q[mask],np.ones(n,bool)))
    fit,check=parts
    before={s:len(v[0]) for s,v in fit.items()}
    if sum(before.values())<3:
        return {'name':row['record'],'status':'insufficient calibration','pairs':before}
    rot,audit=fit_from_pairs(fit)
    held={}
    for mode,(x,y,_) in check.items():
        if len(x):
            original=angle(x,y);aligned=angle(rot.apply(x),y)
            held[mode]={'pairs':len(x),'before_median_deg':float(np.median(original)),
                        'aligned_median_deg':float(np.median(aligned)),'aligned_p90_deg':float(np.quantile(aligned,.90))}
    return {'name':row['record'],'domain':row['dataset'],'duration_s':float(duration),'fit_through_s':float(cut),
            'quality_check_through_s':float(end),'calibration':audit,'check':held}


def main():
    records=json.loads((ROOT/'C001/inventory.json').read_text('utf8'))['records']
    available=[r for r in records if r['side']=='development']
    chosen=[]
    for g in ['handbag','handheld','multi devices','running']:
        chosen.append(next(r for r in available if r['dataset']=='OxIOD' and r['group']==g))
    nexus=next((r for r in available if 'nexus' in r['record'].lower()),None)
    if nexus: chosen.append(nexus)
    for num in (1,2,3):
        chosen.append(next(r for r in available if r['group']=='Pendulum/Test02' and r['record'].endswith('Segment_'+str(num))))
    for g in ['TStick/Test02','TStick/Test08']:
        chosen.append(next(r for r in available if r['group']==g))
    out=[assess(r) for r in chosen]
    (ROOT/'review/mount_development_audit.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
    for r in out:
        if 'calibration' in r:
            print(r['name'],round(r['calibration']['second_over_first'],3),{m:(x['pairs'],round(x['before_median_deg'],1),round(x['aligned_median_deg'],1)) for m,x in r['check'].items()})
        else: print(r)


if __name__=='__main__':
    main()
