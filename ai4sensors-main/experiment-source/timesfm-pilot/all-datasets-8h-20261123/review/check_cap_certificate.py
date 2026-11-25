"""Recalculate the certificate without changing the registered training-side risk module."""
import json

from lock_from_frontier import ROOT,digest
from physical_cap_certificate import ExternalCap


def audit_global_fit():
    b=ROOT/'C037';bank=ExternalCap()
    cert=json.loads((b/'certificate.json').read_text())
    coef=json.loads((b/'coefficient-lock.json').read_text())
    assert cert['solve_count']==1 and cert['neural_fits']==0 and coef['certificate_sha256']==digest(b/'certificate.json')
    evals=[json.loads(s)for s in (b/'solver-evaluations.jsonl').read_text().splitlines()]
    by_cap={r['cap_deg']:bank.risks(r['cap_deg']) for r in evals}
    assert len(evals)==len(by_cap)==cert['evaluated_points']
    discrepancy=max(abs(by_cap[r['cap_deg']].max()-r['objective'])for r in evals)
    leaves=sorted(cert['leaves'],key=lambda e:e['left'])
    assert leaves[0]['left']==0. and abs(leaves[-1]['right']-bank.disagreement.max())<1e-9
    assert all(first['right']==next_['left'] for first,next_ in zip(leaves,leaves[1:]))
    lower=[]
    for node in leaves:
        left,right=node['left'],node['right']
        assert left<right
        bound=bank.interval_bound(left,right,by_cap[left],by_cap[right])
        assert abs(bound-node['lower_bound'])<1.e-9
        lower.append(bound)
    optimum=float(min(v.max()for v in by_cap.values()));cap=coef['cap_deg']
    gap=float(optimum-min(lower))
    assert abs(gap-cert['gap'])<1.e-9 and gap<=1.e-5 and discrepancy<1.e-9
    assert abs(by_cap[cap].max()-optimum)<1.e-10 and len(leaves)==len(by_cap)-1
    out={'cap_deg':cap,'evaluations':len(evals),'leaves':len(leaves),'upper_bound':optimum,'lower_bound':min(lower),
         'global_gap':gap,'max_logged_objective_gap':float(discrepancy),'entire_interval_deg':[0.,float(bank.disagreement.max())],
         'checked_source_ratios':dict(zip(bank.groups,by_cap[cap].tolist())),'source_hashes':bank.provenance,
         'certificate_sha256':digest(b/'certificate.json')}
    (ROOT/'review/C037_certificate_check.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
    print(cap,len(evals),len(leaves),gap)


if __name__=='__main__':audit_global_fit()
