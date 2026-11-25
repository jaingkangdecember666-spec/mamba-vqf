"""Independent angular-risk and lower bounds for a global one-dimensional physical-cap fit on external sources."""
import json

import numpy as np

from anchor_blend_features import included_degrees
from lock_from_frontier import ROOT,digest
from physical_angle_cap import cap_direction


class ExternalCap:
    def __init__(self):
        raw=ROOT/'review/raw-anchor-training-141.npz'
        physical=ROOT/'review/external-physical-anchor-141.npz'
        with np.load(raw)as a,np.load(physical)as p:
            for key in ('domain','record','native_release'):np.testing.assert_array_equal(a[key],p[key])
            self.truth,self.physical,self.model=a['truth'],p['physical'],a['C023']
            ref,dom,self.ids=a['raw_default'],a['domain'],a['record_id']
        self.groups=sorted(set(dom))
        self.members=[np.unique(self.ids[dom==d])for d in self.groups]
        self.N=np.bincount(self.ids)
        self.disagreement=included_degrees(self.physical,self.model)
        self.den=self.group_rmse(included_degrees(ref,self.truth))
        self.provenance={'raw':digest(raw),'physical':digest(physical)}

    def group_rmse(self,values):
        per_record=np.sqrt(np.bincount(self.ids,weights=values**2)/self.N)
        return np.array([per_record[g].mean()for g in self.members])

    def risks(self,cap):
        model=cap_direction(self.physical,self.model,cap)
        return self.group_rmse(included_degrees(model,self.truth))/self.den

    def slopes(self,left):
        # Each unsaturated prediction moves at most one degree per degree of cap.
        # Reverse triangle inequality on the sphere and the RMSE norm preserve this bound.
        return self.group_rmse((self.disagreement>left-1.e-11).astype(float))/self.den

    def interval_bound(self,left,right,left_risks,right_risks):
        slopes=self.slopes(left)
        assert np.all(abs(right_risks-left_risks)<=slopes*(right-left)+2e-9)
        return float(np.max(.5*(left_risks+right_risks-slopes*(right-left))))


def audit_earlier_fit():
    bank=ExternalCap()
    old=ROOT/'C036/solver-evaluations.jsonl'
    trials=[json.loads(s)for s in old.read_text().splitlines()]
    zero=bank.risks(0.);eps=1.e-5;right=bank.risks(eps)
    derivative=(right-zero)/eps
    at_max=np.flatnonzero(zero>=zero.max()-1.e-10)
    detail={'groups':bank.groups,'maximum_native_disagreement_deg':float(bank.disagreement.max()),
            'p90_native_disagreement_deg':float(np.percentile(bank.disagreement,90)),
            'record_macro_VQF_deg':bank.den.tolist(),'endpoint_risks':zero.tolist(),
            'one_sided_derivative_per_degree':derivative.tolist(),'epsilon_deg':eps,
            'endpoint_active_groups':[bank.groups[i]for i in at_max],
            'strictly_better_nearby_on_external_objective':bool(right.max()<zero.max()),
            'min_positive_tested_cap_deg':min(x['cap_deg'] for x in trials if x['cap_deg']>0),
            'max_original_logged_objective_gap':max(abs(bank.risks(x['cap_deg']).max()-x['objective'])for x in trials),
            'source_hashes':bank.provenance,'original_evaluations_sha256':digest(old)}
    assert detail['strictly_better_nearby_on_external_objective'] and detail['max_original_logged_objective_gap']<1.e-9
    (ROOT/'review/C036_solver_gap.json').write_text(json.dumps(detail,indent=2)+'\n',encoding='utf8')
    print(detail['min_positive_tested_cap_deg'],detail['endpoint_active_groups'],derivative[at_max],detail['max_original_logged_objective_gap'])


if __name__=='__main__':audit_earlier_fit()
