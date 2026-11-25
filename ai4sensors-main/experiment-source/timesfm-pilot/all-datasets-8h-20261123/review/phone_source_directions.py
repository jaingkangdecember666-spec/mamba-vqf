"""Quantify the direction and magnitude of existing corrections on four external RIDI training recordings."""
import hashlib
import json

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from check_sources import ROOT, TRANSFER, weights
from native_blocks import block_inputs
from optical_score import infer_canonical, metric


def summarize(record, head6, readout):
    path = ROOT/'external-sources/ridi-phone-train'/record['file']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record['file_sha256']
    with np.load(path) as file:
        x = np.column_stack((file['acc_m_s2'],file['gyr_rad_s']))
        assert file['imu_observed'].all() and np.isfinite(x).all()
        valid = file['reference_valid'] & (file['time_ns']-file['time_ns'][0] >= 10_000_000_000)
        ref = Rotation.from_quat(file['tango_wxyz'],scalar_first=True).inv().apply([0.,0.,1.])
    q,windows,last,release = block_inputs(x,200.)
    ids = np.flatnonzero(valid)
    ref_world = q[ids].apply(ref[ids])
    z = np.array([0.,0.,1.])
    lookup = np.searchsorted(release,ids,side='right')
    def prediction(head,gate):
        delta = .25*q[last].apply(infer_canonical(head,windows))
        if gate:
            delta[windows[:,-1,12].astype(bool)] = 0.
        direction = z+delta
        direction /= np.linalg.norm(direction,axis=1,keepdims=True)
        return np.vstack((z,direction))[lookup], np.vstack((z*0.,delta))[lookup]
    original,do = prediction(head6, False)
    evolved,de = prediction(readout, True)
    baseline = metric(np.broadcast_to(z,ref_world.shape),ref_world)
    # The sign is the negative derivative of mean squared chord error at zero correction.
    target = ref_world-ref_world[:,2,None]*z
    def slope(delta):
        return float(np.einsum('ij,ij->',target,delta)/max(1e-20,np.einsum('ij,ij->',delta,delta)))
    gyro = np.linalg.norm(x[ids,3:],axis=1)
    accel = np.abs(np.linalg.norm(x[ids,:3],axis=1)/9.80665-1.)
    return {'record':record['record'],'placement':record['placement'],'rows':len(ids),
            'full_vqf_deg':baseline,'B6_quarter_deg':metric(original,ref_world),
            'C006_readout_deg':metric(evolved,ref_world),
            'B6_quarter_projected_scale':slope(do),'C006_projected_scale':slope(de),
            'angular_rate_rad_s_median_p90':np.quantile(gyro,[.5,.9]).tolist(),
            'specific_force_norm_deviation_g_median_p90':np.quantile(accel,[.5,.9]).tolist()}


def main():
    torch.set_num_threads(2)
    src = ROOT/'external-sources/ridi-phone-train'/'source-audit.json'
    manifest = json.loads(src.read_text(encoding='utf-8'))
    head6 = weights(TRANSFER/'batch6/mamba-intrinsic-selected.pt','mamba_v1')
    readout = weights(ROOT/'C004/B-readout.pt','mamba_v1')
    rows = [summarize(r,head6,readout) for r in manifest['rows']]
    audit = {'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),
             'purpose':'External training-source check with two existing weight sets; no new fitting or target split.',
             'scope':'Before applying the optional sphere front-end; score only observed IMU and valid same-phone Tango pose, after a 10 s warm-up.',
             'rows':rows}
    (ROOT/'review/ridi_training_directions.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf8')
    print([(r['record'],*[round(r[k],5) for k in ['full_vqf_deg','B6_quarter_deg','C006_readout_deg','C006_projected_scale']]) for r in rows])


if __name__ == '__main__':
    main()
