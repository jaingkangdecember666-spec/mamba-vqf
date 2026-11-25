"""Read PennCOSYVIO original VI-Sensor observations and marker-reference rotations on the publisher's clock."""
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from lock_from_frontier import ROOT, digest, read

BASE = ROOT / 'external-sources/penn-handheld'
UP = np.array([0., 0., 1.])
PUBLISHED_BODY_TO_VI = [[.9989551739,-.0016800380,-.0456698809],
                        [.0005817624,.9997105684,-.0240508012],
                        [.0456970689,.0239991032,.9986670221]]


def rig_to_imu():
    table = np.loadtxt(BASE / 'extrinsic_calibration.txt')
    row, = table[table[:, 0] == 10]
    assert abs(np.linalg.norm(row[1:4]) - 1.) < 2e-9
    rotation = Rotation.from_rotvec(row[1:4] * row[4])
    np.testing.assert_allclose(rotation.as_matrix(), PUBLISHED_BODY_TO_VI, atol=2e-9)
    return rotation


def angular_error(a, b):
    return np.rad2deg(np.arctan2(np.linalg.norm(np.cross(a,b),axis=1),np.einsum('ij,ij->i',a,b)))


class Penn:
    def __init__(self, model_lock=None):
        self.roles = read(BASE / 'protocol.json')['role_by_published_sequence']
        self.formal = False
        if model_lock:
            path = (ROOT / 'selection-lock.json').resolve()
            if Path(model_lock).resolve() != path or not path.is_file():
                raise ValueError('Penn B requires the overall selection lock')
            pre = BASE / 'preprocessing-lock.json'
            assert read(path)['evidence_sha256'][pre.relative_to(ROOT).as_posix()] == digest(pre)
            self.formal = True
        self.body_to_imu = rig_to_imu()

    def native(self, name):
        role = self.roles[name]
        if role == 'sealed_after_model_lock' and not self.formal:
            raise ValueError('Reserved Penn B traversal')
        assert role in ('sealed_after_model_lock','external_training')
        folder = BASE / name
        imu = np.loadtxt(folder / 'imu.txt')
        poses = np.loadtxt(folder / 'pose.txt')
        assert imu.shape[1] == 7 and poses.shape[1] == 13
        assert np.isfinite(imu).all() and np.isfinite(poses).all()
        t, tp = imu[:,0], poses[:,0]
        dt, dp = np.diff(t), np.diff(tp)
        assert np.all(dt > 0.) and np.all(dp > 0.)
        hz = 1. / np.median(dt)
        assert 199. < hz < 201.
        signals = np.ascontiguousarray(imu[:,1:])
        matrices = poses[:,1:].reshape(-1,3,4)[:,:,:3]
        assert np.max(np.abs(np.linalg.det(matrices) - 1.)) < 1e-5
        assert np.max(np.abs(matrices @ matrices.transpose(0,2,1) - np.eye(3))) < 1e-5
        # Sensor-to-world maps VI -> rig -> world in this order.
        vi_to_world = Rotation.from_matrix(matrices) * self.body_to_imu.inv()
        left = np.searchsorted(tp,t,side='right') - 1
        j = np.clip(left,0,len(tp)-2)
        bracket = (left >= 0) & (left+1 < len(tp)) & (dp[j] <= .075)
        ref = np.zeros_like(signals[:,:3])
        ref[bracket] = Slerp(tp,vi_to_world)(t[bracket]).inv().apply(UP)
        mask = bracket & (t-t[0] >= 10.)
        assert mask.any()
        # Integrate measured gyro over three genuine, consecutive optical intervals (~100 ms).
        step = 3
        start = np.arange(0,len(tp)-step)
        gap = np.maximum.reduce([dp[k:k+len(start)] for k in range(step)])
        idx = start[(gap <= .075) & (tp[start] >= t[0]) & (tp[start+step] <= t[-1])]
        span = tp[idx+step] - tp[idx]
        measured = Rotation.identity(len(idx))
        for k in range(20):
            at = tp[idx] + (k+.5) * span / 20.
            rates = np.column_stack([np.interp(at,t,signals[:,axis]) for axis in (3,4,5)])
            measured = measured * Rotation.from_rotvec(rates * (span/20.)[:,None])
        optical = (vi_to_world[idx].inv() * vi_to_world[idx+step]).as_rotvec()
        inertial = measured.as_rotvec()
        lengths = np.linalg.norm(optical, axis=1), np.linalg.norm(inertial, axis=1)
        moving = (lengths[0]/span > .2) & (lengths[1]/span > .2)
        magnitude = np.linalg.norm(signals[:,:3],axis=1)
        quiet = bracket & (np.abs(magnitude-9.80665) < .5) & (np.linalg.norm(signals[:,3:],axis=1) < .12)
        indices = np.flatnonzero(mask).astype('<i8')
        info = {'native_rows':len(t),'native_hz':float(hz),'duration_s':float(t[-1]-t[0]),
                'median_imu_dt_ms':float(1e3*np.median(dt)),'largest_imu_gap_ms':float(1e3*dt.max()),
                'optical_rows':len(tp),'median_optical_hz':float(1./np.median(dp)),'optical_largest_gap_ms':float(1e3*dp.max()),
                'scored_frames':len(indices),'scoring_indices_sha256':hashlib.sha256(indices.tobytes()).hexdigest(),
                'gyro_optical_dynamic_pairs':int(moving.sum()),'quiet_native_rows':int(quiet.sum()),
                'gyro_optical_median_angle_deg':float(np.median(angular_error(inertial[moving],optical[moving]))) if moving.any() else None,
                'optical_over_gyro_median_displacement':float(np.median(lengths[0][moving]/lengths[1][moving])) if moving.any() else None,
                'quiet_reference_vs_force_median_deg':float(np.median(angular_error(ref[quiet],signals[quiet,:3]))) if quiet.any() else None}
        return (signals,float(hz),ref,mask),info


def prepare():
    ds = Penn()
    archive = read(BASE / 'source-files.json')
    assert archive['registered_sha256'] == digest(BASE/'protocol.json')
    assert archive['rig_to_sensor_calibration']['sha256'] == digest(BASE/'extrinsic_calibration.txt')
    entries = {r['record']:r for r in archive['records']}
    assert set(entries) == set(ds.roles)
    folder = BASE/'prepared-train'
    folder.mkdir(exist_ok=True)
    result = []
    for name,role in ds.roles.items():
        entry = entries[name]
        assert entry['role'] == role
        for part in ('pose','imu'):
            assert digest(BASE/entry[part]['path']) == entry[part]['sha256']
        if role != 'external_training':
            try: ds.native(name)
            except ValueError as exc: assert 'Reserved Penn B' in str(exc)
            else: raise AssertionError('Penn held-out traversal accessible')
            continue
        raw,info = ds.native(name)
        assert info['gyro_optical_dynamic_pairs'] > 100 and info['quiet_native_rows'] > 100, (name,info)
        assert info['gyro_optical_median_angle_deg'] < 30 and .7 < info['optical_over_gyro_median_displacement'] < 1.4, (name,info)
        assert info['quiet_reference_vs_force_median_deg'] < 10 and info['largest_imu_gap_ms'] < 10.1, (name,info)
        target = folder/(name+'.npz')
        fields = dict(zip(('six_axis','native_hz','optical_up','mask'),raw))
        if target.is_file():
            with np.load(target) as previous:
                for k,v in fields.items(): np.testing.assert_array_equal(previous[k],v)
        else:
            np.savez_compressed(target,**fields)
        result.append({'record':name,'prepared':target.relative_to(ROOT).as_posix(),'sha256':digest(target),**info})
    assert len(result)==2
    spec = {'source_files_sha256':digest(BASE/'source-files.json'),'protocol_sha256':digest(BASE/'protocol.json'),
            'loader_sha256':digest(Path(__file__)),'source_records':result,'sealed_records':['bf','bs'],
            'reference':'Use original VI acceleration and angular rate, published rig-to-VI and rig-to-world poses. VI-to-world = rig-to-world * inverse(rig-to-VI). Evaluate world +Z at native IMU times after 10s with optical brackets <=75ms.',
            'transfer_scope':'Four recorded runs at two speeds on approximately the same route, with one instrument. The original benchmark assigns A to training and B to testing; its reference poses include smoothing and can span frames without visual markers.'}
    lock = BASE/'preprocessing-lock.json'
    if lock.exists():
        assert {k:v for k,v in read(lock).items() if k!='locked_at'} == spec
    else:
        with lock.open('x',encoding='utf8') as f: json.dump({'locked_at':datetime.now().astimezone().isoformat(),**spec},f,indent=2)
    print([(r['record'],r['scored_frames'],r['gyro_optical_median_angle_deg'],r['quiet_reference_vs_force_median_deg'])for r in result])


if __name__ == '__main__':
    prepare()
