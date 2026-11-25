"""Shared raw-record adapters for independent development diagnostics and final locked replays."""
import hashlib
import json
from functools import cached_property
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from check_array import base_inputs
from check_sources import DATA, ROOT, tum
from optical_reference import extract
from vehicle_reference import prepare_vehicle
from audit_external import load_external


def body_reference(quaternion, world_up):
    valid = np.isfinite(quaternion).all(axis=1) & (np.linalg.norm(quaternion,axis=1) > 0)
    directions = np.zeros((len(valid),3))
    directions[valid] = Rotation.from_quat(quaternion[valid],scalar_first=True).inv().apply(world_up)
    return directions, valid


def shice_record(row):
    path = Path(row['path'])
    with path.open(encoding='utf-8-sig') as f:
        header = f.readline().rstrip().split(',')
        columns = ['sample_index','time_s',*[f'imu_{k}_{a}' for k in ('acc','gyr') for a in 'xyz'],
                   *[f'opt_quat_{a}' for a in 'wxyz'],'is_interpolated']
        table = np.loadtxt(f,delimiter=',',quotechar='"',usecols=[header.index(c) for c in columns],ndmin=2)
    indices = np.arange(len(table))
    np.testing.assert_array_equal(table[:,0],indices)
    dt = np.diff(table[:,1])
    assert np.allclose(dt,np.median(dt),rtol=1e-5,atol=1e-9)
    signal = table[:,2:8].copy()
    signal[table[:,-1]!=0] = np.nan
    truth,valid = body_reference(table[:,8:12],[0.,-1.,0.])
    with np.load(path.parents[1]/'Label_quality_audit_20261116'/row['record']/'screening_masks.npz') as archive:
        np.testing.assert_array_equal(archive['sample_index'],indices)
        valid &= archive['context_ok'].astype(bool) & archive['observed_loss_ok'].astype(bool)
    return signal,1./np.median(dt),truth,valid & (table[:,1]>=10.) & np.isfinite(signal).all(1)


def yt_record(row):
    import h5py
    with h5py.File(DATA/'YT_server/YT_server_200Hz.h5','r') as f:
        series = f[row['source']]
        time = np.asarray(series['ts'], dtype='float64').ravel()
        signal = np.column_stack([series['acc'][:],series['gyr'][:]]).astype('float64')
        quat = series['qua'][:].astype('float64')
    assert len(time) == row['native_frames'] and np.isfinite(time).all() and np.all(np.diff(time)>0.)
    truth, valid = body_reference(quat,[0.,0.,1.])
    return signal,200.,truth,valid & (time-time[0]>=10.) & np.isfinite(signal).all(1)


class Catalog:
    def __init__(self, lock_path=None):
        self.inventory_path = ROOT/'C001/inventory.json'
        records = json.loads(self.inventory_path.read_text(encoding='utf-8-sig'))['records']
        self.rows = {(r['dataset'],r['record']):r for r in records}
        self.lock = None
        if lock_path is not None:
            expected=(ROOT/'selection-lock.json').resolve()
            if Path(lock_path).resolve()!=expected or not expected.is_file():
                raise ValueError('The official eight-domain holdouts require the recorded overall selection')
            self.lock = json.loads(expected.read_text(encoding='utf-8-sig'))
            if self.lock.get('inventory_sha256') != hashlib.sha256(self.inventory_path.read_bytes()).hexdigest():
                raise ValueError('The final selection lock must identify the frozen inventory')
            assert self.lock['selected']['paired_record_count']==213 and len(self.lock['formal_cases'])==365

    @cached_property
    def calibration(self):
        return np.memmap(DATA/'IMU-Array-main/Calibration/Group1_ref_data.bin',dtype='<f8',mode='r').reshape(-1,7)

    @cached_property
    def drive_nav_and_start(self):
        if self.lock is None:
            raise ValueError('Vehicle drive references require the selection lock')
        rows = [r for (d,_),r in self.rows.items() if d=='IMU-Array' and r['side'].startswith('sealed')]
        assert len(rows)==16
        refs = np.loadtxt(DATA/'IMU-Array-main/NavigationTest/RefNavResult.nav',ndmin=2)
        assert refs.shape[1]==11
        starts = [np.fromfile(r['path'],dtype='<f8',count=1)[0] for r in rows]
        return refs,max(*starts,refs[0,1])

    def native(self, domain, name):
        row = self.rows[domain,name]
        side = row['side']
        if self.lock is None and side not in ('development','development_previously_used_for_selection'):
            raise ValueError('This run can read only the frozen development panel')
        if domain=='SHICE':
            return shice_record(row), {}
        if domain in ('BROAD','Sassari'):
            raw,q,mask,hz=load_external({'domain':domain,'name':name,'path':row['path']})
            ref,valid=body_reference(q,[0.,0.,1.])
            return (raw,hz,ref,mask & valid & np.isfinite(raw).all(1)), {}
        if domain=='TUM-VI':
            return tum(int(name.removeprefix('room'))), {}
        if domain=='YT_Server':
            return yt_record(row), {}
        if domain in ('OxIOD','RepoIMU'):
            data = extract(row)
            return data[:4],data[4]
        if domain=='IMU-Array' and row['group']=='calibration_trial':
            return base_inputs(row,self.calibration)
        if domain=='IMU-Array' and row['group']=='vehicle_trial':
            nav,start=self.drive_nav_and_start
            raw=np.memmap(row['path'],dtype='<f8',mode='r').reshape(-1,7)
            sig,hz,ref,mask,nominal,info=prepare_vehicle(raw,nav,start)
            return (sig,hz,ref,mask),dict(info,nominal_reference=nominal)
        raise ValueError(f'Unsupported domain and acquisition: {domain} {row["group"]}')
