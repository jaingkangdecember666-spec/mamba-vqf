"""Read original SHICE CSV and its exclusion masks for independent transfer-score checks."""

import hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

SHICE = Path("E:/数据集/IMU_dynamic_data_SHICE")
CHANNELS = [f"imu_{kind}_{xyz}" for kind in ("acc","gyr") for xyz in "xyz"]
QUATERNION = [f"opt_quat_{c}" for c in "wxyz"]


def source_record(name):
    source = SHICE/(name.split("-")[0]+"_aligned_formal")/(name+".csv")
    with source.open(encoding="utf-8-sig") as handle:
        header=handle.readline().rstrip().split(",")
        cols=["sample_index","time_s",*CHANNELS,*QUATERNION,"is_interpolated"]
        a=np.loadtxt(handle,delimiter=",",quotechar='"',usecols=[header.index(c) for c in cols],ndmin=2)
    np.testing.assert_array_equal(a[:,0],np.arange(len(a)))
    times=a[:,1]
    dt=float(np.median(np.diff(times)))
    packet=np.isfinite(a[:,2:8]).all(axis=1)&(a[:,12]==0)
    prev=np.maximum.accumulate(np.where(packet,np.arange(len(a)),-1))
    imu=a[prev.clip(min=0),2:8].copy()
    imu[prev<0]=0
    labelled=np.isfinite(a[:,8:12]).all(axis=1)&np.any(a[:,8:12]!=0,axis=1)
    ref=np.zeros((len(a),3),dtype=np.float64)
    ref[labelled]=Rotation.from_quat(a[labelled,8:12],scalar_first=True).inv().apply([0.,-1.,0.])
    with np.load(SHICE/"Label_quality_audit_20261116"/name/"screening_masks.npz",allow_pickle=False) as m:
        np.testing.assert_array_equal(m["sample_index"],np.arange(len(a)))
        mask=(times>=10.) & packet & labelled & m["context_ok"].astype(bool) & m["observed_loss_ok"].astype(bool)
    return imu,1/dt,ref,mask


def score_indices(mask):
    return hashlib.sha256(np.flatnonzero(mask).astype("<i8").tobytes()).hexdigest()
