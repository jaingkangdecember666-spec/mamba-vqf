"""Three external Vicon Room 1 trajectories, each with raw gyro and a documented source-only bias variation."""
from pathlib import Path

import numpy as np

from check_euroc_source import load_source
from check_sources import ROOT
from lock_from_frontier import digest,read
from native_catalog import body_reference


NAMES=('V1_01_easy','V1_02_medium','V1_03_difficult')


def paired(name):
    if name not in NAMES:
        raise ValueError('Only the three designated external training files can be read')
    base=ROOT/'external-sources/euroc-original-inputs'
    rows={r['record']:r for r in read(base/'source-audit.json')['records']}
    row=rows[name];path=base/row['prepared_file']
    diagnostic=read(ROOT/'review/euroc_bias_response.json')
    frozen=next(r for r in diagnostic['records'] if r['record']==name)
    hdf5=ROOT/'external-sources/euroc-vicon-room1'/(name+'.hdf5')
    assert row['prepared_sha256']==digest(path)==frozen['original_source_sha256'] and digest(hdf5)==frozen['conversion_sha256']
    with np.load(path)as f:
        a=f['acc_m_s2'];w=f['gyr_rad_s']
        up,valid=body_reference(f['alternative_wxyz'],[0.,0.,1.])
        mask=valid&f['movement']&f['alternative_valid']&(np.arange(len(w))>=2000)
    checked_a,known_debiased,_,_,_=load_source(name)
    np.testing.assert_allclose(a,checked_a,atol=1e-6,rtol=0.)
    offset=np.median(w[:2000]-known_debiased[:2000],axis=0)
    np.testing.assert_allclose(np.degrees(offset),frozen['prefix_bias_deg_s'],rtol=0,atol=1e-12)
    assert np.isfinite(w).all() and np.isfinite(a).all()
    return ((np.column_stack((a,w)),200.,up,mask),(np.column_stack((a,w-offset)),200.,up,mask))
