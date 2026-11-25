"""Causal 100 Hz feature blocks at native rates below as well as above 100 Hz."""
import numpy as np
from scipy.spatial.transform import Rotation
from vqf import VQF

from check_sources import ROOT, TRANSFER


def block_inputs(signal, hz):
    """Use previous observed IMU and VQF state for an empty block, identity prior before first arrival."""
    if hz <= 0 or len(signal)==0 or not np.isfinite(signal).all():
        raise ValueError('Need a nonempty, finite six-axis signal and a positive declared sampling rate')
    nblocks=int(np.floor(len(signal)*100/hz+1e-9))
    edges=np.arange(nblocks+1,dtype=np.float64)*(hz/100)
    stops=np.floor(edges+1e-9).astype('int64')
    sizes=np.diff(stops)
    data=np.ascontiguousarray(signal,dtype='float64')
    out=VQF(1/hz).updateBatch(np.ascontiguousarray(data[:,3:]),np.ascontiguousarray(data[:,:3]))
    q=Rotation.from_quat(out['quat6D'],scalar_first=True)
    block=np.zeros((nblocks,13),dtype='float32')
    block[:,8]=1.
    nonempty=sizes>0
    if nonempty.any():
        block[nonempty,:6]=np.add.reduceat(data[:stops[-1]],stops[:-1][nonempty],axis=0)/sizes[nonempty,None]
    prev=stops[1:]-1
    was_observed=prev>=0
    empty=was_observed & ~nonempty
    block[empty,:6]=data[prev[empty]]
    idx=prev[was_observed]
    block[was_observed,6:9]=q[idx].inv().apply([0.,0.,1.])
    block[was_observed,9:12]=out['bias'][idx]
    block[was_observed,12]=out['restDetected'][idx]
    k=np.arange(69,nblocks,10)
    last=prev[k]
    release=np.ceil(edges[k+1]-1e-9).astype('int64')-1
    assert np.all(last<=release) and np.all(release<len(signal)) and np.all(last>=0)
    return q,block[k[:,None]+np.arange(-63,1)[None,:]],last,release


def verify():
    import json
    from replay_native import features
    rng=np.random.default_rng(20261123)
    signal=rng.normal(0,.17,size=(3501,6));signal[:,2]+=9.80665
    output=[]
    for hz in (25.,48.,90.,1/.011,100.,166.,200.,1/.005016,285.714):
        native,windows,last,release=block_inputs(signal,hz)
        row={'hz':hz,'releases':len(release)}
        assert np.all(np.diff(release)>0)
        if hz>=100:
            previous=features(signal,hz)
            np.testing.assert_array_equal(previous[0].as_quat(),native.as_quat())
            for a,b in zip(previous[1:],(windows,last,release)):np.testing.assert_array_equal(a,b)
            row['high_rate_matches_original_bitwise']=True
        for position in (2,len(release)//2,len(release)-2):
            end=release[position]+1
            q,w,ix,t=block_inputs(signal[:end],hz)
            np.testing.assert_array_equal(q.as_quat(),native.as_quat()[:end])
            for a,b in zip((w,ix,t),(windows,last,release)):np.testing.assert_array_equal(a,b[:position+1])
            corrupted=signal.copy();corrupted[end:]=rng.normal(0,50,size=corrupted[end:].shape)
            q,w,ix,t=block_inputs(corrupted,hz)
            np.testing.assert_array_equal(w[:position+1],windows[:position+1])
        row['truncation_and_future_perturbations']=3
        output.append(row)
    (ROOT/'review/sample_rate_audit.json').write_text(json.dumps(output,indent=2)+'\n',encoding='utf8')
    print(output)


if __name__=='__main__':
    verify()

