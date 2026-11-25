"""Construct a 2.56 s Mamba window at each existing 0.1 s release from the causal 100 Hz block stream."""
import json

import numpy as np

from check_sources import ROOT
from native_blocks import block_inputs


def long_windows(fine):
    if fine.ndim!=3 or fine.shape[1:]!=(64,13):
        raise ValueError('Expected consecutive 64-by-13 release windows')
    result=np.zeros_like(fine)
    result[:,:,8]=1.
    active=np.zeros(len(fine),dtype=bool)
    if not len(fine):
        return result,active
    # The first window contains blocks 6..69. Every following release adds ten new blocks.
    stream=np.concatenate((fine[0],fine[1:,-10:,:].reshape(-1,13)),axis=0)
    last=63+10*np.arange(len(fine))
    active=last>=255
    if active.any():
        ends=last[active,None]-4*np.arange(63,-1,-1)[None,:]
        result[active]=stream[ends]
        cumulative=np.vstack((np.zeros((1,6)),np.cumsum(stream[:,:6],axis=0,dtype=np.float64)))
        result[active,:,:6]=(cumulative[ends+1]-cumulative[ends-3])/4.
        np.testing.assert_array_equal(result[active,-1,6:],fine[active,-1,6:])
    return result,active


def main():
    rng=np.random.default_rng(20261123)
    reports=[]
    for hz in (48.,100.,166.,200.,285.714):
        n=round(18.*hz)
        x=rng.normal(0.,.4,size=(n,6));x[:,2]+=9.80665
        _,fine,at,release=block_inputs(x,hz)
        coarse,valid=long_windows(fine)
        assert valid.any()
        extended=np.concatenate((fine[0],fine[1:,-10:].reshape(-1,13)))
        for j in np.flatnonzero(valid)[::7]:
            end=64+10*j
            groups=extended[end-256:end].reshape(64,4,13)
            expected=groups[:,-1,:].copy()
            expected[:,:6]=groups[:,:,:6].mean(1,dtype=np.float64)
            np.testing.assert_allclose(coarse[j],expected,rtol=1e-7,atol=1e-8)
        for j in (np.flatnonzero(valid)[0],len(fine)//2,len(fine)-2):
            cut=release[j]+1
            changed=x.copy()
            changed[cut:]=rng.normal(0.,20,size=changed[cut:].shape)
            for subset in (x[:cut],changed):
                _,w,_,_=block_inputs(subset,hz)
                history,ready=long_windows(w)
                np.testing.assert_array_equal(history[:j+1],coarse[:j+1])
                np.testing.assert_array_equal(ready[:j+1],valid[:j+1])
        first=np.flatnonzero(valid)[0]
        reports.append({'native_hz':hz,'releases':len(fine),'full_windows':int(valid.sum()),
                        'first_eligible_elapsed_s':(release[first]+1)/hz,'last_visible_native':int(at[first]),
                        'prefix_and_future_changes_checked':6})
    output={'scope':'Synthetic timing and numerical equivalence only. 64 groups of four causal 10 ms blocks, '
                'each group averages measured six-axis inputs and takes the last VQF state. The extra branch stays zero '
                'until 256 visible base blocks are available. No supervision, parameter updates, or formal records.',
            'rates':reports}
    (ROOT/'review/multirate_block_audit.json').write_text(json.dumps(output,indent=2)+'\n',encoding='utf8')
    print(reports)


if __name__=='__main__':
    main()
