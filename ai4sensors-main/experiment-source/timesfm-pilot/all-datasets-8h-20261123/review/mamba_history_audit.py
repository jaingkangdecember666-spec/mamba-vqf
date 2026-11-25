"""Measure learned temporal sensitivity from real external training inputs, holding the intrinsic frame fixed."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat
from check_sources import DATA, ROOT, TRANSFER, weights
from native_blocks import block_inputs
from optical_score import canonical_frame


def external_windows():
    sources=json.loads((ROOT/'C004/protocol.json').read_text('utf8'))['source_records']
    broad=next(r for r in sources if r['domain']=='BROAD')
    sassari=next(r for r in sources if r['domain']=='Sassari')
    for r in (broad,sassari):
        path=DATA/r['relative_data_path']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==r['file_sha256']
        if r['domain']=='BROAD':
            mat=loadmat(path,variable_names=('imu_acc','imu_gyr','sampling_rate'))
            signal=np.c_[mat['imu_acc'],mat['imu_gyr']];hz=float(mat['sampling_rate'].item())
        else:
            sensor=r['record'].split('/')[1]
            mat=loadmat(path,variable_names=(sensor,))
            signal=mat[sensor][:,1:7];hz=100.
        yield r['domain']+'/'+r['record'],signal,hz
    npz=ROOT/'external-sources/euroc-original-inputs/V1_02_medium.npz'
    with np.load(npz) as raw:
        yield 'EuRoC/V1_02_medium',np.column_stack((raw['acc_m_s2'],raw['gyr_rad_s'])),200.


def summary(model,windows):
    network,avg,std,_=model
    framed,_,valid=canonical_frame(windows)
    assert valid.all()
    inputs=((torch.from_numpy(framed)-avg)/std).clamp(-12,12).detach().requires_grad_(True)
    output=network(inputs)
    # The intrinsic +z direction is the estimated vertical. Only the two transverse raw outputs alter tilt.
    energies=[]
    for dim in (0,1):
        gradient,=torch.autograd.grad(output[:,dim].sum(), inputs, retain_graph=dim==0)
        energies.append(gradient.square().sum(-1).detach().numpy())
    energy=np.sum(energies,axis=0)
    ratio=np.cumsum(energy[:,::-1],axis=1)/energy.sum(axis=1,keepdims=True)
    with torch.no_grad():
        masked=inputs.clone()
        masked[:,:-8]=inputs[:,-9:-8]
        effect=(network(masked)[:,:2]-output[:,:2]).norm(dim=-1).numpy()
    return {'windows':len(windows),'median_fraction_gradient_energy_last_blocks':
            {str(n):float(np.median(ratio[:,n-1])) for n in (1,2,4,8,16,32)},
            'median_p90_change_raw_transverse_output_when_oldest_56_blocks_flattened':
            np.quantile(effect,[.5,.9]).tolist()}


def main():
    torch.set_num_threads(2)
    models={'original_B6':weights(TRANSFER/'batch6/mamba-intrinsic-selected.pt','mamba_v1'),
            'C008':weights(ROOT/'C008/mamba-end-to-end.pt','mamba_v1')}
    out=[]
    for label,signal,hz in external_windows():
        finite=np.isfinite(signal).all(1)
        index=np.maximum.accumulate(np.where(finite,np.arange(len(signal)),-1))
        signal=signal[index.clip(min=0)].copy();signal[index<0]=0.
        _,windows,_,when=block_inputs(signal,hz)
        take=np.flatnonzero((when/hz>10.) & (windows[:,-1,12]==0.))
        take=take[np.linspace(0,len(take)-1,min(48,len(take)),dtype=int)]
        row={'source':label,'release_indices':when[take].tolist(),'models':{k:summary(v,windows[take]) for k,v in models.items()}}
        out.append(row)
        print(label, row['models']['C008'],flush=True)
    (ROOT/'review/mamba_history_audit.json').write_text(json.dumps({
        'scope':'External training inputs only. No pose labels, holdouts, optimizer steps or new estimator. Measured local '
        'normalized Jacobian energy through the Mamba at fixed intrinsic basis; this omits baseline/filter memory and '
        'window-based basis selection. Occlusion also holds the basis fixed.', 'block_seconds':.01, 'sources':out},indent=2)+'\n',encoding='utf8')


if __name__=='__main__':
    main()
