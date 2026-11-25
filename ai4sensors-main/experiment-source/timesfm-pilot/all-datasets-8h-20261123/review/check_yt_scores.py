"""Read only the twelve prespecified YT validation rows and replay the unmodified model."""
import json
import sys
import numpy as np
import torch
from check_sources import ROOT, TRANSFER, measured, weights, yt


def main():
    torch.set_num_threads(2)
    split=json.loads((ROOT/'C001/inventory.json').read_text('utf8'))
    chosen=split['yt']['selected_valid']
    assert len(chosen)==len({name.rsplit('_',1)[0] for name in chosen})==12
    head=weights(TRANSFER/'batch6/mamba-intrinsic-selected.pt','mamba_v1')
    out=[]
    for name in chosen:
        raw=yt(name)
        row=measured(raw,head)
        signal,_,reference,eligible=raw
        acc,w=signal[:,:3],signal[:,3:]
        quiet=eligible & (np.linalg.norm(w,axis=1)<.10) & (np.abs(np.linalg.norm(acc,axis=1)-9.80665)<.30)
        a,b=acc[quiet],reference[quiet]
        angles=np.rad2deg(np.arctan2(np.linalg.norm(np.cross(a,b),axis=1),(a*b).sum(axis=1)))
        out.append({'record':name,'valid_quiet_frames':int(quiet.sum()),'median_quiet_deg':float(np.median(angles)) if len(angles) else None,**row})
    (ROOT/'review/yt_development_replay.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
    print({k:float(np.mean([x['rmse_deg'][k] for x in out])) for k in ('candidate','full_vqf')})
    print('quiet_deg',[x['median_quiet_deg'] for x in out])


if __name__=='__main__':
    main()
