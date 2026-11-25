"""Identify matched external source timestamps available before and after adding measured horizontal motion."""
import json

import numpy as np
import torch

from check_sources import ROOT
from euroc_paired_sources import NAMES
from external_translation_variation import motion
from lock_from_frontier import digest
from native_blocks import block_inputs
from optical_score import canonical_frame
from source_motion_router_diagnostic import native


def main():
    torch.set_num_threads(1)
    path=ROOT/'C026/training-inputs.npz';results=[]
    with np.load(path)as f:
        chosen=f['native_release'][:,104:].copy()
    for j,name in enumerate(NAMES):
        sources,translation,mask,info=motion(name)
        prior=chosen[:,2*j]
        np.testing.assert_array_equal(prior,chosen[:,2*j+1])
        assert len(np.unique(prior))==160
        kept=mask[prior].copy()
        for signal,hz,up,_ in sources:
            for scale in (0.,1.):
                xx=signal.copy();xx[:,:3]+=scale*translation
                raw=(xx,hz,up,mask)
                corrected,_=native(raw)
                _,win,inputs,releases=block_inputs(corrected,hz)
                i=np.searchsorted(releases,prior)
                np.testing.assert_array_equal(releases[i],prior)
                kept &= canonical_frame(win[i])[2]
                assert (inputs[i]<=prior).all()
        results.append({'record':name,'original_native_releases':len(prior),'native_releases_after_common_checks':int(kept.sum()),
                        'original_step_indices':np.flatnonzero(kept).tolist(),'native_indices':prior[kept].tolist(),
                        'source_state_sha256':info['source_state_sha256'],'source_stream_sha256':info['source_stream_sha256']})
    out={'scope':'Only the already drawn 160 C026 V1 release timestamps per original recording. Keep a timestamp when it has valid measured velocity coverage, native reference and full intrinsic frames in both gyro bias conditions before and after adding the original observed horizontal translation once. Neither a model nor any held-out recording is used for this check.',
         'C026_source_sha256':digest(path),'variation_adapter_sha256':digest(ROOT/'review/external_translation_variation.py'),'records':results}
    (ROOT/'review/translation_training_slots.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
    print([(r['record'],r['native_releases_after_common_checks'])for r in results])


if __name__=='__main__':main()
