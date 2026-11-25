"""Replay the two previously frozen neural models with their original training-side forward and quaternion transport."""
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

from davis_handheld import Davis
from lock_from_frontier import ROOT,REVIEW,read,digest
from native_blocks import block_inputs
from native_catalog import Catalog
from penn_handheld import Penn
from supplemental_data import Supplemental

TRANSFER=ROOT.parent/"transfer-mamba-20261123"
sys.path.insert(0,str(TRANSFER/"batch8"))
import final_score as archive
from tilt_eval import normalize,rotate,tilt_error

UP=np.array([0.,0.,1.])


@torch.inference_mode()
def score(payload,models):
    raw,hz,body_ref,allowed=payload
    seen=np.isfinite(raw).all(1)
    ids=np.flatnonzero(allowed & seen).astype("<i8")
    if hz>=100.:
        quat,last,release,hist,_,_=archive.base.causal_inputs(raw,hz)
        block_method="original"
    else:
        past=np.maximum.accumulate(np.where(seen,np.arange(len(raw)),-1))
        x=raw[past.clip(min=0)].copy()
        x[past<0]=0.
        orientations,hist,last,release=block_inputs(x,hz)
        quat=orientations.as_quat(scalar_first=True)
        block_method="causal_low_rate_extension"
    vertical=archive.base.body_up(quat[last],archive.base.EST_UP).astype("f4")
    target=rotate(normalize(quat[ids]),body_ref[ids])
    held=np.searchsorted(release,ids,side="right")
    estimates={"full_vqf_deg":np.broadcast_to(UP,target.shape)}
    frames=archive.intrinsic.intrinsic_windows(hist)[0]
    for label,(model,mean,std) in models.items():
        if label=="transfer_mamba":
            body_delta=archive.intrinsic.predict(model,hist,frames,mean,std)
        else:
            body_delta=archive.strict_predict(model,hist,vertical,mean,std)
        world=normalize(UP+rotate(normalize(quat[last]),body_delta))
        estimates[label]=np.vstack((UP,world))[held]
    values={name:float(np.sqrt(np.mean(tilt_error(v,target)**2))) for name,v in estimates.items()}
    return {"scored_frames":len(ids),"scoring_indices_sha256":hashlib.sha256(ids.tobytes()).hexdigest(),
            "block_method":block_method,"rate_hz":hz,**values}


def main():
    torch.set_num_threads(2)
    previous,weights_sha=archive.verify_lock()
    models=archive.load_model_set(previous,torch.device("cpu"))
    spec=read(REVIEW/"historical-baseline-anchors.json")
    models={name:models[name] for name in spec["methods"]}
    lock=ROOT/"selection-lock.json"
    assert spec["selection_sha256"]==digest(lock) and spec["original_lock_sha256"]==archive.LOCK_SHA
    primary=Catalog(lock);extra=Supplemental(lock);davis=Davis(lock);penn=Penn(lock)
    old={}
    with (TRANSFER/"batch8/all-sequences.csv").open(encoding="utf-8-sig")as src:
        old={r["sequence"]:r for r in csv.DictReader(src)}
    reference={}
    for filename in ("main.csv","transfer.csv"):
        with (ROOT/"final"/filename).open(encoding="utf-8-sig")as stream:
            reference.update({(r["dataset"],r["record"]):r for r in csv.DictReader(stream)})
    rows=[]
    for domain,names in spec["anchors"].items():
        for name in names:
            if domain=="DAVIS": raw=davis.native(name)[0]
            elif domain=="PennCOSYVIO":raw=penn.native(name)[0]
            elif domain in ("RIDI","EuRoC"):raw=extra.native(domain,name)[0]
            else:raw=primary.native(domain,name)[0]
            result=score(raw,models)
            frozen=reference[domain,name]
            assert result["scoring_indices_sha256"]==frozen["scoring_indices_sha256"] and result["scored_frames"]==int(frozen["scored_frames"])
            assert abs(result["full_vqf_deg"]-float(frozen["full_vqf_deg"]))<2e-4
            if domain=="SHICE":
                for key in models:
                    assert abs(result[key]-float(old[name][key]))<2e-4
            rows.append({"dataset":domain,"record":name,**result})
            print(domain,name,round(result["transfer_mamba"],5),round(result["strict_tcn"],5),flush=True)
    assert len(rows)==spec["count"]
    out={"selection_sha256":digest(lock),"baseline_lock_sha256":archive.LOCK_SHA,"preregistered_anchor_sha256":digest(REVIEW/"historical-baseline-anchors.json"),
         "forward_scripts":{str(p):digest(p) for p in (Path(__file__),Path(archive.__file__),Path(archive.intrinsic.__file__),Path(archive.base.__file__))},
         "source_checkpoint_hashes":{name:previous["models"][name]["sha256"] for name in spec["methods"]},"records":rows}
    (REVIEW/"previous-model-native-anchors.json").write_text(json.dumps(out,indent=2)+"\n",encoding="utf8")


if __name__=="__main__":
    main()
