"""Register the sole C025 PID before training or opening development inputs."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import runpy
import time
import traceback

H=Path(__file__).resolve().parent
R=H.parent

def sha(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()

def read(n):
    return json.loads((R/n).read_text(encoding="utf-8"))

def write(n,x):
    p=H/n
    tmp=p.with_suffix(".tmp")
    tmp.write_text(json.dumps(x,indent=2)+"\n",encoding="utf-8")
    tmp.replace(p)

receipt={"pid":os.getpid(),"started_at":datetime.now().astimezone().isoformat(),
         "scope":"C025 only: fixed 145-record C024 training panel, one 58-parameter fit, 213 development replays",
         "command":"python -B C025/launch.py"}
with (H/"launch.json").open("x",encoding="utf-8") as f:
    json.dump(receipt,f,indent=2)
print("C025 registered",receipt,flush=True)
try:
    files=("review/router_motion_context.py","review/source_motion_router_features.npz","review/source_motion_router_diagnostic.json",
           "review/vqf_response_controls_development.json","C001/inventory.json","C005/scoring-masks.npz","C006/gate-traces.npz",
           "C019/readout.pt","C020/branch-lock.json","C020/sequences.csv","C022/coefficient-lock.json","C022/protocol.json",
           "C023/router-lock.json","C023/router.pt","C024/process.json","C024/router-lock.json","C024/results.json",
           "C024/router.pt","C024/protocol.json","C024/sampling-lock.json","C024/sampling-steps.jsonl","C024/optimizer-steps.jsonl",
           "C024/training-inputs.npz","C024/training-io.npz","C024/training.json","C024/zero-step-audit.json","C024/davis-new-slots.npz",
           "C024/training-preparation.json","C024/run_davis_router.py")
    stamps={n:sha(R/n) for n in files}
    s,b,sl,x,d=(read(n) for n in ("C024/process.json","C024/router-lock.json","C024/sampling-lock.json",
                                   "C024/training.json","review/source_motion_router_diagnostic.json"))
    assert s["status"]=="complete" and s["results_sha256"]==stamps["C024/results.json"]
    assert s["router_lock_sha256"]==stamps["C024/router-lock.json"] and b["checkpoint_sha256"]==stamps["C024/router.pt"]
    assert b["sampling_lock_sha256"]==stamps["C024/sampling-lock.json"] and sl["training_inputs_sha256"]==stamps["C024/training-inputs.npz"]
    assert b["training_sha256"]==stamps["C024/training.json"] and x["artifacts_sha256"]["training-io.npz"]==stamps["C024/training-io.npz"]
    assert d["native_features_sha256"]==stamps["review/source_motion_router_features.npz"] and len(d["davis"])==6
    write("input-verification.json",{**receipt,"files_sha256":stamps})
    print("C025 verified C024 sampling/weights and source motion feature archive",flush=True)
    worker=H/"run_context_router.py"
    deadline=time.monotonic()+1400
    while not worker.is_file():
        if time.monotonic()>=deadline:
            raise TimeoutError("C025 worker was not provided")
        time.sleep(.25)
    runpy.run_path(str(worker),run_name="__main__")
except BaseException as exc:
    write("failure.json",{**receipt,"failed_at":datetime.now().astimezone().isoformat(),"error":repr(exc)})
    (H/"failure-traceback.log").write_text(traceback.format_exc(),encoding="utf-8")
    raise
