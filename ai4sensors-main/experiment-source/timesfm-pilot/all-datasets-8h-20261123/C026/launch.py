"""Register and execute the only C026 fit in one process."""
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
NAMES=("V1_01_easy","V1_02_medium","V1_03_difficult")

def sha(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()

def read(p):
    return json.loads((R/p).read_text(encoding="utf-8"))

def write(n,value):
    p=H/n
    stage=p.with_suffix(".tmp")
    stage.write_text(json.dumps(value,indent=2)+"\n",encoding="utf-8")
    stage.replace(p)

receipt={"pid":os.getpid(),"started_at":datetime.now().astimezone().isoformat(),
         "scope":"C026 only; three paired EuRoC V1 tracks, exact C025 slots, one training fit, original development panel",
         "command":"python -B C026/launch.py"}
with (H/"launch.json").open("x",encoding="utf-8") as f:
    json.dump(receipt,f,indent=2)
print("C026 registered",receipt,flush=True)
try:
    filenames=["review/euroc_paired_sources.py","review/check_euroc_source.py","review/euroc_bias_response.json",
               "review/router_motion_context.py","review/vqf_response_controls_development.json",
               "external-sources/euroc-original-inputs/source-audit.json","C001/inventory.json","C005/scoring-masks.npz","C006/gate-traces.npz",
               "C019/readout.pt","C020/branch-lock.json","C020/sequences.csv","C022/protocol.json","C022/coefficient-lock.json",
               "C023/router.pt","C024/router.pt","C024/sampling-lock.json","C024/training-inputs.npz","C024/training-io.npz",
               "C024/zero-step-audit.json","C025/run_context_router.py","C025/protocol.json","C025/results.json","C025/process.json",
               "C025/router-lock.json","C025/router.pt","C025/optimizer-steps.jsonl","C025/training.json","C025/training-context.npz",
               "C025/training-io.npz","C025/context-normalization.json"]
    filenames += [f"external-sources/euroc-original-inputs/{n}.npz" for n in NAMES]
    filenames += [f"external-sources/euroc-vicon-room1/{n}.hdf5" for n in NAMES]
    digests={n:sha(R/n) for n in filenames}
    s,b,tr=(read(n) for n in ("C025/process.json","C025/router-lock.json","C025/training.json"))
    assert s["status"]=="complete" and s["results_sha256"]==digests["C025/results.json"]
    assert s["router_lock_sha256"]==digests["C025/router-lock.json"] and b["checkpoint_sha256"]==digests["C025/router.pt"]
    assert b["train_context_sha256"]==digests["C025/training-context.npz"] and b["training_sha256"]==digests["C025/training.json"]
    assert tr["artifacts_sha256"]["training-io.npz"]==digests["C025/training-io.npz"]
    refs={x["record"]:x for x in read("review/euroc_bias_response.json")["records"]}
    for n in NAMES:
        assert digests[f"external-sources/euroc-original-inputs/{n}.npz"]==refs[n]["original_source_sha256"]
        assert digests[f"external-sources/euroc-vicon-room1/{n}.hdf5"]==refs[n]["conversion_sha256"]
    write("input-verification.json",{**receipt,"files_sha256":digests})
    print("C026 locked C025 inputs and verified the three V1 origins",flush=True)
    worker=H/"run_paired_router.py"
    deadline=time.monotonic()+1400
    while not worker.is_file():
        if time.monotonic()>=deadline:
            raise TimeoutError("C026 worker was not provided")
        time.sleep(.25)
    runpy.run_path(str(worker),run_name="__main__")
except BaseException as exc:
    write("failure.json",{**receipt,"failed_at":datetime.now().astimezone().isoformat(),"error":repr(exc)})
    (H/"failure-traceback.log").write_text(traceback.format_exc(),encoding="utf-8")
    raise
