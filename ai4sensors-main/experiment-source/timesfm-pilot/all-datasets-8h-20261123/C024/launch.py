"""Register C024 and launch its only DAVIS-augmented fit in this PID."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import runpy
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DAVIS = "external-sources/davis-handheld"

def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()

def write(name, x):
    path = HERE / name
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(x, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)

r = {"pid": os.getpid(), "started_at": datetime.now().astimezone().isoformat(),
     "scope": "C024 only; six DAVIS external_train captures, one 50-param fit, original 213 development records",
     "command": "python -B C024/launch.py"}
with (HERE / "launch.json").open("x", encoding="utf-8") as f:
    json.dump(r, f, indent=2)
print("C024 registered", r, flush=True)
try:
    files = (f"{DAVIS}/preprocessing-lock.json", f"{DAVIS}/protocol.json", f"{DAVIS}/source-files.json", f"{DAVIS}/external-fixed-vqf.json",
             "review/davis_handheld.py", "review/vqf_response_controls_development.json",
             "C001/inventory.json", "C005/scoring-masks.npz", "C006/gate-traces.npz", "C015/sampling.npz",
             "C019/readout.pt", "C019/sampling-lock.json", "C019/training-inputs.npz", "C019/optimizer-steps.jsonl",
             "C020/sequences.csv", "C020/branch-lock.json", "C022/protocol.json", "C022/coefficient-lock.json", "C022/external-selection.json",
             "C022/training-slots.npz", "C023/process.json", "C023/protocol.json", "C023/results.json", "C023/run_router.py",
             "C023/router-lock.json", "C023/router.pt", "C023/optimizer-steps.jsonl")
    hashes = {n: sha(ROOT / n) for n in files}
    def read(n):
        return json.loads((ROOT/n).read_text(encoding="utf-8"))
    s,b,d=(read(n) for n in ("C023/process.json","C023/router-lock.json",f"{DAVIS}/preprocessing-lock.json"))
    assert s["status"]=="complete" and s["results_sha256"]==hashes["C023/results.json"]
    assert s["router_lock_sha256"]==hashes["C023/router-lock.json"] and b["checkpoint_sha256"]==hashes["C023/router.pt"]
    assert b["C022_coefficient_lock_sha256"]==hashes["C022/coefficient-lock.json"]
    assert d["protocol_sha256"]==hashes[f"{DAVIS}/protocol.json"] and d["source_bytes_manifest_sha256"]==hashes[f"{DAVIS}/source-files.json"]
    assert d["loader_sha256"]==hashes["review/davis_handheld.py"] and d["unseen_numerical_records"]==3 and len(d["source_records"])==6
    write("input-verification.json",{**r,"files_sha256":hashes})
    print("C024 verified the untouched DAVIS training lock and original router inputs", flush=True)
    worker=HERE/"run_davis_router.py"
    end=time.monotonic()+1300
    while not worker.is_file():
        if time.monotonic()>=end:
            raise TimeoutError("C024 worker was not provided")
        time.sleep(.25)
    runpy.run_path(str(worker),run_name="__main__")
except BaseException as exc:
    write("failure.json",{**r,"failed_at":datetime.now().astimezone().isoformat(),"error":repr(exc)})
    (HERE/"failure-traceback.log").write_text(traceback.format_exc(),encoding="utf-8")
    raise
