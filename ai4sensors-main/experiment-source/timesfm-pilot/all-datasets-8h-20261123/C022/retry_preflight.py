"""Re-register C022 after an unused reference diagnostic script changed before any fit."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import traceback

H=Path(__file__).resolve().parent
R=H.parent
def sha(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()

def read(p):
    return json.loads(p.read_text(encoding="utf-8"))

assert not (H/"coefficient-lock.json").exists() and not (H/"training-slots.npz").exists() and not (H/"protocol.json").exists()
orig=read(H/"input-verification.json")
files=dict(orig["files_sha256"])
reference_script="review/auxiliary_filter_directions.py"
original=files.pop(reference_script)
changed=sha(R/reference_script)
assert all(sha(R/name)==old for name,old in files.items())
record={"pid":os.getpid(),"started_at":datetime.now().astimezone().isoformat(),"original_preflight_pid":orig["pid"],
        "command":"python -B C022/retry_preflight.py","scope":orig["scope"],"previous_fit_count":0,
        "reference_script_receipt":{"path":reference_script,"old_sha256":original,"current_sha256":changed,
                                      "usage":"The script is not imported. Its frozen JSON and array outputs are the inputs."},
        "original_failure_sha256":sha(H/"failure.json"),"original_trace_sha256":sha(H/"failure-traceback.log")}
for name,value in (("retry-launch.json",record),("retry-input-verification.json",{**record,"files_sha256":files})):
    with (H/name).open("x",encoding="utf-8") as stream:
        json.dump(value,stream,indent=2)
print("C022 re-registered before any fit",record,flush=True)
sys.argv.append("--preflight-retry")
try:
    runpy.run_path(str(H/"run_filter_pair.py"),run_name="__main__")
except BaseException as exc:
    (H/"retry-failure.json").write_text(json.dumps({**record,"error":repr(exc)},indent=2)+"\n",encoding="utf-8")
    (H/"retry-failure-traceback.log").write_text(traceback.format_exc(),encoding="utf-8")
    raise
