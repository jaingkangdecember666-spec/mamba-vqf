"""Register C022 and run its sole scalar fit and replay in the registered process."""
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

def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()

def write(name, value):
    path = HERE / name
    stage = path.with_suffix(".tmp")
    stage.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    stage.replace(path)

record = {"pid": os.getpid(), "started_at": datetime.now().astimezone().isoformat(),
          "scope": "C022 only, fixed C020, one 2:1 filter-response scalar fit, original 213 development records",
          "command": "python -B C022/launch.py"}
with (HERE / "launch.json").open("x", encoding="utf-8") as f:
    json.dump(record, f, indent=2)
print("C022 registered", record, flush=True)
try:
    inputs = ("review/auxiliary_filter_directions.json", "review/auxiliary_filter_directions.py", "review/filter_response_bank.py",
              "C001/inventory.json", "C003/online_bias.py", "C005/scoring-masks.npz", "C006/gate-traces.npz", "C011/minimax.py",
              "C015/sampling.npz", "C017/coefficient-lock.json", "C017/replay_stable_force.py", "C018/B-readout.pt",
              "C019/readout.pt", "C019/sampling-lock.json", "C019/training-inputs.npz", "C019/optimizer-steps.jsonl",
              "C020/branch-lock.json", "C020/mamba-end-to-end.pt", "C020/protocol.json", "C020/process.json", "C020/results.json",
              "C020/training-preparation.json", "C020/training-canonical-history.npy", "C020/optimizer-steps.jsonl",
              "C020/run_temporal.py")
    files = {n: sha(ROOT / n) for n in inputs}
    read=lambda p:json.loads((ROOT / p).read_text(encoding="utf-8"))
    state, lock = read("C020/process.json"), read("C020/branch-lock.json")
    assert state["status"] == "complete" and state["results_sha256"] == files["C020/results.json"]
    assert state["branch_lock_sha256"] == files["C020/branch-lock.json"]
    assert lock["C020_full_branch_sha256"] == files["C020/mamba-end-to-end.pt"]
    assert lock["C019_sampling_lock_sha256"] == files["C019/sampling-lock.json"]
    assert lock["C017_coefficient_lock_sha256"] == files["C017/coefficient-lock.json"]
    write("input-verification.json", {**record, "files_sha256": files})
    print("C022 verified fixed C020, causal frontend and 139-record external sampling lock", flush=True)
    worker = HERE / "run_filter_pair.py"
    deadline = time.monotonic() + 1200
    while not worker.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError("C022 worker was not provided")
        time.sleep(.25)
    runpy.run_path(str(worker), run_name="__main__")
except BaseException as exc:
    write("failure.json", {**record, "failed_at": datetime.now().astimezone().isoformat(), "error": repr(exc)})
    (HERE / "failure-traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
    raise
