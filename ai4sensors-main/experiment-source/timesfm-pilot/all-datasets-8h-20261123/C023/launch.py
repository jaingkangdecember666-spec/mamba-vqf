"""Register C023 and launch its one readout fit in the same process."""
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
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def write(name, value):
    path = HERE / name
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)

receipt = {"pid": os.getpid(), "started_at": datetime.now().astimezone().isoformat(),
           "scope": "C023 only: one 50-parameter causal softmax mixture; frozen C022 slots; 213 development records after lock.",
           "command": "python -B C023/launch.py"}
with (HERE / "launch.json").open("x", encoding="utf-8") as f:
    json.dump(receipt, f, indent=2)
print("C023 registered", receipt, flush=True)
try:
    files = ("review/source_response_routing.json", "review/vqf_response_controls_development.json",
             "C001/inventory.json", "C005/scoring-masks.npz", "C006/gate-traces.npz", "C019/training-inputs.npz",
             "C019/readout.pt", "C019/sampling-lock.json", "C019/optimizer-steps.jsonl", "C015/sampling.npz",
             "C020/branch-lock.json", "C020/mamba-end-to-end.pt", "C020/sequences.csv", "C022/protocol.json",
             "C022/process.json", "C022/coefficient-lock.json", "C022/training-slots.npz", "C022/external-selection.json",
             "C022/results.json", "C022/run_filter_pair.py")
    hashes = {n: sha(ROOT / n) for n in files}
    read=lambda f:json.loads((ROOT / f).read_text(encoding="utf-8"))
    proc, lock = read("C022/process.json"), read("C022/coefficient-lock.json")
    assert proc["status"] == "complete" and proc["results_sha256"] == hashes["C022/results.json"]
    assert proc["coefficient_lock_sha256"] == hashes["C022/coefficient-lock.json"] and lock["training_slots_sha256"] == hashes["C022/training-slots.npz"]
    assert lock["external_selection_sha256"] == hashes["C022/external-selection.json"] and lock["C020_branch_lock_sha256"] == hashes["C020/branch-lock.json"]
    write("input-verification.json", {**receipt, "files_sha256": hashes})
    print("C023 immutable C022 and C019 locks verified", flush=True)
    worker = HERE / "run_router.py"
    deadline = time.monotonic() + 1200
    while not worker.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError("C023 worker was not provided")
        time.sleep(.25)
    runpy.run_path(str(worker), run_name="__main__")
except BaseException as exc:
    write("failure.json", {**receipt, "failed_at": datetime.now().astimezone().isoformat(), "error": repr(exc)})
    (HERE / "failure-traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
    raise
