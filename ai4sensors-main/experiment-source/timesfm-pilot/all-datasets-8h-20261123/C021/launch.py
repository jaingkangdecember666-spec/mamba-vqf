"""Register C021 and launch its sole configuration in the registered process."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import runpy
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()

def write(name, value):
    target = HERE / name
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    tmp.replace(target)

record = {"pid": os.getpid(), "started_at": datetime.now().astimezone().isoformat(),
          "scope": "C021 only, 64 x 40 ms, exact C020 slots, one fit, 213 development records",
          "command": "python -B C021/launch.py"}
with (HERE / "launch.json").open("x", encoding="utf-8") as f:
    json.dump(record, f, indent=2)
print("C021 registered", record, flush=True)
try:
    inputs = ("review/multirate_blocks.py", "review/multirate_block_audit.json", "review/yt_training_panel.json",
              "C001/inventory.json", "C005/scoring-masks.npz", "C006/gate-traces.npz", "C015/sampling.npz",
              "C017/coefficient-lock.json", "C017/replay_stable_force.py", "C018/B-readout.pt", "C018/run_pair.py",
              "C019/training-inputs.npz", "C019/sampling-lock.json", "C019/readout.pt", "C019/optimizer-steps.jsonl",
              "C020/branch-lock.json", "C020/mamba-end-to-end.pt", "C020/training.json", "C020/training-preparation.json",
              "C020/training-canonical-history.npy", "C020/optimizer-tensors.npz", "C020/optimizer-steps.jsonl",
              "C020/run_temporal.py", "C020/process.json", "C020/protocol.json", "C020/results.json")
    files = {n: sha(ROOT / n) for n in inputs}
    def c20(name):
        return json.loads((ROOT / "C020" / name).read_text(encoding="utf-8"))
    s, branch, prep = (c20(n) for n in ("process.json", "branch-lock.json", "training-preparation.json"))
    assert s["status"] == "complete" and s["results_sha256"] == files["C020/results.json"]
    assert s["branch_lock_sha256"] == files["C020/branch-lock.json"]
    assert branch["C020_full_branch_sha256"] == files["C020/mamba-end-to-end.pt"]
    assert branch["C019_sampling_lock_sha256"] == files["C019/sampling-lock.json"]
    assert branch["C017_coefficient_lock_sha256"] == files["C017/coefficient-lock.json"]
    assert prep["history_sha256"] == files["C020/training-canonical-history.npy"]
    write("input-verification.json", {**record, "files_sha256": files})
    print("C021 verified coarse helper, C020 branch and original C019 sample lock", flush=True)
    worker = HERE / "run_coarse.py"
    deadline = time.monotonic() + 1500
    while not worker.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError("C021 worker did not become available")
        time.sleep(.25)
    runpy.run_path(str(worker), run_name="__main__")
except BaseException as exc:
    write("failure.json", {**record, "failed_at": datetime.now().astimezone().isoformat(), "error": repr(exc)})
    raise
