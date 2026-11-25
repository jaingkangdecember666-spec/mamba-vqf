"""Register C020 and run its single fit and development replay in this process."""
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

def write(name, payload):
    path = HERE / name
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)

record = {"pid": os.getpid(), "started_at": datetime.now().astimezone().isoformat(),
          "scope": "C020 only, one end-to-end Mamba branch, original C019 samples and 213 development masks",
          "command": "python -B C020/launch.py"}
with (HERE / "launch.json").open("x", encoding="utf-8") as f:
    json.dump(record, f, indent=2)
print("C020 registered", record, flush=True)
try:
    paths = ("C008/protocol.json", "C008/fit_backbone.py", "C001/inventory.json", "C005/scoring-masks.npz", "C006/gate-traces.npz",
             "C015/sampling.npz", "C018/B-readout.pt", "C018/pair-lock.json", "C018/run_pair.py", "C017/coefficient-lock.json",
             "C017/replay_stable_force.py", "C019/process.json", "C019/protocol.json", "C019/training.json", "C019/training-inputs.npz",
             "C019/training-preparation.json", "C019/sampling-lock.json", "C019/readout-lock.json", "C019/readout.pt",
             "C019/results.json", "review/yt_training_panel.json")
    files = {p: sha(ROOT / p) for p in paths}
    def previous(name):
        return json.loads((ROOT / "C019" / name).read_text(encoding="utf-8"))
    s, rl, sl = (previous(x) for x in ("process.json", "readout-lock.json", "sampling-lock.json"))
    assert s["status"] == "complete" and s["results_sha256"] == files["C019/results.json"]
    assert s["readout_lock_sha256"] == files["C019/readout-lock.json"]
    assert s["sampling_lock_sha256"] == rl["sampling_lock_sha256"] == files["C019/sampling-lock.json"]
    assert rl["C019_weight_sha256"] == files["C019/readout.pt"] and sl["training_inputs_sha256"] == files["C019/training-inputs.npz"]
    assert rl["C017_coefficient_lock_sha256"] == files["C017/coefficient-lock.json"]
    write("input-verification.json", {**record, "files_sha256": files})
    print("C020 verified C019 training and readout locks", flush=True)
    worker = HERE / "run_temporal.py"
    deadline = time.monotonic() + 1500
    while not worker.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError("C020 worker did not become available")
        time.sleep(.25)
    runpy.run_path(str(worker), run_name="__main__")
except BaseException as exc:
    write("failure.json", {**record, "failed_at": datetime.now().astimezone().isoformat(), "error": repr(exc)})
    raise
