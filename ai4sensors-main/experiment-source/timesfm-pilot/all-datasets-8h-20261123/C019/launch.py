"""Register C019 and dispatch its sole training and replay in the same process."""
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
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def write(name, payload):
    target = HERE / name
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(target)

registration = {"pid": os.getpid(), "started_at": datetime.now().astimezone().isoformat(),
                "scope": "C019 only: one B readout on frozen YT replacement panel, then 213 development records.",
                "command": "python -B C019/launch.py"}
with (HERE / "launch.json").open("x", encoding="utf-8") as stream:
    json.dump(registration, stream, indent=2)
print("C019 registered", registration, flush=True)
try:
    files = ("review/yt_training_panel.json", "C001/inventory.json", "C004/protocol.json",
             "C005/scoring-masks.npz", "C006/gate-traces.npz", "C014/protocol.json", "C015/sampling.npz",
             "C017/coefficient-lock.json", "C017/replay_stable_force.py", "C018/run_pair.py",
             "C018/process.json", "C018/results.json", "C018/pair-lock.json", "C018/B-readout.pt",
             "C018/training-inputs.npz", "C018/training-preparation.json", "C018/training.json")
    hashes = {name: sha(ROOT / name) for name in files}
    with (ROOT / "C018/process.json").open(encoding="utf-8") as stream:
        previous = json.load(stream)
    with (ROOT / "C018/pair-lock.json").open(encoding="utf-8") as stream:
        pair = json.load(stream)
    assert previous["status"] == "complete" and previous["results_sha256"] == hashes["C018/results.json"]
    assert previous["pair_lock_sha256"] == hashes["C018/pair-lock.json"]
    assert pair["readout_sha256"]["B"] == hashes["C018/B-readout.pt"]
    assert pair["C017_coefficient_lock_sha256"] == hashes["C017/coefficient-lock.json"]
    write("input-verification.json", {**registration, "files_sha256": hashes})
    print("C019 verified immutable YT panel, C017 lock, C018-B and development masks", flush=True)
    worker = HERE / "run_panel.py"
    end = time.monotonic() + 1200
    while not worker.is_file():
        if time.monotonic() >= end:
            raise TimeoutError("C019 worker did not become available")
        time.sleep(.25)
    runpy.run_path(str(worker), run_name="__main__")
except BaseException as exc:
    write("failure.json", {**registration, "failed_at": datetime.now().astimezone().isoformat(), "error": repr(exc)})
    raise
