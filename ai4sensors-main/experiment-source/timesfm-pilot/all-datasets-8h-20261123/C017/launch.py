"""Register C017, check its frozen inputs, then run its worker in this process."""
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
import runpy
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
STARTED = datetime.now().astimezone().isoformat()

def write(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")

def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

registration = {"pid": os.getpid(), "started_at": STARTED, "stage": "checking_frozen_inputs",
                "scope": "C017 only. C015 actual external releases, then 213 original development records after scalar lock.",
                "command": "python -B C017/launch.py"}
with (HERE / "launch.json").open("x", encoding="utf-8") as stream:
    json.dump(registration, stream, indent=2)
print("C017 registered", registration, flush=True)
try:
    paths = ["C014/protocol.json", "C014/results.json", "C015/protocol.json", "C015/sampling.npz",
             "C016/training-windows.npz", "C005/scoring-masks.npz", "C006/gate-traces.npz",
             "C011/minimax.py", "C003/online_bias.py"]
    hashes = {name: sha(ROOT / name) for name in paths}
    prior = json.loads((ROOT / "C014/process.json").read_text(encoding="utf-8"))
    assert prior["status"] == "complete" and prior["results_sha256"] == hashes["C014/results.json"]
    write("input-verification.json", {**registration, "files_sha256": hashes, "stage": "initial_hashes_verified"})
    print("C017 frozen input hashing complete", flush=True)
    worker = HERE / "replay_stable_force.py"
    deadline = time.monotonic() + 900
    while not worker.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError("C017 worker did not become available within 900 seconds")
        time.sleep(.25)
    runpy.run_path(str(worker), run_name="__main__")
except BaseException as exc:
    write("failure.json", {"pid": os.getpid(), "started_at": STARTED, "failed_at": datetime.now().astimezone().isoformat(),
                           "error": repr(exc)})
    raise
