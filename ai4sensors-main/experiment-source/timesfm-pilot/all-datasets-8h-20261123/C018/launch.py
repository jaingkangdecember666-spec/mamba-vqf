"""Register and run the two C018 readouts sequentially in this process."""
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
    path = HERE / name
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)

registration = {"pid": os.getpid(), "started_at": datetime.now().astimezone().isoformat(),
                "scope": "C018 only. Sequential A and B fit on C015 slots, then 213 development replays. No final holdout.",
                "command": "python -B C018/launch.py"}
with (HERE / "launch.json").open("x", encoding="utf-8") as stream:
    json.dump(registration, stream, indent=2)
print("C018 registered", registration, flush=True)
try:
    inputs = ("C017/protocol.json", "C017/results.json", "C017/coefficient-lock.json", "C017/replay_stable_force.py",
              "C017/training-slots.npz", "C017/external-selection.json", "C015/sampling.npz",
              "C005/scoring-masks.npz", "C006/gate-traces.npz", "C014/protocol.json")
    hashes = {name: sha(ROOT / name) for name in inputs}
    assert hashes["C017/coefficient-lock.json"] == "72029a76c43fe22d82450a08ccdbcc6ad23dfedfcb2c0f446dbab0e65de2f8bb"
    with (ROOT / "C017/process.json").open(encoding="utf-8") as stream:
        prior = json.load(stream)
    assert prior["status"] == "complete" and prior["results_sha256"] == hashes["C017/results.json"]
    write("input-verification.json", {**registration, "files_sha256": hashes})
    print("C018 frozen inputs verified, including C017 lock", flush=True)
    worker = HERE / "run_pair.py"
    deadline = time.monotonic() + 1200
    while not worker.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError("C018 worker did not become available")
        time.sleep(.25)
    runpy.run_path(str(worker), run_name="__main__")
except BaseException as exc:
    write("failure.json", {**registration, "failed_at": datetime.now().astimezone().isoformat(), "error": repr(exc)})
    raise
