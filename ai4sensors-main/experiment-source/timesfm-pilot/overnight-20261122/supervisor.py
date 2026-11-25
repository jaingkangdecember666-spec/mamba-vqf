"""Run, checkpoint and supervise the bounded overnight SHICE study."""

import importlib
import json
import msvcrt
import os
import socket
import subprocess
import sys
from pathlib import Path
from time import sleep

from common import BATCH, END, EXCLUDED, LOCK, START, fingerprint, names, now, read_json, stamp, write_json


LEDGER = BATCH / "ledger.jsonl"
STATUS = BATCH / "status.json"
ROUNDS = BATCH / "rounds"
FINISHED = {"completed", "reused", "failed", "cutoff", "rejected"}
HOST = socket.gethostname()


def log(event, data):
    with LEDGER.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps({"event": event, "recorded_at": stamp(), **data}, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def heartbeat(phase, item=None, worker_pid=None):
    write_json(STATUS, {"host": HOST, "supervisor_pid": os.getpid(), "worker_pid": worker_pid,
                       "phase": phase, "round_id": item["round_id"] if item else None,
                       "heartbeat_at": stamp(), "window_start": START.isoformat(),
                       "selection_deadline": LOCK.isoformat(), "window_end": END.isoformat()})


def index():
    return [read_json(p) for p in sorted(ROUNDS.glob("*/round.json"))]


def persist(item, event):
    write_json(ROUNDS / item["round_id"] / "round.json", item)
    log(event, item)
    write_json(BATCH / "round_index.json", index())


def schedule(config, hypothesis, batch, relation=None):
    items = index()
    unique = fingerprint(config)
    for item in items:
        if item["config_sha256"] == unique:
            return item
    latest = max((int(row["round_id"][1:]) for row in items if row["round_id"].startswith("R")), default=0)
    identity = f"R{max(latest, len(items)) + 1:05d}"
    item = {"round_id": identity, "batch": batch, "hypothesis": hypothesis, "config": config,
            "config_sha256": unique, "seed": config.get("seed"), "relation": relation,
            "partitions": {"development": names("dev"), "r5": names("r5"), "r6": names("r6"),
                           "excluded": list(EXCLUDED), "test_known_from_history": True},
            "planned_at": stamp(), "started_at": None, "ended_at": None, "status": "planned", "attempts": 0,
            "dev_metrics": None, "result": f"rounds/{identity}/result.json",
            "stdout": f"rounds/{identity}/run.log"}
    persist(item, "planned")
    return item


def lock_occupied(path):
    if not path.is_file():
        return False
    with path.open("a+b") as file:
        file.seek(0)
        try:
            msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            return True
        msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)
        return False


def finish(item, status, returncode=None):
    result_path = BATCH / item["result"]
    item.update(status=status, ended_at=stamp(), return_code=returncode)
    if result_path.is_file():
        item["dev_metrics"] = read_json(result_path).get("dev_metrics")
    else:
        write_json(result_path, {"status": status, "round_id": item["round_id"],
                                 "ended_at": item["ended_at"], "return_code": returncode,
                                 "stdout": item["stdout"]})
    persist(item, status)
    print(f"{stamp()} {item['round_id']} {status} {item['config']['kind']}", flush=True)


def execute(item, cutoff):
    if item["status"] in FINISHED:
        return
    if now() >= cutoff:
        finish(item, "cutoff")
        return
    directory = ROUNDS / item["round_id"]
    result = BATCH / item["result"]
    if item["status"] == "running":
        while lock_occupied(directory / "worker.lock") and now() < cutoff:
            heartbeat("waiting_for_orphan", item)
            sleep(2)
        if result.exists():
            finish(item, "completed", 0)
            return
        if now() >= cutoff:
            finish(item, "cutoff")
            return
        log("interrupted_attempt", item)

    item.update(status="running", started_at=item["started_at"] or stamp(),
                attempt_started_at=stamp(), attempts=item["attempts"] + 1, stop_at=cutoff.isoformat())
    persist(item, "running")
    with (BATCH / item["stdout"]).open("ab", buffering=0) as stdout:
        process = subprocess.Popen([sys.executable, "-u", str(BATCH / "worker.py"),
                                    str(directory / "round.json"), cutoff.isoformat()],
                                   stdout=stdout, stderr=subprocess.STDOUT, cwd=BATCH,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        print(f"{stamp()} {item['round_id']} worker {process.pid} {item['config']['kind']}", flush=True)
        while process.poll() is None:
            heartbeat("running", item, process.pid)
            if now() >= cutoff:
                process.terminate()
                process.wait(timeout=10)
                finish(item, "cutoff", process.returncode)
                return
            sleep(3)
    status = "completed" if process.returncode == 0 and result.is_file() else "failed"
    if process.returncode == 124:
        status = "cutoff"
    finish(item, status, process.returncode)


def supervise():
    if HOST.upper() != "FZR-MINIPC":
        raise RuntimeError(f"Expected FZR-MINIPC, observed {HOST}")
    if not START <= now() < END:
        raise RuntimeError("Runner was started outside the requested experiment window")
    ROUNDS.mkdir(exist_ok=True)
    log("supervisor_started", {"pid": os.getpid(), "host": HOST,
                               "window_start": START.isoformat(), "window_end": END.isoformat()})
    write_json(BATCH / "round_index.json", index())
    first = schedule({"kind": "baseline", "seed": None}, "Recompute causal native-rate FullVQF and verify historical R5", "baseline")
    execute(first, LOCK)
    if read_json(ROUNDS / first["round_id"] / "round.json")["status"] != "completed":
        raise RuntimeError("Baseline did not complete; inspect the round log before starting model search")

    strategy = None
    while now() < END:
        if not (BATCH / "strategy.py").is_file():
            heartbeat("waiting_for_strategy")
            sleep(5)
            continue
        if strategy is None:
            strategy = importlib.import_module("strategy")
        unfinished = [item for item in index() if item["status"] not in FINISHED]
        if unfinished:
            item = unfinished[0]
            execute(item, END if item["config"]["kind"] == "final" else LOCK)
            continue
        decision = strategy.next_job(index())
        if decision is None:
            heartbeat("completed")
            log("supervisor_completed", {"pid": os.getpid(), "host": HOST})
            return
        item = schedule(*decision)
        if item["status"] in FINISHED:
            raise RuntimeError(f"Strategy repeated a finished configuration: {item['round_id']}")
        execute(item, END if item["config"]["kind"] == "final" else LOCK)
    heartbeat("deadline")
    log("deadline", {"pid": os.getpid(), "host": HOST})


if __name__ == "__main__":
    with (BATCH / "supervisor.lock").open("a+b") as lock:
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            raise SystemExit("This batch already has a supervisor")
        try:
            supervise()
        except Exception as error:
            heartbeat("failed")
            log("supervisor_failed", {"error": repr(error), "pid": os.getpid(), "host": HOST})
            raise
