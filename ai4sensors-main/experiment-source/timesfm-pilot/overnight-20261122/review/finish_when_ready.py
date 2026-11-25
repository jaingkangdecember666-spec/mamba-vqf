"""Build the local report once the experiment supervisor has closed its official test."""

import os
import shutil
import subprocess
import time
from datetime import datetime, timedelta

import msvcrt

from audit_rounds import BATCH, HERE, read, write


RECEIPT = HERE / "completion-watch.json"


def stamp():
    return datetime.now().astimezone()


def run():
    end = datetime.fromisoformat(read(BATCH / "status.json")["window_end"])
    grace = end + timedelta(minutes=30)
    while stamp() < grace:
        status = read(BATCH / "status.json")
        receipt = {"pid":os.getpid(), "checked_at":stamp().isoformat(timespec="seconds"),
                   "experiment_phase":status["phase"], "experiment_window_end":end.isoformat(),
                   "status":"waiting_for_final_test"}
        write(RECEIPT, receipt)
        if status["phase"] in {"completed","deadline"}:
            index = read(BATCH / "round_index.json")
            finals = [r for r in index if r["config"]["kind"]=="final" and r["status"]=="completed"]
            if finals:
                assert len(finals)==1
                marker = HERE / "render_check.json"
                final_audit = HERE / "final_audit.json"
                already_built = (marker.is_file() and final_audit.is_file()
                                 and read(marker)["report_status"]=="complete"
                                 and read(final_audit)["final_id"]==finals[0]["round_id"])
                if not already_built:
                    write(RECEIPT, {**receipt,"status":"building_verified_report"})
                    shell = shutil.which("pwsh") or shutil.which("powershell")
                    if not shell:
                        raise RuntimeError("PowerShell is unavailable for the report build.")
                    with (HERE / "final-build.log").open("w",encoding="utf-8") as log:
                        subprocess.run([shell,"-NoProfile","-File",str(HERE / "build_review.ps1")],
                                       cwd=HERE, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=900)
                checked = read(marker)
                assert checked["report_status"]=="complete" and set(checked["viewports_css_px"])=={"1280","390"}
                write(RECEIPT,{**receipt,"status":"finished","final_round":finals[0]["round_id"],
                               "verified_at":stamp().isoformat(),"html":"report-src/dist/index.html"})
                return
        time.sleep(55)
    raise RuntimeError("The study has no completed formal evaluation and report at the review cutoff.")


if __name__=="__main__":
    with (HERE / "completion-watch.lock").open("a+b") as guard:
        guard.seek(0)
        try:
            msvcrt.locking(guard.fileno(),msvcrt.LK_NBLCK,1)
        except OSError:
            raise SystemExit("The one-time report watcher is already running.")
        try:
            run()
        except Exception as error:
            write(RECEIPT,{"pid":os.getpid(),"status":"requires_review","at":stamp().isoformat(),"error":repr(error)})
            raise
