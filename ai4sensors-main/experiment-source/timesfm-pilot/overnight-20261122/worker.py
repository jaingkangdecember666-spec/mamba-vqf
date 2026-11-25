"""Isolate one experiment and enforce its wall-clock deadline."""

import argparse
import msvcrt
import os
import threading
from datetime import datetime
from pathlib import Path
from time import sleep

from common import now, read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path)
    parser.add_argument("cutoff")
    args = parser.parse_args()
    cutoff = datetime.fromisoformat(args.cutoff)
    directory = args.record.parent
    job = read_json(args.record)
    if job["config"]["kind"] not in {"final", "selection"}:
        from datetime import timedelta
        cutoff = min(cutoff, now() + timedelta(minutes=12))
    with (directory / "worker.lock").open("a+b") as lock:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)

        def enforce_cutoff():
            while now() < cutoff:
                sleep(0.5)
            os._exit(124)

        threading.Thread(target=enforce_cutoff, daemon=True).start()
        result = directory / "result.json"
        if job["config"]["kind"] == "baseline":
            from baseline import run
            run(result)
        elif job["config"]["kind"] == "sensor_hold_diagnostic":
            from sensor_hold import run
            run(job, result)
        elif job["config"]["kind"] == "timesfm_encoder_diagnostic":
            from timesfm_encoder_audit import run
            run(job, result)
        elif job["config"]["kind"] == "forecast_backbone_diagnostic":
            from forecast_backbone import run
            run(job, result)
        elif job["config"]["kind"] == "ridge_seed_diagnostic":
            from ridge_seed_diagnostic import run
            run(job, result)
        elif job["config"]["kind"] == "bias_physical_head":
            from calibrated_head import derive
            derive(job, result)
        elif job["config"]["kind"] == "accelerometer_bias_head_audit":
            from bias_light_audit import run
            run(job, result)
        elif job["config"]["kind"] == "accelerometer_bias_audit":
            from bias_calibration import run
            run(job, result)
        elif job["config"]["kind"] == "accelerometer_relative_gain_audit":
            from relative_gain import run
            run(job, result)
        elif job["config"]["kind"] == "quaternion_audit":
            from quaternion_audit import run
            run(job, result)
        elif job["config"]["kind"] == "cpu_readout_audit":
            from cpu_readout import run
            run(job, result)
        elif job["config"]["kind"] == "regime_audit":
            from regime_audit import run
            run(job, result)
        elif job["config"]["kind"] == "development_latency":
            from development_latency import run
            run(job, result)
        elif job["config"]["kind"] == "selection":
            from final_eval import select
            select(job, result)
        elif job["config"]["kind"] == "final":
            from final_eval import run
            run(job, result)
        else:
            from experiment import run
            run(job, result)


if __name__ == "__main__":
    main()
