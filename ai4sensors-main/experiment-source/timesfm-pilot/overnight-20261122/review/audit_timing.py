"""Reconcile ledger wall time and Windows power events for the experiment window."""

import os
import subprocess
from datetime import datetime, timezone
from xml.etree import ElementTree


def system_events(*options):
    result = subprocess.run(["wevtutil.exe", "qe", "System", "/f:xml", "/e:Events", "/uni:true", *options],
                            capture_output=True, check=True, timeout=10)
    return list(ElementTree.fromstring(result.stdout.decode("utf-16")))


def utc_text(value):
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def power_event(entry):
    system = entry.find("{*}System")
    values = {item.get("Name"): item.text for item in entry.findall("{*}EventData/{*}Data")}
    return {"provider": system.find("{*}Provider").get("Name"), "id": int(system.findtext("{*}EventID")),
            "record_id": int(system.findtext("{*}EventRecordID")),
            "recorded_utc": system.find("{*}TimeCreated").get("SystemTime"),
            "sleep_utc": values.get("SleepTime"), "wake_utc": values.get("WakeTime")}


def sleep_overlap(event, start, end):
    if not event["sleep_utc"] or not event["wake_utc"]:
        return None
    try:
        sleep = datetime.fromisoformat(event["sleep_utc"].replace("Z", "+00:00"))
        wake = datetime.fromisoformat(event["wake_utc"].replace("Z", "+00:00"))
    except ValueError:
        return None
    if sleep.tzinfo is None or wake.tzinfo is None:
        return None
    left, right = max(start, sleep), min(end, wake)
    return (left, right) if right > left else None


def inspect_power(start, end):
    evidence = {"host": os.environ.get("COMPUTERNAME"), "log": "Windows System",
                "start_utc": utc_text(start), "through_utc": utc_text(end)}
    if os.name != "nt":
        return {**evidence, "available": False, "error": "Windows System log unavailable on this host"}
    kernel = "Provider[@Name='Microsoft-Windows-Kernel-Power'] and (EventID=42 or EventID=107 or EventID=506 or EventID=507)"
    wake = "Provider[@Name='Microsoft-Windows-Power-Troubleshooter'] and EventID=1"
    try:
        oldest = system_events("/rd:false", "/c:1")
        if not oldest:
            return {**evidence, "available": False, "error": "Windows System log contains no retained records"}
        oldest_utc = oldest[0].find("{*}System/{*}TimeCreated").get("SystemTime")
        bound = f"TimeCreated[@SystemTime>='{utc_text(start)}' and @SystemTime<='{utc_text(end)}']"
        events = [power_event(e) for e in system_events(f"/q:*[System[(({kernel}) or ({wake})) and {bound}]]")]
        if datetime.now(timezone.utc) > end:
            later = f"TimeCreated[@SystemTime>'{utc_text(end)}']"
            for entry in system_events(f"/q:*[System[({wake}) and {later}]]", "/rd:false", "/c:1"):
                event = power_event(entry)
                if sleep_overlap(event, start, end):
                    events.append(event)
        intervals = sorted({period for event in events if (period := sleep_overlap(event, start, end))})
        merged = []
        for left, right in intervals:
            if merged and left <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], right)
            else:
                merged.append([left, right])
        return {**evidence, "available": True, "oldest_retained_system_record_utc": oldest_utc,
                "retention_reaches_start": datetime.fromisoformat(oldest_utc.replace("Z", "+00:00")) <= start,
                "events": events, "sleep_intervals": [[utc_text(a), utc_text(b)] for a,b in merged],
                "confirmed_sleep_seconds": sum((b-a).total_seconds() for a,b in merged)}
    except (OSError, subprocess.SubprocessError, ElementTree.ParseError, UnicodeError, ValueError) as error:
        return {**evidence, "available": False, "error": str(error)}


def timing_summary(merged, attempts, cutoff, status):
    start, end = (datetime.fromisoformat(status[k]) for k in ("window_start", "window_end"))
    gaps = [{"after": left[1].isoformat(), "before": right[0].isoformat(),
             "seconds": (right[0]-left[1]).total_seconds()} for left, right in zip(merged, merged[1:])]
    occupied = sum((b-a).total_seconds() for a,b in merged)
    between = sum(g["seconds"] for g in gaps)
    assert merged and (merged[-1][1]-merged[0][0]).total_seconds() == occupied + between
    return {"first_closed_attempt_started": merged[0][0].isoformat(), "latest_completed_round_ended": cutoff.isoformat(),
            "closed_attempts": attempts, "wall_seconds_in_union_of_closed_attempts": occupied,
            "seconds_before_first_closed_attempt": (merged[0][0]-start).total_seconds(),
            "seconds_between_closed_attempts": between, "longest_gap": max(gaps, key=lambda g:g["seconds"], default=None),
            "gaps_at_least_60_seconds": [g for g in gaps if g["seconds"] >= 60],
            "power": inspect_power(start, min(datetime.now(timezone.utc), end))}


def describe_timing(timing):
    text = (f" 从窗口开始到首轮启动用时 {timing['seconds_before_first_closed_attempt']:.0f} 秒。"
            f"账本中前后两次尝试之间的间隔合计 {timing['seconds_between_closed_attempts']/60:.1f} 分钟")
    if timing["longest_gap"]:
        text += f"，最长一次为 {timing['longest_gap']['seconds']:.0f} 秒"
    power = timing["power"]
    text += "。"
    if not power["available"]:
        return text + " 本次未取得 Windows 系统电源日志，休眠情况待核实。"
    through = datetime.fromisoformat(power["through_utc"].replace("Z", "+00:00")).astimezone(
        datetime.fromisoformat(timing["latest_completed_round_ended"]).tzinfo)
    text += f" 查至 {through:%m-%d %H:%M:%S}，Windows 系统日志"
    if not power["retention_reaches_start"]:
        text += "仅保留了部分实验时段，"
    events = len(power["events"])
    if not events:
        return text + "未见睡眠、恢复或低功耗待机转换记录。"
    text += f"有 {events} 条相关电源记录。"
    if power["sleep_intervals"]:
        text += f" 其中有明确睡眠和唤醒时间的记录与实验窗口重叠 {power['confirmed_sleep_seconds']/60:.1f} 分钟。"
    else:
        text += " 现有日志未给出可配对的睡眠和唤醒时间。"
    return text
