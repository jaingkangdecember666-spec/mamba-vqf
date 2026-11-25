"""Join the 38 frozen choices to the instructions actually delivered to the executing chat."""
import csv
import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import median

from lock_from_frontier import ROOT, REVIEW, read, digest


COORDINATOR = "01a203c6-49fa-7f10-ba57-925c374671f8"
EXECUTOR = "01a20361-ff3b-77c0-90c4-adf20bfd7be2"
CHINA = timezone(timedelta(hours=8))


def load_dispatches(begin, end):
    folder = Path.home() / ".codex/sessions/2026/11/22"
    session, = folder.glob("*" + EXECUTOR + ".jsonl")
    messages = []
    with session.open(encoding="utf8") as stream:
        for line in stream:
            if "<codex_delegation>" not in line or COORDINATOR not in line:
                continue
            entry = json.loads(line)
            stamp = datetime.fromisoformat(entry["timestamp"]).astimezone(CHINA)
            if not begin <= stamp <= end:
                continue
            event = entry.get("payload", {})
            snippet = event.get("output")
            if entry["type"] != "response_item" or event.get("type") != "function_call_output" or event.get("name")!="send_message_to_thread":
                continue
            if not isinstance(snippet, str) or not snippet.startswith("<codex_delegation>"):
                continue
            # The stored text is wrapper-delimited. Plain '<' in a numerical instruction can be literal.
            if f"<source_thread_id>{COORDINATOR}</source_thread_id>" not in snippet:
                continue
            result = re.search(r"<input>(.*)</input>", snippet, re.S)
            assert result is not None
            message = result[1]
            messages.append((stamp, message, hashlib.sha256(message.encode("utf8")).hexdigest(), event["id"]))
    return messages


def main():
    policy = read(ROOT / "protocol.json")
    lock = read(ROOT / "selection-lock.json")
    assert policy["worker_chat"] == EXECUTOR
    ledger = [json.loads(line) for line in (REVIEW / "decisions.jsonl").read_text(encoding="utf-8-sig").splitlines() if line]
    decisions = {r["id"]:r for r in ledger}
    entries = read(REVIEW / "campaign-counts.json")["batches"]
    messages = load_dispatches(datetime.fromisoformat(policy["start"]), datetime.fromisoformat(lock["locked_at"]))
    output = []
    used = set()
    for i, batch in enumerate(entries):
        name = batch["batch"]
        decision = decisions[name]
        registered = datetime.fromisoformat(decision["at"])
        started = datetime.fromisoformat(batch["started_at"])
        stopped = datetime.fromisoformat(batch["completed_at"])
        applicable = [(t, msg, sha, cid) for t, msg, sha, cid in messages
                      if abs((t-registered).total_seconds()) <= 90 and t < started and name in msg]
        assert len(applicable)==1, (name, [(t, x[:45]) for t,x,_,_ in applicable])
        timestamp, text, msg_sha, cid = applicable[0]
        assert cid not in used
        used.add(cid)
        previous = None
        if i:
            earlier = entries[i-1]
            previous = datetime.fromisoformat(earlier["completed_at"])
            assert previous <= started, (name, previous, started)
            if previous > timestamp:
                # C037 published its full score table before an audit-helper hash delayed its summary.
                assert name=="C038" and earlier["batch"]=="C037"
                score_table = ROOT / "C037/sequences.csv"
                written = datetime.fromtimestamp(score_table.stat().st_mtime, CHINA)
                assert written < timestamp and digest(score_table)==lock["evidence_sha256"]["C037/sequences.csv"]
                with score_table.open(encoding="utf-8-sig") as stream:
                    assert sum(1 for _ in csv.DictReader(stream)) == 213
                assert "213" in text and "C037" in text
        output.append({"batch":name, "recorded_decision":decision["at"], "received_in_worker":timestamp.isoformat(),
                       "registered_job_start":started.isoformat(), "registered_result":stopped.isoformat(),
                       "seconds_from_instruction_to_job":(started-timestamp).total_seconds(),
                       "previous_completed_at":previous.isoformat() if previous else "",
                       "previous_evidence_at_dispatch":"initial" if previous is None else "summary" if previous<=timestamp else "full_score_table",
                       "records_scored":batch["scored_records"], "received_message_sha256":msg_sha,
                       "delivery_item_id":cid})
    assert len(output)==len(used)==38
    export=REVIEW/"guidance-receipts.csv"
    with export.open("w", newline="", encoding="utf-8-sig") as stream:
        writer=csv.DictWriter(stream, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    result={"verified_at":datetime.now().astimezone().isoformat(), "coordinator_chat":COORDINATOR, "executing_chat":EXECUTOR,
            "lock_sha256":digest(ROOT/"selection-lock.json"),"ledger_sha256":digest(REVIEW/"decisions.jsonl"),
            "campaign_counts_sha256":digest(REVIEW/"campaign-counts.json"),"receipt_csv_sha256":digest(export),
            "recorded_entries_through_selection":len(ledger), "separate_batch_directives_received":len(used),
            "after_previous_summary":sum(r["previous_evidence_at_dispatch"]=="summary" for r in output),
            "after_previous_full_csv_before_summary":sum(r["previous_evidence_at_dispatch"]=="full_score_table" for r in output),
            "strictly_nonoverlapping_registered_batches":len(output)-1,
            "lead_seconds_min_median_max":[min(r["seconds_from_instruction_to_job"] for r in output),
                                         median(r["seconds_from_instruction_to_job"] for r in output),
                                         max(r["seconds_from_instruction_to_job"] for r in output)]}
    (REVIEW/"guidance-verification.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf8")
    print(result["separate_batch_directives_received"], result["after_previous_summary"],
          result["after_previous_full_csv_before_summary"],result["lead_seconds_min_median_max"])


if __name__=="__main__":
    main()
