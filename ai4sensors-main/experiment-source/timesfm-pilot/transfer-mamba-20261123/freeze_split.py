"""Write the external-only split before opening any sensor samples."""

import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
DATA = Path("E:/数据集")
BROAD = DATA / "broad-main" / "broad-main" / "data_mat"
SASSARI = DATA / "Sassari"
VALIDATION = {
    "undisturbed_slow_rotation_with_breaks",
    "undisturbed_slow_translation",
    "undisturbed_fast_combined",
    "disturbed_phone_vibration",
    "disturbed_office",
}
SENSORS = ("XS1", "XS2", "AP1", "AP2", "SH1", "SH2")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def stamp():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def write(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def main():
    paths = sorted(BROAD.glob("*.mat"))
    trials_path = BROAD / "trials.json"
    trials = json.loads(trials_path.read_text(encoding="utf-8"))["trials"]
    if len(paths) != 39 or set(trials) != {p.stem for p in paths}:
        raise ValueError("BROAD inventory differs from trial definitions")
    rows, sources = [], {}
    for path in paths:
        name = path.stem
        group = re.sub(r"_(?:[A-D]|\d+s)$", "", name.split("_", 1)[1])
        rows.append({"domain": "BROAD", "name": name, "group": group,
                     "split": "validation" if group in VALIDATION else "train", "path": str(path)})
        sources[str(path)] = digest(path)
    if not VALIDATION <= {r["group"] for r in rows}:
        raise ValueError("A requested BROAD validation group is missing")
    for motion in ("slow", "medium", "fast"):
        path = SASSARI / f"{motion}_v5.mat"
        sources[str(path)] = digest(path)
        rows.extend({"domain": "Sassari", "name": f"{motion}/{sensor}", "group": motion,
                     "split": "validation" if motion == "medium" else "train", "path": str(path)}
                    for sensor in SENSORS)
    for domain in ("BROAD", "Sassari"):
        split_groups = [{r["group"] for r in rows if r["domain"] == domain and r["split"] == split}
                        for split in ("train", "validation")]
        if not all(split_groups) or split_groups[0] & split_groups[1]:
            raise ValueError(f"Invalid group separation for {domain}")
    path = HERE / "external-split.json"
    if path.exists():
        raise FileExistsError(path)
    write(path, {"frozen_at": stamp(), "group_rule": {
        "BROAD": "Program after dropping numeric trial prefix and replicate A–D or duration suffix; combined duration variants stay together",
        "Sassari": "Entire motion and all six channels stay together",
    }, "reserved_for_later": ["TUM-VI", "all SHICE"], "broad_trials_sha256": digest(trials_path),
        "source_file_sha256": sources, "records": rows})
    for domain in ("BROAD", "Sassari"):
        print(domain, {s: sum(r["domain"] == domain and r["split"] == s for r in rows) for s in ("train", "validation")})
    print("split SHA-256", digest(path))


if __name__ == "__main__":
    main()
