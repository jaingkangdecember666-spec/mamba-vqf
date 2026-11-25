"""Persist completed sequence rows under the verified formal selection lock."""

from math import isfinite

from common import read_json, stamp, write_json


DATASETS = ("R6", "BROAD", "Sassari")
METHODS = {"full_vqf", "physical_only", "timesfm", "gru", "locked"}


class FinalRecordProgress:
    def __init__(self, directory, lock_sha256, started_at):
        self.path = directory / "final-records-progress.json"
        if self.path.is_file():
            current = read_json(self.path)
            if current["selection_lock_sha256"] != lock_sha256:
                raise ValueError("Per-record progress belongs to another selection lock")
            if set(current["datasets"]) != set(DATASETS):
                raise ValueError("Unexpected datasets in per-record progress")
            for dataset, rows in current["datasets"].items():
                self.index_rows(dataset, rows)
        else:
            current = {"at": stamp(), "started_at": started_at, "selection_lock_sha256": lock_sha256,
                       "datasets": {dataset: [] for dataset in DATASETS}}
            write_json(self.path, current)
        self.current = current
        self.started_at = current["started_at"]

    @staticmethod
    def index_rows(dataset, rows):
        indexed = {}
        for row in rows:
            name = row["name"]
            frames = row["frames"]
            frame_sha = row["score_frame_indices_sha256"]
            scores = row["rmse_deg"]
            if not isinstance(name, str) or not name or name in indexed:
                raise ValueError(f"Invalid or duplicate record name in {dataset}: {name}")
            if type(frames) is not int or frames <= 0:
                raise ValueError(f"Invalid scoring count in {dataset}/{name}")
            if (not isinstance(frame_sha, str) or len(frame_sha) != 64
                    or any(c not in "0123456789abcdef" for c in frame_sha)):
                raise ValueError(f"Invalid score frame fingerprint in {dataset}/{name}")
            if set(scores) != METHODS or any(type(value) not in (int, float) or not isfinite(value) or value < 0
                                             for value in scores.values()):
                raise ValueError(f"Invalid five-method RMSE in {dataset}/{name}")
            indexed[name] = row
        return indexed

    def add(self, dataset, row):
        rows = self.current["datasets"][dataset]
        self.index_rows(dataset, [row])
        previous = next((saved for saved in rows if saved["name"] == row["name"]), None)
        if previous is not None:
            self.require_match(dataset, previous, row)
            return

        updated = {**self.current, "at": stamp(), "datasets": {
            **self.current["datasets"], dataset: [*rows, row]}}
        write_json(self.path, updated)
        self.current = updated

    @staticmethod
    def require_match(dataset, saved, row):
        identity = f"{dataset}/{row['name']}"
        if saved["frames"] != row["frames"] or saved["score_frame_indices_sha256"] != row["score_frame_indices_sha256"]:
            raise ValueError(f"Score frame coverage changed in {identity}")
        changed = [method for method, value in row["rmse_deg"].items() if saved["rmse_deg"][method] != value]
        if changed:
            raise ValueError(f"RMSE changed in {identity}: {', '.join(changed)}")
        if saved != row:
            raise ValueError(f"Per-record metadata changed in {identity}")

    def reconcile(self, dataset, rows):
        current = self.index_rows(dataset, rows)
        saved = self.index_rows(dataset, self.current["datasets"][dataset])
        if current.keys() != saved.keys():
            raise ValueError(f"Formal output record set differs from per-record progress for {dataset}")
        for name, row in current.items():
            self.require_match(dataset, saved[name], row)
