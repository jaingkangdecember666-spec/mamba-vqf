"""Apply the existing training-fold installation fit in the mounted ridge joint audit."""

from hashlib import sha256

import numpy as np

import learning
from common import BATCH, aggregate, by_device, fingerprint, names, read_json
from experiment import baseline_reference
from features import VERSION
from joint_holdout import joint_folds


BATCH_NAME = "mount-50hz-ridge-matched-development"
PROTOCOL_FILE = BATCH / "ridge-50hz-mount-matched-protocol.json"


def run(item, protocol_file=PROTOCOL_FILE):
    planned = read_json(protocol_file)
    joint_config = item["config"]
    parent = joint_config["parent_round"]
    pair = next(row["with_mount"] for row in planned["pairs"] if row["with_mount"]["round"] == parent)
    source = read_json(BATCH / "rounds" / parent / "round.json")
    config = source["config"]
    if (source["status"] != "completed" or pair["joint_round"] != item["round_id"] or pair["config"] != config
            or fingerprint(config) != joint_config["parent_config_sha256"]
            or joint_config["protocol"] != planned["joint_protocol"]):
        raise ValueError("Mounted ridge joint audit requires its preregistered completed parent")
    records = names("dev")
    plan = list(joint_folds(records))
    if plan != [(label, train, held) for label, train, held in planned["joint_validation_folds"]]:
        raise ValueError("Mounted ridge joint fold definitions changed")
    raw = {name: learning.attach_native(name, config) for name in records}
    embeddings = {name: learning.projected(name, config["embedding"], config["seed"]) for name in records}
    baseline = baseline_reference()
    predictions, evidence, context_audit = [], [], {}
    for fold, training, held in plan:
        q = learning.installation_fit([raw[name] for name in training], config)
        data = {name: learning.prepare(learning.apply_install(raw[name], q, include_reference=name in held),
                                      config, embeddings[name]) for name in training + held}
        context_audit.update({name: part["training_context"] for name, part in data.items()})
        state = learning.fit([data[name] for name in training], config)
        rows = [{"name": name, "fold": fold, "frames": len(data[name]["index"]), "baseline_deg": baseline[name],
                 "rmse_deg": learning.evaluate(data[name], state, config),
                 "score_frame_indices_sha256": sha256(np.asarray(data[name]["index"], dtype="<i8").tobytes()).hexdigest()}
                for name in held]
        predictions.extend(rows)
        summary = aggregate(rows)
        evidence.append({"fold": fold, "training": training, "validation": held,
                         "excluded_from_fit": sorted(set(records) - set(training)),
                         "training_log": {"installation_quaternion_wxyz": q.tolist(),
                                          "installation_policy": config["installation"], "mount_shrink": config["mount_shrink"]},
                         "validation_summary": summary})
        print(fold, summary["macro_rmse_deg"], q.tolist(), flush=True)
    if sorted(row["name"] for row in predictions) != sorted(records) or sum(row["frames"] for row in predictions) != 8328711:
        raise ValueError("Mounted ridge joint audit lost historical scoring records or frames")
    return {"dev_metrics": {"joint_holdout": {**aggregate(predictions), "per_device": by_device(predictions)}},
            "sequences": predictions, "folds": evidence, "mask_version": VERSION, "training_context_audit": context_audit,
            "parent_round": parent, "parent_config_sha256": source["config_sha256"],
            "evaluated_model_config": config, "evaluated_model_sha256": fingerprint(config), "uses_heldout_splits": False,
            "new_installation_fits": len(evidence), "new_ridge_fits": len(evidence)}
