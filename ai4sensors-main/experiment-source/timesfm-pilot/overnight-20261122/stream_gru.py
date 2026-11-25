"""A small stateful causal GRU residual over the original FullVQF physical features."""

from time import perf_counter

import numpy as np
import torch
from torch import nn

import learning
from common import BATCH, Z, aggregate, by_device, fingerprint, names, read_json, rmse, rotate, stamp, unit, write_json
from experiment import baseline_reference
from features import VERSION, block_indices, training_support
from joint_holdout import joint_folds
from tilt_model import RESIDUAL_SCALE, contexts, physical_features, residual_loss


MODES = {"continuous": None, "reset_2_seconds": 20, "reset_each_update": 1}
TRAINING_CONTEXT_REVISION = "split-state-at-forbidden-context-v2"


class ResidualGRU(nn.Module):
    def __init__(self, input_mean, input_std, width):
        super().__init__()
        self.register_buffer("input_mean", torch.as_tensor(input_mean))
        self.register_buffer("input_std", torch.as_tensor(input_std))
        dimension = self.input_mean.numel()
        self.recurrent = nn.GRU(dimension, width, num_layers=1, batch_first=True)
        self.readout = nn.Sequential(nn.Linear(dimension+width, 32), nn.GELU(), nn.Linear(32, 3))
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)

    def forward(self, physical, hidden=None):
        x = ((physical-self.input_mean)/self.input_std).clip(-12., 12.)
        states, hidden = self.recurrent(x, hidden)
        delta = torch.tanh(self.readout(torch.cat((states,x),dim=-1))) * (3*RESIDUAL_SCALE)
        up = physical[..., :3]
        return delta - (delta * up).sum(dim=-1,keepdim=True)*up, hidden


def prepare(data, config, for_training=False):
    at = block_indices(len(data["windows"]), config["update_hz"])
    history = contexts(data["windows"], at)
    physical = physical_features(history, data["up"][at], data["bias"][at], data["rest"][at])
    part = {"matrix": physical, "q": data["q"][at], "ready": data["ready"][at],
            "index": data["index"], "ref_world": data["ref_world"], "dt": float(data["dt"])}
    if for_training:
        permitted = max(64, config["context"])
        ends = at + 1
        starts = np.maximum(ends - permitted, 0)
        bad = data["bad_to_block"][ends] - data["bad_to_block"][starts]
        state_ok = (ends >= permitted) & (bad == 0)
        labels, audit = training_support(data, at, config["context"])
        if np.any(labels & ~state_ok):
            raise ValueError("A labeled update has forbidden short-window input")
        part.update(target=data["target"][at], train=labels, state_input_ok=state_ok, training_context=audit)
    return part


def training_segments(record):
    safe = record["state_input_ok"]
    labels = record["train"]
    if np.any(labels & ~safe):
        raise ValueError("A training label lies outside a permitted GRU state segment")
    starts = np.flatnonzero(safe & ~np.r_[False, safe[:-1]])
    ends = 1 + np.flatnonzero(safe & ~np.r_[safe[1:], False])
    for begin, end in zip(starts, ends, strict=True):
        supervised = np.flatnonzero(labels[begin:end])
        if len(supervised):
            yield int(begin), int(begin + supervised[-1] + 1)


def training_arrays(records, device):
    sizes = [int(row["train"].sum()) for row in records]
    if not all(sizes):
        raise ValueError("An entire training record lacks verified update labels")
    num_records = len(records)
    segments = [(index, start, end) for index, row in enumerate(records) for start,end in training_segments(row)]
    longest = max(end - start for _, start,end in segments)
    n = len(segments)
    features = np.zeros((n,longest,records[0]["matrix"].shape[-1]),dtype=np.float32)
    target = np.zeros((n,longest,3),dtype=np.float32)
    weight = np.zeros((n,longest),dtype=np.float32)
    total_labels = sum(sizes)
    for batch_index, (record_index, begin, end) in enumerate(segments):
        row = records[record_index]
        features[batch_index,:end-begin] = row["matrix"][begin:end]
        labeled = np.flatnonzero(row["train"][begin:end])
        target[batch_index,labeled] = row["target"][labeled+begin]
        weight[batch_index,labeled] = total_labels/(num_records*sizes[record_index])
    if np.count_nonzero(weight) != total_labels:
        raise ValueError("Segmentation lost or duplicated original training labels")
    usable = torch.as_tensor(np.concatenate([row["matrix"][row["train"]] for row in records]),device=device)
    mean = usable.mean(0)
    std = usable.std(0,unbiased=False).clamp_min(.003)
    audit = {"labels":total_labels,"per_record_label_counts":sizes,"independent_state_segments":n,
             "segments_per_record":np.bincount([i for i,_,_ in segments],minlength=num_records).tolist(),
             "forbidden_model_inputs_excluded":sum(int((~row["state_input_ok"]).sum()) for row in records),
             "approved_unsupervised_inputs_retained":sum(int((~records[i]["train"][start:end]).sum()) for i,start,end in segments)}
    return tuple(torch.as_tensor(x,device=device) for x in (features,target,weight)), mean, std, audit


def fit(records, config, device):
    started = perf_counter()
    torch.manual_seed(config["seed"])
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    if config.get("recurrent_training_context") != TRAINING_CONTEXT_REVISION:
        raise ValueError("Recurrent training requires the revised context-segmentation protocol")
    (physical,target,weights),mean,std,audit = training_arrays(records,device)
    scoring_labels = audit["labels"]
    model = ResidualGRU(mean,std,config["width"]).to(device)
    if config.get("initial_update_retention"):
        retained = config["initial_update_retention"]
        width = config["width"]
        with torch.no_grad():
            model.recurrent.bias_ih_l0[width:2*width].fill_(np.log(retained/(1-retained)))
            model.recurrent.bias_hh_l0[width:2*width].zero_()
    optimizer = torch.optim.AdamW(model.parameters(),lr=config["lr"],weight_decay=config["weight_decay"])
    span = config["optimizer_span_updates"]
    bptt = config["bptt_updates"]
    if span % bptt:
        raise ValueError("Gradient truncation must divide the fixed optimizer update span")
    n, length, _ = physical.shape
    losses, durations = [], []
    for epoch in range(config["epochs"]):
        epoch_start = perf_counter()
        model.train()
        hidden = None
        total = torch.zeros((),device=device)
        for start in range(0,length,span):
            end = min(start+span,length)
            optimizer.zero_grad(set_to_none=True)
            for at in range(start,end,bptt):
                stop = min(at+bptt,end)
                input_slice = physical[:,at:stop]
                delta,hidden = model(input_slice,hidden)
                hidden = hidden.detach()
                residual = residual_loss(input_slice[..., :3].reshape(-1,3),target[:,at:stop].reshape(-1,3),
                                         delta.reshape(-1,3)).reshape(n,-1)
                sample_loss = (residual * weights[:,at:stop]).sum()
                (sample_loss/(len(records)*span)).backward()
                total += sample_loss.detach()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()
        loss = float(total.item()/scoring_labels)
        if not np.isfinite(loss):
            raise ValueError(f"Residual GRU training diverged during epoch {epoch+1}")
        losses.append(loss)
        durations.append(perf_counter()-epoch_start)
    state = {k:v.detach().cpu().numpy().copy() for k,v in model.state_dict().items()}
    return state, {"weighted_train_residual_loss":losses,"epoch_seconds":durations,"fit_wall_seconds":perf_counter()-started,
                   "train_labels":scoring_labels,"records":len(records),"optimizer_steps_per_epoch":int(np.ceil(length/span)),
                   "parameters":sum(p.numel() for p in model.parameters()),"training_context_audit":audit,
                   "initial_hidden_reset":"zero independently at the start of each context-valid segment and training epoch",
                   "invalid_label_policy":"only missing targets can carry training state; forbidden input context is omitted and splits state and gradients"}


def make_model(state,config,device):
    own = {k:torch.as_tensor(v) for k,v in state.items() if k!="installation_quaternion_wxyz"}
    model = ResidualGRU(own["input_mean"],own["input_std"],config["width"])
    model.load_state_dict(own)
    return model.to(device).eval()


@torch.inference_mode()
def infer_delta(physical,model,device,mode="continuous"):
    x = torch.as_tensor(physical,device=device)
    if not len(x):
        return np.empty((0,3),dtype=np.float32)
    every = MODES[mode]
    if every is not None:
        count = (len(x)+every-1)//every
        padded = torch.nn.functional.pad(x,(0,0,0,count*every-len(x)))
        values,_ = model(padded.reshape(count,every,x.shape[-1]))
        return values.reshape(-1,3)[:len(x)].cpu().numpy()
    result, hidden = [], None
    for chunk in x.split(512):
        predicted, hidden = model(chunk[None],hidden)
        result.append(predicted[0].cpu().numpy())
    return np.concatenate(result)


def score(data,config,model,device,mode="continuous",delay_ms=0.):
    delta = infer_delta(data["matrix"],model,device,mode) * config["gain"]
    direction = unit(Z+rotate(unit(data["q"]),delta))
    delay = int(np.ceil(delay_ms/(1000*data["dt"])))
    active = np.searchsorted(data["ready"]+delay,data["index"],side="right")
    return rmse(np.vstack((Z[None,:],direction))[active],data["ref_world"])


def planned_folds(records):
    regular = [(axis,label,[name for name in records if name not in held],held)
               for axis,label,held in learning.folds(records)]
    return regular + [("joint_holdout",label,training,held) for label,training,held in joint_folds(records)]


def run(item):
    started = perf_counter()
    config = item["config"]
    if config.get("recurrent_training_context") != TRAINING_CONTEXT_REVISION:
        raise ValueError("Superseded recurrent training-context protocol")
    records = names("dev")
    baseline = baseline_reference()
    native = {name:learning.attach_native(name,config) for name in records}
    device = torch.device("cuda")
    scored, folds = {}, []
    for axis,label,training,held in planned_folds(records):
        prep_start = perf_counter()
        mount = learning.installation_fit([native[name] for name in training],config)
        data = {name:prepare(learning.apply_install(native[name],mount,include_reference=name in held),config,
                            for_training=name in training) for name in training+held}
        fit_start = perf_counter()
        state,log = fit([data[name] for name in training],config,device)
        scoring_start = perf_counter()
        model = make_model(state,config,device)
        block = []
        for name in held:
            part = data[name]
            results = {mode:score(part,config,model,device,mode) for mode in MODES}
            block.append({"name":name,"fold":label,"frames":len(part["index"]),"baseline_deg":baseline[name],
                          "rmse_deg":results["continuous"],"same_weight_state_controls_deg":results})
        timing = {"prepare_seconds":fit_start-prep_start,"fit_seconds":scoring_start-fit_start,
                  "score_seconds":perf_counter()-scoring_start}
        scored.setdefault(axis,[]).extend(block)
        folds.append({"axis":axis,"fold":label,"training":training,"validation":held,"training_log":log,
                       "initial_hidden_states":"training segments independently zero; validation resets only at each original recording start",
                      "training_label_counts":{name:int(data[name]["train"].sum()) for name in training},
                      "timing":timing,"validation_summary":aggregate(block)})
        print(axis,label,round(folds[-1]["validation_summary"]["macro_rmse_deg"],6),timing,log["weighted_train_residual_loss"],flush=True)

    metrics,controls = {}, {}
    for axis,rows in scored.items():
        if sorted(r["name"] for r in rows)!=sorted(records):
            raise ValueError(f"Recurrent score frames do not cover all development records in {axis}")
        metrics[axis] = {**aggregate(rows),"per_device":by_device(rows)}
        controls[axis] = {mode:aggregate([{**row,"rmse_deg":row["same_weight_state_controls_deg"][mode]} for row in rows])
                          for mode in MODES}
    metrics["selection_rmse_deg"] = (metrics["device"]["macro_rmse_deg"]+metrics["motion"]["macro_rmse_deg"])/2.
    timing = {key:sum(row["timing"][key] for row in folds) for key in ("prepare_seconds","fit_seconds","score_seconds")}
    timing["total_wall_seconds"] = perf_counter()-started
    return {"dev_metrics":metrics,"state_controls":controls,"sequences":scored,"folds":folds,"timing":timing,
             "mask_version":VERSION,"recurrent_training_context":TRAINING_CONTEXT_REVISION,"update_hz":config["update_hz"],"inference_protocol":"sequential input, original physical features, no device or program identities"}


def bench(config,state,device,benchmark_record):
    from vqf import VQF

    from common import body_up, record, vqf_outputs
    from features import block_indices
    from final_eval import timed
    from tilt_model import physical_features

    raw = learning.attach_native(benchmark_record,config)
    mount = state["installation_quaternion_wxyz"]
    source = learning.apply_install(raw,mount,include_reference=False)
    positions = block_indices(len(source["windows"]),config["update_hz"])[100:320]
    dt = float(source["dt"])
    sample = record(benchmark_record)
    acc = np.ascontiguousarray(sample["imu"][:,:3],dtype=np.float64)
    gyr = np.ascontiguousarray(sample["imu"][:,3:],dtype=np.float64)
    base = VQF(dt,**config["parameters"])
    start = 22000
    base.updateBatch(gyr[:start],acc[:start])

    def native(i):
        at = start+i
        base.update(gyr[at],acc[at])
        return learning.multiply_quaternion(base.getQuat6D(),mount)

    def correction(target_device):
        model = make_model(state,config,target_device)
        hidden = None
        last = Z
        @torch.inference_mode()
        def update(index):
            nonlocal hidden,last
            at = positions[index]
            physical = physical_features(source["windows"][None,at-63:at+1],source["up"][at:at+1],
                                         source["bias"][at:at+1],source["rest"][at:at+1])
            x = torch.as_tensor(physical[None],device=target_device)
            delta,hidden = model(x,hidden)
            last = unit(Z + rotate(unit(source["q"][at]),delta[0,0].cpu().numpy()*config["gain"]))
        timings = timed(update,200,20)
        return timings,last

    native_timing = timed(native,3000,100)
    cuda_timing,held = correction(device)
    cpu_timing,_ = correction(torch.device("cpu"))
    q = vqf_outputs(sample["imu"],dt,config["parameters"])["quat6D"][start:start+3100]
    q = learning.multiply_quaternion(q,mount)
    application = timed(lambda i:body_up(q[i],held),3000,100)
    return {"record":benchmark_record,"nominal_native_hz":1./dt,"correction_hz":1./(100*dt),
            "native_step":native_timing,"correction_update":cuda_timing,"cpu_correction_update":cpu_timing,
            "apply_held_correction":application,"state_preserved_between_successive_correction_windows":True}


def queue():
    from optimizer_study import round_record, schedule

    parent = round_record("P00022")
    config = {**parent["config"],"kind":"stream_gru","width":12,"bptt_updates":96,"optimizer_span_updates":96,
              "epochs":4,"lr":0.0012,"recurrent_training_context":TRAINING_CONTEXT_REVISION}
    variants = (({}, "Small GRU with up to 9.6 seconds of gradient history and continuous inference state"),
                ({"width":24}, "Increase state width at the same gradient history, update span and physical observations"),
                ({"bptt_updates":24}, "Truncate gradient at 2.4 seconds with the same optimization steps and continuous inference state"))
    designs = []
    for i,(change,hypothesis) in enumerate(variants,1):
        identity = schedule(f"C{i+10:05d}",{**config,**change},"small-recurrent-residual-context-v2",f"C{i:05d}",hypothesis)
        designs.append({"round_id":identity,"hypothesis":hypothesis,"supersedes":f"C{i:05d}"})
    write_json(BATCH/"stream-gru-development-stage.json",{"created_at":stamp(),"designs":designs,
                "anchor":parent["round_id"],"other_control":"N00035","sample_rate_hz":10,"state_controls":MODES,
               "recurrent_training_context":TRAINING_CONTEXT_REVISION,
               "historical_unsupported_results":["C00001","C00002","C00003"],"superseded_before_execution":["C00004","C00005"],
               "repeat_rule":{"min_continuous_joint_improvement_over_reset_each_deg":.015,
                              "max_joint_degradation_vs_physical_control_deg":.02,
                              "max_device_degradation_vs_physical_control_deg":.025}})
    print(designs,flush=True)


if __name__ == "__main__":
    queue()
