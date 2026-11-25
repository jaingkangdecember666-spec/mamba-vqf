"""Check the trained Mamba v1 block against a sample-at-a-time CPU recurrence."""

import hashlib
from pathlib import Path

import torch
from torch.nn import functional as F


OFFICIAL_SHA256 = "4a5deb1cc9c22757826ac5dda343f5a0c3b928c50ac5d9c7c860e247bce06cdf"


@torch.no_grad()
def check_head(head, inputs):
    head = head.cpu().eval()
    inputs = inputs.cpu().float()
    block = head.sequence
    raw = Path(__file__).resolve().parents[1]/".deps/transformers/models/mamba/modeling_mamba.py"
    sha = hashlib.sha256(raw.read_bytes().replace(b"\r\n",b"\n")).hexdigest()
    assert sha == OFFICIAL_SHA256 and type(block).__name__ == "MambaMixer"
    features = head.input(inputs)
    normalized = head.norm(features)
    kernel = block.conv1d.weight[:,0,:]
    batch, length = inputs.shape[:2]
    channels, horizon = kernel.shape
    hist = torch.zeros(batch, channels, horizon)
    state = torch.zeros(batch, channels, block.ssm_state_size)
    decay = -torch.exp(block.A_log)
    outputs = []
    for i in range(length):
        drive, gate = block.in_proj(normalized[:,i]).chunk(2,dim=-1)
        hist = torch.cat((hist[:,:,1:],drive.unsqueeze(-1)),dim=-1)
        u = F.silu((hist * kernel[None,:,:]).sum(dim=-1) + block.conv1d.bias)
        times, b, c = block.x_proj(u).split([block.time_step_rank,block.ssm_state_size,block.ssm_state_size],dim=-1)
        dt = F.softplus(block.dt_proj(times)).unsqueeze(-1)
        state = torch.exp(dt * decay[None,:,:]) * state + dt * u.unsqueeze(-1) * b.unsqueeze(1)
        y = (state * c.unsqueeze(1)).sum(-1) + block.D * u
        outputs.append(block.out_proj(y * F.silu(gate)))
    steps = torch.stack(outputs,dim=1)
    native = block(normalized)
    block_error = (native-steps).abs().max().item()
    head_error = (head(inputs,all_steps=True)-head.output(features+steps)).abs().max().item()
    prefix = max(1,length//2)
    altered = inputs.clone()
    altered[:,prefix:] = -inputs[:,prefix:] + 4.
    base_output = head(inputs,all_steps=True)
    prefix_error = (base_output[:,:prefix]-head(altered,all_steps=True)[:,:prefix]).abs().max().item()
    assert block_error < 2e-5 and head_error < 2e-5 and prefix_error < 2e-5
    return {"official_file_sha256":sha,"batch":batch,"time_steps":length,"nonzero_output":float(base_output.abs().max()),
            "block_max_abs_difference":block_error,"head_max_abs_difference":head_error,"future_perturbation_max":prefix_error}
