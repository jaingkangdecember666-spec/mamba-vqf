"""Two compact causal residual heads with identical input and output interfaces."""

import torch
from torch import nn
from torch.nn import functional as F
from transformers import MambaConfig
from transformers.models.mamba.modeling_mamba import MambaMixer, MambaPreTrainedModel


class CausalBlock(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.dilation = dilation
        self.conv = nn.Conv1d(width, width, 2, dilation=dilation)
        self.norm = nn.LayerNorm(width)

    def forward(self, x):
        y = self.conv(F.pad(x.transpose(1, 2), (self.dilation, 0))).transpose(1, 2)
        return x + F.gelu(self.norm(y))


class ResidualHead(nn.Module):
    def __init__(self, kind):
        super().__init__()
        width = 24 if kind == "mamba_v1" else 20
        self.input = nn.Linear(13, width)
        self.kind = kind
        if kind == "mamba_v1":
            config = MambaConfig(hidden_size=width, state_size=8, expand=2, conv_kernel=4,
                                 num_hidden_layers=1, use_mambapy=False, use_cache=False)
            self.norm = nn.LayerNorm(width)
            self.sequence = MambaMixer(config, layer_idx=0)
            initializer = MambaPreTrainedModel(config)
            with torch.no_grad():
                self.sequence.apply(initializer._init_weights)
        elif kind == "tcn":
            self.sequence = nn.Sequential(*(CausalBlock(width, 2**i) for i in range(6)))
        else:
            raise ValueError(kind)
        self.output = nn.Linear(width, 3)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x, *, all_steps=False):
        h = self.input(x)
        h = h + self.sequence(self.norm(h)) if self.kind == "mamba_v1" else self.sequence(h)
        return self.output(h if all_steps else h[:, -1])


def tangent(raw, up, cap_deg):
    delta = 0.1 * raw
    delta = delta - (delta * up).sum(dim=-1, keepdim=True) * up
    cap = torch.tensor(cap_deg * torch.pi / 180, dtype=delta.dtype, device=delta.device).tan()
    return delta * (cap / delta.norm(dim=-1, keepdim=True).clamp_min(1e-8)).clamp_max(1)
