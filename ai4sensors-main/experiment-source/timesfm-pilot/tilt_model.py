"""Causal TimesFM representation and unit-quaternion tilt correction."""

import hashlib

import numpy as np
import torch
from timesfm3.torch.model import TimesFM3Torch
from torch import nn
from torch.nn import functional as F

from run_pilot import CHECKPOINT_SHA256
from tilt_data import CONTEXT, EST_UP


G = 9.80665
RESIDUAL_SCALE = 0.05
INPUT_SCALE = np.array([G, G, G, 1., 1., 1.], dtype=np.float32)


def normalize(vectors):
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True).clip(min=1e-12)


def rotate(quaternions, vectors):
    """Rotate vectors by wxyz unit quaternions with NumPy broadcasting."""
    xyz = quaternions[..., 1:]
    cross = np.cross(xyz, vectors)
    return vectors + 2 * (quaternions[..., :1] * cross + np.cross(xyz, cross))


def correction_quaternions(base, body_deltas):
    """Return shortest world rotations that send the estimated up vector to +Z."""
    world_delta = rotate(normalize(base), body_deltas)
    up = normalize(world_delta + EST_UP)
    return normalize(np.concatenate((1 + up[..., 2:3], np.cross(up, EST_UP)), axis=-1))


def multiply(a, b):
    real = a[..., :1] * b[..., :1] - np.sum(a[..., 1:] * b[..., 1:], axis=-1, keepdims=True)
    imag = a[..., :1] * b[..., 1:] + b[..., :1] * a[..., 1:] + np.cross(a[..., 1:], b[..., 1:])
    return np.concatenate((real, imag), axis=-1)


def contexts(windows, blocks):
    indices = blocks[:, None] - np.arange(CONTEXT - 1, -1, -1)
    return np.ascontiguousarray(windows[indices])


def physical_features(windows, base_up, bias, rest):
    scaled = windows / INPUT_SCALE
    recent = scaled[:, -4:].mean(axis=1)
    return np.concatenate((base_up, bias, rest[:, None], recent,
                           scaled[:, -16:].mean(axis=1), scaled.mean(axis=1),
                           scaled.std(axis=1), normalize(recent[:, :3]) - base_up), axis=1)


def load_backbone(checkpoint, device):
    for name, expected in CHECKPOINT_SHA256.items():
        with (checkpoint / name).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError(f"Checkpoint does not match the pinned TimesFM revision: {name}")
    model = TimesFM3Torch.from_pretrained(str(checkpoint.resolve()), local_files_only=True)
    return model.eval().requires_grad_(False).to(device)


@torch.inference_mode()
def encode(model, histories, device):
    """Extract the last historical patch for each of the six measured channels."""
    signal = torch.as_tensor(histories, device=device)
    values = signal.transpose(1, 2).reshape(len(signal), 6, -1, model.input_patch_len)
    inputs = {"values": values, "masks": torch.zeros_like(values, dtype=torch.bool),
              "patch_is_target": torch.ones(values.shape[:-1], device=device, dtype=torch.bool)}
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        result = model(inputs, return_aux_outputs=True)
    return result["__call__:transformer_output"][:, :, -1].to(torch.float16).contiguous()


class TiltHead(nn.Module):
    """Fit a small supervised readout on frozen TimesFM embeddings and VQF state."""

    def __init__(self, physical_mean, physical_std, embedding_width, dropout=0.1):
        super().__init__()
        self.register_buffer("physical_mean", torch.as_tensor(physical_mean))
        self.register_buffer("physical_std", torch.as_tensor(physical_std))
        self.embedding_width = embedding_width
        input_width = self.physical_mean.numel()
        if embedding_width:
            self.projection = nn.Sequential(nn.LayerNorm(embedding_width, elementwise_affine=False),
                                            nn.Linear(embedding_width, 24), nn.GELU())
            input_width += 6 * 24
        self.readout = nn.Sequential(nn.Linear(input_width, 96), nn.GELU(), nn.Dropout(dropout),
                                     nn.Linear(96, 48), nn.GELU(), nn.Dropout(dropout), nn.Linear(48, 3))
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)

    def forward(self, physical, embedding=None):
        normalized = ((physical - self.physical_mean) / self.physical_std).clamp(-12, 12)
        if self.embedding_width:
            latent = self.projection(embedding.float()).flatten(start_dim=1)
            normalized = torch.cat((normalized, latent), dim=1)
        delta = self.readout(normalized).tanh() * (3 * RESIDUAL_SCALE)
        base = physical[:, :3]
        return delta - (delta * base).sum(dim=1, keepdim=True) * base


def residual_loss(base, target, delta):
    prediction = F.normalize(base + delta, dim=1)
    return ((prediction - target).square().sum(dim=1) / RESIDUAL_SCALE**2)
