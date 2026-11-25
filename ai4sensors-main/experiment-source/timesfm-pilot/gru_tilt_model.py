"""Two-layer time-aware GRU for direct causal six-axis attitude estimation."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


WIDTH = 200
LAYERS = 2
RATES = np.linspace(50., 500., 100)


class AttitudeGRU(nn.Module):
    def __init__(self, input_mean, input_std):
        super().__init__()
        self.register_buffer("input_mean", torch.as_tensor(input_mean, dtype=torch.float32))
        self.register_buffer("input_std", torch.as_tensor(input_std, dtype=torch.float32))
        self.recurrent = nn.GRU(7, WIDTH, num_layers=LAYERS, batch_first=True)
        self.readout = nn.Linear(WIDTH, 4)
        with torch.no_grad():
            self.readout.bias.copy_(torch.tensor([1., 0., 0., 0.]))

    def forward(self, observations, hidden=None):
        x = (observations - self.input_mean) / self.input_std
        states, hidden = self.recurrent(x, hidden)
        return F.normalize(self.readout(states).float(), dim=-1), hidden


def predicted_up(q):
    w, x, y, z = q.unbind(dim=-1)
    return torch.stack((2 * (x * z - w * y), 2 * (y * z + w * x),
                        1 - 2 * (x.square() + y.square())), dim=-1)


def tilt_loss(q, target, mask):
    up = predicted_up(q[mask]).double()
    cosine = (up * target[mask].double()).sum(-1).clamp(-1 + 1e-12, 1 - 1e-12)
    return torch.acos(cosine).square().mean().float()


def rotate_vectors(q, vectors):
    xyz = q[:, None, 1:].expand_as(vectors)
    cross = torch.linalg.cross(xyz, vectors)
    return vectors + 2 * (q[:, None, :1] * cross + torch.linalg.cross(xyz, cross))


def augment(observations, target):
    count = len(observations)
    device = observations.device
    rotation = F.normalize(torch.randn(count, 4, device=device), dim=1)
    acc = rotate_vectors(rotation, observations[:, :, :3])
    gyr = rotate_vectors(rotation, observations[:, :, 3:6])
    acc_noise = torch.rand(count, 1, 1, device=device) * 0.05
    gyr_noise = torch.rand(count, 1, 1, device=device) * np.deg2rad(0.12)
    bias = torch.randn(count, 1, 3, device=device) * np.deg2rad(0.5)
    acc = acc + torch.randn_like(acc) * acc_noise
    gyr = gyr + torch.randn_like(gyr) * gyr_noise + bias
    return torch.cat((acc, gyr, observations[:, :, 6:]), dim=-1), rotate_vectors(rotation, target)


def with_dt(imu, dt):
    return np.concatenate((imu, np.full((len(imu), 1), dt, dtype=np.float32)), axis=1)


@torch.inference_mode()
def infer_sequences(model, sequences, device, chunk_size=2048):
    lengths = [len(sequence) for sequence in sequences]
    max_length = max(lengths)
    padded = np.zeros((len(sequences), max_length, 7), dtype=np.float32)
    for row, sequence in zip(padded, sequences):
        row[:len(sequence)] = sequence
    tensor = torch.as_tensor(padded, device=device)
    result = np.empty((len(sequences), max_length, 3), dtype=np.float32)
    hidden = None
    for start in range(0, max_length, chunk_size):
        stop = min(start + chunk_size, max_length)
        q, hidden = model(tensor[:, start:stop].contiguous(), hidden)
        result[:, start:stop] = predicted_up(q).cpu().numpy()
    return [row[:length] for row, length in zip(result, lengths)]


def load_gru(path, device):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    state = saved["state_dict"]
    model = AttitudeGRU(state["input_mean"], state["input_std"])
    model.load_state_dict(state)
    return model.to(device).eval(), saved
