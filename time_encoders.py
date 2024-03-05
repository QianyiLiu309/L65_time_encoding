"""
Time Encoding Module

Reference:
    - https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/tgn.html
"""

import torch
from torch import Tensor
from torch import nn
from torch.nn import Linear, Parameter

import numpy as np


def get_time_encoder(time_encoder: str, out_channels: int, mul: float = 1):
    if time_encoder == "learned_cos":
        return CosTimeEncoder(out_channels, mul=mul)
    elif time_encoder == "learned_exp":
        return ExpTimeEncoder(out_channels, mul=mul)
    elif time_encoder == "learned_gaussian":
        return GaussianTimeEncoder(out_channels, mul=mul)
    elif time_encoder == "graph_mixer":
        return FixedCosTimeEncoder(out_channels, parameter_requires_grad=False)
    elif time_encoder == "partial":
        return ScaledFixedCosTimeEncoder(out_channels)
    else:
        raise NotImplementedError(f"Unknown time encoder '{time_encoder}'")


class CosTimeEncoder(nn.Module):
    """Learnable cosine time encoder"""
    def __init__(self, out_channels: int, mul: float = 1):
        super().__init__()
        self.out_channels = out_channels
        self.lin = Linear(1, out_channels)
        self.mul = mul

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, timestamp: Tensor) -> Tensor:
        timestamp = timestamp * self.mul
        return self.lin(timestamp.unsqueeze(-1)).cos()


class ExpTimeEncoder(nn.Module):
    """Learnable exponential time encoder"""
    def __init__(self, out_channels: int, mul: float = 1):
        super().__init__()
        self.out_channels = out_channels
        self.lin = Linear(1, out_channels, bias=False)
        self.mul = mul

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, timestamp: Tensor) -> Tensor:
        timestamp = timestamp * self.mul
        # [w1 timestamp, w2 timestamp, w3 timestamp, ...]
        xs = self.lin(timestamp.unsqueeze(-1)).abs()
        return torch.exp(-xs)


class GaussianTimeEncoder(nn.Module):
    """Learnable Gaussian time encoder"""

    def __init__(self, out_channels: int, mul: float = 1):
        super().__init__()
        self.out_channels = out_channels
        self.lin = Linear(1, out_channels, bias=True)
        self.mul = mul

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, timestamp: Tensor) -> Tensor:
        timestamp = timestamp * self.mul
        return torch.exp(-self.lin(timestamp.unsqueeze(-1)) ** 2)


class FixedCosTimeEncoder(nn.Module):
    """Cosine time encoder with non-learnable exponential range of frequencies"""

    def __init__(self, out_channels: int, parameter_requires_grad: bool = True):
        """
        Time encoder.
        :param out_channels: int, dimension of time encodings
        :param parameter_requires_grad: boolean, whether the parameter in TimeEncoder needs gradient
        """
        super().__init__()

        self.out_channels = out_channels
        # trainable parameters for time encoding
        self.lin = Linear(1, out_channels)
        self.lin.weight = Parameter(
            (
                torch.from_numpy(
                    1 / 10 ** np.linspace(0, 9, out_channels, dtype=np.float32)
                )
            ).reshape(out_channels, -1)
        )
        self.lin.bias = Parameter(torch.zeros(out_channels))

        if not parameter_requires_grad:
            self.lin.weight.requires_grad = False
            self.lin.bias.requires_grad = False

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, timestamp: torch.Tensor):
        """
        compute time encodings of time in timestamps
        :param timestamps: Tensor, shape (batch_size, seq_len)
        :return:
        """
        # Tensor, shape (batch_size, seq_len, 1)
        timestamp = timestamp.unsqueeze(-1)

        # Tensor, shape (batch_size, seq_len, out_channels)
        output = torch.cos(self.lin(timestamp))

        return output


class ScaledFixedCosTimeEncoder(nn.Module):
    """Fixed exponential periods but with learnable multipliers"""

    def __init__(self, out_channels: int):
        """
        Time encoder.
        :param out_channels: int, dimension of time encodings
        :param parameter_requires_grad: boolean, whether the parameter in TimeEncoder needs gradient
        """
        super().__init__()

        self.out_channels = out_channels
        # trainable parameters for time encoding
        self.frequencies = Parameter(
            torch.from_numpy(
                1 / 10 ** np.linspace(-2, 7, out_channels, dtype=np.float32)
            ).unsqueeze(0)
        )
        self.frequencies.requires_grad = False

        self.lin = Linear(1, out_channels, bias=True)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, timestamps: torch.Tensor):
        """
        compute time encodings of time in timestamps
        :param timestamps: Tensor, shape (batch_size, seq_len)
        :return:
        """
        # Tensor, shape (batch_size, seq_len, 1)
        timestamps = timestamps.unsqueeze(-1)

        output = torch.matmul(timestamps, self.lin.weight.t())
        if output.shape[0] != 0:
            # TODO handle var shape
            output = output * self.frequencies
        output = output + self.lin.bias
        output = torch.cos(output)

        return output

    def get_parameter_norm(self):
        return torch.norm(self.lin.weight, p=2) + torch.norm(self.lin.bias, p=2)
