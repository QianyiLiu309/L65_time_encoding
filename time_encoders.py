"""
Time Encoding Module

Reference:
    - https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/tgn.html
"""

import torch
from torch import Tensor
from torch import nn
from torch.nn import Linear, Parameter
from abc import ABC, abstractmethod

import numpy as np


class TimeEncoder(ABC):
    @abstractmethod
    def forward(self, timestamps: Tensor) -> Tensor:
        raise NotImplementedError


def get_time_encoder(
    time_encoder: str, out_channels: int, mul: float = 1
) -> TimeEncoder:
    if time_encoder == "learned_cos":
        return CosTimeEncoder(out_channels, mul=mul)
    if time_encoder == "decay_amp":
        return DecayCosTimeEncoder(out_channels, mul=mul, mode='amplitude', learn_power=True, learn_freqs=True)
    if time_encoder == "decay_amp_gm":
        return DecayCosTimeEncoder(out_channels, mul=mul, mode='amplitude', learn_power=True, learn_freqs=False)
    if time_encoder == "decay_freq":
        return DecayCosTimeEncoder(out_channels, mul=mul, mode='frequency', learn_power=True, learn_freqs=True)
    elif time_encoder == "learned_exp":
        return ExpTimeEncoder(out_channels, mul=mul)
    elif time_encoder == "learned_gaussian":
        return GaussianTimeEncoder(out_channels, mul=mul)
    elif time_encoder == "fixed_gaussian":
        return GaussianTimeEncoder(out_channels, mul=mul, graphmixer_freqs=True, learnable=False)
    elif time_encoder == "graph_mixer":
        return FixedCosTimeEncoder(out_channels, mul=mul, parameter_requires_grad=False)
    elif time_encoder == "graph_mixer_exp":
        return GraphMixerTemperature(out_channels, mul=mul, per_channel=False)
    elif time_encoder == "graph_mixer_exp_pc":
        return GraphMixerTemperature(out_channels, mul=mul, per_channel=True)
    elif time_encoder == "scaled_fixed":
        return ScaledFixedCosTimeEncoder(out_channels)
    elif time_encoder == "learned_cos_learned_multiplier":
        return CosTimeEncoderWithLearnedMultiplier(out_channels, mul=mul)
    elif time_encoder == "mlp2":
        return MLPTimeEncoder(out_channels, mul=mul, depth=2)
    elif time_encoder == "mlp3":
        return MLPTimeEncoder(out_channels, mul=mul, depth=3)
    elif time_encoder == "mlp4":
        return MLPTimeEncoder(out_channels, mul=mul, depth=4)
    else:
        raise NotImplementedError(f"Unknown time encoder '{time_encoder}'")


class CosTimeEncoder(nn.Module, TimeEncoder):
    """Learnable cosine time encoder"""

    def __init__(self, out_channels: int, mul: float = 1):
        super().__init__()
        self.out_channels = out_channels
        self.lin = Linear(1, out_channels)
        self.mul = mul

    def forward(self, timestamps: Tensor) -> Tensor:
        timestamps = timestamps * self.mul
        return self.lin(timestamps.unsqueeze(-1)).cos()


class DecayCosTimeEncoder(nn.Module, TimeEncoder):
    """Learnable cosine time encoder"""

    # mode = 'amplitude' or 'frequency'
    def __init__(self, out_channels: int, mul: float = 1, mode: str = 'amplitude', learn_power: bool = True,
                 graphmixer_freq_init: bool = True, learn_freqs: bool = False):
        super().__init__()
        self.out_channels = out_channels
        self.lin = Linear(1, out_channels)
        self.mul = mul

        if graphmixer_freq_init:
            self.lin.weight = Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, out_channels, dtype=np.float32))).reshape(out_channels, -1))

        if not learn_freqs:
            self.lin.weight.requires_grad = False

        self.mode = mode
        self.power = nn.Parameter(torch.ones(1,) / 2, requires_grad=learn_power)

    def forward(self, timestamps: Tensor) -> Tensor:
        timestamps = timestamps.unsqueeze(-1) * self.mul
        t_pow = torch.abs(timestamps) ** self.power

        if self.mode == 'amplitude':
            t_pow = t_pow @ self.lin.weight.T
            return self.lin(timestamps).cos() / (torch.abs(t_pow) + 1)
        elif self.mode == 'frequency':
            return self.lin(t_pow).cos()


class GraphMixerTemperature(nn.Module, TimeEncoder):
    def __init__(self, out_channels: int, mul: float = 1, per_channel: bool = False):
        super().__init__()
        self.out_channels = out_channels
        self.mul = mul
        self.per_channel = per_channel

        if self.per_channel:
            # data = torch.linspace(0, 9, self.out_channels, dtype=torch.float32)
            data = torch.ones(self.out_channels)
            self.multipliers = nn.Parameter(data=data, requires_grad=True)
        else:
            self.temperature = nn.Parameter(data=torch.tensor(1.), requires_grad=True)
            self.temperature_bias = nn.Parameter(data=torch.tensor(0.))

        self.lin = Linear(1, 1)

    def forward(self, timestamps: Tensor) -> Tensor:
        timestamps = timestamps * self.mul

        if self.per_channel:
            exponents = self.multipliers * self.torch.linspace(0, 9, self.out_channels, dtype=torch.float32, device=timestamps.device)
        else:
            exponents = self.temperature * torch.linspace(0, 9, self.out_channels, dtype=torch.float32, device=timestamps.device) + self.temperature_bias
        freqs = 10 ** (-exponents)

        return torch.cos(timestamps.unsqueeze(-1) * freqs.unsqueeze(0))


class MLPTimeEncoder(nn.Module, TimeEncoder):
    """MLP with sinusoidal activation"""

    # depth = number of linear layers total
    def __init__(self, out_channels: int, mul: float = 1, depth: int = 2, hidden_size: int = None):
        super().__init__()
        self.out_channels = out_channels

        if hidden_size is None:
            hidden_size = out_channels * 4

        layers = []
        for i in range(depth):
            in_ch = 1 if i == 0 else hidden_size
            out_ch = hidden_size if i != depth-1 else out_channels
            layers.append(Linear(in_ch, out_ch))

        self.layers = nn.ModuleList(layers)

        self.lin = layers[0]  # silly workaround

        self.mul = mul

    def forward(self, timestamps: Tensor) -> Tensor:
        xs = timestamps.unsqueeze(-1) * self.mul
        for i in range(len(self.layers) - 1):
            xs = self.layers[i](xs)
            xs = torch.cos(xs)
        xs = self.layers[-1](xs)
        return xs


class CosTimeEncoderWithLearnedMultiplier(nn.Module, TimeEncoder):
    """Learnable cosine time encoder"""

    def __init__(self, out_channels: int, mul: float = 1):
        super().__init__()
        self.out_channels = out_channels
        self.lin = Linear(1, out_channels)

        self.init_mul = mul
        self.mul = Parameter(torch.tensor(mul))

    def forward(self, timestamps: Tensor) -> Tensor:
        timestamps = timestamps * self.mul
        return self.lin(timestamps.unsqueeze(-1)).cos()


class ExpTimeEncoder(nn.Module, TimeEncoder):
    """Learnable exponential time encoder"""

    def __init__(self, out_channels: int, mul: float = 1):
        super().__init__()
        self.out_channels = out_channels
        self.lin = Linear(1, out_channels, bias=False)
        self.mul = mul

    def forward(self, timestamps: Tensor) -> Tensor:
        timestamps = timestamps * self.mul
        # [w1 timestamps, w2 timestamps, w3 timestamps, ...]
        xs = self.lin(timestamps.unsqueeze(-1)).abs()
        return torch.exp(-xs)


class GaussianTimeEncoder(nn.Module, TimeEncoder):
    """Learnable Gaussian time encoder"""

    def __init__(self, out_channels: int, mul: float = 1, graphmixer_freqs: bool = False, learnable: bool = True):
        super().__init__()
        self.out_channels = out_channels
        self.lin = Linear(1, out_channels, bias=True)
        self.mul = mul

        if graphmixer_freqs:
            self.lin.weight = Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, out_channels, dtype=np.float32))).reshape(out_channels, -1))
            self.lin.bias = Parameter(torch.zeros(out_channels))

        if not learnable:
            self.lin.weight.requires_grad = False
            # self.lin.bias.requires_grad = False

    def forward(self, timestamps: Tensor) -> Tensor:
        timestamps = timestamps * self.mul
        return torch.exp(-self.lin(timestamps.unsqueeze(-1)) ** 2)


class FixedCosTimeEncoder(nn.Module, TimeEncoder):
    """Cosine time encoder with non-learnable exponential range of frequencies"""

    def __init__(
        self, out_channels: int, mul: float = 1.0, parameter_requires_grad: bool = False
    ):
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
        self.mul = mul

        if not parameter_requires_grad:
            self.lin.weight.requires_grad = False
            self.lin.bias.requires_grad = False

    def forward(self, timestamps: torch.Tensor):
        """
        compute time encodings of time in timestamps
        :param timestamps: Tensor, shape (batch_size, seq_len)
        :return:
        """
        # Tensor, shape (batch_size, seq_len, 1)
        timestamps = timestamps.unsqueeze(-1)
        timestamps = timestamps * self.mul

        # Tensor, shape (batch_size, seq_len, out_channels)
        output = torch.cos(self.lin(timestamps))

        return output


class ScaledFixedCosTimeEncoder(nn.Module, TimeEncoder):
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
                1 / 10 ** np.linspace(0, 9, out_channels, dtype=np.float32)
            ).unsqueeze(0)
        )
        self.frequencies.requires_grad = False

        self.lin = Linear(1, out_channels, bias=True)

    def forward(self, timestamps: torch.Tensor):
        """
        compute time encodings of time in timestamps
        :param timestamps: Tensor, shape (batch_size, seq_len)
        :return:
        """
        # Tensor, shape (batch_size, seq_len, 1)
        timestamps = timestamps.unsqueeze(-1)

        multiplier = self.lin.weight
        output = torch.matmul(timestamps, multiplier.t())
        if output.shape[0] != 0:
            # TODO handle var shape
            output = output * self.frequencies
        output = output + self.lin.bias
        output = torch.cos(output)

        return output

    def get_parameter_norm(self):
        return torch.norm(self.lin.weight, p=2) + torch.norm(self.lin.bias, p=2)
