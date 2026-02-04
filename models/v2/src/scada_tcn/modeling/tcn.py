from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


def receptive_field(kernel_size: int, dilations: List[int]) -> int:
    return 1 + (kernel_size - 1) * int(sum(dilations))


def resolve_dilations(cfg_tcn: dict) -> List[int]:
    mode = str(cfg_tcn.get("dilations", "powers_of_two"))
    num_blocks = int(cfg_tcn.get("num_blocks", 4))
    if mode == "powers_of_two":
        return [2**i for i in range(num_blocks)]
    if mode == "explicit":
        d = list(cfg_tcn.get("dilation_list", []))
        if len(d) != num_blocks:
            raise ValueError(f"Explicit dilation_list length {len(d)} != num_blocks {num_blocks}")
        return [int(x) for x in d]
    raise ValueError(f"Unknown dilations mode: {mode}")


class CausalConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, bias: bool = True) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=self.kernel_size, dilation=self.dilation, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, L)
        pad = (self.kernel_size - 1) * self.dilation
        x = nn.functional.pad(x, (pad, 0))  # left pad only
        return self.conv(x)


class _TimeLayerNorm(nn.Module):
    """LayerNorm over channel dim, per timestep, for (B,C,L)."""

    def __init__(self, C: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(C, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B,C,L) -> (B,L,C) -> LN -> (B,C,L)
        x = x.transpose(1, 2)
        x = self.ln(x)
        return x.transpose(1, 2)


class ResidualBlock(nn.Module):
    def __init__(self, D: int, kernel_size: int, dilation: int, dropout: float, norm: str = "none") -> None:
        super().__init__()
        self.norm = str(norm)

        conv1 = CausalConv1d(D, D, kernel_size=kernel_size, dilation=dilation)
        conv2 = CausalConv1d(D, D, kernel_size=kernel_size, dilation=dilation)

        if self.norm == "weight_norm":
            conv1.conv = weight_norm(conv1.conv)
            conv2.conv = weight_norm(conv2.conv)

        self.conv1 = conv1
        self.conv2 = conv2
        self.dropout = nn.Dropout(float(dropout))
        self.act = nn.ReLU()

        if self.norm == "layer_norm":
            self.ln1 = _TimeLayerNorm(D)
            self.ln2 = _TimeLayerNorm(D)
        else:
            self.ln1 = nn.Identity()
            self.ln2 = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,D,L)
        out = self.conv1(x)
        out = self.ln1(out)
        out = self.act(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.ln2(out)
        out = self.act(out)
        out = self.dropout(out)

        return x + out


class TCNBackbone(nn.Module):
    def __init__(
        self,
        F_in: int,
        D: int,
        kernel_size: int,
        dilations: List[int],
        dropout: float,
        norm: str = "none",
    ) -> None:
        super().__init__()
        self.F_in = int(F_in)
        self.D = int(D)
        self.kernel_size = int(kernel_size)
        self.dilations = [int(d) for d in dilations]
        self.norm = str(norm)

        self.in_proj = nn.Conv1d(self.F_in, self.D, kernel_size=1)
        self.blocks = nn.ModuleList(
            [ResidualBlock(self.D, self.kernel_size, d, dropout=dropout, norm=self.norm) for d in self.dilations]
        )

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        # x_in: (B,L,F_in) -> (B,F_in,L)
        x = x_in.transpose(1, 2)
        x = self.in_proj(x)
        for blk in self.blocks:
            x = blk(x)
        # (B,D,L) -> (B,L,D)
        return x.transpose(1, 2)
