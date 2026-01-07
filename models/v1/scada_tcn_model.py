import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class ResidualBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float = 0.2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        #casual convolution padding maintains sequence length
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=padding)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation, padding=padding)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection.

        Args:
            x: Input tensor (B, C, L)

        Returns:
            Output tensor (B, C_out, L)
        """
        # Remove padding from causal conv to maintain
        out = self.conv1(x)
        out = out[:, :, :x.size(2)]  # Remove future padding
        out = self.relu(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = out[:, :, :x.size(2)]  # Remove future padding
        out = self.relu(out)
        out = self.dropout2(out)

        residual = self.residual_conv(x) if self.residual_conv else x
        return out + residual
class TCNBackbone(nn.Module):
    """
    TCN backbone with dilated residual blocks.
    """
    def __init__(self, input_channels: int, hidden_channels: int, num_layers: int,
                 Kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2 ** i
            in_ch = input_channels if i == 0 else hidden_channels
            out_ch = hidden_channels

            self.layers.append(
                ResidualBlock(in_ch,  out_ch, kernel_size, dilation, dropout)
            )
    def forward(seld, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through TCN layers.
        Args:
            x: Input tensor (B, F_in, L)
        Returns:
            Hidden representation (B, D, L)
        """
        for layer in self.layers
            x = layer(x)
        return x

class ReconstructionHead(nn.Module):
    """
    Reconstruction head: maps hidden to feature reconstruction.
    """
    def __init__(self, hidden_channels: int, output_channels: int):
        super().__init__()
        self.conv = nn.Conv1d(hidden_channels, output_channels, 1)  # 1x1 conv

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: Hidden representation (B, D, L)

        Returns:
            Reconstructed features (B, F, L)
        """
        return self.conv(h)
class ForecastHead(nn.Module):
    """
    Forecast head: predicts future features over horizon K.
    """
    def __init__(self, hidden_channels: int, output_channels: int, forecast_horizon: int):
        super().__init__()
        self.forecast_horizon = forecast_horizon
        self.output_channels = output_channels
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, forecast_horizon * output_channels)
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: Hidden representation (B, D, L)

        Returns:
            Forecast (B, K, F)
        """
        # Use last timestep
        h_last = h[:, :, -1]  # (B, D)

        # MLP to forecast
        out = self.mlp(h_last)  # (B, K * F)

        # Reshape to (B, K, F)
        return out.view(-1, self.forecast_horizon, self.output_channels)


class FaultHead(nn.Module):
    """
    Fault classification head.
    """
    def __init__(self, hidden_channels: int, num_classes: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, num_classes)
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: Hidden representation (B, D, L)

        Returns:
            Fault probabilities (B, C)
        """
        h_pooled = torch.mean(h, dim=2)  # (B, D)
        logits = self.mlp(h_pooled)  # (B, C)
        return F.softmax(logits, dim=1)
