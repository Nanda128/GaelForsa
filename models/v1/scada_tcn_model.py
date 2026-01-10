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
        for layer in self.layers:
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
class SCADATCNModel(nn.Module):
    """
    Complete SCADA Multitask TCN Model with three heads.
    """
    def __init__(self, num_features: int, num_regime_flags: int, hidden_channels: int = 64,
                 num_tcn_layers: int = 4, forecast_horizon: int = 6, num_fault_classes: int = 5,
                 kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()

        # Input channels: F + F + F + R = 3F + R
        self.input_channels = 3 * num_features + num_regime_flags
        self.num_features = num_features
        self.num_regime_flags = num_regime_flags

        # Backbone
        self.backbone = TCNBackbone(self.input_channels, hidden_channels, num_tcn_layers, kernel_size, dropout)

        # Heads
        self.reconstruction_head = ReconstructionHead(hidden_channels, num_features)
        self.forecast_head = ForecastHead(hidden_channels, num_features, forecast_horizon)
        self.fault_head = FaultHead(hidden_channels, num_fault_classes)

    def forward(self, x_corrupt: torch.Tensor, m_miss: torch.Tensor,
                m_mask: torch.Tensor, flags: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            x_corrupt: Corrupted features (B, F, L)
            m_miss: Missingness mask (B, F, L)
            m_mask: Training mask (B, F, L)
            flags: Regime flags (B, R, L)

        Returns:
            Tuple of (reconstruction, forecast, fault_probs)
        """
        # Concatenate inputs: [x_corrupt, m_miss, m_mask, flags]
        x_in = torch.cat([x_corrupt, m_miss, m_mask, flags], dim=1)  # (B, 3F+R, L)

        # Backbone
        h = self.backbone(x_in)  # (B, D, L)

        # Heads
        x_hat = self.reconstruction_head(h)  # (B, F, L)
        y_hat = self.forecast_head(h)  # (B, K, F)
        p_fault = self.fault_head(h)  # (B, C)

        return x_hat, y_hat, p_fault


class SCADATCNTrainer:
    """
    Trainer for the SCADA TCN model with masked reconstruction training.
    """
    def __init__(self, model: SCADATCNModel, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device

        # Loss functions
        self.reconstruction_loss = nn.HuberLoss(delta=1.0)
        self.forecast_loss = nn.HuberLoss(delta=1.0)
        self.fault_loss = nn.CrossEntropyLoss()

        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def train_step(self, batch_data: dict) -> dict:
        """
        Single training step with two streams.

        Args:
            batch_data: Dict containing 'x', 'm_miss', 'flags', 'y_true' (optional), 'y_fault' (optional)

        Returns:
            Dict of losses
        """
        self.model.train()

        x = batch_data['x'].to(self.device)  # (B, F, L)
        m_miss = batch_data['m_miss'].to(self.device)  # (B, F, L)
        flags = batch_data['flags'].to(self.device)  # (B, R, L)
        y_true = batch_data.get('y_true')  # (B, K, F)
        y_fault = batch_data.get('y_fault')  # (B,)

        if y_true is not None:
            y_true = y_true.to(self.device)
        if y_fault is not None:
            y_fault = y_fault.to(self.device)

        # Stream A: Masked reconstruction
        m_mask_a = self._sample_mask(m_miss, p_mask=0.15)  # 15% masking probability
        x_corrupt_a = self._corrupt_features(x, m_mask_a, c=0.0)

        x_hat_a, _, _ = self.model(x_corrupt_a, m_miss, m_mask_a, flags)

        # Reconstruction loss only on masked positions
        l_rec = self.reconstruction_loss(x_hat_a * (1 - m_mask_a), x * (1 - m_mask_a))

        # Stream B: Clean forecasting + fault
        m_mask_b = torch.ones_like(m_miss)  # No masking
        x_corrupt_b = x  # Clean features

        _, y_hat_b, p_fault_b = self.model(x_corrupt_b, m_miss, m_mask_b, flags)

        losses = {'l_rec': l_rec.item()}

        total_loss = 0.5 * l_rec  # Weight reconstruction

        if y_true is not None:
            l_pred = self.forecast_loss(y_hat_b, y_true)
            losses['l_pred'] = l_pred.item()
            total_loss += 0.3 * l_pred

        if y_fault is not None:
            l_fault = self.fault_loss(p_fault_b, y_fault)
            losses['l_fault'] = l_fault.item()
            total_loss += 0.2 * l_fault

        # Backprop
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        losses['total'] = total_loss.item()
        return losses

    def _sample_mask(self, m_miss: torch.Tensor, p_mask: float) -> torch.Tensor:
        """Sample training mask for reconstruction."""
        # Only mask where data is present (m_miss == 1)
        mask_candidates = (m_miss == 1).float()
        random_mask = torch.bernoulli(torch.full_like(mask_candidates, 1 - p_mask))
        return mask_candidates * random_mask

    def _corrupt_features(self, x: torch.Tensor, m_mask: torch.Tensor, c: float = 0.0) -> torch.Tensor:
        """Corrupt features where mask is 0."""
        return m_mask * x + (1 - m_mask) * c

    def predict(self, x: torch.Tensor, m_miss: torch.Tensor, flags: torch.Tensor,
                m_score: Optional[torch.Tensor] = None) -> dict:
        """
        Inference prediction.

        Args:
            x: Features (B, F, L)
            m_miss: Missingness mask (B, F, L)
            flags: Regime flags (B, R, L)
            m_score: Scoring mask for reconstruction (optional)

        Returns:
            Dict with predictions
        """
        self.model.eval()

        x = x.to(self.device)
        m_miss = m_miss.to(self.device)
        flags = flags.to(self.device)

        with torch.no_grad():
            if m_score is not None:
                # Mask-at-test for scoring
                m_score = m_score.to(self.device)
                x_corrupt = self._corrupt_features(x, m_score, c=0.0)
                m_mask = m_score
            else:
                # Clean inference
                x_corrupt = x
                m_mask = torch.ones_like(m_miss)

            x_hat, y_hat, p_fault = self.model(x_corrupt, m_miss, m_mask, flags)

            return {
                'x_hat': x_hat.cpu(),
                'y_hat': y_hat.cpu(),
                'p_fault': p_fault.cpu(),
                'm_score': m_score.cpu() if m_score is not None else None
            }


if __name__ == "__main__":
    # Example usage
    model = SCADATCNModel(
        num_features=20,  # F
        num_regime_flags=5,  # R
        hidden_channels=64,
        num_tcn_layers=4,
        forecast_horizon=6,
        num_fault_classes=5
    )

    trainer = SCADATCNTrainer(model)

    print(f"Model input channels: {model.input_channels}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Dummy batch
    batch_size, seq_len = 4, 100
    dummy_batch = {
        'x': torch.randn(batch_size, 20, seq_len),
        'm_miss': torch.randint(0, 2, (batch_size, 20, seq_len)).float(),
        'flags': torch.randint(0, 2, (batch_size, 5, seq_len)).float(),
        'y_true': torch.randn(batch_size, 6, 20),
        'y_fault': torch.randint(0, 5, (batch_size,))
    }

    losses = trainer.train_step(dummy_batch)
    print(f"Training losses: {losses}")