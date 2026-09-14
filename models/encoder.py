"""RQVAE Encoder implementation."""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Residual block for encoder."""

    def __init__(self, in_channels: int, out_channels: int):
        """Initialize residual block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = out + identity
        out = self.relu(out)
        return out


class Encoder(nn.Module):
    """RQVAE Encoder."""

    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 128,
        num_residual_blocks: int = 2,
    ):
        """Initialize encoder.
        
        Args:
            in_channels: Number of input channels (default: 3 for RGB)
            hidden_channels: Number of hidden channels
            num_residual_blocks: Number of residual blocks
        """
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels

        # Initial convolution layer
        self.initial_conv = nn.Conv2d(
            in_channels, hidden_channels, kernel_size=3, stride=1, padding=1
        )

        # Residual blocks
        self.residual_blocks = nn.Sequential(
            *[
                ResidualBlock(hidden_channels, hidden_channels)
                for _ in range(num_residual_blocks)
            ]
        )

        # Downsampling layers
        self.down1 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Encoded tensor
        """
        x = self.initial_conv(x)
        x = self.residual_blocks(x)
        x = self.down1(x)
        x = self.down2(x)
        return x
