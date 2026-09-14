"""Tests for encoder module."""

import torch
import pytest
from models.encoder import Encoder, ResidualBlock


def test_residual_block():
    """Test residual block forward pass."""
    block = ResidualBlock(128, 128)
    x = torch.randn(2, 128, 32, 32)
    out = block(x)
    assert out.shape == x.shape


def test_encoder_output_shape():
    """Test encoder output shape."""
    encoder = Encoder(in_channels=3, hidden_channels=128, num_residual_blocks=2)
    x = torch.randn(2, 3, 256, 256)
    out = encoder(x)
    
    # After 2 downsampling layers (stride 2 each), spatial dimensions reduce by 4x
    assert out.shape == (2, 512, 64, 64)  # 256 / 4 = 64


def test_encoder_different_batch_sizes():
    """Test encoder with different batch sizes."""
    encoder = Encoder()
    
    for batch_size in [1, 4, 8]:
        x = torch.randn(batch_size, 3, 256, 256)
        out = encoder(x)
        assert out.shape[0] == batch_size


def test_encoder_backward():
    """Test encoder backward pass."""
    encoder = Encoder()
    x = torch.randn(2, 3, 256, 256, requires_grad=True)
    out = encoder(x)
    loss = out.sum()
    loss.backward()
    
    assert x.grad is not None
    for param in encoder.parameters():
        assert param.grad is not None
