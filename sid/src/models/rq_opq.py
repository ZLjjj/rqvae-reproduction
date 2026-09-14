import torch
import torch.nn as nn

from .vq import VectorQuantizer


class PlainRQ(nn.Module):
    """Single VQ without rotation (n_sub=0 case)."""

    def __init__(self, dim: int, codebook_size: int,
                 beta: float = 0.25, kmeans_init: bool = False, kmeans_iters: int = 10,
                 kmeans_init_max_samples: int | None = 200_000, kmeans_init_seed: int = 0, kmeans_init_dedup: bool = True,
                 sk_epsilon: float = 0.003, sk_iters: int = 100):
        super().__init__()
        self.dim = dim
        self.quantizer = VectorQuantizer(
            codebook_size, dim,
            beta=beta,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
            kmeans_init_max_samples=kmeans_init_max_samples,
            kmeans_init_seed=kmeans_init_seed,
            kmeans_init_dedup=kmeans_init_dedup,
            sk_epsilon=sk_epsilon,
            sk_iters=sk_iters,
        )
        self.sk_epsilon = sk_epsilon

    def forward(self, x: torch.Tensor, use_sk: bool = True):
        x_q, loss, indices = self.quantizer(x, use_sk=use_sk)
        return x_q, loss, indices


class OPQ(nn.Module):
    """Optimized Product Quantization with learnable rotation and M sub-quantizers."""

    def __init__(self, dim: int, n_subquantizers: int, codebook_size: int,
                 beta: float = 0.25, kmeans_init: bool = False, kmeans_iters: int = 10,
                 kmeans_init_max_samples: int | None = 200_000, kmeans_init_seed: int = 0, kmeans_init_dedup: bool = True,
                 sk_epsilon: float = 0.003, sk_iters: int = 100):
        super().__init__()
        self.dim = dim
        self.n_subquantizers = n_subquantizers
        self.sub_dim = dim // n_subquantizers
        self.sk_epsilon = sk_epsilon

        if dim % n_subquantizers != 0:
            raise ValueError(f"dim {dim} must be divisible by n_subquantizers {n_subquantizers}")

        self.rotation = nn.Linear(dim, dim, bias=False)
        with torch.no_grad():
            self.rotation.weight.data.copy_(torch.eye(dim))

        self.quantizers = nn.ModuleList([
            VectorQuantizer(
                codebook_size, self.sub_dim,
                beta=beta,
                kmeans_init=kmeans_init,
                kmeans_iters=kmeans_iters,
                kmeans_init_max_samples=kmeans_init_max_samples,
                kmeans_init_seed=kmeans_init_seed,
                kmeans_init_dedup=kmeans_init_dedup,
                sk_epsilon=sk_epsilon,
                sk_iters=sk_iters,
            )
            for _ in range(n_subquantizers)
        ])

    def forward(self, x: torch.Tensor, use_sk: bool = True):
        input_shape = x.shape
        x_flat = x.view(-1, self.dim)

        x_rot = self.rotation(x_flat)
        splits = torch.split(x_rot, self.sub_dim, dim=-1)

        x_q_splits, indices_splits, losses = [], [], []
        for i, quantizer in enumerate(self.quantizers):
            x_sub = splits[i]
            x_q_sub, loss, indices = quantizer(x_sub, use_sk=use_sk)
            x_q_splits.append(x_q_sub)
            indices_splits.append(indices)
            losses.append(loss)

        x_q_rot = torch.cat(x_q_splits, dim=-1)
        x_q_flat = x_q_rot @ self.rotation.weight
        x_q = x_q_flat.view(input_shape)

        mean_loss = torch.stack(losses).mean()
        W = self.rotation.weight
        I = torch.eye(self.dim, device=W.device)
        ortho_loss = torch.norm(W @ W.t() - I) ** 2
        total_loss = mean_loss + 0.1 * ortho_loss

        all_indices = torch.stack(indices_splits, dim=-1)
        if len(input_shape) > 2:
            all_indices = all_indices.view(*input_shape[:-1], self.n_subquantizers)

        return x_q, total_loss, all_indices


class RQOPQ(nn.Module):
    """Residual stack of PlainRQ/OPQ layers."""

    def __init__(self, n_layers: int, dim: int, n_subquantizers, codebook_size: int,
                 beta: float = 0.25, kmeans_init: bool = False, kmeans_iters: int = 10,
                 kmeans_init_max_samples: int | None = 200_000, kmeans_init_seed: int = 0, kmeans_init_dedup: bool = True,
                 sk_epsilon=0.003, sk_iters: int = 100):
        super().__init__()

        if isinstance(n_subquantizers, int):
            n_sub_list = [n_subquantizers] * n_layers
        elif isinstance(n_subquantizers, list):
            if len(n_subquantizers) != n_layers:
                raise ValueError("Length of n_subquantizers list must match n_layers")
            n_sub_list = n_subquantizers
        else:
            raise ValueError("n_subquantizers must be int or list[int]")

        if isinstance(sk_epsilon, list):
            if len(sk_epsilon) != n_layers:
                raise ValueError("Length of sk_epsilon list must match n_layers")
            sk_eps_list = sk_epsilon
        else:
            sk_eps_list = [sk_epsilon] * n_layers

        self.layers = nn.ModuleList([
            PlainRQ(
                dim,
                codebook_size,
                beta=beta,
                kmeans_init=kmeans_init,
                kmeans_iters=kmeans_iters,
                kmeans_init_max_samples=kmeans_init_max_samples,
                kmeans_init_seed=kmeans_init_seed,
                kmeans_init_dedup=kmeans_init_dedup,
                sk_epsilon=sk_eps_list[i],
                sk_iters=sk_iters,
            )
            if n_sub == 0
            else OPQ(
                dim,
                n_sub,
                codebook_size,
                beta=beta,
                kmeans_init=kmeans_init,
                kmeans_iters=kmeans_iters,
                kmeans_init_max_samples=kmeans_init_max_samples,
                kmeans_init_seed=kmeans_init_seed,
                kmeans_init_dedup=kmeans_init_dedup,
                sk_epsilon=sk_eps_list[i],
                sk_iters=sk_iters,
            )
            for i, n_sub in enumerate(n_sub_list)
        ])

    def forward(self, x: torch.Tensor, use_sk: bool = True):
        residual = x
        all_x_q, all_losses, all_indices = [], [], []

        for layer in self.layers:
            x_q, loss, indices = layer(residual, use_sk=use_sk)
            residual = residual - x_q
            all_x_q.append(x_q)
            all_losses.append(loss)
            if indices.dim() == residual.dim() - 1:
                indices = indices.unsqueeze(-1)
            all_indices.append(indices)

        x_out = sum(all_x_q)
        mean_loss = torch.stack(all_losses).mean()
        all_indices = torch.cat(all_indices, dim=-1)
        return x_out, mean_loss, all_indices
