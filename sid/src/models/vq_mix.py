import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import kmeans, sinkhorn_algorithm


class VectorQuantizerMix(nn.Module):
    """VQ with gradient update + post-step EMA smoothing.

    Forward is identical to the standard VQ loss (codebook + beta*commitment), but it stores
    per-batch assignment stats. After optimizer.step(), call apply_post_ema() to update the
    embedding weights with EMA of the batch means.
    """

    def __init__(
        self,
        n_e: int,
        e_dim: int,
        *,
        beta: float = 0.25,
        kmeans_init: bool = False,
        kmeans_iters: int = 10,
        kmeans_init_max_samples: int = 200_000,
        kmeans_init_seed: int = 0,
        kmeans_init_dedup: bool = True,
        sk_epsilon: float = 0.003,
        sk_iters: int = 100,
        ema_decay: float = 0.99,
        ema_eps: float = 1e-5,
    ):
        super().__init__()

        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.kmeans_init_max_samples = kmeans_init_max_samples
        self.kmeans_init_seed = kmeans_init_seed
        self.kmeans_init_dedup = kmeans_init_dedup
        self.sk_epsilon = sk_epsilon
        self.sk_iters = sk_iters
        self.ema_decay = ema_decay
        self.ema_eps = ema_eps

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        if not kmeans_init:
            self.initted = True
            self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        else:
            self.initted = False

        self._last_cluster_size: torch.Tensor | None = None
        self._last_embed_sum: torch.Tensor | None = None

    def init_emb(self, data: torch.Tensor):
        centers = kmeans(
            data,
            self.n_e,
            self.kmeans_iters,
            max_samples=self.kmeans_init_max_samples,
            seed=self.kmeans_init_seed,
            dedup=self.kmeans_init_dedup,
            verbose=False,
        )
        self.embedding.weight.data.copy_(centers)
        self.initted = True

    @staticmethod
    def _center_distance_for_sinkhorn(distances: torch.Tensor):
        min_per_row, _ = torch.min(distances, dim=1, keepdim=True)
        normalized = distances - min_per_row
        max_per_row, _ = torch.max(normalized, dim=1, keepdim=True)
        normalized = normalized / (max_per_row + 1e-8)
        return normalized

    def forward(self, x: torch.Tensor, use_sk: bool = True):
        if not self.initted:
            self.init_emb(x.detach())

        x_shape = x.shape
        x_flat = x.view(-1, self.e_dim)

        # squared euclidean distances to codebook
        d = (
            torch.sum(x_flat**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1, keepdim=True).t()
            - 2 * torch.matmul(x_flat, self.embedding.weight.t())
        )

        if (not use_sk) or self.sk_epsilon <= 0:
            indices = torch.argmin(d, dim=-1)
        else:
            d = self._center_distance_for_sinkhorn(d).double()
            Q = sinkhorn_algorithm(d, self.sk_epsilon, self.sk_iters)
            if torch.isnan(Q).any() or torch.isinf(Q).any():
                raise ValueError("Sinkhorn Algorithm returns nan/inf values")
            indices = torch.argmax(Q, dim=-1)

        # store stats for post-step EMA
        cluster_size = torch.zeros(self.n_e, device=x_flat.device, dtype=x_flat.dtype)
        cluster_size.scatter_add_(0, indices, torch.ones_like(indices, dtype=x_flat.dtype))
        embed_sum = torch.zeros(self.n_e, self.e_dim, device=x_flat.device, dtype=x_flat.dtype)
        embed_sum.scatter_add_(0, indices.unsqueeze(1).expand(-1, self.e_dim), x_flat)
        self._last_cluster_size = cluster_size
        self._last_embed_sum = embed_sum

        x_q = self.embedding(indices).view(x_shape)

        commitment_loss = F.mse_loss(x_q.detach(), x)
        codebook_loss = F.mse_loss(x_q, x.detach())
        loss = codebook_loss + self.beta * commitment_loss

        x_q = x + (x_q - x).detach()
        indices = indices.view(x_shape[:-1])
        return x_q, loss, indices

    @torch.no_grad()
    def apply_post_ema(self):
        if self._last_cluster_size is None or self._last_embed_sum is None:
            return

        decay = float(self.ema_decay)
        cs = self._last_cluster_size
        es = self._last_embed_sum

        n = cs.sum()
        cs_smooth = (cs + self.ema_eps) / (n + self.n_e * self.ema_eps) * n
        embed_mean = es / cs_smooth.unsqueeze(1)

        self.embedding.weight.data.mul_(decay).add_(embed_mean, alpha=1.0 - decay)

        self._last_cluster_size = None
        self._last_embed_sum = None
