import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import kmeans, sinkhorn_algorithm


class VectorQuantizer(nn.Module):
    """Vector quantization with optional KMeans init and Sinkhorn soft assignment.

    Optionally supports EMA-updated codebook (VQ-VAE EMA style).
    In EMA mode, codebook vectors are updated from assignment statistics instead of gradients.
    """

    def __init__(
        self,
        n_e: int,
        e_dim: int,
        beta: float = 0.25,
        kmeans_init: bool = False,
        kmeans_iters: int = 10,
        kmeans_init_max_samples: int | None = 200_000,
        kmeans_init_seed: int = 0,
        kmeans_init_dedup: bool = True,
        sk_epsilon: float = 0.003,
        sk_iters: int = 100,
        *,
        ema_decay: float | None = None,
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
        self.use_ema = ema_decay is not None

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        if self.use_ema:
            # Buffers so they are saved in state_dict, moved with .to(device), but not optimized.
            self.register_buffer("cluster_size_ema", torch.zeros(self.n_e))
            self.register_buffer("embed_avg_ema", torch.zeros(self.n_e, self.e_dim))
        if not kmeans_init:
            self.initted = True
            self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        else:
            self.initted = False
            self.embedding.weight.data.zero_()

    def get_codebook(self):
        return self.embedding.weight

    def get_codebook_entry(self, indices, shape=None):
        z_q = self.embedding(indices)
        if shape is not None:
            z_q = z_q.view(shape)
        return z_q

    def init_emb(self, data: torch.Tensor):
        # Robust init: keep K unchanged, but guard against pathological duplicates/too-few-unique rows
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
        if self.use_ema:
            # Initialize EMA stats from the initial codebook.
            self.embed_avg_ema.data.copy_(centers)
            self.cluster_size_ema.data.fill_(1.0)
        self.initted = True

    @staticmethod
    def center_distance_for_constraint(distances: torch.Tensor):
        max_distance = distances.max()
        min_distance = distances.min()
        middle = (max_distance + min_distance) / 2
        amplitude = max_distance - middle + 1e-5
        centered = (distances - middle) / amplitude
        return centered

    def forward(self, x: torch.Tensor, use_sk: bool = True):
        latent = x.view(-1, self.e_dim)

        if not self.initted and self.training:
            self.init_emb(latent)

        d = torch.sum(latent ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1, keepdim=True).t() - \
            2 * torch.matmul(latent, self.embedding.weight.t())

        if not use_sk or self.sk_epsilon <= 0:
            indices = torch.argmin(d, dim=-1)
        else:
            d = self.center_distance_for_constraint(d).double()
            Q = sinkhorn_algorithm(d, self.sk_epsilon, self.sk_iters)
            if torch.isnan(Q).any() or torch.isinf(Q).any():
                print("Sinkhorn Algorithm returns nan/inf values.")
            indices = torch.argmax(Q, dim=-1)

        if self.use_ema and self.training:
            # EMA update uses hard assignments.
            # Avoid one_hot (N x K) which is very memory-heavy when K is large.
            cluster_size = torch.zeros(self.n_e, device=latent.device, dtype=latent.dtype)
            cluster_size.scatter_add_(0, indices, torch.ones_like(indices, dtype=latent.dtype))

            embed_sum = torch.zeros(self.n_e, self.e_dim, device=latent.device, dtype=latent.dtype)
            embed_sum.scatter_add_(0, indices.unsqueeze(1).expand(-1, self.e_dim), latent)

            decay = float(self.ema_decay)
            self.cluster_size_ema.mul_(decay).add_(cluster_size, alpha=1.0 - decay)
            self.embed_avg_ema.mul_(decay).add_(embed_sum, alpha=1.0 - decay)

            n = self.cluster_size_ema.sum()
            # Laplace smoothing of the cluster size.
            cluster_size = (self.cluster_size_ema + self.ema_eps) / (n + self.n_e * self.ema_eps) * n
            embed_normalized = self.embed_avg_ema / cluster_size.unsqueeze(1)
            self.embedding.weight.data.copy_(embed_normalized)

        x_q = self.embedding(indices).view(x.shape)

        commitment_loss = F.mse_loss(x_q.detach(), x)
        if self.use_ema:
            # In EMA mode, codebook is updated from assignment statistics (no gradient to codebook).
            codebook_loss = x_q.new_tensor(0.0)
            loss = self.beta * commitment_loss
        else:
            codebook_loss = F.mse_loss(x_q, x.detach())
            loss = codebook_loss + self.beta * commitment_loss

        x_q = x + (x_q - x).detach()
        indices = indices.view(x.shape[:-1])
        return x_q, loss, indices
