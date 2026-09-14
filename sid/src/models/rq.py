import torch
import torch.nn as nn

from .vq import VectorQuantizer


class ResidualVectorQuantizer(nn.Module):
    """Cascaded residual VQ layers."""

    def __init__(
        self,
        n_e_list,
        e_dim,
        sk_epsilons,
        beta: float = 0.25,
        kmeans_init: bool = False,
        kmeans_iters: int = 100,
        sk_iters: int = 100,
        kmeans_init_max_samples: int | None = 200_000,
        kmeans_init_seed: int = 0,
        kmeans_init_dedup: bool = True,
        *,
        ema_decay: float | None = None,
        ema_layers: int | None = None,
        ema_eps: float = 1e-5,
    ):
        super().__init__()
        self.n_e_list = n_e_list
        self.e_dim = e_dim
        self.num_quantizers = len(n_e_list)
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.kmeans_init_max_samples = kmeans_init_max_samples
        self.kmeans_init_seed = kmeans_init_seed
        self.kmeans_init_dedup = kmeans_init_dedup
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters

        ema_layers_eff = ema_layers if ema_layers is not None else self.num_quantizers

        self.vq_layers = nn.ModuleList(
            [
                VectorQuantizer(
                    n_e,
                    e_dim,
                    beta=self.beta,
                    kmeans_init=self.kmeans_init,
                    kmeans_iters=self.kmeans_iters,
                    kmeans_init_max_samples=getattr(self, "kmeans_init_max_samples", 200_000),
                    kmeans_init_seed=getattr(self, "kmeans_init_seed", 0),
                    kmeans_init_dedup=getattr(self, "kmeans_init_dedup", True),
                    sk_epsilon=sk_epsilon,
                    sk_iters=sk_iters,
                    ema_decay=ema_decay if (ema_decay is not None and i < ema_layers_eff) else None,
                    ema_eps=ema_eps,
                )
                for i, (n_e, sk_epsilon) in enumerate(zip(n_e_list, sk_epsilons))
            ]
        )

    def get_codebook(self):
        return torch.stack([q.get_codebook() for q in self.vq_layers])

    def forward(self, x: torch.Tensor, use_sk: bool = True):
        all_losses = []
        all_indices = []
        x_q = 0
        residual = x

        for quantizer in self.vq_layers:
            x_res, loss, indices = quantizer(residual, use_sk=use_sk)
            residual = residual - x_res
            x_q = x_q + x_res
            all_losses.append(loss)
            all_indices.append(indices)

        mean_loss = torch.stack(all_losses).mean()
        all_indices = torch.stack(all_indices, dim=-1)
        return x_q, mean_loss, all_indices
