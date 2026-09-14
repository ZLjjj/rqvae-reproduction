import torch
import torch.nn as nn

from .rq_opq import RQOPQ


class RQOPQKMeans(nn.Module):
    """3 RQ + 1 OPQ(2 sub) stack, no Sinkhorn, kmeans-style quantizer (no decoder)."""

    def __init__(
        self,
        num_emb_list=None,
        e_dim: int = 64,
        beta: float = 0.25,
        kmeans_init: bool = False,
        kmeans_iters: int = 100,
        sk_iters: int = 100,
    ):
        super().__init__()
        self.num_emb_list = num_emb_list or [1024] * 4
        self.e_dim = e_dim
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_iters = sk_iters

        self.rqopq = RQOPQ(
            n_layers=4,
            dim=self.e_dim,
            n_subquantizers=[0, 0, 0, 2],
            codebook_size=self.num_emb_list[0],
            beta=self.beta,
            kmeans_init=self.kmeans_init,
            kmeans_iters=self.kmeans_iters,
            sk_epsilon=[0.0, 0.0, 0.0, 0.0],
            sk_iters=self.sk_iters,
        )

    def forward(self, x: torch.Tensor, use_sk: bool = False):
        x_q, rq_loss, indices = self.rqopq(x, use_sk=use_sk)
        return x_q, rq_loss, indices

    @torch.no_grad()
    def get_indices(self, xs: torch.Tensor, use_sk: bool = False):
        _, _, indices = self.rqopq(xs, use_sk=use_sk)
        return indices
