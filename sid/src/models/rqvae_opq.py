import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import MLPLayers
from .rq_opq import RQOPQ


class RQOPQVAE(nn.Module):
    """3 RQ + 1 OPQ (2 sub) stack, no Sinkhorn, with encoder/decoder."""

    def __init__(
        self,
        in_dim: int = 768,
        num_emb_list=None,
        e_dim: int = 64,
        layers=None,
        dropout_prob: float = 0.0,
        bn: bool = False,
        loss_type: str = "mse",
        quant_loss_weight: float = 1.0,
        beta: float = 0.25,
        kmeans_init: bool = False,
        kmeans_iters: int = 100,
        kmeans_init_max_samples: int | None = 200_000,
        kmeans_init_seed: int = 0,
        kmeans_init_dedup: bool = True,
        sk_iters: int = 100,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.num_emb_list = num_emb_list or [1024] * 4  # 3 RQ + 1 OPQ(2sub) = 5 codebooks (3 + 2)
        self.e_dim = e_dim
        self.layers = layers or [512, 256, 128]
        self.dropout_prob = dropout_prob
        self.bn = bn
        self.loss_type = loss_type
        self.quant_loss_weight = quant_loss_weight
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.kmeans_init_max_samples = kmeans_init_max_samples
        self.kmeans_init_seed = kmeans_init_seed
        self.kmeans_init_dedup = kmeans_init_dedup
        self.sk_iters = sk_iters

        self.encode_layer_dims = [self.in_dim] + self.layers + [self.e_dim]
        self.encoder = MLPLayers(
            layers=self.encode_layer_dims,
            dropout=self.dropout_prob,
            bn=self.bn,
        )

        # RQ + OPQ: n_subquantizers=[0,0,0,2] (2 sub-quantizers in last layer)
        self.rqopq = RQOPQ(
            n_layers=4,
            dim=self.e_dim,
            n_subquantizers=[0, 0, 0, 2],
            codebook_size=self.num_emb_list[0],
            beta=self.beta,
            kmeans_init=self.kmeans_init,
            kmeans_iters=self.kmeans_iters,
            kmeans_init_max_samples=self.kmeans_init_max_samples,
            kmeans_init_seed=self.kmeans_init_seed,
            kmeans_init_dedup=self.kmeans_init_dedup,
            sk_epsilon=[0.0, 0.0, 0.0, 0.0],
            sk_iters=self.sk_iters,
        )

        self.decode_layer_dims = self.encode_layer_dims[::-1]
        self.decoder = MLPLayers(
            layers=self.decode_layer_dims,
            dropout=self.dropout_prob,
            bn=self.bn,
        )

    def forward(self, x: torch.Tensor, use_sk: bool = False):
        x = self.encoder(x)
        x_q, rq_loss, indices = self.rqopq(x, use_sk=use_sk)
        out = self.decoder(x_q)
        return out, rq_loss, indices

    @torch.no_grad()
    def get_indices(self, xs: torch.Tensor, use_sk: bool = False):
        x_e = self.encoder(xs)
        _, _, indices = self.rqopq(x_e, use_sk=use_sk)
        return indices

    def compute_loss(self, out: torch.Tensor, quant_loss: torch.Tensor, xs: torch.Tensor = None):
        if self.loss_type == 'mse':
            loss_recon = F.mse_loss(out, xs, reduction='mean')
        elif self.loss_type == 'l1':
            loss_recon = F.l1_loss(out, xs, reduction='mean')
        else:
            raise ValueError('incompatible loss type')
        loss_total = loss_recon + self.quant_loss_weight * quant_loss
        return loss_total, loss_recon
