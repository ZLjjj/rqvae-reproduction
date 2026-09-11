import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import MLPLayers
from .rq import ResidualVectorQuantizer


class RQVAE(nn.Module):
    """5-codebook Residual Quantization VAE.

    Layout: 5 RQ layers, only the 4th uses Sinkhorn balancing, 5th is hard-coded (saletype/publish_year packed into 8 bits).
    """

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
        sk_epsilons=None,
        sk_iters: int = 100,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.num_emb_list = num_emb_list or [1024, 1024, 1024, 1024, 256]
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
        # default: 4th layer balanced, others no Sinkhorn
        self.sk_epsilons = sk_epsilons or [0.0, 0.0, 0.0, 0.003, 0.0]
        self.sk_iters = sk_iters

        self.encode_layer_dims = [self.in_dim] + self.layers + [self.e_dim]
        self.encoder = MLPLayers(
            layers=self.encode_layer_dims,
            dropout=self.dropout_prob,
            bn=self.bn,
        )

        self.rq = ResidualVectorQuantizer(
            self.num_emb_list[:4],  # first 4 learned RQ layers
            self.e_dim,
            beta=self.beta,
            kmeans_init=self.kmeans_init,
            kmeans_iters=self.kmeans_iters,
            kmeans_init_max_samples=self.kmeans_init_max_samples,
            kmeans_init_seed=self.kmeans_init_seed,
            kmeans_init_dedup=self.kmeans_init_dedup,
            sk_epsilons=self.sk_epsilons[:4],
            sk_iters=self.sk_iters,
        )

        # hardcode table for 5th codebook (non-learned)
        self.hardcode = HardCodeMapper()

        self.decode_layer_dims = self.encode_layer_dims[::-1]
        self.decoder = MLPLayers(
            layers=self.decode_layer_dims,
            dropout=self.dropout_prob,
            bn=self.bn,
        )

    def forward(self, x: torch.Tensor, use_sk: bool = True, hard_fields=None):
        x_e = self.encoder(x)
        x_q, rq_loss, indices = self.rq(x_e, use_sk=use_sk)

        if hard_fields is not None:
            hard_vec, hard_idx = self.hardcode.lookup(hard_fields, x_q.shape, x_e.device, x_e.dtype)
            x_q = x_q + hard_vec
            indices = torch.cat([indices, hard_idx], dim=-1)
        else:
            hard_vec = torch.zeros_like(x_q)
            hard_idx = torch.zeros(indices.shape[:-1] + (1,), device=x_q.device, dtype=indices.dtype)
            indices = torch.cat([indices, hard_idx], dim=-1)

        out = self.decoder(x_q)
        return out, rq_loss, indices

    @torch.no_grad()
    def get_indices(self, xs: torch.Tensor, use_sk: bool = False, hard_fields=None):
        x_e = self.encoder(xs)
        _, _, indices = self.rq(x_e, use_sk=use_sk)
        if hard_fields is not None:
            _, hard_idx = self.hardcode.lookup(hard_fields, x_e.shape, x_e.device, x_e.dtype)
            indices = torch.cat([indices, hard_idx], dim=-1)
        else:
            hard_idx = torch.zeros(indices.shape[:-1] + (1,), device=indices.device, dtype=indices.dtype)
            indices = torch.cat([indices, hard_idx], dim=-1)
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


class HardCodeMapper(nn.Module):
    """Pack sale_type (0-3) and publish_year_bucket (0-127) into a single codebook index.

    - publish_year_bucket definition:
        bucket = 2035 - year
        bucket in [0, 125] is kept; otherwise bucket=0
        Examples: 2035->0, 2034->1, ..., 1910->125

    Backward compatible field names for sale_type:
      - saletype, sale_type, pay_type, paytype

    Value mapping (case-insensitive):
      - "FREE" -> 1
      - "PAY"  -> 2
      - numeric strings (e.g. "0", "2") keep the original int mapping

    index = (publish_year_bucket & 0x7F) << 2 | (sale_type & 0x3)
    """

    def __init__(self, bits: int = 9):
        super().__init__()
        self.bits = bits
        self.max_index = (1 << bits) - 1

    @staticmethod
    def _get_first_present(fields: dict, keys, default):
        if not isinstance(fields, dict):
            return default
        for k in keys:
            if k in fields:
                return fields.get(k)
        return default

    @staticmethod
    def _pad_to_batch(vals, batch: int, fill=0):
        if vals is None:
            vals = []
        if not isinstance(vals, (list, tuple)):
            vals = [vals]
        if len(vals) != batch:
            vals = list(vals) + [fill] * max(batch - len(vals), 0)
            vals = vals[:batch]
        return vals

    @staticmethod
    def _parse_sale_type(v) -> int:
        if v is None:
            return 0
        if isinstance(v, str):
            s = v.strip()
            if s.isdigit():
                return int(s)
            u = s.upper()
            if u == "FREE":
                return 1
            if u == "PAY":
                return 2
            return 0
        try:
            return int(v)
        except Exception:
            return 0

    @staticmethod
    def _parse_int(v, default: int = 0) -> int:
        if v is None:
            return default
        if isinstance(v, str):
            s = v.strip()
            if s == "":
                return default
            try:
                return int(s)
            except Exception:
                return default
        try:
            return int(v)
        except Exception:
            return default

    def lookup(self, fields: dict, x_shape: torch.Size, device, dtype):
        batch = x_shape[0] if len(x_shape) > 0 else 0
        if not fields or batch == 0:
            return torch.zeros(x_shape, device=device, dtype=dtype), torch.zeros((batch, 1), device=device, dtype=torch.long)

        sale_vals = self._get_first_present(fields, ("saletype", "sale_type", "pay_type", "paytype"), [0] * batch)
        publish_vals = fields.get("publish_year", [0] * batch)

        sale_vals = self._pad_to_batch(sale_vals, batch, fill=0)
        publish_vals = self._pad_to_batch(publish_vals, batch, fill=0)

        st = torch.tensor([min(max(self._parse_sale_type(v), 0), 3) for v in sale_vals], device=device, dtype=torch.long)

        year = torch.tensor([self._parse_int(v, 0) for v in publish_vals], device=device, dtype=torch.long)
        bucket = 2035 - year
        bucket = torch.where((bucket >= 0) & (bucket <= 125), bucket, torch.zeros_like(bucket))

        idx_val = ((bucket & 0x7F) << 2) | (st & 0x3)
        idx_val = torch.clamp(idx_val, max=self.max_index)
        idx = idx_val.view(batch, 1)

        # hard-coded vector kept zero (broadcast-safe)
        vec = torch.zeros(x_shape, device=device, dtype=dtype)
        return vec, idx
