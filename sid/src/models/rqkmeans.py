import torch
import torch.nn as nn

from .rq import ResidualVectorQuantizer


class RQKMeans(nn.Module):
    """5-codebook residual quantizer: 4 learned RQ layers (+4th balanced), 5th hard-coded.

    Note: the last layer packs (sale_type, publish_year_bucket) into a codebook index.
    If you expand HardCodeMapper bits, make sure num_emb_list[4] matches (e.g. 512 for 9 bits).
    """

    def __init__(
        self,
        num_emb_list=None,
        e_dim: int = 64,
        beta: float = 0.25,
        kmeans_init: bool = False,
        kmeans_iters: int = 100,
        sk_epsilons=None,
        sk_iters: int = 100,
    ):
        super().__init__()
        self.num_emb_list = num_emb_list or [1024] * 5
        self.e_dim = e_dim
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons or [0.0, 0.0, 0.0, 0.003]
        self.sk_iters = sk_iters

        self.rq = ResidualVectorQuantizer(
            self.num_emb_list[:4],
            self.e_dim,
            beta=self.beta,
            kmeans_init=self.kmeans_init,
            kmeans_iters=self.kmeans_iters,
            sk_epsilons=self.sk_epsilons,
            sk_iters=self.sk_iters,
        )
        self.hardcode = HardCodeMapper()

    def forward(self, x: torch.Tensor, use_sk: bool = True, hard_fields=None):
        x_q, rq_loss, indices = self.rq(x, use_sk=use_sk)
        if hard_fields is not None:
            hard_vec, hard_idx = self.hardcode.lookup(hard_fields, x_q.shape, x.device, x.dtype)
            x_q = x_q + hard_vec
            indices = torch.cat([indices, hard_idx], dim=-1)
        else:
            hard_idx = torch.zeros(indices.shape[:-1] + (1,), device=x.device, dtype=indices.dtype)
            indices = torch.cat([indices, hard_idx], dim=-1)
        return x_q, rq_loss, indices

    @torch.no_grad()
    def get_indices(self, xs: torch.Tensor, use_sk: bool = False, hard_fields=None):
        _, _, indices = self.rq(xs, use_sk=use_sk)
        if hard_fields is not None:
            _, hard_idx = self.hardcode.lookup(hard_fields, xs.shape, xs.device, xs.dtype)
            indices = torch.cat([indices, hard_idx], dim=-1)
        else:
            hard_idx = torch.zeros(indices.shape[:-1] + (1,), device=indices.device, dtype=indices.dtype)
            indices = torch.cat([indices, hard_idx], dim=-1)
        return indices


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

        # keep 2 bits for sale_type (0..3) for forward compatibility
        st = torch.tensor([min(max(self._parse_sale_type(v), 0), 3) for v in sale_vals], device=device, dtype=torch.long)

        # year -> bucket in [0, 125], otherwise 0
        year = torch.tensor([self._parse_int(v, 0) for v in publish_vals], device=device, dtype=torch.long)
        bucket = 2035 - year
        bucket = torch.where((bucket >= 0) & (bucket <= 125), bucket, torch.zeros_like(bucket))

        # 7 bits for bucket (0..127) + 2 bits for sale_type (0..3) => 9 bits
        idx_val = ((bucket & 0x7F) << 2) | (st & 0x3)
        idx_val = torch.clamp(idx_val, max=self.max_index)
        idx = idx_val.view(batch, 1)

        vec = torch.zeros(x_shape, device=device, dtype=dtype)
        return vec, idx
