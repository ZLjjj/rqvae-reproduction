import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from torch.nn.init import xavier_normal_


class MLPLayers(nn.Module):
    """Configurable MLP with optional dropout/BN/activation per hidden layer."""

    def __init__(self, layers, dropout: float = 0.0, activation: str = "relu", bn: bool = False):
        super().__init__()
        self.layers = layers
        self.dropout = dropout
        self.activation = activation
        self.use_bn = bn

        modules = []
        for idx, (in_dim, out_dim) in enumerate(zip(self.layers[:-1], self.layers[1:])):
            modules.append(nn.Dropout(p=self.dropout))
            modules.append(nn.Linear(in_dim, out_dim))

            is_last = idx == len(self.layers) - 2
            if self.use_bn and not is_last:
                modules.append(nn.BatchNorm1d(num_features=out_dim))

            act = activation_layer(self.activation)
            if act is not None and not is_last:
                modules.append(act)

        self.mlp_layers = nn.Sequential(*modules)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, x):
        return self.mlp_layers(x)


def activation_layer(name: str = "relu"):
    if name is None:
        return None
    if isinstance(name, str):
        name = name.lower()
        if name == "sigmoid":
            return nn.Sigmoid()
        if name == "tanh":
            return nn.Tanh()
        if name == "relu":
            return nn.ReLU()
        if name == "leakyrelu":
            return nn.LeakyReLU()
        if name == "none":
            return None
    if isinstance(name, type) and issubclass(name, nn.Module):
        return name()
    raise NotImplementedError(f"activation function {name} is not implemented")


def kmeans(
    samples: torch.Tensor,
    num_clusters: int,
    num_iters: int = 10,
    *,
    max_samples: int | None = 200_000,
    seed: int = 0,
    dedup: bool = True,
    verbose: bool = False,
) -> torch.Tensor:
    """Run KMeans on CPU, return centers as torch Tensor on original device.

    Robustness notes:
    - Samples are filtered for non-finite rows.
    - Optionally sub-sampled and de-duplicated to reduce pathological duplicates.
    - If unique samples < num_clusters, falls back to a random/unique-fill init without changing K.
    """

    B, dim, dtype, device = samples.shape[0], samples.shape[-1], samples.dtype, samples.device
    x = samples.detach().cpu().numpy()

    # filter invalid rows early
    finite_mask = (~(torch.isnan(samples).any(dim=1) | torch.isinf(samples).any(dim=1))).detach().cpu().numpy()
    if finite_mask is not None and finite_mask.shape[0] == x.shape[0]:
        x = x[finite_mask]

    if x.shape[0] == 0:
        # nothing valid: random centers
        return torch.empty((num_clusters, dim), device=device, dtype=dtype).normal_(mean=0.0, std=1.0)

    rng = torch.Generator(device="cpu")
    rng.manual_seed(int(seed))

    # subsample (to keep sklearn runtime bounded)
    if max_samples is not None and x.shape[0] > int(max_samples):
        idx = torch.randperm(x.shape[0], generator=rng).numpy()[: int(max_samples)]
        x = x[idx]

    # de-duplicate exactly (byte-wise) to avoid "distinct clusters < K" warnings
    if dedup and x.shape[0] > 1:
        # view float rows as bytes via numpy structured array trick
        xr = x.view(np.dtype((np.void, x.dtype.itemsize * x.shape[1])))
        _, uniq_idx = np.unique(xr, return_index=True)
        x = x[np.sort(uniq_idx)]

    if verbose:
        print(f"kmeans_init: input_rows={B} valid_rows={x.shape[0]} K={num_clusters}")

    if x.shape[0] < num_clusters:
        # fallback: fill centers by sampling (with replacement) from available unique points
        rep = torch.randint(low=0, high=x.shape[0], size=(num_clusters,), generator=rng).numpy()
        centers = torch.from_numpy(x[rep]).to(device=device, dtype=dtype)
        return centers

    cluster = KMeans(n_clusters=num_clusters, max_iter=num_iters, n_init="auto", random_state=int(seed)).fit(x)
    centers = torch.from_numpy(cluster.cluster_centers_).to(device=device, dtype=dtype)
    return centers


@torch.no_grad()
def sinkhorn_algorithm(distances: torch.Tensor, epsilon: float, sinkhorn_iterations: int) -> torch.Tensor:
    """Compute Sinkhorn transport plan for distances matrix."""
    Q = torch.exp(-distances / epsilon)
    B = Q.shape[0]
    K = Q.shape[1]

    sum_Q = Q.sum(-1, keepdim=True).sum(-2, keepdim=True)
    Q /= sum_Q

    for _ in range(sinkhorn_iterations):
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= K

    Q *= B
    return Q
