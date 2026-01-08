"""
Neighbor search utilities with a torch-only fallback.

PocketFlow uses kNN and radius-based neighbor search to construct graphs over
3D coordinates. PyTorch Geometric exposes convenient helpers
(:func:`torch_geometric.nn.knn`, :func:`torch_geometric.nn.radius`,
:func:`torch_geometric.nn.knn_graph`, :func:`torch_geometric.nn.radius_graph`),
but these functions require the optional `torch-cluster` package (conda package
name: `pytorch_cluster`).

This module removes the *hard* runtime dependency on `torch-cluster` by:

1) delegating to PyG when `torch-cluster` is available (fast path), and
2) falling back to a pure PyTorch implementation when it is not.

The fallback is intentionally simple (O(N²) distance computation) and is meant
for portability; installing `torch-cluster` is still recommended for speed.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

try:  # Optional: used only to detect + call the fast torch-cluster-backed path.
    import torch_geometric.typing as pyg_typing

    _WITH_TORCH_CLUSTER = bool(getattr(pyg_typing, "WITH_TORCH_CLUSTER", False))
except Exception:  # pragma: no cover
    pyg_typing = None
    _WITH_TORCH_CLUSTER = False


def _as_2d(x: Tensor) -> Tensor:
    return x.view(-1, 1) if x.dim() == 1 else x


def _pairwise_metric(y: Tensor, x: Tensor, *, cosine: bool = False) -> Tensor:
    """Return a (M, N) matrix where smaller values indicate closer neighbors."""
    y = _as_2d(y)
    x = _as_2d(x)

    if cosine:
        y_norm = F.normalize(y, p=2, dim=-1)
        x_norm = F.normalize(x, p=2, dim=-1)
        # Cosine distance in [0, 2] (up to numerical error).
        return (1.0 - (y_norm @ x_norm.transpose(0, 1))).clamp_min_(0.0)

    # Squared Euclidean distance (avoid sqrt for speed).
    y_norm = (y * y).sum(dim=-1, keepdim=True)  # (M, 1)
    x_norm = (x * x).sum(dim=-1).unsqueeze(0)  # (1, N)
    dist2 = y_norm + x_norm - 2.0 * (y @ x.transpose(0, 1))
    return dist2.clamp_min_(0.0)


def knn(
    x: Tensor,
    y: Tensor,
    k: int,
    batch_x: Tensor | None = None,
    batch_y: Tensor | None = None,
    cosine: bool = False,
    *,
    num_workers: int,  # kept for API parity
    batch_size: int | None = None,  # kept for API parity (PyG>=2.7)
) -> Tensor:
    """Fallback for :func:`torch_geometric.nn.knn` (returns indices into y then x)."""
    if _WITH_TORCH_CLUSTER:
        from torch_geometric.nn import knn as pyg_knn

        return pyg_knn(
            x=x,
            y=y,
            k=k,
            batch_x=batch_x,
            batch_y=batch_y,
            cosine=cosine,
            num_workers=num_workers,
            batch_size=batch_size,
        )

    if batch_x is not None or batch_y is not None:
        if batch_x is None or batch_y is None:
            raise ValueError("Must pass both batch_x and batch_y, or neither.")
        return _knn_batched_fallback(x, y, k, batch_x=batch_x, batch_y=batch_y, cosine=cosine)

    return _knn_fallback(x, y, k, cosine=cosine)


def _knn_fallback(x: Tensor, y: Tensor, k: int, *, cosine: bool) -> Tensor:
    if x.device != y.device:
        y = y.to(x.device)

    x = _as_2d(x)
    y = _as_2d(y)

    num_x = int(x.size(0))
    num_y = int(y.size(0))
    if k <= 0 or num_x == 0 or num_y == 0:
        return torch.empty((2, 0), dtype=torch.long, device=x.device)

    k_eff = min(int(k), num_x)
    metric = _pairwise_metric(y, x, cosine=cosine)  # (num_y, num_x)
    _, nn_idx = metric.topk(k_eff, dim=1, largest=False, sorted=True)  # (num_y, k_eff)

    row = torch.arange(num_y, device=x.device, dtype=torch.long).repeat_interleave(k_eff)
    col = nn_idx.reshape(-1).to(torch.long)
    return torch.stack([row, col], dim=0)


def _knn_batched_fallback(
    x: Tensor,
    y: Tensor,
    k: int,
    *,
    batch_x: Tensor,
    batch_y: Tensor,
    cosine: bool,
) -> Tensor:
    if x.device != y.device:
        y = y.to(x.device)
        batch_y = batch_y.to(x.device)
    if batch_x.device != x.device:
        batch_x = batch_x.to(x.device)

    out: list[Tensor] = []
    for batch_id in torch.unique(batch_y, sorted=True):
        y_mask = batch_y == batch_id
        x_mask = batch_x == batch_id
        if not torch.any(y_mask) or not torch.any(x_mask):
            continue
        y_idx = torch.nonzero(y_mask, as_tuple=False).view(-1)
        x_idx = torch.nonzero(x_mask, as_tuple=False).view(-1)
        edge = _knn_fallback(x[x_mask], y[y_mask], k, cosine=cosine)
        if edge.numel() == 0:
            continue
        edge = torch.stack([y_idx[edge[0]], x_idx[edge[1]]], dim=0)
        out.append(edge)

    if not out:
        return torch.empty((2, 0), dtype=torch.long, device=x.device)
    return torch.cat(out, dim=1)


def knn_graph(
    x: Tensor,
    k: int,
    batch: Tensor | None = None,
    loop: bool = False,
    flow: str = "source_to_target",
    cosine: bool = False,
    *,
    num_workers: int,  # kept for API parity
    batch_size: int | None = None,  # kept for API parity (PyG>=2.7)
) -> Tensor:
    """Fallback for :func:`torch_geometric.nn.knn_graph` (returns source→target)."""
    if _WITH_TORCH_CLUSTER:
        from torch_geometric.nn import knn_graph as pyg_knn_graph

        return pyg_knn_graph(
            x=x,
            k=k,
            batch=batch,
            loop=loop,
            flow=flow,
            cosine=cosine,
            num_workers=num_workers,
            batch_size=batch_size,
        )

    if batch is not None:
        return _knn_graph_batched_fallback(x, k, batch=batch, loop=loop, flow=flow, cosine=cosine)
    return _knn_graph_fallback(x, k, loop=loop, flow=flow, cosine=cosine)


def _knn_graph_fallback(x: Tensor, k: int, *, loop: bool, flow: str, cosine: bool) -> Tensor:
    x = _as_2d(x)
    num_nodes = int(x.size(0))
    if k <= 0 or num_nodes == 0:
        return torch.empty((2, 0), dtype=torch.long, device=x.device)

    if flow not in {"source_to_target", "target_to_source"}:
        raise ValueError("flow must be 'source_to_target' or 'target_to_source'")

    max_neighbors = num_nodes if loop else max(num_nodes - 1, 0)
    k_eff = min(int(k), max_neighbors)
    if k_eff <= 0:
        return torch.empty((2, 0), dtype=torch.long, device=x.device)

    metric = _pairwise_metric(x, x, cosine=cosine)
    if not loop:
        metric.fill_diagonal_(float("inf"))
    _, nn_idx = metric.topk(k_eff, dim=1, largest=False, sorted=True)  # (N, k_eff)

    row = torch.arange(num_nodes, device=x.device, dtype=torch.long).repeat_interleave(k_eff)
    col = nn_idx.reshape(-1).to(torch.long)

    if flow == "source_to_target":
        return torch.stack([col, row], dim=0)
    return torch.stack([row, col], dim=0)


def _knn_graph_batched_fallback(
    x: Tensor, k: int, *, batch: Tensor, loop: bool, flow: str, cosine: bool
) -> Tensor:
    if batch.device != x.device:
        batch = batch.to(x.device)

    out: list[Tensor] = []
    for batch_id in torch.unique(batch, sorted=True):
        mask = batch == batch_id
        if not torch.any(mask):
            continue
        idx = torch.nonzero(mask, as_tuple=False).view(-1)
        edge = _knn_graph_fallback(x[mask], k, loop=loop, flow=flow, cosine=cosine)
        if edge.numel() == 0:
            continue
        out.append(idx[edge])

    if not out:
        return torch.empty((2, 0), dtype=torch.long, device=x.device)
    return torch.cat(out, dim=1)


def radius(
    x: Tensor,
    y: Tensor,
    r: float,
    batch_x: Tensor | None = None,
    batch_y: Tensor | None = None,
    max_num_neighbors: int = 32,
    *,
    num_workers: int,  # kept for API parity
    batch_size: int | None = None,  # kept for API parity (PyG>=2.7)
) -> Tensor:
    """Fallback for :func:`torch_geometric.nn.radius` (returns indices into y then x)."""
    if _WITH_TORCH_CLUSTER:
        from torch_geometric.nn import radius as pyg_radius

        return pyg_radius(
            x=x,
            y=y,
            r=r,
            batch_x=batch_x,
            batch_y=batch_y,
            max_num_neighbors=max_num_neighbors,
            num_workers=num_workers,
            batch_size=batch_size,
        )

    if batch_x is not None or batch_y is not None:
        if batch_x is None or batch_y is None:
            raise ValueError("Must pass both batch_x and batch_y, or neither.")
        return _radius_batched_fallback(
            x,
            y,
            r,
            batch_x=batch_x,
            batch_y=batch_y,
            max_num_neighbors=max_num_neighbors,
        )

    return _radius_fallback(x, y, r, max_num_neighbors=max_num_neighbors)


def _radius_fallback(x: Tensor, y: Tensor, r: float, *, max_num_neighbors: int) -> Tensor:
    if x.device != y.device:
        y = y.to(x.device)

    x = _as_2d(x)
    y = _as_2d(y)

    num_x = int(x.size(0))
    num_y = int(y.size(0))
    if num_x == 0 or num_y == 0:
        return torch.empty((2, 0), dtype=torch.long, device=x.device)

    if max_num_neighbors <= 0:
        return torch.empty((2, 0), dtype=torch.long, device=x.device)

    r2 = float(r) * float(r)
    metric = _pairwise_metric(y, x, cosine=False)  # squared distance
    metric = metric.masked_fill(metric > r2, float("inf"))

    k_eff = min(int(max_num_neighbors), num_x)
    vals, nn_idx = metric.topk(k_eff, dim=1, largest=False, sorted=True)

    row = torch.arange(num_y, device=x.device, dtype=torch.long).repeat_interleave(k_eff)
    col = nn_idx.reshape(-1).to(torch.long)
    valid = torch.isfinite(vals).reshape(-1)
    if valid.numel() != 0:
        row = row[valid]
        col = col[valid]
    return torch.stack([row, col], dim=0)


def _radius_batched_fallback(
    x: Tensor,
    y: Tensor,
    r: float,
    *,
    batch_x: Tensor,
    batch_y: Tensor,
    max_num_neighbors: int,
) -> Tensor:
    if x.device != y.device:
        y = y.to(x.device)
        batch_y = batch_y.to(x.device)
    if batch_x.device != x.device:
        batch_x = batch_x.to(x.device)

    out: list[Tensor] = []
    for batch_id in torch.unique(batch_y, sorted=True):
        y_mask = batch_y == batch_id
        x_mask = batch_x == batch_id
        if not torch.any(y_mask) or not torch.any(x_mask):
            continue
        y_idx = torch.nonzero(y_mask, as_tuple=False).view(-1)
        x_idx = torch.nonzero(x_mask, as_tuple=False).view(-1)
        edge = _radius_fallback(x[x_mask], y[y_mask], r, max_num_neighbors=max_num_neighbors)
        if edge.numel() == 0:
            continue
        edge = torch.stack([y_idx[edge[0]], x_idx[edge[1]]], dim=0)
        out.append(edge)

    if not out:
        return torch.empty((2, 0), dtype=torch.long, device=x.device)
    return torch.cat(out, dim=1)


def radius_graph(
    x: Tensor,
    r: float,
    batch: Tensor | None = None,
    loop: bool = False,
    max_num_neighbors: int = 32,
    flow: str = "source_to_target",
    *,
    num_workers: int,  # kept for API parity
    batch_size: int | None = None,  # kept for API parity (PyG>=2.7)
) -> Tensor:
    """Fallback for :func:`torch_geometric.nn.radius_graph` (returns source→target)."""
    if _WITH_TORCH_CLUSTER:
        from torch_geometric.nn import radius_graph as pyg_radius_graph

        return pyg_radius_graph(
            x=x,
            r=r,
            batch=batch,
            loop=loop,
            max_num_neighbors=max_num_neighbors,
            flow=flow,
            num_workers=num_workers,
            batch_size=batch_size,
        )

    if batch is not None:
        return _radius_graph_batched_fallback(
            x,
            r,
            batch=batch,
            loop=loop,
            max_num_neighbors=max_num_neighbors,
            flow=flow,
        )
    return _radius_graph_fallback(x, r, loop=loop, max_num_neighbors=max_num_neighbors, flow=flow)


def _radius_graph_fallback(
    x: Tensor,
    r: float,
    *,
    loop: bool,
    max_num_neighbors: int,
    flow: str,
) -> Tensor:
    x = _as_2d(x)
    num_nodes = int(x.size(0))
    if num_nodes == 0:
        return torch.empty((2, 0), dtype=torch.long, device=x.device)

    if max_num_neighbors <= 0:
        return torch.empty((2, 0), dtype=torch.long, device=x.device)

    if flow not in {"source_to_target", "target_to_source"}:
        raise ValueError("flow must be 'source_to_target' or 'target_to_source'")

    metric = _pairwise_metric(x, x, cosine=False)  # squared distance
    if not loop:
        metric.fill_diagonal_(float("inf"))

    r2 = float(r) * float(r)
    metric = metric.masked_fill(metric > r2, float("inf"))

    max_neighbors = num_nodes if loop else max(num_nodes - 1, 0)
    k_eff = min(int(max_num_neighbors), max_neighbors)
    if k_eff <= 0:
        return torch.empty((2, 0), dtype=torch.long, device=x.device)

    vals, nn_idx = metric.topk(k_eff, dim=1, largest=False, sorted=True)
    row = torch.arange(num_nodes, device=x.device, dtype=torch.long).repeat_interleave(k_eff)
    col = nn_idx.reshape(-1).to(torch.long)
    valid = torch.isfinite(vals).reshape(-1)
    if valid.numel() != 0:
        row = row[valid]
        col = col[valid]

    if flow == "source_to_target":
        return torch.stack([col, row], dim=0)
    return torch.stack([row, col], dim=0)


def _radius_graph_batched_fallback(
    x: Tensor,
    r: float,
    *,
    batch: Tensor,
    loop: bool,
    max_num_neighbors: int,
    flow: str,
) -> Tensor:
    if batch.device != x.device:
        batch = batch.to(x.device)

    out: list[Tensor] = []
    for batch_id in torch.unique(batch, sorted=True):
        mask = batch == batch_id
        if not torch.any(mask):
            continue
        idx = torch.nonzero(mask, as_tuple=False).view(-1)
        edge = _radius_graph_fallback(
            x[mask],
            r,
            loop=loop,
            max_num_neighbors=max_num_neighbors,
            flow=flow,
        )
        if edge.numel() == 0:
            continue
        out.append(idx[edge])

    if not out:
        return torch.empty((2, 0), dtype=torch.long, device=x.device)
    return torch.cat(out, dim=1)
