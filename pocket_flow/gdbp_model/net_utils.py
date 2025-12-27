"""Neural-network utility layers and helpers for the GDBP model.

This module groups small, reusable components used throughout the GDBP model
implementation:

- Parameter initialization / freezing helpers based on name substrings.
- Forward/reverse passes for simple affine flows.
- Common feature expansions (Gaussian distance smearing, edge direction expansion).
- Lightweight feature transforms (scalarization, rescaling, atom embedding).
- A label-smoothed cross entropy loss.
"""

from collections.abc import Sequence
from typing import Literal, override

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.modules.loss import _WeightedLoss

KEYS: list[str] = [
    "edge_flow.flow_layers.5",
    "atom_flow.flow_layers.5",
    "pos_predictor.mu_net",
    "pos_predictor.logsigma_net",
    "pos_predictor.pi_net",
    "focal_net.net",
]


def reset_parameters[T: nn.Module](model: T, keys: Sequence[str]) -> T:
    """Initialize a subset of parameters selected by name substrings.

    For each parameter whose name contains any substring in `keys`, this applies
    one of several initializations based on the parameter's name:

    - `"bias"`: constant 0.0
    - `"layernorm"`: constant 1.0
    - `"rescale.weight"`: constant 0.0
    - otherwise: Kaiming normal initialization

    Note:
        This function matches by substring containment. If a single parameter
        name matches multiple entries in `keys`, it may be re-initialized more
        than once (later matches "win").

    Args:
        model: Module whose parameters will be (partially) re-initialized.
        keys: Name substrings used to select which parameters to initialize.

    Returns:
        The same `model` object, for fluent chaining.
    """
    for name, para in model.named_parameters():
        for k in keys:
            if k in name and "bias" in name:
                torch.nn.init.constant_(para, 0.0)
            elif k in name and "layernorm" in name:
                torch.nn.init.constant_(para, 1.0)
            elif k in name and "rescale.weight" in name:
                torch.nn.init.constant_(para, 0.0)
            elif k in name:
                torch.nn.init.kaiming_normal_(para)
    return model


def freeze_parameters[T: nn.Module](model: T, keys: Sequence[str]) -> T:
    """Freeze a subset of parameters selected by name substrings.

    Any parameter whose name contains any substring in `keys` will have
    `requires_grad` set to `False`.

    Args:
        model: Module whose parameters will be (partially) frozen.
        keys: Name substrings used to select which parameters to freeze.

    Returns:
        The same `model` object, for fluent chaining.
    """
    for name, para in model.named_parameters():
        for k in keys:
            if k in name:
                para.requires_grad = False
    return model


def flow_reverse(
    flow_layers: nn.ModuleList, latent: Tensor, feat: tuple[Tensor, Tensor]
) -> tuple[Tensor, Tensor]:
    """Apply the inverse of an affine flow parameterized by `flow_layers`.

    Each layer is expected to be callable as `layer(feat)` and return a triple
    `(s_sca, t_sca, vec)`, where `s_sca` and `t_sca` are broadcastable to
    `latent`. The scale is interpreted in log-space and exponentiated before use.

    The inverse transform for each layer is:
        `latent <- (latent / exp(s_sca)) - t_sca`

    Args:
        flow_layers: Sequence of flow layers (applied in reverse order).
        latent: Latent tensor to transform (typically scalar features), shape
            `(..., D)` depending on the caller.
        feat: Conditioning features passed to each layer as-is, as a
            `(scalar, vector)` tuple.

    Returns:
        A `(latent, vec)` tuple where `latent` is the transformed tensor and
        `vec` is the last vector output produced by the flow layers.
    """
    vec = feat[1]
    for i in reversed(range(len(flow_layers))):
        s_sca, t_sca, vec = flow_layers[i](feat)
        s_sca = s_sca.exp()
        latent = (latent / s_sca) - t_sca
    return latent, vec


def flow_forward(
    flow_layers: nn.ModuleList, x_z: Tensor, feature: tuple[Tensor, Tensor]
) -> tuple[Tensor, Tensor, Tensor]:
    """Apply a forward affine flow and accumulate the log-Jacobian.

    Each layer is expected to be callable as `layer(feature)` and return a
    triple `(s_sca, t_sca, vec)`, where `s_sca` and `t_sca` are broadcastable to
    `x_z`. The scale is interpreted in log-space and exponentiated before use.

    The forward transform for each layer is:
        `x_z <- (x_z + t_sca) * exp(s_sca)`

    The per-dimension log-Jacobian contributions are accumulated as:
        `x_log_jacob += log(|exp(s_sca)|)`

    Args:
        flow_layers: Sequence of flow layers (applied in order).
        x_z: Tensor to transform, shape `(..., D)` depending on the caller.
        feature: Conditioning features passed to each layer as-is, as a
            `(scalar, vector)` tuple.

    Returns:
        A `(x_z, x_log_jacob, vec)` tuple where `x_z` is the transformed tensor,
        `x_log_jacob` has the same shape as `x_z`, and `vec` is the last vector
        output produced by the flow layers.
    """
    x_log_jacob = torch.zeros_like(x_z)
    vec = feature[1]
    for i in range(len(flow_layers)):
        s_sca, t_sca, vec = flow_layers[i](feature)
        s_sca = s_sca.exp()
        x_z = (x_z + t_sca) * s_sca
        x_log_jacob += (torch.abs(s_sca) + 1e-20).log()
    return x_z, x_log_jacob, vec


class GaussianSmearing(nn.Module):
    """Expand distances using fixed Gaussian radial basis functions (RBF).

    The expansion is computed against evenly spaced centers between `start` and
    `stop` (inclusive). Input distances are clamped to `stop` before expansion.

    Note:
        This implementation flattens the input to 1D via `dist.view(-1)` and
        returns a 2D tensor of shape `(dist.numel(), num_gaussians)`.
    """

    offset: Tensor
    stop: float
    coeff: float

    def __init__(self, start: float = 0.0, stop: float = 10.0, num_gaussians: int = 50) -> None:
        """Create a Gaussian distance expander.

        Args:
            start: Smallest Gaussian center.
            stop: Largest Gaussian center; also used as a clamp maximum.
            num_gaussians: Number of Gaussian basis functions.
        """
        super().__init__()
        self.stop = stop
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    @override
    def forward(self, dist: Tensor) -> Tensor:
        """Compute the RBF expansion for input distances.

        Args:
            dist: Distance tensor of any shape.

        Returns:
            A tensor of shape `(dist.numel(), num_gaussians)` containing the
            expanded features.
        """
        dist = dist.clamp_max(self.stop)
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class EdgeExpansion(nn.Module):
    """Embed a (normalized) 3D edge direction into `edge_channels` channels.

    This layer first normalizes each edge vector to unit length (with epsilon
    stabilization), then applies a learned linear projection on the per-axis
    scalar component to produce an output of shape `(N_edges, edge_channels, 3)`.
    """

    def __init__(self, edge_channels: int) -> None:
        """Create an edge direction expander.

        Args:
            edge_channels: Number of output channels per edge.
        """
        super().__init__()
        self.nn = nn.Linear(in_features=1, out_features=edge_channels, bias=False)

    @override
    def forward(self, edge_vector: Tensor) -> Tensor:
        """Expand edge directions.

        Args:
            edge_vector: Tensor of shape `(N_edges, 3)` representing 3D edge
                direction vectors.

        Returns:
            Tensor of shape `(N_edges, edge_channels, 3)`.
        """
        # If two points coincide, direction is undefined; emit a zero direction
        # rather than a noisy, potentially destabilising unit vector.
        EPS = 1e-7
        norm = torch.norm(edge_vector, p=2, dim=1, keepdim=True)
        safe_norm = norm.clamp_min(EPS)
        edge_unit = edge_vector / safe_norm
        edge_unit = torch.where(norm < EPS, torch.zeros_like(edge_unit), edge_unit)
        expansion = self.nn(edge_unit.unsqueeze(-1)).transpose(1, -1)
        return expansion


class Scalarize(nn.Module):
    """Map a (scalar, vector) feature tuple to pure scalar features.

    This module concatenates the scalar features with the per-channel L2 norms
    of the vector features, then applies an MLP:

        `concat([sca, ||vec||_2]) -> Linear -> act_fn -> Linear`
    """

    sca_in_dim: int
    vec_in_dim: int
    hidden_dim: int
    out_dim: int
    lin_scalarize_1: nn.Linear
    lin_scalarize_2: nn.Linear
    act_fn: nn.Module

    def __init__(
        self,
        sca_in_dim: int,
        vec_in_dim: int,
        hidden_dim: int,
        out_dim: int,
        act_fn: nn.Module = nn.Sigmoid(),
    ) -> None:
        """Create a scalarization module.

        Args:
            sca_in_dim: Number of scalar input channels.
            vec_in_dim: Number of vector input channels (i.e., vector feature
                channels, each with 3 components).
            hidden_dim: Hidden size of the scalar MLP.
            out_dim: Output scalar dimensionality.
            act_fn: Activation applied between the two linear layers.
        """
        super().__init__()
        self.sca_in_dim = sca_in_dim
        self.vec_in_dim = vec_in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.lin_scalarize_1 = nn.Linear(sca_in_dim + vec_in_dim, hidden_dim)
        self.lin_scalarize_2 = nn.Linear(hidden_dim, out_dim)
        self.act_fn = act_fn

    @override
    def forward(self, x: tuple[Tensor, Tensor]) -> Tensor:
        """Scalarize the input feature tuple.

        Args:
            x: `(sca, vec)` tuple where `sca` has shape `(..., sca_in_dim)` and
                `vec` has shape `(..., vec_in_dim, 3)`.

        Returns:
            Scalar tensor of shape `(-1, out_dim)` (batch/sample dimensions are
            flattened).
        """
        sca, vec = x[0].view(-1, self.sca_in_dim), x[1]
        norm_vec = torch.norm(vec, p=2, dim=-1).view(-1, self.vec_in_dim)
        sca = torch.cat([sca, norm_vec], dim=1)
        sca = self.lin_scalarize_1(sca)
        sca = self.act_fn(sca)
        sca = self.lin_scalarize_2(sca)
        return sca


class Rescale(nn.Module):
    """Learn a single positive rescaling factor `exp(weight)` and apply it.

    This is commonly used to stabilize training by allowing the network to learn
    an overall scale on certain activations.
    """

    weight: nn.Parameter

    def __init__(self) -> None:
        """Initialize the rescale factor to 1.0 via `weight = 0`."""
        super().__init__()
        self.weight = nn.Parameter(torch.zeros([1]))

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Scale `x` by the learned factor.

        Args:
            x: Input tensor of any shape.

        Returns:
            `exp(weight) * x`, with the same shape as `x`.

        Raises:
            RuntimeError: If `exp(weight)` contains NaNs (guard against unstable
                training / corrupted parameters).
        """
        if torch.isnan(torch.exp(self.weight)).any():
            print(self.weight)
            raise RuntimeError("Rescale factor has NaN entries")

        x = torch.exp(self.weight) * x
        return x


class AtomEmbedding(nn.Module):
    """Embed per-atom scalar features and a single 3D vector feature.

    The vector input is expected to contain exactly one 3D vector per atom and
    is normalized prior to projection.
    """

    in_scalar: int
    vector_normalizer: float | Tensor
    emb_sca: nn.Linear
    emb_vec: nn.Linear

    def __init__(
        self,
        in_scalar: int,
        in_vector: int,
        out_scalar: int,
        out_vector: int,
        vector_normalizer: float | Tensor = 20.0,
    ) -> None:
        """Create an atom embedding module.

        Args:
            in_scalar: Number of scalar input channels to embed. Only the first
                `in_scalar` columns of `scalar_input` are used.
            in_vector: Number of vector inputs. Must be 1 (one 3D vector).
            out_scalar: Output scalar embedding dimension.
            out_vector: Output vector embedding channels (each with 3 components).
            vector_normalizer: If a float, divide the vector input by this
                constant. If a tensor, normalize each vector by its L2 norm.
        """
        super().__init__()
        assert in_vector == 1
        self.in_scalar = in_scalar
        self.vector_normalizer = vector_normalizer
        self.emb_sca = nn.Linear(in_scalar, out_scalar)
        self.emb_vec = nn.Linear(in_vector, out_vector)

    @override
    def forward(self, scalar_input: Tensor, vector_input: Tensor) -> tuple[Tensor, Tensor]:
        """Embed scalar and vector atom features.

        Args:
            scalar_input: Tensor of shape `(N, F)` containing per-atom scalar
                features; only `scalar_input[:, :in_scalar]` is embedded.
            vector_input: Tensor of shape `(N, 3)` containing one 3D vector per
                atom (e.g., a coordinate or direction).

        Returns:
            A `(sca_emb, vec_emb)` tuple:
            - `sca_emb`: `(N, out_scalar)`
            - `vec_emb`: `(N, out_vector, 3)`

        Raises:
            AssertionError: If `vector_input` does not have shape `(N, 3)`.
        """
        if isinstance(self.vector_normalizer, float):
            vector_input = vector_input / self.vector_normalizer
        else:
            # Norm-based normalisation: clamp to avoid division by zero.
            denom = torch.norm(vector_input, p=2, dim=-1, keepdim=True).clamp_min(1e-16)
            vector_input = vector_input / denom
        assert vector_input.shape[1:] == (3,), "Not support. Only one vector can be input"
        sca_emb = self.emb_sca(scalar_input[:, : self.in_scalar])  # b, f -> b, f'
        vec_emb = vector_input.unsqueeze(-1)  # b, 3 -> b, 3, 1
        vec_emb = self.emb_vec(vec_emb).transpose(1, -1)  # b, 1, 3 -> b, f', 3
        return sca_emb, vec_emb


def embed_compose(
    compose_feature: Tensor,
    compose_pos: Tensor,
    idx_ligand: Tensor,
    idx_protein: Tensor,
    ligand_atom_emb: AtomEmbedding,
    protein_atom_emb: AtomEmbedding,
    emb_dim: tuple[int, int],
) -> list[Tensor]:
    """Embed ligand/protein atoms and scatter them back into a composed tensor.

    This helper applies two different `AtomEmbedding` modules to the ligand and
    protein subsets of a concatenated "compose" representation, then writes the
    resulting embeddings back into full-length tensors aligned with the original
    indices.

    Args:
        compose_feature: Per-atom scalar features for the concatenated system,
            shape `(N_total, F)`.
        compose_pos: Per-atom vector features, shape `(N_total, 3)` (despite the
            name, this is treated as the single vector input to `AtomEmbedding`).
        idx_ligand: 1D integer indices selecting ligand atoms in the composed
            tensors.
        idx_protein: 1D integer indices selecting protein atoms in the composed
            tensors.
        ligand_atom_emb: Embedding module for ligand atoms.
        protein_atom_emb: Embedding module for protein atoms.
        emb_dim: `(scalar_dim, vector_dim)` for the output tensors.

    Returns:
        A list `[h_sca, h_vec]` where:
        - `h_sca` has shape `(N_total, emb_dim[0])`
        - `h_vec` has shape `(N_total, emb_dim[1], 3)`
    """
    h_ligand = ligand_atom_emb(compose_feature[idx_ligand], compose_pos[idx_ligand])
    h_protein = protein_atom_emb(compose_feature[idx_protein], compose_pos[idx_protein])

    h_sca = torch.zeros(
        [len(compose_pos), emb_dim[0]],
    ).to(h_ligand[0])
    h_vec = torch.zeros(
        [len(compose_pos), emb_dim[1], 3],
    ).to(h_ligand[1])
    h_sca[idx_ligand], h_sca[idx_protein] = h_ligand[0], h_protein[0]
    h_vec[idx_ligand], h_vec[idx_protein] = h_ligand[1], h_protein[1]
    return [h_sca, h_vec]


class SmoothCrossEntropyLoss(_WeightedLoss):
    """Cross entropy loss with optional label smoothing.

    This implements a smoothed-target variant of cross entropy by converting
    integer class labels into a smoothed one-hot distribution and then computing
    `-(p_smooth * log_softmax(logits)).sum(dim=-1)`.

    Compared to `torch.nn.CrossEntropyLoss`, this expects:
    - `inputs`: logits of shape `(N, C, ...)` only for `C == inputs.size(-1)`
      style usage (the implementation uses `inputs.size(-1)` as `n_classes`).
    - `targets`: integer class indices of shape `(N,)`.
    """

    smoothing: float
    weight: Tensor | None
    reduction: str

    def __init__(
        self,
        weight: Tensor | None = None,
        reduction: Literal["none", "mean", "sum"] = "mean",
        smoothing: float = 0.0,
    ) -> None:
        """Create a label-smoothed cross entropy loss.

        Args:
            weight: Optional per-class rescaling weights of shape `(C,)`.
            reduction: Specifies the reduction to apply to the output:
                `"none"` | `"mean"` | `"sum"`.
            smoothing: Label smoothing factor in `[0, 1)`. A value of 0.0
                recovers the standard one-hot targets.
        """
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth_one_hot(targets: Tensor, n_classes: int, smoothing: float = 0.0) -> Tensor:
        """Convert integer labels to a smoothed one-hot distribution.

        Args:
            targets: Integer class indices of shape `(N,)` and integer dtype.
            n_classes: Number of classes `C`.
            smoothing: Smoothing factor in `[0, 1)`. The true class receives
                probability `1 - smoothing` and the remaining mass is distributed
                uniformly over the other classes.

        Returns:
            Tensor of shape `(N, C)` containing smoothed target distributions.
        """
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = (
                torch.empty(size=(targets.size(0), n_classes), device=targets.device)
                .fill_(smoothing / (n_classes - 1))
                .scatter_(1, targets.data.unsqueeze(1), 1.0 - smoothing)
            )
        return targets

    @override
    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """Compute the smoothed cross entropy loss.

        Args:
            inputs: Logits tensor of shape `(N, C)` (classes on the last axis).
            targets: Integer class indices tensor of shape `(N,)`.

        Returns:
            The reduced loss according to `self.reduction`:
            - `"none"`: shape `(N,)`
            - `"mean"` / `"sum"`: scalar tensor
        """
        targets = SmoothCrossEntropyLoss._smooth_one_hot(targets, inputs.size(-1), self.smoothing)
        lsm = F.log_softmax(inputs, -1)

        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)

        loss = -(targets * lsm).sum(-1)

        if self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "mean":
            loss = loss.mean()
        return loss
